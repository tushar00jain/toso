"""Async request lifecycle -- the driver a serving engine would run.

One coroutine per request drives the real request lifecycle on the shared
:class:`~sim_common.async_engine.AsyncEngine` virtual clock:

1. sleep until the request's arrival,
2. ``await scheduler.schedule`` -- consult the **real** directory and route (the
   real ``locate_volumes`` read completes without suspending the loop, so routing
   is atomic: a consistent directory snapshot, like the coordinator's serialized
   mailbox),
3. wait out the prefill queue, then, if the plan pulls a remote prefix, drive a
   **real** ``client.get`` (charging fabric via the cost model), then charge the
   prefill compute,
4. ``await scheduler.on_complete`` -- publish the computed KV blocks back into the
   real directory (read-through) and evict,
5. on the decode-simulating path, admit the request to a decode batch and finalize
   its outcome when its last token is emitted.

Simplification (documented in SPEC): a block becomes reusable at prefill
*completion*, not while in flight, so two requests racing for the same brand-new
prefix may both compute it. Arrivals are typically spaced enough that this is rare.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

from .cluster import Cluster
from .model import Request
from .scheduler import Plan
from .trace import Metrics, RequestResult, Trace


class Driver:
    """Drives request arrivals through a scheduler and records metrics."""

    def __init__(
        self,
        loop,
        cluster: Cluster,
        scheduler,
        block_tokens: int,
        trace: Trace,
        metrics: Metrics,
    ) -> None:
        self.loop = loop
        self.cluster = cluster
        self.scheduler = scheduler
        self.B = block_tokens
        self.trace = trace
        self.metrics = metrics
        # On the decode-simulating path acceptance is provisional until the request
        # is admitted to a decode batch (it may still be shed on the TBT SLO). We
        # stash the accepted row here and publish it once decode finishes (success)
        # or admission is refused (wasted prefill).
        self._pending: Dict[str, RequestResult] = {}
        if getattr(scheduler, "tbt_enabled", False):
            scheduler.on_decode_finish = self._decode_done

    def _now(self) -> float:
        return self.loop.time()

    async def run(self, requests: List[Request]) -> None:
        """Run every request to completion on the shared virtual-clock loop."""
        ordered = sorted(requests, key=lambda r: (r.arrival, r.id))
        with self.cluster.installed():
            await asyncio.gather(*(self._run_request(r) for r in ordered))
            # Request coroutines end at prefill completion; decode continues on its
            # own step tasks, so keep the loop running until every token is emitted.
            engine = getattr(self.scheduler, "engine", None)
            if engine is not None:
                await engine.drain()

    async def _run_request(self, request: Request) -> None:
        delay = request.arrival - self._now()
        if delay > 0:
            await asyncio.sleep(delay)

        plan = await self.scheduler.schedule(request, self._now())
        if plan is None:
            self.trace.record(
                self._now(), "REJECT", f"{request.id} rejected (SLO/overload)"
            )
            self.metrics.add(
                RequestResult(
                    id=request.id, accepted=False, prompt_tokens=request.prompt_tokens
                )
            )
            return

        tbt = getattr(self.scheduler, "tbt_enabled", False)
        self._trace_route(plan)
        if not tbt:
            self._record_accepted(plan)
        else:
            self._pending[request.id] = self._make_accepted(plan)

        # (1) wait out the prefill queue at this instance.
        if plan.queue_wait > 0:
            await asyncio.sleep(plan.queue_wait)
        # (2) pull the remote prefix (a real client.get -> real fabric cost).
        if plan.reuse_source is not None and plan.pull_keys:
            await self.cluster.fetch(plan.prefill, plan.pull_keys)
        # (3) charge the prefill compute for the uncached suffix.
        if plan.prefill_t > 0:
            await asyncio.sleep(plan.prefill_t)

        # (4) publish the computed KV blocks into the real directory; evict.
        evicted = await self.scheduler.on_complete(plan)
        # Keep the analytical queue tail consistent with the real ops' clock.
        self.scheduler.busy_until[plan.prefill] = max(
            self.scheduler.busy_until[plan.prefill], self._now()
        )
        self._prefill_done(plan, evicted)

    # -- outcome bookkeeping ---------------------------------------------- #
    def _make_accepted(self, plan: Plan) -> RequestResult:
        return RequestResult(
            id=plan.request.id,
            accepted=True,
            ttft=plan.ttft,
            prompt_tokens=plan.request.prompt_tokens,
            cached_tokens=plan.cached_tokens,
            uncached_tokens=plan.uncached_tokens,
            transfer_bytes=plan.transfer_bytes,
            prefill=plan.prefill,
            reuse_source=plan.reuse_source,
        )

    def _record_accepted(self, plan: Plan) -> None:
        self.metrics.add(self._make_accepted(plan))

    def _trace_route(self, plan: Plan) -> None:
        if plan.reuse_source is not None:
            src = f" pull {plan.match_blocks}blk from {plan.reuse_source}"
        elif plan.match_blocks:
            src = f" local hit {plan.match_blocks}blk"
        else:
            src = " cold (no reuse)"
        self.trace.record(
            self._now(),
            "ROUTE",
            f"{plan.request.id} -> {plan.prefill}"
            f" (match {plan.match_blocks}blk,{src}, "
            f"compute {plan.uncached_tokens}tok, ttft {plan.ttft:.3f})",
        )

    def _prefill_done(self, plan: Plan, evicted: List[str]) -> None:
        note = ""
        if evicted:
            note = f"; evicted {len(evicted)} blk from {plan.prefill}"
        if not getattr(self.scheduler, "tbt_enabled", False):
            self.trace.record(
                self._now(),
                "DONE",
                f"{plan.request.id} prefill done on {plan.prefill}"
                f" (published {len(plan.request.block_keys)}blk){note}",
            )
            return
        # Decode-simulating path: try to admit the request to a decode batch.
        admitted = self.scheduler.admit_decode(plan, self._now())
        if not admitted:
            result = self._pending.pop(plan.request.id)
            result.accepted = False
            result.decode_rejected = True
            result.wasted_prefill = True
            self.metrics.add(result)
            self.trace.record(
                self._now(),
                "REJECT",
                f"{plan.request.id} decode rejected on {plan.decode}"
                f" (TBT SLO; wasted prefill on {plan.prefill}){note}",
            )
            return
        self.trace.record(
            self._now(),
            "DONE",
            f"{plan.request.id} prefill done on {plan.prefill}"
            f" (published {len(plan.request.block_keys)}blk){note}"
            f"; decoding on {plan.decode}",
        )

    def _decode_done(self, request: Request, tbt: float) -> None:
        """Finalize a request once its last decode token is emitted."""
        result = self._pending.pop(request.id)
        result.tbt = tbt
        self.metrics.add(result)
        self.trace.record(
            self._now(), "DECODE", f"{request.id} decode done (tbt {tbt:.3f})"
        )
