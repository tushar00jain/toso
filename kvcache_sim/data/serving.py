"""The serving loop: turning one routing decision into real store calls.

:class:`ServingPlane` is ``kvcache_sim``'s :class:`~proposed.plane.DataPlane`. The
The run harness releases one work item per request at its arrival
time; everything from there is here:

1. ask the scheduler to route the request (control), and record a rejection if it
   refuses;
2. wait out the prefill queue;
3. if the plan pulls a remote prefix, drive a **real** ``get_batch`` (charging
   fabric via the cost model);
4. charge the prefill compute;
5. ask the scheduler what to publish and evict (control), then make those real
   ``put_batch`` / ``notify_delete_batch`` calls;
6. tell the scheduler the clock the real ops actually reached;
7. on the decode-simulating path, admit the request to a decode batch (if control
   allows) and finalize its outcome when its last token is emitted.

Simplification (documented in SPEC): a block becomes reusable at prefill
*completion*, not while in flight, so two requests racing for the same brand-new
prefix may both compute it. Arrivals are typically spaced enough that this is rare.

Coupling lives here
-------------------
Whether prefill and decode contend for one instance's compute is a fact about the
deployment, not about the policy, so this plane owns it. On a coupled instance it
applies each accepted plan's reservation to the decode engine's timeline
(:meth:`~kvcache_sim.data.decode.DecodeEngine.reserve`) and reports each decode
step's end back to the scheduler
(:meth:`~kvcache_sim.control.scheduler._Base.observe_compute_busy`), so the
control plane's *predicted* prefill queue tracks the timeline decode is actually
using. A disaggregated pool does neither, and prefill never stalls decode. This
replaces the dict the scheduler and the decode engine used to share.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from proposed import DataPlane

from ..control.scheduler import Plan
from ..report.metrics import Metrics, RequestResult
from ..workload.request import Request
from .decode import DecodeEngine
from .store import KVStore


class ServingPlane(DataPlane):
    # Rows are published at rejection, at acceptance, or when the last decode
    # token lands -- never one per item, so the harness must not write them.
    writes_own_outcomes = True

    """Runs one request's lifecycle against the real store.

    Args:
        loop: the virtual-clock loop (its ``time()`` is the only clock).
        store: the :class:`~kvcache_sim.data.store.KVStore` verbs.
        scheduler: the control-plane scheduler (decides; never executes).
        trace: the run's shared trace.
        metrics: the run's :class:`~kvcache_sim.report.metrics.Metrics` ledger.
        coupled: whether prefill shares each decode instance's compute timeline.
        max_batch / profile / model: passed to the decode engine when decode is
            simulated.
    """

    def __init__(
        self,
        loop,
        store: KVStore,
        scheduler,
        *,
        trace,
        metrics: Metrics,
        coupled: bool = False,
        max_batch: int = 8,
        profile=None,
        model=None,
    ) -> None:
        self.loop = loop
        self.store = store
        self.scheduler = scheduler
        self.trace = trace
        self.metrics = metrics
        self.coupled = coupled
        # On the decode-simulating path acceptance is provisional until the request
        # is admitted to a decode batch (it may still be shed on the TBT SLO). We
        # stash the accepted row here and publish it once decode finishes (success)
        # or admission is refused (wasted prefill).
        self._pending: Dict[str, RequestResult] = {}
        self.engine: Optional[DecodeEngine] = None
        if getattr(scheduler, "tbt_enabled", False):
            self.engine = DecodeEngine(
                loop,
                scheduler.decode_ids,
                max_batch=max_batch,
                profile=profile if profile is not None else scheduler.profile,
                model=model if model is not None else scheduler.model,
                on_finish=self._decode_done,
                # Coupled: prefill and decode are one timeline, so every step is
                # reported to control's predicted queue. Disaggregated: never.
                on_compute_busy=(
                    scheduler.observe_compute_busy if coupled else None
                ),
            )
            # Control may *observe* decode occupancy for its TBT gates; it cannot
            # admit, step or drain through this interface.
            scheduler.attach_decode_load(self.engine)

    def _now(self) -> float:
        return self.loop.time()

    async def drain(self) -> None:
        """Keep the loop running until the last decode token is emitted."""
        if self.engine is not None:
            await self.engine.drain()

    # -- the request lifecycle -------------------------------------------- #
    async def execute(self, item) -> None:
        """Serve one request end to end (the runner already waited for arrival)."""
        request: Request = item.payload

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
        # The accepted plan reserved its prefill instance in control's predicted
        # queue. On a coupled instance that reservation occupies the same compute
        # the decode engine steps on, so apply it there too -- immediately, with
        # no await in between, so no decode step can slip past it.
        if self.coupled and self.engine is not None:
            self.engine.reserve(plan.prefill, plan.done_time)

        tbt = getattr(self.scheduler, "tbt_enabled", False)
        self._trace_route(plan)
        if not tbt:
            self.metrics.add(self._make_accepted(plan))
        else:
            self._pending[request.id] = self._make_accepted(plan)

        # (1) wait out the prefill queue at this instance.
        if plan.queue_wait > 0:
            await asyncio.sleep(plan.queue_wait)
        # (2) pull the remote prefix (a real client.get_batch -> real fabric cost).
        if plan.reuse_source is not None and plan.pull_keys:
            await self.store.fetch(plan.prefill, plan.pull_keys)
        # (3) charge the prefill compute for the uncached suffix.
        if plan.prefill_t > 0:
            await asyncio.sleep(plan.prefill_t)

        # (4) publish the computed KV blocks into the real directory; evict.
        completion = self.scheduler.complete(plan)
        await self.store.publish(completion.instance, completion.publish)
        if completion.evict:
            await self.store.evict(completion.instance, completion.evict)
        # (5) tell control the clock the real ops reached, and (coupled only) the
        # decode timeline the same instance now carries.
        now = self._now()
        self.scheduler.observe_prefill_done(completion.instance, now)
        if self.coupled and self.engine is not None:
            self.engine.reserve(
                completion.instance, self.scheduler.busy_until[completion.instance]
            )
        self._prefill_done(plan, completion.evict)

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
        # Decode-simulating path: control decides whether decode can honour the
        # TBT SLO; we perform (or skip) the admission.
        if not self.scheduler.decode_admission(plan):
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
        if self.engine is not None:
            self.engine.admit(plan.request, plan.decode)
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
