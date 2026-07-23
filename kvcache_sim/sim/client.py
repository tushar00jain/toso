"""Simulated client / driver -- the request lifecycle.

Mirrors what a serving engine does against the store: on arrival it calls
``scheduler.schedule`` (the cache-aware scheduler), and on prefill completion it publishes the
computed KV blocks back into the cache via ``scheduler.on_complete`` (read-through,
K4). The scheduler is the only decision-maker; the client just drives events and
records outcomes.

Simplification (documented in SPEC): a block becomes reusable at prefill
*completion*, not while in flight. Unlike the dedup coordinator, this prototype does
not treat an in-flight prefill as a promised cache source, so two requests racing
for the same brand-new prefix may both compute it. Arrivals are typically spaced
enough that this is rare; adding promises (as in ``dedup_sim``) is future work.
"""

from __future__ import annotations

from typing import Dict, List

from sim_common.engine import Sim
from .model import Request
from .scheduler import Plan
from .trace import Metrics, RequestResult, Trace


class Client:
    """Drives request arrivals through a scheduler and records metrics."""

    def __init__(self, sim: Sim, scheduler, block_tokens: int, trace: Trace,
                 metrics: Metrics) -> None:
        self.sim = sim
        self.scheduler = scheduler
        self.B = block_tokens
        self.trace = trace
        self.metrics = metrics
        # When the scheduler simulates decode, acceptance is *provisional* until
        # the request is admitted to a decode batch (it may still be shed on the
        # TBT SLO). We stash the accepted row here and only publish it to metrics
        # once decode finishes (success) or admission is refused (wasted prefill).
        self._pending: Dict[str, RequestResult] = {}
        if getattr(scheduler, "tbt_enabled", False):
            scheduler.on_decode_finish = self._decode_done

    def submit(self, request: Request) -> None:
        """Enqueue a request's arrival event at its arrival time."""
        self.sim.schedule(request.arrival, lambda: self._arrive(request),
                          label=f"arrive:{request.id}")

    def _arrive(self, request: Request) -> None:
        plan = self.scheduler.schedule(request, self.sim.now)
        if plan is None:
            # Pre-prefill rejection (TTFT/predicted-TBT SLO): no compute spent.
            self.trace.record(self.sim.now, "REJECT",
                              f"{request.id} rejected (SLO/overload)")
            self.metrics.add(RequestResult(id=request.id, accepted=False,
                                           prompt_tokens=request.prompt_tokens))
            return
        if not getattr(self.scheduler, "tbt_enabled", False):
            self._record_accepted(plan)
            self._trace_route(plan)
            self.sim.schedule(plan.ttft, lambda: self._prefill_done(plan),
                              label=f"prefill_done:{request.id}")
            return
        # Decode-simulating path: acceptance is provisional until decode admits.
        self._trace_route(plan)
        self._pending[request.id] = self._make_accepted(plan)
        self.sim.schedule(plan.ttft, lambda: self._prefill_done(plan),
                          label=f"prefill_done:{request.id}")

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
            self.sim.now, "ROUTE",
            f"{plan.request.id} -> {plan.prefill}"
            f" (match {plan.match_blocks}blk,{src}, "
            f"compute {plan.uncached_tokens}tok, ttft {plan.ttft:.3f})",
        )

    def _prefill_done(self, plan: Plan) -> None:
        evicted = self.scheduler.on_complete(plan)
        note = ""
        if evicted:
            note = f"; evicted {len(evicted)} blk from {plan.prefill}"
        if not getattr(self.scheduler, "tbt_enabled", False):
            self.trace.record(
                self.sim.now, "DONE",
                f"{plan.request.id} prefill done on {plan.prefill}"
                f" (published {len(plan.request.block_keys)}blk){note}",
            )
            return
        # Decode-simulating path: try to admit the request to a decode batch.
        admitted = self.scheduler.admit_decode(plan, self.sim.now)
        if not admitted:
            # Prefill already spent -> a wasted prefill (late TBT rejection).
            result = self._pending.pop(plan.request.id)
            result.accepted = False
            result.decode_rejected = True
            result.wasted_prefill = True
            self.metrics.add(result)
            self.trace.record(
                self.sim.now, "REJECT",
                f"{plan.request.id} decode rejected on {plan.decode}"
                f" (TBT SLO; wasted prefill on {plan.prefill}){note}",
            )
            return
        # Admitted to decode: the provisional row stays until decode finishes.
        self.trace.record(
            self.sim.now, "DONE",
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
            self.sim.now, "DECODE",
            f"{request.id} decode done (tbt {tbt:.3f})",
        )


def submit_all(client: Client, requests: List[Request]) -> None:
    """Enqueue every request's arrival."""
    for r in requests:
        client.submit(r)
