"""The serving loop: turning one routing decision into real store calls.

:class:`ServingPlane` is ``kvcache_sim``'s :class:`~proposed.plane.DataPlane`. The
The run harness releases one work item per request at its arrival
time; everything from there is here:

1. ask the coordinator to route the request (control), and record a rejection if
   it refuses;
2. wait out the prefill queue;
3. if the plan pulls a remote prefix, drive a **real** ``get_batch`` (charging
   fabric via the cost model);
4. charge the prefill compute;
5. ask the coordinator what to publish and evict (control), then make those real
   ``put_batch`` / ``notify_delete_batch`` calls;
6. tell the coordinator the clock the real ops actually reached;
7. on the decode-simulating path, admit the request to a decode batch (if control
   allows) and finalize its outcome when its last token is emitted.

Simplification (documented in SPEC): a block becomes reusable at prefill
*completion*, not while in flight, so two requests racing for the same brand-new
prefix may both compute it. Arrivals are typically spaced enough that this is rare.

The coordinator is not on this host
-----------------------------------
Control runs as a service holding the cluster-wide picture, so this plane reaches
it exactly the way it reaches the store: through a port, over calls that carry
values. That port is
:class:`~proposed.coordinator.Coordinator`, and it is the *only* thing
this module may touch on the control side -- ``check_structure.py`` rule 6 fails
the build on a field read, a subscript or a ``getattr`` through it, because none
of those survive the two planes being in different processes.

Concretely, this plane owns the decode engine and *reports* it: every batch change
goes out as a :class:`~kvcache_sim.control.scheduler.DecodeState` fact (a
list of estimated finish times -- the whole of what control asks about decode),
rather than control holding the engine and calling it. The engine's callbacks come
back here first, to their owner on this host, and this plane decides what to send
on.

Coupling lives here
-------------------
Whether prefill and decode contend for one instance's compute is a fact about the
deployment, not about the policy, so this plane owns it. On a coupled instance it
applies each accepted plan's reservation to the decode engine's timeline
(:meth:`~kvcache_sim.data._decode.DecodeEngine.reserve`) and reports each decode
step's end on through
a :class:`~kvcache_sim.control.scheduler.ComputeBusy` fact, so the control
plane's *predicted* prefill queue tracks the timeline decode is actually using. A
disaggregated pool does neither, and prefill never stalls decode.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from domain import DEFAULT_MODEL, DEFAULT_PROFILE, Model, prefill_time
from proposed import Coordinator, DataPlane

from ..control.scheduler import (
    AdmitDecode, ComputeBusy, DecodeState, Plan, PrefillFinished, Route,
)
from ..report.metrics import Metrics, RequestResult
from ..control.request import Request
from ._decode import DecodeEngine
from .store import KVStore

__all__ = ["ServingPlane"]


class ServingPlane(DataPlane):
    # Rows are published at rejection, at acceptance, or when the last decode
    # token lands -- never one per item, so the harness must not write them.
    writes_own_outcomes = True

    """Runs one request's lifecycle against the real store.

    The running loop's ``time()`` is the only clock (virtual under simulation).

    Args:
        store: the :class:`~kvcache_sim.data.store.KVStore` verbs.
        coordinator: the control plane, through its
            :class:`~proposed.coordinator.Coordinator` port and nothing
            else. It decides; it never executes, and it is not on this host.
        trace: the run's shared trace.
        metrics: the run's :class:`~kvcache_sim.report.metrics.Metrics` ledger.
        coupled: whether prefill shares each decode instance's compute timeline.
        simulate_decode / decode_ids / max_batch / profile / model: how this plane
            models decode. They come from the run's wiring
            (:func:`kvcache_sim.workload._serving.serving_plane`), which is the
            same place the coordinator got them -- not read off the coordinator,
            which would be this host inspecting another service's fields.
    """

    def __init__(
        self,
        store: KVStore,
        coordinator: Coordinator,
        *,
        trace,
        metrics: Metrics,
        coupled: bool = False,
        simulate_decode: bool = False,
        decode_ids: Optional[List[str]] = None,
        max_batch: int = 8,
        profile=DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
    ) -> None:
        self.store = store
        self.coordinator: Coordinator = coordinator
        self.trace = trace
        self.metrics = metrics
        self.coupled = coupled
        self.simulate_decode = simulate_decode
        # Cost constants, for the one thing this plane has to price itself: a
        # prefill re-priced because the reuse it was planned around is gone.
        self.profile = profile
        self.model = model
        self.block_tokens = store.block_tokens
        # On the decode-simulating path acceptance is provisional until the request
        # is admitted to a decode batch (it may still be shed on the TBT SLO). We
        # stash the accepted row here and publish it once decode finishes (success)
        # or admission is refused (wasted prefill).
        self._pending: Dict[str, RequestResult] = {}
        self.engine: Optional[DecodeEngine] = None
        if simulate_decode:
            self.engine = DecodeEngine(
                decode_ids if decode_ids is not None else [],
                max_batch=max_batch,
                profile=profile,
                model=model,
                on_finish=self._decode_done,
                # Coupled: prefill and decode are one timeline, so every step is
                # reported to control's predicted queue. Disaggregated: never.
                on_compute_busy=self._compute_busy if coupled else None,
                on_state=self._decode_state,
            )

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    # -- what this host tells the coordinator about its decode side -------- #
    def _decode_state(self, inst: str, finishes: List[float]) -> None:
        """Forward a changed decode batch. The engine reports here, not there."""
        self.coordinator.observe.broadcast(DecodeState(inst, tuple(finishes)))

    def _compute_busy(self, inst: str, until: float) -> None:
        """Forward a coupled instance's occupied compute timeline."""
        self.coordinator.observe.broadcast(ComputeBusy(inst, until))

    async def drain(self) -> None:
        """Keep the loop running until the last decode token is emitted."""
        if self.engine is not None:
            await self.engine.drain()

    # -- the request lifecycle -------------------------------------------- #
    async def execute(self, item) -> None:
        """Serve one request end to end (the runner already waited for arrival)."""
        request: Request = item.payload

        plan = await self.coordinator.decide.call_one(Route(request))
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

        tbt = self.simulate_decode
        self._trace_route(plan)
        row = self._make_accepted(plan)
        if not tbt:
            self.metrics.add(row)
        else:
            self._pending[request.id] = row

        # (1) wait out the prefill queue at this instance.
        if plan.queue_wait > 0:
            await asyncio.sleep(plan.queue_wait)
        # (2) the prefix this instance already had is a read the store never sees,
        # so tell it: the volume evicts on what it has observed.
        local_blocks = plan.match_blocks - len(plan.pull_keys)
        if local_blocks:
            await self.store.reuse(
                plan.prefill, list(plan.request.block_keys[:local_blocks])
            )
        # ...then pull the remote prefix (a real get_batch -> real fabric cost).
        prefill_t = plan.prefill_t
        if plan.reuse_source is not None and plan.pull_keys:
            try:
                await self.store.fetch(plan.prefill, plan.pull_keys)
            except KeyError:
                # The peer had those blocks when this was planned and does not now:
                # a volume it shares with other requests ran out of room and dropped
                # its coldest. Nothing is wrong -- a cache that cannot evict is not a
                # cache -- but this plan is stale, so recompute what was going to be
                # reused instead of failing the request. All of it: the pull is
                # all-or-nothing, and half a prefix is not a prefix.
                prefill_t = self._recompute(plan, row)
        # (3) charge the prefill compute for the uncached suffix.
        if prefill_t > 0:
            await asyncio.sleep(prefill_t)

        # (4) publish what this host now holds and did not before: the prefix it
        # pulled, plus the suffix it computed. Which blocks those are is not a
        # decision and not control's to make -- it is everything past what was
        # already local, and the plan says how much that was.
        fresh = list(plan.request.block_keys[local_blocks:])
        await self.store.publish(plan.prefill, fresh)
        # (5) tell control the clock the real ops reached, and (coupled only) the
        # decode timeline the same instance now carries.
        now = self._now()
        busy_until = await self.coordinator.decide.call_one(
            PrefillFinished(plan.prefill, now)
        )
        if self.coupled and self.engine is not None:
            # The reply, not a read of control's queue: prefill just occupied the
            # timeline decode steps on, and only the coordinator knows the tail.
            self.engine.reserve(plan.prefill, busy_until)
        await self._prefill_done(plan)

    def _recompute(self, plan: Plan, row: RequestResult) -> float:
        """Re-price this prefill with the reuse that vanished, and say what it costs.

        The remote prefix is gone, so only what this instance already held is still
        cached: the planned match minus the blocks that were going to be pulled.
        Corrects the row too -- the request really did compute those tokens, and a
        hit rate that counted the plan rather than the outcome would flatter the
        cache that dropped them.
        """
        local_blocks = plan.match_blocks - len(plan.pull_keys)
        cached = min(local_blocks * self.block_tokens, plan.request.prompt_tokens)
        uncached = plan.request.prompt_tokens - cached
        row.cached_tokens = cached
        row.uncached_tokens = uncached
        row.transfer_bytes = 0
        recomputed = prefill_time(uncached, self.profile, self.model)
        row.ttft += recomputed - plan.prefill_t
        self.trace.record(
            self._now(),
            "RESTALE",
            f"{plan.request.id} lost {len(plan.pull_keys)}blk of reuse on "
            f"{plan.reuse_source} (evicted); recomputing on {plan.prefill}",
        )
        return recomputed

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

    async def _prefill_done(self, plan: Plan) -> None:
        note = ""
        if not self.simulate_decode:
            self.trace.record(
                self._now(),
                "DONE",
                f"{plan.request.id} prefill done on {plan.prefill}"
                f" (published {len(plan.request.block_keys)}blk){note}",
            )
            return
        # Decode-simulating path: control decides whether decode can honour the
        # TBT SLO; we perform (or skip) the admission.
        if not await self.coordinator.decide.call_one(AdmitDecode(plan)):
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
