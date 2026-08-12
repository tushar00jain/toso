"""The serving loop: turning one routing decision into real store calls.

:class:`ServingHost` is **one serving instance**: its cache, its decode batch, its
compute. A deployment runs one per host and they reach each other over the same
kind of port they reach the store and the coordinator over.

Every host is also a router
---------------------------
A request can land on any host, and the host it lands on is rarely the host that
should serve it: which instance holds the longest reusable prefix is a
cluster-wide fact. So a host that receives a request asks the coordinator where it
belongs (:meth:`ServingHost.receive`), then either serves it or forwards it to the
host named (:meth:`ServingHost.serve`). Routing is a *role every host plays*, not
a tier in front of them -- a single router object would re-centralize exactly what
running one of these per host decentralizes.

What is left over is which host a request arrives at, and that is a load
balancer's answer rather than a serving decision -- a client SDK, an ingress proxy
or DNS, none of which is part of the serving system and none of which survives
into a deployment of it. So it is not here: the run's wiring stands in for it
(:mod:`kvcache_sim.workload._serving`), and a host is simply told a request has
arrived.

The lifecycle, once a host is serving:

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
Whether prefill and decode contend for this host's compute is a fact about the
deployment, not about the policy, so the host owns it. When coupled it applies each
accepted plan's reservation to its decode engine's timeline
(:meth:`~kvcache_sim.data._decode.DecodeEngine.reserve`) and reports each decode
step's end on through
a :class:`~kvcache_sim.control.scheduler.ComputeBusy` fact, so the control
plane's *predicted* prefill queue tracks the timeline decode is actually using. A
disaggregated host does neither, and prefill never stalls decode.

Reaching another host
---------------------
Two things cross a host boundary: forwarding a request to the host that should
prefill it, and handing a finished prefill to the host that will decode it (which
under disaggregation is always a different host). Both go through ``peers`` -- a
lookup returning an endpoint-shaped reference, the same shape as the coordinator
handle -- so both are charged a hop rather than being a method call on an object
this host should not be able to see. ``peers`` is a lookup and not a dict of
objects for that reason.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from domain import DEFAULT_MODEL, DEFAULT_PROFILE, Model, prefill_time
from proposed import Coordinator

from ..control.scheduler import (
    AdmitDecode, ComputeBusy, DecodeState, Plan, PrefillFinished, Route,
)
from ..report.metrics import Metrics, RequestResult
from ..control.request import Request
from ._decode import DecodeEngine
from .store import KVStore

__all__ = ["ServingHost"]


class ServingHost:
    """One serving instance: its cache, its decode batch, its compute.

    The running loop's ``time()`` is the only clock (virtual under simulation).

    Args:
        me: this host's instance id. The only place an instance id is *this*
            host's; every other one names a peer.
        store: the :class:`~kvcache_sim.data.store.KVStore` verbs.
        coordinator: the control plane, through its
            :class:`~proposed.coordinator.Coordinator` port and nothing
            else. It decides; it never executes, and it is not on this host.
        peers: ``instance id -> a reference to that host``, endpoint-shaped
            (``.serve.call_one(...)``). A lookup rather than a dict of objects,
            because what a host holds for another host is a reference, and the
            hop is charged on the way through it.
        trace: the run's shared trace.
        metrics: the run's :class:`~kvcache_sim.report.metrics.Metrics` ledger.
        coupled: whether prefill shares this host's decode compute timeline.
        simulate_decode / max_batch / profile / model: how this host models
            decode. They come from the run's wiring
            (:func:`kvcache_sim.workload._serving.serving_plane`), which is the
            same place the coordinator got them -- not read off the coordinator,
            which would be this host inspecting another service's fields.
    """

    def __init__(
        self,
        me: str,
        store: KVStore,
        coordinator: Coordinator,
        *,
        peers: Callable[[str], Any],
        trace,
        metrics: Metrics,
        coupled: bool = False,
        simulate_decode: bool = False,
        max_batch: int = 8,
        profile=DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
    ) -> None:
        self.me = me
        self.store = store
        self.coordinator: Coordinator = coordinator
        self.peers = peers
        self.trace = trace
        self.metrics = metrics
        self.coupled = coupled
        self.simulate_decode = simulate_decode
        # Cost constants, for the one thing this host has to price itself: a
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
    def _decode_state(self, finishes: List[float]) -> None:
        """Forward a changed decode batch. The engine reports here, not there."""
        self.coordinator.observe.broadcast(DecodeState(self.me, tuple(finishes)))

    def _compute_busy(self, until: float) -> None:
        """Forward this host's occupied compute timeline (coupled only)."""
        self.coordinator.observe.broadcast(ComputeBusy(self.me, until))

    async def drain(self) -> None:
        """Keep the loop running until this host's last decode token is emitted."""
        if self.engine is not None:
            await self.engine.drain()

    # -- the router role, which every host plays --------------------------- #
    async def receive(self, request: Request) -> None:
        """A client's request landed here. Find out where it belongs, and send it.

        The host a request arrives at is a load balancer's answer; the host that
        should *serve* it is a cluster-wide question about who holds the longest
        reusable prefix, which only the coordinator can answer. So every host asks,
        and most of the time forwards.
        """
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
        if plan.prefill == self.me:
            await self.serve(plan)
            return
        # Not ours: hand it to the host that should prefill it, and pay the hop.
        self.trace.record(
            self._now(), "FWD", f"{request.id} {self.me} -> {plan.prefill}"
        )
        await self.peers(plan.prefill).serve.call_one(plan)

    # -- the request lifecycle, on the host that owns it ------------------- #
    async def serve(self, plan: Plan) -> None:
        """Prefill ``plan``'s request here, then hand it to its decode host."""
        request = plan.request
        # The accepted plan reserved this host in control's predicted queue. When
        # coupled that reservation occupies the same compute the decode engine
        # steps on, so apply it there too -- immediately, with no await in
        # between, so no decode step can slip past it.
        if self.coupled and self.engine is not None:
            self.engine.reserve(plan.done_time)

        tbt = self.simulate_decode
        self._trace_route(plan)
        row = self._make_accepted(plan)
        if not tbt:
            self.metrics.add(row)
        else:
            self._pending[request.id] = row

        # (1) wait out the prefill queue at this host.
        if plan.queue_wait > 0:
            await asyncio.sleep(plan.queue_wait)
        # (2) the prefix this host already had is a read the store never sees,
        # so tell it: the volume evicts on what it has observed.
        local_blocks = plan.match_blocks - len(plan.pull_keys)
        if local_blocks:
            await self.store.reuse(
                self.me, list(plan.request.block_keys[:local_blocks])
            )
        # ...then pull the remote prefix (a real get_batch -> real fabric cost).
        prefill_t = plan.prefill_t
        if plan.reuse_source is not None and plan.pull_keys:
            try:
                await self.store.fetch(self.me, plan.pull_keys)
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
        # A cache fill is allowed to fail -- the request has already been served and
        # the only loss is that nobody reuses this prefix. Recorded rather than
        # dropped, because "cached" and "tried to cache and had no room" are exactly
        # the two outcomes a capacity sweep is measuring between, and a hit rate
        # cannot tell them apart.
        row.published = await self.store.publish(self.me, fresh)
        # (5) tell control the clock the real ops reached, and (coupled only) the
        # decode timeline this host now carries.
        now = self._now()
        busy_until = await self.coordinator.decide.call_one(
            PrefillFinished(self.me, now)
        )
        if self.coupled and self.engine is not None:
            # The reply, not a read of control's queue: prefill just occupied the
            # timeline decode steps on, and only the coordinator knows the tail.
            self.engine.reserve(busy_until)
        await self._prefill_done(plan, row.published)

    def _recompute(self, plan: Plan, row: RequestResult) -> float:
        """Re-price this prefill with the reuse that vanished, and say what it costs.

        The remote prefix is gone, so only what this host already held is still
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
            f"{plan.reuse_source} (evicted); recomputing on {self.me}",
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

    async def _prefill_done(self, plan: Plan, published: bool = True) -> None:
        note = "" if published else " -- NOT cached, no room on the volume"
        # What was published is what this host did not already have: the blocks it
        # pulled plus the suffix it computed, not the request's whole chain.
        stored = len(plan.request.block_keys) - (
            plan.match_blocks - len(plan.pull_keys)
        )
        if not self.simulate_decode:
            self.trace.record(
                self._now(),
                "DONE",
                f"{plan.request.id} prefill done on {self.me}"
                f" (published {stored}blk){note}",
            )
            return
        # Decode-simulating path: control decides whether decode can honour the
        # TBT SLO; the host performs (or skips) the admission.
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
                f" (TBT SLO; wasted prefill on {self.me}){note}",
            )
            return
        # Hand the request to the host that decodes it. Under disaggregation that
        # is always another host, so it is a hop and not a method call: this host
        # cannot reach into another's batch.
        if plan.decode == self.me:
            self._admit_locally(plan.request, self._pending.pop(plan.request.id))
        else:
            await self.peers(plan.decode).admit_decode.call_one(
                plan.request, self._pending.pop(plan.request.id)
            )
        self.trace.record(
            self._now(),
            "DONE",
            f"{plan.request.id} prefill done on {self.me}"
            f" (published {stored}blk){note}"
            f"; decoding on {plan.decode}",
        )

    async def admit_decode(self, request: Request, row: RequestResult) -> None:
        """A peer finished prefill and this host decodes it. Its outcome is ours now.

        The row travels with the request because the host that finishes a request
        is the host that reports it, and after this point that is this one.
        """
        self._admit_locally(request, row)

    def _admit_locally(self, request: Request, row: RequestResult) -> None:
        self._pending[request.id] = row
        if self.engine is not None:
            self.engine.admit(request)

    def _decode_done(self, request: Request, tbt: float) -> None:
        """Finalize a request once its last decode token is emitted."""
        result = self._pending.pop(request.id)
        result.tbt = tbt
        self.metrics.add(result)
        self.trace.record(
            self._now(), "DECODE", f"{request.id} decode done (tbt {tbt:.3f})"
        )
