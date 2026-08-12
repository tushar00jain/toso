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

from proposed import Coordinator

from ..control.scheduler import (
    AdmitDecode, ComputeBusy, DecodeState, Plan, PrefillFinished, Route,
)
from ..report.metrics import Metrics, RequestResult
from ..control.request import Request
from ._decode import DecodeEngine
from ._prefill import PrefillEngine
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
        prefill: this host's :class:`~kvcache_sim.data._prefill.PrefillEngine`,
            or ``None`` if it does not prefill.
        decode: this host's :class:`~kvcache_sim.data._decode.DecodeEngine`, or
            ``None`` if it does not decode. Whether the two were handed the *same*
            :class:`~kvcache_sim.data._compute.ComputeTimeline` is whether they
            contend -- see :attr:`coupled`.
        models_decode: whether this *run* models the request's second half at all.
            Not the same question as whether this host decodes: a prefill-only host
            in a disaggregated run has no decode engine, and its requests still go
            on to decode somewhere else.
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
        prefill: Optional[PrefillEngine] = None,
        decode: Optional[DecodeEngine] = None,
        models_decode: bool = False,
    ) -> None:
        self.me = me
        self.store = store
        self.coordinator: Coordinator = coordinator
        self.peers = peers
        self.trace = trace
        self.metrics = metrics
        self.prefill = prefill
        self.decode = decode
        self.models_decode = models_decode
        self.block_tokens = store.block_tokens
        #: Whether a prefill here delays a decode step here -- true exactly when
        #: both engines run on one accelerator. A run may model two engines on one
        #: host as *not* contending, which is a simplification rather than a
        #: deployment: the wiring says so by handing them separate timelines.
        self.coupled = (
            prefill is not None
            and decode is not None
            and prefill.compute is decode.compute
        )
        # On the decode-simulating path acceptance is provisional until the request
        # is admitted to a decode batch (it may still be shed on the TBT SLO). We
        # stash the accepted row here and publish it once decode finishes (success)
        # or admission is refused (wasted prefill).
        self._pending: Dict[str, RequestResult] = {}
        if self.decode is not None:
            self.decode.on_finish = self._decode_done
            self.decode.on_state = self._decode_state
            # There is something to report only when a decode step can actually
            # collide with a prefill, and that is exactly when the two engines were
            # given the *same* accelerator. Identity, not a flag: the run's wiring
            # answers it by what it hands them, so there is nowhere for a second
            # answer to disagree.
            if self.coupled:
                self.decode.on_compute_busy = self._compute_busy

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
        if self.decode is not None:
            await self.decode.drain()

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
        # The accepted plan reserved this host in control's predicted queue. If a
        # decode engine shares this accelerator that reservation occupies the same
        # timeline its steps run on, so take it -- immediately, with no await in
        # between, so no step can slip past it.
        if self.coupled:
            self.prefill.reserve(plan.done_time)

        tbt = self.models_decode
        self._trace_route(plan)
        row = self._make_accepted(plan)
        if not tbt:
            self.metrics.add(row)
        else:
            self._pending[request.id] = row

        # (1) wait out the prefill queue at this host.
        await self.prefill.wait_turn(plan.queue_wait)
        # (2) the prefix this host already had is a read the store never sees,
        # so tell it: the volume evicts on what it has observed.
        if plan.local_blocks:
            await self.store.reuse(
                self.me, list(plan.request.block_keys[:plan.local_blocks])
            )
        # ...then pull the remote prefix (a real get_batch -> real fabric cost).
        uncached = plan.uncached_tokens
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
                uncached = self._recompute(plan, row)
        # (3) charge the prefill compute for the uncached suffix. The engine is
        # told the work, not a duration: what it costs is the accelerator's answer.
        await self.prefill.run(uncached)

        # (4) publish what this host now holds and did not before: the prefix it
        # pulled, plus the suffix it computed. Which blocks those are is not a
        # decision and not control's to make -- it is everything past what was
        # already local, and the plan says how much that was.
        fresh = list(plan.request.block_keys[plan.local_blocks:])
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
        if self.coupled:
            # The reply, not a read of control's queue: prefill just occupied the
            # timeline decode steps on, and only the coordinator knows the tail.
            self.prefill.reserve(busy_until)
        await self._prefill_done(plan, row.published)

    def _recompute(self, plan: Plan, row: RequestResult) -> int:
        """Re-price this prefill with the reuse that vanished; answer what is left.

        The remote prefix is gone, so only what this host already held is still
        cached: the planned match minus the blocks that were going to be pulled.
        Corrects the row too -- the request really did compute those tokens, and a
        hit rate that counted the plan rather than the outcome would flatter the
        cache that dropped them.
        """
        cached = min(
            plan.local_blocks * self.block_tokens, plan.request.prompt_tokens
        )
        uncached = plan.request.prompt_tokens - cached
        row.cached_tokens = cached
        row.uncached_tokens = uncached
        row.transfer_bytes = 0
        row.ttft += self.prefill.cost(uncached) - plan.prefill_t
        self.trace.record(
            self._now(),
            "RESTALE",
            f"{plan.request.id} lost {len(plan.pull_keys)}blk of reuse on "
            f"{plan.reuse_source} (evicted); recomputing on {self.me}",
        )
        return uncached

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
        stored = len(plan.request.block_keys) - plan.local_blocks
        if not self.models_decode:
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
        # cannot reach into another's batch. Ourselves is the same operation with
        # nothing in between, so it is the same member, called directly.
        row = self._pending.pop(plan.request.id)
        if plan.decode == self.me:
            await self.admit_decode(plan.request, row)
        else:
            await self.peers(plan.decode).admit_decode.call_one(plan.request, row)
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
        self._pending[request.id] = row
        if self.decode is not None:
            self.decode.admit(request)

    def _decode_done(self, request: Request, tbt: float) -> None:
        """Finalize a request once its last decode token is emitted."""
        result = self._pending.pop(request.id)
        result.tbt = tbt
        self.metrics.add(result)
        self.trace.record(
            self._now(), "DECODE", f"{request.id} decode done (tbt {tbt:.3f})"
        )
