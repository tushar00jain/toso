"""The serving loop: turning one routing decision into real store calls.

:class:`ServingHost` is **one serving instance**: its cache, its decode batch, its
compute. A deployment runs one per host, and a host knows about exactly two other
things: the store and the coordinator. It does not know about another host.

Three legs, and the client walks them
-------------------------------------
A request can land on any host, and the host it lands on is rarely the host that
should serve it: which instance holds the longest reusable prefix is a
cluster-wide fact. So a host that receives a request asks the coordinator where it
belongs (:meth:`ServingHost.route`) -- and then *answers with the address*. It does
not call the host it named. The client does::

    client -> A          "serve this"
    A      -> client     "prefill is B"          (a redirect, not a forward)
    client -> B          B prefills and publishes its KV blocks to the store
    B      -> client     "decode is C"           (another redirect)
    client -> C          C fetches that KV back out of the store, decodes, finishes

This used to be two host-to-host RPCs: ``A`` called ``B.serve(plan)`` and ``B``
called ``C.admit_decode(request, row)``. Both are gone, and with them the
``peers`` lookup every host was handed. Three things are better for it, and one
number gets worse -- honestly.

**A host holds no reference to another host.** There is no member on this class
that names a peer and nothing to construct after every host exists, which is what
the old ``peers`` callable was for: a lookup deferred purely so the wiring could
close a cycle it should never have had. A serving instance's whole outward surface
is now the store's client and the coordinator's port, and both of those are things
a deployment already gives it.

**Reporting state stops travelling.** ``admit_decode`` used to carry a
:class:`~kvcache_sim.report.metrics.RequestResult` -- half-filled measurement rows
handed from the host that prefilled to the host that decoded, because whoever
finished a request was expected to report all of it. That is not something one
process sends another; it is telemetry, and telemetry is joined at the collector.
So each host now records what *it* did (the prefill host: the routing decision,
the reuse, the publish; the decode host: the KV it pulled and the inter-token gaps
it produced) into the run's ledger, keyed by request id, and the join happens
there.

**The handoff becomes a real transfer.** This is the point of the change. Under
disaggregation the decode host has none of the request's KV -- another machine
computed it -- so it has to *get* it, and the only thing that knows where it is is
the store the prefill host published to. :meth:`ServingHost.decode` therefore
drives a real ``get_batch`` over the request's whole block chain, priced by the
same cost model as every other fetch. A method call that moved a Python object for
free is now bytes crossing a fabric, which is what a prefill/decode-disaggregated
system actually spends most of its time on. Disaggregation's headline number moves
against it as a result, and that is the more faithful answer: a dedicated decode
pool buys isolation from prefill and pays for it in KV transfer.

What is left over is which host a request arrives at, and that is a load
balancer's answer rather than a serving decision -- a client SDK, an ingress proxy
or DNS, none of which is part of the serving system and none of which survives
into a deployment of it. So it is not here: the run's wiring stands in for it
(:mod:`kvcache_sim.workload._serving`), and that same wiring is what follows the
redirects, because following a redirect is a client's job in every system that
issues one.

The lifecycle, once a host is prefilling:

1. ask the coordinator to route the request (control), and record a rejection if
   it refuses;
2. wait out the prefill queue;
3. if the plan pulls a remote prefix, drive a **real** ``get_batch`` (charging
   fabric via the cost model);
4. charge the prefill compute;
5. publish what this host now holds and did not before -- a real ``put_batch``;
6. tell the coordinator the clock the real ops actually reached;
7. on the decode-simulating path, ask control whether the request may enter a
   decode batch, and answer the client with the host that will run it.

...and then, on that host:

8. fetch the request's KV out of the store (a **real** ``get_batch``, and under
   disaggregation the dominant cost of the request);
9. admit it to the decode batch and record its inter-token gaps when the last
   token lands.

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
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

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

    Three members are the whole surface a client reaches: :meth:`route`,
    :meth:`prefill` and :meth:`decode`, in the order a request visits them. Two of
    the three answer with an *address* rather than doing the next thing themselves,
    which is what makes the set closed: nothing here needs a way to reach another
    host, so nothing here has one.

    Args:
        me: this host's instance id. Under the redirect model it is the only
            instance id this object ever holds -- a plan may name others, but they
            are addresses this host hands back to a client, not things it can call.
        store: the :class:`~kvcache_sim.data.store.KVStore` verbs. Also the *only*
            way KV reaches this host from another one: the prefill host publishes
            and the decode host fetches, with the store in between.
        coordinator: the control plane, through its
            :class:`~proposed.coordinator.Coordinator` port and nothing
            else. It decides; it never executes, and it is not on this host.
        trace: the run's shared trace.
        metrics: the run's :class:`~kvcache_sim.report.metrics.Metrics` ledger --
            the collector each host writes its own half of a request's story into.
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
        trace,
        metrics: Metrics,
        prefill: Optional[PrefillEngine] = None,
        decode: Optional[DecodeEngine] = None,
        models_decode: bool = False,
    ) -> None:
        self.me = me
        self.store = store
        self.coordinator: Coordinator = coordinator
        self.trace = trace
        self.metrics = metrics
        # The engines carry ``_engine`` in their names because the three *methods*
        # are named for the three legs of the redirect, and a host's decode leg and
        # a host's decode engine are different enough to be worth two words: a
        # prefill-only host still has a decode address to hand back, and never has
        # an engine.
        self.prefill_engine = prefill
        self.decode_engine = decode
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
        if self.decode_engine is not None:
            self.decode_engine.on_finish = self._decode_done
            self.decode_engine.on_state = self._decode_state
            # There is something to report only when a decode step can actually
            # collide with a prefill, and that is exactly when the two engines were
            # given the *same* accelerator. Identity, not a flag: the run's wiring
            # answers it by what it hands them, so there is nowhere for a second
            # answer to disagree.
            if self.coupled:
                self.decode_engine.on_compute_busy = self._compute_busy

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
        if self.decode_engine is not None:
            await self.decode_engine.drain()

    # -- leg 1: the router role, which every host plays -------------------- #
    async def route(self, request: Request) -> Optional[Plan]:
        """A client's request landed here. Answer with where it should actually go.

        The host a request arrives at is a load balancer's answer; the host that
        should *serve* it is a cluster-wide question about who holds the longest
        reusable prefix, which only the coordinator can answer. So every host asks
        -- and hands the answer back, rather than acting on it. A host that
        forwarded would be reaching into another host's queue on the strength of a
        decision neither of them made, and would need a reference to every peer to
        do it; a host that redirects needs nothing but the address it was given.

        Returns the :class:`~kvcache_sim.control.scheduler.Plan` for the client to
        take to :meth:`prefill` on ``plan.prefill``, or ``None`` when control
        refused the request outright. A refusal is recorded *here*, because this is
        where it happened and no other host will ever hear about this request.
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
            return None
        if plan.prefill != self.me:
            # Traced only when the answer moves the request, because that is the
            # fact worth having in the trace: a redirect back to the arrival host
            # is the same request in the same place, and the ROUTE line the prefill
            # host writes already records where it landed.
            self.trace.record(
                self._now(), "REDIR", f"{request.id} {self.me} -> {plan.prefill}"
            )
        return plan

    # -- leg 2: the request's prefill, on the host control chose ----------- #
    async def prefill(self, plan: Plan) -> Optional[str]:
        """Prefill ``plan``'s request here; answer with the host that decodes it.

        Returns the instance id the client should take this plan to next, or
        ``None`` when there is no next host -- either because this run does not
        model decode at all, or because control refused the decode admission, which
        makes this a *wasted* prefill and is recorded here as one.
        """
        # The accepted plan reserved this host in control's predicted queue. If a
        # decode engine shares this accelerator that reservation occupies the same
        # timeline its steps run on, so take it -- immediately, with no await in
        # between, so no step can slip past it.
        if self.coupled:
            self.prefill_engine.reserve(plan.done_time)

        self._trace_route(plan)
        row = self._make_accepted(plan)
        # Published straight away, before any of the work below. This host is the
        # only one that will ever know these fields, so there is nothing to wait
        # for: the row is amended in place as the facts land, and the decode host
        # amends the ledger's copy rather than being handed one.
        self.metrics.add(row)

        # (1) wait out the prefill queue at this host.
        await self.prefill_engine.wait_turn(plan.queue_wait)
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
        await self.prefill_engine.run(uncached)

        # (4) publish what this host now holds and did not before: the prefix it
        # pulled, plus the suffix it computed. Which blocks those are is not a
        # decision and not control's to make -- it is everything past what was
        # already local, and the plan says how much that was.
        #
        # Under disaggregation this is no longer only a cache fill for some later
        # request: it is how *this* request's KV reaches the host that will decode
        # it. The store is the handoff.
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
            self.prefill_engine.reserve(busy_until)
        return await self._prefill_done(plan, row)

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
        row.ttft += self.prefill_engine.cost(uncached) - plan.prefill_t
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

    async def _prefill_done(self, plan: Plan, row: RequestResult) -> Optional[str]:
        """Close out the prefill, and answer with the next address (or none)."""
        note = "" if row.published else " -- NOT cached, no room on the volume"
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
            return None
        # Decode-simulating path: control decides whether decode can honour the
        # TBT SLO; the host performs (or skips) the admission.
        if not await self.coordinator.decide.call_one(AdmitDecode(plan)):
            row.accepted = False
            row.decode_rejected = True
            row.wasted_prefill = True
            self.trace.record(
                self._now(),
                "REJECT",
                f"{plan.request.id} decode rejected on {plan.decode}"
                f" (TBT SLO; wasted prefill on {self.me}){note}",
            )
            return None
        self.trace.record(
            self._now(),
            "DONE",
            f"{plan.request.id} prefill done on {self.me}"
            f" (published {stored}blk){note}"
            f"; decoding on {plan.decode}",
        )
        # The address, not the act. The client goes there next; this host is done
        # with the request and has already recorded everything it knows about it.
        return plan.decode

    # -- leg 3: decode, on a host that has to go and get the KV ------------ #
    async def decode(self, plan: Plan) -> None:
        """The client brought a prefilled request here. Fetch its KV and decode it.

        Under disaggregation none of this request's KV is on this host: another
        machine computed it and published it to the store, so getting it is a real
        ``get_batch`` over the request's whole block chain, charged fabric /
        storage / RAM by the same cost model as every other fetch. That is the cost
        a free ``admit_decode(request, row)`` method call used to hide, and in a
        prefill/decode-disaggregated system it is the dominant one.

        The whole chain rather than the suffix the prefill host published: a decode
        step attends over every token of the prompt, so what this host needs is
        every block, not the ones that happened to be new.

        Unless it prefilled the request itself, which the plan says outright. Then
        the chain is already here -- this host computed it and published it -- and
        the store's own rule for a local hit applies: nothing moves and nothing is
        charged, because the instance has the blocks (see
        :meth:`~kvcache_sim.data.store.KVStore.reuse`). Fetching it back would
        charge a storage read and a memory copy for KV that never left, and would
        report it as a handoff, which is the opposite of what the column means.
        That is not a rare corner: with prefill and decode drawn from one pool it
        is most of the traffic, and it was 73% of this scenario's reported handoff
        bytes before the plan was consulted.

        Whether a host that did *not* prefill nonetheless holds some of the chain
        (from another request sharing the prefix) is not asked, and could be: the
        control plane already computes per-instance prefix runs and could price the
        decode side's local match the way it prices the prefill side's. Until it
        does, a cross-host handoff pays for the whole chain, which over-charges by
        whatever the decode host happened to share.

        The fetch finishes **before** the request enters the decode batch, and that
        placement is a decision. Charging it as an inter-token gap instead -- the
        first token came from the prefill host, so arguably the transfer sits
        between token one and token two -- was tried and rejected: it is not how
        prefill/decode-disaggregated systems are measured (DistServe and Mooncake
        both put KV migration in TTFT and leave TPOT/TBT for the decode cadence),
        and on this workload it collapses the whole scenario, taking *both* the
        disaggregated and the coupled column to 0.0% attainment against a target
        set at five decode steps. A one-off migration swamping a per-token metric
        says nothing about per-token behaviour. So the handoff is charged on the
        clock -- it delays this request and everything queued behind it -- and it is
        reported as its own quantity rather than folded into a column it would
        drown. What that leaves genuinely missing is an end-to-end latency
        measurement; the README says so.

        A missing block is possible and is not fatal here. The publish is allowed to
        fail (a full volume) and a volume may drop a block between the publish and
        this fetch, and ``get_batch`` is all-or-nothing, so either shows up as a
        ``KeyError`` over the whole batch. In this model the store holds the only
        copy, so a truthful answer would be that the request dies -- but re-deriving
        KV on a decode-only host is a whole second serving path, and inventing one
        to cover a capacity misconfiguration would be worse than saying so. The
        request decodes, the transfer is recorded as not having happened, and
        :attr:`~kvcache_sim.report.metrics.Metrics.handoff_misses` is the number
        that tells a run its decode pool is being fed by a cache too small to hold
        the handoff.
        """
        request = plan.request
        keys = list(request.block_keys)
        # What the transport will move, from the one definition it charges against:
        # one carrier per block, each the size the model predicts. Derived rather
        # than read back off the ledger, which is the run's collector and mixes
        # every host's transfers together.
        nbytes = len(keys) * self.store.block_nbytes
        if plan.decode == plan.prefill:
            # Ours already. Tell the volume it was read, so its eviction ranking
            # sees a hit the store would otherwise never observe, and record a
            # handoff of nothing -- there was no transfer to make.
            await self.store.reuse(self.me, keys)
            self.metrics.handed_off(request.id, self.me, 0)
            if self.decode_engine is not None:
                self.decode_engine.admit(request)
            return
        try:
            await self.store.fetch(self.me, keys)
        except KeyError:
            self.trace.record(
                self._now(),
                "NOKV",
                f"{request.id} handoff from {plan.prefill} to {self.me} found no "
                f"{len(keys)}blk chain in the store (evicted or never cached); "
                f"decoding without charging a transfer",
            )
            self.metrics.handed_off(request.id, self.me, 0, missed=True)
        else:
            self.trace.record(
                self._now(),
                "HANDOFF",
                f"{request.id} {plan.prefill} -> {self.me} "
                f"({len(keys)}blk, {nbytes}B of KV)",
            )
            self.metrics.handed_off(request.id, self.me, nbytes)
        if self.decode_engine is not None:
            self.decode_engine.admit(request)

    def _decode_done(self, request: Request, tbt: float) -> None:
        """Finalize a request once its last decode token is emitted.

        The gaps are this host's measurement of its own batch, so this host records
        them -- into the ledger's row, not into a row somebody handed over. The
        prefill host wrote the rest of that row before it ever named this one.
        """
        self.metrics.decoded(request.id, tbt)
        self.trace.record(
            self._now(), "DECODE", f"{request.id} decode done (tbt {tbt:.3f})"
        )
