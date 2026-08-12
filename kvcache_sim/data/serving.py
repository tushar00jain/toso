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
    C      -> client     "done" -- after the last token, which is what makes the
                         client's arrival-to-last-token stamp mean anything

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
2. if the plan pulls a remote prefix, drive a **real** ``get_batch`` (charging
   fabric via the cost model);
3. submit the forward pass to this host's accelerator, which runs it when the
   device is free -- so the queue wait is *waited*, not slept from a number, and
   what the request actually waited is recorded next to what control predicted;
4. publish what this host now holds and did not before -- a real ``put_batch``;
5. tell the coordinator the clock the real ops actually reached;
6. on the decode-simulating path, ask control whether the request may enter a
   decode batch, and answer the client with the host that will run it.

...and then, on that host:

7. fetch the request's KV out of the store (a **real** ``get_batch``, and under
   disaggregation the dominant cost of the request);
8. admit it to the decode batch, record its inter-token gaps when the last token
   lands -- and only then answer the client, which is what lets the client stamp
   arrival-to-last-token and what removed the drain hook the run used to need.

Note what moved in that list. The wait used to be step 2, in front of the fetch,
because that is where control's arithmetic puts it (it prices
``queue -> transfer -> prefill`` and reserves the instance for all three). A
forward pass cannot be submitted before its inputs have arrived, so the wait is
now behind the fetch, and the fetch runs on the fabric while the device belongs to
somebody else -- which is also why control's forecast now diverges from what
happens, most for the requests that pull.

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
deployment, not about the policy, so the host owns it -- by handing both engines
one accelerator or two, and by reporting each decode step's end onward as a
:class:`~kvcache_sim.control.scheduler.ComputeBusy` fact so the control plane's
*predicted* prefill queue tracks the device decode is actually using. A
disaggregated host reports nothing, and prefill never stalls decode.

The host used to do a third thing here, and it is worth naming because it is
gone. It pushed each accepted plan's predicted completion onto the accelerator
(``reserve``), so that a decode step would not be scheduled through a prefill that
had been promised but not yet started. A prefill now books its own slot when it is
submitted, on the very occupancy decode steps take, so the reservation had nothing
left to add and something to subtract: it held the device across a KV fetch that
does not use it, and it wrote control's estimate over a completion this host had
measured. Coupling is now entirely a matter of which object a host was handed.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

import torch

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
            :class:`~kvcache_sim.data._compute.Accelerator` is whether they
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
        # Nothing is reserved here, and there used to be two things. This method
        # opened by pushing ``plan.done_time`` -- control's *predicted* completion
        # -- onto the accelerator, so that a decode step sharing it would schedule
        # around a prefill the accelerator itself knew nothing about, and closed by
        # pushing control's corrected tail onto it again. Both are gone with the
        # queue: the accelerator books the pass when it is submitted, decode books
        # its steps on the same occupancy, and one object owns the answer. Applying
        # a forecast on top of that would be worse than redundant -- the first
        # reservation held a device across a fetch that does not use it, and the
        # second overwrote a completion this host had just performed with control's
        # estimate of it.
        self._trace_route(plan)
        row = self._make_accepted(plan)
        # Published straight away, before any of the work below. This host is the
        # only one that will ever know these fields, so there is nothing to wait
        # for: the row is amended in place as the facts land, and the decode host
        # amends the ledger's copy rather than being handed one.
        self.metrics.add(row)

        # (1) the prefix this host already had is a read the store never sees,
        # so tell it: the volume evicts on what it has observed.
        if plan.local_blocks:
            await self.store.reuse(
                self.me, list(plan.request.block_keys[:plan.local_blocks])
            )
        # ...then pull the remote prefix (a real get_batch -> real fabric cost).
        # The KV comes back, because this host is about to hold it: it goes into the
        # forward pass below (which attends over it) and out again in what that pass
        # answers with, which is what gets published.
        uncached = plan.uncached_tokens
        pulled: List[torch.Tensor] = []
        if plan.reuse_source is not None and plan.pull_keys:
            try:
                pulled = await self.store.fetch(self.me, plan.pull_keys)
            except KeyError:
                # The peer had those blocks when this was planned and does not now:
                # a volume it shares with other requests ran out of room and dropped
                # its coldest. Nothing is wrong -- a cache that cannot evict is not a
                # cache -- but this plan is stale, so recompute what was going to be
                # reused instead of failing the request. All of it: the pull is
                # all-or-nothing, and half a prefix is not a prefix.
                uncached = self._recompute(plan, row)
                pulled = []
        # (2) submit the forward pass for the uncached suffix. The engine is told
        # the work, not a duration: what it costs is the accelerator's answer, and
        # *when* it runs is the accelerator's answer too -- the pass waits here for
        # the device, behind whatever prefill or decode step already has it. It
        # answers with the KV: the prefix handed in, then the suffix it computed.
        # The request's id goes with it as the submission's name, which the
        # accelerator uses to break a same-instant tie in its service order.
        submitted_at = self._now()
        kv = await self.prefill_engine.run(uncached, pulled, tag=plan.request.id)
        # Whatever that took beyond the pass itself was queueing for the device.
        # This is the *measured* wait, and the row already carries control's
        # prediction of it: the two used to be the same number by construction,
        # because the wait was slept from the prediction. Derived rather than
        # reported back through the engine -- the pass costs exactly what the
        # accelerator says it costs, so the remainder is the wait, and asking the
        # port to hand a measurement back would put a simulation's bookkeeping in a
        # member a deployment implements.
        row.queue_wait = (
            self._now() - submitted_at - self.prefill_engine.cost(uncached)
        )

        # (3) publish what this host now holds and did not before: the prefix it
        # pulled, plus the suffix it computed. Which blocks those are is not a
        # decision and not control's to make -- it is everything past what was
        # already local, and the plan says how much that was. The store is handed
        # the KV itself, one tensor per key, rather than told how big a block is and
        # left to invent one: it moves bytes and computes none, so what a block is
        # is the accelerator's answer (see kvcache_sim/data/store.py).
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
        row.published = await self.store.publish(self.me, fresh, kv)
        # (4) tell control the clock the real ops actually reached.
        #
        # This is the only thing that closes the loop between control's model of
        # this host's queue and the queue, and it means far more than it used to:
        # when the data plane slept the forecast, ``now`` was that forecast coming
        # back and the correction was a rounding. Now it is the first news control
        # gets that its prediction was wrong, and the next request routed here is
        # priced off what happened.
        #
        # Awaited, and its answer -- control's corrected tail -- deliberately not
        # used. The wait is for the ordering: the decode admission asked a few lines
        # below must be decided by a coordinator that has already recorded this
        # completion, and a one-way report would not guarantee that under a
        # non-zero coordinator hop. The value itself is control's own model, and
        # this host has no use for a model of a completion it just performed.
        await self.coordinator.decide.call_one(PrefillFinished(self.me, self._now()))
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
            plan.local_blocks * self.prefill_engine.block_tokens,
            plan.request.prompt_tokens,
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
            predicted_queue_wait=plan.queue_wait,
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
        """The client brought a prefilled request here. Fetch its KV, decode, finish.

        Returns when the request's **last token** has been emitted, not when it
        entered the batch. It used to return at admission -- the step loop runs as
        its own task, so there was nothing stopping it -- and the shape that fell
        out of that was wrong twice over. The run needed a drain hook (a
        ``ServingHost.drain`` calling into the engine, wired through the harness)
        purely to keep the event loop alive for a tail no coroutine was holding;
        and because the client had already walked away, nothing was still on the
        request when it finished, so nothing could say how long the whole thing
        took. The request's own leg is the honest place to wait: a serving
        endpoint that answers before the answer exists is not answering.

        What that buys is the measurement this method's cost had been falling
        through. The ``get_batch`` below lands in neither headline column -- not
        TTFT, which is control's prediction made before any of it happens, and not
        TBT, which is measured between decode tokens while this finishes before
        the first of them -- so the dominant cost of a disaggregated deployment
        was charged on the clock and reported nowhere. It is inside
        arrival-to-last-token by construction, and the client stamps that once
        this returns (:mod:`kvcache_sim.workload._serving`).

        Waiting cannot deadlock, and the paths are worth naming because a client
        parked on a token that never comes would hang the whole run rather than
        fail it. A request refused at :meth:`route` or at the decode admission
        never reaches this method -- the client stops at the ``None``. A request
        with <= 1 output token has no decode step to run and is retired inside
        :meth:`~kvcache_sim.data._decode.DecodeEngine.admit`, on the clock instant
        it arrived. A request whose handoff found no KV still admits, because it
        still decodes. A queued request enters the batch as a slot frees, and the
        batch always frees slots because every member's ``remaining`` falls by one
        per step. And a host with no engine at all returns below without waiting,
        which is a run that does not model decode rather than a decode that was
        skipped.

        Under disaggregation none of this request's KV is on this host: another
        machine computed it and published it to the store, so getting it is a real
        ``get_batch`` over the request's whole block chain, charged fabric /
        storage / RAM by the same cost model as every other fetch. That is the cost
        a free ``admit_decode(request, row)`` method call used to hide, and in a
        prefill/decode-disaggregated system it is the dominant one.

        The whole chain rather than the suffix the prefill host published: a decode
        step attends over every token of the prompt, so what this host needs is
        every block, not the ones that happened to be new.

        The KV that comes back is what the handoff is *measured* off -- the bytes
        the transport just charged, added up off the tensors it returned. Beyond
        that this method does nothing with it, and a deployment would: it would load
        those blocks into the decode engine's paged cache and the batch would attend
        over them. What is missing to do that here is a decode engine that holds KV
        at all -- :class:`~kvcache_sim.data._decode.DecodeEngine` models a step's
        *duration* and never touches a tensor -- so the honest thing is to say that
        rather than to stash the list on an attribute nothing reads.

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
        drown, with the end-to-end column above as the place it does land in a
        latency.

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
        if plan.decode == plan.prefill:
            # Ours already. Tell the volume it was read, so its eviction ranking
            # sees a hit the store would otherwise never observe, and record a
            # handoff of nothing -- there was no transfer to make.
            await self.store.reuse(self.me, keys)
            self.metrics.handed_off(request.id, self.me, 0)
        else:
            try:
                kv = await self.store.fetch(self.me, keys)
            except KeyError:
                self.trace.record(
                    self._now(),
                    "NOKV",
                    f"{request.id} handoff from {plan.prefill} to {self.me} found "
                    f"no {len(keys)}blk chain in the store (evicted or never "
                    f"cached); decoding without charging a transfer",
                )
                self.metrics.handed_off(request.id, self.me, 0, missed=True)
            else:
                # Measured off the KV that arrived, not predicted from a block size
                # this host was told once: the same tensors the transport just
                # charged for are the ones counted here, so the reported handoff
                # cannot drift from the charged one. Not read back off the ledger
                # either -- that is the run's collector and mixes every host's
                # transfers together.
                nbytes = sum(b.numel() * b.element_size() for b in kv)
                self.trace.record(
                    self._now(),
                    "HANDOFF",
                    f"{request.id} {plan.prefill} -> {self.me} "
                    f"({len(keys)}blk, {nbytes}B of KV)",
                )
                self.metrics.handed_off(request.id, self.me, nbytes)
        # The one place this method can answer without the request having
        # finished, and it is not a decode that was skipped -- it is a run whose
        # decode side is not modelled reaching a host that has no engine. There is
        # no last token coming, so waiting for one would be waiting forever.
        if self.decode_engine is None:
            return
        await self.decode_engine.admit(request)

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
