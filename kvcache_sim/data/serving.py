"""The serving loop: turning one routing decision into real store calls.

:class:`ServingHost` is **one serving instance**: its cache, its decode batch, its
compute. A deployment runs one per host, and a host knows about exactly two other
things: the store and its control plane. It does not know about another host.

Three legs, and the client walks them
-------------------------------------
A request can land on any host, and the host it lands on is rarely the host that
should serve it: which instance holds the longest reusable prefix is a
cluster-wide fact. So a host that receives a request asks control where it
belongs (:meth:`ServingHost.route`) -- and then *answers with the address*. It does
not call the host it named. The client does::

    client -> A          "serve this" (a prompt)
    A      -> client     "prefill is B"          (a redirect, not a forward)
    client -> B          B prefills and publishes its KV blocks to the store
    B      -> client     the first token, and "decode is C"   (another redirect)
    client -> C          C fetches that KV back out of the store, decodes, finishes
    C      -> client     the remaining tokens -- after the last one, which is what
                         makes the client's arrival-to-last-token stamp mean
                         anything

Both ends of that carry tensors, and the split follows the engines' division of
labour: TTFT is the time to the *first* token, which is sampled from the prefill's
last position, so :meth:`prefill` answers with it; everything after it is the decode
batch's, so :meth:`decode` answers with the rest. The client is the only participant
that sees both halves, and therefore the only one that can time the request.

Consequences of the redirect model:

* **A host holds no reference to another host.** A serving instance's whole outward
  surface is the store's client and the two control ports below.
* **Reporting state does not travel.** Each host records what *it* did into the
  run's ledger, keyed by request id (the prefill host: the routing decision, the
  reuse, the publish; the decode host: the KV it pulled and the inter-token gaps),
  and the join happens at the collector.
* **The handoff is a real transfer.** Under disaggregation the decode host has none
  of the request's KV, so :meth:`ServingHost.decode` drives a real ``get_batch``
  over the whole block chain, priced by the same cost model as every other fetch.
  That is what a prefill/decode-disaggregated system spends most of its time on,
  and it moves disaggregation's headline number against it: a dedicated decode pool
  buys isolation from prefill and pays for it in KV transfer.

Which host a request *arrives* at is a load balancer's answer -- a client SDK, an
ingress proxy, DNS -- so it is not here. The run's wiring stands in for it
(:mod:`kvcache_sim.workload._serving`), and that same wiring follows the redirects.

The lifecycle, once a host is prefilling:

1. ask control to route the request, and record a rejection if it refuses;
2. if the plan pulls a remote prefix, drive a **real** ``get_batch`` (charging
   fabric via the cost model);
3. submit the forward pass to this host's accelerator, which runs it when the
   device is free -- so the queue wait is *waited*, not slept from a number, and
   what the request actually waited is recorded next to what control predicted;
4. publish what this host now holds and did not before -- a real ``put_batch``;
5. tell control's sensor the clock the real ops actually reached;
6. answer the client with the first token and, where the run models decode, the
   address of the host the plan named to run the rest.

...and then, on that host:

7. fetch the request's KV out of the store (a **real** ``get_batch``, and under
   disaggregation the dominant cost of the request), and **publish it here**,
   because the bytes are now on this host and this host's volume has to know;
8. admit it to the decode batch, record its inter-token gaps when the last token
   lands, publish the KV the generation left behind -- and only then answer the
   client, with the tokens the batch generated. That is what lets the client stamp
   arrival-to-last-token and what removed the drain hook the run used to need.

A decode host holds KV, so a decode host pays for it
----------------------------------------------------
Steps 7 and 8 close the same hole. Everywhere in this model, KV that lands on a
host is registered on that host's volume: a prefill host publishes the suffix it
computed *and* the prefix it pulled, because it holds both. The decode host pulls
an entire block chain in and generates more KV on top of it, so it publishes both
too, through the same :meth:`~kvcache_sim.data._store.KVStore.publish` -- evictable
by the same LRU, refusable by the same bounded volume. Otherwise a decode host has
unbounded free memory and no capacity, eviction or hit-rate number ever feels it.

Two consequences, both intended. The decode host becomes a **replica**: the
directory maps the chain to the prefill host *and* the decode host, so a later
request can be routed to, or pull from, a host that only ever decoded that prefix
-- which is what a read-through cache is. And decode **competes** for the volume
with the cached prefixes on it: on a bounded instance a decode pool that holds
every chain it has served will evict the prefixes prefill is trying to reuse.

Note where the queue wait sits in that list. Control's arithmetic puts it first (it
prices ``queue -> transfer -> prefill`` and reserves the instance for all three),
but a forward pass cannot be submitted before its inputs have arrived, so here it
is *behind* the fetch and the fetch runs on the fabric while the device belongs to
somebody else. That is why control's forecast diverges from what happens, most for
the requests that pull.

Simplification (documented in SPEC): a block becomes reusable at prefill
*completion*, not while in flight, so two requests racing for the same brand-new
prefix may both compute it. Arrivals are typically spaced enough that this is rare.

Control is not on this host, and it is two ports
-----------------------------------------------
Control runs as a service, so this plane reaches it the way it reaches the store:
through a port, over calls that carry values. There are two ports, and the split is
between asking and telling:

* :class:`~proposed.plane.ControlPlane` -- the **questions**: where should this run
  (``decide``, here), and which peer serves a fetch (``sources``, asked by
  :mod:`kvcache_sim.data._store` where the fetch is). One plane answers both, and a
  question is answered, so it is called and waited for. ``decide``'s answer is a
  :class:`~kvcache_sim.control.scheduler.Response` -- a value naming both of the
  request's hosts, and what prefilling on the first was priced at -- which this plane
  carries to each leg beside the request it already holds;
* :class:`~proposed.deployment.NotifiedSensor` -- the **facts**: this host's decode
  batch, its busy compute, the clock its prefill really reached. Nothing comes
  back, and the reply is waited for anyway, because the next question has to be
  decided against a sensor that has already folded the fact.

Those two are the *only* things this module may touch on the control side --
``check_structure.py`` rule 6 fails the build on a field read, a subscript or a
``getattr`` through either, since none of those survive the planes being in
different processes.

So this plane owns the decode engine and *reports* it: every batch change goes out
as a :class:`~kvcache_sim.control.scheduler.DecodeState` fact. The engine's
callbacks come back here first, to their owner on this host, and this plane decides
what to send on.

Coupling lives here
-------------------
Whether prefill and decode contend for this host's compute is a fact about the
deployment, not the selector, so the host owns it: by handing both engines one
accelerator or two, and by reporting each decode step's end onward as a
:class:`~kvcache_sim.control.scheduler.ComputeBusy` fact so control's *predicted*
prefill queue tracks the device decode is actually using. A disaggregated host
reports nothing, and prefill never stalls decode.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import torch

from proposed import ControlPlane, DataPlane, Deployment, NotifiedSensor

from ..control.scheduler import (
    ComputeBusy, DecodeState, Plan, PrefillFinished, Response,
)
from ..report.metrics import Metrics, RequestResult
from ..control.request import Request
from ._decode import DecodeEngine
from ._prefill import PrefillEngine
from ._store import KVStore

__all__ = ["ServingHost"]


class ServingHost(DataPlane):
    """One serving instance: its cache, its decode batch, its compute.

    The running loop's ``time()`` is the only clock (virtual under simulation).

    Three members are the whole surface a client reaches: :meth:`route`,
    :meth:`prefill` and :meth:`decode`, in the order a request visits them. Two of
    the three answer with an *address* rather than doing the next thing themselves,
    so nothing here needs a way to reach another host.

    A :class:`~proposed.plane.DataPlane`, so the three things it reaches -- the
    store, the control plane it asks, the sensor it reports into -- arrive together
    at :meth:`attach` off the one deployment that has them all, rather than being
    plumbed in by whoever builds the hosts.

    Args:
        me: this host's instance id -- the only one this object holds. A plan may
            name others, but they are addresses to hand back to a client.
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
        *,
        trace,
        metrics: Metrics,
        prefill: Optional[PrefillEngine] = None,
        decode: Optional[DecodeEngine] = None,
        models_decode: bool = False,
    ) -> None:
        self.me = me
        # Filled by attach(): none of the three exists before the deployment does.
        self.store: Optional[KVStore] = None
        self.control: Optional[ControlPlane] = None
        self.cluster: Optional[NotifiedSensor] = None
        self.trace = trace
        self.metrics = metrics
        self.prefill_engine = prefill
        self.decode_engine = decode
        self.models_decode = models_decode
        #: Whether a prefill here delays a decode step here -- true exactly when
        #: both engines run on one accelerator. A run that models two engines on
        #: one host as *not* contending says so by handing them separate timelines.
        self.coupled = (
            prefill is not None
            and decode is not None
            and prefill.compute is decode.compute
        )
        if self.decode_engine is not None:
            self.decode_engine.on_finish = self._decode_done
            self.decode_engine.on_state = self._decode_state
            # Only worth reporting when a decode step can actually collide with a
            # prefill, i.e. when both engines were given the same accelerator.
            if self.coupled:
                self.decode_engine.on_compute_busy = self._compute_busy

    # -- proposed.DataPlane: the deployment hands over what this host reaches -- #
    def attach(self, deployment: Deployment) -> None:
        """Take the store, the control plane and the sensor off ``deployment``.

        One object rather than three arguments: a host reaching its control plane is
        reaching *this deployment's* control plane, so whoever builds the hosts has
        nothing to plumb and cannot pair a host with the wrong run's services.

        The store is built here, over that same deployment, because
        :class:`~kvcache_sim.data._store.KVStore` is the KV-shaped reading of the
        store's verbs and holds nothing else -- a host per process builds its own,
        and so does each of a simulation's.
        """
        self.store = KVStore(deployment)
        self.control = deployment.control_plane_handle
        self.cluster = deployment.sensor_handle

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    # -- what this host tells control's sensor about its decode side -------- #
    async def _decode_state(self, finishes: List[float]) -> None:
        """Forward a changed decode batch. The engine reports here, not there."""
        await self.cluster.notify.call_one(DecodeState(self.me, tuple(finishes)))

    async def _compute_busy(self, until: float) -> None:
        """Forward this host's occupied compute timeline (coupled only)."""
        await self.cluster.notify.call_one(ComputeBusy(self.me, until))

    # -- leg 1: the router role, which every host plays -------------------- #
    async def route(self, request: Request) -> Optional[Response]:
        """A client's request landed here. Answer with where it should actually go.

        Which host should *serve* the request is a cluster-wide question about who
        holds the longest reusable prefix, so every host asks control -- and hands
        the answer back rather than acting on it.

        Returns control's :class:`~kvcache_sim.control.scheduler.Response` for the
        client to take to :meth:`prefill` on ``response.prefill``, or ``None`` when
        control refused the request outright. A refusal is recorded *here*: no other
        host will ever hear about this request.

        The request itself is not in that answer and does not need to be: the client
        holds it already and hands it to each leg beside the decision.
        """
        response = await self.control.decide.call_one(request, self.me)
        if response is None:
            self.trace.record(
                self._now(), "REJECT", f"{request.id} rejected (SLO/overload)"
            )
            self.metrics.add(
                RequestResult(
                    id=request.id, accepted=False, prompt_tokens=request.prompt_tokens
                )
            )
            return None
        if response.prefill != self.me:
            # Traced only when the answer moves the request; the prefill host's own
            # ROUTE line already records where it landed.
            self.trace.record(
                self._now(), "REDIR", f"{request.id} {self.me} -> {response.prefill}"
            )
        return response

    # -- leg 2: the request's prefill, on the host control chose ----------- #
    async def prefill(
        self, request: Request, response: Response
    ) -> Tuple[Optional[str], torch.Tensor]:
        """Prefill ``request`` here as ``response`` priced it; answer with its first
        token and where next.

        Returns ``(next host, first token)``: the instance id the client should take
        this plan to, or ``None`` when there is no next host, which is a run that
        does not model decode at all.

        The address is optional and the token is not: every request that reaches
        this method gets prefilled, and a request that got this far was admitted
        before any of it ran.
        """
        # Nothing is reserved on the accelerator here: it books the pass when it is
        # submitted and decode books its steps on the same occupancy, so one object
        # owns the answer and no forecast is written over it.
        plan = response.plan
        self._trace_route(request, response)
        row = self._make_accepted(request, response)
        # Published before the work below: this host is the only one that will know
        # these fields, and the row is amended in place as the facts land.
        self.metrics.add(row)

        # (1) the prefix this host already had is a read the store never sees,
        # so tell it: the volume evicts on what it has observed.
        if plan.local_blocks:
            await self.store.reuse(
                self.me, list(request.block_keys[:plan.local_blocks])
            )
        # ...then pull the remote prefix (a real get_batch -> real fabric cost). The
        # KV comes back because this host is about to hold it: it goes into the
        # forward pass below and out again in what that pass publishes.
        uncached = plan.uncached_tokens
        pulled: List[torch.Tensor] = []
        if plan.reuse_source is not None and plan.pull_keys:
            try:
                pulled = await self.store.fetch(self.me, plan.pull_keys)
            except KeyError:
                # The peer dropped those blocks between routing and now (its volume
                # ran out of room). The plan is stale, so recompute the whole
                # planned reuse rather than fail: a pull is all-or-nothing, and half
                # a prefix is not a prefix.
                uncached = self._recompute(request, plan, row)
                pulled = []
        # (2) submit the forward pass for the uncached suffix. The engine is told
        # the work, not a duration: what it costs and *when* it runs are both the
        # accelerator's answer, so the pass waits here for the device behind
        # whatever prefill or decode step already has it. It answers with the KV
        # (the prefix handed in, then the suffix it computed) and the first token.
        # The request's id tags the submission, breaking same-instant ties in the
        # accelerator's service order.
        #
        # Sliced by *offset* off the request's own prompt, not as ``prompt[-n:]``,
        # which would hand a fully-cached request its entire prompt back.
        prompt = request.prompt[request.prompt_tokens - uncached:]
        submitted_at = self._now()
        kv, first_token = await self.prefill_engine.run(
            prompt, pulled, tag=request.id
        )
        # Whatever that took beyond the pass itself was queueing for the device: the
        # *measured* wait, next to control's prediction of it already on the row.
        # Derived rather than reported back through the engine, which would put a
        # simulation's bookkeeping in a member a deployment implements.
        row.queue_wait = (
            self._now() - submitted_at - self.prefill_engine.cost(uncached)
        )

        # (3) publish what this host now holds and did not before: the prefix it
        # pulled, plus the suffix it computed -- everything past what was already
        # local. Under disaggregation this is not only a cache fill for some later
        # request; it is how *this* request's KV reaches its decode host.
        fresh = list(request.block_keys[plan.local_blocks:])
        # A cache fill may fail: the request is already served and the only loss is
        # that nobody reuses this prefix. Recorded rather than dropped -- "cached"
        # and "had no room" are the two outcomes a capacity sweep exists to tell
        # apart, and a hit rate cannot.
        row.published = await self.store.publish(self.me, fresh, kv)
        # (4) tell control's sensor the clock the real ops actually reached -- the
        # only thing that closes the loop between the predicted queue for this host
        # and the queue, and the first news control gets that it was wrong.
        #
        # Awaited for the ordering, not the answer, which carries nothing: the next
        # request routed against this host must be priced against a sensor that has
        # already folded this completion, and a report left in flight leaves control
        # answering off a queue it knows to be wrong.
        await self.cluster.notify.call_one(PrefillFinished(self.me, self._now()))
        return await self._prefill_done(request, response, row), first_token

    def _recompute(self, request: Request, plan: Plan, row: RequestResult) -> int:
        """Re-price this prefill with the reuse that vanished; answer what is left.

        The remote prefix is gone, so only what this host already held is still
        cached. Corrects the row too: the request really did compute those tokens,
        and a hit rate counting the plan rather than the outcome would flatter the
        cache that dropped them.
        """
        cached = min(
            plan.local_blocks * self.prefill_engine.block_tokens,
            request.prompt_tokens,
        )
        uncached = request.prompt_tokens - cached
        row.cached_tokens = cached
        row.uncached_tokens = uncached
        row.transfer_bytes = 0
        row.ttft += self.prefill_engine.cost(uncached) - plan.prefill_t
        self.trace.record(
            self._now(),
            "RESTALE",
            f"{request.id} lost {len(plan.pull_keys)}blk of reuse on "
            f"{plan.reuse_source} (evicted); recomputing on {self.me}",
        )
        return uncached

    # -- outcome bookkeeping ---------------------------------------------- #
    def _make_accepted(
        self, request: Request, response: Response
    ) -> RequestResult:
        plan = response.plan
        return RequestResult(
            id=request.id,
            accepted=True,
            ttft=plan.ttft,
            prompt_tokens=request.prompt_tokens,
            cached_tokens=plan.cached_tokens,
            uncached_tokens=plan.uncached_tokens,
            transfer_bytes=plan.transfer_bytes,
            prefill=response.prefill,
            reuse_source=plan.reuse_source,
            predicted_queue_wait=plan.queue_wait,
        )

    def _trace_route(self, request: Request, response: Response) -> None:
        plan = response.plan
        if plan.reuse_source is not None:
            src = f" pull {plan.match_blocks}blk from {plan.reuse_source}"
        elif plan.match_blocks:
            src = f" local hit {plan.match_blocks}blk"
        else:
            src = " cold (no reuse)"
        self.trace.record(
            self._now(),
            "ROUTE",
            f"{request.id} -> {response.prefill}"
            f" (match {plan.match_blocks}blk,{src}, "
            f"compute {plan.uncached_tokens}tok, ttft {plan.ttft:.3f})",
        )

    async def _prefill_done(
        self, request: Request, response: Response, row: RequestResult
    ) -> Optional[str]:
        """Close out the prefill, and answer with the next address (or none)."""
        note = "" if row.published else " -- NOT cached, no room on the volume"
        # What was published: the blocks pulled plus the suffix computed, not the
        # request's whole chain.
        stored = len(request.block_keys) - response.plan.local_blocks
        if not self.models_decode:
            self.trace.record(
                self._now(),
                "DONE",
                f"{request.id} prefill done on {self.me}"
                f" (published {stored}blk){note}",
            )
            return None
        # Decode-simulating path: nothing is asked here. The decision already names
        # the decode host, and the SLO that could have refused this request was
        # decided against before its prefill ran.
        self.trace.record(
            self._now(),
            "DONE",
            f"{request.id} prefill done on {self.me}"
            f" (published {stored}blk){note}"
            f"; decoding on {response.decode}",
        )
        # The address, not the act: the client goes there next.
        return response.decode

    # -- leg 3: decode, on a host that has to go and get the KV ------------ #
    async def decode(
        self, request: Request, response: Response
    ) -> List[torch.Tensor]:
        """The client brought a prefilled request here. Fetch its KV, decode, finish.

        Answers with **the tokens this host generated** -- the request's output
        minus the first, which the prefill host produced. Not a stream: the tokens
        arrive together at the end, which is the ``stream=False`` shape of a serving
        API and matches a leg that already returns at the last token so the client
        can stamp the request end to end.

        Returns when the **last token** has been emitted, not at admission. That is
        what puts this method's cost inside a measured latency: the ``get_batch``
        below lands in neither headline column -- not TTFT, which control predicted
        before any of it happens, and not TBT, which is measured between decode
        tokens while this finishes before the first of them -- so without the wait
        the dominant cost of a disaggregated deployment is charged on the clock and
        reported nowhere. It is inside arrival-to-last-token by construction
        (:mod:`kvcache_sim.workload._serving`).

        Waiting cannot deadlock. A request refused at :meth:`route` or at the decode
        admission never reaches here; one with <= 1 output token is retired inside
        :meth:`~kvcache_sim.data._decode.DecodeEngine.admit` on the instant it
        arrived; one whose handoff found no KV still admits; a queued request enters
        the batch as a slot frees, and slots always free because every member's
        ``remaining`` falls by one per step; and a host with no engine returns below
        without waiting.

        The handoff fetch
        -----------------
        Under disaggregation none of this request's KV is on this host, so getting
        it is a real ``get_batch`` over the request's **whole block chain** (a
        decode step attends over every token of the prompt, not just the blocks the
        prefill host happened to publish), charged fabric / storage / RAM by the
        same cost model as every other fetch.

        **Unless this host prefilled the request**, which the decision says outright.
        Then the chain is already here and the store's local-hit rule applies:
        nothing moves and nothing is charged
        (:meth:`~kvcache_sim.data._store.KVStore.reuse`). Fetching it back would
        charge a storage read for KV that never left and report it as a handoff.
        Not a rare corner -- with prefill and decode drawn from one pool it was 73%
        of this scenario's reported handoff bytes before the plan was consulted.

        Whether a host that did *not* prefill nonetheless holds part of the chain
        (from another request sharing the prefix) is not asked. Until it is, a
        cross-host handoff pays for the whole chain, over-charging by whatever the
        decode host happened to share.

        The fetch completes **before** the request enters the batch. Charging it as
        an inter-token gap instead was tried and rejected: it is not how these
        systems are measured (DistServe and Mooncake both put KV migration in TTFT
        and leave TPOT/TBT for the decode cadence), and on this workload it takes
        *both* the disaggregated and the coupled column to 0.0% attainment against a
        target set at five decode steps.

        A missing block is not fatal. The publish may have failed on a full volume,
        or a volume may have dropped a block since, and ``get_batch`` is
        all-or-nothing, so either surfaces as a ``KeyError`` over the whole batch.
        The request decodes anyway, the transfer is recorded as not having happened,
        and :attr:`~kvcache_sim.report.metrics.Metrics.handoff_misses` is what tells
        a run its decode pool is fed by a cache too small to hold the handoff.
        Re-deriving KV on a decode-only host would be a second serving path, which
        is not worth inventing to cover a capacity misconfiguration.

        What this host then holds
        -------------------------
        The KV that arrives is published **on this host** (:meth:`_reside`), the
        same rule the prefill leg follows for the prefix it pulled. A ``get_batch``
        delivers bytes and stores nothing, so without this a decode host attends
        over a chain its own volume has never heard of: no capacity consumed, no
        directory entry, no eviction pressure. Publishing makes the directory say
        two hosts hold the chain, which is true, and control routes on it.

        Kept after the request finishes rather than dropped. Both models are
        defensible, but keeping them is what this model does with every other block
        it holds -- nothing has a lifetime, the volume's LRU decides -- and a decode
        host that just served a conversation's turn is exactly the host that should
        still have that prefix when the next turn arrives.

        The last thing before answering is publishing the KV the batch **generated**
        (``ceil(n / block_tokens)`` blocks for ``n`` generated tokens,
        :meth:`~kvcache_sim.data._compute.Accelerator.generated_kv`, under keys
        continuing the prompt's chain,
        :meth:`~kvcache_sim.control.request.Request.continuation_keys`). Otherwise
        decoding is free in capacity terms.

        **After the last token, not during the generation.** Awaiting a publish
        between two steps would stall *every other member of the batch* -- the loop
        is one coroutine driving one accelerator -- widening everybody's inter-token
        gap with a TBT effect the hardware does not have. The cost lands where the
        handoff fetch lands: on the clock, inside arrival-to-last-token. What
        deferring costs is intra-generation residency, under-charged until the
        generation ends; no scenario here generates the 512 tokens that would take.

        Generated KV that does not fit is a cache fill like any other -- ``publish``
        answers ``False`` -- and by then the request has already been answered, so
        nothing is dropped mid-generation. Recorded as
        :attr:`~kvcache_sim.report.metrics.Metrics.decode_unpublished`. Preemption
        (a real engine evicting and recomputing a running sequence's blocks) is
        deliberately not modelled -- that is a scheduler this model does not have.
        """
        keys = list(request.block_keys)
        if response.decode == response.prefill:
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
                    f"{request.id} handoff from {response.prefill} to {self.me} "
                    f"found no {len(keys)}blk chain in the store (evicted or never "
                    f"cached); decoding without charging a transfer",
                )
                self.metrics.handed_off(request.id, self.me, 0, missed=True)
            else:
                # Measured off the KV that arrived -- the same tensors the transport
                # just charged for -- so the reported handoff cannot drift from the
                # charged one.
                nbytes = sum(b.numel() * b.element_size() for b in kv)
                self.trace.record(
                    self._now(),
                    "HANDOFF",
                    f"{request.id} {response.prefill} -> {self.me} "
                    f"({len(keys)}blk, {nbytes}B of KV)",
                )
                self.metrics.handed_off(request.id, self.me, nbytes)
                # The chain is on this host now, so the volume must charge itself
                # for it and the directory must know of the second copy. Published
                # *after* the handoff is recorded, so the reported transfer stays
                # the bytes that crossed the fabric.
                await self._reside(keys, kv, request.id, "chain")
        # The only path that answers without the request having finished: a run
        # whose decode side is not modelled, reaching a host with no engine. No last
        # token is coming and nothing here generates any.
        if self.decode_engine is None:
            return []
        generated = await self.decode_engine.admit(request)
        # What the batch produced, under the chain continued past the prompt.
        await self._reside(
            request.continuation_keys(len(generated.kv)),
            generated.kv,
            request.id,
            "generated",
        )
        return generated.tokens

    async def _reside(
        self, keys: List[str], blocks: List[torch.Tensor], request_id: str, why: str
    ) -> None:
        """Register KV this host now holds on this host's volume.

        One method for both of the decode side's publishes -- the chain it pulled in
        and the blocks it generated -- because they are the same act: the bytes are
        here, so the volume accounting for this host's memory has to be told and the
        directory has to know a copy is here.

        ``why`` names which of the two it was, for the trace only; nothing branches
        on it.

        A refusal is not fatal, as on the prefill side: the request is served (or,
        for the chain, is about to be served off KV this host holds in hand either
        way) and the loss is only that the volume will not keep it. Flagged on the
        row rather than counted into it, since blocks the volume threw back occupy
        nothing -- a decode pool that cannot keep what it decodes is mis-sized, and
        the hit rate will not say so.
        """
        if not blocks:
            return
        ok = await self.store.publish(self.me, keys, blocks)
        self.metrics.decode_resident(request_id, len(blocks), published=ok)
        nbytes = sum(b.numel() * b.element_size() for b in blocks)
        self.trace.record(
            self._now(),
            "RESIDE" if ok else "NOROOM",
            f"{request_id} {why} {len(blocks)}blk ({nbytes}B) "
            f"{'now resident on' if ok else 'did not fit'} {self.me}",
        )

    def _decode_done(self, request: Request, tbt: float) -> None:
        """Finalize a request once its last decode token is emitted.

        The gaps are this host's measurement of its own batch, so this host records
        them into the ledger's row -- the prefill host wrote the rest of that row
        before it ever named this one.
        """
        self.metrics.decoded(request.id, tbt)
        self.trace.record(
            self._now(), "DECODE", f"{request.id} decode done (tbt {tbt:.3f})"
        )
