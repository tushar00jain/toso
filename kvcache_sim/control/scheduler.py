"""One scheduler, two presets: ``LoadBalanceScheduler`` / ``CacheAwareScheduler``.

Both are a plain :class:`~proposed.plane.ControlPlane`, and one plane is the whole of
what this capability decides. Two members, asked at the two moments a serving host
has a question::

    await decide(request, me)   -> Optional[Response]   # where should this run
    await sources(keys, me)     -> Selection[int]       # who serves this fetch

The second exists because the first already answered it: routing prices a pull
against recomputing and records the peer it priced, and the fetch that follows asks
which peer to read from. Two objects would have to keep that note in step across a
boundary; one plane just reads its own sensor
(:class:`~kvcache_sim.control._selector.RoutedPull`).

Not a selector, because the decision is made of **two** selections and a selection
holds one: the prefill hosts this scheduler priced, and the decode hosts it ranked
against the winner among them. Both rankings are real, and neither leaves this plane --
what does is the winner of each and the price of the one that won (:class:`Response`),
since nothing outside asks what lost. A refusal (an SLO miss) is ``None``.

One ask settles both halves, before anything runs -- which is what makes a refusal
cost nothing.

What a host *reports* is not asked and is not answered, so it does not come here:
a fact goes to the sensor it corrects (:mod:`kvcache_sim.control._sensor`), which
this scheduler reads and the run gives a service of its own.

The question is about *compute*; data placement is not asked here -- the serving
host already knows which blocks it computed, and a volume that runs out of room
drops its own coldest keys and tells the directory afterwards
(:mod:`realsim.seams._retention`). Every argument and return is a value.

Both names are *presets* of one scheduler, each one choice on each of its two axes --
reuse and the winner, both selectors (:mod:`kvcache_sim.control._selector`, where every
ranking a decision here makes lives) and neither reachable from outside this plane:

* ``LoadBalanceScheduler`` (baseline, ~vLLM) = never pull, least-loaded instance:
  reuse only that instance's **local** cache, whatever a peer may hold.
* ``CacheAwareScheduler`` = pull under a balance threshold, lowest predicted TTFT,
  over the **global** prefix-match directory.

Admission is a setting rather than an axis: what a preset varies is whether the TBT SLO
applies at all, and ``early_rejection`` names which decode occupancy it is judged
against -- the one observed now (``early``) or the one predicted at prefill completion
(``predict``).

What a decision senses
----------------------
Nothing here executes, and nothing here is a live read. Every host this scheduler
ranks, prices or gates, it judges against one
:class:`~kvcache_sim.control._sensor.ClusterSensor` -- the predicted prefill
queues and the observed decode batches, and what keeps each of them true. That sensor
is read through the view this plane and everything it consults senses through
(:class:`~kvcache_sim.control._view.KVView`), beside the prefix runs, the prefills this
plane has promised and not seen land, and the pulls it has already priced
(:class:`~kvcache_sim.control._selector.RoutedPull`). Sensed rather than held because
the hosts are what keep it true: every other fact in it comes from them, over the
service the run fronts it with (:attr:`_Scheduler.sensor`).

An accepted decision is reported back the same way, into each sensor it moves
(:meth:`_Scheduler._admit`): the cluster sensor holds the prefill instance the plan
spoke for (:class:`~kvcache_sim.control._sensor.Committed`), the reservation sensor
stands in for a request no host can report yet, and the routed one remembers the peer
the pull was priced against. A run that judges the TBT SLO against the occupancy
observed now promises nothing and composes no reservation sensor at all, so the two
halves of the prediction cannot come apart.

The TTFT the metrics record is therefore the prediction, not a measurement (the
README says why). Prefill cost is deterministic, so on the default path the two
agree; an evicted block, a pull served by another volume, or ``contention`` can
each move the executed cost off it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from proposed import AnySelector, ControlPlane, Endpoint, Key, KeySelector, Selection
from proposed.selector import FirstMatch, Selector

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, MachineProfile, Model,
    prefill_time,
)

from ._answer import Batched, Plan, Priced, Response
from ._selector import (
    ByBatch, ByLoad, ByTTFT, LocalOnly, LongestPrefixKeySelector, RoutedPull,
)
from ._sensor import (
    ClusterSensor, Committed, ComputeBusy, DecodeState, PrefillFinished,
    ReservationSensor, RoutedPullSensor,
)
from ._view import KVView
from .request import Request

__all__ = [
    "PrefillFinished",
    "ComputeBusy",
    "DecodeState",
    "Plan",
    "Response",
    "LoadBalanceScheduler",
    "CacheAwareScheduler",
]


def _predicts_decode(simulate_decode: bool, early_rejection: str) -> bool:
    """Does this run roll decode occupancy forward to prefill completion?

    ``"predict"`` counts the prefills promised that will have landed by then, which
    moves where decode lands even with no SLO to miss -- a fidelity setting of what a
    decision senses rather than an admission rule, which is why admission applies the
    same two SLOs either way.

    Answered once, in :meth:`_Scheduler.__init__`, and it decides two things together:
    whether a decision records what it promised, and whether the prediction reads
    those promises.
    """
    return simulate_decode and early_rejection == "predict"


# -- pull or recompute: a test the reuse ranking's head is held to ----------- #


def _worth_pulling(
    counts: Dict[str, int], inst: str, threshold: float
) -> Callable[[str], bool]:
    """Does pulling the head's prefix beat recomputing the gap on ``inst``?

    Its run must be more than ``threshold`` times ``inst``'s own -- the balancing
    threshold. A pull is charged to the prefill instance's queue, so one that saves
    little compute still costs the whole wait, and without the threshold every request
    would chase the longest match onto one instance.

    A test for :meth:`~proposed.selector.Selection.require`, which is what makes it a
    test of the *head* and not a filter: see there for why the whole ranking goes.
    """
    return lambda head: counts.get(head, 0) > counts.get(inst, 0) * threshold


class _Scheduler(ControlPlane):
    """The pricing, ranking and admission both schedulers share.

    A :class:`~proposed.plane.ControlPlane` and nothing more specific: a serving host
    asks it as a service, and its two members (:meth:`decide`, :meth:`sources`) are
    the whole of its surface. The second is answered from what the first recorded --
    the peer a pull was priced against -- so a plan and the read that carries it out
    cannot disagree, and nothing is threaded through the data plane to keep them
    together.

    Args:
        reuse / rank: the two axes a preset picks, both selectors, both attached in
            :meth:`attach`, and both used while forming the one answer :meth:`decide`
            gives. ``reuse`` ranks the peers holding a prefix, priced in blocks of that
            prefix (``Selector[Sequence[Key], int]``); what a peer is *worth* to one
            candidate is this scheduler's to work out, off the snapshot it read itself
            (:meth:`_priced_reuse`), so a re-ranking under the axis cannot move a price
            this scheduler compares. ``rank`` ranks the candidates this scheduler
            priced, so its subject is those candidates and its price their
            :class:`~kvcache_sim.control._answer.Plan`.
        balance_threshold: how much longer the head's prefix run must be than a
            candidate's own before pulling beats recomputing (:func:`_worth_pulling`).
            Unread when ``reuse`` names nobody.
        source_selector: ranks the holders of a prefix, and what :meth:`sources`
            answers a fetch with behind the pull it already priced
            (:class:`~kvcache_sim.control._selector.RoutedPull`). ``None`` builds a
            :class:`~kvcache_sim.control._selector.LongestPrefixKeySelector`. A pulling
            preset hands the *same* object to its reuse axis, so the peer priced is
            the peer read from.
        block_tokens: tokens per KV block.
        profile / model: the cost constants prediction is priced against.
        decode_pool / prefill_pool: instance subsets (default: all).
        slo_ttft / slo_tbt: what admission holds a decision to (:meth:`_admit`).
        simulate_decode: whether the run models batched decode at all.
        early_rejection: ``"early"`` | ``"predict"`` -- whether the decode occupancy
            the TBT SLO is judged against is the one observed now or the one predicted
            at prefill completion.
        cluster: the run's one :class:`~kvcache_sim.control._sensor.ClusterSensor`,
            when a caller has to make it first. ``None`` -- the default -- builds it
            in :meth:`attach`, where the instances become known. Either way
            :meth:`attach` composes it into the view and this plane holds it no other
            way.
    """

    def __init__(
        self,
        *,
        reuse: Selector[Sequence[Key], int],
        rank: AnySelector[Sequence[Priced], Plan],
        balance_threshold: float = 1.5,
        source_selector: Optional[KeySelector[int]] = None,
        block_tokens: int,
        profile: MachineProfile = DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        decode_pool: Optional[List[str]] = None,
        prefill_pool: Optional[List[str]] = None,
        slo_ttft: float = float("inf"),
        slo_tbt: float = float("inf"),
        simulate_decode: bool = False,
        early_rejection: str = "early",
        cluster: Optional[ClusterSensor] = None,
    ) -> None:
        self.B = block_tokens
        self.profile = profile
        self.model = model
        self._reuse = reuse
        self._rank = rank
        # Built rather than handed in: not an axis (see the module it comes from).
        self._decode = ByBatch()
        self._threshold = balance_threshold
        self._source = (
            source_selector if source_selector is not None
            else LongestPrefixKeySelector()
        )
        self._prefill_pool = prefill_pool
        self._decode_pool = decode_pool
        self.slo_ttft = slo_ttft
        self.slo_tbt = slo_tbt
        self.tbt_enabled = simulate_decode
        if early_rejection not in ("early", "predict"):
            # Refused rather than read as "not predict": admission is gated at
            # routing whatever a caller passes, so a mode this does not know is a
            # caller expecting a rule this scheduler has no way to apply.
            raise ValueError(
                f"unknown early_rejection {early_rejection!r}: the choice is which "
                f"occupancy the TBT SLO is judged against, 'early' or 'predict'"
            )
        # The admission mode is spent here and never read again: both modes hold a
        # decision to the same two SLOs and differ only in what feeds them. This one
        # answer governs the reservation sensor end to end -- composed in attach(),
        # written on admission, read by the prediction -- so no two of the three can
        # disagree about whether this run predicts.
        self._lookahead = _predicts_decode(simulate_decode, early_rejection)
        # Filled by attach(): the run knows its servers only once its stack exists.
        self.topo: Dict[str, Endpoint] = {}
        self.ids: List[str] = []
        self.prefill_ids: List[str] = []
        self.decode_ids: List[str] = []
        # The caller's argument, not this plane's sensor: attach() spends it composing
        # the view and clears it, so the sensor has one reference here either way.
        self._supplied_cluster: Optional[ClusterSensor] = cluster
        # Both built in attach(), where the sensors they read exist.
        self.view: Optional[KVView] = None
        self._fetch: Optional[FirstMatch[int]] = None

    # -- the stack hands over its ports ----------------------------------- #
    def attach(self, view) -> None:
        """Receive the ports this control plane senses and prices through.

        Two-phase so a scenario can declare a control plane as an object
        (``MyControl(knobs)``) and let the run hand it the stack afterwards.

        The view is composed into a :class:`~kvcache_sim.control._view.KVView` here,
        with the reads this capability's decisions make: prefix runs, the cluster
        sensor, the prefills this plane promised, and the pulls it priced. None of the
        four is the store's notion, so the run supplies none of them. Everything
        downstream then senses one view -- both axes, the fetch chain -- and nothing is
        handed a sensor to read.

        The run's one :class:`~kvcache_sim.control._sensor.ClusterSensor` is built
        here unless a caller made it first (``cluster``): this is where the
        instances become known, and this runs once per run, so nothing else is
        placed to build a second (empty) one -- and an empty one would report every
        host idle, which is a run that looks healthy and is wrong. It goes into the
        view and nowhere else; :attr:`sensor`, which the run harvests to put a
        service in front of, reads it back from there.

        This plane's own two sensors are built here and never supplied, because one
        handed to two planes would have each answering for the other's
        decisions: a second routed-pull sensor would answer every fetch "I decided
        nothing about this", and a second reservation sensor would leave every
        predicted batch short. The reservation sensor is composed in only for a run
        that rolls occupancy forward, so a run that does not predict has no empty
        one to read (:class:`~kvcache_sim.control._view.ReservedView`).
        """
        self.topo = dict(view.topology)
        self.ids = sorted(self.topo)
        cluster, self._supplied_cluster = self._supplied_cluster, None
        if cluster is None:
            # Over ALL instances: the prefill and decode pools may each be a subset.
            cluster = ClusterSensor(self.ids)
        self.view = view.derived(
            KVView,
            cluster=cluster,
            reserved=ReservationSensor() if self._lookahead else None,
            routed=RoutedPullSensor(),
        )
        self.prefill_ids = (
            sorted(self._prefill_pool) if self._prefill_pool else self.ids
        )
        self.decode_ids = (
            sorted(self._decode_pool) if self._decode_pool else self.ids
        )
        # What a fetch is answered with. Both this and the reuse axis sense through
        # the KVView above -- the fetch because its head link reads the routed-pull
        # sensor (:class:`~kvcache_sim.control._selector.RoutedPull`), the reuse axis
        # because one routing decision pins that view's snapshot for the whole of
        # itself (:meth:`~kvcache_sim.control._view.KVView.pinned`) and a ranking
        # consulted inside the pin must not read past it into the live directory. The
        # source ranking is a link of both and gets attached twice, to the same view
        # either way, so no order here is load-bearing.
        self._fetch = FirstMatch([RoutedPull(), self._source]).attach(self.view)
        self._reuse.attach(self.view)
        self._rank.attach(self.view)
        # Senses nothing, but attached like every other ranking here, so wrapping it in
        # one that does needs nothing else moved.
        self._decode.attach(self.view)

    @property
    def sensor(self) -> Optional[ClusterSensor]:
        """:attr:`~proposed.plane.ControlPlane.sensor` -- the one sensor here a host
        writes, which the run fronts with a service.

        Read out of the view rather than stored beside it, so this plane has one path
        to it. ``None`` until :meth:`attach` builds it; the run harvests after. The
        other two sensors this plane holds are not offered: nothing outside this
        process writes them.
        """
        return None if self.view is None else self.view.cluster

    # -- what a serving host asks, at the two moments it has a question ------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection[int]:
        """Which peers should serve ``requester``'s fetch of ``keys``, best first.

        The pull :meth:`decide` already priced for this caller
        (:class:`~kvcache_sim.control._selector.RoutedPull`),
        else whoever holds the longest prefix -- a
        :class:`~proposed.selector.FirstMatch`, so the fall-through is that chain's
        abstention rule rather than an ``if`` here. ``Selection.of([])`` names nobody,
        which leaves the read to the directory's own order.

        Settled before it travels, like any answer this plane gives: neither link
        gates, so there is nothing to wait for, and saying so here is what keeps that
        a property of the ranking rather than of the caller.
        """
        return await (await self._fetch.select(list(keys), requester)).settled()

    async def decide(self, request: Request, requester: str) -> Optional[Response]:
        """Where should ``request`` run? Both selections, or ``None`` if refused.

        The decode side is chosen against the *winning* prefill candidate's predicted
        completion, so the second selection is asked once and after the first -- which
        is the whole reason a decision here is a :class:`Response` and not a
        selection: two answers, and a selection holds one.

        ``None`` is an SLO miss. There is no host this request may run on, and the
        refusal costs nothing because nothing has run.

        ``requester`` is the host the request landed on. Nothing that ranks, prices or
        holds a *candidate host* to an SLO reads it -- where a request should run is a
        fact about the cluster, not about who was asked. Every ranking is handed it
        because every selector is, and only the reuse ranking has a use for it: it is
        the answer to that ranking's own question, "who wants these bytes". The
        candidate under test is what the tests behind that ranking are handed.
        """
        prefill = await self._select_prefill(request, requester)
        decode = await self._select_decode(prefill.winner, requester)
        return self._admit(request, prefill, decode)

    async def _select_prefill(self, request: Request, requester: str) -> Selection[Plan]:
        """Every prefill instance, priced and ranked best first.

        Each one's :class:`Plan` rides under its id in
        :attr:`~proposed.selector.Selection.payload`, so the answer says what was
        compared as well as what won.

        Atomic: every read is off one pinned directory snapshot
        (:meth:`~kvcache_sim.control._view.KVView.pinned`) and one clock read, so the
        prices are comparable, and the bookkeeping it writes needs nothing locked.
        Pricing a candidate reserves nothing (:meth:`_candidate`), so the losers leave
        no trace and the winner is chosen after the whole field is known. Both axes are
        awaited inside the pin and neither suspends, so no second decision can enter it.

        The reuse ranking is asked once and tested per candidate: which peers hold this
        prefix is the same question whoever would prefill it, and only the tests behind
        it -- is that peer me, is its run worth the transfer -- read the candidate.
        """
        now = self.view.now()
        keys = list(request.block_keys)
        with self.view.pinned(keys):
            counts = self.view.prefix_lengths(keys)
            ranked = await self._reuse.select(keys, requester)
            candidates: List[Priced] = []
            for inst in self.prefill_ids:
                # A host is not its own peer, and a peer is only worth the transfer if
                # it holds materially more than this candidate already does.
                peer = (
                    ranked
                    .require(lambda head, me=inst: head != me)
                    .require(_worth_pulling(counts, inst, self._threshold))
                    .take(1)
                )
                match, src, pull = self._priced_reuse(counts, keys, inst, peer)
                candidates.append((inst, self._candidate(
                    request, inst, now, match=match, source=src, pull_keys=pull
                )))
            return await self._rank.select(candidates, requester)

    @staticmethod
    def _priced_reuse(
        counts: Dict[str, int], keys: Sequence[Key], inst: str,
        peer: Selection[int],
    ) -> Tuple[int, Optional[str], Sequence[str]]:
        """What one peer buys ``inst``: ``(match, source, pull_keys)``.

        Derived here and not in the selector, because ranking peers is where a
        selector's job ends: how much of this prompt that peer's prefix covers is
        arithmetic over the snapshot this scheduler already read. A selection naming
        nobody -- a test having dropped the ranking -- leaves the local match to
        recompute from.
        """
        local = counts.get(inst, 0)
        src = peer.head
        if src is None or src == inst:
            return local, None, ()
        return counts[src], src, keys[local:counts[src]]

    # -- prediction (no mutation) ---------------------------------------- #
    def _predict(self, inst: str, now: float, transfer_t: float, prefill_t: float):
        """Return ``(queue_wait, ttft, done_time)`` without reserving the server."""
        avail = max(now, self.view.cluster.busy_until[inst])
        queue_wait = avail - now
        done = avail + transfer_t + prefill_t
        return queue_wait, done - now, done

    def _candidate(
        self,
        request: Request,
        inst: str,
        now: float,
        *,
        match: int,
        source: Optional[str] = None,
        pull_keys: Sequence[str] = (),
    ) -> Plan:
        """Price prefilling ``request`` on ``inst`` reusing ``match`` blocks.

        Reserves nothing and mutates nothing, so a losing candidate leaves no trace
        (:class:`~kvcache_sim.control._sensor.Committed` records a decision actually
        taken). Which candidate wins is the rank axis's business
        (:class:`~kvcache_sim.control._selector.ByLoad` /
        :class:`~kvcache_sim.control._selector.ByTTFT`).
        """
        cached = min(match * self.B, request.prompt_tokens)
        uncached = request.prompt_tokens - cached
        prefill_t = prefill_time(uncached, self.profile, self.model)
        if source is not None and pull_keys:
            xbytes = self.model.block_bytes(len(pull_keys), self.B)
            # The cost model the transport charges, so this prediction is what the
            # real pull will cost.
            transfer_t = self.view.transfer_cost(source, inst, xbytes)
        else:
            source, xbytes, transfer_t = None, 0, 0.0
        queue_wait, ttft, done = self._predict(inst, now, transfer_t, prefill_t)
        plan = Plan(
            match, cached, uncached, source, xbytes, queue_wait, ttft, done,
        )
        plan.prefill_t = prefill_t
        plan.transfer_t = transfer_t
        plan.pull_keys = list(pull_keys)
        return plan

    # -- decode-side TBT prediction / admission -------------------------- #
    def _predicted_batch(self, d: str, done_time: float) -> int:
        """Predicted decode batch size on ``d`` seen by a request admitted at
        ``done_time`` (its prefill completion). Drives TBT prediction.

        The flag that picks between the two readings is the one that decided whether
        the reservation sensor was composed at all (:func:`_predicts_decode`), so the
        second reading finds one to read exactly when it takes it.
        """
        if not self.tbt_enabled:
            return 0
        if not self._lookahead:
            return self.view.cluster.occupancy(d)
        n = self.view.cluster.predict_occupancy(d, done_time)
        # Requests whose prefill has not landed are invisible to the observed
        # decode state; the outstanding reservations stand in for them.
        for res in self.view.reserved.pending(self.view.now()):
            if (
                res.decode_id == d
                and res.prefill_done <= done_time
                and res.prefill_done
                + max(0, res.output_tokens - 1)
                * decode_step_time(1, self.profile, self.model)
                > done_time
            ):
                n += 1
        return n

    async def _select_decode(self, plan: Plan, requester: str) -> Selection[int]:
        """Decode instances, each with the batch a request admitted at ``plan``'s
        completion is predicted to meet there.

        Ranked by :class:`~kvcache_sim.control._selector.ByBatch`. With decode
        unmodelled every candidate prices at zero, so the id tie-break is the whole of
        the choice.
        """
        batches: List[Batched] = [
            (d, self._predicted_batch(d, plan.done_time)) for d in self.decode_ids
        ]
        return await self._decode.select(batches, requester)

    def _admit(
        self,
        request: Request,
        prefill: Selection[Plan],
        decode: Selection[int],
    ) -> Optional[Response]:
        """The two winners as one :class:`Response`, held to the SLOs and committed.

        Assembled before the SLOs are checked rather than after, so the value they
        judge is the value the answer carries. ``None`` == rejected, and rejected here
        costs nothing: this runs before the prefill does, which is the whole of why
        admission is decided at routing.
        """
        instance = decode.sources[0]
        # The decode ranking prices headroom, and what the TBT SLO is judged on is the
        # occupancy behind it (:class:`~kvcache_sim.control._selector.ByBatch`).
        batch = -decode.payload[instance]
        response = Response(
            prefill=prefill.sources[0],
            decode=instance,
            plan=prefill.winner,
            pred_batch=batch,
            pred_tbt=(
                decode_step_time(batch + 1, self.profile, self.model)
                if self.tbt_enabled
                else 0.0
            ),
        )
        if response.plan.ttft > self.slo_ttft:
            return None
        # A run that does not model decode has no batch to hold to a TBT SLO.
        if self.tbt_enabled and response.pred_tbt > self.slo_tbt:
            return None
        # Accepted, so each sensor this decision moves is told: the cluster holds the
        # instance the plan spoke for, the reservation stands in for a request the
        # observed decode state cannot show until its prefill lands, and the routed
        # pull remembers the peer it was priced against for when the fetch asks
        # (:class:`~kvcache_sim.control._selector.RoutedPull`).
        #
        # Local writes, not the endpoint a host reports over: control is in the same
        # process, and plain calls are what keep this decision atomic. All three land in
        # one non-suspending window -- this method has no ``await`` -- and they touch
        # disjoint sensors, so nothing can read a half-committed decision and their
        # order here is unobservable.
        self.view.cluster.notify_sync(Committed(response))
        plan = response.plan
        if self._lookahead:
            self.view.reserved.reserve(
                plan.done_time, response.decode, request.output_tokens
            )
        if plan.reuse_source is not None and plan.pull_keys:
            self.view.routed.route(response.prefill, plan.pull_keys, plan.reuse_source)
        return response


# -- the two schedulers, as the settings that make them ---------------------- #
# Names rather than behaviours: everything either one does is above, and each is
# one choice on each axis. A third combination is composed, not subclassed.


class LoadBalanceScheduler(_Scheduler):
    """Baseline (~vLLM): least-loaded instance, local-only cache reuse."""

    def __init__(self, **knobs: Any) -> None:
        super().__init__(reuse=LocalOnly(), rank=ByLoad(), **knobs)


class CacheAwareScheduler(_Scheduler):
    """Cache-aware: global prefix-match routing under a balance threshold.

    ``replicate=False`` (which isolates replication's contribution in the demo) is
    never pulling at all, so it is the baseline's reuse axis rather than a flag
    the candidate loop tests.

    Args:
        source_selector: ranks peers for a prefix pull. Used *as* the reuse axis and
            handed on to :meth:`~_Scheduler.sources`, which is the point: one object,
            so the peer named while pricing is the peer that serves the read.
            ``replicate=False`` never asks it anything while pricing, and a fetch
            still is.

            Required here, and deliberately without a default: this ranking may keep
            state across the decisions it makes (one under a
            :class:`~proposed.selector.Discount` does), so which one a run uses is the
            run's to choose (:func:`kvcache_sim.workload._serving.scheduler`).
    """

    def __init__(self, *, source_selector: KeySelector[int],
                 balance_threshold: float = 1.5, replicate: bool = True,
                 **knobs: Any) -> None:
        super().__init__(
            reuse=source_selector if replicate else LocalOnly(),
            balance_threshold=balance_threshold,
            rank=ByTTFT(),
            source_selector=source_selector,
            **knobs,
        )
