"""One scheduler, two presets: ``LoadBalanceScheduler`` / ``CacheAwareScheduler``.

Both are a plain :class:`~proposed.plane.ControlPlane`, and one plane is the whole of
what this capability decides. Two members, asked at the two moments a serving host
has a question::

    await decide(request, me)   -> Optional[Response]   # where should this run
    await sources(keys, me)     -> Selection            # who serves this fetch

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
reuse, a selector (:mod:`kvcache_sim.control._selector`, where every ranking a decision
here makes lives), and the winner, the fold that orders the pool this plane keyed
itself; neither reachable from outside this plane:

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
the hosts are what keep it true: every other fact in it comes from them, as actions
dispatched into the one dispatcher the run fronts with a service
(:attr:`_Scheduler.dispatcher`).

A decision that names the host it was asked by goes the same way, as one
:class:`~kvcache_sim.control._sensor.Committed` (:meth:`_Scheduler._admit`) folded into
each sensor it moves: the cluster sensor holds the prefill instance the plan spoke for,
the reservation sensor stands in for a request no host can report yet, and the routed
one remembers the peer the pull was priced against. One that names somebody else is an
address and writes nothing, so a request passed from host to host is priced once per
host and booked once, by the one that serves it. A run that judges the TBT SLO
against the occupancy observed now composes no reservation sensor at all, so that same
action promises nothing and the two halves of the prediction cannot come apart.

The TTFT the metrics record is therefore the prediction, not a measurement (the
README says why). Prefill cost is deterministic, so on the default path the two
agree; an evicted block, a pull served by another volume, or ``contention`` can
each move the executed cost off it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from proposed import (
    ControlPlane, Dispatcher, Endpoint, Key, KeySelector, Selection,
)
from proposed.selector import Balance, Dims, FirstMatch, Selector

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, MachineProfile, Model,
    prefill_time,
)

from ._answer import Batched, Plan, Response
from ._selector import (
    by_prefix_and_load, ByBatch, LocalOnly, LongestPrefixKeySelector, RoutedPull,
)
from ._sensor import (
    ClusterSensor, Committed, ComputeBusy, DecodeState, PrefillFinished,
    ReservationSensor, RoutedPullSensor, SourceLoad,
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


def _source_ranking(name: str) -> Selector[Sequence[Key]]:
    """Which peers may serve a prefix, by name -- a fresh one every call.

    ``"spread"`` is the same ranking under a :class:`~proposed.selector.Balance`,
    carrying the fold that weighs a busy holder against a long match, so every caller
    of it folds the same way and none of them names a fold.
    """
    if name == "prefix":
        return LongestPrefixKeySelector()
    if name == "spread":
        return Balance(LongestPrefixKeySelector(), by_prefix_and_load())
    raise ValueError(
        f"unknown source ranking {name!r}: the choice is 'prefix' or 'spread'"
    )


# -- the two winner folds: what each preset compares of a priced pool ------- #
# Both read the pool :meth:`_Scheduler._select_prefill` keys, and each names the number
# it orders by, so a plan carries no order of its own and neither preset can pick up the
# other's (:class:`Plan`).


def _by_ttft(dims: Dims) -> float:
    """The cache-aware fold: the whole predicted queue + transfer + prefill.

    Why reuse is *priced* rather than preferred: a longer match on a busier instance
    can still lose. Read off the plan in the leading dimension, which is the plan the
    caller of this fold priced.
    """
    return dims[0].ttft


def _by_queue(dims: Dims) -> float:
    """The baseline's fold: the queue a candidate would join, and nothing else.

    Reads the dimension appended behind the plan, so what reuse bought cannot move this
    choice. That dimension is ``busy_until`` and not the plan's own ``queue_wait``,
    which is the same tail clamped at the clock: two instances idle since different
    moments both wait zero, and the pick would fall to the id tie-break instead of to
    the longer-idle host.
    """
    return dims[1]


class _Scheduler(ControlPlane):
    """The pricing, ranking and admission both schedulers share.

    A :class:`~proposed.plane.ControlPlane` and nothing more specific: a serving host
    asks it as a service, and its two members (:meth:`decide`, :meth:`sources`) are
    the whole of its surface. The second is answered from what the first recorded --
    the peer a pull was priced against -- so a plan and the read that carries it out
    cannot disagree, and nothing is threaded through the data plane to keep them
    together.

    Both axes are named rather than handed in, and the ranking of the two is built here
    (:func:`_source_ranking`): which ones a preset picks is the preset's business, and a
    name cannot be shared between two runs the way an object can.

    Args:
        reuse / rank: the two axes a preset picks, both used while forming the one
            answer :meth:`decide` gives. ``reuse`` is a ranking, attached in
            :meth:`attach`: the peers a candidate may pull from -- ``"peers"``, which
            is the ``source`` ranking itself, or ``"none"``, which names nobody; what a
            peer is *worth* to one candidate is this scheduler's to work out, off the
            snapshot it read itself (:meth:`_priced_reuse`), so a stage annotating the
            axis cannot move a price this scheduler compares. ``rank`` is a fold and
            not a ranking, over the pool this scheduler keyed itself -- ``"ttft"`` by
            predicted time to first token (:func:`_by_ttft`), ``"load"`` by what each
            host is already serving (:func:`_by_queue`).
        balance_threshold: how much longer the head's prefix run must be than a
            candidate's own before pulling beats recomputing (:func:`_worth_pulling`).
            Unread when ``reuse`` names nobody.
        source: which holders of a prefix :meth:`sources` answers a fetch with, behind
            the pull it already priced
            (:class:`~kvcache_sim.control._selector.RoutedPull`) -- ``"prefix"``,
            longest match first, or ``"spread"``, the same ranking
            under a :class:`~proposed.selector.Balance` so a host holding a hot prefix
            does not serve every read of it (:func:`_source_ranking`). The fold that
            reads its dimensions rides on its answers, since the two have to agree on
            how many there are, so neither place this ranking is folded names one
            (:meth:`sources`, :meth:`_select_prefill`).
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
        reuse: str,
        rank: str,
        balance_threshold: float = 1.5,
        source: str = "prefix",
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
        # Both sides of a pull build the same ranking, and it is a value: a selector
        # remembers nothing on itself (``check_structure`` rule 7), so two of them
        # answer alike off the sensors they share and neither side has to be handed the
        # other's. A preset that never pulls prices against nobody instead.
        if reuse not in ("peers", "none"):
            raise ValueError(
                f"unknown reuse axis {reuse!r}: a candidate pulls from the peers the "
                f"source ranking names or from nobody, so the choice is 'peers' or "
                f"'none'"
            )
        self._reuse = _source_ranking(source) if reuse == "peers" else LocalOnly()
        self._fetch = FirstMatch([RoutedPull(), _source_ranking(source)])
        if rank not in ("ttft", "load"):
            raise ValueError(
                f"unknown winner ranking {rank!r}: what orders the candidates this "
                f"plane priced is their predicted TTFT or the queue each would join, "
                f"so the choice is 'ttft' or 'load'"
            )
        #: Which of the two folds orders the prefill pool, and with it whether a queue
        #: dimension is appended for that fold to read -- one name, read in the one
        #: place both happen (:meth:`_select_prefill`).
        self._rank = rank
        # Built rather than named: not an axis (see the module it comes from).
        self._decode = ByBatch()
        self._threshold = balance_threshold
        self._prefill_pool = prefill_pool
        self._decode_pool = decode_pool
        #: Requests this plane has sent to another host, and the answer it sent them
        #: with. Private, because the only thing that reads it is the next ask about
        #: the same request (:meth:`decide`) and this is what wrote it: state with one
        #: reader that is also its writer is not something to sense.
        #:
        #: Only a decision that *moves* a request is kept -- one that names the host
        #: that asked is served in that same call, so no second ask about it is coming
        #: -- which gives every entry exactly one reader and needs no clock to forget
        #: by.
        self._placed: Dict[str, Response] = {}
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
        # All built in attach(), where the sensors they read exist.
        self.view: Optional[KVView] = None
        #: :attr:`~proposed.plane.ControlPlane.dispatcher` -- where a host's facts
        #: arrive, and the only thing that writes any sensor here. The run harvests it
        #: after :meth:`attach` to put a service in front of.
        self.dispatcher: Optional[Dispatcher] = None

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

        The dispatcher is built here too, with all three sensors as its reducers, and is
        the only thing that writes any of them (:attr:`dispatcher`). Which is where the
        reservation's condition lives: a run that does not predict composes no
        reservation sensor, so the same :class:`~kvcache_sim.control._sensor.Committed`
        a serving decision dispatches reserves nothing -- the flag is spent on the
        wiring, and no fold reads it.
        """
        self.topo = dict(view.topology)
        self.ids = sorted(self.topo)
        cluster, self._supplied_cluster = self._supplied_cluster, None
        if cluster is None:
            # Over ALL instances: the prefill and decode pools may each be a subset.
            cluster = ClusterSensor(self.ids)
        reserved = ReservationSensor() if self._lookahead else None
        routed = RoutedPullSensor()
        load = SourceLoad()
        self.view = view.derived(
            KVView, cluster=cluster, reserved=reserved, routed=routed, load=load,
        )
        self.dispatcher = Dispatcher()
        for sensor in (cluster, reserved, routed, load):
            # ``None`` is a sensor this run does not hold, so nothing folds for it.
            if sensor is not None:
                self.dispatcher.compose(sensor)
        self.prefill_ids = (
            sorted(self._prefill_pool) if self._prefill_pool else self.ids
        )
        self.decode_ids = (
            sorted(self._decode_pool) if self._decode_pool else self.ids
        )
        # Every ranking senses the view its own header declares
        # (:attr:`~proposed.selector.Selector.sensors`), composed out of the KVView
        # above. Each such subset shares that view's pin
        # (:meth:`~proposed.view.View.subset`), so a ranking consulted inside a routing
        # decision reads the snapshot the decision pinned rather than past it into the
        # live directory.
        for ranking in (self._fetch, self._reuse, self._decode):
            ranking.attach(self.view.subset(*ranking.sensors))

    # -- what a serving host asks, at the two moments it has a question ------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection:
        """Which peers should serve ``requester``'s fetch of ``keys``, best first.

        The pull :meth:`decide` already priced for this caller
        (:class:`~kvcache_sim.control._selector.RoutedPull`),
        else whoever holds the longest prefix -- a
        :class:`~proposed.selector.FirstMatch`, so the fall-through is that chain's
        abstention rule rather than an ``if`` here. ``Selection.of([])`` names nobody,
        which leaves the read to the directory's own order.

        Folded here, by whatever the answer carries, because the links only key what
        they name: a
        chain answering with the memo names one peer and has nothing to order, while the
        ranking behind it keys every holder of the prefix and the caller reads down what
        this returns (:func:`~proposed.selector.prefer`).

        Settled before it travels, like any answer this plane gives: neither link
        gates, so there is nothing to wait for, and saying so here is what keeps that
        a property of the ranking rather than of the caller.
        """
        ranked = (await self._fetch.select(list(keys), requester)).sort()
        return await ranked.settled()

    async def decide(self, request: Request, requester: str) -> Optional[Response]:
        """Where should ``request`` run? Both selections, or ``None`` if refused.

        The decode side is chosen against the *winning* prefill candidate's predicted
        completion, so the second selection is asked once and after the first -- which
        is the whole reason a decision here is a :class:`Response` and not a
        selection: two answers, and a selection holds one.

        ``None`` is an SLO miss. There is no host this request may run on, and the
        refusal costs nothing because nothing has run.

        ``requester`` is the host asking, and nothing that ranks, prices or holds a
        *candidate host* to an SLO reads it -- where a request should run is a fact
        about the cluster, not about who was asked, so two hosts asking about one
        request are answered the same way by an unchanged cluster. Every ranking is
        handed it because every selector is, and only the reuse ranking has a use for
        it: it is the answer to that ranking's own question, "who wants these bytes".
        The candidate under test is what the tests behind that ranking are handed.

        Asked **once per request**, however many hosts the request is passed through: a
        decision that moves it is recorded as it is booked (:attr:`_placed`), and the ask
        from the host it names is answered with it rather than priced again. Which is what settles a request: pricing again would
        price a cluster this decision has already booked, so the host just chosen reads
        as busier and the answer could move -- a request rerouted by its own booking. A
        request admitted once is also not judged again, so "a refusal costs nothing
        because nothing has run" stays true of the one ask that can refuse.
        """
        placed = self._placed.pop(request.id, None)
        if placed is not None:
            return placed
        prefill = await self._select_prefill(request, requester)
        # The winning plan, read once off the dimension it rides in: both halves of the
        # answer are formed against the same one.
        plan: Plan = prefill.key[prefill.head][0]
        decode = await self._select_decode(plan, requester)
        return self._admit(request, requester, prefill, plan, decode)

    async def _select_prefill(self, request: Request, requester: str) -> Selection:
        """Every prefill instance, priced and folded into an order, best first.

        Each one's :class:`Plan` is the leading dimension of its key, so the answer says
        what was compared as well as what won -- which is why the fold here is
        :meth:`~proposed.selector.Selection.sort` and not
        :meth:`~proposed.selector.Selection.max`.

        Atomic: every read is off one pinned directory snapshot
        (:meth:`~proposed.view.View.pinned`) and one clock read, so the
        prices are comparable, and the bookkeeping it writes needs nothing locked.
        Pricing a candidate reserves nothing (:meth:`_candidate`), so the losers leave
        no trace and the winner is chosen after the whole field is known. Both axes are
        awaited inside the pin and neither suspends, so no second decision can enter it.

        The reuse ranking is asked once, folded to its best peer once (by the fold its
        own answer carries, which is why this and :meth:`sources` cannot read one
        ranking two ways), and tested per
        candidate: which peers hold this prefix is the same question whoever would
        prefill it, and only the tests behind it -- is that peer me, is its run worth
        the transfer -- read the candidate. Folded *before* the loop, because each test
        is all-or-nothing on the head (:meth:`~proposed.selector.Selection.require`) and
        the head of an unfolded answer is whatever order the axis built it in.

        The two winner axes are one dimension apart. Both key the pool at the plan and
        both name what they compare of it -- the predicted TTFT (:func:`_by_ttft`), or
        the queue each candidate would join, which the baseline appends first
        (:func:`_by_queue`). The queue dimension goes on only where that fold reads it:
        compared as they stand ``(plan, busy)`` would break a TTFT tie by load rather
        than by id, and two idle instances holding no prefix do price identically.

        Annotated once per dimension rather than once per candidate: annotating rebuilds
        the whole key mapping (:meth:`~proposed.selector.Selection.annotated`), so the
        loop fills a mapping and the appending happens after it.
        """
        now = self.view.now()
        keys = list(request.block_keys)
        with self.view.pinned(keys):
            counts = self.view.prefix_lengths(keys)
            best = (await self._reuse.select(keys, requester)).max()
            plans: Dict[str, Plan] = {}
            for inst in self.prefill_ids:
                # A host is not its own peer, and a peer is only worth the transfer if
                # it holds materially more than this candidate already does.
                peer = (
                    best
                    .require(lambda head, me=inst: head != me)
                    .require(_worth_pulling(counts, inst, self._threshold))
                )
                match, src, pull = self._priced_reuse(counts, keys, inst, peer)
                plans[inst] = self._candidate(
                    request, inst, now, match=match, source=src, pull_keys=pull
                )
            pool = Selection.of(self.prefill_ids).annotated(plans)
            if self._rank == "load":
                busy = self.view.cluster.busy_until
                queued = pool.annotated({i: busy[i] for i in self.prefill_ids})
                return queued.sort(_by_queue)
            return pool.sort(_by_ttft)

    @staticmethod
    def _priced_reuse(
        counts: Dict[str, int], keys: Sequence[Key], inst: str,
        peer: Selection,
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
        taken). Which candidate wins is the winner axis's business
        (:meth:`_select_prefill`).
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

    async def _select_decode(self, plan: Plan, requester: str) -> Selection:
        """Decode instances, each with the batch a request admitted at ``plan``'s
        completion is predicted to meet there.

        Keyed by :class:`~kvcache_sim.control._selector.ByBatch` and folded here, as
        this plane's other answer is. With decode unmodelled every candidate keys at
        zero, so the id tie-break is the whole of the choice.
        """
        batches: List[Batched] = [
            (d, self._predicted_batch(d, plan.done_time)) for d in self.decode_ids
        ]
        return (await self._decode.select(batches, requester)).sort()

    def _admit(
        self,
        request: Request,
        requester: str,
        prefill: Selection,
        plan: Plan,
        decode: Selection,
    ) -> Optional[Response]:
        """The two winners as one :class:`Response`, held to the SLOs and committed.

        Assembled before the SLOs are checked rather than after, so the value they
        judge is the value the answer carries. ``None`` == rejected, and rejected here
        costs nothing: this runs before the prefill does, which is the whole of why
        admission is decided at routing.

        ``plan`` is the winning prefill candidate's, read off its key by :meth:`decide`
        -- the same value the decode side was chosen against.
        """
        instance = decode.sources[0]
        # The decode ranking keys the occupancy, which is what the TBT SLO is judged on
        # (:class:`~kvcache_sim.control._selector.ByBatch`).
        batch = decode.key[instance][0]
        response = Response(
            prefill=prefill.sources[0],
            decode=instance,
            plan=plan,
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
        # Accepted, so it is dispatched: one action, folded into every sensor it moves
        # -- the cluster holds the instance the plan spoke for, the reservation stands
        # in for a request the observed decode state cannot show until its prefill
        # lands, the routed pull remembers the peer it was priced against for when the
        # fetch asks (:class:`~kvcache_sim.control._selector.RoutedPull`). Each fold
        # writes its own sensor and reads no other, so their order is unobservable.
        #
        # At the instant the decision is made, which is what a booking has to be: a
        # request placed here holds this instance from now, so nothing decided after it
        # prices against a queue that does not hold it yet.
        #
        # The synchronous half, not the endpoint a host reports over: control is in the
        # same process, and nothing in this method may suspend -- an ``await`` here
        # would let a second decision interleave with a half-committed one.
        self.dispatcher.dispatch_sync(Committed(response, request.output_tokens))
        if response.prefill != requester:
            # ...and where this request was sent, for the one ask that follows it
            # (:attr:`_placed`). Beside the commit and not in it: nothing outside this
            # plane reads it, so nothing outside this plane needs to hear about it.
            self._placed[request.id] = response
        return response


# -- the two schedulers, as the settings that make them ---------------------- #
# Names rather than behaviours: everything either one does is above, and each is
# one choice on each axis. A third combination is composed, not subclassed.


class LoadBalanceScheduler(_Scheduler):
    """Baseline (~vLLM): least-loaded instance, local-only cache reuse."""

    def __init__(self, **knobs: Any) -> None:
        super().__init__(reuse="none", rank="load", **knobs)


class CacheAwareScheduler(_Scheduler):
    """Cache-aware: global prefix-match routing under a balance threshold.

    ``replicate=False`` (which isolates replication's contribution in the demo) is
    never pulling at all, so it is the baseline's reuse axis rather than a flag
    the candidate loop tests.

    Args:
        replicate: whether a candidate may pull a prefix from a peer at all. The
            ``source`` ranking is the reuse axis when it may, which is the point: one
            object, so the peer named while pricing is the peer that serves the read.
            ``replicate=False`` prices against nobody, and a fetch is still answered
            from that ranking.
    """

    def __init__(self, *, balance_threshold: float = 1.5, replicate: bool = True,
                 **knobs: Any) -> None:
        super().__init__(
            reuse="peers" if replicate else "none",
            balance_threshold=balance_threshold,
            rank="ttft",
            **knobs,
        )
