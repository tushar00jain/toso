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
holds one: the prefill hosts one chain priced, and the decode hosts another ranked
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
here makes lives), and the winner, the fold the prefill chain is stamped with; neither
reachable from outside this plane:

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

from typing import Any, Dict, List, Optional, Sequence

from proposed import (
    ControlPlane, Dispatcher, Key, Selection,
)
from proposed.selector import (
    Annotate, Balance, declared, Dims, FirstMatch, Folded, Max, Selector, Sort,
)

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, MachineProfile, Model,
)

from ._answer import Plan, Response
from ._selector import (
    by_prefix_and_load, DecodeBatch, LocalOnly, LongestPrefixKeySelector, PrefillAsk,
    Priced, RoutedPull,
)
from ._sensor import (
    ClusterSensor, Committed, ComputeBusy, DecodeState, FetchAnswered,
    PrefillFinished, ReservationSensor, RoutedPullSensor, SourceLoad,
)
from ._view import ClusterView, KVView
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


def _source_ranking(name: str) -> Selector[Sequence[Key]]:
    """Which peers may serve a prefix, by name -- a fresh one every call.

    ``"spread"`` is the same ranking under a :class:`~proposed.selector.Balance`, stamped
    with the fold that weighs a busy holder against a long match, so both chains this
    goes into fold it the same way and neither names a fold.
    """
    if name == "prefix":
        return LongestPrefixKeySelector()
    if name == "spread":
        return Folded(Balance(LongestPrefixKeySelector()), by_prefix_and_load())
    raise ValueError(
        f"unknown source ranking {name!r}: the choice is 'prefix' or 'spread'"
    )


# -- the two winner folds: what each preset compares of a priced pool ------- #
# Named here, where the chain that keys that pool is declared (:meth:`_Scheduler.attach`),
# because a fold reads its dimensions by position and only the chain says how many there
# are: the plan :class:`~kvcache_sim.control._selector.Priced` appends, and behind it the
# queue :meth:`_Scheduler.attach` annotates on for one preset alone. Each names the
# number it orders by,
# so a plan carries no order of its own and neither preset can pick up the other's
# (:class:`Plan`).


def _by_ttft(dims: Dims) -> float:
    """The cache-aware fold: the whole predicted queue + transfer + prefill.

    Why reuse is *priced* rather than preferred: a longer match on a busier instance
    can still lose. Read off the plan in the leading dimension.
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
    """The chains, the join and the admission both schedulers share.

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
            peer is *worth* to one candidate is the pricing stage's to work out
            (:class:`~kvcache_sim.control._selector.Priced`), so a stage annotating the
            axis cannot move a price the prefill chain compares. ``rank`` is a fold and
            not a ranking, over the pool the prefill chain keys -- ``"ttft"`` by
            predicted time to first token (:func:`_by_ttft`), ``"load"`` by what each
            host is already serving (:func:`_by_queue`).
        balance_threshold: how much longer the head's prefix run must be than a
            candidate's own before pulling beats recomputing, handed to that stage.
            Unread when ``reuse`` names nobody.
        source: which holders of a prefix :meth:`sources` answers a fetch with, behind
            the pull it already priced
            (:class:`~kvcache_sim.control._selector.RoutedPull`) -- ``"prefix"``,
            longest match first, or ``"spread"``, the same ranking
            under a :class:`~proposed.selector.Balance` so a host holding a hot prefix
            does not serve every read of it (:func:`_source_ranking`). The fold that
            reads its dimensions is stamped on its answers, since the two have to agree
            on how many there are, so neither of the two chains it goes into names one.
        block_tokens: tokens per KV block.
        profile / model: the cost constants prediction is priced against.
        decode_pool / prefill_pool: instance subsets (default: all).
        slo_ttft / slo_tbt: what admission holds a decision to (:meth:`_admit`).
        simulate_decode: whether the run models batched decode at all.
        early_rejection: ``"early"`` | ``"predict"`` -- whether the decode occupancy
            the TBT SLO is judged against is the one observed now or the one predicted
            at prefill completion.
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
        self._reuse = Max(
            _source_ranking(source) if reuse == "peers" else LocalOnly()
        )
        self._fetch = Sort(FirstMatch([RoutedPull(), _source_ranking(source)]))
        if rank not in ("ttft", "load"):
            raise ValueError(
                f"unknown winner ranking {rank!r}: what orders the candidates this "
                f"plane priced is their predicted TTFT or the queue each would join, "
                f"so the choice is 'ttft' or 'load'"
            )
        #: Which of the two folds orders the prefill pool, and with it whether a queue
        #: dimension is appended for that fold to read -- one name, spent in the one
        #: place both happen, the wiring (:meth:`attach`).
        self._rank = rank
        # Built rather than named: not an axis (see the module it comes from).
        self._threshold = balance_threshold
        #: The pools the prefill and decode chains rank. A preset that named no
        #: subset leaves them empty until :meth:`attach` knows every instance.
        self.prefill_ids: List[str] = sorted(prefill_pool) if prefill_pool else []
        self.decode_ids: List[str] = sorted(decode_pool) if decode_pool else []
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
        # Does this run roll decode occupancy forward to prefill completion --
        # counting the prefills promised that will have landed by then? A fidelity
        # setting of what a decision senses, not an admission rule: both modes hold a
        # decision to the same two SLOs and differ only in what feeds them. Spent here
        # and never read again, so this one answer governs the reservation sensor end
        # to end -- composed in attach(), written on admission, read by the prediction
        # -- and no two of the three can disagree about whether this run predicts.
        self._lookahead = simulate_decode and early_rejection == "predict"
        # All built in attach(), where the sensors they read and the pools they rank
        # exist.
        self.view: Optional[KVView] = None
        self._prefill: Optional[Selector[PrefillAsk]] = None
        self._decode: Optional[Selector[Plan]] = None
        #: :attr:`~proposed.plane.ControlPlane.dispatcher` -- where a host's facts
        #: arrive, and the only thing that writes any sensor here. The run harvests it
        #: after :meth:`attach` to put a service in front of.
        self.dispatcher: Optional[Dispatcher] = None

    # -- the stack hands over its ports ----------------------------------- #
    def attach(self, view) -> None:
        """Receive the ports this control plane senses and prices through.

        Two-phase so a scenario can declare a control plane as an object
        (``MyControl(knobs)``) and let the run hand it the stack afterwards.

        Where all four of this plane's decisions are **declared**, as the chains they
        are: which peer a pull comes from, which peer serves a fetch, which host prefills,
        which host decodes. Two of them need a pool this method resolves, and the winner
        axis is spent here rather than at select time.

        The view is composed into a :class:`~kvcache_sim.control._view.KVView` here,
        with the reads this capability's decisions make: prefix runs, the cluster
        sensor, the prefills this plane promised, and the pulls it priced. None of the
        four is the store's notion, so the run supplies none of them. Everything
        downstream then senses one view -- every chain above -- and nothing is handed a
        sensor to read.

        Every sensor is built here and none is ever supplied, because one handed to two
        planes would have each answering for the other's decisions: a second cluster
        sensor would report every host idle -- a run that looks healthy and is wrong --
        a second routed-pull sensor would answer every fetch "I decided nothing about
        this", and a second reservation sensor would leave every predicted batch short.
        This is also where the instances become known, and it runs once per run, so
        nothing else is placed to build any of them. Each goes into the view and
        nowhere else. The reservation sensor is composed in only for a run that rolls
        occupancy forward, so a run that does not predict has no empty one to read
        (:class:`~kvcache_sim.control._view.ReservedView`).

        The dispatcher is built here too, with all three sensors as its reducers, and is
        the only thing that writes any of them (:attr:`dispatcher`). Which is where the
        reservation's condition lives: a run that does not predict composes no
        reservation sensor, so the same :class:`~kvcache_sim.control._sensor.Committed`
        a serving decision dispatches reserves nothing -- the flag is spent on the
        wiring, and no fold reads it.
        """
        ids = sorted(view.topology)
        # Over ALL instances: the prefill and decode pools may each be a subset.
        cluster = ClusterSensor(ids)
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
        self.prefill_ids = self.prefill_ids or ids
        self.decode_ids = self.decode_ids or ids
        self._decode = Sort(DecodeBatch(
            self.decode_ids,
            tbt_enabled=self.tbt_enabled,
            lookahead=self._lookahead,
            profile=self.profile,
            model=self.model,
        ))
        priced = Priced(
            self.prefill_ids,
            block_tokens=self.B,
            profile=self.profile,
            model=self.model,
            threshold=self._threshold,
        )
        if self._rank == "load":
            # The queue dimension goes on only where that fold reads it: compared as
            # they stand ``(plan, busy)`` would break a TTFT tie by load rather than by
            # id, and two idle instances holding no prefix do price identically.
            self._prefill = Sort(Folded(
                Annotate(
                    priced,
                    lambda view, _ask: view.cluster.busy_until,
                    senses=(ClusterView,),
                ),
                _by_queue,
            ))
        else:
            self._prefill = Sort(Folded(priced, _by_ttft))
        # Every ranking senses the view its own header declares
        # (:attr:`~proposed.selector.Selector.sensors`), composed out of the KVView
        # above. Each such subset shares that view's pin
        # (:meth:`~proposed.view.View.subset`), so a ranking consulted inside a routing
        # decision reads the snapshot the decision pinned rather than past it into the
        # live directory.
        for ranking in (self._fetch, self._reuse, self._prefill, self._decode):
            ranking.attach(declared(self.view, ranking))

    # -- what a serving host asks, at the two moments it has a question ------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection:
        """Which peers should serve ``requester``'s fetch of ``keys``, best first.

        The pull :meth:`decide` already priced for this caller
        (:class:`~kvcache_sim.control._selector.RoutedPull`),
        else whoever holds the longest prefix -- a
        :class:`~proposed.selector.FirstMatch`, so the fall-through is that chain's
        abstention rule rather than an ``if`` here. ``Selection.of([])`` names nobody,
        which leaves the read to the directory's own order.

        Ordered by the chain's own :class:`~proposed.selector.Sort`, by whatever fold the
        answer carries, because the links only key what they name: a chain answering with
        the memo names one peer and has nothing to order, while the ranking behind it keys
        every holder of the prefix and the caller reads down what this returns
        (:func:`~proposed.selector.prefer`).

        Settled before it travels, like any answer this plane gives: neither link
        gates, so there is nothing to wait for, and saying so here is what keeps that
        a property of the ranking rather than of the caller.

        Answering is dispatched, unconditionally and whatever answered
        (:class:`~kvcache_sim.control._sensor.FetchAnswered`): a fetch a pull was priced
        for spends that memo, one nothing priced spends nothing, so which link won is
        never this method's question. Nothing between the chain's answer and the dispatch
        suspends, so no second fetch of the same keys can read the memo this one answered
        from.
        """
        answer = await self._fetch.select(list(keys), requester)
        self.dispatcher.dispatch_sync(FetchAnswered(requester, tuple(keys)))
        return await answer.settled()

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

        now = self.view.now()
        keys = list(request.block_keys)
        with self.view.pinned(keys):
            prefill = await self._prefill.select(
                PrefillAsk(
                    request=request,
                    now=now,
                    keys=keys,
                    counts=self.view.prefix_lengths(keys),
                    peer=await self._reuse.select(keys, requester),
                ),
                requester,
            )
            # The winning plan, read once off the dimension it rides in: both halves of the
            # answer are formed against the same one.
            plan: Plan = prefill.key[prefill.head][0]
            decode = await self._decode.select(plan, requester)

        return self._admit(request, requester, prefill, plan, decode)

    # -- admission -------------------------------------------------------- #
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
        # The decode answer keys the occupancy, which is what the TBT SLO is judged
        # on (:class:`~kvcache_sim.control._selector.DecodeBatch`).
        batch = decode.key[instance][0]
        response = Response(
            prefill=prefill.sources[0],
            decode=instance,
            plan=plan,
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
