"""One scheduler, two presets: ``LoadBalanceScheduler`` / ``CacheAwareScheduler``.

One plane is the whole of what this capability decides. Two members, asked at the two
moments a serving host has a question::

    await decide(request, me)   -> Optional[Response]   # where should this run
    await sources(keys, me)     -> Selection            # who serves this fetch

``sources`` answers from the peer ``decide`` already priced
(:class:`~kvcache_sim.control._selector.RoutedPull`), so a plan and the read carrying
it out cannot disagree.

A decision is **two** selections: the prefill hosts, one chain priced; the decode
hosts, another ranked against the winner among them. What leaves this plane is the
winner of each and the price of the one that won (:class:`Response`); a refusal (an
SLO miss) is ``None``, and costs nothing because one ask settles both halves before
anything runs.

What a host *reports* is not asked and not answered, so it does not come here: a fact
goes to the sensor it corrects (:mod:`kvcache_sim.control._sensor`), which this
scheduler reads and the run gives a service of its own. The question is about
*compute*: data placement is not asked, since the serving host knows which blocks it
computed and a volume out of room drops its own coldest keys and tells the directory
afterwards (:mod:`realsim.seams._retention`). Every argument and return is a value.

Both names are *presets* of one scheduler, one choice on each of two axes -- reuse, a
ranking of peers (:mod:`kvcache_sim.control._selector`), and the winner, a fold over
the priced prefill pool:

* ``LoadBalanceScheduler`` (baseline, ~vLLM) = never pull, least-loaded instance:
  reuse only that instance's **local** cache, whatever a peer may hold.
* ``CacheAwareScheduler`` = pull under a balance threshold, lowest predicted TTFT,
  over the **global** prefix-match directory.

Admission is a setting rather than an axis: a preset varies whether the TBT SLO
applies at all, and ``early_rejection`` names which decode occupancy it is judged
against -- the one observed now (``early``) or the one predicted at prefill completion
(``predict``).

What does a decision sense?
---------------------------
Nothing here executes and nothing here is a live read. Every host this plane ranks,
prices or gates is judged against one :class:`~kvcache_sim.control._view.KVView`: the
predicted prefill queues and observed decode batches
(:class:`~kvcache_sim.control._sensor.ClusterSensor`), the prefix runs, the prefills
promised and not yet seen to land, and the pulls already priced. Every fact in it
comes from the hosts, dispatched as actions (:attr:`_Scheduler.dispatcher`).

A decision naming the host that asked is dispatched the same way, as one
:class:`~kvcache_sim.control._sensor.Committed` (:meth:`_Scheduler._admit`) folded into
each sensor it moves: the cluster sensor takes the prefill instance the plan spoke for,
the reservation sensor stands in for a request no host can report yet, the routed one
remembers the peer the pull was priced against. A decision naming somebody else is an
address and writes nothing, so a request passed from host to host is priced once per
host and booked once, by the one that serves it. A run judging the TBT SLO against the
occupancy observed now composes no reservation sensor, so that same action promises
nothing and the two halves of the prediction cannot come apart.

The TTFT the metrics record is therefore the prediction, not a measurement (the README
says why). Prefill cost is deterministic, so on the default path the two agree; an
evicted block, a pull served by another volume, or ``contention`` can each move the
executed cost off it.
"""

from __future__ import annotations

from enum import Enum
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
    "Reuse",
    "Rank",
    "Source",
    "Occupancy",
    "LoadBalanceScheduler",
    "CacheAwareScheduler",
]


class Reuse(Enum):
    """Where a candidate may pull a prefix from while it is priced."""

    PEERS = "peers"  # whoever the source ranking names
    NONE = "none"    # nobody -- reuse only what the candidate already holds


class Rank(Enum):
    """Which number orders the priced prefill pool."""

    TTFT = "ttft"  # predicted time to first token (:func:`_by_ttft`)
    LOAD = "load"  # the queue a candidate would join (:func:`_by_queue`)


class Source(Enum):
    """Which holders of a prefix a fetch is answered with."""

    PREFIX = "prefix"  # longest match first
    SPREAD = "spread"  # ...docked by how busy the holder is


class Occupancy(Enum):
    """Which decode occupancy the TBT SLO is judged against."""

    EARLY = "early"      # the one observed now
    PREDICT = "predict"  # the one predicted at prefill completion


def _source_ranking(source: Source) -> Selector[Sequence[Key]]:
    """Which peers may serve a prefix -- a fresh one every call.

    :attr:`Source.SPREAD` weighs a busy holder against a long match, and carries that
    fold with it, so both chains it goes into weigh them the same way.
    """
    if source is Source.SPREAD:
        return Folded(Balance(LongestPrefixKeySelector()), by_prefix_and_load())
    return LongestPrefixKeySelector()


# -- the two winner folds: what each preset compares of a priced pool ------- #
# Dimension 0 is the candidate's plan; dimension 1 is the queue, annotated on for the
# baseline alone (:meth:`_Scheduler.attach`).
#
# Reuse is *priced* rather than preferred: a longer match on a busier instance can
# still lose. The queue dimension is ``busy_until``, not the plan's ``queue_wait``,
# which is the same tail clamped at the clock -- two instances idle since different
# moments both wait zero, and the pick would fall to the id tie-break rather than to
# the longer-idle host.


def _by_ttft(dims: Dims) -> float:
    """The cache-aware fold: the whole predicted queue + transfer + prefill."""
    return dims[0].ttft


def _by_queue(dims: Dims) -> float:
    """The baseline's fold: the queue a candidate would join, and nothing else."""
    return dims[1]


class _Scheduler(ControlPlane):
    """The chains, the join and the admission both schedulers share.

    Both axes arrive as names, not objects: a name cannot be shared between two runs
    the way an object can.

    Args:
        reuse: :class:`Reuse` -- where a candidate may pull a prefix from.
        rank: :class:`Rank` -- which number orders the priced prefill pool.
        balance_threshold: how much longer the head's prefix run must be than a
            candidate's own before pulling beats recomputing. Unread when ``reuse``
            names nobody.
        source: :class:`Source` -- which holders of a prefix :meth:`sources` answers a
            fetch with, behind the pull it already priced.
        block_tokens: tokens per KV block.
        profile / model: the cost constants prediction is priced against.
        decode_pool / prefill_pool: instance subsets (default: all).
        slo_ttft / slo_tbt: what admission holds a decision to (:meth:`_admit`).
        simulate_decode: whether the run models batched decode at all.
        early_rejection: :class:`Occupancy` -- which decode occupancy the TBT SLO is
            judged against.
    """

    def __init__(
        self,
        *,
        reuse: Reuse,
        rank: Rank,
        balance_threshold: float = 1.5,
        source: Source = Source.PREFIX,
        block_tokens: int,
        profile: MachineProfile = DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        decode_pool: Optional[List[str]] = None,
        prefill_pool: Optional[List[str]] = None,
        slo_ttft: float = float("inf"),
        slo_tbt: float = float("inf"),
        simulate_decode: bool = False,
        early_rejection: Occupancy = Occupancy.EARLY,
    ) -> None:
        self.B = block_tokens
        self.profile = profile
        self.model = model
        # Which peer a candidate may pull the prefix from. One winner, asked once per
        # decision; ``LocalOnly`` names nobody, so the baseline reuses only what a host
        # already holds.
        self._reuse = Max(
            _source_ranking(source) if reuse is Reuse.PEERS else LocalOnly()
        )
        # Which peer serves a fetch: the one this plane already priced the pull
        # against, else whoever holds the longest prefix.
        self._fetch = Sort(FirstMatch([RoutedPull(), _source_ranking(source)]))
        #: Which fold orders the prefill pool, and with it whether the queue is
        #: measured at all (:meth:`attach`).
        self._rank = rank
        self._threshold = balance_threshold
        #: The pools the prefill and decode chains rank. A preset that named no
        #: subset leaves them empty until :meth:`attach` knows every instance.
        self.prefill_ids: List[str] = sorted(prefill_pool) if prefill_pool else []
        self.decode_ids: List[str] = sorted(decode_pool) if decode_pool else []
        #: Requests this plane sent to another host, and the answer it sent them with.
        #: Only a decision that *moves* a request is kept: one naming the host that
        #: asked is served in that same call, so no second ask is coming. Every entry
        #: therefore has exactly one reader (:meth:`decide`) and needs no clock to
        #: forget by.
        self._placed: Dict[str, Response] = {}
        self.slo_ttft = slo_ttft
        self.slo_tbt = slo_tbt
        self.tbt_enabled = simulate_decode
        #: Does this run roll decode occupancy forward to prefill completion, counting
        #: the promised prefills that will have landed by then? Both modes hold a
        #: decision to the same two SLOs and differ only in what feeds them. Read in
        #: three places -- composing the reservation sensor, writing it, predicting off
        #: it -- so no two of them can disagree about whether this run predicts.
        self._lookahead = simulate_decode and early_rejection is Occupancy.PREDICT
        # All built in attach(), where the sensors they read and the pools they rank
        # exist.
        self.view: Optional[KVView] = None
        self._prefill: Optional[Selector[PrefillAsk]] = None
        self._decode: Optional[Selector[Plan]] = None
        #: Where a host's facts arrive, and the only thing that writes any sensor here.
        self.dispatcher: Optional[Dispatcher] = None

    # -- the stack hands over its ports ----------------------------------- #
    def attach(self, view) -> None:
        """Declare this plane's four decisions and build the sensors they read.

        The four: which peer a pull comes from, which peer serves a fetch, which host
        prefills, which host decodes. Two need a pool this method resolves, and the
        winner axis is spent here rather than at select time.

        Every sensor is built here, never supplied. One shared with a second plane
        would have each answering for the other's decisions: a shared cluster sensor
        reports every host idle, a shared routed-pull sensor answers every fetch "I
        decided nothing about this", a shared reservation sensor leaves every predicted
        batch short.

        The reservation sensor exists only in a run that rolls occupancy forward, so
        elsewhere the same :class:`~kvcache_sim.control._sensor.Committed` reserves
        nothing and no fold has an empty sensor to read.
        """
        ids = sorted(view.topology)
        # Held here, not read back off the view: a view raises for a sensor this run
        # composed without, so ``reserved`` could not be tested for ``None`` there.
        sensors = dict(
            # Over ALL instances: the prefill and decode pools may each be a subset.
            cluster=ClusterSensor(ids),
            reserved=ReservationSensor() if self._lookahead else None,
            routed=RoutedPullSensor(),
            load=SourceLoad(),
        )
        self.view = view.derived(KVView, **sensors)
        self.dispatcher = Dispatcher()
        for sensor in sensors.values():
            # ``None`` is a sensor this run does not hold, so nothing folds for it.
            if sensor is not None:
                self.dispatcher.compose(sensor)
        # Every instance, unless the preset named a subset to rank.
        self.prefill_ids = self.prefill_ids or ids
        self.decode_ids = self.decode_ids or ids
        # Which host decodes: every instance in the decode pool, keyed at the batch it
        # would be holding when this request's prefill lands.
        self._decode = Sort(DecodeBatch(
            self.decode_ids,
            tbt_enabled=self.tbt_enabled,
            lookahead=self._lookahead,
            profile=self.profile,
            model=self.model,
        ))
        # Which host prefills: every instance in the prefill pool, keyed at what
        # serving the request there would cost -- queue, then transfer, then prefill --
        # with the peer the reuse ranking named priced in against recomputing.
        priced = Priced(
            self.prefill_ids,
            block_tokens=self.B,
            profile=self.profile,
            model=self.model,
            threshold=self._threshold,
        )
        if self._rank is Rank.LOAD:
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

        for ranking in (self._fetch, self._reuse, self._prefill, self._decode):
            ranking.attach(declared(self.view, ranking))

    # -- what a serving host asks, at the two moments it has a question ------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection:
        """Which peers should serve ``requester``'s fetch of ``keys``, best first.

        The pull :meth:`decide` already priced for this caller
        (:class:`~kvcache_sim.control._selector.RoutedPull`), else whoever holds the
        longest prefix. ``Selection.of([])`` names nobody, leaving the read to the
        directory's order.

        Answering is dispatched unconditionally, whichever link answered
        (:class:`~kvcache_sim.control._sensor.FetchAnswered`): a fetch a pull was priced
        for spends that memo, one nothing priced spends nothing. Nothing between the
        chain's answer and the dispatch suspends, so no second fetch of the same keys
        can read the memo this one answered from.
        """
        answer = self._fetch.select(list(keys), requester)
        self.dispatcher.dispatch_sync(FetchAnswered(requester, tuple(keys)))
        return await answer.settled()

    async def decide(self, request: Request, requester: str) -> Optional[Response]:
        """Where should ``request`` run? Both selections, or ``None`` if refused.

        The decode side is chosen against the *winning* prefill candidate's predicted
        completion, so the second selection is asked once and after the first. ``None``
        is an SLO miss: no host this request may run on, and nothing has run.

        Nothing that ranks, prices or holds a candidate host to an SLO reads
        ``requester``. Where a request should run is a fact about the cluster, not
        about who was asked, so two hosts asking about one request are answered the
        same way by an unchanged cluster. Only the reuse ranking has a use for it, as
        the answer to its own question, "who wants these bytes".

        Priced **once per request**, however many hosts it is passed through: a
        decision that moves it is recorded as it is booked (:attr:`_placed`), and the
        ask from the host it names is answered with the recording. Pricing again would
        read a cluster this decision has already booked, find the host just chosen
        busier, and move the answer -- a request rerouted by its own booking. Judging
        again would also refuse a request whose prefill has by then run.
        """
        placed = self._placed.pop(request.id, None)
        if placed is not None:
            return placed

        now = self.view.now()
        keys = list(request.block_keys)
        with self.view.pinned(keys):
            prefill = self._prefill.select(
                PrefillAsk(
                    request=request,
                    now=now,
                    keys=keys,
                    counts=self.view.prefix_lengths(keys),
                    peer=self._reuse.select(keys, requester),
                ),
                requester,
            )
            # The winning plan, read once off the dimension it rides in, so both halves
            # of the answer are formed against the same one.
            plan: Plan = prefill.key[prefill.head][0]
            decode = self._decode.select(plan, requester)

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

        Assembled before the SLOs are checked, so the value they judge is the value
        the answer carries. ``None`` is a rejection, and costs nothing: this runs
        before the prefill does.

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
        # Accepted: one action, folded into every sensor it moves. Each fold writes its
        # own sensor and reads no other, so their order is unobservable. Booked at the
        # instant the decision is made, so nothing decided after it prices against a
        # queue that does not hold this request yet.
        #
        # Synchronous, not the endpoint a host reports over: nothing in this method may
        # suspend, or a second decision could interleave with a half-committed one.
        self.dispatcher.dispatch_sync(Committed(response, request.output_tokens))
        if response.prefill != requester:
            # Moved elsewhere: remember the answer for the one ask that follows. Kept
            # off the commit, since nothing outside this plane reads it.
            self._placed[request.id] = response
        return response


# -- the two schedulers, as the settings that make them ---------------------- #
# One choice on each axis, no behaviour of their own. A third combination is
# composed, not subclassed.


class LoadBalanceScheduler(_Scheduler):
    """Baseline (~vLLM): least-loaded instance, local-only cache reuse."""

    def __init__(self, **knobs: Any) -> None:
        super().__init__(reuse=Reuse.NONE, rank=Rank.LOAD, **knobs)


class CacheAwareScheduler(_Scheduler):
    """Cache-aware: global prefix-match routing under a balance threshold.

    Args:
        replicate: whether a candidate may pull a prefix from a peer at all. When it
            may, the reuse axis is the ``source`` ranking itself, so the peer named
            while pricing is the peer that serves the read. ``replicate=False``
            (which isolates replication's contribution in the demo) prices against
            nobody, and a fetch is still answered from that ranking.
    """

    def __init__(self, *, balance_threshold: float = 1.5, replicate: bool = True,
                 **knobs: Any) -> None:
        super().__init__(
            reuse=Reuse.PEERS if replicate else Reuse.NONE,
            balance_threshold=balance_threshold,
            rank=Rank.TTFT,
            **knobs,
        )
