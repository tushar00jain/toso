"""One scheduler, two presets: ``LoadBalanceScheduler`` / ``CacheAwareScheduler``.

Both are a :class:`~proposed.policy.Placement`, whose one member -- ``select`` --
is the whole surface the data plane may *ask*. Beside either runs a second control
plane -- a chain headed by :class:`RoutedPull` -- which is what the *directory*
consults; a run installs the two together
(:func:`kvcache_sim.workload._serving.scheduler`) and they meet only at the
cluster model.

This application's one question::

    await select(request, me) -> Selection    # prefill hosts, best first,
                                              # payload {host: Plan}

The ranking is real -- every prefill instance is priced -- and the winner's
:class:`Plan` rides in the payload, so a caller wanting the decision reads
:attr:`~proposed.policy.Selection.winner` and never indexes the ranking. A refusal
(an SLO miss) is the abstention every selector here uses, ``Selection.of([])``.

A plan names both of the request's hosts, so it is asked once and before anything
runs -- which is what makes a refusal cost nothing.

What a host *reports* is not asked and is not answered, so it does not come here:
a fact goes to the model it corrects (:mod:`kvcache_sim.control._cluster`), which
this scheduler reads and the run gives a service of its own.

The question is about *compute*; data placement is not asked here -- the serving
host already knows which blocks it computed, and a volume that runs out of room
drops its own coldest keys and tells the directory afterwards
(:mod:`realsim.seams._retention`). Every argument and return is a value.

Both names are *presets* of one scheduler, parameterized on two axes, each a
:class:`~proposed.policy.Placement`: **reuse**, consulted per candidate for "name a
peer to pull a prefix from, or nobody", and **the ranking**, asked which of the
priced candidates wins.

* ``LoadBalanceScheduler`` (baseline, ~vLLM) = never pull, least-loaded instance:
  reuse only that instance's **local** cache, whatever a peer may hold.
* ``CacheAwareScheduler`` = pull under a balance threshold, lowest predicted TTFT,
  over the **global** prefix-match directory. Which peer serves the gap is a policy
  again (:mod:`kvcache_sim.control._source`).

Admission is a setting rather than an axis: ``early_rejection`` names what the TBT
gate is fed -- the decode occupancy of the moment (``early``) or the occupancy
predicted at prefill completion (``predict``).

Control's model of the cluster
------------------------------
Nothing here executes, and nothing here is a live read. Every host this scheduler
ranks, prices or gates, it judges against one
:class:`~kvcache_sim.control._cluster.KVClusterModel` -- the predicted prefill
queues and the observed decode batches, and what keeps each of them true.

The TTFT the metrics record is therefore the prediction, not a measurement (the
README says why). Prefill cost is deterministic, so on the default path the two
agree; an evicted block, a pull served by another volume, or ``contention`` can
each move the executed cost off it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from proposed import Endpoint, Placement, Policy, Selection

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, MachineProfile, Model,
    prefill_time,
)
from proposed import TransferCost

from ._cluster import (
    Committed, ComputeBusy, DecodeState, KVClusterModel, PrefillFinished,
)
from ._view import KVView
from ._source import LongestPrefixPolicy
from .request import Request

__all__ = [
    "PrefillFinished",
    "ComputeBusy",
    "DecodeState",
    "Plan",
    "RoutedPull",
    "predicts_decode",
    "LoadBalanceScheduler",
    "CacheAwareScheduler",
]


def predicts_decode(simulate_decode: bool, early_rejection: str) -> bool:
    """Does this run roll decode occupancy forward to prefill completion?

    ``"predict"`` counts the prefills promised that will have landed by then, which
    moves where decode lands even with no SLO to miss -- a fidelity setting of the
    model rather than an admission rule, which is why the model is built with it and
    the scheduler installs the same gates either way.

    Here rather than inside the scheduler because whoever builds the model early
    (:func:`kvcache_sim.workload._serving.scheduler`) answers the same question.
    """
    return simulate_decode and early_rejection == "predict"


# -- what this application's control plane answers with ---------------------- #
# The answer is here; the facts a host reports are with the model they write
# (:mod:`kvcache_sim.control._cluster`).


@dataclass
class Plan:
    """A routing decision for one request (or a rejection when ``None``)."""

    request: Request
    prefill: str                 # chosen prefill instance id
    decode: str                  # chosen decode instance id
    match_blocks: int            # reused prefix length (blocks)
    cached_tokens: int
    uncached_tokens: int
    reuse_source: Optional[str]  # remote instance a prefix gap is pulled from
    transfer_bytes: int
    queue_wait: float
    ttft: float                  # time-to-first-token (queue + transfer + prefill)
    done_time: float             # absolute sim time prefill completes
    decode_done: float
    prefill_t: float = 0.0       # prefill compute duration
    transfer_t: float = 0.0      # predicted remote-pull fetch duration
    pull_keys: List[str] = field(default_factory=list)  # gap blocks to fetch
    pred_tbt: float = 0.0        # predicted time-between-tokens at admission
    pred_batch: int = 0          # predicted decode batch size at admission

    @property
    def local_blocks(self) -> int:
        """Blocks the prefill host already held: the match, minus what it pulls.

        Derived rather than a field: the data plane needs it three times over (reuse
        to report, suffix to publish, prefix to fall back on when a planned pull is
        gone) and all three have to agree.
        """
        return self.match_blocks - len(self.pull_keys)


# -- the first axis: which peer a prefix is pulled from ---------------------- #
# Everything a routing decision does is the same whichever scheduler is running.
# What differs is the two axes and which admission gates are installed.
#
# Both axes are :class:`~proposed.policy.Placement`s, because the subject is what
# tells the two selector kinds apart (:mod:`proposed.policy`) and these subjects are
# the application's: this request, and the candidates this scheduler priced. Neither
# is ever installed in the controller. What is not a selection at all is the rest of
# what a scheduler decides -- admission and the SLO gates answer yes or no, and a
# ranked set of sources cannot say that. This first axis answers only the
# store-shaped half -- name a peer to pull from, or nobody -- and the scheduler
# prices what comes back, which is the line :mod:`kvcache_sim.control._source`
# already draws. A placement's ``select`` is awaited inside the decision's pinned
# snapshot (:meth:`_Scheduler.select`), so it runs to completion without
# suspending.


class _LocalOnly(Placement):
    """Name nobody, ever -- the baseline reuses only what a host already holds.

    ``Selection.of([])``, which is :class:`~proposed.policy.FirstMatch`'s
    *abstention*: no source, so the caller recomputes the gap. Deliberately not
    ``Selection()``, which is a decision meaning every holder in directory order.
    """

    name = "local-only"

    async def select(self, keys: Sequence[str], requester: str) -> Selection:
        return Selection.of([])


class _PullLongestPrefix(Placement):
    """Name the best peer when its prefix beats recomputing the gap locally.

    Only when that peer's run is more than ``threshold`` times the requester's own
    -- the balancing threshold. A pull is charged to the prefill instance's queue,
    so one that saves little compute still costs the whole wait, and without the
    threshold every request would chase the longest match onto one instance.

    The peer ranking itself is delegated (:mod:`kvcache_sim.control._source`) and
    only the pull-versus-recompute judgement is here. This ``select`` is consulted
    per candidate while the scheduler prices, and never installed anywhere, unlike
    the store-side chain headed by :class:`RoutedPull`, which answers the *store*
    with the peer this one named.

    Abstains when the ranking's head is the requester itself: a source is a peer,
    and a host does not pull what it already holds -- that prefix is its local
    match, and no shorter peer behind it could be worth pulling either.
    """

    name = "pull-longest-prefix"

    def __init__(self, threshold: float, source_policy: Policy) -> None:
        self.threshold = threshold
        self.source_policy = source_policy

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Sense through the view, and bring up the ranking this defers to."""
        super().attach(view, transfer_cost)
        self.source_policy.attach(view, transfer_cost)

    async def select(self, keys: Sequence[str], requester: str) -> Selection:
        """The one peer worth pulling from, or an abstention.

        Reads the runs off the view rather than being told them: the decision
        pinned it, so this is the same snapshot and costs no directory read.
        """
        counts = self.view.prefix_lengths(list(keys))
        ranked = await self.source_policy.select(keys, requester)
        src = ranked.sources[0] if ranked.sources else None
        if src is None or src == requester:
            return Selection.of([])
        if counts.get(src, 0) <= counts.get(requester, 0) * self.threshold:
            return Selection.of([])  # not worth the transfer: recompute it
        return Selection.of([src])


# -- the second axis: which priced candidate wins ---------------------------- #


def _ranked(candidates: Sequence[Tuple[str, Any]]) -> Selection:
    """``(instance, its price)`` in rank order, as a selection carrying the prices."""
    return Selection.of([i for i, _ in candidates], payload=dict(candidates))


class _Ranking(Placement):
    """Rank candidates the scheduler has already priced, best first.

    Pricing them here instead would pull the reuse placement, the transfer cost and
    the profile in with it, which is most of a scheduler.

    Handed the run's cluster model, since a ranking may want a signal the price has
    already folded the clock into (:class:`_LeastLoaded`). Built where that model is
    (:meth:`_Scheduler.attach`), which is why a preset names the kind rather than
    building one.
    """

    def __init__(self, cluster: KVClusterModel) -> None:
        self.cluster = cluster


class _LeastLoaded(_Ranking):
    """The shortest predicted prefill queue, whatever reuse bought (the baseline).

    Ranks on ``busy_until`` rather than the candidate's ``queue_wait``, which is
    that tail clamped at the clock: two instances idle since different moments both
    wait zero, so the choice would fall to the id tie-break and a different one
    would win. This scheduler's claim is that it picks by load and nothing else.
    """

    name = "least-loaded"

    async def select(self, plans: Sequence[Plan], requester: str) -> Selection:
        tails = self.cluster.busy_until
        ordered = sorted(plans, key=lambda p: (tails[p.prefill], p.prefill))
        return _ranked([(p.prefill, p) for p in ordered])


class _MinTTFT(_Ranking):
    """The lowest predicted queue + transfer + prefill -- why reuse is *priced*
    rather than preferred: a longer match on a busier instance can still lose."""

    name = "min-ttft"

    async def select(self, plans: Sequence[Plan], requester: str) -> Selection:
        ordered = sorted(plans, key=lambda p: (p.ttft, p.prefill))
        return _ranked([(p.prefill, p) for p in ordered])


class _LeastBatch(Placement):
    """The smallest predicted decode batch (instance id tie-break).

    The other host pick of a routing decision, and the same shape as the axis above:
    the scheduler predicts each candidate's batch
    (:meth:`_Scheduler._predicted_batch`) and this ranks what came back. Not an axis
    -- both presets pick a decode instance this way.
    """

    name = "least-batch"

    async def select(
        self, batches: Sequence[Tuple[str, int]], requester: str
    ) -> Selection:
        return _ranked(sorted(batches, key=lambda c: (c[1], c[0])))


#: An admission gate: does this plan clear one SLO? Installed as a list --
#: :attr:`_Scheduler._gates` -- rather than tested behind a mode string, so no
#: decision below has a mode to branch on. A gate holds no settings of its own, so
#: it is handed the model it gates against.
_Gate = Callable[[Plan, "_Scheduler"], bool]


def _ttft_gate(plan: Plan, sched: "_Scheduler") -> bool:
    """The TTFT SLO. The one gate every run installs."""
    return plan.ttft <= sched.slo_ttft


def _predicted_tbt_gate(plan: Plan, sched: "_Scheduler") -> bool:
    """The TBT SLO, on the batch this request is predicted to join.

    Runs at routing, before the prefill: what it refuses costs nothing, which is the
    whole of why admission is decided here and not once the compute has been spent.
    """
    return plan.pred_tbt <= sched.slo_tbt


class RoutedPull(Policy):
    """The peer a fetch's pull was already priced against, or an abstention.

    Answering the store from what routing decided, rather than deciding twice:
    re-deriving would not even agree (routing ranks over the request's whole block
    chain, the fetch names only the gap), and naming a different holder would
    charge a cross-node read for a same-node prediction. A caller with no routed
    pull falls through to the ranking behind this link.

    Reading it **consumes** it (:meth:`~kvcache_sim.control._cluster.KVClusterModel.claim`
    expires the record on a match), so this belongs at the head of a
    :class:`~proposed.policy.FirstMatch` chain and under no combinator that can drop
    an answer. In that one position spending and using coincide: a link that answers
    wins the chain, an abstention matched nothing and spends nothing. Under one that
    could reject the peer, the record would be gone and the fetch would fall through
    to a ranking that never saw it.

    The model is the whole of what this shares with the scheduler that priced the
    pull: one writes it, the other consumes it, and neither holds the other.
    """

    name = "routed-pull"

    def __init__(self, cluster: KVClusterModel) -> None:
        self._cluster = cluster

    async def select(self, keys: Sequence[str], requester: str) -> Selection:
        peer = self._cluster.claim(requester, keys)
        return Selection.of([peer] if peer is not None else [])


class _Scheduler(Placement):
    """The pricing, ranking and admission both schedulers share.

    A :class:`~proposed.policy.Placement`, and only that: a serving host asks it as
    a service (:meth:`select`). What the *directory* is told is a second control
    plane the run installs beside this one
    (a chain headed by :class:`RoutedPull`), and the two meet at the model -- this
    one prices a pull and records it, that one answers the fetch with it -- so
    nothing is threaded through the data plane to carry it and neither object holds
    the other.

    Args:
        reuse / rank: the two axes a preset picks (see above), both
            :class:`~proposed.policy.Placement`s this object consults while forming
            the one answer :meth:`select` gives. ``rank`` is the *kind*, since the
            load it reads exists only once :meth:`attach` has run.
        block_tokens: tokens per KV block.
        profile / model: the cost constants prediction is priced against.
        decode_pool / prefill_pool: instance subsets (default: all).
        slo_ttft / slo_tbt: what the gates hold a plan to.
        simulate_decode: whether the run models batched decode at all.
        early_rejection: ``"early"`` | ``"predict"`` -- whether the decode occupancy
            the TBT gate is fed is the one observed now or the one predicted at
            prefill completion.
        source_policy: ranks peers for a prefix pull
            (default :class:`~kvcache_sim.control._source.LongestPrefixPolicy`).
        cluster: the run's one :class:`~kvcache_sim.control._cluster.KVClusterModel`,
            when a caller is building the store-side plane against it too and so
            has to make it first. ``None`` -- the default -- builds it in
            :meth:`attach`, where the instances become known.
    """

    def __init__(
        self,
        *,
        reuse: Placement,
        rank: Type[_Ranking],
        block_tokens: int,
        profile: MachineProfile = DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        decode_pool: Optional[List[str]] = None,
        prefill_pool: Optional[List[str]] = None,
        slo_ttft: float = float("inf"),
        slo_tbt: float = float("inf"),
        simulate_decode: bool = False,
        early_rejection: str = "early",
        source_policy: Optional[Any] = None,
        cluster: Optional[KVClusterModel] = None,
    ) -> None:
        self.B = block_tokens
        self.profile = profile
        self.model = model
        self.source_policy = (
            source_policy if source_policy is not None else LongestPrefixPolicy()
        )
        self._reuse = reuse
        self._rank_kind = rank
        self._decode_rank = _LeastBatch()
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
                f"unknown early_rejection {early_rejection!r}: the choice is what "
                f"the TBT gate is fed, 'early' or 'predict'"
            )
        # Which gates this run installs. A run that does not model decode has no
        # batch to hold to a TBT SLO, so it installs only the TTFT one.
        self._gates: List[_Gate] = [_ttft_gate]
        if simulate_decode:
            self._gates.append(_predicted_tbt_gate)
        # The admission mode is spent here and never read again: both modes install
        # the same gates (above) and differ only in what the model feeds them.
        self._lookahead = predicts_decode(simulate_decode, early_rejection)
        # Filled by attach(): the run knows its servers only once its stack exists.
        self.transfer_cost: Optional[TransferCost] = None
        self.topo: Dict[str, Endpoint] = {}
        self.ids: List[str] = []
        self.prefill_ids: List[str] = []
        self.decode_ids: List[str] = []
        self.cluster: Optional[KVClusterModel] = cluster
        self._rank: Optional[_Ranking] = None

    # -- the stack hands over its ports ----------------------------------- #
    def attach(self, view, transfer_cost: TransferCost) -> None:
        """Receive the ports this control plane senses and prices through.

        Two-phase so a scenario can declare a control plane as an object
        (``MyControl(knobs)``) and let the run hand it the stack afterwards.

        The view is upgraded to a :class:`~kvcache_sim.control._view.KVView` here:
        prefix runs are this capability's notion, not the store's.

        The run's one :class:`~kvcache_sim.control._cluster.KVClusterModel` is built
        here unless a caller made it first (``cluster``): this is where the
        instances become known, and this runs once per run, so nothing else is
        placed to build a second (empty) one -- and an empty one would report every
        host idle, which is a run that looks healthy and is wrong. Everything that
        ranks, prices or gates is handed that one model.
        """
        self.view = KVView(view.directory, view.topology)
        # A protocol rather than a simulator function: a deployment supplies its own
        # measured numbers.
        self.transfer_cost = transfer_cost
        self.topo = dict(view.topology)
        self.ids = sorted(self.topo)
        if self.cluster is None:
            # Over ALL instances: the prefill and decode pools may each be a subset.
            self.cluster = KVClusterModel(self.ids, lookahead=self._lookahead)
        self._rank = self._rank_kind(self.cluster)
        self.prefill_ids = (
            sorted(self._prefill_pool) if self._prefill_pool else self.ids
        )
        self.decode_ids = (
            sorted(self._decode_pool) if self._decode_pool else self.ids
        )
        # Every selector this one consults senses through the same view, which is
        # what lets one routing decision pin one directory snapshot for all of them
        # (:meth:`~kvcache_sim.control._view.KVView.pinned`).
        #
        # ``source_policy`` is also a link of the store-side chain, which the run
        # attaches to the plain view it hands every control plane. This attach is
        # what leaves it sensing through the pinned one, so a run listing the two
        # planes puts this one **last** -- a source ranking a decision consults
        # inside its pin must not read past the snapshot into the live directory.
        for selector in (self._reuse, self._rank, self._decode_rank):
            selector.attach(self.view, transfer_cost)
        self.source_policy.attach(self.view, transfer_cost)

    # -- proposed.Placement: one member, for the question ------------------ #
    async def select(self, request: Request, requester: str) -> Selection:
        """Where should ``request`` run? Price every prefill instance, rank them.

        The prefill hosts best first, with each one's :class:`Plan` under its id in
        :attr:`~proposed.policy.Selection.payload`; the head's plan is the one that
        was admitted and committed, so a caller reads
        :attr:`~proposed.policy.Selection.winner`. An SLO miss abstains
        (``Selection.of([])``) -- there is no host this request may run on, and a
        refusal costs nothing because nothing has run.

        Atomic: every read is off one pinned directory snapshot
        (:meth:`~kvcache_sim.control._view.KVView.pinned`) and one clock read, so the
        prices are comparable, and the bookkeeping it writes needs nothing locked.
        Pricing a candidate reserves nothing (:meth:`_candidate`), so the losers leave
        no trace and the winner is chosen after the whole field is known.

        ``requester`` is the host the request landed on, which no part of this
        decision reads: where a request *should* run is a fact about the cluster,
        not about who was asked.
        """
        now = self.view.now()
        keys = list(request.block_keys)
        with self.view.pinned(keys):
            counts = self.view.prefix_lengths(keys)
            plans: List[Plan] = []
            for inst in self.prefill_ids:
                chosen = await self._reuse.select(keys, inst)
                match, src, pull = self._priced_reuse(counts, keys, inst, chosen)
                plans.append(self._candidate(
                    request, inst, now, match=match, source=src, pull_keys=pull
                ))
            ranked = await self._rank.select(plans, request.id)
        admitted = await self._admit(ranked.winner)
        return ranked if admitted is not None else Selection.of([])

    @staticmethod
    def _priced_reuse(
        counts: Dict[str, int], keys: Sequence[str], inst: str, chosen: Selection
    ) -> Tuple[int, Optional[str], Sequence[str]]:
        """What a reuse selection buys ``inst``: ``(match, source, pull_keys)``.

        Derived here and not in the policy, because naming a peer is where a
        policy's job ends: how much of this prompt that peer's prefix covers is
        arithmetic over the snapshot this scheduler already read. An abstention --
        and a selection naming ``inst`` itself, which is not a pull -- leaves the
        local match to recompute from.
        """
        local = counts.get(inst, 0)
        src = chosen.sources[0] if chosen.sources else None
        if src is None or src == inst:
            return local, None, ()
        return counts[src], src, keys[local:counts[src]]

    # -- prediction (no mutation) ---------------------------------------- #
    def _predict(self, inst: str, now: float, transfer_t: float, prefill_t: float):
        """Return ``(queue_wait, ttft, done_time)`` without reserving the server."""
        avail = max(now, self.cluster.busy_until[inst])
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
        (:class:`~kvcache_sim.control._cluster.Committed` records a decision actually
        taken). Which candidate wins is the ranking's business (:class:`_Ranking`).
        """
        cached = min(match * self.B, request.prompt_tokens)
        uncached = request.prompt_tokens - cached
        prefill_t = prefill_time(uncached, self.profile, self.model)
        if source is not None and pull_keys:
            xbytes = self.model.block_bytes(len(pull_keys), self.B)
            # The cost model the transport charges, so this prediction is what the
            # real pull will cost.
            transfer_t = self.transfer_cost.get_time(source, inst, xbytes)
        else:
            source, xbytes, transfer_t = None, 0, 0.0
        queue_wait, ttft, done = self._predict(inst, now, transfer_t, prefill_t)
        plan = Plan(
            request, inst, "", match, cached, uncached, source, xbytes,
            queue_wait, ttft, done, 0.0,
        )
        plan.prefill_t = prefill_t
        plan.transfer_t = transfer_t
        plan.pull_keys = list(pull_keys)
        return plan

    # -- decode-side TBT prediction / admission -------------------------- #
    def _predicted_batch(self, d: str, done_time: float) -> int:
        """Predicted decode batch size on ``d`` seen by a request admitted at
        ``done_time`` (its prefill completion). Drives TBT prediction."""
        if not self.tbt_enabled:
            return 0
        if not self._lookahead:
            return self.cluster.occupancy(d)
        n = self.cluster.predict_occupancy(d, done_time)
        # Requests whose prefill has not landed are invisible to the observed
        # decode state; the outstanding reservations stand in for them.
        for res in self.cluster.pending(self.view.now()):
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

    async def _select_decode(self, plan: Plan) -> Tuple[str, int]:
        """Where ``plan`` decodes, and the batch it was predicted to meet there.

        Priced here and ranked by :class:`_LeastBatch`. With decode unmodelled every
        candidate prices at zero, so the ranking's id tie-break is the whole of the
        choice.
        """
        batches = [
            (d, self._predicted_batch(d, plan.done_time)) for d in self.decode_ids
        ]
        chosen = await self._decode_rank.select(batches, plan.request.id)
        d = chosen.sources[0]
        return (d, chosen.payload[d])

    async def _admit(self, plan: Plan) -> Optional[Plan]:
        """Give the won candidate its decode instance, gate it, commit it.

        The decode side is chosen once, against the winning candidate's predicted
        prefill completion -- not per candidate in the loop. ``None`` == rejected,
        and rejected here costs nothing: this runs before the prefill does.
        """
        plan.decode, plan.pred_batch = await self._select_decode(plan)
        plan.pred_tbt = (
            decode_step_time(plan.pred_batch + 1, self.profile, self.model)
            if self.tbt_enabled
            else 0.0
        )
        if not all(gate(plan, self) for gate in self._gates):
            return None
        # Accepted: the model holds the instances this plan spoke for, and remembers
        # the peer its pull was priced against for when the directory asks
        # (:class:`_RoutedPull`).
        #
        # The local write, not the endpoint a host reports over: control is in the
        # model's process, and a plain call is what keeps this decision atomic.
        self.cluster._notify_impl(Committed(plan))
        return plan


# -- the two schedulers, as the settings that make them ---------------------- #
# Names rather than behaviours: everything either one does is above, and each is
# one choice on each axis. A third combination is composed, not subclassed.


class LoadBalanceScheduler(_Scheduler):
    """Baseline (~vLLM): least-loaded instance, local-only cache reuse."""

    def __init__(self, **knobs: Any) -> None:
        super().__init__(reuse=_LocalOnly(), rank=_LeastLoaded, **knobs)


class CacheAwareScheduler(_Scheduler):
    """Cache-aware: global prefix-match routing under a balance threshold.

    ``replicate=False`` (which isolates replication's contribution in the demo) is
    never pulling at all, so it is the baseline's reuse placement rather than a flag
    the candidate loop tests.
    """

    def __init__(self, *, balance_threshold: float = 1.5, replicate: bool = True,
                 **knobs: Any) -> None:
        # One source policy, held by the scheduler (which answers the directory
        # with the peer it chose) and by the reuse placement (which chose it).
        source = knobs.pop("source_policy", None) or LongestPrefixPolicy()
        super().__init__(
            reuse=(
                _PullLongestPrefix(balance_threshold, source) if replicate
                else _LocalOnly()
            ),
            rank=_MinTTFT,
            source_policy=source,
            **knobs,
        )
