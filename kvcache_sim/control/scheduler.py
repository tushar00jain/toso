"""One scheduler, two presets: ``LoadBalanceScheduler`` / ``CacheAwareScheduler``.

Both are a plain :class:`~proposed.plane.ControlPlane`, and one plane is the whole of
what this capability decides. Two members, asked at the two moments a serving host
has a question::

    await decide(request, me)   -> Optional[Response]   # where should this run
    await sources(keys, me)     -> Selection[None]      # who serves this fetch

The second exists because the first already answered it: routing prices a pull
against recomputing and records the peer it priced, and the fetch that follows asks
which peer to read from. Two objects would have to keep that record in step across a
boundary; one plane just reads its own model (:class:`RoutedPull`).

Not a selector, because the decision is made of **two** selections and a selection
holds one: the prefill hosts this scheduler priced, and the decode hosts it ranked
against the winner among them. Both rankings are real, and both stay here -- what
leaves is the winner of each and the price of the one that won (:class:`Response`),
since nothing outside asks what lost. A refusal (an SLO miss) is ``None``.

One ask settles both halves, before anything runs -- which is what makes a refusal
cost nothing.

What a host *reports* is not asked and is not answered, so it does not come here:
a fact goes to the model it corrects (:mod:`kvcache_sim.control._cluster`), which
this scheduler reads and the run gives a service of its own.

The question is about *compute*; data placement is not asked here -- the serving
host already knows which blocks it computed, and a volume that runs out of room
drops its own coldest keys and tells the directory afterwards
(:mod:`realsim.seams._retention`). Every argument and return is a value.

Both names are *presets* of one scheduler, parameterized on two axes, neither of
them reachable from outside it: **reuse**, a selector consulted per candidate for
"name a peer to pull a prefix from, or nobody", and **the rank key**, which sorts
the candidates this scheduler priced.

* ``LoadBalanceScheduler`` (baseline, ~vLLM) = never pull, least-loaded instance:
  reuse only that instance's **local** cache, whatever a peer may hold.
* ``CacheAwareScheduler`` = pull under a balance threshold, lowest predicted TTFT,
  over the **global** prefix-match directory. Which peer serves the gap is a selector
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from proposed import ControlPlane, Endpoint, Key, KeySelector, Selection, VolumeId
from proposed.selector import (
    AbstainOnSelf, KeySelectorChain, Refine, Refinement, Selector, TakeHead,
)

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, MachineProfile, Model,
    prefill_time,
)
from proposed import TransferCost

from ._cluster import (
    Committed, ComputeBusy, DecodeState, KVClusterModel, PrefillFinished,
)
from ._source import LongestPrefixKeySelector
from ._view import KVView
from .request import Request

__all__ = [
    "PrefillFinished",
    "ComputeBusy",
    "DecodeState",
    "Plan",
    "Response",
    "RoutedPull",
    "FetchRouting",
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
    """What prefilling one request on one instance was priced at.

    One candidate's price and nothing else: which instance this is, and which one
    decodes, are the two selections' winners and live on the :class:`Response`. Every
    field here is about the prefill, so a losing candidate is a complete value too.
    """

    match_blocks: int            # reused prefix length (blocks)
    cached_tokens: int
    uncached_tokens: int
    reuse_source: Optional[str]  # remote instance a prefix gap is pulled from
    transfer_bytes: int
    queue_wait: float
    ttft: float                  # time-to-first-token (queue + transfer + prefill)
    done_time: float             # absolute sim time prefill completes
    prefill_t: float = 0.0       # prefill compute duration
    transfer_t: float = 0.0      # predicted remote-pull fetch duration
    pull_keys: List[str] = field(default_factory=list)  # gap blocks to fetch

    @property
    def local_blocks(self) -> int:
        """Blocks the prefill host already held: the match, minus what it pulls.

        Derived rather than a field: the data plane needs it three times over (reuse
        to report, suffix to publish, prefix to fall back on when a planned pull is
        gone) and all three have to agree.
        """
        return self.match_blocks - len(self.pull_keys)


@dataclass(frozen=True)
class Response:
    """Where one request runs: the winner of each selection, and the price of one.

    What :meth:`_Scheduler.decide` answers and the only part of a decision that
    travels. The rankings behind it stay inside the scheduler: nothing outside asks
    what lost.

    Args:
        prefill / decode: the two instances chosen, one from each selection.
        plan: what prefilling on ``prefill`` was priced at.
        pred_batch / pred_tbt: the decode batch this request was predicted to meet
            and the inter-token gap that implies. Read by the TBT gate, which is why
            they are here and not on the plan -- they are the decode side's.
    """

    prefill: VolumeId
    decode: VolumeId
    plan: Plan
    pred_batch: int = 0
    pred_tbt: float = 0.0


# -- the first axis: which peer a prefix is pulled from ---------------------- #
# Everything a routing decision does is the same whichever scheduler is running.
# What differs is the two axes and which admission gates are installed.
#
# Only this first axis is a selector at all, and it answers the store-shaped half
# -- name a peer to pull from, or nobody. So it is the source ranking
# (:mod:`kvcache_sim.control._source`, a real KeySelector) under the one test that is
# not the store's: is pulling worth more than recomputing here. That composition is a
# :class:`~proposed.selector.Refine`, so the composition lives in the object holding
# both rather than inside either. Its ``select`` is awaited inside the pinned snapshot
# (:meth:`_Scheduler._select_prefill`), so it runs to completion without suspending.
#
# The second axis is below, and is a sort key rather than a selector: it ranks what
# this scheduler already priced, reads no directory and answers nobody. What is not
# a selection at all is the rest of what a scheduler decides -- admission and the
# SLO gates answer yes or no, and a ranked set of sources cannot say that.


class _LocalOnly(Selector[Sequence[Key], None]):
    """Name nobody, ever -- the baseline reuses only what a host already holds.

    A plain :class:`~proposed.selector.Selector`: its subject is keys, but the
    scheduler is the only thing that asks it, so it is not fronted by a service at
    all.

    ``Selection.of([])``, which is :class:`~proposed.selector.FirstMatch`'s
    *abstention*: no source, so the caller recomputes the gap. Deliberately not
    ``Selection()``, which is a decision meaning every holder in directory order.
    """

    name = "local-only"

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[None]:
        return Selection.of([])


class _LongerThanLocal(Refinement[Sequence[Key], None]):
    """Abstain unless the head's prefix beats recomputing the gap locally.

    Its run must be more than ``threshold`` times the requester's own -- the
    balancing threshold. A pull is charged to the prefill instance's queue, so one
    that saves little compute still costs the whole wait, and without the threshold
    every request would chase the longest match onto one instance.

    Judges the head and abstains for the whole ranking rather than filtering peer
    by peer. A ranking is not obliged to be in raw-run order
    (:class:`~kvcache_sim.control._source.SpreadReadsKeySelector` discounts a busy
    source), so the sources behind the head are the ones it preferred *less*, and
    promoting one on a longer raw run would overrule the ranking from outside it.

    Reads the runs off the view rather than being told them: the decision pinned
    it, so this is the same snapshot and costs no directory read.
    """

    name = "longer-than-local"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    async def refine(
        self, selection: Selection[None], keys: Sequence[Key], requester: str
    ) -> Selection[None]:
        counts = self.view.prefix_lengths(list(keys))
        head = selection.sources[0]
        if counts.get(head, 0) <= counts.get(requester, 0) * self.threshold:
            return Selection.of([])  # not worth the transfer: recompute it
        return selection


# -- the second axis: which priced candidate wins ---------------------------- #


#: What either ranking here sorts: one candidate as the pair a
#: :class:`~proposed.selector.Selection` is built out of -- the instance, and what
#: this scheduler priced it at. A :class:`Plan` when the choice is which host
#: prefills, a predicted batch size when it is which host decodes.
_Priced = Tuple[VolumeId, Plan]
_Batched = Tuple[VolumeId, int]


#: How a priced candidate sorts: the key ``sorted`` is handed, smallest first.
#: Takes the run's model as its second argument like :data:`_Gate` does, since both
#: judge one candidate against the cluster, and neither is reachable from outside
#: this scheduler, so neither is a selector. The instance is read off the candidate
#: rather than out of what it was priced at, so every key can end in it -- which is
#: what makes a rank total and a run reproducible.
_RankKey = Callable[[_Priced, "_Scheduler"], Tuple]


def _by_load(candidate: _Priced, sched: "_Scheduler") -> Tuple:
    """The shortest predicted prefill queue, whatever reuse bought (the baseline).

    Sorts on ``busy_until`` rather than the candidate's ``queue_wait``, which is
    that tail clamped at the clock: two instances idle since different moments both
    wait zero, so the choice would fall to the id tie-break and a different one
    would win. This scheduler's claim is that it picks by load and nothing else.
    """
    instance, _ = candidate
    return (sched.cluster.busy_until[instance], instance)


def _by_ttft(candidate: _Priced, sched: "_Scheduler") -> Tuple:
    """The lowest predicted queue + transfer + prefill.

    Why reuse is *priced* rather than preferred: a longer match on a busier
    instance can still lose.
    """
    instance, plan = candidate
    return (plan.ttft, instance)


def _by_batch(candidate: _Batched) -> Tuple:
    """The smallest predicted decode batch, instance id breaking the tie.

    The other host pick of a routing decision, over a predicted batch rather than a
    plan. Not an axis -- both presets pick a decode instance this way, so the
    scheduler applies it directly instead of being handed it, which is why this one
    needs no model.
    """
    instance, batch = candidate
    return (batch, instance)


#: An admission gate: does this decision clear one SLO? Installed as a list --
#: :attr:`_Scheduler._gates` -- rather than tested behind a mode string, so no
#: decision below has a mode to branch on. A gate holds no settings of its own, so
#: it is handed the model it gates against.
_Gate = Callable[["Response", "_Scheduler"], bool]


def _ttft_gate(response: "Response", sched: "_Scheduler") -> bool:
    """The TTFT SLO. The one gate every run installs."""
    return response.plan.ttft <= sched.slo_ttft


def _predicted_tbt_gate(response: "Response", sched: "_Scheduler") -> bool:
    """The TBT SLO, on the batch this request is predicted to join.

    Runs at routing, before the prefill: what it refuses costs nothing, which is the
    whole of why admission is decided here and not once the compute has been spent.
    """
    return response.pred_tbt <= sched.slo_tbt


class RoutedPull(KeySelector[None]):
    """The peer a fetch's pull was already priced against, or an abstention.

    Answering the fetch from what routing decided, rather than deciding twice:
    re-deriving would not even agree (routing ranks over the request's whole block
    chain, the fetch names only the gap), and naming a different holder would
    charge a cross-node read for a same-node prediction. A caller with no routed
    pull falls through to the ranking behind this link.

    Reading it **consumes** it (:meth:`~kvcache_sim.control._cluster.KVClusterModel.claim`
    expires the record on a match), so this belongs at the head of a
    :class:`~proposed.selector.FirstMatch` chain and under no combinator that can drop
    an answer. In that one position spending and using coincide: a link that answers
    wins the chain, an abstention matched nothing and spends nothing. Under one that
    could reject the peer, the record would be gone and the fetch would fall through
    to a ranking that never saw it.
    """

    name = "routed-pull"

    def __init__(self, cluster: KVClusterModel) -> None:
        self._cluster = cluster

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[None]:
        peer = self._cluster.claim(requester, keys)
        return Selection.of([peer] if peer is not None else [])


class FetchRouting(KeySelectorChain[None]):
    """How :meth:`_Scheduler.sources` answers: the priced pull, else the ranking.

    The pull this scheduler already priced for the caller (:class:`RoutedPull`), else
    ``source``'s ranking of whoever holds the longest prefix.

    A chain rather than an ``if``, so the fall-through is
    :class:`~proposed.selector.FirstMatch`'s abstention rule and not a second copy of
    it here, and a :class:`~proposed.selector.KeySelectorChain` because the subject is
    keys and both links select over them. A utility the plane holds, not a plane: it
    ranks, and the plane is what a caller reaches.

    Neither link gates on anything, so what it answers is already values.

    Args:
        cluster: the run's model, which this scheduler records a priced pull into.
        source: ranks the holders of a prefix. The same object the reuse axis names a
            peer with, so the peer chosen while pricing is the peer this answers with.
    """

    name = "fetch-routing"

    def __init__(self, cluster: KVClusterModel, source: KeySelector[None]) -> None:
        super().__init__([RoutedPull(cluster), source])


class _Scheduler(ControlPlane):
    """The pricing, ranking and admission both schedulers share.

    A :class:`~proposed.plane.ControlPlane` and nothing more specific: a serving host
    asks it as a service, and its two members (:meth:`decide`, :meth:`sources`) are
    the whole of its surface. The second is answered from what the first recorded --
    the peer a pull was priced against -- so a plan and the read that carries it out
    cannot disagree, and nothing is threaded through the data plane to keep them
    together.

    Args:
        reuse / rank: the two axes a preset picks (see above), both used while
            forming the one answer :meth:`decide` gives. ``reuse`` is a selector
            and ``rank`` a sort key (:data:`_RankKey`), which is why only the first
            is attached. It names a peer for a candidate's block keys and prices
            nothing (``Selector[Sequence[Key], None]``): what a peer is worth is
            this scheduler's to work out (:meth:`_priced_reuse`).
        source_selector: ranks the holders of a prefix, and what :meth:`sources`
            answers a fetch with behind the pull it already priced
            (:class:`FetchRouting`). ``None`` builds a
            :class:`~kvcache_sim.control._source.LongestPrefixKeySelector`. A pulling
            preset hands the *same* object to its reuse axis, so the peer priced is
            the peer read from.
        block_tokens: tokens per KV block.
        profile / model: the cost constants prediction is priced against.
        decode_pool / prefill_pool: instance subsets (default: all).
        slo_ttft / slo_tbt: what the gates hold a plan to.
        simulate_decode: whether the run models batched decode at all.
        early_rejection: ``"early"`` | ``"predict"`` -- whether the decode occupancy
            the TBT gate is fed is the one observed now or the one predicted at
            prefill completion.
        cluster: the run's one :class:`~kvcache_sim.control._cluster.KVClusterModel`,
            when a caller has to make it first. ``None`` -- the default -- builds it
            in :meth:`attach`, where the instances become known.
    """

    def __init__(
        self,
        *,
        reuse: Selector[Sequence[Key], None],
        rank: _RankKey,
        source_selector: Optional[KeySelector[None]] = None,
        block_tokens: int,
        profile: MachineProfile = DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        decode_pool: Optional[List[str]] = None,
        prefill_pool: Optional[List[str]] = None,
        slo_ttft: float = float("inf"),
        slo_tbt: float = float("inf"),
        simulate_decode: bool = False,
        early_rejection: str = "early",
        cluster: Optional[KVClusterModel] = None,
    ) -> None:
        self.B = block_tokens
        self.profile = profile
        self.model = model
        self._reuse = reuse
        self._rank = rank
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
        # Built in attach(), where the model it reads exists.
        self._fetch: Optional[FetchRouting] = None

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
        self.prefill_ids = (
            sorted(self._prefill_pool) if self._prefill_pool else self.ids
        )
        self.decode_ids = (
            sorted(self._decode_pool) if self._decode_pool else self.ids
        )
        # What a fetch is answered with, over the model just settled above.
        self._fetch = FetchRouting(self.cluster, self._source)
        self._fetch.attach(view, transfer_cost)
        # The reuse axis senses through the same view this one does, which is what
        # lets one routing decision pin one directory snapshot for both
        # (:meth:`~kvcache_sim.control._view.KVView.pinned`). The rank keys need no
        # view: they sort what this scheduler already priced.
        #
        # **Second, and that is load-bearing.** A reuse axis that ranks peers brings up
        # the source ranking it funnels (:meth:`~proposed.selector.Refine.attach`), and
        # that same ranking is a link of the chain attached just above, to the plain
        # view. This is the attach that leaves it sensing through the pinned view: a
        # ranking a decision consults inside its pin must not read past the snapshot
        # into the live directory.
        self._reuse.attach(self.view, transfer_cost)

    # -- what a serving host asks, at the two moments it has a question ------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection[None]:
        """Which peers should serve ``requester``'s fetch of ``keys``, best first.

        The pull :meth:`decide` already priced for this caller, else whoever holds the
        longest prefix (:class:`FetchRouting`). ``Selection.of([])`` names nobody,
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

        ``requester`` is the host the request landed on, which no part of this
        decision reads: where a request *should* run is a fact about the cluster,
        not about who was asked.
        """
        prefill = await self._select_prefill(request)
        decode = await self._select_decode(prefill.winner)
        return self._admit(request, prefill, decode)

    async def _select_prefill(self, request: Request) -> Selection[Plan]:
        """Every prefill instance, priced and ranked best first.

        Each one's :class:`Plan` rides under its id in
        :attr:`~proposed.selector.Selection.payload`, so the answer says what was
        compared as well as what won.

        Atomic: every read is off one pinned directory snapshot
        (:meth:`~kvcache_sim.control._view.KVView.pinned`) and one clock read, so the
        prices are comparable, and the bookkeeping it writes needs nothing locked.
        Pricing a candidate reserves nothing (:meth:`_candidate`), so the losers leave
        no trace and the winner is chosen after the whole field is known.
        """
        now = self.view.now()
        keys = list(request.block_keys)
        with self.view.pinned(keys):
            counts = self.view.prefix_lengths(keys)
            candidates: List[_Priced] = []
            for inst in self.prefill_ids:
                chosen = await self._reuse.select(keys, inst)
                match, src, pull = self._priced_reuse(counts, keys, inst, chosen)
                candidates.append((inst, self._candidate(
                    request, inst, now, match=match, source=src, pull_keys=pull
                )))
            return Selection.priced(sorted(
                candidates, key=lambda candidate: self._rank(candidate, self)
            ))

    @staticmethod
    def _priced_reuse(
        counts: Dict[str, int], keys: Sequence[Key], inst: str,
        chosen: Selection[None],
    ) -> Tuple[int, Optional[str], Sequence[str]]:
        """What a reuse selection buys ``inst``: ``(match, source, pull_keys)``.

        Derived here and not in the selector, because naming a peer is where a
        selector's job ends: how much of this prompt that peer's prefix covers is
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
        taken). Which candidate wins is the rank key's business (:data:`_RankKey`).
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
            match, cached, uncached, source, xbytes, queue_wait, ttft, done,
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

    async def _select_decode(self, plan: Plan) -> Selection[int]:
        """Decode instances, each with the batch a request admitted at ``plan``'s
        completion is predicted to meet there.

        Ranked by :func:`_by_batch`. With decode unmodelled every candidate prices at
        zero, so the id tie-break is the whole of the choice.
        """
        batches = [
            (d, self._predicted_batch(d, plan.done_time)) for d in self.decode_ids
        ]
        return Selection.priced(sorted(batches, key=_by_batch))

    def _admit(
        self,
        request: Request,
        prefill: Selection[Plan],
        decode: Selection[int],
    ) -> Optional[Response]:
        """The two winners as one :class:`Response`, gated and committed.

        Assembled before the gates rather than after, so the value the gates judge is
        the value the answer carries. ``None`` == rejected, and rejected here costs
        nothing: this runs before the prefill does.
        """
        instance = decode.sources[0]
        batch = decode.payload[instance]
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
        if not all(gate(response, self) for gate in self._gates):
            return None
        # Accepted: the model holds the instances this plan spoke for, and remembers
        # the peer its pull was priced against for when the fetch asks
        # (:class:`RoutedPull`).
        #
        # The local write, not the endpoint a host reports over: control is in the
        # model's process, and a plain call is what keeps this decision atomic.
        self.cluster._notify_impl(Committed(response, request.output_tokens))
        return response


# -- the two schedulers, as the settings that make them ---------------------- #
# Names rather than behaviours: everything either one does is above, and each is
# one choice on each axis. A third combination is composed, not subclassed.


class LoadBalanceScheduler(_Scheduler):
    """Baseline (~vLLM): least-loaded instance, local-only cache reuse."""

    def __init__(self, **knobs: Any) -> None:
        super().__init__(reuse=_LocalOnly(), rank=_by_load, **knobs)


class CacheAwareScheduler(_Scheduler):
    """Cache-aware: global prefix-match routing under a balance threshold.

    ``replicate=False`` (which isolates replication's contribution in the demo) is
    never pulling at all, so it is the baseline's reuse axis rather than a flag
    the candidate loop tests.

    Args:
        source_selector: ranks peers for a prefix pull. Handed to the reuse axis
            *and* on to :meth:`~_Scheduler.sources`, which is the point: one object,
            so the peer named while pricing is the peer that serves the read.
            ``replicate=False`` never asks it anything while pricing, and a fetch
            still is.

            Required here, and deliberately without a default: this ranking keeps
            state across the decisions it makes
            (:class:`~kvcache_sim.control._source.SpreadReadsKeySelector`), so which
            one a run uses is the run's to choose
            (:func:`kvcache_sim.workload._serving.scheduler`).
    """

    def __init__(self, *, source_selector: KeySelector, balance_threshold: float = 1.5,
                 replicate: bool = True, **knobs: Any) -> None:
        super().__init__(
            reuse=(
                Refine(
                    source_selector,
                    AbstainOnSelf(),
                    _LongerThanLocal(balance_threshold),
                    TakeHead(),
                )
                if replicate
                else _LocalOnly()
            ),
            rank=_by_ttft,
            source_selector=source_selector,
            **knobs,
        )
