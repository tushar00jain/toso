"""Every ranking KV-cache routing decides with.

Three over keys, the store's own question: :class:`LongestPrefixKeySelector` ranks the
peers holding a prefix, :class:`LocalOnly` names nobody, and :class:`RoutedPull` answers
a fetch with the pull that was already priced for it. Two over this plane's own values:
:class:`Priced` keys each prefill candidate at the
:class:`~kvcache_sim.control._answer.Plan` running the request there would cost, and
:class:`DecodeBatch` keys the decode hosts at the batch such a plan's completion would
meet on each.

A **ranking over candidates** is a selector; a **verdict** is not. Holding a plan to an
SLO answers yes or no, which a ranked set of sources cannot say, so admission stays
with the plane. What a ranking measures *with* travels with it: whether a peer's prefix
run beats recomputing (:func:`_worth_pulling`) is a test of the reuse ranking's head.

The **reuse** axis is asked once per decision, since which peers hold a prefix does not
depend on who would prefill it; the tests that *do* depend on the candidate are applied
to that one ranking per candidate (:class:`Priced`). Which host decodes is not an axis
-- both presets rank it the same way, so :class:`DecodeBatch` is unconditional.

Every ranking here **keys** its candidates and orders none of them; the chain the
scheduler declares each in does the ordering, and the instance id is the last thing it
compares, so a rank is total and a run reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from proposed import Key, KeySelector, Selection
from proposed.selector import Dims, Fold, Selector

from domain import decode_step_time, MachineProfile, Model, prefill_time

from ._answer import Plan
from ._view import (
    ClusterView, prefix_lengths_of, PrefixView, ReservedView, RoutedView,
)
from .request import Request

__all__ = [
    "LongestPrefixKeySelector",
    "by_prefix_and_load",
    "LocalOnly",
    "RoutedPull",
    "PrefillAsk",
    "Priced",
    "DecodeBatch",
]


class LongestPrefixKeySelector(KeySelector):
    """Rank instances by how much of the requested block prefix they hold.

    Longest contiguous run first once folded, the instance id breaking that tie there.
    The requester is accepted and ignored: reuse value here is a property of the
    *prefix*, and the scheduler holds the other half of the trade (a nearer peer is
    cheaper to fetch from, a shorter prefix means more recompute).

    The default on both sides of a pull. Spreading reads over the replicas of a hot
    prefix is this ranking docked by :func:`by_prefix_and_load`, opt-in via
    ``python -m kvcache_sim hotspot --spread-reads``.
    """

    name = "longest-prefix"
    sensors = (PrefixView,)

    def select(self, keys: Sequence[Key], requester: str) -> Selection:
        """Instances holding a leading run of ``keys``, keyed at the **negated** run.

        Negated because a fold takes the lowest and a longer run is the better source.
        """
        counts = self._prefix_runs(list(keys))
        if not counts:
            return Selection.of([])
        return Selection.keyed([(inst, (-run,)) for inst, run in counts.items()])

    def _prefix_runs(self, keys: Sequence[Key]) -> Dict[str, int]:
        """Per-instance prefix runs, off whichever view this selector was attached to.

        A prefix run is a KV-cache notion the store has no reason to know, so a view
        carrying none is allowed and the run is derived instead.
        """
        if isinstance(self.view, PrefixView):
            return self.view.prefix_lengths(keys)
        return prefix_lengths_of(self.view.locate(keys), keys)


def by_prefix_and_load(bound: int = 1) -> Fold:
    """Fold a prefix run against the reads lately routed at the host holding it.

    A source is docked one block of prefix run per read routed at it, up to ``bound``
    blocks, so load settles a tie between replicas of one prefix and never outvotes a
    longer match: a host ahead by more than ``bound`` blocks wins however busy it is.

    Dimension 0 is the negated run and dimension 1 the load, so docking is an
    *addition*, and keeping the raw count behind the bound leaves two levelled replicas
    alternating instead of reverting to id order.
    """
    def fold(dims: Dims) -> Tuple[int, int]:
        run, load = dims
        return (run + min(load, bound), load)

    return fold


class LocalOnly(Selector[Sequence[Key]]):
    """Name nobody, ever -- the baseline reuses only what a host already holds.

    An abstention, so the caller recomputes the gap -- not ``Selection()``, which would
    be a decision naming every holder in directory order.
    """

    name = "local-only"
    sensors = ()

    def select(self, keys: Sequence[Key], requester: str) -> Selection:
        return Selection.of([])


class RoutedPull(KeySelector):
    """The peer a fetch's pull was already priced against, or an abstention.

    Deciding twice would not even agree: routing ranks over the request's whole block
    chain while the fetch names only the gap, and naming a different holder would
    charge a cross-node read for a same-node prediction. A caller with no routed pull
    falls through to the ranking behind this link.

    Belongs at the head of the fetch chain, under nothing that could drop this answer
    or rank it down: the memo is spent as the plane answers
    (:class:`~kvcache_sim.control._sensor.FetchAnswered`) whether or not this link won,
    so a memo ranked down is a pull served by a volume nothing charged for.
    """

    name = "routed-pull"
    sensors = (RoutedView,)

    def select(self, keys: Sequence[Key], requester: str) -> Selection:
        peer = self.view.routed.peer(requester, keys)
        return Selection.of([peer] if peer is not None else [])


@dataclass(frozen=True)
class PrefillAsk:
    """What pricing one prefill pool takes: :class:`Priced`'s subject.

    Not beside the values a decision answers with
    (:mod:`kvcache_sim.control._answer`), because ``peer`` may carry a gate, which is a
    closure: this value crosses no boundary.
    """

    request: Request
    #: The moment the pool is priced at.
    now: float
    #: The request's block chain.
    keys: Sequence[Key]
    #: Per-instance prefix run over ``keys``.
    counts: Dict[str, int]
    #: What the reuse ranking named, ordered already, so a per-candidate test applies
    #: to a head that means something.
    peer: Selection


def _worth_pulling(
    counts: Dict[str, int], inst: str, threshold: float
) -> Callable[[str], bool]:
    """Does pulling the head's prefix beat recomputing the gap on ``inst``?

    Its run must be more than ``threshold`` times ``inst``'s own. A pull is charged to
    the prefill instance's queue, so one that saves little compute still costs the whole
    wait, and without the threshold every request would chase the longest match onto one
    instance. A test of the *head*, not a filter.
    """
    return lambda head: counts.get(head, 0) > counts.get(inst, 0) * threshold


class Priced(Selector[PrefillAsk]):
    """Every instance in the prefill pool, keyed at what running this request on it costs.

    The dimension is that candidate's whole :class:`~kvcache_sim.control._answer.Plan`, so
    the answer carries what was compared as well as what won.

    Args:
        instances: the prefill pool, as the plane resolved it.
        block_tokens / profile / model: the cost constants a plan is priced against.
        threshold: how much longer the peer's prefix run must be than a candidate's own
            before pulling beats recomputing (:func:`_worth_pulling`).
    """

    name = "priced"
    sensors = (ClusterView,)

    def __init__(
        self,
        instances: Sequence[str],
        *,
        block_tokens: int,
        profile: MachineProfile,
        model: Model,
        threshold: float,
    ) -> None:
        self.instances = tuple(instances)
        self.B = block_tokens
        self.profile = profile
        self.model = model
        self.threshold = threshold

    def select(self, ask: PrefillAsk, requester: str) -> Selection:
        """Priced as the dimension is appended, so the pool is walked once."""
        return Selection.of(self.instances).annotated(
            lambda inst: self._plan(ask, inst)
        )

    def _plan(self, ask: PrefillAsk, inst: str) -> Plan:
        """Which peer ``inst`` would pull from, if any, and what that comes to."""
        # Worth the transfer only if the peer holds materially more than this candidate
        # already does; a candidate named as its own peer is left to _priced_reuse.
        peer = ask.peer.require(_worth_pulling(ask.counts, inst, self.threshold))
        match, src, pull = self._priced_reuse(ask.counts, ask.keys, inst, peer)
        return self._candidate(
            ask.request, inst, ask.now, match=match, source=src, pull_keys=pull,
        )

    @staticmethod
    def _priced_reuse(
        counts: Dict[str, int], keys: Sequence[Key], inst: str, peer: Selection,
    ) -> Tuple[int, Optional[str], Sequence[str]]:
        """What one peer buys ``inst``: ``(match, source, pull_keys)``.

        A selection naming nobody or naming ``inst`` itself leaves the local match to
        recompute from, so no self-peer test is needed upstream.
        """
        local = counts.get(inst, 0)
        src = peer.head
        if src is None or src == inst:
            return local, None, ()
        return counts[src], src, keys[local:counts[src]]

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
        taken).
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
        return Plan(
            match, cached, uncached, source, xbytes, queue_wait, ttft, done,
            prefill_t=prefill_t, pull_keys=list(pull_keys),
        )


class DecodeBatch(Selector[Plan]):
    """Decode instances, keyed at the batch a request admitted at a plan's completion is
    predicted to meet there -- smallest best, the id breaking a tie.

    The subject is the *winning* prefill candidate's :class:`Plan`, not the request:
    which host decodes is settled against that candidate's predicted completion. With
    decode unmodelled every candidate keys at zero and the tie-break is the whole of the
    choice.

    Args:
        instances: the decode pool, as the plane resolved it.
        tbt_enabled: whether the run models batched decode at all.
        lookahead: whether to roll occupancy forward to the plan's completion. The same
            flag that decided whether a reservation sensor was composed at all
            (:meth:`~kvcache_sim.control.scheduler._Scheduler.attach`), so this read
            finds one exactly when it takes it.
        profile / model: the cost constants a reservation's decode is priced against.
    """

    name = "decode-batch"
    sensors = (ClusterView, ReservedView)

    def __init__(
        self,
        instances: Sequence[str],
        *,
        tbt_enabled: bool,
        lookahead: bool,
        profile: MachineProfile,
        model: Model,
    ) -> None:
        self.instances = tuple(instances)
        self.tbt_enabled = tbt_enabled
        self.lookahead = lookahead
        self.profile = profile
        self.model = model

    def select(self, plan: Plan, requester: str) -> Selection:
        """Every instance in the pool, keyed at its predicted batch."""
        return Selection.of(self.instances).annotated(
            lambda d: self._predicted_batch(d, plan.done_time)
        )

    def _predicted_batch(self, d: str, done_time: float) -> int:
        """The decode batch on ``d`` seen by a request admitted at ``done_time``."""
        if not self.tbt_enabled:
            return 0
        if not self.lookahead:
            return self.view.cluster.occupancy(d)
        n = self.view.cluster.predict_occupancy(d, done_time)
        # Requests whose prefill has not landed are invisible to the observed decode
        # state; the outstanding reservations stand in for them.
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
