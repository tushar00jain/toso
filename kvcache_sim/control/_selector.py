"""Every ranking KV-cache routing decides with, over two subjects.

Over **keys**, the store's own question and the only part of this routing that is one:
:class:`LongestPrefixKeySelector` ranks the peers holding a prefix,
:class:`LocalOnly` names nobody, and :class:`RoutedPull` answers a fetch with the pull
that was already priced for it. Over the scheduler's **own priced candidates**:
:class:`ByTTFT` and :class:`ByLoad` rank the hosts that could prefill,
:class:`ByBatch` the hosts that could decode.

What is a selector and what is not: a **ranking over candidates** is one; a comparison
or a cost function is not. Holding a plan to an SLO answers yes or no, and a ranked set
of sources cannot say that; whether a peer's prefix run beats recomputing is a test of
one ranking's head (:func:`~kvcache_sim.control.scheduler._worth_pulling`); and what a
prefill costs is arithmetic (:func:`domain.prefill_time`).

Two are the **axes** a preset picks. **Reuse** ranks the peers holding this prefix or
names nobody, and is asked once per decision, because which peers hold a prefix does not
depend on who would prefill it; the tests that *do* depend on the candidate are applied
to that one ranking per candidate
(:meth:`~kvcache_sim.control.scheduler._Scheduler._select_prefill`). **The winner**
ranks the candidates the scheduler priced. Which host decodes is *not* an axis -- both
presets rank decode the same way, so the scheduler builds :class:`ByBatch` itself -- and
it is a selector anyway, so it can be re-ranked from outside
(:class:`~proposed.selector.Discount`), which a bare sort key never can.

Every rank here ends in the instance id, read off the candidate rather than out of what
it was priced at, so a rank is total and a run reproduces.
"""

from __future__ import annotations

from typing import Dict, Sequence, TypeVar

from proposed import AnySelector, Key, KeySelector, Selection
from proposed.selector import Selector

from ._answer import Batched, Plan, Priced
from ._view import prefix_lengths_of, PrefixView

__all__ = [
    "LongestPrefixKeySelector",
    "LocalOnly",
    "ByTTFT",
    "ByLoad",
    "ByBatch",
    "RoutedPull",
]

#: The price a selector here answers in, left open on the two that answer without
#: quoting one: :class:`LocalOnly` names nobody, and :class:`RoutedPull` names the peer
#: the scheduler already priced itself. Their payload is empty, and an empty payload is
#: a payload in whatever terms the holder prices in -- so neither claims a unit it never
#: quotes, and both fit a chain pricing in blocks of prefix run.
_P = TypeVar("_P")


class LongestPrefixKeySelector(KeySelector[int]):
    """Rank instances by how much of the requested block prefix they hold.

    Longest contiguous run first, instance id as the tie-break, so the choice is
    deterministic. The requester is accepted and ignored: reuse value here is a
    property of the *prefix*, and only the scheduler holds the other half of the
    trade (a nearer peer is cheaper to fetch from, a shorter prefix means more
    recompute), so it weighs locality itself when it prices the pull.

    The default on both sides of a pull: the reuse axis a cache-aware preset prices
    with, and the ranking a fetch falls through to. Spreading reads over the replicas
    of a hot prefix is this ranking under :class:`~proposed.selector.Discount`, which
    bounds how much load may cancel out of a prefix run, so a host holding a hot prefix
    does not serve every read of it. Opt-in and off by default:
    ``python -m kvcache_sim hotspot --spread-reads`` hands a fresh
    ``Discount(LongestPrefixKeySelector())`` to each of that scenario's cache-aware runs
    as :func:`~kvcache_sim.workload._serving.scheduler`'s ``source_selector``.
    """

    name = "longest-prefix"

    async def select(
        self, keys: Sequence[Key], requester: str
    ) -> Selection[int]:
        """Instances holding a leading run of ``keys``, longest run first.

        ``KeySelector[int]``: the price is the run itself, in blocks -- a measurement,
        not a valuation, and what a source is *worth* remains the scheduler's to
        weigh. Published because the number this ranking already turns on is the only
        honest price for it, and because a re-ranking over it has to weigh something
        (:class:`~proposed.selector.Discount`).
        """
        counts = self._prefix_runs(list(keys))
        if not counts:
            return Selection.of([])
        ranked = sorted(counts, key=lambda inst: (-counts[inst], inst))
        return Selection.priced([(inst, counts[inst]) for inst in ranked])

    def _prefix_runs(self, keys: Sequence[Key]) -> Dict[str, int]:
        """Per-instance prefix runs, off whichever view this selector was attached to.

        The scheduler attaches its :class:`~kvcache_sim.control._view.KVView`, whose
        snapshot a routing decision pins, so the whole decision reads one directory.
        A run that stands this selector up on its own can only attach the plain
        :class:`~proposed.view.View`, since a prefix run is a KV-cache notion the
        store has no reason to know. Read it off the view that carries it, derive it
        otherwise, off one shared definition.
        """
        if isinstance(self.view, PrefixView):
            return self.view.prefix_lengths(keys)
        return prefix_lengths_of(self.view.locate(keys), keys)


class LocalOnly(Selector[Sequence[Key], _P]):
    """Name nobody, ever -- the baseline reuses only what a host already holds.

    A plain :class:`~proposed.selector.Selector`: its subject is keys, but the
    scheduler is the only thing that asks it, so it is not fronted by a service at
    all.

    ``Selection.of([])``, which is :class:`~proposed.selector.FirstMatch`'s
    *abstention*: no source, so the caller recomputes the gap. Deliberately not
    ``Selection()``, which is a decision meaning every holder in directory order.
    """

    name = "local-only"

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        return Selection.of([])


class ByTTFT(AnySelector[Sequence[Priced], Plan]):
    """The lowest predicted queue + transfer + prefill.

    Why reuse is *priced* rather than preferred: a longer match on a busier instance
    can still lose. Senses nothing: the price it ranks by is one the scheduler already
    worked out per candidate.
    """

    name = "by-ttft"

    async def select(
        self, candidates: Sequence[Priced], requester: str
    ) -> Selection[Plan]:
        return Selection.priced(sorted(
            candidates, key=lambda c: (c[1].ttft, c[0])
        ))


class ByLoad(AnySelector[Sequence[Priced], Plan]):
    """The shortest predicted prefill queue, whatever reuse bought (the baseline).

    Sorts on ``busy_until`` rather than the candidate's ``queue_wait``, which is that
    tail clamped at the clock: two instances idle since different moments both wait
    zero, so the choice would fall to the id tie-break and a different one would win.
    This ranking's claim is that it picks by load and nothing else.

    The queue is sensed through the attached view
    (:class:`~kvcache_sim.control._view.ClusterView`) and read once for the whole
    ranking, so every candidate is ranked against one state of the cluster.
    """

    name = "by-load"

    async def select(
        self, candidates: Sequence[Priced], requester: str
    ) -> Selection[Plan]:
        busy = self.view.cluster.busy_until
        return Selection.priced(sorted(
            candidates, key=lambda c: (busy[c[0]], c[0])
        ))


class ByBatch(AnySelector[Sequence[Batched], int]):
    """The smallest predicted decode batch, instance id breaking the tie.

    The other host pick of a routing decision, over a predicted batch rather than a
    plan. Senses nothing: the scheduler predicted every batch before asking.

    Priced at the **negated batch** -- headroom, since a price is better when it is
    higher, which is what lets a re-ranking weigh it
    (:class:`~proposed.selector.Discount` with ``max_discount=1`` says load may cost a
    host one batch slot and no more). Negated rather than counted down from a capacity,
    because only differences are compared and no batch cap is known here -- it is the
    accelerator's. The occupancy itself is what the TBT SLO is judged on, so the one
    caller that needs the number back negates it again
    (:meth:`~kvcache_sim.control.scheduler._Scheduler._admit`).
    """

    name = "by-batch"

    async def select(
        self, candidates: Sequence[Batched], requester: str
    ) -> Selection[int]:
        headroom = [(instance, -batch) for instance, batch in candidates]
        return Selection.priced(sorted(
            headroom, key=lambda c: (-c[1], c[0])
        ))


class RoutedPull(KeySelector[_P]):
    """The peer a fetch's pull was already priced against, or an abstention.

    Answering the fetch from what routing decided, rather than deciding twice:
    re-deriving would not even agree (routing ranks over the request's whole block
    chain, the fetch names only the gap), and naming a different holder would
    charge a cross-node read for a same-node prediction. A caller with no routed
    pull falls through to the ranking behind this link.

    A selector like any other, sensing through the view it is attached to
    (:meth:`~proposed.selector.Selector.attach`) -- the sensor is one that view carries
    (:class:`~kvcache_sim.control._view.RoutedView`), so it arrives the way
    every other read does and nothing hands this one the sensor.

    Reading it **consumes** it
    (:meth:`~kvcache_sim.control._sensor.RoutedPullSensor.claim` expires the entry on
    a match), so this belongs at the head of a
    :class:`~proposed.selector.FirstMatch` chain and under no combinator that can drop
    the answer or rank it down (:class:`~proposed.selector.Discount`). In that one
    position spending and using coincide: a link that answers wins the chain, an
    abstention matched nothing and spends nothing. Under one that could reject the
    peer, the entry would be gone and the fetch would fall through to a ranking that
    never saw it.
    """

    name = "routed-pull"

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        peer = self.view.routed.claim(requester, keys)
        return Selection.of([peer] if peer is not None else [])
