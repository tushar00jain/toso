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
it is a selector anyway, so it can be re-keyed from outside
(:class:`~proposed.selector.Balance`), which a sort written into the scheduler never
could.

Every ranking here **keys** its candidates and orders none of them
(:attr:`~proposed.selector.Selection.key`); the scheduler folds once when it wants a
winner (:meth:`~proposed.selector.Selection.max`,
:meth:`~proposed.selector.Selection.sort`), and the instance id is the last thing that
fold compares, so a rank is total and a run reproduces.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, TypeVar

from proposed import AnySelector, Key, KeySelector, Selection
from proposed.selector import Dims, Fold, Selector

from ._answer import Batched, Plan, Priced
from ._view import ClusterView, prefix_lengths_of, PrefixView, RoutedView

__all__ = [
    "LongestPrefixKeySelector",
    "by_prefix_and_load",
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

    Longest contiguous run first once folded, the instance id breaking that tie there,
    so the choice is deterministic. The requester is accepted and ignored: reuse value
    here is a property of the *prefix*, and only the scheduler holds the other half of
    the trade (a nearer peer is cheaper to fetch from, a shorter prefix means more
    recompute), so it weighs locality itself when it prices the pull.

    The default on both sides of a pull: the reuse axis a cache-aware preset prices
    with, and the ranking a fetch falls through to. Spreading reads over the replicas
    of a hot prefix is this ranking under :class:`~proposed.selector.Balance`, folded by
    :func:`by_prefix_and_load`, so a host holding a hot prefix does not serve every read
    of it. Opt-in and off by default:
    ``python -m kvcache_sim hotspot --spread-reads`` hands a fresh
    ``Balance(LongestPrefixKeySelector())`` to each of that scenario's cache-aware runs
    as :func:`~kvcache_sim.workload._serving.scheduler`'s ``source_selector``, with
    :func:`by_prefix_and_load` as its ``source_fold``.
    """

    name = "longest-prefix"
    sensors = (PrefixView,)

    async def select(
        self, keys: Sequence[Key], requester: str
    ) -> Selection[int]:
        """Instances holding a leading run of ``keys``, keyed at the **negated** run.

        ``KeySelector[int]``: the price is the run itself, in blocks -- a measurement,
        not a valuation, and what a source is *worth* remains the scheduler's to
        weigh. Published because the number this ranking already turns on is the only
        honest price for it, and because a stage appended behind this one has to have
        something to be behind (:class:`~proposed.selector.Balance`).

        Negated in the key and not in the price, since a fold takes the lowest and a
        longer run is the better source.
        """
        counts = self._prefix_runs(list(keys))
        if not counts:
            return Selection.of([])
        return Selection.keyed(
            [(inst, (-run,), run) for inst, run in counts.items()]
        )

    def _prefix_runs(self, keys: Sequence[Key]) -> Dict[str, int]:
        """Per-instance prefix runs, off whichever view this selector was attached to.

        A run that stands this selector up on its own can attach the plain
        :class:`~proposed.view.View`, since a prefix run is a KV-cache notion the store
        has no reason to know. Read it off the view that carries it, derive it
        otherwise, off one shared definition.
        """
        if isinstance(self.view, PrefixView):
            return self.view.prefix_lengths(keys)
        return prefix_lengths_of(self.view.locate(keys), keys)


def by_prefix_and_load(bound: int = 1) -> Fold:
    """Fold a prefix run against the reads lately routed at the host holding it.

    What ``--spread-reads`` folds the source ranking with: a source docked one block of
    prefix run per read routed at it, up to ``bound`` blocks, longest run still first.
    So load settles a tie between replicas of one prefix and can never outvote a
    materially longer match -- a host ahead by more than ``bound`` blocks wins however
    busy it is. The default is one block, which is enough for the tie it exists to
    break; a wider bound does trade reuse away, once it has been fully spent.

    Reads the two dimensions ``Balance(LongestPrefixKeySelector())`` leaves: the run
    negated, then the load. Docking is therefore an *addition*, and behind the bound
    comes the raw count, so two replicas the bound has levelled keep alternating instead
    of reverting to id order.
    """
    def fold(dims: Dims) -> Tuple[int, int]:
        run, load = dims
        return (run + min(load, bound), load)

    return fold


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
    sensors = ()

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        return Selection.of([])


class ByTTFT(AnySelector[Sequence[Priced], Plan]):
    """The lowest predicted queue + transfer + prefill.

    Why reuse is *priced* rather than preferred: a longer match on a busier instance
    can still lose. The price it ranks by is one the scheduler already worked out per
    candidate.
    """

    name = "by-ttft"
    sensors = ()

    async def select(
        self, candidates: Sequence[Priced], requester: str
    ) -> Selection[Plan]:
        """Every candidate, keyed at its predicted TTFT and holding its whole plan."""
        return Selection.keyed(
            [(inst, (plan.ttft,), plan) for inst, plan in candidates]
        )


class ByLoad(AnySelector[Sequence[Priced], Plan]):
    """The shortest predicted prefill queue, whatever reuse bought (the baseline).

    Keys on ``busy_until`` rather than the candidate's ``queue_wait``, which is that
    tail clamped at the clock: two instances idle since different moments both wait
    zero, so the choice would fall to the id tie-break and a different one would win.
    This ranking's claim is that it picks by load and nothing else.

    The queue is read once for the whole ranking, so every candidate is keyed against
    one state of the cluster.
    """

    name = "by-load"
    sensors = (ClusterView,)

    async def select(
        self, candidates: Sequence[Priced], requester: str
    ) -> Selection[Plan]:
        """Every candidate, keyed at the queue it would join and holding its plan."""
        busy = self.view.cluster.busy_until
        return Selection.keyed(
            [(inst, (busy[inst],), plan) for inst, plan in candidates]
        )


class ByBatch(AnySelector[Sequence[Batched], int]):
    """The smallest predicted decode batch, instance id breaking the tie.

    The other host pick of a routing decision, over a predicted batch rather than a
    plan; the scheduler predicted every batch before asking.

    Priced at the **negated batch** -- headroom, which is the figure a fold would read
    load against (:class:`~proposed.selector.Balance`), so it is what this
    publishes. Negated rather than counted down from a capacity, because only
    differences are compared and no batch cap is known here -- it is the accelerator's.
    The occupancy itself is what the TBT SLO is judged on, so the one caller that needs
    the number back negates it again
    (:meth:`~kvcache_sim.control.scheduler._Scheduler._admit`).
    """

    name = "by-batch"
    sensors = ()

    async def select(
        self, candidates: Sequence[Batched], requester: str
    ) -> Selection[int]:
        """Every candidate, keyed at the batch itself and priced at the headroom."""
        return Selection.keyed(
            [(instance, (batch,), -batch) for instance, batch in candidates]
        )


class RoutedPull(KeySelector[_P]):
    """The peer a fetch's pull was already priced against, or an abstention.

    Answering the fetch from what routing decided, rather than deciding twice:
    re-deriving would not even agree (routing ranks over the request's whole block
    chain, the fetch names only the gap), and naming a different holder would
    charge a cross-node read for a same-node prediction. A caller with no routed
    pull falls through to the ranking behind this link.

    A selector like any other: the sensor arrives on the view it is attached to, the way
    every other read does, and nothing hands this one the sensor.

    Reading it **consumes** it
    (:meth:`~kvcache_sim.control._sensor.RoutedPullSensor.claim` expires the entry on
    a match), so this belongs at the head of a
    :class:`~proposed.selector.FirstMatch` chain and under no combinator that can drop
    the answer or rank it down (:class:`~proposed.selector.Balance`). In that one
    position spending and using coincide: a link that answers wins the chain, an
    abstention matched nothing and spends nothing. Under one that could reject the
    peer, the entry would be gone and the fetch would fall through to a ranking that
    never saw it.
    """

    name = "routed-pull"
    sensors = (RoutedView,)

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        peer = self.view.routed.claim(requester, keys)
        return Selection.of([peer] if peer is not None else [])
