"""Every ranking KV-cache routing decides with, all over one subject.

Keys, the store's own question and the only part of this routing that is one:
:class:`LongestPrefixKeySelector` ranks the peers holding a prefix,
:class:`LocalOnly` names nobody, and :class:`RoutedPull` answers a fetch with the pull
that was already priced for it.

What is a selector and what is not: a **ranking over candidates** is one; a comparison
or a cost function is not. Holding a plan to an SLO answers yes or no, and a ranked set
of sources cannot say that; whether a peer's prefix run beats recomputing is a test of
one ranking's head (:func:`~kvcache_sim.control.scheduler._worth_pulling`); and what a
prefill costs is arithmetic (:func:`domain.prefill_time`).

One of the **axes** a preset picks is here. **Reuse** ranks the peers holding this
prefix or names nobody, and is asked once per decision, because which peers hold a
prefix does not depend on who would prefill it; the tests that *do* depend on the
candidate are applied to that one ranking per candidate
(:meth:`~kvcache_sim.control.scheduler._Scheduler._select_prefill`). The other axis,
**the winner**, is not a ranking at all: the scheduler keys the prefill pool with the
plans it priced and its own fold orders them, so there is nothing here to name. Which
host decodes is not an axis either -- both presets rank decode the same way, and the
scheduler predicts every batch itself, so keying them is the whole of what a ranking
would do there (:meth:`~kvcache_sim.control.scheduler._Scheduler._select_decode`).

Every ranking here **keys** its candidates and orders none of them
(:attr:`~proposed.selector.Selection.key`); the scheduler folds once when it wants a
winner (:meth:`~proposed.selector.Selection.max`,
:meth:`~proposed.selector.Selection.sort`), and the instance id is the last thing that
fold compares, so a rank is total and a run reproduces.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from proposed import Key, KeySelector, Selection
from proposed.selector import Dims, Fold, Selector

from ._view import prefix_lengths_of, PrefixView, RoutedView

__all__ = [
    "LongestPrefixKeySelector",
    "by_prefix_and_load",
    "LocalOnly",
    "RoutedPull",
]


class LongestPrefixKeySelector(KeySelector):
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
    of it. Opt-in and off by default: ``python -m kvcache_sim hotspot --spread-reads``
    is that scenario's cache-aware runs asking for the ``"spread"`` source ranking,
    which the scheduler builds with :func:`by_prefix_and_load` as its fold
    (:func:`~kvcache_sim.control.scheduler._source_ranking`).
    """

    name = "longest-prefix"
    sensors = (PrefixView,)

    async def select(self, keys: Sequence[Key], requester: str) -> Selection:
        """Instances holding a leading run of ``keys``, keyed at the **negated** run.

        Blocks of prefix run: a measurement and not a valuation -- what a source is
        *worth* remains the scheduler's to weigh. Keyed at all because a stage appended
        behind this one has to have something to be behind
        (:class:`~proposed.selector.Balance`), and negated because a fold takes the
        lowest while a longer run is the better source.
        """
        counts = self._prefix_runs(list(keys))
        if not counts:
            return Selection.of([])
        return Selection.keyed([(inst, (-run,)) for inst, run in counts.items()])

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


class LocalOnly(Selector[Sequence[Key]]):
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

    async def select(self, keys: Sequence[Key], requester: str) -> Selection:
        return Selection.of([])


class RoutedPull(KeySelector):
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

    async def select(self, keys: Sequence[Key], requester: str) -> Selection:
        peer = self.view.routed.claim(requester, keys)
        return Selection.of([peer] if peer is not None else [])
