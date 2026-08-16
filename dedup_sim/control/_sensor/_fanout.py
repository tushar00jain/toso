"""The read-through tree a dedup plane has planned: :class:`FanoutSensor`."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from proposed import Sensor
from proposed.dispatch import Fold, Stored

from ._action import Asked

__all__ = ["FanoutSensor"]


class FanoutSensor(Sensor):
    """Who is folded in behind whom and which puts are owed. Nobody's wait is here.

    Read through the plane's view (:class:`~dedup_sim.control._view.FanoutView`): the
    ranking senses it to price a source (:mod:`dedup_sim.control._selector`) and the
    decision made out of that ranking writes back to it
    (:mod:`dedup_sim.control._answer`).

    A :class:`proposed.dispatch.Reducer` on this plane's dispatcher (:attr:`folds`),
    which is how the two ends of a debt reach it -- the ask that takes one on and the put
    that settles it: the action is dispatched once, this writes its own state, and the
    commit after it is what wakes anybody. Nothing here reads the directory's state, and
    nothing reads this.

    Args:
        fanout_cap: readers one peer may be planned to feed -- 1 a chain, >= 2 a
            shallow tree. A ceiling and nothing else: which peer a reader takes is
            priced (:class:`~dedup_sim.control._selector.Candidates`), and this only
            says when one stops being offered at all.
    """

    def __init__(self, fanout_cap: int = 1) -> None:
        self.cap = fanout_cap
        # requester -> the source it was routed to (decided once, then reused): who is
        # behind whom (:meth:`routes`).
        self._route: Dict[str, str] = {}
        # source -> how many are behind it (:meth:`named`), carried rather than counted
        # on demand: every decision reads it, and re-deriving it walks the whole tree
        # each time. Moved only by the two members that move a route, so the two cannot
        # disagree.
        self._load: Counter = Counter()
        # The (volume, key) publications planned and not yet seen to land: a routed
        # requester reads the key through into its own volume, so from the moment it
        # is routed it OWES that registration. The only thing that makes waiting for
        # a source safe (:func:`~dedup_sim.control._answer.committed`).
        self._promised: Set[Tuple[str, str]] = set()
        # action type -> the fold that writes this state: a debt taken on, and the same
        # debt settled (:class:`proposed.dispatch.Reducer`).
        self._folds: Dict[type, Fold] = {Asked: self._asked, Stored: self._stored}

    # -- what it folds ------------------------------------------------------- #
    @property
    def folds(self) -> Mapping[type, Fold]:
        """:class:`proposed.dispatch.Reducer` -- what it folds, by action type.

        Read-only, so the one way to move this state is to dispatch something that
        moves it.
        """
        return MappingProxyType(self._folds)

    def _asked(self, action: Asked) -> None:
        """A reader is about to read these keys through: it owes those puts from now."""
        self._promised.update((action.requester, key) for key in action.keys)

    def _stored(self, action: Stored) -> None:
        """A reader's put has landed: settle the debt it owed.

        The directory is what says whether the volume holds the key -- the put wrote it
        before the action was dispatched -- so nothing is recorded here: this drops what
        was owed and stops there. Whoever is parked on that key is woken by the commit
        and re-reads the directory (:meth:`~proposed.dispatch.Dispatcher.gate`), which is
        why nothing here knows that anybody is waiting.
        """
        self._promised.discard((action.host, action.key))

    # -- the tree ------------------------------------------------------------ #
    def planned(self, requester: str) -> Optional[str]:
        """The source ``requester`` is already folded in behind, if any."""
        return self._route.get(requester)

    def routes(self) -> Mapping[str, str]:
        """``requester -> its source``: the tree, as the one map it is kept in.

        What a decision walks to price a peer -- how soon a peer will have the key is
        what the links along these edges cost, up to a volume that holds it now.
        """
        return MappingProxyType(self._route)

    def route(self, requester: str, source: str) -> None:
        """Fold ``requester`` in behind ``source``, and make it a source itself.

        One assignment, one map entry: a requester assigned afresh replaces its own
        edge rather than being offered a second batch of anything, so nothing can
        hand out more slots than the cap by bookkeeping drift. Replacing an edge moves
        the count off the old source and onto the new one, which is what keeps
        :meth:`named` the same fact as :meth:`routes`.
        """
        previous = self._route.get(requester)
        if previous == source:
            return
        if previous is not None:
            self._drop(previous)
        self._route[requester] = source
        self._load[source] += 1

    def named(self) -> Mapping[str, int]:
        """``source -> requesters currently routed to it``. Absent means none.

        The load read of :class:`proposed.view.LoadView`, and the same fact as the tree:
        a route *is* a decision naming a source, so there is no second tally to keep in
        step with this one -- only the one count, moved where a route is. It comes back
        down when a route is retired, which is more than the count that view describes
        promises -- what it still does not observe is a read that has *finished*, so a
        source stays counted for as long as it is planned to serve.
        """
        return MappingProxyType(self._load)

    def retire(self, requester: str, source: str) -> None:
        """Drop ``requester``'s route to a source nothing is coming from.

        No route is kept, so the requester's next ask is priced afresh -- against a
        source that is actually going to have the key. The retired source needs no
        eviction of its own: it is offered only while it still owes the key
        (:meth:`owes`), and it no longer does.
        """
        previous = self._route.pop(requester, None)
        if previous is not None:
            self._drop(previous)

    def _drop(self, source: str) -> None:
        """One reader fewer behind ``source``; absent at zero, as :meth:`named` says."""
        if self._load[source] > 1:
            self._load[source] -= 1
        else:
            del self._load[source]

    # -- the debt ------------------------------------------------------------ #
    def owes(self, facts: Iterable[Tuple[str, str]]) -> bool:
        """Is every one of these ``(volume, key)`` publications still owed?"""
        return all(fact in self._promised for fact in facts)
