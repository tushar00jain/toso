"""The read-through tree a dedup plane has planned: :class:`FanoutSensor`."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

from proposed import Sensor
from proposed.dispatch import Fold, Stored

from ._action import Asked

__all__ = ["FanoutSensor"]


class FanoutSensor(Sensor):
    """Who is folded in behind whom and which puts are owed. Nobody's wait is here.

    Two actions move it: the ask that takes a debt on, and the put that settles it. The
    fold writes this state and the commit *after* it is what wakes anybody, so a woken
    requester never re-reads before the debt is settled. Nothing here reads the
    directory's state, and nothing reads this.

    Args:
        fanout_cap: readers one peer may be planned to feed -- 1 a chain, >= 2 a
            shallow tree. A ceiling and nothing else: which peer a reader takes is
            priced (:class:`~dedup_sim.control._selector.Candidates`), and this only
            says when one stops being offered at all.
    """

    def __init__(self, fanout_cap: int = 1) -> None:
        self.cap = fanout_cap
        # requester -> the source it was routed to, decided once and then reused.
        self._route: Dict[str, str] = {}
        # source -> how many are behind it, carried rather than counted on demand:
        # every decision reads it, and re-deriving it walks the whole tree. Moved only
        # by the two members that move a route, so the two cannot disagree.
        self._load: Counter = Counter()
        # (volume, key) publications planned and not yet seen to land: a routed
        # requester reads the key through into its own volume, so from the moment it is
        # routed it OWES that registration. This is the only thing that makes waiting
        # for a source safe (:func:`~dedup_sim.control._answer.committed`).
        self._promised: Set[Tuple[str, str]] = set()
        # A debt taken on, and the same debt settled.
        self._folds: Dict[type, Fold] = {Asked: self._asked, Stored: self._stored}

    # -- what it folds ------------------------------------------------------- #
    @property
    def folds(self) -> Mapping[type, Fold]:
        """What it folds, by action type."""
        return MappingProxyType(self._folds)

    def _asked(self, action: Asked) -> None:
        """A reader is about to read these keys through: it owes those puts from now."""
        self._promised.update((action.requester, key) for key in action.keys)

    def _stored(self, action: Stored) -> None:
        """A reader's put has landed: settle the debt it owed."""
        # Nothing is recorded: the put wrote the directory before this action was
        # dispatched, and a parked requester re-reads the directory when it wakes.
        self._promised.discard((action.host, action.key))

    # -- the tree ------------------------------------------------------------ #
    def planned(self, requester: str) -> Optional[str]:
        """The source ``requester`` is already folded in behind, if any."""
        return self._route.get(requester)

    def routes(self) -> Mapping[str, str]:
        """``requester -> its source``: the tree, and the edges a decision walks."""
        return MappingProxyType(self._route)

    def route(self, requester: str, source: str) -> None:
        """Fold ``requester`` in behind ``source``, and make it a source itself.

        One requester, one edge: re-routing replaces it rather than adding a second, so
        no bookkeeping drift can hand out more slots than the cap.
        """
        previous = self._route.get(requester)
        # Unmoved: nothing to re-count.
        if previous == source:
            return
        # The count follows the edge, which is what keeps named() the same fact as
        # routes().
        if previous is not None:
            self._drop(previous)
        self._route[requester] = source
        self._load[source] += 1

    def named(self) -> Mapping[str, int]:
        """``source -> requesters currently routed to it``. Absent means none."""
        # The same fact as the tree: one count, moved wherever a route is, so there is
        # no second tally to keep in step. A finished read is not observed, so a source
        # stays counted for as long as it is planned to serve.
        return MappingProxyType(self._load)

    def retire(self, requester: str, source: str) -> None:
        """Drop ``requester``'s route to a source nothing is coming from."""
        # No route is kept, so the next ask is priced afresh. The source needs no
        # eviction of its own: it is offered only while it still owes the key
        # (:meth:`owes`), and it no longer does.
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
