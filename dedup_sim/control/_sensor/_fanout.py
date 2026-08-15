"""The read-through tree a dedup plane has planned: :class:`FanoutSensor`."""

from __future__ import annotations

from collections import deque
from types import MappingProxyType
from typing import (
    Deque, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple,
)

from proposed import Sensor
from proposed.dispatch import Fold, Stored

__all__ = ["FanoutSensor"]


class FanoutSensor(Sensor):
    """Who is folded in behind whom and which puts are owed. Nobody's wait is here.

    Read through the plane's view (:class:`~dedup_sim.control._view.FanoutView`):
    every link of the source chain senses it and writes its own decision back
    (:mod:`dedup_sim.control._selector`).

    A :class:`proposed.dispatch.Reducer` on this plane's dispatcher (:attr:`folds`),
    which is how a landed put reaches it: the action is dispatched once, this writes its
    own state, and the commit after it is what wakes anybody. Nothing here reads the
    directory's state, and nothing reads this.

    Args:
        fanout_cap: peers one source may be planned to feed -- 1 a chain, >= 2 a
            shallow tree. The fabric stays 1x for any cap; the cap only trades
            wallclock against tree depth.
    """

    def __init__(self, fanout_cap: int = 1) -> None:
        self.cap = fanout_cap
        # requester -> the source it was routed to (decided once, then reused).
        self._route: Dict[str, str] = {}
        # One entry per peer a source may still be planned to feed, oldest first: a
        # requester joins with ``cap`` slots and each assignment consumes one. The
        # cap is the queue's own shape rather than a tally compared against it,
        # because a link assigns with no lock -- one popleft cannot leave a
        # half-applied cap behind the way an increment, a comparison and a
        # conditional pop could. See :meth:`claim_slot`.
        self._avail: Deque[str] = deque()
        # Requesters already offered their slots -- once each, however many times
        # they are assigned (see :meth:`route`).
        self._offered: Set[str] = set()
        # The (volume, key) publications planned and not yet seen to land: a routed
        # requester reads the key through into its own volume, so from the moment it
        # is routed it OWES that registration. The only thing that makes waiting for
        # a source safe (:func:`~dedup_sim.control._selector._once_usable`).
        self._promised: Set[Tuple[str, str]] = set()
        # action type -> the fold that writes this state. One entry, because a landed
        # put is the only thing it is told (:class:`proposed.dispatch.Reducer`).
        self._folds: Dict[type, Fold] = {Stored: self._stored}

    # -- what it folds ------------------------------------------------------- #
    @property
    def folds(self) -> Mapping[type, Fold]:
        """:class:`proposed.dispatch.Reducer` -- what it folds, by action type.

        Read-only, so the one way to move this state is to dispatch something that
        moves it.
        """
        return MappingProxyType(self._folds)

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

    def claim_slot(self) -> Optional[str]:
        """The oldest peer with a free slot, spending it; ``None`` if there is none.

        A read-modify-write with no lock, so its caller must not suspend between
        this and the :meth:`route` that follows it.
        """
        return self._avail.popleft() if self._avail else None

    def route(self, requester: str, source: str) -> None:
        """Fold ``requester`` in behind ``source``, and offer it as a source itself.

        Offered once per requester, however many times it is assigned. A requester
        whose source is retired is assigned afresh, and offering it again would hand
        it a second full batch of slots, so one whose first batch was already
        consumed would go on to feed ``2 x cap`` peers. Tracked separately from the
        queue because the queue only remembers the slots that are *left*: an
        exhausted requester is absent from it and would otherwise look exactly like
        one never offered.
        """
        self._route[requester] = source
        if requester in self._offered:
            return
        self._offered.add(requester)
        self._avail.extend([requester] * self.cap)

    def retire(self, requester: str, source: str) -> None:
        """Drop a source nothing is coming from, and ``requester``'s route to it.

        No route is kept, so the requester's next ask is assigned afresh -- to a peer
        that is actually going to have the key.
        """
        self._avail = deque(peer for peer in self._avail if peer != source)
        self._route.pop(requester, None)

    # -- the debt ------------------------------------------------------------ #
    def promise(self, requester: str, keys: Sequence[str]) -> None:
        """``requester`` is about to read ``keys`` through, so it owes those puts.

        Asking is the promise, and a link records it before handing out any source:
        that is what makes a requester offered as a peer only after it has promised,
        and so what bounds the wait on it
        (:func:`~dedup_sim.control._selector._once_usable`).
        """
        self._promised.update((requester, key) for key in keys)

    def owes(self, facts: Iterable[Tuple[str, str]]) -> bool:
        """Is every one of these ``(volume, key)`` publications still owed?"""
        return all(fact in self._promised for fact in facts)
