"""The read-through tree a dedup plane has planned: :class:`FanoutSensor`."""

from __future__ import annotations

from collections import deque
from typing import (
    Deque, Dict, Hashable, Iterable, Optional, Sequence, Set, Tuple,
)

from proposed import Sensor
from proposed.selector import Ready

from ._readiness import Observed, Readiness

__all__ = ["FanoutSensor"]


class FanoutSensor(Sensor):
    """Who is folded in behind whom, which puts are owed, and who waits on them.

    Read through the plane's view (:class:`~dedup_sim.control._view.FanoutView`):
    every link of the source chain senses it and writes its own decision back
    (:mod:`dedup_sim.control._selector`), and the plane folds in the puts it is told
    about (:meth:`~dedup_sim.control.routing.Dedup.published`).

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
        # Waiting for the (volume, key) pairs the real directory has not registered
        # yet. The concurrency lives there, as does the rule that the directory --
        # not a memory of past registrations -- says which are true, since a volume
        # that evicts makes one false again.
        self._ready = Readiness()

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

    def published(self, requester: str, keys: Sequence[str]) -> None:
        """``requester``'s put has landed: release its waiters, settle its debt.

        From here on the directory is what says whether the volume holds the key.
        """
        for key in keys:
            self._promised.discard((requester, key))
            self._ready.record((requester, key))

    async def gate(
        self, facts: Iterable[Hashable], observed: Observed
    ) -> Optional[Ready]:
        """A gate that opens once every one of ``facts`` is true, or ``None``.

        Delegated whole (:mod:`dedup_sim.control._sensor._readiness`); whether these
        facts are coming at all is the caller's question.
        """
        return await self._ready.gate(facts, observed)
