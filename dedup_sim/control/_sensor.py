"""The fan-out tree and pending puts a dedup decision reads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

from proposed import Key, LoadSensor, VolumeId
from proposed.dispatch import Action, Fold, Stored

__all__ = ["Asked", "FanoutSensor", "Retired", "Routed"]


@dataclass(frozen=True)
class Asked(Action):
    """``requester`` is about to read ``keys`` through, so it owes those puts."""

    requester: VolumeId
    keys: Tuple[Key, ...]


@dataclass(frozen=True)
class Routed(Action):
    """``requester`` is routed through ``source``."""

    requester: VolumeId
    source: VolumeId


@dataclass(frozen=True)
class Retired(Action):
    """``requester`` no longer routes through ``source``."""

    requester: VolumeId
    source: VolumeId


class FanoutSensor(LoadSensor):
    """Who is folded in behind whom and which puts are owed. Nobody's wait is here.

    Asks, routes, retirements and stored puts move it through actions. A fold writes
    this state before its commit wakes anybody. Nothing here reads the directory's
    state, and nothing reads this.

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
        # for a source safe (:meth:`~dedup_sim.control.routing.Dedup._committed`).
        self._promised: Set[Tuple[str, str]] = set()
        self._folds: Dict[type, Fold] = {
            Asked: self._asked,
            Retired: self._retired,
            Routed: self._routed,
            Stored: self._stored,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """What it folds, by action type."""
        return MappingProxyType(self._folds)

    def _asked(self, action: Asked) -> None:
        """A reader is about to read these keys through: it owes those puts from now."""
        self._promised.update((action.requester, key) for key in action.keys)

    def _stored(self, action: Stored) -> None:
        """A reader's put has landed: settle the debt it owed."""
        # The put wrote the directory before this action was dispatched.
        self._promised.discard((action.host, action.key))

    def planned(self, requester: str) -> Optional[str]:
        """The source ``requester`` is already folded in behind, if any."""
        return self._route.get(requester)

    def routes(self) -> Mapping[str, str]:
        """``requester -> its source``: the tree, and the edges a decision walks."""
        return MappingProxyType(self._route)

    def _routed(self, action: Routed) -> None:
        """Fold ``requester`` in behind ``source``, and make it a source itself.

        One requester, one edge: re-routing replaces it rather than adding a second, so
        no bookkeeping drift can hand out more slots than the cap.
        """
        previous = self._route.get(action.requester)
        # Unmoved route.
        if previous == action.source:
            return
        if previous is not None:
            self._drop(previous)
        self._route[action.requester] = action.source
        self._load[action.source] += 1

    def named(self) -> Mapping[str, int]:
        """``source -> requesters currently routed to it``. Absent means none."""
        return MappingProxyType(self._load)

    def _retired(self, action: Retired) -> None:
        """Drop ``requester``'s route to a source nothing is coming from."""
        if self._route.get(action.requester) == action.source:
            del self._route[action.requester]
            self._drop(action.source)

    def _drop(self, source: str) -> None:
        """One reader fewer behind ``source``; absent at zero, as :meth:`named` says."""
        if self._load[source] > 1:
            self._load[source] -= 1
        else:
            del self._load[source]

    def owes(self, facts: Iterable[Tuple[str, str]]) -> bool:
        """Is every one of these ``(volume, key)`` publications still owed?"""
        return all(fact in self._promised for fact in facts)
