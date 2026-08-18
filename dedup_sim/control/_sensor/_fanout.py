"""The fan-out tree a dedup decision reads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping

from proposed import Key, LoadSensor, VolumeId
from proposed.dispatch import Action, Fold

from ._directory import Published

__all__ = ["FanoutSensor", "Retired", "Routed"]

_Region = tuple[Key, object]


@dataclass(frozen=True)
class Routed(Action):
    """``requester`` is routed through sources with exact required regions."""

    requester: VolumeId
    sources: tuple[VolumeId, ...]
    pending: tuple[VolumeId, ...] = ()
    required: tuple[tuple[VolumeId, tuple[_Region, ...]], ...] = ()


@dataclass(frozen=True)
class Retired(Action):
    """``requester`` no longer routes through ``source``."""

    requester: VolumeId
    source: VolumeId


class FanoutSensor(LoadSensor):
    """Who is folded in behind whom. Nobody's wait is here.

    Routes, retirements and publications move it through actions. A fold writes this
    state before its commit wakes anybody. Nothing here reads the directory's state,
    and nothing reads this.

    Args:
        fanout_cap: readers one peer may be planned to feed -- 1 a chain, >= 2 a
            shallow tree. A ceiling and nothing else: which peer a reader takes is
            priced (:class:`~dedup_sim.control._selector.Candidates`), and this only
            says when one stops being offered at all.
    """

    def __init__(self, fanout_cap: int = 1) -> None:
        self.cap = fanout_cap
        # requester -> the sources it was routed through.
        self._route: Dict[VolumeId, tuple[VolumeId, ...]] = {}
        self._route_pending: Dict[VolumeId, set[VolumeId]] = {}
        self._route_required: Dict[VolumeId, Dict[VolumeId, Counter[_Region]]] = {}
        # source -> how many are behind it, carried rather than counted on demand:
        # every decision reads it, and re-deriving it walks the whole tree. Moved only
        # by the two members that move a route, so the two cannot disagree.
        self._load: Counter = Counter()
        self._folds: Dict[type, Fold] = {
            Published: self._published,
            Retired: self._retired,
            Routed: self._routed,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """What it folds, by action type."""
        return MappingProxyType(self._folds)

    def _published(self, action: Published) -> None:
        """A reader's batch has landed: its pending route facts are settled."""
        self._route_pending.pop(action.producer, None)
        self._route_required.pop(action.producer, None)

    def planned(self, requester: str) -> tuple[VolumeId, ...]:
        """The sources ``requester`` is already folded in behind."""
        return self._route.get(requester, ())

    def routes(self) -> Mapping[VolumeId, tuple[VolumeId, ...]]:
        """``requester -> its sources``."""
        return MappingProxyType(self._route)

    def _routed(self, action: Routed) -> None:
        """Fold ``requester`` in behind ``sources``, and make it a source itself.

        Re-routing replaces every prior edge, so load and the cap stay aligned.
        """
        if len(set(action.sources)) != len(action.sources):
            raise ValueError("a route names each source once")
        required: Dict[VolumeId, Counter[_Region]] = {}
        for source, regions in action.required:
            if source in required:
                raise ValueError("a route names each source requirement once")
            required[source] = Counter(regions)
        if set(required) != set(action.sources) or any(
            not regions for regions in required.values()
        ):
            raise ValueError("a route requires non-empty regions for every source")
        if not set(action.pending).issubset(action.sources):
            raise ValueError("a pending source must belong to the route")
        previous = self._route.get(action.requester, ())
        # Unmoved route.
        if previous == action.sources:
            self._route_pending[action.requester] = set(action.pending)
            self._route_required[action.requester] = required
            return
        for source in previous:
            self._drop(source)
        self._route[action.requester] = action.sources
        self._route_pending[action.requester] = set(action.pending)
        self._route_required[action.requester] = required
        for source in action.sources:
            self._load[source] += 1

    def named(self) -> Mapping[str, int]:
        """``source -> requesters currently routed to it``. Absent means none."""
        return MappingProxyType(self._load)

    def _retired(self, action: Retired) -> None:
        """Drop ``requester``'s route to a source nothing is coming from."""
        previous = self._route.get(action.requester, ())
        if action.source in previous:
            remaining = tuple(source for source in previous if source != action.source)
            self._route[action.requester] = remaining
            self._route_pending[action.requester].discard(action.source)
            self._route_required[action.requester].pop(action.source, None)
            self._drop(action.source)

    def _drop(self, source: str) -> None:
        """One reader fewer behind ``source``; absent at zero, as :meth:`named` says."""
        if self._load[source] > 1:
            self._load[source] -= 1
        else:
            del self._load[source]

    def route_pending(self, producer: VolumeId) -> set[VolumeId]:
        """Sources ``producer`` is waiting to publish."""
        return set(self._route_pending.get(producer, set()))

    def route_required(self, producer: VolumeId) -> Mapping[VolumeId, Counter[_Region]]:
        """Exact regions each source must provide to ``producer``."""
        return MappingProxyType(self._route_required.get(producer, {}))
