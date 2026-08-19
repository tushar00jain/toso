"""Publication arrival scores and pending fan-out counts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from proposed import LoadSensor, VolumeId
from proposed.dispatch import Action, Fold

from ._directory import Pub, Published

__all__ = ["FanoutSensor", "Routed"]


@dataclass(frozen=True)
class Routed(Action):
    """One publication has a ranked preference and readiness gate."""

    requester_pub: Pub
    sources: tuple[VolumeId, ...]
    gate_pubs: frozenset[Pub]
    arrival: float


class FanoutSensor(LoadSensor):
    """Pending readers per volume and publication arrival scores."""

    def __init__(self, fanout_cap: int = 1) -> None:
        self.cap = fanout_cap
        self._arrival: dict[Pub, float] = {}
        self._behind: dict[VolumeId, int] = {}
        self._assigned: dict[VolumeId, set[VolumeId]] = {}
        self._pending: dict[Pub, dict[VolumeId, set[Pub]]] = {}
        self._folds: dict[type, Fold] = {
            Published: self._published,
            Routed: self._routed,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        return MappingProxyType(self._folds)

    def _routed(self, action: Routed) -> None:
        self._arrival.setdefault(action.requester_pub, action.arrival)
        requester = action.requester_pub[0]
        for source in self._assigned.pop(requester, set()):
            self._decrement(source)
        for pub in tuple(self._pending):
            if pub[0] == requester:
                del self._pending[pub]
        assigned = set(action.sources[:1])
        self._assigned[requester] = assigned
        for source in assigned:
            self._behind[source] = self._behind.get(source, 0) + 1
        by_volume: dict[VolumeId, set[Pub]] = {}
        for pub in action.gate_pubs:
            if pub[0] in assigned:
                by_volume.setdefault(pub[0], set()).add(pub)
        if not by_volume:
            return
        self._pending[action.requester_pub] = by_volume

    def _published(self, action: Published) -> None:
        pub = action.publication
        self._arrival.pop(pub, None)
        for requester, by_volume in tuple(self._pending.items()):
            gates = by_volume.get(action.volume)
            if gates is None or pub not in gates:
                continue
            gates.discard(pub)
            if not gates:
                del by_volume[action.volume]
                self._decrement(action.volume)
                assigned = self._assigned.get(requester[0])
                if assigned is not None:
                    assigned.discard(action.volume)
            if not by_volume:
                del self._pending[requester]

    def _decrement(self, volume: VolumeId) -> None:
        remaining = self._behind[volume] - 1
        if remaining:
            self._behind[volume] = remaining
        else:
            del self._behind[volume]

    def named(self) -> Mapping[VolumeId, int]:
        return MappingProxyType(self._behind)

    def arrival(self, pub: Pub) -> float | None:
        return self._arrival.get(pub)
