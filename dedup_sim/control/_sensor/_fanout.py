"""Publication arrival scores and pending fan-out counts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from proposed import LoadSensor, VolumeId
from proposed.dispatch import Action, Fold
from torchstore import Publication

from ._directory import Published

__all__ = ["FanoutSensor", "Routed"]


@dataclass(frozen=True)
class Routed(Action):
    """One publication has a ranked preference and readiness gate."""

    requester_pub: Publication
    sources: tuple[VolumeId, ...]
    gate_pubs: frozenset[Publication]
    arrival: float


class FanoutSensor(LoadSensor):
    """Pending readers per volume and publication arrival scores."""

    def __init__(self, fanout_cap: int = 1) -> None:
        self.cap = fanout_cap
        self._arrival: dict[Publication, float] = {}
        self._behind: dict[VolumeId, int] = {}
        self._assigned: dict[VolumeId, set[VolumeId]] = {}
        self._gates: dict[
            Publication, dict[VolumeId, set[Publication]]
        ] = {}
        self._folds: dict[type, Fold] = {
            Published: self._published,
            Routed: self._routed,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        return MappingProxyType(self._folds)

    def _routed(self, action: Routed) -> None:
        self._arrival.setdefault(action.requester_pub, action.arrival)
        requester = action.requester_pub[1]
        for source in self._assigned.pop(requester, set()):
            self._decrement(source)
        for pub in tuple(self._gates):
            if pub[1] == requester:
                del self._gates[pub]
        assigned = set(action.sources[:1])
        self._assigned[requester] = assigned
        for source in assigned:
            self._behind[source] = self._behind.get(source, 0) + 1
        by_volume: dict[VolumeId, set[Publication]] = {}
        for pub in action.gate_pubs:
            if pub[1] in assigned:
                by_volume.setdefault(pub[1], set()).add(pub)
        if not by_volume:
            return
        self._gates[action.requester_pub] = by_volume

    def _published(self, action: Published) -> None:
        pub = action.publication
        self._arrival.pop(pub, None)
        volume = pub[1]
        for requester, by_volume in tuple(self._gates.items()):
            gates = by_volume.get(volume)
            if gates is None or pub not in gates:
                continue
            gates.discard(pub)
            if not gates:
                del by_volume[volume]
                self._decrement(volume)
                assigned = self._assigned.get(requester[1])
                if assigned is not None:
                    assigned.discard(volume)
            if not by_volume:
                del self._gates[requester]

    def _decrement(self, volume: VolumeId) -> None:
        remaining = self._behind[volume] - 1
        if remaining:
            self._behind[volume] = remaining
        else:
            del self._behind[volume]

    def named(self) -> Mapping[VolumeId, int]:
        return MappingProxyType(self._behind)

    def arrival(self, publication: Publication) -> float | None:
        return self._arrival.get(publication)
