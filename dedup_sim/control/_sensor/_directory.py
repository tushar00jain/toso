"""Pending publications visible to dedup decisions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from proposed import DirectorySensor, VolumeId
from proposed.dispatch import Action, Fold

__all__ = ["Asked", "DedupDirectorySensor", "Pub", "Published"]

Pub = tuple[VolumeId, int]


@dataclass(frozen=True)
class Asked(Action):
    """A store declaration created one publication."""

    pub: Pub


@dataclass(frozen=True)
class Published(Action):
    """One publication landed or was abandoned."""

    volume: VolumeId
    pub: int

    @property
    def publication(self) -> Pub:
        return self.volume, self.pub


class DedupDirectorySensor(DirectorySensor):
    """Resolve store publication ids to their volume identities."""

    def __init__(self, directory) -> None:
        super().__init__(directory)
        self._in_flight: dict[int, Pub] = {}
        self._folds: dict[type, Fold] = {
            Asked: self._asked,
            Published: self._published,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        return MappingProxyType(self._folds)

    def declare(self, requester: VolumeId, requests: Sequence[Any]) -> Pub:
        pub = self.directory.notify_put_batch(requests, requester, pending=True)
        return requester, pub

    def _asked(self, action: Asked) -> None:
        self._in_flight[action.pub[1]] = action.pub

    def _published(self, action: Published) -> None:
        self.directory.notify_delete_batch(pub=action.pub)
        self._in_flight.pop(action.pub, None)

    def serving_union(
        self, requests: Sequence[Any]
    ) -> tuple[dict[str, set[VolumeId]], dict[str, set[Pub]]]:
        volumes, pub_ids = self.directory.serving_union(requests)
        return volumes, {
            key: {
                self._in_flight[pub_id]
                for pub_id in ids
                if pub_id in self._in_flight
            }
            for key, ids in pub_ids.items()
        }

    def in_flight(self) -> set[Pub]:
        return set(self._in_flight.values())

    def is_in_flight(self, pub: Pub) -> bool:
        return self._in_flight.get(pub[1]) == pub
