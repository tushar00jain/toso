"""Pending publications visible to dedup decisions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from proposed import DirectorySensor, VolumeId
from proposed.dispatch import Action, Fold
from torchstore import Publication

__all__ = ["Asked", "DedupDirectorySensor", "Published"]


@dataclass(frozen=True)
class Asked(Action):
    """A store declaration created one publication."""

    pub: Publication


@dataclass(frozen=True)
class Published(Action):
    """One publication has landed."""

    publication: Publication


class DedupDirectorySensor(DirectorySensor):
    """Resolve store publication ids to their volume identities."""

    def __init__(self, directory) -> None:
        super().__init__(directory)
        self._in_flight: dict[int, Publication] = {}
        self._folds: dict[type, Fold] = {
            Asked: self._asked,
            Published: self._published,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        return MappingProxyType(self._folds)

    def declare(
        self, requester: VolumeId, requests: Sequence[Any]
    ) -> Publication:
        pub = self.directory.notify_put_batch(requests, requester, pending=True)
        return pub, requester

    def _asked(self, action: Asked) -> None:
        self._in_flight[action.pub[0]] = action.pub

    def _published(self, action: Published) -> None:
        pub, _volume = action.publication
        self.directory.notify_delete_batch(pub=pub)
        self._in_flight.pop(pub, None)

    def serving_union(
        self, requests: Sequence[Any]
    ) -> frozenset[Publication]:
        return self.directory.serving_union(requests)

    def greedy_cover(
        self, requests: Sequence[Any], ranked: Iterable[Publication]
    ) -> list[Publication]:
        return self.directory.greedy_cover(requests, ranked)

    def in_flight(self) -> set[Publication]:
        return set(self._in_flight.values())

    def is_in_flight(self, publication: Publication) -> bool:
        return self._in_flight.get(publication[0]) == publication
