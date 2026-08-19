"""What dedup promises the directory, and what a decision reads back out of it.

A reader routed onto a peer is routed against the directory as it *will be*, so this
sensor writes each in-flight batch into the directory as a promise
(:meth:`~proposed.deployment.Controller.project`) and reads it back through the same
lookup as a live holder. No overlay is joined per request: a promise landing on a
volume that already holds part of the key covers both halves in the one entry, and a
real put replaces the promise rather than merging into it.

What stays here is what a ``StorageInfo`` cannot carry: the :class:`Request` a
producer promised, which is what its own route is planned from, and the *shape* of
its batch, which is what answers "does this producer promise what I am asking for"
in one identity test rather than a walk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence

from proposed import (
    Controller,
    DirectorySensor,
    Key,
    VolumeId,
)
from proposed.sensors import FetchPlan
from proposed.dispatch import Action, Fold
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request

__all__ = ["Asked", "DedupDirectorySensor", "PlannedFetch", "Published"]

_Region = tuple[Key, object]
_BatchSpec = tuple[object, ...]


@dataclass(frozen=True)
class Asked(Action):
    """``requester`` is about to read ``requests`` through and publish them."""

    requester: VolumeId
    requests: tuple[Any, ...]

    def __hash__(self) -> int:
        return hash((type(self), self.requester))


@dataclass(frozen=True)
class Published(Action):
    """``producer``'s pending read-through batch has landed."""

    producer: VolumeId


@dataclass(frozen=True)
class PlannedFetch(FetchPlan):
    """A fetch plan and the producers whose copies have not landed yet."""

    pending: frozenset[VolumeId]


@dataclass(frozen=True)
class _Batch:
    """One producer's in-flight promise."""

    #: ``key -> the request it promised``, which is what :meth:`plan` answers.
    requests: Mapping[Key, Request]
    #: ``key -> the metadata handed to the directory``. The same objects the
    #: directory holds, so what is read back out of it is what was promised.
    infos: Mapping[Key, StorageInfo]
    #: One shared object per distinct batch shape, so "does this producer promise
    #: exactly what is being asked for" is an identity test (:meth:`_shape`).
    shape: _BatchSpec


class DedupDirectorySensor(DirectorySensor):
    """Live directory metadata plus dedup's own promises to the same directory."""

    def __init__(self, directory: Controller) -> None:
        super().__init__(directory)
        self._batches: Dict[VolumeId, _Batch] = {}
        self._shapes: Dict[_BatchSpec, _BatchSpec] = {}
        self._shape_uses: Counter[_BatchSpec] = Counter()
        self._folds: Dict[type, Fold] = {
            Asked: self._asked,
            Published: self._published,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """What moves the promised directory, by action type."""
        return MappingProxyType(self._folds)

    def _asked(self, action: Asked) -> None:
        if action.requester in self._batches:
            raise ValueError(f"{action.requester} already has an in-flight publication")
        requests = {request.key: request for request in action.requests}
        if len(requests) != len(action.requests):
            raise ValueError("a publication names each key once")
        infos: Dict[Key, StorageInfo] = {}
        specs = []
        for key, request in requests.items():
            info = StorageInfo(ObjectType.from_request(request), {request.tensor_slice})
            infos[key] = info
            specs.append(self.request_spec(request))
            self.directory.project(action.requester, key, info)
        self._batches[action.requester] = _Batch(
            MappingProxyType(requests), infos, self._shape(tuple(specs))
        )

    def _published(self, action: Published) -> None:
        # A producer may publish fewer keys than it promised, and the directory keeps
        # a promise until it is told otherwise, so the remainder is cleared here.
        self.directory.clear_projections(action.producer)
        batch = self._batches.pop(action.producer, None)
        if batch is None:
            return
        self._shape_uses[batch.shape] -= 1
        if not self._shape_uses[batch.shape]:
            del self._shape_uses[batch.shape]
            del self._shapes[batch.shape]

    def _shape(self, spec: _BatchSpec) -> _BatchSpec:
        """The one object standing for every batch asking for ``spec``."""
        shared = self._shapes.setdefault(spec, spec)
        self._shape_uses[shared] += 1
        return shared

    def plan(self, producer: VolumeId) -> Mapping[Key, Request]:
        """``producer``'s pending read-through requests, by key."""
        return self._batches[producer].requests

    def in_flight(self) -> set[VolumeId]:
        """Producers with promises the directory is still holding."""
        return set(self._batches)

    def promised(
        self, keys: Sequence[Key]
    ) -> Mapping[Key, Mapping[VolumeId, StorageInfo]]:
        """The directory for ``keys`` with promised entries left in."""
        if not keys or not self._batches:
            return {}
        return self.directory.locate_raw(list(keys), missing_ok=True, projected=True)

    def serving_sources(
        self, requests: Sequence[Request]
    ) -> tuple[set[VolumeId], set[VolumeId]]:
        """Sources serving any requested region, and the pending subset.

        Costs one live coverage plus one test per producer, and never a walk of the
        promised entries: a producer that promised this exact batch is answered from
        that fact; one that promised a different shape is checked region by region
        over the keys the two share, so a promise whose slices miss the request is
        not offered.
        """
        candidates = self.coverage(requests).sources
        actual = self.requirements(requests)
        wanted = self._shapes.get(self.batch_spec(requests))
        whole = None
        pending: set[VolumeId] = set()
        for producer, batch in self._batches.items():
            if batch.shape is wanted:
                if whole is None:
                    whole = self._whole_batch(requests)
                regions = whole
            else:
                regions = self._overlap(batch, requests)
            if not regions:
                continue
            candidates.add(producer)
            if regions - actual.get(producer, Counter()):
                pending.add(producer)
        return candidates, pending

    def _whole_batch(self, requests: Sequence[Request]) -> Counter[_Region]:
        """Every region ``requests`` asks for, which is what a matching promise owes."""
        regions: Counter[_Region] = Counter()
        for request in requests:
            regions.update(self.whole_regions(request))
        return regions

    def _overlap(self, batch: _Batch, requests: Sequence[Request]) -> Counter[_Region]:
        """Regions ``batch``'s differently shaped promise serves of ``requests``."""
        regions: Counter[_Region] = Counter()
        for request in requests:
            info = batch.infos.get(request.key)
            if info is not None:
                regions.update(self.expand_regions(request, info))
        return regions

    def servable(
        self,
        key: Key,
        source: VolumeId,
        requests: Sequence[Request],
        live_regions: Mapping[VolumeId, tuple[_Region, ...]],
        promised: Mapping[Key, Mapping[VolumeId, StorageInfo]],
    ) -> Optional[tuple[_Region, ...]]:
        """What ``source`` holds for ``key``, plus what it has promised to hold.

        A promised entry already covers the live one it landed on, so expanding it
        answers for a volume holding part of the key and promising the rest without
        merging anything here.

        A promise this plane did not make is not a source for it. The directory is
        shared, so two dedup planes over one store see each other's promises here;
        the membership test is what keeps a plane answering only from its own
        decisions and waiting only on publications it will hear about.
        """
        if source in self._batches:
            info = promised.get(key, {}).get(source)
            if info is not None:
                return tuple(
                    region
                    for request in requests
                    for region in self.expand_regions(request, info)
                )
        return live_regions.get(source)

    def plan_fetch(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: VolumeId | None = None,
    ) -> PlannedFetch:
        """Materialize an ordered fetch from live and promised metadata."""
        by_key, required = self.project(requests, order, requester=requester)
        actual = self.requirements(requests)
        empty: Counter[_Region] = Counter()
        pending = frozenset(
            source
            for source, expected in required.items()
            if expected - actual.get(source, empty)
        )
        return PlannedFetch(by_key, required, pending)
