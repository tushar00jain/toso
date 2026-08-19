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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import AbstractSet, Any, Dict, Mapping, Optional, Sequence

from proposed import (
    Controller,
    DirectorySensor,
    Key,
    VolumeId,
)
from proposed.planner import region
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
    #: The keys that really landed, or ``None`` where the caller does not say. A
    #: producer may publish fewer keys than it promised, so a route waiting on one of
    #: these is settled by what is named here and not by the action arriving.
    #:
    #: Out of the equality a gate is keyed on: a waiter names ``Published(producer)``
    #: before anybody knows what will land (:meth:`proposed.dispatch.Dispatcher.gate`).
    landed: Optional[frozenset[Key]] = field(default=None, compare=False)


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
        # producer -> the directory's mutation count when its batch landed. Only a
        # deregistration can take a landed region back, and that moves the count, so an
        # unmoved one says the copy is still there without reading for it.
        self._landed: Dict[VolumeId, object] = {}
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
        # Read after the put that landed, and clearing a promise is not a mutation, so
        # this is the count a later decision compares against.
        self._landed[action.producer] = self.moves()
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

    def settled(self, producer: VolumeId) -> bool:
        """Whether ``producer``'s publication is still all the directory says."""
        # Absent, or a directory counting no mutations: nothing to answer from.
        landed = self._landed.get(producer)
        return landed is not None and landed == self.moves()

    def promised(
        self, keys: Sequence[Key]
    ) -> Mapping[Key, Mapping[VolumeId, StorageInfo]]:
        """The directory for ``keys`` with promised entries left in."""
        if not keys or not self._batches:
            return {}
        return self.directory.locate_raw(list(keys), missing_ok=True, projected=True)

    def owners(self) -> AbstractSet[VolumeId]:
        """The producers whose promises this plane may plan a reader onto.

        This plane's own, and that membership is the whole point. The directory is
        shared, so two dedup planes over one store see each other's promises; a plane
        that routed onto a foreign one would gate on a publication it never hears.
        """
        return self._batches.keys()

    def serving_sources(
        self, requests: Sequence[Request]
    ) -> tuple[set[VolumeId], set[VolumeId]]:
        """Sources serving any requested region, and the pending subset.

        Costs one live per-source expansion plus one test per producer, and never a
        walk of the promised entries: a producer that promised this exact batch is
        answered from that fact; one that promised a different shape is planned
        against its own metadata over the keys the two share, so a promise whose
        slices miss the request is not offered.

        The pending subset here is the *pricing* question -- whose copy has not
        arrived, so a wait has to be priced for it. What the chosen sources still owe
        is a different question, answered by :meth:`plan_fetch` once a plan exists.
        """
        actual = self.requirements(requests)
        candidates = set(actual)
        wanted = self._shapes.get(self.batch_spec(requests))
        whole = None
        pending: set[VolumeId] = set()
        for producer, batch in self._batches.items():
            if batch.shape is wanted:
                if whole is None:
                    # A promise of exactly this batch serves exactly its regions.
                    whole = Counter(region(request) for request in requests)
                regions = whole
            else:
                regions = self._overlap(producer, batch, requests)
            if not regions:
                continue
            candidates.add(producer)
            if regions - actual.get(producer, Counter()):
                pending.add(producer)
        return candidates, pending

    def _overlap(
        self, producer: VolumeId, batch: _Batch, requests: Sequence[Request]
    ) -> Counter[_Region]:
        """Regions ``batch``'s differently shaped promise serves of ``requests``."""
        promised = {key: {producer: info} for key, info in batch.infos.items()}
        return self.regions_by_source(requests, promised).get(producer, Counter())

    def plan_fetch(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: VolumeId | None = None,
    ) -> PlannedFetch:
        """Materialize an ordered fetch from live and promised metadata."""
        by_key, required = self._planned(requests, order, requester)
        actual = self.requirements(requests)
        empty: Counter[_Region] = Counter()
        # The gating question: of the sources the plan chose, which owe a region the
        # directory does not live-hold yet. Not the pricing question above -- a
        # producer already holding the half this reader wants owes nothing.
        pending = frozenset(
            source
            for source, expected in required.items()
            if expected - actual.get(source, empty)
        )
        return PlannedFetch(by_key, required, pending)
