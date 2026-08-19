"""Dedup's pending overlay on the live TorchStore directory."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

from proposed import (
    Controller,
    DirectoryCoverage,
    DirectorySensor,
    Key,
    VolumeId,
)
from proposed.sensors import KeyCoverage, SourceCoverage
from proposed.dispatch import Action, Fold
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request

__all__ = ["Asked", "DedupDirectorySensor", "PlannedFetch", "Published"]

_Region = tuple[Key, object]


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
class PlannedFetch:
    """Per-key sources, exact requirements, and pending producers."""

    by_key: Mapping[Key, tuple[VolumeId, ...]]
    required: Mapping[VolumeId, Counter[_Region]]
    pending: frozenset[VolumeId]

    @property
    def sources(self) -> tuple[VolumeId, ...]:
        return tuple(
            dict.fromkeys(
                source for sources in self.by_key.values() for source in sources
            )
        )


@dataclass(frozen=True)
class _PendingEntry:
    request: Request
    info: StorageInfo
    request_spec: object
    regions: tuple[_Region, ...]
    alternate_regions: Dict[object, tuple[_Region, ...]] = field(default_factory=dict)


class DedupDirectorySensor(DirectorySensor):
    """Live directory metadata plus dedup's separately committed pending entries."""

    def __init__(self, directory: Controller) -> None:
        super().__init__(directory)
        self._pending: Dict[VolumeId, Dict[Key, _PendingEntry]] = {}
        self._pending_by_key: Dict[Key, Dict[VolumeId, _PendingEntry]] = {}
        self._pending_generation = 0
        self._folds: Dict[type, Fold] = {
            Asked: self._asked,
            Published: self._published,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """What moves the pending overlay, by action type."""
        return MappingProxyType(self._folds)

    def _asked(self, action: Asked) -> None:
        if action.requester in self._pending:
            raise ValueError(f"{action.requester} already has an in-flight publication")
        requests = {request.key: request for request in action.requests}
        if len(requests) != len(action.requests):
            raise ValueError("a publication names each key once")
        entries = {}
        for key, request in requests.items():
            info = StorageInfo(ObjectType.from_request(request), {request.tensor_slice})
            spec = self.request_spec(request)
            entry = _PendingEntry(
                request,
                info,
                spec,
                self.expand_regions(request, info),
            )
            entries[key] = entry
        self._pending[action.requester] = entries
        for key, entry in entries.items():
            self._pending_by_key.setdefault(key, {})[action.requester] = entry
        self._pending_generation += 1

    def _published(self, action: Published) -> None:
        entries = self._pending.pop(action.producer, {})
        for key in entries:
            producers = self._pending_by_key[key]
            del producers[action.producer]
            if not producers:
                del self._pending_by_key[key]
        if entries:
            self._pending_generation += 1

    def plan(self, producer: VolumeId) -> Mapping[Key, Request]:
        """``producer``'s pending read-through requests, by key."""
        return MappingProxyType(
            {key: entry.request for key, entry in self._pending[producer].items()}
        )

    def pending(self, keys: Sequence[Key]) -> Dict[Key, Mapping[VolumeId, StorageInfo]]:
        """Pending-only directory metadata for ``keys``."""
        return {
            key: MappingProxyType(
                {
                    producer: entry.info
                    for producer, entry in self._pending_by_key.get(key, {}).items()
                }
            )
            for key in keys
        }

    def in_flight(self) -> set[VolumeId]:
        """Producers with pending directory entries."""
        return set(self._pending)

    def serving_sources(
        self, requests: Sequence[Request]
    ) -> tuple[set[VolumeId], set[VolumeId]]:
        """Sources serving any requested region, and the pending subset."""
        combined, live = self._coverages(requests)
        candidates = combined.sources
        pending = live.missing_sources(combined.requirements())
        return candidates, pending

    def plan_fetch(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: VolumeId | None = None,
    ) -> PlannedFetch:
        """Materialize an ordered fetch from live and pending metadata."""
        combined, live = self._coverages(requests)
        fetch = self.plan_coverage(combined, live, order, requester=requester)
        pending = live.missing_sources(fetch.required)
        return PlannedFetch(fetch.by_key, fetch.required, frozenset(pending))

    def _coverages(
        self, requests: Sequence[Request]
    ) -> tuple[DirectoryCoverage, DirectoryCoverage]:
        keys = tuple(dict.fromkeys(request.key for request in requests))
        live = self.locate(keys)
        live_coverage = (
            self.coverage(requests)
            if self._keys is not None
            else self.coverage(requests, live)
        )
        overlay_spec = (
            "dedup-pending",
            tuple(self.request_spec(request) for request in requests),
            self._pending_generation,
        )
        combined = self.decision_coverage(
            overlay_spec,
            lambda: self._overlay_coverage(requests, live, live_coverage),
        )
        return combined, live_coverage

    def _overlay_coverage(
        self,
        requests: Sequence[Request],
        live: Mapping[Key, Mapping[VolumeId, StorageInfo]],
        live_coverage: DirectoryCoverage,
    ) -> DirectoryCoverage:
        requests_by_key: Dict[Key, list[Request]] = {}
        for request in requests:
            requests_by_key.setdefault(request.key, []).append(request)
        combined = []
        for live_entry in live_coverage.entries:
            live_sources = {source.source: source for source in live_entry.sources}
            pending = self._pending_by_key.get(live_entry.key, {})
            sources = tuple(dict.fromkeys((*live_sources, *pending)))
            combined_sources = []
            for source in sources:
                live_source = live_sources.get(source)
                pending_entry = pending.get(source)
                if live_source is None:
                    assert pending_entry is not None
                    regions = tuple(
                        region
                        for request in requests_by_key[live_entry.key]
                        for region in self._pending_regions(request, pending_entry)
                    )
                elif pending_entry is None:
                    regions = live_source.regions
                else:
                    regions = self._merged_regions(
                        requests_by_key[live_entry.key],
                        live[live_entry.key][source],
                        pending_entry.info,
                        live_source.regions,
                    )
                combined_sources.append(SourceCoverage(source, regions))
            combined.append(
                KeyCoverage(live_entry.key, sources, tuple(combined_sources))
            )
        return DirectoryCoverage(tuple(combined))

    def _pending_regions(
        self, request: Request, entry: _PendingEntry
    ) -> tuple[_Region, ...]:
        spec = self.request_spec(request)
        if spec == entry.request_spec:
            return entry.regions
        regions = entry.alternate_regions.get(spec)
        if regions is None:
            regions = self.expand_regions(request, entry.info)
            entry.alternate_regions[spec] = regions
        return regions

    def _merged_regions(
        self,
        requests: Sequence[Request],
        live: StorageInfo,
        pending: StorageInfo,
        live_regions: tuple[_Region, ...],
    ) -> tuple[_Region, ...]:
        if live.object_type != pending.object_type:
            return live_regions
        added = pending.tensor_slices - live.tensor_slices
        if not added:
            return live_regions
        merged_slices = set(live.tensor_slices)
        merged_slices.update(pending.tensor_slices)
        merged = StorageInfo(live.object_type, merged_slices)
        return tuple(
            region
            for request in requests
            for region in self.expand_regions(request, merged)
        )
