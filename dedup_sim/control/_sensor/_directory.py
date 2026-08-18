"""Dedup's pending overlay on the live TorchStore directory."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

from proposed import Controller, DirectorySensor, Key, VolumeId
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


@dataclass(frozen=True)
class _SourceCoverage:
    source: VolumeId
    # Regions visible after overlaying pending entries on the pinned directory.
    combined: tuple[_Region, ...]
    # Regions registered in the pinned directory now.
    live: tuple[_Region, ...]


@dataclass(frozen=True)
class _KeyCoverage:
    key: Key
    # Raw live holders preserve the directory's fallback order.
    holders: tuple[VolumeId, ...]
    sources: tuple[_SourceCoverage, ...]


class DedupDirectorySensor(DirectorySensor):
    """Live directory metadata plus dedup's separately committed pending entries."""

    def __init__(self, directory: Controller) -> None:
        super().__init__(directory)
        self._pending: Dict[VolumeId, Dict[Key, _PendingEntry]] = {}
        self._pending_by_key: Dict[Key, Dict[VolumeId, _PendingEntry]] = {}
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
        entries = {
            key: _PendingEntry(
                request,
                StorageInfo(ObjectType.from_request(request), {request.tensor_slice}),
            )
            for key, request in requests.items()
        }
        self._pending[action.requester] = entries
        for key, entry in entries.items():
            self._pending_by_key.setdefault(key, {})[action.requester] = entry

    def _published(self, action: Published) -> None:
        entries = self._pending.pop(action.producer, {})
        for key in entries:
            producers = self._pending_by_key[key]
            del producers[action.producer]
            if not producers:
                del self._pending_by_key[key]

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
        coverage = self._coverage(requests)
        candidates = {
            source.source
            for entry in coverage
            for source in entry.sources
            if source.combined
        }
        pending = {
            source.source
            for entry in coverage
            for source in entry.sources
            if Counter(source.combined) - Counter(source.live)
        }
        return candidates, pending

    def plan_fetch(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: VolumeId | None = None,
    ) -> PlannedFetch:
        """Materialize an ordered fetch from live and pending metadata."""
        allowed = set(order)
        by_key: Dict[Key, tuple[VolumeId, ...]] = {}
        required: dict[VolumeId, Counter[_Region]] = defaultdict(Counter)
        pending: set[VolumeId] = set()
        for key_coverage in self._coverage(requests):
            holders = set(key_coverage.holders)
            present = {
                source.source
                for source in key_coverage.sources
                if source.source in holders
                or (source.source in allowed and source.source != requester)
            }
            # Ranked-out and self promises cannot become fallback; live regions remain.
            effective = {
                source.source: (
                    source.combined
                    if source.source in allowed and source.source != requester
                    else source.live
                )
                for source in key_coverage.sources
            }
            ranked = [
                (source, effective[source]) for source in order if source in present
            ]
            choices = ranked or [
                (source.source, effective[source.source])
                for source in key_coverage.sources
                if source.source in present
            ]
            seen: set[_Region] = set()
            selected: list[VolumeId] = []
            live = {
                source.source: Counter(source.live) for source in key_coverage.sources
            }
            for source, parts in choices:
                regions = Counter(parts)
                novel = [region for region in regions if region not in seen]
                if not novel:
                    continue
                selected.append(source)
                seen.update(novel)
                required[source].update(regions)
                if regions - live.get(source, Counter()):
                    pending.add(source)
            by_key[key_coverage.key] = tuple(selected)
        return PlannedFetch(by_key, dict(required), frozenset(pending))

    def _coverage(self, requests: Sequence[Request]) -> tuple[_KeyCoverage, ...]:
        keys = tuple(dict.fromkeys(request.key for request in requests))
        live = self.locate(keys)
        combined = self._merged(live, self.pending(keys), keys)
        combined_plan = self.requests_by_source(requests, combined)
        live_plan = self.requests_by_source(requests, live)
        combined_regions = {
            source: self.regions(parts) for source, parts in combined_plan.items()
        }
        live_regions = {
            source: self.regions(parts) for source, parts in live_plan.items()
        }
        return tuple(
            _KeyCoverage(
                key,
                tuple(live.get(key, {})),
                tuple(
                    _SourceCoverage(
                        source,
                        tuple(
                            region
                            for region in combined_regions.get(
                                source, Counter()
                            ).elements()
                            if region[0] == key
                        ),
                        tuple(
                            region
                            for region in live_regions.get(source, Counter()).elements()
                            if region[0] == key
                        ),
                    )
                    for source in combined[key]
                ),
            )
            for key in keys
        )

    @staticmethod
    def _merged(
        live: Mapping[Key, Mapping[VolumeId, StorageInfo]],
        pending: Mapping[Key, Mapping[VolumeId, StorageInfo]],
        keys: Sequence[Key],
    ) -> Dict[Key, Dict[VolumeId, StorageInfo]]:
        merged: Dict[Key, Dict[VolumeId, StorageInfo]] = {}
        for key in keys:
            entries = {
                source: StorageInfo(info.object_type, set(info.tensor_slices))
                for source, info in live.get(key, {}).items()
            }
            for source, info in pending.get(key, {}).items():
                previous = entries.get(source)
                if previous is None:
                    entries[source] = StorageInfo(
                        info.object_type, set(info.tensor_slices)
                    )
                elif previous.object_type == info.object_type:
                    previous.update(info)
            merged[key] = entries
        return merged
