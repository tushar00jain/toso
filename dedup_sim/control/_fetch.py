"""Dedup fetch coverage and materialization."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from proposed import DirectorySensor, Key, VolumeId
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request

__all__ = ["FetchCoverage", "PlannedFetch"]

_Region = tuple[Key, object]


@dataclass(frozen=True)
class PlannedFetch:
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
class _SourceCoverage:
    source: VolumeId
    # Regions visible after overlaying promises on the pinned directory.
    combined: tuple[_Region, ...]
    # Regions registered in the pinned directory now.
    live: tuple[_Region, ...]


@dataclass(frozen=True)
class _KeyCoverage:
    key: Key
    # Raw live holders preserve the directory's fallback order.
    holders: tuple[VolumeId, ...]
    sources: tuple[_SourceCoverage, ...]


@dataclass(frozen=True)
class FetchCoverage:
    """One expansion of live and promised metadata for a read decision."""

    by_key: tuple[_KeyCoverage, ...]

    @classmethod
    def discover(
        cls,
        directory: DirectorySensor,
        requests: Sequence[Request],
        live: Mapping[Key, Mapping[VolumeId, StorageInfo]],
        promised: Mapping[Key, Mapping[VolumeId, Request]],
    ) -> "FetchCoverage":
        """Expand live and promised metadata once for this decision."""
        pending_entries = _promised_entries(promised)
        combined = _merged_entries(live, pending_entries, (r.key for r in requests))
        combined_plan = directory.requests_by_source(requests, combined)
        live_plan = directory.requests_by_source(requests, live)
        combined_regions = {
            source: directory.regions(parts) for source, parts in combined_plan.items()
        }
        live_regions = {
            source: directory.regions(parts) for source, parts in live_plan.items()
        }
        by_key: list[_KeyCoverage] = []
        for key in dict.fromkeys(request.key for request in requests):
            sources = tuple(
                _SourceCoverage(
                    source,
                    tuple(
                        region
                        for region in combined_regions.get(source, Counter()).elements()
                        if region[0] == key
                    ),
                    tuple(
                        region
                        for region in live_regions.get(source, Counter()).elements()
                        if region[0] == key
                    ),
                )
                for source in combined[key]
            )
            by_key.append(_KeyCoverage(key, tuple(live.get(key, {})), sources))
        return cls(tuple(by_key))

    @property
    def keys(self) -> tuple[Key, ...]:
        return tuple(entry.key for entry in self.by_key)

    @property
    def holders(self) -> tuple[VolumeId, ...]:
        return tuple(
            dict.fromkeys(source for entry in self.by_key for source in entry.holders)
        )

    @property
    def candidates(self) -> set[VolumeId]:
        return {
            source.source
            for entry in self.by_key
            for source in entry.sources
            if source.combined
        }

    @property
    def pending(self) -> set[VolumeId]:
        return {
            source.source
            for entry in self.by_key
            for source in entry.sources
            if Counter(source.combined) - Counter(source.live)
        }

    def plan(
        self,
        order: Sequence[VolumeId],
        *,
        requester: VolumeId | None = None,
    ) -> PlannedFetch:
        """Materialize an ordered fetch from this source coverage."""
        allowed = set(order)
        by_key: Dict[Key, tuple[VolumeId, ...]] = {}
        required: dict[VolumeId, Counter[_Region]] = defaultdict(Counter)
        pending: set[VolumeId] = set()
        for key_coverage in self.by_key:
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


def _promised_entries(
    promised: Mapping[Key, Mapping[VolumeId, Request]],
) -> Dict[Key, Dict[VolumeId, StorageInfo]]:
    return {
        key: {
            producer: StorageInfo(
                ObjectType.from_request(request), {request.tensor_slice}
            )
            for producer, request in producers.items()
        }
        for key, producers in promised.items()
    }


def _merged_entries(
    live: Mapping[Key, Mapping[VolumeId, StorageInfo]],
    promised: Mapping[Key, Mapping[VolumeId, StorageInfo]],
    keys: Iterable[Key],
) -> Dict[Key, Dict[VolumeId, StorageInfo]]:
    merged: Dict[Key, Dict[VolumeId, StorageInfo]] = {}
    for key in keys:
        entries = {
            source: StorageInfo(info.object_type, set(info.tensor_slices))
            for source, info in live.get(key, {}).items()
        }
        for source, info in promised.get(key, {}).items():
            previous = entries.get(source)
            if previous is None:
                entries[source] = StorageInfo(info.object_type, set(info.tensor_slices))
            elif previous.object_type == info.object_type:
                previous.update(info)
        merged[key] = entries
    return merged

