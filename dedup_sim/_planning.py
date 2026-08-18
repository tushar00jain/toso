"""TorchStore metadata planning shared by dedup's two planes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from proposed import Key, VolumeId
from proposed.selector import prefer
from torchstore.client import LocalClient
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request

__all__ = ["PlannedFetch", "plan_fetch", "relevant_sources", "requirements_met"]

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


class _NoInplace:
    supports_inplace_resharding = False


_PLANNER: LocalClient = LocalClient.__new__(LocalClient)
_NO_INPLACE = _NoInplace()


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


def _volume_requests(
    requests: Sequence[Request],
    entries: Mapping[Key, Mapping[VolumeId, StorageInfo]],
) -> Dict[VolumeId, list[Request]]:
    maps = {request.key: dict(entries.get(request.key, {})) for request in requests}
    volumes = {source for by_source in maps.values() for source in by_source}
    transports = dict.fromkeys(volumes, _NO_INPLACE)
    planned, _whole = _PLANNER._build_volume_requests(list(requests), maps, transports)
    return planned


def plan_fetch(
    requests: Sequence[Request],
    live: Mapping[Key, Mapping[VolumeId, StorageInfo]],
    promised: Mapping[Key, Mapping[VolumeId, Request]],
    order: Sequence[VolumeId],
) -> PlannedFetch:
    pending_entries = _promised_entries(promised)
    combined = _merged_entries(live, pending_entries, (r.key for r in requests))
    by_key: Dict[Key, tuple[VolumeId, ...]] = {}
    required: dict[VolumeId, Counter[_Region]] = defaultdict(Counter)
    pending: set[VolumeId] = set()

    for request in requests:
        key = request.key
        combined_key = prefer({key: combined[key]}, order)[key]
        planned = _volume_requests([request], {key: combined_key})
        live_planned = _each_source([request], {key: live.get(key, {})})
        live_regions = {
            source: Counter(_region(part) for part in parts)
            for source, parts in live_planned.items()
        }
        seen: set[_Region] = set()
        selected: list[VolumeId] = []
        for source in combined_key:
            regions = Counter(_region(part) for part in planned.get(source, ()))
            novel = [region for region in regions if region not in seen]
            if not novel:
                continue
            selected.append(source)
            seen.update(novel)
            required[source].update(regions)
            if regions - live_regions.get(source, Counter()):
                pending.add(source)
        by_key[key] = tuple(selected)

    return PlannedFetch(by_key, dict(required), frozenset(pending))


def relevant_sources(
    requests: Sequence[Request],
    live: Mapping[Key, Mapping[VolumeId, StorageInfo]],
    promised: Mapping[Key, Mapping[VolumeId, Request]],
) -> tuple[set[VolumeId], set[VolumeId]]:
    pending_entries = _promised_entries(promised)
    combined = _merged_entries(live, pending_entries, (r.key for r in requests))
    combined_plan = _each_source(requests, combined)
    live_plan = _each_source(requests, live)
    live_regions = {
        source: Counter(_region(part) for part in parts)
        for source, parts in live_plan.items()
    }
    pending = {
        source
        for source, parts in combined_plan.items()
        if Counter(_region(part) for part in parts)
        - live_regions.get(source, Counter())
    }
    return set(combined_plan), pending


def _each_source(
    requests: Sequence[Request],
    entries: Mapping[Key, Mapping[VolumeId, StorageInfo]],
) -> Dict[VolumeId, list[Request]]:
    requests_by_source: dict[VolumeId, list[Request]] = defaultdict(list)
    entries_by_source: dict[VolumeId, dict[Key, dict[VolumeId, StorageInfo]]] = (
        defaultdict(dict)
    )
    for request in requests:
        for source, info in entries.get(request.key, {}).items():
            requests_by_source[source].append(request)
            entries_by_source[source][request.key] = {source: info}
    planned: Dict[VolumeId, list[Request]] = {}
    for source, source_requests in requests_by_source.items():
        parts = _volume_requests(source_requests, entries_by_source[source]).get(
            source, []
        )
        if parts:
            planned[source] = parts
    return planned


def requirements_met(
    requests: Sequence[Request],
    live: Mapping[Key, Mapping[VolumeId, StorageInfo]],
    required: Mapping[VolumeId, Counter[_Region]],
) -> bool:
    by_key = {request.key: request for request in requests}
    for source, expected in required.items():
        source_requests = [
            by_key[key] for key in dict.fromkeys(key for key, _region in expected)
        ]
        entries = {
            request.key: {source: live[request.key][source]}
            for request in source_requests
            if source in live.get(request.key, {})
        }
        actual = Counter(
            _region(part)
            for parts in _volume_requests(source_requests, entries).values()
            for part in parts
        )
        if expected - actual:
            return False
    return True


def _region(request: Request) -> _Region:
    tensor_slice = request.tensor_slice
    if tensor_slice is None:
        return request.key, None
    return request.key, (
        tuple(tensor_slice.offsets),
        tuple(tensor_slice.local_shape),
        tuple(tensor_slice.global_shape),
    )
