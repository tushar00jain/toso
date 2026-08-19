"""Domain observations shared by selectors."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from proposed.deployment import Controller, Sensor, VolumeId
from proposed.environment import Environment
from torchstore.client import LocalClient
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request

__all__ = [
    "DirectoryCoverage",
    "DirectorySensor",
    "FetchPlan",
    "KeyCoverage",
    "LoadSensor",
    "Sensing",
    "SourceCoverage",
]

_Attached = TypeVar("_Attached", bound="Sensing")
_S = TypeVar("_S", bound=Sensor)
_Located = Mapping[str, Mapping[VolumeId, StorageInfo]]
_Region = Tuple[str, object]
_RequestSpec = Tuple[str, object, bool]
_StorageSpec = Tuple[ObjectType, Tuple[object, ...]]
_CoverageSpec = Tuple[Tuple[_RequestSpec, ...], Tuple[object, ...]]


class _NoInplace:
    supports_inplace_resharding = False


_PLANNER: LocalClient = LocalClient.__new__(LocalClient)
_NO_INPLACE = _NoInplace()


@dataclass(frozen=True)
class SourceCoverage:
    """One source's independently serviceable regions for a key."""

    source: VolumeId
    regions: Tuple[_Region, ...]


@dataclass(frozen=True)
class KeyCoverage:
    """Directory order and independently serviceable sources for one key."""

    key: str
    holders: Tuple[VolumeId, ...]
    sources: Tuple[SourceCoverage, ...]


@dataclass(frozen=True)
class DirectoryCoverage:
    """Expanded source coverage for one ordered request batch."""

    entries: Tuple[KeyCoverage, ...]

    @property
    def sources(self) -> set[VolumeId]:
        return {
            source.source
            for entry in self.entries
            for source in entry.sources
            if source.regions
        }

    def requirements(self) -> Dict[VolumeId, Counter[_Region]]:
        required: Dict[VolumeId, Counter[_Region]] = defaultdict(Counter)
        for entry in self.entries:
            for source in entry.sources:
                if source.regions:
                    required[source.source].update(source.regions)
        return dict(required)

    def missing_sources(
        self, required: Mapping[VolumeId, Counter[_Region]]
    ) -> set[VolumeId]:
        actual = self.requirements()
        return {
            source
            for source, expected in required.items()
            if expected - actual.get(source, Counter())
        }


@dataclass(frozen=True)
class FetchPlan:
    """Per-key source projection and exact source requirements."""

    by_key: Mapping[str, Tuple[VolumeId, ...]]
    required: Mapping[VolumeId, Counter[_Region]]

    @property
    def sources(self) -> Tuple[VolumeId, ...]:
        return tuple(
            dict.fromkeys(
                source for sources in self.by_key.values() for source in sources
            )
        )


class Sensing:
    """Common attachment for objects that declare the sensor types they read."""

    sensors: Tuple[type, ...] = ()
    environment: Optional[Environment] = None

    def attach(
        self: _Attached,
        environment: Environment,
        sensors: Optional[Mapping[type, Sensor]] = None,
    ) -> _Attached:
        """Bind stable run facts and resolve the declared sensor types."""
        available = sensors or {}
        for registered, sensor in available.items():
            if type(sensor) is not registered:
                raise TypeError(
                    f"{type(sensor).__name__} must be registered by its concrete "
                    f"type, not {registered.__name__}"
                )
        resolved: Dict[type, Sensor] = {}
        for required in self.sensors:
            matches = [
                sensor for sensor in available.values() if isinstance(sensor, required)
            ]
            noun = getattr(required, "__name__", str(required))
            if len(matches) != 1:
                if not matches:
                    raise RuntimeError(f"this object requires one {noun} sensor")
                raise RuntimeError(f"this object received multiple {noun} sensors")
            resolved[required] = matches[0]
        self.environment = environment
        self._sensed = resolved
        return self

    def sensor(self, sensor_type: type[_S]) -> _S:
        """Return one declared sensor by its domain type."""
        try:
            sensor = self._sensed[sensor_type]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"{type(self).__name__} did not declare {sensor_type.__name__}"
            ) from exc
        return sensor  # type: ignore[return-value]

    @property
    def env(self) -> Environment:
        """The attached environment, or raise before lifecycle wiring."""
        if self.environment is None:
            raise RuntimeError(f"{type(self).__name__} is not attached")
        return self.environment


class DirectorySensor(Sensor):
    """A coherent directory read and TorchStore's metadata-only fetch planning."""

    def __init__(self, directory: Controller) -> None:
        self.directory = directory
        self._keys: Optional[FrozenSet[str]] = None
        self._located: Dict[str, Dict[str, Any]] = {}
        self._decision_coverages: Dict[object, DirectoryCoverage] = {}
        self._live_coverage: Optional[Tuple[_CoverageSpec, DirectoryCoverage]] = None
        self._expansions: Dict[
            Tuple[_RequestSpec, _StorageSpec], Tuple[_Region, ...]
        ] = {}

    def locate(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """``key -> {volume_id -> StorageInfo}``, pinned when inside a decision."""
        if self._keys is None:
            return self.locate_live(keys)
        assert all(key in self._keys for key in keys), (
            "a pinned directory answers only for the keys in that decision"
        )
        return {key: self._located[key] for key in keys if key in self._located}

    def locate_live(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Read the raw directory now, with no source preference applied."""
        if not keys:
            return {}
        return self.directory.locate_raw(list(keys), missing_ok=True)

    def holders(
        self, keys: Sequence[str], *, live: bool = False
    ) -> Dict[str, List[VolumeId]]:
        """``key -> volume ids`` from the pinned answer, or a live read when asked."""
        located = self.locate_live(keys) if live else self.locate(keys)
        return {key: list(located.get(key, {})) for key in keys}

    def plan_requests(
        self,
        requests: Sequence[Request],
        located: Optional[_Located] = None,
    ) -> Dict[VolumeId, List[Request]]:
        """Expand requests by volume using TorchStore's own slice planner."""
        if located is None:
            located = self.locate([request.key for request in requests])
        maps = {request.key: dict(located.get(request.key, {})) for request in requests}
        volumes = {source for by_source in maps.values() for source in by_source}
        transports = dict.fromkeys(volumes, _NO_INPLACE)
        planned, _whole = _PLANNER._build_volume_requests(
            list(requests), maps, transports
        )
        return planned

    def requests_by_source(
        self,
        requests: Sequence[Request],
        located: Optional[_Located] = None,
    ) -> Dict[VolumeId, List[Request]]:
        """Return every source's independently serviceable request regions."""
        if located is None:
            located = self.locate([request.key for request in requests])
        planned: Dict[VolumeId, List[Request]] = defaultdict(list)
        for request in requests:
            for source, info in located.get(request.key, {}).items():
                planned[source].extend(self._expand_request(request, info))
        return {source: parts for source, parts in planned.items() if parts}

    def coverage(
        self,
        requests: Sequence[Request],
        located: Optional[_Located] = None,
    ) -> DirectoryCoverage:
        """Expand independent source coverage in one pass over directory entries."""
        keys = tuple(dict.fromkeys(request.key for request in requests))
        live = located is None
        if located is None:
            located = self.locate(keys)
        spec = self._coverage_spec(requests, located, keys)
        if self._keys is not None:
            cached = self._decision_coverages.get(spec)
            if cached is not None:
                return cached
            if live and self._live_coverage is not None:
                cached_spec, cached = self._live_coverage
                if spec == cached_spec:
                    self._decision_coverages[spec] = cached
                    return cached
                self._expansions = {}
        by_key: Dict[str, Dict[VolumeId, List[_Region]]] = {
            key: {source: [] for source in located.get(key, {})} for key in keys
        }
        for request in requests:
            for source, info in located.get(request.key, {}).items():
                by_key[request.key][source].extend(self.expand_regions(request, info))
        coverage = DirectoryCoverage(
            tuple(
                KeyCoverage(
                    key,
                    tuple(located.get(key, {})),
                    tuple(
                        SourceCoverage(source, tuple(regions))
                        for source, regions in by_key[key].items()
                    ),
                )
                for key in keys
            )
        )
        if self._keys is not None:
            self._decision_coverages[spec] = coverage
            if live:
                self._live_coverage = spec, coverage
        return coverage

    def decision_coverage(
        self, spec: object, build: Callable[[], DirectoryCoverage]
    ) -> DirectoryCoverage:
        """Reuse one derived coverage value within the current pin."""
        if self._keys is None:
            return build()
        cached = self._decision_coverages.get(spec)
        if cached is None:
            cached = build()
            self._decision_coverages[spec] = cached
        return cached

    def plan_coverage(
        self,
        combined: DirectoryCoverage,
        live: DirectoryCoverage,
        order: Sequence[VolumeId],
        *,
        requester: Optional[VolumeId] = None,
    ) -> FetchPlan:
        """Project ordered sources from combined coverage with live fallback."""
        allowed = set(order)
        live_by_key = {entry.key: entry for entry in live.entries}
        by_key: Dict[str, Tuple[VolumeId, ...]] = {}
        required: Dict[VolumeId, Counter[_Region]] = defaultdict(Counter)
        for entry in combined.entries:
            live_entry = live_by_key[entry.key]
            holders = set(live_entry.holders)
            live_regions = {
                source.source: source.regions for source in live_entry.sources
            }
            present = {
                source.source
                for source in entry.sources
                if source.source in holders
                or (source.source in allowed and source.source != requester)
            }
            effective = {
                source.source: (
                    source.regions
                    if source.source in allowed and source.source != requester
                    else live_regions.get(source.source, ())
                )
                for source in entry.sources
            }
            ranked = [
                (source, effective[source]) for source in order if source in present
            ]
            choices = ranked or [
                (source.source, effective[source.source])
                for source in entry.sources
                if source.source in present
            ]
            seen: set[_Region] = set()
            selected: List[VolumeId] = []
            for source, parts in choices:
                regions = Counter(parts)
                novel = [region for region in regions if region not in seen]
                if not novel:
                    continue
                selected.append(source)
                seen.update(novel)
                required[source].update(regions)
            by_key[entry.key] = tuple(selected)
        return FetchPlan(by_key, dict(required))

    def plan_fetch(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: Optional[VolumeId] = None,
    ) -> FetchPlan:
        """Plan an ordered fetch from the current live directory."""
        coverage = self.coverage(requests)
        return self.plan_coverage(coverage, coverage, order, requester=requester)

    def _expand_request(
        self, request: Request, storage_info: StorageInfo
    ) -> List[Request]:
        if storage_info.object_type == ObjectType.OBJECT:
            return [Request(key=request.key, is_object=True)]
        if storage_info.object_type == ObjectType.TENSOR:
            return [request]
        return _PLANNER._expand_tensor_slices(request, storage_info, False)

    def expand_regions(
        self, request: Request, storage_info: StorageInfo
    ) -> Tuple[_Region, ...]:
        """Exact regions one metadata entry can serve for ``request``."""
        spec = self.request_spec(request), self._storage_spec(storage_info)
        cached = self._expansions.get(spec)
        if cached is not None:
            return cached
        regions = tuple(
            self._region(part) for part in self._expand_request(request, storage_info)
        )
        self._expansions[spec] = regions
        return regions

    @staticmethod
    def request_spec(request: Request) -> _RequestSpec:
        """Planning-relevant request content, independent of object identity."""
        tensor_slice = request.tensor_slice
        slice_spec: object = None
        if tensor_slice is not None:
            slice_spec = DirectorySensor._slice_spec(tensor_slice)
        return request.key, slice_spec, request.is_object

    @staticmethod
    def _slice_spec(tensor_slice: Any) -> object:
        if tensor_slice is None:
            return None
        return (
            tuple(tensor_slice.offsets),
            tuple(tensor_slice.coordinates),
            tuple(tensor_slice.global_shape),
            tuple(tensor_slice.local_shape),
            tuple(tensor_slice.mesh_shape),
        )

    @classmethod
    def _storage_spec(cls, storage_info: StorageInfo) -> _StorageSpec:
        return (
            storage_info.object_type,
            tuple(cls._slice_spec(part) for part in storage_info.tensor_slices),
        )

    @classmethod
    def _coverage_spec(
        cls,
        requests: Sequence[Request],
        located: _Located,
        keys: Sequence[str],
    ) -> _CoverageSpec:
        request_spec = tuple(cls.request_spec(request) for request in requests)
        located_spec = tuple(
            (
                key,
                tuple(
                    (source, cls._storage_spec(info))
                    for source, info in located.get(key, {}).items()
                ),
            )
            for key in keys
        )
        return request_spec, located_spec

    @staticmethod
    def _region(request: Request) -> _Region:
        tensor_slice = request.tensor_slice
        region: object = None
        if tensor_slice is not None:
            region = (
                tuple(tensor_slice.offsets),
                tuple(tensor_slice.local_shape),
                tuple(tensor_slice.global_shape),
            )
        return request.key, region

    @staticmethod
    def regions(requests: Sequence[Request]) -> Counter[_Region]:
        """Count the exact key regions represented by expanded requests."""
        return Counter(DirectorySensor._region(request) for request in requests)

    def covers(
        self,
        requests: Sequence[Request],
        required: Mapping[VolumeId, Counter[_Region]],
        located: Optional[_Located] = None,
        *,
        live: bool = False,
    ) -> bool:
        """Whether directory metadata covers every required source region."""
        if not required:
            return True
        if located is None:
            keys = [request.key for request in requests]
            located = self.locate_live(keys) if live else self.locate(keys)
        sources = set(required)
        relevant = {
            key: {source: info for source, info in entries.items() if source in sources}
            for key, entries in located.items()
        }
        return not self.coverage(requests, relevant).missing_sources(required)

    @contextmanager
    def pinned(self, keys: Sequence[str]) -> Iterator[None]:
        """Serve one copied directory answer for the duration of a decision."""
        assert self._keys is None, "a decision already holds the directory read"
        located = {
            key: dict(volumes) for key, volumes in self.locate_live(keys).items()
        }
        self._keys, self._located = frozenset(keys), located
        self._decision_coverages = {}
        try:
            yield
        finally:
            self._keys, self._located, self._decision_coverages = None, {}, {}


class LoadSensor(Sensor):
    """Per-volume application load used to break otherwise equal rankings."""

    def named(self) -> Mapping[VolumeId, int]:
        """``volume -> current application load``; absent means zero."""
        raise NotImplementedError
