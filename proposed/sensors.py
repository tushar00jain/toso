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
_V = TypeVar("_V")
_Located = Mapping[str, Mapping[VolumeId, StorageInfo]]
_Region = Tuple[str, object]
_RequestSpec = Tuple[str, object, bool]
_StorageSpec = Tuple[ObjectType, Tuple[object, ...]]


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
    """Independently serviceable sources for one key, in directory order.

    Every source the directory lists appears, including one whose slices serve none
    of the request: ``regions`` empty is "listed here, serves nothing of this".
    """

    key: str
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
        # Derived values keyed by request content. One store, one invalidation rule:
        # :meth:`pinned` clears it when the directory it was derived from has moved.
        self._derived: Dict[object, Any] = {}
        self._stamp: object = None
        self._checked = False
        self._expansions: Dict[
            Tuple[_RequestSpec, _StorageSpec], Tuple[_Region, ...]
        ] = {}
        self._storage_specs: Optional[
            Dict[int, Tuple[StorageInfo, _StorageSpec]]
        ] = None

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
        if located is not None:
            return self._expand_coverage(requests, located, keys)
        pinned = self.locate(keys)
        return self.derived(
            ("coverage", self.batch_spec(requests)),
            lambda: self._expand_coverage(requests, pinned, keys),
        )

    def derived(self, spec: object, build: Callable[[], _V]) -> _V:
        """Reuse one value derived from the pinned directory and ``spec``.

        Only inside a pin: outside one nothing bounds how stale a value may be, and
        :meth:`pinned` is where the directory is checked for movement.
        """
        if self._keys is None:
            return build()
        if not self._checked:
            # Once per pin, and only for a decision that derives anything: a plane
            # that reads holders and no coverage never signs the directory.
            self._checked = True
            stamp = self._directory_stamp()
            if stamp != self._stamp:
                self._derived.clear()
                self._stamp = stamp
        cached = self._derived.get(spec)
        if cached is None:
            cached = build()
            self._derived[spec] = cached
        return cached

    def _expand_coverage(
        self,
        requests: Sequence[Request],
        located: _Located,
        keys: Sequence[str],
    ) -> DirectoryCoverage:
        by_key: Dict[str, Dict[VolumeId, List[_Region]]] = {
            key: {source: [] for source in located.get(key, {})} for key in keys
        }
        for request in requests:
            for source, info in located.get(request.key, {}).items():
                by_key[request.key][source].extend(self.expand_regions(request, info))
        return DirectoryCoverage(
            tuple(
                KeyCoverage(
                    key,
                    tuple(
                        SourceCoverage(source, tuple(regions))
                        for source, regions in by_key[key].items()
                    ),
                )
                for key in keys
            )
        )

    def requirements(
        self, requests: Sequence[Request]
    ) -> Mapping[VolumeId, Counter[_Region]]:
        """``source -> the regions it live-serves`` for ``requests``.

        Answers for keys the pin does not hold, reading those live: a peer's promised
        batch is not the batch this decision asked about.
        """
        return self.derived(
            ("requirements", self.batch_spec(requests)),
            lambda: self._requirements(requests),
        )

    def _requirements(
        self, requests: Sequence[Request]
    ) -> Dict[VolumeId, Counter[_Region]]:
        keys = tuple(dict.fromkeys(request.key for request in requests))
        outside = [key for key in keys if self._keys is None or key not in self._keys]
        if not outside:
            return self.coverage(requests).requirements()
        located = {key: self._located[key] for key in keys if key in self._located}
        located.update(self.locate_live(outside))
        return self.coverage(requests, located).requirements()

    def project(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: Optional[VolumeId] = None,
    ) -> Tuple[Dict[str, Tuple[VolumeId, ...]], Dict[VolumeId, Counter[_Region]]]:
        """Per key, the ranked sources that add regions, and what each must provide.

        Walks ``order`` per key and stops once the key's whole requested value is
        covered (:meth:`whole_regions`), which is after the first serving source
        wherever one source holds a whole value. A key split across sources has no
        such target and is walked to the end of ``order``.

        A key no ranked source serves falls back to its live holders in directory
        order, which is what keeps an unroutable ask answerable.
        """
        live = self.coverage(requests)
        promised = self.promised(tuple(entry.key for entry in live.entries))
        by_request: Dict[str, List[Request]] = defaultdict(list)
        for request in requests:
            by_request[request.key].append(request)
        by_key: Dict[str, Tuple[VolumeId, ...]] = {}
        required: Dict[VolumeId, Counter[_Region]] = defaultdict(Counter)
        for entry in live.entries:
            key_requests = by_request[entry.key]
            live_regions = {source.source: source.regions for source in entry.sources}
            target: set[_Region] = set()
            for request in key_requests:
                target.update(self.whole_regions(request))
            seen: set[_Region] = set()
            selected: List[VolumeId] = []
            offered = False
            for source in order:
                if source == requester:
                    # Its own copy, never its pending promise to itself.
                    regions = live_regions.get(source)
                else:
                    regions = self.servable(
                        entry.key, source, key_requests, live_regions, promised
                    )
                if regions is None:
                    continue
                offered = True
                if self._take(source, regions, seen, selected, required):
                    if target <= seen:
                        break
            if not offered:
                for source, regions in live_regions.items():
                    if self._take(source, regions, seen, selected, required):
                        if target <= seen:
                            break
            by_key[entry.key] = tuple(selected)
        # Sources ranked for a key they do not serve leave no requirement behind.
        return by_key, {
            source: regions for source, regions in required.items() if regions
        }

    @staticmethod
    def _take(
        source: VolumeId,
        parts: Tuple[_Region, ...],
        seen: set[_Region],
        selected: List[VolumeId],
        required: Dict[VolumeId, Counter[_Region]],
    ) -> bool:
        """Select ``source`` if it adds a region nothing ahead of it serves."""
        regions = Counter(parts)
        if all(region in seen for region in regions):
            return False
        selected.append(source)
        seen.update(regions)
        required[source].update(regions)
        return True

    def promised(self, keys: Sequence[str]) -> _Located:
        """Entries volumes have promised for ``keys`` and not yet published.

        Empty here: an ordinary directory read answers with holders, and a sensor
        that promises nothing has nothing more to offer :meth:`servable`. Read once
        per :meth:`project`, so an override pays one directory read per plan.
        """
        return {}

    def servable(
        self,
        key: str,
        source: VolumeId,
        requests: Sequence[Request],
        live_regions: Mapping[VolumeId, Tuple[_Region, ...]],
        promised: _Located,
    ) -> Optional[Tuple[_Region, ...]]:
        """Regions ``source`` can serve of ``requests`` on ``key``.

        ``None`` when the directory does not list ``source`` under ``key`` at all,
        which is different from listing it with nothing of this request to serve.
        """
        return live_regions.get(source)

    def plan_fetch(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: Optional[VolumeId] = None,
    ) -> FetchPlan:
        """Plan an ordered fetch from the current live directory."""
        by_key, required = self.project(requests, order, requester=requester)
        return FetchPlan(by_key, required)

    @classmethod
    def whole_regions(cls, request: Request) -> Tuple[_Region, ...]:
        """The regions ``request`` denotes when one source holds all of it.

        The one region the request itself names: expanding it against a source
        holding exactly it gives that back, for an object, a tensor and a slice
        alike. A key's object type is one per key -- TorchStore's
        ``StorageInfo.update`` refuses to change it -- so where a source holds whole
        values this is what every source of the key serves, and the first of them
        ends the walk in :meth:`project`. Where sources hold pieces, no source serves
        it and the walk runs to the end of the ranking.
        """
        return (cls._region(request),)

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

    def _storage_spec(self, storage_info: StorageInfo) -> _StorageSpec:
        memo = self._storage_specs
        if memo is None:
            return self._build_storage_spec(storage_info)
        # Keyed by identity, so the entry holds the object: an id freed mid-pin could
        # otherwise be handed to a different StorageInfo. Directory metadata does not
        # move while a decision holds the pin.
        cached = memo.get(id(storage_info))
        if cached is None:
            cached = storage_info, self._build_storage_spec(storage_info)
            memo[id(storage_info)] = cached
        return cached[1]

    @classmethod
    def _build_storage_spec(cls, storage_info: StorageInfo) -> _StorageSpec:
        return (
            storage_info.object_type,
            tuple(cls._slice_spec(part) for part in storage_info.tensor_slices),
        )

    def batch_spec(self, requests: Sequence[Request]) -> Tuple[_RequestSpec, ...]:
        """What a batch asks for, independent of the request objects carrying it."""
        return tuple(self.request_spec(request) for request in requests)

    def _directory_stamp(self) -> object:
        """What must not have moved for a derived value to still hold.

        ``O(1)`` where the directory counts its own mutations. A directory that does
        not -- a stub, a fixture -- is signed instead, at ``O(keys x holders)`` once
        per pin.
        """
        revision = getattr(self.directory, "revision", None)
        if revision is not None:
            return "revision", revision
        # Holder order is directory order, and a fallback plan reads it, so a
        # reordering is a move.
        return tuple(
            (
                key,
                tuple(
                    (source, self._storage_spec(info))
                    for source, info in entries.items()
                ),
            )
            for key, entries in self._located.items()
        )

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
        actual: Mapping[VolumeId, Counter[_Region]]
        if located is not None:
            actual = self.coverage(requests, located).requirements()
        elif live:
            keys = tuple(dict.fromkeys(request.key for request in requests))
            actual = self.coverage(requests, self.locate_live(keys)).requirements()
        else:
            actual = self.requirements(requests)
        empty: Counter[_Region] = Counter()
        return not any(
            expected - actual.get(source, empty)
            for source, expected in required.items()
        )

    @contextmanager
    def pinned(self, keys: Sequence[str]) -> Iterator[None]:
        """Serve one copied directory answer for the duration of a decision.

        Nothing suspends between the read and the release, so no mutation can land
        inside a pin: values derived here are checked against the directory once, on
        the way in, and held unchecked for the rest of the decision.
        """
        assert self._keys is None, "a decision already holds the directory read"
        located = {
            key: dict(volumes) for key, volumes in self.locate_live(keys).items()
        }
        self._keys, self._located = frozenset(keys), located
        self._storage_specs = {}
        self._checked = False
        try:
            yield
        finally:
            self._keys, self._located = None, {}
            self._storage_specs = None


class LoadSensor(Sensor):
    """Per-volume application load used to break otherwise equal rankings."""

    def named(self) -> Mapping[VolumeId, int]:
        """``volume -> current application load``; absent means zero."""
        raise NotImplementedError
