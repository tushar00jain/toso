"""Domain observations shared by selectors."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
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
from proposed.planner import plan, region
from torchstore.controller import StorageInfo
from torchstore.transport import Request

__all__ = [
    "DirectorySensor",
    "FetchPlan",
    "LoadSensor",
    "Sensing",
]

_Attached = TypeVar("_Attached", bound="Sensing")
_S = TypeVar("_S", bound=Sensor)
_V = TypeVar("_V")
_Located = Mapping[str, Mapping[VolumeId, StorageInfo]]
_Region = Tuple[str, object]
_RequestSpec = Tuple[str, object, bool]
_EMPTY: Mapping[VolumeId, StorageInfo] = {}


class _Ranked(Mapping):
    """One key's directory entries, iterated best first by a shared ranking.

    Lazily, and that is the point: the planner takes the first entry and stops where
    a key is stored whole, so a key its top-ranked source holds costs one membership
    test rather than a walk of the ranking. Materializing a ranked dict per key would
    be ``O(K x V)`` for every decision.

    A source in ``owners`` is planned against what it *promised*; anything else
    against what it holds. A key no ranked source lists falls back to the directory's
    own order, which is what keeps an unroutable ask answerable.
    """

    __slots__ = ("_live", "_promised", "_owners", "_order")

    def __init__(
        self,
        live: Mapping[VolumeId, StorageInfo],
        promised: Mapping[VolumeId, StorageInfo],
        owners: AbstractSet[VolumeId],
        order: Sequence[VolumeId],
    ) -> None:
        self._live = live
        self._promised = promised
        self._owners = owners
        self._order = order

    def __getitem__(self, source: VolumeId) -> StorageInfo:
        if source in self._owners:
            info = self._promised.get(source)
            if info is not None:
                return info
        return self._live[source]

    def __contains__(self, source: object) -> bool:
        return source in self._live or (
            source in self._owners and source in self._promised
        )

    def __iter__(self) -> Iterator[VolumeId]:
        listed = False
        for source in self._order:
            if source in self:
                listed = True
                yield source
        if not listed:
            yield from self._live

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _Alone(Mapping):
    """One source's entries, each key's map holding only that source.

    A located map for asking what one source could serve by itself. The one-entry
    map per key is built on the lookup and not retained, so ``K x T`` entries cost
    ``T`` dicts rather than ``K x T``.
    """

    __slots__ = ("_source", "_entries")

    def __init__(
        self, source: VolumeId, entries: Mapping[str, StorageInfo]
    ) -> None:
        self._source = source
        self._entries = entries

    def __getitem__(self, key: str) -> Mapping[VolumeId, StorageInfo]:
        return {self._source: self._entries[key]}

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


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
        """``volume -> the sub-requests it is asked for``, as the store would plan it.

        A dry run of the same :class:`~proposed.planner.GreedyClient` method the
        fetch itself runs, so a plan and the fetch it plans cannot disagree about
        which volume serves which region of one located map.
        """
        if located is None:
            located = self.locate([request.key for request in requests])
        return plan(requests, located)

    def requests_by_source(
        self,
        requests: Sequence[Request],
        located: Optional[_Located] = None,
        sources: Optional[Collection[VolumeId]] = None,
    ) -> Dict[VolumeId, List[Request]]:
        """Every source's *independently* serviceable request regions.

        One plan per source over a map holding only that source, so no source is
        dropped for adding nothing over another. That is the question a route's
        recorded requirement asks (:meth:`covers`): can *this* source still serve
        what it was asked for, whoever else could.
        """
        if located is None:
            located = self.locate([request.key for request in requests])
        # Walked entry by entry, not source by key: sparse placement costs the
        # entries that exist rather than every (key, source) pair that could.
        by_source: Dict[VolumeId, Dict[str, StorageInfo]] = defaultdict(dict)
        for key, entries in located.items():
            for source, info in entries.items():
                if sources is None or source in sources:
                    by_source[source][key] = info
        planned: Dict[VolumeId, List[Request]] = {}
        for source, entries in by_source.items():
            parts = plan(requests, _Alone(source, entries)).get(source)
            if parts:
                planned[source] = parts
        return planned

    def ranked(
        self,
        keys: Sequence[str],
        order: Sequence[VolumeId],
        *,
        requester: Optional[VolumeId] = None,
    ) -> Dict[str, Mapping[VolumeId, StorageInfo]]:
        """The located map for ``keys``, each key's entries in ranking order.

        Ranking order is how a preference reaches the planner: it takes the first
        volume listed per key for a whole value, and the first offering each region
        for a sliced one.
        """
        live = self.locate(keys)
        promised = self.promised(keys)
        owners = self.owners() - {requester}
        return {
            key: _Ranked(
                live.get(key, _EMPTY), promised.get(key, _EMPTY), owners, order
            )
            for key in keys
        }

    def owners(self) -> AbstractSet[VolumeId]:
        """Volumes whose *promised* entries this sensor may plan against.

        Empty here, and the membership test rather than the promise itself: the
        directory is shared, so a plane must not plan onto a promise it did not make
        and will never hear the publication of.
        """
        return frozenset()

    def derived(self, spec: object, build: Callable[[], _V]) -> _V:
        """Reuse one value derived from the pinned directory and ``spec``.

        Only inside a pin: outside one nothing bounds how stale a value may be, and
        :meth:`pinned` is where the directory is checked for movement.
        """
        if self._keys is None:
            return build()
        if not self._checked:
            # Once per pin, and only for a decision that derives anything: a plane
            # that reads holders and nothing else never signs the directory.
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

    def requirements(
        self, requests: Sequence[Request]
    ) -> Mapping[VolumeId, Counter[_Region]]:
        """``source -> the regions it live-serves`` for ``requests``.

        Independent per source, so this is what a source *can* serve rather than
        what a plan asked it for. Answers for keys the pin does not hold by reading
        those live: a peer's promised batch is not the batch this decision asked
        about.
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
            return self.regions_by_source(requests, self.locate(keys))
        located = {key: self._located[key] for key in keys if key in self._located}
        located.update(self.locate_live(outside))
        return self.regions_by_source(requests, located)

    def regions_by_source(
        self,
        requests: Sequence[Request],
        located: _Located,
        sources: Optional[Collection[VolumeId]] = None,
    ) -> Dict[VolumeId, Counter[_Region]]:
        """:meth:`requests_by_source` counted by region rather than by sub-request."""
        return {
            source: Counter(region(part) for part in parts)
            for source, parts in self.requests_by_source(
                requests, located, sources
            ).items()
        }

    def _planned(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        requester: Optional[VolumeId],
    ) -> Tuple[Dict[str, Tuple[VolumeId, ...]], Dict[VolumeId, Counter[_Region]]]:
        """Per key the sources the planner chose, in ranking order, and what each owes.

        Taken under the pin, while the real fetch runs later against a fresh
        directory read that this plan is applied to as a *preference*: a source that
        evicted in between drops out of the preference and the read still answers
        from whoever holds the key.
        """
        keys = tuple(dict.fromkeys(request.key for request in requests))
        maps = self.ranked(keys, order, requester=requester)
        required: Dict[VolumeId, Counter[_Region]] = {}
        chosen: Dict[str, set[VolumeId]] = defaultdict(set)
        for source, parts in self.plan_requests(requests, maps).items():
            counts: Counter[_Region] = Counter()
            for part in parts:
                counts[region(part)] += 1
                chosen[part.key].add(source)
            required[source] = counts
        return {key: _in_order(maps[key], chosen[key]) for key in keys}, required

    def promised(self, keys: Sequence[str]) -> _Located:
        """Entries volumes have promised for ``keys`` and not yet published.

        Empty here: an ordinary directory read answers with holders, and a sensor
        that promises nothing has nothing more to offer :meth:`ranked`. Read once per
        plan, so an override pays one directory read per decision.
        """
        return {}

    def plan_fetch(
        self,
        requests: Sequence[Request],
        order: Sequence[VolumeId],
        *,
        requester: Optional[VolumeId] = None,
    ) -> FetchPlan:
        """Plan an ordered fetch from the current live directory."""
        return FetchPlan(*self._planned(requests, order, requester))

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
        # Keyed by identity and holding the object, so an id freed mid-walk cannot be
        # handed to a second StorageInfo. One entry object shared by every slot -- a
        # fixture's usual shape -- is then signed once.
        signed: Dict[int, Tuple[StorageInfo, object]] = {}

        def sign(info: StorageInfo) -> object:
            entry = signed.get(id(info))
            if entry is None:
                entry = info, (info.object_type, frozenset(info.tensor_slices))
                signed[id(info)] = entry
            return entry[1]

        # Holder order is directory order, and a fallback plan reads it, so a
        # reordering is a move.
        return tuple(
            (key, tuple((source, sign(info)) for source, info in entries.items()))
            for key, entries in self._located.items()
        )

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
            actual = self.regions_by_source(requests, located, set(required))
        elif live:
            keys = tuple(dict.fromkeys(request.key for request in requests))
            actual = self.regions_by_source(
                requests, self.locate_live(keys), set(required)
            )
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
        self._checked = False
        try:
            yield
        finally:
            self._keys, self._located = None, {}


def _in_order(
    entries: Mapping[VolumeId, StorageInfo], chosen: AbstractSet[VolumeId]
) -> Tuple[VolumeId, ...]:
    """``chosen`` in ``entries`` order, stopping once all of them are placed.

    One source is the common answer and needs no walk at all, so a whole-value key
    costs nothing here however long the ranking is.
    """
    if len(chosen) <= 1:
        return tuple(chosen)
    ordered = []
    for source in entries:
        if source in chosen:
            ordered.append(source)
            if len(ordered) == len(chosen):
                break
    return tuple(ordered)


class LoadSensor(Sensor):
    """Per-volume application load used to break otherwise equal rankings."""

    def named(self) -> Mapping[VolumeId, int]:
        """``volume -> current application load``; absent means zero."""
        raise NotImplementedError
