"""Domain observations shared by selectors."""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple, TypeVar,
)

from proposed.deployment import Controller, Sensor, VolumeId
from proposed.environment import Environment

__all__ = ["DirectorySensor", "LoadSensor", "Sensing"]

_Attached = TypeVar("_Attached", bound="Sensing")
_S = TypeVar("_S", bound=Sensor)


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
                sensor for sensor in available.values()
                if isinstance(sensor, required)
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
    """The live directory, with one coherent read per decision when pinned."""

    def __init__(self, directory: Controller) -> None:
        self.directory = directory
        self._keys: Optional[FrozenSet[str]] = None
        self._located: Dict[str, Dict[str, Any]] = {}

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

    @contextmanager
    def pinned(self, keys: Sequence[str]) -> Iterator[None]:
        """Serve one copied directory answer for the duration of a decision."""
        assert self._keys is None, "a decision already holds the directory read"
        located = {
            key: dict(volumes) for key, volumes in self.locate_live(keys).items()
        }
        self._keys, self._located = frozenset(keys), located
        try:
            yield
        finally:
            self._keys, self._located = None, {}


class LoadSensor(Sensor):
    """Per-volume application load used to break otherwise equal rankings."""

    def named(self) -> Mapping[VolumeId, int]:
        """``volume -> current application load``; absent means zero."""
        raise NotImplementedError
