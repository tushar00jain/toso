"""Internal metadata and local actions for precomputed tensor routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Mapping, Tuple

from torchstore.transport.types import TensorSlice

__all__ = [
    "LocalRouteTable",
    "RankLayout",
    "RankRole",
    "RelaySignal",
    "RouteEntry",
    "SliceGeometry",
    "Transfer",
    "TransferKind",
    "slice_geometry",
]


@dataclass(frozen=True, order=True)
class SliceGeometry:
    """The axis-aligned geometry of a TorchStore ``TensorSlice``.

    Coordinates and mesh shape identify placement, not the bytes in a slice.
    They are excluded so DP replicas with different coordinates compare as the
    same requested geometry.
    """

    global_shape: Tuple[int, ...]
    offsets: Tuple[int, ...]
    local_shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        ndim = len(self.global_shape)
        if not ndim or len(self.offsets) != ndim or len(self.local_shape) != ndim:
            raise ValueError("TensorSlice shapes and offsets must have equal rank")
        if any(size <= 0 for size in self.global_shape):
            raise ValueError("global dimensions must be positive")
        if any(size <= 0 for size in self.local_shape):
            raise ValueError("local dimensions must be positive")
        if any(offset < 0 for offset in self.offsets):
            raise ValueError("slice offsets must be non-negative")
        if any(end > size for end, size in zip(self.ends, self.global_shape)):
            raise ValueError("TensorSlice extends beyond its global shape")

    @property
    def ends(self) -> Tuple[int, ...]:
        return tuple(
            offset + size for offset, size in zip(self.offsets, self.local_shape)
        )

    @property
    def numel(self) -> int:
        return reduce(mul, self.local_shape, 1)

    def covers(self, other: "SliceGeometry") -> bool:
        return self.global_shape == other.global_shape and all(
            start <= other_start and end >= other_end
            for start, end, other_start, other_end in zip(
                self.offsets, self.ends, other.offsets, other.ends
            )
        )

    def overlaps(self, other: "SliceGeometry") -> bool:
        return self.global_shape == other.global_shape and all(
            start < other_end and other_start < end
            for start, end, other_start, other_end in zip(
                self.offsets, self.ends, other.offsets, other.ends
            )
        )

    def to_tensor_slice(self) -> TensorSlice:
        """Materialize this geometry as real TorchStore metadata."""
        return TensorSlice(
            offsets=self.offsets,
            coordinates=(),
            global_shape=self.global_shape,
            local_shape=self.local_shape,
            mesh_shape=(),
        )


def slice_geometry(tensor_slice: TensorSlice) -> SliceGeometry:
    """Normalize a real TorchStore ``TensorSlice`` to its byte geometry."""
    return SliceGeometry(
        global_shape=tuple(int(x) for x in tensor_slice.global_shape),
        offsets=tuple(int(x) for x in tensor_slice.offsets),
        local_shape=tuple(int(x) for x in tensor_slice.local_shape),
    )


class RankRole(str, Enum):
    TRAINER = "trainer"
    GENERATOR = "generator"


@dataclass(frozen=True)
class RankLayout:
    """The real TorchStore slices one rank publishes or requests."""

    rank: str
    role: RankRole
    slices: Mapping[str, Tuple[TensorSlice, ...]]
    element_sizes: Mapping[str, int]


class TransferKind(str, Enum):
    TRAINER = "trainer-to-generator"
    RELAY = "generator-read-through"


@dataclass(frozen=True)
class Transfer:
    """One fixed source-to-destination transfer in global tensor coordinates."""

    key: str
    source: str
    destination: str
    segment: TensorSlice
    destination_slice: TensorSlice
    nbytes: int
    kind: TransferKind


@dataclass(frozen=True)
class RelaySignal:
    """A readiness-only broadcast after a generator stores a complete slice."""

    key: str
    tensor_slice: TensorSlice
    peers: Tuple[str, ...]


@dataclass(frozen=True)
class RouteEntry:
    """The actions returned by one rank's local key lookup."""

    sends: Tuple[Transfer, ...] = ()
    receives: Tuple[Transfer, ...] = ()
    broadcasts: Tuple[RelaySignal, ...] = ()


@dataclass(frozen=True)
class LocalRouteTable:
    """One rank's immutable route table, installed when layouts are exchanged."""

    rank: str
    role: RankRole
    published: Mapping[str, Tuple[TensorSlice, ...]]
    requested: Mapping[str, Tuple[TensorSlice, ...]]
    entries: Mapping[str, RouteEntry] = field(default_factory=dict)

    def lookup(self, key: str) -> RouteEntry:
        if key not in self.published and key not in self.requested:
            raise KeyError(f"rank {self.rank!r} has no metadata for key {key!r}")
        return self.entries.get(key, RouteEntry())
