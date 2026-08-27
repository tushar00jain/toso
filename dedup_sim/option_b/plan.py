"""Build, inspect, serialize, and distribute an Option B route plan."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from torchstore.transport.types import TensorSlice

from ._model import (
    LocalRouteTable,
    RankLayout,
    RankRole,
    RelaySignal,
    RouteEntry,
    Transfer,
    TransferKind,
)
from ._routing import build_local_routes

__all__ = ["OptionBPlan"]


_Slices = Mapping[str, Mapping[str, Sequence[TensorSlice]]]


class _OptionBPlanBase(ABC):
    """The public API for building and distributing immutable route plans.

    Build one global plan from published and requested ``TensorSlice`` metadata,
    distribute :meth:`for_rank` results, and optionally persist them with
    :meth:`save` and :meth:`load`. Runtime clients use :meth:`lookup`.
    """

    @classmethod
    @abstractmethod
    def build(
        cls,
        publishers: _Slices,
        requesters: _Slices,
        element_sizes: Mapping[str, int],
        *,
        relay_replicas: bool = True,
    ) -> Self:
        """Build a complete plan from publisher and requester geometries."""
        ...

    @property
    @abstractmethod
    def ranks(self) -> tuple[str, ...]:
        """Ranks with a local route table in this plan."""
        ...

    @abstractmethod
    def for_rank(self, rank: str) -> Self:
        """Return a distributable plan containing only ``rank``."""
        ...

    @abstractmethod
    def lookup(self, rank: str, key: str) -> RouteEntry:
        """Return the immutable actions for ``rank`` and ``key``."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this plan."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Restore a plan produced by :meth:`to_dict`."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Write this complete or rank-local plan as deterministic JSON."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Load a plan previously written by :meth:`save`."""
        ...


def _slice_to_dict(tensor_slice: TensorSlice) -> dict[str, list[int]]:
    return {
        "offsets": list(tensor_slice.offsets),
        "coordinates": list(tensor_slice.coordinates),
        "global_shape": list(tensor_slice.global_shape),
        "local_shape": list(tensor_slice.local_shape),
        "mesh_shape": list(tensor_slice.mesh_shape),
    }


def _slice_from_dict(value: Mapping[str, Any]) -> TensorSlice:
    return TensorSlice(
        offsets=tuple(value["offsets"]),
        coordinates=tuple(value["coordinates"]),
        global_shape=tuple(value["global_shape"]),
        local_shape=tuple(value["local_shape"]),
        mesh_shape=tuple(value["mesh_shape"]),
    )


def _slices_to_dict(
    entries: Mapping[str, Sequence[TensorSlice]],
) -> dict[str, list[dict[str, list[int]]]]:
    return {
        key: [_slice_to_dict(tensor_slice) for tensor_slice in slices]
        for key, slices in sorted(entries.items())
    }


def _slices_from_dict(
    entries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[TensorSlice, ...]]:
    return {
        key: tuple(_slice_from_dict(tensor_slice) for tensor_slice in slices)
        for key, slices in entries.items()
    }


def _transfer_to_dict(transfer: Transfer) -> dict[str, Any]:
    return {
        "key": transfer.key,
        "source": transfer.source,
        "destination": transfer.destination,
        "segment": _slice_to_dict(transfer.segment),
        "destination_slice": _slice_to_dict(transfer.destination_slice),
        "nbytes": transfer.nbytes,
        "kind": transfer.kind.value,
    }


def _transfer_from_dict(value: Mapping[str, Any]) -> Transfer:
    return Transfer(
        key=value["key"],
        source=value["source"],
        destination=value["destination"],
        segment=_slice_from_dict(value["segment"]),
        destination_slice=_slice_from_dict(value["destination_slice"]),
        nbytes=int(value["nbytes"]),
        kind=TransferKind(value["kind"]),
    )


class OptionBPlan(_OptionBPlanBase):
    """Immutable per-rank routes produced once from global slice metadata."""

    VERSION = 1

    def __init__(self, routes: Mapping[str, LocalRouteTable]) -> None:
        self._routes = dict(routes)

    @classmethod
    def build(
        cls,
        publishers: _Slices,
        requesters: _Slices,
        element_sizes: Mapping[str, int],
        *,
        relay_replicas: bool = True,
    ) -> "OptionBPlan":
        """Compute the complete plan from publisher and requester geometries."""
        overlap = set(publishers) & set(requesters)
        if overlap:
            raise ValueError(
                f"ranks cannot publish and request in one plan: {sorted(overlap)}"
            )

        def layouts(entries: _Slices, role: RankRole) -> list[RankLayout]:
            result = []
            for rank, keys in entries.items():
                missing = set(keys) - set(element_sizes)
                if missing:
                    raise KeyError(f"missing element sizes for {sorted(missing)}")
                result.append(
                    RankLayout(
                        rank=rank,
                        role=role,
                        slices={key: tuple(slices) for key, slices in keys.items()},
                        element_sizes={key: element_sizes[key] for key in keys},
                    )
                )
            return result

        rank_layouts = layouts(publishers, RankRole.TRAINER)
        rank_layouts += layouts(requesters, RankRole.GENERATOR)
        return cls(
            build_local_routes(rank_layouts, relay_replicas=relay_replicas)
        )

    @property
    def ranks(self) -> tuple[str, ...]:
        """Ranks with a local table in this plan."""
        return tuple(sorted(self._routes))

    def for_rank(self, rank: str) -> "OptionBPlan":
        """Return a distributable plan containing only ``rank``'s local table."""
        return OptionBPlan({rank: self._routes[rank]})

    def lookup(self, rank: str, key: str) -> RouteEntry:
        """Return the immutable local actions for ``rank`` and ``key``."""
        return self._routes[rank].lookup(key)

    def _local(self, rank: str) -> LocalRouteTable:
        return self._routes[rank]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this plan."""
        ranks = {}
        for rank, table in sorted(self._routes.items()):
            entries = {}
            for key, entry in sorted(table.entries.items()):
                entries[key] = {
                    "sends": [_transfer_to_dict(value) for value in entry.sends],
                    "receives": [
                        _transfer_to_dict(value) for value in entry.receives
                    ],
                    "broadcasts": [
                        {
                            "key": value.key,
                            "tensor_slice": _slice_to_dict(value.tensor_slice),
                            "peers": list(value.peers),
                        }
                        for value in entry.broadcasts
                    ],
                }
            ranks[rank] = {
                "role": table.role.value,
                "published": _slices_to_dict(table.published),
                "requested": _slices_to_dict(table.requested),
                "entries": entries,
            }
        return {"version": self.VERSION, "ranks": ranks}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptionBPlan":
        """Restore a plan produced by :meth:`to_dict`."""
        if value.get("version") != cls.VERSION:
            raise ValueError(f"unsupported Option B plan version {value.get('version')}")
        routes = {}
        for rank, table in value["ranks"].items():
            entries = {}
            for key, entry in table["entries"].items():
                entries[key] = RouteEntry(
                    sends=tuple(_transfer_from_dict(item) for item in entry["sends"]),
                    receives=tuple(
                        _transfer_from_dict(item) for item in entry["receives"]
                    ),
                    broadcasts=tuple(
                        RelaySignal(
                            key=item["key"],
                            tensor_slice=_slice_from_dict(item["tensor_slice"]),
                            peers=tuple(item["peers"]),
                        )
                        for item in entry["broadcasts"]
                    ),
                )
            routes[rank] = LocalRouteTable(
                rank=rank,
                role=RankRole(table["role"]),
                published=_slices_from_dict(table["published"]),
                requested=_slices_from_dict(table["requested"]),
                entries=entries,
            )
        return cls(routes)

    def save(self, path: str | Path) -> None:
        """Write this complete or rank-local plan as deterministic JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> "OptionBPlan":
        """Load a plan previously written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))
