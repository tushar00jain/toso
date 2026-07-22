"""Reusable §5 aggregator-response builders over a live TorchStore.

The live socket server (``live_example``) turns the store's introspection
endpoints (controller ``keys`` / ``locate_volumes``; volume ``get_meta``; and
``ts.get`` for peek stats) into the exact §5 JSON the Rust ``toso-tui`` expects.
That logic lives here as plain async functions returning plain ``dict``s — no
wire framing — so the server serializes each to one line from a single source of
truth.

Every builder takes live controller/strategy handles plus the request params
and reads the store at call time (a point-in-time view, SPEC §9): counts can be
approximate under concurrent writes, and a key that races out from under a read
is skipped rather than crashing the response.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from math import prod
from typing import Any

import torch
import torchstore as ts
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport.types import Request

logger: logging.Logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
HEAD_ELEMENTS: int = 8
DEFAULT_LIMIT: int = 200

# The §5 object-type tags the Rust model deserializes (SCREAMING_SNAKE_CASE).
_OBJECT_TYPE_TAG: dict[ObjectType, str] = {
    ObjectType.OBJECT: "OBJECT",
    ObjectType.TENSOR: "TENSOR",
    ObjectType.TENSOR_SLICE: "TENSOR_SLICE",
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _KeyRecord:
    """Everything a §5 response needs about one stored key, gathered once."""

    __slots__ = (
        "key",
        "object_type",
        "dtype",
        "global_shape",
        "mesh_shape",
        "fully_committed",
        "shards",
        "bytes",
        "volume_ids",
    )

    def __init__(
        self,
        *,
        key: str,
        object_type: ObjectType,
        dtype: str | None,
        global_shape: list[int] | None,
        mesh_shape: list[int] | None,
        fully_committed: bool,
        shards: list[dict[str, Any]],
        nbytes: float,
        volume_ids: list[str],
    ) -> None:
        self.key = key
        self.object_type = object_type
        self.dtype = dtype
        self.global_shape = global_shape
        self.mesh_shape = mesh_shape
        self.fully_committed = fully_committed
        self.shards = shards
        self.bytes = nbytes
        self.volume_ids = volume_ids

    @property
    def type_tag(self) -> str:
        return _OBJECT_TYPE_TAG[self.object_type]

    def key_entry(self) -> dict[str, Any]:
        """The §5 ``KeyEntry`` JSON for this key."""
        return {
            "key": self.key,
            "object_type": self.type_tag,
            "dtype": self.dtype,
            "global_shape": self.global_shape,
            "fully_committed": self.fully_committed,
            "mesh_shape": self.mesh_shape,
            "shards": self.shards,
        }


class StoreState:
    """A point-in-time gather of every key + volume, shared by the list ops."""

    __slots__ = ("records", "volumes", "groups")

    def __init__(
        self,
        records: list[_KeyRecord],
        volumes: list[dict[str, Any]],
        groups: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.records = records
        self.volumes = volumes
        self.groups = groups


def _is_fully_committed(
    object_type: ObjectType, volume_map: dict[str, StorageInfo]
) -> bool:
    """Whether every mesh coordinate of a sharded DTensor is present.

    Mirrors ``Controller._is_dtensor_fully_committed``: non-slice keys are
    trivially committed; a slice key needs all ``product(*mesh_shape)`` coords.
    """
    if object_type != ObjectType.TENSOR_SLICE:
        return True

    coords: set[tuple[int, ...]] = set()
    mesh_shape: tuple[int, ...] | None = None
    for storage_info in volume_map.values():
        for tensor_slice in storage_info.tensor_slices:
            if tensor_slice is None:
                continue
            coords.add(tensor_slice.coordinates)
            mesh_shape = tensor_slice.mesh_shape

    if mesh_shape is None:
        return True
    expected = set(product(*(range(dim) for dim in mesh_shape)))
    return coords == expected


def _pick_meta_request(
    key: str, object_type: ObjectType, storage_info: StorageInfo
) -> Request:
    """A ``get_meta`` request shaped for this key's storage form."""
    if object_type == ObjectType.TENSOR_SLICE:
        a_slice = next((s for s in storage_info.tensor_slices if s is not None), None)
        if a_slice is not None:
            return Request.from_tensor_slice(key, a_slice)
    # Plain tensor and object both resolve from the stored value alone.
    return Request.from_any(key, None)


def _shards_for_key(
    object_type: ObjectType,
    volume_map: dict[str, StorageInfo],
    local_shape: list[int] | None,
) -> list[dict[str, Any]]:
    """Build the per-key shard rows the Detail view renders."""
    if object_type == ObjectType.OBJECT:
        return []

    if object_type == ObjectType.TENSOR:
        # A plain tensor lives whole on each volume that holds it.
        return [
            {
                "volume_id": volume_id,
                "coordinates": [],
                "offsets": [],
                "local_shape": local_shape or [],
            }
            for volume_id in volume_map
        ]

    shards: list[dict[str, Any]] = []
    for volume_id, storage_info in volume_map.items():
        for tensor_slice in storage_info.tensor_slices:
            if tensor_slice is None:
                continue
            shards.append(
                {
                    "volume_id": volume_id,
                    "coordinates": list(tensor_slice.coordinates),
                    "offsets": list(tensor_slice.offsets),
                    "local_shape": list(tensor_slice.local_shape),
                }
            )
    return shards


async def _key_meta(
    strategy: Any,
    key: str,
    object_type: ObjectType,
    volume_map: dict[str, StorageInfo],
) -> tuple[list[int] | None, str | None, float]:
    """Fetch (shape, dtype, bytes) via the volume ``get_meta`` endpoint.

    Returns metadata only — never the tensor data. ``shape`` and ``dtype`` are
    ``None`` for objects; ``bytes`` is 0 for objects.
    """
    volume_id = next(iter(volume_map))
    storage_info = volume_map[volume_id]
    volume_ref = strategy.get_storage_volume(volume_id)
    request = _pick_meta_request(key, object_type, storage_info)

    meta = await volume_ref.volume.get_meta.call_one([request])
    entry = meta[0]
    if entry == "obj":
        return None, None, 0.0

    size, dtype = entry
    shape = [int(dim) for dim in size]
    nbytes = float(prod(shape) * _dtype_itemsize(dtype)) if shape else 0.0
    return shape, _dtype_name(dtype), nbytes


async def _peek_stats(key: str) -> dict[str, Any] | None:
    """Compute §5.3 tensor stats near the data via ``ts.get`` (the data path)."""
    try:
        value = await ts.get(key)
    except Exception:
        logger.warning("peek: could not fetch %s for stats", key, exc_info=True)
        return None

    if not isinstance(value, torch.Tensor):
        return None

    flat = value.detach().reshape(-1).to(torch.float64)
    head = [float(x) for x in flat[:HEAD_ELEMENTS].tolist()]
    return {
        "dtype": _dtype_name(value.dtype),
        "shape": [int(dim) for dim in value.shape],
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "l2_norm": float(torch.linalg.vector_norm(flat)),
        "head": head,
    }


async def _collect_keys(
    strategy: Any,
    all_keys: list[str],
    volume_maps: dict[str, dict[str, StorageInfo]],
) -> list[_KeyRecord]:
    records: list[_KeyRecord] = []
    for key in all_keys:
        volume_map = volume_maps.get(key)
        if not volume_map:
            logger.warning("key %s has no located volumes; skipping", key)
            continue

        try:
            object_type = next(iter(volume_map.values())).object_type
            fully_committed = _is_fully_committed(object_type, volume_map)
            shape, dtype, nbytes = await _key_meta(
                strategy, key, object_type, volume_map
            )
        except Exception:
            # A concurrent write/delete can race a key out from under the meta
            # fetch (SPEC §9); skip it rather than failing the whole response.
            logger.warning("key %s raced during read; skipping", key, exc_info=True)
            continue

        is_slice = object_type == ObjectType.TENSOR_SLICE
        records.append(
            _KeyRecord(
                key=key,
                object_type=object_type,
                dtype=dtype,
                # Per §5, global_shape is null for OBJECT and plain TENSOR.
                global_shape=shape if is_slice else None,
                mesh_shape=None,
                fully_committed=fully_committed,
                shards=_shards_for_key(object_type, volume_map, shape),
                nbytes=nbytes,
                volume_ids=list(volume_map),
            )
        )
    return records


def _prefix_entry(prefix: str, members: list[_KeyRecord]) -> dict[str, Any]:
    objects = sum(1 for m in members if m.object_type == ObjectType.OBJECT)
    tensors = sum(1 for m in members if m.object_type == ObjectType.TENSOR)
    dtensors = sum(1 for m in members if m.object_type == ObjectType.TENSOR_SLICE)
    partial = sum(
        1
        for m in members
        if m.object_type == ObjectType.TENSOR_SLICE and not m.fully_committed
    )
    # A node is a leaf iff it is itself a stored key — i.e. exactly one key lives
    # under it and that key IS this node (not a deeper descendant).
    is_leaf = len(members) == 1 and members[0].key == prefix
    return {
        "prefix": prefix,
        "keys": len(members),
        "objects": objects,
        "tensors": tensors,
        "dtensors": dtensors,
        "partial": partial,
        "bytes": sum(m.bytes for m in members),
        "is_leaf": is_leaf,
    }


def _children(records: list[_KeyRecord], parent: str) -> list[dict[str, Any]]:
    """The §5.2 ``expand_prefix`` children one trie level below ``parent``.

    ``parent`` is ``""`` for the top level of the trie.
    """
    parent_dot = f"{parent}." if parent else ""
    groups: dict[str, list[_KeyRecord]] = defaultdict(list)
    for record in records:
        if parent and not record.key.startswith(parent_dot):
            continue
        rest = record.key[len(parent_dot) :]
        segment = rest.split(".", 1)[0]
        groups[f"{parent_dot}{segment}"].append(record)

    return [_prefix_entry(child, members) for child, members in sorted(groups.items())]


def _build_volumes(
    strategy: Any, records: list[_KeyRecord]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Per-volume rows and their grouping by host (§5.1 / §5.2)."""
    per_volume: dict[str, list[_KeyRecord]] = defaultdict(list)
    for record in records:
        for volume_id in record.volume_ids:
            per_volume[volume_id].append(record)

    volumes: list[dict[str, Any]] = []
    for volume_id, hostname in strategy.volume_id_to_hostname.items():
        members = per_volume.get(volume_id, [])
        volumes.append(
            {
                "volume_id": volume_id,
                "hostname": hostname,
                # Transport is only logged, not exposed by an endpoint (§9).
                "transport": None,
                "num_keys": len(members),
                "bytes": sum(m.bytes for m in members),
                "reachable": True,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for volume in volumes:
        groups[f"host:{volume['hostname']}"].append(volume)
    return volumes, groups


def build_summary_dict(
    *,
    store_name: str,
    strategy_name: str,
    records: list[_KeyRecord],
    volumes: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    total_bytes = sum(r.bytes for r in records)
    partial = sum(
        1
        for r in records
        if r.object_type == ObjectType.TENSOR_SLICE and not r.fully_committed
    )
    committed = len(records) - partial

    histogram: list[list[int]] = []
    if committed:
        histogram.append([100, committed])
    if partial:
        histogram.append([0, partial])

    volume_groups = [
        {
            "group": group,
            "volumes": len(group_volumes),
            "keys": sum(v["num_keys"] for v in group_volumes),
            "bytes": sum(v["bytes"] for v in group_volumes),
            "transports": {},
            "reachable": len(group_volumes),
        }
        for group, group_volumes in sorted(groups.items())
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _iso_now(),
        "store_name": store_name,
        "strategy": strategy_name,
        "totals": {
            "volumes": len(volumes),
            "keys": len(records),
            "bytes": total_bytes,
            "partial_dtensors": partial,
        },
        "volume_groups": volume_groups,
        "key_prefixes": _children(records, ""),
        "histograms": {"shard_commit_pct": histogram},
    }


# ---------------------------------------------------------------------------
# Pagination + server-side sort (SPEC §5.2 / §5.4)
# ---------------------------------------------------------------------------


def _decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        return max(0, int(cursor))
    except (TypeError, ValueError):
        return 0


def _paginate(
    items: list[dict[str, Any]], limit: int | None, cursor: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    start = _decode_cursor(cursor)
    lim = int(limit) if limit is not None else DEFAULT_LIMIT
    window = items[start : start + lim]
    next_cursor = str(start + lim) if start + lim < len(items) else None
    return window, next_cursor


def _reverse(sort_by: str, order: str | None) -> bool:
    if order == "asc":
        return False
    if order == "desc":
        return True
    # Default is anomaly/size-first (desc), except name which reads asc.
    return sort_by != "name"


def _sort_prefixes(
    children: list[dict[str, Any]], sort_by: str | None, order: str | None
) -> list[dict[str, Any]]:
    key = sort_by or "partial"
    # KeyPrefix rows carry no bytes/reachable, so those keys fall back to `keys`.
    keyfns = {
        "partial": lambda c: c.get("partial") or 0,
        "keys": lambda c: c["keys"],
        "bytes": lambda c: c["keys"],
        "reachable": lambda c: c["keys"],
        "name": lambda c: c["prefix"],
    }
    keyfn = keyfns.get(key, keyfns["partial"])
    return sorted(children, key=keyfn, reverse=_reverse(key, order))


def _sort_volumes(
    volumes: list[dict[str, Any]], sort_by: str | None, order: str | None
) -> list[dict[str, Any]]:
    key = sort_by or "bytes"
    keyfns = {
        "bytes": lambda v: v["bytes"],
        "keys": lambda v: v["num_keys"],
        "reachable": lambda v: v["reachable"],
        "partial": lambda v: v["num_keys"],
        "name": lambda v: v["volume_id"],
    }
    keyfn = keyfns.get(key, keyfns["bytes"])
    return sorted(volumes, key=keyfn, reverse=_reverse(key, order))


def _sort_key_matches(
    matches: list[dict[str, Any]], sort_by: str | None, order: str | None
) -> list[dict[str, Any]]:
    key = sort_by or "partial"
    keyfns = {
        # Anomaly-first: partially-committed keys sort to the top.
        "partial": lambda m: not m["fully_committed"],
        "name": lambda m: m["key"],
        "keys": lambda m: m["key"],
        "bytes": lambda m: m["key"],
        "reachable": lambda m: m["fully_committed"],
    }
    keyfn = keyfns.get(key, keyfns["partial"])
    return sorted(matches, key=keyfn, reverse=_reverse(key, order))


# ---------------------------------------------------------------------------
# Live state gather + §5 response builders
# ---------------------------------------------------------------------------


async def gather_state(controller: Any, strategy: Any) -> StoreState:
    """Read every key + volume once for the list ops (point-in-time, §9)."""
    all_keys = sorted(await controller.keys.call_one(None))
    volume_maps: dict[
        str, dict[str, StorageInfo]
    ] = await controller.locate_volumes.call_one(
        all_keys, missing_ok=True, require_fully_committed=False
    )
    records = await _collect_keys(strategy, all_keys, volume_maps)
    volumes, groups = _build_volumes(strategy, records)
    return StoreState(records, volumes, groups)


async def build_summary(
    controller: Any, strategy: Any, *, store_name: str = "torchstore"
) -> dict[str, Any]:
    """The §5.1 ``summary`` response over the live store."""
    state = await gather_state(controller, strategy)
    return build_summary_dict(
        store_name=store_name,
        strategy_name=type(strategy).__name__,
        records=state.records,
        volumes=state.volumes,
        groups=state.groups,
    )


async def build_expand_prefix(
    controller: Any, strategy: Any, req: dict[str, Any]
) -> dict[str, Any]:
    """The §5.2 ``expand_prefix`` response: one trie level below ``prefix``."""
    prefix = req.get("prefix") or ""
    state = await gather_state(controller, strategy)
    children = _sort_prefixes(
        _children(state.records, prefix), req.get("sort_by"), req.get("order")
    )
    window, next_cursor = _paginate(children, req.get("limit"), req.get("cursor"))
    return {"children": window, "next_cursor": next_cursor}


async def build_list_volumes(
    controller: Any, strategy: Any, req: dict[str, Any]
) -> dict[str, Any]:
    """The §5.2 ``list_volumes`` response for one volume ``group``."""
    group = req.get("group") or ""
    state = await gather_state(controller, strategy)
    members = state.groups.get(group, [])
    volumes = _sort_volumes(members, req.get("sort_by"), req.get("order"))
    window, next_cursor = _paginate(volumes, req.get("limit"), req.get("cursor"))
    return {"volumes": window, "next_cursor": next_cursor}


async def build_key(
    controller: Any, strategy: Any, req: dict[str, Any]
) -> dict[str, Any]:
    """The §5.2 ``key`` response: full detail for a single key."""
    key = req.get("key")
    if not key:
        raise KeyError("key request missing `key`")
    volume_maps: dict[
        str, dict[str, StorageInfo]
    ] = await controller.locate_volumes.call_one(
        [key], missing_ok=True, require_fully_committed=False
    )
    records = await _collect_keys(strategy, [key], volume_maps)
    if not records:
        raise KeyError(f"key not found: {key}")
    return records[0].key_entry()


async def build_search(
    controller: Any, strategy: Any, req: dict[str, Any]
) -> dict[str, Any]:
    """The §5.2 ``search`` response: substring match across keys or volumes.

    An empty ``pattern`` is match-all (powers the ``:partial`` / jump paths).
    """
    kind = req.get("kind", "key")
    pattern = req.get("pattern") or ""
    state = await gather_state(controller, strategy)

    if kind == "volume":
        matches = [
            v
            for group_volumes in state.groups.values()
            for v in group_volumes
            if pattern in v["volume_id"] or pattern in v["hostname"]
        ]
        matches = _sort_volumes(matches, req.get("sort_by"), req.get("order"))
    else:
        matches = [r.key_entry() for r in state.records if pattern in r.key]
        matches = _sort_key_matches(matches, req.get("sort_by"), req.get("order"))

    window, next_cursor = _paginate(matches, req.get("limit"), req.get("cursor"))
    return {"matches": window, "next_cursor": next_cursor}


async def build_peek(req: dict[str, Any]) -> dict[str, Any]:
    """The §5.3 ``peek`` response: tensor stats computed near the data."""
    key = req.get("key")
    if not key:
        raise KeyError("peek request missing `key`")
    stats = await _peek_stats(key)
    if stats is None:
        raise KeyError(f"no tensor stats for key: {key}")
    return stats
