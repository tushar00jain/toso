"""Build deterministic local routes from multidimensional TensorSlice metadata."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

from torchstore.transport.types import TensorSlice

from ._model import (
    LocalRouteTable,
    RankLayout,
    RankRole,
    RelaySignal,
    RouteEntry,
    SliceGeometry,
    Transfer,
    TransferKind,
    slice_geometry,
)

__all__ = ["build_local_routes"]


def _validate(layouts: Sequence[RankLayout]) -> Dict[str, RankLayout]:
    by_rank: Dict[str, RankLayout] = {}
    key_shapes: Dict[str, Tuple[int, ...]] = {}
    key_sizes: Dict[str, int] = {}
    for layout in layouts:
        if not layout.rank or layout.rank in by_rank:
            raise ValueError(f"invalid or duplicate rank metadata: {layout.rank!r}")
        by_rank[layout.rank] = layout
        if set(layout.slices) != set(layout.element_sizes):
            raise ValueError(
                f"rank {layout.rank!r} must provide one element size per key"
            )
        for key, slices in layout.slices.items():
            if not slices:
                raise ValueError(f"rank {layout.rank!r} has no slices for {key!r}")
            element_size = int(layout.element_sizes[key])
            if element_size <= 0:
                raise ValueError(f"element size for {key!r} must be positive")
            for tensor_slice in slices:
                geometry = slice_geometry(tensor_slice)
                previous_shape = key_shapes.setdefault(key, geometry.global_shape)
                if geometry.global_shape != previous_shape:
                    raise ValueError(f"inconsistent global shape for key {key!r}")
                previous_size = key_sizes.setdefault(key, element_size)
                if element_size != previous_size:
                    raise ValueError(f"inconsistent element size for key {key!r}")
    if not any(x.role == RankRole.TRAINER for x in layouts):
        raise ValueError("at least one trainer layout is required")
    if not any(x.role == RankRole.GENERATOR for x in layouts):
        raise ValueError("at least one generator layout is required")
    return by_rank


def _cells(
    target: SliceGeometry,
    sources: Sequence[Tuple[str, SliceGeometry]],
) -> Iterable[SliceGeometry]:
    boundaries = [{start, end} for start, end in zip(target.offsets, target.ends)]
    for _rank, source in sources:
        if not source.overlaps(target):
            continue
        for axis, (start, end) in enumerate(zip(source.offsets, source.ends)):
            boundaries[axis].add(max(target.offsets[axis], start))
            boundaries[axis].add(min(target.ends[axis], end))

    intervals = []
    for axis in boundaries:
        ordered = sorted(axis)
        intervals.append(tuple(zip(ordered, ordered[1:])))
    for cell in product(*intervals):
        offsets = tuple(start for start, _end in cell)
        shape = tuple(end - start for start, end in cell)
        if all(shape):
            yield SliceGeometry(target.global_shape, offsets, shape)


def _source_segments(
    key: str,
    target: SliceGeometry,
    trainer_slices: Mapping[str, Tuple[TensorSlice, ...]],
    element_size: int,
    source_load: DefaultDict[str, int],
) -> List[Tuple[str, SliceGeometry, int]]:
    sources = []
    for rank, slices in trainer_slices.items():
        for tensor_slice in slices:
            geometry = slice_geometry(tensor_slice)
            if geometry.overlaps(target):
                sources.append((rank, geometry))

    selected: List[Tuple[str, SliceGeometry, int]] = []
    for cell in _cells(target, sources):
        candidates = sorted({rank for rank, source in sources if source.covers(cell)})
        if not candidates:
            raise ValueError(
                f"trainer metadata does not cover {key!r} at "
                f"offsets={cell.offsets}, shape={cell.local_shape}"
            )
        nbytes = cell.numel * element_size
        source = min(candidates, key=lambda rank: (source_load[rank], rank))
        source_load[source] += nbytes
        selected.append((source, cell, nbytes))
    _validate_exact_coverage(key, target, [segment for _rank, segment, _n in selected])
    return selected


def _validate_exact_coverage(
    key: str, target: SliceGeometry, segments: Sequence[SliceGeometry]
) -> None:
    if any(not target.covers(segment) for segment in segments):
        raise ValueError(f"route for {key!r} writes outside its destination slice")
    if sum(segment.numel for segment in segments) != target.numel:
        raise ValueError(f"route for {key!r} does not exactly cover its destination")
    for index, first in enumerate(segments):
        if any(first.overlaps(second) for second in segments[index + 1 :]):
            raise ValueError(f"route for {key!r} writes overlapping destination bytes")


def build_local_routes(
    layouts: Sequence[RankLayout],
    *,
    relay_replicas: bool = True,
) -> Dict[str, LocalRouteTable]:
    """Build per-rank routes from real TorchStore slice metadata.

    Every rank can run this pure setup operation and retain only its own table.
    Replication is inferred from identical byte geometry, independent of mesh
    coordinates. No planner is contacted during an update.
    """
    by_rank = _validate(layouts)
    trainers = {
        rank: layout
        for rank, layout in by_rank.items()
        if layout.role == RankRole.TRAINER
    }
    generators = {
        rank: layout
        for rank, layout in by_rank.items()
        if layout.role == RankRole.GENERATOR
    }
    element_sizes = {
        key: size
        for layout in by_rank.values()
        for key, size in layout.element_sizes.items()
    }

    grouped: DefaultDict[Tuple[str, SliceGeometry], List[Tuple[str, TensorSlice]]] = (
        defaultdict(list)
    )
    for rank, layout in generators.items():
        for key, slices in layout.slices.items():
            for tensor_slice in slices:
                grouped[(key, slice_geometry(tensor_slice))].append((rank, tensor_slice))

    sends: DefaultDict[str, DefaultDict[str, List[Transfer]]] = defaultdict(
        lambda: defaultdict(list)
    )
    receives: DefaultDict[str, DefaultDict[str, List[Transfer]]] = defaultdict(
        lambda: defaultdict(list)
    )
    broadcasts: DefaultDict[str, DefaultDict[str, List[RelaySignal]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_load: DefaultDict[str, int] = defaultdict(int)
    relay_load: DefaultDict[str, int] = defaultdict(int)

    for (key, target), placements in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        members = sorted(placements, key=lambda item: item[0])
        groups = [members] if relay_replicas else [[member] for member in members]
        trainer_slices = {
            rank: tuple(layout.slices.get(key, ()))
            for rank, layout in trainers.items()
        }
        for group in groups:
            ingress_rank, ingress_slice = min(
                group, key=lambda item: (relay_load[item[0]], item[0])
            )
            target_bytes = target.numel * element_sizes[key]
            relay_load[ingress_rank] += target_bytes

            for source, segment, nbytes in _source_segments(
                key, target, trainer_slices, element_sizes[key], source_load
            ):
                transfer = Transfer(
                    key=key,
                    source=source,
                    destination=ingress_rank,
                    segment=segment.to_tensor_slice(),
                    destination_slice=ingress_slice,
                    nbytes=nbytes,
                    kind=TransferKind.TRAINER,
                )
                sends[source][key].append(transfer)
                receives[ingress_rank][key].append(transfer)

            peers = tuple(rank for rank, _slice in group if rank != ingress_rank)
            if not peers:
                continue
            broadcasts[ingress_rank][key].append(
                RelaySignal(key=key, tensor_slice=ingress_slice, peers=peers)
            )
            for peer, peer_slice in group:
                if peer == ingress_rank:
                    continue
                transfer = Transfer(
                    key=key,
                    source=ingress_rank,
                    destination=peer,
                    segment=target.to_tensor_slice(),
                    destination_slice=peer_slice,
                    nbytes=target_bytes,
                    kind=TransferKind.RELAY,
                )
                sends[ingress_rank][key].append(transfer)
                receives[peer][key].append(transfer)

    tables: Dict[str, LocalRouteTable] = {}
    for rank, layout in by_rank.items():
        keys = set(layout.slices) | set(sends[rank]) | set(receives[rank])
        entries = {
            key: RouteEntry(
                sends=tuple(sends[rank][key]),
                receives=tuple(receives[rank][key]),
                broadcasts=tuple(broadcasts[rank][key]),
            )
            for key in sorted(keys)
        }
        tables[rank] = LocalRouteTable(
            rank=rank,
            role=layout.role,
            published=layout.slices if layout.role == RankRole.TRAINER else {},
            requested=layout.slices if layout.role == RankRole.GENERATOR else {},
            entries=entries,
        )
    return tables
