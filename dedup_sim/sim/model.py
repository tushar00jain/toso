"""Domain model: volumes, 1-D regions and atomic-region splitting.

Tensors are kept 1-D for clarity (the running example ``W``). A key has a
global length ``N``; everything is a half-open range ``[start, end)`` over
``[0, N)`` and ``bytes = (end - start) * dtype_bytes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

Region = Tuple[int, int]  # [start, end) over a 1-D flattened tensor


@dataclass(frozen=True)
class Volume:
    """A storage volume (one per rank under ``LocalRankStrategy``).

    Trainer volumes live on the trainer node; each generator rank owns one
    volume on the generator side and reuses it as its own read-through cache.
    """

    id: str
    host: str  # shared-memory domain
    node: str  # NVLink / intra-node domain
    is_trainer: bool = False


def region_bytes(region: Region, dtype_bytes: int) -> int:
    """Return the byte size of ``region`` for the given element width."""
    start, end = region
    return (end - start) * dtype_bytes


def region_str(region: Region) -> str:
    """Render a region as ``[start,end)`` for the trace."""
    start, end = region
    return f"[{start},{end})"


def _covered(seg: Region, regions: Iterable[Region]) -> bool:
    """True if ``seg`` lies entirely inside at least one region."""
    a, b = seg
    return any(s <= a and b <= e for (s, e) in regions)


def split_regions(regions: Iterable[Region]) -> List[Region]:
    """Split a set of (possibly overlapping) regions into atomic segments.

    Collect every breakpoint (all starts and ends), form the consecutive
    segments between adjacent breakpoints, and keep those covered by at least
    one input region. The result is the minimal set of non-overlapping
    segments whose union equals the union of the inputs -- mirroring the
    get-side intersection the store performs. The coordinator then plans per
    atomic region.
    """
    regions = list(regions)
    points = sorted({p for r in regions for p in r})
    segs: List[Region] = []
    for a, b in zip(points, points[1:]):
        if a < b and _covered((a, b), regions):
            segs.append((a, b))
    return segs


def decompose(need: Iterable[Region], atomics: Iterable[Region]) -> List[Region]:
    """Return the atomic regions that make up a reader's ``need``.

    A reader's need is expressed as arbitrary ranges; assembly is the union of
    the atomic regions covering those ranges. The caller can assert this union
    reconstructs the need exactly.
    """
    need = list(need)
    out = [seg for seg in atomics if _covered(seg, need)]
    return sorted(out)


def union_bytes(needs: Iterable[Iterable[Region]], atomics: Iterable[Region],
                dtype_bytes: int) -> int:
    """Total bytes of the *union* of all readers' needs (each region once)."""
    atomics = list(atomics)
    union: Set[Region] = set()
    for need in needs:
        union.update(decompose(need, atomics))
    return sum(region_bytes(r, dtype_bytes) for r in union)
