"""Pure cost model: locality tiers and transfer time.

No measurement. A transfer's duration is a deterministic function of the
source/destination locality and the byte count. Constants are illustrative
(units arbitrary but consistent); the only property that matters is that
cross-node (trainer->generator) is clearly slower than intra-node
(generator<->generator), so the trace shows why peer exchange saves fabric.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Tuple

from .model import Volume


class Tier(IntEnum):
    """Locality tiers, ordered cheapest first (used as a preference key)."""

    SHM = 0     # same host (shared memory)
    NVLINK = 1  # same node, different host
    RDMA = 2    # cross node


# tier -> (latency, bandwidth) in arbitrary-but-consistent units.
TIERS: Dict[Tier, Tuple[float, float]] = {
    Tier.SHM: (0.001, 1500.0),
    Tier.NVLINK: (0.002, 600.0),
    Tier.RDMA: (0.010, 100.0),
}

TIER_LABEL: Dict[Tier, str] = {
    Tier.SHM: "shm",
    Tier.NVLINK: "nvlink",
    Tier.RDMA: "cross-node",
}


def locality(src: Volume, dst: Volume) -> Tier:
    """Return the locality tier between two volumes."""
    if src.host == dst.host:
        return Tier.SHM
    if src.node == dst.node:
        return Tier.NVLINK
    return Tier.RDMA


def transfer_time(src: Volume, dst: Volume, nbytes: int) -> float:
    """Return the simulated time to move ``nbytes`` from ``src`` to ``dst``.

    A same-volume "transfer" is free (the data is already local).
    """
    if src.id == dst.id:
        return 0.0
    lat, bw = TIERS[locality(src, dst)]
    return lat + nbytes / bw
