"""Cost-model skeleton: locality tiers and transfer time.

No measurement. A transfer's duration is a deterministic function of the
source/destination locality and the byte count. This module owns the *shape*
of the cost model -- the :class:`Tier` ordering, the :data:`TIER_LABEL`
rendering, the :func:`locality` classification, and the :func:`transfer_time`
formula. Latency/bandwidth constants are not baked in: the caller supplies them
via the ``tiers`` argument.

``src`` and ``dst`` are duck-typed: they only need ``.id`` (identity),
``.host`` (shared-memory domain) and ``.node`` (intra-node domain) attributes.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Tuple


class Tier(IntEnum):
    """Locality tiers, ordered cheapest first (used as a preference key)."""

    SHM = 0     # same host (shared memory)
    NVLINK = 1  # same node, different host
    RDMA = 2    # cross node


TIER_LABEL: Dict[Tier, str] = {
    Tier.SHM: "shm",
    Tier.NVLINK: "nvlink",
    Tier.RDMA: "cross-node",
}


def locality(src, dst) -> Tier:
    """Return the locality tier between two endpoints (duck-typed on host/node)."""
    if src.host == dst.host:
        return Tier.SHM
    if src.node == dst.node:
        return Tier.NVLINK
    return Tier.RDMA


def transfer_time(src, dst, nbytes: int, tiers: Dict[Tier, Tuple[float, float]]) -> float:
    """Return the simulated time to move ``nbytes`` from ``src`` to ``dst``.

    ``tiers`` maps each :class:`Tier` to its ``(latency, bandwidth)`` pair; the
    caller supplies the per-sim constants. A same-endpoint or zero-byte
    "transfer" is free.
    """
    if src.id == dst.id or nbytes == 0:
        return 0.0
    lat, bw = tiers[locality(src, dst)]
    return lat + nbytes / bw
