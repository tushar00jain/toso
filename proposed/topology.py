"""Where a volume is: the locality types the upstream ask needs.

Part of the proposal, not the simulator (gap 4). A controller cannot rank sources
for a requester without knowing where its volumes sit relative to each other, and
torchstore has no such notion today -- ``StorageInfo`` carries no topology.

``src`` and ``dst`` are duck-typed throughout: they only need ``.id`` (identity),
``.host`` (shared-memory domain) and ``.node`` (intra-node domain).

The *cost* of a tier is emphatically not here -- latency and bandwidth per tier are
simulation constants and live in :mod:`sim_common.topology`, which imports these
types. This module imports nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Sequence

__all__ = ["Endpoint", "Tier", "TIER_LABEL", "locality", "nearest"]

@dataclass(frozen=True)
class Endpoint:
    """A transfer endpoint: the minimal locality identity :func:`transfer_time` needs.

    Only three attributes matter to the cost model: ``id`` (identity -- a
    same-``id`` transfer is free), ``host`` (shared-memory domain) and ``node``
    (intra-node / NVLink domain). This is the shared shape that both the
    ``dedup_sim`` ``Volume`` and ``realsim``'s endpoints reduce to; keep new
    endpoint/locality types reusing this one instead of re-declaring the trio.
    """

    id: str
    host: str
    node: str


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


def nearest(
    topology: Dict[str, "Endpoint"], candidates: Sequence[str], to: str
) -> Optional[str]:
    """The closest of ``candidates`` to ``to``: locality first, id as the tie-break.

    Over the topology map rather than off a :class:`~proposed.view.View`, because
    distance is arithmetic on endpoints and needs no directory read. ``None`` for no
    candidates. The id tie-break is what makes the answer total, hence reproducible.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda v: (int(locality(topology[v], topology[to])), v))
