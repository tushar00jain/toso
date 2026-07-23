"""Pure cost model: locality tiers and transfer time.

No measurement. A transfer's duration is a deterministic function of the
source/destination locality and the byte count. Constants are illustrative
(units arbitrary but consistent); the only property that matters is that
cross-node (trainer->generator) is clearly slower than intra-node
(generator<->generator), so the trace shows why peer exchange saves fabric.
"""

from __future__ import annotations

from typing import Dict, Tuple

from sim_common import topology
from sim_common.topology import locality, Tier, TIER_LABEL  # noqa: F401

from .model import Volume


# tier -> (latency, bandwidth) in arbitrary-but-consistent units.
TIERS: Dict[Tier, Tuple[float, float]] = {
    Tier.SHM: (0.001, 1500.0),
    Tier.NVLINK: (0.002, 600.0),
    Tier.RDMA: (0.010, 100.0),
}


def transfer_time(src: Volume, dst: Volume, nbytes: int) -> float:
    """Return the simulated time to move ``nbytes`` from ``src`` to ``dst``.

    A same-volume "transfer" is free (the data is already local). Delegates to
    the shared skeleton with this sim's own :data:`TIERS` constants.
    """
    return topology.transfer_time(src, dst, nbytes, TIERS)
