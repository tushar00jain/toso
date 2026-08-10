"""Cost-model skeleton: what a locality tier costs.

The locality *types* (:class:`~proposed.topology.Endpoint`, :class:`Tier`,
:func:`locality`) are part of the upstream ask and live in
:mod:`proposed.topology`; they are re-exported here so existing callers keep one
import site. What belongs to the simulator is the formula below: a transfer's
duration as a deterministic function of tier and byte count. Latency/bandwidth
constants are not baked in -- the caller supplies them via ``tiers``.
"""

from __future__ import annotations

from typing import Dict, Tuple

from proposed.topology import Endpoint, locality, Tier, TIER_LABEL

__all__ = ["Endpoint", "Tier", "TIER_LABEL", "locality", "transfer_time"]


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
