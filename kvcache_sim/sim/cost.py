"""Pure cost model: locality tiers, KV transfer time, prefill and decode time.

No measurement. Durations are deterministic functions of size and locality.
Constants are illustrative (units arbitrary but consistent); three properties are
what matter, and they encode the core premise:
  (a) recomputing a token costs time, so reusing a cached prefix is cheaper;
  (b) **transferring a cached KV block is cheaper than recomputing it** -- else
      remote reuse / hot-block replication would never pay off; and
  (c) cross-node KV transfer is slower than intra-node, so *where* a reusable
      prefix lives matters; and
  (d) a decode step's per-token time (TBT) rises with the decode batch size, so
      packing more concurrent requests onto an instance trades throughput for
      latency -- the tension the TBT SLO bounds.
The tier *shape* matches ``dedup_sim`` (SHM < NVLINK < RDMA); bandwidths are scaled
up here so per-token transfer is well under per-token prefill compute (b).
"""

from __future__ import annotations

from typing import Dict, Tuple

from sim_common import topology
from sim_common.topology import locality, Tier, TIER_LABEL  # noqa: F401

from .model import Instance


# tier -> (latency, bandwidth) in arbitrary-but-consistent units. Bandwidths are
# high enough that per-token KV transfer (bytes_per_token / bw) is cheaper than
# per-token prefill compute (PREFILL_PER_TOKEN) -- property (b) above.
TIERS: Dict[Tier, Tuple[float, float]] = {
    Tier.SHM: (0.001, 150000.0),
    Tier.NVLINK: (0.002, 60000.0),
    Tier.RDMA: (0.010, 10000.0),
}

# Prefill: fixed launch cost + per-uncached-token compute. This is the cost a
# cache hit avoids -- the whole point of prefix reuse.
PREFILL_LAT = 0.010
PREFILL_PER_TOKEN = 0.0008

# Decode: per-output-token occupancy on a decode instance (for load / TBT SLO).
DECODE_PER_TOKEN = 0.0006

# Batched decode / TBT (K6). A decode instance generates one token per *step* for
# every request in its batch; the step's duration is the time-between-tokens (TBT)
# every batched request observes for that token. TBT rises with the batch size --
# more concurrent requests => more KV attended per step => a longer step. This is
# the core tension the TBT SLO bounds (Mooncake §4.2, §5.2): larger batches raise
# MFU/throughput but push TBT up.
#
#   decode_step_time(b) = TBT_BASE + (b - 1) * TBT_BATCH_SLOPE
#
# ``TBT_BASE`` is the uninterrupted (batch=1) per-token time -- the baseline a TBT
# SLO is expressed as a multiple of. The slope is deliberately steep enough that a
# handful of batched requests approaches a typical 5x SLO, so the sweet spot is a
# small-but-nonzero batch (as in practice).
TBT_BASE = 0.020
TBT_BATCH_SLOPE = 0.015


def transfer_time(src: Instance, dst: Instance, nbytes: int) -> float:
    """Simulated time to move ``nbytes`` of KV from ``src`` to ``dst`` (0 if same).

    Delegates to the shared skeleton with this sim's own :data:`TIERS` constants.
    """
    return topology.transfer_time(src, dst, nbytes, TIERS)


def prefill_time(uncached_tokens: int) -> float:
    """Simulated prefill compute for the uncached suffix (0 if fully cached)."""
    if uncached_tokens <= 0:
        return 0.0
    return PREFILL_LAT + uncached_tokens * PREFILL_PER_TOKEN


def decode_time(output_tokens: int) -> float:
    """Simulated decode occupancy for ``output_tokens`` generated tokens (batch=1)."""
    return output_tokens * DECODE_PER_TOKEN


def decode_step_time(batch_size: int) -> float:
    """Simulated time to generate one token for every request in a decode batch.

    This is the time-between-tokens (TBT) each batched request observes for that
    step. Monotonically increasing in ``batch_size`` (>= :data:`TBT_BASE`), so a
    request's TBT degrades as its decode instance fills up.
    """
    b = max(1, batch_size)
    return TBT_BASE + (b - 1) * TBT_BATCH_SLOPE
