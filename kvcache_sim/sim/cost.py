"""KV-cache cost layer over the shared analytic cost model.

Every duration here is a deterministic function of a *modeled quantity* (uncached
tokens, decode-batch size, KV byte count) and a target-machine
:class:`~sim_common.cost_model.MachineProfile`. This module does **not** own any
bandwidth/flop constants -- it composes the shared
:mod:`sim_common.cost_model` primitives (``compute_time`` / ``network_time`` /
``storage_time`` / ``mem_copy_time``), exactly as ``realsim``'s transport seam and
``burst_get`` scenario do. The old ad-hoc ``TIERS`` / ``PREFILL_*`` / ``TBT_*``
constants are gone; the profile is the single source of hardware truth.

Three cost premises the KV-cache algorithm rests on, all preserved by the profile:

* recomputing a token costs GPU compute, so reusing a cached prefix is cheaper
  (:func:`prefill_time`);
* moving a cached KV block over the fabric is cheaper than recomputing it, so
  remote reuse / hot-block replication pays off (:func:`fetch_time` vs
  :func:`prefill_time` per token, given :data:`PROFILE`);
* a decode step's per-token time (TBT) rises with the decode-batch size, so
  packing more concurrent requests trades throughput for latency
  (:func:`decode_step_time`).
"""

from __future__ import annotations

from sim_common.cost_model import (
    DEFAULT_PROFILE,
    MachineProfile,
    compute_time,
    mem_copy_time,
    network_time,
    storage_time,
)
from sim_common.topology import Endpoint, Tier, TIER_LABEL, locality  # noqa: F401

# The target-machine profile the simulation is charged against. We reuse the
# shared illustrative DEFAULT_PROFILE (plausible *relative* magnitudes, not
# measured): with it, a KV block's fabric fetch is well under the prefill compute
# it avoids, which is the premise that makes prefix reuse and replication pay off.
PROFILE: MachineProfile = DEFAULT_PROFILE

# KV blocks/tokens are modeled as bytes for the data plane: one modeled byte per
# token (illustrative, keeps fabric numbers readable), so a B-token block is B
# bytes. The metadata carrier (a uint8 TensorDescriptor of shape ``(B,)``) has
# exactly this nbytes with zero real allocation.
BYTES_PER_TOKEN = 1

# Prefill is GPU compute proportional to the uncached-token count; the flop rate
# comes from the profile (``gpu_flops``), so this is a pure cost-model call. The
# per-token flop count is a modeled constant (a stand-in for the attention +
# MLP flops a prefill token costs).
PREFILL_FLOPS_PER_TOKEN = 1.6e6

# A decode step attends over the whole live batch, so its flop count scales with
# the batch size; the step time is that compute charged on the GPU roofline. This
# replaces the ad-hoc ``TBT_BASE + (b-1)*slope`` with a profile-driven compute
# time that is monotonic in the batch (base at batch=1, strictly rising).
DECODE_STEP_FLOPS_PER_REQ = 4.0e7

# Both compute steps are charged on the accelerator that serves the model.
COMPUTE_DEVICE = "cuda"
COMPUTE_DTYPE = "float16"


def block_bytes(num_blocks: int, block_tokens: int) -> int:
    """Modeled KV byte size of ``num_blocks`` blocks of ``block_tokens`` tokens."""
    return num_blocks * block_tokens * BYTES_PER_TOKEN


def prefill_time(uncached_tokens: int, profile: MachineProfile = PROFILE) -> float:
    """GPU prefill compute for the uncached suffix (0 if fully cached).

    Charged through :func:`~sim_common.cost_model.compute_time` on the accelerator:
    the cost a prefix cache hit avoids.
    """
    if uncached_tokens <= 0:
        return 0.0
    flops = PREFILL_FLOPS_PER_TOKEN * uncached_tokens
    return compute_time(flops, COMPUTE_DTYPE, COMPUTE_DEVICE, profile)


def decode_step_time(batch_size: int, profile: MachineProfile = PROFILE) -> float:
    """Time to generate one token for every request in a decode batch.

    This is the time-between-tokens (TBT) each batched request observes for that
    step. Charged as GPU compute proportional to the batch size (clamped to
    ``>= 1``), so it is strictly increasing in the batch -- a request's TBT
    degrades as its decode instance fills up.
    """
    b = max(1, batch_size)
    flops = DECODE_STEP_FLOPS_PER_REQ * b
    return compute_time(flops, COMPUTE_DTYPE, COMPUTE_DEVICE, profile)


def fetch_time(
    src: Endpoint, dst: Endpoint, nbytes: int, profile: MachineProfile = PROFILE
) -> float:
    """Predicted cost of one client ``get`` of ``nbytes`` from ``src`` to ``dst``.

    Mirrors the charges the real transport seam applies on a get -- persistent
    ``storage`` read + host-RAM ``mem`` staging + ``network`` fabric -- so a
    routing prediction made with this matches the time the real fetch advances the
    clock by. A same-endpoint (local) fetch is free.
    """
    if src.id == dst.id or nbytes <= 0:
        return 0.0
    return (
        storage_time(nbytes, "read", profile)
        + mem_copy_time(nbytes, profile)
        + network_time(src, dst, nbytes, profile)
    )
