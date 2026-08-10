"""The cost premises the KV-cache result rests on, asserted rather than assumed.

The sim's headline conclusion -- cache-aware routing with remote prefix pulls
beats load balancing -- depends on one inequality:

    fetching a cached KV block  <  recomputing it

That is **not a law**; it is a property of four numbers spread across two
descriptors (``kv_bytes_per_token`` and ``prefill_flops_per_token`` in
:class:`~domain.llm.Model`, the fabric bandwidth and ``gpu_flops`` in
:class:`~sim_common.cost_model.MachineProfile`). Change any one of them -- e.g.
make ``gpu_flops`` realistic without touching ``kv_bytes_per_token`` -- and the
inequality can flip, which flips what the sim concludes, with nothing failing.

:data:`~domain.llm.DEFAULT_MODEL` is explicitly illustrative and its terms
are *not* mutually derived, so these tests cover both:

1. the illustrative default profiles keep the premise (and by how much);
2. a **dimensionally self-consistent** pairing -- a real model shape via
   ``Model.from_architecture`` on realistic hardware -- also keeps it, so
   the conclusion is not an artifact of the illustrative scaling; and
3. the predicted block size equals the byte count the data plane actually moves,
   the coupling that makes the prediction meaningful at all.
"""

from __future__ import annotations

import math
from dataclasses import replace

from sim_common.async_engine import run_sim
from sim_common.cost_model import DEFAULT_PROFILE, get_time, MachineProfile
from sim_common.topology import Tier

from domain import DEFAULT_MODEL, Model, prefill_time
from kvcache_sim.workload.deploy import make_store
from kvcache_sim.workload.scenarios import BLOCK_TOKENS, make_topology

# Bounds on how much cheaper fetching a block is than recomputing it, under the
# illustrative defaults. Wide on purpose: the point is to catch a *sign* change
# (or an order-of-magnitude drift), not to freeze the exact constants.
MIN_RATIO, MAX_RATIO = 2.0, 100.0

# An 8B-class model with grouped-query attention, fp16: 32 layers x 8 KV heads x
# 128 head dim x (K and V) x 2 bytes = 131072 KV bytes per token.
REAL_MODEL = Model.from_architecture(
    params=8e9, layers=32, kv_heads=8, head_dim=128
)

# Matching real hardware: ~1 PFLOP/s dense fp16 accelerator, ~25 GB/s effective
# cross-node fabric, ~1 TB/s HBM-class host copy. Units are seconds/bytes, so
# these are directly comparable to REAL_MODEL's terms.
#
# ``storage_read_bw`` is the decisive field. The "storage" tier in this sim is the
# serving volume's KV pool -- a real ``InMemoryStore``, i.e. **memory-resident**,
# which is what an inference KV cache is -- so it is priced at DRAM/HBM bandwidth
# here. That choice, not the fabric, is what makes reuse pay off at realistic KV
# sizes: one 512-token block of an 8B-class model is 64 MiB, so the storage-read
# term dominates the fabric term by ~5x. See
# :func:`test_premise_depends_on_where_kv_is_served_from`.
REAL_MACHINE = MachineProfile(
    tiers={
        Tier.SHM: (1e-6, 2.0e11),
        Tier.NVLINK: (2e-6, 4.5e11),
        Tier.RDMA: (5e-6, 2.5e10),
    },
    ram_bandwidth=1.0e12,
    ram_latency=1e-7,
    storage_read_bw=1.0e12,   # memory-resident KV pool
    storage_write_bw=8.0e11,
    storage_latency=1e-7,
    gpu_flops={"float16": 1.0e15, "bfloat16": 1.0e15},
    gpu_flops_default=5.0e14,
    gpu_mem_bandwidth=3.0e12,
    cpu_flops=1.0e11,
)

# The same machine with the KV pool on NVMe-class storage instead of in memory.
# Nothing else differs.
NVME_MACHINE = replace(
    REAL_MACHINE, storage_read_bw=5.0e9, storage_write_bw=3.0e9, storage_latency=1e-5
)


def _endpoints():
    """A local instance, a same-node peer (NVLink) and a cross-node peer (RDMA)."""
    topo = make_topology(4, per_node=2)
    ids = sorted(topo)
    local = topo[ids[0]]
    same_node = next(topo[i] for i in ids[1:] if topo[i].node == local.node)
    cross_node = next(topo[i] for i in ids[1:] if topo[i].node != local.node)
    return local, same_node, cross_node


def _ratio(model: Model, machine: MachineProfile, src, dst) -> float:
    """recompute-one-block / fetch-one-block. > 1 means reuse pays off."""
    nbytes = model.block_bytes(1, BLOCK_TOKENS)
    fetch = get_time(src, dst, nbytes, machine)
    assert fetch > 0.0, "a cross-instance fetch must cost something"
    return prefill_time(BLOCK_TOKENS, machine, model) / fetch


def test_reuse_beats_recompute_under_the_illustrative_defaults():
    """Fetching a block is cheaper than recomputing it, on both locality tiers.

    This is the premise ``domain.llm`` documents. If it inverts, the sim
    still runs and still reports a "winner" -- so assert it explicitly.
    """
    local, same_node, cross_node = _endpoints()
    for peer, label in ((same_node, "same-node"), (cross_node, "cross-node")):
        r = _ratio(DEFAULT_MODEL, DEFAULT_PROFILE, peer, local)
        assert r > 1.0, (
            f"{label}: recompute/fetch = {r:.3f} -- fetching a KV block is no "
            "longer cheaper than recomputing it, so the premise behind "
            "cache-aware routing has inverted"
        )
        assert MIN_RATIO <= r <= MAX_RATIO, (
            f"{label}: recompute/fetch = {r:.3f} drifted outside the documented "
            f"[{MIN_RATIO}, {MAX_RATIO}] band; if this is intentional, update the "
            "band and the premise note in domain/llm.py"
        )
    # Pulling from a same-node peer must be cheaper than from across the fabric,
    # which is what makes the scheduler's locality preference meaningful.
    assert _ratio(DEFAULT_MODEL, DEFAULT_PROFILE, same_node, local) > _ratio(
        DEFAULT_MODEL, DEFAULT_PROFILE, cross_node, local
    )


def test_premise_survives_a_dimensionally_consistent_profile():
    """The premise is not an artifact of the illustrative scaling.

    ``DEFAULT_MODEL``'s flop and byte terms are not derived from one another, so
    on its own it proves little. Re-check with a real model shape on real
    hardware: recompute must still exceed fetch, cross-node included.
    """
    local, same_node, cross_node = _endpoints()
    for peer, label in ((same_node, "same-node"), (cross_node, "cross-node")):
        r = _ratio(REAL_MODEL, REAL_MACHINE, peer, local)
        assert r > 1.0, (
            f"{label}: with a self-consistent 8B-class model on realistic "
            f"hardware, recompute/fetch = {r:.3f} -- fetching is no longer the "
            "cheaper option, so the modeled conclusion would not transfer"
        )


def test_premise_depends_on_where_kv_is_served_from():
    """The premise's real precondition: the KV pool must be memory-resident.

    At realistic KV sizes a 512-token block of an 8B-class model is 64 MiB, so the
    **storage-read** term of a get dominates the fabric term by ~5x. That makes
    ``storage_read_bw`` -- not the interconnect -- the field that decides whether
    remote reuse pays off:

    * pool in DRAM/HBM (~1 TB/s), which is what an inference KV cache is and what
      the real ``InMemoryStore`` behind the volume seam models -> reuse wins;
    * pool on NVMe-class storage (~5 GB/s) -> **recomputing is cheaper** and the
      cache-aware advantage disappears.

    Recording the crossover here means the conclusion is stated with its
    precondition attached, rather than read as unconditional.
    """
    local, _, cross_node = _endpoints()
    in_memory = _ratio(REAL_MODEL, REAL_MACHINE, cross_node, local)
    on_nvme = _ratio(REAL_MODEL, NVME_MACHINE, cross_node, local)

    assert in_memory > 1.0 > on_nvme, (
        "expected the fetch-vs-recompute tradeoff to hinge on the KV pool's "
        f"bandwidth, but got in-memory={in_memory:.3f} and nvme={on_nvme:.3f}"
    )
    # And the ordering is driven by storage bandwidth alone (same fabric, same
    # model, same compute), so a profile change there flips the sim's conclusion.
    assert in_memory > on_nvme


def test_from_architecture_derives_kv_bytes_and_flops_together():
    """The derived profile's two terms come from the same physical description."""
    assert REAL_MODEL.kv_bytes_per_token == 32 * 8 * 128 * 2 * 2
    assert REAL_MODEL.prefill_flops_per_token == 2 * 8e9
    # A forward pass is a forward pass: both flop terms agree when derived.
    assert (
        REAL_MODEL.decode_step_flops_per_request
        == REAL_MODEL.prefill_flops_per_token
    )
    # The illustrative default is deliberately NOT self-consistent, which is why
    # the check above exists; assert that so the distinction cannot rot silently.
    assert (
        DEFAULT_MODEL.decode_step_flops_per_request
        != DEFAULT_MODEL.prefill_flops_per_token
    )


def test_predicted_block_bytes_equal_the_bytes_the_data_plane_moves():
    """What the scheduler prices must be what the transport charges.

    ``Model.block_bytes`` feeds the routing prediction; the carrier's size is what the
    transport actually charges. They are both derived from the model profile, and
    this pins them together for the default and for a much larger model.
    """
    for model in (DEFAULT_MODEL, REAL_MODEL):
        _, cl = make_store(
            make_topology(2), block_tokens=BLOCK_TOKENS, model=model
        )
        assert cl.block_nbytes == model.block_bytes(1, BLOCK_TOKENS), (
            "the KV block carrier and Model.block_bytes() disagree, so every "
            "fetch "
            "prediction would be priced against the wrong byte count"
        )


def test_a_real_pull_costs_what_get_time_predicted():
    """End to end: the clock advance of a real block pull equals the prediction.

    Closes the loop through the actual data plane -- real directory, real
    ``client.get_batch``, real transport charge -- rather than comparing two
    formulas.
    """
    topo = make_topology(4, per_node=2)
    ids = sorted(topo)
    holder, puller = ids[0], next(i for i in ids[1:] if topo[i].node != topo[ids[0]].node)
    keys = ["blk0"]

    async def scenario():
        import asyncio

        mesh, cl = make_store(topo, block_tokens=BLOCK_TOKENS)
        with mesh.installed():
            await cl.publish(holder, list(keys))
            loop = asyncio.get_running_loop()
            before = loop.time()
            await cl.fetch(puller, list(keys))
            return loop.time() - before, cl.block_nbytes

    (advance, nbytes), _ = run_sim(scenario())
    predicted = get_time(topo[holder], topo[puller], nbytes, DEFAULT_PROFILE)
    assert advance > 0.0
    assert math.isclose(advance, predicted, rel_tol=1e-12), (
        f"a real cross-node pull advanced the clock {advance!r} but get_time "
        f"predicted {predicted!r}"
    )
