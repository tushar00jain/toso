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

Where premise 3 is checked moved with the thing it is about. The KV store used to
enforce it at construction -- it was handed a "carrier" and refused one that was not
``Model.block_bytes`` big -- but the store neither produces KV nor knows what a
token costs, so the check sat one object away from every number in it. The
accelerator produces the blocks now
(:class:`~kvcache_sim.workload._accelerator.SimulatedAccelerator`), derives their
size from the model, and refuses a model whose block does not come out whole; the
tests below assert against a **produced block** rather than against anything that
object says about itself.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from sim_common.async_engine import run_sim
from sim_common.cost_model import DEFAULT_PROFILE, _get_time, MachineProfile
from sim_common.topology import Tier

from domain import DEFAULT_MODEL, Model, prefill_time
from kvcache_sim.data.store import KVStore
from kvcache_sim.workload._accelerator import (
    SimulatedAccelerator, TOKEN_DTYPE, token_tensor,
)
from realsim.simulation import Simulation
from kvcache_sim.workload.scenarios import BLOCK_TOKENS, _make_topology

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
    topo = _make_topology(4, per_node=2)
    ids = sorted(topo)
    local = topo[ids[0]]
    same_node = next(topo[i] for i in ids[1:] if topo[i].node == local.node)
    cross_node = next(topo[i] for i in ids[1:] if topo[i].node != local.node)
    return local, same_node, cross_node


def _ratio(model: Model, machine: MachineProfile, src, dst) -> float:
    """recompute-one-block / fetch-one-block. > 1 means reuse pays off."""
    nbytes = model.block_bytes(1, BLOCK_TOKENS)
    fetch = _get_time(src, dst, nbytes, machine)
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

    ``Model.block_bytes`` feeds the routing prediction; a produced block's
    ``numel * element_size`` is what the transport charges. Asserted against the
    tensor a forward pass actually hands back -- not against the accelerator's own
    ``block_nbytes``, which would let the two agree by both being wrong.
    """
    for model in (DEFAULT_MODEL, REAL_MODEL):
        gpu = SimulatedAccelerator(model=model, block_tokens=BLOCK_TOKENS)
        block, = gpu.kv_blocks(1)
        assert block.numel() * block.element_size() == model.block_bytes(
            1, BLOCK_TOKENS
        ), (
            "a produced KV block and Model.block_bytes() disagree, so every fetch "
            "prediction would be priced against the wrong byte count"
        )


def test_a_kv_block_is_a_real_tensor_with_no_storage():
    """The carrier premise's other half: a real tensor, and free to hold.

    Both matter and they pull in opposite directions. It must be a
    ``torch.Tensor``, because ``put_batch`` types its value -- a ``Tensor``/
    ``DTensor`` goes down ``Request.from_any`` and anything else down
    ``Request.from_objects`` -- so a non-tensor carrier puts every KV block on
    torchstore's *object* path, which is not the path a KV deployment takes. And it
    must allocate nothing, because these runs "store" far more KV than the test box
    has memory for. ``device="meta"`` is what satisfies both.
    """
    block, = SimulatedAccelerator(block_tokens=BLOCK_TOKENS).kv_blocks(1)
    assert isinstance(block, torch.Tensor)
    assert block.device.type == "meta"
    assert block.dtype == getattr(torch, DEFAULT_MODEL.compute_dtype)
    # Distinct objects per block: the store keys them separately and a volume
    # evicts them separately, so nothing here may alias.
    a, b = SimulatedAccelerator(block_tokens=BLOCK_TOKENS).kv_blocks(2)
    assert a is not b


def test_a_prompt_and_a_generated_token_are_real_tensors_with_no_storage():
    """The same premise at the two ends of a request, where it is easier to break.

    A KV block has an arithmetic reason to be a real tensor -- the transport prices
    what it moves. The prompt and the generated tokens have no byte count anybody
    charges, which makes them precisely the place a stand-in would go unnoticed: an
    ``int`` count, a list of Python ints, a ``None``. So the shape is asserted
    rather than assumed, and so is the cost of it -- these runs carry a prompt per
    request and a token per decode step per request, and every one of them has to
    be free.
    """
    gpu = SimulatedAccelerator(block_tokens=BLOCK_TOKENS)
    prompt = token_tensor(4 * BLOCK_TOKENS)
    (first,) = gpu.step_tokens(1)
    for name, t in (("prompt", prompt), ("token", first)):
        assert isinstance(t, torch.Tensor), name
        assert t.device.type == "meta", name
        assert t.dtype is TOKEN_DTYPE, name
        # A meta tensor reports the size it *would* occupy and points at nothing:
        # ``nbytes`` is real arithmetic over shape and dtype, while the storage has
        # no address, which is what lets a run "hold" every prompt and every token
        # of a 300-request workload for free.
        assert t.numel() * t.element_size() > 0, name
        assert t.data_ptr() == 0, name
    assert prompt.numel() == 4 * BLOCK_TOKENS
    assert first.numel() == 1               # one token, not a batch of one
    # ...and a step's tokens are one object per member, never one aliased tensor.
    a, b = gpu.step_tokens(2)
    assert a is not b


def test_a_block_that_does_not_come_out_whole_is_refused():
    """The premise fails loudly where the KV is made, instead of rounding.

    The check the ``KVStore`` constructor used to do, in its new home and in the
    only form left to it: the block size is *derived* from
    ``Model.block_bytes``, so it cannot disagree with the prediction -- unless the
    model's bytes are not a whole number of KV elements, in which case a block would
    have to round and be charged a size nobody predicted. Refused rather than
    rounded, because the run would otherwise be internally consistent and wrong.
    """
    odd = replace(DEFAULT_MODEL, kv_bytes_per_token=1)
    with pytest.raises(ValueError, match="whole number"):
        # 3 tokens x 1 byte = 3B, which is not a whole number of float16 elements.
        SimulatedAccelerator(model=odd, block_tokens=3)
    # ...and the even case a scenario actually runs is fine.
    assert SimulatedAccelerator(model=odd, block_tokens=BLOCK_TOKENS).block_nbytes == (
        odd.block_bytes(1, BLOCK_TOKENS)
    )


def test_a_real_pull_costs_what_get_time_predicted():
    """End to end: the clock advance of a real block pull equals the prediction.

    Closes the loop through the actual data plane -- real directory, real
    ``client.get_batch``, real transport charge -- rather than comparing two
    formulas.
    """
    topo = _make_topology(4, per_node=2)
    ids = sorted(topo)
    holder, puller = ids[0], next(i for i in ids[1:] if topo[i].node != topo[ids[0]].node)
    keys = ["blk0"]

    sim = Simulation(topo)
    cl = KVStore(sim.mesh)
    gpu = SimulatedAccelerator(block_tokens=BLOCK_TOKENS)

    async def scenario():
        import asyncio

        with sim.mesh.installed():
            await cl.publish(holder, list(keys), gpu.kv_blocks(len(keys)))
            loop = asyncio.get_running_loop()
            before = loop.time()
            pulled = await cl.fetch(puller, list(keys))
            # Measured off what came back, which is the whole point: the bytes the
            # clock advanced for are the bytes the fetch returned.
            moved = sum(b.numel() * b.element_size() for b in pulled)
            return loop.time() - before, moved

    # Drives one op rather than a workload, so it uses the stack's clock directly
    # instead of Simulation.run.
    try:
        advance, nbytes = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    predicted = _get_time(topo[holder], topo[puller], nbytes, DEFAULT_PROFILE)
    assert advance > 0.0
    assert math.isclose(advance, predicted, rel_tol=1e-12), (
        f"a real cross-node pull advanced the clock {advance!r} but _get_time "
        f"predicted {predicted!r}"
    )
