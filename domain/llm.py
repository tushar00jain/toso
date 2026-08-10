"""The served model, described by the properties a simulation needs.

:class:`Model` reduces a transformer to the quantities a simulation charges
against: how many flops a token costs to prefill, how many a decode step costs per
batched request, and how many bytes of KV cache a token occupies
(:meth:`Model.block_bytes`). :func:`prefill_time` and :func:`decode_step_time`
turn those into seconds. The hardware half is separate and lives in
``sim_common`` (:class:`~sim_common.cost_model.MachineProfile`).

The model is a third axis, independent of both the workload and the hardware: the
same workload on a 7B vs a 70B model has different flops and KV bytes per token,
while the same model serving different workloads has the same ones. It is needed
because a sim's unit of work is a **token** while
:func:`~sim_common.cost_model.compute_time` prices **flops** -- something has to
convert, and the model's architecture *is* that conversion.

Why it is neither in ``realsim`` nor in a capability
----------------------------------------------------
These are **domain facts** -- what a transformer costs -- not simulator
machinery and not a policy. ``realsim`` is about driving the real store; a
capability is about one decision. Both capabilities describe operations on a
model's tensors:

* ``kvcache_sim`` prices prefill/decode compute and KV block bytes from it;
* ``dedup_sim`` syncs a model's **weights**, so the payload it moves should be
  derived from a model too rather than an arbitrary element count -- see the TODO
  below.

:func:`prefill_time` and :func:`decode_step_time` live here for the same reason
and because each is used on **both sides** of the control/data split -- once to
decide, once to charge:

* :func:`prefill_time` -- control compares it against the cost of pulling a
  prefix (:func:`sim_common.cost_model.get_time`) to choose reuse over recompute;
  data then sleeps the chosen value as the actual prefill charge;
* :func:`decode_step_time` -- control predicts TBT with it for the admission/SLO
  decision; data charges it per step as the real time-between-tokens.

Three cost premises the KV-cache algorithm rests on:

* recomputing a token costs GPU compute, so reusing a cached prefix is cheaper
  (:func:`prefill_time`);
* moving a cached KV block over the fabric is cheaper than recomputing it, so
  remote reuse / hot-block replication pays off
  (:func:`sim_common.cost_model.get_time` vs :func:`prefill_time` per token);
* a decode step's per-token time (TBT) rises with the decode-batch size, so
  packing more concurrent requests trades throughput for latency
  (:func:`decode_step_time`).

The premise that constrains *this* module's numbers:
**it is a property of the numbers, not a law**, and its real
precondition is the KV pool's **read bandwidth**, not the interconnect. At
realistic sizes a KV block is large (a 512-token block of an 8B-class model is
64 MiB), so of the three terms :func:`~sim_common.cost_model.get_time` charges,
the *storage read* dominates the fabric transfer by roughly 5x. The premise
therefore holds only while the pool is **memory-resident** -- which it is here
(the volume seam is backed by a real ``InMemoryStore``, and an inference KV cache
lives in HBM/DRAM). Price that pool at NVMe-class bandwidth instead and
recomputing becomes cheaper than fetching, which inverts the sim's headline
conclusion.

:data:`DEFAULT_MODEL` is *illustrative* (see its comment) and its terms are not
derived from any one real model, so editing any single constant -- here or in the
machine profile -- can move that boundary silently.
:meth:`Model.from_architecture` builds a dimensionally self-consistent
profile from a real model's shape, and ``kvcache_sim/tests/test_cost_premises.py``
asserts the premise for the illustrative default, for a real 8B-class model on
realistic hardware, and pins the memory-vs-NVMe crossover -- so an inversion
fails loudly instead of quietly changing what the sim concludes.
"""

from __future__ import annotations

from dataclasses import dataclass

from sim_common.cost_model import compute_time, DEFAULT_PROFILE, MachineProfile


__all__ = ["Model", "DEFAULT_MODEL", "prefill_time", "decode_step_time"]

@dataclass(frozen=True)
class Model:
    """The LLM being served, as the properties the simulation needs.

    Together these turn a *token* count into flops and bytes -- the conversion the
    cost model needs and the workload cannot supply.

    Args:
        prefill_flops_per_token: GPU flops to prefill one token. For a dense
            transformer this is ~``2 * parameter_count`` (one forward pass).
        decode_step_flops_per_request: GPU flops for one decode step for one
            batched request. Physically also ~``2 * parameter_count``; kept as a
            separate field because the modeled decode step deliberately ignores
            the context-length-dependent attention term (a documented
            simplification), so the two need not be equal in an illustrative
            profile.
        kv_bytes_per_token: bytes of KV cache one token occupies:
            ``layers * kv_heads * head_dim * 2 (K and V) * dtype_bytes``. This is
            the quantity that crosses the fabric on a remote prefix pull, so it
            sets the fetch-vs-recompute tradeoff.
        compute_dtype: dtype key used to look up the flop rate in
            :attr:`~sim_common.cost_model.MachineProfile.gpu_flops`.
        compute_device: device the model's compute is charged on.
    """

    prefill_flops_per_token: float
    decode_step_flops_per_request: float
    kv_bytes_per_token: int
    compute_dtype: str = "float16"
    compute_device: str = "cuda"

    @classmethod
    def from_architecture(
        cls,
        *,
        params: float,
        layers: int,
        kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
        compute_dtype: str = "float16",
        compute_device: str = "cuda",
    ) -> "Model":
        """Derive self-consistent properties from a real model's shape.

        Both flop terms become ``2 * params`` (one forward pass per token) and the
        KV size becomes ``layers * kv_heads * head_dim * 2 * dtype_bytes``, so the
        compute and fabric terms are on the same physical footing. Use this (with
        a correspondingly realistic :class:`MachineProfile`) when the *ratio*
        between recompute and fetch has to mean something; ``kv_heads`` is the
        grouped-query (GQA) count, not the attention-head count.

        Example (an 8B-class model, GQA, fp16)::

            Model.from_architecture(
                params=8e9, layers=32, kv_heads=8, head_dim=128
            )
        """
        fwd_flops = 2.0 * params
        return cls(
            prefill_flops_per_token=fwd_flops,
            decode_step_flops_per_request=fwd_flops,
            kv_bytes_per_token=layers * kv_heads * head_dim * 2 * dtype_bytes,
            compute_dtype=compute_dtype,
            compute_device=compute_device,
        )

    def block_bytes(self, num_blocks: int, block_tokens: int) -> int:
        """Modeled KV byte size of ``num_blocks`` blocks of ``block_tokens`` tokens.

        The single source of the block byte count: the KV data plane sizes its
        block carrier from it (so this is what the transport charges) and the
        scheduler prices a prefix pull with it, so both sides of a routing
        decision are the same number.
        """
        return num_blocks * block_tokens * self.kv_bytes_per_token


# The illustrative default model. Like DEFAULT_PROFILE these are plausible
# *relative* magnitudes chosen to keep the printed numbers readable -- NOT any
# real model measured or derived: 1 modeled KV byte per token keeps fabric counts
# legible, and the flop terms are scaled to match the profile's small flop rates.
# They are therefore not mutually derived (contrast Model.from_architecture),
# which is exactly why the premise test exists. Production callers should build
# their own from ``from_architecture`` plus a measured MachineProfile.
DEFAULT_MODEL = Model(
    prefill_flops_per_token=1.6e6,
    decode_step_flops_per_request=4.0e7,
    kv_bytes_per_token=1,
)

# Consumers take the model as an explicit argument, defaulting to this, because it
# changes the *simulated result* -- per the repo convention that only debug/output
# knobs may be read ambiently from the config.


# TODO(next diff): let ``dedup_sim`` derive its payload from a Model.
#
# ``dedup_sim``'s burst currently moves ``n`` elements of one key (the default is
# 64 bytes), which makes its fabric numbers arbitrary. What it actually simulates
# is **weight sync**, so the payload should be a model's weights: an 8B model in
# fp16 is ~14.9 GiB, ~3.7 GiB per rank 4-way sharded, so a 16-reader burst moves
# ~60 GiB naively vs 1x deduped -- numbers that mean something.
#
# What this class needs for that: it currently stores the *derived* per-token terms
# (so an illustrative DEFAULT_MODEL that no real architecture produces can exist),
# and ``from_architecture`` discards the shape it derived them from. Add the shape
# as fields, plus weight-side derivations:
#
#   * ``weight_bytes`` -- total parameter bytes;
#   * ``weight_carriers(world_size)`` -- per-rank **allocation-free** carriers (a
#     ``device="meta"`` tensor or a ``TensorDescriptor``, as
#     ``realsim.scenarios.put_get`` already builds), so a 60 GiB burst still
#     costs no memory;
#   * optionally a whole ``state_dict``'s worth of them (~290 keys for an 8B
#     model), which is what real weight sync moves.
#
# The *resharding* machinery this would feed is already real and already runs: the
# client's ``_expand_tensor_slices`` / ``_assemble_results`` execute unchanged (see
# ``realsim/adapters/real_client.py``), ``Reader.tensor_slice_spec`` already routes
# a per-reader slice, and ``realsim/tests/test_seams.py`` asserts an exact sliced
# get through a real ``TensorSlice``. Only the payloads are missing -- per-rank
# ``TensorSlice``s of a model's weights across two different meshes -- not the
# intersection math.
#
# Keep it opt-in: leave ``put_get``'s ``n``-based default byte-identical (per the
# repo's default-off convention) and select a model explicitly, so the existing
# numbers and recorded fingerprints stand.


# --------------------------------------------------------------------------- #
# Token -> time. Both sides of the control/data split call these (see the module
# docstring), which is why they sit beside the model rather than behind either
# plane.
# --------------------------------------------------------------------------- #


def prefill_time(
    uncached_tokens: int,
    profile: MachineProfile = DEFAULT_PROFILE,
    model: Model = DEFAULT_MODEL,
) -> float:
    """GPU prefill compute for the uncached suffix (0 if fully cached).

    Charged through :func:`~sim_common.cost_model.compute_time` on the model's
    accelerator: the cost a prefix cache hit avoids.
    """
    if uncached_tokens <= 0:
        return 0.0
    flops = model.prefill_flops_per_token * uncached_tokens
    return compute_time(flops, model.compute_dtype, model.compute_device, profile)


def decode_step_time(
    batch_size: int,
    profile: MachineProfile = DEFAULT_PROFILE,
    model: Model = DEFAULT_MODEL,
) -> float:
    """Time to generate one token for every request in a decode batch.

    This is the time-between-tokens (TBT) each batched request observes for that
    step. Charged as GPU compute proportional to the batch size (clamped to
    ``>= 1``), so it is strictly increasing in the batch -- a request's TBT
    degrades as its decode instance fills up.
    """
    b = max(1, batch_size)
    flops = model.decode_step_flops_per_request * b
    return compute_time(flops, model.compute_dtype, model.compute_device, profile)
