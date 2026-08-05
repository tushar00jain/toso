"""Shared helpers: the model compute times both planes need.

Both functions convert a *token count* into GPU time, using the served model's
flop terms (:class:`~realsim.model.Model`) and the target machine's flop rate
(:class:`~sim_common.cost_model.MachineProfile`).

They sit at the package root, in neither role folder, because each is used on
**both sides** of the control/data split -- once to decide, once to charge:

* :func:`prefill_time` -- the **control plane** compares it against the cost of
  pulling a prefix (:func:`sim_common.cost_model.get_time`) to choose reuse over
  recompute; the **data plane** then sleeps the chosen value as the actual prefill
  charge;
* :func:`decode_step_time` -- the **control plane** predicts TBT with it for the
  admission/SLO decision; the **data plane** (:mod:`kvcache_sim.runtime.decode`)
  charges it per step as the real time-between-tokens.

Putting them behind either folder would make the other import through it.
"""

from __future__ import annotations

from sim_common.cost_model import compute_time, DEFAULT_PROFILE, MachineProfile

from realsim.model import DEFAULT_MODEL, Model

__all__ = ["prefill_time", "decode_step_time"]


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
