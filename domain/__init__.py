"""Domain facts: what the thing being simulated actually costs.

Not simulator machinery (that is ``realsim`` / ``sim_common``) and not a selector
(that is a capability). :mod:`domain.llm` describes the served transformer --
flops per prefill token, flops per decode step, KV bytes per token -- and turns
token counts into seconds against a machine profile. Both capabilities describe
operations on a model's tensors, so it belongs to neither of them.
"""

from .llm import (
    DEFAULT_MODEL,
    DEFAULT_PROFILE,
    decode_step_time,
    MachineProfile,
    Model,
    prefill_time,
)

__all__ = [
    "Model",
    "DEFAULT_MODEL",
    "prefill_time",
    "decode_step_time",
    "MachineProfile",
    "DEFAULT_PROFILE",
]
