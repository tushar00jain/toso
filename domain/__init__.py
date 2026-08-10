"""Domain facts: what the thing being simulated actually costs.

Not simulator machinery (that is ``realsim`` / ``sim_common``) and not a policy
(that is a capability). :mod:`domain.llm` describes the served transformer --
flops per prefill token, flops per decode step, KV bytes per token -- and turns
token counts into seconds against a machine profile. Both capabilities describe
operations on a model's tensors, so it belongs to neither of them.
"""
