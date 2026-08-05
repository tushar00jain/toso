"""Discrete-event simulation of a KV cache on TorchStore.

Laid out in role folders, the same set ``dedup_sim`` uses, so the two
capabilities can be compared folder by folder:

* :mod:`kvcache_sim.policy` -- the **control plane**: the algorithm under test
  (scheduler + LRU eviction bookkeeping). Decides; moves nothing;
* :mod:`kvcache_sim.workload` -- what is simulated (domain model, seeded request
  generator, scenario builders + run harness);
* :mod:`kvcache_sim.runtime` -- the **data plane**: what executes those decisions
  on ``realsim``'s real objects (the KV directory verbs over a
  :class:`realsim.mesh.Mesh`, the request driver, the batched decode engine);
* :mod:`kvcache_sim.report` -- outcome metrics and rendering;
* :mod:`kvcache_sim.utils` -- prefill / decode-step GPU times, used by *both*
  planes (the control plane predicts with them, the data plane charges them).

The LLM being served is :class:`realsim.model.Model` (shared with ``dedup_sim``)
and the target machine a :class:`~sim_common.cost_model.MachineProfile`. Both are
explicit arguments wherever they are used, defaulting to
:data:`realsim.model.DEFAULT_MODEL` and
:data:`sim_common.cost_model.DEFAULT_PROFILE`.

See ``dedup_sim/README.md`` for the folder-by-folder comparison of the two.
"""
