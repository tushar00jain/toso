"""Discrete-event simulation of a KV cache on TorchStore.

Laid out by plane, the same split ``dedup_sim`` uses, so the two capabilities can
be compared folder by folder:

* :mod:`kvcache_sim.control` -- what is **decided**: the serving scheduler
  (prefill placement, pull-vs-recompute, TTFT/TBT gates, decode placement), the
  source :class:`proposed.policy.KeySelector` it delegates "which peer" to, the
  prefix-run view it senses through, and LRU eviction bookkeeping. Moves nothing,
  and imports nothing from ``data``;
* :mod:`kvcache_sim.data` -- what **executes** those decisions on ``realsim``'s
  real objects: the per-request serving loop, the batched decode engine, and the
  three KV directory verbs over a :class:`realsim.mesh.Mesh`;
* :mod:`kvcache_sim.workload` -- what is simulated (domain model, seeded request
  generator, and the ``Run`` list each comparison executes);
* :mod:`kvcache_sim.report` -- outcome metrics and rendering, on the shared
  :class:`sim_common.report.Ledger`.

The rule for which folder something belongs in: **does it advance the clock or
move bytes?** The decode engine sleeps and emits tokens, so it is data; the LRU
only picks victims, so it is control; a directory read is control even though it
awaits.

The LLM being served is :class:`domain.llm.Model` (shared with ``dedup_sim``) and
the target machine a :class:`~sim_common.cost_model.MachineProfile`. Both are
explicit arguments wherever they are used, defaulting to
:data:`domain.llm.DEFAULT_MODEL` and
:data:`sim_common.cost_model.DEFAULT_PROFILE`. ``domain.llm`` also owns
``prefill_time`` / ``decode_step_time``, which *both* planes call -- control to
predict, data to charge.

See ``dedup_sim/README.md`` for the folder-by-folder comparison of the two.
"""
