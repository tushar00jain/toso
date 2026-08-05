"""CONTROL PLANE: the decisions, and the metadata they are made from.

Nothing here moves bytes or advances the clock -- these modules decide, and hand
the decision to the data plane (:mod:`kvcache_sim.runtime`) to execute:

- ``scheduler.py`` -- the two policies compared: ``LoadBalanceScheduler``
  (baseline, ~vLLM: least-loaded, local cache only) and ``CacheAwareScheduler``
  (the cache-aware coordinator: route on the global prefix directory, optionally
  pull a remote prefix). Reads the real ``Controller`` directory, returns a
  ``Plan``, and owns the analytical prefill-queue model it predicts TTFT from.
- ``cache.py`` -- per-instance LRU eviction bookkeeping: which keys an instance
  holds and their recency. Metadata mirroring the directory, never the KV bytes.

The cost functions a decision is made with are shared with the data plane -- which
charges the same numbers -- so they live at the package root in
:mod:`kvcache_sim.utils`, over a :class:`realsim.model.Model` and a
:class:`~sim_common.cost_model.MachineProfile`. A transfer's cost is
:func:`sim_common.cost_model.get_time`.
"""
