"""The KV-cache control plane: everything that is decided, nothing that runs.

* :mod:`~kvcache_sim.control.scheduler` -- the serving scheduler: which instance
  prefills, pull-vs-recompute, the TTFT/TBT admission gates, where decode lands.
  These are *compute* decisions the store knows nothing about, which is why they
  are app code and not part of the shared policy interface;
* :mod:`~kvcache_sim.control._source` -- the one part that *is* a store question,
  "which peer serves this prefix gap", as a :class:`proposed.policy.Policy`;
* :mod:`~kvcache_sim.control.view` -- the single derived directory read the
  scheduler needs (per-instance prefix-run lengths), plus the pinned snapshot one
  decision reads it through;
* :mod:`~kvcache_sim.control._cache` -- per-instance LRU. It picks victims; it
  does not delete anything.

Nothing here imports :mod:`kvcache_sim.data`, a deployment or a client, and
nothing here reaches into the simulator -- all checked by the repo's contract lint. Control senses through a view and returns
decisions; what actually happened comes back as an observation.
"""
