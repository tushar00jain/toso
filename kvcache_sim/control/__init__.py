"""The KV-cache control plane: everything that is decided, nothing that runs.

* :mod:`~kvcache_sim.control.request` -- what is being decided *about*: an
  inference ``Request``, carrying the prompt it was submitted with and its
  prefix-hash block keys (each a plain directory key). The keys are not derived
  from the prompt and cannot be -- see that module for the compromise a
  zero-storage prompt forces. It sits here rather than in ``workload/`` because
  all three planes pass it, and ``workload/`` does not exist in production;
* :mod:`~kvcache_sim.control.scheduler` -- the serving scheduler: which instance
  prefills, pull-vs-recompute, the TTFT/TBT admission gates, where decode lands.
  These are *compute* decisions the store knows nothing about, which is why they
  are app code and not part of the shared policy interface;
* :mod:`~kvcache_sim.control._source` -- the one part that *is* a store question,
  "which peer serves this prefix gap", as a :class:`proposed.policy.Policy`. It is
  used twice: the scheduler calls it to *price* a pull against recomputing, and
  the run installs it in the directory so the fetch is *served* by the peer that
  was priced;
* :mod:`~kvcache_sim.control._view` -- the single derived directory read the
  scheduler needs (per-instance prefix-run lengths, and the private prefix walk
  behind them), plus the pinned snapshot one decision reads it through;
* :mod:`~kvcache_sim.control._cache` -- per-instance LRU. It picks victims; it
  does not delete anything.

Nothing here imports :mod:`kvcache_sim.data`, a deployment or a client, and
nothing here reaches into the simulator -- all checked by the repo's contract lint. Control senses through a view and returns
decisions; what actually happened comes back as an observation.
"""
