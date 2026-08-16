"""The KV-cache control plane: everything that is decided, nothing that runs.

* :mod:`~kvcache_sim.control.request` -- what is being decided *about*: an inference
  ``Request``, carrying the prompt it was submitted with and its prefix-hash block
  keys. Here rather than in ``workload/`` because all three planes pass it;
* :mod:`~kvcache_sim.control.scheduler` -- the serving scheduler: which instance
  prefills, pull-vs-recompute, the TTFT/TBT admission gates, where decode lands. These
  are *compute* decisions the store knows nothing about, so what they rank is this
  application's own candidates and not keys;
* :mod:`~kvcache_sim.control._answer` -- what those decisions *are*, as values: the
  ``Plan`` one candidate was priced at and the ``Response`` naming both of a request's
  hosts. The layer under everything else here;
* :mod:`~kvcache_sim.control._selector` -- every ranking those decisions make: which
  peer serves a prefix gap (the one store question, so the only ranking here over
  keys), which host decodes, and the pull a fetch was already answered with. What is
  not a ranking is not here -- an SLO gate
  answers yes or no, a cost is arithmetic, and which candidate prefills is the
  scheduler's own fold over the pool it keyed;
* :mod:`~kvcache_sim.control._sensor` -- what those decisions are made against: one
  sensor per kind of fact this plane holds -- the cluster's load, the prefills it
  promised, the pulls it priced -- each expiring when the thing it stood for happens.
  Beside them the actions that move them, since every write here is one dispatched
  action;
* :mod:`~kvcache_sim.control._view` -- what a decision senses, one class per read:
  prefix-run lengths (with the prefix walk behind them and the pinned snapshot a
  decision reads them through) and each sensor above.

Nothing here imports :mod:`kvcache_sim.data`, a deployment or a client, and nothing
reaches into the simulator -- all checked by the repo's contract lint.
"""
