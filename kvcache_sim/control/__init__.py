"""The KV-cache control plane: everything that is decided, nothing that runs.

* :mod:`~kvcache_sim.control.request` -- what is being decided *about*: an
  inference ``Request``, carrying the prompt it was submitted with and its
  prefix-hash block keys (each a plain directory key). It sits here rather than in
  ``workload/`` because all three planes pass it;
* :mod:`~kvcache_sim.control.scheduler` -- the serving scheduler: which instance
  prefills, pull-vs-recompute, the TTFT/TBT admission gates, where decode lands.
  These are *compute* decisions the store knows nothing about, which is why what they
  rank is this application's own candidates and not keys;
* :mod:`~kvcache_sim.control._answer` -- what those decisions *are*, as values: the
  ``Plan`` one candidate was priced at and the ``Response`` naming both of a request's
  hosts. The layer under everything else here, which is what lets a ranking be typed on
  a plan without a cycle back into the plane that builds one;
* :mod:`~kvcache_sim.control._selector` -- every ranking those decisions make, each a
  :class:`proposed.selector.Selector`: which peer serves a prefix gap (the one part that
  *is* a store question, so the only ranking here over keys), which priced candidate
  prefills, which host decodes, and the pull a fetch was already answered with. What is
  not a ranking is not here -- an SLO gate answers yes or no, and a cost is arithmetic;
* :mod:`~kvcache_sim.control._sensor` -- what those decisions are made against: one
  sensor per kind of fact this plane holds -- the cluster's load, and what this plane
  decided and has not yet seen carried out (the prefills it promised, the pulls it
  priced), each expiring on its own terms as it is read. Beside them, the actions that
  move them and the fold each sensor publishes, since every write here is one action
  dispatched into this plane's :class:`proposed.Dispatcher`;
* :mod:`~kvcache_sim.control._view` -- what a decision senses, as one class per read
  composed onto :class:`proposed.View`: per-instance prefix-run lengths (with the
  private prefix walk behind them and the pinned snapshot one decision reads them
  through) and each of the three sensors above -- so what ranks, prices or gates reads
  the view it needs rather than being handed a sensor.

Nothing here imports :mod:`kvcache_sim.data`, a deployment or a client, and nothing
reaches into the simulator -- all checked by the repo's contract lint. Control
senses through a view and answers questions; what actually happened arrives as an
action its hosts dispatch.
"""
