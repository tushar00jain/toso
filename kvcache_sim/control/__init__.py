"""The KV-cache control plane: everything that is decided, nothing that runs.

* :mod:`~kvcache_sim.control.request` -- what is being decided *about*: an
  inference ``Request``, carrying the prompt it was submitted with and its
  prefix-hash block keys (each a plain directory key). It sits here rather than in
  ``workload/`` because all three planes pass it;
* :mod:`~kvcache_sim.control.scheduler` -- the serving scheduler: which instance
  prefills, pull-vs-recompute, the TTFT/TBT admission gates, where decode lands.
  These are *compute* decisions the store knows nothing about, which is why they
  are app code and not part of the shared selector interface;
* :mod:`~kvcache_sim.control._cluster` -- what those decisions are made against: one
  model of the cluster's load per run, behind :class:`proposed.ClusterModel`'s single
  write verb. The facts a host reports are here with the fold that applies them, and
  so are the reads everything that ranks or prices against that load makes;
* :mod:`~kvcache_sim.control._pending` -- what was decided and not yet carried out,
  each record expiring on its own terms as it is read;
* :mod:`~kvcache_sim.control._source` -- the one part that *is* a store question,
  "which peer serves this prefix gap", as a :class:`proposed.selector.KeySelector`. It is
  used twice: the scheduler calls it to *price* a pull against recomputing, and it
  sits behind the plane a fetch asks, so the read is *served* by the peer that was
  priced. Spreading reads over the replicas of a hot prefix is that ranking under
  :class:`proposed.selector.Discount`, which is a composition and so lives there;
* :mod:`~kvcache_sim.control._view` -- everything a decision senses, as one object:
  the single derived directory read the scheduler needs (per-instance prefix-run
  lengths, and the private prefix walk behind them), the pinned snapshot one decision
  reads it through, and the cluster model above -- so what ranks, prices or gates
  senses it rather than being handed it.

Nothing here imports :mod:`kvcache_sim.data`, a deployment or a client, and nothing
reaches into the simulator -- all checked by the repo's contract lint. Control
senses through a view and answers questions; what actually happened arrives as a
fact its hosts report into the model.
"""
