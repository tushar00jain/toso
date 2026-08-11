"""The KV-cache data plane: everything that advances the clock or moves bytes.

* :mod:`~kvcache_sim.data.serving` -- the per-request serving loop, a
  :class:`proposed.plane.DataPlane`: queue wait, the real prefix pull, the prefill
  charge, the real publish/evict, decode admission, and the outcome rows;
* :mod:`~kvcache_sim.data._decode` -- the batched decode engine. It sleeps and
  emits tokens, so it is data even though the scheduler *reads* its occupancy.
  Underscored because nothing outside this package drives it: the serving plane
  constructs it and owns it;
* :mod:`~kvcache_sim.data.store` -- the three KV verbs (publish / fetch / evict)
  as real ``put_batch`` / ``get_batch`` / ``notify_delete_batch`` calls, plus
  :meth:`~kvcache_sim.data.store.KVStore.for_deployment`, the factory a run (real
  or simulated) builds one with.

The test for what belongs here: does it advance the clock or move bytes? A
directory *read* does neither, so it is a control-plane view, not a verb here.
"""
