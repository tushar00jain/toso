"""The unrouted put/get burst: ``realsim``'s own fixture, as a runnable sim.

``putget_sim`` seeds one key on an origin volume and has ``m`` clients get it,
over the **real** ``LocalClient`` planning core, the **real** ``Controller``
directory and the **real** ``InMemoryStore`` (via ``realsim``). It installs no
policy and no data plane, so every reader locates the origin and pulls from it:
fabric is ``m x`` the payload. That is the baseline ``dedup_sim`` measures its
1x against.

Laid out by role, like ``dedup_sim`` and ``kvcache_sim``:

* :mod:`putget_sim.workload` -- the burst itself: a topology, an allocation-free
  payload, a ``client.put`` and a gather of ``client.get``. Ordinary user code,
  with no policy and no coordinator in it;
* :mod:`putget_sim.report` -- the fabric/wallclock summary and source->dest tree.

There is no ``control/`` and no ``data/``: this capability decides nothing and
executes nothing of its own -- that is the point. Handing the same workload a
:class:`~proposed.policy.Policy` and a :class:`~proposed.plane.DataPlane` is the
only change needed to turn it into a routed run, which is exactly what
``dedup_sim`` does with it. Both sims therefore share this package's
:class:`~putget_sim.workload.put_get.PutGetBurst`, so their comparison is
byte-for-byte the same topology, payload and cost model.

See ``putget_sim/README.md`` for the folder-by-folder comparison.
"""
