"""realsim -- a real-code cooperative DES over the real TorchStore.

``realsim`` drives the **real** TorchStore client planning core and controller
directory logic off-actor -- on a plain asyncio loop or the deterministic
virtual-clock engine (``sim_common.async_engine``) -- with an in-memory transport.
It therefore depends on the real ``torchstore`` / ``torch`` / ``monarch`` install
in the venv; that is the point. ``putget_sim``, ``dedup_sim`` and ``kvcache_sim``
build on it.

It is the foundation only: it has no demo and no scenario of its own. The
unrouted put/get burst that used to live here is ``putget_sim``, a sim package
shaped like the other two.

A capability package should hold only capability code; everything generic it needs
is here:

* the four types a sim is built from, one role each:
  :class:`realsim.workload.Workload` (the work, and nothing else),
  :class:`realsim.run.Run` (a labelled configuration -- the workload plus the
  policy and plane a capability installs around it),
  :class:`realsim.reporting.Report` (a finished run, as text) and
  :class:`realsim.demo.Demo` (a sim's command line, declared). Every run in the
  repo goes through :func:`realsim.run.execute`, so no capability wires a stack
  of its own;
* :mod:`realsim.seams` + :mod:`realsim.adapters` -- run one real client /
  controller / store off-actor;
* :class:`realsim.mesh.Mesh` -- the multi-client wiring: real volumes, one real
  ``LocalClient`` per node, one directory, one resource registry, and the single
  shared ``create_transport_buffer`` substitution. Build capabilities on this
  rather than re-deriving the wiring;
* the four types every capability plugs into:
  :class:`proposed.policy.Policy` (which volume serves these keys for this
  requester, and when -- naive by default, and consulted inside the real
  controller's ``locate_volumes``), :class:`proposed.view.View` (the read-only
  observation a policy is handed), :class:`proposed.plane.DataPlane` (the work
  around and after a transfer, both methods defaulting to real no-op behaviour)
  and :class:`realsim.runner.Runner` (release work items on the virtual clock,
  install the mesh once, drain).

``realsim``'s own tests drive :mod:`putget_sim.workload.put_get` -- the
capability-free fixture (seed a key, then ``m`` clients get it) -- so they
exercise the whole stack without depending on ``dedup_sim`` or ``kvcache_sim``.

See ``docs/realsim_design.md`` for the full design, including exactly how each
real object is driven off-actor.
"""
