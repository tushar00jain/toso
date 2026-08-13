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

* :mod:`realsim.run` -- the whole run lifecycle in one module, one role per type:
  :class:`~realsim.run.Workload` (the work, and nothing else),
  :class:`~realsim.run.Run` (a labelled configuration -- the workload plus the
  selector and plane a capability installs around it, and which knows how to
  :meth:`~realsim.run.Run.execute` itself),
  :class:`~realsim.run.Result` (one type, for every sim) and
  :class:`~realsim.run.Report` (a finished run, as text). Every run in the repo
  goes through ``Run.execute``, so no capability wires a stack of its own;
* :mod:`realsim.demo` -- :class:`~realsim.demo.Demo`, a sim's command line
  declared rather than hand-rolled, plus the run flags every one of them shares;
* :mod:`realsim.seams` + :mod:`realsim.adapters` -- run one real client /
  controller / store off-actor;
* :class:`realsim.mesh.Mesh` -- the multi-client wiring: real volumes, one real
  ``LocalClient`` per node, one directory, one resource registry, and the single
  shared ``create_transport_buffer`` substitution. Build capabilities on this
  rather than re-deriving the wiring;
* the four types every capability plugs into:
  :class:`proposed.selector.KeySelector` (which volume serves these keys for this
  requester, and when -- naive by default, and consulted inside the real
  controller's ``locate_volumes``), :class:`proposed.view.View` (the read-only
  observation a selector is handed), :class:`proposed.plane.DataPlane` (what a
  capability does after a transfer lands) and :class:`realsim.runner.Runner`
  (release work items on the virtual clock, install the mesh once, gather) with
  :class:`realsim.runner.ItemDispatch` (how one run is driven).

``realsim``'s own tests drive :mod:`putget_sim.workload.put_get` -- the
capability-free fixture (seed a key, then ``m`` clients get it) -- so they
exercise the whole stack without depending on ``dedup_sim`` or ``kvcache_sim``.

See ``docs/realsim_design.md`` for the full design, including exactly how each
real object is driven off-actor.
"""
