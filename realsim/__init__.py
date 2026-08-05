"""realsim -- a real-code cooperative DES over the real TorchStore.

``realsim`` drives the **real** TorchStore client planning core and controller
directory logic off-actor -- on a plain asyncio loop or the deterministic
virtual-clock engine (``sim_common.async_engine``) -- with an in-memory transport.
It therefore depends on the real ``torchstore`` / ``torch`` / ``monarch`` install
in the venv; that is the point. ``dedup_sim`` and ``kvcache_sim`` build on it.

A capability package should hold only capability code; everything generic it needs
is here:

* :mod:`realsim.seams` + :mod:`realsim.adapters` -- run one real client /
  controller / store off-actor;
* :class:`realsim.mesh.Mesh` -- the multi-client wiring: real volumes, one real
  ``LocalClient`` per node, one directory, one resource registry, and the single
  shared ``create_transport_buffer`` substitution. Build capabilities on this
  rather than re-deriving the wiring;
* :class:`realsim.coordinator.model.ReadCoordinator` -- a *burst*-shaped consumer
  of a mesh, with the pluggable ``ReadPolicy`` seam ``dedup_sim`` implements.

See ``docs/realsim_design.md`` for the full design, including exactly how each
real object is driven off-actor.
"""
