"""Seams that let the real TorchStore code run off-actor.

Each seam substitutes exactly one Monarch touchpoint the real code reaches for,
with an in-process stand-in that dispatches back into real TorchStore logic:

- ``transport.InMemoryTransport`` -- subclass of the real
  ``MonarchRPCTransportBuffer``; the real put/get lifecycle runs, backed by a
  real ``InMemoryStore``, plus a virtual-clock transfer cost.
- ``volume_handle.FakeVolumeHandle`` -- mimics the ``.put.call`` /
  ``.get.call_one`` / ``.handshake.call_one`` actor surface, delegating to a
  real ``InMemoryStore`` exactly like the real ``StorageVolume`` endpoints.
- ``controller_service.ControllerService`` -- the directory server: the real
  ``Controller``, the policy installed in it, and the endpoint bodies
- ``coordinator_service.CoordinatorService`` -- the same for a coordinator: it
  holds a capability's control plane (a ``proposed.Coordinator``) and answers the
  caller's surface, ``proposed.deployment.Coordinator``
- ``controller_handle.LocalControllerHandle`` -- what a caller holds for that
  service: one endpoint per member (``locate_volumes`` / ``notify_put_batch`` /
  ``keys``), reached through ``call_one`` / ``call`` / ``broadcast``.
- ``coordinator_handle.CoordinatorHandle`` -- the same for a coordinator service.

Each pair is a server and a reference to it, split because they are different
shapes: a service has methods, a reference has endpoints, and in a deployment the
first becomes an actor while the second becomes Monarch's own handle.
- ``dict_directory.DictDirectory`` -- a plain-``dict`` stand-in for the
  controller's ``Trie`` directory, presenting the same ``Mapping`` +
  ``keys().filter_by_prefix`` surface so the opt-in shim adapter can skip the
  per-key trie tax while the real ``Controller`` decision logic runs unchanged.
- ``factory`` -- the single substitution point for the process-wide
  ``create_transport_buffer`` global, plus the contextvar holding the calling
  client's source endpoint. Every install in the repo goes through it, and only
  one owner may hold the patch at a time.
"""
