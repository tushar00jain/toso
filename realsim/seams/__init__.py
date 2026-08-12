"""Seams that let the real TorchStore code run off-actor.

Each seam substitutes exactly one Monarch touchpoint the real code reaches for,
with an in-process stand-in that dispatches back into real TorchStore logic:

- ``transport.InMemoryTransport`` -- subclass of the real
  ``MonarchRPCTransportBuffer``; the real put/get lifecycle runs, backed by a
  real ``InMemoryStore``, plus a virtual-clock transfer cost.
- ``controller_service.ControllerService`` -- the directory server: the real
  ``Controller``, the policy installed in it, and the endpoint bodies
- ``volume_service.VolumeService`` -- the storage server: a real ``InMemoryStore``
  behind the real ``StorageVolume`` endpoint bodies, plus this volume's residency
  and the capacity rule that asks control what to drop
- ``coordinator_service.CoordinatorService`` -- the same for a coordinator: it
  holds a capability's control plane (a ``proposed.Coordinator``) and answers the
  caller's surface, ``proposed.deployment.Coordinator``
- ``controller_handle.LocalControllerHandle`` -- what a caller holds for that
  service: one endpoint per member (``locate_volumes`` / ``notify_put_batch`` /
  ``keys``), reached through ``call_one`` / ``call`` / ``broadcast``.
- ``volume_handle.LocalVolumeHandle`` -- the same for a storage volume
  (``put`` / ``get`` / ``handshake`` / ``delete`` / ``delete_batch`` / ``reset``).
- ``coordinator_handle.LocalCoordinatorHandle`` -- the same for a coordinator
  service (``decide`` / ``observe``), and the one of the three whose hop a run
  gives a duration.

Each pair is a server and a reference to it, split because they are different
shapes: a service has methods, a reference has endpoints, and in a deployment the
first becomes an actor while the second becomes Monarch's own handle. All three
services in a deployment are here, and each one's surface is declared in
``proposed`` (``Controller``, ``StorageVolume``, ``Coordinator``).
- ``dict_directory.DictDirectory`` -- a plain-``dict`` stand-in for the
  controller's ``Trie`` directory, presenting the same ``Mapping`` +
  ``keys().filter_by_prefix`` surface so the opt-in shim adapter can skip the
  per-key trie tax while the real ``Controller`` decision logic runs unchanged.
- ``factory`` -- the single substitution point for the process-wide
  ``create_transport_buffer`` global, plus the contextvar holding the calling
  client's source endpoint. Every install in the repo goes through it, and only
  one owner may hold the patch at a time.
"""
