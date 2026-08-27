"""Seams that let the real TorchStore code run off-actor.

Each seam substitutes exactly one Monarch touchpoint the real code reaches for,
with an in-process stand-in that dispatches back into real TorchStore logic:

- ``transport.InMemoryTransport`` -- subclass of the real
  ``MonarchRPCTransportBuffer``; the real put/get lifecycle runs, backed by a
  real ``InMemoryStore``, plus a virtual-clock transfer cost.
- ``controller_service.ControllerService`` -- the directory server: the real
  ``Controller``, the endpoint bodies, and the one thing it does that torchstore's
  does not -- apply the source preference its caller handed it. Nothing decides in
  there
- ``volume_service.VolumeService`` -- the storage server: a real ``InMemoryStore``
  behind the real ``StorageVolume`` endpoint bodies, plus this volume's residency
  and the capacity rule that asks control what to drop
- ``control_plane_service.ControlPlaneService`` -- the same for the control plane an
  application's own hosts ask: it holds it and forwards whichever members the plane
  itself declares, since what a capability answers is not ``proposed``'s to name
- ``dispatcher_service.DispatcherService`` -- the same for where that control plane's
  facts arrive: it holds the application's ``proposed.Dispatcher`` -- which folds one
  action into every sensor it moves -- and forwards what its hosts report
- ``controller_handle.LocalControllerHandle`` -- what a caller holds for that
  service: one endpoint per member (``locate_volumes`` / ``notify_put_batch`` /
  ``keys``), reached through ``call_one`` / ``call`` / ``broadcast``.
- ``volume_handle.LocalVolumeHandle`` -- the same for a storage volume
  (``put`` / ``get`` / ``handshake`` / ``delete`` / ``delete_batch`` / ``reset``).
- ``control_plane_handle.LocalControlPlaneHandle`` -- the same for a control-plane
  service, one endpoint per member it forwards, whose hop a run gives a duration.
- ``dispatcher_handle.LocalDispatcherHandle`` -- the same for a dispatcher
  (``dispatch``), at that same distance: it is held by the control plane whose sensors
  it folds into.
- ``option_b_service.OptionBService`` and
  ``option_b_handle.LocalOptionBServiceHandle`` -- local server/handle stand-ins
  for the production Option B actor. ``realsim.mesh.LocalActorMesh`` provides its
  mesh-wide endpoint broadcasts.

Each pair is a server and a reference to it, split because they are different
shapes: a service has methods, a reference has endpoints, and in a deployment the
first becomes an actor while the second becomes Monarch's own handle. All four
services a deployment runs are here, and each one's surface is declared in
``proposed`` (``Controller``, ``StorageVolume``, ``ControlPlane``, ``Dispatcher``).
- ``dict_directory.DictDirectory`` -- a plain-``dict`` stand-in for the
  controller's ``Trie`` directory, presenting the same ``Mapping`` +
  ``keys().filter_by_prefix`` surface so the opt-in shim adapter can skip the
  per-key trie tax while the real ``Controller`` decision logic runs unchanged.
- ``factory`` -- the single substitution point for the process-wide
  ``create_transport_buffer`` global, plus the contextvar holding the calling
  client's source endpoint. Every install in the repo goes through it, and only
  one owner may hold the patch at a time.
"""
