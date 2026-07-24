"""Seams that let the real TorchStore code run off-actor.

Each seam substitutes exactly one Monarch touchpoint the real code reaches for,
with an in-process stand-in that dispatches back into real TorchStore logic:

- ``transport.InMemoryTransport`` -- subclass of the real
  ``MonarchRPCTransportBuffer``; the real put/get lifecycle runs, backed by a
  real ``InMemoryStore``, plus a virtual-clock transfer cost.
- ``volume_handle.FakeVolumeHandle`` -- mimics the ``.put.call`` /
  ``.get.call_one`` / ``.handshake.call_one`` actor surface, delegating to a
  real ``InMemoryStore`` exactly like the real ``StorageVolume`` endpoints.
- ``controller_handle.FakeControllerHandle`` -- mimics the controller actor
  surface (``locate_volumes`` / ``notify_put_batch`` / ``keys``), dispatching to
  a real ``Controller`` instance's sync logic.
"""
