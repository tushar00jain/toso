"""Adapters: thin wiring that constructs the real TorchStore objects off-actor
and injects the seams so the real planning + directory logic executes.

- ``real_controller.RealControllerAdapter`` -- constructs a real ``Controller``
  off-actor and exposes it via a :class:`~realsim.seams.controller_handle.FakeControllerHandle`.
- ``real_controller.RealControllerAdapter(shim=True)`` -- same, but swaps the ``Controller``'s
  ``Trie`` directory for a lightweight dict shim; ``make_controller_adapter``
  selects between the two from the ambient ``real_directory`` config flag.
- ``real_client.RealClientAdapter`` -- constructs a real ``LocalClient`` with a
  fake strategy + fake volume handles and substitutes ``create_transport_buffer``
  so the real client planning core drives the in-memory transport.
"""
