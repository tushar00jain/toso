"""Drive the real ``LocalClient`` planning core via the seams.

Wiring:

- A minimal ``FakeStrategy`` provides the ``select_storage_volume`` /
  ``get_storage_volume`` surface the real client uses, returning real
  ``StorageVolumeRef`` objects whose ``.volume`` is a
  :class:`~realsim.seams.volume_handle.LocalVolumeHandle`. We do not construct a
  real ``TorchStoreStrategy`` subclass because its ``set_storage_volumes`` needs
  a live Monarch mesh (``storage_volumes.get_id.call()``); the client only ever
  calls ``select_storage_volume`` / ``get_storage_volume`` / ``transport_context``
  off it.
- ``create_transport_buffer`` is substituted so the client's planning core drives
  :class:`~realsim.seams.transport.InMemoryTransport`. The real client imports the
  factory as a module global, so the substitution is process-wide; it goes through
  :mod:`realsim.seams.factory`, which owns the patch and permits only one owner at
  a time. ``installed()`` below is therefore the **single-client** drive: it pins
  the source endpoint to this adapter's own node. A multi-client drive needs one
  factory shared across clients -- use :class:`realsim.mesh.Mesh`.

The real ``LocalClient`` planning core (``_build_volume_requests``,
``_expand_tensor_slices``, ``_fetch``, ``_assemble_results``, ``_apply_inplace``)
is exactly what executes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from realsim.seams import factory
from realsim.seams.transport import Endpoint, InMemoryTransport
from realsim.seams.volume_handle import LocalVolumeHandle
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile
from sim_common.resources import ResourceRegistry
from sim_common.trace import Trace
from torchstore.client import LocalClient
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext

__all__ = ["FakeStrategy", "RealClientAdapter"]


class FakeStrategy:
    """Minimal stand-in for ``TorchStoreStrategy`` (only the surface the client uses).

    Args:
        client_volume_id: the volume id this client is co-located with (its
            ``select_storage_volume`` target).
        volume_handles: mapping of ``volume_id -> LocalVolumeHandle``.
    """

    def __init__(
        self,
        client_volume_id: str,
        volume_handles: dict[str, LocalVolumeHandle],
    ) -> None:
        self._client_volume_id = client_volume_id
        self._volume_handles = volume_handles
        # Real TransportContext -- LocalClient.delete uses it; harmless otherwise.
        self.transport_context = TransportContext()

    def select_storage_volume(self) -> StorageVolumeRef:
        return self.get_storage_volume(self._client_volume_id)

    def get_storage_volume(self, volume_id: str) -> StorageVolumeRef:
        return StorageVolumeRef(
            self._volume_handles[volume_id],
            volume_id,
            self.transport_context,
            TransportType.Unset,
        )


class RealClientAdapter:
    """Wires a real ``LocalClient`` to the seams.

    Args:
        controller_handle: a :class:`LocalControllerHandle`.
        volume_handles: mapping ``volume_id -> LocalVolumeHandle``.
        client_volume_id: the volume this client is co-located with.
        topology: mapping ``volume_id -> Endpoint`` for transfer-cost locality.
        profile: optional target-machine :class:`~sim_common.cost_model.MachineProfile`
            supplying the cost constants (defaults to
            :data:`~sim_common.cost_model.DEFAULT_PROFILE`).
        trace: optional :class:`sim_common.trace.Trace` for transfer events.
        registry: optional shared :class:`~sim_common.resources.ResourceRegistry`
            for network/storage contention (``None`` -> independent sleeps, the
            historical behavior). Passed to every transport this adapter builds.
    """

    def __init__(
        self,
        controller_handle,
        *,
        volume_handles: dict[str, LocalVolumeHandle],
        client_volume_id: str,
        topology: dict[str, Endpoint],
        profile: MachineProfile | None = None,
        trace: Trace | None = None,
        registry: ResourceRegistry | None = None,
    ) -> None:
        self.strategy = FakeStrategy(client_volume_id, volume_handles)
        self.client = LocalClient(controller_handle, self.strategy)
        self._topology = topology
        # The client is co-located with its own volume, so its endpoint is that
        # volume's endpoint (reading from its own volume is a zero-cost transfer).
        self._client_endpoint = topology[client_volume_id]
        self._profile = profile if profile is not None else DEFAULT_PROFILE
        self._trace = trace
        self._registry = registry

    def _transport_factory(self):
        def factory(storage_volume_ref: StorageVolumeRef) -> InMemoryTransport:
            return InMemoryTransport(
                storage_volume_ref,
                src=self._client_endpoint,
                dst=self._topology[storage_volume_ref.volume_id],
                profile=self._profile,
                trace=self._trace,
                registry=self._registry,
            )

        return factory

    @contextmanager
    def installed(self) -> Iterator["RealClientAdapter"]:
        """Substitute ``create_transport_buffer`` for the duration of the block.

        The substitution is a process-wide monkeypatch (see
        :mod:`realsim.seams.factory`), and this one pins the source endpoint to
        *this* adapter's node -- so it drives a single client. Multi-client drives
        must either scope each client's operations in its own ``installed()``
        block or, better, share one :class:`realsim.mesh.Mesh` factory that
        resolves the source per operation. Overlapping installs raise (see
        ``docs/des_design.md`` under fidelity boundaries).
        """
        with factory.installed(self._transport_factory(), owner=self):
            yield self
