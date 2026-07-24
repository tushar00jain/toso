"""In-memory transport seam over the *real* TorchStore transport lifecycle.

``InMemoryTransport`` subclasses the real ``MonarchRPCTransportBuffer`` so the
whole real put/get lifecycle executes unchanged -- ``put_to_storage_volume`` /
``get_from_storage_volume`` (base ``TransportBuffer``), ``_pre_put_hook`` /
``_pre_get_hook`` / ``handle_put_request`` / ``handle_get_request`` /
``_handle_storage_volume_response`` / ``drop`` (MonarchRPC). The only things we
add are:

1. a real ``InMemoryStore`` behind the fake volume handle (see
   :mod:`realsim.seams.volume_handle`), and
2. **virtual-clock resource costs** -- every put/get charges the full analytic
   resource model from :mod:`sim_common.cost_model` (network fabric, persistent
   storage, and host-RAM staging), each applied as an ``asyncio.sleep`` against
   the running loop's clock -- virtual (free) under the deterministic engine,
   a tiny real sleep under a plain asyncio loop. All constants come from a
   caller-supplied :class:`~sim_common.cost_model.MachineProfile` (the *target*
   machine), never measured on the test box.

The put path charges ``network`` (client->volume fabric) + ``storage write``
(the payload landing in the volume's store). The get path charges ``storage
read`` + ``mem_copy`` (the serving volume reads the payload and stages it through
host RAM) + ``network`` (volume->client fabric). The producer-side ``compute``
cost of generating the payload is charged by the scenario (see
:mod:`realsim.scenarios.burst_get`), which knows the flop model.

We deliberately reuse ``MonarchRPCTransportBuffer`` (not RDMA/gloo/shm) because
it is the transport that already round-trips data as plain in-process Python
references, which is exactly what an in-memory sim needs. No parallel transport
is hand-rolled.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import torch

# ``Endpoint`` is the shared minimal locality identity (id/host/node) used by the
# cost model; it lives in ``sim_common.topology`` so ``realsim`` and ``dedup_sim``
# reduce to one shape instead of each declaring the trio. Re-exported here so the
# seam import path (``from realsim.seams.transport import Endpoint``) keeps
# working.
from sim_common.cost_model import (
    DEFAULT_PROFILE,
    MachineProfile,
    mem_copy_time,
    network_time,
    storage_time,
)
from sim_common.topology import Endpoint
from sim_common.trace import Trace
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer

__all__ = ["Endpoint", "InMemoryTransport", "TensorDescriptor"]


@dataclass(frozen=True)
class TensorDescriptor:
    """Out-of-band ``(shape, dtype)`` metadata for the metadata-only data plane.

    The metadata-only mode carries *no* tensor at all -- not even a zero-storage
    meta tensor. Instead this tiny descriptor stands in for the payload: it is
    passed to ``client.put(key, descriptor)`` as an arbitrary object (so it flows
    through the real object put/get path in ``InMemoryStore`` -- see
    :mod:`realsim.scenarios.burst_get` for why this sidesteps the ``put_batch``
    value-typing gotcha) and :func:`_nbytes` reads the modeled byte count off it.

    It exposes the same size surface as a ``torch.Tensor``
    (``numel``/``element_size``/``nbytes``/``shape``/``dtype``) so callers such as
    ``render_burst_summary`` and the cost model can treat a descriptor and a meta
    tensor uniformly. Constructing it allocates nothing (element size is derived
    from a zero-storage meta tensor).
    """

    shape: tuple[int, ...]
    dtype: torch.dtype

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def element_size(self) -> int:
        # A 0-element meta tensor allocates no storage; element_size() is a pure
        # property of the dtype, so this stays allocation-free.
        return torch.empty(0, dtype=self.dtype, device="meta").element_size()

    @property
    def nbytes(self) -> int:
        return self.numel() * self.element_size()

def _nbytes(value: Any) -> int:
    """Modeled byte count of a payload.

    Handles all three data-plane carriers uniformly:

    * a real or **meta** ``torch.Tensor`` -- ``numel() * element_size()`` (a meta
      tensor has zero storage but exact shape/dtype, so this is correct);
    * a :class:`TensorDescriptor` -- the metadata-only carrier, size read off the
      ``(shape, dtype)`` descriptor even though ``tensor_val is None``;
    * anything else (a genuine object payload / ``None``) -- ``0``.
    """
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, TensorDescriptor):
        return value.nbytes
    return 0


def _request_nbytes(request: Any) -> int:
    """Modeled byte count for a put :class:`~torchstore.transport.types.Request`.

    A tensor put carries the payload in ``tensor_val``; the metadata-only put
    carries a :class:`TensorDescriptor` in ``objects`` (``tensor_val is None``),
    so we fall back to the object payload when there is no tensor.
    """
    if request.tensor_val is not None:
        return _nbytes(request.tensor_val)
    return _nbytes(request.objects)


class InMemoryTransport(MonarchRPCTransportBuffer):
    """Real MonarchRPC transport lifecycle + virtual-clock resource costs.

    A put charges ``network`` (client->volume fabric) then ``storage write`` (the
    payload landing in the volume's store). A get charges ``storage read`` +
    ``mem_copy`` (the serving volume reads the payload and stages it through host
    RAM) then ``network`` (volume->client fabric). Every charge is a deterministic
    function of the modeled byte count and the caller-supplied
    :class:`~sim_common.cost_model.MachineProfile`, applied as an ``asyncio.sleep``
    against the loop's (virtual) clock and recorded into the trace.

    Args:
        storage_volume_ref: the real ``StorageVolumeRef`` (its ``.volume`` is a
            :class:`realsim.seams.volume_handle.FakeVolumeHandle`).
        src: the transferring client's endpoint.
        dst: the target volume's endpoint.
        profile: the target-machine :class:`~sim_common.cost_model.MachineProfile`
            supplying every cost constant (defaults to
            :data:`~sim_common.cost_model.DEFAULT_PROFILE`). Comes from the
            scenario/config, never hard-coded in the seam.
        trace: optional :class:`sim_common.trace.Trace` to record charges into.
        on_transfer: optional callback invoked with
            ``(kind, src_id, dst_id, nbytes, cost)`` after each charged *network*
            transfer, for structured fabric-byte accounting (see
            :class:`realsim.coordinator.model` metrics) without parsing the trace.
    """

    def __init__(
        self,
        storage_volume_ref,
        *,
        src: Endpoint,
        dst: Endpoint,
        profile: MachineProfile | None = None,
        trace: Trace | None = None,
        on_transfer: Optional[Any] = None,
    ) -> None:
        super().__init__(storage_volume_ref)
        self._src = src
        self._dst = dst
        self._profile = profile if profile is not None else DEFAULT_PROFILE
        self._trace = trace
        self._on_transfer = on_transfer

    async def put_to_storage_volume(self, requests) -> None:
        # Real base-class put lifecycle runs first (data actually lands in the
        # real InMemoryStore), then charge the resource costs of the put: the
        # client->volume fabric transfer, then writing the payload to the
        # volume's persistent store.
        nbytes = sum(_request_nbytes(r) for r in requests)
        await super().put_to_storage_volume(requests)
        await self._charge_network(self._src, self._dst, nbytes, "put")
        await self._charge_storage(nbytes, "write")

    async def get_from_storage_volume(self, requests) -> list[Any]:
        # Real base-class get lifecycle runs first, then charge the resource
        # costs of serving the get: the volume reads the payload back from its
        # store and stages it through host RAM, then ships it over the fabric.
        results = await super().get_from_storage_volume(requests)
        nbytes = sum(_nbytes(r) for r in results)
        await self._charge_storage(nbytes, "read")
        await self._charge_mem(nbytes)
        await self._charge_network(self._dst, self._src, nbytes, "get")
        return results

    async def _sleep(self, dt: float) -> float:
        """Advance the (virtual) clock by ``dt`` and return the loop time after.

        Scheduled against the running loop's clock: virtual under the deterministic
        engine (zero wall time), a tiny real sleep under a plain asyncio loop.
        """
        await asyncio.sleep(dt)
        return asyncio.get_running_loop().time()

    async def _charge_network(
        self, src: Endpoint, dst: Endpoint, nbytes: int, kind: str
    ) -> None:
        """Charge (and trace) the fabric cost of moving ``nbytes`` src->dst."""
        dt = network_time(src, dst, nbytes, self._profile)
        now = await self._sleep(dt)
        if self._trace is not None:
            self._trace.record(
                now, "xfer", f"{kind} {src.id}->{dst.id} {nbytes}B cost={dt:.4f}"
            )
        if self._on_transfer is not None:
            self._on_transfer(kind, src.id, dst.id, nbytes, dt)

    async def _charge_storage(self, nbytes: int, kind: str) -> None:
        """Charge (and trace) a persistent-storage read/write of ``nbytes``."""
        dt = storage_time(nbytes, kind, self._profile)
        now = await self._sleep(dt)
        if self._trace is not None and dt > 0:
            self._trace.record(
                now, "store", f"{kind} {self._dst.id} {nbytes}B cost={dt:.4f}"
            )

    async def _charge_mem(self, nbytes: int) -> None:
        """Charge (and trace) a host-RAM staging copy of ``nbytes``."""
        dt = mem_copy_time(nbytes, self._profile)
        now = await self._sleep(dt)
        if self._trace is not None and dt > 0:
            self._trace.record(
                now, "mem", f"copy {self._dst.id} {nbytes}B cost={dt:.4f}"
            )
