"""Production Option B service for direct volume I/O and readiness signaling."""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, Tuple

from monarch.actor import Actor, ProcMesh  # type: ignore[import-untyped]
from proposed import endpoint
from torchstore import TorchStoreStrategy
from torchstore.transport import create_transport_buffer
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import spawn_actors

from ._model import SliceGeometry, Transfer, slice_geometry

__all__ = ["OptionBService"]


_TransportFactory = Callable[[str], TransportBuffer]


class _OptionBServiceBase(Actor):
    """The endpoint contract used by :class:`OptionBClient`.

    A service performs direct volume I/O for one rank and coordinates relay
    readiness with its peers. Production and simulation provide the same
    endpoint-shaped interface.
    """

    @endpoint
    async def put(
        self,
        key: str,
        tensor_slices: Tuple[TensorSlice, ...],
        value: object,
    ) -> None:
        """Put one value directly into this rank's storage volume."""
        ...

    @endpoint
    async def get(self, transfer: Transfer, destination: object) -> object:
        """Read one precomputed transfer directly from its source volume."""
        ...

    @endpoint
    async def wait_ready(
        self, source: str, key: str, tensor_slice: TensorSlice
    ) -> None:
        """Wait until a relay source reports that its slice is available."""
        ...

    @endpoint
    async def notify_ready(
        self,
        source: str,
        peers: Tuple[str, ...],
        key: str,
        tensor_slice: TensorSlice,
    ) -> None:
        """Record a relay source as ready on the addressed peer ranks."""
        ...


class OptionBService(_OptionBServiceBase):
    """Rank-local service used by :class:`OptionBClient`.

    The transport factory resolves a precomputed volume ID directly. It must not
    consult the TorchStore controller. Readiness messages arrive through
    ``notify_ready``; the client owns the service mesh used to broadcast them.
    """

    actor_name = "OptionBServices"

    def __init__(
        self,
        strategy: TorchStoreStrategy | None = None,
        *,
        rank: str | None = None,
        transport_factory: _TransportFactory | None = None,
    ) -> None:
        if strategy is None and transport_factory is None:
            raise ValueError("strategy or transport_factory is required")
        if rank is None:
            if strategy is None:
                raise ValueError("rank is required without a TorchStore strategy")
            rank = str(strategy.get_client_id())
        if transport_factory is None:
            assert strategy is not None

            def transport_factory(volume_id: str) -> TransportBuffer:
                volume = strategy.get_storage_volume(volume_id)
                return create_transport_buffer(volume)

        self.rank = rank
        self._transport_factory = transport_factory
        self._ready: Dict[Tuple[str, str, SliceGeometry], asyncio.Event] = {}

    @classmethod
    def from_strategy(
        cls,
        rank: str,
        strategy: TorchStoreStrategy,
    ) -> "OptionBService":
        """Build from a configured TorchStore strategy's cached volume handles."""

        return cls(strategy, rank=rank)

    @classmethod
    async def spawn(
        cls, mesh: ProcMesh, strategy: TorchStoreStrategy
    ) -> "OptionBService":
        """Spawn one service per mesh rank."""
        return await spawn_actors(
            1,
            cls,
            cls.actor_name,
            mesh,
            strategy=strategy,
        )

    def _event(
        self, source: str, key: str, tensor_slice: TensorSlice
    ) -> asyncio.Event:
        identity = (source, key, slice_geometry(tensor_slice))
        event = self._ready.get(identity)
        if event is None:
            event = asyncio.Event()
            self._ready[identity] = event
        return event

    @endpoint
    async def put(
        self,
        key: str,
        tensor_slices: Tuple[TensorSlice, ...],
        value: object,
    ) -> None:
        """Put one local slice directly into this rank's storage volume."""
        if len(tensor_slices) != 1:
            raise NotImplementedError("one local TensorSlice per key is required")
        request = Request.from_any(key, value, tensor_slice=tensor_slices[0])
        transport = self._transport_factory(self.rank)
        await transport.put_to_storage_volume([request])

    @endpoint
    async def get(self, transfer: Transfer, destination: object) -> object:
        """Get a routed segment directly from its precomputed source volume."""
        if transfer.destination != self.rank:
            raise ValueError(
                f"route destination {transfer.destination!r} does not match "
                f"local rank {self.rank!r}"
            )
        request = Request.from_tensor_slice(transfer.key, transfer.segment)
        transport = self._transport_factory(transfer.source)
        return (await transport.get_from_storage_volume([request]))[0]

    @endpoint
    async def wait_ready(
        self, source: str, key: str, tensor_slice: TensorSlice
    ) -> None:
        """Wait locally until the source reports that its relay slice is stored."""
        await self._event(source, key, tensor_slice).wait()

    @endpoint
    async def notify_ready(
        self,
        source: str,
        peers: Tuple[str, ...],
        key: str,
        tensor_slice: TensorSlice,
    ) -> None:
        """Endpoint invoked by an ingress generator after its direct put."""
        if self.rank not in peers:
            return
        self._event(source, key, tensor_slice).set()
