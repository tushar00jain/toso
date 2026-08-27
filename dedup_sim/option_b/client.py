"""Transport-neutral client for precomputed application-managed routes."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Mapping, Optional, TYPE_CHECKING

from torchstore.transport.types import TensorSlice

from ._model import (
    LocalRouteTable,
    RankRole,
    RouteEntry,
    Transfer,
    TransferKind,
    slice_geometry,
)
from .service import OptionBService

if TYPE_CHECKING:
    from .plan import OptionBPlan

__all__ = ["OptionBClient"]


class _OptionBClientBase(ABC):
    """The runtime API for executing one rank's precomputed routes.

    Construct a client with the rank's plan, its rank-local service handle, and
    the complete service mesh. Trainer ranks call :meth:`publish`; generator
    ranks call :meth:`get`.
    """

    @abstractmethod
    async def publish(self, snapshot: Mapping[str, object]) -> None:
        """Publish this trainer rank's snapshot directly to its volume."""
        ...

    @abstractmethod
    async def get(self, key: str, destination: Optional[object] = None) -> object:
        """Execute this generator rank's precomputed receives for ``key``."""
        ...


class OptionBClient(_OptionBClientBase):
    """Stable ``publish``/``get`` API backed by one immutable local route table."""

    def __init__(
        self,
        rank: str,
        plan: OptionBPlan,
        service: OptionBService,
        services: OptionBService,
    ) -> None:
        service_rank = getattr(service, "rank", rank)
        if service_rank != rank:
            raise ValueError(
                f"service rank {service_rank!r} does not match client rank {rank!r}"
            )
        self.routes: LocalRouteTable = plan._local(rank)
        self.service = service
        self.services = services

    def _lookup(self, key: str) -> RouteEntry:
        return self.routes.lookup(key)

    async def publish(self, snapshot: Mapping[str, object]) -> None:
        """Publish trainer slices without registering controller records."""
        if self.routes.role != RankRole.TRAINER:
            raise RuntimeError("only trainer clients publish snapshots")
        unknown = set(snapshot) - set(self.routes.published)
        if unknown:
            raise KeyError(f"rank {self.routes.rank!r} does not publish {sorted(unknown)}")
        for key, value in snapshot.items():
            self._lookup(key)
            await self.service.put.call_one(
                key,
                tuple(self.routes.published[key]),
                value,
            )

    async def get(self, key: str, destination: Optional[object] = None) -> object:
        """Execute local receives, then publish and signal any relay slices."""
        if self.routes.role != RankRole.GENERATOR:
            raise RuntimeError("only generator clients request model slices")
        requested = tuple(self.routes.requested.get(key, ()))
        if not requested:
            raise KeyError(f"rank {self.routes.rank!r} does not request {key!r}")
        entry = self._lookup(key)

        async def receive_slice(target: TensorSlice) -> object:
            target_geometry = slice_geometry(target)
            incoming = tuple(
                action
                for action in entry.receives
                if slice_geometry(action.destination_slice) == target_geometry
            )
            if not incoming:
                raise RuntimeError(
                    f"local route for {self.routes.rank!r} does not fill "
                    f"{key!r} {target_geometry}"
                )
            async def receive(action: Transfer) -> object:
                if action.kind == TransferKind.RELAY:
                    await self.service.wait_ready.call_one(
                        action.source, action.key, action.segment
                    )
                return await self.service.get.call_one(action, destination)

            pieces = await asyncio.gather(*(receive(action) for action in incoming))
            value = destination
            if value is None:
                value = pieces[0] if len(pieces) == 1 else tuple(pieces)
            for signal in entry.broadcasts:
                if slice_geometry(signal.tensor_slice) != target_geometry:
                    continue
                await self.service.put.call_one(key, (target,), value)
                self.services.notify_ready.broadcast(
                    self.routes.rank,
                    signal.peers,
                    signal.key,
                    signal.tensor_slice,
                )
            return value

        values = await asyncio.gather(*(receive_slice(target) for target in requested))
        if destination is not None:
            return destination
        return values[0] if len(values) == 1 else tuple(values)
