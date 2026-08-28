"""Simulation adapter for TorchStore's production routing client."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

import torch
from torchstore.routing import RoutingClient, RoutingPlan, RoutingService
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport.types import Request

from realsim.mesh import Mesh
from realsim.seams.routing_handle import LocalRoutingServiceHandle

__all__ = ["SimulationRoutingClient"]


class SimulationRoutingClient(RoutingClient):
    """Route through the simulator transport without changing production APIs."""

    def __init__(
        self,
        rank: str,
        plan: RoutingPlan,
        service: LocalRoutingServiceHandle,
        services: Mapping[str, LocalRoutingServiceHandle],
        mesh: Mesh,
    ) -> None:
        volume_id = plan._local(rank).volume_id
        adapter = mesh.adapter(volume_id)
        super().__init__(
            rank,
            plan,
            cast(RoutingService, service),
            cast(Mapping[str, RoutingService], services),
            cast(TorchStoreStrategy, adapter.strategy),
        )
        self._mesh = mesh
        self._volume_id = volume_id
        self._adapter = adapter

    def _local_transport(self):
        return self._adapter.transport_for(self._volume_id)

    async def _fetch(self, requests: list[Request]) -> dict[str, object]:
        self._mesh.bind_source(self._volume_id)
        return await super()._fetch(requests)

    def _assemble_results(
        self,
        requests: list[Request],
        fetch_pairs: list[tuple[Request, Any]],
        whole_keys: set[str],
    ) -> dict[str, Any]:
        meta_keys = {
            request.key
            for request, result in fetch_pairs
            if isinstance(result, torch.Tensor) and result.is_meta
        }
        counts = Counter(request.key for request, _result in fetch_pairs)
        if any(counts[key] != 1 for key in meta_keys):
            raise NotImplementedError("simulation cannot assemble sharded meta tensors")
        return super()._assemble_results(requests, fetch_pairs, whole_keys | meta_keys)
