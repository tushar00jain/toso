"""Simulation adapter for TorchStore's production routing client."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

import torch
from torchstore.routing.client import RoutingClient
from torchstore.routing.plan import RoutingPlan
from torchstore.routing.service import RoutingService
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
        services: Mapping[str, LocalRoutingServiceHandle],
        mesh: Mesh,
    ) -> None:
        table = plan._local(rank)
        volume_id = table.volume_id
        adapter = mesh.adapter(volume_id)
        # The simulator plans up front, so install the routes the coordinator
        # would otherwise hand back from register_state_dict.
        super().__init__(
            rank,
            table.role,
            None,
            cast(TorchStoreStrategy, adapter.strategy),
        )
        self._controller.install(
            plan, cast(Mapping[str, RoutingService], services)
        )
        self._mesh = mesh
        self._volume_id = volume_id

    async def put_batch(self, entries: dict[str, torch.Tensor | Any]) -> None:
        self._mesh.bind_source(self._volume_id)
        await super().put_batch(entries)

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
