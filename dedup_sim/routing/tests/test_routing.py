from __future__ import annotations

import asyncio

import pytest
import torch
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchstore.routing import RoutingClient, RoutingPlan
from torchstore.routing import RoutingService as ProductionRoutingService
from torchstore.transport.types import TensorSlice

from dedup_sim.workload.scenarios import RoutingScenario
from realsim.mesh import Mesh
from realsim.seams.controller_service import ControllerService
from realsim.seams.routing_handle import LocalRoutingServiceHandle
from realsim.seams.routing_service import RoutingService
from realsim.seams.transport import Endpoint


def _slice(
    shape: tuple[int, ...], *, coordinate: int = 0, mesh_size: int = 1
) -> TensorSlice:
    return TensorSlice(
        offsets=(0,) * len(shape),
        coordinates=(coordinate,),
        global_shape=shape,
        local_shape=shape,
        mesh_shape=(mesh_size,),
    )


def _routing_clients(
    plan: RoutingPlan,
    rank_volumes: dict[str, str],
) -> tuple[Mesh, dict[str, RoutingClient]]:
    topology = {
        volume_id: Endpoint(id=volume_id, host=volume_id, node=volume_id)
        for volume_id in rank_volumes.values()
    }
    mesh = Mesh(topology)
    handles = {
        rank: LocalRoutingServiceHandle(
            RoutingService(ProductionRoutingService())
        )
        for rank in plan.ranks
    }
    clients = {
        rank: RoutingClient(
            rank,
            plan.for_rank(rank),
            handles[rank],
            handles,
            transport_factory=mesh.adapter(rank_volumes[rank]).transport_for,
        )
        for rank in plan.ranks
    }
    return mesh, clients


def test_qwen_routing_uses_direct_volume_io(monkeypatch) -> None:
    direct_run, routed_run = RoutingScenario().runs()
    direct = direct_run.execute(quiet=True)

    def reject_controller_io(*args, **kwargs):
        raise AssertionError("routing data movement must bypass the controller")

    monkeypatch.setattr(ControllerService, "locate_volumes", reject_controller_io)
    monkeypatch.setattr(ControllerService, "notify_put_batch", reject_controller_io)

    routed = routed_run.execute(quiet=True)
    payload = 55_600_000_000

    assert direct.ledger.origin_bytes == 2 * payload
    assert direct.ledger.transfer_bytes == 2 * payload
    assert direct.ledger.wallclock == pytest.approx(2 * payload / 17.5e9)

    expected = payload / 17.5e9 + (payload / 4) / 900e9
    assert routed.ledger.origin_bytes == payload
    assert routed.ledger.transfer_bytes == 2 * payload
    assert routed.ledger.wallclock == pytest.approx(expected)
    assert set(direct.sim.mesh.directory.service.keys()) == {
        "W0",
        "W1",
        "W2",
        "W3",
    }
    assert routed.sim.mesh.directory.service.keys() == []


def test_repeated_updates_do_not_reuse_stale_relay_readiness() -> None:
    full = _slice((4,))
    ingress = _slice((4,), mesh_size=2)
    replica = _slice((4,), coordinate=1, mesh_size=2)
    rank_volumes = {
        "trainer": "trainer-volume",
        "ingress": "ingress-volume",
        "replica": "replica-volume",
    }
    plan = RoutingPlan.build(
        {"trainer": {"weight": (full,)}},
        {
            "ingress": {"weight": (ingress,)},
            "replica": {"weight": (replica,)},
        },
        {"weight": 4},
        rank_volumes=rank_volumes,
    )
    _mesh, clients = _routing_clients(plan, rank_volumes)

    async def run() -> torch.Tensor:
        await clients["trainer"].publish({"weight": torch.ones(4)})
        await clients["ingress"].get("weight")
        await clients["replica"].get("weight")

        await clients["trainer"].publish({"weight": torch.full((4,), 2.0)})
        waiting = asyncio.create_task(clients["replica"].get("weight"))
        await asyncio.sleep(0)
        assert not waiting.done()

        await clients["ingress"].get("weight")
        return await waiting

    torch.testing.assert_close(asyncio.run(run()), torch.full((4,), 2.0))


def test_publish_requires_the_exact_local_key_set() -> None:
    full = _slice((2,))
    rank_volumes = {
        "trainer": "trainer-volume",
        "generator": "generator-volume",
    }
    plan = RoutingPlan.build(
        {"trainer": {"a": (full,), "b": (full,)}},
        {"generator": {"a": (full,), "b": (full,)}},
        {"a": 4, "b": 4},
        rank_volumes=rank_volumes,
    )
    _mesh, clients = _routing_clients(plan, rank_volumes)

    with pytest.raises(KeyError, match="missing=.*b"):
        asyncio.run(clients["trainer"].publish({"a": torch.ones(2)}))


def test_state_dict_api_reuses_the_routing_data_path() -> None:
    source = {
        "layer": {
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "bias": torch.arange(2, dtype=torch.float32),
        }
    }
    flattened, mapping = flatten_state_dict(source)
    slices = {
        f"model/{key}": (_slice(tuple(value.shape)),)
        for key, value in flattened.items()
    }
    element_sizes = {
        f"model/{key}": value.element_size()
        for key, value in flattened.items()
    }
    rank_volumes = {
        "trainer": "trainer-volume",
        "generator": "generator-volume",
    }
    plan = RoutingPlan.build(
        {"trainer": slices},
        {"generator": slices},
        element_sizes,
        rank_volumes=rank_volumes,
        state_dict_mappings={"model": mapping},
    )
    _mesh, clients = _routing_clients(plan, rank_volumes)
    destination = {
        "layer": {
            "weight": torch.empty_like(source["layer"]["weight"]),
            "bias": torch.empty_like(source["layer"]["bias"]),
        }
    }

    async def run() -> dict[str, object]:
        await clients["trainer"].put_state_dict(source, "model")
        return await clients["generator"].get_state_dict(
            "model",
            user_state_dict=destination,
        )

    result = asyncio.run(run())
    torch.testing.assert_close(
        result["layer"]["weight"], source["layer"]["weight"]
    )
    torch.testing.assert_close(
        result["layer"]["bias"], source["layer"]["bias"]
    )
