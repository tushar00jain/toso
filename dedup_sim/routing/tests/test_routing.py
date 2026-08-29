from __future__ import annotations

import asyncio

import pytest
import torch
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchstore.routing.plan import RoutingPlan
from torchstore.routing.service import RoutingService as ProductionRoutingService
from torchstore.state_dict_utils import get_state_dict, put_state_dict
from torchstore.transport.types import TensorSlice

from dedup_sim.workload.scenarios import RoutingScenario
from dedup_sim.routing.client import SimulationRoutingClient
from dedup_sim.routing.plans import registrations
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
) -> tuple[Mesh, dict[str, SimulationRoutingClient]]:
    topology = {
        volume_id: Endpoint(id=volume_id, host=volume_id, node=volume_id)
        for volume_id in rank_volumes.values()
    }
    mesh = Mesh(topology)
    handles = {
        rank: LocalRoutingServiceHandle(
            RoutingService(ProductionRoutingService(id_func=lambda r=rank: r))
        )
        for rank in plan.ranks
    }
    clients = {
        rank: SimulationRoutingClient(
            rank,
            plan.for_rank(rank),
            handles,
            mesh,
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
    rank_volumes = {rank: rank for rank in ("trainer", "ingress", "replica")}
    plan = RoutingPlan.build(
        registrations({"trainer": {"weight": (full,)}}, {"weight": 4}),
        registrations(
            {
                "ingress": {"weight": (ingress,)},
                "replica": {"weight": (replica,)},
            },
            {"weight": 4},
        ),
    )
    mesh, clients = _routing_clients(plan, rank_volumes)

    async def run() -> torch.Tensor:
        await clients["trainer"].put_batch({"weight": torch.ones(4)})
        await clients["ingress"].get("weight")
        await clients["replica"].get("weight")

        await clients["trainer"].put_batch({"weight": torch.full((4,), 2.0)})
        waiting = asyncio.create_task(clients["replica"].get("weight"))
        await asyncio.sleep(0)
        assert not waiting.done()

        await clients["ingress"].get("weight")
        return await waiting

    with mesh.installed():
        result = asyncio.run(run())
    torch.testing.assert_close(result, torch.full((4,), 2.0))


def test_simulation_can_validate_an_exact_local_snapshot() -> None:
    full = _slice((2,))
    rank_volumes = {rank: rank for rank in ("trainer", "generator")}
    sizes = {"a": 4, "b": 4}
    plan = RoutingPlan.build(
        registrations({"trainer": {"a": (full,), "b": (full,)}}, sizes),
        registrations({"generator": {"a": (full,), "b": (full,)}}, sizes),
    )
    mesh, clients = _routing_clients(plan, rank_volumes)

    async def put_exact_snapshot(snapshot: dict[str, torch.Tensor]) -> None:
        client = clients["trainer"]
        expected = {"a", "b"}
        actual = set(snapshot)
        if actual != expected:
            raise KeyError(
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        await client.put_batch(snapshot)

    with pytest.raises(KeyError, match="missing=.*b"):
        with mesh.installed():
            asyncio.run(put_exact_snapshot({"a": torch.ones(2)}))


def test_state_dict_helpers_reuse_the_routing_data_path() -> None:
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
    rank_volumes = {rank: rank for rank in ("trainer", "generator")}
    plan = RoutingPlan.build(
        registrations({"trainer": slices}, element_sizes, mapping=mapping),
        registrations({"generator": slices}, element_sizes),
    )
    mesh, clients = _routing_clients(plan, rank_volumes)
    destination = {
        "layer": {
            "weight": torch.empty_like(source["layer"]["weight"]),
            "bias": torch.empty_like(source["layer"]["bias"]),
        }
    }

    async def run() -> dict[str, object]:
        await put_state_dict(clients["trainer"], source, "model")
        return await get_state_dict(
            clients["generator"], "model", destination
        )

    with mesh.installed():
        result = asyncio.run(run())
    torch.testing.assert_close(
        result["layer"]["weight"], source["layer"]["weight"]
    )
    torch.testing.assert_close(
        result["layer"]["bias"], source["layer"]["bias"]
    )
