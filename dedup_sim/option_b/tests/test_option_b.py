from __future__ import annotations

import pytest
import torch
from monarch._src.actor.endpoint import EndpointProperty
from monarch.actor import Actor
from torchstore.transport.types import TensorSlice

from dedup_sim.option_b import OptionBClient, OptionBPlan, OptionBService
from dedup_sim.option_b.client import _OptionBClientBase
from dedup_sim.option_b.plan import _OptionBPlanBase
from dedup_sim.option_b.service import _OptionBServiceBase
from dedup_sim.workload.scenarios import OptionBScenario
from realsim.mesh import LocalActorMesh
from realsim.seams.option_b_handle import LocalOptionBServiceHandle
from realsim.seams.option_b_service import OptionBService as LocalOptionBService
from realsim.seams.controller_service import ControllerService
from sim_common.async_engine import run_sim


def test_option_b_exposes_the_three_production_types() -> None:
    import dedup_sim.option_b as option_b

    assert option_b.__all__ == ["OptionBClient", "OptionBPlan", "OptionBService"]
    assert issubclass(OptionBClient, _OptionBClientBase)
    assert issubclass(OptionBPlan, _OptionBPlanBase)
    assert issubclass(OptionBService, _OptionBServiceBase)
    assert issubclass(OptionBService, Actor)
    for name in (
        "put",
        "get",
        "wait_ready",
        "notify_ready",
    ):
        assert isinstance(getattr(OptionBService, name), EndpointProperty)


def _slice(
    offsets,
    shape,
    *,
    global_shape=(8, 6),
    coordinates=(),
    mesh_shape=(),
) -> TensorSlice:
    return TensorSlice(
        offsets=offsets,
        coordinates=coordinates,
        global_shape=global_shape,
        local_shape=shape,
        mesh_shape=mesh_shape,
    )


def test_cross_axis_reshard_and_replication_are_inferred_from_geometry() -> None:
    left0 = _slice((0, 0), (8, 3), coordinates=(0, 0), mesh_shape=(2, 2))
    left1 = _slice((0, 0), (8, 3), coordinates=(0, 1), mesh_shape=(2, 2))
    right0 = _slice((0, 3), (8, 3), coordinates=(1, 0), mesh_shape=(2, 2))
    right1 = _slice((0, 3), (8, 3), coordinates=(1, 1), mesh_shape=(2, 2))
    plan = OptionBPlan.build(
        {
            "trainer-top": {"w": (_slice((0, 0), (4, 6)),)},
            "trainer-bottom": {"w": (_slice((4, 0), (4, 6)),)},
        },
        {
            "left-dp0": {"w": (left0,)},
            "left-dp1": {"w": (left1,)},
            "right-dp0": {"w": (right0,)},
            "right-dp1": {"w": (right1,)},
        },
        {"w": 2},
    )
    transfers = [
        action
        for rank in plan.ranks
        for action in plan.lookup(rank, "w").sends
    ]
    trainer = [x for x in transfers if x.kind.value == "trainer-to-generator"]
    relay = [x for x in transfers if x.kind.value == "generator-read-through"]

    assert sum(x.nbytes for x in trainer) == 8 * 6 * 2
    assert sum(x.nbytes for x in relay) == 8 * 6 * 2
    assert len(trainer) == 4
    assert len(relay) == 2
    assert sum(
        len(plan.lookup(rank, "w").broadcasts) for rank in plan.ranks
    ) == 2
    assert {x.source for x in trainer} == {"trainer-top", "trainer-bottom"}

    for rank in ("left-dp0", "right-dp0"):
        incoming = plan.lookup(rank, "w").receives
        assert sum(action.nbytes for action in incoming) == 8 * 3 * 2


def test_missing_multidimensional_coverage_is_rejected_at_setup() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        OptionBPlan.build(
            {"producer": {"w": (_slice((0, 0), (7, 6)),)}},
            {"consumer": {"w": (_slice((0, 0), (8, 6)),)}},
            {"w": 2},
        )


def test_plan_can_be_saved_and_distributed_per_rank(tmp_path) -> None:
    full = _slice((0, 0), (8, 6))
    plan = OptionBPlan.build(
        {"trainer": {"w": (full,)}},
        {"generator": {"w": (full,)}},
        {"w": 2},
    ).for_rank("generator")
    path = tmp_path / "generator-plan.json"

    plan.save(path)
    restored = OptionBPlan.load(path)

    assert restored.ranks == ("generator",)
    assert restored.to_dict() == plan.to_dict()
    assert restored.lookup("generator", "w").receives[0].source == "trainer"


class _Transport:
    def __init__(self, requester: str, volume: str, events) -> None:
        self.requester = requester
        self.volume = volume
        self.events = events

    async def put_to_storage_volume(self, requests) -> None:
        self.events.append(("publish", (self.requester, self.volume, requests)))

    async def get_from_storage_volume(self, requests):
        self.events.append(("receive", (self.requester, self.volume, requests)))
        return ["weights"]


def test_client_keeps_route_actions_behind_publish_and_get() -> None:
    full = _slice((0, 0), (8, 6))
    replica = _slice((0, 0), (8, 6), coordinates=(1,), mesh_shape=(2,))
    plan = OptionBPlan.build(
        {"trainer": {"w": (full,)}},
        {
            "generator-a": {"w": (full,)},
            "generator-b": {"w": (replica,)},
        },
        {"w": 2},
    )
    events = []
    service_handles = {
        rank: LocalOptionBServiceHandle(
            LocalOptionBService(
                OptionBService(
                    rank=rank,
                    transport_factory=lambda volume, requester=rank: _Transport(
                        requester, volume, events
                    ),
                )
            )
        )
        for rank in plan.ranks
    }
    services = LocalActorMesh(
        service_handles, ("notify_ready",)
    )
    clients = {
        rank: OptionBClient(
            rank,
            plan.for_rank(rank),
            services.for_rank(rank),
            services,
        )
        for rank in plan.ranks
    }

    async def run():
        await clients["trainer"].publish(
            {"w": torch.empty((8, 6), device="meta")}
        )
        destination = torch.empty((8, 6), device="meta")
        result = await clients["generator-a"].get("w", destination)
        await clients["generator-b"].get(
            "w", torch.empty((8, 6), device="meta")
        )
        return result, destination

    (result, destination), _trace = run_sim(run(), quiet=True)
    assert result is destination

    kinds = [kind for kind, _payload in events]
    assert kinds == ["publish", "receive", "publish", "receive"]


def test_qwen_scenario_uses_direct_volume_io(monkeypatch) -> None:
    def reject_controller_io(*args, **kwargs):
        raise AssertionError("Option B data movement must bypass the controller")

    monkeypatch.setattr(ControllerService, "locate_volumes", reject_controller_io)
    monkeypatch.setattr(ControllerService, "notify_put_batch", reject_controller_io)

    direct, option_b = [run.execute(quiet=True) for run in OptionBScenario().runs()]
    payload = 55_600_000_000

    assert direct.ledger.origin_bytes == 2 * payload
    assert direct.ledger.transfer_bytes == 2 * payload
    assert direct.ledger.wallclock == pytest.approx(2 * payload / 17.5e9)

    expected = payload / 17.5e9 + (payload / 4) / 900e9
    assert option_b.ledger.origin_bytes == payload
    assert option_b.ledger.transfer_bytes == 2 * payload
    assert option_b.ledger.wallclock == pytest.approx(expected)
    assert direct.sim.mesh.directory.service.keys() == []
    assert option_b.sim.mesh.directory.service.keys() == []
