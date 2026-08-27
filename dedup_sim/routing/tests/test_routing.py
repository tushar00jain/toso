from __future__ import annotations

import pytest

from dedup_sim.workload.scenarios import RoutingScenario
from realsim.seams.controller_service import ControllerService


def test_qwen_scenario_uses_direct_volume_io(monkeypatch) -> None:
    def reject_controller_io(*args, **kwargs):
        raise AssertionError("routing data movement must bypass the controller")

    monkeypatch.setattr(ControllerService, "locate_volumes", reject_controller_io)
    monkeypatch.setattr(ControllerService, "notify_put_batch", reject_controller_io)

    direct, routed = [run.execute(quiet=True) for run in RoutingScenario().runs()]
    payload = 55_600_000_000

    assert direct.ledger.origin_bytes == 2 * payload
    assert direct.ledger.transfer_bytes == 2 * payload
    assert direct.ledger.wallclock == pytest.approx(2 * payload / 17.5e9)

    expected = payload / 17.5e9 + (payload / 4) / 900e9
    assert routed.ledger.origin_bytes == payload
    assert routed.ledger.transfer_bytes == 2 * payload
    assert routed.ledger.wallclock == pytest.approx(expected)
    assert direct.sim.mesh.directory.service.keys() == []
    assert routed.sim.mesh.directory.service.keys() == []
