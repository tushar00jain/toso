from __future__ import annotations

from dedup_sim.control._sensor import DedupDirectorySensor
from realsim.tools import benchmark_weight_sync_control as benchmark


def test_dedup_sample_retires_pending_and_keeps_live_rows() -> None:
    workload = benchmark._Workload(
        benchmark._PRESETS["smoke"], "legacy-dedup", fanout_cap=2
    )
    try:
        workload.publish_trainers()
        before = workload.service.serving_union(workload.generator_requests[0])
        assert all(volume.startswith("trainer-") for _pub, volume in before)

        plans = workload.generator_lookups()
        directory = workload.plane.sensor(DedupDirectorySensor)
        assert directory.in_flight() == {plan.publication for plan in plans}

        workload.complete_generators(plans)
        after = workload.service.serving_union(workload.generator_requests[0])
        assert directory.in_flight() == set()
        assert all(pub == 0 for pub, _volume in after)
        assert any(volume.startswith("generator-") for _pub, volume in after)
    finally:
        workload.close()
