from __future__ import annotations

from dedup_sim.control._sensor import DedupDirectorySensor
from realsim.tools import benchmark_weight_sync_control as benchmark
from torchstore import coverage


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


def test_memory_mode_is_separate_and_stops_at_70b(monkeypatch, capsys) -> None:
    calls = []

    def run_case(case, scale, variant, **kwargs):
        calls.append((case, kwargs))
        return benchmark._Result(
            case=case,
            variant=variant,
            scale=scale,
            trainer_publish_cpu_ms=None,
            generator_lookups_cpu_ms=None,
            generator_completions_cpu_ms=None,
            total_cpu_ms=None,
            total_wall_ms=None,
            total_instructions=None,
            peak_python_kib=1.0,
        )

    monkeypatch.setattr(benchmark, "_run_case", run_case)

    assert benchmark.main(
        ["--preset", "suite", "--variant", "legacy", "--metrics", "memory"]
    ) == 0

    assert [case for case, _kwargs in calls] == ["1b", "8b", "qwen-27b", "70b"]
    assert all(not kwargs["measure_cpu"] for _case, kwargs in calls)
    assert all(not kwargs["measure_instructions"] for _case, kwargs in calls)
    assert all(kwargs["measure_memory"] for _case, kwargs in calls)
    output = capsys.readouterr().out
    assert "Peak Python memory" in output
    assert "70b-wide" not in output


def test_legacy_lookup_does_not_run_client_coverage(monkeypatch) -> None:
    workload = benchmark._Workload(
        benchmark._PRESETS["smoke"], "legacy", fanout_cap=2
    )
    try:
        workload.publish_trainers()

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("legacy benchmark must stop at the controller boundary")

        monkeypatch.setattr(coverage, "cover", fail_if_called)
        workload.generator_lookups()
    finally:
        workload.close()
