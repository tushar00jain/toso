"""Baseline gate for the complete weight-sync control lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realsim.tools import benchmark_weight_sync_control as benchmark
from sim_common.perfcount import InstructionCount

_BASELINE = Path(__file__).parent / "baselines" / "weight_sync_8b.json"
_MEMORY_TOLERANCE = 0.02
_INSTRUCTION_TOLERANCE = 0.02


@pytest.mark.parametrize("variant", ["legacy", "legacy-dedup", "indexed-dedup"])
def test_8b_complete_lifecycle_matches_baseline(variant: str) -> None:
    baseline = json.loads(_BASELINE.read_text())
    workload = baseline["workload"]
    result = benchmark._run_case(
        "8b",
        benchmark._Scale(**workload),
        variant,
        fanout_cap=2,
        warmups=0,
        repeats=1,
    )
    expected = baseline["variants"][variant]
    memory_drift = (
        abs(result.peak_python_kib - expected["peak_python_kib"])
        / expected["peak_python_kib"]
    )
    assert memory_drift <= _MEMORY_TOLERANCE, (
        f"peak_python_kib: {result.peak_python_kib} vs "
        f"{expected['peak_python_kib']} "
        f"(drift {memory_drift:.3%} > {_MEMORY_TOLERANCE:.0%})"
    )

    if not InstructionCount.available():
        pytest.skip("hardware instruction counting is unavailable")
    assert result.total_instructions is not None
    instruction_drift = (
        abs(result.total_instructions - expected["total_instructions"])
        / expected["total_instructions"]
    )
    assert instruction_drift <= _INSTRUCTION_TOLERANCE, (
        f"total_instructions: {result.total_instructions} vs "
        f"{expected['total_instructions']} "
        f"(drift {instruction_drift:.3%} > {_INSTRUCTION_TOLERANCE:.0%})"
    )
