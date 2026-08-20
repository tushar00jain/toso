"""Baseline gate for the dedup control benchmark.

The 8B case (``K=290 T=8 G=16``) is one measurement point large enough to be
noise-robust and small enough to stay under a few seconds; its deterministic outputs
are pinned to a JSON baseline. A change that moves the candidate counts or the
tracemalloc peaks or retired instructions fires this test; regenerate the baseline
consciously.

To regenerate on an intentional change, run the benchmark at the same knobs and
copy the emitted columns into :data:`_BASELINE`::

    .venv/bin/python -m realsim.tools.benchmark_dedup_control \\
        --preset 8b --warmups 0 --repeats 1
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realsim.tools.benchmark_dedup_control import _arguments, _memory, _runtime
from sim_common.perfcount import InstructionCount

_BASELINE = Path(__file__).parent / "baselines" / "dedup_8b.json"

#: Python peak-allocation is bit-repeatable within one interpreter build; this
#: absorbs GC-timing jitter without letting a real regression pass.
_MEMORY_TOLERANCE = 0.01
_INSTRUCTION_TOLERANCE = 0.02


def test_8b_matches_baseline():
    baseline = json.loads(_BASELINE.read_text())
    workload = baseline["workload"]
    args = _arguments(
        [
            "--keys", str(workload["keys"]),
            "--source-ranks", str(workload["source_ranks"]),
            "--generators", str(workload["generators"]),
            "--warmups", "0",
            "--repeats", "1",
        ]
    )
    runtime = _runtime(args)
    memory = _memory(args)

    expected_counts = baseline["counts"]
    assert (
        runtime.candidates,
        runtime.pending_candidates,
        runtime.selected_sources,
    ) == (
        expected_counts["candidates"],
        expected_counts["pending_candidates"],
        expected_counts["selected_sources"],
    )

    for field, expected in baseline["memory_kib"].items():
        observed = getattr(memory, f"{field}_kib")
        drift = abs(observed - expected) / expected
        assert drift <= _MEMORY_TOLERANCE, (
            f"{field}_kib: {observed} vs baseline {expected} "
            f"(drift {drift:.3%} > {_MEMORY_TOLERANCE:.0%})"
        )

    expected_instructions = baseline.get("instructions")
    if expected_instructions is None:
        pytest.skip("baseline has no instruction count")
    if not InstructionCount.available():
        pytest.skip("hardware instruction counting is unavailable")
    observed = runtime.full_decision_instructions
    assert observed is not None
    expected = expected_instructions["full_decision"]
    drift = abs(observed - expected) / expected
    assert drift <= _INSTRUCTION_TOLERANCE, (
        f"full_decision_instructions: {observed} vs baseline {expected} "
        f"(drift {drift:.3%} > {_INSTRUCTION_TOLERANCE:.0%})"
    )
