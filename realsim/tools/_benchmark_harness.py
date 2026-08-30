# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared scaffolding for the weight-sync benchmarks.

A benchmark supplies a workload and two printers; everything about *how* a run is
measured lives here, so a change to sampling, gating, or table layout is one edit
rather than one per tool.

Sampling rules the tools depend on:

- each metric runs in its own process-wide-quiet phase, so ``tracemalloc`` never
  perturbs a CPU timing or an instruction count;
- CPU time is thread time on a dedicated worker thread, not wall time;
- a metric that costs an extra pass is capped by workload size, and oversized
  cases are dropped from that metric's table rather than reported as zero.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, TypeVar

from sim_common.perfcount import InstructionCount

__all__ = [
    "MEASURED_METRICS",
    "PRESETS",
    "Scale",
    "Timing",
    "add_common_arguments",
    "duration",
    "instruction_count",
    "measurable_rows",
    "measured",
    "median_ms",
    "memory_kib",
    "memory_size",
    "metadata_rows",
    "resolve_cases",
    "shard_geometry",
    "tables",
    "timed",
    "worker_thread",
]

_T = TypeVar("_T")

MEASURED_METRICS = ("cpu", "instructions", "memory")


@dataclass(frozen=True)
class Scale:
    keys: int
    source_ranks: int
    generators: int
    generator_shards: int


PRESETS = {
    "smoke": Scale(keys=8, source_ranks=2, generators=8, generator_shards=2),
    "1b": Scale(keys=120, source_ranks=2, generators=8, generator_shards=2),
    "8b": Scale(keys=290, source_ranks=8, generators=16, generator_shards=8),
    "qwen-27b": Scale(
        keys=1_199,
        source_ranks=8,
        generators=8,
        generator_shards=4,
    ),
    "70b": Scale(keys=723, source_ranks=8, generators=64, generator_shards=8),
    "70b-wide": Scale(
        keys=723,
        source_ranks=64,
        generators=128,
        generator_shards=64,
    ),
    "405b": Scale(
        keys=1_500,
        source_ranks=32,
        generators=128,
        generator_shards=32,
    ),
    "moe": Scale(
        keys=3_000,
        source_ranks=32,
        generators=128,
        generator_shards=32,
    ),
    "kimi-k2": Scale(
        keys=5_203,
        source_ranks=256,
        generators=128,
        generator_shards=128,
    ),
}

_DEFAULT_SUITE = (
    "1b",
    "8b",
    "qwen-27b",
    "70b",
    "70b-wide",
    "405b",
    "moe",
)

# Tracing every allocation is disproportionately slow past this size.
_MAX_MEMORY_METADATA_ROWS = PRESETS["70b"].keys * (
    PRESETS["70b"].source_ranks + PRESETS["70b"].generators
)

# Retired instructions cost one extra pass rather than tracing allocations, so
# the cap is looser than the memory one but still finite.
_MAX_INSTRUCTION_METADATA_ROWS = PRESETS["moe"].keys * (
    PRESETS["moe"].source_ranks + PRESETS["moe"].generators
)


@dataclass(frozen=True)
class Timing:
    cpu_ns: int
    wall_ns: int


def metadata_rows(scale: Scale) -> int:
    return scale.keys * (scale.source_ranks + scale.generators)


def measurable_rows(metric: str) -> int:
    """Largest workload a metric is measured at; bigger cases are dropped."""
    if metric == "memory":
        return _MAX_MEMORY_METADATA_ROWS
    if metric == "instructions":
        return _MAX_INSTRUCTION_METADATA_ROWS
    return sys.maxsize


def shard_geometry(scale: Scale) -> tuple[int, Callable[[int, int], tuple[int, int]]]:
    """Global extent, and a ``(rank, shards) -> (offset, width)`` slice rule.

    One dimension per key, so a requester shard meets only the source shards
    beside it. A layout sharding a different dimension than the sources would
    make every pair overlap; the presets do not cover that.
    """
    extent = math.lcm(scale.source_ranks, scale.generator_shards)

    def bounds(rank: int, shards: int) -> tuple[int, int]:
        width = extent // shards
        return rank * width, width

    return extent, bounds


def timed(operation: Callable[[], _T]) -> tuple[Timing, _T]:
    wall_started = time.perf_counter_ns()
    cpu_started = time.thread_time_ns()
    result = operation()
    cpu_ns = time.thread_time_ns() - cpu_started
    wall_ns = time.perf_counter_ns() - wall_started
    return Timing(cpu_ns, wall_ns), result


class measured:
    """Quiet the collector around one sample so it cannot skew the numbers."""

    def __enter__(self) -> "measured":
        gc.collect()
        self._enabled = gc.isenabled()
        gc.disable()
        return self

    def __exit__(self, *_exception: object) -> None:
        if self._enabled:
            gc.enable()


def instruction_count(body: Callable[[], object]) -> int | None:
    """Retired instructions for ``body``, or None where perf is unavailable."""
    if not InstructionCount.available():
        return None
    with measured():
        with InstructionCount() as counter:
            body()
    return counter.count


def memory_kib(body: Callable[[], object]) -> float:
    """Peak traced allocation ``body`` adds, above what is already live."""
    with measured():
        tracemalloc.start()
        try:
            baseline, _peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            body()
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return (peak - baseline) / 1024


def median_ms(values: Sequence[int]) -> float:
    return statistics.median(values) / 1_000_000


def worker_thread(prefix: str) -> ThreadPoolExecutor:
    """A single dedicated thread, so thread CPU time measures only the workload."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix=prefix)


def duration(milliseconds: float | None) -> str:
    if milliseconds is None:
        return "—"
    if milliseconds < 1_000:
        return f"{milliseconds:.3f} ms"
    return f"{milliseconds / 1_000:.3f} s"


def memory_size(kibibytes: float | None) -> str:
    if kibibytes is None:
        return "—"
    if kibibytes < 1024:
        return f"{kibibytes:.3f} KiB"
    return f"{kibibytes / 1024:.3f} MiB"


def _metrics_for(selection: str) -> tuple[str, ...]:
    """Metrics to run, each as its own table so cheap ones print first."""
    return MEASURED_METRICS if selection == "all" else (selection,)


def _cases_for(
    metric: str,
    cases: Sequence[tuple[str, Scale]],
) -> list[tuple[str, Scale]]:
    limit = measurable_rows(metric)
    return [(case, scale) for case, scale in cases if metadata_rows(scale) <= limit]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=(*PRESETS, "suite"), default="smoke")
    parser.add_argument("--keys", type=int)
    parser.add_argument("--source-ranks", type=int)
    parser.add_argument("--generators", type=int)
    parser.add_argument("--generator-shards", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--metrics",
        choices=(*MEASURED_METRICS, "all"),
        default="cpu",
    )


def resolve_cases(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[tuple[str, Scale]]:
    """The (name, scale) pairs to run, from a preset or explicit overrides."""
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.preset == "suite":
        return [(case, PRESETS[case]) for case in _DEFAULT_SUITE]

    preset = PRESETS[args.preset]
    scale = Scale(
        keys=args.keys if args.keys is not None else preset.keys,
        source_ranks=(
            args.source_ranks if args.source_ranks is not None else preset.source_ranks
        ),
        generators=(
            args.generators if args.generators is not None else preset.generators
        ),
        generator_shards=(
            args.generator_shards
            if args.generator_shards is not None
            else preset.generator_shards
        ),
    )
    for name in ("keys", "source_ranks", "generators", "generator_shards"):
        if getattr(scale, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if scale.generators < scale.generator_shards:
        parser.error("--generators must be at least --generator-shards")
    if scale.generators % scale.generator_shards:
        parser.error("--generators must be divisible by --generator-shards")
    return [(args.preset, scale)]


def tables(
    selection: str,
    cases: Sequence[tuple[str, Scale]],
) -> Iterator[tuple[str, list[tuple[str, Scale]]]]:
    """Each metric with the cases it is measured at, skipping empty tables."""
    for metric in _metrics_for(selection):
        limited = _cases_for(metric, cases)
        if limited:
            yield metric, limited
