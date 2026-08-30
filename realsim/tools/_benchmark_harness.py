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
from dataclasses import dataclass, replace
from typing import Callable, Iterator, Sequence, TypeVar

from sim_common.perfcount import InstructionCount

__all__ = [
    "LAYOUTS",
    "MEASURED_METRICS",
    "PRESETS",
    "KeySpec",
    "Scale",
    "Timing",
    "Width",
    "add_common_arguments",
    "duration",
    "instruction_count",
    "key_specs",
    "measurable_rows",
    "measured",
    "median_ms",
    "memory_kib",
    "memory_size",
    "metadata_rows",
    "resolve_cases",
    "shard_bounds",
    "shard_geometry",
    "tables",
    "timed",
    "worker_thread",
]

_T = TypeVar("_T")

MEASURED_METRICS = ("cpu", "instructions", "memory")

# "uniform" gives every key one identical 1-D slice; "realistic" builds a
# per-layer transformer tensor table. See :func:`key_specs`.
LAYOUTS = ("uniform", "realistic")


@dataclass(frozen=True)
class Width:
    """Transformer dimensions the realistic layout cuts its tensors from.

    Every dimension either side shards is a multiple of 256, the largest rank
    count in :data:`PRESETS`, so no preset lands on a ragged shard.
    """

    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    vocab: int
    head_dim: int = 128


_WIDTH_1B = Width(hidden=2048, ffn=8192, heads=16, kv_heads=2, vocab=128_256)
_WIDTH_8B = Width(hidden=4096, ffn=14_336, heads=32, kv_heads=8, vocab=128_256)
_WIDTH_27B = Width(hidden=5120, ffn=17_408, heads=40, kv_heads=8, vocab=152_064)
_WIDTH_70B = Width(hidden=8192, ffn=28_672, heads=64, kv_heads=8, vocab=128_256)
_WIDTH_405B = Width(hidden=16_384, ffn=53_248, heads=128, kv_heads=8, vocab=128_256)


@dataclass(frozen=True)
class Scale:
    keys: int
    source_ranks: int
    generators: int
    generator_shards: int
    layout: str = "uniform"
    # Only the realistic layout reads this; uniform keys are sized from the
    # rank counts alone.
    width: Width = _WIDTH_8B


PRESETS = {
    "smoke": Scale(
        keys=8,
        source_ranks=2,
        generators=8,
        generator_shards=2,
        width=_WIDTH_1B,
    ),
    "1b": Scale(
        keys=120,
        source_ranks=2,
        generators=8,
        generator_shards=2,
        width=_WIDTH_1B,
    ),
    "8b": Scale(
        keys=290,
        source_ranks=8,
        generators=16,
        generator_shards=8,
        width=_WIDTH_8B,
    ),
    "qwen-27b": Scale(
        keys=1_199,
        source_ranks=8,
        generators=8,
        generator_shards=4,
        width=_WIDTH_27B,
    ),
    "70b": Scale(
        keys=723,
        source_ranks=8,
        generators=64,
        generator_shards=8,
        width=_WIDTH_70B,
    ),
    "70b-wide": Scale(
        keys=723,
        source_ranks=64,
        generators=128,
        generator_shards=64,
        width=_WIDTH_70B,
    ),
    "405b": Scale(
        keys=1_500,
        source_ranks=32,
        generators=128,
        generator_shards=32,
        width=_WIDTH_405B,
    ),
    "moe": Scale(
        keys=3_000,
        source_ranks=32,
        generators=128,
        generator_shards=32,
        width=_WIDTH_405B,
    ),
    "kimi-k2": Scale(
        keys=5_203,
        source_ranks=256,
        generators=128,
        generator_shards=128,
        width=_WIDTH_405B,
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


@dataclass(frozen=True)
class KeySpec:
    """One storage key: its global shape, and how the two sides split it.

    The trainer is FSDP2/DTensor ``Shard(0)`` on every parameter, so only the
    generator side varies. ``generator_dim`` is the dimension inference
    tensor-parallelism splits, or None where inference replicates the tensor.
    """

    name: str
    global_shape: tuple[int, ...]
    generator_dim: int | None


# Tensors per transformer block, and the embedding/norm/head outside the stack.
_LAYER_KEYS = 9
_NON_LAYER_KEYS = 3


def shard_bounds(
    spec: KeySpec,
    dim: int | None,
    rank: int,
    shards: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Offsets and local shape of ``rank``'s piece, splitting ``dim`` ``shards`` ways.

    ``dim`` of None is a replicated side, which takes the whole tensor. Chunks
    tile their dimension exactly at any size, so publisher slices always cover a
    requested slice with nothing left over and nothing counted twice.
    """
    if dim is None:
        return (0,) * len(spec.global_shape), spec.global_shape
    size = spec.global_shape[dim]
    width = -(-size // shards)
    offset = min(rank * width, size)
    return (
        tuple(offset if axis == dim else 0 for axis in range(len(spec.global_shape))),
        tuple(
            min(width, size - offset) if axis == dim else extent
            for axis, extent in enumerate(spec.global_shape)
        ),
    )


def _realistic_specs(scale: Scale) -> tuple[KeySpec, ...]:
    """A transformer state dict: ``layers`` identical blocks plus embed/norm/head.

    The block count is chosen so the key total lands on ``scale.keys``. Only a
    count of the form ``9n + 3`` lands on it exactly; every preset is within
    five keys.
    """
    width = scale.width
    hidden, ffn = width.hidden, width.ffn
    query = width.heads * width.head_dim
    kv = width.kv_heads * width.head_dim
    layers = max(1, round((scale.keys - _NON_LAYER_KEYS) / _LAYER_KEYS))

    specs = [KeySpec("model.embed_tokens.weight", (width.vocab, hidden), 0)]
    for index in range(layers):
        block = f"model.layers.{index}"
        specs += [
            KeySpec(f"{block}.input_layernorm.weight", (hidden,), None),
            KeySpec(f"{block}.self_attn.q_proj.weight", (query, hidden), 0),
            KeySpec(f"{block}.self_attn.k_proj.weight", (kv, hidden), 0),
            KeySpec(f"{block}.self_attn.v_proj.weight", (kv, hidden), 0),
            # row-parallel: inference splits the input dimension, the trainer dim 0
            KeySpec(f"{block}.self_attn.o_proj.weight", (hidden, query), 1),
            KeySpec(f"{block}.post_attention_layernorm.weight", (hidden,), None),
            KeySpec(f"{block}.mlp.gate_proj.weight", (ffn, hidden), 0),
            KeySpec(f"{block}.mlp.up_proj.weight", (ffn, hidden), 0),
            # row-parallel
            KeySpec(f"{block}.mlp.down_proj.weight", (hidden, ffn), 1),
        ]
    specs += [
        KeySpec("model.norm.weight", (hidden,), None),
        KeySpec("lm_head.weight", (width.vocab, hidden), 0),
    ]
    return tuple(specs)


def key_specs(scale: Scale) -> tuple[KeySpec, ...]:
    """Every storage key of the workload, in the layout ``scale`` selects.

    ``uniform`` repeats one 1-D key, sharded on the same dimension by both
    sides, so a requester shard meets only the source shards beside it and all
    keys share a layout. ``realistic`` varies shape and generator placement per
    tensor, which is what puts cross-axis resharding, replicated publisher
    reads, and byte skew into the plan build.
    """
    if scale.layout == "uniform":
        extent = math.lcm(scale.source_ranks, scale.generator_shards)
        return tuple(
            KeySpec(f"model.weight.{index}", (extent,), 0)
            for index in range(scale.keys)
        )
    return _realistic_specs(scale)


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
    parser.add_argument("--layout", choices=LAYOUTS, default="uniform")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--metrics",
        choices=(*MEASURED_METRICS, "all"),
        default="cpu",
    )


def _in_layout(case: str, scale: Scale, layout: str) -> tuple[str, Scale]:
    placed = replace(scale, layout=layout)
    return case, replace(placed, keys=len(key_specs(placed)))


def resolve_cases(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[tuple[str, Scale]]:
    """The (name, scale) pairs to run, from a preset or explicit overrides.

    ``keys`` comes back as the count the chosen layout actually generates, so
    the per-metric size caps see the workload that will run.
    """
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.preset == "suite":
        return [_in_layout(case, PRESETS[case], args.layout) for case in _DEFAULT_SUITE]

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
        width=preset.width,
    )
    for name in ("keys", "source_ranks", "generators", "generator_shards"):
        if getattr(scale, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if scale.generators < scale.generator_shards:
        parser.error("--generators must be at least --generator-shards")
    if scale.generators % scale.generator_shards:
        parser.error("--generators must be divisible by --generator-shards")
    return [_in_layout(args.preset, scale, args.layout)]


def tables(
    selection: str,
    cases: Sequence[tuple[str, Scale]],
) -> Iterator[tuple[str, list[tuple[str, Scale]]]]:
    """Each metric with the cases it is measured at, skipping empty tables."""
    for metric in _metrics_for(selection):
        limited = _cases_for(metric, cases)
        if limited:
            yield metric, limited
