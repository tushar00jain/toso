# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cost of building precomputed routes, and of one rank's lookups against them.

Unlike the controller lifecycle, routing pays once at setup: every rank's slice
metadata is reconciled into per-rank tables, and each update then reads only the
table it already holds. The build column is therefore whole-job work, while the
lookup column is what a single rank spends per update, not multiplied by ``G``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from realsim.tools._benchmark_harness import (
    Scale,
    Timing,
    add_common_arguments,
    duration,
    instruction_count,
    key_specs,
    measured,
    median_ms,
    memory_kib,
    memory_size,
    resolve_cases,
    shard_bounds,
    tables,
    timed,
    worker_thread,
)

__all__ = ["main"]

# bf16, matching the transfer dtype the routing client registers with.
_ELEMENT_SIZE = 2


@dataclass(frozen=True)
class _Result:
    case: str
    scale: Scale
    build_cpu_ms: float | None
    build_wall_ms: float | None
    lookup_cpu_ms: float | None
    transfers: int
    build_instructions: int | None
    peak_python_kib: float | None


class _Workload:
    """One build of the whole plan, then one rank's per-update lookups."""

    def __init__(self, scale: Scale) -> None:
        from torchstore.routing.plan import KeyRegistration, RoutingPlan
        from torchstore.transport import TensorSlice

        self._plan_type = RoutingPlan
        self._planner = scale.planner
        specs = key_specs(scale)
        # The trainer is FSDP2 Shard(0) throughout; only the generator varies.
        trainer_dims = (0,) * len(specs)
        generator_dims = tuple(spec.generator_dim for spec in specs)

        def registrations(
            rank: int,
            shards: int,
            dims: Sequence[int | None],
        ) -> dict[str, Any]:
            entries: dict[str, Any] = {}
            for spec, dim in zip(specs, dims):
                offsets, local_shape = shard_bounds(spec, dim, rank, shards)
                entries[spec.name] = KeyRegistration(
                    TensorSlice(
                        offsets=offsets,
                        coordinates=(0,) if dim is None else (rank,),
                        global_shape=spec.global_shape,
                        local_shape=local_shape,
                        mesh_shape=(1,) if dim is None else (shards,),
                    ),
                    _ELEMENT_SIZE,
                )
            return entries

        self.publishers = {
            f"trainer-{rank}": registrations(rank, scale.source_ranks, trainer_dims)
            for rank in range(scale.source_ranks)
        }
        self.requesters = {
            f"generator-{rank}": registrations(
                rank % scale.generator_shards, scale.generator_shards, generator_dims
            )
            for rank in range(scale.generators)
        }
        self.rank = "generator-0"
        self.keys = tuple(sorted(next(iter(self.requesters.values()))))
        self.plan: Any | None = None

    def build_plan(self) -> Any:
        """Every rank's plan in one process, or just this rank's, in its own."""
        if self._planner == "central":
            self.plan = self._plan_type.build(self.publishers, self.requesters)
        else:
            self.plan = self._plan_type.build_for(
                self.rank, self.publishers, self.requesters
            )
        return self.plan

    def rank_lookups(self) -> None:
        assert self.plan is not None
        lookup = self.plan.lookup
        rank = self.rank
        for key in self.keys:
            lookup(rank, key)

    def transfers(self) -> int:
        assert self.plan is not None
        return sum(
            len(route.transfers)
            for entry in self.plan._local(self.rank).keys.values()
            for route in entry.routes
        )


def _sample(scale: Scale) -> tuple[Timing, Timing, int]:
    workload = _Workload(scale)
    with measured():
        build, _plan = timed(workload.build_plan)
        lookups, _ = timed(workload.rank_lookups)
        transfers = workload.transfers()
    return build, lookups, transfers


def _instruction_sample(scale: Scale) -> int | None:
    return instruction_count(_Workload(scale).build_plan)


def _memory_sample(scale: Scale) -> float:
    return memory_kib(_Workload(scale).build_plan)


def _run_case(
    case: str,
    scale: Scale,
    *,
    metric: str,
    warmups: int,
    repeats: int,
    progress: Callable[[str], None] | None = None,
) -> _Result:
    builds: list[Timing] = []
    lookups: list[Timing] = []
    transfers = 0
    build_instructions = None
    peak_python_kib = None
    with worker_thread("routing-plan") as worker:
        if progress is not None:
            progress(metric)
        if metric == "cpu":
            for _ in range(warmups):
                worker.submit(_sample, scale).result()
            for _ in range(repeats):
                build, lookup, transfers = worker.submit(_sample, scale).result()
                builds.append(build)
                lookups.append(lookup)
        elif metric == "instructions":
            build_instructions = worker.submit(_instruction_sample, scale).result()
        else:
            peak_python_kib = worker.submit(_memory_sample, scale).result()
    return _Result(
        case=case,
        scale=scale,
        build_cpu_ms=median_ms([x.cpu_ns for x in builds]) if builds else None,
        build_wall_ms=median_ms([x.wall_ns for x in builds]) if builds else None,
        lookup_cpu_ms=median_ms([x.cpu_ns for x in lookups]) if lookups else None,
        transfers=transfers,
        build_instructions=build_instructions,
        peak_python_kib=peak_python_kib,
    )


def _print_table_header(metric: str) -> None:
    print("## Precomputed routing plan", flush=True)
    print(flush=True)
    if metric == "cpu":
        print(
            "| Model | Plan build CPU | Plan build wall | Per-rank lookups CPU "
            "| Per-rank transfers |",
            flush=True,
        )
        print("| --- | ---: | ---: | ---: | ---: |", flush=True)
    elif metric == "instructions":
        print("| Model | Plan build retired instructions |", flush=True)
        print("| --- | ---: |", flush=True)
    else:
        print("| Model | Plan build peak Python memory |", flush=True)
        print("| --- | ---: |", flush=True)


def _print_result(result: _Result, metric: str) -> None:
    if metric == "cpu":
        row = (
            f"| `{result.case}` | {duration(result.build_cpu_ms)} "
            f"| {duration(result.build_wall_ms)} "
            f"| {duration(result.lookup_cpu_ms)} "
            f"| {result.transfers:,} |"
        )
    elif metric == "instructions":
        count = (
            "—"
            if result.build_instructions is None
            else f"{result.build_instructions:,}"
        )
        row = f"| `{result.case}` | {count} |"
    else:
        row = f"| `{result.case}` | {memory_size(result.peak_python_kib)} |"
    print(row, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected cases and print one Markdown table per metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args(argv)
    cases = resolve_cases(parser, args)

    for metric, metric_cases in tables(args.metrics, cases, args.allow_large):
        _print_table_header(metric)
        for case, scale in metric_cases:

            def progress(phase: str) -> None:
                print(f"[routing/{case}] {phase}", file=sys.stderr, flush=True)

            _print_result(
                _run_case(
                    case,
                    scale,
                    metric=metric,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    progress=progress,
                ),
                metric,
            )
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
