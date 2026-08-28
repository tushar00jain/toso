"""Benchmark one complete weight-sync control-plane lifecycle."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from dedup_sim.control._sensor import DedupDirectorySensor, Published
from dedup_sim.control.routing import Dedup, ReadPlan
from proposed import Endpoint, Environment
from realsim.adapters.real_controller import make_controller_adapter
from sim_common import config
from sim_common.perfcount import InstructionCount
from torchstore import coverage
from torchstore.transport import Request, TensorSlice

__all__ = ["main"]

_T = TypeVar("_T")
_MAX_METADATA_ROWS = 1_000_000


@dataclass(frozen=True)
class _Scale:
    keys: int
    source_ranks: int
    generators: int
    generator_shards: int


_PRESETS = {
    "smoke": _Scale(keys=8, source_ranks=2, generators=8, generator_shards=2),
    "1b": _Scale(keys=120, source_ranks=2, generators=8, generator_shards=2),
    "8b": _Scale(keys=290, source_ranks=8, generators=16, generator_shards=8),
    "qwen-27b": _Scale(
        keys=1_199,
        source_ranks=8,
        generators=8,
        generator_shards=4,
    ),
    "70b": _Scale(keys=723, source_ranks=8, generators=64, generator_shards=8),
    "70b-wide": _Scale(
        keys=723,
        source_ranks=64,
        generators=128,
        generator_shards=64,
    ),
    "405b": _Scale(
        keys=1_500,
        source_ranks=32,
        generators=128,
        generator_shards=32,
    ),
    "moe": _Scale(
        keys=3_000,
        source_ranks=32,
        generators=128,
        generator_shards=32,
    ),
    "kimi-k2": _Scale(
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
_VARIANTS = ("legacy", "legacy-dedup", "indexed-dedup")
_VARIANT_TITLES = {
    "legacy": "Legacy controller, no dedupe",
    "legacy-dedup": "Legacy controller + dedupe",
    "indexed-dedup": "Indexed controller + dedupe",
}


class _Profile:
    def read_time(self, src: Endpoint, dst: Endpoint, nbytes: int) -> float:
        del nbytes
        if src.id == dst.id:
            return 0.0
        return 10.0 if src.id.startswith("trainer-") else 1.0


@dataclass(frozen=True)
class _Timing:
    cpu_ns: int
    wall_ns: int


@dataclass(frozen=True)
class _Sample:
    trainer_publish: _Timing
    generator_lookups: _Timing
    generator_completions: _Timing
    total: _Timing


@dataclass(frozen=True)
class _Result:
    case: str
    variant: str
    scale: _Scale
    trainer_publish_cpu_ms: float
    generator_lookups_cpu_ms: float
    generator_completions_cpu_ms: float
    total_cpu_ms: float
    total_wall_ms: float
    total_instructions: int | None
    peak_python_kib: float


class _Workload:
    def __init__(self, scale: _Scale, variant: str, fanout_cap: int) -> None:
        self.scale = scale
        self.variant = variant
        self.trainers = tuple(
            f"trainer-{index}" for index in range(scale.source_ranks)
        )
        self.generators = tuple(
            f"generator-{index}" for index in range(scale.generators)
        )
        extent = math.lcm(scale.source_ranks, scale.generator_shards)
        self.trainer_requests = tuple(
            self._requests_for_shard(rank, scale.source_ranks, extent)
            for rank in range(scale.source_ranks)
        )
        self.generator_requests = tuple(
            self._requests_for_shard(
                rank % scale.generator_shards,
                scale.generator_shards,
                extent,
            )
            for rank in range(scale.generators)
        )
        backend = "indexed" if variant == "indexed-dedup" else "legacy"
        with config.overrides(controller_backend=backend):
            self.service = make_controller_adapter().service
        self.plane: Dedup | None = None
        self.runner: asyncio.Runner | None = None
        if variant != "legacy":
            topology = {
                volume: Endpoint(volume, volume, volume)
                for volume in (*self.trainers, *self.generators)
            }
            directory = DedupDirectorySensor(self.service)
            self.plane = Dedup(fanout_cap=fanout_cap).attach(
                Environment(topology, _Profile()),
                {DedupDirectorySensor: directory},
            )
            self.runner = asyncio.Runner()
            self.runner.get_loop()

    def _requests_for_shard(
        self,
        rank: int,
        shards: int,
        extent: int,
    ) -> tuple[Request, ...]:
        width = extent // shards
        tensor_slice = TensorSlice(
            offsets=(rank * width,),
            coordinates=(rank,),
            global_shape=(extent,),
            local_shape=(width,),
            mesh_shape=(shards,),
        )
        return tuple(
            Request.from_tensor_slice(
                f"model.weight.{index}", tensor_slice
            ).meta_only()
            for index in range(self.scale.keys)
        )

    def publish_trainers(self) -> None:
        for trainer, requests in zip(self.trainers, self.trainer_requests):
            self.service.notify_put_batch(requests, trainer, pending=False)

    def generator_lookups(self) -> list[ReadPlan]:
        if self.variant == "legacy":
            for requests in self.generator_requests:
                volume_maps = self.service.locate_volumes(
                    [request.key for request in requests]
                )
                coverage.cover(requests, volume_maps)
            return []
        assert self.plane is not None
        assert self.runner is not None

        async def decide_all() -> list[ReadPlan]:
            return [
                await self.plane._decide(requests, generator)
                for generator, requests in zip(
                    self.generators, self.generator_requests
                )
            ]

        return self.runner.run(decide_all())

    def complete_generators(self, plans: Sequence[ReadPlan]) -> None:
        if self.variant == "legacy":
            return
        assert self.plane is not None
        for generator, requests, plan in zip(
            self.generators, self.generator_requests, plans
        ):
            self.service.notify_put_batch(requests, generator, pending=False)
            self.plane.dispatcher.dispatch_sync(Published(plan.publication))

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()


def _timed(operation: Callable[[], _T]) -> tuple[_Timing, _T]:
    wall_started = time.perf_counter_ns()
    cpu_started = time.thread_time_ns()
    result = operation()
    cpu_ns = time.thread_time_ns() - cpu_started
    wall_ns = time.perf_counter_ns() - wall_started
    return _Timing(cpu_ns, wall_ns), result


def _sample(scale: _Scale, variant: str, fanout_cap: int) -> _Sample:
    workload = _Workload(scale, variant, fanout_cap)
    workload.publish_trainers()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        total_wall_started = time.perf_counter_ns()
        total_cpu_started = time.thread_time_ns()
        trainer_publish, _ = _timed(workload.publish_trainers)
        generator_lookups, plans = _timed(workload.generator_lookups)
        if variant == "legacy":
            generator_completions = _Timing(0, 0)
        else:
            generator_completions, _ = _timed(
                lambda: workload.complete_generators(plans)
            )
        total = _Timing(
            time.thread_time_ns() - total_cpu_started,
            time.perf_counter_ns() - total_wall_started,
        )
    finally:
        if enabled:
            gc.enable()
        workload.close()
    return _Sample(
        trainer_publish,
        generator_lookups,
        generator_completions,
        total,
    )


def _execute_lifecycle(workload: _Workload) -> None:
    workload.publish_trainers()
    plans = workload.generator_lookups()
    workload.complete_generators(plans)


def _instruction_sample(
    scale: _Scale, variant: str, fanout_cap: int
) -> int | None:
    if not InstructionCount.available():
        return None
    workload = _Workload(scale, variant, fanout_cap)
    workload.publish_trainers()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        with InstructionCount() as counter:
            _execute_lifecycle(workload)
    finally:
        if enabled:
            gc.enable()
        workload.close()
    return counter.count


def _memory_sample(scale: _Scale, variant: str, fanout_cap: int) -> float:
    workload = _Workload(scale, variant, fanout_cap)
    workload.publish_trainers()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    tracemalloc.start()
    try:
        baseline, _peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        _execute_lifecycle(workload)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        if enabled:
            gc.enable()
        workload.close()
    return (peak - baseline) / 1024


def _median_ms(samples: Sequence[_Sample], field: str, clock: str = "cpu_ns") -> float:
    return statistics.median(
        getattr(getattr(sample, field), clock) for sample in samples
    ) / 1_000_000


def _run_case(
    case: str,
    scale: _Scale,
    variant: str,
    *,
    fanout_cap: int,
    warmups: int,
    repeats: int,
) -> _Result:
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="weight-sync-controller"
    ) as worker:
        for _ in range(warmups):
            worker.submit(_sample, scale, variant, fanout_cap).result()
        samples = [
            worker.submit(_sample, scale, variant, fanout_cap).result()
            for _ in range(repeats)
        ]
        total_instructions = worker.submit(
            _instruction_sample, scale, variant, fanout_cap
        ).result()
        peak_python_kib = worker.submit(
            _memory_sample, scale, variant, fanout_cap
        ).result()
    return _Result(
        case=case,
        variant=variant,
        scale=scale,
        trainer_publish_cpu_ms=_median_ms(samples, "trainer_publish"),
        generator_lookups_cpu_ms=_median_ms(samples, "generator_lookups"),
        generator_completions_cpu_ms=_median_ms(samples, "generator_completions"),
        total_cpu_ms=_median_ms(samples, "total"),
        total_wall_ms=_median_ms(samples, "total", "wall_ns"),
        total_instructions=total_instructions,
        peak_python_kib=peak_python_kib,
    )


def _positive(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "keys",
        "source_ranks",
        "generators",
        "generator_shards",
        "fanout_cap",
        "repeats",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.generators < args.generator_shards:
        parser.error("--generators must be at least --generator-shards")
    if args.generators % args.generator_shards:
        parser.error("--generators must be divisible by --generator-shards")


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=(*_PRESETS, "suite"), default="smoke")
    parser.add_argument("--variant", choices=(*_VARIANTS, "all"), default="all")
    parser.add_argument("--keys", type=int)
    parser.add_argument("--source-ranks", type=int)
    parser.add_argument("--generators", type=int)
    parser.add_argument("--generator-shards", type=int)
    parser.add_argument("--fanout-cap", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args(argv)
    customized = any(
        value is not None
        for value in (
            args.keys,
            args.source_ranks,
            args.generators,
            args.generator_shards,
        )
    )
    if args.preset == "suite" and customized:
        parser.error("custom scale flags cannot be combined with --preset suite")
    if args.preset != "suite":
        preset = _PRESETS[args.preset]
        args.keys = preset.keys if args.keys is None else args.keys
        args.source_ranks = (
            preset.source_ranks if args.source_ranks is None else args.source_ranks
        )
        args.generators = (
            preset.generators if args.generators is None else args.generators
        )
        args.generator_shards = (
            preset.generator_shards
            if args.generator_shards is None
            else args.generator_shards
        )
        _positive(parser, args)
        rows = args.keys * (args.source_ranks + args.generators)
        if not args.allow_large and rows > _MAX_METADATA_ROWS:
            parser.error(
                "workload exceeds the default allocation guard; use --allow-large "
                "on a suitably sized host"
            )
        args.cases = (
            (
                "custom" if customized else args.preset,
                _Scale(
                    args.keys,
                    args.source_ranks,
                    args.generators,
                    args.generator_shards,
                ),
            ),
        )
    else:
        _positive(
            parser,
            argparse.Namespace(
                keys=1,
                source_ranks=1,
                generators=1,
                generator_shards=1,
                fanout_cap=args.fanout_cap,
                repeats=args.repeats,
                warmups=args.warmups,
            ),
        )
        args.cases = tuple((name, _PRESETS[name]) for name in _DEFAULT_SUITE)
    args.variants = _VARIANTS if args.variant == "all" else (args.variant,)
    return args


def _duration(milliseconds: float) -> str:
    if milliseconds < 1_000:
        return f"{milliseconds:.3f} ms"
    return f"{milliseconds / 1_000:.3f} s"


def _memory_size(kibibytes: float) -> str:
    if kibibytes < 1024:
        return f"{kibibytes:.3f} KiB"
    return f"{kibibytes / 1024:.3f} MiB"


def _print_results(results: Sequence[_Result]) -> None:
    for variant in _VARIANTS:
        rows = [result for result in results if result.variant == variant]
        if not rows:
            continue
        print(f"## {_VARIANT_TITLES[variant]}")
        print()
        print(
            "| Workload | Trainer publish CPU | G lookups CPU | "
            "G completions CPU | Total CPU | Retired instructions | "
            "Peak Python memory |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in rows:
            completion = (
                "—"
                if result.variant == "legacy"
                else _duration(result.generator_completions_cpu_ms)
            )
            instructions = (
                "—"
                if result.total_instructions is None
                else f"{result.total_instructions:,}"
            )
            print(
                f"| `{result.case}` | "
                f"{_duration(result.trainer_publish_cpu_ms)} | "
                f"{_duration(result.generator_lookups_cpu_ms)} | "
                f"{completion} | {_duration(result.total_cpu_ms)} | "
                f"{instructions} | {_memory_size(result.peak_python_kib)} |"
            )
        print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected comparison cases and print Markdown tables."""
    config.configure()
    args = _arguments(argv)
    results = [
        _run_case(
            case,
            scale,
            variant,
            fanout_cap=args.fanout_cap,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        for variant in args.variants
        for case, scale in args.cases
    ]
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
