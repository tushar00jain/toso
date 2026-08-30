"""Benchmark one complete weight-sync control-plane lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from realsim.tools._benchmark_harness import (
    _DEFAULT_SUITE,
    PRESETS as _PRESETS,
    Scale as _Scale,
    Timing as _Timing,
    duration as _duration,
    instruction_count,
    measurable_rows as _measurable_rows,
    measured,
    memory_kib,
    memory_size as _memory_size,
    metadata_rows as _metadata_rows,
    shard_geometry,
    tables,
    timed as _timed,
    worker_thread,
)
from sim_common import config

__all__ = ["main"]

_MAX_METADATA_ROWS = 1_000_000


_VARIANTS = ("legacy", "legacy-dedup", "indexed-dedup")
_VARIANT_TITLES = {
    "legacy": "Legacy controller, no dedupe",
    "legacy-dedup": "Legacy controller + dedupe",
    "indexed-dedup": "Indexed controller + dedupe",
}


class _Profile:
    def read_time(self, src: Any, dst: Any, nbytes: int) -> float:
        del nbytes
        if src.id == dst.id:
            return 0.0
        return 10.0 if src.id.startswith("trainer-") else 1.0


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
    trainer_publish_cpu_ms: float | None
    generator_lookups_cpu_ms: float | None
    generator_completions_cpu_ms: float | None
    total_cpu_ms: float | None
    total_wall_ms: float | None
    total_instructions: int | None
    peak_python_kib: float | None


class _Workload:
    def __init__(self, scale: _Scale, variant: str, fanout_cap: int) -> None:
        from dedup_sim.control._sensor import DedupDirectorySensor, Published
        from dedup_sim.control.routing import Dedup
        from proposed import Endpoint, Environment
        from realsim.adapters.real_controller import make_controller_adapter
        from torchstore.transport import Request, TensorSlice

        self.scale = scale
        self.variant = variant
        (
            self.trainers,
            self.generators,
            self.trainer_requests,
            self.generator_requests,
        ) = _request_batches(
            scale,
            request_type=Request,
            tensor_slice_type=TensorSlice,
        )
        backend = "indexed" if variant == "indexed-dedup" else "legacy"
        with config.overrides(controller_backend=backend):
            self.service = make_controller_adapter().service
        self.plane: Any | None = None
        self.runner: asyncio.Runner | None = None
        self._published_event = Published
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

    def publish_trainers(self) -> None:
        for trainer, requests in zip(self.trainers, self.trainer_requests):
            self.service.notify_put_batch(requests, trainer, pending=False)

    def generator_lookups(self) -> list[Any]:
        if self.variant == "legacy":
            for requests in self.generator_requests:
                self.service.locate_volumes([request.key for request in requests])
            return []
        assert self.plane is not None
        assert self.runner is not None
        plane = self.plane

        async def decide_all() -> list[Any]:
            return [
                await plane._decide(requests, generator)
                for generator, requests in zip(
                    self.generators, self.generator_requests
                )
            ]

        return self.runner.run(decide_all())

    def complete_generators(self, plans: Sequence[Any]) -> None:
        if self.variant == "legacy":
            return
        assert self.plane is not None
        for generator, requests, plan in zip(
            self.generators, self.generator_requests, plans
        ):
            self.service.notify_put_batch(requests, generator, pending=False)
            self.plane.dispatcher.dispatch_sync(
                self._published_event(plan.publication)
            )

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()


def _request_batches(
    scale: _Scale,
    *,
    request_type: Any,
    tensor_slice_type: Any,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[Any, ...], tuple[Any, ...]]:
    trainers = tuple(f"trainer-{index}" for index in range(scale.source_ranks))
    generators = tuple(f"generator-{index}" for index in range(scale.generators))
    extent, bounds = shard_geometry(scale)

    def requests_for_shard(rank: int, shards: int) -> tuple[Any, ...]:
        offset, width = bounds(rank, shards)
        tensor_slice = tensor_slice_type(
            offsets=(offset,),
            coordinates=(rank,),
            global_shape=(extent,),
            local_shape=(width,),
            mesh_shape=(shards,),
        )
        return tuple(
            request_type.from_tensor_slice(
                f"model.weight.{index}", tensor_slice
            ).meta_only()
            for index in range(scale.keys)
        )

    trainer_requests = tuple(
        requests_for_shard(rank, scale.source_ranks)
        for rank in range(scale.source_ranks)
    )
    generator_requests = tuple(
        requests_for_shard(rank % scale.generator_shards, scale.generator_shards)
        for rank in range(scale.generators)
    )
    return trainers, generators, trainer_requests, generator_requests


def _run_endpoint_body(endpoint: Any, controller: Any, *args: Any) -> Any:
    """Run an old async endpoint body locally, without Monarch or RPC."""
    coroutine = endpoint._method(controller, *args)
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    finally:
        coroutine.close()
    raise RuntimeError("historical controller endpoint unexpectedly suspended")


class _HistoricalWorkload:
    """The same controller-only lifecycle against a pre-dedupe TorchStore checkout."""

    variant = "legacy"

    def __init__(self, scale: _Scale) -> None:
        from torchstore.controller import Controller
        from torchstore.transport import Request, TensorSlice

        self.controller = Controller()
        self.controller.is_initialized = True
        self._controller_type = Controller
        (
            self.trainers,
            self.generators,
            self.trainer_requests,
            self.generator_requests,
        ) = _request_batches(
            scale,
            request_type=Request,
            tensor_slice_type=TensorSlice,
        )

    def publish_trainers(self) -> None:
        endpoint = self._controller_type.__dict__["notify_put_batch"]
        for trainer, requests in zip(self.trainers, self.trainer_requests):
            _run_endpoint_body(endpoint, self.controller, list(requests), trainer)

    def generator_lookups(self) -> list[Any]:
        endpoint = self._controller_type.__dict__["locate_volumes"]
        for requests in self.generator_requests:
            _run_endpoint_body(
                endpoint,
                self.controller,
                [request.key for request in requests],
            )
        return []

    def complete_generators(self, plans: Sequence[Any]) -> None:
        assert not plans

    def close(self) -> None:
        pass


def _workload(
    scale: _Scale,
    variant: str,
    fanout_cap: int,
    historical: bool,
) -> _Workload | _HistoricalWorkload:
    if historical:
        assert variant == "legacy"
        return _HistoricalWorkload(scale)
    return _Workload(scale, variant, fanout_cap)


def _sample(
    scale: _Scale,
    variant: str,
    fanout_cap: int,
    historical: bool = False,
) -> _Sample:
    workload = _workload(scale, variant, fanout_cap, historical)
    workload.publish_trainers()
    with measured():
        try:
            total, (trainer_publish, generator_lookups, generator_completions) = _timed(
                lambda: _lifecycle_timings(workload, variant)
            )
        finally:
            workload.close()
    return _Sample(
        trainer_publish,
        generator_lookups,
        generator_completions,
        total,
    )


def _lifecycle_timings(
    workload: _Workload | _HistoricalWorkload,
    variant: str,
) -> tuple[_Timing, _Timing, _Timing]:
    trainer_publish, _ = _timed(workload.publish_trainers)
    generator_lookups, plans = _timed(workload.generator_lookups)
    if variant == "legacy":
        generator_completions = _Timing(0, 0)
    else:
        generator_completions, _ = _timed(lambda: workload.complete_generators(plans))
    return trainer_publish, generator_lookups, generator_completions


def _execute_lifecycle(workload: _Workload | _HistoricalWorkload) -> None:
    workload.publish_trainers()
    plans = workload.generator_lookups()
    workload.complete_generators(plans)


def _instruction_sample(
    scale: _Scale,
    variant: str,
    fanout_cap: int,
    historical: bool = False,
) -> int | None:
    workload = _workload(scale, variant, fanout_cap, historical)
    workload.publish_trainers()
    try:
        return instruction_count(lambda: _execute_lifecycle(workload))
    finally:
        workload.close()


def _memory_sample(
    scale: _Scale,
    variant: str,
    fanout_cap: int,
    historical: bool = False,
) -> float:
    workload = _workload(scale, variant, fanout_cap, historical)
    workload.publish_trainers()
    try:
        return memory_kib(lambda: _execute_lifecycle(workload))
    finally:
        workload.close()


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
    measure_cpu: bool = True,
    measure_instructions: bool = True,
    measure_memory: bool = True,
    historical: bool = False,
    progress: Callable[[str], None] | None = None,
) -> _Result:
    samples: list[_Sample] = []
    total_instructions = None
    peak_python_kib = None
    with worker_thread("weight-sync-controller") as worker:
        if measure_cpu:
            if progress is not None:
                progress("CPU timing")
            for _ in range(warmups):
                worker.submit(
                    _sample, scale, variant, fanout_cap, historical
                ).result()
            samples = [
                worker.submit(
                    _sample, scale, variant, fanout_cap, historical
                ).result()
                for _ in range(repeats)
            ]
        if measure_instructions:
            if progress is not None:
                progress("retired instructions")
            total_instructions = worker.submit(
                _instruction_sample,
                scale,
                variant,
                fanout_cap,
                historical,
            ).result()
        if measure_memory:
            if progress is not None:
                progress("peak Python memory")
            peak_python_kib = worker.submit(
                _memory_sample,
                scale,
                variant,
                fanout_cap,
                historical,
            ).result()
    return _Result(
        case=case,
        variant=variant,
        scale=scale,
        trainer_publish_cpu_ms=(
            _median_ms(samples, "trainer_publish") if samples else None
        ),
        generator_lookups_cpu_ms=(
            _median_ms(samples, "generator_lookups") if samples else None
        ),
        generator_completions_cpu_ms=(
            _median_ms(samples, "generator_completions") if samples else None
        ),
        total_cpu_ms=_median_ms(samples, "total") if samples else None,
        total_wall_ms=(
            _median_ms(samples, "total", "wall_ns") if samples else None
        ),
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
    parser.add_argument(
        "--metrics",
        choices=("cpu", "instructions", "memory", "all"),
        default="cpu",
    )
    parser.add_argument(
        "--historical-torchstore-root",
        type=Path,
        help=(
            "run the legacy-only benchmark against an older TorchStore checkout; "
            "the checkout must predate the pending-publication API"
        ),
    )
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args(argv)
    if args.historical_torchstore_root is not None:
        root = args.historical_torchstore_root.resolve()
        if not (root / "torchstore" / "controller.py").is_file():
            parser.error(
                "--historical-torchstore-root must contain torchstore/controller.py"
            )
        if args.variant not in ("all", "legacy"):
            parser.error("a historical TorchStore checkout supports only legacy")
        args.historical_torchstore_root = root
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
    if args.metrics in ("memory", "instructions"):
        limited = tuple(
            (case, scale)
            for case, scale in args.cases
            if _metadata_rows(scale) <= _measurable_rows(args.metrics)
        )
        if not limited:
            parser.error(f"{args.metrics} measurement is capped below this scale")
        args.cases = limited
    args.variants = (
        ("legacy",)
        if args.historical_torchstore_root is not None
        else (_VARIANTS if args.variant == "all" else (args.variant,))
    )
    return args


def _print_table_header(variant: str, metrics: str, historical: bool) -> None:
    title = (
        "Historical legacy controller, no dedupe"
        if historical
        else _VARIANT_TITLES[variant]
    )
    print(f"## {title}", flush=True)
    print(flush=True)
    if metrics == "cpu":
        print(
            "| Workload | Trainer publish CPU | G lookups CPU | "
            "G completions CPU | Total CPU |",
            flush=True,
        )
        print("| --- | ---: | ---: | ---: | ---: |", flush=True)
    elif metrics == "instructions":
        print("| Workload | Retired instructions |", flush=True)
        print("| --- | ---: |", flush=True)
    elif metrics == "memory":
        print("| Workload | Peak Python memory |", flush=True)
        print("| --- | ---: |", flush=True)
    else:
        print(
            "| Workload | Trainer publish CPU | G lookups CPU | "
            "G completions CPU | Total CPU | Retired instructions | "
            "Peak Python memory |",
            flush=True,
        )
        print(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            flush=True,
        )


def _print_result(result: _Result, metrics: str) -> None:
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
    if metrics == "cpu":
        row = (
            f"| `{result.case}` | "
            f"{_duration(result.trainer_publish_cpu_ms)} | "
            f"{_duration(result.generator_lookups_cpu_ms)} | "
            f"{completion} | {_duration(result.total_cpu_ms)} |"
        )
    elif metrics == "instructions":
        row = f"| `{result.case}` | {instructions} |"
    elif metrics == "memory":
        row = f"| `{result.case}` | {_memory_size(result.peak_python_kib)} |"
    else:
        row = (
            f"| `{result.case}` | "
            f"{_duration(result.trainer_publish_cpu_ms)} | "
            f"{_duration(result.generator_lookups_cpu_ms)} | "
            f"{completion} | {_duration(result.total_cpu_ms)} | "
            f"{instructions} | {_memory_size(result.peak_python_kib)} |"
        )
    print(
        row,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected comparison cases and print Markdown tables."""
    args = _arguments(argv)
    historical = args.historical_torchstore_root is not None
    if historical:
        imported = tuple(
            name
            for name in sys.modules
            if name == "torchstore" or name.startswith("torchstore.")
        )
        if imported:
            raise RuntimeError(
                "historical mode must start before TorchStore is imported; "
                "run this benchmark as a fresh Python process"
            )
        sys.path.insert(0, str(args.historical_torchstore_root))
    else:
        config.configure()
    for variant in args.variants:
        for metric, metric_cases in tables(args.metrics, args.cases):
            _print_table_header(variant, metric, historical)
            for case, scale in metric_cases:
                def progress(phase: str) -> None:
                    prefix = "historical-" if historical else ""
                    print(
                        f"[{prefix}{variant}/{case}] {phase}",
                        file=sys.stderr,
                        flush=True,
                    )

                result = _run_case(
                    case,
                    scale,
                    variant,
                    fanout_cap=args.fanout_cap,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    measure_cpu=metric == "cpu",
                    measure_instructions=metric == "instructions",
                    measure_memory=metric == "memory",
                    historical=historical,
                    progress=progress,
                )
                _print_result(result, metric)
            print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
