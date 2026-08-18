"""Benchmark dedup control-plane planning at a peak synchronized burst.

Run from the repository root::

    .venv/bin/python -m realsim.tools.benchmark_dedup_control
    .venv/bin/python -m realsim.tools.benchmark_dedup_control --preset planned-8b

The workload uses metadata-only TorchStore requests. Runtime and traced Python peak
allocation are measured in separate passes. Payload allocation, transport, native
allocators, process RSS, and simulated transfer time are outside the measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

# Keep TorchStore's import-time setup warning out of the tabular report.
os.environ.setdefault("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", "910737418240")

from dedup_sim.control._sensor import Asked, DedupDirectorySensor, Routed
from dedup_sim.control.routing import Dedup
from proposed import Endpoint, Environment
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request

__all__ = ["main"]

_T = TypeVar("_T")
_MAX_INDEXED_ENTRIES = 1_000_000
_MAX_REGION_CHECKS = 150_000_000


@dataclass(frozen=True)
class _Scale:
    keys: int
    source_ranks: int
    generators: int


_PRESETS = {
    "smoke": _Scale(keys=8, source_ranks=2, generators=8),
    "current-test": _Scale(keys=1, source_ranks=1, generators=64),
    "planned-8b": _Scale(keys=290, source_ranks=4, generators=16),
    "dense-70b": _Scale(keys=723, source_ranks=64, generators=128),
    "fleet-worst": _Scale(keys=5_203, source_ranks=128, generators=512),
}


class _StaticDirectory:
    def __init__(self, requests: Sequence[Request], sources: Sequence[str]) -> None:
        info = StorageInfo(ObjectType.from_request(requests[0]), {None})
        self._located = {
            request.key: {source: info for source in sources} for request in requests
        }

    def locate_raw(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, StorageInfo]]:
        return {key: self._located[key] for key in keys if key in self._located}


class _Profile:
    def read_time(self, src: Endpoint, dst: Endpoint, nbytes: int) -> float:
        if src.id == dst.id:
            return 0.0
        return 10.0 if src.id.startswith("source-") else 1.0


class _Workload:
    def __init__(self, args: argparse.Namespace) -> None:
        self.requests = tuple(
            Request.from_any(f"model.weight.{index}", None).meta_only()
            for index in range(args.keys)
        )
        self.sources = tuple(f"source-{index}" for index in range(args.source_ranks))
        self.generators = tuple(
            f"generator-{index}" for index in range(args.generators)
        )
        self.probe = "probe"
        topology = {
            volume: Endpoint(id=volume, host=volume, node=volume)
            for volume in (*self.sources, *self.generators, self.probe)
        }
        self.directory = DedupDirectorySensor(
            _StaticDirectory(self.requests, self.sources)
        )
        self.plane = Dedup(fanout_cap=args.fanout_cap).attach(
            Environment(topology, _Profile()),
            {DedupDirectorySensor: self.directory},
        )
        assert self.plane.dispatcher is not None
        self.regions = tuple((request.key, None) for request in self.requests)

    def build_pending(self) -> None:
        assert self.plane.dispatcher is not None
        for index, generator in enumerate(self.generators):
            source = self.sources[index % len(self.sources)]
            self.plane.dispatcher.dispatch_sync(Asked(generator, self.requests))
            self.plane.dispatcher.dispatch_sync(
                Routed(
                    requester=generator,
                    sources=(source,),
                    required=((source, self.regions),),
                )
            )

    def snapshot(self) -> None:
        with self.directory.pinned([request.key for request in self.requests]):
            pass

    def serving_sources(self):
        return self.directory.serving_sources(self.requests)

    def plan_fetch(self):
        order = (*self.generators, *self.sources)
        return self.directory.plan_fetch(self.requests, order, requester=self.probe)

    def decide(self) -> None:
        asyncio.run(self.plane._decide(self.requests, self.probe))


@dataclass(frozen=True)
class _Runtime:
    pending_build_ms: float
    snapshot_ms: float
    serving_sources_ms: float
    plan_fetch_ms: float
    full_decision_ms: float
    candidates: int
    pending_candidates: int
    selected_sources: int


@dataclass(frozen=True)
class _Memory:
    pending_build_python_peak_kib: float
    snapshot_python_peak_kib: float
    serving_sources_python_peak_kib: float
    plan_fetch_python_peak_kib: float
    full_decision_python_peak_kib: float


def _median_ms(
    operation: Callable[[], _T], *, warmups: int, repeats: int
) -> tuple[float, _T]:
    result: _T
    for _ in range(warmups):
        result = operation()
    samples = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            result = operation()
            samples.append((time.perf_counter() - started) * 1_000)
    finally:
        if enabled:
            gc.enable()
    return statistics.median(samples), result


def _python_peak_kib(operation: Callable[[], object]) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        baseline, _peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        operation()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return (peak - baseline) / 1024


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(_PRESETS), default="smoke")
    parser.add_argument("--keys", type=int)
    parser.add_argument("--source-ranks", type=int)
    parser.add_argument("--generators", type=int)
    parser.add_argument("--fanout-cap", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="permit workloads above the default memory or work guard",
    )
    args = parser.parse_args(argv)
    customized = any(
        value is not None for value in (args.keys, args.source_ranks, args.generators)
    )
    preset = _PRESETS[args.preset]
    args.keys = preset.keys if args.keys is None else args.keys
    args.source_ranks = (
        preset.source_ranks if args.source_ranks is None else args.source_ranks
    )
    args.generators = preset.generators if args.generators is None else args.generators
    args.case = "custom" if customized else args.preset
    for name in ("keys", "source_ranks", "generators", "fanout_cap", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    indexed_entries = args.keys * (args.source_ranks + 3 * args.generators)
    region_checks = args.keys * args.keys * (args.source_ranks + args.generators)
    if not args.allow_large and (
        indexed_entries > _MAX_INDEXED_ENTRIES or region_checks > _MAX_REGION_CHECKS
    ):
        parser.error(
            "workload exceeds the default allocation/work guard; use "
            "--allow-large on a suitably sized host"
        )
    return args


def _runtime(args: argparse.Namespace) -> _Runtime:
    workload = _Workload(args)
    gc.collect()
    started = time.perf_counter()
    workload.build_pending()
    pending_build_ms = (time.perf_counter() - started) * 1_000
    snapshot_ms, _ = _median_ms(
        workload.snapshot, warmups=args.warmups, repeats=args.repeats
    )
    with workload.directory.pinned([request.key for request in workload.requests]):
        serving_ms, serving = _median_ms(
            workload.serving_sources,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        plan_ms, fetch = _median_ms(
            workload.plan_fetch,
            warmups=args.warmups,
            repeats=args.repeats,
        )
    gc.collect()
    started = time.perf_counter()
    workload.decide()
    decision_ms = (time.perf_counter() - started) * 1_000
    candidates, pending = serving
    return _Runtime(
        pending_build_ms,
        snapshot_ms,
        serving_ms,
        plan_ms,
        decision_ms,
        len(candidates),
        len(pending),
        len(fetch.sources),
    )


def _memory(args: argparse.Namespace) -> _Memory:
    workload = _Workload(args)
    pending_build = _python_peak_kib(workload.build_pending)
    snapshot = _python_peak_kib(workload.snapshot)
    with workload.directory.pinned([request.key for request in workload.requests]):
        serving = _python_peak_kib(workload.serving_sources)
        plan = _python_peak_kib(workload.plan_fetch)
    decision = _python_peak_kib(workload.decide)
    return _Memory(pending_build, snapshot, serving, plan, decision)


def _run(args: argparse.Namespace) -> None:
    runtime = _runtime(args)
    memory = _memory(args)
    indexed_entries = args.keys * (args.source_ranks + 3 * args.generators)
    print(
        "case\tkeys\tsource_ranks\tgenerators\tindexed_metadata_entries\t"
        "pending_build_ms\tsnapshot_ms\tserving_sources_ms\tplan_fetch_ms\t"
        "full_decision_ms\tpending_build_python_peak_kib\t"
        "snapshot_python_peak_kib\tserving_sources_python_peak_kib\t"
        "plan_fetch_python_peak_kib\tfull_decision_python_peak_kib\t"
        "candidates\tpending_candidates\tselected_sources"
    )
    print(
        f"{args.case}\t{args.keys}\t{args.source_ranks}\t{args.generators}\t"
        f"{indexed_entries}\t{runtime.pending_build_ms:.3f}\t"
        f"{runtime.snapshot_ms:.3f}\t{runtime.serving_sources_ms:.3f}\t"
        f"{runtime.plan_fetch_ms:.3f}\t{runtime.full_decision_ms:.3f}\t"
        f"{memory.pending_build_python_peak_kib:.3f}\t"
        f"{memory.snapshot_python_peak_kib:.3f}\t"
        f"{memory.serving_sources_python_peak_kib:.3f}\t"
        f"{memory.plan_fetch_python_peak_kib:.3f}\t"
        f"{memory.full_decision_python_peak_kib:.3f}\t"
        f"{runtime.candidates}\t{runtime.pending_candidates}\t"
        f"{runtime.selected_sources}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one synthetic scale point and print a tab-separated result."""
    args = _arguments(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
