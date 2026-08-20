"""Benchmark declare, union, rank, and gate at a synchronized dedup burst."""

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

os.environ.setdefault("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", "910737418240")

from dedup_sim.control._sensor import Asked, DedupDirectorySensor, Published, Routed
from dedup_sim.control.routing import Dedup
from proposed import Endpoint, Environment
from realsim.adapters.real_controller import RealControllerAdapter
from sim_common.perfcount import InstructionCount
from torchstore import Publication
from torchstore.transport import Request

__all__ = ["main"]

_T = TypeVar("_T")
_MAX_INDEXED_ENTRIES = 1_000_000


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
        self.declare_probe = "probe-declare"
        self.full_probe = "probe-full"
        ids = (*self.sources, *self.generators, self.declare_probe, self.full_probe)
        topology = {volume: Endpoint(volume, volume, volume) for volume in ids}
        service = RealControllerAdapter().service
        for source in self.sources:
            service.notify_put_batch(self.requests, source, pending=False)
        self.directory = DedupDirectorySensor(service)
        self.plane = Dedup(fanout_cap=args.fanout_cap).attach(
            Environment(topology, _Profile()),
            {DedupDirectorySensor: self.directory},
        )
        self.dispatcher = self.plane.dispatcher

    def build_declared(self) -> None:
        for index, generator in enumerate(self.generators):
            source = self.sources[index % len(self.sources)]
            pub = self.directory.declare(generator, self.requests)
            self.dispatcher.dispatch_sync(Asked(pub))
            self.dispatcher.dispatch_sync(
                Routed(pub, (source,), frozenset(), 10.0)
            )

    def declare(self):
        pub = self.directory.declare(self.declare_probe, self.requests)
        self.dispatcher.dispatch_sync(Asked(pub))
        return pub

    def union(self) -> frozenset[Publication]:
        return self.directory.serving_union(self.requests)

    def rank(self, serving: frozenset[Publication]):
        return self.plane._chain.select(serving, self.full_probe)

    def gate(self, serving: frozenset[Publication]) -> None:
        pending = tuple(
            publication for publication in serving if publication[0] != 0
        )
        actions = tuple(Published(publication) for publication in pending)
        self.dispatcher.gate(lambda: True, actions)

    def decide(self) -> None:
        asyncio.run(self.plane._decide(self.requests, self.full_probe))


@dataclass(frozen=True)
class _Runtime:
    declare_burst_ms: float
    declare_ms: float
    union_ms: float
    rank_ms: float
    gate_ms: float
    full_decision_ms: float
    full_decision_instructions: int | None
    candidates: int
    pending_candidates: int
    selected_sources: int


@dataclass(frozen=True)
class _Memory:
    declare_burst_kib: float
    declare_kib: float
    union_kib: float
    rank_kib: float
    gate_kib: float
    full_decision_kib: float


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


def _once_ms(operation: Callable[[], _T]) -> tuple[float, _T]:
    gc.collect()
    started = time.perf_counter()
    result = operation()
    return (time.perf_counter() - started) * 1_000, result


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
    parser.add_argument("--allow-large", action="store_true")
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
    indexed_entries = args.keys * (args.source_ranks + args.generators)
    if not args.allow_large and indexed_entries > _MAX_INDEXED_ENTRIES:
        parser.error(
            "workload exceeds the default allocation guard; use --allow-large "
            "on a suitably sized host"
        )
    return args


def _runtime(args: argparse.Namespace) -> _Runtime:
    workload = _Workload(args)
    declare_burst_ms, _ = _once_ms(workload.build_declared)
    declare_ms, _pub = _once_ms(workload.declare)
    union_ms, serving = _median_ms(
        workload.union, warmups=args.warmups, repeats=args.repeats
    )
    rank_ms, ranking = _median_ms(
        lambda: workload.rank(serving),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    gate_ms, _ = _median_ms(
        lambda: workload.gate(serving),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    full_decision_instructions = None
    if InstructionCount.available():
        gc.collect()
        with InstructionCount() as counter:
            started = time.perf_counter()
            workload.decide()
            full_decision_ms = (time.perf_counter() - started) * 1_000
        full_decision_instructions = counter.count
    else:
        full_decision_ms, _ = _once_ms(workload.decide)
    return _Runtime(
        declare_burst_ms,
        declare_ms,
        union_ms,
        rank_ms,
        gate_ms,
        full_decision_ms,
        full_decision_instructions,
        len(serving),
        sum(pub != 0 for pub, _volume in serving),
        len(ranking.sources or ()),
    )


def _memory(args: argparse.Namespace) -> _Memory:
    workload = _Workload(args)
    declare_burst = _python_peak_kib(workload.build_declared)
    declare = _python_peak_kib(workload.declare)
    serving = workload.union()
    union = _python_peak_kib(workload.union)
    rank = _python_peak_kib(lambda: workload.rank(serving))
    gate = _python_peak_kib(lambda: workload.gate(serving))
    decision = _python_peak_kib(workload.decide)
    return _Memory(declare_burst, declare, union, rank, gate, decision)


def _run(args: argparse.Namespace) -> None:
    runtime = _runtime(args)
    memory = _memory(args)
    indexed_entries = args.keys * (args.source_ranks + args.generators)
    print(
        "case\tkeys\tsource_ranks\tgenerators\tindexed_metadata_entries\t"
        "declare_burst_ms\tdeclare_ms\tunion_ms\trank_ms\tgate_ms\t"
        "full_decision_ms\tfull_decision_instructions\t"
        "declare_burst_python_peak_kib\t"
        "declare_python_peak_kib\tunion_python_peak_kib\trank_python_peak_kib\t"
        "gate_python_peak_kib\tfull_decision_python_peak_kib\tcandidates\t"
        "pending_candidates\tselected_sources"
    )
    print(
        f"{args.case}\t{args.keys}\t{args.source_ranks}\t{args.generators}\t"
        f"{indexed_entries}\t{runtime.declare_burst_ms:.3f}\t"
        f"{runtime.declare_ms:.3f}\t{runtime.union_ms:.3f}\t"
        f"{runtime.rank_ms:.3f}\t{runtime.gate_ms:.3f}\t"
        f"{runtime.full_decision_ms:.3f}\t{runtime.full_decision_instructions}\t"
        f"{memory.declare_burst_kib:.3f}\t"
        f"{memory.declare_kib:.3f}\t{memory.union_kib:.3f}\t"
        f"{memory.rank_kib:.3f}\t{memory.gate_kib:.3f}\t"
        f"{memory.full_decision_kib:.3f}\t{runtime.candidates}\t"
        f"{runtime.pending_candidates}\t{runtime.selected_sources}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one synthetic scale point and print a tab-separated result."""
    args = _arguments(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
