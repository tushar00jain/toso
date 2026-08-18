"""Benchmark dedup control-plane planning at a peak synchronized burst.

Run from the repository root::

    .venv/bin/python -m realsim.tools.benchmark_dedup_control
    .venv/bin/python -m realsim.tools.benchmark_dedup_control --preset planned-8b

The workload uses metadata-only TorchStore requests. It measures directory and
routing work, not payload allocation, transport, or simulated transfer time.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import statistics
import time
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
            request.key: {source: info for source in sources}
            for request in requests
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
    region_checks = args.keys * args.keys * (
        args.source_ranks + args.generators
    )
    if not args.allow_large and (
        indexed_entries > _MAX_INDEXED_ENTRIES
        or region_checks > _MAX_REGION_CHECKS
    ):
        parser.error(
            "workload exceeds the default allocation/work guard; use "
            "--allow-large on a suitably sized host"
        )
    return args


def _run(args: argparse.Namespace) -> None:
    requests = tuple(
        Request.from_any(f"model.weight.{index}", None).meta_only()
        for index in range(args.keys)
    )
    sources = tuple(f"source-{index}" for index in range(args.source_ranks))
    generators = tuple(f"generator-{index}" for index in range(args.generators))
    probe = "probe"
    topology = {
        volume: Endpoint(id=volume, host=volume, node=volume)
        for volume in (*sources, *generators, probe)
    }
    directory = DedupDirectorySensor(_StaticDirectory(requests, sources))
    plane = Dedup(fanout_cap=args.fanout_cap).attach(
        Environment(topology, _Profile()),
        {DedupDirectorySensor: directory},
    )
    assert plane.dispatcher is not None

    regions = tuple((request.key, None) for request in requests)
    gc.collect()
    started = time.perf_counter()
    for index, generator in enumerate(generators):
        source = sources[index % len(sources)]
        plane.dispatcher.dispatch_sync(Asked(generator, requests))
        plane.dispatcher.dispatch_sync(
            Routed(
                requester=generator,
                sources=(source,),
                required=((source, regions),),
            )
        )
    pending_build_ms = (time.perf_counter() - started) * 1_000

    def snapshot() -> None:
        with directory.pinned([request.key for request in requests]):
            pass

    snapshot_ms, _ = _median_ms(
        snapshot, warmups=args.warmups, repeats=args.repeats
    )
    with directory.pinned([request.key for request in requests]):
        serving_ms, serving = _median_ms(
            lambda: directory.serving_sources(requests),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        order = (*generators, *sources)
        plan_ms, fetch = _median_ms(
            lambda: directory.plan_fetch(requests, order, requester=probe),
            warmups=args.warmups,
            repeats=args.repeats,
        )

    gc.collect()
    started = time.perf_counter()
    asyncio.run(plane._decide(requests, probe))
    decision_ms = (time.perf_counter() - started) * 1_000

    candidates, pending = serving
    indexed_entries = args.keys * (args.source_ranks + 3 * args.generators)
    print(
        "case\tkeys\tsource_ranks\tgenerators\tindexed_metadata_entries\t"
        "pending_build_ms\tsnapshot_ms\tserving_sources_ms\tplan_fetch_ms\t"
        "full_decision_ms\tcandidates\tpending_candidates\tselected_sources"
    )
    print(
        f"{args.case}\t{args.keys}\t{args.source_ranks}\t{args.generators}\t"
        f"{indexed_entries}\t{pending_build_ms:.3f}\t{snapshot_ms:.3f}\t"
        f"{serving_ms:.3f}\t{plan_ms:.3f}\t{decision_ms:.3f}\t"
        f"{len(candidates)}\t{len(pending)}\t{len(fetch.sources)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one synthetic scale point and print a tab-separated result."""
    args = _arguments(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
