"""Scenario builders and a deterministic run harness.

A scenario defines volumes, the trainer-side stored layout, and each generator's
read ``need``. The harness seeds the index, enqueues every generator ``get`` at
``t=0`` (the burst), runs the sim, and returns the ``Trace`` + ``Metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .client import Client
from .coordinator import DedupCoordinator, NaiveCoordinator
from sim_common.engine import Sim
from .model import Region, split_regions, union_bytes, Volume
from .store_index import StoreIndex
from .trace import Metrics, Trace

KEY = "W"
DTYPE_BYTES = 1


@dataclass
class Scenario:
    """A concrete simulation input."""

    name: str
    volumes: List[Volume]
    stored: List[Tuple[str, Region]]  # (trainer_volume_id, region)
    needs: Dict[str, List[Region]]  # generator_volume_id -> needed ranges
    key: str = KEY
    dtype_bytes: int = DTYPE_BYTES

    @property
    def atomics(self) -> List[Region]:
        """All atomic regions across stored + needed ranges."""
        regions = [r for (_v, r) in self.stored]
        for need in self.needs.values():
            regions.extend(need)
        return split_regions(regions)

    @property
    def union_bytes(self) -> int:
        """Bytes of the union of all needs (the 1x fabric target)."""
        return union_bytes(self.needs.values(), self.atomics, self.dtype_bytes)


@dataclass
class RunResult:
    """Output of one coordinator run."""

    trace: Trace
    metrics: Metrics
    peak_serving: int = 0


def _make_topo(scn: Scenario) -> Dict[str, Volume]:
    return {v.id: v for v in scn.volumes}


def run(scn: Scenario, kind: str, fanout_cap: int = 1) -> RunResult:
    """Run ``scn`` under ``kind`` in {"dedup", "naive"} and return the result."""
    sim = Sim()
    index = StoreIndex()
    topo = _make_topo(scn)
    for vol, region in scn.stored:
        index.notify_put(scn.key, vol, region)

    if kind == "dedup":
        coord = DedupCoordinator(sim, index, topo, fanout_cap=fanout_cap)
    elif kind == "naive":
        coord = NaiveCoordinator(sim, index, topo)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown coordinator kind {kind!r}")

    trace = Trace()
    metrics = Metrics()
    client = Client(sim, index, coord, topo, scn.dtype_bytes, scn.atomics,
                    trace, metrics)

    for gen_id in sorted(scn.needs):
        reader = topo[gen_id]
        need = scn.needs[gen_id]
        sim.schedule(
            0.0, lambda r=reader, n=need: client.get(r, scn.key, n),
            label=f"get:{gen_id}",
        )

    sim.run()
    metrics.peak_serving = coord.peak_serving
    return RunResult(trace=trace, metrics=metrics, peak_serving=coord.peak_serving)


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #

def toy_scenario(num_gens: int = 3) -> Scenario:
    """Full-replication burst: every generator needs the whole tensor ``W``.

    One trainer volume ``t0`` (node A) holds ``[0, N)``; ``num_gens`` generator
    volumes on node B (distinct hosts, so gen<->gen is NVLink) each need
    ``[0, N)``. This is where dedup's fan-out tree/chain is visible.
    """
    n = 8
    volumes = [Volume("t0", host="hA", node="A", is_trainer=True)]
    needs: Dict[str, List[Region]] = {}
    for i in range(num_gens):
        gid = f"g{i}"
        volumes.append(Volume(gid, host=f"hB{i}", node="B"))
        needs[gid] = [(0, n)]
    stored = [("t0", (0, n))]
    return Scenario("toy full-replication burst", volumes, stored, needs)


def reshard_scenario() -> Scenario:
    """Reshard: trainer stores halves; generators want a different partition.

    Trainer volumes ``t0`` ([0,4)) and ``t1`` ([4,8)) on node A. Generators want
    overlapping ranges that cross the stored boundary, exercising atomic-region
    splitting. Each atomic still leaves the trainer once (1x fabric).
    """
    volumes = [
        Volume("t0", host="hA0", node="A", is_trainer=True),
        Volume("t1", host="hA1", node="A", is_trainer=True),
        Volume("g0", host="hB0", node="B"),
        Volume("g1", host="hB1", node="B"),
        Volume("g2", host="hB2", node="B"),
    ]
    stored = [("t0", (0, 4)), ("t1", (4, 8))]
    needs = {
        "g0": [(0, 5)],
        "g1": [(3, 8)],
        "g2": [(0, 8)],
    }
    return Scenario("reshard (stored halves, generators want a new partition)",
                    volumes, stored, needs)


@dataclass
class VersioningResult:
    """Output of a two-burst versioning run."""

    trace: Trace
    fabric1: int  # trainer fabric of burst 1
    fabric2: int  # trainer fabric of burst 2 (delta after the bump/no-bump)
    union_bytes: int
    bumped: bool


def run_versioning(bump: bool, fanout_cap: int = 1) -> VersioningResult:
    """Run two bursts on one coordinator; capture the trace + per-burst fabric.

    With ``bump=True`` the version is bumped between bursts (cache invalidated
    -> burst 2 re-pulls the full union from the trainer). With ``bump=False``
    the cache persists, so burst 2 sources entirely from peers and burst-2
    trainer fabric is 0.
    """
    scn = toy_scenario()
    sim = Sim()
    index = StoreIndex()
    topo = _make_topo(scn)
    for vol, region in scn.stored:
        index.notify_put(scn.key, vol, region)
    coord = DedupCoordinator(sim, index, topo, fanout_cap=fanout_cap)
    trace = Trace()
    metrics = Metrics()
    client = Client(sim, index, coord, topo, scn.dtype_bytes, scn.atomics,
                    trace, metrics)

    def burst() -> None:
        for gen_id in sorted(scn.needs):
            reader = topo[gen_id]
            need = scn.needs[gen_id]
            sim.schedule(0.0, lambda r=reader, n=need: client.get(r, scn.key, n))
        sim.run()

    trace.record(sim.now, "MARK", "burst 1 (version 0) -- cold cache")
    burst()
    fabric1 = metrics.fabric_bytes

    if bump:
        coord.bump_version(scn.key)
        # trainer re-publishes the new version's data (idempotent seed).
        for vol, region in scn.stored:
            index.notify_put(scn.key, vol, region)
        trace.record(sim.now, "MARK",
                     "put bumps version 0 -> 1 (stale cache invalidated)")
    else:
        trace.record(sim.now, "MARK",
                     "no version bump (cache from burst 1 still valid)")

    before = metrics.fabric_bytes
    trace.record(sim.now, "MARK", "burst 2")
    burst()
    fabric2 = metrics.fabric_bytes - before
    return VersioningResult(trace, fabric1, fabric2, scn.union_bytes, bump)


def versioning_result(bump: bool, fanout_cap: int = 1) -> Tuple[int, int, int]:
    """Back-compat wrapper: ``(fabric_burst1, fabric_burst2, union_bytes)``."""
    r = run_versioning(bump, fanout_cap)
    return r.fabric1, r.fabric2, r.union_bytes
