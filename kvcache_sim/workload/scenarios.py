"""Scenario builders and a deterministic run harness (on the async engine).

Each scenario fixes a set of serving instances and a synthetic workload, runs it
under a scheduler on the shared deterministic :class:`~sim_common.async_engine.AsyncEngine`,
and returns ``(Metrics, Trace)``. Every scenario drives the **real** TorchStore
directory (block presence via the real ``Controller``) and real per-instance
clients, charging every cost through :mod:`sim_common.cost_model`.

This module is the wiring seam, so it is the one place that touches both planes:
it builds the data plane's :class:`~kvcache_sim.data.store.KVStore`, the control
plane's :class:`~kvcache_sim.control.view.KVView` and scheduler over the same
mesh, hands them to a :class:`~kvcache_sim.data.serving.ServingPlane`, and lets
:class:`realsim.runner.Runner` release the requests on the clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from realsim.runner import Runner, WorkItem
from sim_common.async_engine import AsyncEngine
from sim_common.topology import Endpoint

from sim_common.cost_model import DEFAULT_PROFILE
from domain.llm import decode_step_time
from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from ..control.view import KVView
from ..data.serving import ServingPlane
from ..data.store import KVStore
from ..report.metrics import Metrics, Trace
from .generator import make_workload

BLOCK_TOKENS = 512


def make_topology(num: int, per_node: int = 2) -> Dict[str, Endpoint]:
    """Build ``num`` instances laid out ``per_node`` per node (distinct hosts).

    Same-node instances exchange KV over NVLink; cross-node over RDMA -- so *where*
    a reusable prefix lives changes the cost of pulling it. The instance id is also
    its storage-volume id in the real directory.
    """
    topo: Dict[str, Endpoint] = {}
    for i in range(num):
        node = f"N{i // per_node}"
        topo[f"s{i}"] = Endpoint(id=f"s{i}", host=f"h{i}", node=node)
    return topo


def subset(topology: Dict[str, Endpoint], ids: List[str]) -> Dict[str, Endpoint]:
    """Return the sub-topology for ``ids`` (order-stable)."""
    return {i: topology[i] for i in ids}


@dataclass
class Run:
    """Output of one scheduler run."""

    metrics: Metrics
    trace: Trace


def run(
    topology: Dict[str, Endpoint],
    requests,
    kind: str,
    *,
    capacity: Optional[int] = None,
    balance_threshold: float = 1.5,
    replicate: bool = True,
    slo_ttft: float = float("inf"),
    slo_tbt: float = float("inf"),
    simulate_decode: bool = False,
    max_batch: int = 8,
    coupled: bool = False,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
    early_rejection: str = "off",
) -> Run:
    """Run ``requests`` on ``topology`` under scheduler ``kind``.

    ``kind`` is ``"cache_aware"`` (cache-aware coordinator) or ``"load_balance"``
    (baseline). Returns metrics + trace. The ``simulate_decode`` group of kwargs
    drives the batched-decode / TBT model; ``coupled`` says whether prefill shares
    the decode instances' compute, which is a data-plane fact and so is handed to
    the serving plane, not to the scheduler.
    """
    trace = Trace(time_width=8, kind_width=7)
    metrics = Metrics()
    loop = AsyncEngine(trace=trace)
    try:
        store = KVStore(
            topology, block_tokens=BLOCK_TOKENS, profile=DEFAULT_PROFILE, trace=trace
        )
        # Control senses the same real directory the data plane writes, but only
        # ever reads it.
        view = KVView(store.handle, store.topology)
        common = dict(
            block_tokens=BLOCK_TOKENS,
            capacity=capacity,
            profile=DEFAULT_PROFILE,
            slo_ttft=slo_ttft,
            slo_tbt=slo_tbt,
            simulate_decode=simulate_decode,
            max_batch=max_batch,
            prefill_pool=prefill_pool,
            decode_pool=decode_pool,
            early_rejection=early_rejection,
        )
        if kind == "cache_aware":
            sched = CacheAwareScheduler(
                view, balance_threshold=balance_threshold,
                replicate=replicate, **common,
            )
        elif kind == "load_balance":
            sched = LoadBalanceScheduler(view, **common)
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown scheduler kind {kind!r}")

        plane = ServingPlane(
            loop, store, sched, trace=trace, metrics=metrics,
            coupled=coupled, max_batch=max_batch,
        )
        # Request coroutines end at prefill completion; decode continues on its
        # own step tasks, so the runner drains them before it returns.
        runner = Runner(store.mesh, plane=plane, drain=plane.drain)
        items = [
            WorkItem(id=r.id, release_time=r.arrival, payload=r) for r in requests
        ]
        loop.run_until_complete(runner.run(items))
    finally:
        loop.close()
    return Run(metrics=metrics, trace=trace)


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #

def shared_prefix_workload(seed: int = 0):
    """Many conversations sharing a hot system prompt + per-conv context."""
    return make_workload(
        num_requests=200, num_conversations=8, system_blocks=4,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.1, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )


def run_shared_prefix(seed: int = 0) -> Tuple[Run, Run]:
    """Cache-aware vs load-balance on the shared-prefix workload (ample capacity)."""
    topo = make_topology(4)
    reqs = shared_prefix_workload(seed)
    cache_aware = run(topo, reqs, "cache_aware")
    baseline = run(topo, reqs, "load_balance")
    return cache_aware, baseline


def run_eviction_sweep(seed: int = 0) -> List[Tuple[int, float, int]]:
    """Sweep cache capacity; return ``(capacity, hit_rate, fabric_bytes)`` rows."""
    topo = make_topology(4)
    reqs = make_workload(
        num_requests=400, num_conversations=12, system_blocks=2,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.05, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )
    rows: List[Tuple[int, float, int]] = []
    for cap in (2, 4, 8, 16, 32, 64, 256):
        r = run(topo, reqs, "cache_aware", capacity=cap)
        rows.append((cap, r.metrics.hit_rate, r.metrics.fabric_bytes))
    return rows


def hotspot_workload(seed: int = 0):
    """One dominant conversation (extreme skew) -> a single hot instance."""
    return make_workload(
        num_requests=160, num_conversations=4, system_blocks=6,
        conv_base_blocks=6, query_blocks=2, zipf_s=2.2, arrival_rate=3.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )


def run_hotspot(seed: int = 0) -> Tuple[Run, Run, Run]:
    """Compare (a) baseline, (b) cache-aware no-replication, (c) cache-aware."""
    topo = make_topology(4)
    reqs = hotspot_workload(seed)
    baseline = run(topo, reqs, "load_balance")
    no_repl = run(topo, reqs, "cache_aware", replicate=False)
    repl = run(topo, reqs, "cache_aware", balance_threshold=1.2, replicate=True)
    return baseline, no_repl, repl


def run_overload(seed: int = 0) -> Tuple[Run, Run]:
    """High arrival rate + a TTFT SLO -> some requests must be rejected."""
    topo = make_topology(4)
    reqs = make_workload(
        num_requests=300, num_conversations=6, system_blocks=6,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=9.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )
    slo = 6.0
    cache_aware = run(topo, reqs, "cache_aware", slo_ttft=slo)
    baseline = run(topo, reqs, "load_balance", slo_ttft=slo)
    return cache_aware, baseline


# --------------------------------------------------------------------------- #
# Decode-side scenarios: TBT under batched decode.
# --------------------------------------------------------------------------- #

# Target TBT for the disaggregation scenario: 5 x the batch=1 baseline step time.
DISAGG_TARGET_TBT = 5 * decode_step_time(1, DEFAULT_PROFILE)
DISAGG_MAX_BATCH = 8


def run_disaggregation(seed: int = 0) -> Tuple[Run, Run]:
    """Disaggregating prefill from decode protects TBT (Mooncake's headline).

    Both configs run with admission disabled (``slo_tbt=inf``, ``early_rejection=
    "off"``) so every request is served and measured. Decode capacity is fixed (two
    instances, ``DISAGG_MAX_BATCH`` each); the only difference is whether those two
    instances also do prefill. Returns ``(disaggregated, coupled)``.
    """
    topo = make_topology(4)  # s0..s3
    reqs = make_workload(
        num_requests=120, num_conversations=8, system_blocks=2,
        conv_base_blocks=2, query_blocks=1, zipf_s=1.1, arrival_rate=1.2,
        block_tokens=BLOCK_TOKENS, output_tokens=12, seed=seed,
    )
    common = dict(
        simulate_decode=True, slo_tbt=float("inf"), early_rejection="off",
        max_batch=DISAGG_MAX_BATCH,
    )
    disaggregated = run(
        topo, reqs, "cache_aware",
        prefill_pool=["s0", "s1"], decode_pool=["s2", "s3"], coupled=False,
        **common,
    )
    coupled = run(
        subset(topo, ["s2", "s3"]), reqs, "cache_aware", coupled=True, **common
    )
    return disaggregated, coupled


# TBT SLO for the early-rejection scenario: 3 x baseline step time.
EARLY_SLO_TBT = 3 * decode_step_time(1, DEFAULT_PROFILE)
EARLY_MAX_BATCH = 8


def run_early_rejection(seed: int = 0) -> Tuple[Run, Run, Run]:
    """Predicting decode load avoids wasting prefill (off/early/predict)."""
    topo = make_topology(4)
    reqs = make_workload(
        num_requests=160, num_conversations=6, system_blocks=6,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=20.0,
        block_tokens=BLOCK_TOKENS, output_tokens=32, seed=seed,
    )
    common = dict(
        simulate_decode=True, slo_tbt=EARLY_SLO_TBT, max_batch=EARLY_MAX_BATCH
    )
    off = run(topo, reqs, "cache_aware", early_rejection="off", **common)
    early = run(topo, reqs, "cache_aware", early_rejection="early", **common)
    predict = run(topo, reqs, "cache_aware", early_rejection="predict", **common)
    return off, early, predict
