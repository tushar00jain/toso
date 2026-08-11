"""The kvcache scenarios: what each run simulates.

Each function fixes a topology and a synthetic workload and hands them to
:func:`kvcache_sim.harness.run`, which does the assembling. Nothing here wires a
plane, a scheduler or a clock -- a scenario is a set of choices, not a harness.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from domain import decode_step_time, DEFAULT_PROFILE
from proposed import Endpoint

from realsim.entrypoint import Result

from ..harness import BLOCK_TOKENS, run
from ._generator import make_workload


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


def shared_prefix_workload(seed: int = 0):
    """Many conversations sharing a hot system prompt + per-conv context."""
    return make_workload(
        num_requests=200, num_conversations=8, system_blocks=4,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.1, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )


def run_shared_prefix(seed: int = 0) -> Tuple[Result, Result]:
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
        rows.append((cap, r.ledger.hit_rate, r.ledger.fabric_bytes))
    return rows


def hotspot_workload(seed: int = 0):
    """One dominant conversation (extreme skew) -> a single hot instance."""
    return make_workload(
        num_requests=160, num_conversations=4, system_blocks=6,
        conv_base_blocks=6, query_blocks=2, zipf_s=2.2, arrival_rate=3.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )


def run_hotspot(seed: int = 0) -> Tuple[Result, Result, Result]:
    """Compare (a) baseline, (b) cache-aware no-replication, (c) cache-aware."""
    topo = make_topology(4)
    reqs = hotspot_workload(seed)
    baseline = run(topo, reqs, "load_balance")
    no_repl = run(topo, reqs, "cache_aware", replicate=False)
    repl = run(topo, reqs, "cache_aware", balance_threshold=1.2, replicate=True)
    return baseline, no_repl, repl


def run_overload(seed: int = 0) -> Tuple[Result, Result]:
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


def run_disaggregation(seed: int = 0) -> Tuple[Result, Result]:
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


def run_early_rejection(seed: int = 0) -> Tuple[Result, Result, Result]:
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
