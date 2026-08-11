"""The kvcache scenarios: which configurations each comparison runs.

Each function fixes a topology and a synthetic workload, then returns the
:class:`~realsim.run.Run` values to compare -- same requests, different wiring.
Nothing here builds a clock, a mesh or a plane: :meth:`realsim.run.Run.execute` does
that, the same way for every capability.
"""

from __future__ import annotations

from typing import Dict, List

from domain import decode_step_time, DEFAULT_PROFILE
from proposed import Endpoint

from realsim.run import Run
from sim_common.trace import Trace

from ..report.metrics import Metrics
from ._generator import make_workload
from ._serving import BLOCK_TOKENS, KVWorkload, serving_plane

__all__ = [
    "configure",
    "make_topology",
    "subset",
    "shared_prefix_workload",
    "shared_prefix",
    "EVICTION_CAPACITIES",
    "eviction_sweep",
    "hotspot_workload",
    "hotspot",
    "overload",
    "DISAGG_TARGET_TBT",
    "DISAGG_MAX_BATCH",
    "disaggregation",
    "EARLY_SLO_TBT",
    "EARLY_MAX_BATCH",
    "early_rejection",
]


def configure(label: str, topology, requests, kind: str, **knobs) -> Run:
    """One labelled configuration over ``requests``.

    Every kvcache run is built here -- by the scenarios below and by the tests --
    so the trace format and the metrics ledger a run reports into are chosen once
    and cannot drift between them.
    """
    return Run(
        label,
        KVWorkload(topology, requests),
        plane=serving_plane(kind, **knobs),
        profile=DEFAULT_PROFILE,
        trace=Trace(time_width=8, kind_width=7),
        ledger=Metrics(),
    )


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


def shared_prefix(seed: int = 0) -> List[Run]:
    """Cache-aware vs load-balance on the shared-prefix workload (ample capacity)."""
    topo = make_topology(4)
    reqs = shared_prefix_workload(seed)
    return [
        configure("cache_aware", topo, reqs, "cache_aware"),
        configure("load_balance", topo, reqs, "load_balance"),
    ]


#: Per-instance cache capacities the eviction sweep walks.
EVICTION_CAPACITIES = (2, 4, 8, 16, 32, 64, 256)


def eviction_sweep(seed: int = 0) -> List[Run]:
    """One cache-aware run per capacity; the report reads the hit-rate curve off
    their ledgers. Each run is labelled with its capacity."""
    topo = make_topology(4)
    reqs = make_workload(
        num_requests=400, num_conversations=12, system_blocks=2,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.05, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )
    return [
        configure(str(cap), topo, reqs, "cache_aware", capacity=cap)
        for cap in EVICTION_CAPACITIES
    ]


def hotspot_workload(seed: int = 0):
    """One dominant conversation (extreme skew) -> a single hot instance."""
    return make_workload(
        num_requests=160, num_conversations=4, system_blocks=6,
        conv_base_blocks=6, query_blocks=2, zipf_s=2.2, arrival_rate=3.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )


def hotspot(seed: int = 0) -> List[Run]:
    """Compare (a) baseline, (b) cache-aware no-replication, (c) cache-aware."""
    topo = make_topology(4)
    reqs = hotspot_workload(seed)
    return [
        configure("baseline", topo, reqs, "load_balance"),
        configure("no_replication", topo, reqs, "cache_aware", replicate=False),
        configure("replication", topo, reqs, "cache_aware",
             balance_threshold=1.2, replicate=True),
    ]


def overload(seed: int = 0) -> List[Run]:
    """High arrival rate + a TTFT SLO -> some requests must be rejected."""
    topo = make_topology(4)
    reqs = make_workload(
        num_requests=300, num_conversations=6, system_blocks=6,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=9.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )
    slo = 6.0
    return [
        configure("cache_aware", topo, reqs, "cache_aware", slo_ttft=slo),
        configure("load_balance", topo, reqs, "load_balance", slo_ttft=slo),
    ]


# --------------------------------------------------------------------------- #
# Decode-side scenarios: TBT under batched decode.
# --------------------------------------------------------------------------- #

# Target TBT for the disaggregation scenario: 5 x the batch=1 baseline step time.
DISAGG_TARGET_TBT = 5 * decode_step_time(1, DEFAULT_PROFILE)
DISAGG_MAX_BATCH = 8


def disaggregation(seed: int = 0) -> List[Run]:
    """Disaggregating prefill from decode protects TBT (Mooncake's headline).

    Both configs run with admission disabled (``slo_tbt=inf``, ``early_rejection=
    "off"``) so every request is served and measured. Decode capacity is fixed (two
    instances, ``DISAGG_MAX_BATCH`` each); the only difference is whether those two
    instances also do prefill. Returns ``[disaggregated, coupled]``.
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
    return [
        configure("disaggregated", topo, reqs, "cache_aware",
             prefill_pool=["s0", "s1"], decode_pool=["s2", "s3"], coupled=False,
             **common),
        configure("coupled", subset(topo, ["s2", "s3"]), reqs, "cache_aware",
             coupled=True, **common),
    ]


# TBT SLO for the early-rejection scenario: 3 x baseline step time.
EARLY_SLO_TBT = 3 * decode_step_time(1, DEFAULT_PROFILE)
EARLY_MAX_BATCH = 8


def early_rejection(seed: int = 0) -> List[Run]:
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
    return [
        configure(mode, topo, reqs, "cache_aware", early_rejection=mode, **common)
        for mode in ("off", "early", "predict")
    ]
