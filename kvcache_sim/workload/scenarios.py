"""The kvcache scenarios: which configurations each comparison runs.

Each function fixes a topology and a synthetic workload, then returns the
:class:`~realsim.run.Run` values to compare -- same requests, different wiring.
Nothing here builds a clock, a mesh or a plane: :meth:`realsim.run.Run.execute` does
that, the same way for every capability.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from domain import decode_step_time, DEFAULT_PROFILE
from proposed import Endpoint

from realsim.demo import Console, Scenario
from realsim.run import Result, Run
from sim_common.trace import Trace

from ..report.metrics import Metrics
from ..report.summary import (
    CacheVsBaselineReport,
    DisaggregationReport,
    EarlyRejectionReport,
    EvictionReport,
    HotspotReport,
)
from ._generator import make_workload
from ._serving import BLOCK_TOKENS, KVWorkload, serving_plane

__all__ = [
    "TRACE_LIMIT",
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
    "SharedPrefix",
    "Eviction",
    "Hotspot",
    "Overload",
    "Disaggregation",
    "EarlyRejection",
]


#: Per-event trace lines a scenario dumps under ``-v``; these runs are long.
TRACE_LIMIT = 60


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


class SharedPrefix(Scenario):
    name = "shared_prefix"

    def runs(self, args) -> List[Run]:
        return shared_prefix()

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("SHARED PREFIX: conversations sharing a hot system prompt + context")
        console.info("directory: real torchstore.controller.Controller (off-actor)")
        console.info("4 instances (2 nodes), 200 requests, 8 conversations, Zipf skew.")
        console.info("Cache-aware routes same-prefix requests to the instance holding the")
        console.info("prefix (or pulls it once), so shared prefixes are computed ~once;")
        console.info("load-balance scatters them, recomputing prefixes on every instance.")
        console.trace(results[0].trace, limit=TRACE_LIMIT)
        console.summary(CacheVsBaselineReport("shared_prefix", results))


class Eviction(Scenario):
    name = "eviction"

    def runs(self, args) -> List[Run]:
        return eviction_sweep()

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("EVICTION: hit rate vs cache capacity (LRU)")
        console.info("400 requests, 12 conversations. As per-instance capacity grows, the")
        console.info("hot working set fits and the prefix hit rate rises, then plateaus")
        console.info("(the ~30%%->~50%% shape). Too-small caches also force more KV")
        console.info("re-fetch (fabric).")
        console.info("")
        console.info(EvictionReport(results).render())


class Hotspot(Scenario):
    name = "hotspot"

    def runs(self, args) -> List[Run]:
        return hotspot()

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("HOTSPOT: extreme skew -> hot-block replication spreads load")
        console.info("One dominant conversation. Without replication (balance_threshold huge)")
        console.info("the cache-aware policy piles every hot request on the single instance")
        console.info("holding the prefix; with a moderate threshold it replicates the prefix")
        console.info("to peers (read-through), spreading load and cutting p90 TTFT.")
        console.trace(results[2].trace, limit=TRACE_LIMIT)
        console.summary(HotspotReport(results))
        console.info("(replication swaps recompute for cheap KV transfer when spreading a")
        console.info(" hot prefix to a peer -> fewer prefill tokens, more fabric bytes.)")


class Overload(Scenario):
    name = "overload"

    def runs(self, args) -> List[Run]:
        return overload()

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("OVERLOAD: high arrival + TTFT SLO -> rejections")
        console.info("300 requests at a high rate with a TTFT SLO of 6.0. Prefix reuse")
        console.info("shortens prefill, freeing capacity, so cache-aware admits more")
        console.info("requests (fewer rejections) than the load-balancing baseline.")
        console.summary(CacheVsBaselineReport("overload", results))


class Disaggregation(Scenario):
    name = "disaggregation"

    def runs(self, args) -> List[Run]:
        return disaggregation()

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("DISAGGREGATION: dedicated decode pool protects TBT from prefill")
        console.info("Two decode instances, VRAM cap 8, TBT target %.3f. Admission is",
                     DISAGG_TARGET_TBT)
        console.info("disabled (no TBT SLO gate), so BOTH configs serve every request -- the")
        console.info("contrast is purely the TBT-target attainment among served requests, not")
        console.info("a rejection count. The only difference is prefill placement: disaggregated")
        console.info("prefills on a separate pool (s0/s1) so decode (s2/s3) keeps its own")
        console.info("compute timeline; coupled runs prefill AND decode on s2/s3, so a prefill")
        console.info("can collide with a decode step and spike that request's inter-token gap.")
        console.trace(results[0].trace, limit=TRACE_LIMIT)
        console.summary(DisaggregationReport(results, DISAGG_TARGET_TBT))
        console.info("Attainment is the fraction of served requests whose *worst* inter-token")
        console.info("gap stayed under the target. Disaggregation isolates decode from prefill,")
        console.info("so served requests hold TBT; coupling lets long prefills stall decode, so")
        console.info("a large fraction of served requests blow the target -- same load admitted.")


class EarlyRejection(Scenario):
    name = "early_rejection"

    def runs(self, args) -> List[Run]:
        return early_rejection()

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("EARLY REJECTION: predict decode load, don't waste prefill")
        console.info("Heavy decode load with a tight TBT SLO of %.3f. Three cache-aware runs",
                     EARLY_SLO_TBT)
        console.info("differ only in the admission policy. 'off' late-checks decode load AFTER")
        console.info("prefill and rejects on a violation -- so each rejection is a wasted")
        console.info("prefill (compute already spent). 'early' and 'predict' both gate at")
        console.info("routing, before prefill, so neither ever wastes prefill; here neither")
        console.info("rejects (both admit all). The difference is decode routing: 'early' uses")
        console.info("the current occupancy, which a slow prefill leaves reading ~empty, so it")
        console.info("piles decode onto one instance and blows the SLO; 'predict' routes by the")
        console.info("load foreseen at prefill completion, spreading decode so the SLO holds.")
        console.trace(results[2].trace, limit=TRACE_LIMIT)
        console.summary(EarlyRejectionReport(results, EARLY_SLO_TBT))
        console.info("(Signal: wasted prefill separates 'off' from the rest; TBT attainment")
        console.info(" separates 'predict' (routes on predicted load) from 'early' (stale).)")
