"""The kvcache scenarios: six comparisons, each a :class:`realsim.demo.Scenario`.

Each one fixes a topology and a synthetic request stream, declares the
:class:`~realsim.run.Run` values to compare -- same requests, different wiring --
and narrates the results. Nothing here builds a clock, a mesh or a plane, and
nothing here executes: :meth:`realsim.demo.Demo.main` does that with
:meth:`realsim.run.Run.execute`, the same way for every capability.

Each is parameterized by its seed at construction rather than by a CLI flag, so
``SharedPrefix(seed=1).runs()`` gives a test the same configurations the demo
shows, with no command line involved.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

from domain import decode_step_time, DEFAULT_MODEL, DEFAULT_PROFILE
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
from ._serving import BLOCK_TOKENS, coordinator, KVWorkload, serving_plane

__all__ = [
    "TRACE_LIMIT",
    "EVICTION_CAPACITIES",
    "KV_BLOCKS_PER_INSTANCE",
    "DISAGG_TARGET_TBT",
    "DISAGG_MAX_BATCH",
    "EARLY_SLO_TBT",
    "EARLY_MAX_BATCH",
    "SharedPrefix",
    "Eviction",
    "Hotspot",
    "Overload",
    "Disaggregation",
    "EarlyRejection",
]


#: Per-event trace lines a scenario dumps under ``-v``; these runs are long.
TRACE_LIMIT = 60


def _configure(label: str, topology, requests, kind: str, **knobs) -> Run:
    """One labelled configuration over ``requests``.

    Every kvcache run is built here -- by the scenarios below and by the tests --
    so the trace format and the metrics ledger a run reports into are chosen once
    and cannot drift between them.
    """
    # The pools are a deployment fact and go to both planes: the data plane gives
    # a host the engines its pools put on it, and the coordinator routes to the
    # same pools. ``coupled`` is a separate, fidelity question -- does this run
    # model a prefill and a decode step on one host as colliding? -- so it goes to
    # the data plane alone, which answers it by sharing a timeline or not.
    coupled = knobs.pop("coupled", False)

    # A per-run machine, so a scenario can give its volumes a different capacity.
    # The default is finite: a serving instance's KV memory is a real bound, and a
    # cache that cannot run out of room is not one -- the eviction it never performs
    # would flatter every hit rate here. It is *store* capacity because that is
    # where the blocks are; a store in general is unbounded, which is why this
    # belongs to the capability and not to the profile every sim shares.
    profile = knobs.pop("profile", _KV_INSTANCE_PROFILE)
    return Run(
        label,
        KVWorkload(topology, requests),
        # One control plane, reached from both services: it decides compute
        # placement through the coordinator seam, and answers the store's routing
        # question through the directory it is installed in.
        control=coordinator(kind, **knobs),
        data=serving_plane(
            coupled=coupled,
            simulate_decode=knobs.get("simulate_decode", False),
            max_batch=knobs.get("max_batch", 8),
            prefill_pool=knobs.get("prefill_pool"),
            decode_pool=knobs.get("decode_pool"),
        ),
        profile=profile,
        trace=Trace(time_width=8, kind_width=7),
        ledger=Metrics(),
    )


def _make_topology(num: int, per_node: int = 2) -> Dict[str, Endpoint]:
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


def _subset(topology: Dict[str, Endpoint], ids: List[str]) -> Dict[str, Endpoint]:
    """Return the sub-topology for ``ids`` (order-stable)."""
    return {i: topology[i] for i in ids}


def _shared_prefix_workload(seed: int = 0):
    """Many conversations sharing a hot system prompt + per-conv context."""
    return make_workload(
        num_requests=200, num_conversations=8, system_blocks=4,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.1, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )



#: Per-instance cache capacities the eviction sweep walks.
#: What one serving instance sets aside for KV, in blocks. Chosen where the
#: eviction sweep below has already plateaued: large enough that capacity is not
#: what any other scenario is measuring, finite because the hardware is.
KV_BLOCKS_PER_INSTANCE = 256
_KV_INSTANCE_PROFILE = replace(
    DEFAULT_PROFILE,
    storage_capacity_bytes=DEFAULT_MODEL.block_bytes(
        KV_BLOCKS_PER_INSTANCE, BLOCK_TOKENS
    ),
)

# In blocks, converted to the volume's byte capacity. The smallest must still fit
# one request's own blocks: a volume now enforces a real limit rather than a model
# one, so a put that cannot fit even after evicting everything is refused, not
# absorbed. The largest prompt here is system(2) + conversation(4) + query(2).
EVICTION_CAPACITIES = (8, 16, 32, 64, 128, 256)



def _hotspot_workload(seed: int = 0):
    """One dominant conversation (extreme skew) -> a single hot instance."""
    return make_workload(
        num_requests=160, num_conversations=4, system_blocks=6,
        conv_base_blocks=6, query_blocks=2, zipf_s=2.2, arrival_rate=3.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )





# --------------------------------------------------------------------------- #
# Decode-side scenarios: TBT under batched decode.
# --------------------------------------------------------------------------- #

# Target TBT for the disaggregation scenario: 5 x the batch=1 baseline step time.
DISAGG_TARGET_TBT = 5 * decode_step_time(1, DEFAULT_PROFILE)
DISAGG_MAX_BATCH = 8



# TBT SLO for the early-rejection scenario: 3 x baseline step time.
EARLY_SLO_TBT = 3 * decode_step_time(1, DEFAULT_PROFILE)
EARLY_MAX_BATCH = 8



class SharedPrefix(Scenario):
    """Cache-aware vs load-balance on the shared-prefix workload (ample capacity)."""

    name = "shared_prefix"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        reqs = _shared_prefix_workload(self.seed)
        return [
            _configure("cache_aware", topo, reqs, "cache_aware"),
            _configure("load_balance", topo, reqs, "load_balance"),
        ]

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
    """One cache-aware run per capacity; the report reads the hit-rate curve off
    their ledgers. Each run is labelled with its capacity."""

    name = "eviction"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        reqs = make_workload(
            num_requests=400, num_conversations=12, system_blocks=2,
            conv_base_blocks=4, query_blocks=2, zipf_s=1.05, arrival_rate=2.5,
            block_tokens=BLOCK_TOKENS, output_tokens=64, seed=self.seed,
        )
        return [
            _configure(
                str(cap), topo, reqs, "cache_aware",
                profile=replace(
                    DEFAULT_PROFILE,
                    storage_capacity_bytes=DEFAULT_MODEL.block_bytes(
                        cap, BLOCK_TOKENS
                    ),
                ),
            )
            for cap in EVICTION_CAPACITIES
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("EVICTION: hit rate vs cache capacity (LRU)")
        console.info("400 requests, 12 conversations. As per-instance capacity grows, the")
        console.info("hot working set fits and the prefix hit rate rises, then plateaus")
        console.info("(the ~30%%->~50%% shape). Too-small caches also force more KV")
        console.info("re-fetch (fabric).")
        console.info("")
        console.info(EvictionReport(results).render())


class Hotspot(Scenario):
    """Compare (a) baseline, (b) cache-aware no-replication, (c) cache-aware."""

    name = "hotspot"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        reqs = _hotspot_workload(self.seed)
        return [
            _configure("baseline", topo, reqs, "load_balance"),
            _configure("no_replication", topo, reqs, "cache_aware", replicate=False),
            _configure("replication", topo, reqs, "cache_aware",
                 balance_threshold=1.2, replicate=True),
        ]

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
    """High arrival rate + a TTFT SLO -> some requests must be rejected."""

    name = "overload"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        reqs = make_workload(
            num_requests=300, num_conversations=6, system_blocks=6,
            conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=9.0,
            block_tokens=BLOCK_TOKENS, output_tokens=64, seed=self.seed,
        )
        slo = 6.0
        return [
            _configure("cache_aware", topo, reqs, "cache_aware", slo_ttft=slo),
            _configure("load_balance", topo, reqs, "load_balance", slo_ttft=slo),
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("OVERLOAD: high arrival + TTFT SLO -> rejections")
        console.info("300 requests at a high rate with a TTFT SLO of 6.0. Prefix reuse")
        console.info("shortens prefill, freeing capacity, so cache-aware admits more")
        console.info("requests (fewer rejections) than the load-balancing baseline.")
        console.summary(CacheVsBaselineReport("overload", results))


class Disaggregation(Scenario):
    """Disaggregating prefill from decode protects TBT (Mooncake's headline).

    Both configs run with admission disabled (``slo_tbt=inf``, ``early_rejection=
    "off"``) so every request is served and measured. Decode capacity is fixed (two
    instances, ``DISAGG_MAX_BATCH`` each); the only difference is whether those two
    instances also do prefill. Returns ``[disaggregated, coupled]``.
    """

    name = "disaggregation"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)  # s0..s3
        reqs = make_workload(
            num_requests=120, num_conversations=8, system_blocks=2,
            conv_base_blocks=2, query_blocks=1, zipf_s=1.1, arrival_rate=1.2,
            block_tokens=BLOCK_TOKENS, output_tokens=12, seed=self.seed,
        )
        common = dict(
            simulate_decode=True, slo_tbt=float("inf"), early_rejection="off",
            max_batch=DISAGG_MAX_BATCH,
        )
        return [
            _configure("disaggregated", topo, reqs, "cache_aware",
                 prefill_pool=["s0", "s1"], decode_pool=["s2", "s3"], **common),
            _configure("coupled", _subset(topo, ["s2", "s3"]), reqs, "cache_aware",
                 coupled=True, **common),
        ]

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
    """Predicting decode load avoids wasting prefill (off/early/predict)."""

    name = "early_rejection"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        reqs = make_workload(
            num_requests=160, num_conversations=6, system_blocks=6,
            conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=20.0,
            block_tokens=BLOCK_TOKENS, output_tokens=32, seed=self.seed,
        )
        common = dict(
            simulate_decode=True, slo_tbt=EARLY_SLO_TBT, max_batch=EARLY_MAX_BATCH
        )
        return [
            _configure(mode, topo, reqs, "cache_aware", early_rejection=mode, **common)
            for mode in ("off", "early", "predict")
        ]

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
