"""``python -m kvcache_sim`` -- run the KV-cache scenarios and print results.

Run from the repo root so the package resolves::

    PYTHONPATH=. .venv/bin/python -m kvcache_sim [scenario] [-v]

For each scenario it emits:
  (a) a per-event trace  -> logged at DEBUG (shown only with -v), and
  (b) a summary comparison (cache-aware vs load-balance) -> logged at INFO.

Scenarios (positional arg; omit to run all):
  * shared_prefix   -- multi-turn conversations sharing a hot system prompt; the
                       cache-aware coordinator reuses prefixes across instances.
  * eviction        -- sweep cache capacity; hit rate rises then plateaus (LRU).
  * hotspot         -- extreme skew; hot-block replication spreads load, cuts p90.
  * overload        -- high arrival + TTFT SLO; reuse => fewer rejections.
  * disaggregation  -- dedicated decode pool holds the TBT SLO; coupling prefill
                       into decode stalls it (Mooncake's headline).
  * early_rejection -- predicting decode load avoids wasting prefill (off/early/
                       predict) while still meeting the TBT SLO.
"""

from __future__ import annotations

import argparse
from typing import List

from realsim.demo import Console, Demo, Scenario
from realsim.run import execute, Result, Run

from .report.summary import (
    CacheVsBaselineReport,
    DisaggregationReport,
    EarlyRejectionReport,
    EvictionReport,
    HotspotReport,
)
from .workload import scenarios

TRACE_LIMIT = 60


def _results(runs: List[Run]) -> List[Result]:
    return [execute(run) for run in runs]


def _shared_prefix(console: Console, args: argparse.Namespace) -> None:
    results = _results(scenarios.shared_prefix())
    console.section("SHARED PREFIX: conversations sharing a hot system prompt + context")
    console.info("directory: real torchstore.controller.Controller (off-actor)")
    console.info("4 instances (2 nodes), 200 requests, 8 conversations, Zipf skew.")
    console.info("Cache-aware routes same-prefix requests to the instance holding the")
    console.info("prefix (or pulls it once), so shared prefixes are computed ~once;")
    console.info("load-balance scatters them, recomputing prefixes on every instance.")
    console.trace(results[0].trace, limit=TRACE_LIMIT)
    console.summary(CacheVsBaselineReport("shared_prefix", results))


def _eviction(console: Console, args: argparse.Namespace) -> None:
    results = _results(scenarios.eviction_sweep())
    console.section("EVICTION: hit rate vs cache capacity (LRU)")
    console.info("400 requests, 12 conversations. As per-instance capacity grows, the")
    console.info("hot working set fits and the prefix hit rate rises, then plateaus")
    console.info("(the ~30%%->~50%% shape). Too-small caches also force more KV")
    console.info("re-fetch (fabric).")
    console.info("")
    console.info(EvictionReport(results).render())


def _hotspot(console: Console, args: argparse.Namespace) -> None:
    results = _results(scenarios.hotspot())
    console.section("HOTSPOT: extreme skew -> hot-block replication spreads load")
    console.info("One dominant conversation. Without replication (balance_threshold huge)")
    console.info("the cache-aware policy piles every hot request on the single instance")
    console.info("holding the prefix; with a moderate threshold it replicates the prefix")
    console.info("to peers (read-through), spreading load and cutting p90 TTFT.")
    console.trace(results[2].trace, limit=TRACE_LIMIT)
    console.summary(HotspotReport(results))
    console.info("(replication swaps recompute for cheap KV transfer when spreading a")
    console.info(" hot prefix to a peer -> fewer prefill tokens, more fabric bytes.)")


def _overload(console: Console, args: argparse.Namespace) -> None:
    results = _results(scenarios.overload())
    console.section("OVERLOAD: high arrival + TTFT SLO -> rejections")
    console.info("300 requests at a high rate with a TTFT SLO of 6.0. Prefix reuse")
    console.info("shortens prefill, freeing capacity, so cache-aware admits more")
    console.info("requests (fewer rejections) than the load-balancing baseline.")
    console.summary(CacheVsBaselineReport("overload", results))


def _disaggregation(console: Console, args: argparse.Namespace) -> None:
    results = _results(scenarios.disaggregation())
    console.section("DISAGGREGATION: dedicated decode pool protects TBT from prefill")
    console.info("Two decode instances, VRAM cap 8, TBT target %.3f. Admission is",
                 scenarios.DISAGG_TARGET_TBT)
    console.info("disabled (no TBT SLO gate), so BOTH configs serve every request -- the")
    console.info("contrast is purely the TBT-target attainment among served requests, not")
    console.info("a rejection count. The only difference is prefill placement: disaggregated")
    console.info("prefills on a separate pool (s0/s1) so decode (s2/s3) keeps its own")
    console.info("compute timeline; coupled runs prefill AND decode on s2/s3, so a prefill")
    console.info("can collide with a decode step and spike that request's inter-token gap.")
    console.trace(results[0].trace, limit=TRACE_LIMIT)
    console.summary(DisaggregationReport(results, scenarios.DISAGG_TARGET_TBT))
    console.info("Attainment is the fraction of served requests whose *worst* inter-token")
    console.info("gap stayed under the target. Disaggregation isolates decode from prefill,")
    console.info("so served requests hold TBT; coupling lets long prefills stall decode, so")
    console.info("a large fraction of served requests blow the target -- same load admitted.")


def _early_rejection(console: Console, args: argparse.Namespace) -> None:
    results = _results(scenarios.early_rejection())
    console.section("EARLY REJECTION: predict decode load, don't waste prefill")
    console.info("Heavy decode load with a tight TBT SLO of %.3f. Three cache-aware runs",
                 scenarios.EARLY_SLO_TBT)
    console.info("differ only in the admission policy. 'off' late-checks decode load AFTER")
    console.info("prefill and rejects on a violation -- so each rejection is a wasted")
    console.info("prefill (compute already spent). 'early' and 'predict' both gate at")
    console.info("routing, before prefill, so neither ever wastes prefill; here neither")
    console.info("rejects (both admit all). The difference is decode routing: 'early' uses")
    console.info("the current occupancy, which a slow prefill leaves reading ~empty, so it")
    console.info("piles decode onto one instance and blows the SLO; 'predict' routes by the")
    console.info("load foreseen at prefill completion, spreading decode so the SLO holds.")
    console.trace(results[2].trace, limit=TRACE_LIMIT)
    console.summary(EarlyRejectionReport(results, scenarios.EARLY_SLO_TBT))
    console.info("(Signal: wasted prefill separates 'off' from the rest; TBT attainment")
    console.info(" separates 'predict' (routes on predicted load) from 'early' (stale).)")


class KVCacheDemo(Demo):
    """The KV-cache demo: six comparisons over one real directory."""

    name = "kvcache_sim"
    description = (
        "Discrete-event simulation of a KV cache on TorchStore. Runs the given "
        "scenario (or all) and prints, per scenario, a cache-aware vs "
        "load-balance summary (INFO) and, with -v, the per-event trace (DEBUG)."
    )

    def scenarios(self):
        return [
            Scenario("shared_prefix", _shared_prefix),
            Scenario("eviction", _eviction),
            Scenario("hotspot", _hotspot),
            Scenario("overload", _overload),
            Scenario("disaggregation", _disaggregation),
            Scenario("early_rejection", _early_rejection),
        ]

    def takeaway(self, console: Console) -> None:
        console.section("TAKEAWAY")
        console.info("A cache-aware coordinator over TorchStore's existing volumes+transport")
        console.info("turns shared prefixes into cluster-wide reuse: higher hit rate, less")
        console.info("prefill compute, and lower TTFT than load-balancing that only reuses a")
        console.info("local cache. LRU eviction bounds the cache (hit rate vs capacity is the")
        console.info("sizing knob); hot-block replication spreads skew; SLO admission sheds")
        console.info("overload. All of it is control-plane policy over the same data plane.")
        console.info("On the decode side the same coordinator bounds time-between-tokens:")
        console.info("disaggregating decode onto its own pool keeps a prefill from colliding")
        console.info("with a decode step (TBT target ~100%% vs a large fraction of the SAME")
        console.info("served load missing when coupled), and predicting decode load at")
        console.info("admission avoids spending prefill on requests that can't be decoded in")
        console.info("SLO while routing decode by foreseen load -- no wasted prefill, SLO held.")


def main(argv=None) -> None:
    """Entry point (also used by the demo smoke test)."""
    KVCacheDemo().main(argv)


if __name__ == "__main__":
    main()
