"""``python -m kvcache_sim`` -- run the KV-cache scenarios and print results.

Run from the parent directory so the package resolves::

    cd toso && python -m kvcache_sim [scenario] [-v]

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
import logging

from sim_common.report import configure_logging, section

from .sim.scenarios import (
    DISAGG_TARGET_TBT,
    EARLY_SLO_TBT,
    run_disaggregation,
    run_early_rejection,
    run_eviction_sweep,
    run_hotspot,
    run_overload,
    run_shared_prefix,
)
from .sim.trace import (
    render_disaggregation,
    render_early_rejection,
    render_summary,
)

logger = logging.getLogger("kvcache_sim")


def _section(title: str) -> None:
    section(logger, title)


def _log_trace(trace, limit: int = 60) -> None:
    """Emit the event trace at DEBUG (shown only under -v); cap very long traces."""
    lines = trace.render_lines()
    logger.debug("(a) event trace (%d events%s)", len(lines),
                 f", first {limit} shown" if len(lines) > limit else "")
    for line in lines[:limit]:
        logger.debug(line)
    if len(lines) > limit:
        logger.debug("... (%d more)", len(lines) - limit)
    logger.debug("")


def _shared_prefix() -> None:
    cache_aware, baseline = run_shared_prefix()
    _section("SHARED PREFIX: conversations sharing a hot system prompt + context")
    logger.info("directory: real torchstore.controller.Controller (off-actor)")
    logger.info("4 instances (2 nodes), 200 requests, 8 conversations, Zipf skew.")
    logger.info("Cache-aware routes same-prefix requests to the instance holding the")
    logger.info("prefix (or pulls it once), so shared prefixes are computed ~once;")
    logger.info("load-balance scatters them, recomputing prefixes on every instance.")
    _log_trace(cache_aware.trace)
    logger.info("(b) summary")
    logger.info(render_summary("shared_prefix", cache_aware.metrics, baseline.metrics))


def _eviction() -> None:
    rows = run_eviction_sweep()
    _section("EVICTION: hit rate vs cache capacity (LRU)")
    logger.info("400 requests, 12 conversations. As per-instance capacity grows, the")
    logger.info("hot working set fits and the prefix hit rate rises, then plateaus")
    logger.info("(the ~30%%->~50%% shape). Too-small caches also force more KV")
    logger.info("re-fetch (fabric).")
    logger.info("")
    logger.info("  %10s %12s %14s", "capacity", "hit_rate", "fabric_bytes")
    for cap, hr, fb in rows:
        logger.info("  %10d %11.1f%% %14d", cap, 100.0 * hr, fb)


def _hotspot() -> None:
    baseline, no_repl, repl = run_hotspot()
    _section("HOTSPOT: extreme skew -> hot-block replication spreads load")
    logger.info("One dominant conversation. Without replication (balance_threshold huge)")
    logger.info("the cache-aware policy piles every hot request on the single instance")
    logger.info("holding the prefix; with a moderate threshold it replicates the prefix")
    logger.info("to peers (read-through), spreading load and cutting p90 TTFT.")
    _log_trace(repl.trace)
    logger.info("(b) summary")
    logger.info("  %-26s%12s%12s%12s", "", "load-bal", "cache/no-repl", "cache/repl")
    logger.info("  %-26s%12.3f%12.3f%12.3f", "mean TTFT",
                baseline.metrics.mean_ttft, no_repl.metrics.mean_ttft,
                repl.metrics.mean_ttft)
    logger.info("  %-26s%12.3f%12.3f%12.3f", "p90 TTFT",
                baseline.metrics.pct_ttft(90), no_repl.metrics.pct_ttft(90),
                repl.metrics.pct_ttft(90))
    logger.info("  %-26s%12.1f%12.1f%12.1f", "prefix hit rate %%",
                100 * baseline.metrics.hit_rate, 100 * no_repl.metrics.hit_rate,
                100 * repl.metrics.hit_rate)
    logger.info("  %-26s%12d%12d%12d", "prefill tokens",
                baseline.metrics.compute_tokens, no_repl.metrics.compute_tokens,
                repl.metrics.compute_tokens)
    logger.info("  %-26s%12d%12d%12d", "KV fabric bytes",
                baseline.metrics.fabric_bytes, no_repl.metrics.fabric_bytes,
                repl.metrics.fabric_bytes)
    logger.info("(replication swaps recompute for cheap KV transfer when spreading a")
    logger.info(" hot prefix to a peer -> fewer prefill tokens, more fabric bytes.)")


def _overload() -> None:
    cache_aware, baseline = run_overload()
    _section("OVERLOAD: high arrival + TTFT SLO -> rejections")
    logger.info("300 requests at a high rate with a TTFT SLO of 6.0. Prefix reuse")
    logger.info("shortens prefill, freeing capacity, so cache-aware admits more")
    logger.info("requests (fewer rejections) than the load-balancing baseline.")
    logger.info("(b) summary")
    logger.info(render_summary("overload", cache_aware.metrics, baseline.metrics))


def _disaggregation() -> None:
    disagg, coupled = run_disaggregation()
    _section("DISAGGREGATION: dedicated decode pool protects TBT from prefill")
    logger.info("Two decode instances, VRAM cap 8, TBT target %.3f. Admission is",
                DISAGG_TARGET_TBT)
    logger.info("disabled (no TBT SLO gate), so BOTH configs serve every request -- the")
    logger.info("contrast is purely the TBT-target attainment among served requests, not")
    logger.info("a rejection count. The only difference is prefill placement: disaggregated")
    logger.info("prefills on a separate pool (s0/s1) so decode (s2/s3) keeps its own")
    logger.info("compute timeline; coupled runs prefill AND decode on s2/s3, so a prefill")
    logger.info("can collide with a decode step and spike that request's inter-token gap.")
    _log_trace(disagg.trace)
    logger.info("(b) summary")
    logger.info(render_disaggregation(disagg.metrics, coupled.metrics,
                                      DISAGG_TARGET_TBT))
    logger.info("Attainment is the fraction of served requests whose *worst* inter-token")
    logger.info("gap stayed under the target. Disaggregation isolates decode from prefill,")
    logger.info("so served requests hold TBT; coupling lets long prefills stall decode, so")
    logger.info("a large fraction of served requests blow the target -- same load admitted.")


def _early_rejection() -> None:
    off, early, predict = run_early_rejection()
    _section("EARLY REJECTION: predict decode load, don't waste prefill")
    logger.info("Heavy decode load with a tight TBT SLO of %.3f. Three cache-aware runs",
                EARLY_SLO_TBT)
    logger.info("differ only in the admission policy. 'off' late-checks decode load AFTER")
    logger.info("prefill and rejects on a violation -- so each rejection is a wasted")
    logger.info("prefill (compute already spent). 'early' and 'predict' both gate at")
    logger.info("routing, before prefill, so neither ever wastes prefill; here neither")
    logger.info("rejects (both admit all). The difference is decode routing: 'early' uses")
    logger.info("the current occupancy, which a slow prefill leaves reading ~empty, so it")
    logger.info("piles decode onto one instance and blows the SLO; 'predict' routes by the")
    logger.info("load foreseen at prefill completion, spreading decode so the SLO holds.")
    _log_trace(predict.trace)
    logger.info("(b) summary")
    logger.info(render_early_rejection(off.metrics, early.metrics, predict.metrics,
                                       EARLY_SLO_TBT))
    logger.info("(Signal: wasted prefill separates 'off' from the rest; TBT attainment")
    logger.info(" separates 'predict' (routes on predicted load) from 'early' (stale).)")


def _takeaway() -> None:
    _section("TAKEAWAY")
    logger.info("A cache-aware coordinator over TorchStore's existing volumes+transport")
    logger.info("turns shared prefixes into cluster-wide reuse: higher hit rate, less")
    logger.info("prefill compute, and lower TTFT than load-balancing that only reuses a")
    logger.info("local cache. LRU eviction bounds the cache (hit rate vs capacity is the")
    logger.info("sizing knob); hot-block replication spreads skew; SLO admission sheds")
    logger.info("overload. All of it is control-plane policy over the same data plane.")
    logger.info("On the decode side the same coordinator bounds time-between-tokens:")
    logger.info("disaggregating decode onto its own pool keeps a prefill from colliding")
    logger.info("with a decode step (TBT target ~100%% vs a large fraction of the SAME")
    logger.info("served load missing when coupled), and predicting decode load at")
    logger.info("admission avoids spending prefill on requests that can't be decoded in")
    logger.info("SLO while routing decode by foreseen load -- no wasted prefill, SLO held.")


SCENARIOS = {
    "shared_prefix": _shared_prefix,
    "eviction": _eviction,
    "hotspot": _hotspot,
    "overload": _overload,
    "disaggregation": _disaggregation,
    "early_rejection": _early_rejection,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m kvcache_sim",
        description="Discrete-event simulation of a KV cache on "
                    "TorchStore. Runs the given scenario (or all) and prints, per "
                    "scenario, a cache-aware vs load-balance summary (INFO) and, "
                    "with -v, the per-event trace (DEBUG).",
    )
    parser.add_argument(
        "scenario", nargs="?", choices=sorted(SCENARIOS),
        help="scenario to run (default: all). one of: " + ", ".join(sorted(SCENARIOS)),
    )
    parser.add_argument(
        "-v", "--verbose", "--debug", action="store_true", dest="verbose",
        help="show the per-event trace (log level DEBUG)",
    )
    args = parser.parse_args(argv)

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.scenario is None:
        _shared_prefix()
        _eviction()
        _hotspot()
        _overload()
        _disaggregation()
        _early_rejection()
        _takeaway()
    else:
        SCENARIOS[args.scenario]()


if __name__ == "__main__":
    main()
