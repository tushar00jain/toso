"""``python -m dedup_sim`` -- run the dynamic dedup scenarios and print results.

Run from the parent directory so the package resolves::

    cd toso && python -m dedup_sim [scenario] [-v]

For each scenario it emits:
  (a) the chronological event trace  -> logged at DEBUG (shown only with -v), and
  (b) the summary + ASCII source->dest diagram (§9 of SPEC.md) -> logged at INFO.

Output is routed through the ``logging`` module:
  * INFO (default): section headers, summaries, ASCII diagrams, the takeaway --
    the digest, without the per-event spam.
  * DEBUG (``-v``/``--debug``): additionally the full per-event trace.

Scenarios (positional arg; omit to run all):
  * toy        -- full-replication burst; the dynamic dedup fan-out at
                  FANOUT_CAP=1 (chain) and =2 (tree), plus the naive baseline.
  * reshard    -- trainer partition != generator partition; fabric still 1x.
  * versioning -- burst, then a put that bumps the version, then a second burst
                  that re-pulls from the trainer (cache invalidated).
"""

from __future__ import annotations

import argparse
import logging

from sim_common.report import configure_logging, section

from .sim.scenarios import (
    reshard_scenario,
    run,
    run_versioning,
    toy_scenario,
)
from .sim.store_index import HAVE_REAL, USING_REAL
from .sim.trace import render_summary

logger = logging.getLogger("dedup_sim")


# --------------------------------------------------------------------------- #
# Logging helpers -- INFO for the digest, DEBUG for the per-event trace.
# --------------------------------------------------------------------------- #
def _section(title: str) -> None:
    section(logger, title)


def _log_trace(trace, note: str = "") -> None:
    """Emit every recorded event line at DEBUG (shown only under -v)."""
    logger.debug("(a) event trace%s", f"  {note}" if note else "")
    for line in trace.render_lines():
        logger.debug(line)
    logger.debug("")  # spacer before the summary


def _r(region) -> str:
    start, end = region
    return f"[{start},{end})"


def _dedup_section(title: str, scn, cap: int, naive) -> None:
    """Log one dedup run: (a) event trace (DEBUG), (b) summary + diagram (INFO)."""
    dedup = run(scn, "dedup", fanout_cap=cap)
    _section(title)
    _log_trace(dedup.trace)
    logger.info("(b) summary")
    logger.info(render_summary(dedup.metrics, naive.metrics, scn.union_bytes, cap))
    logger.info("    peak concurrent serves at any source: %d (cap %d)",
                dedup.peak_serving, cap)


def _toy() -> None:
    scn = toy_scenario(num_gens=3)
    naive = run(scn, "naive")

    idx_path = (
        "real torchstore.controller.Controller"
        if USING_REAL
        else f"faithful shim (real import {'available' if HAVE_REAL else 'unavailable'}, "
        "not driveable single-threaded)"
    )

    _section(f"TOY: {scn.name}  ({len(scn.needs)} generators need W[0,8))")
    logger.info("store index: %s", idx_path)
    logger.info("union of needs (1x fabric target): %dB", scn.union_bytes)

    for cap in (1, 2):
        topo = "chain" if cap == 1 else "tree"
        _dedup_section(f"dynamic dedup coordinator  --  FANOUT_CAP={cap} ({topo})",
                       scn, cap, naive)

    _section("NAIVE baseline  --  every reader pulls from the trainer")
    _log_trace(naive.trace)
    logger.info("(b) summary")
    naive_x = naive.metrics.fabric_bytes / scn.union_bytes
    logger.info("fabric(trainer->gen): naive=%dB (%.1fx)   wallclock=%.3f",
                naive.metrics.fabric_bytes, naive_x, naive.metrics.wallclock)
    logger.info("every generator pulls the full region cross-node -> m x fabric; "
                "fewer hops so it can win wallclock, but pays 3x bytes.")


def _reshard() -> None:
    scn = reshard_scenario()
    naive = run(scn, "naive")
    _section(f"RESHARD: {scn.name}")
    logger.info("stored (trainer partition):  %s",
                ", ".join(f"{v}={_r(r)}" for v, r in scn.stored))
    logger.info("needs (generator partition): %s",
                ", ".join(f"{g}={_r(need[0])}"
                          for g, need in sorted(scn.needs.items())))
    logger.info("atomic regions after split:  %s",
                ", ".join(_r(a) for a in scn.atomics))
    logger.info("union of needs (1x fabric target): %dB", scn.union_bytes)
    logger.info("Trainer partition != generator partition, and needs overlap, so the")
    logger.info("coordinator splits into atomic regions and dedups the overlap: each")
    logger.info("unique atomic still leaves the trainer exactly once (1x fabric).")
    _dedup_section("dynamic dedup coordinator  --  FANOUT_CAP=1", scn, 1, naive)
    dedup = run(scn, "dedup", fanout_cap=1)
    assert dedup.metrics.fabric_bytes == scn.union_bytes  # 1x, proven live
    logger.info("    check: dedup fabric %dB == union %dB (1.0x)   naive %dB (%.1fx)",
                dedup.metrics.fabric_bytes, scn.union_bytes,
                naive.metrics.fabric_bytes,
                naive.metrics.fabric_bytes / scn.union_bytes)


def _versioning() -> None:
    res = run_versioning(bump=True, fanout_cap=1)
    _section("VERSIONING: burst, put (version bump), burst -- cache invalidation")
    logger.info("Two bursts of the toy on ONE coordinator. A put between them bumps")
    logger.info("the version, dropping the stale cache; burst 2 must re-pull from the")
    logger.info("trainer instead of sourcing from the (now stale) generator caches.")
    _log_trace(res.trace, note="(MARK lines delimit the phases)")
    logger.info("(b) summary")
    logger.info("burst 1 trainer fabric: %dB (%.1fx union)  -- cold cache, pull once",
                res.fabric1, res.fabric1 / res.union_bytes)
    logger.info("burst 2 trainer fabric: %dB (%.1fx union)  -- version bumped, "
                "cache invalid -> re-pull",
                res.fabric2, res.fabric2 / res.union_bytes)
    nobump = run_versioning(bump=False, fanout_cap=1)
    logger.info("contrast (no bump): burst 2 trainer fabric would be %dB "
                "(cache still valid -> served entirely by peers)", nobump.fabric2)


def _takeaway() -> None:
    _section("TAKEAWAY")
    logger.info("Dynamic dedup moves each unique region across the fabric ONCE "
                "(1x vs mx),")
    logger.info("even across a reshard (trainer partition != generator partition) and")
    logger.info("re-pulling correctly after a version bump. Wallclock depends on")
    logger.info("FANOUT_CAP/topology: FANOUT_CAP=1 is a chain (g0->g1->g2, more hops),")
    logger.info("FANOUT_CAP=2 is a shallower tree that narrows the wallclock gap.")
    logger.info("Naive is fewer hops but pays mx fabric.")


SCENARIOS = {
    "toy": _toy,
    "reshard": _reshard,
    "versioning": _versioning,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dedup_sim",
        description="Discrete-event simulation of the dynamic dedup coordinator. "
                    "Runs the given scenario (or all of them) and prints, per "
                    "scenario, the summary + ASCII diagram (INFO) and, with -v, "
                    "the full per-event trace (DEBUG).",
    )
    parser.add_argument(
        "scenario", nargs="?", choices=sorted(SCENARIOS),
        help="scenario to run (default: run all). one of: "
             + ", ".join(sorted(SCENARIOS)),
    )
    parser.add_argument(
        "-v", "--verbose", "--debug", action="store_true", dest="verbose",
        help="show the full per-event trace (log level DEBUG)",
    )
    args = parser.parse_args(argv)

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.scenario is None:
        _toy()
        _reshard()
        _versioning()
        _takeaway()
    else:
        SCENARIOS[args.scenario]()


if __name__ == "__main__":
    main()
