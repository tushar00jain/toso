"""``python -m dedup_sim`` -- run the dedup demo on the real TorchStore directory.

Run from the repo root so the package resolves::

    PYTHONPATH=. .venv/bin/python -m dedup_sim [-v]

The demo drives a synchronized read burst over the **real** ``Controller``
directory + **real** client/transport (via ``realsim``) under two policies:

  * the unrouted baseline (no policy installed): every reader pulls from the
    origin -- ``m x`` fabric; and
  * the dedup policy (:class:`dedup_sim.control.routing.DedupPolicy`): the
    controller routes each reader to a peer and withholds the answer until that
    peer's read-through put registers, so the burst becomes a chain
    (``fanout_cap=1``) or tree (``fanout_cap=2``) and each unique byte crosses
    the fabric once -- 1x fabric.

Output is routed through the ``logging`` module:
  * INFO (default): section headers, the dedup-vs-naive summary + ASCII
    source->dest diagram, the takeaway.
  * DEBUG (``-v``/``--debug``): additionally the full per-event virtual-time trace.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from sim_common import config
from realsim.cli import add_run_flags, apply_run_flags, log_trace
from sim_common.report import section

from dedup_sim.report.summary import (
    render_baseline_summary,
    render_dedup_summary,
)
from dedup_sim.workload.scenarios import NUM_READERS, run_dedup_vs_baseline

logger = logging.getLogger("dedup_sim")

def _section(title: str) -> None:
    section(logger, title)


def _log_trace(trace, limit: Optional[int] = None) -> None:
    """Dump the trace (shared). Fingerprints are labelled per run below."""
    log_trace(logger, trace, limit=limit)


def _dedup() -> None:
    comparison = run_dedup_vs_baseline()
    naive = comparison.baseline
    payload = comparison.payload_bytes
    if config.current().fingerprint:
        logger.info("naive run fingerprint: %s", naive.trace.fingerprint())

    _section(
        f"DEDUP on the REAL directory  --  {comparison.num_readers} readers get W"
    )
    logger.info("directory: real torchstore.controller.Controller (real Trie state)")
    logger.info("payload(W): %dB   1x-union target (each unique byte once): %dB",
                payload, payload)

    for cap, routed in comparison.routed:
        _routed(cap, routed, comparison)

    _section("NAIVE baseline  --  every reader pulls from the origin")
    _log_trace(naive.trace)
    logger.info("(b) summary")
    logger.info(render_baseline_summary(naive, comparison.num_readers))


def _routed(cap: int, routed, comparison) -> None:
    """One routed configuration: its section, trace, summary and fingerprint."""
    naive, payload = comparison.baseline, comparison.payload_bytes
    _section(f"dedup policy  --  fanout_cap={cap} ({'chain' if cap == 1 else 'tree'})")
    _log_trace(routed.trace)
    logger.info("(b) summary")
    logger.info(render_dedup_summary(routed, naive, cap))
    if config.current().fingerprint:
        logger.info("dedup(cap=%d) run fingerprint: %s", cap, routed.trace.fingerprint())
    # 1x proven live on the real directory.
    assert routed.ledger.origin_bytes == payload
    assert naive.ledger.origin_bytes == comparison.num_readers * payload


def _takeaway() -> None:
    _section("TAKEAWAY")
    logger.info("On the real directory, dedup registers each finished reader as a")
    logger.info("read-through source (real notify_put_batch), so later readers pull")
    logger.info("from a peer, not the origin: each unique byte crosses the fabric")
    logger.info("ONCE (1x vs mx). Wallclock depends on fanout_cap/topology -- a chain")
    logger.info("(cap=1) is more hops, a tree (cap=2) narrows the gap; both stay 1x.")


SCENARIOS = {
    "dedup": _dedup,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dedup_sim",
        description="Dedup read-routing demo on the real TorchStore directory. "
                    "Runs a synchronized read burst under the naive baseline and "
                    "the dedup policy (chain + tree), printing the fabric summary + "
                    "ASCII diagram (INFO) and, with -v, the full per-event "
                    "virtual-time trace (DEBUG).",
    )
    parser.add_argument(
        "scenario", nargs="?", choices=sorted(SCENARIOS),
        help="scenario to run (default: all). one of: " + ", ".join(sorted(SCENARIOS)),
    )
    add_run_flags(parser)
    args = parser.parse_args(argv)

    # Set the process config once from the CLI flags (unset -> env / default);
    # the scenarios' Traces + controller adapters + resource registry + the
    # transport's collapse decision read it ambiently.
    apply_run_flags(args)

    if args.scenario is None:
        _dedup()
        _takeaway()
    else:
        SCENARIOS[args.scenario]()


if __name__ == "__main__":
    main()
