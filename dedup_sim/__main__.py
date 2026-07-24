"""``python -m dedup_sim`` -- run the dedup demo on the real TorchStore directory.

Run from the repo root so the package resolves::

    PYTHONPATH=. .venv/bin/python -m dedup_sim [-v]

The demo drives a synchronized read burst over the **real** ``Controller``
directory + **real** client/transport (via ``realsim``) under two policies:

  * the naive baseline (realsim's ``NaivePolicy``): every reader pulls from the
    origin -- ``m x`` fabric; and
  * the dedup policy (:class:`dedup_sim.policy.DedupPolicy`): the burst is staged
    into a chain (``fanout_cap=1``) or tree (``fanout_cap=2``) of read-through
    peers so each unique byte crosses the fabric once -- 1x fabric.

Output is routed through the ``logging`` module:
  * INFO (default): section headers, the dedup-vs-naive summary + ASCII
    source->dest diagram, the takeaway.
  * DEBUG (``-v``/``--debug``): additionally the full per-event virtual-time trace.
"""

from __future__ import annotations

import argparse
import logging

from sim_common import config
from sim_common.report import configure_logging, section

from dedup_sim.scenario import (
    render_dedup_summary,
    run_dedup_burst,
    run_naive_burst,
)

logger = logging.getLogger("dedup_sim")

NUM_READERS = 3


def _log_trace(trace, note: str = "") -> None:
    """Emit every recorded virtual-time event at DEBUG (shown only under -v)."""
    logger.debug("(a) event trace%s", f"  {note}" if note else "")
    for line in trace.render_lines():
        logger.debug(line)
    logger.debug("")


def _demo() -> None:
    fingerprint = config.current().fingerprint
    naive = run_naive_burst(num_readers=NUM_READERS)
    payload = naive.expected.numel() * naive.expected.element_size()
    if fingerprint:
        logger.info("naive run fingerprint: %s", naive.trace.fingerprint())

    section(logger, f"DEDUP on the REAL directory  --  {NUM_READERS} readers get W")
    logger.info("directory: real torchstore.controller.Controller (real Trie state)")
    logger.info("payload(W): %dB   1x-union target (each unique byte once): %dB",
                payload, payload)

    for cap in (1, 2):
        topo = "chain" if cap == 1 else "tree"
        dedup = run_dedup_burst(num_readers=NUM_READERS, fanout_cap=cap)
        section(logger, f"dedup policy  --  fanout_cap={cap} ({topo})")
        _log_trace(dedup.trace)
        logger.info("(b) summary")
        logger.info(render_dedup_summary(dedup, naive, cap))
        if fingerprint:
            logger.info("dedup(cap=%d) run fingerprint: %s", cap, dedup.trace.fingerprint())
        # 1x proven live on the real directory.
        assert dedup.metrics.fabric_bytes == payload
        assert naive.metrics.fabric_bytes == NUM_READERS * payload

    section(logger, "NAIVE baseline  --  every reader pulls from the origin")
    _log_trace(naive.trace)
    logger.info("(b) summary")
    logger.info("fabric(origin->readers): naive=%dB (%.1fx)   wallclock=%.4f",
                naive.metrics.fabric_bytes,
                naive.metrics.fabric_bytes / payload,
                naive.metrics.wallclock)
    logger.info("every reader pulls the full payload cross-node -> m x fabric; "
                "concurrent so it wins wallclock, but pays %dx the bytes.",
                NUM_READERS)

    section(logger, "TAKEAWAY")
    logger.info("On the real directory, dedup registers each finished reader as a")
    logger.info("read-through source (real notify_put_batch), so later readers pull")
    logger.info("from a peer, not the origin: each unique byte crosses the fabric")
    logger.info("ONCE (1x vs mx). Wallclock depends on fanout_cap/topology -- a chain")
    logger.info("(cap=1) is more hops, a tree (cap=2) narrows the gap; both stay 1x.")


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
        "-v", "--verbose", "--debug", action="store_true", dest="verbose",
        help="show the full per-event virtual-time trace (log level DEBUG)",
    )
    parser.add_argument(
        "--fingerprint", action="store_true",
        help="print each run's trace fingerprint (a determinism-debugging digest, "
        "folded from the trace on demand; off by default -- it is not part of the "
        "performance measurement)",
    )
    parser.add_argument(
        "--shim-directory", action="store_true",
        help="back the controller directory with a lightweight dict shim instead "
        "of the real torchstore Trie (opt-in; skips the per-key trie tax on scale "
        "runs). Metrics are byte-identical either way; the real directory is the "
        "default.",
    )
    args = parser.parse_args(argv)

    # Set the process config once from the CLI flags (unset -> env / default);
    # the scenarios' Traces + controller adapters read it ambiently.
    config.configure(
        fingerprint=args.fingerprint or None,
        real_directory=False if args.shim_directory else None,
    )

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    _demo()


if __name__ == "__main__":
    main()
