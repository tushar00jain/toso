"""``python -m realsim.run_realsim`` -- run the read-burst scenario and print it.

Mirrors the ``dedup_sim`` demo style (``sim_common.report`` headers, a digest at
INFO and the full per-event trace at DEBUG under ``-v``), but every layer below
the coordinator is **real** TorchStore code driven off-actor on the deterministic
virtual-clock engine.

Run from the worktree with the venv interpreter::

    PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.run_realsim [-m N] [-v]
"""

from __future__ import annotations

import argparse
import logging

from sim_common import config
from sim_common.cost_model import DEFAULT_PROFILE
from sim_common.report import configure_logging, section

from realsim.scenarios.burst_get import (
    MODE_META,
    MODE_METADATA,
    render_burst_summary,
    run_burst,
)

# The cost model is driven by a MachineProfile that describes the *target*
# machine being simulated -- never the box this demo runs on. Costs are analytic
# functions of modeled bytes/flops, so the same profile yields the same trace on
# any host. The demo uses the illustrative DEFAULT_PROFILE; a real study would
# swap in a profile measured/spec'd for its target hardware.
PROFILE = DEFAULT_PROFILE

logger = logging.getLogger("realsim")


def _log_trace(trace) -> None:
    """Emit every recorded event line at DEBUG (shown only under -v)."""
    logger.debug("(a) event trace")
    for line in trace.render_lines():
        logger.debug(line)
    logger.debug("")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m realsim.run_realsim",
        description="Deterministic read-burst simulation over the REAL TorchStore "
        "client/controller/transport. Prints the summary + source->dest tree "
        "(INFO) and, with -v, the full per-event trace (DEBUG).",
    )
    parser.add_argument(
        "-m", "--readers", type=int, default=3,
        help="number of readers in the burst (default: 3)",
    )
    parser.add_argument(
        "-n", "--elements", type=int, default=16,
        help="elements in W (float32); payload = 4*N bytes (default: 16)",
    )
    parser.add_argument(
        "--mode", choices=(MODE_META, MODE_METADATA), default=MODE_META,
        help="data-plane carrier: 'meta' (zero-storage meta tensor, default) or "
        "'metadata' (a (shape, dtype) descriptor, no tensor at all). Both are "
        "allocation-free and drive the real store round-trip.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random_seed for the engine's random ready-queue mode "
        "(default: None -> FIFO, reproducible)",
    )
    parser.add_argument(
        "--fingerprint", action="store_true",
        help="print the run's trace fingerprint (a determinism-debugging digest, "
        "folded from the trace on demand; off by default -- it is not part of the "
        "performance measurement)",
    )
    parser.add_argument(
        "-v", "--verbose", "--debug", action="store_true", dest="verbose",
        help="show the full per-event trace (log level DEBUG)",
    )
    args = parser.parse_args(argv)

    # Set the process config once, up front, from the CLI flag (an unset flag
    # defers to the TOSO_* env / default). Every Trace built below reads it
    # ambiently -- no need to thread it through run_burst.
    config.configure(fingerprint=args.fingerprint or None)

    # Keep the ROOT logger at INFO so the real torchstore code's own DEBUG
    # latency logs (emitted via ``logging.log`` on root) stay quiet; then let the
    # stdout handler and the realsim logger drop to DEBUG so ``-v`` shows
    # realsim's virtual-time event trace (propagated records aren't re-gated by
    # the root level). The real code still executes -- this is verbosity only.
    configure_logging(logging.INFO)
    if args.verbose:
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    section(
        logger,
        f"READ BURST: {args.readers} readers each get W over the REAL TorchStore",
    )
    logger.info(
        "real client planning + real controller directory + real InMemoryStore, "
        "on the deterministic virtual-clock engine."
    )
    res = run_burst(
        num_readers=args.readers,
        n=args.elements,
        mode=args.mode,
        profile=PROFILE,
        random_seed=args.seed,
    )
    _log_trace(res.trace)
    if config.current().fingerprint:
        logger.info("run fingerprint: %s", res.trace.fingerprint())
    logger.info("(b) summary")
    logger.info(render_burst_summary(res))
    logger.info(
        "naive policy => %dx fabric (every reader pulls the origin). The pluggable "
        "ReadPolicy seam (coordinator/model.py) would register read-through peers "
        "in the real directory to cut this toward 1x.",
        res.num_readers,
    )


if __name__ == "__main__":
    main()
