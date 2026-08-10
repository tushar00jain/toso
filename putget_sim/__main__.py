"""``python -m putget_sim`` -- run the unrouted put/get burst and print it.

Mirrors the ``dedup_sim`` demo style (``sim_common.report`` headers, a digest at
INFO and the full per-event trace at DEBUG under ``-v``), but every layer under
the scenario is **real** TorchStore code driven off-actor on the deterministic
virtual-clock engine. The scenario itself (``putget_sim/workload/put_get.py``) is
ordinary user code and installs no policy.

Run from the repo root so the package resolves, with the venv interpreter::

    PYTHONPATH=. .venv/bin/python -m putget_sim [-m N] [-v]
"""

from __future__ import annotations

import argparse
import logging

from sim_common import config
from sim_common.cost_model import DEFAULT_PROFILE
from realsim.cli import add_run_flags, apply_run_flags, log_trace
from sim_common.report import section

from putget_sim.harness import run_burst
from putget_sim.report.summary import render_burst_summary
from putget_sim.workload.put_get import MODE_META, MODE_METADATA

# The cost model is driven by a MachineProfile that describes the *target*
# machine being simulated -- never the box this demo runs on. Costs are analytic
# functions of modeled bytes/flops, so the same profile yields the same trace on
# any host. The demo uses the illustrative DEFAULT_PROFILE; a real study would
# swap in a profile measured/spec'd for its target hardware.
PROFILE = DEFAULT_PROFILE

logger = logging.getLogger("putget_sim")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m putget_sim",
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
    add_run_flags(parser)
    args = parser.parse_args(argv)

    # Set the process config once, up front, from the CLI flag (an unset flag
    # defers to the TOSO_* env / default). Every Trace built below reads it
    # ambiently -- no need to thread it through run_burst.
    apply_run_flags(args, logger)


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
    log_trace(logger, res.trace)
    if config.current().fingerprint:
        logger.info("run fingerprint: %s", res.trace.fingerprint())
    logger.info("(b) summary")
    logger.info(render_burst_summary(res))
    logger.info(
        "no routing policy => %dx fabric (every reader pulls the origin). Installing "
        "a Policy (proposed/policy.py) in the controller's locate_volumes -- as "
        "dedup_sim does -- routes later readers to read-through peers and cuts this "
        "toward 1x, with the scenario code above unchanged.",
        res.num_readers,
    )


if __name__ == "__main__":
    main()
