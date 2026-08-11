"""``python -m putget_sim`` -- run the unrouted put/get burst and print it.

Every layer under the scenario is **real** TorchStore code driven off-actor on
the deterministic virtual-clock engine. The scenario itself
(``putget_sim/workload/put_get.py``) is ordinary user code and installs no policy.

Run from the repo root so the package resolves, with the venv interpreter::

    PYTHONPATH=. .venv/bin/python -m putget_sim [-m N] [-v]
"""

from __future__ import annotations

import argparse

from realsim.demo import Console, Demo, Scenario
from sim_common.cost_model import DEFAULT_PROFILE

from .report.summary import BurstReport
from .workload.put_get import MODE_META, MODE_METADATA
from .workload.scenarios import burst, NUM_READERS

# The cost model is driven by a MachineProfile that describes the *target*
# machine being simulated -- never the box this demo runs on. Costs are analytic
# functions of modeled bytes/flops, so the same profile yields the same trace on
# any host. The demo uses the illustrative DEFAULT_PROFILE; a real study would
# swap in a profile measured/spec'd for its target hardware.
PROFILE = DEFAULT_PROFILE


def _runs(args: argparse.Namespace):
    """The one configuration: no policy, no data plane."""
    return burst(args.readers, n=args.elements, mode=args.mode, profile=PROFILE)


def _show(console: Console, results) -> None:
    (result,) = results
    console.section(
        f"READ BURST: {result.workload.num_readers} readers each get W over the "
        f"REAL TorchStore"
    )
    console.info(
        "real client planning + real controller directory + real InMemoryStore, "
        "on the deterministic virtual-clock engine."
    )
    console.trace(result.trace)
    console.summary(BurstReport(result))
    console.info(
        "no routing policy => %dx fabric (every reader pulls the origin). Installing "
        "a Policy (proposed/policy.py) in the controller's locate_volumes -- as "
        "dedup_sim does -- routes later readers to read-through peers and cuts this "
        "toward 1x, with the scenario code above unchanged.",
        result.workload.num_readers,
    )


class PutGetDemo(Demo):
    """The baseline demo: one burst, no policy, no data plane."""

    name = "putget_sim"
    description = (
        "Deterministic read-burst simulation over the REAL TorchStore "
        "client/controller/transport, with no policy installed. Prints the "
        "summary + source->dest tree (INFO) and, with -v, the full per-event "
        "trace (DEBUG)."
    )
    # Keep the root logger at INFO so torchstore's own DEBUG latency logs stay
    # quiet; only this demo's logger drops to DEBUG under -v.
    own_logger = True

    def scenarios(self):
        return [Scenario("burst", _runs, _show)]

    def run_knobs(self, args: argparse.Namespace):
        """``--seed`` is an engine knob, so it reaches execute(), not the scenario."""
        return {"random_seed": args.seed}

    def flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-m", "--readers", type=int, default=NUM_READERS,
            help=f"number of readers in the burst (default: {NUM_READERS})",
        )
        parser.add_argument(
            "-n", "--elements", type=int, default=16,
            help="elements in W (float32); payload = 4*N bytes (default: 16)",
        )
        parser.add_argument(
            "--mode", choices=(MODE_META, MODE_METADATA), default=MODE_META,
            help="data-plane carrier: 'meta' (zero-storage meta tensor, default) "
            "or 'metadata' (a (shape, dtype) descriptor, no tensor at all). Both "
            "are allocation-free and drive the real store round-trip.",
        )
        parser.add_argument(
            "--seed", type=int, default=None,
            help="random_seed for the engine's random ready-queue mode "
            "(default: None -> FIFO, reproducible)",
        )


def main(argv=None) -> None:
    """Entry point (also used by the demo smoke test)."""
    PutGetDemo().main(argv)


if __name__ == "__main__":
    main()
