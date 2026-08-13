"""``python -m putget_sim`` -- run the unrouted put/get burst and print it.

Every layer under the scenario is **real** TorchStore code driven off-actor on
the deterministic virtual-clock engine. The scenario itself
(``putget_sim/workload/put_get.py``) is ordinary user code and installs no selector.

What is run and how it is narrated is
:class:`putget_sim.workload.scenarios.Burst`; this file only declares the demo.

Run from the repo root so the package resolves, with the venv interpreter::

    PYTHONPATH=. .venv/bin/python -m putget_sim [-m N] [-v]
"""

from __future__ import annotations

import argparse

from realsim.demo import Demo

from .workload.put_get import MODE_META, MODE_METADATA
from .workload.scenarios import Burst, NUM_READERS


class PutGetDemo(Demo):
    """The baseline demo: one burst, no selector, no data plane."""

    name = "putget_sim"
    description = (
        "Deterministic read-burst simulation over the REAL TorchStore "
        "client/controller/transport, with no selector installed. Prints the "
        "summary + source->dest tree (INFO) and, with -v, the full per-event "
        "trace (DEBUG)."
    )
    # Keep the root logger at INFO so torchstore's own DEBUG latency logs stay
    # quiet; only this demo's logger drops to DEBUG under -v.
    own_logger = True

    def scenarios(self):
        return [Burst()]

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
