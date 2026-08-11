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

The comparison itself -- which runs, and how they are narrated -- is
:class:`dedup_sim.workload.scenarios.Dedup`. This file only declares the demo.

Output is routed through the ``logging`` module:
  * INFO (default): section headers, the dedup-vs-naive summary + ASCII
    source->dest diagram, the takeaway.
  * DEBUG (``-v``/``--debug``): additionally the full per-event virtual-time trace.
"""

from __future__ import annotations

from realsim.demo import Console, Demo

from .workload.scenarios import Dedup


class DedupDemo(Demo):
    """The dedup demo: one burst, three policies, one comparison."""

    name = "dedup_sim"
    description = (
        "Dedup read-routing demo on the real TorchStore directory. Runs a "
        "synchronized read burst under the naive baseline and the dedup policy "
        "(chain + tree), printing the fabric summary + ASCII diagram (INFO) and, "
        "with -v, the full per-event virtual-time trace (DEBUG)."
    )

    def scenarios(self):
        return [Dedup()]

    def takeaway(self, console: Console) -> None:
        console.section("TAKEAWAY")
        console.info("On the real directory, dedup registers each finished reader as a")
        console.info("read-through source (real notify_put_batch), so later readers pull")
        console.info("from a peer, not the origin: each unique byte crosses the fabric")
        console.info("ONCE (1x vs mx). Wallclock depends on fanout_cap/topology -- a chain")
        console.info("(cap=1) is more hops, a tree (cap=2) narrows the gap; both stay 1x.")


def main(argv=None) -> None:
    """Entry point (also used by the demo smoke test)."""
    DedupDemo().main(argv)


if __name__ == "__main__":
    main()
