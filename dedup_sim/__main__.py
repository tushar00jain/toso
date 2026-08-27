"""``python -m dedup_sim`` -- run the dedup demo on the real TorchStore directory.

Run from the repo root so the package resolves::

    PYTHONPATH=. .venv/bin/python -m dedup_sim [scenario] [-v]

Both scenarios drive a synchronized read burst over the **real** ``Controller``
directory + **real** client/transport (via ``realsim``):

  * ``dedup`` -- one key, one holder. The unrouted baseline has every reader pull
    from the origin (``m x`` fabric); the dedup control plane
    (:class:`dedup_sim.control.routing.Dedup`) routes each reader to a peer and
    withholds the answer until that peer's read-through put registers, so the burst
    becomes a chain (``fanout_cap=1``) or tree (``fanout_cap=2``) and each unique byte
    crosses the fabric once -- 1x fabric.
  * ``weight_sync`` -- one key, two trainer replicas holding it. The chain leaves one
    replica idle and queues the second generator behind the first; ``spread`` sends one
    generator to each, for 1x per replica and one hop of depth.
  * ``routing`` -- Qwen3.6-27B with TP=4 and DP=2, comparing direct trainer
    reads with fixed local routes and readiness-signaled generator read-through.

The comparisons themselves -- which runs, and how they are narrated -- are
:mod:`dedup_sim.workload.scenarios`. This file only declares the demo.

Output is routed through the ``logging`` module:
  * INFO (default): section headers, the dedup-vs-naive summary + ASCII
    source->dest diagram, the takeaway.
  * DEBUG (``-v``/``--debug``): additionally the full per-event virtual-time trace.
"""

from __future__ import annotations

from realsim.demo import Console, Demo

from .workload.scenarios import Dedup, RoutingScenario, WeightSync


class DedupDemo(Demo):
    """Dedup routing plus the fixed-route weight-sync comparison."""

    name = "dedup_sim"
    description = (
        "Dedup read-routing demo on the real TorchStore directory. Runs a "
        "synchronized read burst under the naive baseline and the dedup selector "
        "(chain + tree), and -- over a key two trainer replicas hold -- against a "
        "load-spread variant of the same chain, printing the fabric summary + ASCII "
        "diagram (INFO) and, with -v, the full per-event virtual-time trace (DEBUG). "
        "Also includes the precomputed application-managed routing path."
    )

    def scenarios(self):
        return [Dedup(), WeightSync(), RoutingScenario()]

    def takeaway(self, console: Console) -> None:
        console.section("TAKEAWAY")
        console.info("On the real directory, dedup registers each finished reader as a")
        console.info("read-through source (real notify_put_batch), so later readers pull")
        console.info("from a peer, not the origin: each unique byte crosses the fabric")
        console.info("ONCE (1x vs mx). Wallclock depends on fanout_cap/topology -- a chain")
        console.info("(cap=1) is more hops, a tree (cap=2) narrows the gap; both stay 1x.")
        console.info("With two replicas holding the key, load spreading is the other half of")
        console.info("that trade: one hop per replica instead of one hop overall, and the")
        console.info("depth of the chain divided by however many replicas there are.")


def main(argv=None) -> None:
    """Entry point (also used by the demo smoke test)."""
    DedupDemo().main(argv)


if __name__ == "__main__":
    main()
