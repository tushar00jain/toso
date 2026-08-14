"""``python -m kvcache_sim`` -- run the KV-cache scenarios and print results.

Run from the repo root so the package resolves::

    PYTHONPATH=. .venv/bin/python -m kvcache_sim [scenario] [-v]

For each scenario it emits:
  (a) a per-event trace  -> logged at DEBUG (shown only with -v), and
  (b) a summary comparison (cache-aware vs load-balance) -> logged at INFO.

Each scenario -- which runs it compares and how it narrates them -- is a
:class:`realsim.demo.Scenario` in :mod:`kvcache_sim.workload.scenarios`. This
file only declares the demo:

  * shared_prefix   -- multi-turn conversations sharing a hot system prompt; the
                       cache-aware coordinator reuses prefixes across instances.
  * eviction        -- sweep cache capacity; hit rate rises then plateaus (LRU).
  * hotspot         -- extreme skew; hot-block replication trades recompute for
                       KV transfer (it no longer also cuts p90 -- see its show()).
  * overload        -- high arrival + TTFT SLO; reuse => fewer rejections.
  * disaggregation  -- dedicated decode pool holds the TBT SLO; coupling prefill
                       into decode stalls it (Mooncake's headline).
  * early_rejection -- gating decode admission at routing avoids wasting prefill
                       (early/predict). The TBT half of that comparison does
                       not survive a self-pacing workload -- see its show().
"""

from __future__ import annotations

import argparse

from realsim.demo import Console, Demo

from .workload.scenarios import (
    Disaggregation,
    EarlyRejection,
    Eviction,
    Hotspot,
    Overload,
    SharedPrefix,
)


class KVCacheDemo(Demo):
    """The KV-cache demo: six comparisons over one real directory."""

    name = "kvcache_sim"
    description = (
        "Discrete-event simulation of a KV cache on TorchStore. Runs the given "
        "scenario (or all) and prints, per scenario, a cache-aware vs "
        "load-balance summary (INFO) and, with -v, the per-event trace (DEBUG)."
    )

    def scenarios(self):
        return [
            SharedPrefix(),
            Eviction(),
            Hotspot(),
            Overload(),
            Disaggregation(),
            EarlyRejection(),
        ]

    def flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--spread-reads", action="store_true",
            help="put the hotspot scenario's cache-aware source ranking under a "
            "Discount, so a replica recently routed to is ranked down and one "
            "replica of a hot prefix does not serve every read of it. Off by "
            "default: it changes which replica answers, so it is not "
            "byte-identical.",
        )

    def takeaway(self, console: Console) -> None:
        console.section("TAKEAWAY")
        console.info("A cache-aware coordinator over TorchStore's existing volumes+transport")
        console.info("turns shared prefixes into cluster-wide reuse: higher hit rate, less")
        console.info("prefill compute, and lower TTFT than load-balancing that only reuses a")
        console.info("local cache. The workload is multi-turn -- turn N+1 is turn N's prompt")
        console.info("plus turn N's OUTPUT plus a new message -- so the reusable prefix grows")
        console.info("with the conversation and a miss costs more the deeper the dialogue is,")
        console.info("which is why the gap is wide. It also means the KV a decode host")
        console.info("GENERATES is looked up and hit by the next turn rather than written and")
        console.info("forgotten. LRU eviction bounds the cache (hit rate vs capacity is the")
        console.info("sizing knob); hot-block replication trades recompute for KV transfer;")
        console.info("SLO admission sheds overload. All of it is control-plane selector over the")
        console.info("same data plane.")
        console.info("On the decode side the same coordinator bounds time-between-tokens:")
        console.info("disaggregating decode onto its own pool keeps a prefill from colliding")
        console.info("with a decode step (TBT target ~100% vs a large fraction of the SAME")
        console.info("served load missing when coupled), and gating decode admission at routing")
        console.info("avoids spending prefill on requests that cannot be decoded in SLO.")
        console.info("Two claims this demo used to make no longer hold on a workload that paces")
        console.info("itself, and each says so where it is shown: replication no longer cuts")
        console.info("p90 TTFT (hotspot), and predicted-load decode routing no longer separates")
        console.info("from stale-occupancy routing (early_rejection). Both needed a burst that")
        console.info("a closed loop cannot offer, and neither has been retuned to hide it.")


def main(argv=None) -> None:
    """Entry point (also used by the demo smoke test)."""
    KVCacheDemo().main(argv)


if __name__ == "__main__":
    main()
