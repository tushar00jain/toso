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
  * hotspot         -- extreme skew; hot-block replication spreads load, cuts p90.
  * overload        -- high arrival + TTFT SLO; reuse => fewer rejections.
  * disaggregation  -- dedicated decode pool holds the TBT SLO; coupling prefill
                       into decode stalls it (Mooncake's headline).
  * early_rejection -- predicting decode load avoids wasting prefill (off/early/
                       predict) while still meeting the TBT SLO.
"""

from __future__ import annotations

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

    def takeaway(self, console: Console) -> None:
        console.section("TAKEAWAY")
        console.info("A cache-aware coordinator over TorchStore's existing volumes+transport")
        console.info("turns shared prefixes into cluster-wide reuse: higher hit rate, less")
        console.info("prefill compute, and lower TTFT than load-balancing that only reuses a")
        console.info("local cache. LRU eviction bounds the cache (hit rate vs capacity is the")
        console.info("sizing knob); hot-block replication spreads skew; SLO admission sheds")
        console.info("overload. All of it is control-plane policy over the same data plane.")
        console.info("On the decode side the same coordinator bounds time-between-tokens:")
        console.info("disaggregating decode onto its own pool keeps a prefill from colliding")
        console.info("with a decode step (TBT target ~100%% vs a large fraction of the SAME")
        console.info("served load missing when coupled), and predicting decode load at")
        console.info("admission avoids spending prefill on requests that can't be decoded in")
        console.info("SLO while routing decode by foreseen load -- no wasted prefill, SLO held.")


def main(argv=None) -> None:
    """Entry point (also used by the demo smoke test)."""
    KVCacheDemo().main(argv)


if __name__ == "__main__":
    main()
