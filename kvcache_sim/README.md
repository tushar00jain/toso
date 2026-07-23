# kvcache_sim -- discrete-event simulation of a KV cache

A single-threaded, deterministic discrete-event simulation (DES) of the cache-aware
KV-cache serving design in `../docs/torchstore_kvcache_design.md`: the
`CacheCoordinator` (a cache-aware coordinator) layered over TorchStore's
existing storage volumes + transport, doing prefix-hash addressing, cache-aware
routing, hot-block replication and LRU eviction.

It exercises the *algorithm* and gives a *sense of relative performance* locally —
it is **not** a vendor benchmark. "Time" is a unitless simulated clock; prefill,
transfer and decode durations come from pure cost functions, never measurement.
Pure Python stdlib only; the only randomness is one **seeded** RNG in the synthetic
workload, so the same seed ⇒ byte-identical trace + identical metrics.

This is the sibling of `../dedup_sim/` (the weight-sync dedup coordinator DES): same
engine, same faithful-shim-controller approach, same determinism discipline. The
shared pieces -- the DES engine (`Sim`, `Promise`), the locality/transfer-time cost
skeleton, the generic `Trace` event recorder, the logging/section-header helpers, and
the silenced real-`Controller` import probe (`HAVE_REAL`) -- live in the repo-root
`sim_common/` package (`engine.py`, `topology.py`, `trace.py`, `report.py`,
`controller_probe.py`) that both sims import; each sim keeps its own bandwidth
constants, domain model, index shim, and outcome `Metrics` / summary rendering.

## Environment (uv)

The project uses [uv](https://docs.astral.sh/uv/) with a `.venv` at the repo root
(`toso/.venv`). Run everything from the repo directory (the parent of
`kvcache_sim/`). Either activate the venv:

```
cd toso
source .venv/bin/activate
python -m kvcache_sim
```

or use `uv run` without activating:

```
cd toso
uv run --no-sync python -m kvcache_sim
```

`kvcache_sim` is pure stdlib, so `--no-sync` reuses the existing `.venv` as-is
(avoids re-resolving the repo's heavier optional deps).

## How to run

```
python -m kvcache_sim                  # all scenarios: INFO summaries only
python -m kvcache_sim -v                # add the per-event trace (DEBUG)
python -m kvcache_sim shared_prefix     # run a single scenario
python -m kvcache_sim hotspot -v        # one scenario, with the trace
python -m kvcache_sim --help            # usage + valid scenario names
```

- Positional `scenario` is one of `shared_prefix`, `eviction`, `hotspot`,
  `overload`, `disaggregation`, `early_rejection`; omit it to run all (plus a
  closing takeaway).
- `-v` / `--verbose` / `--debug` raises the log level to DEBUG so the `(a)` event
  trace prints (capped to the first 60 events per scenario); the default INFO level
  prints only the `(b)` summaries. Output is routed through stdlib `logging` with a
  bare `%(message)s` format.

## The scenarios

- **shared_prefix** — many conversations share a hot system prompt + per-conversation
  context. Cache-aware routes same-prefix requests to the instance holding the prefix
  (or pulls it once), so shared prefixes are computed ~once; load-balance scatters
  them and recomputes. Shows higher hit rate, less prefill compute, lower TTFT.
- **eviction** — sweeps per-instance cache capacity and prints the hit-rate curve:
  it rises as the hot working set fits, then plateaus (the ~30%→~50% shape).
  Too-small caches can't even hold a full prefix ⇒ no reuse.
- **hotspot** — one dominant conversation (extreme Zipf skew). Compares load-balance
  vs cache-aware **without** replication (recompute a missing prefix) vs **with**
  replication (pull it once, cheaply). Replication lowers prefill compute and p90
  TTFT at the cost of KV fabric bytes.
- **overload** — high arrival rate with a TTFT SLO. Prefix reuse shortens prefill,
  freeing capacity, so cache-aware sheds fewer requests than load-balance.
- **disaggregation** — batched decode under a TBT target. A dedicated decode pool
  (its own compute timeline) protects served-request TBT from prefill interference;
  coupling prefill and decode on the same instances lets a prefill collide with a
  decode step, so a large fraction of the *same* served load blows the target
  (Mooncake's headline disaggregation result).
- **early_rejection** — heavy decode load under a tight TBT SLO, comparing admission
  policies `off`/`early`/`predict`. `off` late-checks decode load after prefill and
  so wastes prefill on rejects; `early`/`predict` gate before prefill (no waste), but
  only `predict` routes decode by the load foreseen at prefill completion, so it holds
  the SLO where `early`'s stale snapshot cannot.

## Testing

```
uv run --with pytest pytest kvcache_sim/tests -q   # no install (needs a synced project)
# or, if project sync fails / to reuse the existing .venv:
uv pip install pytest
python -m pytest kvcache_sim/tests -q
```

The tests are deterministic — they assert on the DES outcome (hit rate, compute,
eviction bounds, rejections) and on byte-identical traces across runs, never on
wall-clock timing.

## The user-facing entry point mirrors the store

The only call a "serving engine" makes is:

```python
plan = scheduler.schedule(request, now)   # route (cache-aware); None => rejected
...                                        # engine prefills the uncached suffix
scheduler.on_complete(plan)               # publish computed KV blocks (read-through)
```

No promise/pull/replication arguments leak to the caller — routing, remote-prefix
pulls, replication and eviction are entirely internal to the coordinator, exactly as
the design layers them over the existing `put`/`get` plumbing.

## Store-index path

As in `dedup_sim`, we attempt to import the real
`torchstore.controller.Controller` (via the shared `sim_common.controller_probe`,
which exposes `HAVE_REAL`). Even when it imports, its endpoints are
`@endpoint async` Monarch-actor methods needing an actor runtime, so a plain
single-threaded sim uses a **faithful shim** (`BlockIndex`) mirroring the storage
index (`block_key → set[instance_id]`) with matching method names
(`locate`/`notify_put`/`notify_delete`/`keys`) plus `instances_with_prefix` for the
prefix-match query. `__main__` prints which path was taken.

## Honesty notes

- This optimizes **prefix reuse / TTFT under a cost model**; the absolute numbers are
  arbitrary units. Read the scenarios for *relative* wins (cache-aware vs
  load-balance) and *shapes* (hit rate vs capacity), not throughput claims.
- Blocks become reusable at prefill **completion**, not while in flight (the dedup
  sim's promise-based in-flight-as-source is not modelled here); with spaced arrivals
  the difference is small. See SPEC §6.
