# kvcache_sim -- cache-aware KV-cache serving on the real TorchStore directory

A single-threaded, deterministic simulation of the cache-aware KV-cache serving
design in `../docs/torchstore_kvcache_design.md`: a cache-aware coordinator layered
over TorchStore's storage volumes + transport, doing prefix-hash addressing,
cache-aware routing, hot-block replication, LRU eviction, and batched decode.

It runs the scheduling/decode/cache algorithm on the **real** pieces via `realsim`:

- **Real directory.** KV-block presence is the real `torchstore.controller.Controller`
  directory (`keys_to_storage_volumes`), driven off-actor through `realsim`'s
  `RealControllerAdapter` / `FakeControllerHandle`. A KV block is a directory **key**
  (the prefix-hash chain string); "instance X holds block K" is the directory entry
  `K -> volume_X`. Routing consults the real `locate_volumes`.
- **Real clients + types.** Each serving instance is a real storage volume with a
  co-located real `LocalClient` (`realsim`'s `RealClientAdapter`). Publishing a
  prefix after prefill is a real, **metadata-only** `put_batch` (a `(shape, dtype)`
  `TensorDescriptor` per block -- zero real tensor storage) that records presence in
  the real directory; a remote-prefix pull is a real `client.get_batch` driven
  through `realsim`'s transport seam; eviction removes presence via the real
  `notify_delete_batch`. A KV block is a directory key -- real types throughout,
  with no translation layer.
- **Real cost model.** Every duration -- prefill compute, decode-step time, and the
  fabric/storage/RAM cost of a KV fetch -- is charged through
  `sim_common.cost_model` from a target-machine `MachineProfile`, never measured on
  the box running the sim.
- **Real async engine.** The whole request lifecycle runs on `realsim`'s
  deterministic virtual-clock `AsyncEngine`, so torchstore's real `async` client
  code executes under simulated time, single-threaded and reproducibly.

"Time" is a unitless simulated clock; the only randomness is one **seeded** RNG in
the synthetic workload, so the same seed produces a byte-identical trace and
identical metrics. It exercises the *algorithm* and gives a *sense of relative
performance* -- it is **not** a vendor benchmark.

## Environment

Run everything from the repo directory (the parent of `kvcache_sim/`) with the
repo's virtualenv, which has `realsim` + `torchstore` importable:

```
cd toso
PYTHONPATH=. .venv/bin/python -m kvcache_sim
```

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
  prints only the `(b)` summaries.

## The scenarios

- **shared_prefix** — many conversations share a hot system prompt + per-conversation
  context. Cache-aware routes same-prefix requests to the instance holding the prefix
  (or pulls it once), so shared prefixes are computed ~once; load-balance scatters
  them and recomputes. Higher hit rate, less prefill compute, lower TTFT.
- **eviction** — sweeps per-instance cache capacity and prints the hit-rate curve: it
  rises as the hot working set fits, then plateaus. Too-small caches can't even hold
  a full prefix ⇒ no reuse.
- **hotspot** — one dominant conversation (extreme Zipf skew). Compares load-balance
  vs cache-aware **without** replication vs **with** replication. Replication lowers
  prefill compute and p90 TTFT at the cost of KV fabric bytes.
- **overload** — high arrival rate with a TTFT SLO. Prefix reuse shortens prefill,
  freeing capacity, so cache-aware sheds fewer requests than load-balance.
- **disaggregation** — batched decode under a TBT target. A dedicated decode pool (its
  own compute timeline) protects served-request TBT from prefill interference;
  coupling prefill and decode on the same instances lets a prefill collide with a
  decode step, so a fraction of the *same* served load blows the target.
- **early_rejection** — heavy decode load under a tight TBT SLO, comparing admission
  policies `off`/`early`/`predict`. `off` late-checks decode load after prefill and so
  wastes prefill on rejects; `early`/`predict` gate before prefill (no waste), but only
  `predict` routes decode by the load foreseen at prefill completion, so it holds the
  SLO where `early`'s stale snapshot cannot.

## Testing

```
PYTHONPATH=. .venv/bin/python -m pytest kvcache_sim/tests -q
```

The tests are deterministic: they assert on block presence in the **real
directory** (publish → `locate_volumes` → evict), on the outcome (hit rate, compute,
eviction bounds, rejections, TBT), and on byte-identical traces across runs -- never
on wall-clock timing.

## The user-facing entry point mirrors the store

The only calls a "serving engine" makes are:

```python
plan = await scheduler.schedule(request, now)   # route; None => rejected
...                                             # engine pulls any remote prefix + prefills
await scheduler.on_complete(plan)               # publish computed KV blocks (read-through)
```

Routing consults the real directory (`locate_volumes`); remote-prefix pulls
(`client.get`), publishing (`client.put`) and eviction (`notify_delete`) are internal
to the coordinator, layered over the existing `put`/`get` plumbing.

## Layout

```
kvcache_sim/
  utils.py                # prefill / decode-step GPU times. Root, not in a role
                          #   folder, because BOTH planes use them: control
                          #   predicts with them, data charges them.
  policy/                 # CONTROL PLANE -- decides, moves nothing
    scheduler.py          #   LoadBalance (baseline) + CacheAware coordinator;
                          #   owns the analytical prefill-queue model
    cache.py              #   per-instance LRU eviction bookkeeping (metadata)
  workload/               # WHAT IS SIMULATED
    request.py            #   inference Request + prefix-hash chain (str keys)
    generator.py          #   seeded synthetic request stream (Zipf + Poisson)
    scenarios.py          #   scenario builders + the async run harness
  runtime/                # DATA PLANE -- moves bytes, burns compute
    cluster.py            #   the four KV directory verbs over a realsim.mesh.Mesh
    driver.py             #   the serving engine's request loop (turns a Plan into
                          #   clock advances + real ops; decides nothing)
    decode.py             #   async DecodeEngine: batched, stepped decode -> TBT
  report/                 # OUTCOME METRICS
    metrics.py            #   RequestResult, Metrics + summary rendering
  __main__.py             # `python -m kvcache_sim [scenario] [-v]`
  tests/test_sim.py       # deterministic tests (real-directory + outcome assertions)
```

`dedup_sim/` uses the same role folders, so the two capabilities can be compared
folder by folder — see [Comparison with `dedup_sim`](../dedup_sim/README.md#comparison-with-kvcache_sim).

The async engine, the cost model, the topology/`Endpoint` skeleton, the `Trace`
recorder and the report helpers live in the repo-root `sim_common/`; the real
client/controller/transport seams + adapters live in `realsim/`, and the
multi-instance wiring they add up to — per-instance volumes + real clients, one
directory, one resource registry, one shared transport factory — is
`realsim.mesh.Mesh`. This package holds only the KV-cache policy (scheduler,
cache, decode, workload, scenarios) plus the four directory verbs
(`prefix_lengths` / `publish` / `fetch` / `evict`) that express KV caching on a
mesh.

## Honesty notes

- This optimizes **prefix reuse / TTFT / TBT under a cost model**; absolute numbers
  are arbitrary units. Read the scenarios for *relative* wins (cache-aware vs
  load-balance) and *shapes* (hit rate vs capacity), not throughput claims.
- Blocks become reusable at prefill **completion**, not while in flight; with spaced
  arrivals the difference is small.
- A remote pull is routed on the directory snapshot at the request's arrival, but the
  fetch runs after the prefill queue; if a peer evicted a planned block meanwhile,
  the read-through fetches only what remains present (the rest is recomputed) -- the
  faithful real-directory behavior.
```
