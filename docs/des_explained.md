# How the DES works — shared foundation and the two simulations

Both simulations sit on top of one tiny, shared engine (`sim_common/`) and then
build **different domain models on top of it**. This doc walks through the shared
core first, then each simulation's components, and finally how they compare.

---

## 1. The shared DES core (`sim_common/`)

This is the heart that both sims reuse. A **discrete-event simulation** means:
there is no wall-clock, no threads, no `asyncio`, and no randomness in the
*timekeeping*. All "time" is a number that advances only when the next scheduled
event fires.

**`engine.py` — `Sim` + `Promise`** (`sim_common/engine.py`)
- `Sim` holds a binary heap of events ordered by `(time, seq)`. `time` is the
  simulated firing time; `seq` is a monotonic insertion counter that breaks ties
  so the run is *totally ordered and byte-for-byte reproducible*.
- `Sim.schedule(delay, cb, label)` pushes a callback to fire `delay` units after
  `now`.
- `Sim.run()` pops events in `(time, seq)` order, advances `now` to each event's
  time, and calls its callback. A callback can schedule more events — that's how
  the simulation propagates. It stops when the heap is empty.
- `Promise` is a one-shot future. `resolve()` schedules its registered callbacks
  at `delay=0` (so they run *after* the current event, never re-entrantly, and in
  FIFO registration order). This models a store-mediated "done" signal — a
  producer finishing lets parked consumers proceed.

**`topology.py` — the cost model skeleton** (`sim_common/topology.py`)
- Defines three **locality tiers**: `SHM` (same host) < `NVLINK` (same node) <
  `RDMA` (cross-node).
- `locality(src, dst)` classifies a pair of endpoints by comparing their
  `.host`/`.node`.
- `transfer_time(src, dst, nbytes, tiers)` = `latency + nbytes/bandwidth` for the
  relevant tier (free if same endpoint or zero bytes). The *constants* are
  supplied by each sim, not baked in here.

**`trace.py` — event recorder**: appends `(time, kind, msg)` rows and renders them
as aligned text lines. Deterministic sim ⟹ identical trace across runs.

**`controller_probe.py`**: both sims *attempt* to import the real TorchStore
`Controller` (silenced, since the import is noisy). It only records `HAVE_REAL` —
a flag. It's never driven, because the real controller's endpoints are
`@endpoint async` Monarch-actor methods a single-threaded sim can't run. So both
sims use a faithful in-memory **shim** with matching method names (`locate`,
`notify_put`, `keys`), making a later swap mechanical.

`report.py` just sets up logging/section headers for the demo entrypoints.

---

## 2. `dedup_sim` — weight-transfer deduplication

**The question it answers:** when *m* generator ranks all need the same tensor (or
overlapping shards) that lives on a trainer, can we move each unique byte across
the slow cross-node fabric **exactly once** (1×) instead of *m* times?

### Components it creates

| Component | File | Role |
|---|---|---|
| `Volume` | `sim/model.py` | A storage volume (per rank). Has `id/host/node/is_trainer`. Trainer volumes hold the source data; generator volumes are readers that double as read-through caches. |
| Region math | `sim/model.py` | Tensors are 1-D. `split_regions` breaks overlapping ranges into minimal **atomic** non-overlapping segments; `decompose` reconstructs a reader's need from atomics; `union_bytes` = the 1× fabric target. |
| `StoreIndex` | `sim/store_index.py` | Shim of the controller directory: `key -> {volume_id -> set[Region]}`. Metadata only — no bytes. `notify_put` records presence; `locate` looks it up. |
| `cost.py` | `sim/cost.py` | The dedup-specific tier constants (RDMA clearly slower than NVLink) + `transfer_time` wrapper. |
| `NaiveCoordinator` / `DedupCoordinator` | `sim/coordinator.py` | The two policies under test. |
| `Client` | `sim/client.py` | The only user-facing entry point (`get`/`put`), mirroring `ts.get`/`ts.put`. Executes the plan the coordinator returns. |
| `Metrics` + tree renderer | `sim/trace.py` | Fabric bytes, wallclock, and the ASCII source→dest diagram. |
| Scenarios + harness | `sim/scenarios.py` | Builds volumes/needs, seeds the index, bursts every `get` at t=0, runs the sim. |

### How they interact (the event flow)

1. **Harness** (`scenarios.run`) creates a `Sim`, seeds the `StoreIndex` with
   trainer puts, then schedules every generator's `client.get(...)` at `t=0` —
   this is "the burst."
2. **`Client.get`** decomposes the need into atomic regions and calls
   **`coordinator.plan_get`** *synchronously*. This is the crucial DES trick:
   because `plan_get` runs to completion before the next event fires, a burst of
   gets becomes a deterministic **total order**, and the coordinator can treat
   in-flight fetches as *promised* cache sources.
3. **`DedupCoordinator.plan_get`** decides, per atomic region:
   - If a peer already **has** it (present) or **will have** it (a `Promise` from
     an earlier planned fetch) → route to that peer, never the trainer. Source
     choice prefers locality tier, filling each source up to `FANOUT_CAP` before
     spreading (shapes a chain at cap=1, a tree at cap=2).
   - Otherwise → pull from the trainer (`_trainer_holding`). Either way, this
     reader registers its own `publish_promise` so it becomes a source for
     *later* peers (read-through fan-out).
4. **`Client`** executes each `Fetch`:
   - If it has a `data_dep` promise (source doesn't hold the data yet) → **park**
     the reader via `promise.add_callback`.
   - Otherwise → `request_slot` (the execution-time fan-out cap: excess consumers
     **queue** on the source rather than re-pulling the trainer), then `_start`
     schedules the transfer with `sim.schedule(dt, done)` where
     `dt = transfer_time(...)`.
5. When a transfer's `done` fires: if the source was the trainer,
   `fabric_bytes += nbytes` (the metric that matters); the destination is
   registered as a new source (`on_fetch_complete` → `notify_put` +
   `publish_promise.resolve()`), which **releases parked peers**; the serving slot
   frees and wakes the next queued waiter.

The two counters keep fan-out honest: `planned[vol]` (plan-time tally for tree
shaping) and `serving[vol]` (actual concurrent serves, hard-bounded by the cap).

### Scenarios
- **toy** — every generator needs all of `W`; shows the fan-out chain (cap=1) vs
  tree (cap=2) vs naive *m×* baseline.
- **reshard** — trainer stores halves, generators want a different partition;
  exercises atomic splitting; asserts fabric == union (1×) live.
- **versioning** — two bursts; a `put` between them calls `bump_version`,
  invalidating the cache so burst 2 re-pulls from the trainer. Without the bump,
  burst 2 is served entirely by peers (0 trainer fabric).

**The payoff metric is bytes** — dedup moves each unique region 1× vs naive's
*m×*; wallclock depends on the fan-out topology.

---

## 3. `kvcache_sim` — LLM inference KV-cache reuse

**The question it answers:** in LLM serving, requests share prompt prefixes (a
system prompt, a conversation history). Can a **cache-aware scheduler** using a
global prefix directory route requests to reuse cached KV blocks — cutting
recompute, TTFT, and load — vs a plain load-balancer?

### Components it creates

| Component | File | Role |
|---|---|---|
| `Instance`, `Request`, prefix-hash | `sim/model.py` | An `Instance` owns one KV pool. A prompt is chunked into fixed `B`-token blocks, each **content-addressed by a prefix-hash chain** (`m0\|3\|7` etc.) so shared leading blocks get identical keys — dedup/prefix-reuse falls out for free. `longest_prefix_run` counts leading matched blocks. |
| `LRUCache` | `sim/cache.py` | Per-instance bounded cache with **LRU eviction** — the capability weight-sync never needed (inference caches are unbounded/long-lived). Recency is a monotonic counter (deterministic, no wall-clock). |
| `BlockIndex` | `sim/index.py` | Controller shim for KV mode: `block_key -> set[instance_id]`. Key extra query: `instances_with_prefix` → `{instance -> matched prefix length}`, the scheduler's core routing input. `notify_delete` removes evicted blocks. |
| `cost.py` | `sim/cost.py` | Tier transfer times **plus** `prefill_time` (per-uncached-token compute), `decode_time`, and `decode_step_time(batch)` — the per-step **TBT** that rises with decode batch size. Tuned so transferring a cached block is cheaper than recomputing it. |
| `DecodeEngine` | `sim/decode.py` | Batched, stepped decode (the piece that makes **TBT real**). Emits one token per step for every request in an instance's batch; step time = `decode_step_time(batch)`, so TBT degrades as the batch fills. Models the **VRAM `max_batch` cap** (over-cap requests queue, their wait counting against TBT) and **prefill/decode coupling** (a shared vs. private compute timeline). `on_finish(request, tbt_max)` reports each request's worst inter-token gap. |
| `LoadBalanceScheduler` / `CacheAwareScheduler` | `sim/scheduler.py` | The two policies. With `simulate_decode`, they also pick a decode instance by *predicted* batch, predict TBT, and apply the `early_rejection` admission mode (`off`/`early`/`predict`). |
| `Client` | `sim/client.py` | Drives the request lifecycle: arrival → schedule → prefill-done publish, and (when decode is simulated) decode admission → decode-done, recording TBT. |
| `make_workload` | `sim/workload.py` | Seeded synthetic generator: shared system prompt + per-conversation context + unique query suffix, conversations chosen by a **Zipf** popularity law, **Poisson** arrivals. |
| `Metrics` | `sim/trace.py` | Outcome metrics: hit rate, compute/saved tokens, mean/p90 TTFT, fabric bytes, rejections, **plus decode-side TBT** — `mean_tbt`, `pct_tbt`, `tbt_slo_met(slo)`, `wasted_prefills`, `decode_rejections`. |
| Scenarios + harness | `sim/scenarios.py` | Builds instances + workload, runs a scheduler. |

### How they interact (the event flow)

1. **`make_workload`** produces a deterministic, arrival-sorted list of
   `Request`s (seeded RNG drives both Zipf conversation choice and Poisson
   inter-arrivals).
2. **`Client.submit`** schedules each request's `_arrive` at its `arrival` time.
3. On `_arrive`, **`scheduler.schedule(request, now)`** runs synchronously
   (serialized-mailbox model → consistent directory snapshot):
   - **`LoadBalanceScheduler`**: pick least-loaded instance; reuse only *that
     instance's local* cache; never pull a remote prefix.
   - **`CacheAwareScheduler`**: query `instances_with_prefix` for the global best
     match. For each candidate instance, decide whether to **pull a remote
     prefix** (only if `best_len > local_len * balance_threshold`) or recompute
     locally, then predict TTFT (`queue_wait + transfer + prefill`) and pick the
     minimum-TTFT plan. Because prefill cost is deterministic, the *predicted*
     TTFT equals the *actual* completion time.
   - Either scheduler returns `None` if predicted TTFT exceeds the SLO (or, in the
     `early`/`predict` decode-admission modes, predicted TBT) → **rejection**.
4. **`_predict`** computes timing without reserving servers; **`_commit`** then
   reserves the prefill server (`busy_until`) and `touch`es the matched prefix for
   LRU recency. (Decode is admitted later, at prefill completion — see step 7.)
5. **`Client`** schedules `_prefill_done` at `now + ttft`.
6. On `_prefill_done`, **`scheduler.on_complete`** does the **read-through (K4)**:
   the prefill instance now holds KV for the whole prompt, so `cache.admit(keys)`
   inserts them (evicting the coldest past capacity), `index.notify_put`
   registers presence, and `index.notify_delete` removes evicted blocks. This is
   what makes a hot prefix **replicate** across instances over time.
7. When decode is simulated, `_prefill_done` also calls **`scheduler.admit_decode`**,
   entering the request into its decode instance's batch on the **`DecodeEngine`**.
   In `off` mode this is the *only* TBT gate — a rejection here means the prefill was
   already spent (a **wasted prefill**). The engine then runs stepped decode: each
   step emits one token per batched request at `decode_step_time(batch)`, and on the
   request's last token `on_finish` reports its worst inter-token gap, which the
   `Client` records as that request's **TBT**.

> One documented simplification: a block becomes reusable at prefill *completion*,
> not while in flight — unlike `dedup_sim`, there are no promises here, so two
> requests racing for the same brand-new prefix may both compute it. Adding
> promises is noted as future work.

### Scenarios (`sim/scenarios.py`)
- **shared_prefix** — cache-aware vs baseline on the Zipf/shared-prefix workload;
  cache-aware wins on hit rate, compute saved, TTFT.
- **eviction sweep** — sweeps cache capacity; hit rate rises then plateaus once
  the hot working set fits.
- **hotspot** — extreme skew (one dominant conversation); compares baseline vs
  cache-aware-no-replication (piles load on the one hot instance) vs
  cache-aware-with-replication (spreads the hot prefix to peers, cutting p90 TTFT
  for a few KV transfers).
- **overload** — high arrival rate + TTFT SLO; cache-aware reuse shortens prefill
  → frees capacity → rejects fewer requests.
- **disaggregation** — batched decode under a TBT target (admission disabled, so both
  configs serve the *identical* load). A dedicated decode pool (its own compute
  timeline) holds the target for ~every served request; coupling prefill and decode on
  the same instances lets a prefill collide with a decode step, so a large fraction of
  the same served load blows the target. Mooncake's headline disaggregation result.
- **early_rejection** — heavy decode load + a tight TBT SLO, comparing the three
  admission modes. `off` late-checks decode load *after* prefill and rejects on a
  violation → **wasted prefill**; `early`/`predict` gate *before* prefill (no waste),
  but only `predict` routes decode by the load foreseen at prefill completion, so it
  holds the SLO where `early`'s stale current-occupancy snapshot cannot.

**The payoff metrics are TTFT, hit rate, compute saved, rejections, and — on the
decode side — TBT (attainment, wasted prefills)** — never wall-clock.

---

## 4. How the two compare

| Aspect | `dedup_sim` | `kvcache_sim` |
|---|---|---|
| Engine | Same `sim_common.Sim` heap + `(time, seq)` ordering | Same |
| Directory | `StoreIndex`: `key → {volume → regions}` | `BlockIndex`: `block_key → {instances}` |
| Unit of data | 1-D tensor **regions** (atomic splits) | fixed-size **KV blocks** (prefix-hash keys) |
| Decision-maker | `DedupCoordinator.plan_get` | `CacheAwareScheduler.schedule` |
| Concurrency primitive | **`Promise`** parks readers on in-flight fetches; slot queue enforces fan-out cap | No promises; servers reserved via `busy_until`; SLO gate |
| Cache | Read-through, invalidated by `bump_version` | Read-through, bounded by **LRU eviction** |
| Cross-instance transfer | Peer exchange to avoid re-pulling trainer | Remote prefix pull under a balance threshold |
| Decode model | n/a (weight transfer only) | Batched, stepped decode (`DecodeEngine`): per-step **TBT** rises with batch size; VRAM `max_batch` cap; coupled vs. disaggregated compute |
| Randomness | None (fixed scenarios) | Seeded Zipf + Poisson workload |
| Payoff metric | **fabric bytes** (1× vs *m×*) | **TTFT / hit rate / rejections**, and on the decode side **TBT attainment / wasted prefills** |

The common thread: both replace *pulling everything from the source* with
*reusing what a peer already holds*, both model the TorchStore controller as a
metadata-only directory shim, and both are fully deterministic DESs so every
trace and metric is byte-reproducible. The KV-cache sim is essentially the dedup
idea generalized to a long-lived, evicting, workload-driven setting — which is why
the design docs frame eviction and hot-block replication as the capabilities the
weight-sync path never needed.
