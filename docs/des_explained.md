# How the DES works — shared foundation and the two simulations

Both simulations run the **real** TorchStore code — the real `Controller`
directory, the real `LocalClient` planning core, and the real in-memory
transport/store — driven off-actor on a deterministic virtual clock via
[`realsim`](../realsim/). Everything but the new read coordinator is real
TorchStore code and real types (no invented types, no translation layer); see
[`realsim_design.md`](realsim_design.md) for the real-code foundation.

This doc walks through the shared core first, then each simulation's components,
and finally how they compare.

---

## 1. The shared DES core (`sim_common/`)

A **discrete-event simulation** means: there is no wall-clock, no threads, no OS
`asyncio` timing, and no randomness in the *timekeeping*. All "time" is a number
that advances only when the next scheduled event fires. Same input ⇒
byte-identical trace.

**`async_engine.py` — `AsyncEngine`** (what both sims run on)
- A deterministic `asyncio` event loop on a **virtual clock**: a FIFO ready queue
  plus a `heapq` of timers keyed by `(time, seq)` (`seq` a monotonic insertion
  counter that breaks ties, so the run is totally ordered and reproducible).
- `time()` returns simulated seconds; `await asyncio.sleep(dt)` advances the clock
  for free without blocking a real thread. Real `async` torchstore client and
  coordinator code executes under this loop unmodified.
- Every resource cost (network, storage, RAM, CPU, GPU) is applied as a
  virtual-clock `sleep`, so the whole run is free and deterministic.

**`engine.py` — `Sim` + `Promise`** (the original callback engine)
- The ancestor of the async engine: a binary heap of `(time, seq)` events with a
  synchronous `schedule(delay, cb)` and a one-shot `Promise` future. `AsyncEngine`
  is its async sibling and shares its `(time, seq)` tie-break convention. It is not
  on the sim path (the sims are all async) but is kept as the reference callback
  DES.

**`cost_model.py` — the analytic resource cost model**
- `MachineProfile` + functions for network / storage / RAM / CPU / GPU time. Every
  duration a sim charges is an analytic function of *target-machine* constants,
  **never measured** on the box running the sim.

**`topology.py` — the locality skeleton the cost model builds on**
- `Endpoint` (host/node), the locality `Tier` ordering `SHM` (same host) <
  `NVLINK` (same node) < `RDMA` (cross-node), `locality(src, dst)` to classify a
  pair, and `transfer_time(...) = latency + nbytes/bandwidth` for the relevant
  tier.

**`trace.py` — event recorder**: appends `(time, kind, msg)` rows and renders them
as aligned text. A deterministic sim ⟹ an identical trace across runs.

**`report.py`**: logging/section-header helpers and the ASCII source→dest tree
renderer (`render_tree`) the demo entrypoints share.

The real-object plumbing the sims drive — the real client/controller/transport
**seams + adapters**, the `ReadCoordinator` and its pluggable `ReadPolicy`, and
the meta / metadata-only payload carriers — lives in [`realsim`](../realsim/).
Both sims `import realsim` and add only their own policy + scenario code.

Because the sims execute real torchstore code, they depend on the **real
`torchstore` / `torch` / `monarch` install** (the from-source build), not
stdlib-only — the same dependency `realsim` has.

---

## 2. `dedup_sim` — weight-transfer deduplication

**The question it answers:** when *m* generator ranks all need the same tensor (or
overlapping shards) that lives on a trainer, can we move each unique byte across
the slow cross-node fabric **exactly once** (1×) instead of *m* times?

`dedup_sim` implements this as a real `realsim.coordinator.model.ReadPolicy`
(`dedup_sim.policy.DedupPolicy`) plugged into `realsim`'s `ReadCoordinator`. It
consults the **real** `Controller` directory and drives the real `LocalClient`
and in-memory transport, on real types throughout.

### Components it adds (everything else comes from `realsim` / `sim_common`)

| Component | File | Role |
|---|---|---|
| `DedupPolicy` | `policy.py` | A real `ReadPolicy`. Overrides `run_burst` to stage the read-through chain/tree over the real directory, and `after_fetch` for the read-through `put`. Reuses the coordinator's real primitives (`_locate`, `_fetch_one`, shared transport, metrics, trace). |
| `_RoutingControllerHandle` | `policy.py` | Expresses each routing choice: answers `locate_volumes` from the **real** directory, then narrows the result to the policy-chosen volume (returning the real `StorageInfo` unchanged). Every other endpoint — notably `notify_put_batch` — passes straight through, so the real directory stays the single source of truth. |
| dedup scenario + metrics | `scenario.py` | Builds the burst (reusing `realsim`'s wiring), runs it under dedup + naive, and computes fabric bytes (1× vs *m×*) + the source→dest tree. |
| demo entrypoint | `__main__.py` | `python -m dedup_sim` — fabric summary + ASCII diagram (INFO), full per-event trace (DEBUG, `-v`). |

### How it reaches 1× on the real directory

The naive policy (`realsim`'s `NaivePolicy`) fans every reader out concurrently;
in a synchronized burst they all `locate_volumes` the origin before anyone
finishes, so each pulls from the origin — *m×* fabric.

`DedupPolicy` stages the burst into a read-through **chain/tree**:

1. It consults the **real directory** (`locate_volumes` → real `StorageInfo` /
   `TensorSlice`) to find the origin(s) that hold the key.
2. It plans a `cap`-ary tree of sources (`fanout_cap=1` → a chain, `≥2` → a
   shallow tree): the **root** reader pulls from an origin (the single fabric
   hop); every other reader attaches, FIFO, to a **peer** under the cap. No
   non-root reader is ever routed to an origin.
3. After each reader fetches, the **real read-through** fires: the reader `put`s
   the key into its own co-located volume — a zero-fabric local write that, via
   the real `client.put` path, both stores the payload there and calls the real
   `notify_put_batch`. The reader is now a real directory source for the next
   level.
4. Each depth level executes concurrently (a level's sources were populated by
   the previous level), so a wider tree narrows wallclock.

Because exactly one reader ever pulls from an origin, the only origin-sourced
transfer is that first hop: `fabric_bytes == 1×` the payload, for **any** fan-out
cap. The naive baseline stays *m×*.

**The payoff metric is fabric bytes** — dedup moves each unique region 1× vs
naive's *m×*; wallclock depends on the fan-out topology (a `cap=1` chain has more
hops; a `cap=2` tree overlaps siblings and narrows the gap). The demo prints both.

---

## 3. `kvcache_sim` — LLM inference KV-cache reuse

**The question it answers:** in LLM serving, requests share prompt prefixes (a
system prompt, a conversation history). Can a **cache-aware scheduler** using the
global directory route requests to reuse cached KV blocks — cutting recompute,
TTFT, and load — vs a plain load-balancer? And on the decode side, how does
batching and prefill/decode disaggregation shape time-between-tokens (TBT)?

`kvcache_sim` runs the scheduling / caching / decode algorithm on the **real**
pieces via `realsim`: a KV block is a real `Controller` directory **key** (the
prefix-hash chain string), presence is a real directory entry, publishing is a
real metadata-only `put_batch`, a remote-prefix pull is a real `client.get_batch`
through `realsim`'s transport seam, and eviction is the real `notify_delete_batch`.

### Components it adds

| Component | File | Role |
|---|---|---|
| `Instance`, `Request`, prefix-hash | `sim/model.py` | An `Instance` owns one KV pool. A prompt is chunked into fixed `B`-token blocks, each **content-addressed by a prefix-hash chain** so shared leading blocks get identical keys (dedup/prefix-reuse falls out for free). `longest_prefix_run` counts leading matched blocks. |
| `Cluster` | `sim/cluster.py` | The four KV directory verbs (`prefix_lengths` / `publish` / `fetch` / `evict`) over a `realsim.mesh.Mesh`, which supplies the **real** `Controller` directory plus a real per-instance `LocalClient`. `prefix_lengths(block_keys)` — `{instance → leading blocks held}` — is the scheduler's core query, computed from the real `locate_volumes`. |
| `LRUCache` | `sim/cache.py` | Per-instance bounded cache with **LRU eviction**, the local bookkeeping kept in sync with directory presence. Recency is a monotonic counter (deterministic). |
| cost layer | `sim/cost.py` | Over `sim_common.cost_model`: `prefill_time` (per-uncached-token compute), `decode_step_time(batch)` (per-step **TBT**, strictly increasing in batch size), and `fetch_time` (a remote KV fetch, matching what the transport seam charges). |
| `DecodeEngine` | `sim/decode.py` | Batched, stepped decode on the async engine (makes **TBT real**). Emits one token per step per batched request; step time = `decode_step_time(batch)`. Models the VRAM `max_batch` cap (over-cap requests queue, their wait counting against TBT) and prefill/decode coupling (shared vs private compute timeline). |
| `LoadBalanceScheduler` / `CacheAwareScheduler` | `sim/scheduler.py` | The two policies (async). Load-balance: least-loaded instance, local-only cache. Cache-aware: route to minimize predicted TTFT using the **global** prefix directory; optionally pull a remote prefix under a balance threshold (which read-through-replicates the destination); LRU-evict on completion; SLO-reject. With `simulate_decode`, also picks a decode instance by *predicted* batch and applies the `early_rejection` admission mode. |
| `Client` | `sim/client.py` | Drives the request lifecycle: arrival → `schedule` → prefill-done publish (read-through), and (when decode is simulated) decode admission → decode-done, recording TBT. |
| `make_workload` | `sim/workload.py` | Seeded synthetic generator: shared system prompt + per-conversation context + unique query suffix, conversations chosen by a **Zipf** popularity law, **Poisson** arrivals. |
| `Metrics` | `sim/trace.py` | Hit rate, compute/saved tokens, mean/p90 TTFT, fabric bytes, rejections, and decode-side TBT (`mean_tbt`, `pct_tbt`, `tbt_slo_met`, `wasted_prefills`, `decode_rejections`). |
| scenarios + harness | `sim/scenarios.py` | The six scenarios + the async run harness. |

### The event flow

1. **`make_workload`** produces a deterministic, arrival-sorted list of `Request`s.
2. **`Client.submit`** schedules each request's arrival; on arrival
   **`scheduler.schedule(request, now)`** runs (serialized like the Monarch actor
   mailbox — consistent directory snapshot): query `prefix_lengths` for the best
   match, predict TTFT per instance (`queue + transfer + prefill(uncached)`), route
   to the minimum, optionally pulling a remote prefix; return `None` (reject) if
   predicted TTFT/TBT exceeds the SLO.
3. On prefill completion **`scheduler.on_complete`** does the **read-through**: the
   prefill instance now holds KV for the whole prompt, so `cache.admit` inserts the
   keys (evicting the coldest past capacity), the real `notify_put_batch` registers
   presence, and `notify_delete_batch` removes evicted blocks — which makes a hot
   prefix **replicate** across instances over time.
4. When decode is simulated, prefill completion also admits the request into its
   decode instance's batch on the **`DecodeEngine`**; each step emits one token per
   batched request at `decode_step_time(batch)`, and the request's worst inter-token
   gap is recorded as its **TBT**.

> One documented simplification: a block becomes reusable at prefill *completion*,
> not while in flight — unlike `dedup_sim`, there are no promises here, so two
> requests racing for the same brand-new prefix may both compute it. Adding
> promises is noted as future work.

**The payoff metrics are TTFT, hit rate, compute saved, rejections, and — on the
decode side — TBT (attainment, wasted prefills)** — never wall-clock. The six
scenarios (`shared_prefix`, `eviction`, `hotspot`, `overload`, `disaggregation`,
`early_rejection`) are documented in [`../kvcache_sim/README.md`](../kvcache_sim/README.md)
and [`torchstore_kvcache_design.md`](torchstore_kvcache_design.md).

---

## 4. How the two compare

| Aspect | `dedup_sim` | `kvcache_sim` |
|---|---|---|
| Engine | `sim_common.AsyncEngine` (virtual-clock asyncio) | Same |
| Directory | **Real** `Controller` (`locate_volumes` → `StorageInfo`/`TensorSlice`) | **Real** `Controller` (`keys_to_storage_volumes`, key = prefix-hash chain) |
| Client / transport | Real `LocalClient` + real in-memory transport (via `realsim`) | Same |
| Unit of data | 1-D tensor slices, allocation-free carriers | fixed-size **KV blocks** (prefix-hash keys), metadata-only carriers |
| Decision-maker | `DedupPolicy` (a real `ReadPolicy`) | `CacheAwareScheduler.schedule` |
| Concurrency primitive | Read-through chain/tree; the coordinator serializes plan-time so an in-flight peer becomes a source | Servers reserved via `busy_until`; SLO gate; no in-flight-as-source |
| Cache | Read-through peer replication | Read-through + bounded **LRU eviction** |
| Decode model | n/a (weight transfer only) | Batched, stepped decode (`DecodeEngine`): per-step **TBT** rises with batch size; VRAM cap; coupled vs disaggregated |
| Randomness | None (fixed scenarios) | One seeded Zipf + Poisson workload |
| Payoff metric | **fabric bytes** (1× vs *m×*) | **TTFT / hit rate / rejections**, and on the decode side **TBT attainment / wasted prefills** |

The common thread: both replace *pulling everything from the source* with
*reusing what a peer already holds*, both drive the **real** TorchStore controller
directory + client on real types, and both are fully deterministic DESs so every
trace and metric is byte-reproducible. The KV-cache
sim is essentially the dedup idea generalized to a long-lived, evicting,
workload-driven setting — which is why the design docs frame eviction and
hot-block replication as the capabilities the weight-sync path never needed.

For the real-code foundation both sims sit on — exactly which real objects execute
and how they are driven off-actor — see [`realsim_design.md`](realsim_design.md)
and [`../realsim/README.md`](../realsim/README.md).
