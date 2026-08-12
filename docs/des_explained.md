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

**`report.py`**: logging/section-header helpers, the ASCII source→dest tree
renderer (`render_tree`), and `Ledger` — the measurement half every capability
needs: transfer edges, byte counters (`transfer_bytes` delivered vs `origin_bytes`
that had to cross from a pre-existing source), one outcome row per work item, and
the sum/mean/percentile/fraction aggregations every report computes over them.

The real-object plumbing the sims drive — the real client/controller/transport
**seams + adapters**, the `Mesh` that wires per-node volumes + real clients onto
one directory, and the meta / metadata-only payload carriers — lives in
[`realsim`](../realsim/), together with the four types a capability plugs into:

| Type | What it is |
|---|---|
| `Policy.select(view, keys, requester)` | Which volume serves these keys for this requester, and **when** it is usable (ranked sources + an optional readiness gate). Naive — every holder, directory order — is the default, and is exactly what the real directory answers unaided. Consulted *inside* the real `locate_volumes`, so a scenario that just calls `client.get(K)` is routed without knowing a policy exists; an app that wants to *price* alternatives calls it itself. |
| `View` | The read-only observation a policy is handed: `locate`, topology/locality, the virtual clock. Awaited reads, no mutation. |
| `DataPlane` | One member — `after(requester, result)`: what a capability does once a transfer has landed, defaulting to real no-op behaviour. Moving the bytes is an ordinary client call, so no interface declares it. Named by requester, not work item, so a deployment can implement it — how a *run* is driven is `realsim.runner.ItemDispatch`. |
| `Runner` | Releases work items on the virtual clock in `(release_time, id)` order, installs the mesh once, gathers. The gather is the whole wait — there is no drain phase behind it. |

Both sims `import realsim` and add only their own decision + execution code. So
does [`putget_sim`](../putget_sim/), which is not a capability at all: it is the
same put/get workload with **no** `Policy` and **no** `DataPlane` installed, and
therefore the unrouted *m×* baseline dedup measures against. Dedup imports its
`PutGetBurst` unchanged, which is what makes the two runs comparable.

[`domain/llm.py`](../domain/llm.py) is also shared: a `Model` reduces a
transformer to what a sim charges against (flops per prefilled token, flops per
decode step, KV bytes per token via `block_bytes()`), plus `prefill_time` /
`decode_step_time`. `kvcache_sim` prices prefill/decode compute and KV block sizes
from it; `dedup_sim` will size the weights it syncs from it (a TODO in that file
records the plan). It is domain fact, not simulator machinery and not policy,
which is why it sits beside them rather than inside either.

Because the sims execute real torchstore code, they depend on the **real
`torchstore` / `torch` / `monarch` install** (the from-source build), not
stdlib-only — the same dependency `realsim` has.

---

## 2. `dedup_sim` — weight-transfer deduplication

**The question it answers:** when *m* generator ranks all need the same tensor (or
overlapping shards) that lives on a trainer, can we move each unique byte across
the slow cross-node fabric **exactly once** (1×) instead of *m* times?

`dedup_sim` implements this as a real `realsim.policy.Policy`
(`dedup_sim.control.routing.DedupPolicy`), consulted inside the **real**
`Controller`'s `locate_volumes`, over the real `LocalClient` and in-memory
transport, on real types throughout.

### Components it adds (everything else comes from `realsim` / `sim_common`)

| Component | File | Role |
|---|---|---|
| `DedupPolicy` | `control/routing.py` | A real `Policy`. Assigns each requester, as it asks, a source under a fan-out cap, and returns it with a **readiness gate** when that source has not registered yet. Holds no client, no volume, no mesh — and no burst loop. |
| `ReadThroughPlane` | `data/read_through.py` | One `DataPlane.after`: the finished reader `put`s the key into its own co-located volume, which through the real `client.put` path also calls the real `notify_put_batch`. That registration is what opens the next reader's gate. |
| dedup scenario | `workload/scenarios.py` | Runs `putget_sim`'s ordinary put/get fixture twice — unrouted (the *m×* baseline) and with the policy + plane installed (1×) — so the comparison is byte-for-byte the same topology, payload and cost model. |
| demo entrypoint | `__main__.py` | `python -m dedup_sim` — fabric summary + ASCII diagram (INFO), full per-event trace (DEBUG, `-v`). |

### How it reaches 1× on the real directory

With no policy installed, every reader `locate_volumes` the origin before anyone
finishes, so each pulls from the origin — *m×* fabric.

`DedupPolicy` answers that same `locate_volumes` differently:

1. Readers reach the controller in order. The **first** is routed to a volume that
   already holds the key (the single fabric hop), chosen by locality.
2. Every later one is routed to a **peer**: a reader that is *about to* hold the
   key, handed out FIFO under a fan-out cap (`fanout_cap=1` → a chain, `≥2` → a
   shallow tree). No later reader is ever routed to an origin.
3. That peer has not registered yet, so the selection carries a **readiness gate**
   and the controller *withholds its answer* until the peer's read-through put
   lands. No client change is needed, and no client is lied to.
4. The read-through is the data plane's one job: a zero-fabric local `put` that,
   via the real `client.put` path, both stores the payload there and calls the
   real `notify_put_batch` — which opens the next reader's gate.

Because exactly one reader ever pulls from an origin, the only origin-sourced
transfer is that first hop: `origin_bytes == 1×` the payload, for **any** fan-out
cap. The unrouted baseline stays *m×*. There is no burst loop anywhere: the
scenario is a `client.put` and a gather of `client.get`, and the chain is an
emergent consequence of step 4 changing the directory step 1 reads.

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
| `Instance`, `Request`, prefix-hash | `control/request.py`, `workload/_generator.py` | An `Instance` owns one KV pool. A prompt is chunked into fixed `B`-token blocks, each **content-addressed by a prefix-hash chain** so shared leading blocks get identical keys (dedup/prefix-reuse falls out for free). A `Request` carries that chain *and the prompt itself* — one `device="meta"` `int64` per token, so the data plane takes a prompt where it used to take a count. The keys are still built by the generator rather than derived from the prompt, and that is the compromise a meta tensor forces: a content hash needs content, and this prompt deliberately has none. |
| `KVStore` | `data/store.py` | The three verbs that move KV (`publish` / `reuse` / `fetch`), over a `realsim.mesh.Mesh`, which supplies the **real** `Controller` directory plus a real per-instance `LocalClient`. It moves whatever it is handed and holds no notion of what a KV block is or how big one is: a forward pass produces the blocks, so the `Accelerator` owns their size and the premise every fetch is priced against (`block_bytes`) is enforced there. Eviction is not here either — a volume drops its own coldest and tells the directory itself. |
| `Accelerator` / `SimulatedAccelerator` | `data/_compute.py`, `workload/_accelerator.py` | What a forward pass costs, how it is made to take that long, and **what it produces**: `prefill(prompt, cached)` answers with one KV tensor per block this host now holds and did not before — which is exactly what `ServingHost` publishes — **and with the request's first token**, sampled from the pass's last position, which is the token TTFT is the time to. `step_tokens(batch)` is its decode twin: one token per batch member, produced by the step `claim_step` already charged. Under simulation those are `device="meta"` tensors — real `torch.Tensor`s with the model's dtype and exact byte count and zero storage, so publishing takes torchstore's *tensor* path (`Request.from_any`) rather than the object path a `(shape, dtype)` descriptor used to force. A deployment implements the same port by running the model. It also owns the device's **occupancy**, and both kinds of work book on it: a decode step through `claim_step`, a forward pass through the single-server queue `prefill` submits to. So a prefill really queues behind a prefill and behind a decode step, instead of sleeping the wait the scheduler predicted — the wait is emergent, and `RequestResult.queue_wait` vs `predicted_queue_wait` is how wrong the forecast was. |
| `KVView` | `control/_view.py` | The one derived directory *read* the scheduler needs: `prefix_lengths(block_keys)` — `{instance → leading blocks held contiguously}` — computed from the real `locate_volumes` by a prefix walk that stops at the first missing block. `PinnedKVView` fixes one snapshot for the duration of one routing decision. |
| `LongestPrefixPolicy` | `control/_source.py` | The only part of KV routing that is a *store* question, as a real `Policy`: rank instances by how much of the requested prefix they hold (id tie-break). |
| `LRUCache` | `control/_cache.py` | Per-instance bounded cache with **LRU eviction**, the local bookkeeping kept in sync with directory presence. Recency is a monotonic counter (deterministic). It picks victims; the data plane deletes them. |
| compute times | `../domain/llm.py` | Over `sim_common.cost_model`: `prefill_time` (per-uncached-token compute) and `decode_step_time(batch)` (per-step **TBT**, strictly increasing in batch size). Both planes call them — control to predict, data to charge. The cost of the *transfer* is `sim_common.cost_model.get_time`, shared with the transport seam. |
| `DecodeEngine` | `data/_decode.py` | Batched, stepped decode on the async engine (makes **TBT real**). Emits one token per step per batched request — real tensors, accumulated per member and handed to whoever admitted the request when its last one lands; step time = `decode_step_time(batch)`. Models the VRAM `max_batch` cap (over-cap requests queue, their wait counting against TBT) and **owns** the per-instance compute timeline. |
| `LoadBalanceScheduler` / `CacheAwareScheduler` | `control/scheduler.py` | The two policies (async). Load-balance: least-loaded instance, local-only cache. Cache-aware: route to minimize predicted TTFT using the **global** prefix directory; optionally pull a remote prefix under a balance threshold (which read-through-replicates the destination); LRU-evict on completion; SLO-reject. With `simulate_decode`, also picks a decode instance by *predicted* batch and applies the `early_rejection` admission mode. Decides only: it holds a *predicted* prefill queue and is told what actually happened. |
| `ServingHost` | `data/serving.py` | One serving instance: its cache, its decode batch, its compute. Three members, one per leg of a **redirect** — `route` (the coordinator says which host should prefill this; the answer is handed back, not acted on), `prefill` (real prefix pull → submit the forward pass, which waits for the device → real publish → answer with the **first token** and the decode host's address) and `decode` (a **real `get_batch`** of the request's whole block chain out of the store, then the batch, then the inter-token gaps → answer with the **remaining tokens**, `stream=False`-style, at the last one). It holds no reference to any other host and hands no measurement row to one: each host records its own half into the run's ledger, keyed by request id. Also owns **prefill/decode coupling**, which is a deployment fact, not a policy: when coupled it reports each decode step's end back to the scheduler, and the collision itself needs no reporting at all — both engines book on one accelerator. (The `reserve` call that used to push control's *predicted* completion onto that accelerator is gone: a prefill books its own slot now, so one object owns `busy_until`.) |
| `_Client` | `workload/_serving.py` | Submits each request to the host it *lands* on (the run wires client affinity) and then **follows the redirects**: route → prefill → decode, three charged client↔host round trips (`TOSO_CLIENT_RTT`, free by default). It is the only participant that receives the whole answer — the first token from the prefill host, the rest from the decode host — so the produced token count is a client-side join, exactly like the end-to-end latency it stamps. Where a request lands is a load balancer's answer and where it should *run* is the coordinator's — neither is a serving decision, so this is run wiring rather than capability code: a deployment deletes it and keeps the hosts. Reaches the runner through `ItemDispatch`, so wiring never has to declare itself a `DataPlane`. |
| `make_workload` | `workload/_generator.py` | Seeded synthetic generator: shared system prompt + per-conversation context + unique query suffix, conversations chosen by a **Zipf** popularity law, **Poisson** arrivals. |
| `Metrics` | `report/metrics.py` | Hit rate, compute/saved tokens, mean/p90 TTFT, fabric bytes, rejections, and decode-side TBT (`mean_tbt`, `pct_tbt`, `tbt_slo_met`, `wasted_prefills`, `decode_rejections`). |
| scenarios | `workload/scenarios.py` | The six `realsim.demo.Scenario` subclasses. Each declares its `Run` values over one request stream and narrates the results; `Demo.main` executes them with `Run.execute()`. |

### The event flow

1. **`make_workload`** produces a deterministic, arrival-sorted list of `Request`s.
2. **`Runner.run`** releases each request at its arrival in `(arrival, id)` order;
   on release
   **`scheduler.schedule(request, now)`** runs (serialized like the Monarch actor
   mailbox — consistent directory snapshot): query `prefix_lengths` for the best
   match, predict TTFT per instance (`queue + transfer + prefill(uncached)`), route
   to the minimum, optionally pulling a remote prefix; return `None` (reject) if
   predicted TTFT/TBT exceeds the SLO.
3. On prefill completion **`scheduler.complete`** decides the **read-through**: the
   prefill instance now holds KV for the whole prompt, so `cache.admit` inserts the
   keys (evicting the coldest past capacity) and returns what to publish and drop.
   The serving host then makes it real — `notify_put_batch` registers presence,
   `notify_delete_batch` removes evicted blocks — which makes a hot prefix
   **replicate** across instances over time — and reports the clock it reached
   back to the scheduler.
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

Both packages are split the same way, **by plane** — `control/` decides, `data/`
executes, plus `workload/` (what is simulated) and `report/` (outcome metrics) —
so the two can be read folder by folder. How *thick* each folder is is itself
informative; the comparison is tabulated in
[`../dedup_sim/README.md`](../dedup_sim/README.md#comparison-with-kvcache_sim).

The split is enforced, not just documented: `control/` may not import `data/`, the
mesh, or a store client, which `realsim/tools/check_contract.py` checks in the
same AST walk as the concurrency contract. Control receives a `View` and returns a
decision; anything that moves bytes reaches it as an *observation*. The rule for
which folder something belongs in is **does it advance the clock or move bytes?**

| Aspect | `dedup_sim` | `kvcache_sim` |
|---|---|---|
| Engine | `sim_common.AsyncEngine` (virtual-clock asyncio) | Same |
| Directory | **Real** `Controller` (`locate_volumes` → `StorageInfo`/`TensorSlice`) | **Real** `Controller` (`keys_to_storage_volumes`, key = prefix-hash chain) |
| Client / transport | Real `LocalClient` + real in-memory transport (via `realsim`) | Same |
| Unit of data | 1-D tensor slices, allocation-free carriers | fixed-size **KV blocks** (prefix-hash keys), one zero-storage meta tensor each |
| Decision-maker | `DedupPolicy` (a real `Policy`, consulted in the controller) | `CacheAwareScheduler.schedule`, which delegates only "which peer" to `LongestPrefixPolicy` |
| Concurrency primitive | The controller withholds a routed answer until the planned peer registers, so an about-to-be source becomes a real one | Servers reserved in the scheduler's predicted `busy_until`; SLO gate; no in-flight-as-source |
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
