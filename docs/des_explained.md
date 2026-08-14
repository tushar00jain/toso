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
  control-plane code executes under this loop unmodified.
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
| `KeySelector.select(keys, requester)` | Which volume serves these keys for this requester, and **when** it is usable — ranked sources, and it does not answer until they are usable. Naive — every holder, directory order — is the default, and is exactly what the real directory answers unaided. Reached as a service of its own: the data plane asks it and hands the answer to an ordinary `client.get` as a source preference (`prefer()`), so nothing is installed in the store and an app that wants to *price* alternatives asks the same object itself. |
| `View` | The read-only observation a control plane senses through: `locate`, topology/locality, the virtual clock. Synchronous reads, no mutation — a directory read is not a coroutine, so a decision formed against it cannot be interleaved with another. |
| `DataPlane` | One member — `attach(deployment)`, and no verbs: moving bytes is an ordinary client call and what a capability does around it is its own to name. Which member the runner drives is `realsim.runner.ItemDispatch`'s to say. |
| `Runner` | Releases work items on the virtual clock in `(release_time, id)` order, installs the mesh once, gathers. The gather is the whole wait — there is no drain phase behind it. |

Both sims `import realsim` and add only their own decision + execution code. So
does [`putget_sim`](../putget_sim/), which is not a capability at all: it is the
same put/get workload with **no** `KeySelector` and **no** `DataPlane`, and
therefore the unrouted *m×* baseline dedup measures against. Dedup imports its
`PutGetBurst` unchanged, which is what makes the two runs comparable.

[`domain/llm.py`](../domain/llm.py) is also shared: a `Model` reduces a
transformer to what a sim charges against (flops per prefilled token, flops per
decode step, KV bytes per token via `block_bytes()`), plus `prefill_time` /
`decode_step_time`. `kvcache_sim` prices prefill/decode compute and KV block sizes
from it; `dedup_sim` will size the weights it syncs from it (a TODO in that file
records the plan). It is domain fact, not simulator machinery and not selector,
which is why it sits beside them rather than inside either.

Because the sims execute real torchstore code, they depend on the **real
`torchstore` / `torch` / `monarch` install** (the from-source build), not
stdlib-only — the same dependency `realsim` has.

---

## 2. `dedup_sim` — weight-transfer deduplication

**The question it answers:** when *m* generator ranks all need the same tensor (or
overlapping shards) that lives on a trainer, can we move each unique byte across
the slow cross-node fabric **exactly once** (1×) instead of *m* times?

`dedup_sim` implements this as one `proposed.plane.ControlPlane`
(`dedup_sim.control.routing.Dedup`), asked by the data plane before each read and
told when its put lands, over the real `LocalClient`, the **real** `Controller`
directory and the real in-memory transport, on real types throughout.

### Components it adds (everything else comes from `realsim` / `sim_common`)

| Component | File | Role |
|---|---|---|
| `Dedup` | `control/routing.py` | The capability's whole control plane, two members: `sources(keys, me)` answers with a source and does not answer until that source is usable, and `published(me, keys)` is how it hears that a put landed. Holds no client, no volume, no mesh — only the ranking below. |
| `PeerKeySelector` | `control/_source.py` | The routing decision, as a real `KeySelector`: assign each requester a peer under a fan-out cap, gate the answer on the read-through that peer owes, and retire a source that holds nothing and owes nothing. No burst loop, no reader list, no count of how many readers there will be. |
| `ReadThroughPlane` | `data/read_through.py` | Dedup's whole executing half, one member: ask routing who serves this key, `get` from what it named, `put` the key into that reader's co-located volume (which through the real `client.put` path also calls the real `notify_put_batch`), then report that put back to routing. The report is what opens the next reader's gate. |
| dedup scenario | `workload/scenarios.py` | Runs `putget_sim`'s ordinary put/get fixture twice — unrouted (the *m×* baseline) and with the two planes added (1×) — so the comparison is byte-for-byte the same topology, payload and cost model. |
| demo entrypoint | `__main__.py` | `python -m dedup_sim` — fabric summary + ASCII diagram (INFO), full per-event trace (DEBUG, `-v`). |

### How it reaches 1× on the real directory

With no control plane, every reader `locate_volumes` the origin before anyone
finishes, so each pulls from the origin — *m×* fabric.

`Dedup` is asked first, and the read prefers what it named:

1. Readers ask it in order. The **first** is routed to a volume that already holds
   the key (the single fabric hop), chosen by locality.
2. Every later one is routed to a **peer**: a reader that is *about to* hold the
   key, handed out FIFO under a fan-out cap (`fanout_cap=1` → a chain, `≥2` → a
   shallow tree). No later reader is ever routed to an origin.
3. That peer has not registered yet, so the decision carries a **readiness gate**
   and `sources` *does not answer* until the peer's read-through put lands. The read
   itself is then an unmodified `client.get` with a preference passed to it: no
   client change is needed, nothing is installed in the store, and no client is
   lied to.
4. The read-through is the data plane's other job: a zero-fabric local `put` that,
   via the real `client.put` path, both stores the payload there and calls the
   real `notify_put_batch` — and then one call telling routing it landed, which
   opens the next reader's gate.

Because exactly one reader ever pulls from an origin, the only origin-sourced
transfer is that first hop: `origin_bytes == 1×` the payload, for **any** fan-out
cap. The unrouted baseline stays *m×*. There is no burst loop anywhere: the
scenario is a `client.put` and a gather of `client.get`, and the chain is an
emergent consequence of step 4 changing the directory step 1 reads, and saying so.

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
| `Instance`, `Request`, prefix-hash | `control/request.py`, `workload/_generator.py` | An `Instance` owns one KV pool. A prompt is chunked into fixed `B`-token blocks, each **content-addressed by a prefix-hash chain** so shared leading blocks get identical keys (dedup/prefix-reuse falls out for free). A `Request` carries that chain *and the prompt itself* — one `device="meta"` `int64` per token. The keys are built by the generator rather than derived from the prompt, the compromise a meta tensor forces: a content hash needs content, and this prompt deliberately has none. |
| `KVStore` | `data/_store.py` | The three verbs that move KV (`publish` / `reuse` / `fetch`), over a `realsim.mesh.Mesh`, which supplies the **real** `Controller` directory plus a real per-instance `LocalClient`. It moves whatever it is handed and holds no notion of what a KV block is or how big one is: a forward pass produces the blocks, so the `Accelerator` owns their size and the premise every fetch is priced against (`block_bytes`) is enforced there. Eviction is not here either — a volume drops its own coldest and tells the directory itself. |
| `Accelerator` / `SimulatedAccelerator` | `data/_compute.py`, `workload/_accelerator.py` | What a forward pass costs, how it is made to take that long, and **what it produces**: `prefill(prompt, cached)` answers with one KV tensor per block this host now holds and did not before — which is exactly what `ServingHost` publishes — **and with the request's first token**, sampled from the pass's last position, which is the token TTFT is the time to. `step_tokens(batch)` is its decode twin: one token per batch member, produced by the step `claim_step` already charged. Under simulation those are `device="meta"` tensors — real `torch.Tensor`s with the model's dtype and exact byte count and zero storage, so publishing takes torchstore's *tensor* path (`Request.from_any`). A deployment implements the same port by running the model. It also owns the device's **occupancy**, and both kinds of work book on it: a decode step through `claim_step`, a forward pass through the single-server queue `prefill` submits to. So a prefill really queues behind a prefill and behind a decode step: the wait is emergent, and `RequestResult.queue_wait` vs `predicted_queue_wait` is how wrong the forecast was. |
| `KVView` | `control/_view.py` | What a routing decision senses, as one class per sense, each a `proposed.View` itself and each claiming its own `derived()` keyword, so a capability may compose only the senses it reads. `PrefixSense` is the one derived directory *read* the scheduler needs: `prefix_lengths(block_keys)` — `{instance → leading blocks held contiguously}` — computed from the real `locate_volumes` by a prefix walk that stops at the first missing block, with `pinned(block_keys)` fixing one snapshot for the duration of one routing decision so every selector consulted inside it reads one directory. `ClusterSense` carries the run's `KVClusterModel` (`view.cluster`) and `RoutedSense` the pulls already priced (`view.routed`), because predicted queues, observed decode batches and a decision already taken are observations too: a rank key, a gate and the fetch's memo link each read the one sense they need instead of being handed the record. |
| `LongestPrefixKeySelector` | `control/_source.py` | The only part of KV routing that is a *store* question, as a real `KeySelector`: rank instances by how much of the requested prefix they hold (id tie-break), priced in blocks of that run. Wrapping it in `proposed.selector.Discount` re-ranks that answer by a bounded discount for how much the discount has lately routed at each source, so a replicated hot prefix is not always served by whichever holder sorts first — opt-in, and nothing selects it by default. |
| compute times | `../domain/llm.py` | Over `sim_common.cost_model`: `prefill_time` (per-uncached-token compute) and `decode_step_time(batch)` (per-step **TBT**, strictly increasing in batch size). Both planes call them — control to predict, data to charge. The cost of the *transfer* is `sim_common.cost_model.get_time`, shared with the transport seam. |
| `DecodeEngine` | `data/_decode.py` | Batched, stepped decode on the async engine (makes **TBT real**). Emits one token per step per batched request — real tensors, accumulated per member and handed to whoever admitted the request when its last one lands; step time = `decode_step_time(batch)`. Models the VRAM `max_batch` cap (over-cap requests queue, their wait counting against TBT) and **owns** the per-instance compute timeline. |
| `LoadBalanceScheduler` / `CacheAwareScheduler` | `control/scheduler.py` | Two *presets* of one parameterized scheduler (async). It prices every prefill instance and takes one, over two axes: **reuse**, a `proposed` selector asked once per decision ("rank the peers holding this prefix, or name nobody"), and **the rank key**, a function sorting the candidates it priced. Load-balance = never pull + least-loaded instance (local-only cache); cache-aware = pull under a balance threshold (which read-through-replicates the destination) + lowest predicted TTFT over the **global** prefix directory. It answers one question, `decide(request, me)`, and answers it once: a `Response` naming both of the request's hosts and what prefilling on the first was priced at, so admission is settled before anything runs. Behind it are two selections -- the prefill hosts ranked best first with each one's `Plan`, and the decode hosts ranked by predicted batch against the winner among them -- and they stay inside the scheduler; not a selector precisely because two selections do not fit in one `Selection`. A refusal is `None`. Admission is a list of installed **gates** (TTFT, and predicted TBT where the run models decode) rather than a mode string tested inline; `early_rejection` chooses what the TBT gate is fed, and `predict` enables the decode-occupancy lookahead that is all that separates it from `early`. With `simulate_decode` it also picks a decode instance by *predicted* batch. Decides only: it holds a *predicted* prefill queue and is told what actually happened. |
| `ServingHost` | `data/serving.py` | One serving instance: its cache, its decode batch, its compute. Three members, one per leg of a **redirect** — `route` (control says which host should prefill this; the answer is handed back, not acted on), `prefill` (real prefix pull → submit the forward pass, which waits for the device → real publish → answer with the **first token** and the decode host's address) and `decode` (a **real `get_batch`** of the request's whole block chain out of the store, then the batch, then the inter-token gaps → answer with the **remaining tokens**, `stream=False`-style, at the last one). It holds no reference to any other host and hands no measurement row to one: each host records its own half into the run's ledger, keyed by request id. Also owns **prefill/decode coupling**, which is a deployment fact, not a selector: when coupled it reports each decode step's end to control's model, and the collision itself needs no reporting at all — both engines book on one accelerator. |
| `_Client` | `workload/_serving.py` | Submits each request to the host it *lands* on (the run wires client affinity) and then **follows the redirects**: route → prefill → decode, three charged client↔host round trips (`TOSO_CLIENT_RTT`, free by default). It is the only participant that receives the whole answer — the first token from the prefill host, the rest from the decode host — so the produced token count is a client-side join, exactly like the end-to-end latency it stamps. Where a request lands is a load balancer's answer and where it should *run* is control's — neither is a serving decision, so this is run wiring rather than capability code: a deployment deletes it and keeps the hosts. Reaches the runner through `ItemDispatch`, so wiring never has to declare itself a `DataPlane`. |
| `make_workload` | `workload/_generator.py` | Seeded synthetic generator of **multi-turn conversations**: turn 1 is a shared system prompt + per-tenant context + a user message, and turn N+1 is turn N's whole sequence + turn N's **output** + a new message. A **Zipf** law over tenants decides how many turns each contributes; dialogue starts are **Poisson** and the pause between turns is exponential. One work item is a conversation, and its turns are serial. |
| `Metrics` | `report/metrics.py` | Hit rate, compute/saved tokens, mean/p90 TTFT, fabric bytes, rejections, and decode-side TBT (`mean_tbt`, `pct_tbt`, `tbt_slo_met`, `decode_blocks`). |
| scenarios | `workload/scenarios.py` | The six `realsim.demo.Scenario` subclasses. Each declares its `Run` values over one request stream and narrates the results; `Demo.main` executes them with `Run.execute()`. |

### The event flow

1. **`make_workload`** produces a deterministic, arrival-sorted list of
   `Conversation`s, each holding its turns (a `Request` and the user's pause in
   front of it).
2. **`Runner.run`** releases each *conversation* at its first turn's arrival in
   `(arrival, id)` order, and the client walks that dialogue's turns one at a time
   -- turn N+1 contains turn N's output, so it cannot be submitted until turn N has
   answered. Per turn,
   **`scheduler.select(request, me)`** runs (serialized like the Monarch actor
   mailbox — consistent directory snapshot): query `prefix_lengths` for the best
   match, predict TTFT per instance (`queue + transfer + prefill(uncached)`), route
   to the minimum, optionally pulling a remote prefix; abstain (reject) if
   predicted TTFT/TBT exceeds the SLO.
3. On prefill completion the host makes the **read-through** real: it publishes what
   it now holds and did not before, so `notify_put_batch` registers presence and a
   full volume drops its own coldest and reports that with `notify_delete_batch` —
   which makes a hot prefix **replicate** across instances over time. It then reports
   the clock it reached to control's model of the cluster (`PrefillFinished`).
4. When decode is simulated, prefill completion also admits the request into its
   decode instance's batch on the **`DecodeEngine`**; each step emits one token per
   batched request at `decode_step_time(batch)`, and the request's worst inter-token
   gap is recorded as its **TBT**.

> One documented simplification: a block becomes reusable at prefill *completion*,
> not while in flight — unlike `dedup_sim`, there are no promises here, so two
> requests racing for the same brand-new prefix may both compute it. Adding
> promises is noted as future work.

**The payoff metrics are TTFT, hit rate, compute saved, rejections, and — on the
decode side — TBT (attainment, decode-side residency)** — never wall-clock. The six
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
| Decision-maker | `Dedup` — one `ControlPlane`, asked `sources(keys, me)`; it ranks with a `KeySelector` utility (`PeerKeySelector`) | one `ControlPlane` too, asked `decide(request, me)` for compute placement and `sources(keys, me)` for the fetch that plan implies; it ranks with a `KeySelector` utility (`LongestPrefixKeySelector` by default) |
| Concurrency primitive | The plane withholds its answer until the planned peer registers, so an about-to-be source becomes a real one | Servers reserved in the scheduler's predicted `busy_until`; SLO gate; no in-flight-as-source |
| Cache | Read-through peer replication | Read-through + bounded **LRU eviction** |
| Decode model | n/a (weight transfer only) | Batched, stepped decode (`DecodeEngine`): per-step **TBT** rises with batch size; VRAM cap; coupled vs disaggregated |
| Randomness | None (fixed scenarios) | One seeded Zipf + Poisson workload |
| Payoff metric | **fabric bytes** (1× vs *m×*) | **TTFT / hit rate / rejections**, and on the decode side **TBT attainment / decode-side residency** |

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
