# Design: Mooncake-style KV-cache serving on TorchStore

**Status:** draft / proposal · **Scope:** LLM inference KV-cache pool + cache-aware
routing (prefill/decode serving), layered on the existing TorchStore substrate.
The reference architecture is the Mooncake paper (arXiv:2407.00079); this doc adapts
its *concepts* to TorchStore and is concerned only with the design.
See `architecture.md` for how TorchStore's control plane, data plane, and transport
work today.

A design for making **TorchStore double as a Mooncake-style KV cache** for LLM
serving — a disaggregated, prefix-reusing KVCache pool with a cache-aware
scheduler — *without* forking the store. The claim is
that **almost everything such a cache needs already exists in TorchStore's data plane
and transport; what's missing is a new control-plane policy** (a cache-aware
coordinator), a **key-naming convention** (prefix-hash block ids), and one
data-plane capability the weight-sync path never needed (**eviction**).

Mooncake (arXiv:2407.00079, *KVCache-centric Disaggregated Architecture for LLM
Serving*, ToS 2025) is the serving platform for Kimi. Its four moving parts are a
**KVCache pool** (paged KV blocks spread over the cluster's spare CPU/DRAM/SSD), a
**Transfer Engine** ("Messenger", GPUDirect-RDMA block movement), a global
**Conductor** scheduler (cache-aware prefill/decode routing under TTFT/TBT SLOs),
and **prefill/decode disaggregation**. This doc maps each concept onto TorchStore.

---

## 1. Goals / non-goals

**Goals**
- Let a serving engine (vLLM / SGLang / a custom rollout worker) use TorchStore as a
  **shared, cross-instance KV cache**: store paged KV blocks once, **reuse a matched
  prefix** on any instance, and move a block across the fabric **once** when a remote
  instance needs it.
- Provide a **cache-aware location + routing service** (the Conductor analog): given a
  request's prompt, tell the caller which instance has the longest reusable prefix and
  what the predicted TTFT is, so requests land where reuse is highest without
  overloading a hot instance.
- Support a **bounded, long-lived cache**: real **eviction** (LRU/LFU/length-aware) and
  optional **DRAM→SSD tiering**, because an inference cache is unbounded and persistent,
  unlike the versioned, burst-scoped weight-sync cache.
- Be **additive**: the geometry-addressed `put`/`get`/`*_state_dict` weight-sync path
  keeps working unchanged. KV-cache mode is a *different
  control-plane policy over the same volumes and transport*.

**Non-goals**
- **Not** an inference engine. TorchStore stores, locates, and moves KV blocks; it does
  **not** run attention, chunked/layer-wise prefill compute, or decode. Those stay in
  the serving engine (the paper's Chunked Pipeline Parallelism and layer-wise prefill
  overlap are engine concerns; TorchStore only needs to expose async load/store so the
  engine *can* overlap).
- **Not** owning prefill/decode **compute disaggregation** itself. TorchStore is the
  KVCache pool + transfer + location service that *makes* disaggregation cheap; which
  GPU runs prefill vs decode is the engine/orchestrator's call.
- **Not** re-implementing the Transfer Engine. TorchStore's transport (Monarch RDMA /
  TorchComms RDMA / shm / Gloo) already *is* Messenger; we reuse it.
- **Not** a full SLO admission controller in v1. The store exposes the *signals*
  (per-instance prefix-match length, load, predicted TTFT); prediction-based early
  rejection is an optional policy on top (§5.7), not a store primitive.

---

## 2. Background: what exists today, mapped to the reference architecture

TorchStore already has three of Mooncake's four parts. The table is the crux of the
"we only need a control plane" answer:

| Reference component | TorchStore equivalent (today) | Gap |
|--------------------|-------------------------------|-----|
| **KVCache pool** — objects over spare CPU/DRAM/SSD | `StorageVolume` / `InMemoryStore` (`storage_volume.py`), one per rank/host; already a distributed byte pool | add **eviction** + optional **SSD tier** |
| **Transfer Engine** — RDMA block movement, multi-NIC device selection | Transport layer (`transport/*`): shm → Monarch RDMA → TorchComms RDMA → Gloo → RPC, auto-selected per transfer | **none** — reuse as-is; the primitives line up 1:1 |
| **Metadata master** — global directory, no data flow | `Controller` directory `key → {volume_id → StorageInfo}` + prefix `Trie` (`controller.py`) | **none** — reuse; blocks are just keys |
| **Client + transfer submitter** | `LocalClient` (`client.py`) driving `put`/`get` over the transport | **none** — reuse |
| **KV-cache event feed** — stored/removed event stream | *(nothing)* — Controller has no stored/removed event stream | **NEW seam** (optional): Controller emits stored/removed events the coordinator subscribes to (§4) |
| **Conductor** — cache-aware prefill/decode scheduler | *(nothing)* | **NEW control plane**: a `CacheCoordinator` layered in front of the `Controller` (a policy actor over the dumb directory) |
| **Prefix-hash block id** — computed in the serving-engine connector, not the store | opaque string keys (any scheme allowed) | **naming convention** only, no code change |
| **Prefix matching** | `Controller.locate_volumes([keys])` / `keys(prefix)` on the trie | adapt: walk the block-key chain |
| **Eviction / replication policy** | `delete` / `notify_delete` exist; no *policy*, no replica config | **NEW data-plane policy** |
| **Hot-block replication / swap** | read-through population + peer-source fan-out (K4/K3 below) | **reuse the pattern**, drive it by access frequency instead of a burst |

So: **data plane ✓, transport ✓, directory ✓.** The genuinely new work is (a) a
cache-aware **coordinator** (control plane — the Conductor analog), (b) a prefix-hash
**key convention** (a connector concern, §5.1), and (c) **eviction** (a data-plane policy
the weight-sync path never needed).

Three control-plane mechanisms from the reference architecture worth adopting:

- **2-phase commit on writes.** A write begins, streams, then commits (or is revoked on
  failure), so readers never observe a half-written object — the same role as
  TorchStore's `MAPPING` commit marker. Eviction and reads both skip objects not yet
  marked complete.
- **Scheduler-facing capacity query.** The directory should expose per-segment
  `(capacity, used)` so the coordinator has the load/capacity signal it needs (§5.3).
- **Sharded directory.** Under a high request rate, metadata can be split across shards
  routed by `hash(key)`, which answers this doc's open question about sharding the
  coordinator (§5.9, §7).

Two facts from the existing store make this fit cleanly:

- **Volumes are already a reusable, addressable byte pool.** Under `LocalRankStrategy`
  every rank owns a volume; under `HostStrategy` every host owns one. A KV block is just
  a `put(block_key, kv_tensor)` into a volume, and the controller indexes where it lives
  — the same directory that answers `locate_volumes` for weights answers "which instances
  cache this prefix block."
- **The controller is metadata-only and already prefix-structured.** Its `Trie` gives
  cheap prefix listing; `locate_volumes` already returns per-key, per-volume presence.
  That is exactly the query the coordinator needs.

---

## 3. Shared substrate (what KV-cache mode depends on)

Reusable pieces (the substrate this design builds on). Most exist today; K5
(eviction) is the main new data-plane policy.

| # | Piece | What / where |
|---|-------|--------------|
| K1 | **Prefix-hash block addressing** | A KV block's key is a **hash chain**: `block_key[i] = H(block_tokens[i] ‖ block_key[i-1])`. Content-identical prefixes ⇒ identical keys ⇒ **dedup + reuse fall out** for free. **This is a client/connector convention, not a store change** — the serving-engine connector computes the block hash and passes it as the *opaque* object key. The store stays opaque and content-**agnostic**: keys are application-supplied, and the store simply stores each block as a plain tensor under that key. |
| K2 | **Block-level put/get** | Store/fetch one KV block via the existing `put`/`get` on a volume + transport. A block value is the KV for `B` tokens (e.g. `[2, n_layers, B, n_kv_heads, head_dim]`); the store stays geometry-blind (§9 of architecture.md) — a block is just bytes under a key. |
| K2b | **Prefix locate** | `locate_volumes(block_keys)` (existing) returns per-block presence; the **longest present run from the front, per volume** = that instance's reusable prefix length. This is the coordinator's core query, computed from the existing directory. |
| K3 | **Peer-source transfer** | A block resolves to *any* volume holding it, incl. a peer instance's volume; the transport moves it once. This is already indexable in the Controller directory. |
| K4 | **Read-through population** | After an instance fetches/computes a block, it `put`s the block into its own volume and the controller registers it — so peers can reuse it. Persistent: the block stays cached (no version window). |
| K5 | **Eviction policy** | Per-volume LRU (default; the paper found LRU best on their traces) / LFU / length-aware, driven by access recency the coordinator observes on the request stream. Evicting a block = `delete(block_key)` on that volume + `notify_delete`. **New** relative to weight-sync (which pins within a version window). |
| K6 | **Locality cost model** | Rank sources/instances by locality: shm > NVLink peer > cross-node RDMA. These tiers drive which instance's prefix is "cheap" to reuse and whether to transfer vs recompute. |
| K7 | **Client-side local hot cache** (optional) | A per-client hot cache admitted by frequency: only keys whose access count clears a threshold are cached locally, so repeat fetches of very hot blocks short-circuit before reaching the coordinator. An optional client-side tier in front of the coordinator. |

K1+K2b+K3+K4 are the reusable core (mostly existing); K5 is the new data-plane policy;
the coordinator (§5) adds cache-aware routing on top.

---

## 4. Layered controllers (the key structural change)

The key structural move: **split policy from storage.** Keep the
existing `Controller` as a dumb directory; add a **`CacheCoordinator`** actor in front
that owns prefix matching, cache-aware routing, replication, and eviction policy. It
**delegates** block-location lookups to the `Controller` and only *instructs* clients —
KV bytes still move client↔volume over the existing transport.

```
   serving instance i                   NEW LAYER                      EXISTING
 ┌───────────────────────┐     ┌──────────────────────────┐   ┌───────────────────┐
 │ schedule(request):    │route│    CacheCoordinator       │   │  Controller       │
 │  prompt -> block_keys │────▶│  (Conductor analog):      │──▶│  (block directory:│
 │                       │◀────│   prefix-match + TTFT      │   │   key→{vol→info}, │
 │  prefill uncached     │(p,d)│   routing + replication    │   │   trie)           │
 │  suffix, put blocks   │     │   + eviction policy        │   └───────────────────┘
 │  (K4), decode         │     │  — one actor, serialized   │            │
 └──────────┬────────────┘     └──────────────────────────┘            ▼
            └────────── KV block transfer (existing transport) ──▶┌───────────────────┐
                                                                  │  StorageVolume(s) │
                                                                  │  (KV block pool)  │
                                                                  └───────────────────┘
```

- **Controller (unchanged):** block-location index, prefix trie, presence. No policy.
- **CacheCoordinator (new):** answers "for this prompt, which instance has the longest
  reusable prefix, what's the predicted TTFT on each, where should this request run, and
  should I replicate a hot block?" Owns the routing/replication/eviction policy. This is
  precisely the role of Mooncake's **Conductor** — a global scheduler that maintains a
  prefix table and routes requests. The `CacheCoordinator` is TorchStore's in-tree
  equivalent.
- **Fallback:** with no coordinator, the store is the plain geometry-addressed KV store;
  the weight-sync path is unaffected (KV-cache mode is a *different* policy over the
  same volumes).

**Integration seam — how the coordinator learns block locations.** Two options:

- **(a) Synchronous directory query** (what §5.4 describes): each `schedule` calls
  `Controller.locate_volumes(block_keys)` inline. Simplest; the coordinator reads
  presence on demand.
- **(b) KV-event feed** (decouples the router): the Controller emits **stored/removed**
  events, and the `CacheCoordinator` subscribes and maintains its own prefix index, so
  routing no longer blocks on a directory round-trip. **This requires TorchStore's
  Controller to grow an event-emit hook** on `notify_put`/`notify_delete` — a small,
  additive change. Start with (a); adopt (b) when the synchronous query becomes the
  bottleneck.

---

## 5. The CacheCoordinator (Conductor analog)

### 5.1 Prefix-hash addressing (K1), precisely

A prompt of `T` tokens is chunked into blocks of `B` tokens (paper uses `B=512`). Block
`i`'s key encodes the whole prefix up to `i`:

```
block_key[0] = H(model_id ‖ tokens[0:B])
block_key[i] = H(block_key[i-1] ‖ tokens[i*B:(i+1)*B])
```

Two requests sharing the first `k` blocks produce **identical** `block_key[0..k-1]`, so
their shared prefix is one set of entries in the store — automatic dedup and reuse
(the paper's "share prefix caching for the first 12·512 = 6144 tokens" is exactly a
12-key run match). Keys are opaque to the store, so this needs **no store change** — it's
a convention the **client/connector layer** computes and passes as an opaque object key,
while the store itself stays content-agnostic. Attributing the hashing to the connector
*reinforces* "no store change needed" — TorchStore's client library owns K1, the
volumes/directory are untouched.

### 5.2 API

Two coordinator endpoints, plus ordinary block `put`/`get` underneath:

```python
# 1) locate + route: the Conductor query
plan = await cache.schedule(request)     # request = (model_id, prompt_tokens, out_len, slos)
# plan -> (prefill_instance, decode_instance, prefix_match_len,
#          reuse_source_instance_or_None, predicted_ttft)  |  Reject(429)

# 2) after prefill, the instance publishes the blocks it computed (read-through, K4)
await cache.publish(prefill_instance, new_block_keys)   # -> notify_put_batch under the hood
```

The serving engine calls `schedule`, runs prefill on `plan.prefill_instance` for the
**uncached suffix only** (fetching `plan.prefix_match_len` blocks from
`reuse_source_instance` if remote), then `publish`es the blocks it produced.

### 5.3 Coordinator state

```
# delegated to Controller (existing): block_key -> {volume_id -> present}
# owned by CacheCoordinator:
access_recency[volume][block_key]   : clock          # for LRU eviction (K5)
capacity[volume]                    : int (blocks)   # per-instance cache budget
load[instance]                      : queue of pending prefill (predicted busy-until)
decode_load[decode_instance]        : batch occupancy (for TBT SLO / early reject)
replicas[block_key]                 : set[volume]    # for hot-block spreading (§5.6)
```

`access_recency` and `capacity` drive eviction; `load`/`decode_load` drive TTFT/TBT
prediction and admission.

### 5.4 Per-request algorithm (cache-aware scheduling — Mooncake Algorithm 1)

Runs to completion per request (serialized actor mailbox). Mirrors the
paper's Algorithm 1:

```
on schedule(request):                          # runs atomically
  block_keys   = prefix_hash(request.prompt, B)             # K1
  presence     = controller.locate_volumes(block_keys)      # K2b (existing directory)
  best_len, best_inst = global_longest_prefix(presence)     # blocks matched cluster-wide

  best_ttft, pick, src = +inf, None, None
  for p in prefill_instances:
    local_len = longest_prefix_on(p, presence)              # p's own reusable prefix
    T_queue   = predicted_busy_until(p) - now               # from load[p]
    if best_len / max(local_len,1) < balance_threshold:     # local reuse good enough
        uncached = request.T - local_len*B
        T_prefill = prefill_cost(uncached)                  # cost model (offline-fit)
        ttft, from_src = T_queue + T_prefill, None
    else:                                                    # remote prefix worth pulling
        transfer_len = (best_len - local_len) * B
        uncached     = request.T - best_len*B
        ttft = transfer_cost(transfer_len, best_inst→p) + T_queue + prefill_cost(uncached)
        from_src = best_inst
    if ttft < best_ttft: best_ttft, pick, src = ttft, p, from_src

  d, tbt = select_decode_instance(decode_load)              # load-balanced decode pick
  if best_ttft > SLO_ttft or tbt > SLO_tbt:  return Reject(429)   # §5.7
  if best_len / prefix_on(pick) > balance_threshold:        # §5.6 hot-block migration
      schedule_replicate(best_inst, pick, block_keys[prefix_on(pick):best_len])
  load[pick].add(best_ttft); decode_load[d].add(...)
  return Plan(prefill=pick, decode=d, prefix_match_len=..., reuse_source=src, ttft=best_ttft)
```

`prefill_cost` / `transfer_cost` are offline-fit predictors — the paper notes prefill
time is very predictable (regular Transformer compute) while transfer time is noisier
(depends on live congestion), which is *why* it prefers replicating hot blocks over
always pulling (§5.6).

### 5.5 Eviction & tiering (K5) — the piece weight-sync never needed

The weight-sync path needs no eviction — its cache is bounded and short-lived (blocks
are pinned within a version window). An inference cache is neither, so eviction is
mandatory. Plain LRU is the baseline, but a richer scheme pays off:

- **Watermark-triggered batch eviction (approximate LRU).** Rather than evicting on every
  `put`, an eviction pass triggers at a **high watermark** (e.g. ~95% of capacity) and
  frees a batch (e.g. ~5%) in one pass. Eviction is **metadata-only** — evicted objects
  are just *marked deleted* + `notify_delete`, no data move. This avoids per-put LRU
  churn. LRU remains the recency order (the paper found LRU best on their traces:
  "temporal proximity in request utilization"); a near-LRU two-pass scan prefers objects
  with the least remaining protection.
- **Leases — the concrete "never evict a block being read."** Every successful presence
  check / fetch grants or refreshes a per-object lease (e.g. TTL ~5s); while active, the
  object is protected from deletion/eviction. Objects not yet marked complete by the
  commit marker are likewise skipped. The coordinator grants a lease on each
  `schedule`/fetch so a block cannot be evicted out from under an in-flight read.
- **Soft pin / hard pin — protection tiers above LRU.** *Soft pin* marks important,
  frequently-used objects (e.g. system prompts): evicted only as a last resort when no
  unpinned object is eligible, and the pin auto-expires after a TTL (e.g. ~30 min) if
  untouched. *Hard pin* is never evicted (e.g. weights). Both are set at put time. This
  turns §5.5 from plain LRU into **LRU + soft/hard-pin protection tiers**: the first
  eviction pass skips soft-pinned objects; a second pass may touch them only under
  pressure.
- **Hit-ratio vs capacity** is the key tuning curve (paper: ~30% hit @ 1k blocks →
  ~50% @ 50k, then plateau; >50% of blocks unused, a few accessed tens-of-thousands of
  times → heavy skew). The sim (§ below) reproduces this shape so capacity can be sized
  before deployment.
- **DRAM→SSD tiering.** A spill tier plus promotion-on-hit: a demoted block stays in the
  directory with a tier tag so `locate` still finds it (slower fetch), and a hit promotes
  it back to DRAM. v1 can ship a single DRAM tier + evict; tiering is a data-plane
  extension, not a control-plane change.

### 5.6 Hot-block replication / swap (peer-source + read-through, K3/K4)

A single instance holding a very hot prefix becomes a serving bottleneck (the paper's
motivation for hot-spot migration). The coordinator replicates on demand, driven by the
`balance_threshold` heuristic: **if the best remote prefix match is more than
`threshold`× the picked instance's local match, forward the block location so the picked
instance pulls and stores it locally** (read-through, K4) — otherwise prefer local
recompute. This reuses the peer-source + read-through pattern (K3/K4), triggered by
*access skew over time*, and the replica **persists**
in the cache (subject to eviction) rather than living only for one version window.

Make replication **config- and strategy-driven** rather than an ad-hoc heuristic: a
per-block replica target plus a pluggable **placement strategy**. Two useful strategies:
local-preference random, and a free-ratio-first pick that samples several candidate
segments and chooses those with the highest free ratio (a best-of-N load-spreading
pick). Replication is best-effort with slice-level placement guarantees (each replica's
slices in different segments). The coordinator should also mark a genuinely hot prefix
**soft-pinned** (§5.5) so replicas survive eviction pressure.

### 5.7 Overload / prediction-based early rejection (optional policy)

The store exposes the load signals; the rejection *policy* is a thin layer:

- **SLO as load measure:** compare predicted max TTFT (prefill) and TBT (decode) against
  `l_ttft` / `l_tbt`; reject (429) when unachievable — the `schedule` return above.
- **Early rejection:** assess **decode** load *before* committing prefill (route on the
  greater of prefill/decode pool load), so we don't burn prefill compute on a request
  the decode pool can't sustain.
- **Prediction-based (anti-oscillation):** naive early rejection oscillates (prefill and
  decode loads swing anti-phase because decode load is predicted before prefill
  finishes). The paper's fix predicts decode load *after* the incoming request's prefill
  stage (system-level: assume uniform decode time, roll the batch forward, compare
  average TBT ratio to `l_tbt`). v1 can ship SLO-reject only and add prediction later.

**How the design accounts for TBT (batched decode).** TTFT is a prefill-side cost;
**TBT (time-between-tokens)** is a decode-side cost, and it is not a fixed per-request
number — it is set by *batching*. A decode instance generates one token per step for
**every** request in its batch, so the step's wall time is the TBT every batched
request observes for that token, and it **rises with the batch size** (more concurrent
requests ⇒ more KV attended per step). This is the tension the TBT SLO bounds
(Mooncake arXiv:2407.00079 §4.2): a larger batch lifts MFU/throughput but pushes TBT
up, so the sweet spot is a small-but-nonzero batch. Two levers set a served request's
TBT:

- **VRAM cap on the batch.** Aggregated KVCache is bounded, so a decode instance's
  batch can only grow so far; requests over the cap queue, and that wait counts
  against their TBT. VRAM pressure therefore shows up as TBT violations — which is why
  admission must reason about a *predicted* batch, not accept blindly.
- **Prefill/decode disaggregation.** If prefill and decode share an instance's compute,
  a long prefill delays the next decode step and spikes that token's TBT. **Placing
  decode on its own pool** (its own compute timeline) means a prefill can never stall a
  decode step — the central Mooncake result, and the reason disaggregation protects
  served-request TBT even at identical admitted load.

The **early-rejection** policy above (§5.7) is exactly the admission decision that
keeps TBT in SLO without burning prefill: reject *before* prefill on the decode load
**predicted at prefill completion** (including in-flight prefills that will have landed
by then), rather than late-checking *after* prefill (a wasted prefill) or gating on a
stale *current*-occupancy snapshot (which slow prefills leave reading empty, so decode
piles onto one instance and blows the SLO). The `../kvcache_sim/` DES now models all of
this — a batched `DecodeEngine` (per-step TBT rising with batch size, a `max_batch`
VRAM cap, coupled vs. disaggregated compute timelines), a TBT SLO/target on the
worst per-request inter-token gap, and the three admission modes (`off`/`early`/
`predict`) — in its `disaggregation` and `early_rejection` scenarios (§5.8).

### 5.8 Sequence (two requests sharing a system prompt)

```
B=512. Req1: 3 blocks (system[0], userA[1..2]). Req2: 3 blocks (system[0], userB[1..2]).
t=0  schedule Req1: nothing cached -> pick p0 (least load); prefix_match=0
     p0 prefills 3 blocks, publish(p0, [k0,k1,k2])              (compute 1536 tok)
t=..  schedule Req2: block_keys=[k0,k1',k2']; locate -> k0 present on p0 (len=1 block)
     best_len=1 (system prompt hot). local on p0 also 1. route p0 (same reuse, low load)
     p0 reuses k0 (512 tok cached), prefills only k1',k2'  (compute 1024 tok, saved 512)
     publish(p0, [k1',k2'])
 => system prompt computed ONCE; Req2's TTFT drops by the prefill of 512 cached tokens.
    If p0 were overloaded, the coordinator routes Req2 to p1 and (if k0 match ≫ p1's) pulls
    k0 from p0 over NVLink once (K3) instead of recomputing it.
```

(A discrete-event simulation of exactly this — cache-aware vs round-robin routing,
eviction-capacity sweeps, hot-block replication under load, and the decode-side TBT
model above (batched decode, prefill/decode disaggregation, and off/early/predict
early rejection) — lives in `../kvcache_sim/`; run `python -m kvcache_sim`.)

**Fidelity note (sim vs. this design).** The `../kvcache_sim/` DES models the
**cache-aware router** — the routing decision this design's `CacheCoordinator` owns. It
deliberately **abstracts the cache-lifecycle mechanisms** that §5.5/§5.6 describe: the
sim approximates eviction as **per-instance LRU at a capacity bound**, and does *not*
model leases, watermark+batch eviction, soft/hard pins, or DRAM↔SSD offload/promotion.
Those are faithfully *described* here but simplified in the sim to keep the routing study
tractable; treat the sim as a router/capacity-sizing tool, not a storage-fidelity model.

### 5.9 Failure modes / costs
- **Coordinator hot path:** every request serializes through one actor doing O(#blocks)
  directory ops. Cheap, but a very high request rate could back-pressure — shard the
  coordinator by prefix-hash (the first block key) if needed. Sharding metadata by
  `hash(key)` is a standard mitigation.
- **Stale directory after eviction:** if a volume evicts a block but the `notify_delete`
  races a concurrent `locate`, a reader could be routed to a now-empty volume. The fetch
  must fall back (re-locate or recompute) — the coordinator treats a missed fetch as a
  cache miss, not an error. A **lease** granted on read closes most of this race: it
  blocks eviction of a block being read, and a lease that expires mid-fetch fails the
  read rather than returning torn data. Pairing eviction with leases (§5.5) plus a
  miss-tolerant fetch removes the corruption window.
- **Transfer-time misprediction:** the paper flags transfer time as congestion-sensitive
  and hard to predict; over-eager remote-prefix pulls can miss TTFT. The
  `balance_threshold` and preferring local recompute when matches are close mitigate it.
- **Eviction thrash under skew:** a too-small capacity on a hot instance evicts blocks
  that are immediately re-requested. The capacity sweep (§5.5, sim) sizes this.
- **Correctness of dedup by hash:** two distinct prefixes must never collide to the same
  `block_key`. Use a strong hash and include `model_id` (and dtype/quantization) in the
  chain so caches from different models/precisions never alias.

---

## 6. Recommendation & phasing

**Recommendation:** build KV-cache mode as a **second layered controller** over the
existing substrate, reusing volumes + transport + directory. It's a natural fit: the
store is already a distributed, RDMA-connected, metadata-indexed byte pool — precisely
the KVCache pool + Transfer Engine + global metadata the reference architecture calls
for. Only the *policy* is new.

**Phasing (each independently shippable):**
1. **K1 + K2/K2b (prefix-hash blocks + prefix locate)** — a client-side key convention
   plus a `locate`-based prefix-match query. Enables *local* prefix reuse (vLLM-parity)
   and cross-instance *discovery* with no new actor.
2. **K3 + K4 (peer-source fetch + read-through)** — a matched
   prefix on a peer is fetched once and republished. Cross-instance reuse works.
3. **`CacheCoordinator` (cache-aware routing)** — the Conductor: TTFT-predicting,
   prefix-maximizing routing (§5.4). The headline throughput win.
4. **K5 (eviction)** — LRU + capacity; makes the cache long-lived and bounded. Ship with
   the capacity-sizing sim.
5. **§5.6 hot-block replication** — access-skew-driven spreading.
6. **§5.7 early rejection / SLO admission** — overload handling (SLO-reject first,
   prediction-based later).
7. **DRAM→SSD tiering** — data-plane spill tier (future).

## 7. Open questions
- **Coordinator placement/sharding** for high request rates (one actor vs. prefix-hash
  sharded). Sharding the directory by `hash(key)` is the standard answer; the open
  question is mostly *when* to shard, not *how*.
- **Block granularity `B`** and whether it should match the engine's paged-attention
  page size (avoids a repack on load/store).
- **Cost-model fidelity:** how much offline profiling to fit `prefill_cost` /
  `transfer_cost`, and per-model vs. per-cluster fits.
- **Eviction coordination:** per-volume-local LRU (simple, may evict a globally-hot block
  that only that volume holds) vs. coordinator-aware eviction (avoid evicting the last
  replica of a hot block). Start local, revisit.
- **Interaction with weight sync on the same store:** a serving instance is often *also*
  a generator being weight-synced (RL). Do KV-cache keys and weight keys share a store
  (separate namespaces) or separate stores? Namespacing is cheapest.
- **Consistency across model versions:** a weight update (new policy in RL) invalidates
  the KV cache. Tie KV `block_key`s to the weight `version` (MAPPING marker) so a sync
  bump auto-invalidates stale KV.
