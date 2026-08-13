# Design: replica-aware weight-transfer dedup on TorchStore

**Status:** draft / proposal · **Scope:** trainer→generator weight sync for RL.
See `architecture.md` for how TorchStore's control plane and resharding work today.

A design for making replica-aware, **deduplicated** weight transfer a capability of
**TorchStore** itself, rather than something each application re-implements or a separate
point-to-point library must own.

The approach is **dynamic**: a routing layer builds the transfer plan *incrementally as
requests arrive*, exploiting Monarch's serialized actor mailbox to treat in-flight
fetches as promised cache sources. There is no barrier and no cohort registration —
readers just call `get` — yet a synchronized burst still dedups to 1× fabric.

It sits on a shared substrate and a new **layered controller**, and needs no knowledge of
the model or its parallelism (tensor/pipeline/expert parallel degrees, layer types,
fusion conventions). It needs only two things:

- **layout geometry** — how a tensor is sharded (offsets/shapes/replica classes), which
  the controller already holds from DTensor metadata; and
- **read demand** — who wants which region, which the store can *observe* from the
  request stream rather than require declared up front.

---

## 1. Goals / non-goals

**Goals**
- Move each unique weight region across the fabric **once** per hop (eliminate the `m×`
  broadcast and the single-source bottleneck) for a trainer → `m`-replica-generator sync.
- Reshard across differing trainer/generator meshes (already TorchStore's job).
- Require **no** model/parallelism knowledge. Use only the geometry the controller
  already holds and the read demand it can observe.
- Be **additive**: existing `put`/`get`/`put_state_dict`/`get_state_dict` keep working;
  the store falls back to today's behavior when the feature is off.

**Non-goals**
- Matching the wall-clock of a purpose-built one-hop transfer engine — that also needs
  fused packing, pipelined rounds, and one-sided-RDMA flag signaling. We target
  **bytes-moved** optimality plus reasonable compute/transfer overlap, not the last
  microsecond.
- Cross-name / operator-fusion / transpose **transforms** between differently-named or
  differently-fused parameters — a separate concern from transfer/dedup.
- Changing the `direct_rdma` handle path (it reshards/dedups nothing by design).

---

## 2. Background: what exists today (grounding)

- **Controller** (`controller.py`) — single Monarch actor, obtained via
  `get_or_spawn_controller(store_name, Controller)`. Holds the storage index
  `keys_to_storage_volumes: key → {volume_id → StorageInfo(object_type, {TensorSlice})}`.
  Metadata only. Endpoints: `locate_volumes`, `notify_put_batch`, `keys`, `notify_delete*`.
- **StorageVolume** (`storage_volume.py`) — data-plane actors holding bytes; one per
  rank under `LocalRankStrategy` (`volume_id = str(rank)`). So **every generator rank
  already owns a volume** — usable as a cache with no new spawn.
- **LocalClient** (`client.py`) — in-process; `get` → `locate_volumes` → per-volume
  fetch (`_expand_tensor_slices` intersects, `_assemble_results` reassembles via
  `assemble_tensor`). This is the reshard-on-get path the design reuses.
- **Commit marker** — `put_state_dict` writes all tensors then the `.../MAPPING` key
  **last**; `get_state_dict` reads MAPPING first. Natural **version/commit** signal.
- **TensorSlice** — `(offsets, coordinates, global_shape, local_shape, mesh_shape)` in
  global coords. Replicated shards appear as identical `(offsets, local_shape)` under
  different `coordinates` → replica classes are already visible to the controller.

---

## 3. Shared substrate (the design depends on these)

| # | Piece | What / where |
|---|-------|--------------|
| S1 | **Read-each-region-once** | On read, collapse identical stored slices; fetch each unique `(offsets, local_shape)` from **one** volume, not all `m`. Change in read request-planning (`client._build_volume_requests`). Fixes today's DP over-fetch (`_expand_tensor_slices` TODO). |
| S2 | **Put-side de-replication** (optional) | Replicated writers store only a disjoint `1/k` sub-slice (an even split of the shared shard across the `k` replicas); `get` reassembles via existing machinery. Cuts store footprint + put traffic. |
| S3 | **Peer-source routing** | A region resolves to *any* volume holding it — including a peer generator's volume. Already indexable; needs the client to accept an explicit source. |
| S4 | **Read-through population** | After a reader fetches region `R`, it `put`s `R` into its **own** volume and the controller registers it (version-keyed), so peers can source from it. |
| S5 | **Source-preference cost model** | Rank sources by locality: same-host shared-mem > intra-node NVLink peer > cross-node RDMA peer > trainer. Drives who-serves-whom. |

S1+S3+S4+S5 are the reusable core; the coordinator (§5) adds the demand-aware
source→destination assignment on top.

---

## 4. Layered controllers (the key structural change)

Split selector from storage. Keep the existing `Controller` as a dumb directory; add a
**`TransferCoordinator`** actor in front that owns demand-aware routing, dedup,
dependencies, and cache lifecycle. It **delegates** storage lookups to the `Controller`
and only *instructs* clients — bytes still move client↔volume as today.

```
   generator rank i                    NEW LAYER                     EXISTING
 ┌───────────────────┐        ┌──────────────────────────┐   ┌───────────────────┐
 │ LocalClient.get(  │  plan  │   TransferCoordinator    │   │  Controller       │
 │   key, dtensor )  │ ─────▶ │   (routing / dedup /     │──▶│  (storage index:  │
 │                   │ ◀───── │    promises / cache mgmt)│   │   key→{vol→slice})│
 │  executes byte    │  {src, │   — one actor, serialized│   └───────────────────┘
 │  fetches (S3/S5), │  wait} │     mailbox              │            │
 │  re-publishes(S4) │        └──────────────────────────┘            ▼
 └─────────┬─────────┘                                        ┌───────────────────┐
           └──────────── byte transfer (transport) ──────────▶│  StorageVolume(s) │
                                                              └───────────────────┘
```

- **Controller (unchanged):** index, commit tracking, key listing. No selector.
- **TransferCoordinator (new):** answers "for this reader's needed regions, which source
  should each come from, and what must I wait for?" Owns the plan, the promise/dependency
  table, and cache-volume registration.
- **Fallback:** if no coordinator is configured, `get` behaves exactly as today.

The coordinator is the only new selector surface; the client/transport/publish plumbing is
shared with ordinary `get`/`put`.

---

## 5. The coordinator (dynamic cache + routing + queuing)

### 5.1 The Monarch insight that makes it work

The coordinator is a **single Monarch actor**, so its endpoint calls are **serialized**:
a burst of `m` simultaneous gets becomes a **total order** `req_0, req_1, …` at the
mailbox. Therefore the coordinator can treat an **in-flight** fetch as a valid *promised*
source — it does not need the bytes to be *present*, only *promised*. This closes the
pure-lazy cache's failure mode (in a burst, no peer has the data *yet*): the mailbox is
the rendezvous. **No barrier, no cohort, no `world_size`.**

### 5.2 API

Ordinary `get` / `get_state_dict`, with an opt-in flag (or store-level selector):

```python
sd = await ts.get_state_dict(key, user_state_dict=sd, dedup=True)
```

Readers don't know about each other; they just read.

### 5.3 Coordinator state (per `(key, version)`)

```
sources[region]      : set of volume_ids that HAVE region        (present)
promises[region]     : list of (volume_id, ready_future)         (in-flight; will HAVE it)
waiters[region]      : queue of pending reader requests deferred until a promise resolves
version              : commit epoch (bumped on new MAPPING for key)
```

`version` is derived from the `.../MAPPING` commit marker (a new `put_state_dict` bumps
it and invalidates stale `sources`/`promises`).

### 5.4 Per-request algorithm (serial, so race-free)

**The 1×-fabric invariant:** once *any* peer holds **or promises** region `R`, `R` is
**never re-pulled from the trainer** — the trainer branch fires only for the very first
requester of `R`. Every later reader is routed to a present-or-promised peer; the fan-out
cap only chooses *which* peer (and may make a reader wait), it never redirects back to the
trainer.

```
on get_plan(reader r, needed_regions):          # runs to completion before next msg
  plan = []
  for R in needed_regions:
    if r already holds R: continue
    peers = sources[R] ∪ {v for (v,_) in promises[R]}   # present OR promised (exclude r)
    if peers nonempty:
      src = choose_under_cap(peers, r)          # S5 locality; prefer a source below cap,
                                                # else the least-loaded (consumer will queue)
      dep = promise_of(src) if src is a promise else None
      plan.append((R, src, dep))                # route to a PEER — never the trainer
    else:
      # first requester for R -> it pulls from the trainer AND becomes the cache source
      tvol = one_trainer_volume_holding(R)      # from Controller.locate_volumes
      fut  = new_future()
      promises[R].append((r.volume, fut))       # r will HAVE R soon
      plan.append((R, tvol, dep=None, publish=fut))   # r publishes R, resolving fut
  return plan
```

The reader executes `plan` with existing transport (S3/S5): entries with `dep=None` fetch
immediately; entries with a `dep` **wait** for that promise (§5.5). After fetching a
region it was designated to publish, the reader `put`s it into its own volume; the
resulting `notify_put_batch` reaches the coordinator and **resolves the future**
(`promises[R] → sources[R]`, wake `waiters[R]`).

### 5.5 The queuing / dependency mechanic (why it needs its own layer)

Two kinds of waiting, both owned by the coordinator (not the dumb directory — hence the
layer):

- **Data dependency (`dep`)** — a producer→consumer readiness handshake (the
  "done"/"consumed" signaling any one-sided transfer needs). Preferred implementation is
  **defer-reply:** the coordinator doesn't answer `req_j` for region `R` until `R`'s
  promise resolves; it parks `req_j` in `waiters[R]` and replies when the puller's
  `notify_put_batch(R)` arrives. The reader's `get` simply blocks in the coordinator
  call. (Alternative: reply with `(peer_volume, ready_token)` and let the reader poll —
  more client complexity, avoids holding actor state.)
- **Fan-out slot (cap)** — when the chosen source is already serving `FANOUT_CAP`
  consumers concurrently, the excess consumer **queues on that source's slot** and is
  released as a serve completes. It does **not** get redirected to the trainer (that
  would break the 1×-fabric invariant). A plan-time counter shapes the tree; an
  execution-time per-source slot queue enforces the cap.

### 5.6 Balancing / fan-out (avoid the single-source bottleneck)

If every later reader is routed to the *first* fetcher, that fetcher's NIC becomes the
bottleneck. Because requests are serialized, the coordinator builds a **fan-out tree
incrementally**: it prefers a present/promised source still below the cap, so once a
source is "full," later readers are routed to another peer that already has (or will have)
`R` — turning earlier consumers into secondary sources. This yields a balanced tree
without any global plan, and no source ever exceeds `FANOUT_CAP` concurrent serves.

> **Why not "re-pull from the trainer when the cache source is at cap"?** It looks
> simpler but breaks the 1×-fabric guarantee: a region that overlaps several readers gets
> pulled from the trainer more than once (measured: fabric = 16B vs. union = 8B, i.e. 2×,
> on the reshard case). Queue on the peer instead.

### 5.7 Cache lifecycle & coherence
- **Reuse existing volumes:** under `LocalRankStrategy` each generator rank already has a
  volume; "create a volume on `g`" is just publishing into it under the weight key.
- **No eviction within a version window:** once `R` is promised/present at version `v`, it
  stays until `v` is superseded — the coordinator knows the burst is in progress, so the
  cache won't evict mid-sync (it's routing selector + a pin, not an LRU).
- **Invalidation:** a new `put_state_dict` (new MAPPING) bumps `version`; the coordinator
  drops stale `sources`/`promises`. Optionally `delete`s stale cached slices.

### 5.8 Sequence (burst of 3 generators, region A needed by all)

```
t=0  req from g0 for A:  no peer -> g0 pulls A from trainer; promise(A -> g0)    (t->g, 1x)
t=1  req from g1 for A:  promise(A@g0) exists -> plan: get A from g0, WAIT dep0  (deferred)
t=2  req from g2 for A:  g0 at cap 1 -> plan: get A from g1 (promised), WAIT dep1  (tree!)
...  g0 fetches A, publishes -> notify resolves dep0 -> g1 unblocked, gets A from g0
...  g1 publishes A          -> notify resolves dep1 -> g2 unblocked, gets A from g1
 => trainer shipped A once; g0->g1->g2 is a balanced chain over NVLink/intra-node
```

(A discrete-event simulation of exactly this algorithm — including reshard and
version-bump scenarios — lives in `../dedup_sim/`; run `python -m dedup_sim`.)

### 5.9 Failure modes / costs
- **Puller failure:** a designated puller dies before publishing → its promise never
  resolves → dependents hang. Need a timeout that re-designates another reader (or falls
  back to trainer).
- **Greedy vs. global optimum:** online, incremental assignment can be slightly less
  balanced than a hypothetical global plan (mitigated by the fan-out cap).
- **Coordinator hot path:** all gets serialize through one actor; the per-request work is
  O(regions) directory ops — cheap, but a very large fan-in could back-pressure. Shard
  the coordinator by key-hash if needed.
- **Coherence bugs** if version bumping is wrong → serving stale weights. The MAPPING
  marker must gate reads (it already does in `get_state_dict`).

---

## 6. Recommendation & phasing

**Recommendation:** build the **shared substrate first**, then the coordinator. The
dynamic design fits TorchStore's decoupled, elastic, mediated nature: it needs no barrier,
handles spread/async demand as a plain cache, and (thanks to actor serialization) dedups a
synchronized burst just as well.

**Phasing (each independently shippable):**
1. **S1 (read-each-region-once)** — pure win, no new API; fixes the current DP over-fetch.
2. **S3 + S5 + S4** — peer-source routing, cost model, read-through population. Enables a
   plain lazy cache (helps async/spread demand immediately).
3. **`TransferCoordinator` layer** — promises/dependencies/fan-out for the sync burst.
4. **S2 (put-side de-replication)** — trims store footprint + put traffic.

## 7. Open questions
- Coordinator placement/sharding for very wide fan-in (one actor vs. key-hash sharded).
- Fan-out cap selector (fixed vs. bandwidth-aware) and topology awareness (does the
  coordinator know NVLink vs. cross-node? `strategy` has `volume_hostname` — enough for
  same-host detection; NVLink domain needs more).
- Whether read-through population should be automatic for every `get` or opt-in per key
  (memory cost of caching on readers).
- Interaction with `direct_rdma` (handles path) — likely a separate mode, not unified.
- Version/epoch source of truth (MAPPING marker) for multi-key state dicts written
  non-atomically.
</content>
