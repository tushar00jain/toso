# TorchStore

TorchStore is a distributed, async KV store for PyTorch
tensors built on [Monarch](https://github.com/meta-pytorch/monarch) actors. Its
headline use case is **weight sync between a trainer/learner and a generator in
RL**, including **resharding** the weights across two *different* device meshes.

---

## 0. Glossary

Every term the rest of the doc leans on, defined with a tiny concrete example.
The running example throughout is a single 1-D tensor **`W = [0 1 2 3 4 5 6 7]`**
(8 elements) that a trainer splits across GPUs and a generator wants back.

### Background PyTorch / distributed terms

- **Tensor** — an n-dimensional array (the weights). Our example `W` is a 1-D
  tensor of length 8.

- **Rank** — the integer id of one process in a distributed job. 4 processes →
  ranks `0,1,2,3`. Each rank usually drives one GPU.

- **World size** — total number of ranks in a job. 4 processes → world size 4.

- **Shard** — one contiguous *piece* of a tensor held by one rank. If `W` is split
  across 4 ranks, rank 1's shard is `[2 3]`.

- **Placement** — *how* a tensor is split across a mesh axis. The two that matter:
  - `Shard(d)` = cut along dimension `d` (e.g. `Shard(0)` on `W` gives each rank a
    different slice).
  - `Replicate()` = every rank holds a full copy (used for data parallelism).

- **Device mesh** — the logical grid of ranks a tensor is distributed over, given
  as a `mesh_shape`. `mesh_shape=(4,)` is a 1-D grid of 4 ranks; `mesh_shape=(2,2)`
  is a 2-D grid of 4 ranks.

- **Mesh coordinate (`coordinates`)** — a rank's position *in the mesh grid*. In a
  `(2,2)` mesh, rank 3 sits at coordinate `(1,1)`. This is how a shard is uniquely
  identified.

- **DTensor** ("distributed tensor") — PyTorch's object that bundles a local shard
  **plus** the metadata describing how it fits into the global tensor (mesh,
  placements, coordinate). `ts.put(key, dtensor)` stores just that rank's shard.

- **`state_dict`** — a plain `{parameter_name -> tensor}` dictionary describing a
  whole model (e.g. `{"layer0.weight": <tensor>, ...}`). What you sync in RL.

- **Global vs. local coordinates** — *global* = position within the full tensor
  `W`; *local* = position within one shard. Rank 1's shard `[2 3]` sits at **global
  offset 2**, but its own **local** indices are `0,1`. TorchStore always records
  slices in *global* coordinates — that's the trick that makes resharding work.

### TorchStore-specific terms

- **Store** — one named TorchStore instance, identified by `store_name` (default
  `"torchstore"`). You can run several independent stores in one job.

- **Key** — the string you store a value under, e.g. `"selector/layer0.weight"`.
  Like a dictionary key. State-dicts expand into many keys of the form
  `"<key>/<param_name>"`.

- **`TensorSlice`** — TorchStore's metadata record describing *one shard's
  geometry*, in **global** coordinates. Fields (all tuples, one entry per
  dimension):

  | field | meaning | example (rank 1 of a 4-way split of `W`) |
  |-------|---------|------------------------------------------|
  | `global_shape` | shape of the full tensor | `(8,)` |
  | `offsets` | where this shard starts in the global tensor | `(2,)` |
  | `local_shape` | shape of this shard | `(2,)` — i.e. `[2 3]` |
  | `coordinates` | this shard's mesh coordinate | `(1,)` |
  | `mesh_shape` | shape of the device mesh | `(4,)` |

  Read as: "this piece is `W[2:4]`, it's mesh-cell `(1,)` of a 4-cell mesh."

- **`Request`** — the internal envelope for one operation: a `key` plus optionally
  the tensor bytes (`tensor_val`), its `TensorSlice`, or an arbitrary `objects`
  payload. `meta_only()` produces a copy with the bytes stripped (`tensor_val=None`)
  — that stripped copy is what's sent to the control plane.

- **Storage Volume** — a **data-plane** actor that actually *holds bytes*. Think of
  it as one shard-server. Its `volume_id` is a string (e.g. `"0"`, `"1"`, or a
  hostname). Internally it's an `InMemoryStore`: a dict `key -> {coordinate ->
  {slice, tensor}}`. With `LocalRankStrategy` there's one volume per rank.

- **`volume_id` / storage ID** — the unique string naming a storage volume. With
  `LocalRankStrategy` it's the rank (`"0"`, `"1"`, …); with `HostStrategy` it's the
  hostname. The control plane's index maps each key to the set of `volume_id`s that
  hold its shards.

- **`StorageInfo`** — the **control-plane** metadata the controller keeps *per key,
  per volume*: an `object_type` (is this an `OBJECT`, a whole `TENSOR`, or a
  `TENSOR_SLICE`/shard?) plus the set of `TensorSlice`s that volume holds for that
  key. It records *what lives where* — never the bytes themselves. Example:
  `StorageInfo(object_type=TENSOR_SLICE, tensor_slices={TensorSlice(offsets=(2,), local_shape=(2,), ...)})`.

- **Controller** — the single **control-plane** actor. Holds the index
  `key -> {volume_id -> StorageInfo}` and answers "which volumes hold this key?"
  (`locate_volumes`). Metadata only; no tensor bytes ever pass through it.

- **Strategy** — the selector deciding *which volume a given client talks to*.
  `LocalRankStrategy` = one volume per rank; `HostStrategy` = one per host;
  `ControllerStorageVolumes` = a single shared volume (deprecated).

- **`LocalClient`** — the object living in *your* process that runs the actual
  put/get: it asks the controller where things are, moves bytes to/from volumes,
  and does the reshard math. You rarely touch it directly — the `ts.*` functions
  wrap it.

- **Transport / `TransportBuffer`** — the mechanism that physically moves bytes
  between your process and a volume (shared memory, RDMA, Gloo, or RPC), chosen
  automatically. The `TransportBuffer` is the handle for one such transfer.

- **Control plane vs. data plane** — *control plane* = the small metadata service
  that knows *where* data is (the Controller). *Data plane* = the components that
  hold and move the actual bytes (StorageVolumes + transports). TorchStore keeps
  them strictly separate so tensor bytes never bottleneck on the metadata actor.

- **Intersection** — on `get`, the overlap between a *stored* `TensorSlice` and the
  *requested* `TensorSlice`, computed in global coordinates. Only overlapping bytes
  are fetched. E.g. stored `W[0:2]` ∩ requested `W[0:4]` = `W[0:2]`.

- **Reshard** — reading a tensor back into a *different* split than it was written
  with (e.g. written 4-way, read 2-way). In TorchStore this isn't a separate call;
  it falls out of "store slices in global coords, then intersect + reassemble on
  `get`."

- **Commit / fully-committed** — a DTensor key is "committed" only once *every*
  shard (every mesh coordinate) has been written. The controller checks this before
  letting a reader see the key, so nobody reads a half-written tensor.

---

## 1. TL;DR

- **Control plane.** A single Monarch actor called
  `Controller` (`torchstore/controller.py`). It holds **only metadata** — a trie
  mapping `key -> {storage_volume_id -> StorageInfo}`. Tensor bytes never pass
  through it.
- **Data plane** = a set of `StorageVolume` actors (`torchstore/storage_volume.py`),
  each an in-memory KV store, one per rank/host depending on the *strategy*.
- **Resharding.** It is implicit and happens on `get`.
  Every `put` records the **`TensorSlice`** (offsets / shape / mesh
  coords) of the local shard, and every `get` describes the shard the *reader*
  wants. The client computes the **intersection** of stored slices with the
  requested slice, fetches only the overlapping bytes from the relevant volumes,
  and **reassembles** them into the reader's local shard. Because the store keeps
  slices in *global* coordinates, the put-side mesh and get-side mesh never have to
  match.
- **API contract**: module-level async functions in `torchstore/api.py`
  (`ts.put/get/put_batch/get_batch/put_state_dict/get_state_dict/delete/exists/keys`),
  all keyed by a `store_name`. State-dict sync (the RL path) is layered on top of
  `put`/`get` in `torchstore/state_dict_utils.py`.

---

## 2. Component map

```
                           ┌────────────────────────────────────────────┐
                           │            CONTROL PLANE                   │
                           │   Controller  (single Monarch actor)       │
   locate_volumes(keys) ─▶ │   keys_to_storage_volumes: Trie            │
   notify_put_batch(...) ─▶│     key -> { vol_id -> StorageInfo(        │
   get_controller_strategy │              object_type, {TensorSlice} ) }│
                           │   *** METADATA ONLY — no tensor bytes ***  │
                           └────────────────────────────────────────────┘
                              ▲  (1) locate     ▲ (3) notify (meta only)
                              │                 │
        ┌─────────────────────┴─────────────────┴──────────────────────┐
        │                      LocalClient (runs in caller's process)  │
        │  strategy.select_storage_volume() / get_storage_volume(id)   │
        │  create_transport_buffer(volume_ref)                         │
        └───────┬───────────────────────────────────────────┬──────────┘
                │ (2) put_to / get_from storage volume      │
                │     via TransportBuffer (bytes move here) │
                ▼                                           ▼
        ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
        │ StorageVolume │   │ StorageVolume │   │ StorageVolume │  ...  DATA PLANE
        │  vol_id="0"   │   │  vol_id="1"   │   │  vol_id="2"   │
        │ InMemoryStore │   │ InMemoryStore │   │ InMemoryStore │
        │  kv: key ->   │   │  kv: key ->   │   │  kv: key ->   │
        │   {coords:    │   │   {coords:    │   │   {coords:    │
        │    {slice,    │   │    {slice,    │   │    {slice,    │
        │     tensor}}  │   │     tensor}}  │   │     tensor}}  │
        └───────────────┘   └───────────────┘   └───────────────┘
```

| Piece | File | Role |
|-------|------|------|
| `Controller` | `controller.py` | **Control plane**: key→volume index, commit tracking, key listing. Metadata only. |
| `StorageVolume` / `InMemoryStore` | `storage_volume.py` | **Data plane**: actually holds tensor bytes / DTensor shards / objects. |
| `LocalClient` | `client.py` | Runs in the user's process; orchestrates put/get, does the resharding math. |
| `TorchStoreStrategy` | `strategy.py` | Maps a client process → which volume it writes to (`LocalRankStrategy`, `HostStrategy`, `ControllerStorageVolumes`). |
| `Request` / `TensorSlice` | `transport/types.py` | The wire contract: what to store/fetch + shard geometry. |
| `TransportBuffer` | `transport/*` | Moves the actual bytes (shared-mem / RDMA / Gloo / RPC). Auto-selected. |
| `state_dict_utils` | `state_dict_utils.py` | Layers `state_dict` put/get + direct-RDMA weight sync on top of the KV API. |
| module funcs | `api.py` | Public `ts.*` API surface. |

---

## 3. The control plane in detail (`Controller`)

The `Controller` is a Monarch actor obtained via
`get_or_spawn_controller(store_name, Controller)` — so it is a **named singleton
per store**. It never sees tensor data; the client explicitly strips data before
notifying it (`request.meta_only()`), and the controller *asserts* that
`tensor_val is None`:

```python
# controller.py  _notify_put
assert request.tensor_val is None, \
    "request should not contain tensor data, as this will significantly increase e2e latency"
```

Its state is one structure:

```
keys_to_storage_volumes : Trie
    "selector/layer0.weight" -> {
        "0": StorageInfo(object_type=TENSOR_SLICE, tensor_slices={TensorSlice(...)}),
        "1": StorageInfo(object_type=TENSOR_SLICE, tensor_slices={TensorSlice(...)}),
        ...
    }
```

Control-plane endpoints (all `@endpoint`, i.e. remote-callable):

| Endpoint | Purpose |
|----------|---------|
| `init(strategy, num_storage_volumes, storage_volumes)` | One-time setup; stores the strategy + volume handles. |
| `get_controller_strategy()` | Client fetches the strategy so it can address volumes directly. |
| `locate_volumes(keys, missing_ok, require_fully_committed)` | **The lookup that drives every get/delete.** Returns `{key -> {vol_id -> StorageInfo}}`. |
| `notify_put_batch(requests, vol_id)` | Client tells the controller "I wrote these (meta-only) requests to volume X". |
| `notify_delete` / `notify_delete_batch` | Idempotent index cleanup. |
| `keys(prefix)` | Prefix listing straight from the trie. |
| `teardown()` | Clears the index and resets volumes. |

### Commit semantics (important for correctness)

A DTensor is spread over many ranks, so a key isn't "ready" until **every** shard
has landed. `locate_volumes(..., require_fully_committed=True)` rejects a
partially-written DTensor by checking that the union of stored mesh coordinates
equals the full mesh:

```python
# controller.py  _is_dtensor_fully_committed
expected_coords = set(product(*(range(s) for s in mesh_shape)))
return all_slices == expected_coords     # every coordinate present?
```

For plain state-dicts, `state_dict_utils.put_state_dict` writes all tensors first
and then writes a `.../MAPPING` key **last**, so the mapping doubles as a commit
marker — `get_state_dict` reads the mapping first and fails cleanly if the writer
never finished.

---

## 4. The API contract

### 4.1 Public functions (`api.py`)

All are `async`, all take `store_name=DEFAULT_TORCHSTORE_NAME`.

```python
await ts.initialize(num_storage_volumes=1, strategy=None, store_name=..., mesh=None)
await ts.shutdown(store_name=...)

# KV core
await ts.put(key, value)                       # tensor | DTensor | any pickleable object
await ts.put_batch({key: value, ...})
val   = await ts.get(key, inplace_tensor=None, tensor_slice_spec=None)
vals  = await ts.get_batch([keys] | {key: inplace_tensor})   # all-or-nothing
await ts.delete(key)  /  ts.delete_batch(keys)
ok    = await ts.exists(key)
keys  = await ts.keys(prefix)                  # prefix search

# state_dict / weight sync (built on the above)
await ts.put_state_dict(state_dict, key, direct_rdma=False, transfer_dtype=None)
sd = await ts.get_state_dict(key, user_state_dict=None, strict=True, direct_rdma=False)
```

Contract highlights, straight from the code:

- **`put(key, DTensor)` stores only the caller's *local shard*** plus its
  `TensorSlice` (global offsets + mesh coords). It does **not** gather the full
  tensor. (`Request.from_dtensor`)
- A DTensor whose placements are all `Replicate()` or whose mesh has size 1 is
  treated as a **plain tensor** (no shard bookkeeping) — see
  `_is_dtensor_fully_local`.
- **`get(key, dtensor)`** uses the passed DTensor's sharding to fetch *only the
  slice that rank needs*. `get(key)` with no arg reassembles and returns the
  **full** tensor.
- `get_batch` is **all-or-nothing**: any missing key raises `KeyError`, no partial
  results.
- `tensor_slice_spec` lets you fetch an arbitrary rectangular region of a stored
  tensor without a DTensor at all.

### 4.2 The wire types (`transport/types.py`)

```python
@dataclass
class TensorSlice:            # geometry of one shard, in GLOBAL coordinates
    offsets:      tuple       # where this slice starts in the full tensor
    coordinates:  tuple       # device-mesh coordinate of this shard
    global_shape: tuple       # shape of the full (unsharded) tensor
    local_shape:  tuple       # shape of THIS slice
    mesh_shape:   tuple       # shape of the device mesh

@dataclass
class Request:               # one KV op
    key:          str
    tensor_val:   Tensor|None    # the actual bytes (stripped to None for control plane)
    tensor_slice: TensorSlice|None
    objects:      Any|None
    is_object:    bool
```

`TensorSlice` is the **single most important contract** for resharding: because
offsets and shapes are always expressed relative to the *global* tensor, two
processes that sharded the same tensor with completely different meshes still
describe overlapping regions in the same coordinate system.

---

## 5. How resharding actually works

There is **no explicit "reshard" call**. Resharding is an emergent property of
"store slices in global coords, then intersect on read." Three phases:

### Phase A — PUT (writer records its shard)

Each writer rank calls `ts.put(key, dtensor)`:

1. `Request.from_dtensor` computes this rank's `TensorSlice` via
   `_compute_local_shape_and_global_offset(...)` and attaches `dtensor._local_tensor`.
2. `strategy.select_storage_volume()` picks this rank's volume; the bytes are
   pushed with a `TransportBuffer`.
3. The volume's `InMemoryStore._handle_dtensor` stores it keyed by mesh
   coordinate: `kv[key][coords] = {"slice": TensorSlice, "tensor": shard}`.
4. The client notifies the controller **meta-only** (`notify_put_batch`).

### Phase B — LOCATE (control plane)

`get` calls `controller.locate_volumes([key])` → `{key -> {vol_id -> StorageInfo}}`,
i.e. "which volumes hold slices of this key, and what slices."

### Phase C — GET (reader pulls + reassembles its shard)

In `LocalClient._fetch` / `_expand_tensor_slices` / `_assemble_results`:

1. For the reader's requested `TensorSlice`, walk every stored slice and compute
   `get_slice_intersection(stored, requested)` (`utils.py`). Skip non-overlapping
   shards entirely — only overlapping bytes are transferred.
2. Fetch each intersecting sub-slice from its volume, **in parallel across volumes**
   (`asyncio.gather`).
3. Reassemble the pieces into the reader's local shard with `assemble_tensor`
   (computes the bounding box of the fetched pieces and copies each into place).
4. If the reader passed an in-place tensor and the transport supports it, bytes are
   written **directly** into a contiguous destination view (`get_destination_view`),
   skipping the reassembly allocation.

### ASCII: 1-D reshard, `put` on a 4-rank mesh → `get` on a 2-rank mesh

Global tensor `[0 1 2 3 4 5 6 7]`, `Shard(0)`.

```
PUT world (mesh_shape=(4,))          Stored in volumes, keyed by coords, GLOBAL offsets:
 rankP0: local=[0 1]  offset=0  ─┐
 rankP1: local=[2 3]  offset=2  ─┤   vol0: key -> {(0,): slice(off=0,len=2) tensor=[0 1]}
 rankP2: local=[4 5]  offset=4  ─┤   vol1: key -> {(1,): slice(off=2,len=2) tensor=[2 3]}
 rankP3: local=[6 7]  offset=6  ─┘   vol2: key -> {(2,): slice(off=4,len=2) tensor=[4 5]}
                                     vol3: key -> {(3,): slice(off=6,len=2) tensor=[6 7]}

GET world (mesh_shape=(2,))  wants Shard(0) over 2 ranks:
 rankG0 wants global[0..4) = [0 1 2 3]      rankG1 wants global[4..8) = [4 5 6 7]

 rankG0 requested slice: off=0 len=4
   ∩ vol0 stored off=0 len=2  -> [0 1]   (intersection off=0 len=2)
   ∩ vol1 stored off=2 len=2  -> [2 3]   (intersection off=2 len=2)
   ∩ vol2/vol3               -> no overlap, skipped
   assemble([0 1] @0, [2 3] @2)  ->  [0 1 2 3]   ✓

 rankG1 requested slice: off=4 len=4
   ∩ vol2 -> [4 5] @4 ,  ∩ vol3 -> [6 7] @6
   assemble  ->  [4 5 6 7]   ✓
```

The same machinery covers **grow** (2→4 ranks), **2-D→1-D**, **1-D→2-D**, and
**mixed Replicate/Shard** — see `tests/test_resharding_basic.py`, which
parametrizes exactly these cases.

### Worked example: a 2-D DTensor on a `(2,2)` mesh

Take a **4×4** tensor `W` and a mesh of 4 ranks arranged as `mesh_shape=(2,2)`:

```
Full tensor W (global_shape=(4,4))        Device mesh (2,2), rank at each cell:
 ┌────┬────┬────┬────┐                     ┌──────────┬──────────┐
 │  0 │  1 │  2 │  3 │                     │ rank0    │ rank1    │   mesh axis 1 →
 ├────┼────┼────┼────┤                     │ (0,0)    │ (0,1)    │
 │  4 │  5 │  6 │  7 │                     ├──────────┼──────────┤
 ├────┼────┼────┼────┤                     │ rank2    │ rank3    │   mesh
 │  8 │  9 │ 10 │ 11 │                     │ (1,0)    │ (1,1)    │   axis 0
 ├────┼────┼────┼────┤                     └──────────┴──────────┘     ↓
 │ 12 │ 13 │ 14 │ 15 │
 └────┴────┴────┴────┘
```

#### Case 1 — `placements=[Shard(0), Shard(1)]` (2-D sharding, the classic case)

"mesh axis 0 cuts tensor **rows**, mesh axis 1 cuts tensor **columns**." Each rank
ends up with a **2×2 block**:

```
 coord (0,0) rows0-1,cols0-1     coord (0,1) rows0-1,cols2-3
   ┌────┬────┐                     ┌────┬────┐
   │  0 │  1 │                     │  2 │  3 │
   ├────┼────┤                     ├────┼────┤
   │  4 │  5 │                     │  6 │  7 │
   └────┴────┘                     └────┴────┘

 coord (1,0) rows2-3,cols0-1     coord (1,1) rows2-3,cols2-3
   ┌────┬────┐                     ┌────┬────┐
   │  8 │  9 │                     │ 10 │ 11 │
   ├────┼────┤                     ├────┼────┤
   │ 12 │ 13 │                     │ 14 │ 15 │
   └────┴────┘                     └────┴────┘
```

The `TensorSlice` TorchStore records for each rank (offsets are `(row, col)` in the
**global** tensor):

| rank | `coordinates` | `offsets` | `local_shape` | `global_shape` | `mesh_shape` |
|------|---------------|-----------|---------------|----------------|--------------|
| 0 | `(0,0)` | `(0,0)` | `(2,2)` | `(4,4)` | `(2,2)` |
| 1 | `(0,1)` | `(0,2)` | `(2,2)` | `(4,4)` | `(2,2)` |
| 2 | `(1,0)` | `(2,0)` | `(2,2)` | `(4,4)` | `(2,2)` |
| 3 | `(1,1)` | `(2,2)` | `(2,2)` | `(4,4)` | `(2,2)` |

So with `LocalRankStrategy` the store holds (control-plane view):

```
Controller index for key "W":
  { "0": StorageInfo(TENSOR_SLICE, {slice off=(0,0) shape=(2,2)}),
    "1": StorageInfo(TENSOR_SLICE, {slice off=(0,2) shape=(2,2)}),
    "2": StorageInfo(TENSOR_SLICE, {slice off=(2,0) shape=(2,2)}),
    "3": StorageInfo(TENSOR_SLICE, {slice off=(2,2) shape=(2,2)}) }
```

Commit check: `mesh_shape=(2,2)` ⇒ expected coords
`{(0,0),(0,1),(1,0),(1,1)}`. Only once all four have been `put` is `"W"` readable.

#### Case 2 — `placements=[Shard(0), Replicate()]` (FSDP-style: shard rows, replicate cols)

"mesh axis 0 cuts **rows** into two halves; mesh axis 1 **replicates**." Now each
rank holds a full-width **2×4** row-block, and the two ranks in the same mesh row
hold **identical** data:

```
 coord (0,0) AND (0,1)  →  rows 0-1, all cols     coord (1,0) AND (1,1)  →  rows 2-3, all cols
   ┌────┬────┬────┬────┐                            ┌────┬────┬────┬────┐
   │  0 │  1 │  2 │  3 │                            │  8 │  9 │ 10 │ 11 │
   ├────┼────┼────┼────┤                            ├────┼────┼────┼────┤
   │  4 │  5 │  6 │  7 │                            │ 12 │ 13 │ 14 │ 15 │
   └────┴────┴────┴────┘                            └────┴────┴────┴────┘
```

| rank | `coordinates` | `offsets` | `local_shape` | note |
|------|---------------|-----------|---------------|------|
| 0 | `(0,0)` | `(0,0)` | `(2,4)` | rows 0-1 |
| 1 | `(0,1)` | `(0,0)` | `(2,4)` | **same bytes as rank 0** (replicated on axis 1) |
| 2 | `(1,0)` | `(2,0)` | `(2,4)` | rows 2-3 |
| 3 | `(1,1)` | `(2,0)` | `(2,4)` | **same bytes as rank 2** |

Note ranks 0 and 1 have **identical `offsets`/`local_shape`** — that's what
"replicated along a mesh axis" looks like in slice-space (distinguished only by
their `coordinates`). This is the case where `get` over-fetches replicated copies —
the DP inefficiency flagged in `client._expand_tensor_slices`.

> If **all** placements were `Replicate()` (or the mesh had size 1), TorchStore
> skips the shard bookkeeping entirely and stores `W` as a plain tensor
> (`_is_dtensor_fully_local`).

#### Reading it back into a different layout

A generator that constructs its DTensor with, say, `mesh_shape=(4,)` and
`placements=[Shard(1)]` (1-D, column-sharded over 4 ranks) just calls
`ts.get("W", its_dtensor)`. For its rank 2 (wanting global columns `[2:3]`, all
rows → `offsets=(0,2)`, `local_shape=(4,1)`), the client intersects that request
against the stored 2×2 blocks from Case 1:

```
 wanted off=(0,2) shape=(4,1)   (column 2, all rows)
   ∩ (0,1) off=(0,2) shape=(2,2) -> rows0-1,col2 = [2, 6]      @ (0,2)
   ∩ (1,1) off=(2,2) shape=(2,2) -> rows2-3,col2 = [10, 14]    @ (2,2)
   ∩ (0,0),(1,0)                 -> no column overlap, skipped
   assemble -> [2, 6, 10, 14]  (a 4×1 column)   ✓
```

No agreement between the `(2,2)`-mesh writer and the `(4,)`-mesh reader was
needed — global-coordinate slices make the reshard fall out automatically.

### The intersection math (the heart of it)

```python
# utils.get_slice_intersection  (per dimension)
intersect_start = max(stored_start, requested_start)
intersect_end   = min(stored_end,   requested_end)
if intersect_start >= intersect_end:
    return None                       # no overlap -> skip this shard
# result keeps GLOBAL offsets, new local_shape = intersect_end - intersect_start
```

---

## 6. The RL weight-sync path (learner → generator)

This is the motivating use case. `state_dict_utils` builds it on the KV API.

```
        LEARNER (trainer)                         GENERATOR (inference/rollout)
   ┌──────────────────────────┐             ┌──────────────────────────────┐
   │ optim.step()             │             │ serving_model                │
   │ sd = model.state_dict()  │             │ sd = model.state_dict()      │
   │ ts.put_state_dict(sd,"v")│             │ ts.get_state_dict("v",       │
   └───────────┬──────────────┘             │        user_state_dict=sd)   │
               │                            └───────────────┬──────────────┘
   flatten sd → put each param (put_batch)                  │ read ".../MAPPING" first
   → put ".../MAPPING" LAST  (commit marker)                │ (fails if writer unfinished)
               │                                            │ get_batch(all param keys,
               ▼                                            │           inplace=sd tensors)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                      TorchStore (Controller + StorageVolumes)            │
   │   resharding happens transparently if learner mesh ≠ generator mesh      │
   └──────────────────────────────────────────────────────────────────────────┘
```

Key facts:

- `put_state_dict` **flattens** the state dict, `put_batch`es all tensors, then
  writes the `MAPPING` key **last** as an atomic commit marker.
- `get_state_dict` reads the mapping first (guaranteeing completeness), then
  `get_batch`es every param — passing the generator's own tensors as **in-place**
  destinations, so if the generator is sharded differently the reshard-on-get
  logic fills each local shard correctly.
- `transfer_dtype` lets the learner keep fp32 master weights but ship bf16.
- **Direct RDMA** (`direct_rdma=True`, `direct_weight_sync.py`): instead of copying
  bytes into a StorageVolume, the learner registers **RDMA handles** to its GPU
  memory and stores only the handles as objects (`key/rank_{r}`, `key/num_ranks`).
  The generator fetches the handles and does **one-sided RDMA reads straight from
  the learner's GPU** — zero intermediate storage, for synchronous RL. Note this
  path stores raw handles per rank and does **not** go through the
  `TensorSlice`-intersection resharding path.

The shipped example is `example/torchstore_rl.py` (a `Learner` actor calling
`ts.put_state_dict` after each `optim.step()`, and a `Generator` actor calling
`ts.get_state_dict` to refresh its weights).

---

## 7. Strategies (who writes where)

`ts.initialize(num_storage_volumes=N, strategy=...)` picks the volume topology:

- **`LocalRankStrategy`** — one volume per rank (`volume_id = str(rank)`); client
  picks its volume from `RANK`/`LOCAL_RANK`. Default for multi-volume / RL.
- **`HostStrategy`** — one volume per host (keyed by `HOSTNAME`).
- **`ControllerStorageVolumes`** — single volume co-located on the controller
  (default when `num_storage_volumes==1`; now **deprecated** — it's a bottleneck).

Transport is auto-selected per transfer (README priority): shared-mem (same host)
→ Monarch RDMA → TorchComms RDMA → Gloo → Monarch RPC. Force with
`strategy(default_transport_type=TransportType.Gloo)`.

---

## 8. Toy code examples

### 8.1 Faithful TorchStore snippet (needs Monarch runtime + torchstore installed)

`toy_reshard.py` — put a tensor sharded 4 ways, get it sharded 2 ways:

```python
import asyncio, math, os, tempfile
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed._tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torchstore.utils import spawn_actors


class ShardActor(Actor):
    KEY = "weights"

    def __init__(self, mesh_shape, tensor, placements, store_file):
        self.rank = current_rank().rank
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.tensor = tensor
        self.placements = placements
        self.store_file = store_file
        os.environ["LOCAL_RANK"] = str(self.rank)   # LocalRankStrategy reads this

    def _pg(self):
        torch.distributed.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size,
            init_method=f"file://{self.store_file}",
        )

    @endpoint
    async def do_put(self):
        self._pg()
        mesh = init_device_mesh("cpu", self.mesh_shape)
        dt = distribute_tensor(self.tensor, mesh, placements=self.placements)
        await ts.put(self.KEY, dt)          # stores only THIS rank's shard + TensorSlice

    @endpoint
    async def do_get(self):
        self._pg()
        mesh = init_device_mesh("cpu", self.mesh_shape)
        dt = distribute_tensor(torch.zeros(8), mesh, placements=self.placements)
        out = await ts.get(self.KEY, dt)    # reshard-on-get fills this rank's shard
        return out.full_tensor()

    @endpoint
    async def stop_pg(self):
        torch.distributed.destroy_process_group()


async def main():
    original = torch.arange(8, dtype=torch.float32)          # [0..7]
    await ts.initialize(num_storage_volumes=4, strategy=ts.LocalRankStrategy())
    with tempfile.TemporaryDirectory() as d:
        put = await spawn_actors(4, ShardActor, "put",
                                 mesh_shape=(4,), tensor=original,
                                 placements=[Shard(0)],
                                 store_file=f"{d}/put")
        await put.do_put.call()                              # 4-way shard write

        get = await spawn_actors(2, ShardActor, "get",
                                 mesh_shape=(2,), tensor=torch.zeros(8),
                                 placements=[Shard(0)],
                                 store_file=f"{d}/get")
        results = await get.do_get.call()                    # 2-way shard read (reshard!)
        for _, full in results:
            assert torch.equal(full, original)
        await put.stop_pg.call(); await get.stop_pg.call()
    await ts.shutdown()
    print("resharded 4-way -> 2-way OK")


asyncio.run(main())
```

### 8.2 Self-contained simulation of the reshard *algorithm* (no deps, runs anywhere)

This reproduces the exact intersect-then-assemble logic TorchStore uses on `get`,
so you can see the control-plane index + data-plane fetch without Monarch:

```python
# toy_reshard_sim.py  — pure python, mirrors client._fetch + utils.get_slice_intersection
from dataclasses import dataclass

@dataclass(frozen=True)
class Slice:            # 1-D for clarity; TorchStore generalizes per-dimension
    offset: int
    length: int

def intersect(stored: Slice, wanted: Slice):
    lo = max(stored.offset, wanted.offset)
    hi = min(stored.offset + stored.length, wanted.offset + wanted.length)
    if lo >= hi:
        return None                       # no overlap -> skip (utils.get_slice_intersection)
    return Slice(lo, hi - lo)

# ---- DATA PLANE: volumes hold (slice, data) keyed by mesh coord ----
volumes = {
    "vol0": {(0,): (Slice(0, 2), [0, 1])},
    "vol1": {(1,): (Slice(2, 2), [2, 3])},
    "vol2": {(2,): (Slice(4, 2), [4, 5])},
    "vol3": {(3,): (Slice(6, 2), [6, 7])},
}
# ---- CONTROL PLANE: key -> {vol_id -> stored slices}  (Controller.keys_to_storage_volumes) ----
index = {"weights": {v: [s for s, _ in shards.values()] for v, shards in volumes.items()}}

def fetch_shard(key, wanted: Slice):
    """What LocalClient does on get(key, dtensor) for one reader rank."""
    pieces = []                                            # (global_offset, data)
    for vol_id, stored_slices in index[key].items():       # (Phase B) locate_volumes
        for stored in stored_slices:
            hit = intersect(stored, wanted)               # (Phase C) intersection
            if hit is None:
                continue
            # pull only the overlapping bytes from the volume (data plane)
            for s, data in volumes[vol_id].values():
                if s == stored:
                    start = hit.offset - s.offset
                    pieces.append((hit.offset, data[start:start + hit.length]))
    # assemble into contiguous local shard (utils.assemble_tensor)
    pieces.sort()
    base = pieces[0][0]
    out = [None] * wanted.length
    for off, data in pieces:
        out[off - base: off - base + len(data)] = data
    return out

# reader mesh = 2 ranks over the 8-element tensor, Shard(0)
print("rankG0:", fetch_shard("weights", Slice(0, 4)))   # -> [0, 1, 2, 3]
print("rankG1:", fetch_shard("weights", Slice(4, 4)))   # -> [4, 5, 6, 7]
```

Running `8.2` prints:

```
rankG0: [0, 1, 2, 3]
rankG1: [4, 5, 6, 7]
```

which is the 4-way→2-way reshard, purely from stored global-coordinate slices —
exactly what the real store does, minus the actor/transport plumbing.

---

## 9. How application-aware is TorchStore? (parallelism / sharding scheme)

Short answer: **deliberately almost blind.** TorchStore knows tensor *geometry*,
not application *semantics*. It has **no concept of TP / FSDP / PP / DP / EP**, no
model, no optimizer, no parallelism plan. The design bet is exactly this: by
lowering everything to *global-coordinate rectangles*, resharding works between any
two layouts **without either side ever declaring its strategy**.

### What it knows vs. what it throws away

The DTensor's rich sharding info is consumed **once, at the client boundary**
(`Request.from_dtensor`) and immediately lowered to plain numbers:

```python
# transport/types.py  Request.from_dtensor
coordinates = dtensor.device_mesh.get_coordinate()
_, offsets  = _compute_local_shape_and_global_offset(          # placements + mesh
    dtensor.shape, mesh_shape=..., my_coordinate=..., placements=dtensor.placements)
tensor_slice = TensorSlice(offsets, coordinates,
                           dtensor.shape, dtensor._local_tensor.shape,
                           dtensor.device_mesh.shape)
```

After this line, **`placements` are gone.** `Shard(0)` vs `Shard(1)` vs
`Replicate()` all collapse into "a rectangle at these `offsets` with this
`local_shape`." The store cannot tell you *why* a tensor was cut that way.

| Property | Stored / used? | Interpreted semantically? |
|----------|----------------|---------------------------|
| Global shape, per-shard offsets, local shape | ✅ stored in `TensorSlice` | geometry only |
| Mesh **shape** + shard **coordinates** | ✅ stored | only to check *completeness* (see below) — not to infer TP/DP |
| `placements` (`Shard(dim)`, `Replicate`) | ❌ discarded after lowering | no |
| Which mesh axis is TP vs DP vs PP | ❌ never conveyed | no |
| Model architecture / layer types / FQNs | ❌ (keys are opaque strings) | no |
| dtype / device of the shard | ✅ carried with the bytes | no selector attached |

### The two places it *does* peek at mesh structure

1. **Commit completeness.** The controller stores `mesh_shape` + `coordinates`
   purely to verify *all* shards arrived — it enumerates the full coordinate grid
   and checks the set is complete. It's counting shards, not understanding
   parallelism:

   ```python
   # controller._is_dtensor_fully_committed
   expected_coords = set(product(*(range(s) for s in mesh_shape)))
   return all_slices == expected_coords
   ```

2. **A "fully-local" shortcut.** If placements are all `Replicate()` or the mesh has
   one device, the DTensor is stored as a plain tensor. This is the *one* spot with
   a hint of application-awareness, and it's an explicit accommodation for a
   *torchtitan* quirk, flagged in a comment:

   ```python
   # transport/types.py  _is_dtensor_fully_local  (paraphrased comment)
   # torchtitan uses Replicate() that isn't actually replicated along the mesh;
   # treat all-Replicate / size-1 mesh as a regular tensor.  TODO: revisit if fixed upstream.
   ```

### Where the blindness costs something

Because it doesn't model **data parallelism / replication semantics**, the reader
can't tell a replicated copy from a distinct shard and re-fetches redundant
Replicate copies — called out as a TODO:

```python
# client._expand_tensor_slices
# TODO: ... This is extra inefficient in the case of DP, where we fetch all
# Replicate shards unnecessarily
```

### Net

- **Application → store:** the app injects geometry (via DTensor or an explicit
  `TensorSlice`/`tensor_slice_spec`). That's the whole contract.
- **Store → application:** the store hands back the exact region requested; the
  *application* is responsible for knowing its own parallelism and asking for the
  right slice (the generator constructs a DTensor with *its* mesh/placements and
  the store fills it).
- The store is a **geometry-addressed blob store**, not a parallelism-aware
  checkpoint engine. That is precisely what lets a TP-sharded learner feed an
  FSDP-sharded generator with no handshake about strategies — but it also means the
  store can't optimize for or validate a specific parallelism scheme on your behalf.
</content>
</invoke>
