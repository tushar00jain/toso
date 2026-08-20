# Indexed TensorSlice reshard planning

**Status:** proposal · **Scope:** TorchStore directory lookup, DTensor reshard
planning, and TOSO source binding.

TorchStore should index logical tensor regions separately from the volumes that hold
them. A read can then find the few regions that overlap its requested
`TensorSlice`, reuse the resulting reshard geometry across model keys and requests,
and choose a live or pending source only when the read is routed.

The key trie remains useful. It finds an FQN without scanning unrelated keys. The
scaling problem is the unindexed value behind each key:

```text
Trie[key] -> volume -> StorageInfo[tensor_slices]
```

For `K` requested keys and `H` recorded holders per key, the controller validates
`K·H` entries and the client examines up to `K·H` entries again. The proposed
directory changes the inner lookup from physical holders to indexed logical regions.

See [`torchstore.md`](torchstore.md) for the existing store path,
[`torchstore_dedup_design.md`](torchstore_dedup_design.md) for pending publications
and source ranking, and [`dedup_scaling.md`](dedup_scaling.md) for the synchronized
burst envelope.

## 1. Boundary

The design must:

- find only stored regions that overlap a requested `TensorSlice`;
- collapse volumes holding the same region into one replica class;
- build indexes incrementally from ordinary put, delete, declare, and publish events;
- cache intersection geometry without freezing source availability or routing load;
- give the controller and client one common fetch plan;
- preserve objects, whole tensors, DTensor completeness checks, preferred sources,
  in-place resharding, and pending-publication gates;
- retain deterministic source ordering and output parity with the current planner.

It does not define tensor transforms across keys, transposes, dtype conversion,
collective redistribution, or a new storage transport.

## 2. Why the current lookup is `K·H`

A generator `get_batch` carries one requested `TensorSlice` per key. The controller
receives only the keys, returns every recorded holder, and scans each holder to check
DTensor completeness. The client then tests every returned stored slice against the
requested slice.

For a 6,400-element tensor:

```text
trainer layout:   8 shards, 800 elements each
generator layout: 64 shards, 100 elements each
generator rank 19 requests [1900:2000]
```

The request overlaps trainer region `[1600:2400]` and generator region
`[1900:2000]`. A directory containing all 72 physical holders still examines 72
entries to discover those two logical regions. At `K = 723`, one pass visits 52,056
`(key, holder)` entries.

The work is necessary only when all 72 regions contribute bytes. When two regions
cover the request, the other 70 intersection checks are discovery overhead.

## 3. Logical regions and physical sources

The directory should give geometry and residency different identities.

```python
RegionKey = (
    tensor_slice.global_shape,
    tensor_slice.offsets,
    tensor_slice.local_shape,
)
```

`coordinates` and `mesh_shape` describe the DTensor placement that produced a slice.
They do not change the global rectangle. Volumes with the same `RegionKey` are
alternative physical sources for one logical region.

```text
Trie[key] -> KeyEntry

KeyEntry
  object_type
  global_shape
  geometry_epoch
  region_index
  layouts

RegionEntry
  region: RegionKey
  live_volumes: set[volume_id]
  pending: map[publication_id, volume_id]
```

Objects and whole tensors use one whole-value region and need no spatial query.
DTensor slices enter `region_index`, which supports an overlap query. `layouts`
tracks completeness independently for each placement layout.

Separating region from source also prevents replicated slices from multiplying the
geometry work. Source ranking sees the replica set after the region query identifies
what bytes are needed.

## 4. Dynamic index maintenance

`notify_put_batch` already supplies every field required for an exact region index.
For each request it:

1. derives the `RegionKey`;
2. inserts a new logical region into the spatial index when absent;
3. adds the volume or publication to the region's source set;
4. updates the completeness state for the slice's layout.

Delete and publication retirement remove a source. The region leaves the index only
when no live or pending source names it.

Two versions keep cache invalidation narrow:

```text
geometry_epoch
  changes when the set or geometry of logical regions changes

availability
  changes when sources appear, land, disappear, or change load
```

Adding another replica of an existing region changes availability but not geometry.
Cached reshard plans therefore survive replica churn and pending-to-live transitions.

New regions may be inserted into a balanced interval index in `O(log S)`, where `S`
is the number of distinct regions for the key. Adding a source to an existing region
is an `O(1)` set update.

## 5. Region lookup

The controller needs the requested meta-only `Request` values, not only their keys.
A slice-aware endpoint can coexist with `locate_volumes`:

```python
locate_slices(requests, prefer=None) -> SlicePlan
```

For each requested slice, the index chooses one dimension for pruning. The dimension
with the largest extent is a useful default. Stored regions are ordered by their
start in that dimension. A query discards regions ending before the request and stops
before regions starting at or beyond the request end. Only the active regions receive
the full multidimensional intersection test.

This is the sweep-line structure used by PyTorch Distributed Checkpoint planning:

```text
stored regions sorted by start
requested regions sorted by start
active stored regions ordered by end
```

Sorting on every TorchStore request would turn a linear scan into extra work. The
controller must either retain the ordered index or reuse one batch plan across many
keys. With a persistent index, a query costs approximately:

```text
O(log S + A·d)
```

`A` is the candidate count surviving the sweep dimension and `d` is tensor rank. The
result contains `C <= A` overlapping logical regions.

For regular DTensor layouts, a placement-aware owner calculation can be cheaper than
an interval query. `TensorSlice` is sufficient for exact indexing; explicit DTensor
placements such as `Shard(dim)` and `Replicate()` are needed for reliable layout
templates and direct coordinate-to-region formulas.

## 6. Cached reshard geometry

The first query between source geometry and a requested region produces immutable
intersection offers:

```text
PlanOffer
  source_region
  requested_region
  storage_offsets
  destination_offsets
  lengths
```

For the generator rank 19 example:

```text
requested [1900:2000]

offer from trainer region [1600:2400]
  read [1900:2000]
  write destination [0:100]

offer from generator region [1900:2000]
  read [1900:2000]
  write destination [0:100]
```

The offers are alternatives because either one covers the requested region. A
request spanning disjoint source shards receives several offers whose union covers
the destination.

An exact per-key cache key is:

```text
(key, geometry_epoch, requested RegionKey)
```

This removes repeated geometry work for recurring reads. A layout template removes
the key multiplier as well:

```text
(source layout ids, target layout id, target coordinate, shape class)
```

The template describes coordinate relationships and is instantiated with each
key's global shape. Uneven shards require the exact boundary table for their shape
class.

The cache stores geometry only. It never stores the winning volume, arrival time, or
load score.

## 7. Binding a plan to sources

Routing binds each required logical region to a current source:

```text
cached intersection offers
        |
        v
live volumes + pending publications
        |
        v
readiness, transfer time, load, stable id
        |
        v
resolved fetch plan
```

TOSO ranks the physical sources attached to each offer and chooses a cover of the
requested destination. A pending choice contributes its publication to the readiness
gate. A live choice can execute immediately.

The resolved plan names:

- the chosen volume and publication, if any;
- the exact stored intersection;
- the destination offset and length;
- the key and object type.

The client converts those entries into transport requests and destination views. It
does not call `get_slice_intersection` or rediscover coverage. Transport capability
and destination contiguity remain client concerns.

One resolved plan therefore defines both the control-plane gate and the data-plane
fetch. They cannot disagree about which publications or regions are required.

## 8. Complexity

Let:

- `K` be requested keys;
- `H` be physical holders recorded per key;
- `S` be distinct logical regions per key;
- `A` be regions surviving the sweep-dimension filter;
- `C` be regions in the selected cover;
- `d` be tensor rank.

| Operation | Current path | Indexed plan |
| --- | ---: | ---: |
| Key lookup | `O(K)` | `O(K)` |
| Completeness check | `O(K·H)` | `O(K)` from incremental layout state |
| Region discovery | Client `O(K·H·d)` | `O(K·(log S + A·d))` |
| Replica discovery | Included in `H` scan | `O(C)` region lookups plus source sets |
| Geometry on cache hit | `O(K·H·d)` | `O(K·C)` plan instantiation |
| Source ranking | None on main; TOSO adds it | Output-sensitive in eligible sources |
| Metadata response | `O(K·H)` | `O(K·C)` resolved entries |

A template shared by keys with one placement pattern reduces the common geometry
path toward:

```text
first layout pair: O(S log S + K·C)
cache hit:         O(K·C)
```

`K·C` is an output lower bound when every selected region produces one fetch entry.
A full-span request requiring all `H` distinct regions therefore remains `O(K·H)`.
The index improves sparse-overlap and replica-heavy requests, where `C << H`.

## 9. Correctness invariants

- A `RegionKey` names one immutable global rectangle.
- Every live or pending source belongs to exactly one region entry per stored slice.
- A resolved plan covers the requested destination without gaps.
- Replica alternatives do not cause the same destination region to be fetched twice.
- A plan is used only while its `geometry_epoch` matches the directory.
- Availability changes are checked while binding, so a cached plan cannot name a
  departed source.
- Pending sources enter the gate selected by the same plan the client executes.
- Stable region and volume ordering preserves deterministic ties.
- Incremental completeness state is equivalent to a full coordinate scan.

The implementation should assert destination coverage in tests and validate cached
plans against the existing exhaustive planner during rollout.

## 10. Delivery order

1. Introduce `RegionKey` and group identical regions into replica sets while keeping
   the existing `locate_volumes` response.
2. Maintain per-layout DTensor completeness incrementally on put and delete.
3. Add a persistent interval index and `locate_slices` using meta-only requests.
4. Define `PlanOffer`, `SlicePlan`, and the exact per-key geometry cache.
5. Make TOSO gate and rank sources from `SlicePlan`.
6. Make the client execute the resolved plan without recomputing intersections.
7. Add placement-aware layout templates shared across state-dict keys and ranks.

Each step retains the exhaustive planner as a parity oracle until randomized slice
tests and scale benchmarks cover the new path.

## 11. Verification

Correctness tests should cover:

- exact, partial, disjoint, nested, and multidimensional rectangles;
- uneven shards and replicated placements;
- two layouts contributing alternatives for the same destination region;
- one destination assembled from several disjoint regions;
- region insertion, replica addition, deletion, and pending retirement;
- geometry invalidation without invalidation on replica-only changes;
- equality between resolved-plan assembly and the exhaustive client planner;
- equality between incremental and full-scan completeness checks.

A saved benchmark should measure:

- `K = 723`, `H = 72`, and a generator-local request with a small `C`;
- the same directory with a full-span request where `C = H`;
- cold plan construction and warm cache hits;
- controller time, client planning time, metadata response size, and retained index
  memory;
- mutation cost for a synchronized `G`-publication burst.

The expected result is a large reduction in intersection calls and response metadata
when `C << H`, with limited improvement when the fetch must emit all `K·H` regions.

## 12. Open choices

- Store one interval index per key or group keys under a shared layout manifest.
- Carry explicit DTensor placements in `TensorSlice` metadata or introduce a separate
  layout descriptor.
- Use one sweep dimension, an interval tree, or a multidimensional spatial index for
  irregular layouts.
- Cache exact key plans, normalized layout templates, or both with separate budgets.
- Return a resolved per-volume fetch plan or logical offers plus a client-side source
  binding token.
- Bound plan-cache memory and choose eviction independently from directory residency.
- Shard the controller by key while keeping all replicas of one logical region under
  the same authority.
