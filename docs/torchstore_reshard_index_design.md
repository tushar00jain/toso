# Indexed TensorSlice reshard planning

**Status:** proposal · **Scope:** TorchStore directory lookup, DTensor reshard
planning, and TOSO source binding.

TorchStore should index `RegionKey` values derived from stored `TensorSlice` metadata
separately from the `Publication` values that name their volumes. A read can then
find the few stored slices that overlap `Request.tensor_slice`, reuse the resulting
`PlanOffer` values across model keys and requests, and choose a live or pending
`Publication` only when the read is routed.

The key trie remains useful. It finds an FQN without scanning unrelated keys. The
scaling problem is the unindexed value behind each key:

```text
Trie[key] -> volume -> StorageInfo[tensor_slices]
```

For `K` requested keys and `H` recorded `(publication_id, volume_id)` entries per key,
the controller validates
`K·H` entries and the client examines up to `K·H` entries again. The proposed
directory changes the inner lookup from source entries to indexed `RegionKey` values.

See [`torchstore.md`](torchstore.md) for the existing store path,
[`torchstore_dedup_design.md`](torchstore_dedup_design.md) for pending publications
and source ranking, and [`dedup_scaling.md`](dedup_scaling.md) for the synchronized
burst envelope.

## Terminology

The proposal retains TorchStore names for existing concepts:

- `TensorSlice` is stored shard geometry;
- `StorageInfo.tensor_slices` is the set of slices held by one volume;
- `Publication = (publication_id, volume_id)` identifies one live or pending source;
- publication id zero means the volume currently holds the data;
- a positive publication id means the volume has declared a pending batch;
- `Request.tensor_slice` is the region a client wants;
- `Controller`, `LocalClient`, `locate_volumes`, and `get_slice_intersection` retain
  their existing meanings.

The indexed controller adds names for state TorchStore does not currently model:

- `RegionKey` is the `(global_shape, offsets, local_shape)` projection of one
  `TensorSlice`; coordinates and mesh shape do not change that global region;
- `IntervalIndex` finds `RegionKey` values overlapping `Request.tensor_slice`;
- `PlanOffer` is one cached `get_slice_intersection` result, including source,
  destination, and length metadata;
- `BoundPlanOffer` attaches current live and pending `Publication` values to a
  `PlanOffer`;
- `SlicePlan` is the controller answer containing those bound offers;
- `geometry_epoch` invalidates cached `PlanOffer` values when stored `RegionKey`
  values change;
- a normalized topology is the interned mapping from normalized `RegionKey` values
  to complete `frozenset[Publication]` source sets used to share coverage work across
  compatible keys.

“Region,” “source,” “geometry,” and “binding” below are prose abbreviations for these
types, not additional APIs.

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
receives only the keys, returns every recorded `volume_id -> StorageInfo` entry, and
scans those entries to check DTensor completeness. The client then tests every
returned stored slice against the requested slice.

For a 6,400-element tensor:

```text
trainer layout:   8 shards, 800 elements each
generator layout: 64 shards, 100 elements each
generator rank 19 requests [1900:2000]
```

The request overlaps trainer region `[1600:2400]` and generator region
`[1900:2000]`. A directory containing all 72 `Publication` source entries still
examines 72 entries to discover those two `RegionKey` values. At `K = 723`, one pass
visits 52,056 `(key, Publication)` entries.

The work is necessary only when all 72 regions contribute bytes. When two regions
cover the request, the other 70 intersection checks are discovery overhead.

## 3. RegionKey values and Publication sources

The directory should give `TensorSlice` geometry and `Publication` residency
different identities.

```python
RegionKey = (
    tensor_slice.global_shape,
    tensor_slice.offsets,
    tensor_slice.local_shape,
)
```

`coordinates` and `mesh_shape` describe the DTensor placement that produced a slice.
They do not change the global rectangle. `Publication` values attached to the same
`RegionKey` are alternative sources for that stored region.

```text
Trie[key] -> _KeyEntry

_KeyEntry
  object_type
  geometry_epoch
  interval_index: IntervalIndex
  regions: dict[RegionKey, _RegionEntry]
  source_infos: dict[Publication, StorageInfo]
  layouts
  topology: NormalizedTopology

_RegionEntry
  sources: dict[Publication, set[TensorSlice]]
  source_ids: frozenset[Publication]
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
2. inserts a new `RegionKey` into `IntervalIndex` when absent;
3. adds the volume or publication to the region's source set;
4. updates the completeness state for the slice's layout.

Delete and publication retirement remove a source. The region leaves the index only
when no live or pending source names it.

Two versions keep cache invalidation narrow:

```text
geometry_epoch
  changes when the set of stored RegionKey values changes

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
result contains `C <= A` overlapping `RegionKey` values.

For regular DTensor layouts, a placement-aware owner calculation can be cheaper than
an interval query. `TensorSlice` is sufficient for exact indexing; explicit DTensor
placements such as `Shard(dim)` and `Replicate()` are needed for reliable
`TensorSliceLayout` records and direct coordinate-to-`RegionKey` formulas.

## 6. Cached reshard geometry

The first query between source geometry and a requested region produces immutable
intersection offers:

```text
PlanOffer
  source_region
  intersection
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

Each `_KeyEntry` has an exact geometry cache:

```text
cache[requested RegionKey] -> (geometry_epoch, tuple[PlanOffer])
```

This removes repeated intersection work for recurring reads. The implemented
normalized topology cache removes the key multiplier when keys have compatible
layouts and source incidence:

```text
NormalizedTopology
  normalized RegionKey -> frozenset[Publication]

coverage cache key
  (NormalizedTopology, normalized requested RegionKey, global-shape match)
```

Offsets and lengths are normalized by `TensorSlice.global_shape`, so proportionally
equivalent keys share one coverage template without key-name or model-specific rules.
Each key retains its exact `RegionKey` and `PlanOffer` values for client execution.
Uneven or incompatible boundaries form different normalized topologies.

The `_KeyEntry` cache stores `PlanOffer` geometry. The coverage cache stores interned
`frozenset[Publication]` incidence for one normalized topology. Neither cache stores
the winning publication, arrival time, or load score.

## 7. Binding a plan to sources

Routing binds each required `RegionKey` to current `Publication` values:

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

TOSO ranks the `Publication` values attached to each offer and chooses a cover of the
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

### 7.1 Ranking contract

The controller and TOSO have separate responsibilities:

```text
IndexedController.serving_union
  every live or pending (publication, volume) overlapping any request
        |
        v
TOSO ranking
  one total best-first order over that complete union
        |
        v
IndexedController.greedy_cover
  the first ranked sources whose regions cover every request
```

The directory must not remove replicas, choose one source per region, or otherwise
narrow the union before ranking. Indexing may share the work used to construct the
union, but the returned `frozenset[Publication]` names every eligible source.

For each `(pub, volume)` in the union, TOSO computes:

```text
if volume == requester:
  exclude

if pub == 0:
  wait = 0
else:
  exclude when pending_load[volume] >= fanout_cap
  exclude when arrival[pub, volume] is unknown
  wait = arrival[pub, volume]

hop = read_time(volume, requester, payload_bytes)
base = wait + (1 + fabric_weight) * hop
queued = pending_load[volume]
```

The chain mode uses `fabric_weight = 10` and orders lexicographically by:

```text
(base, queued, (pub, volume))
```

Load therefore breaks equal readiness-and-fabric prices. The spread mode uses
`fabric_weight = 0` and folds the first two dimensions into:

```text
base * (1 + queued)
```

It then orders by `(folded_score, (pub, volume))`. The source tuple is the final
stable tie-break in both modes.

`greedy_cover` consumes that order without repricing it. It selects a `Publication`
when the publication contributes at least one uncovered `RegionKey`, marks every
region it contributes, and continues until all requested regions are covered. Its
answer is an ordered subset of the ranking. Every selected positive publication
enters the readiness gate; live sources have publication id zero and need no gate.

An indexed implementation may precompute source-to-region incidence, intern identical
key layouts, and traverse the ranking once. Those changes are valid only when
`serving_union` and the ordered greedy subset remain identical to the exhaustive
contract above.

## 8. Complexity

Let:

- `K` be requested keys;
- `H` be `Publication` source entries recorded per key;
- `S` be distinct `RegionKey` values per key;
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
7. Add placement-aware `TensorSliceLayout` records shared across state-dict keys and
   ranks.

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
- Cache exact `RegionKey` plans, normalized topology templates, or both with separate
  budgets.
- Return a resolved per-volume fetch plan or logical offers plus a client-side source
  binding token.
- Bound plan-cache memory and choose eviction independently from directory residency.
- Shard the controller by key while keeping every `Publication` for one `RegionKey`
  under the same authority.

## 13. Regular-layout fast path

The interval index is the general path for arbitrary `StorageInfo.tensor_slices`.
When a complete set of `TensorSlice` values forms a regular DTensor layout, the
controller can also retain a TorchStore layout record:

```text
TensorSliceLayout
  TensorSlice.global_shape
  TensorSlice.mesh_shape
  sharded tensor dimensions
  mesh coordinate -> RegionKey(offsets, local_shape, global_shape)
  RegionKey -> live (0, volume_id) sources
            -> pending (publication_id, volume_id) sources
```

The controller reads the requested region from `Request.tensor_slice`. For each
sharded tensor dimension it computes the first and last stored mesh coordinates that
intersect `offsets:offsets + local_shape`, enumerates those coordinates, and looks up
their `RegionKey` entries directly. It does not call `get_slice_intersection` against
every `StorageInfo`. The cost is:

```text
O(d + C·d)
```

`d` is tensor rank and `C` is the number of intersecting `RegionKey` entries. Replica
count does not affect geometry discovery because all volumes and publications for
one region share its source set.

For even shards, the controller derives the coordinate boundaries from
`global_shape`, `mesh_shape`, and the sharded tensor dimension. Uneven shards retain
the exact ordered `TensorSlice.offsets` and `local_shape` boundaries. The layout
record must name the sharded tensor dimensions directly; `TensorSlice` does not carry
the DTensor `Shard(dim)` placements, and inferring them from key names is not sound.

The controller selects this path only when its live or pending `TensorSlice` metadata
proves one complete regular layout. Irregular slices, incomplete layouts, and
overlapping application-defined regions use `IntervalIndex` without changing
`locate_slices`, `serving_union`, or `greedy_cover` semantics.

The same separation applies to execution caching:

```text
stable SlicePlan geometry
  PlanOffer.source_region
  PlanOffer.storage_offsets
  PlanOffer.destination_offsets
  PlanOffer.lengths

dynamic BoundPlanOffer binding
  live volume_id values
  pending Publication values
  TOSO readiness, locality, and load ranking
```

The controller may construct the `PlanOffer` geometry once for a stored-layout and
requested-layout pair and reuse it for later weight versions. It binds live volumes
and pending publications afterward, so source changes do not rebuild geometry.
`LocalClient` may attach transport buffers and destination views to a resolved
`SlicePlan`, provided those objects are versioned separately from controller source
availability.

The regular-layout path is an optimization of region discovery. A request that
genuinely consumes `C` disjoint shards still emits `K·C` fetch entries; direct lookup
removes search work but cannot remove that output-size lower bound.
