# Dedup control-plane scaling

This note defines the synchronized weight-sync workload the dedup control plane must
survive and the benchmark used to measure it. The benchmark exercises metadata-only
TorchStore planning; payload bytes never enter the process.

## Scale dimensions

Let:

- `K` be keys in each generator's request;
- `T` be trainer/source ranks visible for each requested key;
- `G` be generator requests in the synchronized burst;
- `V = T + G` be candidate volumes at the peak of the burst;
- `D` be distinct pending batch shapes across the burst;
- `P` be pending publications named by one returned preference (`P ≤ K`);
- `Wₐ` be waiters released when action `a` commits.

The worst metadata shape is a synchronized full-state-dict request: every generator
asks for all `K` keys before earlier generators publish, and every source rank is
visible for every key. Sharded or sparse placement reduces the holders per key, so `T`
in the benchmark is the dense upper bound. Identical synchronized requests keep `D = 1`
even at large `G` -- one interned shape bucket answers candidate discovery in one
lookup.

Dense-placement measurements below use `K/T/G`. Allocation is the largest additional
Python peak reported for any measured phase, not process RSS.

| Workload (`K/T/G`) | Peak decision | Whole burst, `G` × peak | Declare burst | Largest Python phase peak | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| Current executable test (`1/1/64`) | 0.8 ms | 49 ms | 0.7 ms | 0.04 MiB | Trivial |
| Planned 8B (`290/4/16`) | 12 ms | 192 ms | 22 ms | 1.1 MiB | Comfortable |
| 70B key count, small burst (`723/4/16`) | 59 ms | 942 ms | 52 ms | 2.8 MiB | Comfortable |
| Wider 8B burst (`290/16/64`) | 16 ms | 1.03 s | 183 ms | 2.3 MiB | Comfortable |
| Wide 8B placement (`290/64/128`) | 28 ms | 3.6 s | 421 ms | 5.6 MiB | Usable for burst routing |
| Dense 70B (`723/64/128`) | 94 ms | 12.1 s | 1.15 s | 13.9 MiB | One decision is cheap; the burst is not |
| Fleet/MoE worst (`5,203/128/512`) | Projected ~1.3 s † | Projected ~11 min † | Projected 25–40 s † | Projected ~200 MiB † | Guarded; unsupported |

† extrapolated from the dense 70B row rather than measured; the guard rejects the
workload by default.

The whole-burst column is `G` × the peak decision and is an **upper bound**: peak
decision prices union against `T + G` candidates, while the first generator ranks
against `T` alone. It is also the number that decides whether a burst is servable at
all, because one serialized control plane answers the `G` requests one after another.
Nothing here overlaps it: a decision dispatches its own `Asked` and `Routed`, so the
fold work is already inside it.

Declare burst is the setup cost -- constructing all `G` pending publications and
staging their routes into the fan-out sensor before the peak decision runs. It is not
an addition to the whole-burst column: it is the state that column priced against.

The measured rows are synthetic capacity points, not observations of a particular
deployment. The fleet row extrapolates from dense 70B: 7.2 times the keys, 4 times the
burst and twice the holders per key. The union term (`K·T` live plus one shape probe
into `K·G` pending) scales roughly with `K·(T+G)`, so peak decision follows that; the
declare burst carries `G·K` writes and scales with those. The projected trie holds
3,329,920 indexed rows. The default benchmark guard prevents running it accidentally.
Production runs should substitute their state-dict key count, holders per requested
region, and synchronized generator count.

## State and work at the burst peak

The tables assume one independently serviceable region per `(key, source)`. Slicing
increases the per-key candidate set without adding new state kinds.

### Whole burst

| Component | Total time across `G` requests | Peak or retained space | Benchmark coverage |
| --- | --- | --- | --- |
| Declare | `O(GK)` pending-row inserts, one shape bucket per distinct shape | `O(GK)` pending rows in the trie, `O(G)` publication records, one shape reference per publication | `declare_burst_ms` |
| Route staging | `O(G(S + P))` | `O(GS)` charged load, `O(GP)` waiter links at the ceiling | `declare_burst_ms` |
| Serving union across `G` decisions | `O(GK·T)` live plus `O(G·D)` shape probes | Live-view cache retains at most one entry per key carrying a pending row | Peak decision `union_ms` |
| Total burst construction | `O(G(K·T + S + P) + G·D)` | `O(K·(T+G) + G + V + GP)` | `declare_burst_ms` + peak decision |
| Generator completions (`Published`) | `O(GK)` pending-row retirements plus `ΣWₐ` waiter releases | Each completion clears its publication and its waiter links | Not measured |
| Total lifecycle including completion | Adds `O(GK)` retirement work over the burst construction | Same construction peak; completion releases the pending state | Not measured |

A generator completion means its read-through finished, its local `put_batch` landed
the keys it fetched, and it dispatched `Published(volume, pub)`. Each landed put
replaces the pending row on its slot; `Published` retires whatever the publication
declared and did not land, releases the load charge, and wakes every gate that named
this publication. Retirement stays inside the `O(GK)` term: it is one pass over the
publication's own keys.

### One request at peak

This is one additional generator request evaluated after the benchmark has constructed
all `G` pending publications above.

| Component | Time | Peak space | Benchmark coverage |
| --- | --- | --- | --- |
| Declare the requester's publication | `O(K)` pending-row inserts | `O(K)` trie rows plus one publication record | `declare_ms` |
| Live serving union | `O(K·T)`: one directory read | `O(K·T)` transient candidate map | Part of `union_ms` |
| Pending serving union | Exact shape: one bucket lookup returning `O(G)` publications; other shapes: one probe per distinct shape | `O(D)` shape probes | Part of `union_ms` |
| Candidate scoring | `O(V)` reads of stored arrivals plus `O(V)` `read_time` calls | `O(V)` priced tuples | Part of `rank_ms` |
| Balance and ordering | `O(V + V log V)` | `O(V)` keyed selection | Part of `rank_ms` |
| Head-per-key gate set | `O(K)` head scans, `O(P)` pending pubs named | `O(P)` publication references | `gate_ms` |
| Route dispatch and arrival record | `O(S)` load edges plus one arrival float | `O(1)` retained on the requester publication | `full_decision_ms` |
| Gate registration | `O(P)` waiter links | `O(P)` links, up to `O(GP)` across a blocked burst | `gate_ms` |
| Total synchronous control decision | `O(K·T + G·D + V log V + K + P)` | Adds `O(K + P + S)` retained requester state plus `O(K·T)` transient union state | `full_decision_ms` |

The total request row ends after the route and readiness gate are constructed. It does
not include waiting for a pending publication, transferring payloads, or landing the
fetched batch.

Peer readiness is `O(1)` per publication: an arrival score was computed when the peer
was routed and is read back through `FanoutSensor.arrival`. Dependencies point at
publications declared earlier in the serialized decision stream, so the recursion the
old readiness walk carried is unnecessary -- the score is defined by construction.

Candidate discovery scales with distinct shapes rather than pending publications. The
common synchronized burst uses one exact-shape bucket lookup for all `G` publications
in one probe. A workload with several slice layouts pays for those distinct layouts;
slice intersection remains proportional to the metadata in each probed shape.

`indexed_metadata_entries` counts entries in the large key-multiplied structures, not
bytes or Python objects:

```text
K·T live rows in the trie
+ K·G pending rows in the same trie
= K·(T + G) indexed rows
```

It omits dictionary headers, `StorageInfo` objects, publication records, shape
buckets, arrival scores, load counts and waiter links, so it is a scale indicator
rather than a memory estimate. At the fleet envelope it is `5,203 × (128 + 512)`, or
`3,329,920` rows.

The benchmark constructs all `G` pending publications by declaring each and
dispatching its `Asked` and `Routed`. `declare_burst_ms` measures that setup;
`full_decision_ms` measures one additional declare/union/rank/route/gate operation
after the whole burst state exists.

## Reusable benchmark

Run the small default case from the repository root:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control
```

One smoke row is split below only to keep the example readable. Values vary by host
and Python build.

| case | keys | sources | burst requests | indexed entries | declare burst ms | declare ms | union ms | rank ms | gate ms | full decision ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 2 | 8 | 80 | 0.468 | 0.132 | 0.043 | 0.028 | 0.016 | 0.735 |

| declare burst peak KiB | declare peak KiB | union peak KiB | rank peak KiB | gate peak KiB | full decision peak KiB | candidates | pending candidates | selected sources |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 67.445 | 11.945 | 15.148 | 5.656 | 5.543 | 32.211 | 11 | 9 | 10 |

Run the intended 8B scale or supply a deployment-specific point:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control --preset planned-8b
.venv/bin/python -m realsim.tools.benchmark_dedup_control \
  --keys 290 --source-ranks 8 --generators 256 --repeats 5
```

The fleet preset is opt-in. A guard rejects cases whose estimated metadata is too large
for a routine developer run. Use `--allow-large` only on a host sized for the reported
workload:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control \
  --preset fleet-worst --repeats 1 --warmups 0 --allow-large
```

Output is one tab-separated header and row with fixed column order. Commit or archive
that row alongside the code revision and host description; compare the same preset,
Python build, and machine. The tool measures runtime first, then creates a fresh
workload for the traced-memory pass. Tracing starts separately for each phase, so its
overhead cannot affect the `*_ms` columns. Median values are used where multiple
repeats are requested; declare-burst construction and single-decision measurements are
each one observation. Memory phases run once. The declare-burst construction dominates
the noisiest column on a shared host; read the burst as an order of magnitude.

The tool does not measure controller RPC, payload transfer, publication latency, or
the simulation scheduler. It answers whether one serialized dedup controller can
construct a routing decision at the requested metadata scale.

## Anatomy of one request

Toy scale for this section: keys `w0, w1` (`K=2`), live sources `s0, s1` (`T=2`),
generators `g0, g1` already declared (`G=2`), requester `r`, so `V=4`.

State before `r` declares. Pending rows sit in the trie beside the live ones, tagged
with the publication that owns them, and one interned shape represents both generators
because they declared the same batch:

```text
trie                                   controller sidecars
  w0 -> {s0, s1, g0*p0, g1*p1}           publications
  w1 -> {s0, s1, g0*p0, g1*p1}             p0 -> generator=g0, keys={w0,w1}, shape=#1
        * = Pending(pub=..., shadowed=...)  p1 -> generator=g1, keys={w0,w1}, shape=#1
                                         shape #1 -> {p0, p1}
```

A `Pending` row is a `StorageInfo` subclass, so `isinstance` is the whole filter and
ordinary reads subtract them through the controller's live-view cache. A volume
holding part of a key and declaring the rest keeps its live entry as `shadowed`
underneath the pending row; retirement restores it.

`Dedup.sources([w0, w1], "r")` is one synchronous turn: nothing suspends between
declaration and gate.

```text
declare(r, [w0, w1])                    K pending-row inserts, one publication record
    |
dispatch_sync(Asked(pub_r))             fanout sensor sees the new publication
    |
serving_union([w0, w1])                 one directory read: K·T live entries;
                                        exact-shape lookup: {p0, p1} in one bucket
    |
Candidates.select
    |  price V candidates                stored arrival per pub, read_time per hop
    |  cap check against _behind         O(1) per priced pub
    |
Balance -> WithFold -> Ordered           V-entry keyed selection, sort V log V
    |
head-per-key gate set                   for each key, the ranked head if pending;
                                        for slices, every intersecting pending pub
    |
record arrival on requester_pub         source_arrival(head) + read_time(head, r)
    |
dispatch_sync(Routed(pub_r, ...))       load charged to head; gate assignments
    |
gate(not any is_in_flight(pub))         O(P) waiter links; probe once
```

The gate is per-key, not per-preference: `r` waits for the ranked head that serves each
requested key, not for every pending publication anywhere in its ranking. Gating on the
whole preference would collapse a wider fan-out cap to a chain, and the head is the
only source that gets read.

### Reusable state

Outlives the decision; sizes are for one requester unless stated.

| Structure | Owner | Size | Released by |
| --- | --- | --- | --- |
| `keys_to_storage_volumes` live rows | Controller trie | `K·T` slots | `notify_delete*` |
| `Pending` rows | Same trie | `K·G` slots | Landing puts, `retire_publication` |
| Publication records and shape buckets | Controller sidecars | `O(G + D)` plus each publication's accepted keys | `retire_publication` |
| Live-view cache | Controller sidecars | At most one entry per key carrying a pending row | Live mutation of the same key |
| `_arrival` | `FanoutSensor` | one float per pending publication | `Published(publication)` |
| `_behind` | `FanoutSensor` | one int per volume currently loaded | `Published`, reroute |
| `_assigned`, `_pending` gate bookkeeping | `FanoutSensor` | `O(V)` for the head charge and `O(GP)` for outstanding gate links across the burst | `Published(publication)` |
| `_waiters` | `Dispatcher` | one link per (pending publication, gate) | Commit |

The largest retained item is the pending trie rows themselves, `K·G`. There is no
per-region route requirement structure; readiness is per publication, `O(1)` per peer.
A `Pending` row's `shadowed` field carries at most one live entry to restore, so a
volume holding part of a key and declaring the rest costs one trie slot and one
`StorageInfo`.

### Per-request state

Dies when the pin releases; there is no pin because there is no derived cache to
invalidate.

| Structure | Size |
| --- | --- |
| `serving_union` answer | Live: `K·T` slots across `K` maps. Pending: one `{pub}` set per key |
| Priced candidates | `V` tuples |
| Keyed selection | `V` entries, rebuilt by `annotated` and `only` |
| `gate_pubs` | `O(P)` publication references |
| `ReadPlan.sources` | The ranked flat preference |

### What that costs in calls

One decision at `planned-8b` (`290/4/16`), counted by wrapping the sensor:

| Operation | Calls | Per key |
| --- | ---: | ---: |
| `serving_union` | 1 | |
| `arrival` reads | one per priced pub | |
| `read_time` | one per priced pub | |
| `is_in_flight` (gate probe) | one per pending pub named | |

There is no cache to key, no batch-spec build, no `covers` fallback, no per-source live
expansion. One directory read, `V` price lookups, one arrival record.

## Potential improvement

In profile order against the numbers above:

- Union work at wide bursts (`183 ms` at `290/16/64`, `421 ms` at `290/64/128`) is
  dominated by the live-directory read and its per-key hashing. Interning `(key,
  region)` to an int shrinks the retained state and speeds the reads; nothing else
  touches the region contents.
- `V log V` in `Ordered` is a stable-order sort. Scores collide by construction --
  every price is `wait + hop(1 + fabric)` over a handful of link classes -- so a
  counting-sort over the bucketed score is `O(V)` exact, not approximate.
- `read_time` is called `V` times per decision, once for each priced candidate. Caching
  the per-(source, requester) cost across a burst removes one function call per
  candidate; the cost has no key term, so a burst that reads the same requester many
  times amortizes it.
- The head-per-key gate can over-wait for a sliced key with several intersecting
  pending publications, because the plan does not narrow to the subset the client will
  actually pull from. Whole values do not take this path (`selected_sources = 1` at
  both smoke and `planned-8b`). If a DTensor-heavy sync shows wall-clock loss from
  conservative gating, `serving_union` can return per-key candidates ranked, and the
  gate then names only the publications that offer regions the head does not.
