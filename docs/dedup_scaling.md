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
- `C` be publications retained in one requester's greedy cover (`C ≤ K`, `C = 1`
  for whole-value keys with any single-source that covers the request);
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
| Current executable test (`1/1/64`) | 0.7 ms | 46 ms | 0.5 ms | 0.04 MiB | Trivial |
| Planned 8B (`290/4/16`) | 2.5 ms | 40 ms | 6 ms | 0.34 MiB | Trivial |
| 70B key count, small burst (`723/4/16`) | 5.1 ms | 81 ms | 16 ms | 0.84 MiB | Trivial |
| Wider 8B burst (`290/16/64`) | 4.5 ms | 285 ms | 128 ms | 0.35 MiB | Comfortable |
| Wide 8B placement (`290/64/128`) | 9 ms | 1.15 s | 173 ms | 0.38 MiB | Comfortable |
| Dense 70B (`723/64/128`) | 21 ms | 2.74 s | 602 ms | 0.86 MiB | Comfortable |
| Fleet/MoE worst (`5,203/128/512`) | Projected ~0.65 s † | Projected ~5.5 min † | Projected 15–20 s † | Projected ~30 MiB † | Guarded; unsupported |

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
| Declare | `O(GK)` pending-map inserts, one shape bucket per distinct shape | `O(GK)` pending entries in `_pending`, `O(G)` publication records, one shape reference per publication | `declare_burst_ms` |
| Route staging | `O(GC)` load edges plus waiter links | `O(GC)` charged load, `O(GC)` waiter links at the ceiling | `declare_burst_ms` |
| Serving union across `G` decisions | `O(G·K·T)` live plus `O(G·D)` shape probes | Flat `frozenset[Publication]` per decision; publication records reused across decisions | Peak decision `union_ms` |
| Total burst construction | `O(G(K·T + C) + G·D)` | `O(K·(T+G) + G + V + GC)` | `declare_burst_ms` + peak decision |
| Generator completions (`Published`) | `O(GK)` pending-map retirements plus `ΣWₐ` waiter releases | Each completion clears its publication and its waiter links | Not measured |
| Total lifecycle including completion | Adds `O(GK)` retirement work over the burst construction | Same construction peak; completion releases the pending state | Not measured |

A generator completion means its read-through finished, its local `put_batch` landed
the keys it fetched, and it dispatched `Published(publication)`. Each landed put moves
its slot from `_pending` into the live trie; `Published` retires whatever the
publication declared and did not land, releases the load charge, and wakes every gate
that named this publication. Retirement stays inside the `O(GK)` term: it is one pass
over the publication's own keys.

### One request at peak

This is one additional generator request evaluated after the benchmark has constructed
all `G` pending publications above.

| Component | Time | Peak space | Benchmark coverage |
| --- | --- | --- | --- |
| Declare the requester's publication | `O(K)` pending-map inserts | `O(K)` entries in `_pending` plus one publication record | `declare_ms` |
| Serving union | `O(K·T)` live overlap plus `O(D)` shape probes; returns `frozenset[Publication]` with `V + G` sources at the peak | `O(V + G)` flat set of `(pub_id, volume)` tuples | `union_ms` |
| Candidate scoring | `O(V + G)` `arrival` reads plus `O(V + G)` `read_time` calls; per-volume max collapses `V + G → V` | `O(V)` priced tuples | `rank_ms` |
| Balance and ordering | `O(V + V log V)` | `O(V)` keyed selection | `rank_ms` |
| `greedy_cover` (per key, per slice via the shared walker) | `O(C · K)` region-overlap checks; breaks early at full coverage. `C = 1` under whole-value single-source coverage | `O(C)` chosen publications, `O(K)` covered-region set | Part of `full_decision_ms` |
| Route dispatch and arrival record | `O(C)` load edges plus one arrival float | `O(1)` retained on the requester publication | `full_decision_ms` |
| Gate registration | `O(C)` waiter links | `O(C)` links, up to `O(GC)` across a blocked burst | `gate_ms` |
| Total synchronous control decision | `O(K·T + D + V log V + C·K)` | Adds `O(K + C)` retained requester state plus `O(V + G)` transient union state | `full_decision_ms` |

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

The greedy walk on the control plane is `Controller.greedy_cover`. It builds ranked
source maps and delegates to the same per-key, per-slice walker the client uses for
volume requests. Under whole-value synchronized bursts the first pick covers the whole
request and `greedy_cover` returns after one source -- `C = 1`. Sliced or sparse
workloads walk until every requested slice region is covered.

`indexed_metadata_entries` counts entries in the large key-multiplied structures, not
bytes or Python objects:

```text
K·T live rows in the controller trie
+ K·G pending entries in the controller's _pending map
= K·(T + G) indexed entries
```

It omits dictionary headers, `StorageInfo` objects, publication records, shape
buckets, arrival scores, load counts and waiter links, so it is a scale indicator
rather than a memory estimate. At the fleet envelope it is `5,203 × (128 + 512)`, or
`3,329,920` entries.

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
| smoke | 8 | 2 | 8 | 80 | 0.229 | 0.067 | 0.029 | 0.031 | 0.011 | 0.650 |

| full decision instructions | declare burst peak KiB | declare peak KiB | union peak KiB | rank peak KiB | gate peak KiB | full decision peak KiB | candidates | pending candidates | selected sources |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,594,571 | 63.055 | 7.414 | 5.000 | 5.430 | 5.629 | 23.195 | 11 | 9 | 10 |

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

The `full_decision_instructions` column is retired hardware instructions from
`perf_event_open` (Linux x86_64 only, skipped elsewhere). It is deterministic within
one Python build and complements the peak-KiB columns; the benchmark gate at
`realsim/tests/test_benchmark_gate.py` compares against the recorded baseline.

The tool does not measure controller RPC, payload transfer, publication latency, or
the simulation scheduler. It answers whether one serialized dedup controller can
construct a routing decision at the requested metadata scale.

## Anatomy of one request

Toy scale for this section: keys `w0, w1` (`K=2`), live sources `s0, s1` (`T=2`),
generators `g0, g1` already declared (`G=2`), requester `r`, so `V=4`.

State before `r` declares. Live entries sit in the controller trie; pending entries
live in a separate `_pending` map keyed by `(key, volume, pub)`. One interned shape
represents both generators because they declared the same batch:

```text
trie (live only)                       controller sidecars
  w0 -> {s0, s1}                         _pending[w0]
  w1 -> {s0, s1}                           g0 -> {1: StorageInfo}
                                           g1 -> {2: StorageInfo}
                                         _pending[w1]
                                           g0 -> {1: StorageInfo}
                                           g1 -> {2: StorageInfo}
                                         publications
                                           1 -> volume=g0, entries={w0,w1}, shape=#1
                                           2 -> volume=g1, entries={w0,w1}, shape=#1
                                         shape_pubs
                                           #1 -> {1, 2}
```

Ordinary reads (`locate_volumes`, `get`, `keys`, DTensor commit check) traverse only
the trie. Pending entries live in the separate `_pending` map and are consulted by
`serving_union`, `regions_covered`, and `greedy_cover` when it builds source maps.

`Dedup.sources([w0, w1], "r")` is one synchronous turn: nothing suspends between
declaration and gate.

```text
declare(r, [w0, w1])                    K pending-map inserts, one publication record
    |
dispatch_sync(Asked(pub_r))             fanout sensor sees the new publication
    |
serving_union([w0, w1])                 one directory read: K·T live entries;
                                        exact-shape lookup: {(1,g0),(2,g1)} in one
                                        bucket; return frozenset[Publication] flat
    |
Candidates.select
    |  price every publication           `wait = 0 if pub_id == 0 else arrival[pub]`
    |  cap check pending sources         O(1) per priced publication
    |  aggregate per volume              max wait across a volume's publications
    |
Balance -> WithFold -> Ordered           V-entry keyed selection, sort V log V
    |
greedy_cover(requests, ranked)          per key, per slice via the shared walker;
                                        take each source with a fresh overlap and
                                        stop each whole-value key at its first source.
                                        For whole-value keys the first pick covers
                                        everything -- C = 1.
    |
record arrival on requester_pub         max over chosen of arrival + hop(source,r)
    |
dispatch_sync(Routed(pub_r, ...))       load charged to chosen volumes
    |
gate(all pending in chosen have landed) O(C) waiter links; probe once
```

The gate names exactly the pending publications in the chosen cover. Nothing else --
the greedy walk skips publications superseded by an earlier live pick, so the gate
never over-includes. The client's `_build_volume_requests` runs the same walk over the
returned volume preference at fetch time; both sides pick from the same argmin, so the
control plane's precomputed gate and the client's actual fetch agree.

### Reusable state

Outlives the decision; sizes are for one requester unless stated.

| Structure | Owner | Size | Released by |
| --- | --- | --- | --- |
| `keys_to_storage_volumes` (live trie) | Controller | `K·T` slots | `notify_delete*` |
| `_pending` map | Controller | `K·G` entries; `dict[key, dict[vol, dict[pub, StorageInfo]]]` | Landing puts, `retire_publication` |
| Publication records and shape buckets | Controller | `O(G + D)` plus each publication's accepted keys | `retire_publication` |
| `_arrival` | `FanoutSensor` | one float per pending publication | `Published(publication)` |
| `_behind` | `FanoutSensor` | one int per volume currently loaded | `Published`, reroute |
| `_assigned` gate bookkeeping | `FanoutSensor` | `O(V)` for outstanding load charges plus `O(GC)` for waiter links across the burst | `Published(publication)` |
| `_waiters` | `Dispatcher` | one link per (pending publication, gate) | Commit |

The largest retained item is the pending map itself, `K·G` entries. There is no
per-region route requirement structure; readiness is per publication, `O(1)` per peer.

### Per-request state

Dies with the decision.

| Structure | Size |
| --- | --- |
| `serving_union` answer | `frozenset[Publication]`, `O(V + G)` tuples |
| Priced candidates | `V` tuples after per-volume aggregation |
| Keyed selection | `V` entries, rebuilt by `annotated` and `only` |
| `chosen` from `greedy_cover` | `O(C)` publications |
| `covered` region set (transient inside `greedy_cover`) | `O(K)` regions |
| `gate_pubs` | `O(C)` publication references |
| `ReadPlan.sources` | The ranked flat preference over `V` volumes |

### What that costs in calls

One decision at `planned-8b` (`290/4/16`), counted by wrapping the sensor:

| Operation | Calls | Per key |
| --- | ---: | ---: |
| `serving_union` | 1 | |
| `arrival` reads | one per priced publication | |
| `read_time` | one per priced publication | |
| `greedy_cover` shared-walker overlap | one per requested key and visited source | |
| `is_in_flight` (gate probe) | one per pending publication named | |

There is no cache to key, no batch-spec build, no `covers` fallback, no per-source
live expansion. One directory read, `V + G` price lookups, one greedy walk with early
break, one arrival record.

## Potential improvement

In profile order against the numbers above:

- `read_time` is called `V + G` times per decision, once for each priced source.
  Caching the per-(volume, requester) cost across a burst removes one function call
  per source; the cost has no key term, so a burst that reads the same requester many
  times amortizes it.
- `V log V` in `Ordered` is a stable-order sort. Scores collide by construction --
  every price is `wait + hop(1 + fabric)` over a handful of link classes -- so a
  counting-sort over the bucketed score is `O(V)` exact, not approximate.
- Union work at the widest bursts (`128 ms` declare burst at `290/16/64`, `602 ms` at
  `723/64/128`) is dominated by the per-key live-directory read and its per-key
  hashing. Interning `(key, region)` to an int shrinks the retained state and speeds
  the reads; nothing else touches the region contents.
