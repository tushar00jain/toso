# Dedup control-plane scaling

This note defines the synchronized weight-sync workload that the dedup control
plane must survive and the benchmark used to measure it. The benchmark exercises
metadata-only TorchStore planning; payload bytes never enter the process.

## Scale dimensions

Let:

- `K` be keys in each generator's batch request;
- `T` be trainer/source ranks visible for each requested key;
- `G` be generator requests in the synchronized burst;
- `V = T + G` be candidate volumes at the peak of the burst;
- `R` be region requirements stored for one route, `K` in the dense benchmark;
- `S` be sources selected for one route;
- `P` be pending sources selected by one decision;
- `Fₐ` be folds registered for action `a`;
- `Wₐ` be waiters released when action `a` commits.

The worst metadata shape is a synchronized full-state-dict request: every generator
asks for all `K` keys before earlier generators publish, and every source rank is
visible for every key. Sharded or sparse placement reduces the holders per key, so
`T` in the benchmark is deliberately the dense upper bound.

These dense-placement measurements use `K/T/G`: keys per generator request, live
sources per key, and synchronized generator requests. Allocation is the largest
additional Python peak reported for any measured phase, not process RSS.

| Workload (`K/T/G`) | Peak decision | Promise burst build | Largest Python phase peak | Assessment |
| --- | ---: | ---: | ---: | --- |
| Current executable test (`1/1/64`) | 1.3 ms | 0.7 ms | 0.1 MiB | Trivial |
| Planned 8B (`290/4/16`) | 4.4 ms | 9.3 ms | 2.3 MiB | Comfortable |
| 70B key count, small burst (`723/4/16`) | 9.2 ms | 21 ms | 6.3 MiB | Comfortable |
| Wider 8B burst (`290/16/64`) | 8.2 ms | 33 ms | 9.1 MiB | Comfortable |
| Wide 8B placement (`290/64/128`) | 17 ms | 212 ms | 18 MiB | Usable for burst routing |
| Dense 70B (`723/64/128`) | 40 ms | 500–750 ms | 53 MiB | Inside a 100 ms envelope |
| Fleet/MoE worst (`5,203/128/512`) | Projected 0.8–1.2 s † | Projected ~15 s † | Projected ~1.5 GiB † | Guarded; unsupported |

† extrapolated from the dense 70B row rather than measured; the guard rejects the
workload by default.

Promise burst build is the noisiest column on a shared host: repeated single
observations of dense 70B ranged 362–745 ms, and three timings inside one process
spread 492–602 ms. Read it as an order of magnitude, not a delta. It is dominated by
the `G*K` promise writes into the directory's own map — 92,544 of them at dense 70B —
and staging the same `G` routes with no pin and no stamp measures the same, so the
column says nothing about anything else a change touched.

The measured rows are synthetic capacity points, not observations of a particular
deployment. The fleet row is an extrapolation from dense 70B: 7.2 times the keys,
4 times the burst and twice the holders per key, so the `K*T` live-expansion term
scales 14x and the `G*K` promise terms 29x. It has 6.0 million indexed entries. The default
benchmark guard prevents running it accidentally. Production runs should substitute
their state-dict key count, holders per requested region, and synchronized generator
count.

## State and work at the burst peak

The tables assume one independently serviceable region per `(key, source)`. More
slices increase `R` and the TorchStore expansion terms without changing the ownership
of the state.

### Whole burst

| Component | Total time across `G` requests | Peak or retained space | Benchmark coverage |
| --- | --- | --- | --- |
| Dispatcher commits | `O(GFₐ + ΣWₐ)` for the synthetic `Asked` and `Routed` actions | Registered folds plus outstanding waiter links | `pending_build_ms` |
| Promised directory entries | `O(GK)` writes into the directory's own key-to-volume map | `O(GK)` entries in that one map, plus `O(G)` promise records naming the requests | `pending_build_ms` |
| Fan-out route state | `O(G(R + S))` | `O(GR + GS + V)` requirements, route edges, and source loads | `pending_build_ms` |
| Total burst construction | `O(G(K + R + S + Fₐ) + ΣWₐ)` | `O(KT + GK + GR + GS + GP + V)` | `pending_build_ms` |
| Generator completions (`Published`) | `O(G(K + R) + ΣWₐ)` | Each completion releases its promises, its requirements and its waiter links | Not measured |
| Total lifecycle including completion | `O(G(K + R + S + Fₐ) + ΣWₐ)` | Same construction peak; completion releases the promised state | Not measured |

A generator completion means its read-through finished, its local `put_batch`
registered the keys it fetched, and it dispatched `Published(generator, landed)`. Each
of those puts replaces the promise it lands on, and `Published` clears whatever the
generator promised and did not publish, releases the leg of every route routed onto it
that `landed` covers, then wakes requests that selected it as a source. A burst has at
most `G` such completions. Releasing legs stays inside the `O(GR)` term: the readers
behind one producer are capped by `fanout_cap`, and each is one pass over the `R`
regions it was owed. In each total row, `ΣWₐ` is the waiter wake-up work across the
actions included by that row.

### One request at peak

This is one additional generator request evaluated after the benchmark has constructed
all `G` pending requests above.

| Component | Time | Peak space | Benchmark coverage |
| --- | --- | --- | --- |
| Pinned live-directory snapshot | `O(KT)`; promised entries are not walked | `O(KT)` transient copied mapping slots | `snapshot_ms` |
| Cold live requirements | `O(KT)`: one planner run per source over that source's own entries, so slice intersection runs once per `(key, source)` | `O(KT)` region counters and one `K`-entry map per source | Cold `serving_sources_ms` |
| Reused live requirements | `O(KT)` metadata signature, or `O(1)` where the directory counts its own mutations, plus `O(G)` overlap tests | Reuses the `O(KT)` counters | Reused `serving_sources_ms` |
| Candidate readiness and scoring | `O(PS + V log V)` flag lookups where the directory has not moved since each peer's route was read; `O(P(S + K) + V log V)` region checks against the cached live requirements where it has | `O(V)` transient wait memo | Part of `full_decision_ms` |
| Fetch materialization | One planner run over `K` lazily ranked maps: `O(K)` where one source holds a whole value, `O(KV)` where every volume of a sliced key is walked | Up to `O(KS)` required-region output, and `K` map views rather than `K` ranked dicts | `plan_fetch_ms` |
| Gate registration | `O(P)` | `O(P)` waiter links, up to `O(GP)` for blocked burst requests | Part of `full_decision_ms` |
| Total synchronous control decision | `O(KT + G + PS + KS + V log V)` | Adds `O(K + R + P)` retained requester state plus `O(KT)` transient planning state | `full_decision_ms` |

The total request row ends after the route and readiness gate are constructed. It
does not include waiting for a pending source, transferring payloads, or publishing
the fetched batch.

Readiness carries no `K` term while the directory is what each peer's route was read
against, because then the route's own pending flags describe live coverage exactly: a
flag is set from the plan the route was made from and cleared by the publication that
carried every region it named. Where the directory has moved since, the flags are not
trusted and the leg is re-read, at one `K`-cell subtraction per `(peer, source)` — the
`O(P(S + K))` term. A burst with no eviction in it never takes that path; a run that
evicts between a route and a decision takes it for the routes the eviction outdates.

`O(G)` rather than `O(KG)` for candidate discovery rests on the batch shape being
interned: every generator in a synchronized burst promises the same `K` keys with the
same slices, so "does this generator promise what I am asking for" is one pointer
comparison. A generator promising a *different* shape is instead planned against its
own promised metadata, at one expansion per key the two batches share.

Live reuse is keyed by ordered sources, `ObjectType`, the stored slice set, and the
request slice. A put, delete, eviction, slice mutation, source reorder, or request
mutation changes that signature before the cached requirements can be returned.
Making or clearing a promise does not: it cannot change which volumes hold a key, so
the live view of a promised key is rebuilt only by a live mutation of that key.

`indexed_metadata_entries` counts entries in the large key-multiplied mappings and
counters, not bytes or Python objects:

```text
K*T live placements
+ K*G promised entries, in the same map as the live ones
+ K*G fan-out route requirements
= K*T + 2*K*G indexed entries
```

It omits dictionary headers, `StorageInfo` objects, route edges, load sets, and
waiter links, so it is a scale indicator rather than a memory estimate. At the fleet
envelope it is 5,203 × (128 + 2 × 512), or 5,993,856 cells.

`pending_build_ms` constructs all `G` promised batches by dispatching both `Asked`
and `Routed` for every synthetic generator. The cold/reused serving, fetch, and full
decision columns measure one additional request after that whole burst state exists.
`full_decision_ms` includes that requester's `Asked`, final `Routed`, and gate
registration. These numbers include dispatcher lookup, fold invocation, and commit
bookkeeping, but do not isolate fixed framework cost from sensor folds.
Each matching `*_python_peak_kib` column is `tracemalloc`'s additional peak for
Python metadata allocated during that phase. It excludes native allocations, tensor
allocators, process RSS, and the workload state live before tracing starts. The
cached live requirements built by `serving_sources` are therefore baseline state for
the `plan_fetch_python_peak_kib` phase.

## Reusable benchmark

Run the small default case from the repository root:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control
```

One smoke row is split below only to keep the example readable. Values vary by host
and Python build.

| case | keys | sources | burst requests | indexed entries | pending build ms | cold serving ms | reused serving ms | plan fetch ms | full decision ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 2 | 8 | 144 | 0.322 | 0.187 | 0.043 | 0.050 | 0.834 |

| pending build peak KiB | cold serving peak KiB | reused serving peak KiB | plan fetch peak KiB | full decision peak KiB | candidates | pending candidates | selected sources |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 47.344 | 8.281 | 5.672 | 6.234 | 27.578 | 10 | 8 | 1 |

The planned-8B measurements below isolate the implementation steps on one host. The
first three rows are carried over from earlier hosts and runs; values are approximate
rather than a performance contract. The last three were measured back to back on one
host, the newest against the current tree.

| Planner | pending build ms | serving ms | plan fetch ms | full decision ms | serving peak KiB | full peak KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Singleton-map planning | 5.56 | 29.68 | 10.10 | 55.27 | 3046 | 3387 |
| One-pass TorchStore expansion | 5.92 | 13.06 | 6.31 | 42.59 | 3361 | 3727 |
| Shared expansions and live reuse | 13.06 | 10.46 | 7.09 | 37.81 | 1318 | 2484 |
| Promises held in the directory | 8.00 | 1.68 | 1.67 | 6.56 | 145 | 436 |
| Greedy planner as the dry run | 8.71 | 1.39 | 1.09 | 5.87 | 144 | 462 |
| Readiness off the route's own flags | 9.27 | 1.41 | 1.14 | 4.43 | 144 | 462 |

Run the intended 8B scale or supply a deployment-specific point:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control --preset planned-8b
.venv/bin/python -m realsim.tools.benchmark_dedup_control \
  --keys 290 --source-ranks 8 --generators 256 --repeats 5
```

The fleet preset is opt-in. A guard rejects cases whose estimated metadata or
conservative work bound is too large for a routine developer run. Use `--allow-large`
only on a host sized for the reported workload:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control \
  --preset fleet-worst --repeats 1 --warmups 0 --allow-large
```

Output is one tab-separated header and row with fixed column order. Commit or archive
that row alongside the code revision and host description; compare the same preset,
Python build, and machine before and after a change. The tool measures runtime first,
then creates a fresh workload for the traced-memory pass. Tracing starts separately
for each phase, so its overhead cannot affect the `*_ms` columns. The first planning
sample reports `cold_*`; later fresh-pin samples report reuse against unchanged live
metadata. Every sample measures `serving_sources`, then `plan_fetch` against the live
requirements that call cached, before leaving its pin. Wall-clock reuse values are
medians; pending-state construction,
cold planning, and the full decision are single observations. Memory phases run once
in the same cold-then-reused order.

The tool does not measure controller RPC, payload transfer, publication latency, or
the simulation scheduler. It answers whether one serialized dedup controller can
construct a routing decision at the requested metadata scale.

## Anatomy of one request

Toy scale for this section: keys `w0, w1` (`K=2`), live sources `s0, s1` (`T=2`),
generators `g0, g1` already asked (`G=2`), requester `r`, so `V=4`.

State before `r` asks. There is one map, the store's own, and an entry in it is
either a holder or a promise:

```text
controller.keys_to_storage_volumes        dedup's promise records
  w0 -> {s0, s1, g0*, g1*}                  g0 -> requests {w0, w1}, shape #1
  w1 -> {s0, s1, g0*, g1*}                  g1 -> requests {w0, w1}, shape #1
        * = Promised(owner=...)
```

A `*` entry is a `StorageInfo` subclass, so `isinstance` is the whole filter, and
`locate_volumes`, `get`, the DTensor commit check and `keys()` subtract them. Only a
read that asks (`locate_raw(..., projected=True)`) sees them. A volume holding part
of a key and promising the rest has one entry covering both: it carries the live
slices it landed on, so the filter answers with those and clearing the promise
restores them. The promise records
hold what a `StorageInfo` cannot: the `Request` each generator promised, which is
what its own route was planned from, and one interned *shape* object per distinct
batch — `g0` and `g1` share `#1`, so "does this generator promise what I am asking
for" is a pointer comparison.

`Dedup.sources([w0, w1], "r")` is one synchronous turn: the pin is taken after `Asked`
commits and released after `Routed` commits, with no await between, so no put, publish
or eviction can interleave.

```text
dispatch_sync(Asked(r, [w0,w1]))    K project() calls into the one map
    |
pinned([w0,w1])                     copy of the live read: K dicts, K*T slots
    |                               (promised entries are filtered, not walked)
Candidates.select
    |  plan(r)                      the promise record's own mapping, not rebuilt
    |  serving_sources
    |     live requirements         one GreedyClient run per source over that
    |                               source's own K entries -> K*T regions, reused
    |     one test per generator    shape identity, then K-cell Counter subtraction
    |  _wait(g) per pending peer    per route source: one flag lookup, or a K-cell
    |                               subtraction where the directory has since moved
    |  price V candidates           env.read_time per candidate
    |
Balance -> WithFold -> Ordered      V-entry key mapping rebuilt three times, sort V log V
    |
plan_fetch(order)                   one projected read, then one GreedyClient run
    |                               over K ranked map views: it takes the first
    |                               listed volume for a whole value and every
    |                               volume offering a fresh region for a sliced one
dispatch_sync(Routed(r, ...))       Counter -> elements() tuple -> Counter again
    |
gate(not fetch.pending)             no second directory read
```

The ranked views are why the middle of that step is `O(K)` and not `O(K*V)`: a view
yields its key's entries in ranking order without materializing a dict, so the
planner's `break` on a whole value stops it after one membership test.

The plan is a dry run of the method the fetch itself runs, so control and the reader
cannot disagree about which volume serves which region of one located map. They can
still see *different* maps — the plan is taken under the pin and the reader's
`locate_volumes` happens after the gate opens — and that is why the plan reaches the
reader as a preference rather than a filter: a source that evicted in between drops
out and the read still answers from whoever holds the key.

### Reusable state

Outlives the decision; sizes are for one requester unless stated.

| Structure | Owner | Size | Released by |
| --- | --- | --- | --- |
| `_derived` | `DirectorySensor` | one live requirements map per batch spec | the directory's `revision` moving |
| `Promised` entries | the controller's own directory | `K` slots, one each | a put on the same slot, or `Published` |
| `_batches` | `DedupDirectorySensor` | `K` requests plus `K` `StorageInfo`, and one shape reference | `Published` |
| `_route`, `_route_pending`, `_route_stamp` | `FanoutSensor` | `S` + `P` + one stamp reference | `Routed`, `Retired`, `Published` |
| `_route_required` | `FanoutSensor` | `S` Counters of up to `K` region cells; `G*K*S` cells across the burst | `Published` |
| `_load` | `FanoutSensor` | one set per source, `S` entries across all of them per route | `_drop` |
| `_landed` | `DedupDirectorySensor` | one mutation count per volume that has published | never |
| `_waiters` | `Dispatcher` | one link per (pending source, gate) | commit |

`_route_required` is the largest retained item: 92,544 cells at dense 70B, 2.66 million
at the fleet envelope, each a `(key, slice spec)` tuple. A decision reads none of them
while it can answer readiness from the flags; the fallback probe hashes every cell of
the leg it re-reads.

The `StorageInfo` a generator promised is one object, referenced by both its promise
record and the directory entry, so a promised `(key, generator)` costs one map slot
and one dict entry.

### Per-request state

Dies with the pin, except the derived cache, which outlives it.

| Structure | Size |
| --- | --- |
| Pinned `_located` copy | `K` dicts, `K*T` slots |
| `promised()` read | `K` references into the directory's own maps, one per decision |
| Live requirements | `T` `K`-entry maps and `T` Counters of `K` cells, reused across decisions |
| Ranked map views | `K` slotted views over the pinned and promised maps, no ranked dict |
| `_wait` memo and its probes | `V` memo entries; no allocation per (peer, source) until a leg falls back to a `K`-cell subtraction |
| `Selection.key` | `V` entries, rebuilt by `priced`, `annotated` and `only` |
| `FetchPlan` | `K` source tuples plus `S` Counters |
| `Routed.required` | every required region flattened into one tuple |

### What that costs in calls

One decision at `planned-8b` (`290/4/16`), counted by wrapping the sensor:

| Operation | Calls | Per key |
| --- | ---: | ---: |
| `request_spec` | 1,160 | 4 |
| `requirements` | 2 | |
| `covers` | 0 | |
| `GreedyClient` runs | 5 | |
| `locate_live` | 1 | |
| `promised` | 1 | |

No `covers` at all: every peer's route was read against this same directory, so the
sixteen readiness questions are sixteen flag lookups. That is most of the other two
lines as well — a `covers` probe derives the batch spec that keys the requirements
cache, which is one `request_spec` per key each time. The two `requirements` calls left
are the planning reads, and one of them hits the cache. Four of the five planner runs
are the per-source live expansion and the fifth is the fetch plan.

## Potential improvement

In profile order against the numbers above:

- `_route_required` is still `O(P(S + K))` region cells retained, hashed on every probe
  the fallback path takes. Interning a region to an int shrinks the retained state and
  speeds those probes; nothing else touches the region's contents.
- `request_spec` is rebuilt 4 times per key, all of it to key the derived cache.
  Caching it on the request would make that lookup a pair of pointers.
- `Routed` flattens each source's Counter through `elements()` so `_routed` can rebuild
  the same Counter. Carry the counts.
- a sliced key is walked to the end of the ranking, because the planner has no early
  stop once a request is fully covered — upstream's own limitation, kept. Stopping
  needs box subtraction across misaligned grids; walking the rest costs metadata only,
  and is invisible in this whole-value benchmark.
- the ranked view falls back to directory order by walking the whole ranking first and
  finding nothing. A key no ranked source lists therefore costs `O(V)`, which is the
  unroutable case and rare.
