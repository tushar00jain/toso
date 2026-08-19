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

| Workload (`K/T/G`) | Peak decision | Pending burst build | Largest Python phase peak | Assessment |
| --- | ---: | ---: | ---: | --- |
| Current executable test (`1/1/64`) | 1.7 ms | 0.7 ms | <0.1 MiB | Trivial |
| Planned 8B (`290/4/16`) | 38 ms | 13 ms | 3 MiB | Comfortable |
| 70B key count, small burst (`723/4/16`) | 95 ms | 33 ms | 8 MiB | Near a 100 ms envelope |
| Wider 8B burst (`290/16/64`) | 142 ms | 52 ms | 11 MiB | Usable, noticeable |
| Wide 8B placement (`290/64/128`) | 662 ms | 199 ms | 26 MiB | Too slow for burst routing |
| Dense 70B (`723/64/128`) | 1.51 s | 588 ms | 65 MiB | Completes, but is not practical |
| Fleet/MoE worst (`5,203/128/512`) | Projected 30–40 s | Not run | Projected >1 GiB per phase | Guarded; currently unsupported |

The measured rows are synthetic capacity points, not observations of a particular
deployment. The fleet row is an extrapolation from dense 70B: it has 8.66 million
indexed entries and about 24 times as many key/source pairs. The default benchmark
guard prevents running it accidentally; its combined retained and transient metadata
would likely require several GiB. Production runs should substitute their state-dict
key count, holders per requested region, and synchronized generator count.

## State and work at the burst peak

The tables assume one independently serviceable region per `(key, source)`. More
slices increase `R` and the TorchStore expansion terms without changing the ownership
of the state.

### Whole burst

| Component | Total time across `G` requests | Peak or retained space | Benchmark coverage |
| --- | --- | --- | --- |
| Dispatcher commits | `O(GFₐ + ΣWₐ)` for the synthetic `Asked` and `Routed` actions | Registered folds plus outstanding waiter links | `pending_build_ms` |
| Pending directory indexes and primed coverage | `O(GK)` | `O(GK)` index/cache references; equal request/metadata pairs share immutable region tuples | `pending_build_ms` |
| Fan-out route state | `O(G(R + S))` | `O(GR + GS + V)` requirements, route edges, and source loads | `pending_build_ms` |
| Total burst construction | `O(G(K + R + S + Fₐ) + ΣWₐ)` | `O(KT + GK + GR + GS + GP + V)` | `pending_build_ms` |
| Generator completions (`Published`) | `O(G(K + R) + ΣWₐ)` | Each completion releases its pending requirements and waiter links | Not measured |
| Total lifecycle including completion | `O(G(K + R + S + Fₐ) + ΣWₐ)` | Same construction peak; completion releases pending state | Not measured |

A generator completion means its read-through finished, its local `put_batch`
registered all fetched keys, and it dispatched `Published(generator)`. That action
removes the generator's pending directory and route requirements and wakes requests
that selected it as a source. A burst has at most `G` such completions. In each total
row, `ΣWₐ` is the waiter wake-up work across the actions included by that row.

### One request at peak

This is one additional generator request evaluated after the benchmark has constructed
all `G` pending requests above.

| Component | Time | Peak space | Benchmark coverage |
| --- | --- | --- | --- |
| Pinned live-directory snapshot | `O(KT)` | `O(KT)` transient copied mapping slots | `snapshot_ms` |
| Cold source coverage | `O(KV)`; slice intersection work is `O(U)` expansions for `U` distinct request/metadata pairs | `O(KV)` source coverage plus `O(U)` retained expansion tuples | Cold `serving_sources_ms` |
| Reused live coverage | `O(KT)` metadata signature plus `O(KG)` pending overlay assembly | Reuses `O(KT)` live coverage and `O(U)` expansion tuples | Reused `serving_sources_ms` |
| Candidate readiness and scoring | `O(GK + V + V log V)` for dense pending routes | `O(V + K)` transient wait memo and one pending route's coverage work | Part of `full_decision_ms` |
| Fetch materialization | `O(KV)` over pinned combined coverage | Up to `O(KV)` required-region output | `plan_fetch_ms` |
| Gate registration | `O(P)` | `O(P)` waiter links, up to `O(GP)` for blocked burst requests | Part of `full_decision_ms` |
| Total synchronous control decision | `O(KV + GK + V log V)` | Adds `O(K + R + P)` retained requester state plus `O(KV)` transient planning state | `full_decision_ms` |

The total request row ends after the route and readiness gate are constructed. It
does not include waiting for a pending source, transferring payloads, or publishing
the fetched batch.

Live reuse is keyed by ordered sources, `ObjectType`, every stored slice field, and
the request slice. A put, delete, eviction, slice mutation, source reorder, or request
mutation changes that signature before cached coverage can be returned.

`indexed_metadata_entries` counts entries in the large key-multiplied mappings and
counters, not bytes or Python objects:

```text
K*T live placements
+ K*G directory entries indexed by generator
+ K*G directory entries indexed by key
+ K*G fan-out route requirements
= K*T + 3*K*G indexed entries
```

The two directory indexes point to the same `_PendingEntry`; the formula counts both
mapping slots but not the object twice. It omits dictionary headers, `StorageInfo`
objects, route edges, load counters, and waiter links, so it is a scale indicator
rather than a memory estimate. At the fleet envelope it is 5,203 × (128 + 3 × 512),
or 8,657,792 cells.

`pending_build_ms` constructs all `G` pending requests by dispatching both `Asked`
and `Routed` for every synthetic generator. The cold/reused serving, fetch, and full
decision columns measure one additional request after that whole burst state exists.
`full_decision_ms` includes that requester's `Asked`, final `Routed`, and gate
registration. These numbers include dispatcher lookup, fold invocation, and commit
bookkeeping, but do not isolate fixed framework cost from sensor folds.
Each matching `*_python_peak_kib` column is `tracemalloc`'s additional peak for
Python metadata allocated during that phase. It excludes native allocations, tensor
allocators, process RSS, and the workload state live before tracing starts. The
cached coverage built by `serving_sources` is therefore baseline state for the
`plan_fetch_python_peak_kib` phase.

## Reusable benchmark

Run the small default case from the repository root:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control
```

One smoke row is split below only to keep the example readable. Values vary by host
and Python build.

| case | keys | sources | burst requests | indexed entries | pending build ms | cold serving ms | reused serving ms | plan fetch ms | full decision ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 2 | 8 | 208 | 0.600 | 0.450 | 0.292 | 0.208 | 1.695 |

| pending build peak KiB | cold serving peak KiB | reused serving peak KiB | plan fetch peak KiB | full decision peak KiB | candidates | pending candidates | selected sources |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 52.453 | 31.547 | 26.125 | 9.984 | 69.383 | 10 | 8 | 1 |

The planned-8B measurements below isolate the two implementation steps on one host.
The one-pass row predates cross-decision reuse; values are approximate rather than a
performance contract.

| Planner | pending build ms | serving ms | plan fetch ms | full decision ms | serving peak KiB | full peak KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Singleton-map planning | 5.56 | 29.68 | 10.10 | 55.27 | 3046 | 3387 |
| One-pass TorchStore expansion | 5.92 | 13.06 | 6.31 | 42.59 | 3361 | 3727 |
| Shared expansions and live reuse | 13.06 | 10.46 | 7.09 | 37.81 | 1318 | 2484 |

Run the intended 8B scale or supply a deployment-specific point:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control --preset planned-8b
.venv/bin/python -m realsim.tools.benchmark_dedup_control \
  --keys 290 --source-ranks 8 --generators 256 --repeats 5
```

The dense 70B and fleet presets are opt-in. A guard rejects cases whose estimated
metadata or conservative work bound is too large for a routine developer run. Use
`--allow-large` only on a host sized for the reported workload:

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
metadata. Every sample measures `serving_sources`, then cached `plan_fetch`, before
leaving its pin. Wall-clock reuse values are medians; pending-state construction,
cold planning, and the full decision are single observations. Memory phases run once
in the same cold-then-reused order.

The tool does not measure controller RPC, payload transfer, publication latency, or
the simulation scheduler. It answers whether one serialized dedup controller can
construct a routing decision at the requested metadata scale.

## Anatomy of one request

Toy scale for this section: keys `w0, w1` (`K=2`), live sources `s0, s1` (`T=2`),
generators `g0, g1` already asked (`G=2`), requester `r`, so `V=4`.

State before `r` asks:

```text
live directory              dedup pending overlay
  w0 -> {s0, s1}              _pending      g0 -> {w0: e0, w1: e1}
  w1 -> {s0, s1}                            g1 -> {w0: e2, w1: e3}
                              _pending_by_key  w0 -> {g0: e0, g1: e2}
                                               w1 -> {g0: e1, g1: e3}
```

The two pending indexes point at the same `_PendingEntry` objects.

`Dedup.sources([w0, w1], "r")` is one synchronous turn: the pin is taken after `Asked`
commits and released after `Routed` commits, with no await between, so no put, publish
or eviction can interleave.

```text
dispatch_sync(Asked(r, [w0,w1]))    K entries, each written to both pending indexes
    |
pinned([w0,w1])                     copy of the live read: K dicts, K*T slots
    |
Candidates.select
    |  plan(r)                      fresh K-entry dict
    |  serving_sources
    |     live coverage             K KeyCoverage + K*T SourceCoverage
    |     overlay coverage          K KeyCoverage + K*V SourceCoverage
    |     requirements x2           V Counters, K*V cells, for combined and live
    |  _wait(g) per pending peer    plan(g) + covers(): a fresh K-key coverage each
    |  price V candidates           env.read_time per candidate
    |
Balance -> WithFold -> Ordered      V-entry key mapping rebuilt three times, sort V log V
    |
plan_fetch(order)                   per key: holders, present, effective, a walk of
    |                               the whole order -> K*V, plus required Counters
dispatch_sync(Routed(r, ...))       Counter -> elements() tuple -> Counter again
    |
gate(covers(..., live=True))        a second directory read of the same K keys
```

### Reusable state

Outlives the decision; sizes are for one requester unless stated.

| Structure | Owner | Size | Released by |
| --- | --- | --- | --- |
| `_expansions` | `DirectorySensor` | one region tuple per distinct (request spec, storage spec) | first live-coverage signature change |
| `_pending`, `_pending_by_key` | `DedupDirectorySensor` | `K` entries, two index slots each | `Published` |
| `_PendingEntry.regions`, `alternate_regions` | `DedupDirectorySensor` | one tuple per key, shared by every reader that sees it | `Published` |
| `_route`, `_route_pending` | `FanoutSensor` | `S` + `P` | `Routed`, `Retired`, `Published` |
| `_route_required` | `FanoutSensor` | `S` Counters of up to `K` region cells; `G*K*S` cells across the burst | `Published` |
| `_load` | `FanoutSensor` | one counter per source | `_drop` |
| `_waiters` | `Dispatcher` | one link per (pending source, gate) | commit |

`_route_required` is the largest retained item: 92,544 cells at dense 70B, 2.66 million
at the fleet envelope, each a `(key, slice spec)` tuple hashed on every probe.

### Per-request state

Dies with the pin, except the decision cache which is cleared on pin exit.

| Structure | Size |
| --- | --- |
| Pinned `_located` copy | `K` dicts, `K*T` slots |
| Live `DirectoryCoverage` | `K` + `K*T` objects |
| Combined overlay coverage | `K` + `K*V` objects |
| Coverage cache keys (`_coverage_spec`) | `K` request-spec plus `K*T` storage-spec tuples, per `coverage()` call |
| `requirements()` Counters | `V` Counters, `K*V` cells, per coverage asked |
| `_wait` memo and its probes | `V` memo entries; one `K`-key coverage per pending peer |
| `Selection.key` | `V` entries, rebuilt by `priced`, `annotated` and `only` |
| `FetchPlan` | `K` source tuples plus `S` Counters |
| `Routed.required` | every required region flattened into one tuple |

### What that costs in calls

One decision at `planned-8b` (`290/4/16`), counted by wrapping the sensor:

| Operation | Calls | Per key |
| --- | ---: | ---: |
| `request_spec` | 13,920 | 48 |
| `_storage_spec` | 9,570 | 33 |
| `expand_regions` | 2,610 | 9 |
| `coverage` | 19 | |
| `covers` | 17 | |
| `locate_live` | 2 | |

Sixteen of the seventeen `covers` calls are `_wait` probes, one per pending peer, and
each rebuilds a whole `K`-key coverage plus its `O(K*T)` cache key. That is where the
48 spec constructions per key come from.

## Potential improvement

Two localized changes, prototyped by monkeypatching and measured on the full decision:

| Change | `planned-8b` | `dense-70b` |
| --- | ---: | ---: |
| Probe readiness from the pin's live requirements instead of rebuilding a coverage | 1.4–1.5x | 1.5–1.7x |
| Memoize `_storage_spec` per `StorageInfo` identity | 1.1x | 1.2–1.3x |
| Both | 1.6x | 2.0–2.1x |

Ratios are stable across runs; absolute times are not, so only the ratios are reported.
That would put wide 8B placement near 330 ms and dense 70B near 750 ms.

Remaining, in profile order after those two:

- `_overlay_coverage` is rebuilt from scratch every decision (`O(K*V)`, 0.59 s of a 1.7 s
  dense-70B decision) although one `Asked` or `Published` changes one generator's
  entries. An overlay retained across decisions and moved by those two folds makes it
  `O(K)` per decision, and is the only change that removes the `G` factor from the
  burst.
- `_coverage_spec` is `O(K*T)` per call and is paid to *look up* a cache. Within a pin
  the located mapping cannot change, so a pin generation counter plus request identity
  is an equivalent key.
- The gate's `covers(..., live=True)` reads the directory a second time for the same `K`
  keys. `_committed` runs inside the pin with no await, so the pinned snapshot is
  provably identical.
- `plan()` builds a fresh `K`-entry dict per call, `G+2` times per decision. Cache the
  mapping per producer and invalidate it in the two folds that move it.
- `Routed` flattens each source's Counter through `elements()` so `_routed` can rebuild
  the same Counter. Carry the counts.
- A region is a nested tuple hashed on every Counter and set operation. Interning
  regions to ints shrinks `_route_required` and speeds every probe.
- `plan_coverage` scans the whole ranking per key (`O(K*V)`). A rank dict makes it
  `O(t log t)` for `t` sources present on a key, which matters for sparse placement and
  is invisible in this dense benchmark.
