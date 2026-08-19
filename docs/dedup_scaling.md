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

The repository currently establishes these burst envelopes. Each row means `K` keys
per generator request, `G` synchronized requests, and `T` live sources per key.

| Workload | Keys | Source ranks | Generator ranks | Status |
| --- | ---: | ---: | ---: | --- |
| Largest executable dedup test | 1 | 1 | 64 | Covered by `test_a_chain_deeper_than_the_fabric_charge_starts_a_new_one` |
| Planned dense 8B sync | about 290 | 4 | 16 | Key and reader counts documented in `domain/llm.py`; not yet an executable scale test |
| Dense 70B planning case | about 723 | 64 | 128 | Capacity-planning assumption |
| Fleet/MoE worst case | 5,203 | 128 | 512 | Deliberately severe planning envelope; the key count matches the fleet example in `tui_design.md` |

The 70B and fleet rows are planning envelopes, not measurements of a particular
deployment. Production runs should replace them with the state-dict key count,
holders per requested region, and synchronized generator count from that deployment.

## State and work at the burst peak

The table assumes one independently serviceable region per `(key, source)`. More
slices increase `R` and the TorchStore expansion terms without changing the ownership
of the state.

| Scope | Component | Time | Peak space | Benchmark coverage |
| --- | --- | --- | --- | --- |
| Whole burst | Dispatcher commits | `O(G(Fₐ + Wₐ))` for the synthetic `Asked` and `Routed` actions | Registered folds plus outstanding waiter links | `pending_build_ms` |
| Whole burst | Pending directory indexes and primed coverage | `O(GK)` | `O(GK)` index/cache references; equal request/metadata pairs share immutable region tuples | `pending_build_ms` |
| Whole burst | Fan-out route state | `O(G(R + S))` | `O(GR + GS + V)` requirements, route edges, and source loads | `pending_build_ms` |
| One request at peak | Pinned live-directory snapshot | `O(KT)` | `O(KT)` transient copied mapping slots | `snapshot_ms` |
| One request at peak | Source coverage visits | `O(KV)`; slice intersection work is `O(U)` expansions for `U` distinct request/metadata pairs | `O(KV)` source coverage plus `O(U)` retained expansion tuples | `pending_build_ms` and cold `serving_sources_ms` |
| Later request, unchanged live directory | Live coverage validation and reuse | `O(KT)` metadata signature plus `O(KG)` pending overlay assembly | Reuses `O(KT)` live coverage and `O(U)` expansion tuples | Reused `serving_sources_ms` |
| One request at peak | Candidate readiness and scoring | `O(GK + V + V log V)` for dense pending routes | `O(V + K)` transient wait memo and one pending route's coverage work | Part of `full_decision_ms` |
| One request at peak | Fetch materialization | `O(KV)` over pinned combined coverage | Up to `O(KV)` required-region output | `plan_fetch_ms` |
| One request at peak | Gate registration | `O(P)` | `O(P)` waiter links, up to `O(GP)` for blocked burst requests | Part of `full_decision_ms` |
| One publication | `Published` commit | `O(K + R + Wₐ)` | Releases pending requirements and waiter links | Not measured |
| One request at peak | Whole serialized decision | `O(KV + GK + V log V)` | Adds `O(K + R + P)` retained requester state plus `O(KV)` transient planning state | `full_decision_ms` |

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
`plan_fetch_python_peak_kib` phase. The benchmark does not dispatch `Published`, so
it does not cover completion cleanup or waking `Wₐ` blocked decisions.

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
