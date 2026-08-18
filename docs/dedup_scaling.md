# Dedup control-plane scaling

This note defines the synchronized weight-sync workload that the dedup control
plane must survive and the benchmark used to measure it. The benchmark exercises
metadata-only TorchStore planning; payload bytes never enter the process.

## Scale dimensions

Let:

- `K` be keys requested in one state-dict batch;
- `T` be trainer/source ranks visible for each key;
- `G` be generator ranks with a read-through publication in flight;
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

The repository currently establishes these scale points:

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

| Component | Time | Peak space | Benchmark coverage |
| --- | --- | --- | --- |
| Dispatcher lookup and commit | `O(Fₐ + Wₐ)` per action | Registered action/fold pairs plus outstanding waiter links | Included in `pending_build_ms` and `full_decision_ms`, not isolated |
| `Asked` → directory pending indexes | `O(K)` per generator; `O(GK)` for the burst | `O(GK)` entries in each of two indexes; both reference the same `_PendingEntry` | Included in `pending_build_ms` and `full_decision_ms` |
| `Routed` → fan-out route state | `O(R + S)` per generator for `S` selected sources | `O(GR + GS + V)` requirements, route edges, and source loads | Included in `pending_build_ms` and `full_decision_ms` |
| Pinned live-directory snapshot | `O(KT)` | `O(KT)` transient copied mappings | `snapshot_ms` |
| Pending overlay and live merge | `O(KV)` | `O(KV)` transient `StorageInfo` mappings | Part of `serving_sources_ms` and `plan_fetch_ms` |
| TorchStore request expansion | `O(KV)` for the benchmark's one region per key/source; higher with slice intersections | `O(KV)` transient requests and region counters | Part of `serving_sources_ms` and `plan_fetch_ms` |
| Coverage regrouping | Currently up to `O(K²V)` because each key filters source-wide mixed-key regions | `O(KV)` coverage records and counters | Dominant part of `serving_sources_ms` and `plan_fetch_ms` |
| Candidate readiness and scoring | `O(GK + V + V log V)` for dense pending routes, including ordering | `O(V + K)` transient wait memo and one pending route's coverage work | Part of `full_decision_ms` |
| Fetch materialization | `O(K²V + KV)` including its own coverage construction | `O(KV)` transient coverage; up to `O(KV)` required-region output | `plan_fetch_ms` |
| Gate registration | `O(P)` | `O(P)` waiter links per blocked decision, up to `O(GP)` at the burst peak | Included in `full_decision_ms` |
| `Published` commit | `O(K + R + Wₐ)` to remove pending keys and requirements and release waiters | Releases `O(K + R + Wₐ)` pending requirements and waiter links; route/load history remains until rerouted or retired | Not measured |
| Whole serialized decision | Currently `O(K²V + GK + V log V)` | Adds `O(K + R + P)` retained state for the requester plus `O(KV)` transient planning state | `full_decision_ms` |

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

`pending_build_ms` dispatches both `Asked` and `Routed` for every synthetic generator.
`full_decision_ms` includes the requester's `Asked`, its final `Routed`, and gate
registration. These numbers include dispatcher lookup, fold invocation, and commit
bookkeeping, but they do not separate that fixed framework cost from the sensor folds.
The benchmark does not dispatch `Published`, so it does not cover completion cleanup
or waking `Wₐ` blocked decisions.

## Reusable benchmark

Run the small default case from the repository root:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control
```

One smoke run produces this shape; timings vary by host and Python build:

| case | keys | source ranks | generators | indexed metadata entries | pending build ms | snapshot ms | serving sources ms | plan fetch ms | full decision ms | candidates | pending candidates | selected sources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 2 | 8 | 208 | 0.355 | 0.006 | 0.614 | 0.631 | 2.686 | 10 | 8 | 1 |

Run the intended 8B scale or supply a deployment-specific point:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control --preset planned-8b
.venv/bin/python -m realsim.tools.benchmark_dedup_control \
  --keys 290 --source-ranks 8 --generators 256 --repeats 5
```

The dense 70B and fleet presets are opt-in. A guard rejects cases whose estimated
metadata or current coverage work is too large for a routine developer run. Use
`--allow-large` only on a host sized for the reported workload:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control \
  --preset fleet-worst --repeats 1 --warmups 0 --allow-large
```

Output is one tab-separated header and row with fixed column order. Commit or archive
that row alongside the code revision and host description; compare the same preset,
Python build, and machine before and after a change. Wall-clock values are medians for
the repeated phase measurements, while pending-state construction and the full
decision are single peak-state observations to avoid multiplying large allocations.

The tool does not measure controller RPC, payload transfer, publication latency, or
the simulation scheduler. It answers whether one serialized dedup controller can
construct a routing decision at the requested metadata scale.
