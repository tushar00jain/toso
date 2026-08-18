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
| Pending overlay and live merge | `O(KV)` | `O(KV)` transient source mappings; unchanged live `StorageInfo` is shared | Part of `serving_sources_ms` |
| TorchStore request expansion | `O(KV)` for the benchmark's one region per key/source; higher with slice intersections | `O(KV)` transient requests and region counters | Part of `serving_sources_ms` |
| Coverage regrouping and decision cache | `O(KV)` by indexing expanded regions once by key and source | `O(KV)` coverage retained until the pinned decision exits | Built in `serving_sources_ms`, reused by `plan_fetch_ms` |
| Candidate readiness and scoring | `O(GK + V + V log V)` for dense pending routes, including ordering | `O(V + K)` transient wait memo and one pending route's coverage work | Part of `full_decision_ms` |
| Fetch materialization | `O(KV)` over cached coverage | Up to `O(KV)` required-region output | `plan_fetch_ms` |
| Gate registration | `O(P)` | `O(P)` waiter links per blocked decision, up to `O(GP)` at the burst peak | Included in `full_decision_ms` |
| `Published` commit | `O(K + R + Wₐ)` to remove pending keys and requirements and release waiters | Releases `O(K + R + Wₐ)` pending requirements and waiter links; route/load history remains until rerouted or retired | Not measured |
| Whole serialized decision | `O(KV + GK + V log V)` | Adds `O(K + R + P)` retained state for the requester plus `O(KV)` transient planning state | `full_decision_ms` |

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

| case | keys | source ranks | generators | indexed entries | pending build ms | snapshot ms | serving sources ms | plan fetch ms | full decision ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 2 | 8 | 208 | 0.273 | 0.007 | 0.449 | 0.155 | 1.664 |

| pending build peak KiB | snapshot peak KiB | serving sources peak KiB | plan fetch peak KiB | full decision peak KiB | candidates | pending candidates | selected sources |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 37.688 | 3.953 | 64.719 | 9.492 | 85.367 | 10 | 8 | 1 |

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
for each phase, so its overhead cannot affect the `*_ms` columns. Each planning sample
opens a fresh pin, measures cold `serving_sources`, then measures cached `plan_fetch`
before leaving that pin. Wall-clock values are medians for the repeated phase
measurements, while pending-state construction and the full decision are single
peak-state observations to avoid multiplying large allocations. Each memory phase
runs once in the same cold-then-cached order.

The tool does not measure controller RPC, payload transfer, publication latency, or
the simulation scheduler. It answers whether one serialized dedup controller can
construct a routing decision at the requested metadata scale.
