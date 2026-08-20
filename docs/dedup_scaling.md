# Dedup control-plane scaling

This note defines the synchronized weight-sync workload the dedup control plane must
survive and the benchmark used to measure it. The benchmark exercises metadata-only
TorchStore planning; payload bytes never enter the process.

## Scale dimensions

Let:

- `K` be keys in each generator's request;
- `T` be trainer/source ranks indexed for each requested key;
- `G` be generator requests in the synchronized burst;
- `V = T + G` be indexed volumes at the peak of the burst;
- `D` be distinct pending batch shapes across the burst;
- `C` be publications retained in one requester's greedy cover (`C = G` for the
  benchmark's full-span probe);
- `Wₐ` be waiters released when action `a` commits.

Every generator asks for all `K` FQNs before earlier generators publish. Generator
rank `g` requests offset `g` in a one-dimensional `G`-rank mesh, and trainer rank `t`
publishes offset `t` in its own `T`-rank mesh. The pending burst therefore has `D = G`:
each generator has a distinct DTensor placement signature. The peak probe covers the
full generator span and carries an unused coordinate, so it overlaps every generator
without matching a pending shape bucket.

Sliced-placement measurements below use `K/T/G`. Allocation is the largest additional
Python peak reported for any measured phase, not process RSS.

| Workload (`K/T/G`) | Peak decision | Whole burst, `G` × peak | Declare burst | Largest Python phase peak | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| `smoke` synthetic sanity (`8/2/8`) | 1.4 ms | 11 ms | 0.4 ms | 0.06 MiB | Trivial |
| `1b` Llama-1B-class (`120/2/8`) | 10 ms | 83 ms | 3.4 ms | 0.84 MiB | Comfortable |
| `8b` Llama-8B-class, JSON baseline (`290/8/16`) | 51 ms | 0.81 s | 17 ms | 3.9 MiB | Comfortable |
| `70b` Llama-70B RL (`723/8/64`) | 500 ms | 32.0 s | 562 ms | 38.8 MiB | Burst latency matters |
| `70b-wide` 70B with wider fanout (`723/64/128`) | 994 ms | 2.12 min | 786 ms | 79.9 MiB | Expensive burst |
| `405b` Llama-405B (`1,500/32/128`) | 2.02 s | 4.31 min | 1.50 s | 168.8 MiB | Expensive burst |
| `moe` Mixtral / DeepSeek scale (`3,000/32/128`) | 4.59 s | 9.79 min | 3.54 s | 321.4 MiB | Expensive burst |
| `fleet-worst` fleet-scale envelope (`5,203/128/512`) | Projected ~31.8 s † | Projected ~4.53 h † | Projected ~24.6 s † | Projected ~2.18 GiB † | Guarded; unsupported |

† extrapolated by `K·G` from the `moe` row rather than measured; the guard rejects
the workload by default.

The whole-burst column is `G` × the full-span peak decision and is an **upper bound**
for a serialized controller. It repeats the most expensive coverage walk for every
generator; local-shard requests can terminate after covering one rank slice. Nothing
inside one decision overlaps: it dispatches its own `Asked` and `Routed`, so the fold
work is already included.

Declare burst is the setup cost -- constructing all `G` pending publications and
staging their routes into the fan-out sensor before the peak decision runs. It is not
an addition to the whole-burst column: it is the state that column priced against.

The measured rows are synthetic capacity points, not observations of a particular
deployment. The fleet row extrapolates from `moe`: 1.7 times the keys and 4 times the
burst, with the same `T/G` ratio. The peak probe scans `K·T` live slots, probes `D = G`
pending shapes, and walks `K·G` generator slices for coverage. The declare burst
carries `G·K` writes. The projected trie holds 3,329,920 indexed rows. The default
benchmark guard prevents running it accidentally. Production runs should substitute
their state-dict key count, trainer mesh size, and synchronized generator count.

## State and work at the burst peak

The tables assume one independently serviceable region per `(key, source)`. Slicing
increases the per-key candidate set without adding new state kinds.

### Whole burst

| Component | Total time across `G` requests | Peak or retained space | Benchmark coverage |
| --- | --- | --- | --- |
| Declare | `O(GK)` trie-slot inserts, one shape bucket per distinct shape | `O(GK)` pending entries in the unified trie, `O(G)` publication records, one shape reference per publication | `declare_burst_ms` |
| Route staging | `O(G)` load edges for the constructed pending burst | `O(G)` charged load | `declare_burst_ms` |
| Serving union across `G` decisions | `O(G·(K·T + D))`; each distinct pending shape runs one overlap check | Flat `frozenset[Publication]` per decision; publication records reused across decisions | Peak decision `union_ms` |
| Total burst construction | `O(G·(K·T + D + K·C))`, with `D = C = G` at the peak probe | `O(K·(T+G) + G + V + GC)` | `declare_burst_ms` + peak decision |
| Generator completions (`Published`) | `O(GK)` trie-slot retirements plus `ΣWₐ` waiter releases | Each completion clears its publication and its waiter links | Not measured |
| Total lifecycle including completion | Adds `O(GK)` retirement work over the burst construction | Same construction peak; completion releases the pending state | Not measured |

A generator completion means its read-through finished, its local `put_batch` landed
the keys it fetched, and it dispatched `Published(publication)`. Each landed put writes
the `pub_id=0` live entry beside pending entries in the same trie slot; `Published`
retires the publication's positive id, releases the load charge, and wakes every gate
that named this publication. Retirement stays inside the `O(GK)` term: it is one pass
over the publication's own keys.

### One request at peak

This is one full-span probe evaluated after the benchmark has constructed all `G`
pending publications above.

| Component | Time | Peak space | Benchmark coverage |
| --- | --- | --- | --- |
| Declare the requester's publication | `O(K)` trie-slot inserts | `O(K)` pending entries in the unified trie plus one publication record | `declare_ms` |
| Serving union | `O(K·T + K·D)`; each of the `D = G` non-matching shape buckets iterates its `K` entries running `_overlaps` until it finds an overlapping entry or exhausts the shape, and the declare-probe's own bucket resolves in one identity check | `O(G)` flat set of `(pub_id, volume)` tuples | `union_ms` |
| Candidate scoring | `O(G)` `arrival` reads plus `O(G)` `read_time` calls | `O(G)` priced tuples | `rank_ms` |
| Balance and ordering | `O(G + G log G)` | `O(G)` keyed selection | `rank_ms` |
| `greedy_cover` (per key, per slice via the shared walker) | `O(K·G)` region-overlap checks; the full-span probe needs every generator shard | `O(G)` chosen publications, `O(K·G)` covered-region set | Part of `full_decision_ms` |
| Route dispatch and arrival record | `O(C)` load edges plus one arrival float | `O(1)` retained on the requester publication | `full_decision_ms` |
| Gate registration | `O(C)` waiter links | `O(C)` links, up to `O(GC)` across a blocked burst | `gate_ms` |
| Total synchronous control decision | `O(K·(T + D) + G log G + K·G)`; with `D = G` the union and the greedy walker each contribute a `K·G` term, so the ceiling is `O(K·(T + G) + G log G)` | Adds `O(K + G)` retained requester state plus `O(G)` transient union state | `full_decision_ms` |

The total request row ends after the route and readiness gate are constructed. It does
not include waiting for a pending publication, transferring payloads, or landing the
fetched batch.

Peer readiness is `O(1)` per publication: an arrival score is computed when the peer
is routed and read back through `FanoutSensor.arrival`. Dependencies point at
publications declared earlier in the serialized decision stream, so the score is
defined by construction with no readiness walk at decision time.

The benchmark exercises `D = G`: every generator publication occupies a distinct
shape bucket, and the full-span probe matches none of them exactly. Pending discovery
therefore visits every bucket and runs `_overlaps` once per publication. The exact-
shape fast path applies only when generators share a request shape; the hand-crafted
fast-path test covers that case separately.

The greedy walk on the control plane is `Controller.greedy_cover`. It builds ranked
source maps and delegates to the same per-key, per-slice walker the client uses for
volume requests. The peak probe spans the full generator tensor, so every key needs
all `G` rank slices and `C = G`.

`indexed_metadata_entries` counts entries in the large key-multiplied structures, not
bytes or Python objects:

```text
K·T live rows in the controller trie
+ K·G pending rows in the same trie slots
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
| smoke | 8 | 2 | 8 | 80 | 0.353 | 0.098 | 0.069 | 0.025 | 0.011 | 1.358 |

| full decision instructions | declare burst peak KiB | declare peak KiB | union peak KiB | rank peak KiB | gate peak KiB | full decision peak KiB | candidates | pending candidates | selected sources |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,762,079 | 66.484 | 11.570 | 5.953 | 4.898 | 5.629 | 40.680 | 9 | 9 | 8 |

Run the intended 8B scale or supply a deployment-specific point:

```bash
.venv/bin/python -m realsim.tools.benchmark_dedup_control --preset 8b
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

Toy scale for this section: keys `w0, w1` (`K=2`), live trainer shard `s0` (`T=1`),
generators `g0, g1` already declared (`G=2`), and full-span requester `r`.

State before `r` declares. Live and pending entries share each controller trie slot.
`pub_id=0` marks live storage; positive ids name outstanding publications. Each
generator rank has its own shape:

```text
keys_to_storage_volumes: Trie[k, {vol: {pub_id: StorageInfo}}]
  w0 -> s0:{0: live}, g0:{1: pending}, g1:{2: pending}
  w1 -> s0:{0: live}, g0:{1: pending}, g1:{2: pending}

controller sidecars
  publications
    1 -> volume=g0, keys=frozenset({w0,w1}), shape=#g0
    2 -> volume=g1, keys=frozenset({w0,w1}), shape=#g1
  shape_pubs
    #g0 -> {1}
    #g1 -> {2}
```

Ordinary reads (`locate_volumes`, `get`, `keys`, DTensor commit check) project each
slot to its `pub_id=0` entry. `serving_union` probes live slot values and shape
buckets; `greedy_cover` reads the ranked publication id directly from each slot.

`Dedup.sources([w0, w1], "r")` is one synchronous turn: nothing suspends between
declaration and gate.

```text
declare(r, [w0, w1])                    K trie-slot inserts, one publication record
    |
dispatch_sync(Asked(pub_r))             directory sensor records the publication
    |
serving_union([w0, w1])                 one directory read: K·T live slots;
                                        D=G distinct pending shape probes, each
                                        checked with _overlaps; return flat set
    |
Candidates.select
    |  price every publication           `wait = arrival[pub]`
    |  cap check pending sources         O(1) per priced publication
    |  aggregate per volume              max wait across a volume's publications
    |
Balance -> WithFold -> Ordered           G-entry keyed selection, sort G log G
    |
greedy_cover(requests, ranked)          per key, per slice via the shared walker;
                                        take all G generator shards needed to cover
                                        the full-span request -- C = G.
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
| `keys_to_storage_volumes` unified trie | Controller | `K·(T+G)` entries; `Trie[key, dict[vol, dict[pub_id, StorageInfo]]]` | Live deletes, publication retirement |
| Publication records and shape buckets | Controller | `O(G + D)` plus each publication's key frozenset | Publication retirement |
| `_arrival` | `FanoutSensor` | one float per pending publication | `Published(publication)` |
| `_behind` | `FanoutSensor` | one int per volume currently loaded | `Published`, reroute |
| `_assigned` gate bookkeeping | `FanoutSensor` | `O(G)` for outstanding load charges plus `O(GC)` for waiter links across the burst | `Published(publication)` |
| `_waiters` | `Dispatcher` | one link per (pending publication, gate) | Commit |

The largest retained item is the unified trie, `K·(T+G)` entries. There is no
per-region route requirement structure; readiness is per publication, `O(1)` per peer.

### Per-request state

Dies with the decision.

| Structure | Size |
| --- | --- |
| `serving_union` answer | `frozenset[Publication]`, `O(G)` tuples |
| Priced candidates | `O(G)` tuples after per-volume aggregation |
| Keyed selection | `O(G)` entries, rebuilt by `annotated` and `only` |
| `chosen` from `greedy_cover` | `O(G)` publications |
| `covered` region set (transient inside `greedy_cover`) | `O(K·G)` regions |
| `gate_pubs` | `O(G)` publication references |
| `ReadPlan.sources` | The ranked flat preference over `G` generator volumes |

### What that costs in calls

One decision at `8b` (`290/8/16`), counted by wrapping the sensor:

| Operation | Calls | Per key |
| --- | ---: | ---: |
| `serving_union` | 1 | |
| pending-discovery `_overlaps` | `G + 1` | |
| `arrival` reads | `G` | |
| `read_time` | `G` | |
| `greedy_cover` shared-walker overlap | `K·G` | `G` |
| `is_in_flight` (gate probe) | `G` | |

There is no cache to key, no batch-spec build, no `covers` fallback, no per-source
live expansion. One directory read, `G + 1` pending-shape checks, `G` price lookups, one
`K·G` greedy walk, and one arrival record.

## Potential improvement

In profile order against the numbers above:

- `read_time` is called `G` times per peak decision, once for each generator source.
  Caching the per-(volume, requester) cost across a burst removes one function call
  per source; the cost has no key term, so a burst that reads the same requester many
  times amortizes it.
- `G log G` in `Ordered` is a stable-order sort. Scores collide by construction --
  every price is `wait + hop(1 + fabric)` over a handful of link classes -- so a
  counting-sort over the bucketed score is `O(G)` exact, not approximate.
- Union work at the widest bursts is 87 ms at `70b-wide` and 284 ms at `moe`;
  their declare bursts are 786 ms and 3.54 s. The union combines the
  per-key live-slot walk with one overlap check for each of the `G` pending shapes.
  Interning `(key, region)` to an int shrinks retained state and speeds the reads.

## Indexed-controller comparison, 2026-08-19

These measurements use the TOSO `e50ba5f` and TorchStore `97afc30` base revisions
with the current indexed-controller integration in both worktrees. A construction
preflight resolved `legacy` to `torchstore.controller.Controller` and `indexed` to
`torchstore.controllers.indexed.controller.IndexedController` before the benchmark
ran.

Each measured row used the saved workload unchanged on the same host and Python:

```bash
TOSO_CONTROLLER_BACKEND=legacy \
  .venv/bin/python -m realsim.tools.benchmark_dedup_control --preset PRESET
TOSO_CONTROLLER_BACKEND=indexed \
  .venv/bin/python -m realsim.tools.benchmark_dedup_control --preset PRESET
```

`PRESET` was each of `smoke`, `1b`, `8b`, `70b`, `70b-wide`, `405b`, and
`moe`. The defaults were retained: one warmup and three repeats. `fleet-worst` was
not run. The benchmark reports medians for repeated phases, but burst construction
and memory phases remain single observations. In particular, `declare_burst_ms` is
noisy on a shared host and should be read as an order of magnitude.

### Measured backend comparison

Count parity shows `candidates/pending_candidates/selected_sources`; every pair is
identical. Speedup is legacy full-decision time divided by indexed full-decision
time.

| Workload (`K/T/G`) | Legacy full decision | Indexed full decision | Speedup | Legacy declare burst | Indexed declare burst | Union legacy -> indexed | Count parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke` (`8/2/8`) | 1.309 ms | 1.018 ms | 1.29x | 0.392 ms | 1.253 ms | 0.066 -> 0.046 ms | Yes (`9/9/8`) |
| `1b` (`120/2/8`) | 10.175 ms | 3.864 ms | 2.63x | 3.309 ms | 14.285 ms | 0.533 -> 0.710 ms | Yes (`9/9/8`) |
| `8b` (`290/8/16`) | 48.985 ms | 7.368 ms | 6.65x | 16.461 ms | 67.488 ms | 3.103 -> 1.610 ms | Yes (`17/17/16`) |
| `70b` (`723/8/64`) | 461.261 ms | 17.737 ms | 26.01x | 424.387 ms | 1,277.724 ms | 17.203 -> 4.190 ms | Yes (`65/65/64`) |
| `70b-wide` (`723/64/128`) | 982.389 ms | 19.172 ms | 51.24x | 779.878 ms | 2,928.057 ms | 86.186 -> 4.299 ms | Yes (`129/129/128`) |
| `405b` (`1,500/32/128`) | 2,012.535 ms | 38.061 ms | 52.88x | 1,539.403 ms | 5,727.941 ms | 109.162 -> 8.770 ms | Yes (`129/129/128`) |
| `moe` (`3,000/32/128`) | 4,573.494 ms | 71.801 ms | 63.70x | 3,411.008 ms | 13,138.340 ms | 285.156 -> 17.936 ms | Yes (`129/129/128`) |

The index moves work from the decision path to publication writes. At `moe`, the
full decision is 63.70x faster while burst declaration is 3.85x slower. Maintaining
the region index and normalized topology costs more than inserting the same rows in
the legacy trie, so the indexed backend is useful when the directory serves enough
decisions to amortize that write cost.

The decision gain comes from sharing normalized topology and source-set incidence
across compatible keys. That removes the avoidable exhaustive `K·H` controller walk;
the widening from `G=64` to `G=128` at fixed `K=723` changes the indexed decision
only from 17.737 ms to 19.172 ms. It does not remove the output lower bound. The
full-span probe still selects all `G` source publications, and a client plan that
materializes one fetch for every key and selected region still emits `K·C` entries.
The benchmark stops after metadata routing and gate construction, so these numbers
do not claim a payload-transfer or end-to-end fetch speedup.

### Indexed scale dimensions

The largest Python phase is the indexed declare-burst peak for every row.

| Workload (`K/T/G`) | Peak decision | Whole burst, `G` × peak | Declare burst | Largest Python phase peak | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| `smoke` synthetic sanity (`8/2/8`) | 1.018 ms | 8.1 ms | 1.253 ms | 0.11 MiB | Trivial |
| `1b` Llama-1B-class (`120/2/8`) | 3.864 ms | 31 ms | 14.285 ms | 1.37 MiB | Comfortable |
| `8b` Llama-8B-class (`290/8/16`) | 7.368 ms | 118 ms | 67.488 ms | 6.82 MiB | Comfortable |
| `70b` Llama-70B RL (`723/8/64`) | 17.737 ms | 1.14 s | 1.278 s | 58.40 MiB | Decision cheap; writes dominate |
| `70b-wide` 70B with wider fanout (`723/64/128`) | 19.172 ms | 2.45 s | 2.928 s | 126.49 MiB | Decision cheap; writes dominate |
| `405b` Llama-405B (`1,500/32/128`) | 38.061 ms | 4.87 s | 5.728 s | 247.53 MiB | Declaration is expensive |
| `moe` Mixtral / DeepSeek scale (`3,000/32/128`) | 71.801 ms | 9.19 s | 13.138 s | 494.12 MiB | Declaration and memory matter |
| `fleet-worst` fleet-scale envelope (`5,203/128/512`) | Projected ~0.13 s | Projected ~1.12 min | Projected ~91 s | Projected ~3.35 GiB | Guarded; unsupported |

The fleet projection uses separate models for the three quantities rather than the
legacy decision's `K·G` extrapolation:

- Indexed decision measurements at fixed `K` and fixed `G` fit
  `full_decision_ms = 0.023113·K + 0.022422·G - 0.409`. The fit predicts the
  measured scale rows from `8b` through `405b` within 10% and gives 131 ms at
  `5,203/128/512`. A 15% model band is about 0.11--0.15 s. Multiplying the peak by
  `G=512` gives the 67.2 s serialized upper bound.
- Declaration retains `K·G` publication writes. Scaling the measured `moe` burst by
  `(5,203·512)/(3,000·128) = 6.9373` gives 91.1 s. Per-write rates across
  `70b-wide`, `405b`, and `moe` imply roughly 80--95 s before host noise.
- The largest peak follows retained `K·(T+G)` indexed state. The fleet and `moe`
  points have the same `T/G` ratio, so the same 6.9373 factor gives 3.35 GiB from
  the measured 494.12 MiB. Allowing for allocator and shape-mix effects gives a
  rough 3.0--3.7 GiB range.

The projection prices controller metadata only. `fleet-worst` remains guarded
because its declaration and retained-memory costs are large even though normalized
topology sharing keeps the projected peak decision below one second.
