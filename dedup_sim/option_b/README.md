# Option B: precomputed application routes

Option B exposes three production classes with separate responsibilities:

```python
from dedup_sim.option_b import OptionBClient, OptionBPlan, OptionBService

plan = OptionBPlan.build(publishers, requesters, element_sizes)
plan.for_rank(rank).save("routes.json")

local_plan = OptionBPlan.load("routes.json")
services = await OptionBService.spawn(mesh, strategy)
client = OptionBClient(rank, local_plan, services.slice(...), services)

await client.publish(snapshot)  # trainer rank
value = await client.get(key, destination)  # generator rank
```

The plan is built once and then distributed. A rank may receive the complete
plan or the smaller result of `plan.for_rank(rank)`; it does not recompute global
routing during startup or an update. Geometry records and route actions remain
private implementation details. Simulation code lives in `dedup_sim/workload`.

## Route construction

`OptionBPlan.build(...)` consumes each rank's real TorchStore `TensorSlice`
metadata and tensor element sizes:

```text
(rank, role, key, TensorSlice)
  -> group generator slices with identical byte geometry
  -> partition each distinct generator slice at overlapping trainer boundaries
  -> choose one covering trainer per segment using assigned byte load
  -> choose one ingress generator per replica group and relay to its peers
  -> install immutable local send, receive, and readiness actions
```

Replication compares `(global_shape, offsets, local_shape)`. Mesh coordinates
are placement metadata, so DP replicas may have different `coordinates` and
`mesh_shape` while requesting the same bytes.

There are no TP-, DP-, model-, key-, or rank-specific branches in the route
compiler. Missing coverage, inconsistent shapes, and inconsistent element sizes
fail during setup. Every rank can build the same deterministic tables and keep
only its own entry; no central planner participates in an update.

Trainer weights may use any axis-aligned rectangular partition or replication
expressible with `TensorSlice`, including uneven shards, multiple slices per
rank, and cross-axis resharding. Generator requests may use any such sharding
or replication as well; the planner does not interpret TP, DP, PP, EP, or other
parallelism labels.

Generator read-through applies whenever multiple generator ranks request the
same key and exact byte geometry. DP is the common case, but it is not a special
case in the algorithm: weights replicated across any generator mesh dimension
can share a read-through. Disjoint TP or EP shards route independently, while
weights replicated across those ranks may relay. Partially overlapping
generator slices are currently planned independently.

Route selection is deterministic but heuristic. It greedily assigns each
segment to the eligible trainer with the least assigned bytes and chooses the
least-loaded generator as the replica group's ingress. One trainer ingress per
exact replica group minimizes trainer-origin bytes for this two-hop design, but
the algorithm does not globally minimize completion time, transfer count, or
network contention, and it does not yet include topology or bandwidth in its
cost.

## Runtime

`OptionBService` owns direct volume I/O and readiness state. `OptionBClient`
receives a plan plus the rank-local service and exposes stable `publish`/`get`
operations:

- put slices directly into a rank's local volume;
- get bytes directly from the source volume already named by a route;
- broadcast a readiness-only message to the peer clients named by the route.

Each service keeps its own wait state behind its `notify_ready` endpoint; there
is no central readiness map or control-plane service. Like a TorchStore client,
`OptionBClient` receives its remote handles at construction: the rank-local
service for direct I/O and the complete service mesh for readiness broadcasts.
Tensor transports are constructed from the strategy's cached volume handles.

The broadcast is a signal only. The ingress generator first receives and stores
its complete slice, then invokes the peers' readiness endpoints. A DP peer's
`get` waits in its rank-local service and fetches the bytes directly from the
precomputed generator source.
Broadcasting the bytes and then calling `get` would transfer the slice twice;
performing a normal TorchStore lookup would restore the per-update control path
that Option B removes.

## Simulation result

The ordinary one-key dedupe scenario also runs through `OptionB`:

| Path | Origin bytes | Total delivered | Completion |
| --- | ---: | ---: | ---: |
| Naive | 192 B | 192 B | 0.0195 s |
| Online dedupe, fan-out 2 | 64 B | 192 B | 0.0401 s |
| Precomputed `OptionB` | 64 B | 192 B | 0.0327 s |

Both dedupe implementations reduce origin traffic from 3× to 1×. The online
path chooses sources from live directory state; `OptionB` installs the same
one-ingress tree before the burst and performs only local lookups during it.

The integrated scenario reuses the existing `WeightSync` workload, `Run`,
`ItemDispatch`, real TorchStore volumes and transport lifecycle, virtual clock,
machine profile, ledger, and shared-egress resource model:

```text
PYTHONPATH=. .venv/bin/python -m dedup_sim option_b
```

For Qwen3.6-27B (55.6 GB), TP=4, DP=2, effective trainer-to-generator
bandwidth of 17.5 GB/s, and generator relay bandwidth of 900 GB/s:

| Path | Trainer bytes | Generator relay bytes | Completion |
| --- | ---: | ---: | ---: |
| Every generator reads trainers | 111.2 GB | 0 GB | 6.354 s |
| Option B | 55.6 GB | 55.6 GB | 3.193 s |

```text
55.6 GB / 17.5 GB/s + (55.6 GB / 4) / 900 GB/s = 3.193 s
```

The simulation adapter constructs the existing TorchStore transport directly
for the source volume named by the local route. Trainer publication and
generator read-through use direct volume puts, and receives use direct volume
gets. They execute the normal transport and storage-volume lifecycle but never
call controller lookup or publication; the scenario asserts that the controller
directory remains empty.

The scenario represents the trainer side as one aggregate constrained egress
endpoint. If trainer ranks have independent links, the same route tables and
resource model allow their assigned segments to progress in parallel.

## Files

```text
option_b/
  __init__.py    # exports the three production classes
  plan.py        # OptionBPlan: build, inspect, save, load, and shard the plan
  service.py     # OptionBService: direct volume I/O + readiness endpoint
  client.py      # OptionBClient: stable publish/get API
  _model.py      # internal TensorSlice geometry and route actions
  _routing.py    # internal multidimensional route compiler
  tests/         # geometry, client, validation, and integrated scenario tests
```

Run the focused tests with:

```text
PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/option_b/tests -q
```
