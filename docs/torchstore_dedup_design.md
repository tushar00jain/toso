# Replica-aware weight-transfer dedup

<!-- Generated from torchstore_dedup_design.diagram.xml by realsim.tools.text_diagram. -->

**Status:** proposal · **Scope:** trainer-to-generator weight sync for RL.

TorchStore can turn a synchronized burst of replica reads into a shallow,
load-balanced transfer tree. Readers need no cohort, barrier, or model-parallelism
metadata: each asks normally, while a control plane prices present and promised
sources by readiness, transfer time, fabric cost, and load.

This document covers the capability-specific design. See
[`architecture.md`](architecture.md) for the shared control/view/data feedback loop,
[`torchstore.md`](torchstore.md) for TorchStore internals, and
[`des_design.md`](des_design.md) for the simulation stack.

## 1. Boundary

The design must:

- target one trainer transfer per unique region without extending a peer branch past
  the point where a fresh trainer transfer is cheaper;
- bound source fan-out and spread transfers across eligible trainer and generator
  volumes;
- preserve TorchStore resharding across different trainer and generator meshes;
- derive replica classes from `TensorSlice` geometry, not model structure;
- keep ordinary `put`, `get`, and state-dict APIs as the fallback path.

It does not provide tensor transforms across names, fusion layouts, or transposes;
optimize the final microseconds of a fused one-hop engine; or change the
`direct_rdma` handle path.

## 2. Capability shape

<!-- text-diagram:shape:start -->
```
┌────────────── CONTROL ──────────────┐      ┌──────────── DATA ─────────────┐      ┌────────── TORCHSTORE ──────────┐
│ Dedup ControlPlane                  │      │ ReadThroughPlane              │      │ Controller: current holders    │
│ source rank + readiness + load      │answer│ preferred get → local put     │─────►│ LocalClient: slice planning    │
│ FanoutSensor: tree / promises / load│      │ dispatch Stored               │      │ StorageVolume: resident bytes  │
│ reads directory through View        │◄─────│ moves bytes through Deployment│─────►│ transport: peer / origin copy  │
└─────────────────────────────────────┘      └───────────────────────────────┘      └────────────────────────────────┘
┌──────────────── DIRECTORY TRUTH ─────────────────┐   ┌──────────────────── SENSOR TRUTH ────────────────────┐
│ local put registers reader as a current holder   │   │ Stored settles the promised copy and wakes dependents│
└──────────────────────────────────────────────────┘   └──────────────────────────────────────────────────────┘
```
<!-- text-diagram:shape:end -->

The `Controller` remains the authority for current residency. The dedup
`ControlPlane` holds only the planned fan-out tree, source load, and readiness of
copies that have not registered yet. Its selectors read both through a `View`; they
move no bytes.

The data plane executes an ordinary preferred `get`, writes the result into the
reader's co-located volume, and reports `Stored`. The put changes directory truth;
the action settles the planned copy in `FanoutSensor`. A waiting reader is answered
only when both agree that its source can serve the key.

Every generator already has a volume under `LocalRankStrategy`, so read-through
population needs no new storage actor.

## 3. How one burst becomes a tree

<!-- text-diagram:burst:start -->
```
┌─────────────────── ROUTES ────────────────────┐   ┌────────────── WHY EACH EDGE WINS ──────────────┐
│ t0 and t1 both hold A and B                   │   │ A first pull: t0 wins the stable tie           │
│                                               │   │ B first pull: t1 wins after t0 is reserved     │
│ A: expensive trainer fabric                   │   │                                                │
│ t0 ──► g0 ──► g1                              │   │ g1: wait(g0) + peer hop < direct t0 hop        │
│          └──► g2                              │   │ g2: g0 has a slot; sibling keeps depth at 2    │
│                                               │   │                                                │
│ B: cheap direct path from t1                  │   │ g1′: promised g0′ beats direct t1              │
│ t1 ──► g0′ ──► g1′                            │   │ g2′: direct t1 hop < peer wait + hop           │
│  └───────────► g2′                            │   │      so B starts a second shallow root         │
└───────────────────────────────────────────────┘   └────────────────────────────────────────────────┘
┌────────────────────────────────────────────── RESULT ───────────────────────────────────────────────┐
│ one burst balances trainers and tree depth: A moves 1× from origin; B moves 2× when latency wins    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:burst:end -->

The production control service is serialized. Requests therefore acquire a total
order even when the reads arrive together. The first reader selects a trainer volume
and becomes a promised peer; later decisions may select that promise before its bytes
arrive. This actor mailbox is the rendezvous, so the design needs neither a declared
world size nor a registration barrier.

A readiness gate withholds a peer-routed answer until the peer's local put registers.
The waiting client then performs a normal `get` against a truthful directory; the
selector never invents residency.

`fanout_cap` bounds the number of readers planned behind one peer. Larger caps form a
shallower tree and allow sibling transfers to overlap. The score also stops a branch
from growing merely to preserve 1x bytes: a reader starts another trainer-rooted
branch when its direct path is cheaper than the peer's readiness wait plus hop.

The diagram combines both choices. `t0` and `t1` hold A and B; reserving A on `t0`
makes `t1` the lower-load first source for B. A's expensive trainer path keeps later
readers behind nearby `g0`, with fan-out producing `g1` and `g2` as siblings. For B,
`g1′` follows promised `g0′`, but the cheap `t1 → g2′` path beats waiting for another
peer transfer and starts a second root. A uses 1x origin bytes; B pays 2x to avoid a
longer or slower branch.

## 4. Source selection

For every requested region, one ranking considers:

- current holders from `Controller.locate_volumes`;
- readers already routed to fetch the region, while they still owe their local put;
- locality and transfer cost from the shared topology view;
- planned load and the fan-out cap.

The useful cost is expressed in seconds, not abstract topology tiers:

```text
score = source_ready_wait + transfer_time + fabric_weight * transfer_time
```

`source_ready_wait` is zero for a holder and the remaining branch cost for a promised
peer. A high `fabric_weight` favors a nearby peer and minimizes origin traffic; a low
direct-transfer time can still make a trainer the faster source. Active source load
breaks otherwise similar choices, and `fanout_cap` prevents one early reader from
serving the whole burst. The first pulls of different regions therefore spread across
trainer volumes, while later reads extend only the peer branches that remain cheaper.

Stable volume-id tie-breaking makes equal-cost decisions deterministic. An explicit
source preference is passed into the ordinary read planner; if the preferred peer
evicts before the read, later ranked sources remain valid fallbacks.

### 4.1 Per-request algorithm (serial, so race-free)

```text
on plan(reader r, regions):                    # one control-plane turn
  for R in regions:
    present = directory_holders(R) - {r}
    promised = promised_peers_below_cap(R) - {r}
    src = min(present | promised, key=(score, active_load, stable_id))

    promise(R, r)                              # r will publish after its get
    reserve_edge(src, r, R)                    # charges fan-out and source load
    answer(R, src, wait=ready(src, R) + slot(src))

on Stored(reader r, region R):
  settle_promise(R, r); release_incoming_edge(r, R); wake_waiters(R)
```

`directory_holders` returns registered sources; `promised_peers_below_cap` returns
readers whose copies are in flight and still have fan-out capacity. `score` is the
readiness-plus-transfer formula above; `active_load` and `stable_id` are its load and
deterministic tie-breaks. `ready` and `slot` are the source-copy and fan-out wait gates.
`promise` and `reserve_edge` record the planned copy and load; `Stored` settles both.

Every request is decided against all earlier promises and reservations. The source
load is therefore part of the same atomic decision as the route; two readers cannot
both observe the last free slot.

## 5. Regions, versions, and store changes

Dedup operates on global tensor regions. Identical `(offsets, local_shape)` slices are
one replica class even when their mesh coordinates differ. Four store capabilities
support the control plane:

1. **Read each region once.** Collapse identical stored slices during request
   planning instead of fetching every replica.
2. **Accept preferred sources.** A read may rank any volume that currently holds the
   region, including a generator peer.
3. **Populate on read.** A reader publishes the fetched region into its own volume.
4. **Rank by locality and load.** Same-host and peer links can beat a remote trainer
   without special cases in the client.

Optional put-side de-replication splits a replicated writer's shared shard across its
replicas. Existing assembly reconstructs the full value. This reduces put traffic and
storage footprint but is independent of read routing.

State-dict `MAPPING` is the commit and version boundary. A new marker invalidates the
previous version's planned and cached copies. Copies stay pinned within one sync
window; optional deletion reclaims them after the version changes.

## 6. Correctness and failure handling

- **Fabric objective:** peer routing reaches 1x origin bytes while its scored wait and
  hop remain cheaper; any extra trainer root is an explicit lower-latency choice.
- **No torn source:** a peer becomes readable only after its put has registered and
  `Stored` has settled the promise.
- **Deterministic order:** serialized decisions and stable tie-breaks define the same
  tree for the same inputs.
- **Load bound:** reservations count before the answer is returned, and a source never
  serves more than `fanout_cap` readers concurrently.
- **Resharding:** source choice changes where bytes come from, not TorchStore's slice
  intersection and assembly.
- **Puller failure:** a promise needs a timeout. On expiry, cancel its dependent routes
  and designate another reader or trainer source.
- **Stale version:** reads remain gated by the committed `MAPPING`; version changes
  discard promises before any new plan is issued.

The online tree can be deeper than a global optimum, and one control service may
back-pressure at very wide fan-in. Key-hash sharding is the scale-out path, provided
all requests for one region share a shard.

## 7. Delivery order

1. Collapse identical slices on read.
2. Add preferred peer sources, locality pricing, and read-through population.
3. Add the dedup control plane, readiness gates, load-aware routing, and fan-out
   accounting.
4. Add optional put-side de-replication.

Open choices are the default fan-out policy, request-count versus byte-weighted source
load, topology detail beyond host locality, automatic versus opt-in read-through,
promise timeout behavior, and the exact state-dict epoch source for non-atomic
multi-key writes.

The executable design and its invariant tests live in
[`dedup_sim`](../dedup_sim/README.md).
