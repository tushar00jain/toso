# Toso: A Read-Through Cache

Weight distribution, LLM scheduling, load balancing, and admission control look like
different systems problems. Toso expresses them with one question:

> Which copy should serve this request, when will it be usable, and is using it better
> than the alternatives?

TorchStore supplies the directory, storage, transport, and tensor resharding. Toso
adds application policy. It is a library for building a cache service per use case,
not one shared cache deployment for every workload.

## One read-through loop

```text
┌──────────────────────────── OBSERVE ────────────────────────────┐
│ directory: key -> current holders                               │
│ sensors: promises, queues, reservations, load, budgets          │
└───────────────────────────────┬─────────────────────────────────┘
                                v
┌───────────────────────────── DECIDE ────────────────────────────┐
│ rank candidates -> apply cost function -> gate or reject        │
└───────────────────────────────┬─────────────────────────────────┘
                                v
┌───────────────────────────── EXECUTE ───────────────────────────┐
│ get / compute / put / route application work                    │
└───────────────────────────────┬─────────────────────────────────┘
                                v
┌──────────────────────────── FEEDBACK ───────────────────────────┐
│ publish a new copy -> report facts -> inform the next decision  │
└───────────────────────────────┴─────────────────────────────────┘
```

- A key can identify a tensor, weight shard, KV block, checkpoint fragment, or
  immutable application object.
- A source can hold the key now or be a peer whose read-through copy is in flight.
- A miss can fetch from an origin, read another storage tier, or recompute the value.
- A successful read can populate a local copy, create a replica, and change future
  placement decisions.
- The data path remains ordinary `get`, `put`, `get_batch`, and `put_batch`.

## Dedup: turning a read burst into a transfer tree

Without routing, synchronized readers all observe the same origin before any transfer
finishes:

```text
NAIVE: m origin reads                 TOSO: one read-through tree

        ┌────> g0                                    ┌────> g1
trainer ├────> g1                      trainer ───> g0
        └────> g2                                    └────> g2

origin traffic: 3 copies              origin traffic: 1 copy
```

- The first generator reads from a current holder.
- Once routed, that generator promises to publish the value locally.
- Later readers rank both current holders and promised peers as possible sources.
- A fan-out cap produces a chain at `1` or a shallower tree at larger values.
- No reader cohort, barrier, or model-parallel layout is required.

The implemented selector prices every candidate in seconds:

```text
score = ready_wait + hop_time + fabric_weight * hop_time

ready_wait    0 for a holder; remaining branch time for a promised peer
hop_time      modeled transfer time from the source to this reader
fabric_weight value of scarce link occupancy relative to reader latency
```

- A nearby peer wins when waiting for it is cheaper than another origin transfer.
- A fresh origin edge wins when extending the branch becomes too slow.
- Within the selected cost regime, only the first reader pulls from the origin.
- Every destination still receives the full value; dedup removes repeated origin
  traffic, not destination bytes.

### A readiness gate makes future sources safe

```text
B asks for W
  -> control records: B owes a local copy
  -> B starts reading W

C asks for W
  -> selector chooses B
  -> answer waits at a readiness gate

B finishes get_batch -> local put_batch -> directory registers B -> Published(B)
  -> C receives B as a usable source
```

- The gate delays the control-plane answer, not the storage operation.
- A promised source is released only after the directory confirms every requested key.
- One producer has one publication in flight, and its completion releases all
  compatible waiters without repeating the request plan in each gate.
- The local `put` turns a consumer into a source; repeated read-through fills create
  the tree.

## Replicas make the cache a load balancer

When two trainers already hold the same weight, locality alone may still overload one
of them. Toso can annotate equivalent sources with current application load:

```text
WITHOUT LOAD SPREADING             WITH LOAD SPREADING

t0 ───> g0 ───> g1                t0 ───> g0 ───> g2
t1          idle                  t1 ───> g1 ───> g3

one hot source                     one tree per useful replica
```

- `Balance` adds source load to an existing ranking.
- Dedup reuses its fan-out state as the load signal; no second load tracker is needed.
- The capability decides how load trades against transfer cost and readiness.
- Spreading spends one first-hop read per replica, reduces tree depth, and avoids a
  single serving hotspot.

This is useful for cross-cluster weight caching. Versioned weight
shards are cache keys; consuming clusters read through and retain them. The directory
then supports both reuse and load balancing across the resulting replicas.

## The reusable primitives

```text
┌───────────────┐   reads   ┌────────────────────────────────────┐
│ ControlPlane  │ <──────── │ DirectorySensor + app Sensors      │
│ selector      │           │ residency        promises / load   │
│ chains        │           └────────────────────────────────────┘
└───────┬───────┘
        │ decision
        v
┌───────────────┐  store calls  ┌───────────────────────────────┐
│ DataPlane     │ ────────────> │ TorchStore clients / volumes  │
│ app lifecycle │               └───────────────────────────────┘
└───────┬───────┘
        │ typed actions
        v
┌───────────────┐  atomic fold  ┌───────────────────────────────┐
│ Dispatcher    │ ────────────> │ affected Sensors              │
└───────────────┘               └───────────────────────────────┘
```

- **Directory sensor:** reads current holders and can pin one coherent snapshot for a
  decision.
- **Application sensors:** hold facts the storage directory cannot express, such as
  promised copies, queues, reservations, or routed load.
- **Dispatcher:** folds one reported action into every affected sensor, then commits
  the updates together.
- **Selectors:** produce candidates, add measurements, fold them into a cost, order
  them, take the best, or fall back to another strategy.
- **Data plane:** executes the decision through ordinary store and application APIs.

Most policies reduce to a workload-specific cost function:

```text
total cost = transfer + readiness wait + queueing + recomputation
             + penalties for fabric, memory, load, or priority
```

The selector algebra supplies composition and deterministic tie-breaking. The
application supplies the measurements and tradeoffs.

### Two kinds of gate

```text
candidate ranking
      |
      +--> admission bound: unacceptable? reject or fall back
      |
      +--> readiness gate: selected but not usable? wait
      v
safe, admitted answer
```

- A **readiness gate** waits for a fact that is expected to become true. Dedup waits
  for a promised cache fill.
- An **admission bound** rejects an uneconomic or unsafe result. KV serving bounds
  predicted TTFT and TBT.
- Other capabilities can bound memory pressure, deadlines, tenant budgets,
  cross-cluster traffic, or transfer-versus-recompute cost.
- A bound can filter candidates before selection or reject the winner after selection.

## Customizing Toso for another workload

```text
APPLICATION DEFINES                         TOSO REUSES

keys + miss behavior                 |      directory and topology
sensors + reported actions           |      dispatcher and lifecycle
candidate cost + admission bounds    |      selector composition
data-plane application steps         |      routed calls and deployment wiring
```

The minimum read-through path is small:

```python
requests = tuple(
    Request.from_any(key, value).meta_only()
    for key, value in entries.items()
)
plan = await control.sources.call_one(requests, requester)
client = deployment.client_for(requester)
routed = LocalClient(
    _ScopedController(client._controller, plan.by_key), client.strategy
)
values = await routed.get_batch(entries)
await deployment.client_for(requester).put_batch(values)
await deployment.dispatcher_handle.dispatch.call_one(Published(requester))
```

- Change the selector chain to change placement or source preference.
- Add a sensor when the cost function needs a new application fact.
- Add a bound for admission control.
- Add application steps around the same read-through data path.
- Keep each use case in its own deployment and policy boundary.

## KV cache: writing application logic with the same machinery

The KV implementation uses the cache to coordinate a full serving lifecycle:

```text
request
  |
  v
rank prefix holders + prefill hosts
  |
  +--> pull remote prefix or recompute locally
  |
  v
prefill -> publish KV -> choose decode host
  |
  v
fetch full KV chain if remote -> decode -> publish another copy
  |
  v
report queues, completion, occupancy, and source load
```

- Prefix-block keys turn reusable computation into directory entries.
- Each prefill candidate is priced as queue wait + transfer + uncached prefill.
- Pulling a remote prefix is chosen only when it saves enough recomputation.
- Decode placement uses predicted occupancy at prefill completion.
- TTFT and TBT bounds reject a request before it consumes compute.
- Prefill and decode hosts publish KV locally, creating sources for later requests.

The control plane therefore does more than choose bytes. It places compute, reserves
capacity, enforces SLOs, and coordinates request stages while remaining a read-through
cache underneath.

## `@routed`: placement that creates replicas

```text
client -> host A: prefill(request)
host A -> client: redirect to B
client -> host B: prefill(request)

existing holder H ──KV pull──> B ──publish──> directory adds B
decode host D     <──handoff── B ──publish──> directory adds D

later source set: H, B, D
```

- `@routed` declares where an answer carries the next host address.
- The caller repeats the same method at that host; the server does not forward to a
  peer data plane.
- The decorator does not copy data.
- Replication emerges because the routed host reads through and publishes locally.
- Later requests can route to or pull from the new holder, spreading both data and
  compute load.
- A hop cap detects cycles or a placement policy that does not converge.

This is demand-driven, best-effort replication. Copies are created by useful work and
remain subject to the cache’s capacity and eviction policy.

## Other use cases

```text
reusable work -> stable key -> rank ways to obtain it -> gate -> publish near consumer
```

- **Cross-cluster model weights:** key by model, version, and shard; optimize topology,
  startup time, replica load, and tier placement.
- **RL weight synchronization:** add freshness and bounded fan-out while retaining
  TorchStore’s resharding path.
- **Checkpoint restart:** prefer warm distributed-memory copies over durable storage
  and spread recovery reads across replicas.
- **Dataset preprocessing:** compare transfer cost with decode or transform cost and
  admit only reusable results.
- **Compilation artifacts:** key by graph, shapes, compiler, and target architecture;
  compare build time with fetch time.
- **Immutable object caching:** combine capacity, eviction, shard placement, and
  replica-aware load balancing.

These are mappings of the model, not claims that every capability is implemented.

## What exists today

- `dedup_sim` implements bounded fan-out, cost-based source routing, readiness gates,
  read-through population, eviction-aware recovery, and optional replica spreading.
- `kvcache_sim` implements prefix reuse, pull-versus-recompute pricing, prefill/decode
  placement, source spreading, bounded storage, TTFT/TBT admission, `@routed`
  prefill, KV handoff, and publication on prefill and decode hosts.
- `proposed` contains the shared planes, sensors, dispatcher, selector algebra, gates,
  routed calls, and deployment interfaces.
- `realsim` exercises the real TorchStore directory, client planning, volumes, and
  transport under a deterministic virtual clock and target-machine cost model.

These are executable design implementations, not a production-owned cache service.
Together they show that one read-through substrate can express a small transfer
optimization and a complete cache-aware application lifecycle.
