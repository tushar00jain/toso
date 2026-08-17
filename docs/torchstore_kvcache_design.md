# Mooncake-style KV-cache serving on TorchStore

<!-- Generated from torchstore_kvcache_design.diagram.xml by realsim.tools.text_diagram. -->

**Status:** proposal · **Scope:** shared LLM KV cache, cache-aware routing, and
prefill/decode serving.

TorchStore already provides the byte pool, directory, clients, and transports needed
by a distributed KV cache. The capability adds prefix-addressed blocks, a control
plane that places work using cache and load state, and bounded cache retention.

The reference is Mooncake, *KVCache-centric Disaggregated Architecture for LLM
Serving* (arXiv:2407.00079). This document maps its ideas onto TorchStore. See
[`architecture.md`](architecture.md) for the shared control/sensor/data loop,
[`torchstore.md`](torchstore.md) for store internals, and
[`des_design.md`](des_design.md) for the simulation stack.

## 1. Boundary

The design must:

- reuse a prompt prefix across serving instances, but never across incompatible model
  identities;
- choose prefill and decode hosts using prefix residency, queue state, transfer cost,
  and TTFT/TBT limits;
- keep the long-lived cache within capacity through eviction and optional tiering;
- leave ordinary TorchStore weight and object APIs unchanged.

TorchStore does not run attention, implement chunked or layer-wise prefill, own the
serving engine, or replace its transport with a separate Messenger. It exposes the
storage and scheduling seams that let an engine perform those operations. Full SLO
admission can remain an optional selector.

## 2. What is reused and what is added

<!-- text-diagram:mapping:start -->
```
┌─────────── REUSE TORCHSTORE ───────────┐   ┌─────────── ADD KV CAPABILITY ───────────┐
│ Controller: key → current holders      │   │ connector: prefix-hash block keys       │
│ StorageVolume: DRAM byte pool          │   │ ControlPlane: route + admission         │
│ LocalClient: put/get/delete batches    │   │ Sensors: queues / batches / reservations│
│ transport: local / peer / RDMA fallback│   │ eviction + replication selectors        │
└────────────────────────────────────────┘   └─────────────────────────────────────────┘
┌────────────────────────────── SERVING ENGINE KEEPS ───────────────────────────────┐
│ attention and paged KV layout                                                     │
│ prefill / decode compute  +  batching / streaming  +  request lifecycle           │
└───────────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────── MOONCAKE MAPPING ─────────────────────────────────┐
│ KVCache pool + Messenger + metadata = TorchStore; Conductor = KV ControlPlane     │
└───────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:mapping:end -->

The `Controller` remains a metadata-only directory. A KV block is an opaque key with
one or more volume holders, so existing `locate_volumes`, `put_batch`, `get_batch`,
and delete registration supply the storage path.

The KV `ControlPlane` is Mooncake's Conductor analog. It reads directory presence and
capability sensors directly, selects prefill/decode placement and fetch
sources, then commits the decision. Serving hosts execute the answer through normal
clients and report facts through the dispatcher.

Directory presence can be read synchronously for each decision. If that becomes the
hot path, `notify_put` and `notify_delete` may emit stored/removed events into a
coordinator-owned prefix index. The event feed is an optimization; the directory
remains the source of truth.

## 3. Prefix-addressed blocks

A connector chunks a prompt into `B`-token blocks and hashes a chain:

```text
k[0] = H(model_id, weight_version, dtype, quantization, tokens[0:B])
k[i] = H(k[i-1], tokens[i*B:(i+1)*B])
```

Requests with the same first `n` blocks produce the same first `n` keys. Deduplication
and prefix reuse therefore follow from ordinary key equality; TorchStore never parses
tokens or hashes content itself.

For a request's ordered keys, `locate_volumes(keys)` supplies per-block presence. The
longest consecutive run from the first key on each volume is that instance's reusable
prefix. A strong hash and all representation-changing fields must participate in the
chain so different models, weights, or precisions cannot alias.

A value is the KV tensor for one block, for example
`[2, layers, B, kv_heads, head_dim]`. The store treats it as bytes under a key; block
geometry belongs to the serving connector.

## 4. One serving request

<!-- text-diagram:lifecycle:start -->
```
┌─────────── CONTROL ───────────┐    ┌─────── PREFILL HOST ────────┐       ┌────────── DECODE HOST ──────────┐
│ 1 locate prefix holders       │    │ 5 fetch remote gap          │       │ 9 fetch complete KV chain       │
│ 2 price prefill candidates    │    │ 6 reuse local prefix        │       │ 10 join decode batch            │
│ 3 choose decode + admit       │plan│ 7 prefill uncached suffix   │address│ 11 generate remaining tokens    │
│ 4 commit route + pull source  │    │ 8 publish fresh KV + report │       │ 12 publish generated KV + report│
└───────────────────────────────┘    └─────────────────────────────┘       └─────────────────────────────────┘
┌─────────────── DIRECTORY FEEDBACK ────────────────┐   ┌──────────────────── SENSOR FEEDBACK ─────────────────────┐
│ publish / eviction change current prefix residency│   │ actions change queues, reservations, batches and load    │
└───────────────────────────────────────────────────┘   └──────────────────────────────────────────────────────────┘
```
<!-- text-diagram:lifecycle:end -->

The serving surface can stay small:

```python
response = await control.decide(request, requester)
kv = await store.fetch(response.prefill, response.plan.pull_keys)
fresh_kv = await engine.prefill(request, kv)
await store.publish(response.prefill, fresh_keys, fresh_kv)
await dispatcher.dispatch(PrefillFinished(...))
```

`decide` returns both prefill and decode placement before work begins, or rejects the
request. A request sent to another host is rerouted by address, not forwarded through
the first host. The client follows that address, and the chosen host receives the same
booked decision rather than pricing the request again against state that its own
booking changed.

The remote fetch names only the **gap** between the chosen host's local prefix and the
remote reusable prefix. After prefill, fresh blocks are published locally. A separate
decode host fetches the complete chain through the same store, making disaggregation's
KV handoff cost explicit.

## 5. Cache-aware placement

<!-- text-diagram:scheduling:start -->
```
┌──────────── READ ────────────┐     ┌────── PRICE EACH PREFILL HOST ───────┐     ┌─────────── COMMIT ────────────┐
│ prefix runs by instance      │     │ queue wait                           │     │ best prefill host             │
│ prefill queues + reservations│     │ + transfer gap or local recompute    │     │ best decode host at completion│
│ decode batches               │────►│ + uncached-suffix prefill            │────►│ priced pull source            │
│ topology + transfer costs    │     │ = predicted TTFT                     │     │ or reject before compute      │
└──────────────────────────────┘     └──────────────────────────────────────┘     └───────────────────────────────┘
┌──────────────────────────────────────────── NEXT REQUEST ─────────────────────────────────────────────┐
│ committed reservations affect the next decision; completion replaces prediction with observation      │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:scheduling:end -->

For each prefill candidate `p`, the selector estimates:

```text
TTFT(p) = queue_wait(p)
        + read_time(remote_prefix_gap → p)
        + prefill_cost(uncached_suffix on p)
```

### 5.1 Per-request reference algorithm (serial)

```text
on decide(request):                            # one control-plane turn
  keys = prefix_hash(request); presence = locate(keys)
  best_peer, best_len = longest_prefix(presence)

  for p in prefill_pool:
    local = prefix_on(p, presence)
    reuse = best_len if remote_worth_it(best_len, local) else local
    pull = keys[local:reuse]
    ttft = queue[p] + transfer(best_peer, p, pull) + prefill(tokens_after(reuse))
    plans[p] = Plan(host=p, pull=pull, ttft=ttft, done_time=now + ttft)

  plan = min(plans, key=(ttft, prefill_load, id))
  decode = min(decode_pool, key=predicted_batch_at(plan.done_time))
  if plan.ttft > SLO_ttft or predicted_tbt(decode) > SLO_tbt: reject
  commit(plan, decode, routed_pull=(best_peer, plan.pull)); return plan, decode

on sources(pull_keys, requester):
  return recorded_routed_pull(...) or rank_holders(pull_keys, locality, load)
```

`prefix_hash` builds the request's block-key chain, and `locate` reads its current
holders. `longest_prefix` and `prefix_on` measure consecutive cached blocks;
`remote_worth_it` applies the reuse threshold. `transfer`, `prefill`,
`predicted_batch_at`, and `predicted_tbt` use the shared cost model. `tokens_after`
selects the uncached suffix, and `Plan` stores one candidate's host, pull, and times.
`commit` records the placement and reservation. `recorded_routed_pull` returns that
decision's source; `rank_holders` is the fallback ordering of current holders by
prefix coverage, locality, source load, and stable id.

The remote-prefix term is used only when the best cluster-wide match is sufficiently
longer than `p`'s local match. `balance_threshold` prevents a congestion-sensitive
transfer from replacing a similar local recompute. Offline model and machine profiles
provide the prefill and transfer estimates.

The winning prefill plan is then paired with the decode host whose predicted batch at
prefill completion best satisfies the TBT limit. The control plane commits both
choices together, so later requests see their reservations immediately.

The routing decision also records the peer whose transfer it priced. When the data
plane later asks `sources(pull_keys)`, `RoutedPullSensor` returns that peer before the
generic longest-prefix ranking. Pricing the whole prefix chain and selecting a holder
for an isolated fetch are different questions; recording the decision prevents them
from disagreeing.

The capability-specific sensor state is:

- predicted prefill queues and observed decode batches;
- reservations for prefills that will affect future decode occupancy;
- routed pulls waiting for their matching fetch;
- recent source load for spreading reads across equivalent replicas.

Directory residency stays separate from these predictions. Every new decision reads
both truths through its declared sensors.

## 6. Retention, eviction, and replication

<!-- text-diagram:retention:start -->
```
┌────────── WRITE ───────────┐     ┌────────── RESIDENT ───────────┐     ┌─────────── PRESSURE ───────────┐
│ stream block               │     │ lease: in-flight read         │     │ high watermark starts a batch  │
│ commit registration        │────►│ hard pin: never evict         │────►│ evict to low watermark         │
│ incomplete blocks invisible│     │ soft pin: last-resort eviction│     │ or demote DRAM → SSD           │
└────────────────────────────┘     │ ordinary: LRU ordered         │     │ notify directory per holder    │
                                   └───────────────────────────────┘     └────────────────────────────────┘
┌─────────────────────────────────────────── HOT PREFIX ────────────────────────────────────────────┐
│ remote reuse publishes another replica; placement spreads failure domains and source load         │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:retention:end -->

Only committed blocks are visible. A write begins, streams its value, then registers
completion; an aborted write is revoked. Reads grant a short lease so eviction cannot
delete a block while it is being fetched.

The baseline policy is watermark-triggered batch LRU: crossing a high watermark
selects enough old blocks to return to a low watermark, reducing per-put churn.
Eviction deletes the local holder and notifies the directory; another replica may
remain. Per-volume `(capacity, used)` is exposed to placement and eviction selectors.
A local prefix hit explicitly touches its blocks because no transport read occurs to
refresh their recency.

Protection tiers sit above LRU:

- an active lease or incomplete write is never eligible;
- a hard pin is never evicted;
- a soft pin is skipped on the first pass and expires without renewed use;
- ordinary blocks follow recency order.

DRAM-to-SSD demotion is an optional later tier. The directory retains a tier tag, a
hit promotes the block, and source pricing includes the slower tier. A client-local
frequency-admitted hot cache may sit in front of the shared pool without changing the
control plane.

A hot prefix can bottleneck on one holder. When remote reuse repeatedly wins, the
chosen instance pulls the missing gap and keeps it locally. Replica placement should
be policy-driven: choose distinct failure/locality domains, prefer free capacity, and
soft-pin genuinely hot prefixes. Replication is best effort and remains subject to
eviction.

If a planned source evicts before serving, the fetch is a cache miss, not corruption.
The host may re-locate or recompute the missing gap. Leases make this race uncommon;
commit markers prevent torn data.

## 7. Decode load and admission

TTFT is dominated by queueing, transfer, and prefill. TBT is a decode-batch property:
each step emits one token per batch member, and step time rises with the KV attended.
A VRAM cap bounds the batch; excess requests queue and can violate TBT.

Prefill/decode disaggregation protects decode steps from long prefills by giving the
decode pool a separate compute timeline. It also adds a full-chain KV transfer and
duplicates residency on the decode host. The placement decision must price both
effects.

Admission compares predicted TTFT and TBT with their SLOs before prefill begins. A
simple early gate reads current decode occupancy. A predictive gate rolls occupancy
forward to the candidate's prefill completion and includes already reserved prefills.
The latter avoids admitting many slow prefills against an apparently empty decode
pool. Rejection after prefill is excluded because it burns compute without serving a
request.

## 8. Correctness and failure handling

- **Identity:** prefix keys include model representation and weight version.
- **Visibility:** only committed blocks appear in location results.
- **Read safety:** leases protect in-flight fetches; a miss falls back without returning
  partial data.
- **Bounded storage:** capacity enforcement eventually reaches the low watermark
  unless every resident block is leased or hard-pinned.
- **One decision:** rerouted hosts consume the recorded placement; they do not book the
  request twice.
- **One priced source:** a routed pull is spent by exactly one matching fetch.
- **Determinism:** stable ids break equal-cost ties.

Control-plane work is `O(number of prompt blocks × candidate instances)`. Very high
request rates can shard metadata and control by the first prefix key, provided a
request's chain and reservations remain on one shard. Transfer-time error is handled
by the pull/recompute threshold; eviction thrash is handled by capacity sizing and
replication policy.

## 9. Delivery order

1. Add connector-side prefix keys and longest-prefix location queries.
2. Add peer-preferred fetch and read-through publication.
3. Add cache-aware prefill/decode placement and routed-pull recording.
4. Add capacity, leases, and LRU eviction.
5. Add hot-prefix replication and load-spread source ranking.
6. Add TTFT/TBT admission with predictive decode occupancy.
7. Add DRAM-to-SSD tiering if the capacity curve warrants it.

Open choices are block size, model-specific cost fitting, local versus globally aware
eviction, coordinator sharding thresholds, and whether KV and weight namespaces share
one store. Weight version must be part of the KV key so a weight update naturally
invalidates incompatible cache entries.

The executable routing and serving model lives in
[`kvcache_sim`](../kvcache_sim/README.md). It models bounded per-instance LRU and the
serving lifecycle; leases, pin tiers, batch-watermark eviction, and SSD promotion
remain storage-design details rather than simulation fidelity.
