# kvcache_sim -- cache-aware KV-cache serving on the real TorchStore directory

A single-threaded, deterministic simulation of the cache-aware KV-cache serving
design in `../docs/torchstore_kvcache_design.md`: a cache-aware coordinator layered
over TorchStore's storage volumes + transport, doing prefix-hash addressing,
cache-aware routing, hot-block replication, LRU eviction, and batched decode.

It runs the scheduling/decode/cache algorithm on the **real** pieces via `realsim`:

- **Real directory.** KV-block presence is the real `torchstore.controller.Controller`
  directory (`keys_to_storage_volumes`), driven off-actor through `realsim`'s
  `RealControllerAdapter` / `LocalControllerHandle`. A KV block is a directory **key**
  (the prefix-hash chain string); "instance X holds block K" is the directory entry
  `K -> volume_X`. Routing consults the real `locate_volumes`.
- **Real clients + real tensors.** Each serving instance is a real storage volume
  with a co-located real `LocalClient` (`realsim`'s `RealClientAdapter`). Prefill
  returns the KV it produced -- one `torch.Tensor` per block, `device="meta"`, so it
  has the model's dtype and exact byte count and **zero** storage -- and publishing
  it is a real `put_batch` of those tensors, which records presence in the real
  directory. Being tensors is the point: `put_batch` types its value, sending a
  `Tensor`/`DTensor` down `Request.from_any` and anything else down
  `Request.from_objects`, so a run that published a descriptor exercised the object
  path no KV deployment takes. A remote-prefix pull is a real `client.get_batch`
  driven through `realsim`'s transport seam and it hands the KV back; eviction
  removes presence via the real `notify_delete_batch`. A KV block is a directory key
  holding a tensor -- real types throughout, with no translation layer.
- **Real cost model.** Every duration -- prefill compute, decode-step time, and the
  fabric/storage/RAM cost of a KV fetch -- is charged through
  `sim_common.cost_model` from a target-machine `MachineProfile`, never measured on
  the box running the sim.
- **Real async engine.** The whole request lifecycle runs on `realsim`'s
  deterministic virtual-clock `AsyncEngine`, so torchstore's real `async` client
  code executes under simulated time, single-threaded and reproducibly.

"Time" is a unitless simulated clock; the only randomness is one **seeded** RNG in
the synthetic workload, so the same seed produces a byte-identical trace and
identical metrics. It exercises the *algorithm* and gives a *sense of relative
performance* -- it is **not** a vendor benchmark.

## Environment

Run everything from the repo directory (the parent of `kvcache_sim/`) with the
repo's virtualenv, which has `realsim` + `torchstore` importable:

```
cd toso
PYTHONPATH=. .venv/bin/python -m kvcache_sim
```

## How to run

```
python -m kvcache_sim                  # all scenarios: INFO summaries only
python -m kvcache_sim -v                # add the per-event trace (DEBUG)
python -m kvcache_sim shared_prefix     # run a single scenario
python -m kvcache_sim hotspot -v        # one scenario, with the trace
python -m kvcache_sim --help            # usage + valid scenario names
```

- Positional `scenario` is one of `shared_prefix`, `eviction`, `hotspot`,
  `overload`, `disaggregation`, `early_rejection`; omit it to run all (plus a
  closing takeaway).
- `-v` / `--verbose` / `--debug` raises the log level to DEBUG so the `(a)` event
  trace prints (capped to the first 60 events per scenario); the default INFO level
  prints only the `(b)` summaries.

## The scenarios

- **shared_prefix** — many conversations share a hot system prompt + per-conversation
  context. Cache-aware routes same-prefix requests to the instance holding the prefix
  (or pulls it once), so shared prefixes are computed ~once; load-balance scatters
  them and recomputes. Higher hit rate, less prefill compute, lower TTFT.
- **eviction** — sweeps per-instance cache capacity and prints the hit-rate curve: it
  rises as the hot working set fits, then plateaus. Too-small caches can't even hold
  a full prefix ⇒ no reuse.
- **hotspot** — one dominant conversation (extreme Zipf skew). Compares load-balance
  vs cache-aware **without** replication vs **with** replication. Replication lowers
  prefill compute and p90 TTFT at the cost of KV fabric bytes.
- **overload** — high arrival rate with a TTFT SLO. Prefix reuse shortens prefill,
  freeing capacity, so cache-aware sheds fewer requests than load-balance.
- **disaggregation** — batched decode under a TBT target. A dedicated decode pool (its
  own compute timeline) protects served-request TBT from prefill interference;
  coupling prefill and decode on the same instances lets a prefill collide with a
  decode step, so a fraction of the *same* served load blows the target. What it also
  shows, since the KV handoff goes through the store, is the bill: every request pays
  a real `get_batch` of its whole block chain to reach the host that decodes it, and
  the `KV handoff bytes` row is what that costs. See the honesty note on where that
  cost does and does not land.
- **early_rejection** — heavy decode load under a tight TBT SLO, comparing admission
  policies `off`/`early`/`predict`. `off` late-checks decode load after prefill and so
  wastes prefill on rejects; `early`/`predict` gate before prefill (no waste), but only
  `predict` routes decode by the load foreseen at prefill completion, so it holds the
  SLO where `early`'s stale snapshot cannot.

## Testing

```
PYTHONPATH=. .venv/bin/python -m pytest kvcache_sim/tests -q
```

The tests are deterministic: they assert on block presence in the **real
directory** (publish → `locate_volumes` → evict), on the outcome (hit rate, compute,
eviction bounds, rejections, TBT), and on byte-identical traces across runs -- never
on wall-clock timing.

## A request is redirected, not forwarded

No serving host calls another serving host. A host answers with an *address* and the
client goes there:

```
client -> A          "serve this"
A      -> client     "prefill is B"     # A asks the coordinator; A does not call B
client -> B          B prefills, publishes its KV blocks to the store
B      -> client     "decode is C"
client -> C          C fetches that KV back out of the store, decodes, finishes
```

Three consequences, and they are the reason for the shape:

- **A host's whole outward surface is the store and the coordinator.** There is no
  peer lookup and nothing to wire up after all the hosts exist. `workload/_serving.py`
  holds the client that walks the chain, which is where a client belongs.
- **No measurement row crosses a host boundary.** Each host records what *it* did into
  the run's ledger (`report/metrics.py`), keyed by request id, and the ledger joins the
  two halves — the prefill host's routing/reuse/publish facts and the decode host's
  handoff and inter-token gaps. Shipping a half-filled `RequestResult` from one host to
  another was telemetry pretending to be a payload.
- **The store *is* the handoff.** A decode host that did not prefill the prompt has
  none of its KV, so it fetches the whole block chain with a real `get_batch`, priced
  by the same cost model as every other transfer. That is the dominant cost in a real
  prefill/decode-disaggregated system, and it used to be a free method call.

## The user-facing entry point mirrors the store

The only calls a "serving engine" makes are:

```python
plan = await coordinator.decide(Route(request))        # route; None => rejected
...                                                    # pull remote prefix + prefill
completion = await coordinator.decide(Published(plan))  # which blocks to publish/evict
await store.publish(completion.instance, completion.publish)
busy = await coordinator.decide(PrefillFinished(completion.instance, now))
```

That is the whole `Coordinator` port: two members, `decide` and `observe`, with this
application's questions carried as values (plus `AdmitDecode`, and the `ComputeBusy` /
`DecodeState` facts). It is deliberately all a serving host may touch: control
holds every instance's queue, cache and decode occupancy, so it runs as a service,
not here. Everything crossing is a value, which is what lets the in-process call
become an actor endpoint without either side changing shape. The scheduler only
ever *decides*: it reads the real directory through a view (`locate_volumes`),
returns a plan, and is told the outcome. Remote-prefix pulls
(`client.get_batch`), publishing (`client.put_batch`) and eviction
(`notify_delete_batch`) are the data plane's, layered over the existing
`put`/`get` plumbing.

## Layout

Split by plane: `control/` decides, `data/` executes, and neither imports the
simulator — `control/` takes a `View`, a `TransferCost` and machine facts from
`domain`; `data/` calls torchstore APIs against a `Deployment`. Both are enforced
by `realsim/tools/check_contract.py`, which also forbids either of them from
importing `workload/` -- that is the run's scaffolding and has no counterpart in
production, so a type all three planes pass (`Request`) belongs in `control/`.
The test for which folder something belongs in is **does it advance the clock or
move bytes?** — the decode engine sleeps and
emits tokens, so it is data; the LRU only picks victims, so it is control; a
directory read is control even though it awaits.

```
kvcache_sim/
  control/                # DECIDES -- moves nothing, holds no client
    scheduler.py          #   LoadBalance (baseline) + CacheAware coordinator,
                          #   behind proposed.Coordinator, the port the data
                          #   plane calls:
                          #   prefill placement, pull-vs-recompute, SLO gates,
                          #   decode placement; owns the PREDICTED prefill queue
                          #   and its model of the decode load
    _pending.py           #   Reservations / RoutedPulls: what was decided and
                          #   not yet done. Each expires on its own terms, when
                          #   read -- so no decision method carries a sweep
    _source.py            #   LongestPrefixPolicy: the one store question
                          #   ("which peer serves this gap"), a proposed.Policy
    _view.py              #   KVView: per-instance prefix-run lengths, plus the
                          #   pinned snapshot one routing decision reads through
                          #   (underscored: the coordinator builds its own, so
                          #   nothing outside control/ names this)
    request.py            #   inference Request, carrying its prefix-hash chain
                          #   (str keys): what is decided about, and what data/
                          #   is handed
  data/                   # EXECUTES -- advances the clock, moves bytes
    serving.py            #   one ServingHost per instance, as three things a
                          #   client asks it: route (which host should prefill
                          #   this -- answered with an address, not a forward),
                          #   prefill (real pull, compute, publish -> answered
                          #   with the decode host's address) and decode (fetch
                          #   the KV back out of the store, then batch it). No
                          #   host holds a reference to another host
    _compute.py           #   Accelerator: the port an engine runs its work on --
                          #   what it costs, making it take that long, and the KV
                          #   a forward pass hands back. Both engines get one; the
                          #   SAME one is what coupling means
    _prefill.py           #   PrefillEngine: the queue a request waits behind and
                          #   the forward pass, run on the accelerator -> the KV
                          #   blocks this host now holds and did not before
    _decode.py            #   async DecodeEngine: batched, stepped decode -> TBT
                          #   (all three underscored: nothing outside data/ drives
                          #   them)
    store.py              #   publish / reuse / fetch over a Deployment's clients,
                          #   moving whatever KV it is handed. It holds no notion
                          #   of what a block is or how big one is -- that is the
                          #   accelerator's, which produces them
  workload/               # WHAT IS SIMULATED
    _generator.py         #   seeded synthetic request stream (Zipf + Poisson),
                          #   incl. the prompt's prefix-hash chain (str keys)
    _accelerator.py       #   SimulatedAccelerator: the Accelerator port, answered
                          #   by a roofline, a sleep, and one zero-storage meta
                          #   tensor per KV block at the size the scheduler priced.
                          #   Owns BLOCK_TOKENS. The one piece of the compute story
                          #   a deployment replaces outright
    _serving.py           #   KVWorkload (the request stream) + serving_plane,
                          #   the wiring a run installs around it, incl. the
                          #   client that submits a request and follows the two
                          #   redirects it gets back
    scenarios.py          #   the six Scenarios: each declares its Runs over one
                          #   request stream, and narrates the results
  report/                 # OUTCOME METRICS
    metrics.py            #   RequestResult rows on a sim_common Ledger + rendering
    summary.py            #   one realsim.Report per comparison, over those rows
  __main__.py             # `python -m kvcache_sim [scenario] [-v]`: a Demo
  tests/test_sim.py       # deterministic tests (real-directory + outcome assertions)
```

`dedup_sim/` uses the same plane split, so the two capabilities can be compared
folder by folder — see [Comparison with `dedup_sim`](../dedup_sim/README.md#comparison-with-kvcache_sim).

The async engine, the cost model, the topology/`Endpoint` skeleton, the `Trace`
recorder and the `Ledger`/report helpers live in the repo-root `sim_common/`; the
served model's flop terms, KV block bytes and token→time conversions live in
`domain/llm.py` (both planes call them: control predicts, data charges); the real
client/controller/transport seams + adapters, the `Mesh`, the `Policy` / `View` /
`DataPlane` / `Runner` / `ItemDispatch` types live in `realsim/`. This package holds only the
KV-cache decisions and the three directory verbs (`publish` / `fetch` / `evict`)
plus the prefix-run read that express KV caching on a mesh.

## Honesty notes

- This optimizes **prefix reuse / TTFT / TBT under a cost model**; absolute numbers
  are arbitrary units. Read the scenarios for *relative* wins (cache-aware vs
  load-balance) and *shapes* (hit rate vs capacity), not throughput claims.
- Blocks become reusable at prefill **completion**, not while in flight; with spaced
  arrivals the difference is small.
- A remote pull is routed on the directory snapshot at the request's arrival, but the
  fetch runs after the prefill queue; if a peer evicted a planned block meanwhile,
  the read-through fetches only what remains present (the rest is recomputed) -- the
  faithful real-directory behavior. The peer it pulls *from* is the one the
  coordinator priced: the run installs `LongestPrefixPolicy` in the directory and the
  fetch names its source, so `locate_volumes` narrows to that peer. Without it the
  client takes whichever holder the directory lists first, which for a block several
  instances hold (a shared system prompt, anything replicated) can be a different
  locality tier than the one the TTFT prediction was built on.
- **The coordinator hop is free by default.** Control is a service, reached through
  `realsim/seams/coordinator_handle.py` — so there is now somewhere to charge the round trip,
  but `--coordinator-rtt` defaults to `0` and every call is inline. Turn it up and it
  is paid out and back before prefill can start: at `0.5` on the shared-prefix
  workload, mean TTFT goes 2.56 → 4.90 and the hit rate 0.734 → 0.704, because routing
  reads a directory snapshot one hop old and a just-published prefix is not there to
  reuse yet. Both schedulers pay the same hop, so the comparison holds either way. Two
  things the seam still does not model: the one-way `observe_*` sends are delivered
  instantly (a real bus would leave control acting on a slightly stale decode picture,
  and on a coupled instance there is one per decode step), and the recorded TTFT is
  control's own prediction, so it moves with queueing rather than by exactly one RTT.
- **The directory hop is free by default too**, and it is charged the same way
  (`--controller-rtt`, one `ServiceHop` per boundary). It is the hop every capability
  crosses on every `locate_volumes` / `notify_put_batch`, the baseline included. Note
  what it does *not* move: TTFT here is control's prediction, made before any store
  call, so directory latency shows up in the run's wall clock and in later requests'
  queue waits but never directly in the TTFT column. Recording a measured TTFT instead
  would fix that and would also fold in fetch-vs-prediction divergence -- a different
  decision, not made.
- **The KV handoff is real, but it lands in neither headline column.** The decode host
  fetches the request's whole block chain through the real `get_batch`, so the bytes,
  the fabric time and the storage/RAM staging are all charged — and then land nowhere
  the disaggregation table looks directly. Not in TTFT, which is control's prediction
  made before any of it happens; and not in TBT, which is measured *between* decode
  tokens, while the handoff finishes before the first one. What it does move is *when*
  a request joins its decode batch, which changes who is batched with whom and
  therefore everybody's step times — so the numbers shift, indirectly (`mean TBT`
  0.028 → 0.029 disaggregated, 0.162 → 0.159 coupled), and the early-rejection
  scenario shifts a lot more, because later admissions mean a lower occupancy at each
  admission check. Folding the handoff into TBT instead was tried: the first token
  comes from the prefill host, so the transfer arguably *is* an inter-token gap. It
  takes both disaggregation columns to `0.0%` attainment against a target of five
  decode steps, which is a one-off migration swamping a per-token metric rather than a
  finding, and it is not how the disaggregation literature measures either
  (DistServe/Mooncake put KV migration in TTFT and keep TPOT for decode cadence). So:
  charged on the clock, reported in its own column. What is genuinely missing is an
  end-to-end latency measurement (arrival → last token), which is where the handoff
  would show up honestly. That column does not exist yet; adding it is the obvious
  next thing and is not part of this change.
- **A handoff can find nothing, and the run says so rather than pretending.** The
  publish before it is allowed to fail (a full volume) and a volume may drop a block
  in between, and `get_batch` is all-or-nothing, so either shows up as the whole
  handoff missing. The store holds the only copy in this model, so the truthful
  consequence would be that the request dies — but re-deriving KV on a decode-only
  host is a second serving path that does not exist here, so instead the request
  decodes, no transfer is charged, and `handoff misses` counts it. A non-zero count in
  that column means the run's cache is too small for the store to be a credible
  handoff, not that a request failed.
- **The client hops are free by default**, and are the only hops a request's journey
  now has besides the coordinator's and the directory's: `TOSO_CLIENT_RTT` prices one
  client↔host round trip, and a request makes three of them (route, prefill, decode).
  It replaces the old `TOSO_HOST_RTT`, which priced a host-to-host forward — a boundary
  that no longer exists, because hosts do not call each other. Same caveat as the
  directory hop: TTFT is a prediction made before any of it is paid, so client latency
  shows up in the wall clock and in later requests' queue waits, not in the TTFT column.
