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
  a real `get_batch` of its whole block chain to reach the host that decodes it, the
  `KV handoff bytes` row is what that moves and the `end-to-end` rows are what it
  costs. Read together they are the actual trade — the disaggregated column wins every
  per-token metric and loses the wall clock.
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
C      -> client     "done"             # at the LAST token, not at admission
```

Four consequences, and they are the reason for the shape:

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
- **The client is still there at the last token, so it can time the request.** The
  decode leg returns when the request finishes, not when it is admitted to a batch, so
  the client stamps `now - arrival` onto the row — the one measurement no host can
  make, since it spans both of them. That also deleted the run's drain plumbing: decode
  no longer outlives the coroutine that asked for it, so the harness has nothing left
  to wait for after the requests.

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
                          #   what it costs, making it take that long, the KV a
                          #   forward pass hands back, and the one occupancy both
                          #   engines book on. Both engines get one; the SAME one
                          #   is what coupling means
    _prefill.py           #   PrefillEngine: what a forward pass costs and
                          #   submitting it -> the KV blocks this host now holds
                          #   and did not before. It no longer sleeps a queue
                          #   wait: the pass waits for the device, so the wait is
                          #   measured rather than taken from control's forecast
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
                          #   Owns BLOCK_TOKENS, and the single-server queue that
                          #   makes a forward pass wait for the device. The one
                          #   piece of the compute story a deployment replaces
                          #   outright
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
- **The prefill queue is real, and it disagrees with the scheduler that predicted
  it.** A request used to wait by *sleeping `plan.queue_wait`* — the number the
  scheduler produced when it routed the request — and then sleeping its forward
  pass. Nothing measured the queue, so a scheduler that mispredicted its own backlog
  was right by construction in every column the wait lands in: TTFT, end-to-end, and
  the next request's predicted queue. Now the forward pass is *submitted* to the
  host's accelerator, after the KV fetch, and runs when that accelerator is free,
  behind whatever prefill or decode step already has it. The wait is emergent, both
  numbers are recorded per request (`predicted_queue_wait` and `queue_wait`), and the
  run can contradict control. It does. Mean over accepted requests:

  | run | predicted | actual | requests differing |
  |---|---|---|---|
  | `shared_prefix` cache-aware | 0.987 | 0.898 | 14.5% |
  | `shared_prefix` load-balance | 1.585 | 1.585 | 0% |
  | `eviction` cap 8 | 27.640 | 23.989 | 95.0% |
  | `eviction` cap 32 | 0.519 | 0.477 | 17.8% |
  | `hotspot` cache/repl | 11.096 | 10.309 | 80.6% |
  | `hotspot` cache/no-repl | 11.433 | 11.433 | 0% |
  | `overload` cache-aware | 4.590 | 4.259 | 78.2% |
  | `early_rejection` predict | 85.047 | 85.041 | 8.1% |

  The error is nearly all one-signed and has one cause: control prices a candidate as
  `queue → transfer → prefill` and reserves the instance until the end of all three,
  so a **remote prefix pull is charged to the prefill device's occupancy** while a real
  device is idle during it. Every configuration that never pulls — both `load_balance`
  runs, `hotspot`'s no-replication run — predicts its queue exactly, which is what
  makes the cause legible rather than inferred. The other direction exists but is rare
  (0.2–1.2% of requests, up to +2.9s): a request queued behind one whose planned reuse
  had been evicted and became a full-length recompute. Saturated queues predict
  themselves well — at an arrival rate of 20/s (`early_rejection`) the backlog dwarfs
  every transfer and the mean error is 0.007% — so this is a light-to-moderate-load
  effect, not an overload one.

  What it moved, against the same runs before the queue existed: `early_rejection`'s
  `predict` mode loses TBT attainment (86.2% → 81.9%), because routing decode by the
  load foreseen *at prefill completion* is only as good as the prefill completion it
  foresees — the clearest sign that a self-fulfilling wait was propping a result up.
  `eviction` hit rates shift in both directions (34.9% → 32.7% at capacity 8,
  51.5% → 55.2% at 16) as a small cache's publish/evict interleaving moves with the
  queue. End-to-end barely moves where the queue is short (disaggregation
  1.269 → 1.266, coupled 1.126 → 1.117). Everything else — the cache-aware-vs-baseline
  story, the eviction curve's shape, the disaggregation trade — is unchanged.

  There is no flag for the old behaviour, deliberately. A flag is for a genuine
  alternative (`contention="none"` is a defensible model of an uncontended fabric);
  sleeping a forecast is not an alternative model, it is just less true, and a second
  path nobody runs is a second path everybody maintains. The run stays deterministic
  because the queue is served in an explicitly sorted order — submission instant, then
  request id — rather than in whichever order the event loop resumed its waiters.
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
- **The KV handoff lands in one column, and it had to be built.** The decode host
  fetches the request's whole block chain through the real `get_batch`, so the bytes,
  the fabric time and the storage/RAM staging are all charged — and none of it reaches
  either per-token column. Not TTFT, which is control's prediction made before any of
  it happens; not TBT, which is measured *between* decode tokens while the handoff
  finishes before the first one. It used to land nowhere at all: the cost that
  dominates a prefill/decode-disaggregated deployment was paid on the clock and
  reported in no table, visible only as an indirect nudge to *when* requests joined
  their batches. The `mean/p90 end-to-end` rows are the fix — arrival to last token,
  measured by the client, the only interval that contains the handoff by construction.
  On the disaggregation scenario it is **~32% of the mean** (1.269 charged vs 0.860
  with the transfer made free), against ~1% for the coupled run, which mostly decodes
  where it prefilled and pays nothing. It is also enough to flip the comparison:
  disaggregation wins TBT (0.029 vs 0.146) and *loses* end-to-end (1.269 vs 1.126),
  which is the trade a dedicated decode pool actually makes.
  Folding the handoff into TBT instead was tried: the first token
  comes from the prefill host, so the transfer arguably *is* an inter-token gap. It
  takes both disaggregation columns to `0.0%` attainment against a target of five
  decode steps, which is a one-off migration swamping a per-token metric rather than a
  finding, and it is not how the disaggregation literature measures either
  (DistServe/Mooncake put KV migration in TTFT and keep TPOT for decode cadence). So:
  charged on the clock, its bytes in their own column, its time in end-to-end.
- **End-to-end is measured only where there is a last token to measure to.** The four
  prefill-only scenarios (`shared_prefix`, `eviction`, `hotspot`, `overload`) do not
  model decode, so the client's walk ends at prefill completion; stamping *that* under
  the same name would make one column mean two different intervals depending on the
  scenario. It is left unstamped and those tables do not offer the column. Nor is a
  rejected request given one: it has no last token either, at either gate.
  Two things the number does include, deliberately, because a caller pays them: the
  client↔host round trips (free by default) and the queueing behind other requests.
  One thing it does not: a *queue delay a request would have seen in a real front
  end*, since arrivals here are released onto the clock rather than shaped by a load
  balancer's own backlog.
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
