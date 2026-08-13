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
  `Request.from_objects`, so publishing a descriptor would exercise an object path
  no KV deployment takes. A remote-prefix pull is a real `client.get_batch`
  driven through `realsim`'s transport seam and it hands the KV back; eviction
  removes presence via the real `notify_delete_batch`. A KV block is a directory key
  holding a tensor -- real types throughout, with no translation layer.
- **Real tensors at both ends of the request, too.** A `Request` carries its **prompt**
  (one `device="meta"` `int64` per prompt token), `Accelerator.prefill(prompt, cached)`
  takes it and answers with the KV *and* the request's **first token**, and each decode
  step emits one token per batch member — so the client that walks the redirect chain
  receives the whole output, first token from the prefill host and the rest from the
  decode host, and the run *counts* what it produced instead of reporting the
  `output_tokens` the workload asked for. One thing that deliberately does **not**
  follow: a block key is a hash of a prefix's content and a meta tensor has no
  content, so the chain is still generated alongside the prompt rather than derived
  from it — the same compromise the KV blocks make (real object, real shape, no data).
  Not streaming: decode answers with the remaining tokens when the request finishes,
  which is the `stream=False` shape. Decode answers with the **KV** those tokens left
  behind as well (`Accelerator.generated_kv`), because that is what makes generating
  cost memory on the host generating it — see the decode-residency bullet below.
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
python -m kvcache_sim hotspot --spread-reads   # spread reads over hot replicas
python -m kvcache_sim --help            # usage + valid scenario names
```

- Positional `scenario` is one of `shared_prefix`, `eviction`, `hotspot`,
  `overload`, `disaggregation`, `early_rejection`; omit it to run all (plus a
  closing takeaway).
- `-v` / `--verbose` / `--debug` raises the log level to DEBUG so the `(a)` event
  trace prints (capped to the first 60 events per scenario); the default INFO level
  prints only the `(b)` summaries.
- `--spread-reads` gives the **hotspot** scenario's cache-aware runs
  `SpreadReadsKeySelector` as their source ranking instead of longest-prefix-then-id, so
  one replica of a hot prefix does not serve every read of it. Off by default: it
  changes which replica answers, so it is not byte-identical.

## The scenarios

- **shared_prefix** — multi-turn conversations sharing a hot system prompt +
  per-tenant context. Cache-aware routes a turn to the instance holding that
  conversation's history (or pulls it once), so a dialogue is prefilled ~once;
  load-balance scatters the turns and recomputes the whole history. Higher hit rate,
  much less prefill compute, lower TTFT — and the gap widens with turn depth, because
  what a miss costs is the conversation so far.
- **eviction** — sweeps per-instance cache capacity and prints the hit-rate curve: it
  rises as the hot working set fits, then plateaus. The sweep starts at 48 blocks
  because a conversation's last turn is a 29-block chain and a volume below that
  cannot hold one request's own working set.
- **hotspot** — one dominant tenant (extreme Zipf skew). Compares load-balance
  vs cache-aware **without** replication vs **with** replication. Replication lowers
  prefill compute at the cost of KV fabric bytes. It does not also lower p90 TTFT:
  a dominant *tenant* is many dialogues with their own growing histories, which
  cache-aware routing already scatters, so there is no pile left to spread. The
  scenario says so where it prints.
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
  per-token metric and loses the wall clock. The `decode KV blocks` row is the same
  bill in memory: a decode host keeps the chain it pulled in *and* the KV it
  generated, so the disaggregated pool accumulates ~2.7x the blocks the coupled one
  does for identical load.
- **early_rejection** — heavy decode load under a tight TBT SLO, comparing admission
  modes `early` and `predict`. Both gate at routing, before the prefill runs, so a
  refusal costs no compute — late-checking after the prefill is the behaviour the
  design argues against and is not implemented. A mode is not a branch the coordinator
  tests either: it names whether the decode occupancy the **gate** is fed is predicted
  forward (`predict`) or read off the last report (`early`), which is all that
  separates the two. Neither rejects anything at this load, so what the table shows is
  where each one's decode selection landed — the `decode KV blocks` row separates them
  by more than any other, unlooked-for: routing by foreseen load happens to keep decode
  nearer the prompt, so `predict` leaves 2179 blocks resident against `early`'s 2665.

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
client -> A          "serve this" (a prompt tensor)
A      -> client     "prefill is B"     # A asks the coordinator; A does not call B
client -> B          B prefills, publishes its KV blocks to the store
B      -> client     the FIRST token + "decode is C"
client -> C          C fetches that KV back out of the store, decodes, finishes
C      -> client     the remaining tokens   # at the LAST token, not at admission
```

Five consequences, and they are the reason for the shape:

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
  prefill/decode-disaggregated system.
- **The client is still there at the last token, so it can time the request.** The
  decode leg returns when the request finishes, not when it is admitted to a batch, so
  the client stamps `now - arrival` onto the row — the one measurement no host can
  make, since it spans both of them. Decode therefore never outlives the coroutine
  that asked for it, so the harness needs no drain phase after the requests.
- **...and it is the only thing that holds the whole answer.** TTFT means time to the
  *first* token, and that token is sampled from the prefill's last position, so the
  prefill leg answers with it and the decode leg answers with the remaining
  `output_tokens - 1`. No host holds both halves, which is why the produced token
  count is a client-side join in exactly the way end-to-end latency is
  (`RequestResult.output_tokens`, counted; `Request.output_tokens`, asked for).

## The user-facing entry point mirrors the store

The only calls a "serving engine" makes are:

```python
answer = await placement.decide(request, me)           # route; None => rejected
...                                                    # pull remote prefix + prefill
await store.publish(me, fresh, kv)                     # cache fill + decode handoff
await cluster.notify(PrefillFinished(me, now))
```

One question, asked once: the answer names the prefill host *and* the decode host, so
everything a request needs is settled before any of it runs and no refusal can cost
a prefill. Behind it are two selections -- the prefill hosts control priced and the
decode hosts it ranked against the winner among them -- and what comes back is the
winner of each plus the price of the one that won (a `Response`). `None` is the
refusal.

Two ports, one member each, split between asking and telling: the scheduler's `decide`
for the question (a `Request`) and `ClusterModel.notify` for the facts
(`PrefillFinished`, `ComputeBusy`, `DecodeState`), every one of them a value. They
are deliberately all a serving host may touch: control holds every instance's
queue, cache and decode occupancy, so it runs as a service, not here. Everything crossing is a value, which is what lets the in-process call
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
directory read is control -- it does neither.

```
kvcache_sim/
  control/                # DECIDES -- moves nothing, holds no client
    scheduler.py          #   ONE scheduler behind proposed.ControlPlane, the
                          #   port the data plane calls: prefill placement,
                          #   pull-vs-recompute, SLO gates, decode placement,
                          #   every one of them priced against the cluster model
                          #   below. LoadBalance (baseline) and CacheAware are
                          #   presets of it -- two Placements, one naming a peer
                          #   to pull from and one ranking the priced
                          #   candidates, and admission as a list of gates
    _cluster.py           #   KVClusterModel behind proposed.ClusterModel: the
                          #   PREDICTED prefill queue, the observed decode
                          #   batches, and what was promised against them. One
                          #   per run, built in attach() and written only by
                          #   notify(fact) -- the facts a host reports live here
                          #   with the fold that applies them, and so do the
                          #   reads everything that ranks hosts by load makes
    _pending.py           #   Reservations / RoutedPulls: what was decided and
                          #   not yet done. Each expires on its own terms, when
                          #   read -- so no decision method carries a sweep
    _source.py            #   LongestPrefixKeySelector (+ the opt-in
                          #   SpreadReadsKeySelector, which spreads reads over
                          #   equally good replicas): the one store question
                          #   ("which peer serves this gap"), a proposed.KeySelector
    _view.py              #   KVView: per-instance prefix-run lengths, plus the
                          #   pinned snapshot one routing decision reads through
                          #   (underscored: the coordinator builds its own, so
                          #   nothing outside control/ names this)
    request.py            #   inference Request, carrying its prompt (a
                          #   zero-storage meta tensor of token ids) and its
                          #   prefix-hash chain (str keys). The keys are NOT
                          #   derived from the prompt -- a meta tensor has no
                          #   content to hash -- so they are generated with it
  data/                   # EXECUTES -- advances the clock, moves bytes
    serving.py            #   one ServingHost per instance, as three things a
                          #   client asks it: route (which host should prefill
                          #   this -- answered with an address, not a forward),
                          #   prefill (real pull, compute, publish -> answered
                          #   with the FIRST token and the decode host's address)
                          #   and decode (fetch the KV back out of the store and
                          #   become resident for it, batch it -> answered with
                          #   the remaining tokens, publish what they left). No
                          #   host holds a reference to another host
    _compute.py           #   Accelerator: the port an engine runs its work on --
                          #   what it costs, making it take that long, the KV and
                          #   first token a forward pass over a prompt hands back,
                          #   the token a decode step emits per batch member, the
                          #   KV a generation leaves behind it, and
                          #   the one occupancy both engines book on. Both engines
                          #   get one; the SAME one is what coupling means
    _prefill.py           #   PrefillEngine: what a forward pass costs and
                          #   submitting it -> the KV blocks this host now holds
                          #   and did not before, plus the request's first token.
                          #   The pass waits for the device, so the queue wait is
                          #   measured rather than taken from control's forecast
    _decode.py            #   async DecodeEngine: batched, stepped decode -> TBT,
                          #   and the tokens each member generated plus the KV
                          #   they left on this host, handed to whoever admitted
                          #   it (all three underscored: nothing outside data/
                          #   drives them)
    _store.py             #   publish / reuse / fetch over a Deployment's clients,
                          #   moving whatever KV it is handed. It holds no notion
                          #   of what a block is or how big one is -- that is the
                          #   accelerator's, which produces them
  workload/               # WHAT IS SIMULATED
    _generator.py         #   seeded synthetic stream of multi-turn Conversations
                          #   (Zipf over turn depth, Poisson dialogue starts,
                          #   exponential think time), incl. each prompt's tensor
                          #   and the prefix-hash chain (str keys) generated
                          #   beside it. Turn N+1's chain is turn N's chain plus
                          #   turn N's continuation keys plus a new message
    _accelerator.py       #   SimulatedAccelerator: the Accelerator port, answered
                          #   by a roofline, a sleep, one zero-storage meta tensor
                          #   per KV block at the size the scheduler priced, and
                          #   one per token produced. Owns BLOCK_TOKENS,
                          #   TOKEN_DTYPE, and the single-server queue that makes
                          #   a forward pass wait for the device. The one piece of
                          #   the compute story a deployment replaces outright
    _serving.py           #   KVWorkload (one work item per conversation) +
                          #   serving_plane, the wiring a run installs around it,
                          #   incl. the client that walks a dialogue's turns one
                          #   at a time and follows the two redirects each gets
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
client/controller/transport seams + adapters, the `Mesh`, the `KeySelector` / `View` /
`DataPlane` / `Runner` / `ItemDispatch` types live in `realsim/`. This package holds only the
KV-cache decisions and the three directory verbs (`publish` / `fetch` / `evict`)
plus the prefix-run read that express KV caching on a mesh.

## Honesty notes

- **The workload is multi-turn, and that is a closed loop.** A work item is a
  *conversation*, not a request: turn N+1's prompt is turn N's prompt plus turn N's
  **output** plus a new user message, so it cannot be submitted until turn N has
  answered, and the client walks a dialogue's turns one at a time with a think-time
  pause between them. The runner's `gather` gives concurrency across conversations;
  turns within one are strictly serial. Three consequences worth stating up front.
  (1) The reusable prefix **grows** and contains generated tokens, so the KV a decode
  host publishes under `Request.continuation_keys` is looked up and hit — ~15% of
  every matched block in the prefill-side scenarios and ~30% in the decode-side ones.
  (2) Only a conversation's first turn has an arrival this workload can state; every
  later one arrives when the run says it does. What stays fixed by the seed alone is
  *which* turns exist, what they contain and how long each user pauses, which is what
  keeps "same workload, different wiring" a fair comparison.
  (3) The offered load is **paced by the system**: at most one request per open
  dialogue can be in flight, and anything that slows a turn down delays its successor.
  Two scenario claims do not hold under it, and the scenarios say so where they
  print (hotspot's replication win, early_rejection's predicted-vs-stale decode
  routing); both need a burst a closed loop cannot offer. A related surprise:
  adding a coordinator or client hop *lowers* mean TTFT on the prefill-side workload,
  because it throttles the load faster than it lengthens a queue. The hop's cost is
  measured end to end instead, which is the interval that contains it.
- This optimizes **prefix reuse / TTFT / TBT under a cost model**; absolute numbers
  are arbitrary units. Read the scenarios for *relative* wins (cache-aware vs
  load-balance) and *shapes* (hit rate vs capacity), not throughput claims.
- **A refused turn does not end its conversation.** The user is told no and the next
  turn is offered anyway, as though the refused one had been served. Ending the
  dialogue instead is what a discouraged user does, but it makes *which requests
  exist* depend on the selector under test, and a rejection count is only worth
  anything when both columns are shedding from the same offered load. What the
  simplification costs is one block of query and one of output in the next turn's
  prompt that a real transcript would not carry — an over-charge, never an invented
  reuse, since a refused turn published nothing and the prefix run stops there.
- **A batch bigger than the volume desynchronizes the directory.** A publish is one
  `put_batch`, and torchstore registers a batch's keys *after* every one has landed,
  while the volume evicts and reports its evictions key by key as they land. A volume
  with less slack than the batch is writing therefore drops a key out of the batch it
  is still landing, reports that drop before the key was ever registered, and is then
  registered for it anyway — so the directory names a volume for blocks it threw away.
  It is self-healing in behaviour (the later read raises and the request recomputes,
  the `RESTALE` path) and not in the directory. Closing it means changing when a batch
  is registered, which is upstream of this repo; the eviction sweep's floor keeps the
  scenarios out of that regime, and `test_the_directory_and_the_volumes_agree_on_who_holds_what`
  is what noticed.
- Blocks become reusable at prefill **completion**, not while in flight; with spaced
  arrivals the difference is small.
- A remote pull is routed on the directory snapshot at the request's arrival, but the
  fetch runs after the prefill queue; if a peer evicted a planned block meanwhile,
  the read-through fetches only what remains present (the rest is recomputed) -- the
  faithful real-directory behavior. The peer it pulls *from* is the one the
  coordinator priced: the run installs the *scheduler* in the directory, and its
  `select` answers a fetch with the pull it already routed (falling back to
  `LongestPrefixKeySelector` when it routed none), so `locate_volumes` narrows to that
  peer. Without it the
  client takes whichever holder the directory lists first, which for a block several
  instances hold (a shared system prompt, anything replicated) can be a different
  locality tier than the one the TTFT prediction was built on.
- **The prefill queue is real, and it disagrees with the scheduler that predicted
  it.** The forward pass is *submitted* to the host's accelerator, after the KV
  fetch, and runs when that accelerator is free, behind whatever prefill or decode
  step already has it. Nothing sleeps a forecast, so the wait is emergent, both
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

  It bounds what the decode-side results can be worth: `early_rejection`'s `predict`
  mode routes decode by the load foreseen *at prefill completion*, so its TBT
  attainment is only as good as the prefill completion it foresees. The run
  stays deterministic because the queue is served in an explicitly sorted order —
  submission instant, then request id — rather than in whichever order the event loop
  resumed its waiters.
- **The coordinator hop is free by default.** Control is a service, reached through
  `realsim/seams/control_plane_handle.py` — so there is somewhere to charge the round
  trip,
  but `--coordinator-rtt` defaults to `0` and every call is inline. Turn it up and it
  is paid out and back before prefill can start: at `0.5` on the shared-prefix
  workload, mean TTFT goes 2.56 → 4.90 and the hit rate 0.734 → 0.704, because routing
  reads a directory snapshot one hop old and a just-published prefix is not there to
  reuse yet. Both schedulers pay the same hop, so the comparison holds either way.
  Reports pay it too, over the seam in front of the model they correct
  (`realsim/seams/cluster_model_handle.py`, same distance): a decode batch change is
  a round trip inside the step loop, so on a coupled instance every step pays one.
  What the seam still does not model is that the recorded TTFT is control's own
  prediction, so it moves with queueing rather than by exactly one RTT.
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
  finishes before the first one. The `mean/p90 end-to-end` rows are where it lands —
  arrival to last token, measured by the client, the only interval that contains the
  handoff by construction. On the disaggregation scenario the transfer itself is
  **~0.41s, ~26% of the 1.573 mean**, against ~1% for the coupled run, which mostly
  decodes where it prefilled and pays nothing. It is enough to flip the comparison:
  disaggregation wins TBT (0.028 vs 0.147) and *loses* end-to-end (1.573 vs 1.212),
  which is the trade a dedicated decode pool actually makes. Folding the handoff into
  TBT instead — the first token comes from the prefill host, so the transfer arguably
  *is* an inter-token gap — takes both disaggregation columns to `0.0%` attainment
  against a target of five decode steps, a one-off migration swamping a per-token
  metric rather than a finding, and it is not how the disaggregation literature
  measures either (DistServe/Mooncake put KV migration in TTFT and keep TPOT for
  decode cadence). So: charged on the clock, its bytes in their own column, its time
  in end-to-end.
- **A decode host holds KV, so a decode host pays for it.** Two things land on one:
  the block chain it pulls in to attend over (a `get_batch` delivers bytes and stores
  nothing) and the KV its own generation appends (one position per step,
  `ceil(n / block_tokens)` blocks, the trailing partial one charged whole because a
  paged cache allocates whole blocks). Both are published on the decode host through
  the same `publish` the prefill side uses — under the prompt's keys and under keys
  continuing its chain (`Request.continuation_keys`) respectively — and both are
  evictable and refusable like anything else there. Published rather than held as
  unlookupable residency, for the same reason the prefill leg publishes a prefix it
  pulled: a host that holds a block says so, and the alternative is a second kind of
  occupancy the directory cannot see. Two consequences, both real: the decode host
  becomes a **replica** (a read-through cache, which is what the hotspot scenario's
  `replicate=True` buys deliberately on the prefill side), and decode **competes**
  with cached prefixes for the volume. Nearly all of that pressure is the *chain*
  rather than the generation, which is one block per request at this block size, and
  the four prefill-only scenarios never feel it because a run that does not model
  decode never reaches a decode host. One limit, stated rather than fixed: the
  generation's bytes are charged when it ends rather than as it grows (a publish
  inside the step loop would stall every other batch member and invent a TBT effect
  the hardware does not have).
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
- **The tokens are real objects with nothing in them, and there is no stopping
  rule.** A prompt, the first token and every decode token are `device="meta"`
  `int64` tensors: right dtype, right shape, right `nbytes`, no data — the same
  compromise the KV blocks make, and the reason a block key still cannot be a hash of
  its prefix's content. What follows is that nothing here can *decide* to stop: there
  is no vocabulary, no logits and no EOS, so a request generates exactly the
  `output_tokens` it asked for and the produced count equals the requested one in
  every run. That is why both are recorded (`RequestResult.output_tokens` counted
  against `Request.output_tokens` asked for): today the pair is a consistency check,
  and the day a stopping rule or a preemption exists it is the answer.
- **The client hops are free by default**, and are the only hops a request's journey
  has besides the coordinator's and the directory's: `TOSO_CLIENT_RTT` prices one
  client↔host round trip, and a request makes three of them (route, prefill, decode).
  Same caveat as the directory hop: TTFT is a prediction made before any of it is
  paid, so client latency shows up in the wall clock and in later requests' queue
  waits, not in the TTFT column.
