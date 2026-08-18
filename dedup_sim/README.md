# dedup_sim -- dedup read-routing on the real TorchStore directory

`dedup_sim` runs the **dedup algorithm on the real TorchStore directory and real
types** (via [`realsim`](../realsim/)): a synchronized read burst is routed so
that each unique byte crosses the fabric **exactly once (1x)**, versus **`m x`**
for the unrouted baseline. The routing is one `proposed.plane.ControlPlane`
(`dedup_sim.control.routing.Dedup`), asked by the data plane before each read, over the
real `LocalClient` planning core, the real `Controller` directory and the real in-memory
transport, all on `realsim`'s deterministic virtual-clock async engine.

Everything is single-threaded, deterministic (byte-identical trace across runs),
and **allocation-free**: the payload is carried by a `device="meta"` tensor (real
tensor, zero storage) or a `(shape, dtype)` descriptor, so no real tensor bytes
ever move no matter how large the modeled payload.

For the capability's design see
[`../docs/torchstore_dedup_design.md`](../docs/torchstore_dedup_design.md); for how
the DES foundation works, [`../docs/des_design.md`](../docs/des_design.md).

## How dedup gets to 1x on the real directory

With no control plane, every reader `locate_volumes` the origin before anyone
finishes, so each pulls from the origin volume -- `m x` fabric.

`Dedup` is asked first, and the read then prefers what it named. It is this
capability's whole control plane, reached as a service of its own:

1. Readers ask it in order, and one ranking prices everything that could serve them:
   the volumes holding the key, and the readers already routed to fetch it. The
   **first** reader has only the holders to choose from -- the single fabric hop.
2. Every later reader finds a **peer** cheaper: a reader that is about to hold the
   key, one tier away instead of across the fabric. A peer stops being offered once
   `fanout_cap` readers are behind it (`1` -> a chain, `>=2` -> a shallow tree).
3. A selected peer that has not registered adds `Published(peer)` to the decision's
   **readiness gate**. `sources` answers after every selected peer lands.
4. The read-through is the data plane's one job
   (`dedup_sim.data.read_through`): after a reader's batch returns, it stores the
   keys in its own co-located volume -- a zero-fabric local write through the real
   `client.put_batch` path, which registers every key before it returns -- and then
   commits `Published(reader)`. The plane's own fan-out folds that one action and
   releases every gate waiting on the completed producer.

Because a peer outprices a holder, exactly one reader ever pulls from a pre-existing
holder: the only origin-sourced transfer is that first hop, `origin_bytes == 1x` the
payload, for **any** fan-out cap. The baseline stays `m x`.

The price is where that preference lives, rather than an order the chain is written in
(`dedup_sim.control._selector`), and it is in **seconds off the run's own cost model**,
so nothing weighs a distance against a delay:

```
score = wait + hop + fabric * hop        (seconds; lower wins)

  wait     seconds until that source holds the key -- 0 for a holder, and for a peer
           the sum of the real link times up its own branch
  hop      what the transfer to the requester costs over the link between them
  fabric   the one dial: what a second of the link this read occupies is worth against
           a second of the requester's own waiting (10 chained, 0 spread)
```

A near peer wins because its link is cheap, and 1x holds while the branch's accumulated
link time stays under the fabric charge -- on the default profile a cap-1 chain seven
readers deep, past which a fresh hop off a holder really is the better answer. That is a
trade the price makes rather than a promise it keeps; any cap above 1 keeps the tree
shallow enough that a burst of any size never reaches it.

There is no burst loop anywhere. `dedup_sim/workload/scenarios.py` runs
[`putget_sim`](../putget_sim/)'s ordinary put/get fixture -- a `client.put` and a
gather of `client.get` -- and the
chain/tree is an emergent consequence of step 4 changing the directory, and saying
so, that step 1 reads.

## Environment (uv)

The project uses [uv](https://docs.astral.sh/uv/) with a `.venv` at the repo root
(`toso/.venv`). Run everything from the repo root so packages resolve, with the
repo on `PYTHONPATH` and the venv interpreter:

```
cd toso
PYTHONPATH=. .venv/bin/python -m dedup_sim
```

This sim imports `torch` + `torchstore` (through `realsim`), so use the project's
`.venv` interpreter.

## How to run

```
PYTHONPATH=. .venv/bin/python -m dedup_sim              # both scenarios
PYTHONPATH=. .venv/bin/python -m dedup_sim weight_sync  # just the two-replica one
PYTHONPATH=. .venv/bin/python -m dedup_sim -v           # add the per-event trace (DEBUG)
PYTHONPATH=. .venv/bin/python -m dedup_sim --help
```

- `-v` / `--verbose` / `--debug` raises the log level to DEBUG so the `(a)`
  per-event virtual-time trace prints; the default INFO level prints only the
  `(b)` fabric summaries and the ASCII source->dest diagram.

The demo runs one synchronized read burst under three selectors and prints, for
each, the fabric summary (dedup 1x vs naive `m x`), the wallclock, and the
who-served-whom diagram:

- **dedup, `fanout_cap=1` (chain)** -- `origin -> r0 -> r1 -> r2`;
- **dedup, `fanout_cap=2` (tree)** -- `origin -> r0 -> {r1, r2}` (narrower wallclock);
- **unrouted baseline** -- every reader pulls from the origin (`m x` fabric).

## Load spreading: the same chain over a key with more than one holder

`python -m dedup_sim weight_sync` is the second scenario: two trainer replicas hold
`W` and two generators want it. The replicas are equidistant, so distance prices them
identically and the id tie-break puts every first hop on `t0`: the unrouted baseline
sends both generators there, and the chain sends one and queues the other behind it.
Neither reads `t1` at all.

`Dedup(spread=True)` is the same ranking with the dial at `0` -- a trainer is charged
nothing for the fabric a read of it burns, so what is left of the score is how soon each
source can serve me. The queue is then what separates two replicas:
[`proposed.selector.Balance`](../proposed/selector.py) appends it to the ranking's sort
key and the plane's own fold charges it -- a reader already routed at a source costs one
more read of that source, so waiting behind one is worth a hop somewhere else. No new
fact and no new link -- a route *is* a decision naming a source, so the load is a read of
the fan-out the plane already keeps.

|                       | baseline | dedup | dedup+spread |
|---|---|---|---|
| bytes off the trainers | 2x | **1x** | 2x |
| reads off any one trainer | 2 | 1 | **1** |
| depth (2 generators) | 1 hop | 2 hops | **1 hop** |
| depth (4 generators) | 1 hop | 4 hops | **2 hops** |

So spreading buys **1x per replica** and a tree of depth `ceil(m / n)` in place of a
chain of depth `m`, and pays one origin hop per replica for it. Both routed runs read
through, so a generator that has finished is a holder like any other -- which is why the
third generator is folded in behind a peer rather than becoming a third trainer hop.

What the load number does *not* say is on
[`proposed.sensors.LoadSensor`](../proposed/sensors.py): it counts requesters currently routed
to a source, not bytes in flight, so it comes down when a route is retired but not when
a read finishes.

## Testing

```
PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests -q
```

The tests assert the dedup **outcome** on the real directory (not wall-clock
timing): every reader receives the payload; fabric is the 1x union vs `m x`
baseline; the fan-out cap shapes a chain/tree; and the trace is byte-identical
across runs (default and under a fixed random-scheduling seed). The exact-byte
reassembly guarantee of the real client is covered separately in
`../realsim/tests/test_correctness.py`.

`test_evicts.py` runs the harder half: volumes with room for one payload, so a
new version displaces the one a reader cached and the directory *unregisters* it
mid-run. The same key is read twice with that eviction in between, and the fabric
is 1x per read -- the chain re-forms because each answer is withheld against what
the directory holds now, not against a registration that has since been dropped.

`test_spread.py` asserts the other outcome: which trainer each generator read from,
that each replica is read once and no more, and that a generator past the replicas is
folded in behind a peer.

## Module layout

Split by plane: `control/` decides, `data/` executes, and neither imports the
simulator — `control/` takes an `Environment` and typed sensors, `data/` calls torchstore APIs
against a `Deployment` (enforced by `realsim/tools/check_contract.py`).

```
dedup_sim/
  control/                # DECIDES
    routing.py            #   Dedup: a proposed.ControlPlane -- sources() answers
                          #   with per-key sources once they are usable, off the chain it
                          #   builds and the one fold that orders what the chain
                          #   keyed; a completed batch is an action it folds, not a
                          #   question it is asked. `spread` picks the fabric dial
                          #   and the fold that reads the queue at a source
    _selector.py          #   Candidates: one ranking over every relevant region holder
                          #   and the peers already
                          #   routed to fetch it -- priced in seconds: the wait
                          #   until a source has the key, the hop to me, and the
                          #   fabric that hop burns
    _sensor/              #   pending directory, fan-out state, and shared actions
      _directory.py       #     live + pending metadata, publication facts, fetch plans
      _fanout.py          #     route facts, dependencies, and source load
  data/                   # EXECUTES
    read_through.py       #   ReadThroughPlane: apply the per-key directory scope to
                          #   LocalClient._fetch, put, then commit Published
  workload/               # WHAT IS SIMULATED
    scenarios.py          #   the Dedup and WeightSync Scenarios: the Runs to compare
                          #   (the fixture as it is, and with the two planes added)
                          #   + narration
    _weight_sync.py       #   WeightSync: the same burst over a key that n trainer
                          #   replicas hold, which is where spreading has a choice
  report/                 # OUTCOME METRICS
    summary.py            #   DedupReport / BaselineReport / WeightSyncReport: fabric
                          #   summary + tree
  __main__.py             # `python -m dedup_sim`: a realsim.Demo declaration
  tests/                  # the dedup-outcome assertions (pytest, deterministic)
```

All the real-object plumbing -- adapters, seams, mesh, runner, cost model, async
engine, meta/metadata carriers -- is imported from `realsim` / `sim_common`;
`dedup_sim` adds only the routing decision and the read-through write.

## Comparison with `kvcache_sim`

Both capability packages use the same role folders, so what each one *needs* is
visible from which folders exist and how thick they are:

| role | `dedup_sim` | `kvcache_sim` |
|---|---|---|
| `control/` — what is decided | `routing.py`: one plane, `sources` + `_selector.py` (the chain behind it) + `_sensor/` (pending directory, fan-out state, and their actions) | `scheduler.py` (prefill placement, pull-vs-recompute, SLO gates, decode placement, and which peer serves a fetch) + `_selector.py` (the rankings it decides with) + `_answer.py` (the values it answers with) + `_sensor/` (the model) + `_prefix.py` (prefix runs) |
| `data/` — what executes | `read_through.py`: one member — ask, get, local put, commit | `serving.py` (the per-request lifecycle) + `_decode.py` (the batched decode engine) + `_store.py` (the KV directory verbs) |
| `workload/` — what is simulated | `scenarios.py`: **one fixed synchronized burst** (`putget_sim`'s fixture), parameterized by reader count | `request.py` (domain model) + `generator.py` (seeded Zipf/Poisson stream) + `scenarios.py` (six scenarios) |
| `report/` — outcome metrics | `summary.py`: rendering only; the measurements are a shared `sim_common.report.Ledger` | `metrics.py`: its **own** per-request outcome row (TTFT/TBT percentiles, hit rate, rejections) on the same `Ledger` |
| domain model + cost layer | **absent** — no served model to describe; charges realsim's cost model directly through the transport seam | `domain/llm.py` (shared — the LLM's flop terms, KV block byte size, and token→time) |

The short version: dedup is a *source decision*, so its control plane is one member
and its data plane one. KV-cache serving is a *continuous arrival stream with
per-instance compute state*, so its plane also answers where a request runs, and it
keeps its own serving loop and outcome model — while the "which peer" half of it is the
same kind of `KeySelector` chain dedup's is built on.

## Honesty note

Dedup optimizes **fabric bytes**. Both selectors deliver the full payload to every
reader (`total delivered` is `m x` in both); dedup cuts the *origin* fabric to
1x. Wallclock depends on `fanout_cap`/topology: a `cap=1` chain has more hops
(more wallclock, still 1x fabric); a `cap=2` tree overlaps siblings and narrows
the gap. The demo prints both so the tradeoff is visible.
