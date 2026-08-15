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
the DES foundation works, [`../docs/des_explained.md`](../docs/des_explained.md).

## How dedup gets to 1x on the real directory

With no control plane, every reader `locate_volumes` the origin before anyone
finishes, so each pulls from the origin volume -- `m x` fabric.

`Dedup` is asked first, and the read then prefers what it named. It is this
capability's whole control plane, reached as a service of its own:

1. Readers ask it in order. The **first** is routed to a volume that already holds
   the key -- the single fabric hop.
2. Every later reader is routed to a **peer**: a reader that is about to hold the
   key. Peers are handed out FIFO under a fan-out cap (`fanout_cap=1` -> a chain,
   `>=2` -> a shallow tree).
3. That peer has not registered yet, so the decision carries a **readiness gate**
   and `sources` *does not answer* until the peer's read-through put lands. The
   caller's read is then an unmodified `client.get` with a preference passed to it:
   no client change is needed, nothing is installed in the store, and no client is
   lied to.
4. The read-through is the data plane's one job
   (`dedup_sim.data.read_through`): after a reader's `get` returns, it `put`s the
   key into its own co-located volume -- a zero-fabric local write through the real
   `client.put` path, which registers the key before it returns -- and then **commits
   one action**, `Stored`, which the plane's own fan-out folds. That commit wakes every
   parked reader, each of which re-reads the directory: one of them now finds its peer
   there, and the gate is open.

Because exactly one reader ever pulls from a pre-existing holder, the only
origin-sourced transfer is that first hop: `origin_bytes == 1x` the payload, for
**any** fan-out cap. The baseline stays `m x`.

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
PYTHONPATH=. .venv/bin/python -m dedup_sim        # INFO: fabric summaries + ASCII
PYTHONPATH=. .venv/bin/python -m dedup_sim -v     # add the full per-event trace (DEBUG)
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

## Module layout

Split by plane: `control/` decides, `data/` executes, and neither imports the
simulator — `control/` takes a read-only `View`, `data/` calls torchstore APIs
against a `Deployment` (enforced by `realsim/tools/check_contract.py`).

```
dedup_sim/
  control/                # DECIDES
    routing.py            #   Dedup: a proposed.ControlPlane -- sources() answers
                          #   with a source once it is usable, off the chain it
                          #   builds; a landed put is an action it folds, not a
                          #   question it is asked
    _selector.py          #   PlannedPeer / HolderRanking: the routing itself -- the
                          #   next peer under a fan-out cap, else the holders that
                          #   could open a tree, nearest first and priced by tier;
                          #   both gated on the put the head owes
    _sensor/              #   the one kind of fact this plane holds
      _fanout.py          #     FanoutSensor: who is folded in behind whom, which
                          #     puts are owed, who waits on them -- a
                          #     proposed.Sensor, the links' one record, and the
                          #     proposed.dispatch.Reducer that folds a landed put
    _view.py              #   FanoutView: that sensor as a read of the plane's
                          #   View, which is how a link reaches it
  data/                   # EXECUTES
    read_through.py       #   ReadThroughPlane: one DataPlane method -- ask, read,
                          #   put, commit one Stored action, over the Deployment's
                          #   client and ports
  workload/               # WHAT IS SIMULATED
    scenarios.py          #   the Dedup Scenario: the Runs to compare (the fixture
                          #   as it is, and with the two planes added) + narration
  report/                 # OUTCOME METRICS
    summary.py            #   DedupReport / BaselineReport: fabric summary + tree
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
| `control/` — what is decided | `routing.py`: one plane, `sources` + `_selector.py` (the chain behind it) + `_sensor/`/`_view.py` (the fan-out it senses, and folds a landed put into) | `scheduler.py` (prefill placement, pull-vs-recompute, SLO gates, decode placement, and which peer serves a fetch) + `_selector.py` (the rankings it decides with) + `_answer.py` (the values it answers with) + `_sensor/` (the model) + `_view.py` (prefix runs) |
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
