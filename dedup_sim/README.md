# dedup_sim -- dedup read-routing on the real TorchStore directory

`dedup_sim` runs the **dedup algorithm on the real TorchStore directory and real
types** (via [`realsim`](../realsim/)): a synchronized read burst is routed so
that each unique byte crosses the fabric **exactly once (1x)**, versus **`m x`**
for the unrouted baseline. The routing is a real `proposed.selector.KeySelector`
(`dedup_sim.control.routing.DedupKeySelector`) consulted *inside* the real
`Controller`'s `locate_volumes`, over the real `LocalClient` planning core and the
real in-memory transport, all on `realsim`'s deterministic virtual-clock async
engine.

Everything is single-threaded, deterministic (byte-identical trace across runs),
and **allocation-free**: the payload is carried by a `device="meta"` tensor (real
tensor, zero storage) or a `(shape, dtype)` descriptor, so no real tensor bytes
ever move no matter how large the modeled payload.

For the capability's design see
[`../docs/torchstore_dedup_design.md`](../docs/torchstore_dedup_design.md); for how
the DES foundation works, [`../docs/des_explained.md`](../docs/des_explained.md).

## How dedup gets to 1x on the real directory

With no selector installed, every reader `locate_volumes` the origin before anyone
finishes, so each pulls from the origin volume -- `m x` fabric.

`DedupKeySelector` answers that same `locate_volumes` differently. It is a
`proposed.selector.KeySelector`, consulted inside the real controller endpoint body:

1. Readers reach the controller in order. The **first** is routed to a volume
   that already holds the key -- the single fabric hop.
2. Every later reader is routed to a **peer**: a reader that is about to hold the
   key. Peers are handed out FIFO under a fan-out cap (`fanout_cap=1` -> a chain,
   `>=2` -> a shallow tree).
3. That peer has not registered yet, so the selection carries a **readiness
   gate** and the controller *withholds its answer* until the peer's
   read-through put lands. No client change is needed, and no client is lied to.
4. The read-through is the data plane's one job
   (`dedup_sim.data.read_through`): after a reader's `get` returns, it `put`s the
   key into its own co-located volume -- a zero-fabric local write that, through
   the real `client.put` path, both stores the payload there and calls the real
   `notify_put_batch`. That registration opens the next reader's gate.

Because exactly one reader ever pulls from a pre-existing holder, the only
origin-sourced transfer is that first hop: `origin_bytes == 1x` the payload, for
**any** fan-out cap. The baseline stays `m x`.

There is no burst loop anywhere. `dedup_sim/workload/scenarios.py` runs
[`putget_sim`](../putget_sim/)'s ordinary put/get fixture -- a `client.put` and a
gather of `client.get` -- and the
chain/tree is an emergent consequence of step 4 changing the directory that step
1 reads.

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
    routing.py            #   DedupKeySelector: a proposed.KeySelector -- ranked source
                          #   + a readiness gate, from a read-only View
    _readiness.py         #   Readiness: a gate per fact not true yet, opened
                          #   against the directory (nothing remembered, because
                          #   volumes evict) -- the waiting, out of the routing
  data/                   # EXECUTES
    read_through.py       #   ReadThroughPlane: one DataPlane method -- the
                          #   reader's put, via the Deployment's client for it
  workload/               # WHAT IS SIMULATED
    scenarios.py          #   the Dedup Scenario: the Runs to compare (the fixture
                          #   unwrapped, and with the selector installed) + narration
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
| `control/` — what is decided | `routing.py`: one `KeySelector.select` — a ranked source plus a readiness gate | `scheduler.py` (prefill placement, pull-vs-recompute, SLO gates, decode placement) + `selector.py` (the source `KeySelector`) + `cache.py` (LRU) + `view.py` (prefix runs) |
| `data/` — what executes | `read_through.py`: one member — the reader's get, then a local put | `serving.py` (the per-request lifecycle) + `decode.py` (the batched decode engine) + `_store.py` (the KV directory verbs) |
| `workload/` — what is simulated | `scenarios.py`: **one fixed synchronized burst** (`putget_sim`'s fixture), parameterized by reader count | `request.py` (domain model) + `generator.py` (seeded Zipf/Poisson stream) + `scenarios.py` (six scenarios) |
| `report/` — outcome metrics | `summary.py`: rendering only; the measurements are a shared `sim_common.report.Ledger` | `metrics.py`: its **own** per-request outcome row (TTFT/TBT percentiles, hit rate, rejections) on the same `Ledger` |
| domain model + cost layer | **absent** — no served model to describe; charges realsim's cost model directly through the transport seam | `domain/llm.py` (shared — the LLM's flop terms, KV block byte size, and token→time) |

The short version: dedup is a *source decision*, so it is one `KeySelector` and one
`DataPlane` method. KV-cache serving is a *continuous arrival stream with
per-instance compute state*, so it keeps its own scheduler, its own serving loop
and its own outcome model — and delegates only the "which peer" part to the same
`KeySelector.select`.

## Honesty note

Dedup optimizes **fabric bytes**. Both selectors deliver the full payload to every
reader (`total delivered` is `m x` in both); dedup cuts the *origin* fabric to
1x. Wallclock depends on `fanout_cap`/topology: a `cap=1` chain has more hops
(more wallclock, still 1x fabric); a `cap=2` tree overlaps siblings and narrows
the gap. The demo prints both so the tradeoff is visible.
