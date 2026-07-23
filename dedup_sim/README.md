# dedup_sim -- discrete-event simulation of dynamic dedup

A single-threaded, fully deterministic discrete-event simulation (DES) of the
dynamic dedup coordinator in `../docs/torchstore_dedup_design.md`: the dynamic
cache / routing / queuing coordinator that turns a burst of reads into a
**1x-fabric**, balanced transfer.

This exercises the *algorithm*, not performance. "Time" is a unitless simulated
clock; transfer durations come from a pure cost function, never measurement.
Pure Python stdlib only (`heapq`, `dataclasses`, `typing`) -- no
torch/numpy/asyncio/threads/sleeps/randomness. Same input => byte-identical
event trace.

## Environment (uv)

The project uses [uv](https://docs.astral.sh/uv/) with a `.venv` at the repo
root (`toso/.venv`). Run everything from the repo directory (the parent of
`dedup_sim/`) so the package resolves. Either activate the venv:

```
cd toso
source .venv/bin/activate
python -m dedup_sim
```

or use `uv run` without activating:

```
cd toso
uv run python -m dedup_sim
```

`dedup_sim` itself is pure stdlib, so if the repo's heavier optional
dependencies aren't built in your checkout and `uv run` tries (and fails) to
sync them, add `--no-sync` to reuse the existing `.venv` as-is:

```
uv run --no-sync python -m dedup_sim
```

## How to run

```
python -m dedup_sim                 # all scenarios, INFO: summaries + ASCII only
python -m dedup_sim -v               # add the full per-event trace (DEBUG)
python -m dedup_sim reshard          # run only the 'reshard' scenario
python -m dedup_sim versioning -v    # one scenario, with the event trace
python -m dedup_sim --help           # usage + the valid scenario names
```

- Positional `scenario` is one of `toy`, `reshard`, `versioning`; omit it to run
  all of them (plus a closing takeaway). An unknown name errors with the list of
  valid names.
- `-v` / `--verbose` / `--debug` raises the log level to DEBUG so the `(a)` event
  trace prints; the default INFO level prints only the `(b)` summaries and ASCII
  diagrams (no per-event spam). Output is routed through the stdlib `logging`
  module with a bare `%(message)s` format, so it reads the same as plain prints.

The demo prints three clearly-titled sections, each with (a) the event trace
(DEBUG) and (b) the summary + ASCII diagram (INFO):

- **toy** full-replication burst -- dedup vs naive, `FANOUT_CAP=1` (a chain) and
  `FANOUT_CAP=2` (a shallow tree), with fabric bytes (1x vs mx) and wallclock.
- **reshard** -- trainer partition != generator partition (with overlap); shows
  fabric stays 1x (== union of needs) after atomic-region splitting.
- **versioning** -- a burst, then a `put` that bumps the version, then a second
  burst that re-pulls from the trainer (cache invalidated), vs a no-bump
  contrast where the cache serves the second burst.

## Testing

`pytest` may not be installed in `.venv`. Run the tests with a one-off
environment or install pytest into the venv:

```
uv run --with pytest pytest dedup_sim/tests -q     # no install (needs a synced project)
# or, if project sync fails / to reuse the existing .venv:
uv pip install pytest
python -m pytest dedup_sim/tests -q
```

The tests are deterministic (they assert on the DES outcome and compare recorded
trace strings, not on logging output or wall-clock timing).

## The user-facing entry point takes no extra args

The only call a "user" makes is:

```python
client.get(reader, key, need)
```

exactly mirroring `ts.get` -- **no** promise/wait/dedup arguments leak to the
caller. The coordinator (routing, promises, parking, fan-out) is invoked
entirely *inside* `get`. `client.put(key, volume_id, region)` seeds the index.

## What the coordinator does (as simulated here)

- The first reader to need an atomic region pulls it from a **trainer** volume
  and becomes its (promised) cache source.
- Later readers of the same region are routed to a **present or promised** peer
  -- never re-pulling from the trainer -- and are **parked** until the peer's
  promise resolves (the store-mediated "done" = the puller's `notify_put`).
- Source choice prefers locality (a pure cost model: shm < nvlink < cross-node).
- A **fan-out cap** bounds concurrent serves per source, producing a balanced
  chain (`cap=1`) or shallow tree (`cap>=1`) incrementally. Two counters keep
  this honest: a plan-time tally shapes the tree; an execution-time slot queue
  guarantees no source ever exceeds the cap (excess consumers queue instead of
  re-pulling from the trainer, so fabric stays 1x).
- State is per `(key, version)`; a new `put` bumps the version and drops stale
  cache.

## Module layout (SPEC §10)

```
sim_common/            reusable DES library (repo root)
  engine.py            Sim (DES event loop) + Promise (dependency primitive)
  topology.py          Tier / TIER_LABEL / locality / transfer_time skeleton
  controller_probe.py  silenced real-Controller import probe -> HAVE_REAL
  trace.py             generic Trace event recorder (record/render, tunable widths)
  report.py            configure_logging + section header helpers for the demo
dedup_sim/
  SPEC.md              source-of-truth spec
  __main__.py          `python -m dedup_sim` demo (trace + summary + ASCII)
  sim/store_index.py   real-Controller probe (HAVE_REAL from sim_common) + faithful shim
  sim/model.py         Volume, Region, atomic-region splitting
  sim/cost.py          dedup TIERS + transfer_time (delegates to sim_common.topology)
  sim/coordinator.py   NaiveCoordinator (baseline) + DedupCoordinator (dynamic dedup)
  sim/client.py        Client.get/put -- the no-API-change entry point
  sim/trace.py         dedup Metrics + event/summary/ASCII rendering (Trace from sim_common)
  sim/scenarios.py     toy burst, reshard, versioning scenarios + run harness
  tests/test_sim.py    SPEC §8 assertions (pytest, deterministic)
```

The DES engine, the locality/transfer-time cost skeleton, the generic `Trace`
event recorder, the logging/section-header helpers, and the silenced
real-`Controller` import probe live in the repo-root `sim_common/` package (a
reusable DES library); this sim keeps its own bandwidth constants, domain model,
and the dedup-specific `Metrics` + ASCII rendering.

## Store-index path

We attempt to import the real `torchstore.controller.Controller` (via the shared
`sim_common.controller_probe`, which exposes `HAVE_REAL`). Even when it
imports, its endpoints are `@endpoint async` Monarch-actor methods that need an
actor runtime to drive and operate on torchstore-internal
`Request`/`TensorSlice`/`Trie` types -- neither of which a plain single-threaded
sim can use without spawning Monarch (out of scope). So the sim uses a
**faithful shim** (`StoreIndex`) that mirrors the controller's storage-index
semantics (`key -> {volume_id -> set[Region]}`) with matching method names
(`locate` ~ `locate_volumes`, `notify_put` ~ `notify_put_batch`, `keys`), so a
later swap to the real controller is mechanical. The `__main__` output prints
which path was taken.

## Scenarios & tests (SPEC §8)

1. **Correctness** -- every generator ends holding exactly its `need`.
2. **1x fabric** -- dedup trainer->gen bytes == union of needs; naive == mx;
   dedup < naive.
3. **Fan-out respected** -- no source exceeds `FANOUT_CAP` concurrent serves.
4. **Determinism** -- two runs yield byte-identical trace strings.
5. **Versioning** -- a `put` bump invalidates the cache (next burst re-pulls);
   without a bump, the cache serves the second burst (zero trainer fabric).
6. **Reshard** -- trainer stores one partition, generators want another (with
   overlap); assembled result is correct and fabric is still 1x.

Note (honesty): dedup optimizes **bytes moved**. Wallclock depends on
`FANOUT_CAP`/topology -- a `cap=1` chain has more hops and can be slower in
wallclock than the naive broadcast, while `cap=2` narrows the gap. The demo
prints both so the tradeoff is visible.
