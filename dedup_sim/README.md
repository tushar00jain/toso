# dedup_sim -- dedup read-routing on the real TorchStore directory

`dedup_sim` runs the **dedup algorithm on the real TorchStore directory and real
types** (via [`realsim`](../realsim/)): a synchronized read burst is routed so
that each unique byte crosses the fabric **exactly once (1x)**, versus **`m x`**
for the naive baseline. The routing is a real
`realsim.coordinator.model.ReadPolicy` (`dedup_sim.policy.routing.DedupPolicy`) driving
the real `Controller` directory, the real `LocalClient` planning core, and the
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

The naive policy (`realsim`'s `NaivePolicy`) fans every reader out concurrently;
in a synchronized burst they all `locate_volumes` the origin before anyone
finishes, so each pulls from the origin volume -- `m x` fabric.

`DedupPolicy` stages the burst into a read-through **chain/tree** over the real
directory:

1. It consults the **real directory** (`locate_volumes` -> real `StorageInfo` /
   `TensorSlice`) to find the origin(s) that hold the key.
2. It plans a `cap`-ary tree of sources (`fanout_cap=1` -> a chain, `>=2` -> a
   shallow tree): the **root** reader pulls from an origin (the single fabric
   hop); every other reader is attached to a **peer**.
3. After each reader fetches, the **real read-through** fires: the reader `put`s
   the key into its own co-located volume -- a zero-fabric local write that, via
   the real `client.put` path, both stores the payload there and calls the real
   `notify_put_batch`. The reader is now a real directory source for the next
   level.
4. Each depth level executes concurrently (a level's sources were populated by
   the previous level), so a wider tree narrows wallclock.

Because exactly one reader ever pulls from an origin, the only origin-sourced
transfer is that first hop: `fabric_bytes == 1x` the payload, for **any** fan-out
cap. The naive baseline stays `m x`.

The real `LocalClient` chooses a read source purely from what `locate_volumes`
returns and takes no source argument, so the policy expresses each routing choice
by giving that reader's client a `_RoutingControllerHandle` that answers
`locate_volumes` from the **real** directory and then narrows the result to the
chosen volume (returning the real `StorageInfo` unchanged). Every other endpoint
-- notably `notify_put_batch` for read-through -- passes straight through, so the
real directory stays the single source of truth.

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

The demo runs one synchronized read burst under three policies and prints, for
each, the fabric summary (dedup 1x vs naive `m x`), the wallclock, and the
who-served-whom diagram:

- **dedup, `fanout_cap=1` (chain)** -- `origin -> r0 -> r1 -> r2`;
- **dedup, `fanout_cap=2` (tree)** -- `origin -> r0 -> {r1, r2}` (narrower wallclock);
- **naive baseline** -- every reader pulls from the origin (`m x` fabric).

## Testing

```
PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests -q
```

The tests assert the dedup **outcome** on the real directory (not wall-clock
timing): every reader receives the payload; fabric is the 1x union vs `m x`
naive; the fan-out cap shapes a chain/tree; and the trace is byte-identical
across runs (default and under a fixed random-scheduling seed). The exact-byte
reassembly guarantee of the real client is covered separately in
`../realsim/tests/test_correctness.py`.

## Module layout

```
dedup_sim/
  policy/                 # THE ALGORITHM UNDER TEST
    routing.py            #   DedupPolicy (a real realsim ReadPolicy)
                          #   + the per-reader routing directory view
  workload/               # WHAT IS SIMULATED
    scenarios.py          #   build/run the dedup burst (reuses realsim wiring)
                          #   + realsim's own burst as the naive baseline
  report/                 # OUTCOME METRICS
    summary.py            #   dedup-vs-naive fabric summary + source->dest tree
  __main__.py             # `python -m dedup_sim` demo (summary + ASCII + trace)
  tests/                  # the dedup-outcome assertions (pytest, deterministic)
```

All the real-object plumbing -- adapters, seams, coordinator, cost model, async
engine, meta/metadata carriers -- is imported from `realsim` / `sim_common`;
`dedup_sim` adds only the dedup policy and its scenario.

## Comparison with `kvcache_sim`

Both capability packages use the same role folders, so what each one *needs* is
visible from which folders exist and how thick they are:

| role | `dedup_sim` | `kvcache_sim` |
|---|---|---|
| `policy/` — the algorithm under test | `routing.py`: one `ReadPolicy` override | `scheduler.py` + `cache.py` + `decode.py`: routing, eviction and a batched decode engine |
| `workload/` — what is simulated | `scenarios.py` only; the workload is **one fixed synchronized burst**, parameterized by reader count | `request.py` (domain model) + `generator.py` (seeded Zipf/Poisson stream) + `scenarios.py` (six scenarios) |
| `report/` — outcome metrics | `summary.py`: rendering only; the metrics are realsim's `BurstMetrics` | `metrics.py`: its **own** per-request outcome model (TTFT/TBT percentiles, hit rate, rejections) |
| `runtime/` — running on the real objects | **absent** — realsim's `ReadCoordinator` already drives a burst, so the `ReadPolicy` seam is the only hook needed | `cluster.py` (four KV directory verbs on a `Mesh`) + `driver.py` (per-request lifecycle on the virtual clock) |
| domain model + cost layer | **absent** — no served model to describe; charges realsim's cost model directly through the transport seam | `realsim.model.Model` (shared — the LLM's flop terms and KV block byte size) plus `utils.py` (`prefill_time` / `decode_step_time`, used by both planes) |

The short version: dedup is a *routing decision inside an existing burst*, so it
fits realsim's policy seam and needs no runtime or cost layer of its own.
KV-cache serving is a *continuous arrival stream with per-instance state*, so it
brings its own driver, its own directory verbs, its own cost layer and its own
metrics.

## Honesty note

Dedup optimizes **fabric bytes**. Both policies deliver the full payload to every
reader (`total delivered` is `m x` in both); dedup cuts the *origin* fabric to
1x. Wallclock depends on `fanout_cap`/topology: a `cap=1` chain has more hops
(more wallclock, still 1x fabric); a `cap=2` tree overlaps siblings and narrows
the gap. The demo prints both so the tradeoff is visible.
