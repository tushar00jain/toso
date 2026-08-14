# putget_sim -- the unrouted put/get burst on the real TorchStore directory

`putget_sim` seeds one key on an origin volume and has `m` clients `get` it, over
the **real** `LocalClient` planning core, the **real** `Controller` directory and
the **real** in-memory transport/store (via [`realsim`](../realsim/)), on
`realsim`'s deterministic virtual-clock async engine.

It has **no control plane and no data plane**. Every reader locates the origin
before anyone finishes, so each pulls from it and fabric is **`m x`** the
payload. That is deliberately the uninteresting outcome: this is the baseline
[`dedup_sim`](../dedup_sim/) measures its **1x** against, and the smallest thing
that exercises everything `realsim` provides while owning no capability code of
its own.

Everything is single-threaded, deterministic (byte-identical trace across runs),
and **allocation-free**: the payload is carried by a `device="meta"` tensor (real
tensor, zero storage) or a `(shape, dtype)` descriptor, so no real tensor bytes
ever move no matter how large the modeled payload.

For how the DES foundation works see
[`../docs/des_explained.md`](../docs/des_explained.md); for the real-code
foundation, [`../docs/realsim_design.md`](../docs/realsim_design.md).

## What one run exercises

One burst charges every resource class through one `MachineProfile`:

- **compute/GPU** — the producer's generate step for `W` (a roofline over
  `FLOPS_PER_ELEMENT` per element and the payload's bytes);
- **network** — the client↔volume fabric on the seed put and on every get;
- **storage** — the write on put and the read on serve;
- **RAM** — host staging on serve.

The scenario is ordinary user code top to bottom: a `client.put` and a gather of
`client.get`. There is no selector, no coordinator and no execution loop in it.
Handing it a `proposed.selector.KeySelector` (and, if the capability needs one, a
`proposed.plane.DataPlane`) is the *only* change needed to make it a routed run —
which is exactly what `dedup_sim` does, importing this package's `PutGetBurst`
unchanged so its comparison is byte-for-byte the same topology, payload and cost
model.

## Environment (uv)

The project uses [uv](https://docs.astral.sh/uv/) with a `.venv` at the repo root
(`toso/.venv`). Run everything from the repo root so packages resolve, with the
repo on `PYTHONPATH` and the venv interpreter:

```
cd toso
PYTHONPATH=. .venv/bin/python -m putget_sim
```

This sim imports `torch` + `torchstore` (through `realsim`), so use the project's
`.venv` interpreter.

## How to run

```
PYTHONPATH=. .venv/bin/python -m putget_sim          # INFO: fabric summary + ASCII
PYTHONPATH=. .venv/bin/python -m putget_sim -m 4 -v  # 4 readers + the full trace
PYTHONPATH=. .venv/bin/python -m putget_sim --help
```

- `-m/--readers N` — readers in the burst (default 3).
- `-n/--elements N` — elements in `W` (float32; payload = `4*N` bytes).
- `--mode meta|metadata` — the allocation-free data-plane carrier: `meta`
  (zero-storage meta tensor, default) or `metadata` (a `(shape, dtype)`
  descriptor, no tensor at all).
- `--seed S` — switch the engine to seeded-random ready-queue mode (default:
  FIFO, reproducible).
- `-v` — also print the full per-event virtual-time trace (DEBUG).

Output: the fabric/wallclock summary + an ASCII source→dest tree at INFO. With no
selector every reader pulls the origin (`m×` fabric) — the baseline a read-through
selector cuts toward the 1× union.

## Testing

This package has no test folder of its own: its workload is the fixture
`realsim`'s suite drives, so the assertions about it live there
(`../realsim/tests/` — determinism, off-sim correctness, the perf guard, storage
capacity, the contract lint).

```
PYTHONPATH=. .venv/bin/python -m pytest realsim/tests -q
```

## Module layout

Same role folders as the other two sims, so what this one *needs* is visible from
which folders exist — and the two that decide and execute are simply absent.

```
putget_sim/
  control/                # DECIDES -- absent: no selector, that is the point
  data/                   # EXECUTES -- absent: no data plane either
  workload/               # WHAT IS SIMULATED
    put_get.py            #   PutGetBurst, a realsim.Workload: seed W on the
                          #   origin, then m clients get it. dedup_sim reuses it
    scenarios.py          #   the Burst Scenario: the single unrouted Run + narration
  report/                 # OUTCOME METRICS
    summary.py            #   BurstReport: fabric/wallclock + source->dest tree
  __main__.py             # `python -m putget_sim`: a realsim.Demo declaration
```

All the real-object plumbing — adapters, seams, mesh, runner, cost model, async
engine, meta/metadata carriers — is imported from `realsim` / `sim_common`.

## Comparison with the capability sims

| role | `putget_sim` | `dedup_sim` | `kvcache_sim` |
|---|---|---|---|
| `control/` — what is decided | **absent** — no preference is passed, so the directory's own order stands | `routing.py` + `_selector.py` + `_sensor/` + `_view.py`: a `KeySelector` chain behind one plane — a ranked source, once it is usable | `scheduler.py` + `_source.py` + `_sensor/` + `_view.py` |
| `data/` — what executes | **absent** — a `get` and nothing around it | `read_through.py`: one member — ask, get, local put, report | `serving.py` + `_decode.py` + `_store.py` |
| `workload/` — what is simulated | `put_get.py`: one synchronized burst, parameterized by reader count | the same `put_get.py`, with the two planes added | `request.py` + `generator.py` + `scenarios.py` |
| `report/` — outcome metrics | `summary.py`: rendering only, over a shared `Ledger` | `summary.py`: rendering only, over the same `Ledger` | `metrics.py`: its **own** per-request outcome row |
| outcome | `m x` fabric | **1x** fabric, same workload | TTFT/TBT under an arrival stream |
