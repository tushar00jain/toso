# toso — TorchStore designs, simulations, and examples

A workspace for exploring [TorchStore](https://github.com/meta-pytorch/torchstore):
its architecture, proposed new capabilities, deterministic simulations of those
capabilities, and a runnable live example with a terminal UI to inspect a real
store.

TorchStore is a distributed, async KV store for PyTorch tensors built on
[Monarch](https://github.com/meta-pytorch/monarch) actors. Its headline use case
is **weight sync between a trainer/learner and a generator in RL**, including
**resharding** weights across two different device meshes. See
[`docs/architecture.md`](docs/architecture.md) for how it works today.

## What's here

The repo has three independent workstreams.

**1. Design docs** (`docs/`) — how TorchStore works and two proposed capabilities
layered on it.

- [`architecture.md`](docs/architecture.md) — a ground-up explanation of
  TorchStore's control plane, data plane, and resharding, with a glossary.
- [`torchstore_dedup_design.md`](docs/torchstore_dedup_design.md) — replica-aware,
  **deduplicated** trainer→generator weight transfer: a dynamic routing layer that
  turns a synchronized read burst into a 1× fabric transfer with no barrier.
- [`torchstore_kvcache_design.md`](docs/torchstore_kvcache_design.md) — making
  TorchStore double as a **Mooncake-style KV-cache** pool for LLM serving, with a
  cache-aware coordinator, prefix-hash addressing, and eviction.

**2. Discrete-event simulations** — each design has a companion deterministic DES
that exercises the *algorithm* (not performance) on a simulated clock: no
wall-clock, no threads, no randomness in timekeeping; same input ⇒ byte-identical
trace. All three run the **real** TorchStore code (real
client/controller/transport), so they depend on the from-source
`torchstore`/`torch`/`monarch` build — not stdlib-only.

- [`realsim/`](realsim/) — the **real-code** cooperative DES foundation: it drives
  the **real** TorchStore client planning core, controller directory, and
  in-memory transport/store off-actor on a deterministic virtual clock, modeling
  only the new read coordinator (with a pluggable `ReadPolicy` seam). `Mesh` is the
  multi-client wiring the capability sims build on — per-node volumes + real
  clients, one directory, one resource registry, and the single shared transport
  factory — so a capability package holds only capability code. It runs
  **allocation-free** (zero-storage meta tensors / metadata-only descriptors, so a
  modeled payload of any size costs no memory) and charges **every** resource —
  network, storage, RAM, CPU, and GPU/compute — as analytic functions of a
  *target-machine* `MachineProfile`, never measured on the box running the sim.
- [`dedup_sim/`](dedup_sim/) — the dedup capability **on the real directory**:
  `import realsim` and implement dedup routing as a real `ReadPolicy` (a 1× peer
  read-through against the real `Controller`), plus the dedup scenario + metrics
  (fabric bytes 1× vs naive *m×*).
- [`kvcache_sim/`](kvcache_sim/) — the cache-aware KV-cache capability **on the
  real directory**: the scheduler/decode/cache logic consults real KV-block
  presence per instance (real `Controller`) and drives real fetches via
  `realsim`'s `Mesh`/client/engine/cost model, expressed as four directory verbs
  (`prefix_lengths` / `publish` / `fetch` / `evict`).
- [`sim_common/`](sim_common/) — the shared building blocks all three sims use:
  the deterministic virtual-clock `AsyncEngine` (the sim path) plus the original
  callback engine (`engine.py`: `Sim`/`Promise`), the locality/topology skeleton,
  an analytic resource cost model (`cost_model.py`: `MachineProfile` +
  network/RAM/storage/CPU/GPU functions), a trace recorder, and reporting helpers.
- [`docs/des_explained.md`](docs/des_explained.md) — how the shared core works and
  how the two sims differ.

**3. Live example + TUI** — a real single-host store you can write to and watch.

- [`live_example.py`](live_example.py) — an SPMD (`torchrun`) async-RL workload
  (trainers + generators + aggregator, one role per rank) that writes to a real
  store and serves a JSON query protocol over a socket.
- [`toso_store_reader.py`](toso_store_reader.py) — turns the store's introspection
  endpoints into that protocol's JSON.
- [`tui/`](tui/) — a read-only [`ratatui`](https://ratatui.rs) terminal UI that
  reads the same protocol, live over TCP or from bundled JSON fixtures. See
  [`tui/README.md`](tui/README.md) to run it and
  [`docs/tui_design.md`](docs/tui_design.md) for its design.

## Running the simulations

All three sims drive the **real** TorchStore code, so they need the from-source
build below (`torchstore`/`torch`/`monarch` in the repo-root `.venv`). Once built,
run from the repo directory with the venv interpreter and the repo on
`PYTHONPATH`:

```bash
PYTHONPATH=. .venv/bin/python -m dedup_sim
PYTHONPATH=. .venv/bin/python -m kvcache_sim
PYTHONPATH=. .venv/bin/python -m realsim.run_realsim
```

See each capability sim's `README.md` for flags, [`realsim/README.md`](realsim/README.md)
for the real-code foundation, and the `docs/` design docs for how each capability
works ([`realsim_design.md`](docs/realsim_design.md),
[`torchstore_dedup_design.md`](docs/torchstore_dedup_design.md),
[`torchstore_kvcache_design.md`](docs/torchstore_kvcache_design.md), and
[`des_explained.md`](docs/des_explained.md)).

## Building the live example from source

The live example depends on `torchstore` and `torchmonarch`, both **built from the
in-repo source** (editable installs) so you can edit either and iterate. Check out
`torchstore` and `monarch` next to this repo, then build from the repo root:

```bash
# layout: a parent dir holding toso/, torchstore/, and monarch/ side by side
cd toso
./build_from_source.sh
```

Monarch has a Rust native extension, so the first build needs a Rust toolchain and
network access and compiles Monarch's full Rust workspace (~800 crates) — slow.
Later rebuilds are incremental:

- **Python edits** (Monarch or TorchStore) are live — just re-run.
- **Rust edits** (Monarch) need `./rebuild_monarch.sh`, which reuses a pinned
  `CARGO_TARGET_DIR` and only recompiles what you touched.

It builds CPU-only (`USE_TENSOR_ENGINE=0 MONARCH_GPU_PLATFORM=none`) — enough for
this gloo example.

> Why from source? `initialize_spmd` needs a Monarch symbol that landed after the
> newest published `torchmonarch-nightly` wheel, so no wheel has it yet. Building
> from the in-repo source gets the current code and lets you modify it.

## Running the live example + TUI

Launch the store under `torchrun` (one role per rank), then point the TUI at the
aggregator socket:

```bash
# shell 1 — 2 trainers + 2 generators + 1 aggregator
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=5 live_example.py --port 8099

# shell 2 — the live TUI (re-polls every 2s)
cd tui
cargo run --offline --bin toso-tui -- --aggregator 127.0.0.1:8099 --refresh 2
```

The TUI can also read the bundled `tui/fixtures/` directly with no Python or store
at all (`--fixtures fixtures/`), covering states a single-host store can't produce
(partial DTensors, multiple hosts, unreachable volumes).
[`tui/README.md`](tui/README.md) has the full launch matrix and a tour of the
pages.

## SPMD vs Monarch-actor

TorchStore has two entry points; the live example uses the first:

- **SPMD** (`ts.initialize_spmd()` + `torchrun`) — the classic `torch.distributed`
  model: `torchrun` launches N identical ranks, each calls
  `dist.init_process_group(...)`, and TorchStore attaches to that process group.
  This is the shape for dropping TorchStore into an existing torchrun training job.
- **Monarch-actor** (`ts.initialize()` + `spawn_actors`) — a driver process asks
  Monarch to spawn worker actors and `ts.*` runs inside actor endpoints, launched
  with a plain `python …`. This is the shape for a Monarch-native application.
