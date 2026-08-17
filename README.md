# toso 🫖

Application-level caching, control planes, deterministic simulations, and live
inspection for [TorchStore](https://github.com/meta-pytorch/torchstore), a
distributed PyTorch tensor store built on
[Monarch](https://github.com/meta-pytorch/monarch).

TorchStore provides the directory, tensor transport, storage volumes, and
resharding path. Toso stays separate because it adds experimental application
policy rather than storage mechanisms, without expanding TorchStore's API for each
experiment:

- **Deduplicated weight transfer:** bounded peer fan-out, read-through replicas,
  `1×` origin traffic, and load spreading across copies.
- **Cache-aware LLM serving:** global prefix reuse, pull-versus-recompute pricing,
  prefill/decode placement, load balancing, and TTFT/TBT admission.
- **Reusable selection policy:** a typed selector algebra composes candidates,
  measurements, load annotations, folds, ordering, bounds, and fallbacks. The same
  `KeySelector` ranks sources for dedup and KV caching.
- **Live inspection:** a real SPMD workload and Rust TUI for topology, keys, DTensor
  shards, health, and bounded tensor statistics.

## Capability pattern

Capabilities share one feedback loop:

```text
data-plane facts -> Dispatcher -> Sensors -> Selectors
        ^                                      |
        +------ execute the control-plane answer <---------+
```

Sensors fold application facts such as promised replicas, queue occupancy,
reservations, and routed pulls. An `Environment` supplies topology, time, and read
pricing; a `DirectorySensor` supplies coherent residency reads. Selectors declare and
resolve only the sensor types they read, without mutating them. The control plane
commits a choice, and the data plane executes it through ordinary TorchStore clients
and reports the resulting facts. New capabilities supply only their sensors, selector
chains, questions, and execution steps.

## Deterministic simulation framework

`realsim` runs the real TorchStore client, controller, volume, transport-buffer, and
store paths under a deterministic virtual clock. In-process seams replace actor RPC,
not the production planning and storage logic.

- Identical inputs produce byte-identical traces; seeded scheduling explores
  repeatable alternative interleavings.
- A target-machine profile analytically charges network, storage, RAM, CPU, and GPU
  resources instead of measuring the host running the simulation.
- Metadata-only tensors model large payloads without allocating their bytes.
- Shared workload, runner, ledger, trace, and report types make baseline and
  capability runs directly comparable.

This makes algorithm changes fast to test, reproducible, and independent of host
noise while still exercising the real TorchStore path.

## Where to look

- Designs: [TorchStore overview](docs/torchstore.md),
  [generic cache](docs/toso.md),
  [dedup](docs/torchstore_dedup_design.md),
  [KV cache](docs/torchstore_kvcache_design.md),
  [DES](docs/des_design.md), and [control flow](docs/architecture.md).
- Simulations: [`putget_sim/`](putget_sim/), [`dedup_sim/`](dedup_sim/),
  [`kvcache_sim/`](kvcache_sim/), and their [`realsim/`](realsim/) foundation.
- Live inspection: [`live_example.py`](live_example.py), [`tui/`](tui/), and the
  [TUI design](docs/tui_design.md).

## Setup

The simulations and live example use editable source builds of TorchStore and
Monarch. Place `toso/`, `torchstore/`, and `monarch/` beside one another, then run:

```bash
cd toso
./build_from_source.sh
```

The first build needs Rust and network access. It builds a CPU-only environment in
`.venv`. Python source edits are immediately visible; Monarch Rust edits require
`./rebuild_monarch.sh`.

## Run the simulations

```bash
PYTHONPATH=. .venv/bin/python -m putget_sim
PYTHONPATH=. .venv/bin/python -m dedup_sim
PYTHONPATH=. .venv/bin/python -m kvcache_sim
```

Each package README documents its scenarios and flags.

## Run the live store and TUI

Start two trainers, two generators, and one query aggregator:

```bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=5 \
  live_example.py --port 8099
```

In another shell:

```bash
cd tui
cargo run --offline --bin toso-tui -- \
  --aggregator 127.0.0.1:8099 --refresh 2
```

The TUI can run without TorchStore by loading its fixtures:

```bash
cd tui
cargo run --offline --bin toso-tui -- --fixtures fixtures/
```

See [`tui/README.md`](tui/README.md) for the full launch matrix and UI tour.
