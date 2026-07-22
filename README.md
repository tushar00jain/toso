# toso — TorchStore examples

Runnable [TorchStore](https://github.com/meta-pytorch/torchstore) examples driven
by a [uv](https://docs.astral.sh/uv/) environment, with both TorchStore and
[Monarch](https://github.com/meta-pytorch/monarch) **built from the in-repo
source** so you can edit either and iterate.

The example is written in the **SPMD** style: launched under `torchrun`, every
rank runs the same script. It **writes toso data** to a real store and serves the
§5 query protocol over a local socket, which the Rust `tui/toso-tui` reads live.

For states a single-host local store can't produce (partial DTensors, multiple
hosts, unreachable volumes), the TUI reads the hand-written `tui/fixtures/`
directly — no Python needed. See `tui/README.md`.

## Files

- `live_example.py` — writes a live async-RL workload (trainers + generators) and
  serves the §5 query protocol over a socket for the TUI to read live.
- `toso_store_reader.py` — the §5 response builders `live_example.py` uses to turn
  the store's introspection endpoints into §5 JSON.
- `pyproject.toml` — uv project. `torchstore` and `torchmonarch` are editable
  source dependencies, expected as sibling checkouts (`../torchstore`, `../monarch`).
- `build_from_source.sh` — one-time build of both from source into the venv.
- `rebuild_monarch.sh` — incremental recompile after editing Monarch's Rust.

## Build (once)

Monarch has a Rust native extension, so the first build needs a Rust toolchain and
pulls crates over the network. Check out `torchstore` and `monarch` next to this
repo, then run the build script from the repo root:

```bash
# layout: a parent dir holding toso/, torchstore/, and monarch/ side by side
cd toso
./build_from_source.sh
```

It installs Rust nightly (via rustup if missing), creates `.venv/`, installs the
build + runtime deps, then editable-installs `torchmonarch` and `torchstore`. The
**first** build compiles Monarch's full Rust workspace (~800 crates) and is slow;
later rebuilds are incremental (see below). It builds CPU-only
(`USE_TENSOR_ENGINE=0 MONARCH_GPU_PLATFORM=none`) — enough for this gloo example.

## Run

`live_example.py` assigns **one role per rank**: with `--nproc-per-node=5` you get
2 trainers, 2 generators, and 1 aggregator, each on its own rank (and its own
storage volume). The aggregator is the last rank; it serves the §5 protocol on the
`--port`. `--no-sync` keeps `uv run` from re-resolving/rebuilding against the
lockfile — the venv is managed by `build_from_source.sh`, not `uv sync`.

```bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=5 live_example.py --port 8099
```

Fewer ranks work too — `--nproc-per-node=3` gives 1 trainer + 1 generator +
aggregator, and `--nproc-per-node=1` collapses every role onto a single rank.

This writes a live workload and serves the §5 query protocol on
`127.0.0.1:8099`. Point the TUI at it and watch the store mutate:

```bash
cd tui
cargo run --offline --bin toso-tui -- --aggregator 127.0.0.1:8099 --refresh 2
```

See `tui/README.md` for the full launch matrix (including reading the bundled
`fixtures/` directly) and a tour of the TUI's pages.

## Editing the source

Both packages are **editable** installs, so:

- **Python edits** (Monarch *or* TorchStore) are live — just re-run the example,
  no rebuild.
- **Rust edits** (Monarch) need a recompile. Run `./rebuild_monarch.sh`. Because
  `CARGO_TARGET_DIR` is pinned (`~/.cache/monarch-cargo-target`), cargo reuses the
  prior build and only recompiles the crates you touched — seconds-to-minutes, not
  the full ~800.

Incrementality depends on two things the scripts set up for you: an **editable**
install (so the build runs in the source tree, not an isolated temp copy) and a
**stable `CARGO_TARGET_DIR`**. A plain `uv sync` of a path dep would rebuild from
scratch each time; that's why the venv is `uv pip`-managed here instead.

## Why build from source at all?

`initialize_spmd` needs `monarch._src.spmd.host_mesh.host_mesh_from_store`, which
landed in Monarch on **2026-04-23**. The newest published `torchmonarch-nightly`
is **2026-01-09**, so no pip wheel has it yet. Building from the in-repo source
gets the current code (and lets you modify it). Before this, the example used a
stopgap that copied the two new pure-Python files into the wheel's install; the
source build supersedes it.

## SPMD vs Monarch-actor

TorchStore has two entry points:

- **SPMD** (`ts.initialize_spmd()` + `torchrun`) — used here. The classic
  `torch.distributed` model: `torchrun` launches N identical ranks, each calls
  `dist.init_process_group(...)`, and TorchStore attaches to that process group.
  Ordering via `dist.barrier()`. This is the shape for dropping TorchStore into an
  existing torchrun training job.
- **Monarch-actor** (`ts.initialize()` + `spawn_actors`) — a driver process asks
  Monarch to spawn worker actors and `ts.*` runs inside actor endpoints; launched
  with a plain `python …`. This is the shape for a Monarch-native application.
