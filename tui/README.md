# `toso-tui`

A [`ratatui`](https://ratatui.rs) terminal UI for inspecting a running
**TorchStore** — its topology (strategy, volumes, hosts) and contents (keys,
tensor/object metadata, DTensor shard layout). Read-only. See
[`../docs/tui_design.md`](../docs/tui_design.md) for the design.

It reads from a `Provider`: either a directory of JSON snapshot files
(`--fixtures`) or a live agent over TCP (`--aggregator`). Both speak the same §5
JSON contract, so the UI is identical either way.

## Build

Standalone cargo crate. Deps come from crates.io, so the **first** build
needs network:

```bash
cd toso/tui
cargo build            # first time: fetches deps
```

After that you can work offline with `cargo build --offline` / `cargo test --offline`.

## Run it

Two ways. **A** drives the local single-host store in `../` (built once via
`../build_from_source.sh` — see `../README.md`); **B** needs no Python or store at
all.

### A. Live — trainers and generators writing while you watch (recommended)

`../live_example.py` runs an async-RL workload with **one role per rank** (see its
module docstring): under `--nproc-per-node=5` ranks 0–1 are **trainers**
(`trainer.<i>.*`, rewriting policy tensors each step and rotating an optimizer key
in/out), ranks 2–3 are **generators** (`generator.<j>.*`, pulling each trainer's
latest policy version and writing fresh rollouts), and rank 4 is the
**aggregator** — the §5 socket server the TUI connects to. Each rank hosts its own
storage volume, so the TUI shows the keys sharded across five volumes.

```bash
# shell 1 — 2 trainers + 2 generators + 1 aggregator, one per rank
cd toso
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=5 live_example.py --port 8099

# shell 2 — the live TUI (re-polls every 2s)
cd tui
cargo run --offline --bin toso-tui -- --aggregator 127.0.0.1:8099 --refresh 2
```

Fewer ranks work too: `--nproc-per-node=3` is 1 trainer + 1 generator +
aggregator, and `--nproc-per-node=1` collapses every role onto one rank.

Watch the per-worker key counts and `step`/`metadata` change, and each
`trainer.<i>.optimizer.momentum` appear and disappear. Stop the run with
`pkill -TERM -f live_example.py` (or Ctrl-C in shell 1).

### B. Fixtures — bundled §5 snapshot (no Python, no store)

The bundled `fixtures/` dir is a hand-written snapshot covering tensors, objects,
and committed/partial DTensors — states a single-host local store can't produce.
Read it directly:

```bash
cargo run --offline --bin toso-tui -- --fixtures fixtures/
```

You can also replay it over the network via the stub server (exercises the same
`--aggregator` path as A):

```bash
cargo run --offline --bin agg_stub -- --fixtures fixtures/ --port 0   # prints the bound port
cargo run --offline --bin toso-tui -- --aggregator 127.0.0.1:<port>
```

## What each page means

The header is always shown: `toso-tui · <store> · <strategy> · <N> vols · <N> keys
· [live 2s]`, with a breadcrumb of where you are and the current `scope`/`sort`.
`l`/`Enter` drills into the selected row; `h`/`Esc` goes back. The examples below
use the store from `live_example.py`, whose top-level prefixes are `trainer.0`,
`trainer.1`, `generator.0`, and `generator.1` (each holding `step`, `metadata`,
and policy/rollout tensors).

### Health board (landing)

The first screen. Fleet health as numbers, not rows: total volumes/keys/bytes, a
count of ⚠ partial DTensors and ↓ unreachable volumes, and a `shard commit %`
histogram. Below it is a drillable list of the **top-level key prefixes** and the
**volume groups**.

```
health
volumes 5   keys 16   bytes 168.1 KB
✓ 0 partial dtensors   ↓ 0 unreachable volumes
shard commit %:  100% ██████████ 16

NAME           KEYS  DTENSORS  PARTIAL  BYTES      STATUS
▸ generator.0    4     0        —       64.0 KB    ok      ← branch, drills into Keys
▸ generator.1    4     0        —       64.0 KB    ok
▸ trainer.0      4     0        —       20.0 KB    ok
▸ trainer.1      4     0        —       20.0 KB    ok
▤ host:myhost…   16     —        —       168.1 KB   ok      ← volume group (▤)
```

A `▸` row is a key-trie node; a `▤` row is a volume group. Drill a group to scope
the other views to it.

### Keys (the key trie)

One level of the key trie, reached by drilling a `▸` prefix. Each row rolls up the
keys beneath it. Drilling descends one level at a time (lazy, paged); drilling a
**leaf** (a row that is itself a stored key) opens its Detail.

```
keys · trainer.0.policy
NAME       KEYS  DTENSORS  PARTIAL  BYTES     STATUS
▸ layer0    1     0         —       16.0 KB   ok     ← intermediate node → drills deeper
▸ layer1    1     0         —       4.0 KB    ok
```
Drilling `layer0` → `keys · trainer.0.policy.layer0` → row `weight` (a leaf) → Detail.

### Detail (a single key)

Everything about one key: type (`OBJECT` / `TENSOR` / `TENSOR_SLICE`), dtype,
global shape, mesh shape, committed status, and a shard table (which volume holds
each shard, with coordinates/offsets/local shape). `p` peeks tensor stats.

```
detail
trainer.0.policy.layer0.weight
type TENSOR   dtype float32   committed ✓ fully committed
global_shape —   mesh_shape —
shards (1)
VOLUME   COORDINATES  OFFSETS  LOCAL_SHAPE
vol-0    []           []       [64, 64]
```
For an **object** (e.g. `trainer.0.metadata`) there is no tensor shape or sharding, so the
shard panel reads `(object — stored whole; no tensor shape or shards)`. Press `p`
on a tensor shard to add a `peek` block (dtype/shape/min/max/mean/l2_norm and a
short head), computed near the data — never the full tensor.

### Topology / Volumes

Drilling a `▤` volume group lists its volumes: id, host, transport, key count,
bytes, and reachability. This also sets the `scope` shown in the header.

### Results (search / jump)

`/` filters keys by substring; `:` jumps directly (`:key <k>`, `:group <g>`,
`:partial`, `:unreachable`, `:peek <k>`). Both open a Results page of matches you
can drill into — e.g. `/rollout` lists `generator.0.rollout.tokens`,
`generator.0.rollout.rewards`, and the `generator.1.*` pair; `:key trainer.0.step`
jumps straight to that key's Detail.

## Keys

| Key | Action |
|-----|--------|
| `↑`/`↓`, `j`/`k` | move |
| `l` / `Enter` | drill into the selected row (lazy-loads a level) |
| `h` / `Esc` | back up the drill-stack |
| `g` / `G` | top / bottom |
| `PgUp` / `PgDn` | page |
| `/` | filter (pushed down as a search) |
| `:` | jump — `:key <k>`, `:group <g>`, `:partial`, `:unreachable`, `:peek <k>` |
| `s` | cycle sort (anomaly-first by default) |
| `p` | peek — tensor stats for the selected shard |
| `r` | refresh |
| `?` | help |
| `q` | quit |

## CLI

```
toso-tui --fixtures <dir>         # read a snapshot directory
toso-tui --aggregator <host:port> # connect to a live agent / stub
toso-tui ... --refresh <secs>     # summary auto-refresh interval (default 5)
toso-tui ... --headless           # render frames to stdout and exit (no TTY; for tests/CI)
```
