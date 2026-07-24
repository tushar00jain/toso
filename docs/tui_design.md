# `toso-tui` — a terminal UI for inspecting a TorchStore

**Status:** living spec
**Goal:** A Rust [`ratatui`](https://ratatui.rs) terminal app to explore a running
TorchStore — its **topology** (controller, storage volumes, hosts, transports)
and its **contents** (keys, tensor/object metadata, DTensor shard layout) — in the
spirit of interactive DB browsers like `harlequin` / `sqlite-web` / `gitui`.

---

## 1. Motivation

TorchStore ships no UI, no dashboard, and no CLI — introspection today is
async Python calls (`ts.keys()`, `ts.exists()`) and stdout logging. When a
training / RL-weight-sync job is running, there is no way to *see* what's in the
store: which volumes exist, where they live, which keys are stored, whether a
DTensor is fully committed, or how shards are spread across volumes.

This tool gives that a live, navigable view.

### Goals
- Show the **topology**: controller → N storage volumes → hosts, plus the
  strategy in use and the transport each volume negotiated.
- Browse **keys** as a tree (the store's keyspace is already a trie, e.g.
  `model.layers.0.attn.wq.weight`).
- Inspect a **key**: object type (object / tensor / DTensor-shard), dtype+shape,
  which volumes hold it, and — for DTensors — the shard/mesh layout and whether
  all shards are committed.
- Optionally **peek** at data: summary stats (shape, dtype, min/max/mean/norm,
  first-N elements) for a selected tensor, not full dumps.

### Non-goals (v1)
- Mutating the store (no put/delete from the UI). Read-only.
- Rendering full tensor contents or large blobs.
- A hosted web dashboard — this is a terminal tool.
- Cross-job / historical views — this inspects one live store at a time.

---

## 2. Background: what TorchStore actually exposes

The introspection surface already exists as Monarch actor endpoints. The spec is
built entirely on these — no new torchstore APIs are required for v1.

**Controller** (`torchstore/controller.py`) — the global index:
- `keys(prefix=None) -> list[str]` — all keys (from a `Trie`), optionally
  prefix-filtered.
- `locate_volumes(keys, missing_ok, require_fully_committed) -> {key: {volume_id: StorageInfo}}`
  — which volumes hold each key's data. `StorageInfo` carries `object_type`
  (`OBJECT` / `TENSOR` / `TENSOR_SLICE`) and a set of `tensor_slices`.
- `get_controller_strategy() -> TorchStoreStrategy` — the topology strategy
  (`LocalRankStrategy`, `HostStrategy`, `ControllerStorageVolumes`, …).
- `_is_dtensor_fully_committed(...)` — logic for "are all shards present"
  (compares stored `coordinates` against `product(*mesh_shape)`).

**StorageVolume** (`torchstore/storage_volume.py`) — per volume:
- `get_id() -> (volume_id, hostname)` — identity + host placement.
- `get_meta(requests) -> [(torch.Size, torch.dtype) | "obj"]` — per-key shape/dtype
  without transferring tensor data.
- underlying `InMemoryStore.kv`: `key -> tensor | {"obj": ...} | {coords: {"slice", "tensor"}}`.

**TensorSlice** (`torchstore/transport/types.py`) — shard descriptor:
`coordinates`, `mesh_shape`, `global_shape`, `local_shape`, `offsets`.

**The transport** each volume resolves (SharedMemory / RDMA / gloo / …) is logged
at init (`[ts-transport] resolved=...`) — capturable for the topology view.

---

## 3. The core problem: Rust ⇄ Python

TorchStore's live state is **in-process in the Python training job** — the
controller's `Trie` and each volume's `kv` dict live inside Monarch actors, not in
a file or a server. A separate Rust process cannot read them directly, and it
cannot trivially join the Monarch actor mesh.

There are **two** boundaries to cross, not one:

1. **Python/Monarch world → out-of-world.** State is in Monarch actors; a
   non-Monarch process can't read it. Solved by an **in-world Python agent**.
2. **Store nodes → a separate machine.** The store is spread across many training
   nodes, but the **TUI runs elsewhere**. So the agent's output has to travel over
   one network connection.

### Boundary 2 needs an aggregation tier

The TUI must **never** connect to the training nodes directly. At real scale
(potentially millions of nodes) one machine fanning out to N nodes is a
non-starter — connection count, and the sheer volume of state, both explode. So
the TUI talks to exactly **one** endpoint: an **aggregator**.

**Decision (v1): the aggregator *is* the in-world Python agent at N=1.** We do not
build a separate distributed rollup service. The in-world agent (§3 boundary 1)
that already reads the store via the controller + volume endpoints simply *answers
the §5 query contract directly* — for a real small/moderate store it is both the
data source and the "aggregator". "Aggregator" throughout this doc therefore means
"whatever serves the §5 contract on one endpoint"; for v1 that is one agent. The
hierarchical rollup below is the **fleet-scale swap** behind the *same* contract,
not a prerequisite — the Rust side (`AggregatorClient`) is identical either way.

Note the torchstore `Controller` is *not* that aggregator: it holds only the
keys→volumes index (not per-volume stats/data), and a single controller is itself
a known bottleneck at scale (torchstore even deprecates the controller-hosted
volume strategy for this reason). The aggregator is a separate introspection
tier that *sources* the index from the controller but adds its own rollup.

At **fleet scale**, that single agent is replaced (behind the same §5 contract) by
a rollup tier made tractable by two properties:

1. **Hierarchical fan-in.** Node/rank agents emit local partials → intermediate
   aggregators (per host / rack / region) roll them up → a root aggregator. Each
   level has bounded fan-in, so no single hop sees all N nodes. This is standard
   telemetry tree-aggregation; the store's own controller can seed the index but
   the stats/rollup ride the tree.
2. **Summarize, don't ship the firehose.** You cannot render millions of volumes
   or billions of keys, and you shouldn't move them. The aggregator keeps a
   *reduced* view — counts, group-bys (per host / volume-group / key-prefix),
   histograms, top-N — and answers **drill-down queries** on demand ("expand
   prefix `model.layers`", "volumes on rack R", "stats for group G"). The TUI
   pulls summaries first and fetches detail lazily as the user navigates.

That fleet-scale fan-in/rollup is its own component and out of scope for the TUI
deliverable — it is a **later swap behind the §5 contract**, not something v1
depends on. What the TUI owns, at either scale, is: one connection, a
summary+drill-down protocol, and rendering.

### One transport: the agent serves §5 over a socket

The agent and the TUI speak over **one transport — a TCP socket carrying
line-delimited JSON** (the §5 contract). The agent binds a port on its host; the
TUI holds a single connection to it (`--aggregator <host:port>`). When the two are
on different machines, reach the host the way you already do — e.g. an SSH tunnel
or a port-forward — rather than teaching the tool a second wire protocol.

The same JSON can also be read straight from a directory of files
(`--fixtures <dir>`) for offline work — the bundled fixtures use this. Both live
behind one `Provider` trait, so the UI never knows whether it is reading a socket
or files.

Authentication is out of scope for v1: the socket is plain TCP, meant for a local
or otherwise trusted endpoint (a tunnel, a dev box). A production deployment would
wrap it in whatever transport security its environment already provides.

### Does SPMD (torchrun) change this?

No — **SPMD TorchStore still runs on Monarch.** `torchstore/spmd.py` uses
`host_mesh_from_store` to build a Monarch host mesh from the torchrun `TCPStore`,
then global rank 0 spawns the Monarch storage-volume actors and broadcasts the
`Controller` handle through the same `TCPStore`. So there is still a single
Monarch `Controller` with the global keys→volumes index and Monarch volume actors
— the centralized-agent design above is unchanged. The only SPMD-specific detail:
the agent must be spawned **from within the SPMD-created Monarch context** (e.g.
on the primary rank, reusing the broadcast controller handle) rather than from a
`ts.initialize()` driver.

### If there were genuinely no Monarch (no central controller)

TorchStore hard-depends on Monarch today, so this is hypothetical — but the design
shouldn't assume a central aggregator exists. If a deployment had per-rank local
state and **no controller to fan out from**, switch from *central collection* to
**per-rank collection**: each rank runs a tiny collector that emits its *local
partial* snapshot (its own volume's keys/meta) and sends it to a collector. That
collector (or the TUI) merges the partials into the global view — key union,
per-shard volume attribution, committed-status computed from the merged mesh
coordinates. The fan-in already supports this: N ranks send, one consumer
merges. Centralized (controller) collection is just the N=1-producer special case,
so the schema (§5) and providers don't change — only *who* produces snapshots. The
Snapshot gains an optional `partial: true` + `rank` marker so the merger knows to
combine rather than replace.

```
  Training nodes (store's Monarch world) ── up to millions
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │ vol +  │ │ vol +  │ │ vol +  │ │ vol +  │ │ vol +  │ │ vol +  │   agent.py per
  │ agent  │ │ agent  │ │ agent  │ │ agent  │ │ agent  │ │ agent  │   node/rank emits
  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘   a LOCAL partial
      └────┬─────┘          └────┬─────┘          └────┬─────┘        (push outbound)
           ▼                     ▼                     ▼
     ┌───────────┐         ┌───────────┐         ┌───────────┐        intermediate
     │ aggregator│         │ aggregator│         │ aggregator│        aggregators
     │ (rack/rgn)│         │ (rack/rgn)│         │ (rack/rgn)│        (bounded fan-in)
     └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
           └───────────────┬─────┴─────────────────────┘
                           ▼
              ┌──────────────────────────────┐    also pulls the keys→volumes
              │      ROOT aggregator          │◄── index from the torchstore
              │  • rolled-up summaries        │    Controller (seed), adds rollup
              │  • group-bys / histograms     │
              │  • answers drill-down queries │   ◄── ASSUMED to exist; its
              └───────────────┬───────────────┘       internal tree is out of
                              │  query / summary contract (§5),      TUI scope
                              │  one connection, line-JSON
                              ▼
 ┌──────────────────────── a machine elsewhere ────────────────────────┐
 │  toso-tui (Rust) — ONE connection, never touches nodes              │
 │    data::Provider (trait)                                           │
 │      ├─ FileProvider   (dev: static JSON fixtures)                  │
 │      └─ AggregatorClient (TCP; summary + drill-down queries)        │
 │    model:: Summary / Group / KeyEntry / ShardMap  (serde)           │
 │    ui:: views + drill-stack + event loop (ratatui + crossterm)      │
 └─────────────────────────────────────────────────────────────────────┘
```

The Rust binary never speaks Monarch or Python, never contacts a training node,
and never pulls the whole store — it holds **one** connection to the aggregator
and speaks a summary + drill-down protocol. That keeps the sides
decoupled and testable (the UI develops against hand-written JSON fixtures).

---

## 5. Data contract: aggregator query protocol

The contract is **summary-first with lazy drill-down**, not one big snapshot — at
scale the TUI can't receive (or render) every key/volume. Metadata only; tensor
data never crosses the boundary except via an explicit `peek` (stats computed
aggregator/agent-side). Request/response is line-delimited JSON over the one
connection.

### 5.1 `summary` — the landing view (bounded size)

```jsonc
// request
{"op": "summary"}
// response — rolled up by the aggregator; O(groups), never O(nodes/keys)
{
  "schema_version": 1,
  "captured_at": "2026-07-17T17:40:00Z",
  "store_name": "torchstore",
  "strategy": "LocalRankStrategy",
  "totals": { "volumes": 1048576, "keys": 5203, "bytes": 9.2e15,
              "partial_dtensors": 3 },       // store-wide counts
  "volume_groups": [                          // grouped, e.g. by host/rack/region
    { "group": "rack:A12", "volumes": 4096, "keys": 5203, "bytes": 3.1e14,
      "transports": { "RDMA": 4090, "SharedMemory": 6 }, "reachable": 4096 }
  ],
  "key_prefixes": [                           // top level of the key trie only
    { "prefix": "model", "keys": 4820, "objects": 0, "tensors": 12,
      "dtensors": 4808, "partial": 2, "bytes": 8.9e15 },   // bytes: rolled-up tensor bytes
    { "prefix": "optimizer", "keys": 380, "dtensors": 380, "partial": 1, "bytes": 3.0e14 },
    { "prefix": "metadata", "keys": 3, "objects": 3, "bytes": 0 }
  ],
  "histograms": {                             // optional, for at-a-glance health
    "shard_commit_pct": [[0, 5], [50, 12], [100, 4790]]   // bucket -> key count
  }
}
```

### 5.2 Drill-down — fetch detail only for what the user opened

```jsonc
{"op": "expand_prefix", "prefix": "model.layers", "limit": 200, "cursor": null,
 "sort_by": "partial", "order": "desc"}    // sort/order applied SERVER-SIDE (§6.3)
// -> { "children": [ {"prefix": "model.layers.0", "keys": 120, ...}, ... ],
//      "next_cursor": "..." }          // one trie level deeper; paginated

{"op": "list_volumes", "group": "rack:A12", "limit": 200, "cursor": null,
 "sort_by": "bytes", "order": "desc"}
// -> { "volumes": [ {"volume_id","hostname","transport","num_keys","bytes","reachable"}... ],
//      "next_cursor": "..." }

{"op": "key", "key": "model.layers.0.attn.wq.weight"}
// -> KeyEntry (below): full per-key detail, fetched only when selected

{"op": "search", "kind": "key", "pattern": "attn.wq", "limit": 200, "cursor": null}
// -> { "matches": [ {"key": "model.layers.0.attn.wq.weight", ...}, ... ],
//      "next_cursor": "..." }    // substring/glob match ACROSS the whole keyspace,
                                  // no prefix required — powers /-filter and : jump
{"op": "search", "kind": "volume", "pattern": "rack:A12", "limit": 200}
// -> { "matches": [ {"volume_id","hostname","transport",...}, ... ] }
```

**Sort is server-side by necessity.** The TUI holds only a window of any list
(§6.1), so it cannot sort locally — the sort key must be applied by the aggregator
before pagination. `sort_by` accepts the anomaly/size keys the UI ranks by
(`partial`, `reachable`, `bytes`, `keys`, `name`); default is anomaly-first
(unhealthy rows top). Every list op (`expand_prefix`, `list_volumes`, `search`)
accepts `sort_by`/`order`.

**`search` walks no path.** Unlike `expand_prefix` (which descends one known trie
level), `search` finds matches anywhere in the keyspace/volume set without the
caller knowing where they live — this is what a `/` filter and the `:` jump
(§6.2) compile to. It is paginated like every other list op.

`KeyEntry` (returned by `key`, and inline in fixtures):
```jsonc
{
  "key": "model.layers.0.attn.wq.weight",
  "object_type": "TENSOR_SLICE",              // OBJECT | TENSOR | TENSOR_SLICE
  "dtype": "float32",                         // null for OBJECT
  "global_shape": [4096, 4096],               // null for OBJECT / plain TENSOR
  "fully_committed": true,                    // all mesh coords present
  "mesh_shape": [2, 2],                       // null unless TENSOR_SLICE
  "shards": [                                 // paginate if huge
    { "volume_id": "vol-0", "coordinates": [0,0], "offsets": [0,0], "local_shape": [2048,2048] }
  ]
}
```

### 5.3 `peek` — the only path that touches tensor data
```jsonc
{"op": "peek", "key": "model.layers.0.attn.wq.weight", "coordinates": [0,0]}
// -> { "dtype", "shape", "min", "max", "mean", "l2_norm", "head": [ ...first N... ] }
```
Computed near the data (agent side), so a large tensor never traverses the tree —
only its stats + a small head do.

### 5.4 Notes
- Every list op is **paginated** (`limit` + `cursor`) so no single response is
  unbounded, regardless of store size.
- Every list op takes **`sort_by`/`order`**, applied server-side before paging so
  anomaly-first / by-size ordering survives windowed rendering.
- `expand_prefix` mirrors the store's key trie exactly one level per call — the
  TUI's tree lazily requests children as nodes are opened. `search` is the
  path-free counterpart for jump/filter across the whole set.
- Per-node partials feeding the aggregator (§3) use the same `KeyEntry`/volume
  shapes with a `partial`/`rank` marker; merging is the aggregator's job, so the
  TUI-facing summary/drill-down responses are always pre-merged.
- **The same contract serves both scales.** For a small store (≤ a few thousand
  volumes — the common case today) a single-node agent can answer every op from an
  in-memory snapshot; at fleet scale the identical ops are served by the rollup
  aggregator (§3). The UI and `Provider` are unchanged — the aggregator is a
  scale-out swap, not a prerequisite for the tool to be useful.

---

## 6. UI design

**One primary view at a time, full terminal width, with a drill-stack and a
command bar** — the k9s model, not a fixed multi-pane DB browser. This is a
deliberate departure from an earlier `gitui`/`harlequin`-style three-pane sketch:
the tools that actually scale to huge object counts (k9s for Kubernetes, `btop`/
`htop` for thousands of processes, `ranger`/`nnn` for large filesystems) all use a
**single focused view + a navigation stack + a jump/command primitive**, never
three simultaneous narrow columns. The reason is concrete for *this* data: keys
like `model.layers.0.attn.wq.weight`, mesh coords, and volume ids are wide, and a
~24-column pane truncates them into uselessness. A full-width view keeps them
legible and the layout responsive on any terminal size.

Navigation is a **drill-stack** (like `ranger`/Miller columns / k9s breadcrumbs):
`Enter`/`l` pushes into the thing under the cursor (group → volumes; prefix → next
trie level; key → detail), `Esc`/`h` pops back. A breadcrumb in the header shows
where you are. Detail is a **pushed screen**, not a permanent pane — so it gets the
whole width for the shard table and stats.

Summary-first: the landing is a **health board** (§6.1), not a table; the trees
expand lazily via `expand_prefix` / `list_volumes` as the user drills in.

```
┌ toso-tui · torchstore · LocalRankStrategy · 1,048,576 vols · 5,203 keys · [live 2s] ┐
│ keys ▸ model ▸ layers                              scope: (all)   sort: partial↓    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  NAME                    KEYS    DTENSORS   PARTIAL   BYTES     STATUS              │
│  ▸ 0                       120        120       2 ⚠   1.4 TB    degraded            │
│  ▸ 7                       120        120       1 ⚠   1.4 TB    degraded            │
│  ▸ 1                       120        120       —     1.4 TB    ok                  │
│  ▸ 2                       120        120       —     1.4 TB    ok                  │
│  … 28 more (window 4/32 · PgDn for next page — lazy, paged)                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ ↑↓ move · l/⏎ drill · h/esc back · / filter · : jump · s sort · p peek · r · ? · q  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Views** (each full-width; `:` jumps between them, drill-stack moves within):
- **Health board** (landing, §6.1): rolled-up counts + anomaly gauges + histograms
  from `summary`. The first thing on screen; problems visible before any drilling.
- **Topology:** **volume groups** (host/rack/region) with counts and transport mix,
  not individual volumes — drilling a group calls `list_volumes` (paged). A ⚠ flags
  unreachable/partial. Selecting a group sets **scope** (shown in the header), which
  filters the keys/detail views to it — the analog of a k9s namespace.
- **Keys:** the key trie, each row showing rolled-up counts (`keys`, `dtensors`,
  `partial ⚠`, `bytes`). Drilling a row calls `expand_prefix` for just that level
  (lazy, paged) — never the whole keyspace.
- **Detail:** the selected key's `KeyEntry` (via `key`) — type, dtype, global
  shape, mesh, committed status, full-width paged shard table. `p` → `peek` renders
  tensor stats.

**Contextual footer.** The hint bar changes with the focused view (k9s/lazygit
style): a shard table offers `p peek`; the trie offers `l drill`; a group offers
drill-into-volumes. Static hints go stale and lie about what a key does *here*.

**Modes:** browse (default); filter (`/`, pushed down as a `search`, §5.2);
command/jump (`:`, §6.2); help (`?`). A status line shows errors, staleness
("summary 2s old"), and an in-flight spinner while a drill-down request is
outstanding.

**Keybindings (initial):** `↑↓`/`j k` move, `l`/`Enter` drill (lazy load),
`h`/`Esc` back up the stack, `g/G` top/bottom, `PgUp/PgDn` page, `/` filter,
`:` command/jump, `s` cycle sort, `p` peek, `r` refresh, `?` help, `q` quit.

### 6.1 Landing = a health board, and sort-by-problem everywhere

The landing view is a **Pulse-style health board** (after k9s Pulses), not a
table: big colored counts and gauges sourced from `summary` —
`⚠ 3 partial dtensors`, `6 unreachable volumes`, total bytes, and a
`shard_commit_pct` histogram rendered as a bar/sparkline. An operator sees *what's
wrong* in one glance, then drills into it. Health is perceived through numbers and
distribution, never enumeration.

Everywhere else, **anomaly-first is a real, toggleable sort**, not a fixed order.
`s` cycles the sort key (by `partial` ⚠, `bytes`, `reachable`, name); because the
TUI only holds a window (§6.3), the sort is **pushed to the aggregator** via
`sort_by`/`order` (§5.2) and applied before pagination. Default is anomaly-first
(unhealthy rows on top) — the `btop`/`htop` "sort by the column that matters so the
interesting few float up" pattern. The active sort is shown in the header
(`sort: partial↓`).

### 6.2 Command mode / jump — "jump, don't scroll" at the view level

Filtering (`/`) narrows *within* a view; `:` **jumps directly** to a target without
navigating there — the primitive that makes k9s usable on a huge cluster,
generalized by Textual's command palette. At million-key scale the operator usually
already knows what they want:

- `:key model.layers.0.attn.wq.weight` → straight to that key's detail.
- `:group rack:A12` → topology scoped to that group.
- `:partial` / `:unreachable` → an anomaly view (a `search` + anomaly sort).
- `:peek <key>` → straight to tensor stats.

Each command compiles to one bounded aggregator op (`key`, `search`, `list_volumes`,
`peek`), so "find node X among a million" collapses from a region→rack→host drill to
a single keystroke when the target is known.

### 6.3 Rendering at scale (millions of nodes)

The UI **never displays millions of nodes** — a terminal shows ~40–60 rows, and
the data can't be held or shipped wholesale. "Display millions" is reframed as
"always show ≤ a screenful of *meaningful* rows, and let the user reach any one."
Five techniques stack:

1. **Aggregate by default — nodes as numbers, not rows.** The landing view shows
   *groups*, never individual nodes: a million volumes collapse to a few hundred
   rack rows, each one line (`rack:A12 · 4096 vols · RDMA·4090 shm·6 · 3 ⚠`). Scale
   is perceived via counts / transport mix / histograms, not enumeration. Fan-out
   is bounded at every level (region → rack → host → volume), so the user only ever
   faces the *hundreds of children* of the node they opened — never a flat million.

2. **Lazy, paginated fetch — pull only what's about to be on screen.** Expanding a
   node calls `expand_prefix` / `list_volumes` for that level only, `limit`+`cursor`
   paged; the next page is fetched as the user scrolls toward it. Nothing off-screen
   is fetched.

3. **Virtualized (windowed) rendering.** Even one group of 4096 hosts is never fully
   materialized: the UI keeps a scroll offset + a small windowed buffer and draws
   only the visible slice (ratatui's stateful `List`/`Table` render only on-screen
   rows). Draw cost and memory are O(visible), independent of total size.

4. **Search/jump, not scroll (§6.2).** At scale you don't scroll to find a node —
   `/` filters by hostname / rack / key-prefix (pushed down as a `search`, §5.2) and
   `:` jumps straight to a known key/group. Both return only the small matching set;
   reaching a specific node is one query, not a million-row page-through.

5. **Anomaly-first ordering (§6.1).** The interesting rows are the sick few, not the
   healthy majority. Server-side `sort_by` floats problem rows (unreachable /
   partial-DTensor / hot) to the top so the rows worth seeing fit on one screen;
   histograms show the fleet-wide *distribution* (e.g. shard-commit %, bytes/rack) in
   a few cells.

**Memory:** the TUI holds only the expanded path + visible windows, with LRU
eviction on collapse — footprint is O(what's open on screen), never O(fleet). A
"find node X among a million" flow is: land on `summary` → either search-jump or
drill region → rack → host, each step one bounded paginated request in a
virtualized list — a handful of pages touched total.

Implementation note: the data layer exposes each level as a `LazyList` — a windowed
view over `(total_count, loaded_pages)` that requests pages on demand and reports
`total_count` to the widget for an accurate scrollbar without holding all rows.

---

## 7. Tech stack

- **`ratatui`** + **`crossterm`** — TUI framework + terminal backend.
- **`serde` / `serde_json`** — the query/response schema structs (§5).
- **`tokio`** — async event loop + aggregator client; drill-down requests are
  async so the UI never blocks on a round-trip.
- Line-delimited JSON over a plain TCP stream is the wire format to the aggregator
  — no TLS/gRPC deps; `tokio` + `serde_json` cover it.
- **`anyhow` / `color-eyre`** — errors; **`clap`** — CLI args
  (`--aggregator <host:port>`, `--fixtures <dir>`, `--refresh <secs>`).
- Layout: a small crate — `main.rs`, `model.rs` (schema), `data.rs`
  (`Provider` trait + `FileProvider`/`AggregatorClient`), `ui/` (views, drill-stack,
  lazy tree, event loop).

**Architecture — Elm-style Model–Update–View.** A single `App` state, an input/
event/data-arrival message enum, an `update(msg)` reducer, and a pure `view(state)`
render. This is the idiom most robust `ratatui` apps use and it keeps the
fixtures-driven tests clean — feed messages, assert on state.

**Async / render discipline.** The render thread never blocks on IO: drill-down
requests go out on `tokio` and their results land back via a channel as messages
that mark the state dirty. Render only on input, data-arrival, or the refresh tick
— not at a fixed FPS. Each list level is a `LazyList` (§6.3) whose in-flight pages
render a spinner row so a partial load is visible, never a freeze.

**Text handling.** Middle-elide long keys (`model.…attn.wq.weight`, keeping the
distinguishing tail) rather than right-truncating; the full-width Detail view shows
the whole key. Color rows by status (ok / degraded / unreachable) and by object
type, so anomalies read at a glance.

**Build.** A standalone `cargo` crate under `tui/` — one `cargo build`, deps from
crates.io, no extra build system.

---

## 8. Risks & open questions

- **Attaching to a live store.** Cleanest is for the job to *opt in* by spawning
  the agent actor (it already has controller + volume handles). Whether a fully
  external process can attach to an existing Monarch world is a Monarch-addressing
  question — deferred; MVP assumes the job cooperates (the agent runs in-world).
- **Transport field.** Not exposed by an endpoint today; it's only logged. v1 may
  parse the resolved-transport log line or leave `transport: null` until an
  endpoint exists.
- **Bytes / sizes.** `get_meta` gives shape+dtype (so size is computable), but
  summing across shards costs calls — make `bytes` optional/lazy.
- **Consistency.** Summaries are point-in-time and roll up across many nodes
  collected at slightly different instants; a training job mutates concurrently.
  Show age; treat counts as approximate, not transactional.
- **`peek` cost.** `ts.get` moves a real tensor; cap by numel and only ever send
  computed stats + a small head across the tree.
- **Aggregator is a scale-out swap, not a v1 prerequisite.** The fleet-scale rollup
  tier (hierarchical fan-in + summarization) is larger than the TUI and assumed as
  infra. But the tool is useful *before* it exists: the identical `Provider`/ops are
  served by a single-node agent from an in-memory snapshot for the common case (≤ a
  few thousand volumes, cf. k9s serving thousands of objects from a watch cache
  without server-side paging). The single-node leaf agent + the query contract are
  what make the tool useful; the tree is a later swap behind the same contract (§5.4).
- **Rollup fidelity.** Group-bys/histograms lose detail by construction (that's
  the point). The drill-down must be able to reach any individual key/volume the
  user asks for, or the tool feels blind — so `expand_prefix`/`list_volumes` must
  page all the way down, not cap at top-N.

## 9. Future ideas
- Throughput/latency sparklines by tapping `LatencyTracker` (`torchstore/logging.py`).
- A non-interactive `dump` subcommand (JSON/plain) for scripting.
- Watch mode that diffs snapshots (keys added/removed, shards filling in).
- Optional export of a snapshot to an external metrics or dashboard system.
