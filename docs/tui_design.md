# `toso-tui`: inspect a live TorchStore

**Status:** living specification

`toso-tui` is a Rust [`ratatui`](https://ratatui.rs) application for exploring a
running TorchStore: its topology, keys, object metadata, and DTensor shard layout.
It is an interactive database browser for one live store.

## Scope

The UI should:

- Show the controller strategy and volumes grouped by host, rack, or region,
  including reachability and negotiated transports.
- Browse the store's trie-shaped keyspace.
- Show a key's object type, dtype, shape, holders, DTensor mesh, shards, and commit
  status.
- Optionally compute bounded tensor statistics near the data: min, max, mean, norm,
  and the first N values.

Version 1 is read-only. It does not dump tensors or large objects, retain history,
compare jobs, or provide a hosted web dashboard.

## TorchStore introspection surface

The required state already exists behind Monarch endpoints:

- `Controller.keys(prefix=None)` reads the key trie.
- `Controller.locate_volumes(...)` maps keys to `{volume_id: StorageInfo}`.
  `StorageInfo` identifies `OBJECT`, `TENSOR`, or `TENSOR_SLICE` data and carries
  tensor slices.
- `Controller.get_controller_strategy()` identifies the topology strategy.
- The controller's DTensor commit check compares stored coordinates with
  `product(*mesh_shape)`.
- `StorageVolume.get_id()` returns volume identity and hostname.
- `StorageVolume.get_meta()` returns shape and dtype, or `"obj"`, without moving
  tensor data.
- `TensorSlice` provides coordinates, mesh/global/local shapes, and offsets.

The negotiated transport is only logged during volume initialization, so it may
remain unknown until TorchStore exposes it through an endpoint.

## Process boundary and aggregation

TorchStore state lives in Python objects inside Monarch actors. A separate Rust
process cannot read that state or simply join the actor mesh. An in-world Python
agent therefore queries the controller and volumes, then serves the TUI over one
TCP connection using line-delimited JSON.

For version 1, that agent is also the aggregator: it answers the protocol below
directly for a small or moderate store. The TUI never connects to storage nodes.
At fleet scale, the same endpoint can be backed by hierarchical fan-in:

```text
volume agents -> host/rack/region rollups -> root aggregator -> one TUI connection
```

Each tier combines local partials into counts, group-bys, histograms, and bounded
pages. Detail remains queryable on demand. This keeps fan-in bounded and avoids
moving a complete fleet snapshot to the TUI. The rollup service is future
infrastructure behind the protocol, not a version 1 dependency.

The TorchStore controller is a source, not the aggregator: it has the global
keys-to-volumes index but not all per-volume metadata or tensor statistics.

### Runtime variants

SPMD does not change the design. `torchstore/spmd.py` builds a Monarch host mesh,
rank 0 spawns volume actors, and the controller handle is broadcast through
`TCPStore`. The agent must start inside that Monarch context and reuse the handle.

A deployment without a controller would run one collector per rank. Each collector
would emit a local snapshot marked `partial: true` and `rank`; an aggregator would
merge key unions, volume attribution, and shard coordinates before serving the same
TUI protocol. Controller collection is the single-producer form of this model.

### Transport and providers

The live provider connects with `--aggregator <host:port>`. Remote deployments use
an existing SSH tunnel or port-forward rather than a second wire protocol. Plain
TCP assumes a trusted endpoint; production security belongs to the deployment.

An offline `--fixtures <dir>` provider reads the same JSON contract. Both implement
one Rust `Provider` trait, so rendering and navigation are transport-independent.

## Query protocol

The protocol is summary-first and metadata-only. It sends a bounded landing view,
then fetches detail as the user navigates. Tensor values cross the boundary only as
the bounded result of an explicit `peek`.

### Summary

`summary` returns store-wide health and the first aggregation level. Its size is
`O(groups)`, never `O(nodes + keys)`.

```jsonc
{"op": "summary"}
// ->
{
  "schema_version": 1,
  "captured_at": "2026-07-17T17:40:00Z",
  "store_name": "torchstore",
  "strategy": "LocalRankStrategy",
  "totals": {
    "volumes": 1048576,
    "keys": 5203,
    "bytes": 9.2e15,
    "partial_dtensors": 3
  },
  "volume_groups": [
    {
      "group": "rack:A12",
      "volumes": 4096,
      "keys": 5203,
      "bytes": 3.1e14,
      "transports": {"RDMA": 4090, "SharedMemory": 6},
      "reachable": 4096
    }
  ],
  "key_prefixes": [
    {
      "prefix": "model",
      "keys": 4820,
      "objects": 0,
      "tensors": 12,
      "dtensors": 4808,
      "partial": 2,
      "bytes": 8.9e15
    }
  ],
  "histograms": {
    "shard_commit_pct": [[0, 5], [50, 12], [100, 4790]]
  }
}
```

### Drill-down

Every list operation accepts `limit`, `cursor`, `sort_by`, and `order`. Sorting is
server-side because the TUI owns only one page and cannot correctly sort the full
set. The default order is anomaly-first.

```jsonc
{"op": "expand_prefix", "prefix": "model.layers", "limit": 200,
 "cursor": null, "sort_by": "partial", "order": "desc"}
// -> {"children": [{"prefix": "model.layers.0", "keys": 120}],
//     "next_cursor": "..."}

{"op": "list_volumes", "group": "rack:A12", "limit": 200,
 "cursor": null, "sort_by": "bytes", "order": "desc"}
// -> {"volumes": [{"volume_id": "vol-0", "hostname": "host-0",
//                  "transport": "RDMA", "num_keys": 42,
//                  "bytes": 1024, "reachable": true}],
//     "next_cursor": "..."}

{"op": "key", "key": "model.layers.0.attn.wq.weight"}
// -> KeyEntry

{"op": "search", "kind": "key", "pattern": "attn.wq", "limit": 200,
 "cursor": null, "sort_by": "partial", "order": "desc"}
// -> {"matches": [{"key": "model.layers.0.attn.wq.weight"}],
//     "next_cursor": "..."}

{"op": "search", "kind": "volume", "pattern": "rack:A12", "limit": 200}
// -> {"matches": [{"volume_id": "vol-0", "hostname": "host-0"}]}
```

`expand_prefix` descends one known trie level. `search` finds keys or volumes
without a known path and powers filtering and direct jumps. Both are paginated.
Supported sort keys include `partial`, `reachable`, `bytes`, `keys`, and `name`.

`key` returns full detail only for the selected key:

```jsonc
{
  "key": "model.layers.0.attn.wq.weight",
  "object_type": "TENSOR_SLICE",
  "dtype": "float32",
  "global_shape": [4096, 4096],
  "fully_committed": true,
  "mesh_shape": [2, 2],
  "shards": [
    {
      "volume_id": "vol-0",
      "coordinates": [0, 0],
      "offsets": [0, 0],
      "local_shape": [2048, 2048]
    }
  ]
}
```

Large shard lists are paginated. Object entries omit tensor fields; plain tensors
omit mesh fields.

### Peek

```jsonc
{"op": "peek", "key": "model.layers.0.attn.wq.weight",
 "coordinates": [0, 0]}
// -> {"dtype": "float32", "shape": [2048, 2048], "min": -1.2,
//     "max": 1.3, "mean": 0.01, "l2_norm": 42.0, "head": [0.1, 0.2]}
```

The agent computes these values near the tensor. It caps the work by element count
and sends only statistics plus a small head.

### Contract invariants

- Every list response is paginated and bounded.
- Sorting precedes pagination at the server.
- The TUI receives merged responses; partial-snapshot merging belongs upstream.
- The same operations serve a single-agent deployment and a fleet rollup.
- Tensor data is inaccessible except through bounded `peek` results.

## UI

The application shows one full-width primary view with a drill stack and command
bar. A fixed multi-pane layout would truncate long keys, volume IDs, mesh
coordinates, and shard metadata. Detail is therefore a pushed screen, not a narrow
permanent pane.

`Enter` or `l` pushes the selected group, prefix, or key. `Esc` or `h` pops. The
header shows breadcrumbs, scope, sort, and data age.

```text
┌ toso-tui · torchstore · LocalRankStrategy · 1,048,576 vols · 5,203 keys · live 2s ┐
│ keys ▸ model ▸ layers                              scope: all   sort: partial↓    │
├───────────────────────────────────────────────────────────────────────────────────┤
│  NAME                    KEYS    DTENSORS   PARTIAL   BYTES     STATUS            │
│  ▸ 0                       120        120       2 ⚠   1.4 TB    degraded          │
│  ▸ 7                       120        120       1 ⚠   1.4 TB    degraded          │
│  ▸ 1                       120        120       —     1.4 TB    ok                │
│  … 29 more · window 3/32 · PgDn loads the next page                               │
├───────────────────────────────────────────────────────────────────────────────────┤
│ ↑↓ move · l/⏎ drill · h/esc back · / filter · : jump · s sort · p peek · ? · q    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Views

- **Health** is the landing screen. It renders totals, partial DTensors,
  unreachable volumes, bytes, and commit histograms from `summary`.
- **Topology** shows volume groups and transport mix. Drilling calls
  `list_volumes`. Selecting a group sets a scope used by other views.
- **Keys** shows one trie level with key, DTensor, partial, and byte counts.
  Drilling calls `expand_prefix`.
- **Detail** shows a selected `KeyEntry` and its paginated shard table. `p` opens
  tensor statistics from `peek`.

The footer is contextual: it advertises only actions valid for the focused view.
A status line reports errors, staleness, and in-flight requests.

### Navigation and modes

Browse is the default mode. `/` filters through `search`; `:` opens direct commands:

- `:key model.layers.0.attn.wq.weight` opens a key.
- `:group rack:A12` scopes topology to a group.
- `:partial` and `:unreachable` open anomaly searches.
- `:peek <key>` opens tensor statistics.

Known targets should take one bounded query rather than a long drill path.

Initial bindings are `↑↓`/`j k` to move, `l`/`Enter` to drill, `h`/`Esc` to go
back, `g/G` for top/bottom, `PgUp/PgDn` for pages, `/` to filter, `:` to jump, `s`
to sort, `p` to peek, `r` to refresh, `?` for help, and `q` to quit.

### Scale invariants

The UI never materializes the fleet. It keeps a screenful of useful rows and makes
any specific item reachable:

1. The landing screen and topology start with groups, counts, and histograms rather
   than individual nodes.
2. Drill-down fetches one level and one page at a time.
3. A windowed `LazyList` stores visible pages, total count, scroll position, and
   in-flight requests. Collapsed pages are LRU-evicted.
4. Search and jump avoid enumeration; anomaly-first server sorting keeps failures
   above the healthy majority.

Render and memory costs are `O(open path + visible windows)`, independent of total
fleet size. Widgets receive `total_count` so scrollbars remain accurate without
loading every row.

## Implementation

- `ratatui` and `crossterm` provide rendering and terminal input.
- `serde` and `serde_json` define the protocol types.
- `tokio` runs the socket client and background requests.
- `clap` exposes `--aggregator`, `--fixtures`, and `--refresh`.
- `anyhow` or `color-eyre` reports errors.

The standalone crate lives under `tui/` with `main.rs`, `model.rs`, `data.rs`, and
`ui/`. `data.rs` defines `Provider`, `FileProvider`, and `AggregatorClient`.

Use an Elm-style Model-Update-View loop: one `App` state, a message enum for input
and data arrival, an `update` reducer, and a pure render function. This makes fixture
tests deterministic: feed messages and assert on state.

Rendering never blocks on I/O. Tokio tasks return responses through a channel;
input, responses, and refresh ticks mark the state dirty. In-flight pages render a
spinner row. There is no fixed-FPS redraw loop.

Middle-elide long keys while preserving their distinguishing suffix; the detail
view shows the full key. Color indicates object type and health, but text and
symbols must carry the same meaning.

## Risks and open questions

- **Live attachment.** Version 1 requires the job to spawn the agent inside its
  Monarch world. Attaching an external process depends on Monarch addressing.
- **Transport identity.** The resolved transport is logged but not queryable; leave
  it null or parse the log until an endpoint exists.
- **Byte counts.** Shape and dtype make size computable, but collecting every
  volume's metadata is expensive. Keep bytes optional or lazy.
- **Snapshot consistency.** A rollup combines observations from different instants
  while the store mutates. Show capture age and treat totals as approximate.
- **Peek cost.** Computing statistics touches tensor data. Enforce element and time
  limits and return only bounded output.
- **Rollup fidelity.** Summaries intentionally lose detail. Drill-down must still
  page to every requested key or volume rather than stop at a top-N result.

## Future work

- Throughput and latency sparklines from `LatencyTracker`.
- A non-interactive JSON or text `dump` command.
- Snapshot diffs for added keys, removed keys, and shards becoming committed.
- Export to an external metrics or dashboard system.
