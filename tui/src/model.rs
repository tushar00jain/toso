//! Serde types mirroring the aggregator query protocol (SPEC §5).
//!
//! These types are the stable data contract between the TUI and whatever
//! serves it (a `FileProvider` over fixtures, or a real `AggregatorClient`).
//! The wire form is line-delimited JSON: requests carry an `"op"` tag, and
//! responses are the plain JSON objects documented in §5.

use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Shared enums
// ---------------------------------------------------------------------------

/// The object kind a stored key resolves to (mirrors torchstore `StorageInfo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ObjectType {
    Object,
    Tensor,
    TensorSlice,
}

/// Server-side sort key for every list op. Default (anomaly-first) is `Partial`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortKey {
    Partial,
    Bytes,
    Reachable,
    Keys,
    Name,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    Asc,
    Desc,
}

/// What a `search` op walks over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchKind {
    Key,
    Volume,
}

// ---------------------------------------------------------------------------
// summary (§5.1)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub schema_version: u32,
    /// ISO-8601 instant the rollup was captured; the UI shows its age.
    pub captured_at: String,
    pub store_name: String,
    pub strategy: String,
    pub totals: Totals,
    pub volume_groups: Vec<VolumeGroup>,
    /// Top level of the key trie only; descend with `expand_prefix`.
    pub key_prefixes: Vec<KeyPrefix>,
    #[serde(default)]
    pub histograms: Histograms,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Totals {
    pub volumes: u64,
    pub keys: u64,
    /// Byte counts can exceed u64-friendly literals and arrive as floats
    /// (e.g. `9.2e15`), so they are `f64` throughout.
    pub bytes: f64,
    pub partial_dtensors: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeGroup {
    /// Group label, e.g. `rack:A12` (grouped by host/rack/region).
    pub group: String,
    pub volumes: u64,
    pub keys: u64,
    pub bytes: f64,
    /// Transport name -> count of volumes that negotiated it.
    pub transports: HashMap<String, u64>,
    /// Count of reachable volumes in the group (not a bool).
    pub reachable: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyPrefix {
    pub prefix: String,
    pub keys: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objects: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensors: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtensors: Option<u64>,
    /// Count of partially-committed DTensors under this prefix (the ⚠ column).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial: Option<u64>,
    /// Total bytes of the tensors under this prefix (the BYTES column). `None`
    /// for older snapshots that predate the field; rendered as `—` then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<f64>,
    /// True when this trie node is itself a stored key (a terminal leaf), not
    /// just an intermediate prefix. The UI drills a leaf straight to Detail and
    /// an internal node one level deeper — a node's `keys` count alone can't
    /// distinguish the two (a single key can live below an intermediate node).
    #[serde(default)]
    pub is_leaf: bool,
}

/// At-a-glance distributions. Each histogram is a list of `[bucket, count]`
/// pairs (e.g. `shard_commit_pct` maps commit-% bucket -> key count).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Histograms {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shard_commit_pct: Vec<[u64; 2]>,
}

// ---------------------------------------------------------------------------
// expand_prefix (§5.2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpandPrefixResponse {
    /// One trie level deeper than the requested prefix.
    pub children: Vec<KeyPrefix>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// list_volumes (§5.2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Volume {
    pub volume_id: String,
    pub hostname: String,
    /// Negotiated transport (SharedMemory / RDMA / gloo / …). Null until an
    /// endpoint exposes it (SPEC §9).
    #[serde(default)]
    pub transport: Option<String>,
    pub num_keys: u64,
    pub bytes: f64,
    /// Per-volume reachability (a bool, unlike `VolumeGroup::reachable`).
    pub reachable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListVolumesResponse {
    pub volumes: Vec<Volume>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// key (§5.2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shard {
    pub volume_id: String,
    /// Position of this shard within the device mesh.
    pub coordinates: Vec<u64>,
    /// Element offset of this shard into the global tensor.
    pub offsets: Vec<u64>,
    pub local_shape: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyEntry {
    pub key: String,
    pub object_type: ObjectType,
    /// Null for `OBJECT`.
    #[serde(default)]
    pub dtype: Option<String>,
    /// Null for `OBJECT` and plain `TENSOR`.
    #[serde(default)]
    pub global_shape: Option<Vec<u64>>,
    /// True when every mesh coordinate is present.
    pub fully_committed: bool,
    /// Null unless `TENSOR_SLICE`.
    #[serde(default)]
    pub mesh_shape: Option<Vec<u64>>,
    #[serde(default)]
    pub shards: Vec<Shard>,
}

// ---------------------------------------------------------------------------
// search (§5.2)
// ---------------------------------------------------------------------------

/// A single `search` hit — a key entry or a volume, depending on `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SearchMatch {
    Key(KeyEntry),
    Volume(Volume),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub matches: Vec<SearchMatch>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

// ---------------------------------------------------------------------------
// peek (§5.3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeekResult {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub l2_norm: f64,
    /// First N elements only — never the full tensor.
    pub head: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Request parameter structs (ergonomic Provider inputs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpandPrefixRequest {
    pub prefix: String,
    #[serde(default)]
    pub limit: Option<u64>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub sort_by: Option<SortKey>,
    #[serde(default)]
    pub order: Option<Order>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListVolumesRequest {
    pub group: String,
    #[serde(default)]
    pub limit: Option<u64>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub sort_by: Option<SortKey>,
    #[serde(default)]
    pub order: Option<Order>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyRequest {
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub kind: SearchKind,
    pub pattern: String,
    #[serde(default)]
    pub limit: Option<u64>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub sort_by: Option<SortKey>,
    #[serde(default)]
    pub order: Option<Order>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeekRequest {
    pub key: String,
    #[serde(default)]
    pub coordinates: Option<Vec<u64>>,
}

// ---------------------------------------------------------------------------
// Wire request enum (the `{"op": "..."}` form, SPEC §5)
// ---------------------------------------------------------------------------

/// Tagged union matching the on-the-wire request form. `AggregatorClient`
/// (task #4) serializes one of these per query; the `From` impls below turn a
/// parameter struct into its wire request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Summary,
    ExpandPrefix {
        prefix: String,
        limit: Option<u64>,
        cursor: Option<String>,
        sort_by: Option<SortKey>,
        order: Option<Order>,
    },
    ListVolumes {
        group: String,
        limit: Option<u64>,
        cursor: Option<String>,
        sort_by: Option<SortKey>,
        order: Option<Order>,
    },
    Key {
        key: String,
    },
    Search {
        kind: SearchKind,
        pattern: String,
        limit: Option<u64>,
        cursor: Option<String>,
        sort_by: Option<SortKey>,
        order: Option<Order>,
    },
    Peek {
        key: String,
        coordinates: Option<Vec<u64>>,
    },
}

impl From<ExpandPrefixRequest> for Request {
    fn from(r: ExpandPrefixRequest) -> Self {
        Request::ExpandPrefix {
            prefix: r.prefix,
            limit: r.limit,
            cursor: r.cursor,
            sort_by: r.sort_by,
            order: r.order,
        }
    }
}

impl From<ListVolumesRequest> for Request {
    fn from(r: ListVolumesRequest) -> Self {
        Request::ListVolumes {
            group: r.group,
            limit: r.limit,
            cursor: r.cursor,
            sort_by: r.sort_by,
            order: r.order,
        }
    }
}

impl From<KeyRequest> for Request {
    fn from(r: KeyRequest) -> Self {
        Request::Key { key: r.key }
    }
}

impl From<SearchRequest> for Request {
    fn from(r: SearchRequest) -> Self {
        Request::Search {
            kind: r.kind,
            pattern: r.pattern,
            limit: r.limit,
            cursor: r.cursor,
            sort_by: r.sort_by,
            order: r.order,
        }
    }
}

impl From<PeekRequest> for Request {
    fn from(r: PeekRequest) -> Self {
        Request::Peek {
            key: r.key,
            coordinates: r.coordinates,
        }
    }
}
