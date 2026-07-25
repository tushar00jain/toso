//! The data layer: a `Provider` trait the UI talks to, and a `FileProvider`
//! that answers every op from hand-written JSON fixtures on disk (SPEC §8
//! milestone 1). `AggregatorClient` implements the same trait over a TCP
//! connection, so the UI never learns which one it holds.

mod aggregator;

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use serde::de::DeserializeOwned;

pub use crate::data::aggregator::AggregatorClient;
use crate::model::ExpandPrefixRequest;
use crate::model::ExpandPrefixResponse;
use crate::model::KeyEntry;
use crate::model::KeyRequest;
use crate::model::ListVolumesRequest;
use crate::model::ListVolumesResponse;
use crate::model::PeekRequest;
use crate::model::PeekResult;
use crate::model::SearchRequest;
use crate::model::SearchResponse;
use crate::model::Summary;

/// The one interface the UI depends on. Each method maps to a §5 op; list ops
/// carry their pagination/sort params in the request struct.
#[async_trait]
pub trait Provider: Send + Sync {
    async fn summary(&self) -> Result<Summary>;
    async fn expand_prefix(&self, req: ExpandPrefixRequest) -> Result<ExpandPrefixResponse>;
    async fn list_volumes(&self, req: ListVolumesRequest) -> Result<ListVolumesResponse>;
    async fn key(&self, req: KeyRequest) -> Result<KeyEntry>;
    async fn search(&self, req: SearchRequest) -> Result<SearchResponse>;
    async fn peek(&self, req: PeekRequest) -> Result<PeekResult>;
}

/// Serves fixtures from `dir`, using this file layout (params become filenames):
///
/// - `summary`        -> `summary.json`
/// - `expand_prefix`  -> `expand_prefix/<prefix>.json`
/// - `list_volumes`   -> `list_volumes/<group>.json`
/// - `key`            -> `key/<key>.json`
/// - `search`         -> `search/<pattern>.json`
/// - `peek`           -> `peek/<key>.json`, falling back to `peek.json`
///
/// Params map to filenames near-verbatim: keys like `model.layers.0.attn.wq`
/// are used as-is, but `:` is rewritten to `-` for the on-disk name (a colon is
/// awkward in filenames and illegal on some filesystems). So group `rack:A12`
/// reads `list_volumes/rack-A12.json` while the group label keeps its colon.
pub struct FileProvider {
    dir: PathBuf,
}

/// Map a request param to its on-disk filename component. `:` is the one
/// character we rewrite (illegal in filenames on Windows/exFAT, awkward
/// elsewhere); every other char a param can hold is filename-safe on unix.
fn fixture_component(param: &str) -> String {
    param.replace(':', "-")
}

impl FileProvider {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    async fn read_fixture<T: DeserializeOwned>(&self, rel: impl AsRef<Path>) -> Result<T> {
        let path = self.dir.join(rel);
        let bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("reading fixture {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing fixture {}", path.display()))
    }
}

#[async_trait]
impl Provider for FileProvider {
    async fn summary(&self) -> Result<Summary> {
        self.read_fixture("summary.json").await
    }

    async fn expand_prefix(&self, req: ExpandPrefixRequest) -> Result<ExpandPrefixResponse> {
        // A page request (cursor set) is served from a per-cursor fixture; when
        // none exists, pagination terminates cleanly with an empty page rather
        // than re-serving page one forever.
        match &req.cursor {
            Some(cursor) => {
                let rel = format!("expand_prefix/{}.json", fixture_component(cursor));
                if self.dir.join(&rel).exists() {
                    self.read_fixture(rel).await
                } else {
                    Ok(ExpandPrefixResponse {
                        children: Vec::new(),
                        next_cursor: None,
                    })
                }
            }
            None => {
                self.read_fixture(format!("expand_prefix/{}.json", fixture_component(&req.prefix)))
                    .await
            }
        }
    }

    async fn list_volumes(&self, req: ListVolumesRequest) -> Result<ListVolumesResponse> {
        match &req.cursor {
            Some(cursor) => {
                let rel = format!("list_volumes/{}.json", fixture_component(cursor));
                if self.dir.join(&rel).exists() {
                    self.read_fixture(rel).await
                } else {
                    Ok(ListVolumesResponse {
                        volumes: Vec::new(),
                        next_cursor: None,
                    })
                }
            }
            None => {
                self.read_fixture(format!("list_volumes/{}.json", fixture_component(&req.group)))
                    .await
            }
        }
    }

    async fn key(&self, req: KeyRequest) -> Result<KeyEntry> {
        self.read_fixture(format!("key/{}.json", fixture_component(&req.key)))
            .await
    }

    async fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        match &req.cursor {
            Some(cursor) => {
                let rel = format!("search/{}.json", fixture_component(cursor));
                if self.dir.join(&rel).exists() {
                    self.read_fixture(rel).await
                } else {
                    Ok(SearchResponse {
                        matches: Vec::new(),
                        next_cursor: None,
                    })
                }
            }
            None => {
                self.read_fixture(format!("search/{}.json", fixture_component(&req.pattern)))
                    .await
            }
        }
    }

    async fn peek(&self, req: PeekRequest) -> Result<PeekResult> {
        let rel = format!("peek/{}.json", fixture_component(&req.key));
        if self.dir.join(&rel).exists() {
            return self.read_fixture(rel).await;
        }
        self.read_fixture("peek.json").await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ObjectType;
    use crate::model::SearchKind;
    use crate::model::SearchMatch;

    fn provider() -> FileProvider {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        FileProvider::new(dir)
    }

    #[tokio::test]
    async fn summary_deserializes() {
        let s = provider().summary().await.expect("summary fixture");
        assert_eq!(s.schema_version, 1);
        assert_eq!(s.store_name, "torchstore");
        assert!(!s.volume_groups.is_empty(), "expected volume groups");
        assert!(!s.key_prefixes.is_empty(), "expected key prefixes");
        assert!(
            !s.histograms.shard_commit_pct.is_empty(),
            "expected a shard_commit_pct histogram"
        );
    }

    #[tokio::test]
    async fn expand_prefix_paginates() {
        let resp = provider()
            .expand_prefix(ExpandPrefixRequest {
                prefix: "model".to_string(),
                limit: Some(200),
                cursor: None,
                sort_by: None,
                order: None,
            })
            .await
            .expect("expand_prefix model fixture");
        assert!(
            !resp.children.is_empty(),
            "expected children one level deeper"
        );
        assert!(resp.next_cursor.is_some(), "model fixture should page");
    }

    #[tokio::test]
    async fn expand_prefix_deeper_level() {
        let resp = provider()
            .expand_prefix(ExpandPrefixRequest {
                prefix: "model.layers".to_string(),
                limit: Some(200),
                cursor: None,
                sort_by: None,
                order: None,
            })
            .await
            .expect("expand_prefix model.layers fixture");
        assert!(resp.children.iter().any(|c| c.prefix == "model.layers.0"));
    }

    #[tokio::test]
    async fn list_volumes_deserializes() {
        let resp = provider()
            .list_volumes(ListVolumesRequest {
                group: "rack:A12".to_string(),
                limit: Some(200),
                cursor: None,
                sort_by: None,
                order: None,
            })
            .await
            .expect("list_volumes rack:A12 fixture");
        assert!(!resp.volumes.is_empty());
        assert!(resp.volumes.iter().any(|v| !v.reachable), "expect a ⚠ row");
    }

    #[tokio::test]
    async fn list_volumes_follows_the_page_cursor() {
        // Page one advertises `next_cursor: rack:A12:page2`; requesting that
        // cursor must serve the *second* page's fixture (distinct volumes), not
        // re-serve page one. Page two is terminal.
        let page2 = provider()
            .list_volumes(ListVolumesRequest {
                group: "rack:A12".to_string(),
                limit: Some(200),
                cursor: Some("rack:A12:page2".to_string()),
                sort_by: None,
                order: None,
            })
            .await
            .expect("second page fixture");
        assert!(
            page2.volumes.iter().any(|v| v.volume_id == "vol-4"),
            "page two serves new volumes, not page one again"
        );
        assert!(
            page2.next_cursor.is_none(),
            "page two is the last page — pagination terminates"
        );
    }

    #[tokio::test]
    async fn list_volumes_unknown_page_cursor_terminates() {
        // A cursor with no fixture must return an empty, terminal page — this is
        // what stops the volume list from growing without bound when the user
        // scrolls past the last authored page.
        let resp = provider()
            .list_volumes(ListVolumesRequest {
                group: "rack:A12".to_string(),
                limit: Some(200),
                cursor: Some("rack:A12:nonexistent-page".to_string()),
                sort_by: None,
                order: None,
            })
            .await
            .expect("a missing page cursor yields an empty page, not an error");
        assert!(resp.volumes.is_empty(), "no fixture for this cursor => empty");
        assert!(resp.next_cursor.is_none(), "empty page terminates pagination");
    }

    #[tokio::test]
    async fn key_committed_dtensor() {
        let entry = provider()
            .key(KeyRequest {
                key: "model.layers.0.attn.wq.weight".to_string(),
            })
            .await
            .expect("committed dtensor fixture");
        assert_eq!(entry.object_type, ObjectType::TensorSlice);
        assert!(entry.fully_committed);
        assert_eq!(entry.mesh_shape, Some(vec![2, 2]));
        assert_eq!(entry.shards.len(), 4, "mesh [2,2] => 4 shards present");
    }

    #[tokio::test]
    async fn key_partial_dtensor() {
        let entry = provider()
            .key(KeyRequest {
                key: "model.layers.7.attn.wq.weight".to_string(),
            })
            .await
            .expect("partial dtensor fixture");
        assert_eq!(entry.object_type, ObjectType::TensorSlice);
        assert!(!entry.fully_committed, "partial dtensor is not committed");
        assert!(
            entry.shards.len() < 4,
            "partial mesh [2,2] should be missing shards"
        );
    }

    #[tokio::test]
    async fn key_plain_tensor() {
        let entry = provider()
            .key(KeyRequest {
                key: "model.embed.weight".to_string(),
            })
            .await
            .expect("plain tensor fixture");
        assert_eq!(entry.object_type, ObjectType::Tensor);
        assert!(entry.dtype.is_some());
        assert!(
            entry.global_shape.is_none(),
            "plain tensor has no global_shape"
        );
        assert!(entry.mesh_shape.is_none());
    }

    #[tokio::test]
    async fn key_object() {
        let entry = provider()
            .key(KeyRequest {
                key: "metadata.config".to_string(),
            })
            .await
            .expect("object fixture");
        assert_eq!(entry.object_type, ObjectType::Object);
        assert!(entry.dtype.is_none(), "OBJECT has null dtype");
        assert!(entry.global_shape.is_none(), "OBJECT has null global_shape");
    }

    #[tokio::test]
    async fn search_deserializes() {
        let resp = provider()
            .search(SearchRequest {
                kind: SearchKind::Key,
                pattern: "attn.wq".to_string(),
                limit: Some(200),
                cursor: None,
                sort_by: None,
                order: None,
            })
            .await
            .expect("search fixture");
        assert!(!resp.matches.is_empty());
        assert!(matches!(resp.matches[0], SearchMatch::Key(_)));
    }

    #[tokio::test]
    async fn peek_deserializes() {
        let resp = provider()
            .peek(PeekRequest {
                key: "model.layers.0.attn.wq.weight".to_string(),
                coordinates: Some(vec![0, 0]),
            })
            .await
            .expect("peek fixture");
        assert_eq!(resp.dtype, "float32");
        assert!(!resp.head.is_empty(), "peek carries a small head");
        assert!(resp.max >= resp.min);
    }
}
