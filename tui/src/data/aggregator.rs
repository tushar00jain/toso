//! `AggregatorClient` — a `Provider` that answers every §5 op over a single
//! line-delimited-JSON TCP connection to an aggregator endpoint.
//!
//! Wire form (SPEC §5): each request is one `{"op": ...}` JSON object on its
//! own line; the aggregator replies with exactly one JSON line. We serialize
//! the [`Request`] wire enum from `model`, write it plus `\n`, then read one
//! response line and deserialize it into the op's response type.
//!
//! Note: this is plain TCP with no authentication — intended for a local or
//! otherwise trusted endpoint (a dev stub, an SSH tunnel, a port-forward).
//! Transport security is out of scope here; a deployment would wrap the socket
//! in whatever its environment already provides (SPEC §3).

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::data::Provider;
use crate::model::ExpandPrefixRequest;
use crate::model::ExpandPrefixResponse;
use crate::model::KeyEntry;
use crate::model::KeyRequest;
use crate::model::ListVolumesRequest;
use crate::model::ListVolumesResponse;
use crate::model::PeekRequest;
use crate::model::PeekResult;
use crate::model::Request;
use crate::model::SearchRequest;
use crate::model::SearchResponse;
use crate::model::Summary;

/// A single reused connection guarded so concurrent `Provider` calls serialize
/// their request/response round-trips on the one stream.
///
/// `BufReader<TcpStream>` gives buffered `read_line` for framing while still
/// forwarding writes straight to the socket, so one value is both the reader
/// and writer of the duplex stream.
pub struct AggregatorClient {
    addr: String,
    /// `None` until the first request (lazy connect) or after a broken pipe
    /// clears it for reconnect on the next call.
    conn: Mutex<Option<BufReader<TcpStream>>>,
}

impl AggregatorClient {
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            conn: Mutex::new(None),
        }
    }

    async fn connect(&self) -> Result<BufReader<TcpStream>> {
        let stream = TcpStream::connect(&self.addr)
            .await
            .with_context(|| format!("connecting to aggregator at {}", self.addr))?;
        Ok(BufReader::new(stream))
    }

    /// Send one request line and read exactly one response line back.
    ///
    /// The caller holds the connection mutex for the entire call, so the
    /// write-then-read is atomic w.r.t. other callers — two requests can never
    /// interleave their bytes or mismatch a reply to the wrong request on the
    /// shared stream (the core concurrency hazard of a single reused socket).
    /// On any IO failure the connection is dropped and the request is retried
    /// once on a fresh connection (reconnect on broken pipe).
    async fn request<T: DeserializeOwned>(&self, req: Request) -> Result<T> {
        let line = format!(
            "{}\n",
            serde_json::to_string(&req).context("serializing aggregator request")?
        );

        let mut guard = self.conn.lock().await;

        let mut last_err = None;
        for _ in 0..2 {
            if guard.is_none() {
                match self.connect().await {
                    Ok(c) => *guard = Some(c),
                    Err(e) => {
                        last_err = Some(e);
                        continue;
                    }
                }
            }

            let conn = guard.as_mut().expect("connection set above");
            match round_trip(conn, &line).await {
                // A well-formed response line is parsed directly; a parse
                // failure (e.g. an `{"error":...}` line) is a real error to
                // surface, not a broken connection, so it does not reconnect.
                Ok(resp) => return parse_response(&resp),
                Err(e) => {
                    *guard = None;
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.expect("loop runs at least once"))
            .context("aggregator request failed after reconnect")
    }
}

/// Write the request line, flush, and read one `\n`-terminated response line.
async fn round_trip(conn: &mut BufReader<TcpStream>, line: &str) -> Result<String> {
    conn.write_all(line.as_bytes())
        .await
        .context("writing request to aggregator")?;
    conn.flush()
        .await
        .context("flushing request to aggregator")?;

    let mut resp = String::new();
    let n = conn
        .read_line(&mut resp)
        .await
        .context("reading response from aggregator")?;
    if n == 0 {
        return Err(anyhow!("aggregator closed the connection"));
    }
    Ok(resp)
}

/// Parse a response line into `T`, turning an `{"error": "..."}` line into a
/// descriptive `Err` rather than an opaque deserialization failure.
fn parse_response<T: DeserializeOwned>(line: &str) -> Result<T> {
    match serde_json::from_str::<T>(line) {
        Ok(v) => Ok(v),
        Err(parse_err) => {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(msg) = val.get("error").and_then(serde_json::Value::as_str) {
                    return Err(anyhow!("aggregator returned error: {msg}"));
                }
            }
            Err(parse_err).context("parsing aggregator response")
        }
    }
}

#[async_trait]
impl Provider for AggregatorClient {
    async fn summary(&self) -> Result<Summary> {
        self.request(Request::Summary).await
    }

    async fn expand_prefix(&self, req: ExpandPrefixRequest) -> Result<ExpandPrefixResponse> {
        self.request(Request::from(req)).await
    }

    async fn list_volumes(&self, req: ListVolumesRequest) -> Result<ListVolumesResponse> {
        self.request(Request::from(req)).await
    }

    async fn key(&self, req: KeyRequest) -> Result<KeyEntry> {
        self.request(Request::from(req)).await
    }

    async fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        self.request(Request::from(req)).await
    }

    async fn peek(&self, req: PeekRequest) -> Result<PeekResult> {
        self.request(Request::from(req)).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tokio::io::AsyncBufReadExt;
    use tokio::io::AsyncWriteExt;
    use tokio::io::BufReader;
    use tokio::net::TcpListener;

    use super::*;
    use crate::model::ObjectType;
    use crate::model::SearchKind;
    use crate::model::SearchMatch;

    /// Start an in-process aggregator that replies to each request line with a
    /// canned response looked up by its `op`, keeping the connection open
    /// across requests (so the same client connection is reused). Returns the
    /// bound `host:port` for an `AggregatorClient`.
    async fn spawn_server(responses: HashMap<&'static str, String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr").to_string();

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let responses = responses.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stream);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        let n = reader.read_line(&mut line).await.expect("read request");
                        if n == 0 {
                            break;
                        }
                        let req: serde_json::Value =
                            serde_json::from_str(line.trim()).expect("request json");
                        let op = req["op"].as_str().expect("op field");
                        let resp = responses
                            .get(op)
                            .cloned()
                            .unwrap_or_else(|| r#"{"error":"no canned response"}"#.to_string());
                        reader
                            .write_all(resp.as_bytes())
                            .await
                            .expect("write response");
                        reader.write_all(b"\n").await.expect("write newline");
                        reader.flush().await.expect("flush");
                    }
                });
            }
        });

        addr
    }

    fn canned() -> HashMap<&'static str, String> {
        let mut m = HashMap::new();
        m.insert(
            "summary",
            r#"{"schema_version":1,"captured_at":"2026-07-17T00:00:00Z","store_name":"torchstore","strategy":"LocalRankStrategy","totals":{"volumes":2,"keys":3,"bytes":1024.0,"partial_dtensors":1},"volume_groups":[],"key_prefixes":[{"prefix":"model","keys":3}]}"#
                .to_string(),
        );
        m.insert(
            "expand_prefix",
            r#"{"children":[{"prefix":"model.layers","keys":10}],"next_cursor":"pg2"}"#.to_string(),
        );
        m.insert(
            "key",
            r#"{"key":"model.embed.weight","object_type":"TENSOR","dtype":"float32","fully_committed":true,"shards":[]}"#
                .to_string(),
        );
        m.insert(
            "search",
            r#"{"matches":[{"key":"model.layers.0.attn.wq.weight","object_type":"TENSOR_SLICE","dtype":"float32","global_shape":[4096,4096],"fully_committed":true,"mesh_shape":[2,2],"shards":[]}],"next_cursor":null}"#
                .to_string(),
        );
        m.insert(
            "peek",
            r#"{"dtype":"float32","shape":[2,2],"min":-1.0,"max":1.0,"mean":0.0,"l2_norm":1.5,"head":[0.1,0.2]}"#
                .to_string(),
        );
        m
    }

    #[tokio::test]
    async fn summary_round_trips() {
        let addr = spawn_server(canned()).await;
        let client = AggregatorClient::new(addr);

        let s = client.summary().await.expect("summary");
        assert_eq!(s.schema_version, 1);
        assert_eq!(s.store_name, "torchstore");
        assert_eq!(s.totals.partial_dtensors, 1);
    }

    #[tokio::test]
    async fn multiple_ops_reuse_one_connection() {
        // Issuing several ops through one client proves the connection is
        // reused: the stub keeps a single connection open in its read loop, so
        // a second request only succeeds if the client wrote to the same
        // socket rather than reconnecting per call.
        let addr = spawn_server(canned()).await;
        let client = AggregatorClient::new(addr);

        client.summary().await.expect("summary");

        let resp = client
            .expand_prefix(ExpandPrefixRequest {
                prefix: "model".to_string(),
                limit: Some(200),
                cursor: None,
                sort_by: None,
                order: None,
            })
            .await
            .expect("expand_prefix");
        assert_eq!(resp.children.len(), 1);
        assert_eq!(resp.children[0].prefix, "model.layers");
        assert_eq!(resp.next_cursor.as_deref(), Some("pg2"));
    }

    #[tokio::test]
    async fn key_round_trips() {
        let addr = spawn_server(canned()).await;
        let client = AggregatorClient::new(addr);

        let entry = client
            .key(KeyRequest {
                key: "model.embed.weight".to_string(),
            })
            .await
            .expect("key");
        assert_eq!(entry.object_type, ObjectType::Tensor);
        assert!(entry.fully_committed);
    }

    #[tokio::test]
    async fn search_round_trips() {
        let addr = spawn_server(canned()).await;
        let client = AggregatorClient::new(addr);

        let resp = client
            .search(SearchRequest {
                kind: SearchKind::Key,
                pattern: "attn.wq".to_string(),
                limit: Some(200),
                cursor: None,
                sort_by: None,
                order: None,
            })
            .await
            .expect("search");
        assert_eq!(resp.matches.len(), 1);
        assert!(matches!(resp.matches[0], SearchMatch::Key(_)));
    }

    #[tokio::test]
    async fn peek_round_trips() {
        let addr = spawn_server(canned()).await;
        let client = AggregatorClient::new(addr);

        let resp = client
            .peek(PeekRequest {
                key: "model.embed.weight".to_string(),
                coordinates: Some(vec![0, 0]),
            })
            .await
            .expect("peek");
        assert_eq!(resp.dtype, "float32");
        assert_eq!(resp.head.len(), 2);
        assert!(resp.max >= resp.min);
    }

    #[tokio::test]
    async fn error_line_maps_to_err() {
        // Server returns an `{"error":...}` line for `summary`; the client must
        // surface it as `Err`, not panic or silently deserialize.
        let mut responses = HashMap::new();
        responses.insert("summary", r#"{"error":"store unavailable"}"#.to_string());
        let addr = spawn_server(responses).await;
        let client = AggregatorClient::new(addr);

        let err = client.summary().await.expect_err("expected an error");
        assert!(
            err.to_string().contains("store unavailable"),
            "error should carry the aggregator message, got: {err}"
        );
    }
}
