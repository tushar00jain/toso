//! `agg_stub` — a throwaway aggregator that replays the `fixtures/` directory
//! over the §5 line-delimited-JSON wire protocol, so `AggregatorClient` (and
//! the TUI's `--aggregator` path) can be exercised end-to-end with no Python,
//! no Monarch, and no real store.
//!
//! It deliberately knows nothing about the `model` types — it maps each request
//! `op` to a fixture file using the SAME scheme as `FileProvider`, reads the
//! file, and emits it as one compact JSON line. Missing fixtures become a JSON
//! error line rather than a crash, so one bad request never kills the server.
//!
//! This is a dev tool, so `eprintln!` logging is acceptable here (the library
//! `aggregator.rs` stays silent).

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use clap::Parser;
use serde_json::Value;
use serde_json::json;
use tokio::fs;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::net::TcpListener;
use tokio::net::TcpStream;

#[derive(Parser, Debug)]
#[command(
    name = "agg_stub",
    about = "Replay toso-tui fixtures over the aggregator wire protocol"
)]
struct Args {
    /// Directory of JSON fixtures to serve (defaults to the crate's `fixtures/`).
    #[arg(long)]
    fixtures: Option<PathBuf>,

    /// TCP port to listen on (`0` picks a free ephemeral port).
    #[arg(long, default_value_t = 0)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let fixtures = args
        .fixtures
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"));

    let listener = TcpListener::bind(("127.0.0.1", args.port))
        .await
        .with_context(|| format!("binding 127.0.0.1:{}", args.port))?;
    let bound = listener.local_addr().context("resolving bound address")?;
    eprintln!(
        "agg_stub listening on {bound} (fixtures: {})",
        fixtures.display()
    );

    loop {
        let (stream, peer) = listener.accept().await.context("accepting connection")?;
        let dir = fixtures.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_conn(stream, &dir).await {
                eprintln!("connection {peer} ended: {e:#}");
            }
        });
    }
}

/// Serve one connection: read request lines until EOF, reply one line each.
async fn handle_conn(stream: TcpStream, fixtures: &Path) -> Result<()> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .await
            .context("reading request line")?;
        if n == 0 {
            return Ok(());
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match build_response(trimmed, fixtures).await {
            Ok(value) => value,
            Err(e) => {
                eprintln!("request {trimmed:?} -> error: {e:#}");
                json!({ "error": "not found" })
            }
        };

        // `Value::to_string` is compact, so the response never contains an
        // embedded newline that would corrupt the line framing.
        reader
            .write_all(response.to_string().as_bytes())
            .await
            .context("writing response")?;
        reader.write_all(b"\n").await.context("writing newline")?;
        reader.flush().await.context("flushing response")?;
    }
}

/// Map a request line to its fixture value, mirroring `FileProvider`'s layout.
async fn build_response(req_line: &str, fixtures: &Path) -> Result<Value> {
    let req: Value = serde_json::from_str(req_line).context("parsing request json")?;
    let op = req
        .get("op")
        .and_then(Value::as_str)
        .context("request missing string `op` field")?;

    match op {
        "summary" => read_fixture(&fixtures.join("summary.json")).await,
        "expand_prefix" => {
            let prefix = str_field(&req, "prefix")?;
            read_fixture(&fixtures.join(format!("expand_prefix/{prefix}.json"))).await
        }
        "list_volumes" => {
            let group = str_field(&req, "group")?;
            read_fixture(&fixtures.join(format!("list_volumes/{group}.json"))).await
        }
        "key" => {
            let key = str_field(&req, "key")?;
            read_fixture(&fixtures.join(format!("key/{key}.json"))).await
        }
        "search" => {
            // Empty pattern (`:partial` / `:unreachable` jumps) is match-all:
            // serve `search/all.json` if present, else empty matches — never a
            // 404 for a legitimately empty query.
            let pattern = req.get("pattern").and_then(Value::as_str).unwrap_or("");
            if pattern.is_empty() {
                let all = fixtures.join("search/all.json");
                if fs::try_exists(&all).await.unwrap_or(false) {
                    return read_fixture(&all).await;
                }
                return Ok(json!({ "matches": [], "next_cursor": null }));
            }
            read_fixture(&fixtures.join(format!("search/{pattern}.json"))).await
        }
        "peek" => {
            let key = str_field(&req, "key")?;
            let keyed = fixtures.join(format!("peek/{key}.json"));
            if fs::try_exists(&keyed).await.unwrap_or(false) {
                return read_fixture(&keyed).await;
            }
            read_fixture(&fixtures.join("peek.json")).await
        }
        other => Err(anyhow!("unknown op: {other}")),
    }
}

fn str_field<'a>(req: &'a Value, field: &str) -> Result<&'a str> {
    req.get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("request missing string `{field}` field"))
}

/// Read a fixture file and re-parse it to a `Value` (so the reply is one
/// compact line regardless of how the fixture is formatted on disk).
async fn read_fixture(path: &Path) -> Result<Value> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("reading fixture {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing fixture {}", path.display()))
}
