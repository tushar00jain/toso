//! `toso-tui` entrypoint. Builds a `Provider` (a `FileProvider` over fixtures
//! for now; the `AggregatorClient` is task #4) and runs the ratatui event loop.

mod data;
mod model;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use color_eyre::eyre::Result;
use color_eyre::eyre::bail;
use color_eyre::eyre::eyre;

use crate::data::AggregatorClient;
use crate::data::FileProvider;
use crate::data::Provider;

#[derive(Parser, Debug)]
#[command(name = "toso-tui", about = "Terminal UI for inspecting a TorchStore")]
struct Args {
    /// Connect to an aggregator at `host:port` (line-delimited JSON over TCP).
    #[arg(long)]
    aggregator: Option<String>,

    /// Serve from a directory of JSON fixtures instead of the network.
    #[arg(long)]
    fixtures: Option<PathBuf>,

    /// Seconds between automatic `summary` refreshes.
    #[arg(long, default_value_t = 5)]
    refresh: u64,

    /// Render the landing frame and one drill to stdout (via an in-memory
    /// backend) then exit, instead of starting the interactive TUI. Works with
    /// either provider and needs no TTY.
    #[arg(long)]
    headless: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    let provider: Arc<dyn Provider> = match (args.aggregator, args.fixtures) {
        (Some(_), Some(_)) | (None, None) => {
            bail!("pass exactly one of --aggregator or --fixtures");
        }
        (Some(addr), None) => Arc::new(AggregatorClient::new(addr)),
        (None, Some(dir)) => Arc::new(FileProvider::new(dir)),
    };

    let refresh = Duration::from_secs(args.refresh);
    if args.headless {
        ui::run_headless(provider, refresh)
            .await
            .map_err(|e| eyre!("headless render failed: {e:#}"))?;
        return Ok(());
    }

    ui::run(provider, refresh)
        .await
        .map_err(|e| eyre!("UI exited with error: {e:#}"))?;

    Ok(())
}
