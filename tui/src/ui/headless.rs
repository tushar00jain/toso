//! A non-interactive render path (verification for SPEC §8 milestone 2): build
//! the provider, fetch the `summary`, drill one level into the key trie, and
//! render each resulting frame to a `ratatui` `TestBackend`, printing the buffer
//! as plain text. This exercises the exact pure [`view`] the interactive loop
//! uses, but with no TTY — so rendering over real snapshot data (or fixtures) is
//! observable and assertable.

use std::io::Write;
use std::io::{self};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use tokio::sync::mpsc;

use super::App;
use super::Cmd;
use super::update;
use crate::data::Provider;
use crate::ui::view::view;

/// Size the headless frames render at. Wide enough that keys, the strategy, and
/// volume rows are not middle-elided or clipped out of the frame.
const WIDTH: u16 = 120;
const HEIGHT: u16 = 40;

/// Render the walkthrough and write it to stdout.
pub(crate) async fn run(provider: Arc<dyn Provider>, refresh: Duration) -> Result<()> {
    let out = walkthrough(provider, refresh).await?;
    let mut stdout = io::stdout();
    stdout
        .write_all(out.as_bytes())
        .context("writing headless render to stdout")?;
    stdout.flush().context("flushing stdout")?;
    Ok(())
}

/// Drive the app through `summary` + one drill and return the rendered frames as
/// text. Shared by [`run`] and the render regression test so the committed test
/// covers the same path the binary runs.
pub(crate) async fn walkthrough(provider: Arc<dyn Provider>, refresh: Duration) -> Result<String> {
    // The reducer needs a sender for `Frame`/`App` construction, but the
    // headless path runs commands inline (via `App::run_cmd`) rather than
    // through this channel, so the receiver is intentionally dropped.
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut app = App::new(provider, tx, refresh);
    app.body_rows = HEIGHT as usize;

    let mut out = String::new();

    // 1. Landing / health board, populated from `summary`.
    let msg = app.run_cmd(Cmd::FetchSummary).await;
    update(&mut app, msg);
    out.push_str("=== frame 1: health board (summary) ===\n");
    out.push_str(&render(&app));

    // 2. Drill one real level into the key trie via `expand_prefix`.
    let cmds = app.drill_best_landing_prefix();
    for cmd in cmds {
        let msg = app.run_cmd(cmd).await;
        update(&mut app, msg);
    }
    out.push_str(&format!("\n=== frame 2: {} ===\n", app.breadcrumb()));
    out.push_str(&render(&app));

    Ok(out)
}

/// Render the current app state into an in-memory backend and return the buffer
/// as text. Rendering to a `TestBackend` cannot perform IO, so the draw is
/// infallible here.
fn render(app: &App) -> String {
    let backend = TestBackend::new(WIDTH, HEIGHT);
    let mut terminal = Terminal::new(backend).expect("in-memory terminal never fails to build");
    terminal
        .draw(|f| view(app, f))
        .expect("rendering to an in-memory backend never performs IO");
    buffer_to_string(terminal.backend().buffer())
}

/// Flatten a `ratatui` cell buffer into newline-separated rows, trimming the
/// trailing padding each row carries.
fn buffer_to_string(buf: &Buffer) -> String {
    let area = buf.area();
    let mut out = String::new();
    for y in 0..area.height {
        let mut line = String::new();
        for x in 0..area.width {
            line.push_str(buf[(x, y)].symbol());
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::data::FileProvider;

    async fn fixtures_walkthrough() -> String {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let provider: Arc<dyn Provider> = Arc::new(FileProvider::new(dir));
        walkthrough(provider, Duration::from_secs(5))
            .await
            .expect("headless walkthrough over fixtures")
    }

    #[tokio::test]
    async fn headless_renders_summary_and_drill_from_fixtures() {
        let out = fixtures_walkthrough().await;

        // Header, sourced from summary.json.
        assert!(out.contains("torchstore"), "store name in header:\n{out}");
        assert!(
            out.contains("LocalRankStrategy"),
            "strategy in header:\n{out}"
        );

        // Landing health board renders the top-level key-trie prefixes and a
        // volume group.
        assert!(
            out.contains("model"),
            "top-level `model` prefix on the board"
        );
        assert!(out.contains("metadata"), "top-level `metadata` prefix");
        assert!(out.contains("rack:A12"), "a volume group row from summary");

        // Drilling the branchiest prefix expands one trie level deeper via
        // `expand_prefix`, and the breadcrumb tracks it.
        assert!(
            out.contains("health ▸ model"),
            "breadcrumb reflects the drill:\n{out}"
        );
        assert!(
            out.contains("layers"),
            "expand_prefix child (model.layers) rendered:\n{out}"
        );
    }

    #[tokio::test]
    async fn headless_output_has_two_frames() {
        let out = fixtures_walkthrough().await;
        assert!(out.contains("=== frame 1"), "first frame labelled");
        assert!(out.contains("=== frame 2"), "second frame labelled");
    }
}
