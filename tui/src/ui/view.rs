//! The pure render: `view(&App, &mut Frame)`. Draws a full-width single view
//! (SPEC §6) — a breadcrumb header, the current drill-stack frame's body, and a
//! contextual footer. Lists are windowed via `LazyList` (SPEC §6.3).
//!
//! Everything visual routes through the small [theme][self] palette + helpers at
//! the top (rounded panels, status badges, gauges, themed selection/scrollbars)
//! so the whole app reads as one cohesive, colored surface rather than a set of
//! ad-hoc `Color::Red`s.

use ratatui::Frame;
use ratatui::layout::Alignment;
use ratatui::layout::Constraint;
use ratatui::layout::Flex;
use ratatui::layout::Layout;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::BorderType;
use ratatui::widgets::Cell;
use ratatui::widgets::Clear;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Row;
use ratatui::widgets::Scrollbar;
use ratatui::widgets::ScrollbarOrientation;
use ratatui::widgets::ScrollbarState;
use ratatui::widgets::Table;
use ratatui::widgets::Wrap;

use super::App;
use super::FrameKind;
use super::LandingRow;
use super::Mode;
use super::Scope;
use super::lazy_list::LazyList;
use super::human_bytes;
use super::middle_elide;
use super::thousands;
use crate::model::KeyEntry;
use crate::model::KeyPrefix;
use crate::model::ObjectType;
use crate::model::PeekResult;
use crate::model::SearchMatch;
use crate::model::Summary;
use crate::model::Volume;

// ---------------------------------------------------------------------------
// Theme — one cohesive palette applied everywhere
// ---------------------------------------------------------------------------

/// Primary brand accent (panel titles, active breadcrumb, key hints).
const ACCENT: Color = Color::Rgb(96, 165, 250);
/// Secondary accent (strategy, scope, object kinds).
const ACCENT_ALT: Color = Color::Rgb(45, 212, 191);
/// Healthy / committed.
const OK: Color = Color::Rgb(74, 194, 133);
/// Degraded / partial.
const WARN: Color = Color::Rgb(230, 179, 64);
/// Unreachable / errors.
const DANGER: Color = Color::Rgb(240, 105, 105);
/// De-emphasized text (labels, hints, separators).
const MUTED: Color = Color::Rgb(130, 140, 160);
/// Panel borders and faint separators.
const BORDER: Color = Color::Rgb(72, 82, 104);
/// Primary reading text (numbers, values).
const HEADING: Color = Color::Rgb(226, 232, 240);
/// Selected-row background tint.
const SEL_BG: Color = Color::Rgb(41, 54, 82);
/// Text drawn on top of an accent-filled pill.
const INK: Color = Color::Rgb(16, 20, 30);
/// The app background — a soft near-black (#181818), forced so the UI looks the
/// same regardless of the terminal's own theme.
const BG: Color = Color::Rgb(24, 24, 24);
/// DTensor-shard object kind.
const SLICE: Color = Color::Rgb(196, 141, 255);

/// A longer braille cycle than the SPEC sketch — animates smoothly while any
/// request is in flight (`app.spinner` only advances then).
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// A rounded, border-tinted panel with an accent title — the frame every view
/// draws inside.
fn panel(title: &str) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .style(Style::new().bg(BG))
        .title(Line::from(vec![
            Span::styled("▍", Style::new().fg(ACCENT)),
            Span::styled(
                format!(" {title} "),
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
        ]))
}

/// Subtle blue-tinted selection that keeps a row's health color legible (the
/// highlight sets only the background + bold, never the foreground).
fn selection_style() -> Style {
    Style::new().bg(SEL_BG).add_modifier(Modifier::BOLD)
}

/// A bold, muted column header shared by every table.
fn header_row<'a, const N: usize>(labels: [&'a str; N]) -> Row<'a> {
    Row::new(labels).style(Style::new().fg(MUTED).add_modifier(Modifier::BOLD))
}

/// Health states rendered as a colored dot + label so anomalies read at a glance.
enum Status {
    Ok,
    Degraded,
    Unreachable,
}

fn status_cell(s: Status) -> Cell<'static> {
    let (glyph, label, color) = match s {
        Status::Ok => ("●", "ok", OK),
        Status::Degraded => ("▲", "degraded", WARN),
        Status::Unreachable => ("✕", "unreachable", DANGER),
    };
    Cell::from(Line::from(vec![Span::styled(
        format!("{glyph} {label}"),
        Style::new().fg(color).add_modifier(Modifier::BOLD),
    )]))
}

pub(crate) fn view(app: &App, f: &mut Frame) {
    // Paint the whole surface black up front; every panel keeps this bg, so the
    // UI reads identically on any terminal theme.
    f.render_widget(Block::new().style(Style::new().bg(BG)), f.area());

    let chunks = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(1),
        Constraint::Length(4),
    ])
    .split(f.area());

    render_header(app, f, chunks[0]);
    render_body(app, f, chunks[1]);
    render_footer(app, f, chunks[2]);

    if matches!(app.mode, Mode::Help) {
        render_help(f);
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

fn render_header(app: &App, f: &mut Frame, area: Rect) {
    let (store, strategy, vols, keys) = match &app.summary {
        Some(s) => (
            s.store_name.clone(),
            s.strategy.clone(),
            thousands(s.totals.volumes),
            thousands(s.totals.keys),
        ),
        None => (
            "torchstore".to_string(),
            "…".to_string(),
            "…".to_string(),
            "…".to_string(),
        ),
    };

    let sep = || Span::styled("  ·  ", Style::new().fg(BORDER));
    let title = Line::from(vec![
        Span::styled(
            " toso-tui ",
            Style::new().fg(INK).bg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(store, Style::new().fg(HEADING).add_modifier(Modifier::BOLD)),
        sep(),
        Span::styled(strategy, Style::new().fg(ACCENT_ALT)),
        sep(),
        Span::styled(vols, Style::new().fg(HEADING)),
        Span::styled(" vols", Style::new().fg(MUTED)),
        sep(),
        Span::styled(keys, Style::new().fg(HEADING)),
        Span::styled(" keys", Style::new().fg(MUTED)),
        Span::raw(" "),
        liveness_badge(app),
        Span::raw(" "),
    ]);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .style(Style::new().bg(BG))
        .title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let scope = match &app.scope {
        Scope::All => "(all)".to_string(),
        Scope::Group(g) => g.clone(),
    };
    let sort = format!(
        "{:?}{}",
        app.sort,
        if matches!(app.order, crate::model::Order::Desc) {
            "↓"
        } else {
            "↑"
        }
    );

    let lines = vec![
        breadcrumb_line(app),
        Line::from(vec![
            Span::styled("scope ", Style::new().fg(MUTED)),
            Span::styled(scope, Style::new().fg(ACCENT_ALT).add_modifier(Modifier::BOLD)),
            Span::raw("    "),
            Span::styled("sort ", Style::new().fg(MUTED)),
            Span::styled(sort, Style::new().fg(HEADING).add_modifier(Modifier::BOLD)),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

/// The drill-stack as a breadcrumb: earlier levels muted, the current level in
/// bold accent, chevrons faint.
fn breadcrumb_line(app: &App) -> Line<'static> {
    let n = app.stack.len();
    let mut spans = Vec::with_capacity(n * 2);
    for (i, fr) in app.stack.iter().enumerate() {
        let last = i + 1 == n;
        let style = if last {
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(MUTED)
        };
        spans.push(Span::styled(fr.title.clone(), style));
        if !last {
            spans.push(Span::styled(" ▸ ", Style::new().fg(BORDER)));
        }
    }
    Line::from(spans)
}

/// A colored liveness pill: amber while loading, green when fresh, red when stale.
fn liveness_badge(app: &App) -> Span<'static> {
    let (text, color) = match app.summary_loaded_at {
        None => ("loading".to_string(), ACCENT),
        Some(t) => {
            let secs = t.elapsed().as_secs();
            if app.summary.is_some() && t.elapsed() > app.refresh_threshold() {
                (format!("stale {secs}s"), DANGER)
            } else {
                (format!("live {secs}s"), OK)
            }
        }
    };
    Span::styled(
        format!(" ● {text} "),
        Style::new().fg(color).add_modifier(Modifier::BOLD),
    )
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

fn render_footer(app: &App, f: &mut Frame, area: Rect) {
    let hints: &[(&str, &str)] = match app.stack.last().map(|fr| &fr.kind) {
        Some(FrameKind::Health { .. }) => &[
            ("↑↓", "move"),
            ("l/⏎", "drill"),
            ("/", "filter"),
            (":", "jump"),
            ("s", "sort"),
            ("r", "refresh"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Some(FrameKind::Keys { .. }) => &[
            ("↑↓", "move"),
            ("l/⏎", "drill"),
            ("h/esc", "back"),
            ("/", "filter"),
            (":", "jump"),
            ("s", "sort"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Some(FrameKind::Volumes { .. }) => &[
            ("↑↓", "move"),
            ("⏎", "more"),
            ("h/esc", "back"),
            ("s", "sort"),
            ("/", "filter"),
            (":", "jump"),
            ("r", "refresh"),
            ("q", "quit"),
        ],
        Some(FrameKind::Results { .. }) => &[
            ("↑↓", "move"),
            ("l/⏎", "open"),
            ("h/esc", "back"),
            ("s", "sort"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Some(FrameKind::Detail { .. }) => &[
            ("↑↓", "shards"),
            ("p", "peek"),
            ("h/esc", "back"),
            ("r", "refresh"),
            ("?", "help"),
            ("q", "quit"),
        ],
        None => &[("q", "quit")],
    };

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .style(Style::new().bg(BG));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let lines = vec![hint_line(hints), status_line(app)];
    f.render_widget(Paragraph::new(lines), inner);
}

/// Key hints with the key in accent-bold and its action muted.
fn hint_line(pairs: &[(&str, &str)]) -> Line<'static> {
    let mut spans = Vec::with_capacity(pairs.len() * 4);
    for (i, (k, d)) in pairs.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ·  ", Style::new().fg(BORDER)));
        }
        spans.push(Span::styled(
            k.to_string(),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(d.to_string(), Style::new().fg(MUTED)));
    }
    Line::from(spans)
}

/// The line editor (`/`, `:`) or a status message. Errors read red, transient
/// info reads accent, everything else muted.
fn status_line(app: &App) -> Line<'static> {
    match &app.mode {
        Mode::Filter { buffer } => Line::from(vec![
            Span::styled("/", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(buffer.clone(), Style::new().fg(HEADING)),
            Span::styled("▏", Style::new().fg(ACCENT)),
        ]),
        Mode::Command { buffer } => Line::from(vec![
            Span::styled(":", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(buffer.clone(), Style::new().fg(HEADING)),
            Span::styled("▏", Style::new().fg(ACCENT)),
        ]),
        Mode::Browse | Mode::Help => match &app.status {
            None => Line::from(""),
            Some(msg) => {
                let low = msg.to_ascii_lowercase();
                let is_err = ["failed", "error", "cannot", "could not", "unknown"]
                    .iter()
                    .any(|w| low.contains(w));
                let color = if is_err { DANGER } else { ACCENT_ALT };
                Line::from(Span::styled(msg.clone(), Style::new().fg(color)))
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Body dispatch
// ---------------------------------------------------------------------------

fn render_body(app: &App, f: &mut Frame, area: Rect) {
    match app.stack.last().map(|fr| &fr.kind) {
        Some(FrameKind::Health { .. }) => render_health(app, f, area),
        Some(FrameKind::Keys { .. }) => render_keys(app, f, area),
        Some(FrameKind::Volumes { .. }) => render_volumes(app, f, area),
        Some(FrameKind::Results { .. }) => render_results(app, f, area),
        Some(FrameKind::Detail { .. }) => render_detail(app, f, area),
        None => {}
    }
}

/// A centered spinner + message — the loading state a panel shows before its
/// first data arrives.
fn render_loading(app: &App, f: &mut Frame, area: Rect, title: &str, msg: &str) {
    let block = panel(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let ch = SPINNER[app.spinner % SPINNER.len()];
    let line = Line::from(vec![
        Span::styled(
            format!("{ch} "),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(msg.to_string(), Style::new().fg(MUTED)),
    ]);
    let rows = Layout::vertical([
        Constraint::Percentage(45),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .split(inner);
    f.render_widget(Paragraph::new(line).alignment(Alignment::Center), rows[1]);
}

// ---------------------------------------------------------------------------
// Health board (landing)
// ---------------------------------------------------------------------------

fn render_health(app: &App, f: &mut Frame, area: Rect) {
    let Some(summary) = &app.summary else {
        render_loading(app, f, area, "health", "loading summary…");
        return;
    };

    let board = health_board_lines(summary);
    let board_height = (board.len() as u16 + 2).min(area.height.saturating_sub(3));
    let chunks =
        Layout::vertical([Constraint::Length(board_height), Constraint::Min(1)]).split(area);

    f.render_widget(
        // `trim: false` preserves each line's leading padding — the gauge labels
        // are right-aligned (`  0%` / ` 50%` / `100%`) so the bars line up.
        Paragraph::new(board)
            .block(panel("health"))
            .wrap(Wrap { trim: false }),
        chunks[0],
    );

    render_landing_list(app, f, chunks[1]);
}

fn health_board_lines(summary: &Summary) -> Vec<Line<'static>> {
    let t = &summary.totals;
    let unreachable: u64 = summary
        .volume_groups
        .iter()
        .map(|g| g.volumes.saturating_sub(g.reachable))
        .sum();

    // Metric tiles: muted label, bright value.
    let tile = |label: &str, value: String| {
        vec![
            Span::styled(format!("{label} "), Style::new().fg(MUTED)),
            Span::styled(value, Style::new().fg(HEADING).add_modifier(Modifier::BOLD)),
        ]
    };
    let mut top = tile("volumes", thousands(t.volumes));
    top.push(Span::styled("     ", Style::new().fg(MUTED)));
    top.extend(tile("keys", thousands(t.keys)));
    top.push(Span::styled("     ", Style::new().fg(MUTED)));
    top.extend(tile("bytes", human_bytes(t.bytes)));

    let (pg, pc) = if t.partial_dtensors > 0 {
        ("⚠", WARN)
    } else {
        ("✓", OK)
    };
    let (ug, uc) = if unreachable > 0 {
        ("↓", DANGER)
    } else {
        ("✓", OK)
    };
    let alerts = Line::from(vec![
        Span::styled(
            format!("{pg} {} partial dtensors", t.partial_dtensors),
            Style::new().fg(pc).add_modifier(Modifier::BOLD),
        ),
        Span::styled("      ", Style::new().fg(MUTED)),
        Span::styled(
            format!("{ug} {unreachable} unreachable volumes"),
            Style::new().fg(uc).add_modifier(Modifier::BOLD),
        ),
    ]);

    let mut lines = vec![
        Line::from(top),
        alerts,
        Line::from(""),
        Line::from(Span::styled(
            "shard commit %",
            Style::new().fg(MUTED).add_modifier(Modifier::BOLD),
        )),
    ];

    let width = 30;
    let hist = &summary.histograms.shard_commit_pct;
    let max = hist.iter().map(|[_, c]| *c).max().unwrap_or(1).max(1);
    lines.extend(hist.iter().map(|[bucket, count]| {
        let frac = *count as f64 / max as f64;
        let filled = gauge_cells(frac, width, *count > 0);
        Line::from(vec![
            Span::styled(format!("{bucket:>4}% "), Style::new().fg(MUTED)),
            // Solid fill in the bucket color …
            Span::styled("█".repeat(filled), Style::new().fg(commit_color(*bucket))),
            // … over a dim, recessive track for the empty remainder.
            Span::styled("░".repeat(width - filled), Style::new().fg(BORDER)),
            Span::styled(format!(" {}", thousands(*count)), Style::new().fg(HEADING)),
        ])
    }));
    lines
}

/// Bucket → color: low commit is dangerous, mid is a warning, full is healthy.
fn commit_color(bucket: u64) -> Color {
    match bucket {
        b if b >= 100 => OK,
        b if b >= 50 => WARN,
        _ => DANGER,
    }
}

/// Filled cell count for a bar `width` cells wide. Whole cells only (no
/// fractional glyph — those leave a half-lit cell that reads as a gap), but any
/// non-empty bucket keeps at least one lit cell so it stays visible next to a
/// truly-empty one.
fn gauge_cells(frac: f64, width: usize, nonzero: bool) -> usize {
    let cells = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    let cells = cells.min(width);
    if nonzero {
        cells.max(1)
    } else {
        cells
    }
}

fn render_landing_list(app: &App, f: &mut Frame, area: Rect) {
    let FrameKind::Health { list } = &app.stack[0].kind else {
        return;
    };

    let block = panel("summary");
    let inner = block.inner(area);
    let rows_avail = inner.height as usize;
    let (offset, slice) = list.window(rows_avail);

    let header = header_row(["NAME", "KEYS", "DTENSORS", "PARTIAL", "BYTES", "STATUS"]);
    let mut rows: Vec<Row> = slice.iter().map(landing_row).collect();
    push_trailing(&mut rows, app, list, offset, rows_avail);

    let widths = [
        Constraint::Min(20),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(9),
        Constraint::Length(12),
        Constraint::Length(14),
    ];
    let total = list.total_for_scrollbar();
    let bottom = window_label(offset, slice.len(), list.len(), list.has_more());
    let table = Table::new(rows, widths)
        .header(header)
        .block(block.title_bottom(bottom.right_aligned()))
        .row_highlight_style(selection_style());

    let mut state = ratatui::widgets::TableState::default()
        .with_selected(Some(list.selected_index().saturating_sub(offset)));
    f.render_stateful_widget(table, area, &mut state);

    render_scrollbar(f, inner, total, offset);
}

fn landing_row(row: &LandingRow) -> Row<'static> {
    match row {
        LandingRow::Prefix(kp) => key_prefix_row("▸", kp),
        LandingRow::Group(g) => {
            let unreachable = g.volumes.saturating_sub(g.reachable);
            let (partial_cell, status) = if unreachable > 0 {
                (
                    Cell::from(Span::styled(
                        format!("{unreachable} ↓"),
                        Style::new().fg(DANGER).add_modifier(Modifier::BOLD),
                    )),
                    Status::Degraded,
                )
            } else {
                (Cell::from(dash()), Status::Ok)
            };
            Row::new(vec![
                Cell::from(Line::from(vec![
                    Span::styled("▤ ", Style::new().fg(ACCENT_ALT)),
                    Span::styled(g.group.clone(), Style::new().fg(HEADING)),
                ])),
                Cell::from(num(thousands(g.keys))),
                Cell::from(dash()),
                partial_cell,
                Cell::from(num(human_bytes(g.bytes))),
                status_cell(status),
            ])
        }
    }
}

// ---------------------------------------------------------------------------
// Keys view
// ---------------------------------------------------------------------------

fn render_keys(app: &App, f: &mut Frame, area: Rect) {
    let Some(FrameKind::Keys { prefix, list }) = app.stack.last().map(|fr| &fr.kind) else {
        return;
    };

    let block = panel(&format!("keys · {}", middle_elide(prefix, 60)));
    let inner = block.inner(area);
    let rows_avail = inner.height as usize;
    let (offset, slice) = list.window(rows_avail);

    let header = header_row(["NAME", "KEYS", "DTENSORS", "PARTIAL", "BYTES", "STATUS"]);
    let mut rows: Vec<Row> = slice.iter().map(|kp| key_prefix_row("▸", kp)).collect();
    if list.has_trailing() {
        push_trailing(&mut rows, app, list, offset, rows_avail);
    } else if list.is_loaded() && list.is_empty() {
        rows.push(empty_row("(empty)"));
    }

    let widths = [
        Constraint::Min(20),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(9),
        Constraint::Length(12),
        Constraint::Length(14),
    ];
    let total = list.total_for_scrollbar();
    let bottom = window_label(offset, slice.len(), list.len(), list.has_more());
    let table = Table::new(rows, widths)
        .header(header)
        .block(block.title_bottom(bottom.right_aligned()))
        .row_highlight_style(selection_style());

    let mut state = ratatui::widgets::TableState::default()
        .with_selected(Some(list.selected_index().saturating_sub(offset)));
    f.render_stateful_widget(table, area, &mut state);
    render_scrollbar(f, inner, total, offset);
}

fn key_prefix_row(marker: &str, kp: &KeyPrefix) -> Row<'static> {
    let partial = kp.partial.unwrap_or(0);
    let name = last_of(&kp.prefix);
    let (partial_cell, status) = if partial > 0 {
        (
            Cell::from(Span::styled(
                format!("{partial} ⚠"),
                Style::new().fg(WARN).add_modifier(Modifier::BOLD),
            )),
            Status::Degraded,
        )
    } else {
        (Cell::from(dash()), Status::Ok)
    };
    Row::new(vec![
        Cell::from(Line::from(vec![
            Span::styled(format!("{marker} "), Style::new().fg(ACCENT)),
            Span::styled(name, Style::new().fg(HEADING)),
        ])),
        Cell::from(num(thousands(kp.keys))),
        Cell::from(num(opt_num(kp.dtensors))),
        partial_cell,
        Cell::from(num(kp.bytes.map_or_else(|| "—".to_string(), human_bytes))),
        status_cell(status),
    ])
}

// ---------------------------------------------------------------------------
// Volumes (topology) view
// ---------------------------------------------------------------------------

fn render_volumes(app: &App, f: &mut Frame, area: Rect) {
    let Some(FrameKind::Volumes { group, list }) = app.stack.last().map(|fr| &fr.kind) else {
        return;
    };

    let block = panel(&format!("volumes · {group}"));
    let inner = block.inner(area);
    let rows_avail = inner.height as usize;
    let (offset, slice) = list.window(rows_avail);

    let header = header_row(["VOLUME", "HOST", "TRANSPORT", "KEYS", "BYTES", "STATUS"]);
    let mut rows: Vec<Row> = slice.iter().map(volume_row).collect();
    push_trailing(&mut rows, app, list, offset, rows_avail);

    let widths = [
        Constraint::Length(12),
        Constraint::Min(16),
        Constraint::Length(14),
        Constraint::Length(10),
        Constraint::Length(12),
        Constraint::Length(14),
    ];
    let total = list.total_for_scrollbar();
    let bottom = window_label(offset, slice.len(), list.len(), list.has_more());
    let table = Table::new(rows, widths)
        .header(header)
        .block(block.title_bottom(bottom.right_aligned()))
        .row_highlight_style(selection_style());

    let mut state = ratatui::widgets::TableState::default()
        .with_selected(Some(list.selected_index().saturating_sub(offset)));
    f.render_stateful_widget(table, area, &mut state);
    render_scrollbar(f, inner, total, offset);
}

fn volume_row(v: &Volume) -> Row<'static> {
    let (transport, tstyle) = match &v.transport {
        Some(t) => (t.clone(), Style::new().fg(ACCENT_ALT)),
        None => ("—".to_string(), Style::new().fg(MUTED)),
    };
    Row::new(vec![
        Cell::from(Span::styled(v.volume_id.clone(), Style::new().fg(HEADING))),
        Cell::from(Span::styled(v.hostname.clone(), Style::new().fg(HEADING))),
        Cell::from(Span::styled(transport, tstyle)),
        Cell::from(num(thousands(v.num_keys))),
        Cell::from(num(human_bytes(v.bytes))),
        status_cell(if v.reachable {
            Status::Ok
        } else {
            Status::Unreachable
        }),
    ])
}

// ---------------------------------------------------------------------------
// Results view (search: / filter and : jump)
// ---------------------------------------------------------------------------

fn render_results(app: &App, f: &mut Frame, area: Rect) {
    let Some(FrameKind::Results { list, .. }) = app.stack.last().map(|fr| &fr.kind) else {
        return;
    };

    let title = app.stack.last().map_or("results", |fr| fr.title.as_str());
    let block = panel(&format!("results · {title}"));
    let inner = block.inner(area);
    let rows_avail = inner.height as usize;
    let (offset, slice) = list.window(rows_avail);

    let header = header_row(["NAME", "TYPE", "SHAPE / HOST", "STATUS"]);
    let mut rows: Vec<Row> = slice.iter().map(result_row).collect();
    if list.has_trailing() {
        push_trailing(&mut rows, app, list, offset, rows_avail);
    } else if list.is_loaded() && list.is_empty() {
        rows.push(empty_row("(no matches)"));
    }

    let widths = [
        Constraint::Min(30),
        Constraint::Length(14),
        Constraint::Min(16),
        Constraint::Length(14),
    ];
    let total = list.total_for_scrollbar();
    let bottom = window_label(offset, slice.len(), list.len(), list.has_more());
    let table = Table::new(rows, widths)
        .header(header)
        .block(block.title_bottom(bottom.right_aligned()))
        .row_highlight_style(selection_style());

    let mut state = ratatui::widgets::TableState::default()
        .with_selected(Some(list.selected_index().saturating_sub(offset)));
    f.render_stateful_widget(table, area, &mut state);
    render_scrollbar(f, inner, total, offset);
}

fn result_row(m: &SearchMatch) -> Row<'static> {
    match m {
        SearchMatch::Key(k) => Row::new(vec![
            Cell::from(Span::styled(
                middle_elide(&k.key, 44),
                Style::new().fg(HEADING),
            )),
            Cell::from(object_type_span(k.object_type)),
            Cell::from(num(shape_label(k.global_shape.as_deref()))),
            status_cell(if k.fully_committed {
                Status::Ok
            } else {
                Status::Degraded
            }),
        ]),
        SearchMatch::Volume(v) => Row::new(vec![
            Cell::from(Span::styled(v.volume_id.clone(), Style::new().fg(HEADING))),
            Cell::from(Span::styled("VOLUME", Style::new().fg(MUTED))),
            Cell::from(Span::styled(v.hostname.clone(), Style::new().fg(HEADING))),
            status_cell(if v.reachable {
                Status::Ok
            } else {
                Status::Unreachable
            }),
        ]),
    }
}

// ---------------------------------------------------------------------------
// Detail view
// ---------------------------------------------------------------------------

fn render_detail(app: &App, f: &mut Frame, area: Rect) {
    let Some(FrameKind::Detail {
        key,
        entry,
        error,
        peek,
        shard_cursor,
    }) = app.stack.last().map(|fr| &fr.kind)
    else {
        return;
    };

    let Some(entry) = entry else {
        match error {
            Some(e) => {
                let block = panel("detail");
                let inner = block.inner(area);
                f.render_widget(block, area);
                let body = Line::from(vec![
                    Span::styled("✕ ", Style::new().fg(DANGER).add_modifier(Modifier::BOLD)),
                    Span::styled(
                        format!("could not load {key}: {e}"),
                        Style::new().fg(DANGER),
                    ),
                ]);
                f.render_widget(Paragraph::new(body).wrap(Wrap { trim: true }), inner);
            }
            None => render_loading(app, f, area, "detail", &format!("loading {key}…")),
        }
        return;
    };

    let mut info = detail_info_lines(entry);
    if let Some(peek) = peek {
        info.extend(peek_lines(peek));
    }
    let info_height = (info.len() as u16 + 2).min(area.height.saturating_sub(3));
    let chunks =
        Layout::vertical([Constraint::Length(info_height), Constraint::Min(1)]).split(area);

    f.render_widget(
        Paragraph::new(info)
            .block(panel("detail"))
            .wrap(Wrap { trim: true }),
        chunks[0],
    );

    render_shard_table(entry, *shard_cursor, f, chunks[1]);
}

fn detail_info_lines(entry: &KeyEntry) -> Vec<Line<'static>> {
    let (committed, cstyle) = if entry.fully_committed {
        ("✓ fully committed", Style::new().fg(OK).add_modifier(Modifier::BOLD))
    } else {
        ("⚠ partial", Style::new().fg(WARN).add_modifier(Modifier::BOLD))
    };
    let label = |s: &str| Span::styled(format!("{s} "), Style::new().fg(MUTED));
    let type_line = Line::from(vec![
        label("type"),
        object_type_span(entry.object_type),
        Span::raw("    "),
        label("dtype"),
        Span::styled(
            entry.dtype.clone().unwrap_or_else(|| "—".to_string()),
            Style::new().fg(HEADING),
        ),
        Span::raw("    "),
        label("committed"),
        Span::styled(committed, cstyle),
    ]);
    vec![
        // The full key is shown untruncated here (SPEC §7).
        Line::from(Span::styled(
            entry.key.clone(),
            Style::new().fg(HEADING).add_modifier(Modifier::BOLD),
        )),
        type_line,
        Line::from(vec![
            label("global_shape"),
            Span::styled(shape_label(entry.global_shape.as_deref()), Style::new().fg(HEADING)),
            Span::raw("    "),
            label("mesh_shape"),
            Span::styled(shape_label(entry.mesh_shape.as_deref()), Style::new().fg(HEADING)),
        ]),
    ]
}

/// Render `peek` tensor stats with a size-guardrail note (SPEC §5.3, §9).
fn peek_lines(p: &PeekResult) -> Vec<Line<'static>> {
    let head = p
        .head
        .iter()
        .map(|x| format!("{x:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    let label = |s: &str| Span::styled(format!("{s} "), Style::new().fg(MUTED));
    let val = |s: String| Span::styled(s, Style::new().fg(HEADING));
    vec![
        Line::from(""),
        Line::from(Span::styled(
            "peek — stats computed near the data; head is first-N only, never the full tensor",
            Style::new().fg(MUTED).add_modifier(Modifier::ITALIC),
        )),
        Line::from(vec![
            Span::styled(" ▸ peek", Style::new().fg(SLICE).add_modifier(Modifier::BOLD)),
            Span::raw("   "),
            label("dtype"),
            val(p.dtype.clone()),
            Span::raw("    "),
            label("shape"),
            val(nums(&p.shape)),
        ]),
        Line::from(vec![
            label("min"),
            val(format!("{:.4}", p.min)),
            Span::raw("    "),
            label("max"),
            val(format!("{:.4}", p.max)),
            Span::raw("    "),
            label("mean"),
            val(format!("{:.4}", p.mean)),
            Span::raw("    "),
            label("l2_norm"),
            val(format!("{:.4}", p.l2_norm)),
        ]),
        Line::from(vec![label("head"), val(format!("[{head}]"))]),
    ]
}

fn render_shard_table(entry: &KeyEntry, cursor: usize, f: &mut Frame, area: Rect) {
    let block = panel(&format!("shards ({})", entry.shards.len()));

    // Objects (and plain tensors without recorded placement) have no shards —
    // show why rather than an empty table.
    if entry.shards.is_empty() {
        let note = match entry.object_type {
            ObjectType::Object => "(object — stored whole; no tensor shape or shards)",
            _ => "(no shard placement recorded in this snapshot)",
        };
        f.render_widget(
            Paragraph::new(Span::styled(note, Style::new().fg(MUTED))).block(block),
            area,
        );
        return;
    }

    let inner = block.inner(area);
    let height = inner.height as usize;
    let offset = if cursor < height {
        0
    } else {
        cursor + 1 - height
    };
    let end = (offset + height).min(entry.shards.len());
    let slice = entry.shards.get(offset..end).unwrap_or(&[]);

    let header = header_row(["VOLUME", "COORDINATES", "OFFSETS", "LOCAL_SHAPE"]);
    let rows: Vec<Row> = slice
        .iter()
        .map(|s| {
            Row::new(vec![
                Cell::from(Span::styled(s.volume_id.clone(), Style::new().fg(HEADING))),
                Cell::from(num(nums(&s.coordinates))),
                Cell::from(num(nums(&s.offsets))),
                Cell::from(num(nums(&s.local_shape))),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(14),
        Constraint::Min(14),
        Constraint::Min(14),
        Constraint::Min(14),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .row_highlight_style(selection_style());
    let mut state =
        ratatui::widgets::TableState::default().with_selected(Some(cursor.saturating_sub(offset)));
    f.render_stateful_widget(table, area, &mut state);
    render_scrollbar(f, inner, entry.shards.len(), offset);
}

// ---------------------------------------------------------------------------
// Shared bits
// ---------------------------------------------------------------------------

fn spinner_row(app: &App) -> Row<'static> {
    let ch = SPINNER[app.spinner % SPINNER.len()];
    Row::new(vec![Cell::from(Line::from(vec![Span::styled(
        format!("{ch} loading…"),
        Style::new().fg(ACCENT),
    )]))])
}

/// The row shown below a paged list: a spinner while a page is in flight, or a
/// selectable "load more" affordance the user presses Enter on to fetch the
/// next page (paging is manual — see `LazyList`).
fn trailing_row(app: &App, pending: bool) -> Row<'static> {
    if pending {
        return spinner_row(app);
    }
    Row::new(vec![Cell::from(Line::from(vec![
        Span::styled("↵ ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(
            "load more",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  press enter", Style::new().fg(MUTED)),
    ]))])
}

/// Append the trailing row when the window has scrolled to the end of the
/// loaded items and the list has a spinner/"load more" row to show.
fn push_trailing<T>(
    rows: &mut Vec<Row<'static>>,
    app: &App,
    list: &LazyList<T>,
    offset: usize,
    rows_avail: usize,
) {
    if list.has_trailing() && offset + rows_avail > list.len() {
        rows.push(trailing_row(app, list.is_pending()));
    }
}

fn empty_row(text: &str) -> Row<'static> {
    Row::new(vec![Cell::from(Span::styled(
        text.to_string(),
        Style::new().fg(MUTED).add_modifier(Modifier::ITALIC),
    ))])
}

/// A right-value cell in reading (not muted) color.
fn num(s: String) -> Span<'static> {
    Span::styled(s, Style::new().fg(HEADING))
}

fn dash() -> Span<'static> {
    Span::styled("—", Style::new().fg(MUTED))
}

fn render_scrollbar(f: &mut Frame, area: Rect, total: usize, offset: usize) {
    if total <= area.height as usize {
        return;
    }
    let mut state = ScrollbarState::new(total).position(offset);
    let bar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .thumb_style(Style::new().fg(ACCENT))
        .track_style(Style::new().fg(BORDER));
    f.render_stateful_widget(bar, area, &mut state);
}

fn window_label(offset: usize, shown: usize, loaded: usize, more: bool) -> Line<'static> {
    let end = offset + shown;
    let plus = if more { "+" } else { "" };
    Line::from(Span::styled(
        format!(" {}-{} of {loaded}{plus} ", offset + 1, end.max(offset)),
        Style::new().fg(MUTED),
    ))
}

fn object_type_label(t: ObjectType) -> &'static str {
    match t {
        ObjectType::Object => "OBJECT",
        ObjectType::Tensor => "TENSOR",
        ObjectType::TensorSlice => "TENSOR_SLICE",
    }
}

/// Color by object type (SPEC §7) so a key's kind reads at a glance.
fn object_type_color(t: ObjectType) -> Color {
    match t {
        ObjectType::Object => ACCENT_ALT,
        ObjectType::Tensor => OK,
        ObjectType::TensorSlice => SLICE,
    }
}

fn object_type_span(t: ObjectType) -> Span<'static> {
    Span::styled(
        object_type_label(t),
        Style::new().fg(object_type_color(t)).add_modifier(Modifier::BOLD),
    )
}

fn shape_label(shape: Option<&[u64]>) -> String {
    shape.map_or_else(|| "—".to_string(), nums)
}

fn nums(v: &[u64]) -> String {
    let inner = v
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn opt_num(n: Option<u64>) -> String {
    n.map_or_else(|| "—".to_string(), thousands)
}

fn last_of(prefix: &str) -> String {
    prefix.rsplit('.').next().unwrap_or(prefix).to_string()
}

// ---------------------------------------------------------------------------
// Help overlay
// ---------------------------------------------------------------------------

fn render_help(f: &mut Frame) {
    let area = centered_rect(f.area(), 62, 60);

    let entries = [
        ("↑↓ / j k", "move the cursor"),
        ("l / ⏎", "drill into the row under the cursor"),
        ("h / esc", "pop back up the drill-stack"),
        ("g / G", "jump to top / bottom"),
        ("PgUp / PgDn", "page up / down"),
        ("/", "filter — search keys by substring"),
        (
            ":",
            "command / jump (:key :group :partial :unreachable :peek)",
        ),
        (
            "s",
            "cycle sort (partial · bytes · reachable · keys · name)",
        ),
        ("p", "peek — tensor stats for the selected shard"),
        ("r", "refresh the summary"),
        ("?", "toggle this help"),
        ("q", "quit"),
    ];
    let body: Vec<Line> = entries
        .iter()
        .map(|(k, d)| {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{k:<12}"),
                    Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(d.to_string(), Style::new().fg(HEADING)),
            ])
        })
        .collect();

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(ACCENT))
        .style(Style::new().bg(BG))
        .title(Line::from(Span::styled(
            " help — any key to dismiss ",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(area);
    f.render_widget(Clear, area);
    f.render_widget(block, area);
    f.render_widget(Paragraph::new(body), inner);
}

/// A `pct_x`×`pct_y` percentage rectangle centered in `area`.
fn centered_rect(area: Rect, pct_x: u16, pct_y: u16) -> Rect {
    let [h] = Layout::horizontal([Constraint::Percentage(pct_x)])
        .flex(Flex::Center)
        .areas(area);
    let [v] = Layout::vertical([Constraint::Percentage(pct_y)])
        .flex(Flex::Center)
        .areas(h);
    v
}
