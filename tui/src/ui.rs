//! The UI layer: an Elm-style Model-Update-View over `ratatui` + `crossterm`
//! (SPEC §6, §7).
//!
//! - `App` is the whole model: a nav stack of `Frame`s, the landing `Summary`,
//!   and cursor/scope/status bookkeeping.
//! - `Msg` is every input, data-arrival, and tick event.
//! - [`update`] is a pure reducer (no IO) mapping `(App, Msg) -> Vec<Cmd>`; the
//!   returned [`Cmd`]s are the side effects the event loop performs.
//! - `view` (in [`view`]) is a pure render of `&App` into a `Frame`.
//!
//! The render path never blocks on a provider: [`run`] spawns each `Cmd` on a
//! tokio task that sends its result back as a `Msg` over an `mpsc` channel, and
//! the loop redraws only on input, data, or the refresh tick.

mod headless;
mod lazy_list;
mod view;

use std::io::Stdout;
use std::io::{self};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use crossterm::cursor;
use crossterm::event::Event;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyEventKind;
use crossterm::event::KeyModifiers;
use crossterm::event::{self};
use crossterm::execute;
use crossterm::terminal::EnterAlternateScreen;
use crossterm::terminal::LeaveAlternateScreen;
use crossterm::terminal::disable_raw_mode;
use crossterm::terminal::enable_raw_mode;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

use crate::data::Provider;
use crate::model::ExpandPrefixRequest;
use crate::model::ExpandPrefixResponse;
use crate::model::KeyEntry;
use crate::model::KeyPrefix;
use crate::model::KeyRequest;
use crate::model::ListVolumesRequest;
use crate::model::ListVolumesResponse;
use crate::model::ObjectType;
use crate::model::Order;
use crate::model::PeekRequest;
use crate::model::PeekResult;
use crate::model::SearchKind;
use crate::model::SearchMatch;
use crate::model::SearchRequest;
use crate::model::SearchResponse;
use crate::model::SortKey;
use crate::model::Summary;
use crate::model::VolumeGroup;
pub(crate) use crate::ui::headless::run as run_headless;
use crate::ui::lazy_list::LazyList;
use crate::ui::view::view;

/// Page size for every drill-down request.
const PAGE: u64 = 200;

type Tui = Terminal<CrosstermBackend<Stdout>>;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A row on the health-board landing: either a key-trie prefix (drill into Keys)
/// or a volume group (drill into Topology).
pub(crate) enum LandingRow {
    Prefix(KeyPrefix),
    Group(VolumeGroup),
}

/// One level of the drill-stack. Each frame owns the list it drilled into so a
/// late-arriving page can be routed back to it by `id` even after the user has
/// moved on.
pub(crate) struct Frame {
    pub(crate) id: u64,
    pub(crate) title: String,
    pub(crate) kind: FrameKind,
}

pub(crate) enum FrameKind {
    /// Landing: health gauges/histogram plus a drillable list of prefixes+groups.
    Health { list: LazyList<LandingRow> },
    /// A key-trie level fetched via `expand_prefix`.
    Keys {
        prefix: String,
        list: LazyList<KeyPrefix>,
    },
    /// A volume group's volumes fetched via `list_volumes`.
    Volumes {
        group: String,
        list: LazyList<crate::model::Volume>,
    },
    /// A single key's detail (`KeyEntry`) with its shard table.
    Detail {
        key: String,
        entry: Option<KeyEntry>,
        /// Set when the key fetch fails, so the view shows the error instead of
        /// sitting on "loading…" forever.
        error: Option<String>,
        /// Tensor stats from a `peek` (SPEC §5.3); `None` until requested/arrived.
        peek: Option<PeekResult>,
        shard_cursor: usize,
    },
    /// A `search` result set powering `/` filter and `:` jump (SPEC §6.2/§6.3).
    Results {
        kind: SearchKind,
        pattern: String,
        list: LazyList<SearchMatch>,
    },
}

/// The active input mode. Keystrokes route to the buffer while a line editor is
/// open (`/` filter, `:` command); `Help` shows the overlay and dismisses on any
/// key (SPEC §6 "Modes").
pub(crate) enum Mode {
    Browse,
    Filter { buffer: String },
    Command { buffer: String },
    Help,
}

/// The active scope (a k9s-style namespace), set by drilling into a group.
pub(crate) enum Scope {
    All,
    Group(String),
}

pub(crate) struct App {
    provider: Arc<dyn Provider>,
    tx: mpsc::UnboundedSender<Msg>,
    refresh: Duration,

    pub(crate) should_quit: bool,
    pub(crate) summary: Option<Summary>,
    pub(crate) summary_loaded_at: Option<Instant>,
    pub(crate) stack: Vec<Frame>,
    next_id: u64,
    pub(crate) status: Option<String>,
    pub(crate) inflight: usize,
    pub(crate) spinner: usize,
    pub(crate) scope: Scope,
    pub(crate) sort: SortKey,
    pub(crate) order: Order,
    pub(crate) mode: Mode,
    /// Rows available to the body view; the reducer uses it as the PgUp/PgDn
    /// jump distance. The event loop keeps it in sync with the terminal.
    pub(crate) body_rows: usize,
}

impl App {
    pub(crate) fn new(
        provider: Arc<dyn Provider>,
        tx: mpsc::UnboundedSender<Msg>,
        refresh: Duration,
    ) -> Self {
        let mut list = LazyList::new();
        list.begin_page();
        let root = Frame {
            id: 0,
            title: "health".to_string(),
            kind: FrameKind::Health { list },
        };
        Self {
            provider,
            tx,
            refresh,
            should_quit: false,
            summary: None,
            summary_loaded_at: None,
            stack: vec![root],
            next_id: 1,
            status: None,
            inflight: 0,
            spinner: 0,
            scope: Scope::All,
            sort: SortKey::Partial,
            order: Order::Desc,
            mode: Mode::Browse,
            body_rows: 20,
        }
    }

    fn new_frame_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // -- input handling -----------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) -> Vec<Cmd> {
        // While a line editor or the help overlay is open, keystrokes belong to
        // the mode, not the browse keymap.
        match self.mode {
            Mode::Filter { .. } | Mode::Command { .. } => return self.on_input_key(key),
            Mode::Help => {
                self.mode = Mode::Browse;
                return vec![];
            }
            Mode::Browse => {}
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('q') => {
                self.should_quit = true;
                vec![]
            }
            KeyCode::Char('c') if ctrl => {
                self.should_quit = true;
                vec![]
            }
            KeyCode::Char('r') => {
                self.status = Some("refreshing".to_string());
                vec![Cmd::FetchSummary]
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.select_up();
                vec![]
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.select_down();
                vec![]
            }
            KeyCode::Char('g') => {
                self.select_top();
                vec![]
            }
            KeyCode::Char('G') => {
                self.select_bottom();
                vec![]
            }
            KeyCode::PageDown => {
                self.page_down();
                vec![]
            }
            KeyCode::PageUp => {
                self.page_up();
                vec![]
            }
            KeyCode::Enter | KeyCode::Char('l') | KeyCode::Right => self.drill(),
            KeyCode::Esc | KeyCode::Char('h') | KeyCode::Left => {
                self.pop();
                vec![]
            }
            KeyCode::Char('/') => {
                self.mode = Mode::Filter {
                    buffer: String::new(),
                };
                self.status = None;
                vec![]
            }
            KeyCode::Char(':') => {
                self.mode = Mode::Command {
                    buffer: String::new(),
                };
                self.status = None;
                vec![]
            }
            KeyCode::Char('s') => self.cycle_sort(),
            KeyCode::Char('p') => self.peek(),
            KeyCode::Char('?') => {
                self.mode = Mode::Help;
                vec![]
            }
            _ => vec![],
        }
    }

    // -- input modes (filter / command / help) ------------------------------

    /// Edit the active line buffer; `Enter` submits, `Esc` cancels.
    fn on_input_key(&mut self, key: KeyEvent) -> Vec<Cmd> {
        match key.code {
            KeyCode::Esc => {
                self.mode = Mode::Browse;
                vec![]
            }
            KeyCode::Enter => self.submit_input(),
            KeyCode::Backspace => {
                if let Mode::Filter { buffer } | Mode::Command { buffer } = &mut self.mode {
                    buffer.pop();
                }
                vec![]
            }
            KeyCode::Char(c) => {
                if let Mode::Filter { buffer } | Mode::Command { buffer } = &mut self.mode {
                    buffer.push(c);
                }
                vec![]
            }
            _ => vec![],
        }
    }

    fn submit_input(&mut self) -> Vec<Cmd> {
        match std::mem::replace(&mut self.mode, Mode::Browse) {
            Mode::Filter { buffer } => self.start_filter(buffer),
            Mode::Command { buffer } => self.run_command(&buffer),
            _ => vec![],
        }
    }

    /// A `/` filter compiles to a `search` over keys (SPEC §6.3).
    fn start_filter(&mut self, pattern: String) -> Vec<Cmd> {
        let pattern = pattern.trim().to_string();
        if pattern.is_empty() {
            return vec![];
        }
        let title = format!("filter: {pattern}");
        self.open_search(SearchKind::Key, pattern, title)
    }

    /// Parse a `:` command into exactly one bounded op (SPEC §6.2).
    fn run_command(&mut self, input: &str) -> Vec<Cmd> {
        let input = input.trim();
        if input.is_empty() {
            return vec![];
        }
        let (cmd, arg) = match input.split_once(char::is_whitespace) {
            Some((c, a)) => (c, a.trim()),
            None => (input, ""),
        };
        match cmd {
            "key" if !arg.is_empty() => self.open_key_detail(arg.to_string()),
            "group" if !arg.is_empty() => self.open_group(arg.to_string()),
            "peek" if !arg.is_empty() => self.open_peek(arg.to_string()),
            "partial" => {
                self.sort = SortKey::Partial;
                self.order = Order::Desc;
                self.open_search(SearchKind::Key, String::new(), ":partial".to_string())
            }
            "unreachable" => {
                self.sort = SortKey::Reachable;
                self.order = Order::Asc;
                self.open_search(
                    SearchKind::Volume,
                    String::new(),
                    ":unreachable".to_string(),
                )
            }
            _ => {
                self.status = Some(format!("unknown command: :{input}"));
                vec![]
            }
        }
    }

    // -- sort / peek --------------------------------------------------------

    /// Cycle the server-side sort key and re-issue the current level's first
    /// page under the new ordering (SPEC §6.1). The header reflects `app.sort`.
    fn cycle_sort(&mut self) -> Vec<Cmd> {
        self.sort = next_sort(self.sort);
        self.reissue_first_page()
    }

    fn reissue_first_page(&mut self) -> Vec<Cmd> {
        let Some(frame) = self.stack.last_mut() else {
            return vec![];
        };
        let id = frame.id;
        match &mut frame.kind {
            FrameKind::Keys { prefix, list } => {
                let prefix = prefix.clone();
                *list = LazyList::new();
                list.begin_page();
                vec![Cmd::ExpandPrefix {
                    prefix,
                    cursor: None,
                    frame_id: id,
                }]
            }
            FrameKind::Volumes { group, list } => {
                let group = group.clone();
                *list = LazyList::new();
                list.begin_page();
                vec![Cmd::ListVolumes {
                    group,
                    cursor: None,
                    frame_id: id,
                }]
            }
            FrameKind::Results {
                kind,
                pattern,
                list,
            } => {
                let kind = *kind;
                let pattern = pattern.clone();
                *list = LazyList::new();
                list.begin_page();
                vec![Cmd::Search {
                    kind,
                    pattern,
                    cursor: None,
                    frame_id: id,
                }]
            }
            // Health is a fixed snapshot; Detail is a single key — nothing to re-page.
            FrameKind::Health { .. } | FrameKind::Detail { .. } => vec![],
        }
    }

    /// `peek` the selected shard of the current Detail key (SPEC §5.3). Guards
    /// the OBJECT case (no tensor data) and the not-yet-loaded case.
    fn peek(&mut self) -> Vec<Cmd> {
        let Some(frame) = self.stack.last() else {
            return vec![];
        };
        let id = frame.id;
        let FrameKind::Detail {
            key,
            entry,
            shard_cursor,
            ..
        } = &frame.kind
        else {
            self.status = Some("peek only works on a key's detail".to_string());
            return vec![];
        };
        let Some(entry) = entry else {
            self.status = Some("key still loading — peek unavailable".to_string());
            return vec![];
        };
        if entry.object_type == ObjectType::Object {
            self.status = Some("cannot peek an OBJECT (no tensor data)".to_string());
            return vec![];
        }
        let coordinates = entry
            .shards
            .get(*shard_cursor)
            .map(|s| s.coordinates.clone());
        self.status = Some("peeking…".to_string());
        vec![Cmd::Peek {
            key: key.clone(),
            coordinates,
            frame_id: id,
        }]
    }

    // -- cursor movement (per frame kind) -----------------------------------

    fn select_up(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        match &mut frame.kind {
            FrameKind::Health { list } => list.select_up(),
            FrameKind::Keys { list, .. } => list.select_up(),
            FrameKind::Volumes { list, .. } => list.select_up(),
            FrameKind::Results { list, .. } => list.select_up(),
            FrameKind::Detail { shard_cursor, .. } => {
                *shard_cursor = shard_cursor.saturating_sub(1);
            }
        }
    }

    fn select_down(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        match &mut frame.kind {
            FrameKind::Health { list } => list.select_down(),
            FrameKind::Keys { list, .. } => list.select_down(),
            FrameKind::Volumes { list, .. } => list.select_down(),
            FrameKind::Results { list, .. } => list.select_down(),
            FrameKind::Detail {
                entry,
                shard_cursor,
                ..
            } => {
                if let Some(e) = entry {
                    if *shard_cursor + 1 < e.shards.len() {
                        *shard_cursor += 1;
                    }
                }
            }
        }
    }

    fn select_top(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        match &mut frame.kind {
            FrameKind::Health { list } => list.select_top(),
            FrameKind::Keys { list, .. } => list.select_top(),
            FrameKind::Volumes { list, .. } => list.select_top(),
            FrameKind::Results { list, .. } => list.select_top(),
            FrameKind::Detail { shard_cursor, .. } => *shard_cursor = 0,
        }
    }

    fn select_bottom(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        match &mut frame.kind {
            FrameKind::Health { list } => list.select_bottom(),
            FrameKind::Keys { list, .. } => list.select_bottom(),
            FrameKind::Volumes { list, .. } => list.select_bottom(),
            FrameKind::Results { list, .. } => list.select_bottom(),
            FrameKind::Detail {
                entry,
                shard_cursor,
                ..
            } => {
                *shard_cursor = entry
                    .as_ref()
                    .map_or(0, |e| e.shards.len().saturating_sub(1));
            }
        }
    }

    fn page_down(&mut self) {
        let height = self.body_rows.max(1);
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        match &mut frame.kind {
            FrameKind::Health { list } => list.page_down(height),
            FrameKind::Keys { list, .. } => list.page_down(height),
            FrameKind::Volumes { list, .. } => list.page_down(height),
            FrameKind::Results { list, .. } => list.page_down(height),
            FrameKind::Detail {
                entry,
                shard_cursor,
                ..
            } => {
                let last = entry
                    .as_ref()
                    .map_or(0, |e| e.shards.len().saturating_sub(1));
                *shard_cursor = (*shard_cursor + height).min(last);
            }
        }
    }

    fn page_up(&mut self) {
        let height = self.body_rows.max(1);
        let Some(frame) = self.stack.last_mut() else {
            return;
        };
        match &mut frame.kind {
            FrameKind::Health { list } => list.page_up(height),
            FrameKind::Keys { list, .. } => list.page_up(height),
            FrameKind::Volumes { list, .. } => list.page_up(height),
            FrameKind::Results { list, .. } => list.page_up(height),
            FrameKind::Detail { shard_cursor, .. } => {
                *shard_cursor = shard_cursor.saturating_sub(height);
            }
        }
    }

    /// Whether the current frame's list has its cursor on the trailing "load
    /// more" row.
    fn on_more_row(&self) -> bool {
        match self.stack.last().map(|f| &f.kind) {
            Some(FrameKind::Keys { list, .. }) => list.on_more_row(),
            Some(FrameKind::Volumes { list, .. }) => list.on_more_row(),
            Some(FrameKind::Results { list, .. }) => list.on_more_row(),
            _ => false,
        }
    }

    /// Fetch the next page of the current level and mark it pending. A no-op if
    /// there is nothing more to load or a request is already in flight (so
    /// hammering Enter on the "load more" row can't queue duplicate fetches).
    fn load_more(&mut self) -> Vec<Cmd> {
        let Some(frame) = self.stack.last_mut() else {
            return vec![];
        };
        let id = frame.id;
        match &mut frame.kind {
            FrameKind::Keys { prefix, list } => {
                if !list.has_more() || list.is_pending() {
                    return vec![];
                }
                let cursor = list.next_cursor();
                list.begin_page();
                vec![Cmd::ExpandPrefix {
                    prefix: prefix.clone(),
                    cursor,
                    frame_id: id,
                }]
            }
            FrameKind::Volumes { group, list } => {
                if !list.has_more() || list.is_pending() {
                    return vec![];
                }
                let cursor = list.next_cursor();
                list.begin_page();
                vec![Cmd::ListVolumes {
                    group: group.clone(),
                    cursor,
                    frame_id: id,
                }]
            }
            FrameKind::Results {
                kind,
                pattern,
                list,
            } => {
                if !list.has_more() || list.is_pending() {
                    return vec![];
                }
                let cursor = list.next_cursor();
                list.begin_page();
                vec![Cmd::Search {
                    kind: *kind,
                    pattern: pattern.clone(),
                    cursor,
                    frame_id: id,
                }]
            }
            _ => vec![],
        }
    }

    // -- drill / pop --------------------------------------------------------

    fn drill(&mut self) -> Vec<Cmd> {
        // Enter/l on the trailing "load more" row fetches the next page instead
        // of drilling into an item.
        if self.on_more_row() {
            return self.load_more();
        }
        let action = {
            let Some(frame) = self.stack.last() else {
                return vec![];
            };
            match &frame.kind {
                FrameKind::Health { list } => list.selected().map(|row| match row {
                    LandingRow::Prefix(kp) => Drill::Prefix(kp.clone()),
                    LandingRow::Group(g) => Drill::Group(g.group.clone()),
                }),
                FrameKind::Keys { list, .. } => list.selected().map(|kp| Drill::Prefix(kp.clone())),
                FrameKind::Results { list, .. } => list.selected().map(|m| match m {
                    SearchMatch::Key(k) => Drill::Key(k.key.clone()),
                    SearchMatch::Volume(_) => Drill::Volume,
                }),
                FrameKind::Volumes { .. } => Some(Drill::Volume),
                FrameKind::Detail { .. } => None,
            }
        };
        match action {
            Some(Drill::Prefix(kp)) => self.open_prefix(kp),
            Some(Drill::Group(group)) => self.open_group(group),
            Some(Drill::Key(key)) => self.open_key_detail(key),
            Some(Drill::Volume) => {
                self.status = Some("no drill-down for a volume yet".to_string());
                vec![]
            }
            None => vec![],
        }
    }

    /// Push a Detail frame for `key` and fetch its `KeyEntry` (SPEC §5.2).
    fn open_key_detail(&mut self, key: String) -> Vec<Cmd> {
        let id = self.new_frame_id();
        self.stack.push(Frame {
            id,
            title: middle_elide(&key, 40),
            kind: FrameKind::Detail {
                key: key.clone(),
                entry: None,
                error: None,
                peek: None,
                shard_cursor: 0,
            },
        });
        vec![Cmd::FetchKey { key, frame_id: id }]
    }

    /// `:peek <key>` — jump to a key's Detail and request its stats at once.
    fn open_peek(&mut self, key: String) -> Vec<Cmd> {
        let id = self.new_frame_id();
        self.stack.push(Frame {
            id,
            title: middle_elide(&key, 40),
            kind: FrameKind::Detail {
                key: key.clone(),
                entry: None,
                error: None,
                peek: None,
                shard_cursor: 0,
            },
        });
        vec![
            Cmd::FetchKey {
                key: key.clone(),
                frame_id: id,
            },
            Cmd::Peek {
                key,
                coordinates: None,
                frame_id: id,
            },
        ]
    }

    /// Push a Results frame and fetch its first `search` page (SPEC §5.2).
    fn open_search(&mut self, kind: SearchKind, pattern: String, title: String) -> Vec<Cmd> {
        let id = self.new_frame_id();
        let mut list = LazyList::new();
        list.begin_page();
        self.stack.push(Frame {
            id,
            title,
            kind: FrameKind::Results {
                kind,
                pattern: pattern.clone(),
                list,
            },
        });
        vec![Cmd::Search {
            kind,
            pattern,
            cursor: None,
            frame_id: id,
        }]
    }

    fn open_prefix(&mut self, kp: KeyPrefix) -> Vec<Cmd> {
        let id = self.new_frame_id();
        // Only a terminal key (a leaf) drills straight to Detail; an intermediate
        // node expands one level deeper. `keys <= 1` is NOT a leaf test — a single
        // key can live below an intermediate prefix (e.g. `model.layer0` holding
        // only `model.layer0.weight`).
        if kp.is_leaf {
            self.stack.push(Frame {
                id,
                title: middle_elide(&kp.prefix, 40),
                kind: FrameKind::Detail {
                    key: kp.prefix.clone(),
                    entry: None,
                    error: None,
                    peek: None,
                    shard_cursor: 0,
                },
            });
            return vec![Cmd::FetchKey {
                key: kp.prefix,
                frame_id: id,
            }];
        }
        let mut list = LazyList::new();
        list.begin_page();
        self.stack.push(Frame {
            id,
            title: last_segment(&kp.prefix),
            kind: FrameKind::Keys {
                prefix: kp.prefix.clone(),
                list,
            },
        });
        vec![Cmd::ExpandPrefix {
            prefix: kp.prefix,
            cursor: None,
            frame_id: id,
        }]
    }

    fn open_group(&mut self, group: String) -> Vec<Cmd> {
        let id = self.new_frame_id();
        let mut list = LazyList::new();
        list.begin_page();
        self.scope = Scope::Group(group.clone());
        self.stack.push(Frame {
            id,
            title: group.clone(),
            kind: FrameKind::Volumes {
                group: group.clone(),
                list,
            },
        });
        vec![Cmd::ListVolumes {
            group,
            cursor: None,
            frame_id: id,
        }]
    }

    fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
        self.recompute_scope();
    }

    fn recompute_scope(&mut self) {
        self.scope = self
            .stack
            .iter()
            .rev()
            .find_map(|f| match &f.kind {
                FrameKind::Volumes { group, .. } => Some(Scope::Group(group.clone())),
                _ => None,
            })
            .unwrap_or(Scope::All);
    }

    // -- data arrival -------------------------------------------------------

    fn on_summary(&mut self, res: Result<Summary, String>) {
        self.inflight = self.inflight.saturating_sub(1);
        match res {
            Ok(summary) => {
                let rows: Vec<LandingRow> = summary
                    .key_prefixes
                    .iter()
                    .cloned()
                    .map(LandingRow::Prefix)
                    .chain(summary.volume_groups.iter().cloned().map(LandingRow::Group))
                    .collect();
                if let Some(frame) = self.stack.first_mut() {
                    if let FrameKind::Health { list } = &mut frame.kind {
                        let keep = list.selected_index();
                        let mut fresh = LazyList::new();
                        fresh.append_page(rows, None);
                        fresh.set_cursor(keep);
                        *list = fresh;
                    }
                }
                self.summary = Some(summary);
                self.summary_loaded_at = Some(Instant::now());
                self.status = None;
            }
            Err(e) => {
                self.status = Some(format!("summary failed: {e}"));
                if let Some(frame) = self.stack.first_mut() {
                    if let FrameKind::Health { list } = &mut frame.kind {
                        list.fail_page();
                    }
                }
            }
        }
    }

    fn on_prefix_page(&mut self, id: u64, resp: Result<ExpandPrefixResponse, String>) {
        self.inflight = self.inflight.saturating_sub(1);
        let mut err = None;
        if let Some(frame) = self.stack.iter_mut().find(|f| f.id == id) {
            if let FrameKind::Keys { list, .. } = &mut frame.kind {
                match resp {
                    Ok(r) => list.append_page(r.children, r.next_cursor),
                    Err(e) => {
                        list.fail_page();
                        err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = err {
            self.status = Some(format!("expand_prefix failed: {e}"));
        }
    }

    fn on_volumes_page(&mut self, id: u64, resp: Result<ListVolumesResponse, String>) {
        self.inflight = self.inflight.saturating_sub(1);
        let mut err = None;
        if let Some(frame) = self.stack.iter_mut().find(|f| f.id == id) {
            if let FrameKind::Volumes { list, .. } = &mut frame.kind {
                match resp {
                    Ok(r) => list.append_page(r.volumes, r.next_cursor),
                    Err(e) => {
                        list.fail_page();
                        err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = err {
            self.status = Some(format!("list_volumes failed: {e}"));
        }
    }

    fn on_key_loaded(&mut self, id: u64, resp: Result<KeyEntry, String>) {
        self.inflight = self.inflight.saturating_sub(1);
        let mut err = None;
        if let Some(frame) = self.stack.iter_mut().find(|f| f.id == id) {
            if let FrameKind::Detail { entry, error, .. } = &mut frame.kind {
                match resp {
                    Ok(k) => {
                        *entry = Some(k);
                        *error = None;
                    }
                    Err(e) => {
                        *error = Some(e.clone());
                        err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = err {
            self.status = Some(format!("key failed: {e}"));
        }
    }

    fn on_search_page(&mut self, id: u64, resp: Result<SearchResponse, String>) {
        self.inflight = self.inflight.saturating_sub(1);
        let mut err = None;
        if let Some(frame) = self.stack.iter_mut().find(|f| f.id == id) {
            if let FrameKind::Results { list, .. } = &mut frame.kind {
                match resp {
                    Ok(r) => list.append_page(r.matches, r.next_cursor),
                    Err(e) => {
                        list.fail_page();
                        err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = err {
            self.status = Some(format!("search failed: {e}"));
        }
    }

    fn on_peek_loaded(&mut self, id: u64, resp: Result<PeekResult, String>) {
        self.inflight = self.inflight.saturating_sub(1);
        let mut err = None;
        if let Some(frame) = self.stack.iter_mut().find(|f| f.id == id) {
            if let FrameKind::Detail { peek, .. } = &mut frame.kind {
                match resp {
                    Ok(p) => {
                        *peek = Some(p);
                        self.status = None;
                    }
                    Err(e) => err = Some(e),
                }
            }
        }
        if let Some(e) = err {
            self.status = Some(format!("peek failed: {e}"));
        }
    }

    /// A summary older than this is rendered as stale in the header.
    pub(crate) fn refresh_threshold(&self) -> Duration {
        self.refresh.saturating_mul(2).max(Duration::from_secs(2))
    }

    pub(crate) fn breadcrumb(&self) -> String {
        self.stack
            .iter()
            .map(|f| f.title.as_str())
            .collect::<Vec<_>>()
            .join(" ▸ ")
    }

    // -- headless driving ---------------------------------------------------

    /// Execute one command against the provider and return the resulting
    /// message. The interactive loop spawns each `Cmd` on a task (see
    /// [`run_cmds`]); the headless path has no loop, so it awaits each command
    /// inline through this helper and feeds the message straight back to
    /// [`update`].
    pub(crate) async fn run_cmd(&self, cmd: Cmd) -> Msg {
        let provider = self.provider.clone();
        let sort = self.sort;
        let order = self.order;
        match cmd {
            Cmd::FetchSummary => {
                Msg::SummaryLoaded(provider.summary().await.map_err(|e| format!("{e:#}")))
            }
            Cmd::ExpandPrefix {
                prefix,
                cursor,
                frame_id,
            } => {
                let req = ExpandPrefixRequest {
                    prefix,
                    limit: Some(PAGE),
                    cursor,
                    sort_by: Some(sort),
                    order: Some(order),
                };
                let resp = provider
                    .expand_prefix(req)
                    .await
                    .map_err(|e| format!("{e:#}"));
                Msg::PrefixPage { frame_id, resp }
            }
            Cmd::ListVolumes {
                group,
                cursor,
                frame_id,
            } => {
                let req = ListVolumesRequest {
                    group,
                    limit: Some(PAGE),
                    cursor,
                    sort_by: Some(sort),
                    order: Some(order),
                };
                let resp = provider
                    .list_volumes(req)
                    .await
                    .map_err(|e| format!("{e:#}"));
                Msg::VolumesPage { frame_id, resp }
            }
            Cmd::FetchKey { key, frame_id } => {
                let resp = provider
                    .key(KeyRequest { key })
                    .await
                    .map_err(|e| format!("{e:#}"));
                Msg::KeyLoaded { frame_id, resp }
            }
            Cmd::Search {
                kind,
                pattern,
                cursor,
                frame_id,
            } => {
                let req = SearchRequest {
                    kind,
                    pattern,
                    limit: Some(PAGE),
                    cursor,
                    sort_by: Some(sort),
                    order: Some(order),
                };
                let resp = provider.search(req).await.map_err(|e| format!("{e:#}"));
                Msg::SearchResults { frame_id, resp }
            }
            Cmd::Peek {
                key,
                coordinates,
                frame_id,
            } => {
                let resp = provider
                    .peek(PeekRequest { key, coordinates })
                    .await
                    .map_err(|e| format!("{e:#}"));
                Msg::PeekLoaded { frame_id, resp }
            }
        }
    }

    /// Index of the landing prefix row best suited to demonstrate a drill: the
    /// branch (more than one key) with the most keys. `None` when the landing
    /// has no branch prefix (only leaves/groups).
    pub(crate) fn landing_branch_index(&self) -> Option<usize> {
        let FrameKind::Health { list } = &self.stack[0].kind else {
            return None;
        };
        let (_, rows) = list.window(list.len().max(1));
        rows.iter()
            .enumerate()
            .filter_map(|(i, row)| match row {
                LandingRow::Prefix(kp) if kp.keys > 1 => Some((i, kp.keys)),
                _ => None,
            })
            .max_by_key(|(_, keys)| *keys)
            .map(|(i, _)| i)
    }

    /// Select the branchiest landing prefix (falling back to the current
    /// selection) and drill into it, returning the fetch commands to run. Used
    /// by the headless walkthrough to descend one real trie level.
    pub(crate) fn drill_best_landing_prefix(&mut self) -> Vec<Cmd> {
        if let Some(idx) = self.landing_branch_index() {
            if let FrameKind::Health { list } = &mut self.stack[0].kind {
                list.set_cursor(idx);
            }
        }
        self.drill()
    }
}

/// What a drill resolves to once the borrow on the current frame is released.
enum Drill {
    Prefix(KeyPrefix),
    Group(String),
    Key(String),
    Volume,
}

/// Cycle the sort key: partial -> bytes -> reachable -> keys -> name -> partial
/// (SPEC §6.1).
fn next_sort(s: SortKey) -> SortKey {
    match s {
        SortKey::Partial => SortKey::Bytes,
        SortKey::Bytes => SortKey::Reachable,
        SortKey::Reachable => SortKey::Keys,
        SortKey::Keys => SortKey::Name,
        SortKey::Name => SortKey::Partial,
    }
}

// ---------------------------------------------------------------------------
// Messages and commands
// ---------------------------------------------------------------------------

/// Every event the reducer consumes: terminal input, data arrival, and the tick.
pub(crate) enum Msg {
    Key(KeyEvent),
    /// A terminal resize — no state change, just forces a redraw.
    Resize,
    RefreshTick,
    SummaryLoaded(Result<Summary, String>),
    PrefixPage {
        frame_id: u64,
        resp: Result<ExpandPrefixResponse, String>,
    },
    VolumesPage {
        frame_id: u64,
        resp: Result<ListVolumesResponse, String>,
    },
    KeyLoaded {
        frame_id: u64,
        resp: Result<KeyEntry, String>,
    },
    SearchResults {
        frame_id: u64,
        resp: Result<SearchResponse, String>,
    },
    PeekLoaded {
        frame_id: u64,
        resp: Result<PeekResult, String>,
    },
}

/// A side effect the event loop performs off the render path. Every variant is a
/// provider call whose result comes back as a `Msg`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Cmd {
    FetchSummary,
    ExpandPrefix {
        prefix: String,
        cursor: Option<String>,
        frame_id: u64,
    },
    ListVolumes {
        group: String,
        cursor: Option<String>,
        frame_id: u64,
    },
    FetchKey {
        key: String,
        frame_id: u64,
    },
    Search {
        kind: SearchKind,
        pattern: String,
        cursor: Option<String>,
        frame_id: u64,
    },
    Peek {
        key: String,
        coordinates: Option<Vec<u64>>,
        frame_id: u64,
    },
}

/// The pure reducer: mutate `app`, return the side effects to run. No IO here so
/// it is fully unit-testable.
pub(crate) fn update(app: &mut App, msg: Msg) -> Vec<Cmd> {
    match msg {
        Msg::Key(k) => app.on_key(k),
        Msg::Resize => vec![],
        Msg::RefreshTick => vec![Cmd::FetchSummary],
        Msg::SummaryLoaded(res) => {
            app.on_summary(res);
            vec![]
        }
        Msg::PrefixPage { frame_id, resp } => {
            app.on_prefix_page(frame_id, resp);
            vec![]
        }
        Msg::VolumesPage { frame_id, resp } => {
            app.on_volumes_page(frame_id, resp);
            vec![]
        }
        Msg::KeyLoaded { frame_id, resp } => {
            app.on_key_loaded(frame_id, resp);
            vec![]
        }
        Msg::SearchResults { frame_id, resp } => {
            app.on_search_page(frame_id, resp);
            vec![]
        }
        Msg::PeekLoaded { frame_id, resp } => {
            app.on_peek_loaded(frame_id, resp);
            vec![]
        }
    }
}

// ---------------------------------------------------------------------------
// Text helpers (shared with the view)
// ---------------------------------------------------------------------------

pub(crate) fn last_segment(prefix: &str) -> String {
    prefix.rsplit('.').next().unwrap_or(prefix).to_string()
}

/// Middle-elide a long string, keeping the distinguishing tail (SPEC §7).
pub(crate) fn middle_elide(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    if max <= 1 {
        return "…".to_string();
    }
    let keep = max - 1;
    let head = keep / 2;
    let tail = keep - head;
    let h: String = chars[..head].iter().collect();
    let t: String = chars[chars.len() - tail..].iter().collect();
    format!("{h}…{t}")
}

pub(crate) fn human_bytes(b: f64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    if b <= 0.0 {
        return "0 B".to_string();
    }
    let mut v = b;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{v:.0} {}", UNITS[i])
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

pub(crate) fn thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let len = bytes.len();
    bytes
        .iter()
        .enumerate()
        .flat_map(|(i, c)| {
            let sep = (i > 0 && (len - i) % 3 == 0).then_some(',');
            sep.into_iter().chain(std::iter::once(*c as char))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Event loop + terminal lifecycle
// ---------------------------------------------------------------------------

/// Run the interactive UI until the user quits. Restores the terminal on exit,
/// error, or panic.
pub(crate) async fn run(provider: Arc<dyn Provider>, refresh: Duration) -> Result<()> {
    install_panic_hook();
    let mut terminal = init_terminal()?;
    let _guard = TerminalGuard;

    let (tx, mut rx) = mpsc::unbounded_channel::<Msg>();
    spawn_input(tx.clone());

    let mut app = App::new(provider, tx, refresh);
    run_cmds(&mut app, vec![Cmd::FetchSummary]);

    let refresh_period = refresh.max(Duration::from_secs(1));
    let mut refresh_tick = tokio::time::interval(refresh_period);
    refresh_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut spinner = tokio::time::interval(Duration::from_millis(120));
    spinner.set_missed_tick_behavior(MissedTickBehavior::Skip);

    app.body_rows = body_rows(&terminal);
    terminal.draw(|f| view(&app, f))?;

    loop {
        let dirty;
        tokio::select! {
            maybe = rx.recv() => {
                let Some(msg) = maybe else { break };
                app.body_rows = body_rows(&terminal);
                let cmds = update(&mut app, msg);
                run_cmds(&mut app, cmds);
                dirty = true;
            }
            _ = refresh_tick.tick() => {
                let cmds = update(&mut app, Msg::RefreshTick);
                run_cmds(&mut app, cmds);
                dirty = true;
            }
            // Only animate the spinner while work is outstanding — no fixed FPS
            // when idle.
            _ = spinner.tick(), if app.inflight > 0 => {
                app.spinner = app.spinner.wrapping_add(1);
                dirty = true;
            }
        }
        if app.should_quit {
            break;
        }
        if dirty {
            terminal.draw(|f| view(&app, f))?;
        }
    }
    Ok(())
}

/// Spawn each command on a tokio task; its result returns as a `Msg`. Bumps the
/// in-flight counter so the spinner runs; the matching `on_*` handler clears it.
fn run_cmds(app: &mut App, cmds: Vec<Cmd>) {
    for cmd in cmds {
        app.inflight += 1;
        let provider = app.provider.clone();
        let tx = app.tx.clone();
        let sort = app.sort;
        let order = app.order;
        match cmd {
            Cmd::FetchSummary => {
                tokio::spawn(async move {
                    let resp = provider.summary().await.map_err(|e| format!("{e:#}"));
                    let _ = tx.send(Msg::SummaryLoaded(resp));
                });
            }
            Cmd::ExpandPrefix {
                prefix,
                cursor,
                frame_id,
            } => {
                tokio::spawn(async move {
                    let req = ExpandPrefixRequest {
                        prefix,
                        limit: Some(PAGE),
                        cursor,
                        sort_by: Some(sort),
                        order: Some(order),
                    };
                    let resp = provider
                        .expand_prefix(req)
                        .await
                        .map_err(|e| format!("{e:#}"));
                    let _ = tx.send(Msg::PrefixPage { frame_id, resp });
                });
            }
            Cmd::ListVolumes {
                group,
                cursor,
                frame_id,
            } => {
                tokio::spawn(async move {
                    let req = ListVolumesRequest {
                        group,
                        limit: Some(PAGE),
                        cursor,
                        sort_by: Some(sort),
                        order: Some(order),
                    };
                    let resp = provider
                        .list_volumes(req)
                        .await
                        .map_err(|e| format!("{e:#}"));
                    let _ = tx.send(Msg::VolumesPage { frame_id, resp });
                });
            }
            Cmd::FetchKey { key, frame_id } => {
                tokio::spawn(async move {
                    let resp = provider
                        .key(KeyRequest { key })
                        .await
                        .map_err(|e| format!("{e:#}"));
                    let _ = tx.send(Msg::KeyLoaded { frame_id, resp });
                });
            }
            Cmd::Search {
                kind,
                pattern,
                cursor,
                frame_id,
            } => {
                tokio::spawn(async move {
                    let req = SearchRequest {
                        kind,
                        pattern,
                        limit: Some(PAGE),
                        cursor,
                        sort_by: Some(sort),
                        order: Some(order),
                    };
                    let resp = provider.search(req).await.map_err(|e| format!("{e:#}"));
                    let _ = tx.send(Msg::SearchResults { frame_id, resp });
                });
            }
            Cmd::Peek {
                key,
                coordinates,
                frame_id,
            } => {
                tokio::spawn(async move {
                    let req = PeekRequest { key, coordinates };
                    let resp = provider.peek(req).await.map_err(|e| format!("{e:#}"));
                    let _ = tx.send(Msg::PeekLoaded { frame_id, resp });
                });
            }
        }
    }
}

fn body_rows(terminal: &Tui) -> usize {
    // Terminal height minus the 4-row header and 4-row footer chrome.
    terminal
        .size()
        .map(|s| (s.height as usize).saturating_sub(8))
        .unwrap_or(20)
        .max(1)
}

/// Read terminal events on a dedicated OS thread (crossterm's `read` blocks) and
/// forward them into the async loop's channel.
fn spawn_input(tx: mpsc::UnboundedSender<Msg>) {
    std::thread::spawn(move || {
        loop {
            match event::read() {
                Ok(Event::Key(k)) if k.kind == KeyEventKind::Press => {
                    if tx.send(Msg::Key(k)).is_err() {
                        break;
                    }
                }
                Ok(Event::Resize(_, _)) => {
                    if tx.send(Msg::Resize).is_err() {
                        break;
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });
}

fn init_terminal() -> Result<Tui> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal() -> Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, cursor::Show)?;
    Ok(())
}

/// Restores the terminal on scope exit (normal return or `?` propagation).
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = restore_terminal();
    }
}

/// Ensure a panic leaves the terminal usable before printing the message.
fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        default(info);
    }));
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;
    use crate::data::FileProvider;
    use crate::model::KeyPrefix;
    use crate::model::ObjectType;
    use crate::model::Shard;
    use crate::model::Totals;
    use crate::model::VolumeGroup;

    fn test_app() -> App {
        let (tx, _rx) = mpsc::unbounded_channel();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let provider: Arc<dyn Provider> = Arc::new(FileProvider::new(dir));
        let mut app = App::new(provider, tx, Duration::from_secs(5));
        app.body_rows = 20;
        app
    }

    fn prefix(name: &str, keys: u64, partial: u64) -> KeyPrefix {
        KeyPrefix {
            prefix: name.to_string(),
            keys,
            objects: Some(0),
            tensors: Some(0),
            dtensors: Some(keys),
            partial: Some(partial),
            bytes: Some(keys as f64 * 1024.0),
            is_leaf: false,
        }
    }

    fn vol(id: &str) -> crate::model::Volume {
        crate::model::Volume {
            volume_id: id.to_string(),
            hostname: "host".to_string(),
            transport: None,
            num_keys: 1,
            bytes: 1.0,
            reachable: true,
        }
    }

    fn sample_summary() -> Summary {
        Summary {
            schema_version: 1,
            captured_at: "2026-07-17T17:40:00Z".to_string(),
            store_name: "torchstore".to_string(),
            strategy: "LocalRankStrategy".to_string(),
            totals: Totals {
                volumes: 4096,
                keys: 4821,
                bytes: 3.1e14,
                partial_dtensors: 2,
            },
            volume_groups: vec![VolumeGroup {
                group: "rack:A12".to_string(),
                volumes: 4096,
                keys: 4821,
                bytes: 3.1e14,
                transports: HashMap::new(),
                reachable: 4094,
            }],
            key_prefixes: vec![
                prefix("model", 4820, 2),
                KeyPrefix {
                    prefix: "metadata.config".to_string(),
                    keys: 1,
                    objects: Some(1),
                    tensors: Some(0),
                    dtensors: Some(0),
                    partial: Some(0),
                    bytes: None,
                    is_leaf: true,
                },
            ],
            histograms: Default::default(),
        }
    }

    fn key(code: KeyCode) -> Msg {
        Msg::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn health_len(app: &App) -> usize {
        match &app.stack[0].kind {
            FrameKind::Health { list } => list.len(),
            _ => panic!("root is not health"),
        }
    }

    #[test]
    fn summary_populates_landing_list() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        // 2 key prefixes + 1 volume group.
        assert_eq!(health_len(&app), 3, "landing rows = prefixes + groups");
        assert!(app.summary.is_some());
        assert!(app.summary_loaded_at.is_some());
    }

    #[test]
    fn drill_branch_prefix_pushes_keys_and_fetches() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        // Cursor starts on "model" (a branch, keys=4820).
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert_eq!(app.stack.len(), 2, "drilled one level");
        match &app.stack[1].kind {
            FrameKind::Keys { prefix, .. } => assert_eq!(prefix, "model"),
            _ => panic!("expected a Keys frame"),
        }
        assert_eq!(
            cmds,
            vec![Cmd::ExpandPrefix {
                prefix: "model".to_string(),
                cursor: None,
                frame_id: 1,
            }],
            "drilling a branch fetches its first page"
        );
    }

    #[test]
    fn drill_leaf_prefix_pushes_detail_and_fetches_key() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Down)); // move to "metadata.config" (leaf, keys=1)
        let cmds = update(&mut app, key(KeyCode::Char('l')));
        assert_eq!(app.stack.len(), 2);
        match &app.stack[1].kind {
            FrameKind::Detail { key, entry, .. } => {
                assert_eq!(key, "metadata.config");
                assert!(entry.is_none(), "detail entry loads asynchronously");
            }
            _ => panic!("expected a Detail frame"),
        }
        assert_eq!(
            cmds,
            vec![Cmd::FetchKey {
                key: "metadata.config".to_string(),
                frame_id: 1,
            }]
        );
    }

    #[test]
    fn drill_intermediate_single_key_prefix_expands_not_detail() {
        // Regression: an intermediate node like `model.layer0` can hold exactly
        // one key (`model.layer0.weight`) while NOT being a key itself. `keys==1`
        // must not be mistaken for a leaf — drilling it must expand, not fetch a
        // (non-existent) key `model.layer0` that would stick on "loading".
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Enter)); // Keys{model}, frame 1
        let mut intermediate = prefix("model.layer0", 1, 0);
        intermediate.is_leaf = false;
        update(
            &mut app,
            Msg::PrefixPage {
                frame_id: 1,
                resp: Ok(ExpandPrefixResponse {
                    children: vec![intermediate],
                    next_cursor: None,
                }),
            },
        );
        let cmds = update(&mut app, key(KeyCode::Enter));
        match &app.stack.last().unwrap().kind {
            FrameKind::Keys { prefix, .. } => assert_eq!(prefix, "model.layer0"),
            _ => panic!("intermediate single-key prefix must expand into a Keys frame"),
        }
        assert!(
            matches!(cmds.as_slice(), [Cmd::ExpandPrefix { .. }]),
            "must expand, not FetchKey"
        );
    }

    #[test]
    fn failed_key_load_sets_error_not_stuck_loading() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Down)); // metadata.config (leaf)
        update(&mut app, key(KeyCode::Char('l'))); // Detail frame 1 + FetchKey
        update(
            &mut app,
            Msg::KeyLoaded {
                frame_id: 1,
                resp: Err("boom".to_string()),
            },
        );
        match &app.stack.last().unwrap().kind {
            FrameKind::Detail { entry, error, .. } => {
                assert!(entry.is_none(), "no entry on failure");
                assert_eq!(
                    error.as_deref(),
                    Some("boom"),
                    "error is recorded for the view"
                );
            }
            _ => panic!("expected a Detail frame"),
        }
    }

    #[test]
    fn prefix_page_appends_to_matching_frame() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Enter)); // push Keys{model}, frame 1
        let resp = ExpandPrefixResponse {
            children: vec![prefix("model.layers", 4800, 2), prefix("model.embed", 2, 0)],
            next_cursor: Some("model:page2".to_string()),
        };
        update(
            &mut app,
            Msg::PrefixPage {
                frame_id: 1,
                resp: Ok(resp),
            },
        );
        match &app.stack[1].kind {
            FrameKind::Keys { list, .. } => {
                assert_eq!(list.len(), 2);
                assert!(list.has_more(), "next_cursor => more pages");
                assert!(!list.is_pending(), "page arrival clears the spinner");
            }
            _ => panic!("expected a Keys frame"),
        }
    }

    #[test]
    fn late_page_for_popped_frame_is_ignored() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Enter)); // frame 1
        update(&mut app, key(KeyCode::Esc)); // pop back to root
        // A page for the now-gone frame 1 must not panic or leak into root.
        update(
            &mut app,
            Msg::PrefixPage {
                frame_id: 1,
                resp: Ok(ExpandPrefixResponse {
                    children: vec![prefix("model.layers", 4800, 2)],
                    next_cursor: None,
                }),
            },
        );
        assert_eq!(app.stack.len(), 1, "still at root");
    }

    #[test]
    fn pop_returns_to_root_and_clears_scope() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        // Drill into the volume group (last landing row) to set scope.
        update(&mut app, key(KeyCode::Char('G')));
        update(&mut app, key(KeyCode::Enter));
        assert!(matches!(app.scope, Scope::Group(_)), "group sets scope");
        update(&mut app, key(KeyCode::Esc));
        assert_eq!(app.stack.len(), 1);
        assert!(matches!(app.scope, Scope::All), "popping clears scope");
    }

    #[test]
    fn pop_at_root_is_a_noop() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Esc));
        assert_eq!(app.stack.len(), 1, "cannot pop past the landing view");
    }

    #[test]
    fn quit_sets_flag() {
        let mut app = test_app();
        assert!(!app.should_quit);
        update(&mut app, key(KeyCode::Char('q')));
        assert!(app.should_quit);
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = test_app();
        update(
            &mut app,
            Msg::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        );
        assert!(app.should_quit);
    }

    #[test]
    fn refresh_and_tick_fetch_summary() {
        let mut app = test_app();
        assert_eq!(
            update(&mut app, key(KeyCode::Char('r'))),
            vec![Cmd::FetchSummary]
        );
        assert_eq!(update(&mut app, Msg::RefreshTick), vec![Cmd::FetchSummary]);
    }

    /// Type each char of `s` into whatever input mode is open.
    fn type_str(app: &mut App, s: &str) {
        for c in s.chars() {
            update(app, key(KeyCode::Char(c)));
        }
    }

    #[test]
    fn slash_opens_filter_mode_and_esc_cancels() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char('/')));
        assert!(
            matches!(app.mode, Mode::Filter { .. }),
            "/ opens the filter editor"
        );
        type_str(&mut app, "attn");
        match &app.mode {
            Mode::Filter { buffer } => assert_eq!(buffer, "attn", "chars accumulate in the buffer"),
            _ => panic!("expected filter mode"),
        }
        update(&mut app, key(KeyCode::Esc));
        assert!(
            matches!(app.mode, Mode::Browse),
            "esc cancels back to browse"
        );
        assert_eq!(app.stack.len(), 1, "cancelling opens no results frame");
    }

    #[test]
    fn filter_submits_a_search_command() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char('/')));
        type_str(&mut app, "attn.wq");
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert!(
            matches!(app.mode, Mode::Browse),
            "submitting closes the editor"
        );
        assert_eq!(app.stack.len(), 2, "a results frame was pushed");
        assert!(matches!(app.stack[1].kind, FrameKind::Results { .. }));
        assert_eq!(
            cmds,
            vec![Cmd::Search {
                kind: SearchKind::Key,
                pattern: "attn.wq".to_string(),
                cursor: None,
                frame_id: 1,
            }],
            "/ filter compiles to a key search"
        );
    }

    #[test]
    fn empty_filter_is_a_noop() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char('/')));
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert!(cmds.is_empty(), "an empty pattern issues no search");
        assert_eq!(app.stack.len(), 1, "no frame pushed");
    }

    #[test]
    fn search_results_populate_and_drill_to_detail() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char('/')));
        type_str(&mut app, "attn.wq");
        update(&mut app, key(KeyCode::Enter)); // Results frame 1
        let resp = SearchResponse {
            matches: vec![SearchMatch::Key(KeyEntry {
                key: "model.layers.0.attn.wq.weight".to_string(),
                object_type: ObjectType::TensorSlice,
                dtype: Some("float32".to_string()),
                global_shape: Some(vec![4096, 4096]),
                fully_committed: true,
                mesh_shape: Some(vec![2, 2]),
                shards: vec![],
            })],
            next_cursor: None,
        };
        update(
            &mut app,
            Msg::SearchResults {
                frame_id: 1,
                resp: Ok(resp),
            },
        );
        match &app.stack[1].kind {
            FrameKind::Results { list, .. } => assert_eq!(list.len(), 1, "match arrived"),
            _ => panic!("expected a Results frame"),
        }
        // Drilling a key match opens its Detail and fetches the entry.
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert!(matches!(
            app.stack.last().map(|f| &f.kind),
            Some(FrameKind::Detail { .. })
        ));
        assert_eq!(
            cmds,
            vec![Cmd::FetchKey {
                key: "model.layers.0.attn.wq.weight".to_string(),
                frame_id: 2,
            }]
        );
    }

    #[test]
    fn command_key_jumps_to_detail() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char(':')));
        assert!(matches!(app.mode, Mode::Command { .. }));
        type_str(&mut app, "key model.embed.weight");
        let cmds = update(&mut app, key(KeyCode::Enter));
        match &app.stack[1].kind {
            FrameKind::Detail { key, .. } => assert_eq!(key, "model.embed.weight"),
            _ => panic!("expected a Detail frame"),
        }
        assert_eq!(
            cmds,
            vec![Cmd::FetchKey {
                key: "model.embed.weight".to_string(),
                frame_id: 1,
            }]
        );
    }

    #[test]
    fn command_group_jumps_to_volumes_and_sets_scope() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char(':')));
        type_str(&mut app, "group rack:A12");
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert!(matches!(app.stack[1].kind, FrameKind::Volumes { .. }));
        assert!(matches!(app.scope, Scope::Group(ref g) if g == "rack:A12"));
        assert_eq!(
            cmds,
            vec![Cmd::ListVolumes {
                group: "rack:A12".to_string(),
                cursor: None,
                frame_id: 1,
            }]
        );
    }

    #[test]
    fn enter_on_load_more_row_fetches_next_page_and_scrolling_does_not() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Char('G'))); // select the volume group row
        let cmds = update(&mut app, key(KeyCode::Enter)); // open Volumes
        let id = app.stack.last().expect("volumes frame").id;
        assert!(
            matches!(cmds.as_slice(), [Cmd::ListVolumes { cursor: None, .. }]),
            "opening a group fetches its first page"
        );

        // First page arrives, and another page is available.
        update(
            &mut app,
            Msg::VolumesPage {
                frame_id: id,
                resp: Ok(ListVolumesResponse {
                    volumes: vec![vol("vol-0"), vol("vol-1")],
                    next_cursor: Some("rack:A12:page2".to_string()),
                }),
            },
        );

        // Scrolling down onto items must NOT auto-fetch — paging is manual now.
        assert!(
            update(&mut app, key(KeyCode::Down)).is_empty(),
            "moving the cursor issues no fetch"
        );

        // Land on the trailing "load more" row and press Enter.
        update(&mut app, key(KeyCode::Char('G')));
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert_eq!(
            cmds,
            vec![Cmd::ListVolumes {
                group: "rack:A12".to_string(),
                cursor: Some("rack:A12:page2".to_string()),
                frame_id: id,
            }],
            "Enter on the load-more row fetches the next page at the saved cursor"
        );
    }

    #[test]
    fn command_partial_searches_keys_with_anomaly_sort() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char(':')));
        type_str(&mut app, "partial");
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert_eq!(
            app.sort,
            SortKey::Partial,
            ":partial forces the anomaly sort"
        );
        assert_eq!(app.order, Order::Desc);
        assert_eq!(
            cmds,
            vec![Cmd::Search {
                kind: SearchKind::Key,
                pattern: String::new(),
                cursor: None,
                frame_id: 1,
            }]
        );
    }

    #[test]
    fn command_unreachable_searches_volumes() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char(':')));
        type_str(&mut app, "unreachable");
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert_eq!(app.sort, SortKey::Reachable);
        assert_eq!(
            cmds,
            vec![Cmd::Search {
                kind: SearchKind::Volume,
                pattern: String::new(),
                cursor: None,
                frame_id: 1,
            }]
        );
    }

    #[test]
    fn command_peek_fetches_key_and_peek() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char(':')));
        type_str(&mut app, "peek model.layers.0.attn.wq.weight");
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert!(matches!(app.stack[1].kind, FrameKind::Detail { .. }));
        assert_eq!(
            cmds,
            vec![
                Cmd::FetchKey {
                    key: "model.layers.0.attn.wq.weight".to_string(),
                    frame_id: 1,
                },
                Cmd::Peek {
                    key: "model.layers.0.attn.wq.weight".to_string(),
                    coordinates: None,
                    frame_id: 1,
                },
            ]
        );
    }

    #[test]
    fn unknown_command_sets_an_error_and_pushes_no_frame() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char(':')));
        type_str(&mut app, "wat");
        let cmds = update(&mut app, key(KeyCode::Enter));
        assert!(cmds.is_empty());
        assert_eq!(app.stack.len(), 1, "no navigation on an unknown command");
        assert!(
            app.status
                .as_deref()
                .unwrap_or_default()
                .contains("unknown command")
        );
    }

    #[test]
    fn sort_key_cycles_and_reissues_the_current_level() {
        let mut app = test_app();
        assert_eq!(app.sort, SortKey::Partial, "default is anomaly-first");
        // On the Health frame there is nothing to re-page.
        let cmds = update(&mut app, key(KeyCode::Char('s')));
        assert_eq!(app.sort, SortKey::Bytes, "s advances the sort key");
        assert!(cmds.is_empty(), "the fixed summary is not re-paged");
        // Full cycle returns to Partial after five presses.
        for _ in 0..4 {
            update(&mut app, key(KeyCode::Char('s')));
        }
        assert_eq!(app.sort, SortKey::Partial, "the cycle wraps around");
    }

    #[test]
    fn sort_on_a_keys_frame_reissues_first_page() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Enter)); // Keys{model}, frame 1
        let cmds = update(&mut app, key(KeyCode::Char('s')));
        assert_eq!(app.sort, SortKey::Bytes);
        assert_eq!(
            cmds,
            vec![Cmd::ExpandPrefix {
                prefix: "model".to_string(),
                cursor: None,
                frame_id: 1,
            }],
            "cycling sort re-issues the level's first page under the new order"
        );
    }

    #[test]
    fn peek_on_a_tensor_detail_emits_peek_for_the_selected_shard() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Enter)); // into "model" prefix (Keys)
        // Simulate opening a key detail directly via a command for determinism.
        app.mode = Mode::Command {
            buffer: String::new(),
        };
        type_str(&mut app, "key model.layers.0.attn.wq.weight");
        update(&mut app, key(KeyCode::Enter));
        let detail_id = app.stack.last().expect("detail frame").id;
        let entry = KeyEntry {
            key: "model.layers.0.attn.wq.weight".to_string(),
            object_type: ObjectType::TensorSlice,
            dtype: Some("float32".to_string()),
            global_shape: Some(vec![4096, 4096]),
            fully_committed: true,
            mesh_shape: Some(vec![2, 2]),
            shards: vec![Shard {
                volume_id: "vol-0".to_string(),
                coordinates: vec![0, 0],
                offsets: vec![0, 0],
                local_shape: vec![2048, 2048],
            }],
        };
        update(
            &mut app,
            Msg::KeyLoaded {
                frame_id: detail_id,
                resp: Ok(entry),
            },
        );
        let cmds = update(&mut app, key(KeyCode::Char('p')));
        assert_eq!(
            cmds,
            vec![Cmd::Peek {
                key: "model.layers.0.attn.wq.weight".to_string(),
                coordinates: Some(vec![0, 0]),
                frame_id: detail_id,
            }],
            "peek targets the selected shard's coordinates"
        );
    }

    #[test]
    fn peek_on_an_object_is_refused() {
        let mut app = test_app();
        app.mode = Mode::Command {
            buffer: String::new(),
        };
        type_str(&mut app, "key metadata.config");
        update(&mut app, key(KeyCode::Enter));
        let detail_id = app.stack.last().expect("detail frame").id;
        let entry = KeyEntry {
            key: "metadata.config".to_string(),
            object_type: ObjectType::Object,
            dtype: None,
            global_shape: None,
            fully_committed: true,
            mesh_shape: None,
            shards: vec![],
        };
        update(
            &mut app,
            Msg::KeyLoaded {
                frame_id: detail_id,
                resp: Ok(entry),
            },
        );
        let cmds = update(&mut app, key(KeyCode::Char('p')));
        assert!(cmds.is_empty(), "an OBJECT has no tensor data to peek");
        assert!(app.status.as_deref().unwrap_or_default().contains("OBJECT"));
    }

    #[test]
    fn peek_result_lands_on_the_detail_frame() {
        let mut app = test_app();
        app.mode = Mode::Command {
            buffer: String::new(),
        };
        type_str(&mut app, "peek model.layers.0.attn.wq.weight");
        update(&mut app, key(KeyCode::Enter));
        let detail_id = app.stack.last().expect("detail frame").id;
        let peek = PeekResult {
            dtype: "float32".to_string(),
            shape: vec![2048, 2048],
            min: -0.4,
            max: 0.4,
            mean: 0.0,
            l2_norm: 128.0,
            head: vec![0.1, -0.2],
        };
        update(
            &mut app,
            Msg::PeekLoaded {
                frame_id: detail_id,
                resp: Ok(peek),
            },
        );
        match &app.stack.last().expect("detail").kind {
            FrameKind::Detail { peek, .. } => {
                assert!(peek.is_some(), "peek stats stored on detail")
            }
            _ => panic!("expected a Detail frame"),
        }
    }

    #[test]
    fn help_toggles_and_any_key_dismisses() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char('?')));
        assert!(matches!(app.mode, Mode::Help), "? opens the overlay");
        // Any key dismisses without acting on the browse keymap.
        update(&mut app, key(KeyCode::Char('j')));
        assert!(matches!(app.mode, Mode::Browse), "any key dismisses help");
    }

    #[test]
    fn typing_q_in_filter_does_not_quit() {
        let mut app = test_app();
        update(&mut app, key(KeyCode::Char('/')));
        type_str(&mut app, "q");
        assert!(!app.should_quit, "q is text while the editor is open");
        match &app.mode {
            Mode::Filter { buffer } => assert_eq!(buffer, "q"),
            _ => panic!("expected filter mode"),
        }
    }

    #[test]
    fn middle_elide_keeps_the_distinguishing_tail() {
        let elided = middle_elide("model.layers.0.attn.wq.weight", 16);
        assert!(elided.contains('…'), "long keys are middle-elided");
        assert!(
            elided.ends_with("weight"),
            "the distinguishing tail is kept"
        );
        assert!(elided.chars().count() <= 16, "respects the width budget");
        // Short strings pass through untouched.
        assert_eq!(middle_elide("wq.weight", 40), "wq.weight");
    }

    #[test]
    fn detail_entry_lands_on_key_loaded() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        update(&mut app, key(KeyCode::Down));
        update(&mut app, key(KeyCode::Enter)); // Detail frame 1
        let entry = KeyEntry {
            key: "metadata.config".to_string(),
            object_type: ObjectType::Object,
            dtype: None,
            global_shape: None,
            fully_committed: true,
            mesh_shape: None,
            shards: vec![],
        };
        update(
            &mut app,
            Msg::KeyLoaded {
                frame_id: 1,
                resp: Ok(entry),
            },
        );
        match &app.stack[1].kind {
            FrameKind::Detail { entry, .. } => {
                assert!(entry.is_some(), "key detail arrived");
            }
            _ => panic!("expected a Detail frame"),
        }
    }

    #[test]
    fn breadcrumb_tracks_the_stack() {
        let mut app = test_app();
        update(&mut app, Msg::SummaryLoaded(Ok(sample_summary())));
        assert_eq!(app.breadcrumb(), "health");
        update(&mut app, key(KeyCode::Enter)); // into model
        assert_eq!(app.breadcrumb(), "health ▸ model");
    }

    #[test]
    fn helpers_format_as_expected() {
        assert_eq!(thousands(1_048_576), "1,048,576");
        assert_eq!(thousands(5), "5");
        assert_eq!(last_segment("model.layers.0"), "0");
        assert_eq!(middle_elide("short", 40), "short");
        let elided = middle_elide("model.layers.0.attn.wq.weight", 12);
        assert!(elided.contains('…') && elided.len() <= 14);
        assert_eq!(human_bytes(0.0), "0 B");
        assert_eq!(human_bytes(512.0), "512 B");
        assert_eq!(human_bytes(1536.0), "1.5 KB");
    }
}
