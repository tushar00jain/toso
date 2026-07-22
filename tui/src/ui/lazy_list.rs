//! A windowed, cursor-paginated list (SPEC §6.3).
//!
//! `LazyList<T>` holds only the pages loaded so far plus a `next_cursor` marking
//! whether more exist. It never materializes a whole level: the widget draws the
//! visible slice (`window`) and sizes its scrollbar from `total_for_scrollbar`.
//! Paging is **manual**: when more pages exist the list carries a trailing
//! "load more" row (`has_trailing`/`on_more_row`) the user activates with Enter
//! to fetch the next page. An in-flight page is signalled by `pending` so the
//! view draws a spinner in that trailing row instead.

pub(crate) struct LazyList<T> {
    items: Vec<T>,
    /// `Some` => more pages exist behind this cursor; `None` => fully loaded.
    next_cursor: Option<String>,
    /// True once at least one page (even empty) has arrived.
    loaded: bool,
    /// True while a page request is outstanding (drives the spinner row).
    pending: bool,
    /// Selected row, an index into `items`.
    cursor: usize,
}

impl<T> LazyList<T> {
    pub(crate) fn new() -> Self {
        Self {
            items: Vec::new(),
            next_cursor: None,
            loaded: false,
            pending: false,
            cursor: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.pending
    }

    pub(crate) fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub(crate) fn has_more(&self) -> bool {
        self.next_cursor.is_some()
    }

    /// Whether a trailing row is shown below the loaded items: a spinner while a
    /// page is in flight, or a selectable "load more" row when another page can
    /// be fetched.
    pub(crate) fn has_trailing(&self) -> bool {
        self.pending || self.next_cursor.is_some()
    }

    /// Total displayed rows: the loaded items plus the trailing row (if any).
    /// Doubles as the cursor's upper bound so it can land on the trailing row.
    pub(crate) fn row_count(&self) -> usize {
        self.items.len() + usize::from(self.has_trailing())
    }

    /// True when the cursor rests on the trailing row (`items.len()`), i.e. the
    /// spinner or the "load more" affordance rather than an item.
    pub(crate) fn on_more_row(&self) -> bool {
        self.has_trailing() && self.cursor == self.items.len()
    }

    pub(crate) fn next_cursor(&self) -> Option<String> {
        self.next_cursor.clone()
    }

    pub(crate) fn selected(&self) -> Option<&T> {
        self.items.get(self.cursor)
    }

    pub(crate) fn selected_index(&self) -> usize {
        self.cursor
    }

    /// Mark a page request as dispatched so the view draws a spinner row and the
    /// reducer won't request the same page twice.
    pub(crate) fn begin_page(&mut self) {
        self.pending = true;
    }

    /// A page arrived: append it and record whether more remain.
    pub(crate) fn append_page(&mut self, mut items: Vec<T>, next_cursor: Option<String>) {
        self.items.append(&mut items);
        self.next_cursor = next_cursor;
        self.pending = false;
        self.loaded = true;
        self.clamp_cursor();
    }

    /// A page request failed: stop the spinner but keep what we have.
    pub(crate) fn fail_page(&mut self) {
        self.pending = false;
        self.loaded = true;
    }

    pub(crate) fn select_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub(crate) fn select_down(&mut self) {
        // `row_count` includes the trailing "load more" row, so the cursor can
        // reach it (one past the last item).
        if self.cursor + 1 < self.row_count() {
            self.cursor += 1;
        }
    }

    pub(crate) fn select_top(&mut self) {
        self.cursor = 0;
    }

    pub(crate) fn select_bottom(&mut self) {
        self.cursor = self.row_count().saturating_sub(1);
    }

    pub(crate) fn page_down(&mut self, height: usize) {
        self.cursor = (self.cursor + height.max(1)).min(self.row_count().saturating_sub(1));
    }

    pub(crate) fn page_up(&mut self, height: usize) {
        self.cursor = self.cursor.saturating_sub(height.max(1));
    }

    pub(crate) fn set_cursor(&mut self, cursor: usize) {
        self.cursor = cursor;
        self.clamp_cursor();
    }

    fn clamp_cursor(&mut self) {
        let rows = self.row_count();
        if rows == 0 {
            self.cursor = 0;
        } else if self.cursor >= rows {
            self.cursor = rows - 1;
        }
    }

    /// Top row index of the window that keeps the cursor visible in `height` rows.
    pub(crate) fn view_offset(&self, height: usize) -> usize {
        if height == 0 || self.cursor < height {
            0
        } else {
            self.cursor + 1 - height
        }
    }

    /// The visible slice for a `height`-row viewport, with its top offset.
    pub(crate) fn window(&self, height: usize) -> (usize, &[T]) {
        let offset = self.view_offset(height);
        let end = (offset + height).min(self.items.len());
        (offset, &self.items[offset..end])
    }

    /// Row count the scrollbar spans: loaded rows plus one for the unloaded tail.
    pub(crate) fn total_for_scrollbar(&self) -> usize {
        self.items.len() + usize::from(self.next_cursor.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded(items: Vec<u32>, more: bool) -> LazyList<u32> {
        let mut l = LazyList::new();
        l.append_page(items, more.then(|| "c1".to_string()));
        l
    }

    #[test]
    fn new_list_is_empty_and_unloaded() {
        let l: LazyList<u32> = LazyList::new();
        assert!(!l.is_loaded());
        assert_eq!(l.row_count(), 0, "no items and nothing in flight");
        assert!(!l.on_more_row());
    }

    #[test]
    fn pending_adds_a_trailing_spinner_row() {
        let mut l: LazyList<u32> = LazyList::new();
        l.begin_page();
        assert!(l.is_pending());
        assert!(l.has_trailing(), "an in-flight page shows a trailing row");
        assert_eq!(l.row_count(), 1, "the trailing spinner occupies a row");
    }

    #[test]
    fn append_page_sets_items_and_more() {
        let l = loaded(vec![1, 2, 3], true);
        assert_eq!(l.len(), 3);
        assert!(l.has_more());
        assert!(l.is_loaded());
        assert!(!l.is_pending());
        // scrollbar spans loaded rows plus one for the unloaded tail.
        assert_eq!(l.total_for_scrollbar(), 4);
    }

    #[test]
    fn fully_loaded_list_has_no_more_row() {
        let mut l = loaded(vec![1, 2, 3], false);
        l.select_bottom();
        assert!(!l.has_trailing(), "no next_cursor => no trailing row");
        assert_eq!(l.row_count(), 3, "just the three items");
        assert!(!l.on_more_row());
        assert_eq!(
            l.total_for_scrollbar(),
            3,
            "no tail placeholder when complete"
        );
    }

    #[test]
    fn cursor_can_reach_the_trailing_more_row() {
        let mut l = loaded((0..10).collect(), true);
        assert_eq!(l.row_count(), 11, "10 items + one load-more row");
        l.select_bottom();
        assert_eq!(l.selected_index(), 10, "bottom is the load-more row");
        assert!(l.on_more_row(), "cursor rests on the load-more row");
        assert!(l.selected().is_none(), "the more row is not an item");
    }

    #[test]
    fn cursor_movement_clamps() {
        let mut l = loaded(vec![10, 20, 30], false);
        l.select_up();
        assert_eq!(l.selected_index(), 0, "can't move above the top");
        l.select_down();
        l.select_down();
        l.select_down();
        assert_eq!(l.selected_index(), 2, "can't move past the last row");
        assert_eq!(l.selected(), Some(&30));
        l.select_top();
        assert_eq!(l.selected_index(), 0);
        l.select_bottom();
        assert_eq!(l.selected_index(), 2);
    }

    #[test]
    fn paging_moves_by_height_and_clamps() {
        let mut l = loaded((0..20).collect(), false);
        l.page_down(5);
        assert_eq!(l.selected_index(), 5);
        l.page_down(100);
        assert_eq!(l.selected_index(), 19, "page down clamps to the last row");
        l.page_up(4);
        assert_eq!(l.selected_index(), 15);
        l.page_up(100);
        assert_eq!(l.selected_index(), 0, "page up clamps to the top");
    }

    #[test]
    fn window_follows_cursor() {
        let l = {
            let mut l = loaded((0..100).collect(), false);
            l.set_cursor(0);
            l
        };
        // At the top the window starts at 0.
        let (off, slice) = l.window(10);
        assert_eq!(off, 0);
        assert_eq!(slice.len(), 10);
        assert_eq!(slice[0], 0);

        // Cursor deep in the list bottom-anchors the window so the cursor shows.
        let mut l = l;
        l.set_cursor(50);
        let (off, slice) = l.window(10);
        assert_eq!(off, 41, "offset = cursor + 1 - height");
        assert_eq!(slice.len(), 10);
        assert_eq!(*slice.last().expect("non-empty window"), 50);
    }

    #[test]
    fn window_shorter_than_viewport() {
        let l = loaded(vec![1, 2, 3], false);
        let (off, slice) = l.window(10);
        assert_eq!(off, 0);
        assert_eq!(slice.len(), 3, "window never exceeds the loaded rows");
    }

    #[test]
    fn append_second_page_keeps_cursor() {
        let mut l = loaded((0..5).collect(), true);
        l.set_cursor(4);
        l.begin_page();
        l.append_page((5..10).collect(), None);
        assert_eq!(l.len(), 10);
        assert!(!l.has_more());
        assert_eq!(l.selected_index(), 4, "appending keeps the selection");
    }
}
