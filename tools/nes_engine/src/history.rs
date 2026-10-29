//! Engine-side edit history.
//!
//! The editor sends whole-buffer snapshots (on every prediction request and on
//! cheap observe calls). Consecutive snapshots of the same region inside a short
//! pause are coalesced into one logical edit, so the model sees "renamed foo to
//! bar" rather than ten single-keystroke diffs. Hunks carry up to two context
//! lines on each side; the prompt builder decides how many to show.

use similar::{Algorithm, DiffOp, capture_diff_slices, capture_diff_slices_deadline};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Edits closer than this many lines, without a longer pause, are one edit.
const PROXIMITY_LINES: usize = 8;
const PAUSE: Duration = Duration::from_millis(2500);
const MAX_FINALIZED: usize = 24;
const MAX_AGE: Duration = Duration::from_secs(15 * 60);
const ORIGINAL_REFRESH_AGE: Duration = Duration::from_secs(15 * 60);
const CONTEXT_LINES: usize = 2;
/// With session bases, a live group is not split by pauses, only by edits
/// elsewhere, a file change, or this age.
const SESSION_MAX_AGE: Duration = Duration::from_secs(60);
/// The original snapshot is advanced past edits the history no longer shows
/// once they change more than this many lines (`refresh_original_at`).
const REFRESH_DRIFT_LINES: usize = 12;
/// ... or once the snapshot has fewer than half the current lines and the
/// advance adds at least this many.
const REFRESH_MIN_LINES: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffHunk {
    pub path: PathBuf,
    pub display: String,
    /// 0-based line of the first `ctx_before` line in the old text.
    pub old_start: usize,
    /// 0-based line of the first `ctx_before` line in the new text.
    pub new_start: usize,
    pub ctx_before: Vec<String>,
    /// Old lines from first to last change, including interior equal lines.
    pub old: Vec<String>,
    pub new: Vec<String>,
    pub ctx_after: Vec<String>,
    /// Edit group the hunk came from (hunks of one group share line
    /// coordinates: old in the group's base, new in its latest text).
    pub group: u64,
}

impl DiffHunk {
    /// Old and new code with `ctx` context lines, joined by newlines (no
    /// trailing newline), mirroring Sweep's `extract_diff_parts`.
    pub fn parts(&self, ctx: usize) -> (String, String) {
        let before = &self.ctx_before[self.ctx_before.len().saturating_sub(ctx)..];
        let after = &self.ctx_after[..ctx.min(self.ctx_after.len())];
        let join = |body: &[String]| {
            before
                .iter()
                .chain(body)
                .chain(after)
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join("\n")
        };
        (join(&self.old), join(&self.new))
    }

    /// 1-based first and last line numbers of the shown new region.
    pub fn shown_new_range(&self, ctx: usize) -> (usize, usize) {
        let before = ctx.min(self.ctx_before.len());
        let after = ctx.min(self.ctx_after.len());
        let start = self.new_start + (self.ctx_before.len() - before) + 1;
        let len = before + self.new.len() + after;
        (start, start + len.max(1) - 1)
    }

    /// Old lines that were removed (Sweep: present in input, absent in output),
    /// with their 1-based old line number.
    pub fn deleted_lines(&self) -> Vec<(String, usize)> {
        let first_old = self.old_start + self.ctx_before.len() + 1;
        self.old
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty() && !self.new.contains(l))
            .map(|(i, l)| (l.trim().to_string(), first_old + i))
            .collect()
    }

    pub fn is_whitespace_only(&self) -> bool {
        let (old, new) = self.parts(0);
        old.trim() == new.trim()
    }
}

#[derive(Clone)]
struct FileState {
    content: String,
    original: String,
    original_at: Instant,
    cursor_line: usize,
    last_visit: Instant,
    display: String,
}

#[derive(Clone)]
struct Group {
    id: u64,
    path: PathBuf,
    display: String,
    base: Arc<str>,
    latest: String,
    started: Instant,
    last_ts: Instant,
    lo: usize,
    hi: usize,
    cached: Option<Vec<DiffHunk>>,
}

#[derive(Default, Clone)]
pub struct Tracker {
    files: HashMap<PathBuf, FileState>,
    finalized: VecDeque<(Instant, DiffHunk)>,
    group: Option<Group>,
    /// Text before each recent edit group, oldest first: (group, path, base).
    bases: VecDeque<(u64, PathBuf, Arc<str>)>,
    next_group: u64,
    /// Group edits by place only (pauses do not split them), see
    /// `SESSION_MAX_AGE`.
    session_base: bool,
}

pub struct Visit {
    pub path: PathBuf,
    pub display: String,
    pub content: String,
    pub cursor_line: usize,
}

impl Tracker {
    pub fn new(session_base: bool) -> Self {
        Self { session_base, ..Default::default() }
    }

    /// Record a snapshot. Returns true when the content changed.
    pub fn observe(
        &mut self,
        path: &Path,
        display: &str,
        content: &str,
        cursor_line: usize,
        saved: bool,
    ) -> bool {
        self.observe_at(path, display, content, cursor_line, saved, Instant::now())
    }

    pub fn observe_at(
        &mut self,
        path: &Path,
        display: &str,
        content: &str,
        cursor_line: usize,
        saved: bool,
        now: Instant,
    ) -> bool {
        let Some(state) = self.files.get_mut(path) else {
            self.files.insert(
                path.to_path_buf(),
                FileState {
                    content: content.to_string(),
                    original: content.to_string(),
                    original_at: now,
                    cursor_line,
                    last_visit: now,
                    display: display.to_string(),
                },
            );
            return false;
        };
        state.cursor_line = cursor_line;
        state.last_visit = now;
        state.display = display.to_string();
        if saved && now.duration_since(state.original_at) > ORIGINAL_REFRESH_AGE {
            state.original = content.to_string();
            state.original_at = now;
        }
        if state.content == content {
            return false;
        }
        let (lo, hi) = changed_region(&state.content, content);
        let previous = std::mem::replace(&mut state.content, content.to_string());
        let extend = match &self.group {
            Some(g) => {
                let recent = if self.session_base {
                    now.duration_since(g.started) < SESSION_MAX_AGE
                } else {
                    now.duration_since(g.last_ts) < PAUSE
                };
                g.path == path
                    && recent
                    && lo <= g.hi + PROXIMITY_LINES
                    && hi + PROXIMITY_LINES >= g.lo
            }
            None => false,
        };
        if extend {
            let g = self.group.as_mut().unwrap();
            g.latest = content.to_string();
            g.last_ts = now;
            g.lo = g.lo.min(lo);
            g.hi = g.hi.max(hi);
            g.cached = None;
        } else {
            self.finalize_group();
            self.next_group += 1;
            let base: Arc<str> = Arc::from(previous);
            self.bases.push_back((self.next_group, path.to_path_buf(), Arc::clone(&base)));
            while self.bases.len() > MAX_FINALIZED + 1 {
                self.bases.pop_front();
            }
            self.group = Some(Group {
                id: self.next_group,
                path: path.to_path_buf(),
                display: display.to_string(),
                base,
                latest: content.to_string(),
                started: now,
                last_ts: now,
                lo,
                hi,
                cached: None,
            });
        }
        true
    }

    fn finalize_group(&mut self) {
        if let Some(mut g) = self.group.take() {
            let hunks = g.cached.take().unwrap_or_else(|| g.hunks());
            for h in hunks {
                self.finalized.push_back((g.last_ts, h));
            }
            while self.finalized.len() > MAX_FINALIZED {
                self.finalized.pop_front();
            }
        }
    }

    /// Recent hunks, oldest first, after age decay, inverse-pair collapsing,
    /// deduplication and whitespace filtering.
    pub fn recent_hunks(&mut self, limit: usize) -> Vec<DiffHunk> {
        self.recent_hunks_at(limit, Instant::now())
    }

    pub fn recent_hunks_at(&mut self, limit: usize, now: Instant) -> Vec<DiffHunk> {
        let mut out: Vec<DiffHunk> = self
            .finalized
            .iter()
            .filter(|(ts, _)| now.duration_since(*ts) <= MAX_AGE)
            .map(|(_, h)| h.clone())
            .collect();
        out.extend(self.live_hunks());
        let out = collapse_inverse(out);
        let mut out = dedupe(out);
        out.retain(|h| !h.is_whitespace_only());
        let skip = out.len().saturating_sub(limit);
        out.split_off(skip)
    }

    /// Up to `limit` finalized hunks (filtered as in `recent_hunks_at`)
    /// followed by the live group's raw hunks, oldest first, and the number
    /// of live hunks. The live group changes on every keystroke; filtering it
    /// together with the finalized hunks made hunks vanish and reappear
    /// while typing. The caller hides hunks shown by the window and caps the
    /// rest, so `limit` should leave room for hidden ones.
    pub fn recent_split_at(&mut self, limit: usize, now: Instant) -> (Vec<DiffHunk>, usize) {
        let finalized: Vec<DiffHunk> = self
            .finalized
            .iter()
            .filter(|(ts, _)| now.duration_since(*ts) <= MAX_AGE)
            .map(|(_, h)| h.clone())
            .collect();
        let mut out = dedupe(collapse_inverse(finalized));
        out.retain(|h| !h.is_whitespace_only());
        let mut out = out.split_off(out.len().saturating_sub(limit));
        let live = self.live_hunks();
        let n = live.len();
        out.extend(live);
        (out, n)
    }

    fn live_hunks(&mut self) -> Vec<DiffHunk> {
        match self.group.as_mut() {
            Some(g) => {
                if g.cached.is_none() {
                    g.cached = Some(g.hunks());
                }
                g.cached.clone().unwrap()
            }
            None => Vec::new(),
        }
    }

    /// Text of `path` before its live edit group, when the group is in it.
    pub fn session_base(&self, path: &Path) -> Option<&str> {
        self.group.as_ref().filter(|g| g.path == path).map(|g| &*g.base)
    }

    pub fn original(&self, path: &Path) -> Option<&str> {
        self.files.get(path).map(|s| s.original.as_str())
    }

    /// Text of `path` just before edit group `from`, the oldest group a
    /// prompt shows: every edit the prompt leaves out applied, none it shows.
    /// The current text when `from` is None (nothing shown) or newer than
    /// every group of the file.
    pub fn state_before_group(&self, path: &Path, from: Option<u64>) -> Option<&str> {
        let state = self.files.get(path)?;
        let base = from.and_then(|from| self.bases.iter().find(|(id, p, _)| *id >= from && p == path));
        Some(base.map_or(state.content.as_str(), |(_, _, b)| &**b))
    }

    /// Advance the original snapshot of `path` to `state_before_group` when
    /// the snapshot is clearly stale: the edits between them change more
    /// than `REFRESH_DRIFT_LINES` lines, or the snapshot has fewer than half
    /// the current lines. The original opens the prompt, so a new one
    /// re-evaluates nearly all of it; small drift is left alone. Returns
    /// true when the snapshot changed.
    pub fn refresh_original_at(&mut self, path: &Path, from: Option<u64>, now: Instant) -> bool {
        let Some(candidate) = self.state_before_group(path, from).map(str::to_string) else {
            return false;
        };
        let state = self.files.get_mut(path).unwrap();
        if !original_is_stale(&state.original, &candidate, &state.content) {
            return false;
        }
        state.original = candidate;
        state.original_at = now;
        true
    }

    /// Byte ranges of `content` (the current text of `path`) typed during the
    /// live edit group: characters inserted or replaced since its base.
    /// Empty when the live group is in another file or `content` is not its
    /// latest text.
    pub fn live_typed(&mut self, path: &Path, content: &str) -> Vec<(usize, usize)> {
        if !self.group.as_ref().is_some_and(|g| g.path == path && g.latest == content) {
            return Vec::new();
        }
        let starts: Vec<usize> = std::iter::once(0)
            .chain(content.split_inclusive('\n').scan(0, |at, l| {
                *at += l.len();
                Some(*at)
            }))
            .collect();
        let mut out = Vec::new();
        for h in self.live_hunks() {
            let first = h.new_start + h.ctx_before.len();
            let Some(&base) = starts.get(first) else { continue };
            let old = h.old.join("\n");
            let new = h.new.join("\n");
            out.extend(inserted_ranges(&old, &new).into_iter().map(|(a, b)| (base + a, base + b)));
        }
        out
    }

    /// Most recently visited files other than `exclude`, newest first.
    pub fn recent_visits(&self, exclude: &Path, limit: usize) -> Vec<Visit> {
        let mut v: Vec<_> = self
            .files
            .iter()
            .filter(|(p, _)| p.as_path() != exclude)
            .collect();
        v.sort_by(|a, b| b.1.last_visit.cmp(&a.1.last_visit));
        v.into_iter()
            .take(limit)
            .map(|(p, s)| Visit {
                path: p.clone(),
                display: s.display.clone(),
                content: s.content.clone(),
                cursor_line: s.cursor_line,
            })
            .collect()
    }

    /// Current snapshot of every tracked file (open buffers).
    pub fn open_contents(&self) -> Vec<(PathBuf, String)> {
        self.files
            .iter()
            .map(|(p, s)| (p.clone(), s.content.clone()))
            .collect()
    }
}

impl Group {
    fn hunks(&self) -> Vec<DiffHunk> {
        let mut hunks = diff_hunks(&self.path, &self.display, &self.base, &self.latest);
        for h in &mut hunks {
            h.group = self.id;
        }
        hunks
    }
}

/// Line of `from` (0-based) mapped to the corresponding line of `to`:
/// unchanged lines map exactly, changed lines map to the start of the
/// region that replaced them.
pub fn map_line(from: &str, to: &str, line: usize) -> usize {
    if from == to {
        return line;
    }
    let a = split_lines(from);
    let b = split_lines(to);
    let deadline = Instant::now() + Duration::from_millis(20);
    for op in capture_diff_slices_deadline(Algorithm::Myers, &a, &b, Some(deadline)) {
        let (old, new) = (op.old_range(), op.new_range());
        if old.contains(&line) {
            return match op {
                DiffOp::Equal { .. } => new.start + (line - old.start),
                _ => new.start,
            };
        }
    }
    // Past the last line of `from` (the empty line after a final newline).
    b.len() + line.saturating_sub(a.len())
}

fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut v: Vec<&str> = text.split('\n').collect();
    if text.ends_with('\n') {
        v.pop();
    }
    v
}

/// Lines changed between `a` and `b` (per change, the larger side).
fn changed_lines(a: &str, b: &str) -> usize {
    let (a, b) = (split_lines(a), split_lines(b));
    let deadline = Instant::now() + Duration::from_millis(20);
    capture_diff_slices_deadline(Algorithm::Myers, &a, &b, Some(deadline))
        .iter()
        .filter(|op| !matches!(op, DiffOp::Equal { .. }))
        .map(|op| op.old_range().len().max(op.new_range().len()))
        .sum()
}

/// See `Tracker::refresh_original_at`.
fn original_is_stale(original: &str, candidate: &str, current: &str) -> bool {
    if original == candidate {
        return false;
    }
    let drift = changed_lines(original, candidate);
    let (o, k, c) = (split_lines(original).len(), split_lines(candidate).len(), split_lines(current).len());
    drift > REFRESH_DRIFT_LINES || (o * 2 < c && k > o && drift >= REFRESH_MIN_LINES)
}

/// Byte ranges of `new` holding characters that are not in `old` (inserted
/// or replacing old ones), by a character diff.
fn inserted_ranges(old: &str, new: &str) -> Vec<(usize, usize)> {
    let a: Vec<char> = old.chars().collect();
    let b: Vec<char> = new.chars().collect();
    let mut at: Vec<usize> = new.char_indices().map(|(i, _)| i).collect();
    at.push(new.len());
    let deadline = Instant::now() + Duration::from_millis(20);
    let mut out: Vec<(usize, usize)> = Vec::new();
    for op in capture_diff_slices_deadline(Algorithm::Myers, &a, &b, Some(deadline)) {
        let r = op.new_range();
        if matches!(op, DiffOp::Equal { .. }) || r.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.1 == at[r.start] => last.1 = at[r.end],
            _ => out.push((at[r.start], at[r.end])),
        }
    }
    out
}

/// First and last changed line (0-based, in the new text).
fn changed_region(old: &str, new: &str) -> (usize, usize) {
    let a = split_lines(old);
    let b = split_lines(new);
    let mut start = 0;
    while start < a.len() && start < b.len() && a[start] == b[start] {
        start += 1;
    }
    let (mut ea, mut eb) = (a.len(), b.len());
    while ea > start && eb > start && a[ea - 1] == b[eb - 1] {
        ea -= 1;
        eb -= 1;
    }
    (start, eb.max(start + 1) - 1)
}

/// Unified-diff style hunks with up to two context lines on each side.
pub fn diff_hunks(path: &Path, display: &str, old: &str, new: &str) -> Vec<DiffHunk> {
    let a = split_lines(old);
    let b = split_lines(new);
    let ops = capture_diff_slices(Algorithm::Myers, &a, &b);
    // Merge change ops separated by at most 2 * CONTEXT_LINES equal lines.
    let mut groups: Vec<(usize, usize, usize, usize)> = Vec::new(); // old lo..hi, new lo..hi
    for op in &ops {
        let (ol, oh, nl, nh) = match *op {
            DiffOp::Equal { .. } => continue,
            DiffOp::Delete { old_index, old_len, new_index } => {
                (old_index, old_index + old_len, new_index, new_index)
            }
            DiffOp::Insert { old_index, new_index, new_len } => {
                (old_index, old_index, new_index, new_index + new_len)
            }
            DiffOp::Replace { old_index, old_len, new_index, new_len } => {
                (old_index, old_index + old_len, new_index, new_index + new_len)
            }
        };
        if let Some(last) = groups.last_mut() {
            if ol <= last.1 + 2 * CONTEXT_LINES {
                last.1 = last.1.max(oh);
                last.3 = last.3.max(nh);
                continue;
            }
        }
        groups.push((ol, oh, nl, nh));
    }
    groups
        .into_iter()
        .map(|(ol, oh, nl, nh)| {
            let before = CONTEXT_LINES.min(ol).min(nl);
            let after = CONTEXT_LINES.min(a.len() - oh).min(b.len() - nh);
            DiffHunk {
                path: path.to_path_buf(),
                display: display.to_string(),
                old_start: ol - before,
                new_start: nl - before,
                ctx_before: b[nl - before..nl].iter().map(|s| s.to_string()).collect(),
                old: a[ol..oh].iter().map(|s| s.to_string()).collect(),
                new: b[nl..nh].iter().map(|s| s.to_string()).collect(),
                ctx_after: b[nh..nh + after].iter().map(|s| s.to_string()).collect(),
                group: 0,
            }
        })
        .collect()
}

fn collapse_inverse(entries: Vec<DiffHunk>) -> Vec<DiffHunk> {
    let mut removed = vec![false; entries.len()];
    for i in 0..entries.len() {
        if removed[i] {
            continue;
        }
        for j in i + 1..entries.len() {
            if removed[j] {
                continue;
            }
            let (ao, an) = entries[i].parts(0);
            let (bo, bn) = entries[j].parts(0);
            if entries[i].path == entries[j].path && an.trim() == bo.trim() && bn.trim() == ao.trim() {
                removed[i] = true;
                removed[j] = true;
                break;
            }
        }
    }
    entries
        .into_iter()
        .zip(removed)
        .filter(|(_, r)| !r)
        .map(|(e, _)| e)
        .collect()
}

fn dedupe(entries: Vec<DiffHunk>) -> Vec<DiffHunk> {
    let keys: Vec<(PathBuf, String, String)> = entries
        .iter()
        .map(|e| {
            let (o, n) = e.parts(0);
            (e.path.clone(), o.trim().to_string(), n.trim().to_string())
        })
        .collect();
    entries
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !keys[i + 1..].contains(&keys[*i]))
        .map(|(_, e)| e)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hunk_parts_and_ranges() {
        let old = "a\nb\nc\nd\ne\n";
        let new = "a\nb\nC\nd\ne\n";
        let h = diff_hunks(Path::new("f"), "f", old, new);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].parts(0), ("c".into(), "C".into()));
        assert_eq!(h[0].parts(1), ("b\nc\nd".into(), "b\nC\nd".into()));
        assert_eq!(h[0].shown_new_range(1), (2, 4));
        assert_eq!(h[0].deleted_lines(), vec![("c".to_string(), 3)]);
    }

    #[test]
    fn coalesces_nearby_quick_edits() {
        let mut t = Tracker::default();
        let p = Path::new("/r/f.rs");
        let now = Instant::now();
        t.observe_at(p, "f.rs", "fn a() {}\n", 0, false, now);
        t.observe_at(p, "f.rs", "fn ab() {}\n", 0, false, now + Duration::from_millis(300));
        t.observe_at(p, "f.rs", "fn abc() {}\n", 0, false, now + Duration::from_millis(600));
        let h = t.recent_hunks_at(6, now + Duration::from_millis(700));
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].parts(0), ("fn a() {}".into(), "fn abc() {}".into()));
    }

    #[test]
    fn separate_after_pause_and_inverse_collapse() {
        let mut t = Tracker::default();
        let p = Path::new("/r/f.rs");
        let now = Instant::now();
        t.observe_at(p, "f.rs", "x\n", 0, false, now);
        t.observe_at(p, "f.rs", "y\n", 0, false, now + Duration::from_secs(1));
        t.observe_at(p, "f.rs", "x\n", 0, false, now + Duration::from_secs(10));
        assert!(t.recent_hunks_at(6, now + Duration::from_secs(11)).is_empty());
    }

    #[test]
    fn far_edits_are_separate() {
        let mut t = Tracker::default();
        let p = Path::new("/r/f.rs");
        let base: String = (0..40).map(|i| format!("l{i}\n")).collect();
        let now = Instant::now();
        t.observe_at(p, "f.rs", &base, 0, false, now);
        let e1 = base.replace("l1\n", "L1\n");
        t.observe_at(p, "f.rs", &e1, 1, false, now + Duration::from_millis(100));
        let e2 = e1.replace("l30\n", "L30\n");
        t.observe_at(p, "f.rs", &e2, 30, false, now + Duration::from_millis(200));
        let h = t.recent_hunks_at(6, now + Duration::from_millis(300));
        assert_eq!(h.len(), 2);
        assert_eq!(h[1].parts(0).1, "L30");
    }

    #[test]
    fn live_group_skips_filters() {
        let mut t = Tracker::default();
        let p = Path::new("/r/f.rs");
        let now = Instant::now();
        t.observe_at(p, "f.rs", "x\ny\n", 0, false, now);
        t.observe_at(p, "f.rs", "X\ny\n", 0, false, now + Duration::from_secs(1));
        // Undoing the edit while still typing: the filters would collapse the
        // pair and the hunks would reappear on the next keystroke.
        t.observe_at(p, "f.rs", "x\ny\n", 0, false, now + Duration::from_secs(10));
        assert!(t.recent_hunks_at(6, now + Duration::from_secs(11)).is_empty());
        let (h, live) = t.recent_split_at(6, now + Duration::from_secs(11));
        assert_eq!((h.len(), live), (2, 1));
        assert_eq!(h[1].parts(0), ("X".into(), "x".into()));
        assert_ne!(h[0].group, h[1].group);
        // A blank line being typed is kept too.
        t.observe_at(p, "f.rs", "x\n\ny\n", 1, false, now + Duration::from_secs(20));
        let (h, live) = t.recent_split_at(6, now + Duration::from_secs(21));
        assert_eq!(live, 1);
        assert!(h.last().unwrap().is_whitespace_only());
    }

    #[test]
    fn split_caps_finalized_only() {
        let mut t = Tracker::default();
        let p = Path::new("/r/f.rs");
        let base: String = (0..60).map(|i| format!("l{i}\n")).collect();
        let now = Instant::now();
        t.observe_at(p, "f.rs", &base, 0, false, now);
        let mut text = base.clone();
        for (k, line) in [1, 20, 40, 55].iter().enumerate() {
            text = text.replace(&format!("l{line}\n"), &format!("L{line}\n"));
            t.observe_at(p, "f.rs", &text, *line, false, now + Duration::from_secs(10 * (k as u64 + 1)));
        }
        let (h, live) = t.recent_split_at(2, now + Duration::from_secs(41));
        assert_eq!(live, 1);
        let new: Vec<String> = h.iter().map(|h| h.parts(0).1).collect();
        assert_eq!(new, ["L20", "L40", "L55"]);
    }

    #[test]
    fn session_base_ignores_pauses() {
        let p = Path::new("/r/f.rs");
        let now = Instant::now();
        let mut t = Tracker::new(true);
        t.observe_at(p, "f.rs", "a\nb\n", 0, false, now);
        t.observe_at(p, "f.rs", "A\nb\n", 0, false, now + Duration::from_secs(1));
        t.observe_at(p, "f.rs", "A\nB\n", 1, false, now + Duration::from_secs(20));
        assert_eq!(t.session_base(p), Some("a\nb\n"));
        assert_eq!(t.recent_split_at(6, now + Duration::from_secs(21)).1, 1);
        // The session is folded after its maximum age.
        t.observe_at(p, "f.rs", "A\nBB\n", 1, false, now + Duration::from_secs(70));
        assert_eq!(t.session_base(p), Some("A\nB\n"));
        assert!(t.session_base(Path::new("/r/g.rs")).is_none());
    }

    #[test]
    fn original_advances_past_hunks_no_longer_shown() {
        let p = Path::new("/r/d.yaml");
        let now = Instant::now();
        let at = |s: u64| now + Duration::from_secs(s);
        let mut t = Tracker::default();
        // A new, empty file: the snapshot is taken before anything is typed.
        t.observe_at(p, "d.yaml", "", 0, false, now);
        let block = "tasks:\n  db:\n    cmd: [postgres]\n    ready: yes\n";
        t.observe_at(p, "d.yaml", block, 3, false, at(1));
        // Nothing has dropped out of the history yet: the paste is shown.
        let (h, _) = t.recent_split_at(3, at(2));
        assert_eq!(t.state_before_group(p, Some(h[0].group)), Some(""));
        assert!(!t.refresh_original_at(p, Some(h[0].group), at(2)));
        let mut text = block.to_string();
        for (k, line) in ["  api:\n", "    cmd: [cargo]\n", "    deps: [db]\n"].iter().enumerate() {
            text.push_str(line);
            t.observe_at(p, "d.yaml", &text, 4 + k, false, at(10 * (k as u64 + 1)));
        }
        // Two finalized groups and the live one shown.
        let (h, _) = t.recent_split_at(2, at(40));
        assert_eq!(h.len(), 3);
        // Only the three newest groups are shown: the snapshot moves to the
        // text just before the oldest of them, the paste applied.
        assert_eq!(t.state_before_group(p, Some(h[0].group)), Some(block));
        assert!(t.refresh_original_at(p, Some(h[0].group), at(40)));
        assert_eq!(t.original(p), Some(block));
        // Small drift that no shown hunk covers is left alone (KV cache).
        text.push_str("  web:\n");
        t.observe_at(p, "d.yaml", &text, 7, false, at(50));
        let (h, _) = t.recent_split_at(2, at(51));
        assert!(!t.refresh_original_at(p, Some(h[0].group), at(51)));
        assert_eq!(t.original(p), Some(block));
        // Nothing shown at all: the current text.
        assert_eq!(t.state_before_group(p, None), Some(text.as_str()));
        assert!(t.state_before_group(Path::new("/r/other"), None).is_none());
    }

    #[test]
    fn large_drift_refreshes_a_full_snapshot() {
        let p = Path::new("/r/f.rs");
        let now = Instant::now();
        let base: String = (0..60).map(|i| format!("l{i}\n")).collect();
        let mut t = Tracker::default();
        t.observe_at(p, "f.rs", &base, 0, false, now);
        // Fourteen separate one-line edits, far apart in time.
        let mut text = base.clone();
        for k in 0..14u64 {
            text = text.replace(&format!("l{}\n", k * 4), &format!("L{}\n", k * 4));
            t.observe_at(p, "f.rs", &text, k as usize * 4, false, now + Duration::from_secs(10 * (k + 1)));
        }
        let later = now + Duration::from_secs(200);
        let (h, _) = t.recent_split_at(2, later);
        // Eleven dropped edits is not enough drift; the original is full size.
        assert!(!original_is_stale(&base, t.state_before_group(p, Some(h[0].group)).unwrap(), &text));
        text = text.replace("l57\n", "L57\n");
        t.observe_at(p, "f.rs", &text, 57, false, now + Duration::from_secs(300));
        text = text.replace("l59\n", "L59\n");
        t.observe_at(p, "f.rs", &text, 59, false, now + Duration::from_secs(310));
        let (h, _) = t.recent_split_at(2, now + Duration::from_secs(311));
        assert!(t.refresh_original_at(p, Some(h[0].group), now + Duration::from_secs(311)));
        let o = t.original(p).unwrap();
        assert!(o.contains("L48\n") && o.contains("l52\n") && !o.contains("L52\n"));
    }

    #[test]
    fn live_typed_ranges_cover_the_live_group_only() {
        let p = Path::new("/r/d.yaml");
        let now = Instant::now();
        let mut t = Tracker::default();
        let base = "tasks:\n  db:\n    cmd: x\n";
        t.observe_at(p, "d.yaml", base, 0, false, now);
        let typed = format!("{base}  web:\n");
        t.observe_at(p, "d.yaml", &typed, 3, false, now + Duration::from_millis(300));
        let r = t.live_typed(p, &typed);
        assert_eq!(r, vec![(base.len(), base.len() + "  web:".len())]);
        // A replaced word inside a line.
        let renamed = typed.replace("cmd: x", "cmd: yz");
        t.observe_at(p, "d.yaml", &renamed, 2, false, now + Duration::from_millis(600));
        let r = t.live_typed(p, &renamed);
        // "x" became "yz", then the typed line follows.
        let at = renamed.find("yz").unwrap();
        assert_eq!(r, vec![(at, renamed.len() - 1)]);
        assert!(t.live_typed(p, &typed).is_empty());
        assert!(t.live_typed(Path::new("/r/g.rs"), &renamed).is_empty());
    }

    #[test]
    fn map_line_follows_edits() {
        let original: String = (0..20).map(|i| format!("l{i}\n")).collect();
        let content = format!("new0\nnew1\nnew2\n{}", original.replace("l9\n", "L9\nextra\n"));
        assert_eq!(map_line(&content, &original, 1), 0);
        assert_eq!(map_line(&content, &original, 5), 2);
        assert_eq!(map_line(&content, &original, 12), 9);
        assert_eq!(map_line(&content, &original, 13), 9);
        assert_eq!(map_line(&content, &original, 15), 11);
        assert_eq!(map_line(&original, &original, 7), 7);
    }
}
