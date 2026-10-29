//! Sweep next-edit prompt format and completion post-processing.
//!
//! Ported from Sweep's reference server (`sweep-autocomplete` 0.1.1,
//! `next_edit_autocomplete.py`), which is what Sweep ships for running the
//! 0.5B model locally. Offsets are UTF-8 byte offsets instead of Python
//! character offsets, and a few `str.find`-based offset lookups are replaced
//! by exact block offsets.

use crate::history::DiffHunk;
use serde::Serialize;

pub const NUM_LINES_BEFORE: usize = 2;
pub const NUM_LINES_AFTER: usize = 5;
pub const CHARS_PER_TOKEN: f64 = 3.5;
pub const OUTPUT_MAX_TOKENS: usize = 1024;
pub const TRUNCATION_LINE_LENGTH: usize = 600;
pub const MAXIMUM_LINE_LENGTH: usize = 1000;
pub const MAX_RETRIEVAL_CHUNK_LINES: usize = 25;
pub const CURSOR: &str = "<|cursor|>";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LineNumbers {
    /// Real 1-based line numbers of the window in the file.
    Absolute,
    /// Sweep's local server: cursor line within the block, plus one.
    SweepLocal,
    /// No `:start:end` suffix (the model card's `run_model.py` format).
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffFormat {
    /// `<|file_sep|>{path}:{start}:{end}` (Sweep's server).
    Server,
    /// `<|file_sep|>{path}.diff` (the model card's `run_model.py`).
    Card,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Prefill {
    None,
    /// First block line plus following blank lines (Sweep's server with
    /// `changes_above_cursor`).
    FirstLine,
    /// Every block line above the cursor line (cursortab.nvim's default).
    BeforeCursorLine,
    /// Everything before the cursor, backed off to the start of the word
    /// (and its leading spaces) so the prompt does not end mid-token.
    ThroughCursor,
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptOptions {
    pub line_numbers: LineNumbers,
    pub diff_format: DiffFormat,
    pub prefill: Prefill,
    pub cursor_marker: bool,
    /// Leave edits shown by the window out of the edit-history section.
    pub hide_active_edit: bool,
    pub lines_before: usize,
    pub lines_after: usize,
    /// Whole original file is used when it has at most this many lines.
    pub whole_file_lines: usize,
    /// Otherwise a stride-aligned chunk of this many lines around the cursor.
    pub chunk_lines: usize,
    pub max_retrieval_tokens: usize,
    pub max_hunks: usize,
    pub context_order: ContextOrder,
    /// Hide window edits before capping the history, keep the live edit
    /// group out of the history filters, and revert by line position.
    pub stable_history: bool,
    /// Show the live edit group only through the window, reverted to the
    /// text before the group (`original/`), never in the history.
    pub session_base: bool,
}

/// Where retrieved context sits relative to the edit history. Anything that
/// changes between keystrokes invalidates the KV cache from its position on,
/// so volatile context placed before the history forces the history to be
/// re-evaluated as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextOrder {
    /// Sweep's reference order: all retrieval before the history.
    Sweep,
    /// Definitions before the history; diagnostics and clipboard (which change
    /// while typing) after it, next to the window.
    VolatileLate,
    /// All retrieval after the history.
    RetrievalLate,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            line_numbers: LineNumbers::Absolute,
            diff_format: DiffFormat::Server,
            prefill: Prefill::FirstLine,
            cursor_marker: true,
            hide_active_edit: false,
            lines_before: NUM_LINES_BEFORE,
            lines_after: NUM_LINES_AFTER,
            whole_file_lines: 240,
            chunk_lines: 240,
            max_retrieval_tokens: 2048,
            max_hunks: 6,
            context_order: ContextOrder::Sweep,
            stable_history: false,
            session_base: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Chunk {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub source: &'static str,
    pub score: f32,
}

impl Chunk {
    pub fn render(&self) -> String {
        format!("<|file_sep|>{}\n{}\n", self.path, self.content)
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    /// Block text without the cursor marker.
    pub text: String,
    /// Byte offset of the block in the file.
    pub start: usize,
    /// 0-based line of the block start.
    pub start_line: usize,
}

pub fn estimate_tokens(text: &str) -> usize {
    (text.len() as f64 / CHARS_PER_TOKEN) as usize
}

/// Lines with their terminators. A trailing newline yields a final empty
/// line, so a cursor at EOF always has a line.
pub fn lines_keepends(text: &str) -> Vec<&str> {
    let mut out: Vec<&str> = text.split_inclusive('\n').collect();
    if text.is_empty() || text.ends_with('\n') {
        out.push("");
    }
    out
}

pub fn line_of_offset(text: &str, offset: usize) -> usize {
    text.as_bytes()[..offset.min(text.len())]
        .iter()
        .filter(|b| **b == b'\n')
        .count()
}

/// Sweep `get_block_at_cursor`: 2 lines above, 5 below, skipping leading and
/// trailing blank lines (never past the cursor line), then token-capped.
pub fn block_at_cursor(content: &str, cursor: usize, before: usize, after: usize) -> Block {
    let lines = lines_keepends(content);
    let cursor_line = line_of_offset(content, cursor).min(lines.len() - 1);
    block_around_line(&lines, cursor_line, before, after)
}

fn block_around_line(lines: &[&str], cursor_line: usize, before: usize, after: usize) -> Block {
    let n = lines.len();
    let mut start = cursor_line.saturating_sub(before);
    let mut end = (cursor_line + after + 1).min(n);
    while start < cursor_line && lines[start].trim().is_empty() {
        start += 1;
        if end < n {
            end += 1;
        }
    }
    while end > cursor_line + 1 && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    // Token cap: keep the first NUM_LINES_BEFORE lines, cap the rest.
    let max_chars = ((OUTPUT_MAX_TOKENS / 2) as f64 * CHARS_PER_TOKEN) as usize;
    let head_end = (start + before).min(end);
    let mut tail_chars = 0;
    let mut capped_end = head_end;
    for (i, line) in lines.iter().enumerate().take(end).skip(head_end) {
        tail_chars += line.len();
        if tail_chars > max_chars && i > cursor_line {
            break;
        }
        capped_end = i + 1;
    }
    let start_offset: usize = lines[..start].iter().map(|l| l.len()).sum();
    Block {
        text: lines[start..capped_end].concat(),
        start: start_offset,
        start_line: start,
    }
}

/// A chunk start is kept while the cursor is at least this many lines inside.
const CHUNK_MARGIN: usize = 10;

/// First line of the stride-aligned chunk shown for a file of `total` lines
/// with the cursor on `line`. The previous start is kept while the cursor
/// stays `CHUNK_MARGIN` lines inside its chunk: a new start changes the
/// whole file section, which opens the prompt.
pub fn chunk_start(total: usize, line: usize, opts: &PromptOptions, previous: Option<usize>) -> usize {
    let chunk = opts.chunk_lines.max(2);
    let stride = chunk / 2;
    if let Some(p) = previous.filter(|p| *p < total) {
        let end = (p + chunk).min(total);
        let lo = if p == 0 { 0 } else { p + CHUNK_MARGIN };
        let hi = if end == total { usize::MAX } else { end.saturating_sub(CHUNK_MARGIN) };
        if line >= lo && line < hi {
            return p;
        }
    }
    let ideal = line as i64 - (chunk / 2) as i64;
    let index = ((ideal as f64) / stride as f64).round().max(0.0) as usize;
    (index * stride).min(total.saturating_sub(1))
}

/// The whole file when it is short, else the chunk starting at `start`.
pub fn lines_from(content: &str, start: usize, opts: &PromptOptions) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= opts.whole_file_lines {
        return content.to_string();
    }
    let start = start.min(lines.len().saturating_sub(1));
    let end = (start + opts.chunk_lines.max(2)).min(lines.len());
    lines[start..end].join("\n")
}

/// Sweep `get_lines_around_cursor` with configurable sizes.
pub fn lines_around_cursor(content: &str, cursor: usize, opts: &PromptOptions) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= opts.whole_file_lines {
        return content.to_string();
    }
    let chunk = opts.chunk_lines.max(2);
    let stride = chunk / 2;
    let cursor_line = line_of_offset(content, cursor) as i64;
    let ideal = cursor_line - (chunk / 2) as i64;
    let index = ((ideal as f64) / stride as f64).round().max(0.0) as usize;
    let start = (index * stride).min(lines.len().saturating_sub(1));
    let end = (start + chunk).min(lines.len());
    lines[start..end].join("\n")
}

fn strip_leading_empty_newlines(text: &str) -> String {
    let parts: Vec<&str> = text.split('\n').collect();
    let first = parts.iter().position(|l| !l.trim().is_empty()).unwrap_or(parts.len());
    parts[first..].join("\n")
}

fn truncate_long_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        if line.len() > TRUNCATION_LINE_LENGTH {
            let mut cut = TRUNCATION_LINE_LENGTH.min(body.len());
            while !body.is_char_boundary(cut) {
                cut -= 1;
            }
            out.push_str(&body[..cut]);
            out.push_str("...");
            if line.ends_with('\n') {
                out.push('\n');
            }
        } else {
            out.push_str(line);
        }
    }
    out
}

pub fn should_disable_for_block(block: &str) -> bool {
    block.lines().any(|l| l.len() > MAXIMUM_LINE_LENGTH)
}

/// Sweep `should_disable_autocomplete` (file-level guards).
pub fn should_disable_file(content: &str) -> Option<&'static str> {
    if content.len() > 10_000_000 {
        return Some("file_too_large");
    }
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return None;
    }
    if lines.len() > 50_000 {
        return Some("too_many_lines");
    }
    let total: usize = lines.iter().map(|l| l.len()).sum();
    if total as f64 / lines.len() as f64 > 240.0 {
        return Some("long_average_line");
    }
    if lines.len() > 1000 && lines.iter().filter(|l| l.len() > 120).count() * 10 > lines.len() * 3 {
        return Some("many_long_lines");
    }
    None
}

pub struct PromptInput<'a> {
    pub path: &'a str,
    pub content: &'a str,
    pub original: &'a str,
    pub cursor: usize,
    pub block: &'a Block,
    pub hunks: &'a [DiffHunk],
    pub file_chunks: &'a [Chunk],
    /// Retrieval chunks ordered so the most relevant is last.
    pub retrieval: &'a [Chunk],
    pub opts: &'a PromptOptions,
    /// Trailing `hunks` from the live edit group (`stable_history`).
    pub live: usize,
    /// Text before the live edit group, when it is in this file
    /// (`session_base`).
    pub session_base: Option<&'a str>,
    /// First line of the original-file chunk; None maps the cursor's byte
    /// offset into the original (the old behaviour).
    pub initial_start: Option<usize>,
}

#[derive(Debug)]
pub struct Built {
    pub prompt: String,
    /// Byte length of the stable head (file chunks and the original file).
    pub stable_prefix: usize,
    pub prefill: String,
    pub prev_sections: Vec<String>,
    pub retrieval_used: usize,
    pub file_chunks_used: usize,
    pub hunks_used: usize,
    pub initial_file_lines: usize,
    /// Oldest edit group the prompt shows, in the history or through the
    /// window; None when it shows no edit.
    pub shown_from: Option<u64>,
}

/// Sweep `format_recent_changes_and_prev_section`, restricted to hunks of the
/// current file when reverting (matching is substring-based).
fn recent_changes_and_prev(
    path: &str,
    hunks: &[DiffHunk],
    cleaned_block: &str,
    max_hunks: usize,
    diff_format: DiffFormat,
    hide_active: bool,
) -> (String, String, Vec<String>) {
    let mut prev = cleaned_block.to_string();
    let mut prev_sections = Vec::new();
    for h in hunks.iter().rev() {
        if h.display != path {
            break;
        }
        let (old0, new0) = h.parts(0);
        let (old1, new1) = h.parts(1);
        if !new1.trim().is_empty() && prev.contains(&new1) {
            prev = prev.replacen(&new1, &old1, 1);
            prev_sections.push(prev.clone());
        } else if !new0.trim().is_empty() && prev.contains(&new0) {
            prev = prev.replacen(&new0, &old0, 1);
            prev_sections.push(prev.clone());
        } else {
            break;
        }
    }
    // The newest hunks that were reverted above lie inside the window, whose
    // original/current pair already shows them. Hiding them keeps this
    // section (and the KV cache through the original window) unchanged while
    // the user keeps typing in the same place.
    let shown = if hide_active { &hunks[..hunks.len() - prev_sections.len()] } else { hunks };
    let mut result = String::new();
    let skip = shown.len().saturating_sub(max_hunks);
    for h in &shown[skip..] {
        let (old, new) = h.parts(1);
        if old.trim().is_empty() && new.trim().is_empty() {
            continue;
        }
        let (start, end) = h.shown_new_range(1);
        let header = match diff_format {
            DiffFormat::Server => format!("{}:{}:{}", h.display, start, end),
            DiffFormat::Card => format!("{}.diff", h.display),
        };
        result.push_str(&format!(
            "<|file_sep|>{}\noriginal:\n{}\nupdated:\n{}\n",
            header,
            old.trim_matches('\n'),
            new.trim_matches('\n')
        ));
    }
    let result = result.trim_end_matches('\n').to_string();
    (result, prev, prev_sections)
}

fn render_hunk(h: &DiffHunk, diff_format: DiffFormat) -> Option<String> {
    let (old, new) = h.parts(1);
    if old.trim().is_empty() && new.trim().is_empty() {
        return None;
    }
    let (start, end) = h.shown_new_range(1);
    let header = match diff_format {
        DiffFormat::Server => format!("{}:{}:{}", h.display, start, end),
        DiffFormat::Card => format!("{}.diff", h.display),
    };
    Some(format!(
        "<|file_sep|>{}\noriginal:\n{}\nupdated:\n{}\n",
        header,
        old.trim_matches('\n'),
        new.trim_matches('\n')
    ))
}

/// The last `max_hunks` renderable hunks, with the number used and the
/// group of the oldest one.
fn render_history<'h>(hunks: impl DoubleEndedIterator<Item = &'h DiffHunk>, max_hunks: usize, diff_format: DiffFormat) -> (String, usize, Option<u64>) {
    let mut shown: Vec<(u64, String)> = hunks
        .rev()
        .filter(|h| !h.is_whitespace_only())
        .filter_map(|h| render_hunk(h, diff_format).map(|r| (h.group, r)))
        .take(max_hunks)
        .collect();
    shown.reverse();
    let n = shown.len();
    let oldest = shown.first().map(|(g, _)| *g);
    let text: String = shown.into_iter().map(|(_, r)| r).collect();
    (text.trim_end_matches('\n').to_string(), n, oldest)
}

/// Lines `[w0, w1)` of `lines` joined like the window text `like`.
fn join_window(lines: &[String], like: &str) -> String {
    let mut out: String = lines.iter().map(|l| format!("{l}\n")).collect();
    if !like.ends_with('\n') {
        out.pop();
    }
    out
}

/// Window line range `[w0, w1)` of a block in its file.
fn window_lines(block: &Block) -> (usize, usize) {
    (block.start_line, block.start_line + block.text.split_inclusive('\n').count())
}

/// `recent_changes_and_prev` with positional reverts. Hunk groups are
/// reverted newest first by their line positions, which is exact for
/// chained same-file groups and cannot pick a wrong repeated occurrence. A
/// group is reverted while any of its hunks overlaps the window; those
/// hunks are hidden from the history (the window pair shows them), and the
/// history is capped after hiding so hidden hunks do not take its slots.
/// Returns (history, window before the reverted edits, revert steps, hunks
/// shown, oldest group shown in either).
fn stable_changes_and_prev(
    path: &str,
    hunks: &[DiffHunk],
    content: &str,
    block: &Block,
    max_hunks: usize,
    diff_format: DiffFormat,
    hide_active: bool,
) -> (String, String, Vec<String>, usize, Option<u64>) {
    let mut file: Vec<String> = content.split('\n').map(str::to_string).collect();
    let (mut w0, mut w1) = window_lines(block);
    let mut hidden = vec![false; hunks.len()];
    let mut sections = Vec::new();
    let mut end = hunks.len();
    while end > 0 && hunks[end - 1].display == path {
        let group = hunks[end - 1].group;
        let start = hunks[..end].iter().rposition(|h| h.group != group || h.display != path).map_or(0, |i| i + 1);
        let span = |h: &DiffHunk| {
            let c0 = h.new_start + h.ctx_before.len();
            (c0, c0 + h.new.len())
        };
        let applies = hunks[start..end].iter().all(|h| {
            let (c0, c1) = span(h);
            c1 <= file.len() && file[c0..c1] == h.new[..]
        });
        let overlaps = |h: &DiffHunk, w0: usize, w1: usize| {
            let (c0, c1) = span(h);
            if c0 == c1 { w0 < c0 && c0 < w1 } else { c0 < w1 && c1 > w0 }
        };
        if !applies || !hunks[start..end].iter().any(|h| overlaps(h, w0, w1)) {
            break;
        }
        // Bottom-up, so earlier hunks of the group keep their positions.
        for i in (start..end).rev() {
            let h = &hunks[i];
            let (c0, c1) = span(h);
            let inside = overlaps(h, w0, w1);
            let delta = h.old.len() as isize - h.new.len() as isize;
            let map = |w: usize| {
                if w <= c0 {
                    w
                } else if w >= c1 {
                    (w as isize + delta) as usize
                } else {
                    c0 + h.old.len()
                }
            };
            // A window edge inside the hunk widens the window to all of it.
            w0 = if w0 > c0 && w0 < c1 { c0 } else { map(w0) };
            w1 = map(w1);
            file.splice(c0..c1, h.old.iter().cloned());
            if inside {
                hidden[i] = true;
                sections.push(join_window(&file[w0..w1], &block.text));
            }
        }
        end = start;
    }
    let prev = join_window(&file[w0..w1.min(file.len())], &block.text);
    let shown = hunks.iter().zip(&hidden).filter(|(_, hide)| !(hide_active && **hide)).map(|(h, _)| h);
    let (history, used, oldest) = render_history(shown, max_hunks, diff_format);
    let in_window = hunks.iter().zip(&hidden).filter(|(_, hide)| **hide).map(|(h, _)| h.group).min();
    let oldest = match (oldest, in_window) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    (history, prev, sections, used, oldest)
}

/// The window's lines `[w0, w1)` of `content` with every region changed
/// since `base` that touches the window replaced by its `base` lines.
fn window_from_base(base: &str, content: &str, block: &Block) -> String {
    let a: Vec<&str> = base.split('\n').collect();
    let b: Vec<&str> = content.split('\n').collect();
    let (w0, w1) = window_lines(block);
    let mut out: Vec<String> = Vec::new();
    for op in similar::capture_diff_slices(similar::Algorithm::Myers, &a, &b) {
        let (old, new) = (op.old_range(), op.new_range());
        if let similar::DiffOp::Equal { .. } = op {
            let (lo, hi) = (new.start.max(w0), new.end.min(w1));
            if lo < hi {
                out.extend(b[lo..hi].iter().map(|l| l.to_string()));
            }
        } else if (new.start < w1 && new.end > w0) || (new.is_empty() && w0 < new.start && new.start < w1) {
            out.extend(a[old].iter().map(|l| l.to_string()));
        }
    }
    join_window(&out, &block.text)
}

fn compute_prefill(cleaned: &str, rel_cursor: usize, mode: Prefill) -> String {
    let before = &cleaned[..rel_cursor];
    match mode {
        Prefill::None => return String::new(),
        Prefill::BeforeCursorLine => {
            return before.rfind('\n').map_or(String::new(), |i| before[..=i].to_string());
        }
        Prefill::ThroughCursor => {
            let word = before.trim_end_matches(|c: char| c.is_alphanumeric() || c == '_');
            let cut = word.trim_end_matches([' ', '\t']);
            return cut.to_string();
        }
        Prefill::FirstLine => {}
    }
    let mut parts = before.split_inclusive('\n');
    let mut prefill = parts.next().unwrap_or("").to_string();
    let rest: String = parts.collect();
    for ch in rest.chars() {
        if ch == '\n' {
            prefill.push('\n');
        } else {
            break;
        }
    }
    prefill
}

/// Pack chunks from the end backwards within a character budget (Sweep
/// `pack_items_for_prompt` with `truncate_from_end=False`).
fn pack_from_end(chunks: &[Chunk], token_limit: usize) -> &[Chunk] {
    let limit = (token_limit as f64 * CHARS_PER_TOKEN) as usize;
    let mut total = 0;
    let mut start = chunks.len();
    for (i, c) in chunks.iter().enumerate().rev() {
        let len = c.render().len();
        if total + len > limit {
            break;
        }
        total += len;
        start = i;
    }
    &chunks[start..]
}

pub fn build_prompt(input: &PromptInput<'_>) -> Built {
    let block = input.block;
    let rel_cursor = input.cursor - block.start;
    let cleaned = block.text.as_str();
    let rel_cursor_line = line_of_offset(cleaned, rel_cursor);
    let marker = if input.opts.cursor_marker { CURSOR } else { "" };
    let mut code_block = format!("{}{}{}", &cleaned[..rel_cursor], marker, &cleaned[rel_cursor..]);
    let opts = input.opts;
    let (recent, mut prev, prev_sections, hunks_used, shown_from) = if opts.session_base {
        // Only finalized groups are history; the live one is in the window.
        let finalized = &input.hunks[..input.hunks.len() - input.live];
        let (history, used, oldest) = render_history(finalized.iter(), opts.max_hunks, opts.diff_format);
        let prev = match input.session_base {
            Some(base) => window_from_base(base, input.content, block),
            None => cleaned.to_string(),
        };
        let sections = if prev != cleaned { vec![prev.clone()] } else { Vec::new() };
        let live = input.hunks[input.hunks.len() - input.live..].iter().map(|h| h.group).min();
        (history, prev, sections, used, oldest.or(live))
    } else if opts.stable_history {
        stable_changes_and_prev(input.path, input.hunks, input.content, block, opts.max_hunks, opts.diff_format, opts.hide_active_edit)
    } else {
        let (recent, prev, sections) = recent_changes_and_prev(
            input.path,
            input.hunks,
            cleaned,
            opts.max_hunks,
            opts.diff_format,
            opts.hide_active_edit,
        );
        let n = input.hunks.len();
        let shown = n.saturating_sub(opts.max_hunks + if opts.hide_active_edit { sections.len() } else { 0 });
        let oldest = input.hunks[shown..].iter().map(|h| h.group).min();
        (recent, prev, sections, n.min(opts.max_hunks), oldest)
    };
    let prefill = compute_prefill(cleaned, rel_cursor, input.opts.prefill);
    let retrieval = pack_from_end(input.retrieval, input.opts.max_retrieval_tokens);
    let late = |c: &Chunk| match input.opts.context_order {
        ContextOrder::Sweep => false,
        ContextOrder::VolatileLate => matches!(c.source, "diagnostics" | "clipboard"),
        ContextOrder::RetrievalLate => true,
    };
    let retrieval_text: String = retrieval.iter().filter(|c| !late(c)).map(|c| format!("\n{}", c.render())).collect();
    let late_text: String = retrieval.iter().filter(|c| late(c)).map(|c| format!("\n{}", c.render())).collect();
    if code_block.ends_with('\n') && prev.ends_with('\n') {
        code_block.pop();
        prev.pop();
    }
    let initial = match input.initial_start {
        Some(start) => lines_from(input.original, start, input.opts),
        None => lines_around_cursor(input.original, input.cursor.min(input.original.len()), input.opts),
    };
    let block_lines = code_block.split('\n').count();
    let range = match input.opts.line_numbers {
        LineNumbers::Absolute => format!(":{}:{}", block.start_line + 1, block.start_line + block_lines),
        LineNumbers::SweepLocal => format!(":{}:{}", rel_cursor_line + 1, rel_cursor_line + block_lines + 1),
        LineNumbers::None => String::new(),
    };
    let mut prompt = String::new();
    for c in input.file_chunks {
        prompt.push_str(&c.render());
    }
    prompt.push_str(&format!("<|file_sep|>{}\n{}", input.path, initial));
    let stable_prefix = prompt.len();
    prompt.push_str(&retrieval_text);
    prompt.push('\n');
    prompt.push_str(&recent);
    prompt.push_str(&late_text);
    prompt.push('\n');
    let fp = input.path;
    prompt.push_str(&format!(
        "<|file_sep|>original/{fp}{range}\n{prev}\n<|file_sep|>current/{fp}{range}\n{code_block}\n<|file_sep|>updated/{fp}{range}"
    ));
    prompt.push('\n');
    prompt.push_str(&prefill);
    let prompt = truncate_long_lines(&prompt);
    Built {
        stable_prefix: stable_prefix.min(prompt.len()),
        prompt,
        prefill,
        prev_sections,
        retrieval_used: retrieval.len(),
        file_chunks_used: input.file_chunks.len(),
        hunks_used,
        initial_file_lines: initial.lines().count(),
        shown_from,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Edit {
    /// Byte range in the file being replaced.
    pub start: usize,
    pub end: usize,
    pub text: String,
}

fn ghost_text_at<'a>(completion: &'a str, cleaned: &str, pos: usize) -> &'a str {
    let (prefix, suffix) = cleaned.split_at(pos);
    if completion.len() >= prefix.len() + suffix.len()
        && completion.starts_with(prefix)
        && completion.ends_with(suffix)
    {
        return &completion[prefix.len()..completion.len() - suffix.len()];
    }
    ""
}

fn find_ghost_text_non_local<'a>(completion: &'a str, cleaned: &str) -> Option<(&'a str, usize)> {
    if cleaned.len() > completion.len() {
        return None;
    }
    let mut best = None;
    for pos in 0..=cleaned.len() {
        if !cleaned.is_char_boundary(pos) {
            continue;
        }
        let g = ghost_text_at(completion, cleaned, pos);
        if !g.is_empty() && completion.is_char_boundary(pos) {
            best = Some((g, pos));
        }
    }
    best
}

fn is_single_line_ghost_text<'a>(completion: &'a str, cleaned: &str, rel_cursor: usize) -> &'a str {
    if cleaned.len() < rel_cursor {
        return "";
    }
    let g = ghost_text_at(completion, cleaned, rel_cursor);
    if !g.is_empty() && !g.contains('\n') { g } else { "" }
}

/// Line hunks between block and completion, as (old line index, old lines,
/// new lines) with every line newline-terminated (difflib n=0 semantics).
fn line_hunks(old: &str, new: &str) -> Vec<(usize, usize, Vec<String>)> {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let ops = similar::capture_diff_slices(similar::Algorithm::Myers, &a, &b);
    let mut out: Vec<(usize, usize, Vec<String>)> = Vec::new();
    let mut last_end: Option<usize> = None;
    for op in ops {
        let (oi, ol, ni, nl) = match op {
            similar::DiffOp::Equal { .. } => {
                last_end = None;
                continue;
            }
            similar::DiffOp::Delete { old_index, old_len, new_index } => (old_index, old_len, new_index, 0),
            similar::DiffOp::Insert { old_index, new_index, new_len } => (old_index, 0, new_index, new_len),
            similar::DiffOp::Replace { old_index, old_len, new_index, new_len } => {
                (old_index, old_len, new_index, new_len)
            }
        };
        let new_lines: Vec<String> = b[ni..ni + nl].iter().map(|l| format!("{l}\n")).collect();
        if last_end == Some(oi) {
            let last = out.last_mut().unwrap();
            last.1 += ol;
            last.2.extend(new_lines);
        } else {
            out.push((oi, ol, new_lines));
        }
        last_end = Some(oi + ol);
    }
    out
}

/// Sweep `select_best_hunk_from_completion`. `block_start` is the byte offset
/// of `cleaned` in `file`.
pub fn select_best_hunks(
    completion: &str,
    cleaned: &str,
    block_start: usize,
    file: &str,
    cursor: usize,
) -> Vec<Edit> {
    let completion = strip_leading_empty_newlines(completion);
    let stripped = strip_leading_empty_newlines(cleaned);
    if completion == stripped {
        return Vec::new();
    }
    let block_start = block_start + (cleaned.len() - stripped.len());
    let cleaned = stripped.as_str();
    if cursor < block_start {
        return Vec::new();
    }
    let rel_cursor = cursor - block_start;

    if let Some((ghost, pos)) = find_ghost_text_non_local(&completion, cleaned) {
        let at = pos + block_start;
        let insert_next_line = pos == rel_cursor + 1 && pos >= 1 && cleaned.as_bytes()[pos - 1] == b'\n';
        let starts_with_newline = ghost.starts_with('\n') && pos == rel_cursor;
        if insert_next_line || starts_with_newline {
            return vec![Edit { start: at, end: at, text: ghost.to_string() }];
        }
        let mut parts = ghost.split_inclusive('\n');
        let mut first = parts.next().unwrap_or("").to_string();
        let mut remaining: String = parts.collect();
        let trailing = first.len() - first.trim_end_matches('\n').len();
        if trailing > 0 {
            remaining = format!("{}{}", &first[first.len() - trailing..], remaining);
            first.truncate(first.len() - trailing);
        }
        if !remaining.is_empty() {
            let remaining = remaining.trim_end().to_string();
            if pos < cleaned.len() && cleaned.as_bytes()[pos] == b'\n' {
                return vec![
                    Edit { start: at, end: at, text: first },
                    Edit { start: at, end: at, text: remaining },
                ];
            }
        } else {
            return vec![Edit { start: at, end: at, text: ghost.to_string() }];
        }
    }

    let hunks = line_hunks(cleaned, &completion);
    if hunks.is_empty() {
        return Vec::new();
    }
    let original_lines: Vec<&str> = cleaned.split_inclusive('\n').collect();
    let block_end = block_start + cleaned.len();
    let mut processed: Vec<Edit> = Vec::new();
    for (start_idx, old_len, new_lines) in hunks {
        let start: usize = block_start + original_lines[..start_idx.min(original_lines.len())].iter().map(|l| l.len()).sum::<usize>();
        let end: usize = start
            + original_lines[start_idx.min(original_lines.len())..(start_idx + old_len).min(original_lines.len())]
                .iter()
                .map(|l| l.len())
                .sum::<usize>();
        let mut text: String = new_lines.concat();
        if end == block_end && !cleaned.ends_with('\n') && !completion.ends_with('\n') && text.ends_with('\n') {
            text.pop();
        }
        if start == file.len() && start > 0 && file.as_bytes()[start - 1] != b'\n' {
            text = format!("\n{text}");
        }
        if file.get(start..end) != Some(text.as_str()) {
            processed.push(Edit { start, end, text });
        }
    }

    let mut after: Vec<Edit> = processed.iter().filter(|h| h.end >= cursor).cloned().collect();
    let before: Vec<Edit> = processed.iter().filter(|h| h.end < cursor).cloned().collect();

    if !after.is_empty() {
        after.sort_by_key(|h| h.start);
        let first = after.remove(0);
        let line_start = file[..cursor].rfind('\n').map_or(0, |i| i + 1);
        let mut results = Vec::new();
        let mut split = line_start == first.start && first.start <= cursor && cursor < first.end && first.text.contains('\n');
        if split {
            let mut it = first.text.split_inclusive('\n');
            let first_line = it.next().unwrap_or("").to_string();
            let remaining: String = it.collect();
            let section = &file[cursor..first.end];
            let first_line_end = match section.find('\n') {
                Some(i) => cursor + i + 1,
                None => first.end,
            };
            let line_end = file[cursor..].find('\n').map_or(file.len(), |i| cursor + i);
            let current_line = &file[line_start..line_end];
            if !first_line.starts_with(current_line) {
                split = false;
            }
            if split {
                results.push(Edit { start: first.start, end: first_line_end, text: first_line });
                if !remaining.is_empty() {
                    results.push(Edit { start: first_line_end, end: first.end, text: remaining });
                }
            }
        }
        if !split {
            results.push(first);
        }
        results.extend(after);
        return results;
    }

    if before.is_empty() {
        return Vec::new();
    }
    // Fuse hunks before the cursor that are within two lines of each other.
    let mut before = before;
    before.sort_by_key(|h| h.start);
    let mut groups: Vec<Vec<Edit>> = vec![vec![before[0].clone()]];
    for h in before.into_iter().skip(1) {
        let prev_end_line = line_of_offset(file, groups.last().unwrap().last().unwrap().end);
        let start_line = line_of_offset(file, h.start);
        if start_line <= prev_end_line + 2 {
            groups.last_mut().unwrap().push(h);
        } else {
            groups.push(vec![h]);
        }
    }
    let mut results: Vec<Edit> = groups
        .into_iter()
        .map(|g| {
            let start = g[0].start;
            let end = g.last().unwrap().end;
            let mut text = String::new();
            let mut at = start;
            for h in &g {
                if at < h.start {
                    text.push_str(&file[at..h.start]);
                }
                text.push_str(&h.text);
                at = h.end;
            }
            Edit { start, end, text }
        })
        .collect();
    results.sort_by_key(|e| e.start.abs_diff(cursor));
    results
}

fn apply_edits_to_block(edits: &[Edit], block_start: usize, cleaned: &str) -> String {
    let mut out = cleaned.to_string();
    let mut sorted: Vec<&Edit> = edits.iter().collect();
    sorted.sort_by(|a, b| b.start.cmp(&a.start));
    for e in sorted {
        if e.start < block_start {
            continue;
        }
        let (s, t) = (e.start - block_start, e.end - block_start);
        if s <= cleaned.len() && t <= cleaned.len() && s <= t && out.is_char_boundary(s) && out.is_char_boundary(t) {
            out.replace_range(s..t, &e.text);
        }
    }
    out
}

fn normalize_newlines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_newline = false;
    for c in text.chars() {
        if c == '\n' {
            if !last_newline {
                out.push(c);
            }
            last_newline = true;
        } else {
            out.push(c);
            last_newline = false;
        }
    }
    out
}

fn is_pure_insertion_above_cursor(cleaned: &str, completion: &str, rel_cursor: usize) -> bool {
    if cleaned.trim() == completion.trim() {
        return false;
    }
    let lines: Vec<&str> = cleaned.split_inclusive('\n').collect();
    let current = cleaned[..rel_cursor].split_inclusive('\n').count().max(1);
    let Some(cursor_line) = lines.get(current - 1) else {
        return false;
    };
    if cursor_line.trim().is_empty() {
        return false;
    }
    let prefix: String = lines[..current - 1].concat();
    let suffix: String = lines[current..].concat();
    completion.starts_with(&prefix) && completion.ends_with(&format!("{cursor_line}{suffix}"))
}

fn is_large_diff_above_cursor(original: &str, completion: &str, rel_cursor: usize) -> bool {
    if original == completion {
        return false;
    }
    let above_completion = completion.get(..rel_cursor.min(completion.len()));
    if Some(&original[..rel_cursor]) == above_completion {
        return false;
    }
    let a: Vec<&str> = original.lines().collect();
    let b: Vec<&str> = completion.lines().collect();
    let (mut adds, mut dels) = (0, 0);
    for op in similar::capture_diff_slices(similar::Algorithm::Myers, &a, &b) {
        match op {
            similar::DiffOp::Delete { old_len, .. } => dels += old_len,
            similar::DiffOp::Insert { new_len, .. } => adds += new_len,
            similar::DiffOp::Replace { old_len, new_len, .. } => {
                dels += old_len;
                adds += new_len;
            }
            similar::DiffOp::Equal { .. } => {}
        }
    }
    adds > 5 && dels > 5
}

#[derive(Debug)]
pub struct Processed {
    pub edits: Vec<Edit>,
    pub exit_reason: &'static str,
    pub completion: String,
}

/// Early stop for window rewrites: once the output has diverged from the
/// window and its last two complete lines match two consecutive window lines
/// after the divergence, the remainder of the window is assumed unchanged.
/// Returns the window text to append. Only the first edit is ever shown, so
/// later edits in the window are not lost in practice.
pub fn splice_rest(prefill: &str, generated: &str, cleaned: &str) -> Option<String> {
    splice_rest_n(prefill, generated, cleaned, 2)
}

/// `splice_rest` with a configurable number of matching anchor lines.
pub fn splice_rest_n(prefill: &str, generated: &str, cleaned: &str, anchors: usize) -> Option<String> {
    let anchors = anchors.max(1);
    let out = format!("{prefill}{generated}");
    let complete = &out[..out.rfind('\n')? + 1];
    let a = cleaned.as_bytes();
    let b = complete.as_bytes();
    let mut p = 0;
    while p < a.len() && p < b.len() && a[p] == b[p] {
        p += 1;
    }
    if p == b.len() {
        return None; // no divergence yet
    }
    let out_lines: Vec<&str> = complete[p..].split_inclusive('\n').collect();
    if out_lines.len() < anchors + 1 {
        return None; // need the changed line plus the anchor lines
    }
    let tail = &out_lines[out_lines.len() - anchors..];
    if tail.iter().map(|l| l.trim().len()).sum::<usize>() < 8 {
        return None; // braces and blank lines are ambiguous anchors
    }
    let div_line = cleaned[..p].matches('\n').count();
    let block_lines: Vec<&str> = cleaned.split_inclusive('\n').collect();
    let mut found = None;
    for j in div_line..block_lines.len().saturating_sub(anchors - 1) {
        if (0..anchors).all(|k| block_lines[j + k] == tail[k]) {
            if found.is_some() {
                return None; // ambiguous anchor
            }
            found = Some(j + anchors);
        }
    }
    found.map(|k| block_lines[k..].concat())
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Confidence {
    /// Minimum token log-probability over tokens that change the window.
    pub min_logprob: f32,
    pub mean_logprob: f32,
    pub changed_tokens: usize,
}

/// Confidence over the generated tokens that differ from the window.
/// `generated` excludes the prefill; offsets index into `generated`.
pub fn change_confidence(prefill: &str, generated: &str, cleaned: &str, logprobs: &[f32], offsets: &[usize]) -> Confidence {
    let completion = format!("{prefill}{generated}");
    let a = cleaned.as_bytes();
    let b = completion.as_bytes();
    let mut p = 0;
    while p < a.len() && p < b.len() && a[p] == b[p] {
        p += 1;
    }
    let mut s = 0;
    while s < a.len() - p && s < b.len() - p && a[a.len() - 1 - s] == b[b.len() - 1 - s] {
        s += 1;
    }
    let (lo, hi) = (p.saturating_sub(prefill.len()), (b.len() - s).saturating_sub(prefill.len()));
    let mut picked: Vec<f32> = Vec::new();
    for (i, off) in offsets.iter().enumerate() {
        let end = offsets.get(i + 1).copied().unwrap_or(generated.len());
        // Tokens overlapping the changed span, or the token at a pure deletion point.
        if (end > lo && *off < hi.max(lo + 1)) || (lo == hi && *off <= lo && lo < end) {
            picked.push(logprobs[i]);
        }
    }
    if picked.is_empty() {
        return Confidence { min_logprob: 0.0, mean_logprob: 0.0, changed_tokens: 0 };
    }
    Confidence {
        min_logprob: picked.iter().copied().fold(f32::INFINITY, f32::min),
        mean_logprob: picked.iter().sum::<f32>() / picked.len() as f32,
        changed_tokens: picked.len(),
    }
}

/// Post-processing after generation (Sweep `_fetch_next_edits_core` tail).
pub fn process_completion(
    generated: &str,
    finish_reason: &str,
    built: &Built,
    block: &Block,
    file: &str,
    cursor: usize,
) -> Processed {
    let cleaned = block.text.as_str();
    let rel_cursor = cursor - block.start;
    let empty = |reason: &'static str, completion: String| Processed {
        edits: Vec::new(),
        exit_reason: reason,
        completion,
    };
    if generated.is_empty() && built.prefill.is_empty() {
        return empty("empty_completion", String::new());
    }
    if generated.starts_with("<|") {
        return empty("special_token_in_completion", generated.to_string());
    }
    let mut completion = format!("{}{}", built.prefill, generated);
    if is_pure_insertion_above_cursor(cleaned, &completion, rel_cursor) {
        return empty("pure_insertion_above_cursor", completion);
    }
    if is_large_diff_above_cursor(cleaned, &completion, rel_cursor) {
        return empty("large_diff_above_cursor", completion);
    }
    if let Some(i) = completion.find(" No newline at end of file") {
        if completion.trim_end_matches('\n').ends_with(" No newline at end of file") {
            completion.truncate(i);
        }
    }
    let mut c = strip_leading_empty_newlines(&completion);
    if let Some(s) = c.strip_suffix("<|file_sep|>") {
        c = s.to_string();
    }
    if c.is_empty() {
        c = cleaned.to_string();
    }
    if !cleaned.contains(CURSOR) {
        c = c.replace(CURSOR, "");
    }
    let cleaned_lines = cleaned.split_inclusive('\n').count();
    let lines: Vec<&str> = c.split_inclusive('\n').collect();
    if lines.len() > cleaned_lines + 20 {
        c = lines[..cleaned_lines + 20].concat();
    }
    let completion = c;
    if finish_reason == "length" {
        return empty("hit_max_tokens", completion);
    }
    let _ = finish_reason; // "spliced" completions are complete windows
    let edits = select_best_hunks(&completion, cleaned, block.start, file, cursor);
    if completion.trim_matches('\n') == cleaned.trim_matches('\n') {
        return Processed { edits, exit_reason: "no_changes_made", completion };
    }
    let applied = apply_edits_to_block(&edits, block.start, cleaned);
    for section in &built.prev_sections {
        if normalize_newlines(&applied) == normalize_newlines(section) {
            return empty("revert_detected", completion);
        }
    }
    let cleaned_lines: Vec<&str> = cleaned.split_inclusive('\n').collect();
    let rel_line = line_of_offset(cleaned, rel_cursor);
    let current_line = cleaned_lines.get(rel_line).copied().unwrap_or("");
    if current_line.trim().is_empty() && !current_line.is_empty() {
        if let Some(first) = edits.first() {
            let whitespace_deleted = first.text.is_empty()
                && file[first.start..first.end].trim().is_empty()
                && (first.end == cursor || first.end == cursor + 1);
            if whitespace_deleted {
                return empty("pure_whitespace_deleted", completion);
            }
            let with_first = apply_edits_to_block(std::slice::from_ref(first), block.start, cleaned);
            let ghost = is_single_line_ghost_text(&with_first, cleaned, rel_cursor);
            if ghost.starts_with(' ') {
                return empty("ghost_text_starts_with_whitespace", completion);
            }
        }
    }
    if edits.is_empty() {
        return empty("no_edits", completion);
    }
    Processed { edits, exit_reason: "success", completion }
}

/// Bytes `a` and `b` share at the start, at a character boundary.
fn common_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).map(|(x, _)| x.len_utf8()).sum()
}

/// Bytes `a` and `b` share at the end, at a character boundary.
fn common_suffix(a: &str, b: &str) -> usize {
    a.chars().rev().zip(b.chars().rev()).take_while(|(x, y)| x == y).map(|(x, _)| x.len_utf8()).sum()
}

/// Whether replacing `old`, at byte `at` of the file, by `new` touches
/// `typed`: the part between their common prefix and suffix overlaps a
/// typed range. Adding text touches nothing.
fn touches_typed(old: &str, new: &str, at: usize, typed: &[(usize, usize)]) -> bool {
    let p = common_prefix(old, new);
    let s = common_suffix(&old[p..], &new[p..]);
    let (lo, hi) = (at + p, at + old.len() - s);
    lo < hi && typed.iter().any(|&(a, b)| lo < b && hi > a)
}

/// Typed characters a line rewrite may delete or replace before it counts
/// as rewriting typed text: small corrections (a stray slash, `-` for `_`)
/// are often right, overwriting what was typed is not.
const TYPED_CHANGE_CHARS: usize = 3;

/// Whether rewriting the line `old`, at byte `at` of the file, as `new`
/// deletes or replaces at least `TYPED_CHANGE_CHARS` typed characters (by a
/// character diff). Text added between typed characters rewrites nothing.
fn rewrites_typed_line(old: &str, new: &str, at: usize, typed: &[(usize, usize)]) -> bool {
    if !touches_typed(old, new, at, typed) {
        return false;
    }
    let pos: Vec<usize> = old.char_indices().map(|(i, _)| at + i).collect();
    let a: Vec<char> = old.chars().collect();
    let b: Vec<char> = new.chars().collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(10);
    let changed = similar::capture_diff_slices_deadline(similar::Algorithm::Myers, &a, &b, Some(deadline))
        .into_iter()
        .filter(|op| matches!(op, similar::DiffOp::Delete { .. } | similar::DiffOp::Replace { .. }))
        .flat_map(|op| op.old_range())
        .filter(|&i| typed.iter().any(|&(a, b)| a <= pos[i] && pos[i] < b))
        .count();
    changed >= TYPED_CHANGE_CHARS
}

/// Trim `edit` so it leaves text typed during the live edit (`typed`, byte
/// ranges of `file`) as it is. Changed lines that would rewrite typed text
/// (`rewrites_typed_line`) stay unchanged, deleted lines holding typed text
/// stay, and lines the edit adds next to them stay as an insertion: a
/// header the user just typed, rewritten with a new body below it, keeps
/// the header and gets the body. None when nothing is left.
pub fn protect_typed(file: &str, edit: &Edit, typed: &[(usize, usize)]) -> Option<Edit> {
    let old = &file[edit.start..edit.end];
    if !touches_typed(old, &edit.text, edit.start, typed) {
        return Some(edit.clone());
    }
    let a: Vec<&str> = old.split_inclusive('\n').collect();
    let b: Vec<&str> = edit.text.split_inclusive('\n').collect();
    let mut offs = vec![edit.start];
    for l in &a {
        offs.push(offs.last().unwrap() + l.len());
    }
    let alike = |x: &str, y: &str| {
        let p = common_prefix(x, y);
        p + common_suffix(&x[p..], &y[p..])
    };
    let mut out: Vec<&str> = Vec::new();
    for op in similar::capture_diff_slices(similar::Algorithm::Myers, &a, &b) {
        let (o, ol, n, nl) = match op {
            similar::DiffOp::Equal { old_index, len, .. } => {
                out.extend(&a[old_index..old_index + len]);
                continue;
            }
            similar::DiffOp::Insert { new_index, new_len, .. } => {
                out.extend(&b[new_index..new_index + new_len]);
                continue;
            }
            similar::DiffOp::Delete { old_index, old_len, .. } => (old_index, old_len, 0, 0),
            similar::DiffOp::Replace { old_index, old_len, new_index, new_len } => (old_index, old_len, new_index, new_len),
        };
        if !touches_typed(&a[o..o + ol].concat(), &b[n..n + nl].concat(), offs[o], typed) {
            out.extend(&b[n..n + nl]);
            continue;
        }
        // Pair old and new lines from the head or the tail of the change,
        // whichever pairs more alike lines. A pair keeps the old line if the
        // new one rewrites typed text; unpaired new lines are inserted and
        // unpaired old lines deleted unless they hold typed text.
        let m = ol.min(nl);
        let head: usize = (0..m).map(|k| alike(a[o + k], b[n + k])).sum();
        let tail: usize = (0..m).map(|k| alike(a[o + ol - m + k], b[n + nl - m + k])).sum();
        let (po, pn) = if tail > head { (o + ol - m, n + nl - m) } else { (o, n) };
        let pairs: Vec<&str> = (0..m)
            .map(|k| {
                let (i, j) = (po + k, pn + k);
                if rewrites_typed_line(a[i], b[j], offs[i], typed) { a[i] } else { b[j] }
            })
            .collect();
        let mut extras: Vec<&str> = (o..o + ol)
            .filter(|i| (*i < po || *i >= po + m) && touches_typed(a[*i], "", offs[*i], typed))
            .map(|i| a[i])
            .collect();
        extras.extend((n..n + nl).filter(|j| *j < pn || *j >= pn + m).map(|j| b[j]));
        if po == o && pn == n {
            out.extend(pairs);
            out.extend(extras);
        } else {
            out.extend(extras);
            out.extend(pairs);
        }
    }
    // The smallest whole-line edit with the same result.
    let mut p = 0;
    while p < a.len() && p < out.len() && a[p] == out[p] {
        p += 1;
    }
    let mut s = 0;
    while s < a.len() - p && s < out.len() - p && a[a.len() - 1 - s] == out[out.len() - 1 - s] {
        s += 1;
    }
    if p == a.len() && p == out.len() {
        return None;
    }
    Some(Edit { start: offs[p], end: offs[a.len() - s], text: out[p..out.len() - s].concat() })
}

/// What a partial completion says about the edit before decoding ends.
#[derive(Debug, PartialEq)]
pub enum Provisional {
    /// Nothing to show yet; check again after more tokens.
    Pending,
    /// Nothing can be shown early for this completion.
    Never,
    /// Ghost text at the cursor. While the cursor line is still being
    /// written, the final edit's text starts with this one if the final is
    /// an insertion at the cursor at all. `complete` once the line ended.
    Ghost { edit: Edit, min_logprob: f32, complete: bool },
}

/// Ghost text at the cursor from a partial completion, gated like the final
/// edit on the tokens decoded so far.
///
/// The early-stop splice ends generation as soon as the first changed region
/// is closed by an anchor line, so the first hunk is never known before the
/// final answer. What can be shown earlier is the cursor line while it is
/// written: when everything before the cursor is unchanged, an insertion at
/// the cursor only grows as tokens arrive. Text that may turn out to be the
/// original rest of the cursor line is held back. The final answer can still
/// differ (later tokens fail the confidence gate, the line does not end with
/// the original rest, or a revert); the caller must reconcile.
#[allow(clippy::too_many_arguments)]
pub fn provisional_ghost(
    generated: &str,
    logprobs: &[f32],
    offsets: &[usize],
    built: &Built,
    block: &Block,
    file: &str,
    cursor: usize,
    min_confidence: f32,
) -> Provisional {
    let cleaned = block.text.as_str();
    if cursor < block.start || cursor > block.start + cleaned.len() {
        return Provisional::Never;
    }
    let rel = cursor - block.start;
    let prefill = built.prefill.as_str();
    if prefill.len() > rel || !cleaned.starts_with(prefill) {
        return Provisional::Never;
    }
    let so_far = format!("{prefill}{generated}");
    if so_far.len() <= rel {
        return if cleaned.starts_with(&so_far) { Provisional::Pending } else { Provisional::Never };
    }
    if so_far.as_bytes()[..rel] != cleaned.as_bytes()[..rel] {
        return Provisional::Never; // the rewrite changes text before the cursor
    }
    let line_end = cleaned[rel..].find('\n').map_or(cleaned.len(), |i| rel + i);
    let rest = &cleaned[rel..line_end];
    let tail = &so_far[rel..];
    let (line, complete) = match tail.find('\n') {
        Some(i) => (&tail[..i], true),
        None => (tail, false),
    };
    if line.contains("<|") {
        return Provisional::Never;
    }
    let ghost = if complete {
        match line.strip_suffix(rest) {
            Some(g) => g,
            None => return Provisional::Never, // a replacement, not an insertion
        }
    } else {
        let mut keep = line.len().saturating_sub(rest.len());
        while !line.is_char_boundary(keep) {
            keep -= 1;
        }
        &line[..keep]
    };
    if ghost.is_empty() {
        return if complete { Provisional::Never } else { Provisional::Pending };
    }
    // An insertion starting with the original rest of the line is placed
    // after it by the final hunk selection.
    if rest.chars().next().is_some_and(|c| ghost.starts_with(c)) {
        return Provisional::Never;
    }
    // Close the line with its original rest and rejoin the window, as the
    // splice would, and require the final pipeline to agree.
    let hyp = format!("{}{ghost}{}", &cleaned[..rel], &cleaned[rel..]);
    let applied = normalize_newlines(&hyp);
    let toward_revert = built.prev_sections.iter().any(|s| {
        let s = normalize_newlines(s);
        s != applied && s.starts_with(&normalize_newlines(&hyp[..rel + ghost.len()])) && s.ends_with(&normalize_newlines(&cleaned[rel..]))
    });
    if toward_revert {
        // The line may be completing toward an earlier state of the window.
        return if complete { Provisional::Never } else { Provisional::Pending };
    }
    let processed = process_completion(&hyp[prefill.len()..], "spliced", built, block, file, cursor);
    let edit = Edit { start: cursor, end: cursor, text: ghost.to_string() };
    if processed.edits.first() != Some(&edit) {
        return if complete { Provisional::Never } else { Provisional::Pending };
    }
    // Every token touching the cursor line after the cursor, including
    // held-back text: a superset of what the final gate sees for this line.
    let (lo, hi) = (rel - prefill.len(), rel - prefill.len() + line.len());
    let mut min_logprob = f32::INFINITY;
    for (i, off) in offsets.iter().enumerate() {
        let end = offsets.get(i + 1).copied().unwrap_or(generated.len());
        if end > lo && *off < hi.max(lo + 1) {
            min_logprob = min_logprob.min(logprobs[i]);
        }
    }
    if !min_logprob.is_finite() {
        return Provisional::Pending;
    }
    if min_logprob < min_confidence {
        // The minimum only falls as tokens arrive: the final is withheld too.
        return Provisional::Never;
    }
    Provisional::Ghost { edit, min_logprob, complete }
}

/// Result of Sweep's `find_best_matching_block` for jump-to-edit.
#[derive(Debug, Clone)]
pub struct JumpBlock {
    pub block: Block,
    pub cursor: usize,
    pub reason: &'static str,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub line: usize,
    pub col: usize,
    pub severity: String,
    pub message: String,
}

fn words(text: &str) -> Vec<(&str, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        let w = c.is_alphanumeric() || c == '_';
        match (w, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((&text[s..i], s));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((&text[s..], s));
    }
    out
}

fn diff_tokens(text: &str) -> Vec<&str> {
    // \w+|\s+|[^\w\s]
    let mut out = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let class = |c: char| {
            if c.is_alphanumeric() || c == '_' { 0 } else if c.is_whitespace() { 1 } else { 2 }
        };
        let k = class(c);
        let mut end = i + c.len_utf8();
        if k != 2 {
            while let Some(&(j, d)) = chars.peek() {
                if class(d) != k {
                    break;
                }
                end = j + d.len_utf8();
                chars.next();
            }
        }
        out.push(&text[i..end]);
    }
    out
}

fn added_deleted_words(h: &DiffHunk) -> (Vec<String>, Vec<String>) {
    let (old, new) = h.parts(0);
    let a = diff_tokens(&old);
    let b = diff_tokens(&new);
    let mut added = Vec::new();
    let mut deleted = Vec::new();
    for op in similar::capture_diff_slices(similar::Algorithm::Myers, &a, &b) {
        match op {
            similar::DiffOp::Delete { old_index, old_len, .. } => deleted.extend(&a[old_index..old_index + old_len]),
            similar::DiffOp::Insert { new_index, new_len, .. } => added.extend(&b[new_index..new_index + new_len]),
            similar::DiffOp::Replace { old_index, old_len, new_index, new_len } => {
                deleted.extend(&a[old_index..old_index + old_len]);
                added.extend(&b[new_index..new_index + new_len]);
            }
            similar::DiffOp::Equal { .. } => {}
        }
    }
    let uniq = |v: Vec<&str>| {
        let mut out: Vec<String> = Vec::new();
        for w in v {
            if w.chars().count() > 1 && !out.iter().any(|x| x == w) {
                out.push(w.to_string());
            }
        }
        out
    };
    (uniq(added), uniq(deleted))
}

/// Sweep `find_best_matching_block`: pick a block elsewhere in the file that
/// the recent edits suggest changing next.
pub fn find_jump_block(
    file: &str,
    hunks: &[DiffHunk],
    path: &str,
    cursor: usize,
    diagnostics: &[Diagnostic],
    before: usize,
    after: usize,
) -> Option<JumpBlock> {
    let file_lines: Vec<&str> = file.split_inclusive('\n').collect();
    let line_offset = |idx: usize| -> usize { file_lines[..idx.min(file_lines.len())].iter().map(|l| l.len()).sum() };
    let cursor_line = line_of_offset(file, cursor);
    let local: Vec<&DiffHunk> = hunks.iter().filter(|h| h.display == path).collect();

    // 1. A deleted line (3+ distinct words) that still exists elsewhere.
    for h in &local {
        for (deleted, number) in h.deleted_lines() {
            let distinct: std::collections::HashSet<&str> = words(&deleted).into_iter().map(|w| w.0).collect();
            if distinct.len() < 3 {
                continue;
            }
            for (i, line) in file_lines.iter().enumerate() {
                if i + 1 == number {
                    continue;
                }
                if line.trim().contains(deleted.as_str()) {
                    let end = (i + 4).min(file_lines.len());
                    return Some(make_jump(before, after, &file_lines, i, end, line_offset(i), "deleted_line_match", None));
                }
            }
        }
    }

    // 2. A renamed or newly used identifier that occurs elsewhere.
    let file_words = words(file);
    let file_set: std::collections::HashSet<&str> = file_words.iter().map(|w| w.0).collect();
    let mut added = Vec::new();
    let mut deleted = Vec::new();
    for h in local.iter().rev() {
        let (a, d) = added_deleted_words(h);
        added = a;
        deleted = d.into_iter().filter(|w| file_set.contains(w.as_str())).collect();
        if deleted.len() == 1 {
            break;
        }
    }
    let far = |offset: usize| line_of_offset(file, offset).abs_diff(cursor_line) > 10;
    let query: Option<String> = if deleted.len() == 1 {
        Some(deleted[0].clone())
    } else {
        added.iter().find(|w| {
            let occ: Vec<usize> = file_words.iter().filter(|x| x.0 == w.as_str()).map(|x| x.1).collect();
            occ.len() > 1 && occ.len() <= 5 && occ.iter().any(|o| far(*o))
        }).cloned()
    };
    let error = diagnostics
        .iter()
        .filter(|d| d.severity.eq_ignore_ascii_case("error") && d.line.abs_diff(cursor_line) > 10)
        .min_by_key(|d| d.line.abs_diff(cursor_line));
    if let Some(d) = error {
        let i = d.line.min(file_lines.len().saturating_sub(1));
        let message = format!("{} at line {}:\n{}", d.message, d.line, file_lines.get(i).unwrap_or(&"").trim_end_matches('\n'));
        return Some(make_jump(before, after, &file_lines, i, i + 1, line_offset(i), "diagnostic", Some(message)));
    }
    if let Some(q) = query {
        let best = file_words
            .iter()
            .filter(|w| w.0 == q && far(w.1))
            .min_by_key(|w| w.1.abs_diff(cursor));
        if let Some(w) = best {
            let i = line_of_offset(file, w.1);
            return Some(make_jump(before, after, &file_lines, i, i + 1, line_offset(i), "identifier_match", None));
        }
    }
    // 3. The block right after the current one.
    let next_line = cursor_line + 1 + 6;
    if next_line >= file_lines.len() {
        return None;
    }
    let end = (next_line + 6).min(file_lines.len());
    Some(make_jump(before, after, &file_lines, next_line, end, line_offset(next_line), "block_after_cursor", None))
}

fn make_jump(
    before: usize,
    after: usize,
    file_lines: &[&str],
    start: usize,
    end: usize,
    start_offset: usize,
    reason: &'static str,
    diagnostic: Option<String>,
) -> JumpBlock {
    let prefix_start = start.saturating_sub(before);
    let retrieved = end - start;
    let suffix_lines = (after + 1).saturating_sub(retrieved);
    let block_end = (end + suffix_lines).min(file_lines.len());
    // Token cap as in truncate_code_block_by_tokens.
    let max_chars = ((OUTPUT_MAX_TOKENS / 2) as f64 * CHARS_PER_TOKEN) as usize;
    let mut capped = block_end;
    let mut chars = 0;
    for (i, l) in file_lines.iter().enumerate().take(block_end).skip(start) {
        chars += l.len();
        if chars > max_chars && i > start {
            capped = i;
            break;
        }
    }
    let block_start = start_offset - file_lines[prefix_start..start].iter().map(|l| l.len()).sum::<usize>();
    let text = file_lines[prefix_start..capped].concat();
    let first_len = file_lines[start].trim_end_matches('\n').len();
    JumpBlock {
        block: Block { text, start: block_start, start_line: prefix_start },
        cursor: start_offset + first_len,
        reason,
        diagnostic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn opts() -> PromptOptions {
        PromptOptions::default()
    }

    #[test]
    fn block_window_matches_sweep_shape() {
        let content = "l0\nl1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\n";
        let cursor = content.find("l4").unwrap() + 1;
        let b = block_at_cursor(content, cursor, 2, 5);
        assert_eq!(b.text, "l2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\n");
        assert_eq!(b.start_line, 2);
        assert_eq!(&content[b.start..b.start + 2], "l2");
    }

    #[test]
    fn through_cursor_prefill_stops_at_word_start() {
        let text = "fn f() {\n    let value = fo";
        let p = compute_prefill(text, text.len(), Prefill::ThroughCursor);
        assert_eq!(p, "fn f() {\n    let value =");
        let text = "fn f() {\n    re";
        assert_eq!(compute_prefill(text, text.len(), Prefill::ThroughCursor), "fn f() {\n");
        let text = "call(a, ";
        assert_eq!(compute_prefill(text, text.len(), Prefill::ThroughCursor), "call(a,");
    }

    #[test]
    fn volatile_context_follows_history() {
        let content = "fn main() {\n    let x = 1;\n}\n";
        let cursor = content.find("let").unwrap();
        let block = block_at_cursor(content, cursor, 2, 5);
        let chunk = |path: &str, source: &'static str| Chunk {
            path: path.into(),
            start_line: 1,
            end_line: 1,
            content: "x".into(),
            source,
            score: 0.0,
        };
        let retrieval = [chunk("lib.rs", "definition"), chunk("diagnostics", "diagnostics")];
        let hunks = [DiffHunk {
            path: "other.rs".into(),
            display: "other.rs".into(),
            old_start: 0,
            new_start: 0,
            ctx_before: vec![],
            old: vec!["let a = 1;".into()],
            new: vec!["let a = 2;".into()],
            ctx_after: vec![],
            group: 0,
        }];
        let mut o = opts();
        o.context_order = ContextOrder::VolatileLate;
        let built = build_prompt(&PromptInput {
            path: "src/main.rs",
            content,
            original: content,
            cursor,
            block: &block,
            hunks: &hunks,
            file_chunks: &[],
            retrieval: &retrieval,
            opts: &o,
            live: 0,
            session_base: None,
            initial_start: None,
        });
        let p = &built.prompt;
        let (def, diag, window) = (p.find("lib.rs").unwrap(), p.find("<|file_sep|>diagnostics").unwrap(), p.find("original/").unwrap());
        let history = p.find("original:").unwrap();
        assert!(def < history && history < diag && diag < window, "{p}");
    }

    #[test]
    fn block_skips_leading_blank_lines() {
        let content = "a\n\n\nb\nc\nd\ne\nf\ng\nh\n";
        let cursor = content.find("b").unwrap();
        let b = block_at_cursor(content, cursor, 2, 5);
        assert!(b.text.starts_with("b\n"));
    }

    #[test]
    fn prompt_layout() {
        let content = "fn main() {\n    let x = 1;\n    println!(\"{}\", x);\n}\n";
        let cursor = content.find("let").unwrap();
        let block = block_at_cursor(content, cursor, 2, 5);
        let o = opts();
        let built = build_prompt(&PromptInput {
            path: "src/main.rs",
            content,
            original: content,
            cursor,
            block: &block,
            hunks: &[],
            file_chunks: &[],
            retrieval: &[],
            opts: &o,
            live: 0,
            session_base: None,
            initial_start: None,
        });
        assert!(built.prompt.starts_with("<|file_sep|>src/main.rs\nfn main()"));
        assert!(built.prompt.contains("<|file_sep|>current/src/main.rs:1:4\nfn main() {\n    <|cursor|>let x"));
        assert!(built.prompt.ends_with("<|file_sep|>updated/src/main.rs:1:4\nfn main() {\n"));
        assert_eq!(built.prefill, "fn main() {\n");
    }

    #[test]
    fn prev_section_reverts_recent_edit() {
        let old = "fn f(a: i32) {\n    a\n}\n";
        let new = "fn f(a: i64) {\n    a\n}\n";
        let hunks = crate::history::diff_hunks(Path::new("/x/f.rs"), "f.rs", old, new);
        let (recent, prev, sections) = recent_changes_and_prev("f.rs", &hunks, new, 6, DiffFormat::Server, false);
        assert_eq!(prev, old);
        assert_eq!(sections.len(), 1);
        assert!(recent.starts_with("<|file_sep|>f.rs:1:2\noriginal:\nfn f(a: i32) {\n    a\nupdated:\nfn f(a: i64) {\n    a"));
    }

    #[test]
    fn hide_active_edit_keeps_history_stable() {
        let old = "fn f(a: i32) {\n    a\n}\n";
        let new = "fn f(a: i64) {\n    a\n}\n";
        let hunks = crate::history::diff_hunks(Path::new("/x/f.rs"), "f.rs", old, new);
        let (recent, prev, _) = recent_changes_and_prev("f.rs", &hunks, new, 6, DiffFormat::Server, true);
        assert_eq!(recent, "");
        assert_eq!(prev, old);
    }

    #[test]
    fn ghost_text_insertion_at_cursor() {
        let file = "let x = foo(\nlet y = 2;\n";
        let cursor = "let x = foo(".len();
        let block = block_at_cursor(file, cursor, 2, 5);
        let completion = "let x = foo(bar);\nlet y = 2;\n";
        let edits = select_best_hunks(completion, &block.text, block.start, file, cursor);
        assert_eq!(edits, vec![Edit { start: cursor, end: cursor, text: "bar);".into() }]);
    }

    #[test]
    fn replacement_hunk_after_cursor() {
        let file = "a = 1\nb = old\nc = 3\n";
        let cursor = 0;
        let block = block_at_cursor(file, cursor, 2, 5);
        let completion = "a = 1\nb = new\nc = 3\n";
        let edits = select_best_hunks(completion, &block.text, block.start, file, cursor);
        assert_eq!(edits.len(), 1);
        assert_eq!(&file[edits[0].start..edits[0].end], "b = old\n");
        assert_eq!(edits[0].text, "b = new\n");
    }

    #[test]
    fn jump_finds_renamed_identifier() {
        let mut file = String::from("fn compute(total: i32) -> i32 {\n    total\n}\n");
        for i in 0..20 {
            file.push_str(&format!("// filler {i}\n"));
        }
        file.push_str("let v = compute_old(3);\n");
        let old = file.replace("fn compute(", "fn compute_old(");
        let hunks = crate::history::diff_hunks(Path::new("/x/a.rs"), "a.rs", &old, &file);
        let j = find_jump_block(&file, &hunks, "a.rs", 3, &[], 2, 5).unwrap();
        assert_eq!(j.reason, "identifier_match");
        assert!(j.block.text.contains("compute_old(3)"));
    }

    #[test]
    fn splice_after_reconvergence() {
        let cleaned = "fn a() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n    tail();\n}\n";
        // Still copying: no splice.
        assert_eq!(splice_rest("fn a() {\n", "    let x = 1;\n", cleaned), None);
        // Changed line then two matching lines: splice the rest.
        let rest = splice_rest("fn a() {\n", "    let x = 10;\n    let y = 2;\n    let z = 3;\n", cleaned);
        assert_eq!(rest.as_deref(), Some("    tail();\n}\n"));
    }

    #[test]
    fn confidence_covers_changed_tokens() {
        let c = change_confidence("", "ab XY cd", "ab cd", &[-0.1, -0.2, -2.0, -0.1], &[0, 2, 3, 5]);
        assert_eq!(c.changed_tokens, 2);
        assert!((c.min_logprob + 2.0).abs() < 1e-6);
    }

    #[test]
    fn process_rejects_revert() {
        let old = "x = 1\n";
        let new = "x = 2\n";
        let hunks = crate::history::diff_hunks(Path::new("/x/a.py"), "a.py", old, new);
        let block = block_at_cursor(new, 0, 2, 5);
        let o = opts();
        let built = build_prompt(&PromptInput {
            path: "a.py",
            content: new,
            original: old,
            cursor: 0,
            block: &block,
            hunks: &hunks,
            file_chunks: &[],
            retrieval: &[],
            opts: &o,
            live: 0,
            session_base: None,
            initial_start: None,
        });
        let p = process_completion("x = 1\n", "stop", &built, &block, new, 0);
        assert_eq!(p.exit_reason, "revert_detected");
    }

    /// Prompt for `file` with the cursor at the end of `marker`, after an
    /// edit from `old` when given.
    fn staged(file: &str, marker: &str, old: Option<&str>) -> (Built, Block, usize) {
        let cursor = file.find(marker).unwrap() + marker.len();
        let block = block_at_cursor(file, cursor, 2, 5);
        let hunks = old.map(|o| crate::history::diff_hunks(Path::new("/x/a.rs"), "a.rs", o, file)).unwrap_or_default();
        let o = opts();
        let built = build_prompt(&PromptInput {
            path: "a.rs",
            content: file,
            original: old.unwrap_or(file),
            cursor,
            block: &block,
            hunks: &hunks,
            file_chunks: &[],
            retrieval: &[],
            opts: &o,
            live: 0,
            session_base: None,
            initial_start: None,
        });
        (built, block, cursor)
    }

    /// One token per byte, all at `lp` except `low` at byte `low_at`.
    fn early(generated: &str, s: &(Built, Block, usize), file: &str, low_at: Option<usize>) -> Provisional {
        let offsets: Vec<usize> = (0..generated.len()).collect();
        let logprobs: Vec<f32> = offsets.iter().map(|i| if Some(*i) == low_at { -3.0 } else { -0.1 }).collect();
        provisional_ghost(generated, &logprobs, &offsets, &s.0, &s.1, file, s.2, -1.0)
    }

    fn ghost(p: &Provisional) -> Option<(&str, bool)> {
        match p {
            Provisional::Ghost { edit, complete, .. } => Some((edit.text.as_str(), *complete)),
            _ => None,
        }
    }

    const FILE: &str = "fn f() {\n    let v = 1;\n    let x = foo(\n    let y = 2;\n}\n";

    #[test]
    fn provisional_ghost_grows_with_the_cursor_line() {
        let s = staged(FILE, "let x = foo(", None);
        assert_eq!(s.0.prefill, "fn f() {\n");
        // Still copying the window up to the cursor.
        assert_eq!(early("    let v = 1;\n    let x = fo", &s, FILE, None), Provisional::Pending);
        let p = early("    let v = 1;\n    let x = foo(ba", &s, FILE, None);
        assert_eq!(ghost(&p), Some(("ba", false)));
        if let Provisional::Ghost { edit, .. } = &p {
            assert_eq!((edit.start, edit.end), (s.2, s.2));
        }
        let p = early("    let v = 1;\n    let x = foo(bar);\n    let", &s, FILE, None);
        assert_eq!(ghost(&p), Some(("bar);", true)));
        // The provisional text is what the final pipeline shows for the
        // completed rewrite.
        let done = process_completion("    let v = 1;\n    let x = foo(bar);\n    let y = 2;\n}\n", "spliced", &s.0, &s.1, FILE, s.2);
        assert_eq!(done.edits[0], Edit { start: s.2, end: s.2, text: "bar);".into() });
    }

    #[test]
    fn provisional_requires_unchanged_text_before_cursor() {
        let s = staged(FILE, "let x = foo(", None);
        assert_eq!(early("    let v = 2;\n", &s, FILE, None), Provisional::Never);
        assert_eq!(early("    let v = 1;\n    let z", &s, FILE, None), Provisional::Never);
        // The cursor line is closed without an insertion.
        assert_eq!(early("    let v = 1;\n    let x = foo(\n", &s, FILE, None), Provisional::Never);
    }

    #[test]
    fn provisional_gates_on_tokens_so_far() {
        let s = staged(FILE, "let x = foo(", None);
        let generated = "    let v = 1;\n    let x = foo(bar";
        let at = generated.len() - 2;
        assert_eq!(early(generated, &s, FILE, Some(at)), Provisional::Never);
        // A weak token before the cursor does not count.
        assert!(ghost(&early(generated, &s, FILE, Some(3))).is_some());
    }

    #[test]
    fn provisional_holds_back_the_rest_of_the_line() {
        let file = "fn f() {\n    let v = 1;\n    call()\n    let y = 2;\n}\n";
        let s = staged(file, "call(", None);
        // "a)" may be the whole insertion "a" plus the original ")".
        assert_eq!(ghost(&early("    let v = 1;\n    call(a)", &s, file, None)), Some(("a", false)));
        assert_eq!(ghost(&early("    let v = 1;\n    call(a, b", &s, file, None)), Some(("a, ", false)));
        assert_eq!(ghost(&early("    let v = 1;\n    call(a, b)\n", &s, file, None)), Some(("a, b", true)));
        // The line ends without the original rest: a replacement.
        assert_eq!(early("    let v = 1;\n    call(a, b]\n", &s, file, None), Provisional::Never);
        // An insertion starting with the rest is placed after it by the final.
        assert_eq!(early("    let v = 1;\n    call())", &s, file, None), Provisional::Never);
    }

    #[test]
    fn provisional_ghosts_are_prefixes_of_the_final_edit() {
        let mid = "fn f() {\n    let v = 1;\n    call()\n    let y = 2;\n}\n";
        for (file, marker, generated) in [
            (FILE, "let x = foo(", "    let v = 1;\n    let x = foo(bar(1), \"é\");\n    let y = 2;\n}\n"),
            (mid, "call(", "    let v = 1;\n    call(a(b), c)\n    let y = 2;\n}\n"),
        ] {
            let s = staged(file, marker, None);
            let done = process_completion(generated, "spliced", &s.0, &s.1, file, s.2);
            let last = done.edits[0].text.clone();
            let mut seen = Vec::new();
            for cut in (0..=generated.len()).filter(|i| generated.is_char_boundary(*i)) {
                if let Some((g, _)) = ghost(&early(&generated[..cut], &s, file, None)) {
                    assert!(last.starts_with(g), "{g:?} is not a prefix of {last:?}");
                    seen.push(g.to_string());
                }
            }
            assert_eq!(seen.last(), Some(&last));
        }
    }

    #[test]
    fn provisional_waits_on_a_possible_revert() {
        let old = "fn f() {\n    let v = 1;\n    let x = foo(bar);\n    let y = 2;\n}\n";
        let s = staged(FILE, "let x = foo(", Some(old));
        assert!(!s.0.prev_sections.is_empty());
        assert_eq!(early("    let v = 1;\n    let x = foo(ba", &s, FILE, None), Provisional::Pending);
        assert_eq!(early("    let v = 1;\n    let x = foo(bar);\n", &s, FILE, None), Provisional::Never);
        // A different insertion is shown.
        assert_eq!(ghost(&early("    let v = 1;\n    let x = foo(baz", &s, FILE, None)), Some(("baz", false)));
    }

    fn lines(range: std::ops::Range<usize>) -> String {
        range.map(|i| format!("l{i}\n")).collect()
    }

    #[test]
    fn hidden_hunks_do_not_take_history_slots() {
        use crate::history::Tracker;
        use std::time::{Duration, Instant};
        let p = Path::new("/x/f.rs");
        let mut t = Tracker::default();
        let now = Instant::now();
        let base = lines(0..60);
        let mut text = base.clone();
        t.observe_at(p, "f.rs", &text, 0, false, now);
        for (k, line) in [1, 20, 40, 50].iter().enumerate() {
            text = text.replace(&format!("l{line}\n"), &format!("L{line}\n"));
            t.observe_at(p, "f.rs", &text, *line, false, now + Duration::from_secs(10 * (k as u64 + 1)));
        }
        let (hunks, live) = t.recent_split_at(3 + 4, now + Duration::from_secs(41));
        let cursor = text.find("L50").unwrap();
        let block = block_at_cursor(&text, cursor, 2, 3);
        let mut o = opts();
        o.max_hunks = 3;
        o.hide_active_edit = true;
        o.stable_history = true;
        let built = build_prompt(&PromptInput {
            path: "f.rs",
            content: &text,
            original: &base,
            cursor,
            block: &block,
            hunks: &hunks,
            file_chunks: &[],
            retrieval: &[],
            opts: &o,
            live,
            session_base: None,
            initial_start: None,
        });
        // The window edit is hidden and three older edits remain visible.
        assert_eq!(built.hunks_used, 3);
        assert_eq!(built.shown_from, Some(hunks[0].group));
        let history = &built.prompt[..built.prompt.find("original/").unwrap()];
        assert!(history.contains("L1") && history.contains("L20") && history.contains("L40"));
        assert!(!history.contains("L50"));
        assert!(built.prompt.contains("original/f.rs:49:54\nl48\nl49\nl50\n"));
    }

    #[test]
    fn positional_revert_picks_the_edited_occurrence() {
        let old = "{\n    total += 1;\n}\n{\n    total += 2;\n}\n";
        let new = "{\n    total += 1;\n}\n{\n    total += 1;\n}\n";
        let hunks = crate::history::diff_hunks(Path::new("/x/f.rs"), "f.rs", old, new);
        // The substring revert rewrites the first, unedited block.
        let (_, prev, _) = recent_changes_and_prev("f.rs", &hunks, new, 6, DiffFormat::Server, true);
        assert_ne!(prev, old);
        let block = Block { text: new.into(), start: 0, start_line: 0 };
        let (history, prev, sections, used, _) = stable_changes_and_prev("f.rs", &hunks, new, &block, 6, DiffFormat::Server, true);
        assert_eq!(prev, old);
        assert_eq!((history.as_str(), sections.len(), used), ("", 1, 0));
    }

    #[test]
    fn positional_revert_chains_groups_and_hides_overlaps() {
        use crate::history::Tracker;
        use std::time::{Duration, Instant};
        let p = Path::new("/x/f.rs");
        let mut t = Tracker::default();
        let now = Instant::now();
        let base = lines(0..20);
        t.observe_at(p, "f.rs", &base, 0, false, now);
        let first = base.replace("l5\n", "A1\nA2\nA3\n");
        t.observe_at(p, "f.rs", &first, 5, false, now + Duration::from_secs(1));
        let second = first.replace("l8\n", "B\n");
        t.observe_at(p, "f.rs", &second, 10, false, now + Duration::from_secs(10));
        let (hunks, live) = t.recent_split_at(6, now + Duration::from_secs(11));
        assert_eq!((hunks.len(), live), (2, 1));
        // The window starts inside the first edit (at A3) and holds the second.
        let cursor = second.find("l7").unwrap();
        let block = block_at_cursor(&second, cursor, 2, 3);
        assert_eq!(block.text, "A3\nl6\nl7\nB\nl9\nl10\n");
        let (history, prev, sections, used, _) = stable_changes_and_prev("f.rs", &hunks, &second, &block, 6, DiffFormat::Server, true);
        assert_eq!(prev, lines(5..11));
        assert_eq!((history.as_str(), sections.len(), used), ("", 2, 0));
        // Without hiding both stay in the history, oldest first.
        let (history, _, _, used, _) = stable_changes_and_prev("f.rs", &hunks, &second, &block, 6, DiffFormat::Server, false);
        assert_eq!(used, 2);
        assert!(history.find("A1").unwrap() < history.find("\nB\n").unwrap());
    }

    #[test]
    fn session_window_reverts_to_base() {
        let base = lines(0..10);
        let content = base.replace("l4\n", "X\nY\n");
        let cursor = content.find('Y').unwrap();
        let block = block_at_cursor(&content, cursor, 2, 3);
        assert_eq!(window_from_base(&base, &content, &block), "l3\nl4\nl5\nl6\nl7\n");
    }

    #[test]
    fn shown_from_is_the_oldest_history_or_window_group() {
        use crate::history::Tracker;
        use std::time::{Duration, Instant};
        let p = Path::new("/x/f.rs");
        let mut t = Tracker::default();
        let now = Instant::now();
        let base = lines(0..60);
        let mut text = base.clone();
        t.observe_at(p, "f.rs", &text, 0, false, now);
        for (k, line) in [1, 20, 40, 50].iter().enumerate() {
            text = text.replace(&format!("l{line}\n"), &format!("L{line}\n"));
            t.observe_at(p, "f.rs", &text, *line, false, now + Duration::from_secs(10 * (k as u64 + 1)));
        }
        let (hunks, live) = t.recent_split_at(6, now + Duration::from_secs(41));
        let mut o = opts();
        o.max_hunks = 2;
        o.hide_active_edit = true;
        o.stable_history = true;
        let build = |cursor: usize| {
            let block = block_at_cursor(&text, cursor, 2, 3);
            build_prompt(&PromptInput {
                path: "f.rs",
                content: &text,
                original: &base,
                cursor,
                block: &block,
                hunks: &hunks,
                file_chunks: &[],
                retrieval: &[],
                opts: &o,
                live,
                session_base: None,
                initial_start: None,
            })
            .shown_from
        };
        // The live edit is in the window; the history shows L20 and L40.
        assert_eq!(build(text.find("L50").unwrap()), Some(hunks[1].group));
        // Away from every edit the history shows the newest two.
        assert_eq!(build(text.find("l10").unwrap()), Some(hunks[2].group));
    }

    #[test]
    fn typed_header_keeps_its_text_and_gets_the_body() {
        let file = "tasks:\n  api:\n    cmd: x\n    deps: [db]\n  web:\n";
        let at = file.find("  web:").unwrap();
        let typed = [(at, at + "  web:".len())];
        // The model clones the previous block over the typed header.
        let clone = Edit { start: at, end: file.len(), text: "  api:\n    cmd: x\n    deps: [db]\n".into() };
        let trimmed = protect_typed(file, &clone, &typed).unwrap();
        assert_eq!(trimmed, Edit { start: file.len(), end: file.len(), text: "    cmd: x\n    deps: [db]\n".into() });
        // A rename of typed text leaves nothing.
        let rename = Edit { start: at, end: file.len(), text: "  api:\n".into() };
        assert_eq!(protect_typed(file, &rename, &typed), None);
        // Small corrections and text added between typed characters stay.
        let fix = Edit { start: at, end: file.len(), text: "  wbe:\n".into() };
        assert_eq!(protect_typed(file, &fix, &typed).as_ref(), Some(&fix));
        let fill = Edit { start: at, end: file.len(), text: "  w(e)b: x\n".into() };
        assert_eq!(protect_typed(file, &fill, &typed).as_ref(), Some(&fill));
        // Completing typed text and editing other lines are untouched.
        let append = Edit { start: at, end: file.len(), text: "  web: {}\n".into() };
        assert_eq!(protect_typed(file, &append, &typed).as_ref(), Some(&append));
        let other = Edit { start: 0, end: 7, text: "jobs:\n".into() };
        assert_eq!(protect_typed(file, &other, &typed).as_ref(), Some(&other));
        // New lines above the typed one, completing it: kept whole.
        let file = "a: 1\n    \n";
        let at = file.find("    ").unwrap();
        let typed = [(at - 1, at + 4)];
        let above = Edit { start: at, end: file.len(), text: "  b:\n    c: 2\n".into() };
        assert_eq!(protect_typed(file, &above, &typed).as_ref(), Some(&above));
        // Lines removed with typed text: the typed line stays, the rest goes.
        let file = "x\ny\nz\n";
        let typed = [(2, 3)];
        let gone = Edit { start: 0, end: 6, text: "x\n".into() };
        assert_eq!(protect_typed(file, &gone, &typed), Some(Edit { start: 4, end: 6, text: String::new() }));
    }

    #[test]
    fn chunk_choice_has_hysteresis() {
        let mut o = opts();
        o.whole_file_lines = 120;
        o.chunk_lines = 120;
        assert_eq!(chunk_start(400, 150, &o, None), 120);
        assert_eq!(chunk_start(400, 150, &o, Some(60)), 60);
        assert_eq!(chunk_start(400, 175, &o, Some(60)), 120);
        assert_eq!(chunk_start(400, 65, &o, Some(60)), 0);
        assert_eq!(chunk_start(400, 395, &o, Some(300)), 300);
        let file = lines(0..400);
        assert_eq!(lines_from(&file, 60, &o).lines().next(), Some("l60"));
        assert_eq!(lines_from(&file, 60, &o).lines().count(), 120);
    }
}
