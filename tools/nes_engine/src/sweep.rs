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
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptOptions {
    pub line_numbers: LineNumbers,
    pub diff_format: DiffFormat,
    pub prefill: Prefill,
    pub cursor_marker: bool,
    pub lines_before: usize,
    pub lines_after: usize,
    /// Whole original file is used when it has at most this many lines.
    pub whole_file_lines: usize,
    /// Otherwise a stride-aligned chunk of this many lines around the cursor.
    pub chunk_lines: usize,
    pub max_retrieval_tokens: usize,
    pub max_hunks: usize,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            line_numbers: LineNumbers::Absolute,
            diff_format: DiffFormat::Server,
            prefill: Prefill::FirstLine,
            cursor_marker: true,
            lines_before: NUM_LINES_BEFORE,
            lines_after: NUM_LINES_AFTER,
            whole_file_lines: 240,
            chunk_lines: 240,
            max_retrieval_tokens: 2048,
            max_hunks: 6,
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
}

/// Sweep `format_recent_changes_and_prev_section`, restricted to hunks of the
/// current file when reverting (matching is substring-based).
fn recent_changes_and_prev(
    path: &str,
    hunks: &[DiffHunk],
    cleaned_block: &str,
    max_hunks: usize,
    diff_format: DiffFormat,
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
    let mut result = String::new();
    let skip = hunks.len().saturating_sub(max_hunks);
    for h in &hunks[skip..] {
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

fn compute_prefill(cleaned: &str, rel_cursor: usize, mode: Prefill) -> String {
    let before = &cleaned[..rel_cursor];
    match mode {
        Prefill::None => return String::new(),
        Prefill::BeforeCursorLine => {
            return before.rfind('\n').map_or(String::new(), |i| before[..=i].to_string());
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
    let (recent, mut prev, prev_sections) =
        recent_changes_and_prev(input.path, input.hunks, cleaned, input.opts.max_hunks, input.opts.diff_format);
    let prefill = compute_prefill(cleaned, rel_cursor, input.opts.prefill);
    let retrieval = pack_from_end(input.retrieval, input.opts.max_retrieval_tokens);
    let retrieval_text: String = retrieval.iter().map(|c| format!("\n{}", c.render())).collect();
    if code_block.ends_with('\n') && prev.ends_with('\n') {
        code_block.pop();
        prev.pop();
    }
    let initial = lines_around_cursor(input.original, input.cursor.min(input.original.len()), input.opts);
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
        hunks_used: input.hunks.len().min(input.opts.max_hunks),
        initial_file_lines: initial.lines().count(),
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
    if out_lines.len() < 3 {
        return None; // need the changed line plus two anchor lines
    }
    let tail = &out_lines[out_lines.len() - 2..];
    if tail.iter().map(|l| l.trim().len()).sum::<usize>() < 8 {
        return None; // braces and blank lines are ambiguous anchors
    }
    let div_line = cleaned[..p].matches('\n').count();
    let block_lines: Vec<&str> = cleaned.split_inclusive('\n').collect();
    let mut found = None;
    for j in div_line..block_lines.len().saturating_sub(1) {
        if block_lines[j] == tail[0] && block_lines[j + 1] == tail[1] {
            if found.is_some() {
                return None; // ambiguous anchor
            }
            found = Some(j + 2);
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
        let (recent, prev, sections) = recent_changes_and_prev("f.rs", &hunks, new, 6, DiffFormat::Server);
        assert_eq!(prev, old);
        assert_eq!(sections.len(), 1);
        assert!(recent.starts_with("<|file_sep|>f.rs:1:2\noriginal:\nfn f(a: i32) {\n    a\nupdated:\nfn f(a: i64) {\n    a"));
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
        });
        let p = process_completion("x = 1\n", "stop", &built, &block, new, 0);
        assert_eq!(p.exit_reason, "revert_detected");
    }
}
