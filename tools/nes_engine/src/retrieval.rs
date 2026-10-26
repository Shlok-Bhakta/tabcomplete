//! Retrieval: choose the few cross-file snippets worth prompt tokens.
//!
//! Identifiers near the cursor and in recent edits are looked up in the
//! workspace index (Zed's edit-prediction context and Sweep's IDE plugins both
//! resolve definitions of nearby identifiers). Candidates are scored by
//! proximity, edit recency and ambiguity, and the selection is kept sticky
//! across requests so the KV cache prefix survives cursor movement.

use crate::history::DiffHunk;
use crate::index::Workspace;
use crate::sweep::{Chunk, Diagnostic, MAX_RETRIEVAL_CHUNK_LINES};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const STOPWORDS: &[&str] = &[
    "self", "Self", "this", "let", "mut", "const", "return", "true", "false", "None", "Some", "Ok", "Err", "fn", "def",
    "function", "var", "if", "else", "for", "while", "in", "of", "and", "or", "not", "pub", "use", "import", "from",
    "class", "struct", "impl", "new", "type", "string", "String", "str", "int", "i32", "i64", "u32", "u64", "usize",
    "bool", "Vec", "Option", "Result", "local", "end", "then", "nil", "null", "undefined", "async", "await",
];

pub struct RetrievalInput<'a> {
    pub path: &'a Path,
    pub content: &'a str,
    pub cursor_line: usize,
    /// Lines of the current file already present in the prompt.
    pub visible: (usize, usize),
    pub hunks: &'a [DiffHunk],
    pub recent_files: &'a [PathBuf],
    pub previous: &'a [String],
    pub max_chunks: usize,
}

fn identifiers(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.len() >= 2 && !w.chars().next().unwrap().is_ascii_digit() && !STOPWORDS.contains(w))
}

pub fn chunk_key(c: &Chunk) -> String {
    format!("{}:{}", c.path, c.start_line)
}

/// Rank definition snippets for the current request.
pub fn definition_chunks(ws: &Workspace, input: &RetrievalInput<'_>) -> Vec<Chunk> {
    let mut weights: HashMap<&str, f32> = HashMap::new();
    let lines: Vec<&str> = input.content.lines().collect();
    let lo = input.cursor_line.saturating_sub(8);
    let hi = (input.cursor_line + 9).min(lines.len());
    for (i, line) in lines.iter().enumerate().take(hi).skip(lo) {
        let distance = i.abs_diff(input.cursor_line) as f32;
        let w = if distance == 0.0 { 2.0 } else { 1.0 / (1.0 + distance / 2.0) };
        for id in identifiers(line) {
            *weights.entry(id).or_default() += w;
        }
    }
    // Owned strings from edits outlive this function's borrow of `input`.
    let edit_text: Vec<String> = input
        .hunks
        .iter()
        .rev()
        .take(3)
        .map(|h| h.parts(0).1)
        .collect();
    for (rank, text) in edit_text.iter().enumerate() {
        let w = 1.5 / (1.0 + rank as f32);
        for id in identifiers(text) {
            *weights.entry(id).or_default() += w;
        }
    }
    let mut scored: HashMap<(PathBuf, usize), (f32, Chunk)> = HashMap::new();
    for (id, w) in weights {
        let defs: Vec<_> = ws.lookup(id).collect();
        if defs.is_empty() || defs.len() > 12 {
            continue;
        }
        let share = w / (defs.len() as f32).sqrt();
        for (file, def) in defs {
            let same_file = file.path == input.path;
            if same_file {
                let overlaps = def.start_line <= input.visible.1 && def.end_line >= input.visible.0;
                let encloses = def.start_line <= input.cursor_line && def.end_line >= input.cursor_line;
                if overlaps || encloses {
                    continue;
                }
            }
            let mut score = share;
            if input.recent_files.iter().any(|p| p == &file.path) {
                score *= 1.3;
            }
            if file.path.parent() == input.path.parent() {
                score *= 1.1;
            }
            let key = (file.path.clone(), def.start_line);
            let entry = scored.entry(key).or_insert_with(|| {
                let content: String = def.snippet.lines().take(MAX_RETRIEVAL_CHUNK_LINES).collect::<Vec<_>>().join("\n");
                let n = content.lines().count().max(1);
                (
                    0.0,
                    Chunk {
                        path: file.rel.clone(),
                        start_line: def.start_line + 1,
                        end_line: def.start_line + n,
                        content,
                        source: "definition",
                        score: 0.0,
                    },
                )
            });
            entry.0 += score;
        }
    }
    let mut ranked: Vec<(f32, Chunk)> = scored.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.path.cmp(&b.1.path)));
    for (s, c) in ranked.iter_mut() {
        c.score = *s;
    }
    let pool: Vec<Chunk> = ranked.into_iter().take(input.max_chunks * 2).map(|(_, c)| c).collect();
    // Sticky: keep previously selected chunks that are still competitive.
    let mut chosen: Vec<Chunk> = Vec::new();
    for key in input.previous {
        if let Some(c) = pool.iter().find(|c| &chunk_key(c) == key) {
            if chosen.len() < input.max_chunks {
                chosen.push(c.clone());
            }
        }
    }
    for c in &pool {
        if chosen.len() >= input.max_chunks {
            break;
        }
        if !chosen.iter().any(|x| chunk_key(x) == chunk_key(c)) {
            chosen.push(c.clone());
        }
    }
    chosen
}

/// Diagnostics near the cursor in Sweep's VS Code text format.
pub fn diagnostics_chunk(path: &str, diagnostics: &[Diagnostic], cursor_line: usize) -> Option<Chunk> {
    let near: Vec<&Diagnostic> = diagnostics
        .iter()
        .filter(|d| d.line.abs_diff(cursor_line) <= 15)
        .filter(|d| matches!(d.severity.to_ascii_lowercase().as_str(), "error" | "warning"))
        .take(5)
        .collect();
    if near.is_empty() {
        return None;
    }
    let content: String = near
        .iter()
        .map(|d| format!("{}:{}:{}: {}: {}", path, d.line + 1, d.col + 1, d.severity.to_ascii_lowercase(), d.message.lines().next().unwrap_or("")))
        .collect::<Vec<_>>()
        .join("\n");
    Some(Chunk { path: "diagnostics".into(), start_line: 1, end_line: near.len(), content, source: "diagnostics", score: 0.0 })
}

pub fn clipboard_chunk(text: &str) -> Option<Chunk> {
    let lines: Vec<&str> = text.trim().lines().take(20).collect();
    if lines.is_empty() || lines.iter().all(|l| l.trim().is_empty()) {
        return None;
    }
    Some(Chunk {
        path: "clipboard.txt".into(),
        start_line: 1,
        end_line: lines.len(),
        content: lines.join("\n"),
        source: "clipboard",
        score: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_cross_file_definition_and_skips_visible() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("util.rs"), "pub fn parse_config(s: &str) -> Config {\n    todo!()\n}\n").unwrap();
        let main = "fn local_helper() {}\nfn main() {\n    let c = parse_config(\"x\");\n    local_helper();\n}\n";
        std::fs::write(dir.path().join("main.rs"), main).unwrap();
        let ws = Workspace::default().refresh(dir.path()).unwrap();
        let input = RetrievalInput {
            path: &dir.path().join("main.rs"),
            content: main,
            cursor_line: 2,
            visible: (0, 10),
            hunks: &[],
            recent_files: &[],
            previous: &[],
            max_chunks: 2,
        };
        let chunks = definition_chunks(&ws, &input);
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        assert_eq!(chunks[0].path, "util.rs");
        assert!(chunks[0].content.starts_with("pub fn parse_config"));
    }
}
