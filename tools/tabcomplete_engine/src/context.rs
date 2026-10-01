use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeSet;

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Edit {
    pub row: usize,
    pub old_text: String,
    pub new_text: String,
}
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct EditorState {
    pub file_id: String,
    pub filetype: String,
    pub source: String,
    pub target_row: usize,
    pub cursor_col: usize,
    #[serde(default)]
    pub history: Vec<Edit>,
    #[serde(default)]
    pub relevant: Vec<String>,
}
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Buffer {
    pub path: String,
    pub source: String,
    #[serde(default)]
    pub recency: u64,
}
#[derive(Clone, Deserialize, Debug)]
pub struct ContextRequest {
    pub state: EditorState,
    #[serde(default)]
    pub buffers: Vec<Buffer>,
    #[serde(default)]
    pub repository_identity: String,
}
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Window {
    pub source: String,
    pub target_row: usize,
    pub start_row: usize,
}
#[derive(Serialize)]
pub struct Prepared {
    pub prompt: String,
    pub prompt_tokens: usize,
    pub context_policy_version: String,
    pub context_layout: String,
    pub model_protocol: String,
    pub window: Option<Window>,
    pub selected_buffers: Vec<String>,
}

fn lines(source: &str) -> Result<Vec<(&str, &str)>> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            let crlf = i > start && source.as_bytes()[i - 1] == b'\r';
            let end = if crlf { i - 1 } else { i };
            ensure!(!source[start..end].contains('\r'), "unsupported lone CR");
            out.push((&source[start..end], if crlf { "CRLF" } else { "LF" }));
            start = i + 1;
        }
    }
    if start < source.len() {
        ensure!(!source[start..].contains('\r'), "unsupported lone CR");
        out.push((&source[start..], "EOF"));
    }
    Ok(out)
}
fn validate(s: &EditorState) -> Result<()> {
    ensure!(
        s.source.len() <= 1024 * 1024 && !s.source.contains('\0'),
        "source outside size contract"
    );
    ensure!(
        matches!(s.filetype.as_str(), "python" | "typescript" | "rust" | "go"),
        "unsupported filetype"
    );
    ensure!(
        !excluded(&s.file_id) && s.file_id.len() < 4096,
        "excluded path"
    );
    ensure!(
        s.history.len() <= 32 && s.relevant.len() <= 16,
        "context history too large"
    );
    let ls = lines(&s.source)?;
    ensure!(s.target_row <= ls.len(), "row outside source");
    let target = ls.get(s.target_row).map_or("", |l| l.0);
    ensure!(
        s.cursor_col <= target.len() && target.is_char_boundary(s.cursor_col),
        "invalid UTF-8 cursor"
    );
    Ok(())
}
pub fn excluded(path: &str) -> bool {
    let base = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    base == ".env"
        || base.starts_with(".env.")
        || base == "credentials"
        || base.ends_with(".pem")
        || base.ends_with(".key")
        || base.starts_with("id_rsa")
        || base.starts_with("id_ed25519")
        || base.contains("credentials")
        || base.contains("secrets")
        || [".npmrc", ".netrc", ".pypirc"].contains(&base.as_str())
        || [".p12", ".pfx", ".crt", ".cer"]
            .iter()
            .any(|ext| base.ends_with(ext))
        || path.contains("/.ssh/")
        || path.contains("/.gnupg/")
        || path.contains("/.aws/")
        || path.contains("/.git/")
}
fn quote(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}
fn render(
    s: &EditorState,
    ls: &[(&str, &str)],
    rows: &BTreeSet<usize>,
    history: &BTreeSet<usize>,
    relevant: &BTreeSet<usize>,
) -> String {
    let mut out = vec![
        "<single-line-edit-v1>".into(),
        "Actions: N, D, R\t, I\t. R/I prefixes take one source line of text.".into(),
        "The gap after R/I is a literal tab. No CR, LF, or explanation; end with tokenizer EOS."
            .into(),
        format!("File: {}", quote(&s.file_id)),
        format!("Filetype: {}", s.filetype),
        format!("Target row (zero based): {}", s.target_row),
        format!("Cursor byte column: {}", s.cursor_col),
        format!("Physical line count: {}", ls.len()),
        "Current source lines (JSON strings; indices are original rows):".into(),
    ];
    for row in rows {
        out.push(format!("{} [{}] {}", row, ls[*row].1, quote(ls[*row].0)));
    }
    if s.target_row == ls.len() {
        out.push(format!("{} [APPEND AT EOF]", s.target_row));
    }
    out.push("Recent edits, oldest to newest (JSON old and new text):".into());
    for i in history {
        let e = &s.history[*i];
        out.push(format!(
            "{} row={} old={} new={}",
            i,
            e.row,
            quote(&e.old_text),
            quote(&e.new_text)
        ));
    }
    out.push("Relevant definitions and imports (JSON strings):".into());
    for i in relevant {
        out.push(format!("{} {}", i, quote(&s.relevant[*i])));
    }
    out.extend(["</single-line-edit-v1>".into(), "Action:".into()]);
    out.join("\n")
}
pub fn prepare(
    req: &ContextRequest,
    protocol: &str,
    budget: usize,
    count: impl Fn(&str) -> Result<usize>,
) -> Result<Prepared> {
    validate(&req.state)?;
    ensure!(
        req.repository_identity.len() <= 4096,
        "repository identity too large"
    );
    ensure!(req.buffers.len() <= 8, "too many context buffers");
    let mut s = req.state.clone();
    let mut selected = Vec::new();
    let mut buffers = req.buffers.clone();
    buffers.sort_by_key(|b| std::cmp::Reverse(b.recency));
    // Only bounded, explicitly supplied open/recent buffers. No filesystem walk,
    // embedding model, synchronous Git, or unrelated repository reads.
    for b in buffers {
        if b.path == s.file_id || excluded(&b.path) || b.source.len() > 65536 {
            continue;
        }
        let snippets: Vec<_> = b
            .source
            .lines()
            .take(128)
            .filter(|l| {
                let l = l.trim_start();
                l.starts_with("import ")
                    || l.starts_with("use ")
                    || l.starts_with("from ")
                    || l.starts_with("def ")
                    || l.starts_with("fn ")
                    || l.starts_with("func ")
                    || l.starts_with("export ")
                    || l.starts_with("class ")
            })
            .take(4)
            .map(|l| l.chars().take(256).collect::<String>())
            .collect();
        if !snippets.is_empty() && s.relevant.len() < 16 {
            s.relevant
                .push(format!("{}\n{}", b.path, snippets.join("\n")));
            selected.push(b.path);
        }
    }
    if protocol == "sweep-full-file-v1" {
        let ls = lines(&s.source)?;
        ensure!(
            s.target_row < ls.len(),
            "Sweep requires an existing target line"
        );
        // A complete contiguous window is the output scope, not a silently
        // truncated whole-file generation. Exact source outside the target
        // line must survive before a canonical action is returned.
        for radius in (0..=12).rev() {
            let start = s.target_row.saturating_sub(radius);
            let end = (s.target_row + radius + 1).min(ls.len());
            let source = ls[start..end]
                .iter()
                .map(|(l, t)| {
                    format!(
                        "{}{}",
                        l,
                        match *t {
                            "LF" => "\n",
                            "CRLF" => "\r\n",
                            _ => "",
                        }
                    )
                })
                .collect::<String>();
            let window = Window {
                source: source.clone(),
                target_row: s.target_row - start,
                start_row: start,
            };
            // Leave output room for the copied window and an actual edit.
            // This changes the explicitly versioned window scope only.
            if count(&source)? > 160 {
                continue;
            }
            let mut original = source.clone();
            if let Some(e) = s.history.last().filter(|e| {
                e.row >= start
                    && e.row < end
                    && !e.old_text.contains(['\r', '\n'])
                    && !e.new_text.contains(['\r', '\n'])
            }) {
                let mut old = ls[start..end]
                    .iter()
                    .map(|(l, t)| ((*l).to_string(), *t))
                    .collect::<Vec<_>>();
                if old[e.row - start].0 == e.new_text {
                    old[e.row - start].0 = e.old_text.clone();
                    original = old
                        .iter()
                        .map(|(l, t)| {
                            format!(
                                "{}{}",
                                l,
                                match *t {
                                    "LF" => "\n",
                                    "CRLF" => "\r\n",
                                    _ => "",
                                }
                            )
                        })
                        .collect();
                }
            }
            let mut parts = Vec::new();
            for item in s.relevant.iter().take(2) {
                parts.push(format!("<|file_sep|>context\n{}", item));
            }
            if original != source {
                parts.extend([
                    format!("<|file_sep|>{}.diff", s.file_id),
                    "original:".into(),
                    original.clone(),
                    "updated:".into(),
                    source.clone(),
                ]);
            }
            parts.extend([
                format!("<|file_sep|>original/{}", s.file_id),
                original,
                format!("<|file_sep|>current/{}", s.file_id),
                source,
                format!("<|file_sep|>updated/{}\n", s.file_id),
            ]);
            let prompt = parts.join("\n");
            let n = count(&prompt)?;
            if n <= budget {
                return Ok(Prepared {
                    prompt,
                    prompt_tokens: n,
                    context_policy_version: "sweep-window-context-v1".into(),
                    context_layout: "sweep-window-v1".into(),
                    model_protocol: protocol.into(),
                    window: Some(window),
                    selected_buffers: selected,
                });
            }
        }
        bail!("mandatory Sweep window exceeds token budget");
    }
    ensure!(
        protocol == "single-line-edit-v1",
        "unsupported model protocol"
    );
    let ls = lines(&s.source)?;
    let mut rows = BTreeSet::new();
    let mut history = BTreeSet::new();
    let mut relevant = BTreeSet::new();
    if s.target_row < ls.len() {
        rows.insert(s.target_row);
    }
    let mut items = Vec::new();
    for distance in 1..=3 {
        if let Some(row) = s.target_row.checked_sub(distance) {
            items.push((0, row));
        }
        items.push((0, s.target_row + distance));
    }
    for index in (0..s.history.len()).rev() {
        items.push((1, index));
    }
    for index in 0..s.relevant.len() {
        items.push((2, index));
    }
    for distance in 4..=20 {
        if let Some(row) = s.target_row.checked_sub(distance) {
            items.push((0, row));
        }
        items.push((0, s.target_row + distance));
    }
    ensure!(
        count(&render(&s, &ls, &rows, &history, &relevant))? <= budget,
        "mandatory markers exceed token budget"
    );
    for (bucket, i) in items {
        if bucket == 0 && i >= ls.len() {
            continue;
        }
        let set = match bucket {
            0 => &mut rows,
            1 => &mut history,
            _ => &mut relevant,
        };
        if !set.insert(i) {
            continue;
        }
        if count(&render(&s, &ls, &rows, &history, &relevant))? > budget {
            match bucket {
                0 => &mut rows,
                1 => &mut history,
                _ => &mut relevant,
            }
            .remove(&i);
        }
    }
    let prompt = render(&s, &ls, &rows, &history, &relevant);
    let n = count(&prompt)?;
    Ok(Prepared {
        prompt,
        prompt_tokens: n,
        context_policy_version: "single-line-context-v2".into(),
        context_layout: "trained-v2".into(),
        model_protocol: protocol.into(),
        window: None,
        selected_buffers: selected,
    })
}
pub fn effective_layout(protocol: &str, requested: &str) -> Result<&'static str> {
    ensure!(
        matches!(requested, "trained-v2" | "cursor-last-v1"),
        "unsupported context layout"
    );
    match protocol {
        "single-line-edit-v1" => match requested {
            "trained-v2" => Ok("trained-v2"),
            "cursor-last-v1" => Ok("cursor-last-v1"),
            _ => unreachable!(),
        },
        "sweep-full-file-v1" => Ok("sweep-window-v1"),
        _ => bail!("unsupported model protocol"),
    }
}

pub fn prepare_layout(
    req: &ContextRequest,
    protocol: &str,
    requested_layout: &str,
    budget: usize,
    count: impl Fn(&str) -> Result<usize>,
) -> Result<Prepared> {
    let actual_layout = effective_layout(protocol, requested_layout)?;
    let mut prepared = prepare(req, protocol, budget, |prompt| count(prompt))?;
    if actual_layout == "cursor-last-v1" {
        prepared.prompt = move_cursor_header_before_action(&prepared.prompt)?;
        prepared.context_policy_version = "single-line-cursor-last-context-v1".into();
        prepared.prompt_tokens = count(&prepared.prompt)?;
        ensure!(
            prepared.prompt_tokens <= budget,
            "cursor-last layout exceeds token budget"
        );
    }
    prepared.context_layout = actual_layout.into();
    Ok(prepared)
}

fn move_cursor_header_before_action(prompt: &str) -> Result<String> {
    let mut lines = prompt.split('\n').collect::<Vec<_>>();
    let cursor_positions = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("Cursor byte column: "))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    ensure!(
        cursor_positions.len() == 1,
        "expected exactly one cursor byte column header"
    );
    let cursor_index = cursor_positions[0];
    let cursor_line = lines[cursor_index];
    let cursor_value = cursor_line
        .strip_prefix("Cursor byte column: ")
        .ok_or_else(|| anyhow::anyhow!("malformed cursor byte column header"))?;
    ensure!(
        !cursor_value.is_empty() && cursor_value.bytes().all(|byte| byte.is_ascii_digit()),
        "malformed cursor byte column value"
    );
    ensure!(
        cursor_index > 0
            && lines[cursor_index - 1].starts_with("Target row (zero based): ")
            && cursor_index + 1 < lines.len()
            && lines[cursor_index + 1].starts_with("Physical line count: "),
        "cursor byte column header is outside the trained layout"
    );
    let action_positions = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| **line == "Action:")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    ensure!(
        action_positions.len() == 1
            && action_positions[0] + 1 == lines.len()
            && action_positions[0] > 0
            && lines[action_positions[0] - 1] == "</single-line-edit-v1>",
        "malformed single-line action marker"
    );
    let cursor_line = lines.remove(cursor_index);
    let action_index = lines.len() - 1;
    lines.insert(action_index, cursor_line);
    Ok(lines.join("\n"))
}

pub fn decode_action(raw: &str) -> Result<serde_json::Value> {
    Ok(match raw {
        "N" => json!({"kind":"keep","text":null}),
        "D" => json!({"kind":"delete_line","text":null}),
        text if text.starts_with("R\t") && !text[2..].contains(['\r', '\n']) => {
            json!({"kind":"replace_line","text":&text[2..]})
        }
        text if text.starts_with("I\t") && !text[2..].contains(['\r', '\n']) => {
            json!({"kind":"insert_before","text":&text[2..]})
        }
        _ => bail!("invalid single-line wire"),
    })
}
pub fn map_sweep(raw: &str, window: &Window) -> Result<serde_json::Value> {
    validate_window(window)?;
    if raw == window.source {
        return decode_action("N");
    }
    let ls = lines(&window.source)?;
    let prefix = ls[..window.target_row]
        .iter()
        .map(|(l, t)| {
            format!(
                "{}{}",
                l,
                match *t {
                    "LF" => "\n",
                    "CRLF" => "\r\n",
                    _ => "",
                }
            )
        })
        .collect::<String>();
    let target = ls[window.target_row];
    let term = match target.1 {
        "LF" => "\n",
        "CRLF" => "\r\n",
        _ => "",
    };
    let suffix = &window.source[prefix.len() + target.0.len()..];
    ensure!(
        raw.starts_with(&prefix)
            && raw.ends_with(suffix)
            && raw.len() >= prefix.len() + suffix.len(),
        "Sweep changed outside target line"
    );
    let text = &raw[prefix.len()..raw.len() - suffix.len()];
    ensure!(
        !text.contains(['\r', '\n']),
        "Sweep produced a multiline edit"
    );
    // Empty replacement retains the newline, distinct from deleting a line.
    let _ = term;
    decode_action(&format!("R\t{}", text))
}

pub fn validate_window(window: &Window) -> Result<()> {
    ensure!(window.source.len() <= 65536, "window too large");
    ensure!(
        window.target_row < lines(&window.source)?.len(),
        "window row outside source"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_context_golden() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../tests/context_golden.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let req = ContextRequest {
                state: serde_json::from_value(case["state"].clone()).unwrap(),
                buffers: vec![],
                repository_identity: "golden".into(),
            };
            let prepared =
                prepare(&req, "single-line-edit-v1", 20000, |text| Ok(text.len())).unwrap();
            assert_eq!(prepared.prompt, case["expected"].as_str().unwrap());
        }
    }
    fn state() -> EditorState {
        EditorState {
            file_id: "sample.py".into(),
            filetype: "python".into(),
            source: "def f():\n    return α\n".into(),
            target_row: 1,
            cursor_col: 11,
            history: vec![],
            relevant: vec![],
        }
    }
    #[test]
    fn trained_layout_preserves_the_control_prompt() {
        let req = ContextRequest {
            state: state(),
            buffers: vec![],
            repository_identity: "layout-test".into(),
        };
        let control = prepare(&req, "single-line-edit-v1", 20000, |s| Ok(s.len())).unwrap();
        let trained = prepare_layout(&req, "single-line-edit-v1", "trained-v2", 20000, |s| {
            Ok(s.len())
        })
        .unwrap();
        assert_eq!(trained.prompt, control.prompt);
        assert_eq!(trained.prompt_tokens, control.prompt_tokens);
        assert_eq!(
            trained.context_policy_version,
            control.context_policy_version
        );
        assert_eq!(trained.context_layout, "trained-v2");
    }

    #[test]
    fn cursor_last_layout_moves_only_the_cursor_header_and_keeps_unicode() {
        let req = ContextRequest {
            state: state(),
            buffers: vec![],
            repository_identity: "layout-test".into(),
        };
        let control = prepare(&req, "single-line-edit-v1", 20000, |s| Ok(s.len())).unwrap();
        let moved = prepare_layout(&req, "single-line-edit-v1", "cursor-last-v1", 20000, |s| {
            Ok(s.len())
        })
        .unwrap();
        let cursor_line = "Cursor byte column: 11";
        let control_without_cursor = control
            .prompt
            .lines()
            .filter(|line| *line != cursor_line)
            .collect::<Vec<_>>();
        let moved_without_cursor = moved
            .prompt
            .lines()
            .filter(|line| *line != cursor_line)
            .collect::<Vec<_>>();
        assert_eq!(control_without_cursor, moved_without_cursor);
        assert_eq!(moved.prompt.matches(cursor_line).count(), 1);
        assert!(
            moved
                .prompt
                .ends_with("</single-line-edit-v1>\nCursor byte column: 11\nAction:")
        );
        assert!(moved.prompt.contains("\"    return α\""));
        assert_eq!(moved.prompt_tokens, moved.prompt.len());
        assert_eq!(moved.context_layout, "cursor-last-v1");
        assert_eq!(
            moved.context_policy_version,
            "single-line-cursor-last-context-v1"
        );
        assert_ne!(moved.context_policy_version, control.context_policy_version);
    }

    #[test]
    fn cursor_last_layout_recounts_and_rejects_over_budget() {
        let req = ContextRequest {
            state: state(),
            buffers: vec![],
            repository_identity: "layout-test".into(),
        };
        let control = prepare(&req, "single-line-edit-v1", 20000, |s| Ok(s.len())).unwrap();
        let budget = control.prompt.len();
        let result = prepare_layout(&req, "single-line-edit-v1", "cursor-last-v1", budget, |s| {
            Ok(s.len()
                + if s.contains("</single-line-edit-v1>\nCursor byte column: ") {
                    1
                } else {
                    0
                })
        });
        assert!(result.is_err());
    }

    #[test]
    fn cursor_last_layout_rejects_malformed_prompts_and_sweep_stays_unchanged() {
        let req = ContextRequest {
            state: state(),
            buffers: vec![],
            repository_identity: "layout-test".into(),
        };
        let control = prepare(&req, "single-line-edit-v1", 20000, |s| Ok(s.len())).unwrap();
        assert!(
            move_cursor_header_before_action(
                &control.prompt.replace("Cursor byte column: 11\n", "")
            )
            .is_err()
        );
        assert!(
            move_cursor_header_before_action(&control.prompt.replacen(
                "Cursor byte column: 11\n",
                "Cursor byte column: 11\nCursor byte column: 11\n",
                1
            ))
            .is_err()
        );
        assert!(
            move_cursor_header_before_action(
                &control
                    .prompt
                    .replace("Physical line count:", "Physical lines:")
            )
            .is_err()
        );
        assert!(effective_layout("single-line-edit-v1", "cursor-lats-v1").is_err());

        let sweep_req = ContextRequest {
            state: EditorState {
                source: "a\nold\nz\n".into(),
                target_row: 1,
                cursor_col: 0,
                ..state()
            },
            buffers: vec![],
            repository_identity: "layout-test".into(),
        };
        let sweep_control =
            prepare(&sweep_req, "sweep-full-file-v1", 20000, |s| Ok(s.len())).unwrap();
        let sweep_layout = prepare_layout(
            &sweep_req,
            "sweep-full-file-v1",
            "cursor-last-v1",
            20000,
            |s| Ok(s.len()),
        )
        .unwrap();
        assert_eq!(sweep_layout.prompt, sweep_control.prompt);
        assert_eq!(sweep_layout.prompt_tokens, sweep_control.prompt_tokens);
        assert_eq!(
            sweep_layout.context_policy_version,
            "sweep-window-context-v1"
        );
        assert_eq!(sweep_layout.context_layout, "sweep-window-v1");
    }

    #[test]
    fn unicode_boundary() {
        let mut s = state();
        s.cursor_col = 12;
        assert!(validate(&s).is_err());
        s.cursor_col = 13;
        assert!(validate(&s).is_ok());
    }
    #[test]
    fn strict_wire() {
        for good in ["N", "D", "R\t", "R\t  ", "I\tα"] {
            assert!(decode_action(good).is_ok());
        }
        for bad in ["N\n", "R x", "R\tα\n", " ``` "] {
            assert!(decode_action(bad).is_err());
        }
    }
    #[test]
    fn safe_sweep_mapping() {
        let w = Window {
            source: "a\nold\nz\n".into(),
            target_row: 1,
            start_row: 0,
        };
        assert_eq!(map_sweep("a\nnew\nz\n", &w).unwrap()["text"], "new");
        assert!(map_sweep("b\nnew\nz\n", &w).is_err());
        assert!(map_sweep("a\nnew\nother\nz\n", &w).is_err());
        let invalid = Window {
            target_row: usize::MAX,
            ..w.clone()
        };
        assert!(map_sweep(&invalid.source, &invalid).is_err());
        assert!(map_sweep("new", &invalid).is_err());
    }
    #[test]
    fn budget_and_exclusions() {
        let r = ContextRequest {
            state: state(),
            buffers: vec![Buffer {
                path: ".env".into(),
                source: "export secret=1".into(),
                recency: 1,
            }],
            repository_identity: "test".into(),
        };
        let p = prepare(&r, "single-line-edit-v1", 2000, |s| Ok(s.len())).unwrap();
        assert!(p.selected_buffers.is_empty());
        assert!(p.prompt.ends_with("Action:"));
        assert!(prepare(&r, "single-line-edit-v1", 10, |s| Ok(s.len())).is_err());
    }
}
