//! Bounded syntax regression guard. This is not a compiler or a quality score.
//! Source bytes are never unescaped or rewritten to rescue model output.
use crate::context::EditorState;
use anyhow::{Result, bail, ensure};
use serde_json::Value;
use tree_sitter::{Node, Parser};

const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_ERROR_NODES: usize = 4096;

fn error_fragments(parser: &mut Parser, source: &str) -> Result<Vec<String>> {
    parser.reset();
    let tree = parser
        .parse(source.as_bytes(), None)
        .ok_or_else(|| anyhow::anyhow!("syntax_budget_exceeded"))?;
    let mut stack: Vec<Node<'_>> = vec![tree.root_node()];
    let mut errors = Vec::new();
    while let Some(node) = stack.pop() {
        if node.is_error() || node.is_missing() {
            ensure!(errors.len() < MAX_ERROR_NODES, "syntax_error_limit");
            let text = source.get(node.byte_range()).unwrap_or("");
            // Keep exact fragments. Equal error counts with different bad text
            // must not turn a newly introduced error into an approved action.
            errors.push(format!("{}:{}", node.kind(), text));
        }
        if node.has_error() {
            let mut cursor = node.walk();
            stack.extend(node.children(&mut cursor));
        }
    }
    Ok(errors)
}

fn apply(state: &EditorState, action: &Value) -> Result<String> {
    let kind = action["kind"].as_str().unwrap_or("");
    if kind == "keep" {
        return Ok(state.source.clone());
    }
    let rows: Vec<&str> = state.source.split_inclusive('\n').collect();
    ensure!(
        state.target_row <= rows.len(),
        "syntax_target_outside_source"
    );
    let begin: usize = rows.iter().take(state.target_row).map(|s| s.len()).sum();
    let line = rows.get(state.target_row).copied().unwrap_or("");
    let content = line.trim_end_matches('\n').trim_end_matches('\r');
    let crlf = rows.iter().filter(|row| row.ends_with("\r\n")).count();
    let lf = rows
        .iter()
        .filter(|row| row.ends_with('\n') && !row.ends_with("\r\n"))
        .count();
    let style = if crlf > lf {
        "\r\n"
    } else if crlf == lf {
        rows.iter()
            .find(|row| row.ends_with('\n'))
            .map_or(
                "\n",
                |row| if row.ends_with("\r\n") { "\r\n" } else { "\n" },
            )
    } else {
        "\n"
    };
    let text = action["text"].as_str().unwrap_or("");
    ensure!(!text.contains(['\r', '\n', '\0']), "syntax_not_single_line");
    let (end, replacement) = match kind {
        "replace_line" => {
            ensure!(state.target_row < rows.len(), "syntax_replace_at_eof");
            (
                begin + content.len(),
                if text.is_empty() && !line.ends_with('\n') {
                    style.to_string()
                } else {
                    text.to_string()
                },
            )
        }
        "delete_line" => {
            ensure!(state.target_row < rows.len(), "syntax_delete_at_eof");
            (begin + line.len(), String::new())
        }
        "insert_before" => {
            if state.target_row < rows.len() {
                let separator = if line.ends_with("\r\n") {
                    "\r\n"
                } else if line.ends_with('\n') {
                    "\n"
                } else {
                    style
                };
                (begin, format!("{text}{separator}"))
            } else {
                let prefix = if !rows.is_empty() && !state.source.ends_with('\n') {
                    style
                } else {
                    ""
                };
                let suffix = if text.is_empty() { style } else { "" };
                (begin, format!("{prefix}{text}{suffix}"))
            }
        }
        _ => bail!("syntax_unknown_action"),
    };
    let mut result = String::with_capacity(state.source.len() + replacement.len());
    result.push_str(&state.source[..begin]);
    result.push_str(&replacement);
    result.push_str(&state.source[end..]);
    Ok(result)
}

pub fn check(state: &EditorState, action: &Value) -> Result<()> {
    if state.filetype != "rust" || action["kind"] == "keep" {
        return Ok(());
    }
    ensure!(
        state.source.len() <= MAX_SOURCE_BYTES,
        "syntax_source_limit"
    );
    let after = apply(state, action)?;
    ensure!(after.len() <= MAX_SOURCE_BYTES, "syntax_source_limit");
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_rust::LANGUAGE.into())?;
    // The pinned API retains this bounded parse facility. On timeout we suppress
    // the proposal rather than block the editor or assume its syntax is valid.
    #[allow(deprecated)]
    parser.set_timeout_micros(30_000);
    let mut existing = error_fragments(&mut parser, &state.source)?;
    for error in error_fragments(&mut parser, &after)? {
        if let Some(index) = existing.iter().position(|old| old == &error) {
            existing.swap_remove(index);
        } else {
            bail!("rust_syntax_regression");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn state(line: &str) -> EditorState {
        EditorState {
            file_id: "src/main.rs".into(),
            filetype: "rust".into(),
            source: format!("fn main() {{\n{line}\n}}\n"),
            target_row: 1,
            cursor_col: 0,
            history: vec![],
            relevant: vec![],
        }
    }
    #[test]
    fn screenshot_escaped_quotes_are_rejected_without_repair() {
        let s = state("    println!(\"{:?}\", Message::Resize);");
        let bad =
            json!({"kind":"replace_line","text":r#"    println!(\"{:?}\", Message::Resize);"#});
        assert!(check(&s, &bad).is_err());
        assert!(
            check(
                &s,
                &json!({"kind":"replace_line","text":"    println!(\"{:?}\", Message::Resize);"})
            )
            .is_ok()
        );
    }
    #[test]
    fn legitimate_backslashes_raw_strings_unicode_comments_are_preserved() {
        for line in [
            r#"    println!("quote: \" and path: C:\\tmp");"#,
            r##"    let s = r#"\"東京"#;"##,
            r#"    // \" is literal documentation"#,
            "    let λ = 'λ';",
        ] {
            assert!(
                check(&state(""), &json!({"kind":"replace_line","text":line})).is_ok(),
                "{line}"
            );
        }
    }
    #[test]
    fn preexisting_unrelated_errors_do_not_disqualify_valid_edit() {
        let mut s = state("    let value = 1;");
        s.source.push_str("fn broken( {\n");
        assert!(
            check(
                &s,
                &json!({"kind":"replace_line","text":"    let value = 2;"})
            )
            .is_ok()
        );
    }
    #[test]
    fn fixing_bad_source_is_allowed_and_crlf_bytes_survive() {
        let mut s = state(r#"    println!(\"broken\");"#);
        s.source = s.source.replace('\n', "\r\n");
        let a = json!({"kind":"replace_line","text":"    println!(\"fixed\");"});
        assert!(check(&s, &a).is_ok());
        assert!(apply(&s, &a).unwrap().contains("\r\n"));
    }
    #[test]
    fn other_languages_and_keep_do_not_claim_rust_validation() {
        let mut s = state("");
        s.filetype = "python".into();
        assert!(check(&s, &json!({"kind":"replace_line","text":"value = 1"})).is_ok());
        assert!(check(&state("fn broken("), &json!({"kind":"keep","text":null})).is_ok());
    }
    #[test]
    fn guard_application_matches_python_lua_golden_bytes() {
        let cases: Vec<Value> =
            serde_json::from_str(include_str!("../tests/action_golden.json")).unwrap();
        for case in cases {
            let s: EditorState = serde_json::from_value(case["state"].clone()).unwrap();
            assert_eq!(
                apply(&s, &case["action"]).unwrap(),
                case["after_source"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
}
