//! Detached, byte-exact adapter for the Qwen PSM line-completion contract.
//!
//! This module is intentionally not registered as a serving model protocol.
//! A selected model digest and a frozen tokenizer-control inventory are needed
//! before a server profile can safely expose it.

use anyhow::{Result, ensure};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub const WIRE_VERSION: &str = "q25-fim-line-completion-v1";
pub const CONTEXT_POLICY_VERSION: &str = "q25-fim-psm-cursor-to-line-end-v1";
pub const CONTEXT_LAYOUT: &str = "q25-fim-psm-v1";
pub const OUTPUT_TOKEN_CAP: usize = 96;

pub const EOS_TOKEN_ID: i32 = 151_643;
pub const FIM_PREFIX_TOKEN_ID: i32 = 151_659;
pub const FIM_MIDDLE_TOKEN_ID: i32 = 151_660;
pub const FIM_SUFFIX_TOKEN_ID: i32 = 151_661;

pub const FIM_PREFIX: &str = "<|fim_prefix|>";
pub const FIM_SUFFIX: &str = "<|fim_suffix|>";
pub const FIM_MIDDLE: &str = "<|fim_middle|>";
pub const EOS_SPELLING: &str = "<|endoftext|>";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecialToken {
    pub id: i32,
    pub spelling: String,
}

#[derive(Clone, Debug)]
pub struct TokenContract {
    tokenizer_sha256: String,
    special_by_id: BTreeMap<i32, String>,
    special_spellings: BTreeSet<String>,
}

impl TokenContract {
    /// Bind the PSM markers and EOS to the exact selected tokenizer, and carry
    /// its complete added-special-token inventory for source/output rejection.
    pub fn new(
        tokenizer_sha256: &str,
        prefix_encoding: &[i32],
        suffix_encoding: &[i32],
        middle_encoding: &[i32],
        eos_encoding: i32,
        special_tokens: &[SpecialToken],
    ) -> Result<Self> {
        ensure!(
            tokenizer_sha256.len() == 64
                && tokenizer_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "invalid tokenizer digest"
        );
        ensure!(
            prefix_encoding == [FIM_PREFIX_TOKEN_ID],
            "fim_prefix tokenizer identity mismatch"
        );
        ensure!(
            suffix_encoding == [FIM_SUFFIX_TOKEN_ID],
            "fim_suffix tokenizer identity mismatch"
        );
        ensure!(
            middle_encoding == [FIM_MIDDLE_TOKEN_ID],
            "fim_middle tokenizer identity mismatch"
        );
        ensure!(
            eos_encoding == EOS_TOKEN_ID,
            "EOS tokenizer identity mismatch"
        );

        let mut special_by_id = BTreeMap::new();
        let mut special_spellings = BTreeSet::new();
        for token in special_tokens {
            ensure!(token.id >= 0, "negative special-token id");
            ensure!(!token.spelling.is_empty(), "empty special-token spelling");
            ensure!(
                special_by_id
                    .insert(token.id, token.spelling.clone())
                    .is_none(),
                "duplicate special-token id"
            );
            ensure!(
                special_spellings.insert(token.spelling.clone()),
                "duplicate special-token spelling"
            );
        }
        for (id, spelling) in [
            (EOS_TOKEN_ID, EOS_SPELLING),
            (FIM_PREFIX_TOKEN_ID, FIM_PREFIX),
            (FIM_SUFFIX_TOKEN_ID, FIM_SUFFIX),
            (FIM_MIDDLE_TOKEN_ID, FIM_MIDDLE),
        ] {
            ensure!(
                special_by_id
                    .get(&id)
                    .is_some_and(|actual| actual == spelling),
                "required FIM/EOS token missing from special-token inventory"
            );
        }
        Ok(Self {
            tokenizer_sha256: tokenizer_sha256.to_ascii_lowercase(),
            special_by_id,
            special_spellings,
        })
    }

    fn reject_special_spellings(&self, text: &str, where_: &str) -> Result<()> {
        ensure!(
            !self
                .special_spellings
                .iter()
                .any(|spelling| text.contains(spelling)),
            "{where_} contains a tokenizer special-token spelling"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LineEnding {
    #[serde(rename = "LF")]
    Lf,
    #[serde(rename = "CRLF")]
    Crlf,
    #[serde(rename = "EOF")]
    Eof,
}

impl LineEnding {
    fn bytes(self) -> &'static str {
        match self {
            Self::Lf => "\n",
            Self::Crlf => "\r\n",
            Self::Eof => "",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ByteRange {
    /// Zero-based, inclusive byte offset.
    pub start_byte: usize,
    /// Zero-based, exclusive byte offset.
    pub end_byte: usize,
    pub end_exclusive: bool,
}

impl ByteRange {
    fn new(start_byte: usize, end_byte: usize) -> Self {
        Self {
            start_byte,
            end_byte,
            end_exclusive: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Prepared {
    pub prompt: String,
    pub model_protocol: String,
    pub context_policy_version: String,
    pub context_layout: String,
    pub tokenizer_sha256: String,
    pub target_row: usize,
    pub cursor_col: usize,
    /// Cursor through the physical line ending, matching the model hole.
    pub model_hole_range: ByteRange,
    /// Whole-line replacement interval used by the existing canonical applier.
    /// It excludes the line ending, which Neovim retains as buffer metadata.
    pub apply_range: ByteRange,
    pub line_ending: LineEnding,
    #[serde(skip)]
    prefix_before_cursor: String,
    #[serde(skip)]
    virtual_empty_file: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CanonicalAction {
    pub kind: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DecodedCompletion {
    /// Unmodified decoded model text, including the expected trailing EOL.
    pub raw_text: String,
    /// One declared trailing line ending is removed only for the old line applier.
    pub stripped_line_ending: String,
    /// The exact canonical whole-line action consumed by the existing applier.
    pub action: CanonicalAction,
    pub model_hole_range: ByteRange,
    pub apply_range: ByteRange,
    /// Number of model sampling steps including the verified EOS token.
    pub generated_steps: usize,
}

#[derive(Clone, Copy, Debug)]
struct PhysicalLine {
    start: usize,
    content_end: usize,
    end: usize,
    ending: LineEnding,
}

fn physical_lines(source: &str) -> Result<Vec<PhysicalLine>> {
    ensure!(source.len() <= 1024 * 1024, "source outside size contract");
    ensure!(!source.contains('\0'), "source contains NUL");
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\r' => {
                ensure!(
                    bytes.get(index + 1) == Some(&b'\n'),
                    "lone CR is outside the FIM line model"
                );
                out.push(PhysicalLine {
                    start,
                    content_end: index,
                    end: index + 2,
                    ending: LineEnding::Crlf,
                });
                index += 2;
                start = index;
            }
            b'\n' => {
                out.push(PhysicalLine {
                    start,
                    content_end: index,
                    end: index + 1,
                    ending: LineEnding::Lf,
                });
                index += 1;
                start = index;
            }
            _ => index += 1,
        }
    }
    if start < bytes.len() {
        out.push(PhysicalLine {
            start,
            content_end: bytes.len(),
            end: bytes.len(),
            ending: LineEnding::Eof,
        });
    }
    Ok(out)
}

/// Build exact PSM context for one physical line. The prompt carries all bytes
/// before the cursor and after the target line, while the model hole includes
/// the original line ending when one exists.
pub fn prepare(
    source: &str,
    target_row: usize,
    cursor_col: usize,
    tokens: &TokenContract,
) -> Result<Prepared> {
    tokens.reject_special_spellings(source, "source")?;
    let lines = physical_lines(source)?;
    let virtual_empty_file = source.is_empty();
    let (line, content, ending) = if virtual_empty_file {
        ensure!(
            target_row == 0 && cursor_col == 0,
            "target outside empty file"
        );
        (None, "", LineEnding::Eof)
    } else {
        let line = *lines
            .get(target_row)
            .ok_or_else(|| anyhow::anyhow!("target row outside source"))?;
        let content = &source[line.start..line.content_end];
        ensure!(
            cursor_col <= content.len() && content.is_char_boundary(cursor_col),
            "cursor is outside a UTF-8 boundary"
        );
        (Some(line), content, line.ending)
    };

    let (line_start, content_end, line_end) = line
        .map(|l| (l.start, l.content_end, l.end))
        .unwrap_or((0, 0, 0));
    let cursor_byte = line_start + cursor_col;
    let prefix_before_cursor = content[..cursor_col].to_string();
    let prompt = format!(
        "{FIM_PREFIX}{}{FIM_SUFFIX}{}{FIM_MIDDLE}",
        &source[..cursor_byte],
        &source[line_end..]
    );
    Ok(Prepared {
        prompt,
        model_protocol: WIRE_VERSION.into(),
        context_policy_version: CONTEXT_POLICY_VERSION.into(),
        context_layout: CONTEXT_LAYOUT.into(),
        tokenizer_sha256: tokens.tokenizer_sha256.clone(),
        target_row,
        cursor_col,
        model_hole_range: ByteRange::new(cursor_byte, line_end),
        apply_range: ByteRange::new(line_start, content_end),
        line_ending: ending,
        prefix_before_cursor,
        virtual_empty_file,
    })
}

/// Decode one completed PSM response. `content_token_ids` excludes the terminal
/// token. The terminal ID is passed separately because llama.cpp's generic
/// end-of-generation predicate also recognizes non-EOS control tokens.
pub fn decode_completion(
    prepared: &Prepared,
    model_protocol: &str,
    output_cap: usize,
    raw_text: &str,
    content_token_ids: &[i32],
    terminal_token_id: Option<i32>,
    tokens: &TokenContract,
) -> Result<DecodedCompletion> {
    ensure!(
        model_protocol == WIRE_VERSION,
        "FIM model protocol mismatch"
    );
    ensure!(
        prepared.tokenizer_sha256 == tokens.tokenizer_sha256,
        "FIM context tokenizer identity mismatch"
    );
    ensure!(output_cap == OUTPUT_TOKEN_CAP, "FIM output cap mismatch");
    let terminal = terminal_token_id.ok_or_else(|| anyhow::anyhow!("missing terminal token"))?;
    ensure!(
        terminal == EOS_TOKEN_ID,
        "generation did not terminate with EOS"
    );
    let generated_steps = content_token_ids
        .len()
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("generated token count overflow"))?;
    ensure!(
        generated_steps <= OUTPUT_TOKEN_CAP,
        "FIM generation exceeds the EOS-inclusive token cap"
    );
    for id in content_token_ids {
        ensure!(*id >= 0, "negative generated token id");
        ensure!(
            !tokens.special_by_id.contains_key(id),
            "generated added-special token is forbidden"
        );
    }
    ensure!(!raw_text.contains('\0'), "completion contains NUL");
    tokens.reject_special_spellings(raw_text, "completion")?;

    let line_ending = prepared.line_ending.bytes();
    let body = if line_ending.is_empty() {
        ensure!(
            !raw_text.contains(['\r', '\n']),
            "EOF completion contains a line ending"
        );
        raw_text
    } else {
        ensure!(
            raw_text.ends_with(line_ending),
            "completion line ending mismatch"
        );
        let body = &raw_text[..raw_text.len() - line_ending.len()];
        ensure!(
            !body.contains(['\r', '\n']),
            "completion contains more than its one declared line ending"
        );
        body
    };
    let replacement = format!("{}{body}", prepared.prefix_before_cursor);
    Ok(DecodedCompletion {
        raw_text: raw_text.into(),
        stripped_line_ending: line_ending.into(),
        action: CanonicalAction {
            kind: if prepared.virtual_empty_file {
                "insert_before"
            } else {
                "replace_line"
            }
            .into(),
            text: replacement,
        },
        model_hole_range: prepared.model_hole_range,
        apply_range: prepared.apply_range,
        generated_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOKENIZER_SHA256: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn tokens() -> TokenContract {
        TokenContract::new(
            TEST_TOKENIZER_SHA256,
            &[FIM_PREFIX_TOKEN_ID],
            &[FIM_SUFFIX_TOKEN_ID],
            &[FIM_MIDDLE_TOKEN_ID],
            EOS_TOKEN_ID,
            &[
                SpecialToken {
                    id: EOS_TOKEN_ID,
                    spelling: EOS_SPELLING.into(),
                },
                SpecialToken {
                    id: 151_644,
                    spelling: "<|im_start|>".into(),
                },
                SpecialToken {
                    id: FIM_PREFIX_TOKEN_ID,
                    spelling: FIM_PREFIX.into(),
                },
                SpecialToken {
                    id: FIM_MIDDLE_TOKEN_ID,
                    spelling: FIM_MIDDLE.into(),
                },
                SpecialToken {
                    id: FIM_SUFFIX_TOKEN_ID,
                    spelling: FIM_SUFFIX.into(),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn golden_psm_prompt_uses_utf8_byte_offsets_and_separate_ranges() {
        let prepared = prepare("α=1\nnext()\n", 0, 2, &tokens()).unwrap();
        assert_eq!(
            prepared.prompt,
            "<|fim_prefix|>α<|fim_suffix|>next()\n<|fim_middle|>"
        );
        assert_eq!(prepared.model_hole_range, ByteRange::new(2, 5));
        assert_eq!(prepared.apply_range, ByteRange::new(0, 4));
        assert_eq!(prepared.line_ending, LineEnding::Lf);
        let metadata = serde_json::to_value(&prepared).unwrap();
        assert_eq!(metadata["line_ending"], "LF");
        assert_eq!(metadata["model_hole_range"]["end_exclusive"], true);
    }

    #[test]
    fn crlf_hole_includes_exact_terminator_and_suffix_starts_after_it() {
        let prepared = prepare("a🙂b\r\ntail\r\n", 0, 5, &tokens()).unwrap();
        assert_eq!(
            prepared.prompt,
            "<|fim_prefix|>a🙂<|fim_suffix|>tail\r\n<|fim_middle|>"
        );
        assert_eq!(prepared.model_hole_range, ByteRange::new(5, 8));
        assert_eq!(prepared.apply_range, ByteRange::new(0, 6));
        assert_eq!(prepared.line_ending, LineEnding::Crlf);
        let decoded = decode_completion(
            &prepared,
            WIRE_VERSION,
            OUTPUT_TOKEN_CAP,
            "value\r\n",
            &[17],
            Some(EOS_TOKEN_ID),
            &tokens(),
        )
        .unwrap();
        assert_eq!(decoded.raw_text, "value\r\n");
        assert_eq!(decoded.stripped_line_ending, "\r\n");
        assert_eq!(decoded.action.text, "a🙂value");
        assert_eq!(decoded.model_hole_range, prepared.model_hole_range);
        assert_eq!(decoded.apply_range, prepared.apply_range);
    }

    #[test]
    fn eof_and_empty_file_have_no_invented_terminator() {
        let prepared = prepare("ab", 0, 2, &tokens()).unwrap();
        let decoded = decode_completion(
            &prepared,
            WIRE_VERSION,
            OUTPUT_TOKEN_CAP,
            "cd",
            &[19],
            Some(EOS_TOKEN_ID),
            &tokens(),
        )
        .unwrap();
        assert_eq!(decoded.action.kind, "replace_line");
        assert_eq!(decoded.action.text, "abcd");
        assert_eq!(decoded.stripped_line_ending, "");
        assert_eq!(prepared.model_hole_range, ByteRange::new(2, 2));

        let empty = prepare("", 0, 0, &tokens()).unwrap();
        assert_eq!(empty.prompt, "<|fim_prefix|><|fim_suffix|><|fim_middle|>");
        assert_eq!(empty.apply_range, ByteRange::new(0, 0));
        let insert = decode_completion(
            &empty,
            WIRE_VERSION,
            OUTPUT_TOKEN_CAP,
            "fn main() {}",
            &[20],
            Some(EOS_TOKEN_ID),
            &tokens(),
        )
        .unwrap();
        assert_eq!(insert.action.kind, "insert_before");
        assert_eq!(insert.action.text, "fn main() {}");
    }

    #[test]
    fn eos_at_step_96_is_valid_but_missing_or_late_eos_is_not() {
        let prepared = prepare("x\n", 0, 1, &tokens()).unwrap();
        let ninety_five = vec![17; OUTPUT_TOKEN_CAP - 1];
        let accepted = decode_completion(
            &prepared,
            WIRE_VERSION,
            OUTPUT_TOKEN_CAP,
            "\n",
            &ninety_five,
            Some(EOS_TOKEN_ID),
            &tokens(),
        )
        .unwrap();
        assert_eq!(accepted.generated_steps, 96);
        assert!(
            decode_completion(
                &prepared,
                WIRE_VERSION,
                OUTPUT_TOKEN_CAP,
                "\n",
                &[],
                None,
                &tokens(),
            )
            .is_err()
        );
        assert!(
            decode_completion(
                &prepared,
                WIRE_VERSION,
                OUTPUT_TOKEN_CAP,
                "\n",
                &vec![17; OUTPUT_TOKEN_CAP],
                Some(EOS_TOKEN_ID),
                &tokens(),
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_eog_other_than_eos_added_tokens_and_old_wire_is_not_decoded() {
        let prepared = prepare("x", 0, 1, &tokens()).unwrap();
        for (wire, terminal, content, raw) in [
            (WIRE_VERSION, Some(151_645), vec![17], "R\tvalue"),
            (WIRE_VERSION, Some(151_645), vec![], ""),
            (
                "single-line-edit-v1",
                Some(EOS_TOKEN_ID),
                vec![17],
                "R\tvalue",
            ),
            (
                WIRE_VERSION,
                Some(EOS_TOKEN_ID),
                vec![FIM_PREFIX_TOKEN_ID],
                "R\tvalue",
            ),
            (WIRE_VERSION, Some(EOS_TOKEN_ID), vec![151_644], "R\tvalue"),
        ] {
            assert!(
                decode_completion(
                    &prepared,
                    wire,
                    OUTPUT_TOKEN_CAP,
                    raw,
                    &content,
                    terminal,
                    &tokens(),
                )
                .is_err()
            );
        }
        let literal_wire_text = decode_completion(
            &prepared,
            WIRE_VERSION,
            OUTPUT_TOKEN_CAP,
            "R\tvalue",
            &[17],
            Some(EOS_TOKEN_ID),
            &tokens(),
        )
        .unwrap();
        assert_eq!(literal_wire_text.action.text, "xR\tvalue");
    }

    #[test]
    fn rejects_ambiguous_source_specials_invalid_ranges_and_line_endings() {
        assert!(prepare("x<|fim_middle|>y", 0, 0, &tokens()).is_err());
        assert!(prepare("x<|im_start|>y", 0, 0, &tokens()).is_err());
        assert!(prepare("a\rb", 0, 0, &tokens()).is_err());
        assert!(prepare("🙂", 0, 1, &tokens()).is_err());
        assert!(prepare("\0", 0, 0, &tokens()).is_err());
        assert!(prepare("x\n", 1, 0, &tokens()).is_err());

        let crlf = prepare("x\r\n", 0, 1, &tokens()).unwrap();
        for raw in ["body\n", "body\r", "body\r\n\r\n"] {
            assert!(
                decode_completion(
                    &crlf,
                    WIRE_VERSION,
                    OUTPUT_TOKEN_CAP,
                    raw,
                    &[17],
                    Some(EOS_TOKEN_ID),
                    &tokens(),
                )
                .is_err()
            );
        }
        let eof = prepare("x", 0, 1, &tokens()).unwrap();
        assert!(
            decode_completion(
                &eof,
                WIRE_VERSION,
                OUTPUT_TOKEN_CAP,
                "body\n",
                &[17],
                Some(EOS_TOKEN_ID),
                &tokens(),
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_wrong_marker_tokenization_and_control_inventory() {
        assert!(
            TokenContract::new(
                TEST_TOKENIZER_SHA256,
                &[FIM_PREFIX_TOKEN_ID + 1],
                &[FIM_SUFFIX_TOKEN_ID],
                &[FIM_MIDDLE_TOKEN_ID],
                EOS_TOKEN_ID,
                &[]
            )
            .is_err()
        );
        let missing = [SpecialToken {
            id: EOS_TOKEN_ID,
            spelling: EOS_SPELLING.into(),
        }];
        assert!(
            TokenContract::new(
                TEST_TOKENIZER_SHA256,
                &[FIM_PREFIX_TOKEN_ID],
                &[FIM_SUFFIX_TOKEN_ID],
                &[FIM_MIDDLE_TOKEN_ID],
                EOS_TOKEN_ID,
                &missing
            )
            .is_err()
        );
        let tokens = tokens();
        let prepared = prepare("x", 0, 1, &tokens).unwrap();
        assert!(
            decode_completion(
                &prepared,
                WIRE_VERSION,
                OUTPUT_TOKEN_CAP,
                "<|im_start|>",
                &[17],
                Some(EOS_TOKEN_ID),
                &tokens,
            )
            .is_err()
        );
    }

    #[test]
    fn context_and_decode_are_bound_to_tokenizer_file_identity() {
        let correct = tokens();
        let prepared = prepare("x", 0, 1, &correct).unwrap();
        let mut mismatched = correct.clone();
        mismatched.tokenizer_sha256 = "b".repeat(64);
        assert!(
            decode_completion(
                &prepared,
                WIRE_VERSION,
                OUTPUT_TOKEN_CAP,
                "y",
                &[17],
                Some(EOS_TOKEN_ID),
                &mismatched,
            )
            .is_err()
        );
    }
}
