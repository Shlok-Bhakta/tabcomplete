//! Byte-exact adapter for the Qwen PSM line-completion contract.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::{BTreeMap, BTreeSet};
use unicode_normalization::is_nfc;

pub const WIRE_VERSION: &str = "q25-fim-line-completion-v1";
pub const CONTEXT_POLICY_VERSION: &str = "q25-fim-psm-cursor-to-line-end-bounded640-256-v2";
pub const ALIGNED_CONTEXT_POLICY_VERSION: &str =
    "q25-fim-psm-cursor-to-line-end-bounded640-256-align128-v1";
pub const CONTEXT_LAYOUT: &str = "q25-fim-psm-bounded-v2";
pub const PREFIX_CONTEXT_TOKEN_LIMIT: usize = 640;
pub const SUFFIX_CONTEXT_TOKEN_LIMIT: usize = 256;
pub const OUTPUT_TOKEN_CAP: usize = 96;

pub const EOS_TOKEN_ID: i32 = 151_643;
pub const FIM_PREFIX_TOKEN_ID: i32 = 151_659;
pub const FIM_MIDDLE_TOKEN_ID: i32 = 151_660;
pub const FIM_SUFFIX_TOKEN_ID: i32 = 151_661;

pub const FIM_PREFIX: &str = "<|fim_prefix|>";
pub const FIM_SUFFIX: &str = "<|fim_suffix|>";
pub const FIM_MIDDLE: &str = "<|fim_middle|>";
pub const EOS_SPELLING: &str = "<|endoftext|>";
pub const COMPLETION_MODE: &str = "remaining_logical_line_after_utf8_cursor";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ContextWindow {
    SlidingV2,
    Aligned128V1,
}

impl ContextWindow {
    pub fn is_sliding(&self) -> bool {
        *self == Self::SlidingV2
    }
    pub fn policy(self) -> &'static str {
        match self {
            Self::SlidingV2 => CONTEXT_POLICY_VERSION,
            Self::Aligned128V1 => ALIGNED_CONTEXT_POLICY_VERSION,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PrefixWindowMetadata {
    pub full_prefix_tokens: usize,
    pub control_start_byte: usize,
    pub control_prefix_tokens: usize,
    pub aligned_start_token: usize,
    pub fallback_reason: Option<&'static str>,
}

/// The default path is exactly the existing crop. The opt-in path sacrifices
/// older context to hold an absolute native-token boundary across nearby edits.
pub fn crop_prefix_context<F>(
    text: &str,
    editable_line_start: usize,
    window: ContextWindow,
    mut tokenize: F,
) -> Result<(CroppedContext, Option<PrefixWindowMetadata>)>
where
    F: FnMut(&str) -> Result<Vec<Vec<u8>>>,
{
    ensure!(
        editable_line_start <= text.len() && text.is_char_boundary(editable_line_start),
        "invalid editable line start"
    );
    if window == ContextWindow::SlidingV2 {
        return Ok((
            crop_context(
                text,
                PREFIX_CONTEXT_TOKEN_LIMIT,
                ContextSide::KeepRight,
                tokenize,
            )?,
            None,
        ));
    }
    let full_pieces = tokenize(text)?;
    let boundaries = piece_boundaries(text, &full_pieces)?;
    ensure!(
        full_pieces.iter().flatten().eq(text.as_bytes().iter()),
        "native prefix pieces do not round-trip the source bytes"
    );
    let control = crop_context(
        text,
        PREFIX_CONTEXT_TOKEN_LIMIT,
        ContextSide::KeepRight,
        |part| {
            if part == text {
                Ok(full_pieces.clone())
            } else {
                tokenize(part)
            }
        },
    )?;
    let full_count = full_pieces.len();
    let aligned_start_token = full_count
        .saturating_sub(PREFIX_CONTEXT_TOKEN_LIMIT)
        .div_ceil(128)
        * 128;
    let start_byte = ceil_char_boundary(text, boundaries[aligned_start_token]);
    let mut metadata = PrefixWindowMetadata {
        full_prefix_tokens: full_count,
        control_start_byte: control.start_byte,
        control_prefix_tokens: control.token_count,
        aligned_start_token,
        fallback_reason: None,
    };
    // Do not remove any editable-line prefix retained by the control, including
    // the tail of a line already longer than the control budget.
    if start_byte > editable_line_start && start_byte > control.start_byte {
        metadata.fallback_reason = Some("editable_line_prefix");
        return Ok((control, Some(metadata)));
    }
    if full_count <= PREFIX_CONTEXT_TOKEN_LIMIT {
        return Ok((control, Some(metadata)));
    }
    let retained = &text[start_byte..];
    let pieces = tokenize(retained)?;
    piece_boundaries(retained, &pieces)?;
    ensure!(
        pieces.iter().flatten().eq(retained.as_bytes().iter()),
        "native retained prefix pieces do not round-trip the source bytes"
    );
    if pieces.len() > PREFIX_CONTEXT_TOKEN_LIMIT {
        metadata.fallback_reason = Some("retokenized_budget");
        return Ok((control, Some(metadata)));
    }
    Ok((
        CroppedContext {
            start_byte,
            end_byte: text.len(),
            token_count: pieces.len(),
        },
        Some(metadata),
    ))
}

pub fn filetype_training_scope(filetype: &str) -> &'static str {
    if matches!(
        filetype,
        "python" | "typescript" | "typescriptreact" | "rust" | "go"
    ) {
        "trained_language"
    } else {
        "uncalibrated_language"
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextSide {
    KeepRight,
    KeepLeft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CroppedContext {
    /// Byte range into the supplied segment, end-exclusive.
    pub start_byte: usize,
    pub end_byte: usize,
    pub token_count: usize,
}

fn piece_boundaries(text: &str, pieces: &[Vec<u8>]) -> Result<Vec<usize>> {
    let mut boundaries = Vec::with_capacity(pieces.len() + 1);
    let mut offset = 0usize;
    boundaries.push(offset);
    for piece in pieces {
        offset = offset
            .checked_add(piece.len())
            .ok_or_else(|| anyhow::anyhow!("token-piece offset overflow"))?;
        ensure!(offset <= text.len(), "token pieces exceed source segment");
        boundaries.push(offset);
    }
    ensure!(
        offset == text.len(),
        "native tokenizer pieces do not round-trip the source segment"
    );
    Ok(boundaries)
}

fn floor_char_boundary(text: &str, mut offset: usize) -> usize {
    while !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn ceil_char_boundary(text: &str, mut offset: usize) -> usize {
    while !text.is_char_boundary(offset) {
        offset += 1;
    }
    offset
}

/// Crop one source-side segment from token-piece byte offsets. `tokenize`
/// must return pieces in source order; their concatenation is checked exactly
/// before any offsets are used. The result never splits a UTF-8 scalar.
pub fn crop_context<F>(
    text: &str,
    token_limit: usize,
    side: ContextSide,
    mut tokenize: F,
) -> Result<CroppedContext>
where
    F: FnMut(&str) -> Result<Vec<Vec<u8>>>,
{
    let mut start = 0usize;
    let mut end = text.len();
    if token_limit == 0 {
        let point = match side {
            ContextSide::KeepRight => text.len(),
            ContextSide::KeepLeft => 0,
        };
        return Ok(CroppedContext {
            start_byte: point,
            end_byte: point,
            token_count: 0,
        });
    }

    for _ in 0..=text.chars().count() {
        let candidate = &text[start..end];
        let pieces = tokenize(candidate)?;
        let boundaries = piece_boundaries(candidate, &pieces)?;
        if pieces.len() <= token_limit {
            return Ok(CroppedContext {
                start_byte: start,
                end_byte: end,
                token_count: pieces.len(),
            });
        }

        match side {
            ContextSide::KeepRight => {
                let token_index = pieces.len() - token_limit;
                let mut cut = floor_char_boundary(candidate, boundaries[token_index]);
                if cut == 0 {
                    cut = boundaries
                        .iter()
                        .copied()
                        .find(|offset| *offset > 0 && candidate.is_char_boundary(*offset))
                        .or_else(|| candidate.char_indices().nth(1).map(|(offset, _)| offset))
                        .unwrap_or(candidate.len());
                }
                ensure!(
                    cut > 0 && cut <= candidate.len(),
                    "prefix context crop stalled"
                );
                start += cut;
            }
            ContextSide::KeepLeft => {
                let token_index = (token_limit - 1).min(pieces.len() - 1);
                let mut cut = ceil_char_boundary(candidate, boundaries[token_index + 1]);
                if cut >= candidate.len() {
                    cut = boundaries
                        .iter()
                        .copied()
                        .filter(|offset| *offset > 0 && *offset < candidate.len())
                        .filter(|offset| candidate.is_char_boundary(*offset))
                        .next_back()
                        .unwrap_or_else(|| {
                            candidate
                                .char_indices()
                                .next_back()
                                .map_or(0, |(offset, _)| offset)
                        });
                }
                if cut == 0 {
                    cut = candidate
                        .char_indices()
                        .next_back()
                        .map_or(0, |(offset, _)| offset);
                }
                ensure!(cut < candidate.len(), "suffix context crop stalled");
                end = start + cut;
            }
        }
    }
    anyhow::bail!("context crop failed to converge")
}

/// Frozen identity supplied by a selected research-model profile. The digest
/// covers these tokenizer fields and the complete control/user-defined token
/// inventory using `tokenizer_contract_bytes` below.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TokenizerProfile {
    pub tokenizer_id: String,
    pub tokenizer_revision: String,
    pub tokenizer_sha256: String,
    pub tokenizer_contract_sha256: String,
    pub tokenizer_vocab_size: usize,
    pub tokenizer_vocab_ids_sha256: String,
    #[serde(default, skip_serializing)]
    pub tokenizer_vocab_ids: Vec<i32>,
    pub eos_id: i32,
    pub fim_prefix_id: i32,
    pub fim_suffix_id: i32,
    pub fim_middle_id: i32,
    pub completion_mode: String,
    pub special_tokens: Vec<SpecialToken>,
}

/// Immutable identity of the selected model artifact and its matching FIM
/// tokenizer. The model-file digest is also checked directly at startup.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ServingProfile {
    pub artifact_manifest_sha256: String,
    pub tokenizer: TokenizerProfile,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecialToken {
    pub id: i32,
    pub spelling: String,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Text representation used by both Rust and Lua to bind every tokenizer
/// identity field and inventory entry without relying on JSON key ordering.
pub fn tokenizer_contract_bytes(profile: &TokenizerProfile) -> Result<Vec<u8>> {
    ensure!(
        !profile.tokenizer_id.is_empty()
            && !profile.tokenizer_revision.is_empty()
            && !profile.tokenizer_id.contains(['\n', '\r', '\t'])
            && !profile.tokenizer_revision.contains(['\n', '\r', '\t']),
        "invalid tokenizer identity fields"
    );
    ensure!(
        valid_sha256(&profile.tokenizer_sha256),
        "invalid tokenizer digest"
    );
    ensure!(
        profile.tokenizer_vocab_size > 0
            && profile.tokenizer_vocab_size == profile.tokenizer_vocab_ids.len()
            && valid_sha256(&profile.tokenizer_vocab_ids_sha256),
        "invalid tokenizer vocabulary identity"
    );
    ensure!(
        profile
            .tokenizer_vocab_ids
            .windows(2)
            .all(|pair| pair[0] >= 0 && pair[0] < pair[1]),
        "tokenizer vocabulary IDs must be sorted and unique"
    );
    ensure!(
        profile
            .tokenizer_vocab_ids
            .last()
            .is_some_and(|id| *id >= 0),
        "tokenizer vocabulary ID set is empty"
    );
    let vocab_bytes = profile
        .tokenizer_vocab_ids
        .iter()
        .map(|id| format!("{id}\n"))
        .collect::<String>();
    ensure!(
        format!("{:x}", sha2::Sha256::digest(vocab_bytes.as_bytes()))
            == profile.tokenizer_vocab_ids_sha256.to_ascii_lowercase(),
        "tokenizer vocabulary ID digest mismatch"
    );
    ensure!(
        [
            profile.eos_id,
            profile.fim_prefix_id,
            profile.fim_suffix_id,
            profile.fim_middle_id
        ]
        .iter()
        .all(|id| profile.tokenizer_vocab_ids.binary_search(id).is_ok()),
        "FIM markers are missing from the tokenizer vocabulary"
    );
    ensure!(
        profile.eos_id == EOS_TOKEN_ID
            && profile.fim_prefix_id == FIM_PREFIX_TOKEN_ID
            && profile.fim_suffix_id == FIM_SUFFIX_TOKEN_ID
            && profile.fim_middle_id == FIM_MIDDLE_TOKEN_ID,
        "FIM marker identity mismatch"
    );
    ensure!(
        profile.completion_mode == COMPLETION_MODE,
        "unsupported FIM completion mode"
    );
    let mut inventory = profile.special_tokens.clone();
    inventory.sort_by_key(|token| token.id);
    let mut ids = BTreeSet::new();
    let mut spellings = BTreeSet::new();
    for token in &inventory {
        ensure!(token.id >= 0, "negative special-token id");
        ensure!(
            !token.spelling.is_empty()
                && token.spelling.is_ascii()
                && !token.spelling.contains(['\n', '\r', '\t']),
            "invalid special-token spelling"
        );
        ensure!(ids.insert(token.id), "duplicate special-token id");
        ensure!(
            spellings.insert(token.spelling.as_str()),
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
            inventory
                .iter()
                .any(|token| token.id == id && token.spelling == spelling),
            "required FIM/EOS token missing from inventory"
        );
    }

    let mut bytes = format!(
        "q25-fim-tokenizer-contract-v1\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
        profile.tokenizer_id,
        profile.tokenizer_revision,
        profile.tokenizer_sha256.to_ascii_lowercase(),
        profile.eos_id,
        profile.fim_prefix_id,
        profile.fim_suffix_id,
        profile.fim_middle_id,
        profile.completion_mode,
        profile.tokenizer_vocab_size,
        profile.tokenizer_vocab_ids_sha256.to_ascii_lowercase(),
    )
    .into_bytes();
    for token in inventory {
        bytes.extend_from_slice(token.id.to_string().as_bytes());
        bytes.push(b'\t');
        bytes.extend_from_slice(token.spelling.as_bytes());
        bytes.push(b'\n');
    }
    Ok(bytes)
}

pub fn validate_serving_profile(profile: &ServingProfile) -> Result<()> {
    ensure!(
        valid_sha256(&profile.artifact_manifest_sha256),
        "invalid model artifact manifest digest"
    );
    let bytes = tokenizer_contract_bytes(&profile.tokenizer)?;
    ensure!(
        valid_sha256(&profile.tokenizer.tokenizer_contract_sha256)
            && format!("{:x}", sha2::Sha256::digest(bytes))
                == profile
                    .tokenizer
                    .tokenizer_contract_sha256
                    .to_ascii_lowercase(),
        "FIM tokenizer contract digest mismatch"
    );
    Ok(())
}

pub fn context_digest(
    request_id: &str,
    source: &str,
    target_row: usize,
    cursor_col: usize,
    prepared: &Prepared,
) -> Result<String> {
    ensure!(
        !request_id.is_empty()
            && request_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "invalid request identity"
    );
    ensure!(
        valid_sha256(&prepared.tokenizer_contract_sha256),
        "invalid tokenizer contract digest"
    );
    let canonical = format!(
        "q25-fim-context-v2\n{}\n{:x}\n{}\n{}\n{:x}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
        request_id,
        sha2::Sha256::digest(source.as_bytes()),
        target_row,
        cursor_col,
        sha2::Sha256::digest(prepared.prompt.as_bytes()),
        prepared.prefix_range.start_byte,
        prepared.prefix_range.end_byte,
        prepared.prefix_token_count,
        prepared.suffix_range.start_byte,
        prepared.suffix_range.end_byte,
        prepared.suffix_token_count,
        prepared.context_policy_version,
        prepared.context_layout,
        prepared.tokenizer_contract_sha256.to_ascii_lowercase(),
    );
    Ok(format!("{:x}", sha2::Sha256::digest(canonical.as_bytes())))
}

#[derive(Clone, Debug)]
pub struct TokenContract {
    tokenizer_sha256: String,
    tokenizer_contract_sha256: String,
    known_vocab_ids: BTreeSet<i32>,
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
        let known_vocab_ids = special_by_id
            .keys()
            .next_back()
            .map_or_else(BTreeSet::new, |maximum| (0..=*maximum).collect());
        Ok(Self {
            tokenizer_sha256: tokenizer_sha256.to_ascii_lowercase(),
            tokenizer_contract_sha256: String::new(),
            known_vocab_ids,
            special_by_id,
            special_spellings,
        })
    }

    pub fn from_profile(profile: &TokenizerProfile) -> Result<Self> {
        let mut contract = Self::new(
            &profile.tokenizer_sha256,
            &[profile.fim_prefix_id],
            &[profile.fim_suffix_id],
            &[profile.fim_middle_id],
            profile.eos_id,
            &profile.special_tokens,
        )?;
        contract.tokenizer_contract_sha256 = profile.tokenizer_contract_sha256.to_ascii_lowercase();
        contract.known_vocab_ids = profile.tokenizer_vocab_ids.iter().copied().collect();
        Ok(contract)
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
    pub tokenizer_contract_sha256: String,
    pub target_row: usize,
    pub cursor_col: usize,
    /// Retained source prefix, ending at the cursor byte.
    pub prefix_range: ByteRange,
    pub prefix_token_count: usize,
    /// Retained source suffix, starting after the target line terminator.
    pub suffix_range: ByteRange,
    pub suffix_token_count: usize,
    /// Cursor through the physical line ending, matching the model hole.
    pub model_hole_range: ByteRange,
    /// Whole-line replacement interval used by the existing canonical applier.
    /// It excludes the line ending, which Neovim retains as buffer metadata.
    pub apply_range: ByteRange,
    pub line_ending: LineEnding,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_window: Option<PrefixWindowMetadata>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionFailure {
    MissingEos,
    TerminalNotEos,
    OutputCapExceeded,
    TokenIdOutsideVocabulary,
    ControlTokenInBody,
    NulInBody,
    LiteralControlSpelling,
    LineEndingMismatch,
    MultipleLines,
}

impl CompletionFailure {
    pub fn code(self) -> &'static str {
        match self {
            Self::MissingEos => "fim_missing_eos",
            Self::TerminalNotEos => "fim_terminal_not_eos",
            Self::OutputCapExceeded => "fim_output_cap_exceeded",
            Self::TokenIdOutsideVocabulary => "fim_token_id_outside_vocabulary",
            Self::ControlTokenInBody => "fim_control_token_in_body",
            Self::NulInBody => "fim_nul_in_body",
            Self::LiteralControlSpelling => "fim_literal_control_spelling",
            Self::LineEndingMismatch => "fim_line_ending_mismatch",
            Self::MultipleLines => "fim_multiline_completion",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompletionCheck {
    Valid(DecodedCompletion),
    Invalid(CompletionFailure),
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
    let prefix_range = ByteRange::new(0, cursor_byte);
    let suffix_range = ByteRange::new(line_end, source.len());
    let prompt = format!(
        "{FIM_PREFIX}{}{FIM_SUFFIX}{}{FIM_MIDDLE}",
        &source[prefix_range.start_byte..prefix_range.end_byte],
        &source[suffix_range.start_byte..suffix_range.end_byte]
    );
    Ok(Prepared {
        prompt,
        model_protocol: WIRE_VERSION.into(),
        context_policy_version: CONTEXT_POLICY_VERSION.into(),
        context_layout: CONTEXT_LAYOUT.into(),
        tokenizer_sha256: tokens.tokenizer_sha256.clone(),
        tokenizer_contract_sha256: tokens.tokenizer_contract_sha256.clone(),
        target_row,
        cursor_col,
        prefix_range,
        prefix_token_count: 0,
        suffix_range,
        suffix_token_count: 0,
        model_hole_range: ByteRange::new(cursor_byte, line_end),
        apply_range: ByteRange::new(line_start, content_end),
        line_ending: ending,
        prefix_window: None,
        prefix_before_cursor,
        virtual_empty_file,
    })
}

impl Prepared {
    /// Replace the full-file PSM segments with exact bounded source ranges.
    /// The editor source itself and the canonical whole-line apply range stay
    /// unchanged. Retained text must remain byte-exact and NFC, matching the
    /// training tokenizer's normalization behavior.
    pub fn set_bounded_window(
        &mut self,
        source: &str,
        prefix_start_byte: usize,
        suffix_end_byte: usize,
        prefix_token_count: usize,
        suffix_token_count: usize,
    ) -> Result<()> {
        let prefix_end_byte = self.model_hole_range.start_byte;
        let suffix_start_byte = self.model_hole_range.end_byte;
        ensure!(
            prefix_start_byte <= prefix_end_byte
                && prefix_end_byte <= source.len()
                && source.is_char_boundary(prefix_start_byte)
                && source.is_char_boundary(prefix_end_byte),
            "invalid retained FIM prefix range"
        );
        ensure!(
            suffix_start_byte <= suffix_end_byte
                && suffix_end_byte <= source.len()
                && source.is_char_boundary(suffix_start_byte)
                && source.is_char_boundary(suffix_end_byte),
            "invalid retained FIM suffix range"
        );
        ensure!(
            prefix_token_count <= PREFIX_CONTEXT_TOKEN_LIMIT
                && suffix_token_count <= SUFFIX_CONTEXT_TOKEN_LIMIT,
            "retained FIM context exceeds its per-side token limit"
        );
        let prefix = &source[prefix_start_byte..prefix_end_byte];
        let suffix = &source[suffix_start_byte..suffix_end_byte];
        ensure!(
            is_nfc(prefix),
            "retained FIM prefix is not NFC-normalized; tokenizer parity is unavailable"
        );
        ensure!(
            is_nfc(suffix),
            "retained FIM suffix is not NFC-normalized; tokenizer parity is unavailable"
        );
        self.prefix_range = ByteRange::new(prefix_start_byte, prefix_end_byte);
        self.prefix_token_count = prefix_token_count;
        self.suffix_range = ByteRange::new(suffix_start_byte, suffix_end_byte);
        self.suffix_token_count = suffix_token_count;
        self.prompt = format!("{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}");
        Ok(())
    }
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
    match check_completion(
        prepared,
        model_protocol,
        output_cap,
        raw_text,
        content_token_ids,
        terminal_token_id,
        tokens,
    )? {
        CompletionCheck::Valid(decoded) => Ok(decoded),
        CompletionCheck::Invalid(failure) => {
            anyhow::bail!("FIM completion validation failed ({})", failure.code())
        }
    }
}

pub fn check_completion(
    prepared: &Prepared,
    model_protocol: &str,
    output_cap: usize,
    raw_text: &str,
    content_token_ids: &[i32],
    terminal_token_id: Option<i32>,
    tokens: &TokenContract,
) -> Result<CompletionCheck> {
    ensure!(
        model_protocol == WIRE_VERSION,
        "FIM model protocol mismatch"
    );
    ensure!(
        prepared.tokenizer_sha256 == tokens.tokenizer_sha256,
        "FIM context tokenizer identity mismatch"
    );
    ensure!(output_cap == OUTPUT_TOKEN_CAP, "FIM output cap mismatch");
    let Some(terminal) = terminal_token_id else {
        return Ok(CompletionCheck::Invalid(CompletionFailure::MissingEos));
    };
    if terminal != EOS_TOKEN_ID {
        return Ok(CompletionCheck::Invalid(CompletionFailure::TerminalNotEos));
    }
    let Some(generated_steps) = content_token_ids.len().checked_add(1) else {
        return Ok(CompletionCheck::Invalid(
            CompletionFailure::OutputCapExceeded,
        ));
    };
    if generated_steps > OUTPUT_TOKEN_CAP {
        return Ok(CompletionCheck::Invalid(
            CompletionFailure::OutputCapExceeded,
        ));
    }
    for id in content_token_ids {
        if *id < 0 || !tokens.known_vocab_ids.contains(id) {
            return Ok(CompletionCheck::Invalid(
                CompletionFailure::TokenIdOutsideVocabulary,
            ));
        }
        if tokens.special_by_id.contains_key(id) {
            return Ok(CompletionCheck::Invalid(
                CompletionFailure::ControlTokenInBody,
            ));
        }
    }
    if raw_text.contains('\0') {
        return Ok(CompletionCheck::Invalid(CompletionFailure::NulInBody));
    }
    if tokens
        .reject_special_spellings(raw_text, "completion")
        .is_err()
    {
        return Ok(CompletionCheck::Invalid(
            CompletionFailure::LiteralControlSpelling,
        ));
    }

    let line_ending = prepared.line_ending.bytes();
    let body = if line_ending.is_empty() {
        if raw_text.contains(['\r', '\n']) {
            return Ok(CompletionCheck::Invalid(
                CompletionFailure::LineEndingMismatch,
            ));
        }
        raw_text
    } else {
        if !raw_text.ends_with(line_ending) {
            return Ok(CompletionCheck::Invalid(
                CompletionFailure::LineEndingMismatch,
            ));
        }
        let body = &raw_text[..raw_text.len() - line_ending.len()];
        if body.contains(['\r', '\n']) {
            return Ok(CompletionCheck::Invalid(CompletionFailure::MultipleLines));
        }
        body
    };
    let replacement = format!("{}{body}", prepared.prefix_before_cursor);
    Ok(CompletionCheck::Valid(DecodedCompletion {
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
    }))
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

    fn serving_profile() -> ServingProfile {
        let tokenizer_vocab_ids = vec![17, 151_643, 151_644, 151_645, 151_659, 151_660, 151_661];
        let tokenizer_vocab_ids_sha256 = format!(
            "{:x}",
            sha2::Sha256::digest(
                tokenizer_vocab_ids
                    .iter()
                    .map(|id| format!("{id}\n"))
                    .collect::<String>()
                    .as_bytes()
            )
        );
        let mut tokenizer = TokenizerProfile {
            tokenizer_id: "synthetic/fim-fixture".into(),
            tokenizer_revision: "synthetic-revision".into(),
            tokenizer_sha256: TEST_TOKENIZER_SHA256.into(),
            tokenizer_contract_sha256: String::new(),
            tokenizer_vocab_size: tokenizer_vocab_ids.len(),
            tokenizer_vocab_ids_sha256,
            tokenizer_vocab_ids,
            eos_id: EOS_TOKEN_ID,
            fim_prefix_id: FIM_PREFIX_TOKEN_ID,
            fim_suffix_id: FIM_SUFFIX_TOKEN_ID,
            fim_middle_id: FIM_MIDDLE_TOKEN_ID,
            completion_mode: COMPLETION_MODE.into(),
            special_tokens: vec![
                SpecialToken {
                    id: EOS_TOKEN_ID,
                    spelling: EOS_SPELLING.into(),
                },
                SpecialToken {
                    id: 151_644,
                    spelling: "<|im_start|>".into(),
                },
                SpecialToken {
                    id: 151_645,
                    spelling: "<|im_end|>".into(),
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
        };
        tokenizer.tokenizer_contract_sha256 = format!(
            "{:x}",
            sha2::Sha256::digest(tokenizer_contract_bytes(&tokenizer).unwrap())
        );
        ServingProfile {
            artifact_manifest_sha256: "b".repeat(64),
            tokenizer,
        }
    }

    #[test]
    fn serving_profile_binds_tokenizer_and_known_vocabulary_ids() {
        let profile = serving_profile();
        assert!(validate_serving_profile(&profile).is_ok());
        let mut bad_vocab = profile.clone();
        bad_vocab.tokenizer.tokenizer_vocab_ids[0] = 18;
        assert!(validate_serving_profile(&bad_vocab).is_err());
        let mut bad_markers = profile;
        bad_markers.tokenizer.fim_prefix_id += 1;
        assert!(validate_serving_profile(&bad_markers).is_err());
    }

    #[test]
    fn context_digest_binds_request_source_cursor_prompt_and_tokenizer() {
        let request = "12345678-1234-1234-1234-123456789abc";
        let source = "x=1\n";
        let mut prepared = prepare(source, 0, 2, &tokens()).unwrap();
        prepared.tokenizer_contract_sha256 = "c".repeat(64);
        prepared
            .set_bounded_window(source, 0, source.len(), 2, 0)
            .unwrap();
        let baseline = context_digest(request, source, 0, 2, &prepared).unwrap();
        assert_ne!(
            context_digest(
                "22345678-1234-1234-1234-123456789abc",
                source,
                0,
                2,
                &prepared
            )
            .unwrap(),
            baseline
        );
        assert_ne!(
            context_digest(request, "y=1\n", 0, 2, &prepared).unwrap(),
            baseline
        );
        assert_ne!(
            context_digest(request, source, 1, 2, &prepared).unwrap(),
            baseline
        );
        assert_ne!(
            context_digest(request, source, 0, 3, &prepared).unwrap(),
            baseline
        );

        let mut changed = prepared.clone();
        changed.prompt.push('!');
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
        let mut changed = prepared.clone();
        changed.prefix_range.start_byte = 1;
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
        let mut changed = prepared.clone();
        changed.prefix_token_count += 1;
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
        let mut changed = prepared.clone();
        changed.suffix_range.end_byte = 1;
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
        let mut changed = prepared.clone();
        changed.suffix_token_count += 1;
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
        let mut changed = prepared.clone();
        changed.context_policy_version.push_str("-changed");
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
        let mut changed = prepared.clone();
        changed.context_layout.push_str("-changed");
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
        let mut changed = prepared;
        changed.tokenizer_contract_sha256 = "d".repeat(64);
        assert_ne!(
            context_digest(request, source, 0, 2, &changed).unwrap(),
            baseline
        );
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
        assert_eq!(prepared.prefix_range, ByteRange::new(0, 2));
        assert_eq!(prepared.suffix_range, ByteRange::new(5, 12));
        let metadata = serde_json::to_value(&prepared).unwrap();
        assert_eq!(metadata["line_ending"], "LF");
        assert_eq!(metadata["model_hole_range"]["end_exclusive"], true);
    }

    #[test]
    fn aligned_prefix_boundaries_keep_absolute_token_buckets() {
        let scalar_pieces = |text: &str| -> Result<Vec<Vec<u8>>> {
            Ok(text.chars().map(|c| c.to_string().into_bytes()).collect())
        };
        for (count, start, retained) in [
            (640, 0, 640),
            (641, 128, 513),
            (768, 128, 640),
            (769, 256, 513),
        ] {
            let source = "a".repeat(count - 1) + "\n";
            let (aligned, metadata) = crop_prefix_context(
                &source,
                source.len(),
                ContextWindow::Aligned128V1,
                scalar_pieces,
            )
            .unwrap();
            assert_eq!(aligned.start_byte, start);
            assert_eq!(aligned.token_count, retained);
            assert_eq!(metadata.unwrap().fallback_reason, None);
            let (default, metadata) = crop_prefix_context(
                &source,
                source.len(),
                ContextWindow::SlidingV2,
                scalar_pieces,
            )
            .unwrap();
            assert_eq!(
                default,
                crop_context(&source, 640, ContextSide::KeepRight, scalar_pieces).unwrap()
            );
            assert!(metadata.is_none());
        }
    }

    #[test]
    fn aligned_prefix_utf8_pieces_and_editable_line_fallback_are_exact() {
        let byte_pieces = |text: &str| -> Result<Vec<Vec<u8>>> {
            Ok(text.as_bytes().iter().map(|byte| vec![*byte]).collect())
        };
        let source = "a".repeat(127) + "🙂" + &"b".repeat(514);
        let (crop, metadata) = crop_prefix_context(
            &source,
            source.len(),
            ContextWindow::Aligned128V1,
            byte_pieces,
        )
        .unwrap();
        assert_eq!(crop.start_byte, 131);
        assert_eq!(crop.token_count, 514);
        assert!(source.is_char_boundary(crop.start_byte));
        assert_eq!(metadata.unwrap().aligned_start_token, 128);

        for (source, line_start) in [
            ("a".repeat(50) + "\n" + &"b".repeat(620), 51),
            ("b".repeat(700), 0),
        ] {
            let (crop, metadata) = crop_prefix_context(
                &source,
                line_start,
                ContextWindow::Aligned128V1,
                byte_pieces,
            )
            .unwrap();
            assert_eq!(
                crop,
                crop_context(&source, 640, ContextSide::KeepRight, byte_pieces).unwrap()
            );
            assert_eq!(
                metadata.unwrap().fallback_reason,
                Some("editable_line_prefix")
            );
        }
    }

    #[test]
    fn aligned_prefix_retokenization_budget_falls_back_without_repair() {
        let source = "x".repeat(1300);
        let tokenize = |part: &str| -> Result<Vec<Vec<u8>>> {
            Ok(if part == source {
                part.as_bytes()
                    .chunks(2)
                    .map(|piece| piece.to_vec())
                    .collect()
            } else {
                part.as_bytes().iter().map(|byte| vec![*byte]).collect()
            })
        };
        let (crop, metadata) =
            crop_prefix_context(&source, source.len(), ContextWindow::Aligned128V1, tokenize)
                .unwrap();
        assert_eq!(crop.token_count, 640);
        assert_eq!(crop.start_byte, 660);
        assert_eq!(
            metadata.unwrap().fallback_reason,
            Some("retokenized_budget")
        );
        let bad_pieces = |part: &str| -> Result<Vec<Vec<u8>>> {
            Ok(part.as_bytes().iter().map(|_| vec![b'z']).collect())
        };
        assert!(
            crop_prefix_context(
                &source,
                source.len(),
                ContextWindow::Aligned128V1,
                bad_pieces
            )
            .is_err()
        );
    }

    #[test]
    fn aligned_policy_keeps_crlf_eof_application_ranges_and_changes_digest() {
        for ending in ["\r\n", ""] {
            let source = "a".repeat(700) + "\nleft" + ending + "tail";
            let mut prepared = prepare(&source, 1, 2, &tokens()).unwrap();
            prepared.tokenizer_contract_sha256 = "c".repeat(64);
            let original = prepared.clone();
            let scalar_pieces = |text: &str| -> Result<Vec<Vec<u8>>> {
                Ok(text.chars().map(|c| c.to_string().into_bytes()).collect())
            };
            let (prefix, metadata) = crop_prefix_context(
                &source[..prepared.model_hole_range.start_byte],
                prepared.apply_range.start_byte,
                ContextWindow::Aligned128V1,
                scalar_pieces,
            )
            .unwrap();
            prepared
                .set_bounded_window(
                    &source,
                    prefix.start_byte,
                    source.len(),
                    prefix.token_count,
                    source[prepared.model_hole_range.end_byte..].chars().count(),
                )
                .unwrap();
            let baseline_digest = context_digest("test", &source, 1, 2, &prepared).unwrap();
            prepared.context_policy_version = ALIGNED_CONTEXT_POLICY_VERSION.into();
            prepared.prefix_window = metadata;
            assert_ne!(
                baseline_digest,
                context_digest("test", &source, 1, 2, &prepared).unwrap()
            );
            assert_eq!(prepared.apply_range, original.apply_range);
            assert_eq!(prepared.model_hole_range, original.model_hole_range);
            assert_eq!(prepared.line_ending, original.line_ending);
            assert!(prepared.prompt.starts_with(FIM_PREFIX));
            assert!(prepared.prompt.ends_with(FIM_MIDDLE));
        }
    }

    #[test]
    fn bounded_crop_matches_utf8_offset_windows_for_unicode_whitespace_and_crlf() {
        let scalar_pieces = |text: &str| -> Result<Vec<Vec<u8>>> {
            Ok(text
                .chars()
                .map(|character| character.to_string().into_bytes())
                .collect())
        };
        let source = "ab🙂cd";
        let prefix = crop_context(source, 3, ContextSide::KeepRight, scalar_pieces).unwrap();
        assert_eq!(&source[prefix.start_byte..prefix.end_byte], "🙂cd");
        assert_eq!(prefix.start_byte, 2);
        assert_eq!(prefix.token_count, 3);
        let suffix = crop_context(source, 3, ContextSide::KeepLeft, scalar_pieces).unwrap();
        assert_eq!(&source[suffix.start_byte..suffix.end_byte], "ab🙂");
        assert_eq!(suffix.end_byte, 6);

        let combining = "a\u{301}bc";
        let prefix = crop_context(combining, 2, ContextSide::KeepRight, scalar_pieces).unwrap();
        assert_eq!(&combining[prefix.start_byte..prefix.end_byte], "bc");
        assert!(combining.is_char_boundary(prefix.start_byte));
        let whitespace_crlf = "left   \r\n  right";
        let suffix =
            crop_context(whitespace_crlf, 7, ContextSide::KeepLeft, scalar_pieces).unwrap();
        assert_eq!(
            &whitespace_crlf[suffix.start_byte..suffix.end_byte],
            "left   "
        );
        let prefix =
            crop_context(whitespace_crlf, 4, ContextSide::KeepRight, scalar_pieces).unwrap();
        assert_eq!(&whitespace_crlf[prefix.start_byte..prefix.end_byte], "ight");
    }

    #[test]
    fn byte_fallback_offsets_expand_to_utf8_boundaries_then_retokenize() {
        let byte_pieces = |text: &str| -> Result<Vec<Vec<u8>>> {
            Ok(text.as_bytes().iter().map(|byte| vec![*byte]).collect())
        };
        let source = "x🙂y";
        let prefix = crop_context(source, 4, ContextSide::KeepRight, byte_pieces).unwrap();
        assert_eq!(&source[prefix.start_byte..prefix.end_byte], "y");
        assert_eq!(prefix.start_byte, 5);
        let suffix = crop_context(source, 4, ContextSide::KeepLeft, byte_pieces).unwrap();
        assert_eq!(&source[suffix.start_byte..suffix.end_byte], "x");
        assert_eq!(suffix.end_byte, 1);
    }

    #[test]
    fn nfc_guard_checks_only_the_retained_prompt_segments() {
        let decomposed = "e\u{301}=1\nnext()\n";
        let mut prepared = prepare(decomposed, 0, 3, &tokens()).unwrap();
        assert!(
            prepared
                .set_bounded_window(decomposed, 0, decomposed.len(), 3, 0)
                .is_err()
        );

        let retained_ascii = "e\u{301}=1\nnext()\n";
        let mut prepared = prepare(retained_ascii, 1, 2, &tokens()).unwrap();
        let line_start = "e\u{301}=1\n".len();
        prepared
            .set_bounded_window(retained_ascii, line_start, retained_ascii.len(), 2, 0)
            .unwrap();
        assert_eq!(prepared.prefix_range.start_byte, line_start);
        assert!(prepared.prompt.contains("<|fim_prefix|>ne"));
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
    fn bound_invalid_completions_have_stable_failure_codes() {
        let prepared = prepare("x\n", 0, 1, &tokens()).unwrap();
        let token_contract = tokens();
        let cases = [
            ("\n", vec![], None, "fim_missing_eos"),
            ("\n", vec![], Some(151_644), "fim_terminal_not_eos"),
            (
                "\n",
                vec![17; OUTPUT_TOKEN_CAP],
                Some(EOS_TOKEN_ID),
                "fim_output_cap_exceeded",
            ),
            (
                "\n",
                vec![999_999],
                Some(EOS_TOKEN_ID),
                "fim_token_id_outside_vocabulary",
            ),
            (
                "<|im_start|>\n",
                vec![151_644],
                Some(EOS_TOKEN_ID),
                "fim_control_token_in_body",
            ),
            ("value\0\n", vec![17], Some(EOS_TOKEN_ID), "fim_nul_in_body"),
            (
                "<|im_start|>\n",
                vec![17],
                Some(EOS_TOKEN_ID),
                "fim_literal_control_spelling",
            ),
            (
                "value",
                vec![17],
                Some(EOS_TOKEN_ID),
                "fim_line_ending_mismatch",
            ),
            (
                "value\nmore\n",
                vec![17],
                Some(EOS_TOKEN_ID),
                "fim_multiline_completion",
            ),
        ];
        for (raw, ids, terminal, expected_code) in cases {
            let CompletionCheck::Invalid(failure) = check_completion(
                &prepared,
                WIRE_VERSION,
                OUTPUT_TOKEN_CAP,
                raw,
                &ids,
                terminal,
                &token_contract,
            )
            .unwrap() else {
                panic!("expected bound invalid completion {expected_code}");
            };
            assert_eq!(failure.code(), expected_code);
        }
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
