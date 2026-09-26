# Frozen shared schema: `single-line-edit-v1`, suite revision 3

This document fixes the interface shared by the contract, data, evaluation, and training work. Revision 2 was made before any cases or model results: revision 1's EOF rule collapsed an empty inserted/replaced line into zero bytes. Revision 3 was also made before cases or model results: revision 2's final-line deletion rule could remove a preceding blank physical line as well. Prior frozen plans are retained. A result-changing correction requires another numbered suite/plan revision before more results are produced.

## State and action

An `EditState` contains `file_id` (opaque public-source identity), `filetype` (`python`, `typescript`, `rust`, or `go`), exact UTF-8 `source`, zero-based `target_row`, zero-based UTF-8 byte `cursor_col`, ordered prior edits with exact old/new text, and bounded relevant definitions/imports. `target_row` must identify an existing physical line, except insertion may use `line_count` to append. The model prompt includes the visible state and history, but never provenance, split, mechanism, target action, test specification, or future state.

Canonical actions are `keep`, `replace_line(text)`, `delete_line`, and `insert_before(text)`. `text` may be empty or contain tabs and Unicode; it contains no CR or LF. `replace_line("")` creates a blank physical line. An empty file has zero lines: only `keep` or `insert_before` at row zero is valid. A trailing line terminator does not create an additional physical line.

The reference application operates on UTF-8 bytes and preserves untouched bytes. Replacement retains the target's terminator. Replacing an unterminated line with empty text materializes a blank line with the file's predominant terminator or LF. Deletion removes the target's content and terminator. For an unterminated final line, deletion removes only that line's content and leaves the preceding line's terminator; this guarantees that exactly one physical line disappears even when the preceding line is blank. Insertion before an existing row writes `text` plus that row's terminator style, defaulting to the file's predominant style then LF if the row is unterminated. EOF append writes nonempty `text` without an additional trailing terminator; if the existing last line is unterminated, prepend its predominant terminator or LF. Appending empty text writes a physical blank line by adding one terminator after the empty payload, including for an initially empty file. Existing final newline remains before appended text. The file's existing LF/CRLF bytes are never normalized. Cursor byte column must fall on a UTF-8 boundary and within the target line's content.

## Wire and tokenization

The response is exactly `N`, `D`, `R\t` plus one-line payload, or `I\t` plus one-line payload, immediately followed by the tokenizer's actual EOS token. No literal newline is part of the response. The raw response must be complete and terminated; a 64-token cap or malformed delimiter is failure. Header and payload are decoded without trimming. Prompt and response are tokenized with the immutable q25 tokenizer revision `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`. Prompt positions and padding are masked by position; response and EOS are supervised. Default input limit is 1,024 tokens; total sequence limit is 2,048.

## Record and isolation

Each data row has stable `id`, `state`, `action`, exact `after_source`, `source_type`, `source_repo`, `source_revision`, `source_license`, `session_or_commit`, `mechanism`, `generator_family`, `template_id`, `provenance`, and validation evidence. Source group, commit/session, template family, and normalized near duplicates are grouped before train/development/test assignment. A test row is immutable after the split manifest is frozen. The evaluator reads training metadata for scoring and grouping, but constructs model input only through the reference serializer.

## Provider boundary

Current OpenCode Terms of Use (effective 2026-08-15) bar programmatic extraction of output and use of output to develop AI models competing with the service or third-party models. Accordingly, OpenCode output is excluded from student labels and automated benchmark scoring pending a separate compatible rights basis. Public source edit history and local deterministic checks remain authorized routes. No provider request may contain private editor feedback, secrets, or sealed test data.
