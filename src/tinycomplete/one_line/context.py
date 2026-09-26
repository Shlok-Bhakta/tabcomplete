"""Canonical, answer-free single-line edit prompt construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tinycomplete.one_line.contract import EditState, physical_lines

CONTEXT_POLICY_VERSION = "single-line-context-v2"
DEFAULT_INPUT_TOKENS = 1024
MAX_TOTAL_TOKENS = 2048


@dataclass(frozen=True)
class SerializedContext:
    text: str
    input_tokens: int | None
    included_rows: tuple[int, ...]
    included_history: int
    included_relevant: int
    omitted_rows: int


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render(
    state: EditState,
    *,
    rows: set[int],
    history_indices: set[int],
    relevant_indices: set[int],
) -> str:
    lines = physical_lines(state.source.encode("utf-8"))
    section = [
        "<single-line-edit-v1>",
        "Actions: N, D, R\t, I\t. R/I prefixes take one source line of text.",
        "The gap after R/I is a literal tab. No CR, LF, or explanation; end with tokenizer EOS.",
        f"File: {_quoted(state.file_id)}",
        f"Filetype: {state.filetype}",
        f"Target row (zero based): {state.target_row}",
        f"Cursor byte column: {state.cursor_col}",
        f"Physical line count: {len(lines)}",
        "Current source lines (JSON strings; indices are original rows):",
    ]
    for row in sorted(rows):
        line = lines[row]
        terminator = "CRLF" if line.terminator == b"\r\n" else "LF" if line.terminator else "EOF"
        section.append(f"{row} [{terminator}] {_quoted(line.content.decode('utf-8'))}")
    if state.target_row == len(lines):
        section.append(f"{state.target_row} [APPEND AT EOF]")
    section.append("Recent edits, oldest to newest (JSON old and new text):")
    for index in sorted(history_indices):
        edit = state.history[index]
        section.append(
            f"{index} row={edit.row} old={_quoted(edit.old_text)} new={_quoted(edit.new_text)}"
        )
    section.append("Relevant definitions and imports (JSON strings):")
    for index in sorted(relevant_indices):
        section.append(f"{index} {_quoted(state.relevant[index])}")
    section.extend(("</single-line-edit-v1>", "Action:"))
    return "\n".join(section)


def _tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def serialize_state_bounded(
    state: EditState,
    tokenizer: Any | None = None,
    *,
    max_input_tokens: int = DEFAULT_INPUT_TOKENS,
) -> SerializedContext:
    """Select whole evidence items with exact tokenizer accounting.

    The target line is mandatory. Nearby lines, latest edits, relevant symbols,
    then the remaining source are included in that order. No marker or line is
    partially truncated. Without a tokenizer, every item is included for tests.
    """
    if max_input_tokens < 1 or max_input_tokens > MAX_TOTAL_TOKENS:
        raise ValueError("input token limit is outside the validated range")
    lines = physical_lines(state.source.encode("utf-8"))
    target = state.target_row
    rows: set[int] = {target} if target < len(lines) else set()
    history_indices: set[int] = set()
    relevant_indices: set[int] = set()

    def render() -> str:
        return _render(
            state,
            rows=rows,
            history_indices=history_indices,
            relevant_indices=relevant_indices,
        )

    if tokenizer is not None and _tokens(tokenizer, render()) > max_input_tokens:
        raise ValueError("mandatory target and control markers exceed input token budget")

    def try_add(bucket: set[int], index: int) -> None:
        bucket.add(index)
        if tokenizer is not None and _tokens(tokenizer, render()) > max_input_tokens:
            bucket.remove(index)

    neighborhood = []
    for distance in range(1, 4):
        for row in (target - distance, target + distance):
            if 0 <= row < len(lines) and row not in rows:
                neighborhood.append(row)
    for row in neighborhood:
        try_add(rows, row)
    for index in range(len(state.history) - 1, -1, -1):
        try_add(history_indices, index)
    for index in range(len(state.relevant)):
        try_add(relevant_indices, index)
    # Nearby source is already represented. Bound further tokenization work per
    # state; distant symbols should arrive through the explicit relevant set.
    for distance in range(4, 21):
        for row in (target - distance, target + distance):
            if 0 <= row < len(lines) and row not in rows:
                try_add(rows, row)
    text = render()
    token_count = _tokens(tokenizer, text) if tokenizer is not None else None
    return SerializedContext(
        text=text,
        input_tokens=token_count,
        included_rows=tuple(sorted(rows)),
        included_history=len(history_indices),
        included_relevant=len(relevant_indices),
        omitted_rows=len(lines) - len(rows),
    )


def serialize_state(
    state: EditState,
    tokenizer: Any | None = None,
    *,
    max_input_tokens: int = DEFAULT_INPUT_TOKENS,
) -> str:
    return serialize_state_bounded(state, tokenizer, max_input_tokens=max_input_tokens).text
