"""Byte-exact reference contract for single-line-edit-v1.

Rows name physical lines separated by LF or CRLF. A final terminator belongs to
the preceding line and does not create a phantom empty line.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

WIRE_VERSION = "single-line-edit-v1"
MAX_ACTION_TOKENS = 64
ActionKind = Literal["keep", "replace_line", "delete_line", "insert_before"]
Filetype = Literal["python", "typescript", "rust", "go"]


@dataclass(frozen=True)
class RecentEdit:
    row: int
    old_text: str
    new_text: str

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("history row must be nonnegative")
        self.old_text.encode("utf-8")
        self.new_text.encode("utf-8")


@dataclass(frozen=True)
class EditState:
    file_id: str
    filetype: Filetype
    source: str
    target_row: int
    cursor_col: int
    history: tuple[RecentEdit, ...] = field(default_factory=tuple)
    relevant: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.file_id or self.filetype not in ("python", "typescript", "rust", "go"):
            raise ValueError("file identity and supported filetype are required")
        source = self.source.encode("utf-8")
        lines = physical_lines(source)
        if self.target_row < 0 or self.target_row > len(lines):
            raise ValueError("target row is outside the file")
        if self.target_row == len(lines):
            if self.cursor_col != 0:
                raise ValueError("EOF insertion cursor column must be zero")
        else:
            content = lines[self.target_row].content
            if self.cursor_col < 0 or self.cursor_col > len(content):
                raise ValueError("cursor byte column is outside target line content")
            try:
                content[: self.cursor_col].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("cursor byte column splits a UTF-8 character") from error
        for item in self.history:
            if not isinstance(item, RecentEdit):
                raise TypeError("history must contain RecentEdit values")
        for snippet in self.relevant:
            snippet.encode("utf-8")

    @property
    def line_count(self) -> int:
        return len(physical_lines(self.source.encode("utf-8")))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> EditState:
        """Read JSONL state rows without accepting answer-bearing fields as input."""

        def integer(value: object) -> int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("state row/column must be integers")
            return value

        history_raw = raw.get("history", ())
        relevant_raw = raw.get("relevant", ())
        if not isinstance(history_raw, (list, tuple)) or not isinstance(
            relevant_raw, (list, tuple)
        ):
            raise TypeError("history and relevant must be sequences")
        history = []
        for entry in history_raw:
            if not isinstance(entry, Mapping):
                raise TypeError("history entry must be a mapping")
            history.append(
                RecentEdit(
                    row=integer(entry["row"]),
                    old_text=str(entry["old_text"]),
                    new_text=str(entry["new_text"]),
                )
            )
        return cls(
            file_id=str(raw["file_id"]),
            filetype=cast(Filetype, str(raw["filetype"])),
            source=str(raw["source"]),
            target_row=integer(raw["target_row"]),
            cursor_col=integer(raw["cursor_col"]),
            history=tuple(history),
            relevant=tuple(str(snippet) for snippet in relevant_raw),
        )


@dataclass(frozen=True)
class PhysicalLine:
    content: bytes
    terminator: bytes

    @property
    def raw(self) -> bytes:
        return self.content + self.terminator


def physical_lines(source: bytes) -> tuple[PhysicalLine, ...]:
    """Split only on LF/CRLF; reject lone CR instead of normalizing it."""
    if not source:
        return ()
    if b"\r" in source.replace(b"\r\n", b""):
        raise ValueError("lone CR is outside the v1 line model")
    result = []
    start = 0
    for index, byte in enumerate(source):
        if byte == 10:
            crlf = index > start and source[index - 1] == 13
            content = source[start : index - 1 if crlf else index]
            result.append(PhysicalLine(content, b"\r\n" if crlf else b"\n"))
            start = index + 1
    if start < len(source):
        result.append(PhysicalLine(source[start:], b""))
    return tuple(result)


@dataclass(frozen=True)
class EditAction:
    kind: ActionKind
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind in ("replace_line", "insert_before"):
            if self.text is None or "\r" in self.text or "\n" in self.text:
                raise ValueError("line action requires one physical line of text")
            self.text.encode("utf-8")
        elif self.kind in ("keep", "delete_line"):
            if self.text is not None:
                raise ValueError("keep/delete actions have no text")
        else:
            raise ValueError("unsupported action")


def _predominant_terminator(lines: tuple[PhysicalLine, ...]) -> bytes:
    lf = sum(line.terminator == b"\n" for line in lines)
    crlf = sum(line.terminator == b"\r\n" for line in lines)
    if lf == crlf:
        return next((line.terminator for line in lines if line.terminator), b"\n")
    return b"\r\n" if crlf > lf else b"\n"


def apply_action(state: EditState, action: EditAction) -> str:
    """Apply one action to the target row, preserving all untouched UTF-8 bytes."""
    source = state.source.encode("utf-8")
    lines = list(physical_lines(source))
    row = state.target_row
    if action.kind == "keep":
        return state.source
    if action.kind in ("replace_line", "delete_line") and row == len(lines):
        raise ValueError("replace/delete require an existing target line")
    if action.kind == "replace_line":
        assert action.text is not None
        terminator = lines[row].terminator
        if not action.text and not terminator:
            # An unterminated empty payload would otherwise collapse to the
            # empty file or lose the newly created final blank physical line.
            terminator = _predominant_terminator(tuple(lines))
        lines[row] = PhysicalLine(action.text.encode("utf-8"), terminator)
    elif action.kind == "delete_line":
        del lines[row]
    elif action.kind == "insert_before":
        assert action.text is not None
        payload = action.text.encode("utf-8")
        if row < len(lines):
            style = lines[row].terminator or _predominant_terminator(tuple(lines))
            lines.insert(row, PhysicalLine(payload, style))
        else:
            if lines and not lines[-1].terminator:
                lines[-1] = PhysicalLine(lines[-1].content, _predominant_terminator(tuple(lines)))
            lines.append(
                PhysicalLine(payload, _predominant_terminator(tuple(lines)) if not payload else b"")
            )
    else:
        raise ValueError("unsupported action")
    return b"".join(line.raw for line in lines).decode("utf-8")


def encode_action(action: EditAction) -> str:
    """Return the payload before the tokenizer's real EOS token."""
    if action.kind == "keep":
        return "N"
    if action.kind == "delete_line":
        return "D"
    assert action.text is not None
    return ("R" if action.kind == "replace_line" else "I") + "\t" + action.text


@dataclass(frozen=True)
class DecodedAction:
    status: Literal["ok", "incomplete", "malformed"]
    action: EditAction | None
    terminated: bool


def decode_action(
    text: str,
    *,
    terminated: bool,
    generated_tokens: int | None = None,
    max_tokens: int = MAX_ACTION_TOKENS,
) -> DecodedAction:
    """A cutoff or cancellation cannot be mistaken for a valid no-edit."""
    if max_tokens < 1 or max_tokens > MAX_ACTION_TOKENS:
        raise ValueError("invalid action token ceiling")
    if generated_tokens is not None and (generated_tokens < 1 or generated_tokens > max_tokens):
        return DecodedAction("incomplete", None, False)
    if not terminated:
        return DecodedAction("incomplete", None, False)
    if text == "N":
        return DecodedAction("ok", EditAction("keep"), True)
    if text == "D":
        return DecodedAction("ok", EditAction("delete_line"), True)
    if len(text) >= 2 and text[:2] in ("R\t", "I\t"):
        try:
            action = EditAction("replace_line" if text[0] == "R" else "insert_before", text[2:])
        except (UnicodeError, ValueError):
            return DecodedAction("malformed", None, True)
        return DecodedAction("ok", action, True)
    return DecodedAction("malformed", None, True)
