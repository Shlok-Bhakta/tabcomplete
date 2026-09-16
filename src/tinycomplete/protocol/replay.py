"""Deterministic replay: fold an append-only event stream into editor state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .events import (
    AcceptCompletion,
    BaseEvent,
    CloseFile,
    Cursor,
    Delete,
    Diagnostics,
    Insert,
    OpenFile,
    PartialAcceptCompletion,
    Predict,
    RejectCompletion,
    Replace,
    RetrievedContext,
    Snapshot,
    byte_len,
    byte_splice,
    check_boundary,
)

__all__ = ["EditorState", "replay", "apply_event"]


@dataclass
class EditorState:
    files: dict[str, str] = field(default_factory=dict)
    cursors: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, tuple] = field(default_factory=dict)
    open_files: list[str] = field(default_factory=list)
    retrieved: list[RetrievedContext] = field(default_factory=list)
    completions: dict[str, str] = field(default_factory=dict)
    last_predict: Predict | None = None

    def file_bytes(self, path: str) -> int:
        return byte_len(self.files.get(path, ""))


def _require_open(state: EditorState, path: str, kind: str) -> str:
    if path not in state.files:
        raise ValueError(f"{kind}: file not open: {path!r}")
    return state.files[path]


def apply_event(state: EditorState, event: BaseEvent) -> EditorState:
    """Apply one event to mutable draft state. Returns the same state."""
    if isinstance(event, OpenFile):
        if event.path not in state.files:
            state.files[event.path] = ""
        if event.path not in state.open_files:
            state.open_files.append(event.path)
    elif isinstance(event, CloseFile):
        if event.path in state.open_files:
            state.open_files.remove(event.path)
    elif isinstance(event, Snapshot):
        state.files[event.path] = event.content
        if event.path not in state.open_files:
            state.open_files.append(event.path)
    elif isinstance(event, Insert):
        current = _require_open(state, event.path, "INSERT")
        check_boundary(current, event.offset)
        state.files[event.path] = byte_splice(current, event.offset, event.offset, event.text)
    elif isinstance(event, Delete):
        current = _require_open(state, event.path, "DELETE")
        state.files[event.path] = byte_splice(current, event.start, event.end, "")
    elif isinstance(event, Replace):
        current = _require_open(state, event.path, "REPLACE")
        state.files[event.path] = byte_splice(current, event.start, event.end, event.text)
    elif isinstance(event, Cursor):
        _require_open(state, event.path, "CURSOR")
        check_boundary(state.files[event.path], event.offset)
        state.cursors[event.path] = event.offset
    elif isinstance(event, Diagnostics):
        _require_open(state, event.path, "DIAGNOSTICS")
        for entry in event.entries:
            if entry.start > entry.end:
                raise ValueError(f"DIAGNOSTICS: start {entry.start} > end {entry.end}")
            check_boundary(state.files[event.path], entry.start)
            check_boundary(state.files[event.path], entry.end)
        state.diagnostics[event.path] = tuple(event.entries)
    elif isinstance(event, RetrievedContext):
        state.retrieved.append(event)
    elif isinstance(event, AcceptCompletion):
        state.completions[event.completion_id] = "accepted"
    elif isinstance(event, PartialAcceptCompletion):
        state.completions[event.completion_id] = "partial"
    elif isinstance(event, RejectCompletion):
        state.completions[event.completion_id] = "rejected"
    elif isinstance(event, Predict):
        _require_open(state, event.path, "PREDICT")
        check_boundary(state.files[event.path], event.offset)
        state.last_predict = event
    else:
        raise ValueError(f"unknown event type: {type(event).__name__}")
    return state


def replay(events: Sequence[BaseEvent]) -> EditorState:
    """Fold events into a fresh EditorState.

    Validates: seq starts at 0 and increases by exactly 1 per event; every
    edit/cursor offset lies inside its file on a UTF-8 boundary.
    """
    state = EditorState()
    for index, event in enumerate(events):
        if event.seq != index:
            raise ValueError(
                f"invalid sequence number: event seq={event.seq} at position {index} "
                f"(expected {index})"
            )
        apply_event(state, event)
    return state
