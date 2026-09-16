"""Typed append-only editor event protocol.

Position convention: **byte offsets into the UTF-8 encoding** of the file text.
Every offset (insert position, delete/replace bounds, cursor) counts bytes, not
code points or graphemes. Offsets must land on UTF-8 code-point boundaries;
replay rejects offsets that split a multi-byte sequence. Rationale: byte
offsets are unambiguous across languages/runtimes and match how model token
spans map back to source bytes.

Events are immutable (frozen Pydantic models). An INSERT never mutates a
historical event; replay only folds new events onto derived state.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EventKind",
    "BaseEvent",
    "OpenFile",
    "CloseFile",
    "Snapshot",
    "Insert",
    "Delete",
    "Replace",
    "Cursor",
    "DiagnosticEntry",
    "Diagnostics",
    "RetrievedContext",
    "AcceptCompletion",
    "PartialAcceptCompletion",
    "RejectCompletion",
    "Predict",
    "Event",
    "EVENT_CLASSES",
    "event_from_dict",
    "event_to_dict",
    "byte_len",
    "check_boundary",
    "byte_splice",
]


class EventKind(StrEnum):
    OPEN_FILE = "OPEN_FILE"
    CLOSE_FILE = "CLOSE_FILE"
    SNAPSHOT = "SNAPSHOT"
    INSERT = "INSERT"
    DELETE = "DELETE"
    REPLACE = "REPLACE"
    CURSOR = "CURSOR"
    DIAGNOSTICS = "DIAGNOSTICS"
    RETRIEVED_CONTEXT = "RETRIEVED_CONTEXT"
    ACCEPT_COMPLETION = "ACCEPT_COMPLETION"
    PARTIAL_ACCEPT_COMPLETION = "PARTIAL_ACCEPT_COMPLETION"
    REJECT_COMPLETION = "REJECT_COMPLETION"
    PREDICT = "PREDICT"


def _now() -> float:
    return time.time()


class BaseEvent(BaseModel):
    """Common envelope: monotonic seq + wall clock. Order is defined by seq."""

    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=0)
    clock: float = Field(default_factory=_now)
    kind: EventKind


class OpenFile(BaseEvent):
    kind: Literal[EventKind.OPEN_FILE] = EventKind.OPEN_FILE
    path: str


class CloseFile(BaseEvent):
    kind: Literal[EventKind.CLOSE_FILE] = EventKind.CLOSE_FILE
    path: str


class Snapshot(BaseEvent):
    """Establishes a new source-of-truth state for a file."""

    kind: Literal[EventKind.SNAPSHOT] = EventKind.SNAPSHOT
    path: str
    content: str


class Insert(BaseEvent):
    kind: Literal[EventKind.INSERT] = EventKind.INSERT
    path: str
    offset: int = Field(ge=0)
    text: str


class Delete(BaseEvent):
    kind: Literal[EventKind.DELETE] = EventKind.DELETE
    path: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class Replace(BaseEvent):
    kind: Literal[EventKind.REPLACE] = EventKind.REPLACE
    path: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str


class Cursor(BaseEvent):
    kind: Literal[EventKind.CURSOR] = EventKind.CURSOR
    path: str
    offset: int = Field(ge=0)


class DiagnosticEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    message: str
    severity: str = "error"


class Diagnostics(BaseEvent):
    kind: Literal[EventKind.DIAGNOSTICS] = EventKind.DIAGNOSTICS
    path: str
    entries: tuple[DiagnosticEntry, ...] = ()


class RetrievedContext(BaseEvent):
    kind: Literal[EventKind.RETRIEVED_CONTEXT] = EventKind.RETRIEVED_CONTEXT
    path: str = ""
    content: str
    source: str = "manual"


class AcceptCompletion(BaseEvent):
    kind: Literal[EventKind.ACCEPT_COMPLETION] = EventKind.ACCEPT_COMPLETION
    completion_id: str
    text: str = ""


class PartialAcceptCompletion(BaseEvent):
    kind: Literal[EventKind.PARTIAL_ACCEPT_COMPLETION] = EventKind.PARTIAL_ACCEPT_COMPLETION
    completion_id: str
    text: str = ""


class RejectCompletion(BaseEvent):
    kind: Literal[EventKind.REJECT_COMPLETION] = EventKind.REJECT_COMPLETION
    completion_id: str
    reason: str = ""


class Predict(BaseEvent):
    kind: Literal[EventKind.PREDICT] = EventKind.PREDICT
    path: str
    offset: int = Field(ge=0)


Event = Annotated[
    OpenFile
    | CloseFile
    | Snapshot
    | Insert
    | Delete
    | Replace
    | Cursor
    | Diagnostics
    | RetrievedContext
    | AcceptCompletion
    | PartialAcceptCompletion
    | RejectCompletion
    | Predict,
    Field(discriminator="kind"),
]

EVENT_CLASSES: dict[str, type[BaseEvent]] = {
    EventKind.OPEN_FILE.value: OpenFile,
    EventKind.CLOSE_FILE.value: CloseFile,
    EventKind.SNAPSHOT.value: Snapshot,
    EventKind.INSERT.value: Insert,
    EventKind.DELETE.value: Delete,
    EventKind.REPLACE.value: Replace,
    EventKind.CURSOR.value: Cursor,
    EventKind.DIAGNOSTICS.value: Diagnostics,
    EventKind.RETRIEVED_CONTEXT.value: RetrievedContext,
    EventKind.ACCEPT_COMPLETION.value: AcceptCompletion,
    EventKind.PARTIAL_ACCEPT_COMPLETION.value: PartialAcceptCompletion,
    EventKind.REJECT_COMPLETION.value: RejectCompletion,
    EventKind.PREDICT.value: Predict,
}


def event_from_dict(data: dict[str, Any]) -> BaseEvent:
    """Parse a plain dict into the typed event for its ``kind``."""
    kind = data.get("kind")
    try:
        cls = EVENT_CLASSES[kind]
    except KeyError:
        raise ValueError(f"unknown event kind: {kind!r}") from None
    return cls.model_validate(data)


def event_to_dict(event: BaseEvent) -> dict[str, Any]:
    return event.model_dump(mode="json")


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def check_boundary(text: str, offset: int) -> None:
    """Reject offsets outside the file or inside a multi-byte sequence."""
    raw = text.encode("utf-8")
    if offset < 0 or offset > len(raw):
        raise ValueError(f"offset {offset} outside file of {len(raw)} bytes")
    try:
        raw[:offset].decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"offset {offset} splits a UTF-8 sequence") from None


def byte_splice(text: str, start: int, end: int, replacement: str) -> str:
    """Replace byte range [start, end) with replacement text."""
    if start > end:
        raise ValueError(f"start {start} > end {end}")
    check_boundary(text, start)
    check_boundary(text, end)
    raw = text.encode("utf-8")
    return (raw[:start] + replacement.encode("utf-8") + raw[end:]).decode("utf-8")
