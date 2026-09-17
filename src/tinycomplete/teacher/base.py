"""Teacher abstraction: narrow next-edit labeling interface.

Money is never spent through this module alone: every paid adapter consults
the budget gate (teacher/budget.py) and requires ALLOW_PAID_SYNTHETIC=1.
Secrets arrive only via process environment and are never stored in records.
"""

from __future__ import annotations

import hashlib
import time
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EditableRegion",
    "TeacherRequest",
    "Candidate",
    "ValidationSummary",
    "TeacherResponse",
    "TeacherProvider",
    "provider_from_name",
]

MAX_CANDIDATES_HARD_CAP = 5


class EditableRegion(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str = ""
    start: int = Field(ge=0, default=0)  # byte offset in current file
    end: int = Field(ge=0, default=0)
    text: str = ""


class TeacherRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    state_id: str
    serialized_state: str
    region: EditableRegion
    repo_context: str = ""
    recent_edits: tuple[str, ...] = ()
    num_candidates: int = Field(default=3, ge=1, le=MAX_CANDIDATES_HARD_CAP)
    language: str = "python"
    # Raw current file text the region offsets index into. Validation MUST use
    # this (never the serialized wrapper) as the syntax/apply base.
    file_text: str = ""


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["replace", "noop"]
    replacement: str = ""


class ValidationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    reasons: tuple[str, ...] = ()


class TeacherResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    timestamp: float = Field(default_factory=time.time)
    state_id: str
    candidates: tuple[Candidate, ...]
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    reported_cost_usd: float | None = None
    request_id: str = ""
    raw_sha256: str = ""
    validation: ValidationSummary = ValidationSummary(ok=False, reasons=("unchecked",))


def raw_sha256_of(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TeacherProvider(Protocol):
    provider_name: str

    async def predict(self, request: TeacherRequest) -> TeacherResponse:
        ...


def provider_from_name(name: str, **kwargs):
    """Construct a provider by name. Imported lazily to keep deps light."""
    if name == "fake":
        from .fake import FakeProvider

        return FakeProvider(**kwargs)
    if name == "openrouter":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider(**kwargs)
    if name == "deepseek":
        from .deepseek import DeepSeekProvider

        return DeepSeekProvider(**kwargs)
    raise ValueError(f"unknown teacher provider: {name!r}")
