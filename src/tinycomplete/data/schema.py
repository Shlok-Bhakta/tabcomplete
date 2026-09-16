"""Shared training-example schema."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Provenance", "Example"]

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


class Provenance(StrEnum):
    STATIC_FIM = "static_fim"
    GIT_SYNTHETIC = "git_synthetic"
    AGENT_TRACE = "agent_trace"
    HUMAN_TRACE = "human_trace"
    TEACHER_MUSE = "teacher_muse"
    TEACHER_DEEPSEEK = "teacher_deepseek"


class Example(BaseModel):
    """One training example.

    FIM modes: ``input_text`` holds the formatted prompt (PSM/SPM with Qwen
    FIM sentinels); ``target`` is the middle span.
    next_edit mode: ``input_text`` holds repo context + recent edits +
    surrounding file with the editable region marked; ``target`` is the
    replacement region text ("" when ``action`` is "noop").
    """

    model_config = ConfigDict(frozen=True)

    id: str
    provenance: Provenance
    mode: Literal["psm", "spm", "next_edit"]
    language: str = "python"
    source_path: str = ""
    input_text: str
    target: str
    action: Literal["replace", "noop"] = "replace"
    hole_type: str = ""
    region_start: int = Field(default=0, ge=0)
    region_end: int = Field(default=0, ge=0)
    recent_edits: tuple[str, ...] = ()
    seed: int = 0
