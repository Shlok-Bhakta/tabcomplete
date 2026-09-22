"""Evaluation helpers with optional Torch-backed latency imports."""

from typing import TYPE_CHECKING, Any

from .metrics import *  # noqa: F401,F403

if TYPE_CHECKING:
    from .latency import GenerationStats as GenerationStats


def __getattr__(name: str) -> Any:
    if name in {"GenerationStats", "benchmark"}:
        from . import latency

        return getattr(latency, name)
    raise AttributeError(name)
