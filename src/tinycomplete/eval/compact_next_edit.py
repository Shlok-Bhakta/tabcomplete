"""Compact v1 wire format for canonical v2 next-edit actions."""

from __future__ import annotations

from dataclasses import dataclass

from tinycomplete.eval.next_edit_protocol import MAX_NEW_TOKENS, NextEditAction

WIRE_VERSION = "compact-next-edit-v1"


@dataclass(frozen=True)
class CompactResult:
    status: str
    action: NextEditAction | None
    terminated: bool


def encode(action: NextEditAction) -> str:
    if action.action == "no_edit":
        return "N\n"
    assert action.text is not None
    return "R\n" + action.text


def decode(
    text: str,
    *,
    finish_reason: str | None,
    generated_tokens: int | None = None,
    max_tokens: int = MAX_NEW_TOKENS,
) -> CompactResult:
    """Require an observed EOS and preserve every replacement byte.

    `text` excludes EOS because model tokenizers do not decode it as ordinary
    response content. A token cap is never accepted as EOS.
    """
    if max_tokens < 1 or max_tokens > MAX_NEW_TOKENS:
        raise ValueError("invalid token ceiling")
    if finish_reason != "eos" or (generated_tokens is not None and generated_tokens > max_tokens):
        return CompactResult("incomplete", None, False)
    if text == "N\n":
        return CompactResult("ok", NextEditAction(action="no_edit"), True)
    if text.startswith("R\n"):
        return CompactResult("ok", NextEditAction(action="replace", text=text[2:]), True)
    return CompactResult("malformed", None, True)
