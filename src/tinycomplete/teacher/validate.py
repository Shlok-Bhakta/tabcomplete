"""Automatic validation for teacher candidates (stores rejects too)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Candidate

__all__ = ["MAX_REWRITE_BYTES", "ValidationResult", "validate_candidate", "apply_replacement"]

MAX_REWRITE_BYTES = 8192


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def apply_replacement(full_text: str, start: int, end: int, replacement: str) -> str:
    raw = full_text.encode("utf-8")
    return (raw[:start] + replacement.encode("utf-8") + raw[end:]).decode("utf-8")


def _error_nodes(source: str, language: str) -> int:
    if language != "python":
        return 0
    try:
        from tree_sitter_language_pack import get_parser
    except Exception:
        return 0
    try:
        root = get_parser("python").parse(source.encode("utf-8")).root_node
    except Exception:
        return 0
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            count += 1
        stack.extend(node.children)
    return count


def validate_candidate(
    candidate: Candidate,
    *,
    region_text: str,
    full_text: str,
    region_start: int,
    region_end: int,
    language: str = "python",
    prev_region_text: str | None = None,
) -> ValidationResult:
    reasons: list[str] = []
    if candidate.action == "noop":
        if candidate.replacement != "":
            reasons.append("noop must carry an empty replacement")
        return ValidationResult(ok=not reasons, reasons=reasons)
    if candidate.action != "replace":
        return ValidationResult(ok=False, reasons=[f"unknown action: {candidate.action}"])
    rep = candidate.replacement
    try:
        rep.encode("utf-8")
    except Exception:
        return ValidationResult(ok=False, reasons=["replacement is not valid text"])
    if rep == region_text:
        reasons.append("replacement identical to region (use noop instead)")
    if not rep:
        reasons.append("empty replacement with action=replace")
    if len(rep.encode("utf-8")) > MAX_REWRITE_BYTES:
        reasons.append(f"replacement exceeds {MAX_REWRITE_BYTES} bytes")
    floor = max(512, 4 * len(region_text.encode("utf-8")))
    if len(rep.encode("utf-8")) > floor:
        reasons.append("replacement exceeds 4x region size")
    if prev_region_text is not None and rep == prev_region_text:
        reasons.append("exact reversal of immediately preceding edit")
    if region_text and rep.count(region_text) > 1:
        reasons.append("region duplicated inside replacement")
    if not reasons:
        try:
            applied = apply_replacement(full_text, region_start, region_end, rep)
        except Exception as exc:
            reasons.append(f"replacement does not apply cleanly: {exc}")
            return ValidationResult(ok=False, reasons=reasons)
        if _error_nodes(applied, language) > _error_nodes(full_text, language):
            reasons.append("replacement introduces syntax errors")
    return ValidationResult(ok=not reasons, reasons=reasons)
