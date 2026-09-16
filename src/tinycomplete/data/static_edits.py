"""Synthetic next-edit region-rewrite examples (deterministic, seeded).

Construction: pick an editable region plus 1-3 disjoint "recent edit" spans
from the same source. Recent edits are applied to form the current file
context; the region itself is left untouched and its future rewrite is the
target. With a seeded probability the region needs no change (NO_EDIT is a
first-class target with action "noop").

Input format:

    <repo path>
    - replace [s,e): before -> after     (recent edits)
    <file path>
    ... current file with region marked [[EDIT]] ... [[/EDIT]] ...
    </file>
    <P path offset>                      (prediction point at region start)
"""

from __future__ import annotations

import random
import re

from .fim import HOLE_TYPES, make_hole
from .schema import Example, Provenance

__all__ = ["make_next_edit", "generate_next_edits", "NOOP_PROBABILITY"]

NOOP_PROBABILITY = 0.2

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS = {"def", "return", "import", "from", "for", "in", "if", "else", "while"}


def _mutate_region(text: str, rng: random.Random) -> str:
    """Deterministic synthetic rewrite: rename one identifier in the region."""
    names = [m.group(0) for m in _IDENT_RE.finditer(text) if m.group(0) not in _KEYWORDS]
    if not names:
        return text + "  # touched"
    victim = names[rng.randrange(len(names))]
    return re.sub(rf"\b{victim}\b", victim + "_v2", text)


def _apply_non_overlapping(
    source: bytes, replacements: list[tuple[int, int, bytes]]
) -> bytes:
    """Apply disjoint replacements to source bytes."""
    ordered = sorted(replacements)
    out = bytearray()
    cursor = 0
    for s, e, new in ordered:
        if s < cursor:
            raise ValueError("overlapping replacements")
        out += source[cursor:s]
        out += new
        cursor = e
    out += source[cursor:]
    return bytes(out)


def make_next_edit(
    source: str,
    seed: int,
    language: str = "python",
    source_path: str = "",
    example_id: str = "",
) -> Example:
    raw = source.encode("utf-8")
    if len(raw) == 0:
        raise ValueError("empty source")
    rng = random.Random(f"{seed}:next-edit:{len(raw)}")

    region_hole = make_hole(source, HOLE_TYPES[rng.randrange(len(HOLE_TYPES))], seed, language)
    rs, re_ = region_hole.start, region_hole.end

    # Recent-edit spans: short spans disjoint from the region.
    recent: list[tuple[int, int, bytes]] = []
    attempts = 0
    while len(recent) < rng.randint(1, 3) and attempts < 50:
        attempts += 1
        cand = make_hole(source, "span", seed * 7919 + attempts, language)
        if cand.end <= rs or cand.start >= re_:
            if all(cand.end <= s or cand.start >= e for s, e, _ in recent):
                before = cand.middle.encode("utf-8")
                after = _mutate_region(cand.middle, rng).encode("utf-8")
                if after != before:
                    recent.append((cand.start, cand.end, after))

    current = _apply_non_overlapping(raw, recent)
    shift = sum(len(new) - (e - s) for s, e, new in sorted(recent) if s < rs)
    region_raw = raw[rs:re_]
    new_rs = rs + shift
    new_re = new_rs + len(region_raw)
    assert current[new_rs:new_re] == region_raw, "region misaligned after edits"
    region_text = region_raw.decode("utf-8")

    is_noop = rng.random() < NOOP_PROBABILITY
    if is_noop:
        target, action = "", "noop"
    else:
        mutated = _mutate_region(region_text, rng)
        if mutated == region_text:
            target, action = "", "noop"
        else:
            target, action = mutated, "replace"

    label = source_path or "file"
    recent_lines = [
        f"- replace [{s},{e}): {raw[s:e].decode('utf-8')!r} -> {new.decode('utf-8')!r}"
        for s, e, new in sorted(recent)
    ]
    cur_text = current.decode("utf-8")
    marked = (
        cur_text.encode("utf-8")[:new_rs].decode("utf-8")
        + "[[EDIT]]"
        + region_text
        + "[[/EDIT]]"
        + cur_text.encode("utf-8")[new_re:].decode("utf-8")
    )
    input_text = (
        f"<repo {label}>\n"
        + "\n".join(recent_lines)
        + f"\n<file {label}>\n{marked}\n</file>\n<P {label} {new_rs}>\n"
    )
    return Example(
        id=example_id or f"next-edit-{seed}",
        provenance=Provenance.STATIC_FIM,
        mode="next_edit",
        language=language,
        source_path=source_path,
        input_text=input_text,
        target=target,
        action=action,  # type: ignore[arg-type]
        hole_type=region_hole.hole_type,
        region_start=new_rs,
        region_end=new_re,
        recent_edits=tuple(recent_lines),
        seed=seed,
    )


def generate_next_edits(
    source: str, count: int, seed: int, language: str = "python", source_path: str = ""
) -> list[Example]:
    return [
        make_next_edit(source, seed * 100003 + k, language, source_path, f"next-edit-{k}-{seed}")
        for k in range(count)
    ]
