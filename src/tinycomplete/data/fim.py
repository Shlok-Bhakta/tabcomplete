"""Deterministic FIM hole generation (PSM + SPM).

Holes are carved from fixture source with a seeded RNG: same seed + same
input yields byte-identical output. Spans are byte offsets (consistent with
the editor protocol) and always land on UTF-8 boundaries. Syntactically
meaningful spans come from Tree-sitter when a grammar is available; otherwise
deterministic line/regex fallbacks are used.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .schema import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, Example, Provenance

__all__ = [
    "HOLE_TYPES",
    "Hole",
    "format_psm",
    "format_spm",
    "make_hole",
    "make_example",
    "generate_examples",
]

HOLE_TYPES = (
    "identifier",
    "expression",
    "partial_statement",
    "statement",
    "block",
    "function_body",
    "import",
    "span",
)

_STATEMENT_TYPES = {
    "expression_statement",
    "return_statement",
    "assignment",
    "augmented_assignment",
    "assert_statement",
    "break_statement",
    "continue_statement",
    "pass_statement",
    "raise_statement",
    "yield",
    "await",
}

_EXPRESSION_TYPES = {
    "binary_operator",
    "unary_operator",
    "comparison_operator",
    "boolean_operator",
    "call",
    "attribute",
    "subscript",
    "parenthesized_expression",
    "conditional_expression",
    "named_expression",
    "lambda",
    "string",
    "integer",
    "float",
    "true",
    "false",
    "none",
}

_IMPORT_TYPES = {"import_statement", "import_from_statement"}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Hole:
    hole_type: str
    start: int  # byte offset
    end: int  # byte offset
    prefix: str
    middle: str
    suffix: str


def format_psm(prefix: str, suffix: str) -> str:
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def format_spm(prefix: str, suffix: str) -> str:
    # Suffix first: keeps the live prefix immediately before generation.
    return f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"


def _get_parser(language: str):
    if language != "python":
        return None
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser("python")
    except Exception:
        return None


def _iter_nodes(node):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _collect_spans(source: bytes, language: str) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {t: [] for t in HOLE_TYPES}
    spans["span"] = []
    spans["partial_statement"] = []
    parser = _get_parser(language)
    if parser is not None:
        try:
            tree = parser.parse(source)
        except Exception:
            tree = None
        if tree is not None and not tree.root_node.has_error:
            for node in _iter_nodes(tree.root_node):
                s, e = node.start_byte, node.end_byte
                if s >= e:
                    continue
                if node.type == "identifier" and e - s <= 64:
                    spans["identifier"].append((s, e))
                elif node.type in _EXPRESSION_TYPES:
                    spans["expression"].append((s, e))
                elif node.type in _STATEMENT_TYPES and b"\n" not in source[s:e]:
                    spans["statement"].append((s, e))
                elif node.type == "block":
                    parent = node.parent
                    if parent is not None and parent.type == "function_definition":
                        spans["function_body"].append((s, e))
                    else:
                        spans["block"].append((s, e))
                elif node.type in _IMPORT_TYPES:
                    spans["import"].append((s, e))
            for node in _iter_nodes(tree.root_node):
                stmt = node.type in _STATEMENT_TYPES
                single = b"\n" not in source[node.start_byte : node.end_byte]
                if stmt and single:
                    spans["partial_statement"].append((node.start_byte, node.end_byte))
    if not spans["identifier"]:
        # regex fallback on characters -> byte offsets
        text = source.decode("utf-8", errors="replace")
        for m in _IDENT_RE.finditer(text):
            s = len(text[: m.start()].encode("utf-8"))
            e = s + len(m.group(0).encode("utf-8"))
            spans["identifier"].append((s, e))
    if not spans["statement"] and not spans["partial_statement"]:
        text = source.decode("utf-8", errors="replace")
        off = 0
        for line in text.split("\n"):
            raw = line.encode("utf-8")
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                spans["statement"].append((off, off + len(raw)))
                spans["partial_statement"].append((off, off + len(raw)))
            off += len(raw) + 1
    for key in ("identifier", "expression", "statement", "block", "function_body", "import"):
        spans[key] = sorted(set(spans[key]))
    return spans


def _decode_slice(source: bytes, start: int, end: int) -> str:
    return source[start:end].decode("utf-8")


def make_hole(source: str, hole_type: str, seed: int, language: str = "python") -> Hole:
    """Carve one deterministic hole. Falls back to ``span`` when empty."""
    if hole_type not in HOLE_TYPES:
        raise ValueError(f"unknown hole type: {hole_type!r}")
    raw = source.encode("utf-8")
    if len(raw) == 0:
        raise ValueError("cannot carve a hole from empty source")
    rng = random.Random(f"{seed}:{hole_type}:{len(raw)}")
    spans = _collect_spans(raw, language)
    actual = hole_type
    candidates = spans.get(hole_type, [])
    if hole_type == "partial_statement" and candidates:
        s, e = candidates[rng.randrange(len(candidates))]
        width = e - s
        cut = s + max(1, int(width * rng.choice((0.3, 0.5, 0.7))))
        # move cut back to a UTF-8 boundary
        while cut > s and (raw[cut] & 0xC0) == 0x80:
            cut -= 1
        s, e = s, cut
    elif hole_type == "span" or not candidates:
        actual = "span"
        start = rng.randrange(len(raw))
        while start > 0 and (raw[start] & 0xC0) == 0x80:
            start -= 1
        max_len = min(32, len(raw) - start)
        length = rng.randint(1, max(1, max_len))
        end = start + length
        while end < len(raw) and (raw[end] & 0xC0) == 0x80:
            end += 1
        s, e = start, end
    else:
        s, e = candidates[rng.randrange(len(candidates))]
    prefix = _decode_slice(raw, 0, s)
    middle = _decode_slice(raw, s, e)
    suffix = _decode_slice(raw, e, len(raw))
    return Hole(hole_type=actual, start=s, end=e, prefix=prefix, middle=middle, suffix=suffix)


def make_example(
    source: str,
    hole_type: str,
    seed: int,
    mode: str = "psm",
    language: str = "python",
    source_path: str = "",
    example_id: str = "",
) -> Example:
    if mode not in ("psm", "spm"):
        raise ValueError(f"unknown FIM mode: {mode!r}")
    hole = make_hole(source, hole_type, seed, language)
    if mode == "psm":
        input_text = format_psm(hole.prefix, hole.suffix)
    else:
        input_text = format_spm(hole.prefix, hole.suffix)
    return Example(
        id=example_id or f"{hole.hole_type}-{mode}-{seed}",
        provenance=Provenance.STATIC_FIM,
        mode=mode,  # type: ignore[arg-type]
        language=language,
        source_path=source_path,
        input_text=input_text,
        target=hole.middle,
        action="replace",
        hole_type=hole.hole_type,
        region_start=hole.start,
        region_end=hole.end,
        seed=seed,
    )


def generate_examples(
    source: str,
    count: int,
    seed: int,
    language: str = "python",
    source_path: str = "",
) -> list[Example]:
    """Generate ``count`` deterministic examples cycling hole types and modes."""
    rng = random.Random(f"{seed}:schedule:{len(source)}")
    out = []
    for k in range(count):
        hole_type = HOLE_TYPES[rng.randrange(len(HOLE_TYPES))]
        mode = "psm" if rng.randrange(2) == 0 else "spm"
        out.append(
            make_example(
                source,
                hole_type,
                seed * 100003 + k,
                mode=mode,
                language=language,
                source_path=source_path,
                example_id=f"fim-{k}-{hole_type}-{mode}-{seed}",
            )
        )
    return out
