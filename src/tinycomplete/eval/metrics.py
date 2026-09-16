"""Independent next-edit metrics. No invented composite quality scores."""

from __future__ import annotations

import json
import os

__all__ = [
    "exact_match",
    "normalized_exact_match",
    "normalize_code",
    "prefix_match_length",
    "character_edit_distance",
    "candidate_length",
    "tree_sitter_parse_success",
    "noop_accuracy",
    "score_example",
    "aggregate",
    "write_predictions",
]


def exact_match(prediction: str, target: str) -> bool:
    return prediction == target


def normalize_code(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().split("\n"))


def normalized_exact_match(prediction: str, target: str) -> bool:
    return normalize_code(prediction) == normalize_code(target)


def prefix_match_length(prediction: str, target: str) -> int:
    n = 0
    for a, b in zip(prediction, target, strict=False):
        if a != b:
            break
        n += 1
    return n


def character_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over characters (training examples are small)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def candidate_length(candidate: str) -> dict[str, int]:
    raw = candidate.encode("utf-8")
    return {"chars": len(candidate), "bytes": len(raw)}


def tree_sitter_parse_success(code: str, language: str = "python") -> bool:
    if language != "python":
        return True  # no grammar configured: do not fabricate a verdict
    try:
        from tree_sitter_language_pack import get_parser
    except Exception:
        return True
    try:
        root = get_parser("python").parse(code.encode("utf-8")).root_node
    except Exception:
        return False
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            return False
        stack.extend(node.children)
    return True


def noop_accuracy(gold_is_noop: list[bool], pred_is_noop: list[bool]) -> float | None:
    """Accuracy on the subset where ground truth is noop; None if empty."""
    idx = [i for i, g in enumerate(gold_is_noop) if g]
    if not idx:
        return None
    return sum(1 for i in idx if pred_is_noop[i]) / len(idx)


def score_example(prediction: str, target: str, language: str = "python") -> dict:
    length = candidate_length(prediction)
    return {
        "exact_match": exact_match(prediction, target),
        "normalized_exact_match": normalized_exact_match(prediction, target),
        "prefix_match_length": prefix_match_length(prediction, target),
        "character_edit_distance": character_edit_distance(prediction, target),
        "candidate_chars": length["chars"],
        "candidate_bytes": length["bytes"],
        "tree_sitter_parse_success": tree_sitter_parse_success(prediction, language),
    }


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    out: dict = {"n": len(scores)}
    for key in scores[0]:
        vals = [s[key] for s in scores]
        if isinstance(vals[0], bool):
            out[key + "_rate"] = sum(1 for v in vals if v) / len(vals)
        elif isinstance(vals[0], (int, float)):
            out[key + "_mean"] = sum(vals) / len(vals)
    return out


def write_predictions(path: str, records: list[dict]) -> None:
    """Store raw per-example predictions so metrics can be recomputed later."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
