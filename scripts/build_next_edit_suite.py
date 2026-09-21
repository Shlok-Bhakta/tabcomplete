"""Merge reviewed buggy-region suggestions into a deterministic next-edit suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from tinycomplete.eval.code_benchmark import load_suite
from tinycomplete.eval.next_edit_benchmark import NextEditCase


class RegionSuggestion(BaseModel):
    id: str
    current_region: str = Field(min_length=1, max_length=16_384)
    recent_edits: tuple[str, ...] = Field(min_length=1, max_length=2)


_EXPECTED_STDOUT = {
    "c/stable_unique_id_2": "3\n",
    "c/stable_unique_id_3": "5\n",
    "c/stable_unique_id_4": "123\n",
    "c/stable_unique_id_6": "30\n",
    "c/stable_unique_id_7": "two\n",
    "c/stable_unique_id_8": "5\n",
}

_EXPECTED_REPLACEMENTS = {
    "c/stable_13": (
        "int64_t g = gcd(a, b);\n"
        "    int64_t value = (a / g) * b;\n"
        "    return value < 0 ? -value : value;"
    )
}


def _executable_check(base):
    """Turn C/C++ link-only test commands into link-then-run checks."""
    check = base.check
    if base.id in _EXPECTED_STDOUT:
        check = check.model_copy(update={"expected_stdout": _EXPECTED_STDOUT[base.id]})
    if base.language not in {"c", "cpp"} or not check.test or check.run:
        return check
    command = list(check.test)
    if "-o" not in command:
        return check
    output_index = command.index("-o") + 1
    command[output_index] = ".tabcomplete-test-bin"
    return check.model_copy(
        update={"test": command, "run": ["./.tabcomplete-test-bin"]}
    )


def load_suggestions(root: Path) -> dict[str, RegionSuggestion]:
    suggestions: dict[str, RegionSuggestion] = {}
    paths = sorted(root.glob("chunk_*/suggestions.jsonl"))
    if not paths:
        raise ValueError(f"no suggestions.jsonl files found below {root}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                suggestion = RegionSuggestion.model_validate_json(line)
                if suggestion.id in suggestions:
                    raise ValueError(
                        f"duplicate suggestion {suggestion.id!r} in {path}:{line_number}"
                    )
                suggestions[suggestion.id] = suggestion
    return suggestions


def _assemble_cases(
    base_cases: list, suggestions: dict[str, RegionSuggestion], noop_every: int
) -> list[NextEditCase]:
    if noop_every < 2:
        raise ValueError("noop_every must be at least 2")
    expected_ids = {case.id for case in base_cases}
    if set(suggestions) != expected_ids:
        missing = sorted(expected_ids - set(suggestions))
        extra = sorted(set(suggestions) - expected_ids)
        raise ValueError(f"suggestion id mismatch: missing={missing[:3]} extra={extra[:3]}")

    cases = []
    for index, base in enumerate(base_cases):
        suggestion = suggestions[base.id]
        is_noop = index % noop_every == 0
        expected = _EXPECTED_REPLACEMENTS.get(base.id, base.expected)
        current_region = expected if is_noop else suggestion.current_region
        if not is_noop and current_region == expected:
            raise ValueError(f"replacement suggestion is identical to gold: {base.id}")
        if "[[EDIT]]" in current_region or "[[/EDIT]]" in current_region:
            raise ValueError(f"suggestion contains reserved edit markers: {base.id}")
        prefix_bytes = base.prefix.encode("utf-8")
        current = base.prefix + current_region + base.suffix
        cases.append(
            NextEditCase(
                id=base.id,
                language=base.language,
                path=base.path,
                current=current,
                region_start=len(prefix_bytes),
                region_end=len(prefix_bytes) + len(current_region.encode("utf-8")),
                expected="" if is_noop else expected,
                action="noop" if is_noop else "replace",
                recent_edits=suggestion.recent_edits,
                context_files=base.context_files,
                check=_executable_check(base),
                category=base.category,
                repository_context=base.repository_context,
            )
        )
    return cases


def build_cases(base_path: Path, suggestion_root: Path, noop_every: int) -> list[NextEditCase]:
    return _assemble_cases(
        load_suite(base_path), load_suggestions(suggestion_root), noop_every
    )


def apply_repairs(
    suggestions: dict[str, RegionSuggestion], repair_path: Path | None
) -> dict[str, RegionSuggestion]:
    if repair_path is None:
        return suggestions
    updated = dict(suggestions)
    with repair_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            repair = RegionSuggestion.model_validate_json(line)
            if repair.id not in updated:
                raise ValueError(f"repair id is not in base suggestions: {repair.id}")
            updated[repair.id] = repair
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path, default=Path("data/benchmarks/code_completion_v1.jsonl")
    )
    parser.add_argument("--suggestions", type=Path, required=True)
    parser.add_argument("--repairs", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("data/benchmarks/next_edit_v1.jsonl")
    )
    parser.add_argument("--noop-every", type=int, default=5)
    args = parser.parse_args()

    suggestions = apply_repairs(load_suggestions(args.suggestions), args.repairs)
    cases = _assemble_cases(load_suite(args.base), suggestions, args.noop_every)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(mode="json"), sort_keys=True) + "\n")
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    actions = {
        action: sum(case.action == action for case in cases) for action in ("replace", "noop")
    }
    print(f"wrote {len(cases)} cases sha256={digest} actions={actions}")


if __name__ == "__main__":
    main()
