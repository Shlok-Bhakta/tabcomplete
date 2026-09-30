#!/usr/bin/env python3
"""Postscore frozen pilot predictions with the existing isolated CPU evaluator.

Public line-completion exactness and synthetic functional success are separate
denominators. This never makes a provider call or repairs a generated action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinycomplete.eval.code_benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    CheckSpec,
    Prediction,
    evaluate_prediction,
)
from tinycomplete.one_line.contract import EditState, apply_action, decode_action


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def state_digest(state: dict[str, Any]) -> str:
    # Matches the GPU evaluator's state encoding, including default separators.
    return digest(json.dumps(state, sort_keys=True, ensure_ascii=False).encode())


def _index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("missing or duplicate case identity")
        result[identifier] = row
    return result


def score_predictions(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    *,
    work_root: Path,
    evaluate: Callable[..., BenchmarkResult] = evaluate_prediction,
) -> dict[str, Any]:
    data = _index(rows, "id")
    observed = _index(predictions, "id")
    oracles = _index(fixtures, "id")
    if set(data) != set(observed) or not set(oracles) <= set(data):
        raise ValueError("prediction or fixture case coverage mismatch")
    outcomes = []
    counts: Counter[str] = Counter()
    per_action: dict[str, Counter[str]] = {}
    for position, (identifier, row) in enumerate(data.items()):
        prediction = observed[identifier]
        state = EditState.from_mapping(row["state"])
        if prediction.get("state_sha256") != state_digest(row["state"]) or prediction.get(
            "source_sha256"
        ) != digest(state.source.encode()):
            raise ValueError("prediction pre-state identity mismatch")
        wire = prediction.get("wire")
        terminated = prediction.get("terminated_by_eos")
        tokens = prediction.get("generated_tokens")
        if (
            not isinstance(wire, str)
            or type(terminated) is not bool
            or type(tokens) is not int
            or tokens < 0
        ):
            raise ValueError("prediction termination evidence is malformed")
        decoded = decode_action(wire, terminated=terminated, generated_tokens=tokens)
        action = decoded.action
        try:
            after = apply_action(state, action) if action is not None else None
        except ValueError:
            action, after = None, None
        canonical = asdict(action) if action is not None else None
        if prediction.get("canonical_action") != canonical:
            raise ValueError("prediction wire and canonical action disagree")
        gold_kind = row["action"]["kind"]
        if (gold_kind != "replace_line") != (identifier in oracles):
            raise ValueError("functional fixture coverage does not match the action track")
        bucket = per_action.setdefault(gold_kind, Counter())
        bucket["cases"] += 1
        exact = after is not None and after == row["after_source"]
        no_edit = gold_kind == "keep"
        changed = after is not None and after != state.source
        functional: bool | None = None
        check_statuses = None
        if identifier in oracles:
            fixture = oracles[identifier]
            test_source = fixture.get("test_source")
            image = fixture.get("image_identity")
            if (
                not isinstance(test_source, str)
                or fixture.get("fixture_sha256") != digest(test_source.encode())
                or fixture.get("gold_after_source_sha256") != digest(row["after_source"].encode())
                or not isinstance(image, str)
                or "@sha256:" not in image
            ):
                raise ValueError("functional fixture identity mismatch")
            functional = False
            if after is not None:
                case = BenchmarkCase(
                    id=identifier,
                    language="python",
                    path="solution.py",
                    prefix="",
                    # BenchmarkCase requires a nonempty reference. Exactness is
                    # computed above against the original bytes, including empty.
                    expected=row["after_source"] or "\n",
                    category=gold_kind,
                    check=CheckSpec(
                        compile=["python", "-m", "py_compile", "solution.py"],
                        test=["python", "tests.py"],
                        files={"tests.py": test_source},
                        container_image=image,
                    ),
                )
                result = evaluate(
                    case,
                    Prediction(case_id=identifier, completion=after),
                    work_root=work_root / f"case-{position:04d}",
                    execution_backend="container",
                )
                check_statuses = {
                    "parse": result.parse.status,
                    "compile": result.compile.status,
                    "test": result.test.status,
                }
                functional = all(value == "pass" for value in check_statuses.values())
            # A behavior-preserving gratuitous edit does not count as no-edit recall.
            if no_edit:
                functional = functional and action is not None and action.kind == "keep"
            counts["functional_cases"] += 1
            counts["functional_success"] += int(functional)
            bucket["functional_cases"] += 1
            bucket["functional_success"] += int(functional)
        else:
            counts["public_prefix_cases"] += 1
            counts["public_prefix_exact"] += int(exact)
        counts["cases"] += 1
        counts["valid"] += int(action is not None)
        counts["terminated"] += int(terminated)
        counts["exact"] += int(exact)
        counts["nonempty_edit"] += int(changed)
        bucket["exact"] += int(exact)
        if no_edit:
            counts["no_edit_cases"] += 1
            counts["no_edit_recalled"] += int(action is not None and action.kind == "keep")
            counts["no_edit_false_edits"] += int(changed)
        else:
            counts["edit_required_cases"] += 1
            counts["edit_required_exact"] += int(exact)
        outcomes.append(
            {
                "id": identifier,
                "gold_action": gold_kind,
                "canonical_action": canonical,
                "valid": action is not None,
                "terminated": terminated,
                "exact_after": exact,
                "source_changed": changed,
                "functional_success": functional,
                "checks": check_statuses,
            }
        )
    return {
        "schema": "functional-pilot-cpu-postscore-v1",
        "counts": dict(counts),
        "per_action": {key: dict(value) for key, value in per_action.items()},
        "observations": outcomes,
        "human_feedback": False,
        "personalization_enabled": False,
    }


def _read_pinned(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if digest(payload) != expected:
        raise ValueError("input artifact hash mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "predictions", "fixtures"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("postscore output already exists")
    artifacts = {
        name: _read_pinned(getattr(args, name), getattr(args, name + "_sha256"))
        for name in ("development", "predictions", "fixtures")
    }
    rows = [json.loads(line) for line in artifacts["development"].splitlines()]
    fixtures = [json.loads(line) for line in artifacts["fixtures"].splitlines()]
    predictions = json.loads(artifacts["predictions"])["observations"]
    args.output.mkdir(mode=0o700, parents=True)
    result = score_predictions(rows, predictions, fixtures, work_root=args.output / "checks")
    result["input_sha256"] = {key: digest(value) for key, value in artifacts.items()}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
