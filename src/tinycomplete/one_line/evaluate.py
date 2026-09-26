"""Deterministic scoring for the frozen single-line-edit-v1 contract.

The evaluator receives completed model responses and objective checks. It never
uses gold actions, provenance, or test specifications to construct model input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tinycomplete.one_line.contract import EditAction, EditState, apply_action, decode_action

Split = Literal["train", "development", "test", "exploratory"]
Outcome = Literal["pass", "fail", "unknown"]
ObjectiveCheck = Callable[[str], bool]


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    state: EditState
    gold_action: EditAction
    after_source: str
    split: Split
    source_repo: str
    generator_family: str
    mechanism: str
    source_type: str
    input_tokens: int | None = None
    objective_check: ObjectiveCheck | None = None
    objective_name: str | None = None
    ambiguity: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.source_repo or not self.generator_family:
            raise ValueError("case, repository, and family identities are required")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens must be nonnegative")
        if self.objective_check is not None and not self.objective_name:
            raise ValueError("objective checks require an auditable name")
        if apply_action(self.state, self.gold_action) != self.after_source:
            raise ValueError("gold action does not produce after_source")
        if self.gold_action.kind == "keep" and self.after_source != self.state.source:
            raise ValueError("keep gold must preserve source")
        if self.gold_action.kind != "keep" and self.after_source == self.state.source:
            raise ValueError("edit-required gold must change source")
        if self.objective_check is not None:
            if not self.objective_check(self.after_source):
                raise ValueError("gold fails its independent objective check")
            if (
                self.gold_action.kind != "keep"
                and self.ambiguity is None
                and self.objective_check(self.state.source)
            ):
                raise ValueError("unchanged source passes an edit-required objective")


@dataclass(frozen=True)
class Prediction:
    case_id: str
    wire: str
    terminated: bool
    generated_tokens: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    split: Split
    language: str
    source_repo: str
    generator_family: str
    mechanism: str
    source_type: str
    input_tokens: int | None
    gold_action: str
    predicted_action: str | None
    parse_status: str
    explicit_termination: bool
    cap_hit: bool
    generated_tokens: int | None
    output_bytes: int
    valid_action: bool
    action_correct: bool
    exact_after: bool
    alternative_valid: bool | None
    edit_success: Outcome
    keep_correct: bool
    false_positive_edit: bool
    hard_scored: bool
    ambiguity: str | None
    confidence: float | None

    @property
    def edit_required(self) -> bool:
        return self.gold_action != "keep"


@dataclass(frozen=True)
class SealedClaim:
    suite_sha256: str
    case_ids_sha256: str
    decision_sha256: str


def case_ids_sha256(case_ids: Sequence[str]) -> str:
    """Hash only stable identities; test contents remain in the sealed file."""
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs")
    payload = json.dumps(sorted(case_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def claim_sealed_evaluation(
    *, suite_manifest: Path, locked_decision: Path, claim_path: Path
) -> SealedClaim:
    """Atomically allow one locked test pass before opening the sealed cases.

    The suite manifest contains IDs and hashes, not test prompts or answers.
    A failed or interrupted test remains claimed and requires a documented new
    suite/plan revision rather than silent repeated selection on test outcomes.
    """
    suite_bytes = suite_manifest.read_bytes()
    decision_bytes = locked_decision.read_bytes()
    suite = json.loads(suite_bytes)
    decision = json.loads(decision_bytes)
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()
    if decision.get("decision_locked") is not True:
        raise ValueError("selection must be locked before test evaluation")
    if decision.get("suite_sha256") != suite_hash:
        raise ValueError("selection is not bound to this sealed suite manifest")
    if not decision.get("selected_artifact_sha256"):
        raise ValueError("selection lacks an artifact identity")
    ids = suite.get("case_ids")
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError("sealed suite manifest lacks case IDs")
    digest = case_ids_sha256(ids)
    if suite.get("case_ids_sha256") != digest:
        raise ValueError("sealed suite case ID digest mismatch")
    claim = SealedClaim(suite_hash, digest, hashlib.sha256(decision_bytes).hexdigest())
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(claim.__dict__, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return claim


def _require_split_permission(cases: Sequence[EvaluationCase], claim: SealedClaim | None) -> None:
    test_ids = [case.id for case in cases if case.split == "test"]
    if not test_ids:
        return
    if claim is None or len(test_ids) != len(cases):
        raise ValueError("sealed test requires an exclusive one-shot claim")
    if case_ids_sha256(test_ids) != claim.case_ids_sha256:
        raise ValueError("sealed test cases differ from the claimed manifest")


def score_case(
    case: EvaluationCase,
    prediction: Prediction,
    *,
    sealed_claim: SealedClaim | None = None,
    max_tokens: int = 64,
) -> CaseScore:
    _require_split_permission([case], sealed_claim)
    return _score_case(case, prediction, max_tokens=max_tokens)


def _score_case(case: EvaluationCase, prediction: Prediction, *, max_tokens: int) -> CaseScore:
    if prediction.case_id != case.id:
        raise ValueError("prediction case ID mismatch")
    parsed = decode_action(
        prediction.wire,
        terminated=prediction.terminated,
        generated_tokens=prediction.generated_tokens,
        max_tokens=max_tokens,
    )
    action = parsed.action if parsed.status == "ok" else None
    after: str | None = None
    if action is not None:
        try:
            after = apply_action(case.state, action)
        except ValueError:
            action = None
    valid = action is not None
    exact = valid and after == case.after_source
    alternative: bool | None = None
    if valid and not exact and case.objective_check is not None:
        assert after is not None
        alternative = bool(case.objective_check(after))
    if case.gold_action.kind == "keep":
        edit_success: Outcome = "unknown"
    elif exact or alternative is True:
        edit_success = "pass"
    elif action is not None and action.kind == "keep":
        edit_success = "fail"
    elif case.objective_check is None and valid and not exact:
        edit_success = "unknown"
    else:
        edit_success = "fail"
    if case.ambiguity is not None:
        edit_success = "unknown"
    return CaseScore(
        case_id=case.id,
        split=case.split,
        language=case.state.filetype,
        source_repo=case.source_repo,
        generator_family=case.generator_family,
        mechanism=case.mechanism,
        source_type=case.source_type,
        input_tokens=case.input_tokens,
        gold_action=case.gold_action.kind,
        predicted_action=action.kind if action is not None else None,
        parse_status=parsed.status
        if valid
        else ("invalid_range" if parsed.status == "ok" else parsed.status),
        explicit_termination=prediction.terminated,
        cap_hit=(
            prediction.generated_tokens is not None
            and (
                prediction.generated_tokens > max_tokens
                or (not prediction.terminated and prediction.generated_tokens >= max_tokens)
            )
        ),
        generated_tokens=prediction.generated_tokens,
        output_bytes=len(prediction.wire.encode("utf-8")),
        valid_action=valid,
        action_correct=action is not None and action.kind == case.gold_action.kind,
        exact_after=exact,
        alternative_valid=alternative,
        edit_success=edit_success,
        keep_correct=case.gold_action.kind == "keep"
        and action is not None
        and action.kind == "keep",
        false_positive_edit=(
            case.gold_action.kind == "keep" and action is not None and action.kind != "keep"
        ),
        hard_scored=case.ambiguity is None,
        ambiguity=case.ambiguity,
        confidence=prediction.confidence,
    )


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    predictions: Sequence[Prediction],
    *,
    sealed_claim: SealedClaim | None = None,
    max_tokens: int = 64,
) -> list[CaseScore]:
    _require_split_permission(cases, sealed_claim)
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate evaluation case IDs")
    by_id = {prediction.case_id: prediction for prediction in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {case.id for case in cases}:
        raise ValueError("predictions must cover every case exactly once")
    return [_score_case(case, by_id[case.id], max_tokens=max_tokens) for case in cases]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(scores: Sequence[CaseScore], *, incorrect_edit_cost: int = 3) -> dict:
    """Retain every denominator; ambiguous cases are exploratory only."""
    if incorrect_edit_cost < 0:
        raise ValueError("incorrect_edit_cost must be nonnegative")
    hard = [row for row in scores if row.hard_scored]
    required = [row for row in hard if row.edit_required]
    keep = [row for row in hard if not row.edit_required]
    verified_required = [row for row in required if row.edit_success != "unknown"]
    correct = sum(row.edit_success == "pass" for row in verified_required)
    incorrect = sum(
        row.valid_action and row.predicted_action != "keep" and row.edit_success == "fail"
        for row in required
    )
    false_edits = sum(row.false_positive_edit for row in keep)
    output_lengths = sorted(
        row.generated_tokens for row in scores if row.generated_tokens is not None
    )
    return {
        "cases": len(scores),
        "hard_cases": len(hard),
        "ambiguous_cases": len(scores) - len(hard),
        "valid_wire": sum(row.parse_status == "ok" for row in scores),
        "valid_action": sum(row.valid_action for row in scores),
        "explicit_termination": sum(row.explicit_termination for row in scores),
        "cap_hits": sum(row.cap_hit for row in scores),
        "edit_required": len(required),
        "edit_required_verified": len(verified_required),
        "edit_success": correct,
        "edit_success_rate": _rate(correct, len(verified_required)),
        "edit_required_unknown": len(required) - len(verified_required),
        "exact_after": sum(row.exact_after for row in required),
        "alternative_valid": sum(row.alternative_valid is True for row in required),
        "keep_cases": len(keep),
        "keep_correct": sum(row.keep_correct for row in keep),
        "keep_recall": _rate(sum(row.keep_correct for row in keep), len(keep)),
        "false_positive_edits": false_edits,
        "false_positive_edit_rate": _rate(false_edits, len(keep)),
        "incorrect_edits": incorrect,
        "utility_cost": incorrect_edit_cost,
        "utility": correct - incorrect_edit_cost * (incorrect + false_edits),
        "actions": dict(Counter(row.predicted_action or "invalid" for row in scores)),
        "by_gold_action": {
            action: summarize_action([row for row in hard if row.gold_action == action])
            for action in ("keep", "insert_before", "replace_line", "delete_line")
        },
        "output_tokens": {
            "known": len(output_lengths),
            "min": output_lengths[0] if output_lengths else None,
            "median": output_lengths[len(output_lengths) // 2] if output_lengths else None,
            "max": output_lengths[-1] if output_lengths else None,
        },
    }


def summarize_action(scores: Sequence[CaseScore]) -> dict:
    return {
        "count": len(scores),
        "valid": sum(row.valid_action for row in scores),
        "action_correct": sum(row.action_correct for row in scores),
        "exact_after": sum(row.exact_after for row in scores),
        "verified_success": sum(row.edit_success == "pass" for row in scores),
        "unknown_success": sum(row.edit_success == "unknown" for row in scores),
    }


def stratified_summary(scores: Sequence[CaseScore], field: str) -> dict[str, dict]:
    if field not in {"language", "mechanism", "source_type", "input_bucket"}:
        raise ValueError("unsupported summary field")
    groups: dict[str, list[CaseScore]] = defaultdict(list)
    for row in scores:
        if field == "input_bucket":
            value = (
                "unknown"
                if row.input_tokens is None
                else (
                    "<=512"
                    if row.input_tokens <= 512
                    else "<=1024"
                    if row.input_tokens <= 1024
                    else ">1024"
                )
            )
        else:
            value = str(getattr(row, field))
        groups[value].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def control_predictions(
    cases: Sequence[EvaluationCase],
    strategy: Literal["gold", "keep", "random", "trivial"],
    *,
    seed: int = 271828,
) -> list[Prediction]:
    """Deterministic controls; trivial uses only visible history and source."""
    from tinycomplete.one_line.contract import encode_action

    rng = random.Random(seed)
    output = []
    for case in cases:
        if strategy == "gold":
            action = case.gold_action
        elif strategy == "keep":
            action = EditAction("keep")
        elif strategy == "trivial":
            action = _trivial_visible_rule(case.state)
        elif strategy == "random":
            action = rng.choice(
                [
                    EditAction("keep"),
                    EditAction("delete_line"),
                    EditAction("replace_line", ""),
                    EditAction("insert_before", ""),
                ]
            )
        else:
            raise ValueError("unknown control")
        output.append(Prediction(case.id, encode_action(action), terminated=True))
    return output


def _trivial_visible_rule(state: EditState) -> EditAction:
    """Copy an exact latest visible substitution on the selected line, else keep."""
    if not state.history:
        return EditAction("keep")
    latest = state.history[-1]
    old = getattr(latest, "old_text", None)
    new = getattr(latest, "new_text", None)
    if (
        not isinstance(old, str)
        or not isinstance(new, str)
        or not old
        or "\n" in new
        or "\r" in new
    ):
        return EditAction("keep")
    source_lines = state.source.splitlines()
    if state.target_row >= len(source_lines):
        return EditAction("keep")
    line = source_lines[state.target_row]
    if line.count(old) != 1 or old == new:
        return EditAction("keep")
    return EditAction("replace_line", line.replace(old, new, 1))


def calibrate_display_threshold(
    scores: Sequence[CaseScore],
    *,
    incorrect_edit_cost: int = 3,
) -> dict:
    """Fit a simple abstention threshold on development predictions only."""
    if not scores or any(row.split != "development" for row in scores):
        raise ValueError("display calibration requires development cases only")
    if any(row.confidence is None for row in scores):
        raise ValueError("calibration requires a confidence for every case")
    if any(
        row.confidence is not None
        and (not math.isfinite(row.confidence) or row.confidence < 0 or row.confidence > 1)
        for row in scores
    ):
        raise ValueError("confidence must be finite and within [0, 1]")
    thresholds = sorted({float(row.confidence) for row in scores if row.confidence is not None})
    candidates = [float("inf"), *thresholds]
    ranked = []
    for threshold in candidates:
        shown = [
            row
            for row in scores
            if row.hard_scored
            and row.valid_action
            and row.predicted_action != "keep"
            and row.confidence is not None
            and row.confidence >= threshold
        ]
        correct = sum(row.edit_success == "pass" for row in shown)
        incorrect = sum(row.edit_success == "fail" or row.false_positive_edit for row in shown)
        unknown = len(shown) - correct - incorrect
        required = sum(row.hard_scored and row.edit_required for row in scores)
        shown_required = sum(row.edit_required for row in shown)
        ranked.append(
            {
                "threshold": threshold if threshold != float("inf") else None,
                "displayed": len(shown),
                "correct_displayed": correct,
                "incorrect_displayed": incorrect,
                "unknown_displayed": unknown,
                "precision_verified": _rate(correct, correct + incorrect),
                "coverage_of_edit_required": _rate(shown_required, required),
                "utility": correct - incorrect_edit_cost * (incorrect + unknown),
            }
        )
    return max(
        ranked, key=lambda row: (row["utility"], row["correct_displayed"], -(row["displayed"]))
    )


def clustered_interval(
    scores: Sequence[CaseScore],
    *,
    cluster_by: Literal["repository", "family"],
    samples: int = 2_000,
    seed: int = 271828,
) -> dict:
    """Bootstrap edit success over repository or generated-task-family units."""
    if samples < 1:
        raise ValueError("samples must be positive")
    field = "source_repo" if cluster_by == "repository" else "generator_family"
    groups: dict[str, list[CaseScore]] = defaultdict(list)
    for row in scores:
        if row.hard_scored and row.edit_required and row.edit_success != "unknown":
            groups[str(getattr(row, field))].append(row)
    keys = sorted(groups)
    if not keys:
        raise ValueError("no verified edit-required groups")
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sampled = [groups[rng.choice(keys)] for _ in keys]
        flat = [row for group in sampled for row in group]
        values.append(sum(row.edit_success == "pass" for row in flat) / len(flat))
    values.sort()
    point_rows = [row for group in groups.values() for row in group]
    return {
        "cluster_by": cluster_by,
        "cluster_count": len(keys),
        "samples": samples,
        "seed": seed,
        "point": sum(row.edit_success == "pass" for row in point_rows) / len(point_rows),
        "interval_95": [values[int(0.025 * (samples - 1))], values[int(0.975 * (samples - 1))]],
    }


def paired_outcomes(
    first: Sequence[CaseScore],
    second: Sequence[CaseScore],
    *,
    samples: int = 2_000,
    seed: int = 271828,
) -> dict:
    """Compare matched verified edit tasks without claiming null-result equivalence."""
    if samples < 1:
        raise ValueError("samples must be positive")
    left = {row.case_id: row for row in first}
    right = {row.case_id: row for row in second}
    if len(left) != len(first) or len(right) != len(second) or set(left) != set(right):
        raise ValueError("paired scores must have identical unique case IDs")
    paired = []
    for case_id in sorted(left):
        a, b = left[case_id], right[case_id]
        if (a.source_repo, a.generator_family, a.gold_action, a.split) != (
            b.source_repo,
            b.generator_family,
            b.gold_action,
            b.split,
        ):
            raise ValueError("paired case metadata differ")
        if (
            a.hard_scored
            and b.hard_scored
            and a.edit_required
            and a.edit_success != "unknown"
            and b.edit_success != "unknown"
        ):
            paired.append((a, b))
    if not paired:
        raise ValueError("no paired verified edit-required cases")
    wins = sum(a.edit_success != "pass" and b.edit_success == "pass" for a, b in paired)
    losses = sum(a.edit_success == "pass" and b.edit_success != "pass" for a, b in paired)
    groups: dict[str, list[tuple[CaseScore, CaseScore]]] = defaultdict(list)
    for a, b in paired:
        groups[a.source_repo].append((a, b))
    keys = sorted(groups)
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        rows = [pair for _ in keys for pair in groups[rng.choice(keys)]]
        differences.append(
            sum((b.edit_success == "pass") - (a.edit_success == "pass") for a, b in rows)
            / len(rows)
        )
    differences.sort()
    return {
        "definition": "second minus first, verified edit-required success",
        "paired_cases": len(paired),
        "repository_clusters": len(keys),
        "second_wins": wins,
        "second_losses": losses,
        "ties": len(paired) - wins - losses,
        "difference": (wins - losses) / len(paired),
        "interval_95": [
            differences[int(0.025 * (samples - 1))],
            differences[int(0.975 * (samples - 1))],
        ],
        "samples": samples,
        "seed": seed,
    }
