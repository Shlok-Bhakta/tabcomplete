from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tinycomplete.one_line.contract import EditAction, EditState, RecentEdit, encode_action
from tinycomplete.one_line.evaluate import (
    EvaluationCase,
    Prediction,
    calibrate_display_threshold,
    case_ids_sha256,
    claim_sealed_evaluation,
    clustered_interval,
    control_predictions,
    evaluate_cases,
    paired_outcomes,
    score_case,
    stratified_summary,
    summarize,
)


def make_case(
    case_id: str,
    *,
    source: str = "value = 1\n",
    action: EditAction | None = None,
    after: str = "value = 2\n",
    split: str = "development",
    repo: str = "public/example",
    family: str = "change-value",
    check=None,
    ambiguity: str | None = None,
    history: tuple[RecentEdit, ...] = (),
) -> EvaluationCase:
    state = EditState(
        file_id=f"{repo}/{case_id}.py",
        filetype="python",
        source=source,
        target_row=0,
        cursor_col=0,
        history=history,
    )
    return EvaluationCase(
        id=case_id,
        state=state,
        gold_action=action or EditAction("replace_line", "value = 2"),
        after_source=after,
        split=split,  # type: ignore[arg-type]
        source_repo=repo,
        generator_family=family,
        mechanism="assignment-update",
        source_type="public-history",
        input_tokens=513,
        objective_check=check,
        objective_name="independent-output-check" if check else None,
        ambiguity=ambiguity,
    )


def pred(
    case: EvaluationCase, action: EditAction, *, confidence: float | None = None
) -> Prediction:
    return Prediction(
        case.id, encode_action(action), True, generated_tokens=4, confidence=confidence
    )


def test_gold_keep_wrong_edit_and_distinct_denominators() -> None:
    edit = make_case("edit")
    keep = make_case("keep", action=EditAction("keep"), after="value = 1\n")
    rows = evaluate_cases(
        [edit, keep],
        [
            pred(edit, EditAction("replace_line", "value = 2")),
            pred(keep, EditAction("delete_line")),
        ],
    )
    summary = summarize(rows)
    assert summary["edit_required"] == 1
    assert summary["edit_success"] == 1
    assert summary["keep_cases"] == 1
    assert summary["false_positive_edits"] == 1
    assert summary["keep_recall"] == 0.0
    assert summary["utility"] == -2  # one correct edit, one false edit at cost three
    assert summary["by_gold_action"]["replace_line"]["exact_after"] == 1


def test_alternative_valid_requires_independent_objective() -> None:
    case = make_case("alt", check=lambda source: source in {"value = 2\n", "value = 1 + 1\n"})
    alternative = score_case(case, pred(case, EditAction("replace_line", "value = 1 + 1")))
    assert alternative.exact_after is False
    assert alternative.alternative_valid is True
    assert alternative.edit_success == "pass"

    no_check = make_case("no-check")
    uncertain = score_case(no_check, pred(no_check, EditAction("replace_line", "value = 1 + 1")))
    assert uncertain.alternative_valid is None
    assert uncertain.edit_success == "unknown"
    assert summarize([uncertain])["edit_required_verified"] == 0


def test_objective_validation_rejects_trivial_or_incorrect_gold() -> None:
    with pytest.raises(ValueError, match="unchanged source"):
        make_case("bad", check=lambda source: True)
    with pytest.raises(ValueError, match="gold fails"):
        make_case("bad", check=lambda source: False)
    with pytest.raises(ValueError, match="gold action"):
        make_case("bad", after="value = 3\n")


def test_ambiguous_cases_are_reported_but_not_hard_failures() -> None:
    case = make_case("ambiguous", ambiguity="multiple plausible next edits")
    row = score_case(case, pred(case, EditAction("keep")))
    summary = summarize([row])
    assert row.edit_success == "unknown"
    assert summary["ambiguous_cases"] == 1
    assert summary["edit_required"] == 0


def test_incomplete_invalid_wire_and_out_of_range_actions_do_not_pass() -> None:
    case = make_case("cutoff")
    cutoff = score_case(case, Prediction(case.id, "R\tvalue = 2", False, generated_tokens=64))
    malformed = score_case(case, Prediction(case.id, "R value = 2", True, generated_tokens=4))
    assert cutoff.parse_status == "incomplete"
    assert cutoff.cap_hit is True
    assert cutoff.valid_action is False
    assert malformed.parse_status == "malformed"
    eof = EvaluationCase(
        id="eof",
        state=EditState("public/eof.py", "python", "value = 1", 1, 0),
        gold_action=EditAction("insert_before", "value = 2"),
        after_source="value = 1\nvalue = 2",
        split="development",
        source_repo="public/eof",
        generator_family="append",
        mechanism="append",
        source_type="synthetic",
    )
    out_of_range = score_case(eof, pred(eof, EditAction("delete_line")))
    assert out_of_range.parse_status == "invalid_range"
    assert out_of_range.valid_action is False


def test_controls_are_deterministic_and_trivial_uses_only_visible_history() -> None:
    case = make_case(
        "rename",
        source="old_name = 1\n",
        action=EditAction("replace_line", "new_name = 1"),
        after="new_name = 1\n",
        history=(RecentEdit(0, "old_name", "new_name"),),
    )
    assert control_predictions([case], "trivial")[0].wire == "R\tnew_name = 1"
    assert control_predictions([case], "keep")[0].wire == "N"
    assert control_predictions([case], "gold")[0].wire == "R\tnew_name = 1"
    assert control_predictions([case] * 4, "random", seed=5) == control_predictions(
        [case] * 4, "random", seed=5
    )


def test_strata_cluster_uncertainty_and_paired_outcomes() -> None:
    a = make_case("a", repo="r1", check=lambda source: source == "value = 2\n")
    b = make_case("b", repo="r2", family="other", check=lambda source: source == "value = 2\n")
    left = evaluate_cases([a, b], [pred(a, EditAction("keep")), pred(b, EditAction("keep"))])
    right = evaluate_cases(
        [a, b], [pred(a, EditAction("replace_line", "value = 2")), pred(b, EditAction("keep"))]
    )
    summary = stratified_summary(right, "input_bucket")
    assert summary["<=1024"]["cases"] == 2
    ci = clustered_interval(right, cluster_by="repository", samples=100, seed=7)
    assert ci["cluster_count"] == 2
    assert ci["point"] == 0.5
    paired = paired_outcomes(left, right, samples=100, seed=7)
    assert paired["second_wins"] == 1
    assert paired["second_losses"] == 0
    assert paired["ties"] == 1
    assert paired["difference"] == 0.5


def test_display_calibration_is_development_only() -> None:
    a = make_case("a", check=lambda source: source == "value = 2\n")
    b = make_case("b", check=lambda source: source == "value = 2\n")
    rows = evaluate_cases(
        [a, b],
        [
            pred(a, EditAction("replace_line", "value = 2"), confidence=0.9),
            pred(b, EditAction("delete_line"), confidence=0.1),
        ],
    )
    selection = calibrate_display_threshold(rows)
    assert selection["threshold"] == 0.9
    assert selection["displayed"] == 1
    assert selection["precision_verified"] == 1.0
    test_row = replace(rows[0], split="test")
    with pytest.raises(ValueError, match="development"):
        calibrate_display_threshold([test_row])


def test_sealed_claim_is_one_shot_and_matches_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "sealed-manifest.json"
    manifest.write_text(
        json.dumps({"case_ids": ["sealed-1"], "case_ids_sha256": case_ids_sha256(["sealed-1"])})
    )
    decision = tmp_path / "locked-selection.json"
    decision.write_text(
        json.dumps(
            {
                "decision_locked": True,
                "suite_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "selected_artifact_sha256": "a" * 64,
            }
        )
    )
    case = make_case("sealed-1", split="test")
    prediction = pred(case, EditAction("keep"))
    with pytest.raises(ValueError, match="one-shot"):
        evaluate_cases([case], [prediction])
    with pytest.raises(ValueError, match="one-shot"):
        score_case(case, prediction)
    claim = claim_sealed_evaluation(
        suite_manifest=manifest, locked_decision=decision, claim_path=tmp_path / "claim.json"
    )
    assert evaluate_cases([case], [prediction], sealed_claim=claim)[0].case_id == "sealed-1"
    with pytest.raises(FileExistsError):
        claim_sealed_evaluation(
            suite_manifest=manifest, locked_decision=decision, claim_path=tmp_path / "claim.json"
        )
    wrong = make_case("other", split="test")
    with pytest.raises(ValueError, match="claimed manifest"):
        evaluate_cases([wrong], [pred(wrong, EditAction("keep"))], sealed_claim=claim)
