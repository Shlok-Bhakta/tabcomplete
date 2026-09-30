from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tinycomplete.one_line.data import near_duplicate_key
from tinycomplete.one_line.pilot_data import (
    CONSTRUCTIVE,
    INSTINCT,
    constructive_row_bindings,
    policy_for_schema,
    validate_aggregate_budget,
    validate_constructive_manifest,
    validate_constructive_review,
    validate_constructive_splits,
    validate_pilot_row,
)


def verified_row(kind: str = "replace_line") -> dict:
    return {
        "source_type": CONSTRUCTIVE.source_type,
        "source_license": "MIT",
        "source_group_id": "fixture-group",
        "template_id": "fixture-template",
        "independent_reviewer_id": "synthetic-test-reviewer",
        "oracle_sha256": "a" * 64,
        "action": {"kind": kind},
        "validation": {
            "replay_verified": True,
            "history_replays_to_state": True,
            "apply_reconstructs_after": True,
            "objective_verified": True,
            "inferability_reviewed": True,
            "accepted_training": True,
            "wrong_action_controls_rejected": True,
            "history_leakage_check": True,
            "before_objective_passes": kind == "keep",
            "after_objective_passes": True,
        },
    }


def test_explicit_constructive_policy_has_no_source_fallback():
    row = verified_row()
    validate_pilot_row(row, policy_for_schema(CONSTRUCTIVE.data_schema))
    with pytest.raises(ValueError, match="source type"):
        validate_pilot_row(row, INSTINCT)
    with pytest.raises(ValueError, match="unapproved"):
        policy_for_schema("unknown-pilot")


def test_direct_trainer_cannot_reset_prior_campaign_exposure():
    budgets = {"prior_training_input_tokens": 538_275, "prior_session_wall_seconds": 14_400}
    validate_aggregate_budget(
        budgets,
        planned_tokens=100_000,
        session_seconds=7200,
        external_campaign_tokens=538_275,
    )
    with pytest.raises(ValueError, match="aggregate campaign"):
        validate_aggregate_budget(
            budgets,
            planned_tokens=100_000,
            session_seconds=7200,
            external_campaign_tokens=0,
        )


def test_replay_alone_does_not_promote_a_constructive_candidate():
    row = verified_row()
    row["validation"]["accepted_training"] = False
    with pytest.raises(ValueError, match="accepted_training"):
        validate_pilot_row(row, CONSTRUCTIVE)
    row = verified_row()
    row["validation"]["wrong_action_controls_rejected"] = False
    with pytest.raises(ValueError, match="wrong_action_controls_rejected"):
        validate_pilot_row(row, CONSTRUCTIVE)


def test_keep_requires_a_positive_completed_state_objective():
    row = verified_row("keep")
    validate_pilot_row(row, CONSTRUCTIVE)
    row["validation"]["before_objective_passes"] = False
    with pytest.raises(ValueError, match="edit/keep"):
        validate_pilot_row(row, CONSTRUCTIVE)
    row = verified_row()
    row["validation"]["before_objective_passes"] = True
    with pytest.raises(ValueError, match="edit/keep"):
        validate_pilot_row(row, CONSTRUCTIVE)


def split_row(identifier: str, split: str, source: str) -> dict:
    return {
        "id": identifier,
        "split": split,
        "source_repo": CONSTRUCTIVE.dataset_id,
        "source_group_id": identifier,
        "session_or_commit": identifier,
        "generator_family": identifier,
        "template_id": identifier,
        "state": {
            "file_id": "sample/main.py",
            "filetype": "python",
            "source": source,
            "target_row": 0,
            "cursor_col": 0,
        },
        "action": {"kind": "replace_line", "text": "return 0"},
    }


def test_one_authored_source_keeps_real_families_isolated():
    train = split_row("train-family", "train", "result = seed + step\n")
    dev = split_row("dev-family", "development", "if values:\n    return values[0]\n")
    result = validate_constructive_splits([train, dev])
    assert result["authored_source_count"] == 1
    assert result["repository_diversity_claimed"] is False
    dev["generator_family"] = train["generator_family"]
    with pytest.raises(ValueError, match="crosses splits"):
        validate_constructive_splits([train, dev])


def test_metadata_cannot_hide_a_duplicate_constructive_model_state():
    train = split_row("first", "train", "result = seed + step\n")
    duplicate = split_row("second", "train", "result = seed + step\n")
    duplicate["state"]["metadata_nonce"] = "different"
    with pytest.raises(ValueError, match="duplicate constructive model input"):
        validate_constructive_splits([train, duplicate])


def test_constructive_manifest_cannot_omit_independent_evidence():
    manifest = {
        "schema": CONSTRUCTIVE.data_schema,
        "dataset_id": CONSTRUCTIVE.dataset_id,
        "dataset_license": CONSTRUCTIVE.dataset_license,
        "source_file_license_status": CONSTRUCTIVE.file_license_status,
        "generator_sha256": "a" * 64,
        "oracle_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="independent_review_sha256"):
        validate_constructive_manifest(manifest)
    manifest["independent_review_sha256"] = "c" * 64
    validate_constructive_manifest(manifest)


def review_fixture(tmp_path: Path) -> tuple[dict, list[dict], dict, Path]:
    rows = [
        split_row("a", "train", "result = seed + step\n"),
        split_row("b", "development", "if values:\n    return values[0]\n"),
        split_row("c", "test_new_mechanism", "for item in items:\n    yield item\n"),
    ]
    for row in rows:
        row["independent_reviewer_id"] = "disposable-test-reviewer"
    review = {
        "schema": "one-line-constructive-independent-review-v1",
        "independent_reviewer_id": "disposable-test-reviewer",
        "generator_sha256": "a" * 64,
        "oracle_sha256": "b" * 64,
        "full_split_audit_complete": True,
        "rows": [
            {
                **{
                    key: row[key]
                    for key in (
                        "id",
                        "split",
                        "source_group_id",
                        "session_or_commit",
                        "generator_family",
                        "template_id",
                    )
                },
                **constructive_row_bindings(row),
                "near_duplicate_sha256": near_duplicate_key(row),
                **{
                    key: True
                    for key in (
                        "accepted_training",
                        "inferability_reviewed",
                        "objective_verified",
                        "wrong_action_controls_rejected",
                        "history_leakage_check",
                    )
                },
            }
            for row in rows
        ],
    }
    path = tmp_path / "independent_review.json"
    path.write_text(json.dumps(review))
    manifest = {
        "generator_sha256": "a" * 64,
        "oracle_sha256": "b" * 64,
        "independent_review_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate_split_counts": {"train": 1, "development": 1, "test_new_mechanism": 1},
    }
    return manifest, rows[:2], review, path


def test_review_binds_the_actual_supervised_action_not_a_plausible_digest(tmp_path: Path):
    manifest, rows, _review, path = review_fixture(tmp_path)
    validate_constructive_review(manifest, rows, path)
    rows[0]["action"]["text"] = "return 999"
    with pytest.raises(ValueError, match="reviewed state/action"):
        validate_constructive_review(manifest, rows, path)


def test_full_review_detects_family_leakage_into_reserved_holdout(tmp_path: Path):
    manifest, rows, review, path = review_fixture(tmp_path)
    review["rows"][2]["generator_family"] = rows[0]["generator_family"]
    path.write_text(json.dumps(review))
    manifest["independent_review_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="crosses splits"):
        validate_constructive_review(manifest, rows, path)


def test_silent_model_is_not_scored_as_success_on_required_edits():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_one_line_gpu_pilot.py"
    spec = importlib.util.spec_from_file_location("constructive_outcome_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcomes = [
        {
            "gold_action": gold,
            "predicted_action": "keep",
            "exact_after": gold == "keep",
            "valid_action": True,
            "source_changed": False,
        }
        for gold in ("keep", "keep", "replace_line", "insert_before", "delete_line")
    ]
    result = module.action_outcome_counts(outcomes)
    assert result["edit_required_cases"] == 3
    assert result["edit_required_exact"] == 0
    assert result["edit_required_predicted_keep"] == 3
    assert result["no_edit_cases"] == result["no_edit_recalled"] == 2
    assert result["no_edit_false_positive_changes"] == 0
    assert result["functional_success"] is None
