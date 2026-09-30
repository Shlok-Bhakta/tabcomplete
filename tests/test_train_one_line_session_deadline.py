from __future__ import annotations

import runpy
from pathlib import Path

_TRAINER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "train_one_line.py"),
    run_name="tabcomplete_train_one_line_deadline_test",
)


def test_pilot_deadline_includes_preparation_and_model_loading() -> None:
    deadline_from_invocation = _TRAINER["session_deadline_from_invocation"]
    reserve_seconds = _TRAINER["finalization_reserve_seconds"]
    remaining_seconds = _TRAINER["remaining_training_seconds"]

    invocation_started = 100.0
    deadline = deadline_from_invocation(invocation_started, 120.0)
    assert deadline == 7_300.0

    # After 99 minutes of preparation and loading, the frozen 20-minute reserve
    # leaves one minute for training.
    reserve = reserve_seconds(20.0, 0.0)
    assert reserve == 1_200.0
    assert remaining_seconds(deadline, invocation_started + 99 * 60, reserve) == 60.0

    # At exactly the reserve boundary the worker must bail before its first
    # update. A slow prior save expands the reserve instead of moving deadline.
    assert remaining_seconds(deadline, invocation_started + 100 * 60, reserve) == 0.0
    slow_save_reserve = reserve_seconds(20.0, 600.0)
    assert slow_save_reserve == 1_260.0
    assert remaining_seconds(deadline, invocation_started + 99 * 60, slow_save_reserve) == 0.0


def test_disposable_fixture_v2_plan_pins_updates_batch_and_viability_budget() -> None:
    validate = _TRAINER["validate_disposable_fixture_plan"]
    plan = {
        "training": {
            "disposable_fixture": {
                "schema": "single-line-disposable-training-fixture-v2",
                "examples": 64,
                "epochs": 1,
                "effective_batch_examples": 2,
                "microbatch_examples": 2,
                "peak_learning_rate": 1e-4,
                "expected_updates": 32,
                "nonpadding_training_input_tokens": 10_592,
                "quality_evidence": False,
                "implementation_viability_schema": (
                    "single-line-disposable-implementation-viability-v1"
                ),
                "decode_examples_per_action": 4,
                "minimum_exact_actions_per_action": 3,
                "eos_required": True,
            }
        }
    }
    assert validate(plan, phase="fixture", epochs=1, microbatch=2) == plan[
        "training"
    ]["disposable_fixture"]
    assert validate(plan, phase="pilot", epochs=1, microbatch=2) is None
    assert validate(
        {
            "training": {
                "disposable_fixture": {
                    "schema": "single-line-disposable-training-fixture-v1"
                }
            }
        },
        phase="fixture",
        epochs=1,
        microbatch=2,
    ) is None


def test_disposable_fixture_viability_requires_every_action_header_and_eos() -> None:
    score = _TRAINER["disposable_fixture_implementation_viability"]
    kinds = ("keep", "replace_line", "insert_before", "delete_line")
    observations = []
    for kind in kinds:
        for index in range(4):
            passed = index < 3
            observations.append(
                {
                    "gold_action": kind,
                    "valid_action": passed,
                    "terminated_by_eos": passed,
                    "exact_action": passed,
                }
            )
    result = score(observations, training_complete=True)
    assert result["passed"] is True
    assert result["quality_evidence"] is False
    assert result["scope"] == "answer-cued_disposable_codec_only"
    assert all(
        result["per_action"][kind]["exact_actions"] == 3 for kind in kinds
    )

    failed = [dict(row) for row in observations]
    failed[-4]["exact_action"] = False
    failed[-3]["valid_action"] = False
    failed[-3]["terminated_by_eos"] = False
    result = score(failed, training_complete=True)
    assert result["passed"] is False
    assert "delete_line:below_3_of_4_decode_threshold" in result["failure_reasons"]
    assert score(observations, training_complete=False)["passed"] is False


def test_disposable_fixture_identity_survives_json_tuple_roundtrip() -> None:
    import json

    from tinycomplete.one_line.train import disposable_fixture_rows

    rows = disposable_fixture_rows()
    loaded = json.loads(json.dumps(rows))
    digest = _TRAINER["canonical_sha256"]
    assert digest(loaded) == digest(rows)
    loaded[0]["state"]["source"] = "altered = True\n"
    assert digest(loaded) != digest(rows)
