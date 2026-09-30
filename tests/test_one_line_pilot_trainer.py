from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tinycomplete.one_line.pilot_data import LICENSE_MIXED, LICENSE_MIXED_HISTORY

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_one_line.py"
SPEC = importlib.util.spec_from_file_location("train_one_line_pilot_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def _row(index: int) -> dict[str, object]:
    source = f"const value = {index};\n"
    return {
        "id": f"instinct-{index}",
        "split": "train",
        "source_type": "continue_instinct_observed",
        "source_license": "Apache-2.0",
        "state": {
            "file_id": f"core/example-{index}.ts",
            "filetype": "typescript",
            "source": source,
            "target_row": 0,
            "cursor_col": 0,
            "history": [],
            "relevant": [],
        },
        "action": {"kind": "replace_line", "text": f"const value = {index + 1};"},
        "after_source": f"const value = {index + 1};\n",
        "validation": {"replay_verified": True, "inferability_reviewed": False},
    }


def test_pilot_phase_accepts_replayed_rows_without_weakening_main_gate(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [_row(index) for index in range(128)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = trainer.sha256_file(path)
    loaded = trainer.load_training_rows(path, digest, phase="pilot", minimum_main_train=20_000)
    assert len(loaded) == 128
    assert trainer.peak_learning_rate("pilot", None) == 1e-5
    with pytest.raises(ValueError, match="inferability-reviewed"):
        trainer.load_training_rows(path, digest, phase="main", minimum_main_train=20_000)


def test_pilot_rejects_unverified_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = _row(0)
    row["validation"] = {"replay_verified": False}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="replay-verified"):
        trainer.load_training_rows(
            path, trainer.sha256_file(path), phase="pilot", minimum_main_train=20_000
        )


@pytest.mark.parametrize(
    ("policy", "source_type"),
    [
        (LICENSE_MIXED, "public_source_supervised_candidate"),
        (LICENSE_MIXED_HISTORY, "reviewed_public_history_candidate"),
    ],
)
def test_license_mixed_policies_dispatch_through_the_existing_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy, source_type: str
) -> None:
    rows = []
    for index in range(128):
        source = f"value = {index}\n"
        rows.append(
            {
                "id": f"candidate-{index}",
                "candidate_id": f"candidate-{index}",
                "split": "train",
                "source_type": source_type,
                "source_license": "MIT",
                "state": {
                    "file_id": f"src/example-{index}.py",
                    "filetype": "python",
                    "source": source,
                    "target_row": 0,
                    "cursor_col": 0,
                    "history": [],
                    "relevant": [],
                },
                "action": {"kind": "keep"},
                "after_source": source,
            }
        )
    path = tmp_path / f"{policy.data_schema}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    observed = []
    monkeypatch.setattr(
        trainer,
        "validate_pilot_row",
        lambda _row, selected, *, package_root: observed.append(selected),
    )

    loaded = trainer.load_training_rows(
        path,
        trainer.sha256_file(path),
        phase="pilot",
        minimum_main_train=20_000,
        pilot_schema=policy.data_schema,
        package_root=tmp_path,
    )
    assert len(loaded) == 128
    assert observed and all(selected is policy for selected in observed)


def test_history_pilot_uses_declared_128_loss_scale_without_changing_v1() -> None:
    history_plan = {
        "data": {"schema": LICENSE_MIXED_HISTORY.data_schema},
        "training": {"disposable_fixture": {"initial_loss_scale": 128.0}},
    }
    assert (
        trainer.initial_loss_scale_for_run(
            plan=history_plan, phase="pilot", disposable_fixture_plan=None
        )
        == 128.0
    )
    history_plan["training"]["disposable_fixture"]["initial_loss_scale"] = 256.0
    with pytest.raises(ValueError, match="128 initial loss scale"):
        trainer.initial_loss_scale_for_run(
            plan=history_plan, phase="pilot", disposable_fixture_plan=None
        )

    legacy_plan = {"data": {"schema": LICENSE_MIXED.data_schema}, "training": {}}
    assert (
        trainer.initial_loss_scale_for_run(
            plan=legacy_plan, phase="pilot", disposable_fixture_plan=None
        )
        == 256.0
    )


def test_history_fixture_v2_requires_128_and_reader_keeps_strict_json(tmp_path: Path) -> None:
    fixture = {
        "schema": trainer.FIXTURE_V2_PLAN_SCHEMA,
        "examples": 64,
        "epochs": 1,
        "effective_batch_examples": 2,
        "microbatch_examples": 2,
        "peak_learning_rate": 1e-4,
        "expected_updates": 32,
        "quality_evidence": False,
        "implementation_viability_schema": trainer.FIXTURE_VIABILITY_SCHEMA,
        "decode_examples_per_action": 4,
        "minimum_exact_actions_per_action": 3,
        "eos_required": True,
        "nonpadding_training_input_tokens": 10_000,
        "initial_loss_scale": 128.0,
    }
    plan = {
        "data": {"schema": LICENSE_MIXED_HISTORY.data_schema},
        "training": {"disposable_fixture": fixture},
    }
    assert (
        trainer.validate_disposable_fixture_plan(
            plan, phase="fixture", epochs=1, microbatch=2
        )
        == fixture
    )
    fixture["initial_loss_scale"] = 256.0
    with pytest.raises(ValueError, match="128 initial loss scale"):
        trainer.validate_disposable_fixture_plan(
            plan, phase="fixture", epochs=1, microbatch=2
        )

    duplicate_keys = tmp_path / "duplicate-history-row.jsonl"
    duplicate_keys.write_text('{"id":"a","id":"b"}\n')
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        trainer.load_training_rows(
            duplicate_keys,
            trainer.sha256_file(duplicate_keys),
            phase="pilot",
            minimum_main_train=20_000,
            pilot_schema=LICENSE_MIXED_HISTORY.data_schema,
            package_root=tmp_path,
        )
