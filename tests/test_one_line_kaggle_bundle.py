from __future__ import annotations

import ast
import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle/one_line_r1"))
from build_bundle import (  # noqa: E402
    check_campaign_budget,
    prepare_bundle,
    sha256_file,
    validate_data_gate,
    validate_source_rows,
    verify_pulled_output,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"approved-q25-weight-fixture")
    (model / "tokenizer.json").write_bytes(b"{}")
    (model / "config.json").write_bytes(b'{"model_type":"qwen2"}')
    data = tmp_path / "train.jsonl"
    rows = []
    for index in range(2):
        source = f"value_{index} = 1\n"
        rows.append(
            {
                "id": f"synthetic-{index}",
                "split": "train",
                "state": {
                    "file_id": f"public/fixture_{index}.py",
                    "filetype": "python",
                    "source": source,
                    "target_row": 0,
                    "cursor_col": 0,
                    "history": [],
                    "relevant": [],
                },
                "action": {"kind": "keep", "text": None},
                "after_source": source,
                "source_type": "synthetic_training_diagnostic",
                "source_license": "CC0-1.0",
                "validation": {"inferability_reviewed": True},
            }
        )
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data_manifest = tmp_path / "data_manifest.json"
    _write_json(
        data_manifest,
        {
            "accepted_train": 2,
            "source_groups": 2,
            "mechanisms": 2,
            "maximum_repository_fraction": 0.5,
            "maximum_template_fraction": 0.5,
            "synthetic_repair_fraction": 0.0,
            "real_source_fraction": 0.0,
            "real_source_exception": "training-only synthetic bundle test",
            "train_sha256": sha256_file(data),
            "split_manifest_sha256": "a" * 64,
            "planned_training_input_tokens": 100,
        },
    )
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {"selected_peak_lr": 1e-5, "development_evidence_sha256": "b" * 64},
    )
    config = {
        "suite_revision": 3,
        "student": {
            "initializer": "untouched_pretrained",
            "model_id": "Qwen/Qwen2.5-Coder-0.5B",
            "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
            "weight_sha256": sha256_file(model / "model.safetensors"),
            "tokenizer_sha256": sha256_file(model / "tokenizer.json"),
        },
        "data": {
            "minimum_main_train": 2,
            "minimum_source_groups": 2,
            "minimum_mechanisms": 2,
            "maximum_repository_fraction": 0.5,
            "maximum_template_fraction": 0.5,
            "maximum_synthetic_repair_fraction": 0.1,
            "target_real_source_fraction": 0.5,
        },
        "budget": {
            "maximum_kaggle_t4x2_session_wall_hours": 24,
            "maximum_nonpadding_training_input_tokens": 100_000_000,
            "maximum_new_persistent_local_research_bytes": 12 * 1024**3,
            "minimum_session_finalization_minutes": 15,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = {
        "suite_revision": 3,
        "config_sha256": sha256_file(config_path),
        "quota_at_freeze": {"renewal": "2026-10-03T00:00:00"},
        "budgets": config["budget"],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    ledger_path = tmp_path / "ledger.json"
    _write_json(ledger_path, {"session_wall_seconds": 0, "training_input_tokens": 0})
    return {
        "model": model,
        "train": data,
        "data_manifest": data_manifest,
        "selection": selection,
        "config": config,
        "config_path": config_path,
        "plan": plan,
        "plan_path": plan_path,
        "ledger_path": ledger_path,
        "quota": {"remaining": 10.0, "active_jobs": [], "renewal": "2026-10-03T00:00:00"},
    }


def test_data_gate_blocks_current_zero_accepted_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture["data_manifest"].read_text())
    manifest["accepted_train"] = 0
    with pytest.raises(ValueError, match="20,000"):
        validate_data_gate(manifest, fixture["config"], fixture["train"])


def test_source_egress_rejects_private_or_unreviewed_rows(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    validate_source_rows(fixture["train"], expected_count=2)
    first = json.loads(fixture["train"].read_text().splitlines()[0])
    first["source_type"] = "private_editor_feedback"
    fixture["train"].write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved source"):
        validate_source_rows(fixture["train"], expected_count=1)


def test_quota_and_campaign_caps_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    check_campaign_budget(
        plan=fixture["plan"],
        ledger={"session_wall_seconds": 0, "training_input_tokens": 0},
        quota=fixture["quota"],
        session_seconds=3600,
        planned_tokens=100,
    )
    bad = {**fixture["quota"], "active_jobs": [{"reference": "owned/active"}]}
    with pytest.raises(ValueError, match="active"):
        check_campaign_budget(
            plan=fixture["plan"],
            ledger={"session_wall_seconds": 0, "training_input_tokens": 0},
            quota=bad,
            session_seconds=3600,
            planned_tokens=100,
        )
    unknown = {**fixture["quota"], "remaining": None}
    with pytest.raises(ValueError, match="two-hour"):
        check_campaign_budget(
            plan=fixture["plan"],
            ledger={"session_wall_seconds": 1, "training_input_tokens": 0},
            quota=unknown,
            session_seconds=3600,
            planned_tokens=100,
        )
    renewed = {**fixture["quota"], "renewal": "2026-10-10T00:00:00"}
    with pytest.raises(ValueError, match="renewed"):
        check_campaign_budget(
            plan=fixture["plan"],
            ledger={"session_wall_seconds": 0, "training_input_tokens": 0},
            quota=renewed,
            session_seconds=3600,
            planned_tokens=100,
        )


def test_prepared_bundle_is_private_pinned_and_has_no_fallback_model(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    # Hugging Face snapshot files are symlinks into its blob cache. The staged
    # private dataset must contain the actual weight bytes, not a broken link.
    weight = fixture["model"] / "model.safetensors"
    blob = fixture["model"] / "weight-blob"
    weight.rename(blob)
    weight.symlink_to(blob.name)
    output = tmp_path / "bundle"
    record = prepare_bundle(
        config_path=fixture["config_path"],
        plan_path=fixture["plan_path"],
        ledger_path=fixture["ledger_path"],
        model_dir=fixture["model"],
        train_path=fixture["train"],
        data_manifest_path=fixture["data_manifest"],
        selection_path=fixture["selection"],
        output=output,
        dataset_id="shlokbhakta/test-one-line-inputs",
        kernel_id="shlokbhakta/test-one-line-main-s01",
        commit="c" * 40,
        quota=fixture["quota"],
        session_seconds=3600,
    )
    assert record["kernel_metadata"]["is_private"] is True
    assert record["kernel_metadata"]["machine_shape"] == "NvidiaTeslaT4"
    assert record["kernel_metadata"]["dataset_sources"] == ["shlokbhakta/test-one-line-inputs"]
    assert record["dataset_metadata"]["licenses"] == [{"name": "other"}]
    assert not (output / "dataset/model.safetensors").is_symlink()
    input_manifest = json.loads((output / "dataset/input-manifest.json").read_text())
    assert set(input_manifest["files"]) == {
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "train.jsonl",
        "data_manifest.json",
        "lr_selection.json",
    }
    for name, item in input_manifest["files"].items():
        assert sha256_file(output / "dataset" / name) == item["sha256"]
    generated = (output / "kernel/run.py").read_text()
    ast.parse(generated)
    loaded = runpy.run_path(str(output / "kernel/run.py"), run_name="bundle_test")
    assert loaded["SESSION"]["commit"] == "c" * 40
    assert '"commit": "' + "c" * 40 + '"' in generated
    assert "Qwen/Qwen3" not in generated
    with pytest.raises(FileExistsError):
        prepare_bundle(
            config_path=fixture["config_path"],
            plan_path=fixture["plan_path"],
            ledger_path=fixture["ledger_path"],
            model_dir=fixture["model"],
            train_path=fixture["train"],
            data_manifest_path=fixture["data_manifest"],
            selection_path=fixture["selection"],
            output=output,
            dataset_id="shlokbhakta/test-one-line-inputs",
            kernel_id="shlokbhakta/test-one-line-main-s01",
            commit="c" * 40,
            quota=fixture["quota"],
            session_seconds=3600,
        )


def test_pulled_checkpoint_requires_hash_and_cursor_match(tmp_path: Path) -> None:
    training = tmp_path / "pulled/training"
    training.mkdir(parents=True)
    fingerprint = "f" * 64
    _write_json(training / "run_manifest.json", {"fingerprint": fingerprint})
    checkpoint = training / "resume-step-000001.pt"
    checkpoint.write_bytes(b"complete resumable state")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cursor = {"attempted_updates": 1, "next_example_index": 32}
    _write_json(
        training / "resume-step-000001.pt.complete.json",
        {
            "sha256": digest,
            "fingerprint": fingerprint,
        },
    )
    _write_json(
        training / "latest.json",
        {
            "checkpoint": "/kaggle/working/one_line_r1/training/resume-step-000001.pt",
            "sha256": digest,
            "cursor": cursor,
        },
    )
    _write_json(
        training / "run_result.json",
        {
            "fingerprint": fingerprint,
            "cursor": cursor,
            "status": "deadline_stop",
        },
    )
    verified = verify_pulled_output(tmp_path / "pulled", expected_fingerprint=fingerprint)
    assert verified["checkpoint_sha256"] == digest
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        verify_pulled_output(tmp_path / "pulled", expected_fingerprint=fingerprint)
