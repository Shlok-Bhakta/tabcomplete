from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "kaggle/one_line_gpu_pilot_r1"
sys.path.insert(0, str(PILOT))

import build_pilot as builder  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    config = {
        "student": {
            "model_id": builder.MODEL_ID,
            "revision": builder.MODEL_REVISION,
            "weight_sha256": builder.WEIGHT_SHA256,
            "tokenizer_sha256": builder.TOKENIZER_SHA256,
            "config_sha256": builder.MODEL_CONFIG_SHA256,
            "initializer": "untouched_pretrained",
        }
    }
    plan: dict[str, Any] = {
        "schema": builder.PLAN_SCHEMA,
        "suite_revision": 3,
        "branch": "research/one-line-gpu-pilot-r1",
        "base_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "student": {
            "model_id": builder.MODEL_ID,
            "revision": builder.MODEL_REVISION,
            "weight_sha256": builder.WEIGHT_SHA256,
            "tokenizer_sha256": builder.TOKENIZER_SHA256,
            "config_sha256": builder.MODEL_CONFIG_SHA256,
        },
        "data": {
            "schema": builder.DATA_SCHEMA,
            "dataset_id": "continuedev/instinct-data",
            "dataset_revision": "c" * 40,
            "dataset_license": "Apache-2.0",
            "source_file_license_status": "unverified",
            "train_sha256": "1" * 64,
            "development_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
            "train_count": 128,
            "dev_count": 64,
            "file_groups_disjoint": True,
        },
        "training": {
            "phase": "pilot",
            "epochs": 1,
            "peak_learning_rate": 1e-5,
            "planned_nonpadding_input_tokens": 1_000_000,
            "max_nonpadding_input_tokens": 2_000_000,
        },
        "budgets": {
            "max_session_seconds": builder.SESSION_SECONDS,
            "reserve_seconds": builder.RESERVE_SECONDS,
            "max_new_storage_bytes": builder.MAX_NEW_STORAGE_BYTES,
            "quota_gpu_hours_multiplier": builder.QUOTA_GPU_HOURS_MULTIPLIER,
            "no_automatic_renewal": True,
        },
        "quota_at_freeze": {
            "observed_at": "2026-09-29T05:51:33Z",
            "remaining": 45.0,
            "renewal": "2026-10-03T00:00:00Z",
            "units": "Kaggle account GPU-hours",
            "source": "authenticated Kaggle quota and statuses",
            "active_jobs": [],
            "job_statuses": [],
        },
    }
    return plan, config


def test_frozen_plan_requires_exact_untouched_model_and_pilot_budget() -> None:
    plan, config = _valid_plan()
    builder._validate_plan(plan, config)

    bad_model = json.loads(json.dumps(plan))
    bad_model["student"]["model_id"] = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    with pytest.raises(ValueError, match="untouched q25"):
        builder._validate_plan(bad_model, config)

    too_many_tokens = json.loads(json.dumps(plan))
    too_many_tokens["training"]["planned_nonpadding_input_tokens"] = 2_000_001
    with pytest.raises(ValueError, match="one-pass/token"):
        builder._validate_plan(too_many_tokens, config)

    licensed_claim = json.loads(json.dumps(plan))
    licensed_claim["data"]["source_file_license_status"] = "verified"
    with pytest.raises(ValueError, match="source, split"):
        builder._validate_plan(licensed_claim, config)


def test_quota_gate_requires_live_remaining_and_no_active_job() -> None:
    plan, _ = _valid_plan()
    good = {
        "remaining": 4.5,
        "renewal": plan["quota_at_freeze"]["renewal"],
        "active_jobs": [],
        "job_statuses": [],
    }
    builder._check_live_quota(plan, good)
    with pytest.raises(RuntimeError, match="active"):
        builder._check_live_quota(plan, {**good, "active_jobs": [{"status": "RUNNING"}]})
    with pytest.raises(RuntimeError, match="renewal changed"):
        builder._check_live_quota(plan, {**good, "renewal": "changed"})
    with pytest.raises(RuntimeError, match="insufficient"):
        builder._check_live_quota(plan, {**good, "remaining": 3.9})
    with pytest.raises(RuntimeError, match="active-job status"):
        builder._check_live_quota(plan, {**good, "job_statuses": [{"status": "unknown"}]})
    with pytest.raises(RuntimeError, match="could not be verified"):
        builder._check_live_quota(plan, {**good, "remaining": None})


def test_deleted_kernel_is_accepted_only_with_explicit_authenticated_404(monkeypatch) -> None:
    quota = {
        "job_statuses": [
            {"reference": "owner/deleted-kernel", "status": "unknown"},
        ],
        "active_jobs": [],
    }

    class Response:
        returncode = 1
        stdout = ""
        stderr = "HTTP 404 Not Found"

    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: Response())
    builder._resolve_unknown_kernel_statuses(quota)
    row = quota["job_statuses"][0]
    assert row["not_found_verified"] is True
    assert row["http_status"] == 404
    assert builder._job_statuses_verified(quota)

    quota["job_statuses"][0]["http_status"] = 403
    assert not builder._job_statuses_verified(quota)


def _worker_module(tmp_path: Path, session: dict[str, Any]):
    source = (PILOT / "run.py").read_text(encoding="utf-8")
    rendered = source.replace('"__SESSION_LITERAL__"', repr(json.dumps(session, sort_keys=True)))
    compile(rendered, "kaggle-pilot-run.py", "exec")
    script = tmp_path / "generated_run.py"
    script.write_text(rendered, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"pilot_worker_{id(session)}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_worker_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    source = tmp_path / "input" / "bundle"
    source.mkdir(parents=True)
    payloads = {
        "model.safetensors": b"approved-test-weights",
        "config.json": b'{"model_type":"qwen2"}\n',
        "tokenizer.json": b"approved-test-tokenizer",
        "train.jsonl": b'{"split":"train"}\n',
        "development.jsonl": b'{"split":"development"}\n',
        "config.yaml": b"student: {}\n",
    }
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)
    train_hash = _sha(source / "train.jsonl")
    dev_hash = _sha(source / "development.jsonl")
    data_manifest = {
        "schema": builder.DATA_SCHEMA,
        "dataset_id": "continuedev/instinct-data",
        "dataset_revision": "c" * 40,
        "dataset_license": "Apache-2.0",
        "source_file_license_status": "unverified",
        "train_sha256": train_hash,
        "development_sha256": dev_hash,
        "train_count": 128,
        "dev_count": 64,
        "file_groups_disjoint": True,
    }
    data_path = source / "data_manifest.json"
    data_path.write_text(json.dumps(data_manifest, sort_keys=True) + "\n")
    plan: dict[str, Any] = {
        "schema": builder.PLAN_SCHEMA,
        "suite_revision": 3,
        "branch": "research/one-line-gpu-pilot-r1",
        "base_commit": "a" * 40,
        "config_sha256": _sha(source / "config.yaml"),
        "student": {
            "model_id": builder.MODEL_ID,
            "revision": builder.MODEL_REVISION,
            "weight_sha256": _sha(source / "model.safetensors"),
            "tokenizer_sha256": _sha(source / "tokenizer.json"),
            "config_sha256": _sha(source / "config.json"),
        },
        "data": {
            **data_manifest,
            "manifest_sha256": _sha(data_path),
        },
        "training": {
            "phase": "pilot",
            "epochs": 1,
            "max_nonpadding_input_tokens": 2_000_000,
            "planned_nonpadding_input_tokens": 1_000_000,
        },
        "budgets": {
            "max_session_seconds": builder.SESSION_SECONDS,
            "reserve_seconds": builder.RESERVE_SECONDS,
            "max_new_storage_bytes": builder.MAX_NEW_STORAGE_BYTES,
            "quota_gpu_hours_multiplier": builder.QUOTA_GPU_HOURS_MULTIPLIER,
            "no_automatic_renewal": True,
        },
    }
    plan_path = source / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    filenames = payloads.keys() | {"data_manifest.json", "plan.json"}
    file_records = {
        name: {"bytes": (source / name).stat().st_size, "sha256": _sha(source / name)}
        for name in filenames
    }
    input_manifest = {
        "schema": builder.INPUT_SCHEMA,
        "plan_sha256": _sha(plan_path),
        "branch": plan["branch"],
        "commit": "d" * 40,
        "files": file_records,
    }
    manifest_path = source / "input-manifest.json"
    manifest_path.write_text(json.dumps(input_manifest, sort_keys=True) + "\n")
    session = {
        "input_manifest_sha256": _sha(manifest_path),
        "plan_sha256": _sha(plan_path),
        "branch": plan["branch"],
        "commit": input_manifest["commit"],
        "base_commit": plan["base_commit"],
        "train_sha256": train_hash,
        "development_sha256": dev_hash,
        "manifest_sha256": _sha(data_path),
        "config_sha256": _sha(source / "config.yaml"),
        "model_weight_sha256": plan["student"]["weight_sha256"],
        "tokenizer_sha256": plan["student"]["tokenizer_sha256"],
        "model_config_sha256": plan["student"]["config_sha256"],
        "model_id": builder.MODEL_ID,
        "model_revision": builder.MODEL_REVISION,
        "dataset_revision": data_manifest["dataset_revision"],
        "dataset_license": "Apache-2.0",
        "train_count": 128,
        "development_count": 64,
        "session_seconds": builder.SESSION_SECONDS,
        "reserve_seconds": builder.RESERVE_SECONDS,
    }
    return session, source


def test_worker_verifies_frozen_input_bundle_and_rejects_extra_weights(tmp_path: Path) -> None:
    session, source = _write_worker_fixture(tmp_path)
    module = _worker_module(tmp_path, session)
    module.INPUT_ROOT = source.parent
    found, manifest = module._safe_input_manifest(source.parent)
    assert found == source
    assert manifest["commit"] == session["commit"]

    (source / "other-model.gguf").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="unapproved model"):
        module._safe_input_manifest(source.parent)


def test_worker_template_is_syntax_valid_and_trainer_owns_output_creation(tmp_path: Path) -> None:
    session, _source = _write_worker_fixture(tmp_path)
    module = _worker_module(tmp_path, session)
    assert module.SESSION["model_id"] == builder.MODEL_ID
    assert module.SESSION["development_count"] == 64
    text = (PILOT / "run.py").read_text(encoding="utf-8")
    assert '"HF_HUB_OFFLINE": "1"' in text
    assert '"--phase",\n            "pilot"' in text
    assert '"HF_HUB_OFFLINE"' in text
    assert "TRAIN_OUT.mkdir(parents=True, exist_ok=False)" not in text
    assert "logits_to_keep=positions" in text


def test_kaggle_submission_cli_flags_match_installed_cli() -> None:
    import subprocess

    datasets = subprocess.run(
        ["kaggle", "datasets", "create", "--help"], capture_output=True, text=True, check=True
    ).stdout
    kernels = subprocess.run(
        ["kaggle", "kernels", "push", "--help"], capture_output=True, text=True, check=True
    ).stdout
    # CLI help names the older plural file, but the installed create command
    # rejected it and explicitly requires the singular filename.
    assert "datasets-metadata.json" in datasets
    assert 'dataset_dir / "dataset-metadata.json"' in (PILOT / "build_pilot.py").read_text()
    assert "-u, --public" in datasets and "default is private" in datasets
    assert "--accelerator ACC" in kernels
    assert "NvidiaTeslaT4" not in kernels  # values are validated by the server, not local choices
    assert "-t TIMEOUT" in kernels
