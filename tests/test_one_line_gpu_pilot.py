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


def _license_mixed_plan(policy) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, config = _valid_plan()
    plan.update(schema=policy.plan_schema, branch=policy.branch)
    plan["data"].update(
        schema=policy.data_schema,
        dataset_id=policy.dataset_id,
        dataset_license=policy.dataset_license,
        source_file_license_status=policy.file_license_status,
    )
    plan["training"]["disposable_fixture"] = {
        "schema": "single-line-disposable-training-fixture-v2",
        "sha256": "4" * 64,
        "examples": 64,
        "epochs": 1,
        "peak_learning_rate": 1e-4,
        "nonpadding_training_input_tokens": 10_000,
        "supervised_response_and_eos_tokens": 400,
        "quality_evidence": False,
        "effective_batch_examples": 2,
        "microbatch_examples": 2,
        "expected_updates": 32,
        "decode_examples_per_action": 4,
        "minimum_exact_actions_per_action": 3,
        "eos_required": True,
        "implementation_viability_schema": (
            "single-line-disposable-implementation-viability-v1"
        ),
        "initial_loss_scale": 128.0,
    }
    plan["budgets"].update(
        prior_training_input_tokens=538_275,
        prior_session_wall_seconds=14_400,
    )
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


def test_constructive_plan_enforces_prior_campaign_exposure() -> None:
    plan, config = _valid_plan()
    policy = builder.CONSTRUCTIVE
    plan.update(schema=policy.plan_schema, branch=policy.branch)
    plan["data"].update(
        schema=policy.data_schema,
        dataset_id=policy.dataset_id,
        dataset_license=policy.dataset_license,
        source_file_license_status=policy.file_license_status,
    )
    plan["training"]["disposable_fixture"] = {
        "schema": "single-line-disposable-training-fixture-v1",
        "sha256": "4" * 64,
        "examples": 64,
        "epochs": 1,
        "peak_learning_rate": 1e-4,
        "nonpadding_training_input_tokens": 10_000,
        "supervised_response_and_eos_tokens": 400,
        "quality_evidence": False,
    }
    with pytest.raises(ValueError, match="aggregate campaign"):
        builder._validate_plan(plan, config)
    plan["budgets"].update(
        prior_training_input_tokens=538_275,
        prior_session_wall_seconds=14_400,
    )
    builder._validate_plan(plan, config)
    for key, value in (
        ("prior_training_input_tokens", 99_000_001),
        ("prior_session_wall_seconds", 24 * 3600),
        ("prior_training_input_tokens", True),
    ):
        invalid = json.loads(json.dumps(plan))
        invalid["budgets"][key] = value
        with pytest.raises(ValueError, match="aggregate campaign"):
            builder._validate_plan(invalid, config)


def test_license_mixed_v1_and_history_v2_plan_dispatch_keep_fixed_gates() -> None:
    legacy, config = _license_mixed_plan(builder.LICENSE_MIXED)
    legacy["training"]["disposable_fixture"].pop("initial_loss_scale")
    builder._validate_plan(legacy, config)

    history, config = _license_mixed_plan(builder.LICENSE_MIXED_HISTORY)
    builder._validate_plan(history, config)
    invalid_scale = json.loads(json.dumps(history))
    invalid_scale["training"]["disposable_fixture"]["initial_loss_scale"] = 256.0
    with pytest.raises(ValueError, match="128 initial loss scale"):
        builder._validate_plan(invalid_scale, config)
    invalid_fixture = json.loads(json.dumps(history))
    invalid_fixture["training"]["disposable_fixture"]["expected_updates"] = 31
    with pytest.raises(ValueError, match="fixture-v2 viability"):
        builder._validate_plan(invalid_fixture, config)


def test_mixed_proof_inventory_requires_frozen_safe_files(tmp_path: Path) -> None:
    review = tmp_path / "artifacts" / "review.json"
    review.parent.mkdir()
    review.write_bytes(b'{"schema":"review"}\n')
    entry = {
        "path": "artifacts/review.json",
        "bytes": review.stat().st_size,
        "sha256": _sha(review),
    }
    plan = {"data": {"proof_files": [entry]}}
    manifest = {
        "independent_review_path": entry["path"],
        "independent_review_sha256": entry["sha256"],
        "independent_review_bytes": entry["bytes"],
    }
    assert builder._mixed_proof_files(plan, manifest, tmp_path) == {entry["path"]: review}

    for bad_entry in (
        {**entry, "path": "../review.json"},
        {**entry, "sha256": "0" * 64},
        {**entry, "extra": True},
    ):
        with pytest.raises(ValueError, match="mixed-license proof"):
            builder._mixed_proof_files(
                {"data": {"proof_files": [bad_entry]}}, manifest, tmp_path
            )


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


def test_worker_fixture_requires_actual_updates_and_counts_both_phases(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    fixture_file = dataset / "training-fixture.jsonl"
    fixture_file.write_text("synthetic test payload\n")
    session = {
        "session_seconds": 7200,
        "reserve_seconds": 1200,
        "plan_sha256": "a" * 64,
        "prior_training_input_tokens": 538_275,
        "disposable_fixture": {
            "sha256": _sha(fixture_file),
            "nonpadding_training_input_tokens": 10_000,
            "supervised_response_and_eos_tokens": 400,
        },
    }
    worker = _worker_module(tmp_path, session)
    worker.OUT = tmp_path / "outputs"
    worker.OUT.mkdir()
    checkpoint = tmp_path / "fixture-checkpoint"
    checkpoint.write_bytes(b"synthetic checkpoint")
    commands = []
    monkeypatch.setattr(worker, "stage", lambda command, *a, **kw: commands.append(command))
    result = {
        "status": "complete",
        "examples": 64,
        "cursor": {
            "completed_updates": 2,
            "skipped_updates": 0,
            "training_input_tokens": 10_000,
            "supervised_target_tokens": 400,
        },
        "disposable_fixture": {
            "response_and_eos_positions_supervised": True,
            "changed_parameter_elements": 3,
            "greedy_generation_observations": [
                {"gold_action": kind, "terminated_by_eos": False}
                for kind in ("keep", "replace_line", "insert_before", "delete_line")
            ],
        },
    }
    monkeypatch.setattr(worker, "_verify_training_output", lambda *a, **kw: (result, checkpoint))
    assert worker._run_disposable_training_fixture(dataset, {}) == 10_000
    assert commands[0][commands[0].index("--phase") + 1] == "fixture"
    assert commands[0][commands[0].index("--model") + 1] == str(dataset)
    evidence = json.loads((worker.OUT / "disposable-fixture-verification.json").read_text())
    assert evidence["quality_evidence"] is False
    result["cursor"]["skipped_updates"] = 1
    with pytest.raises(ValueError, match="prove the declared updates"):
        worker._run_disposable_training_fixture(dataset, {})
    result["cursor"]["skipped_updates"] = 0
    result["disposable_fixture"]["changed_parameter_elements"] = 0
    with pytest.raises(ValueError, match="prove the declared updates"):
        worker._run_disposable_training_fixture(dataset, {})


def test_python311_bootstrap_preserves_deadline_and_uses_isolated_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker_module(tmp_path, {"session_seconds": 7200, "reserve_seconds": 1200})
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(
        worker,
        "stage",
        lambda command, label, **kwargs: calls.append((command, {"label": label, **kwargs})),
    )
    executed: dict[str, Any] = {}

    def execve(path, args, env):
        executed.update(path=path, args=args, env=env)
        raise RuntimeError("test process replacement")

    monkeypatch.setattr(worker.os, "execve", execve)
    with pytest.raises(RuntimeError, match="test process replacement"):
        worker._prepare_python311()
    assert len(calls) == 3
    assert "uv==0.12.3" in calls[0][0]
    assert "3.11.15" in calls[1][0]
    assert "torch==2.10.0" in calls[2][0]
    assert calls[2][1]["timeout"] == 1200
    assert executed["env"]["TABCOMPLETE_PILOT_STARTED_MONOTONIC"] == str(worker.STARTED)
    assert executed["env"]["TABCOMPLETE_PILOT_PYTHON311_READY"] == "1"
    assert executed["path"] == "/kaggle/temp/tabcomplete-python311/venv/bin/python"
    assert calls[1][1]["env"]["UV_CACHE_DIR"].startswith("/kaggle/temp/")


@pytest.mark.parametrize(
    "schema", ["one-line-constructive-pilot-v1", builder.PUBLIC_SYNTHETIC.data_schema]
)
def test_constructive_runtime_rejects_wrong_python_before_cuda_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: str
) -> None:
    worker = _worker_module(
        tmp_path,
        {
            "session_seconds": 7200,
            "reserve_seconds": 1200,
            "data_schema": schema,
        },
    )
    monkeypatch.setattr(worker.sys, "version_info", (3, 12, 0))
    with pytest.raises(RuntimeError, match="Python 3.11"):
        worker._check_t4_and_logits_support()


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
            "peak_learning_rate": 1e-5,
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
        "peak_learning_rate": 1e-5,
    }
    return session, source


def _write_mixed_worker_fixture(
    tmp_path: Path, policy
) -> tuple[dict[str, Any], Path, str]:
    session, source = _write_worker_fixture(tmp_path)
    proof_name = "artifacts/role-proof.json"
    proof_path = source / proof_name
    proof_path.parent.mkdir()
    proof_path.write_bytes(b'{"proof":"synthetic"}\n')
    fixture_path = source / "training-fixture.jsonl"
    fixture_path.write_bytes(b"synthetic fixture input\n")

    data_path = source / "data_manifest.json"
    data_manifest = json.loads(data_path.read_text())
    data_manifest.update(
        schema=policy.data_schema,
        dataset_id=policy.dataset_id,
        dataset_license=policy.dataset_license,
        source_file_license_status=policy.file_license_status,
        artifact_root=".",
    )
    data_path.write_text(json.dumps(data_manifest, sort_keys=True) + "\n")
    proof_identity = {
        "bytes": proof_path.stat().st_size,
        "sha256": _sha(proof_path),
    }
    fixture = {
        "schema": "single-line-disposable-training-fixture-v2",
        "sha256": _sha(fixture_path),
        "examples": 64,
        "epochs": 1,
        "peak_learning_rate": 1e-4,
        "nonpadding_training_input_tokens": 10_000,
        "supervised_response_and_eos_tokens": 400,
        "quality_evidence": False,
        "effective_batch_examples": 2,
        "microbatch_examples": 2,
        "expected_updates": 32,
        "decode_examples_per_action": 4,
        "minimum_exact_actions_per_action": 3,
        "eos_required": True,
        "implementation_viability_schema": (
            "single-line-disposable-implementation-viability-v1"
        ),
        "initial_loss_scale": 128.0,
    }
    plan_path = source / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan.update(schema=policy.plan_schema, branch=policy.branch)
    plan["data"] = {
        **data_manifest,
        "manifest_sha256": _sha(data_path),
        "proof_files": [
            {"path": proof_name, **proof_identity},
        ],
    }
    plan["training"].update(disposable_fixture=fixture)
    plan["budgets"].update(
        prior_training_input_tokens=538_275,
        prior_session_wall_seconds=14_400,
    )
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")

    input_path = source / "input-manifest.json"
    input_manifest = json.loads(input_path.read_text())
    file_records = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in source.iterdir()
        if path.is_file() and path.name != "input-manifest.json"
    }
    file_records[proof_name] = proof_identity
    input_manifest.update(
        plan_sha256=_sha(plan_path),
        branch=policy.branch,
        files=file_records,
    )
    input_path.write_text(json.dumps(input_manifest, sort_keys=True) + "\n")

    session.update(
        data_schema=policy.data_schema,
        plan_schema=policy.plan_schema,
        source_dataset_id=policy.dataset_id,
        branch=policy.branch,
        input_manifest_sha256=_sha(input_path),
        plan_sha256=_sha(plan_path),
        manifest_sha256=_sha(data_path),
        dataset_license=policy.dataset_license,
        source_file_license_status=policy.file_license_status,
        prior_training_input_tokens=538_275,
        prior_session_wall_seconds=14_400,
        disposable_fixture=fixture,
        proof_files={proof_name: proof_identity},
    )
    return session, source, proof_name


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


@pytest.mark.parametrize("policy", [builder.LICENSE_MIXED, builder.LICENSE_MIXED_HISTORY])
def test_worker_dispatches_mixed_policy_and_rejects_bad_proof_inventory(
    tmp_path: Path, policy
) -> None:
    session, source, proof_name = _write_mixed_worker_fixture(tmp_path, policy)
    module = _worker_module(tmp_path, session)
    module.INPUT_ROOT = source.parent
    found, manifest = module._safe_input_manifest(source.parent)
    assert found == source
    assert manifest["files"][proof_name]["sha256"] == session["proof_files"][proof_name]["sha256"]

    module.SESSION["proof_files"][proof_name]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="proof inventory is unsafe or inconsistent"):
        module._safe_input_manifest(source.parent)

    module.SESSION["proof_files"] = {
        "artifacts/unapproved.gguf": session["proof_files"][proof_name]
    }
    with pytest.raises(ValueError, match="proof inventory is unsafe or inconsistent"):
        module._safe_input_manifest(source.parent)


@pytest.mark.parametrize("policy", [builder.LICENSE_MIXED, builder.LICENSE_MIXED_HISTORY])
def test_worker_review_dispatch_passes_the_selected_license_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy
) -> None:
    import types

    import tinycomplete.one_line.pilot_data as pilot_data

    data = tmp_path / "input"
    data.mkdir()
    (data / "data_manifest.json").write_text("{}\n")
    row = {"id": "reviewed-row", "split": "train"}
    for name in ("train.jsonl", "development.jsonl"):
        (data / name).write_text(json.dumps(row) + "\n")
    worker = _worker_module(
        tmp_path,
        {"session_seconds": 7200, "reserve_seconds": 1200, "data_schema": policy.data_schema},
    )
    worker.REPO = tmp_path / "frozen-repo"
    observed: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        pilot_data,
        "validate_license_mixed_manifest",
        lambda _manifest, *, policy: observed.append(("manifest", policy)),
    )
    monkeypatch.setattr(pilot_data, "validate_license_mixed_splits", lambda _rows: None)
    monkeypatch.setattr(
        pilot_data,
        "validate_license_mixed_review",
        lambda _manifest, _rows, *, tokenizer, package_root, policy: observed.append(
            ("review", policy)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: object()
            )
        ),
    )

    worker._verify_reviewed_inputs(data)
    assert observed == [("manifest", policy), ("review", policy)]


def test_worker_evaluation_uses_the_export_tokenizer_hash(tmp_path: Path) -> None:
    session, _source = _write_worker_fixture(tmp_path)
    module = _worker_module(tmp_path, session)
    evaluation = tmp_path / "adapted-evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "identity": {
                    "model_weight_sha256": "export-weight",
                    "source_weight_sha256": session["model_weight_sha256"],
                    "development_sha256": session["development_sha256"],
                    "tokenizer_sha256": "export-tokenizer",
                },
                "cases": session["development_count"],
                "synthetic_calibration": {},
            }
        ),
        encoding="utf-8",
    )
    module._verify_evaluation(
        evaluation, model_sha="export-weight", tokenizer_sha="export-tokenizer"
    )
    with pytest.raises(ValueError, match="evaluation identity"):
        module._verify_evaluation(
            evaluation, model_sha="export-weight", tokenizer_sha=session["tokenizer_sha256"]
        )


def test_worker_template_is_syntax_valid_and_trainer_owns_output_creation(tmp_path: Path) -> None:
    session, _source = _write_worker_fixture(tmp_path)
    module = _worker_module(tmp_path, session)
    assert module.SESSION["model_id"] == builder.MODEL_ID
    assert module.SESSION["development_count"] == 64
    text = (PILOT / "run.py").read_text(encoding="utf-8")
    assert '"HF_HUB_OFFLINE": "1"' in text
    assert '"tree-sitter-language-pack==1.20.0"' in text
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


def test_dataset_slug_is_checked_before_private_upload() -> None:
    builder.validate_kaggle_refs(
        "shlokbhakta/tc-oline-instinct-r1-retry-inputs", "shlokbhakta/tc-oline-instinct-r1-retry"
    )
    with pytest.raises(ValueError, match="between 6 and 50"):
        builder.validate_kaggle_refs(
            "shlokbhakta/tabcomplete-one-line-instinct-pilot-r1-retry-inputs",
            "shlokbhakta/tc-oline-instinct-r1-retry",
        )


@pytest.mark.parametrize('lr', [1e-5, 3e-5])
def test_public_functional_worker_verifies_the_selected_pushed_plan_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lr: float
) -> None:
    import types

    suffix = '1e5' if lr == 1e-5 else '3e5'
    worker = _worker_module(tmp_path, {
        'session_seconds': 7200, 'reserve_seconds': 1200,
        'data_schema': builder.PUBLIC_SYNTHETIC.data_schema,
        'branch': 'prototype/product-r2', 'commit': 'a' * 40,
        'base_commit': 'b' * 40, 'peak_learning_rate': lr,
        'plan_sha256': hashlib.sha256(b'correct plan').hexdigest(),
        'config_sha256': hashlib.sha256(b'correct config').hexdigest(),
    })
    worker.REPO = tmp_path / 'cloned'

    def clone(command, *_args, **_kwargs):
        if command[:2] == ['git', 'clone']:
            plan = worker.REPO / (
                f'reports/research/public_synthetic_quality_pilot_r1/plan_lr{suffix}.json'
            )
            plan.parent.mkdir(parents=True)
            plan.write_bytes(b'correct plan')
            config = worker.REPO / 'configs/research/public_synthetic_quality_pilot_r1.yaml'
            config.parent.mkdir(parents=True)
            config.write_bytes(b'correct config')

    monkeypatch.setattr(worker, 'stage', clone)
    monkeypatch.setattr(worker, '_run', lambda *a, **kw: types.SimpleNamespace(
        returncode=0, stdout='a' * 40
    ))
    worker._clone_frozen_commit()


def test_public_functional_worker_reopens_portable_proofs_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tinycomplete.one_line.public_synthetic_pilot as pilot

    data = tmp_path / 'input'
    data.mkdir()
    (data / 'data_manifest.json').write_text('{"id":"portable"}')
    (data / 'train.jsonl').write_text('{"id":"train"}\n')
    (data / 'development.jsonl').write_text('{"id":"dev"}\n')
    worker = _worker_module(tmp_path, {
        'session_seconds': 7200, 'reserve_seconds': 1200,
        'data_schema': builder.PUBLIC_SYNTHETIC.data_schema,
    })
    observed = []
    monkeypatch.setattr(pilot, 'validate_manifest', lambda manifest, rows, *, package_root:
                        observed.append((manifest, rows, package_root)))
    worker._verify_reviewed_inputs(data)
    assert observed == [({'id': 'portable'}, [{'id': 'train'}, {'id': 'dev'}], data)]


def test_remote_input_inventory_checks_all_pages_and_rejects_skipped_proofs(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []
    pages = iter([
        'Next Page Token = next-page\n' + json.dumps([{'name': 'model.safetensors', 'size': 10}]),
        json.dumps([{'name': 'proofs/source.json', 'size': 20}]),
    ])
    monkeypatch.setattr(builder, '_run', lambda command, **kw:
                        (commands.append(command), next(pages))[1])
    expected = {'model.safetensors': {'bytes': 10}, 'proofs/source.json': {'bytes': 20}}
    result = builder.verify_remote_inputs('owner/private-inputs', expected)
    assert result['verified_files'] == 2
    assert commands[1][-2:] == ['--page-token', 'next-page']
    monkeypatch.setattr(builder, '_run', lambda *a, **kw:
                        json.dumps([{'name': 'model.safetensors', 'size': 10}]))
    with pytest.raises(ValueError, match='missing files'):
        builder.verify_remote_inputs('owner/private-inputs', expected)


def test_public_functional_main_bootstraps_python311_before_setup_or_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, dataset = _write_worker_fixture(tmp_path)
    session['data_schema'] = builder.PUBLIC_SYNTHETIC.data_schema
    worker = _worker_module(tmp_path, session)
    worker.OUT = tmp_path / 'outputs'
    monkeypatch.delenv('TABCOMPLETE_PILOT_PYTHON311_READY', raising=False)
    monkeypatch.setattr(worker, '_safe_input_manifest', lambda _root: (dataset, {'files': {}}))
    monkeypatch.setattr(worker, '_verify_model', lambda _path: None)
    calls = []

    def bootstrap():
        calls.append('python311')
        raise RuntimeError('bounded test interruption')

    monkeypatch.setattr(worker, '_prepare_python311', bootstrap)
    monkeypatch.setattr(worker, 'stage', lambda *a, **kw: calls.append('unexpected setup'))
    assert worker.main() == 1
    assert calls == ['python311']
    assert json.loads((worker.OUT / 'worker-status.json').read_text())['state'] == 'failed'
