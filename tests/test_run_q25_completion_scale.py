from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_q25_completion_scale as scale  # noqa: E402


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode() if isinstance(value, str) else value)


def _prior_charge_fixture() -> dict[str, Any]:
    return {
        "revision": 2,
        "session_wall_seconds_upper_bound": 120,
        "conservative_account_gpu_hours": 120 / 3600 * 2,
        "processed_input_tokens_conservative": 0,
        "maximum_additional_repeat_allocations": 1,
        "automatic_retry": False,
    }


def _freeze_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repository = tmp_path / "repo"
    report = repository / "reports/research/q25_completion_scale_r1"
    artifacts = tmp_path / "artifacts"
    corpus = artifacts / "corpus"
    base_report = repository / "reports/research/q25_code_cpt_r2"
    training_plans = {
        "repeat": report / "training_plan_repeat-r3.json",
        "scaled": report / "training_plan_scaled-r3.json",
    }
    for name in ("repeat_train", "scaled_train", "development_new", "development_previous"):
        _write(corpus / f"{name}.jsonl", f"fixture:{name}\n")

    model = {
        "id": "Qwen/Qwen2.5-Coder-0.5B",
        "initializer": "untouched_pretrained",
        "license": "Apache-2.0",
        "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
        "tokenizer_sha256": "c" * 64,
        "weight_sha256": "a" * 64,
    }
    split_counts = {
        "repeat_train": (4096, 1000, 120),
        "scaled_train": (8192, 2000, 240),
        "development_new": (512, 400, 80),
        "development_previous": (240, 300, 50),
    }
    records: dict[str, dict[str, Any]] = {}
    splits: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}
    for split, (rows, input_tokens, target_tokens) in split_counts.items():
        filename = f"{split}.jsonl"
        path = corpus / filename
        data = path.read_bytes()
        record = {
            "file": filename,
            "sha256": _sha_bytes(data),
            "bytes": len(data),
            "row_count": rows,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
        }
        records[split] = record
        splits[split] = record
        files[filename] = {"sha256": record["sha256"], "bytes": record["bytes"]}

    previous_train_sha = records["repeat_train"]["sha256"]
    previous_dev_sha = records["development_previous"]["sha256"]
    experiment = {
        "repeat": {"distinct_states": 4096, "epochs": 2, "example_exposures": 8192},
        "scaled": {"distinct_states": 8192, "epochs": 1, "example_exposures": 8192},
        "matched_batch_schedule": "frozen two-stage ordered batches",
        "limitations": ["synthetic completion is not human intent"],
    }
    preparation_path = report / "preparation_plan-r2.json"
    runtime_lock_path = base_report / "fim_runtime_lock.json"
    runtime_requirements_path = base_report / "fim_runtime_requirements.lock"
    analysis_path = report / "analysis_plan-r2.json"
    _write(runtime_lock_path, json.dumps({"schema": "runtime-lock-test"}))
    _write(runtime_requirements_path, "sha256 pinned runtime requirements\n")
    _write(
        analysis_path,
        json.dumps({"schema": "q25-completion-scale-analysis-v1", "plan_revision": 2}),
    )
    prep = {
        "schema": "q25-completion-scale-preparation-plan-v1",
        "plan_revision": 2,
        "budget": {"gpu_allocation_authorized_by_this_plan": False},
        "parent_cpt_plan_sha256": "d" * 64,
        "model": model,
        "runtime_lock": {"sha256": scale.digest(runtime_lock_path)},
        "previous_training": {"sha256": previous_train_sha},
        "previous_development": {"sha256": previous_dev_sha},
        "experiment": experiment,
        "evaluation": {
            "primary": "new development fixture",
            "secondary": "historical development",
            "max_output_tokens": 96,
            "decoding": "greedy EOS only",
            "decision": "no automatic promotion",
        },
    }
    scale.save(preparation_path, prep)
    metadata = {
        "schema": "q25-completion-scale-corpus-v1",
        "preparation_plan_sha256": scale.digest(preparation_path),
        "parent_cpt_plan_sha256": prep["parent_cpt_plan_sha256"],
        "tokenizer_sha256": model["tokenizer_sha256"],
        "raw_source_content_emitted": False,
        "sealed_test_accessed": False,
        "splits": splits,
        "files": files,
    }
    scale.save(corpus / "corpus_metadata.json", metadata)

    line_path = tmp_path / "line180.jsonl"
    causal_path = repository / "data/benchmarks/code_completion_v2.jsonl"
    _write(line_path, "line fixture\n")
    _write(causal_path, "causal fixture\n")
    fixture_records = {
        "causal": {"sha256": scale.digest(causal_path), "bytes": causal_path.stat().st_size},
        "line": {"sha256": scale.digest(line_path), "bytes": line_path.stat().st_size},
    }
    lock_digest = "b" * 64
    parent = {
        "schema": "q25-fim-training-plan-v1",
        "gpu_execution_authorized": True,
        "parent_cpt_runtime_observation": {"runtime": "pinned"},
        "configuration": {
            "training": {
                "epochs": 1,
                "sequence_length": 1024,
                "effective_batch": 16,
                "microbatch_examples": 1,
                "checkpoint_every_updates": 64,
                "seed": 314159,
                "attention": "sdpa",
                "gradient_checkpointing": True,
                "master_weights": "fp32",
                "compute": "fp16",
                "optimizer": "AdamW8bit",
                "objective": "example_mean_response_only_FIM_target_and_EOS",
                "learning_rate": 1e-5,
                "warmup_fraction": 0.03,
                "cosine_floor_fraction": 0.1,
                "gradient_clip": 1.0,
                "weight_decay": 0.01,
                "initial_loss_scale": 128,
                "max_input_tokens": 4_194_304,
                "max_output_bytes": 10 * 1024**3,
            },
            "runtime": {
                "python": "3.11.15",
                "torch": "2.11.0+cu128",
                "transformers": "5.17.0",
                "bitsandbytes": "0.50.2",
                "cuda_runtime": "12.8",
            },
            "runtime_lock": {
                "runtime_lock_sha256": scale.digest(runtime_lock_path),
                "requirements_lock_sha256": lock_digest,
                "bootstrap_uv_version": "0.12.3",
                "bootstrap_uv_wheel_sha256": (
                    "1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1"
                ),
            },
        },
        "initializers": {
            scale.TRAIN_ARM: {
                "kind": "untouched_pretrained",
                "model_id": model["id"],
                "revision": model["revision"],
                "files": {"model.safetensors": {"sha256": model["weight_sha256"], "bytes": 10}},
            }
        },
        "evaluation": {
            "attention_backend": "torch-efficient-sdpa-explicit-kv-repeat-v1",
            "source_syntax": {"protocol": "q25-fim-source-syntax-v1", "selection_gate": False},
            "paired_analysis": {"primary": "old primary", "bootstrap_seed": 271828},
            "fixtures": fixture_records,
        },
    }
    base_plan_path = base_report / "fim_training_plan.json"
    scale.save(base_plan_path, parent)
    config = {
        "schema": "q25-completion-scale-config-v1",
        "branch": "research/q25-completion-scale-r1",
        "artifact_root": str(artifacts),
        "corpus_dir": str(corpus),
        "report_root": report.relative_to(repository).as_posix(),
        "preparation_plan": preparation_path.relative_to(repository).as_posix(),
        "preparation_plan_sha256": scale.digest(preparation_path),
        "model": model,
        "budget": {
            "aggregate_session_seconds": scale.AGGREGATE_SESSION_SECONDS,
            "session_seconds": scale.SESSION_SECONDS,
            "conservative_account_gpu_hours": scale.CONSERVATIVE_ACCOUNT_GPU_HOURS,
            "conservative_quota_multiplier": scale.QUOTA_MULTIPLIER,
            "maximum_campaign_input_tokens": scale.CAMPAIGN_TOKEN_CAP,
            "maximum_discarded_replay_input_tokens": scale.DISCARDED_TOKEN_CAP,
            "new_artifact_bytes_cap": scale.ARTIFACT_CAP_BYTES,
            "minimum_free_bytes": scale.MIN_FREE_BYTES,
            "finalization_reserve_seconds": scale.FINALIZATION_RESERVE_SECONDS,
            "minimum_finalization_reserve_seconds": scale.FINALIZATION_RESERVE_SECONDS,
            "runtime_setup_reserve_seconds": 1800,
            "quota_renewal": "2026-10-10T00:00:00",
            "automatic_renewal_use": False,
            "paid_compute": False,
            "gpu_allocation_authorized_by_this_plan": False,
        },
    }
    config_path = repository / "configs/research/q25_completion_scale_r1.yaml"
    _write(config_path, yaml.safe_dump(config))

    source_paths = (
        repository / "scripts/run_q25_completion_scale.py",
        repository / "scripts/analyze_q25_completion_scale.py",
        repository / "kaggle/q25_code_cpt_r2/run_fim.py",
        repository / "src/tinycomplete/code_cpt/q25_fim.py",
        repository / "src/tinycomplete/code_cpt/q25.py",
        repository / "src/tinycomplete/eval/q25_fim_attention.py",
        repository / "scripts/evaluate_q25_fim.py",
        repository / "scripts/evaluate_q25_fim_regression.py",
        repository / "scripts/prepare_q25_completion_scale.py",
        repository / "reports/research/q25_code_cpt_r2/fim_runtime_requirements.lock",
        runtime_lock_path,
        analysis_path,
    )
    for path in source_paths:
        if not path.exists():
            _write(path, "source identity fixture\n")

    monkeypatch.setattr(scale, "ROOT", repository)
    monkeypatch.setattr(scale, "REPORT", report)
    monkeypatch.setattr(scale, "ARTIFACTS", artifacts)
    monkeypatch.setattr(scale, "CONFIG", config_path)
    monkeypatch.setattr(scale, "PREPARATION_PLAN", preparation_path)
    monkeypatch.setattr(scale, "ANALYSIS_PLAN", analysis_path)
    monkeypatch.setattr(scale, "TRAINING_PLANS", training_plans)
    monkeypatch.setattr(scale, "CORPUS", corpus)
    monkeypatch.setattr(scale, "_prior_failure_charge", _prior_charge_fixture)
    monkeypatch.setattr(scale, "BASE_REPORT", base_report)
    monkeypatch.setattr(scale, "BASE_PLAN", base_plan_path)
    monkeypatch.setattr(scale, "RUNTIME_LOCK", runtime_lock_path)
    monkeypatch.setattr(scale, "WORKER", repository / "kaggle/q25_code_cpt_r2/run_fim.py")
    monkeypatch.setattr(scale, "LINE_FIXTURE", line_path)
    monkeypatch.setattr(
        scale,
        "cli",
        lambda *args, **_kwargs: "f" * 40 if args[-2:] == ("rev-parse", "HEAD") else "",
    )
    return {"plan": parent, "records": records, "metadata": metadata, "prep": prep}


def test_freeze_preserves_initializer_evaluation_contract_and_analysis_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _freeze_fixture(tmp_path, monkeypatch)

    plan = scale.freeze("repeat")

    assert plan["plan_revision"] == 3
    assert scale.TRAINING_PLANS["repeat"].name == "training_plan_repeat-r3.json"
    assert plan["experiment"]["variants"] == {
        variant: fixture["prep"]["experiment"][variant] for variant in scale.VARIANTS
    }
    assert "repeat" not in plan["experiment"]
    assert "scaled" not in plan["experiment"]
    assert plan["experiment"]["limitations"] == fixture["prep"]["experiment"]["limitations"]
    assert plan["initializers"][scale.TRAIN_ARM]["model_id"] == "Qwen/Qwen2.5-Coder-0.5B"
    assert (
        plan["initializers"][scale.TRAIN_ARM]["revision"]
        == "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
    )
    assert plan["initializers"][scale.TRAIN_ARM]["files"]["model.safetensors"]["sha256"] == "a" * 64
    assert plan["evaluation"]["attention_backend"] == "torch-efficient-sdpa-explicit-kv-repeat-v1"
    assert plan["evaluation"]["source_syntax"]["protocol"] == "q25-fim-source-syntax-v1"
    assert plan["evaluation"]["source_syntax"]["development_splits"] == [
        "development_new", "development_previous"
    ]
    assert "development_splits" not in fixture["plan"]["evaluation"]["source_syntax"]
    assert (
        plan["evaluation"]["fixtures"]["causal"]
        == fixture["plan"]["evaluation"]["fixtures"]["causal"]
    )
    assert plan["analysis_plan_sha256"] == scale.digest(scale.ANALYSIS_PLAN)
    assert (
        plan["evaluation"]["paired_analysis"]["primary"] == "development_new exact_and_terminated"
    )
    assert (
        plan["configuration"]["budget"]["aggregate_session_seconds"]
        == scale.AGGREGATE_SESSION_SECONDS
    )
    assert (
        plan["configuration"]["budget"]["conservative_account_gpu_hours"]
        == scale.CONSERVATIVE_ACCOUNT_GPU_HOURS
    )


def test_scale_quota_resolves_legacy_unknown_kernel_rows_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = {
        "job_statuses": [{"reference": "owner/old-job", "status": "unknown"}],
        "active_jobs": [],
    }
    monkeypatch.setattr(
        scale, "_campaign_module", lambda: SimpleNamespace(quota=lambda: observation)
    )

    def resolve(value: dict[str, Any]) -> None:
        value["job_statuses"][0].update(
            status="unknown",
            not_found_verified=True,
            http_status=404,
            verified_at="2026-10-04T15:00:00+00:00",
            verification_source="authenticated kaggle kernels status HTTP 404",
        )

    monkeypatch.setattr(
        scale, "_pilot_module", lambda: SimpleNamespace(_resolve_unknown_kernel_statuses=resolve)
    )

    assert scale.quota()["job_statuses"][0]["not_found_verified"] is True


def test_git_identity_accepts_report_receipts_after_frozen_commit_and_checks_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    report = repository / "reports/research/q25_completion_scale_r1"
    source = repository / "scripts/source.py"
    _write(source, "pinned source\n")
    _write(report / "training_plan_repeat.json", "frozen report\n")
    monkeypatch.setattr(scale, "ROOT", repository)
    monkeypatch.setattr(scale, "REPORT", report)
    current = "b" * 40
    monkeypatch.setattr(
        scale,
        "cli",
        lambda *args, **_kwargs: (
            "?? reports/research/q25_completion_scale_r1/training_plan_repeat.json"
            if args[-1:] == ("--porcelain",)
            else "research/q25-completion-scale-r1"
            if args[-1:] == ("--show-current",)
            else current
            if args[-2:] == ("rev-parse", "HEAD")
            else f"{current}\trefs/heads/research/q25-completion-scale-r1"
        ),
    )
    monkeypatch.setattr(
        scale.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    )
    plan = {
        "branch": "research/q25-completion-scale-r1",
        "base_commit": "a" * 40,
        "source_identity": {"files": {"scripts/source.py": scale.digest(source)}},
    }

    assert scale._git_identity(plan) == (plan["branch"], current)

    _write(source, "changed source\n")
    with pytest.raises(RuntimeError, match="source file changed"):
        scale._git_identity(plan)


def _complete_collection_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, Path]:
    report = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    plan_path = report / "training_plan_repeat.json"
    input_manifest = artifacts / "input-bundle-repeat-r3/input-manifest.json"
    _write(plan_path, "plan")
    _write(input_manifest, '{"schema":"test"}\n')
    plan = {
        "schema": scale.PLAN_SCHEMA,
        "scale_variant": "repeat",
        "base_commit": "a" * 40,
        "data": {
            "repeat_train": {"input_tokens": 100},
            "scaled_train": {"input_tokens": 100},
        },
        "initializers": {
            scale.TRAIN_ARM: {
                "kind": "untouched_pretrained",
                "model_id": "Qwen/Qwen2.5-Coder-0.5B",
                "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
                "files": {
                    name: {"sha256": "d" * 64, "bytes": 100}
                    for name in ("config.json", "model.safetensors", "tokenizer.json")
                },
            }
        },
    }
    job = {
        "status": "submitted",
        "scale_variant": "repeat",
        "arm": scale.TRAIN_ARM,
        "attempt": 1,
        "plan_sha256": scale.digest(plan_path),
        "input_manifest_sha256": scale.digest(input_manifest),
        "commit": plan["base_commit"],
        "reference": scale.KERNELS["repeat"],
        "external_campaign_tokens": 0,
        "session_seconds": 10680,
    }
    scale.save(report / "job-repeat-r3.json", job)
    output = artifacts / "output-repeat-r3/q25_completion_scale_r1-repeat"
    training = output / "training"
    checkpoint_bytes = b"checkpoint-data"
    checkpoint = training / "resume-step-000512.pt"
    _write(checkpoint, checkpoint_bytes)
    cursor = {
        "next_example_index": 8192,
        "completed_updates": 512,
        "attempted_updates": 512,
        "skipped_updates": 0,
        "training_input_tokens": 200,
        "supervised_target_tokens": 30,
        "epoch": 2,
    }
    checkpoint_sha = scale.digest(checkpoint)
    identity = {
        "schema": "q25-completion-scale-resume-v1",
        "plan_sha256": job["plan_sha256"],
        "scale_variant": "repeat",
        "arm": scale.TRAIN_ARM,
        "initializer": plan["initializers"][scale.TRAIN_ARM],
    }
    from tinycomplete.code_cpt.q25 import canonical_sha256

    fingerprint = canonical_sha256(identity)
    scale.save(training / "run_manifest.json", {"fingerprint": fingerprint, "identity": identity})
    scale.save(
        training / "latest.json",
        {
            "schema": "q25-fim-latest-checkpoint-v1",
            "path": checkpoint.name,
            "sha256": checkpoint_sha,
            "fingerprint": fingerprint,
            "cursor": cursor,
        },
    )
    scale.save(
        training / f"{checkpoint.name}.complete.json",
        {"version": 1, "sha256": checkpoint_sha, "fingerprint": fingerprint},
    )
    scale.save(
        training / "run_result.json",
        {
            "status": "complete",
            "scale_variant": "repeat",
            "cursor": cursor,
            "fingerprint": fingerprint,
            "identity": identity,
        },
    )
    _write(training / "updates.jsonl", json.dumps({"cumulative_input_tokens": 200}) + "\n")
    scale.save(
        output / "worker-status.json",
        {
            "schema": "q25-completion-scale-kaggle-worker-status-v1",
            "state": "complete",
            "commit": job["commit"],
            "attempt": 1,
            "arm": scale.TRAIN_ARM,
            "scale_variant": "repeat",
            "plan_sha256": job["plan_sha256"],
            "input_manifest_sha256": job["input_manifest_sha256"],
            "training_started": True,
            "stages": [{"name": "training", "exit_code": 0}],
        },
    )
    ledger = {
        "variants": {
            "repeat": {"state": "submitted"},
            "scaled": {"state": "reserved"},
        }
    }
    scale.save(report / "campaign_budget-r3.json", ledger)
    monkeypatch.setattr(scale, "REPORT", report)
    monkeypatch.setattr(scale, "ARTIFACTS", artifacts)
    monkeypatch.setattr(
        scale,
        "TRAINING_PLANS",
        {"repeat": plan_path, "scaled": report / "training_plan_scaled.json"},
    )
    monkeypatch.setattr(scale, "load_plan", lambda variant: plan)
    monkeypatch.setattr(scale, "_check_storage", lambda **_kwargs: None)
    monkeypatch.setattr(
        scale,
        "cli",
        lambda *args, **_kwargs: (
            "KernelWorkerStatus.COMPLETE" if args[1:3] == ("kernels", "status") else ""
        ),
    )
    return job, report, artifacts


def test_collect_persists_terminal_state_and_releases_only_completed_variant_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _job, report, _artifacts = _complete_collection_fixture(tmp_path, monkeypatch)

    verified = scale.collect("repeat")

    saved_job = scale.read_json(report / "job-repeat-r3.json")
    ledger = scale.read_json(report / "campaign_budget-r3.json")
    assert verified["training_status"] == "complete"
    assert saved_job["status"] == "collected"
    assert saved_job["terminal_status"] == "KernelWorkerStatus.COMPLETE"
    assert ledger["variants"]["repeat"]["state"] == "collected"
    assert ledger["variants"]["repeat"]["processed_input_tokens_conservative"] == 200
    assert scale._has_unresolved_job() is False


def test_collect_is_idempotent_after_report_write_before_ledger_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _job, report, _artifacts = _complete_collection_fixture(tmp_path, monkeypatch)
    result = scale.collect("repeat")
    scale.save(report / "verified-repeat-r3.json", result)
    job = scale.read_json(report / "job-repeat-r3.json")
    job["status"] = "submitted"
    scale.save(report / "job-repeat-r3.json", job)

    repeated = scale.collect("repeat")

    assert repeated["reference"] == scale.KERNELS["repeat"]
    assert scale.read_json(report / "job-repeat-r3.json")["status"] == "collected"
    assert scale._has_unresolved_job() is False


def test_frozen_plan_rechecks_configuration_and_complete_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_fixture(tmp_path, monkeypatch)
    plan = scale.freeze("repeat")
    assert scale.load_plan("repeat") == plan
    plan["source_identity"]["files"] = {}
    scale.save(scale.TRAINING_PLANS["repeat"], plan)
    with pytest.raises(ValueError, match="plan identity differs"):
        scale.load_plan("repeat")


def test_frozen_plan_rejects_configuration_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_fixture(tmp_path, monkeypatch)
    scale.freeze("repeat")
    configuration = yaml.safe_load(scale.CONFIG.read_text())
    configuration["model"]["revision"] = "0" * 40
    _write(scale.CONFIG, yaml.safe_dump(configuration))
    with pytest.raises(ValueError, match="plan identity differs"):
        scale.load_plan("repeat")


def test_uncertain_submission_remains_reserved_and_blocks_new_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_fixture(tmp_path, monkeypatch)
    plans = {variant: scale.freeze(variant) for variant in scale.VARIANTS}
    ledger = scale._load_or_create_ledger(plans)
    ledger["variants"]["repeat"]["state"] = "submission_unknown"
    scale.save(scale.REPORT / "campaign_budget-r3.json", ledger)
    scale.save(
        scale.REPORT / "job-repeat-r3.json",
        {"reference": scale.KERNELS["repeat"], "status": "submission_unknown"},
    )
    assert scale._load_or_create_ledger(plans)["variants"]["repeat"]["state"] == (
        "submission_unknown"
    )
    assert scale._has_unresolved_job() is True


@pytest.mark.parametrize("mutation", ["marker_version", "result_fingerprint", "manifest_identity"])
def test_collect_rejects_checkpoint_identity_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    _job, _report, artifacts = _complete_collection_fixture(tmp_path, monkeypatch)
    training = artifacts / "output-repeat-r3/q25_completion_scale_r1-repeat/training"
    if mutation == "marker_version":
        path = training / "resume-step-000512.pt.complete.json"
        value = scale.read_json(path)
        value["version"] = 99
    elif mutation == "result_fingerprint":
        path = training / "run_result.json"
        value = scale.read_json(path)
        value["fingerprint"] = "0" * 64
    else:
        path = training / "run_manifest.json"
        value = scale.read_json(path)
        value["identity"]["scale_variant"] = "scaled"
    scale.save(path, value)
    with pytest.raises(ValueError, match="checkpoint marker differs"):
        scale.collect("repeat")


def test_collect_records_actual_checkpoint_bytes_without_claiming_payload_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _job, _report, _artifacts = _complete_collection_fixture(tmp_path, monkeypatch)
    result = scale.collect("repeat")
    assert result["checkpoint"]["bytes"] == len(b"checkpoint-data")
    assert result["checkpoint"]["sha256"] == _sha_bytes(b"checkpoint-data")
    assert result["checkpoint"]["payload_reload_verified"] is False


def test_collect_uses_canonical_reference_recorded_by_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, report, _artifacts = _complete_collection_fixture(tmp_path, monkeypatch)
    job["requested_reference"] = scale.KERNELS["repeat"]
    job["reference"] = scale.KERNELS["repeat"] + "-canonical"
    scale.save(report / "job-repeat-r3.json", job)
    result = scale.collect("repeat")
    assert result["reference"] == job["reference"]
    assert scale._has_unresolved_job() is False


def test_kernel_reference_cannot_change_owner_or_lose_requested_identity() -> None:
    with pytest.raises(ValueError, match="allocation receipt"):
        scale._job_reference({"reference": "other/kernel"}, "repeat")
    with pytest.raises(ValueError, match="allocation receipt"):
        scale._job_reference({"reference": scale.KERNELS["repeat"] + "-other"}, "repeat")


def test_watch_process_restart_observes_existing_job_without_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, report, _artifacts = _complete_collection_fixture(tmp_path, monkeypatch)
    job["submitted_at"] = datetime.now(UTC).isoformat()
    scale.save(report / "job-repeat-r3.json", job)
    calls: list[tuple[str, ...]] = []

    def run(*args: str, **_kwargs: Any) -> str:
        calls.append(args)
        return "KernelWorkerStatus.COMPLETE"

    monkeypatch.setattr(scale, "cli", run)
    monkeypatch.setattr(scale, "collect", lambda variant: {"variant": variant, "collected": True})
    assert scale.watch("repeat") == {"variant": "repeat", "collected": True}
    assert len(calls) == 1
    assert calls[0][1:3] == ("kernels", "status")


def test_watch_does_not_consume_renewed_quota_after_original_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, report, _artifacts = _complete_collection_fixture(tmp_path, monkeypatch)
    job["submitted_at"] = "2020-01-01T00:00:00+00:00"
    scale.save(report / "job-repeat-r3.json", job)
    with pytest.raises(TimeoutError, match="observer deadline"):
        scale.watch("repeat")


@pytest.mark.parametrize("variant", ["repeat", "scaled"])
def test_actual_preparation_shape_freezes_into_worker_compatible_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    _freeze_fixture(tmp_path, monkeypatch)
    plan = scale.freeze(variant)
    worker_path = ROOT / "kaggle/q25_code_cpt_r2/run_fim.py"
    spec = importlib.util.spec_from_file_location("scale_controller_worker_roundtrip", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    source = worker_path.read_text().replace("__SESSION_JSON__", "{}")
    exec(compile(source, str(worker_path), "exec"), worker.__dict__)
    files = {
        name: {"sha256": scale.digest(path), "bytes": path.stat().st_size}
        for name, path in scale._bundle_paths(plan).items()
    }
    # Toy fixtures cannot reproduce historical corpus bytes. Substitute only
    # these two pinned baseline references; all generated protocol fields pass
    # directly from freeze into the real worker validator.
    plan["data"]["previous_training_sha256"] = (
        "341f2d54da2d3c64299c18a918049ded75235ed375137012df71a9ce737fd690"
    )
    plan["data"]["previous_development_sha256"] = (
        "43c56d113a819256c7e175ef1f863b9f622a8119f4d3d01b24d455a010fed4ac"
    )
    session = {
        "commit": plan["base_commit"],
        "plan_sha256": scale.digest(scale.TRAINING_PLANS[variant]),
        "input_manifest_sha256": "c" * 64,
        "arm": scale.TRAIN_ARM,
        "attempt": 1,
        "session_seconds": scale.SESSION_SECONDS,
        "resume_source": None,
        "external_campaign_tokens": 0,
        "scale_variant": variant,
    }
    worker.validate_session(session)
    worker.verify_scale_plan(plan, session, files)


@pytest.mark.parametrize("counts", [{}, {"epochs": True}, {"distinct_states": 4095}])
def test_freeze_validates_both_preparation_variant_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counts: dict[str, Any]
) -> None:
    fixture = _freeze_fixture(tmp_path, monkeypatch)
    preparation = fixture["prep"]
    preparation["experiment"]["scaled"] = counts
    with pytest.raises(ValueError, match="variant counts differ"):
        scale._normalized_experiment(preparation)


@pytest.mark.parametrize("variant", ["repeat", "scaled"])
def test_revision_three_bundle_preserves_historical_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    _freeze_fixture(tmp_path, monkeypatch)
    historical = scale.ARTIFACTS / f"input-bundle-{variant}/plan.json"
    _write(historical, "historical invalid revision one\n")
    historical_bytes = historical.read_bytes()
    plan = scale.freeze(variant)
    monkeypatch.setattr(scale, "_check_storage", lambda **_kwargs: None)

    directory = scale.build_bundle(plan)

    assert directory == scale.ARTIFACTS / f"input-bundle-{variant}-r3"
    assert historical.read_bytes() == historical_bytes
    assert scale.read_json(directory / "input-manifest.json")["plan_sha256"] == scale.digest(
        scale.TRAINING_PLANS[variant]
    )


def test_revision_three_ledger_preserves_failure_charge_and_total_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_fixture(tmp_path, monkeypatch)
    plans = {variant: scale.freeze(variant) for variant in scale.VARIANTS}
    prior_ledger = scale.REPORT / "campaign_budget.json"
    _write(prior_ledger, "historical acknowledged failure\n")
    prior_bytes = prior_ledger.read_bytes()

    ledger = scale._load_or_create_ledger(plans)

    assert scale._report_record("campaign_budget").name == "campaign_budget-r3.json"
    assert prior_ledger.read_bytes() == prior_bytes
    prior_seconds = ledger["prior_failed_allocation"]["session_wall_seconds_upper_bound"]
    repeat_seconds = ledger["variants"]["repeat"]["reserved_session_seconds"]
    scaled_seconds = ledger["variants"]["scaled"]["reserved_session_seconds"]
    assert (prior_seconds, repeat_seconds, scaled_seconds) == (120, 10680, 10800)
    assert prior_seconds + repeat_seconds + scaled_seconds == scale.AGGREGATE_SESSION_SECONDS
    assert (prior_seconds + repeat_seconds + scaled_seconds) / 3600 * 2 == 12
    assert ledger["prior_failed_allocation"]["maximum_additional_repeat_allocations"] == 1
    assert ledger["prior_failed_allocation"]["automatic_retry"] is False


def _failed_prior_fixture(report: Path) -> None:
    reference = "shlokbhakta/tc-q25-completion-scale-r1-repeat"
    receipt = {
        "schema": "q25-completion-scale-verified-output-v1",
        "attempt": 1,
        "reference": reference,
        "training_status": "no_training_executed",
        "checkpoint_verified": False,
        "processed_input_tokens_conservative": 0,
        "discarded_input_tokens_conservative": 0,
        "plan_sha256": "a" * 64,
        "input_manifest_sha256": "b" * 64,
        "status": "KernelWorkerStatus.ERROR",
        "observed_at": "2026-10-05T03:26:03.295129+00:00",
    }
    scale.save(report / "verified-repeat.json", receipt)
    receipt_sha = scale.digest(report / "verified-repeat.json")
    scale.save(
        report / "job-repeat.json",
        {
            "attempt": 1,
            "status": "collected",
            "reference": reference,
            "verified_output_sha256": receipt_sha,
            "plan_sha256": receipt["plan_sha256"],
            "input_manifest_sha256": receipt["input_manifest_sha256"],
            "submitted_at": "2026-10-05T03:24:49.357085+00:00",
        },
    )
    scale.save(
        report / "campaign_budget.json",
        {
            "variants": {
                "repeat": {
                    "state": "collected",
                    "verified_output_sha256": receipt_sha,
                    "processed_input_tokens_conservative": 0,
                },
                "scaled": {"state": "reserved"},
            }
        },
    )


def test_prior_failure_charge_uses_conservative_elapsed_receipt_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scale, "REPORT", tmp_path)
    _failed_prior_fixture(tmp_path)
    charge = scale._prior_failure_charge()
    assert charge["session_wall_seconds_upper_bound"] == 74
    assert charge["prior_scaled_allocation"] == "cancelled_before_allocation"
    assert charge["processed_input_tokens_conservative"] == 0


@pytest.mark.parametrize("failure", ["trained", "missing_timezone", "too_long", "scaled_started"])
def test_retry_requires_zero_work_failure_with_remaining_bounded_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(scale, "REPORT", tmp_path)
    _failed_prior_fixture(tmp_path)
    if failure == "scaled_started":
        _write(tmp_path / "job-scaled.json", "an allocation exists")
    else:
        path = tmp_path / "job-repeat.json"
        job = scale.read_json(path)
        if failure == "trained":
            path = tmp_path / "verified-repeat.json"
            job = scale.read_json(path)
            job["processed_input_tokens_conservative"] = 1
        elif failure == "missing_timezone":
            job["submitted_at"] = "2026-10-05T03:24:49"
        else:
            job["submitted_at"] = "2026-10-04T03:24:49+00:00"
        scale.save(path, job)
    with pytest.raises(ValueError):
        scale._prior_failure_charge()
