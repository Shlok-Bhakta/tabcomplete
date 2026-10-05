from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "kaggle/q25_code_cpt_r2/run_fim.py"


def _load_worker() -> ModuleType:
    source = WORKER_PATH.read_text(encoding="utf-8").replace("__SESSION_JSON__", "{}")
    spec = importlib.util.spec_from_file_location("q25_completion_scale_worker_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, str(WORKER_PATH), "exec"), module.__dict__)
    return module


worker = _load_worker()


def _sha(value: bytes | str) -> str:
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _file_record(name: str, data: bytes) -> dict[str, Any]:
    return {"sha256": _sha(data), "bytes": len(data)}


def _session(variant: str = "repeat") -> dict[str, Any]:
    return {
        "commit": "a" * 40,
        "plan_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "arm": worker.TRAIN_ARM,
        "attempt": 1,
        "session_seconds": 7200,
        "resume_source": None,
        "external_campaign_tokens": 0,
        "scale_variant": variant,
    }


def _full_plan(files: dict[str, dict[str, Any]], variant: str = "repeat") -> dict[str, Any]:
    return {
        "schema": worker.SCALE_PLAN_SCHEMA,
        "branch": worker.SCALE_BRANCH,
        "gpu_execution_authorized": True,
        "scale_variant": variant,
        "base_commit": _session(variant)["commit"],
        "configuration": {
            "budget": {
                "maximum_campaign_input_tokens": worker.SCALE_MAX_CAMPAIGN_TOKENS,
                "aggregate_session_seconds": 21_600,
                "conservative_account_gpu_hours": 12,
                "conservative_quota_multiplier": 2,
                "session_seconds": worker.MAX_SESSION_SECONDS,
                "new_artifact_bytes_cap": worker.MAX_ARTIFACT_BYTES,
                "minimum_free_bytes": worker.MINIMUM_FREE_BYTES,
                "minimum_finalization_reserve_seconds": worker.MINIMUM_FINAL_RESERVE_SECONDS,
                "runtime_setup_reserve_seconds": 1800,
                "paid_compute": False,
                "automatic_renewal_use": False,
            },
            "training": {
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
                "max_input_tokens": worker.SCALE_MAX_ARM_TOKENS,
            },
            "runtime": worker.EXPECTED_RUNTIME,
            "runtime_lock": {
                "runtime_lock_sha256": "d" * 64,
                "requirements_lock_sha256": "e" * 64,
                "bootstrap_uv_version": "0.12.3",
                "bootstrap_uv_wheel_sha256": worker.UV_WHEEL_SHA256,
            },
        },
        "data": {
            "corpus_metadata_sha256": files["corpus_metadata.json"]["sha256"],
            "previous_training_sha256": (
                "341f2d54da2d3c64299c18a918049ded75235ed375137012df71a9ce737fd690"
            ),
            "previous_development_sha256": (
                "43c56d113a819256c7e175ef1f863b9f622a8119f4d3d01b24d455a010fed4ac"
            ),
            "repeat_train": {
                "sha256": files["repeat_train.jsonl"]["sha256"],
                "bytes": files["repeat_train.jsonl"]["bytes"],
                "row_count": 4096,
                "input_tokens": 1000,
                "target_tokens": 100,
            },
            "scaled_train": {
                "sha256": files["scaled_train.jsonl"]["sha256"],
                "bytes": files["scaled_train.jsonl"]["bytes"],
                "row_count": 8192,
                "input_tokens": 2000,
                "target_tokens": 200,
            },
            "development_new": {
                "sha256": files["development_new.jsonl"]["sha256"],
                "bytes": files["development_new.jsonl"]["bytes"],
                "row_count": 512,
                "input_tokens": 150,
                "target_tokens": 20,
            },
            "development_previous": {
                "sha256": files["development_previous.jsonl"]["sha256"],
                "bytes": files["development_previous.jsonl"]["bytes"],
                "row_count": 240,
                "input_tokens": 100,
                "target_tokens": 10,
            },
        },
        "initializers": {
            worker.TRAIN_ARM: {
                "kind": "untouched_pretrained",
                "model_id": "Qwen/Qwen2.5-Coder-0.5B",
                "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
            }
        },
        "experiment": {
            "variants": {
                "repeat": {"distinct_states": 4096, "epochs": 2, "example_exposures": 8192},
                "scaled": {"distinct_states": 8192, "epochs": 1, "example_exposures": 8192},
            }
        },
        "evaluation": {
            "attention_backend": "torch-efficient-sdpa-explicit-kv-repeat-v1",
            "fixtures": {
                "causal": {"sha256": files["causal200.jsonl"]["sha256"]},
                "line": {"sha256": files["line180.jsonl"]["sha256"]},
            }
        },
    }


def _write_bundle(root: Path, variant: str = "repeat") -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_contents = {
        "repeat_train.jsonl": b"repeat\n",
        "scaled_train.jsonl": b"scaled\n",
        "development_new.jsonl": b"new-dev\n",
        "development_previous.jsonl": b"old-dev\n",
        "causal200.jsonl": b"causal\n",
        "line180.jsonl": b"line\n",
    }
    metadata: dict[str, Any] = {
        "schema": "q25-completion-scale-corpus-v1",
        "raw_source_content_emitted": False,
        "sealed_test_accessed": False,
        "preparation_plan_sha256": "f" * 64,
        "parent_cpt_plan_sha256": "1" * 64,
        "tokenizer_id": "Qwen/Qwen2.5-Coder-0.5B",
    }
    (root / "corpus_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    artifact_contents["corpus_metadata.json"] = (root / "corpus_metadata.json").read_bytes()
    files = {}
    for name, payload in artifact_contents.items():
        (root / name).write_bytes(payload)
        files[name] = _file_record(name, payload)
    plan = _full_plan(files, variant)
    # Tie plan metadata and selected records to the actual bundle files.
    plan["preparation_plan_sha256"] = metadata["preparation_plan_sha256"]
    plan["parent_cpt_plan_sha256"] = metadata["parent_cpt_plan_sha256"]
    plan_bytes = json.dumps(plan, sort_keys=True).encode() + b"\n"
    (root / "plan.json").write_bytes(plan_bytes)
    files["plan.json"] = _file_record("plan.json", plan_bytes)
    manifest = {"files": files}
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode() + b"\n"
    (root / "input-manifest.json").write_bytes(manifest_bytes)
    session = _session(variant)
    session["plan_sha256"] = _sha(plan_bytes)
    session["input_manifest_sha256"] = _sha(manifest_bytes)
    return session, plan


def test_scale_session_selects_a_separate_validated_output_namespace() -> None:
    session = worker.validate_session(_session("scaled"))
    instance = worker.Worker(session)
    assert instance.out == worker.WORK_ROOT / "q25_completion_scale_r1-scaled"
    assert instance.status["scale_variant"] == "scaled"
    with pytest.raises(worker.WorkerError, match="scale_session_identity_invalid"):
        worker.validate_session({**_session("repeat"), "arm": worker.CPT_ARM})
    with pytest.raises(worker.WorkerError, match="scale_session_identity_invalid"):
        worker.validate_session({**_session("repeat"), "scale_variant": "other"})


def test_scale_input_bundle_binds_plan_variant_and_all_four_data_files(tmp_path: Path) -> None:
    session, _plan = _write_bundle(tmp_path, "repeat")
    manifest = json.loads((tmp_path / "input-manifest.json").read_text())
    paths, plan = worker.verify_input_bundle(tmp_path, manifest, session)
    assert set(paths) == worker.SCALE_REQUIRED_INPUT_FILES
    assert plan["scale_variant"] == "repeat"


def test_scale_plan_rejects_variant_or_epoch_drift() -> None:
    contents = {
        name: _file_record(name, name.encode())
        for name in worker.SCALE_REQUIRED_INPUT_FILES
    }
    contents["corpus_metadata.json"] = _file_record("corpus_metadata.json", b"meta")
    plan = _full_plan(contents, "repeat")
    worker.verify_scale_plan(plan, _session("repeat"), contents)
    plan["base_commit"] = "d" * 40
    with pytest.raises(worker.WorkerError, match="frozen_scale_plan_identity_invalid"):
        worker.verify_scale_plan(plan, _session("repeat"), contents)
    plan["base_commit"] = _session()["commit"]
    plan["evaluation"]["attention_backend"] = "math"
    with pytest.raises(worker.WorkerError, match="frozen_scale_budget_or_training_invalid"):
        worker.verify_scale_plan(plan, _session("repeat"), contents)
    plan["evaluation"]["attention_backend"] = "torch-efficient-sdpa-explicit-kv-repeat-v1"
    with pytest.raises(worker.WorkerError, match="frozen_scale_plan_identity_invalid"):
        worker.verify_scale_plan(plan, _session("scaled"), contents)
    plan["configuration"]["training"]["epochs"] = 2
    with pytest.raises(worker.WorkerError, match="frozen_scale_budget_or_training_invalid"):
        worker.verify_scale_plan(plan, _session("repeat"), contents)


def test_scale_checkout_checks_frozen_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    source = repo / "trainer.py"
    source.write_bytes(b"frozen trainer\n")
    monkeypatch.setattr(worker, "REPO", repo)
    monkeypatch.setattr(worker, "WORK_ROOT", tmp_path / "output")
    instance = worker.Worker(_session())
    (instance.out / "logs").mkdir(parents=True)
    (instance.out / "logs" / "verify-repository-commit.log").write_text(
        _session()["commit"] + "\n"
    )
    monkeypatch.setattr(instance, "_setup_stage", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(instance, "_runtime_artifact_bytes", lambda: 100_000)
    plan = {
        "source_identity": {
            "commit": _session()["commit"],
            "files": {"trainer.py": _file_record("trainer.py", source.read_bytes())},
        }
    }
    instance._checkout_repository(plan)
    source.write_bytes(b"changed trainer\n")
    with pytest.raises(worker.WorkerError, match="scale_repository_source_hash_mismatch"):
        instance._checkout_repository(plan)
    plan["source_identity"]["files"] = {"../escape.py": _file_record("escape.py", b"")}
    with pytest.raises(worker.WorkerError, match="scale_repository_source_identity_invalid"):
        instance._checkout_repository(plan)


def test_scale_trainer_command_selects_variant_paths_without_changing_arm(tmp_path: Path) -> None:
    paths = {
        "train.jsonl": tmp_path / "scaled_train.jsonl",
        "repeat_train.jsonl": tmp_path / "repeat_train.jsonl",
        "scaled_train.jsonl": tmp_path / "scaled_train.jsonl",
        "development.jsonl": tmp_path / "development_new.jsonl",
        "development_previous.jsonl": tmp_path / "development_previous.jsonl",
        "corpus_metadata.json": tmp_path / "corpus_metadata.json",
    }
    command = worker.trainer_command(
        python=Path("python"),
        model=Path("model"),
        paths=paths,
        plan=Path("plan.json"),
        arm=worker.TRAIN_ARM,
        output=Path("out"),
        session_seconds=7200,
        reserve_seconds=1800,
        external_campaign_tokens=0,
        mounted_bytes=0,
        resume=None,
        execute=False,
        scale_variant="scaled",
    )
    assert command[command.index("--arm") + 1] == worker.TRAIN_ARM
    assert command[command.index("--train") + 1] == str(paths["scaled_train.jsonl"])
    assert command[command.index("--development") + 1] == str(paths["development.jsonl"])
    assert command[command.index("--scale-variant") + 1] == "scaled"
    assert command[command.index("--historical-development") + 1] == str(
        paths["development_previous.jsonl"]
    )


def test_scale_checkout_accepts_actual_controller_source_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_path = ROOT / "scripts/run_q25_completion_scale.py"
    spec = importlib.util.spec_from_file_location(
        "scale_controller_source_contract", controller_path
    )
    assert spec is not None and spec.loader is not None
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    # Use the real producer rather than hand-constructing its wire shape.
    actual_sources = controller._source_files()
    assert actual_sources and all(isinstance(value, str) for value in actual_sources.values())
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    for relative in actual_sources:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    monkeypatch.setattr(worker, "REPO", repo)
    monkeypatch.setattr(worker, "WORK_ROOT", tmp_path / "output")
    instance = worker.Worker(_session())
    (instance.out / "logs").mkdir(parents=True)
    (instance.out / "logs" / "verify-repository-commit.log").write_text(
        _session()["commit"] + "\n"
    )
    monkeypatch.setattr(instance, "_setup_stage", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(instance, "_runtime_artifact_bytes", lambda: 2**30)
    plan = {
        "source_identity": {"commit": _session()["commit"], "files": actual_sources}
    }
    instance._checkout_repository(plan)
    changed_path = repo / "kaggle/q25_code_cpt_r2/run_fim.py"
    changed_path.write_bytes(changed_path.read_bytes() + b"\n# changed fixture\n")
    with pytest.raises(worker.WorkerError, match="scale_repository_source_hash_mismatch"):
        instance._checkout_repository(plan)
    changed_path.unlink()
    changed_path.symlink_to(ROOT / "kaggle/q25_code_cpt_r2/run_fim.py")
    with pytest.raises(worker.WorkerError, match="scale_repository_source_hash_mismatch"):
        instance._checkout_repository(plan)
