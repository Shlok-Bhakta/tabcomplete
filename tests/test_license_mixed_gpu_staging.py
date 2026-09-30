from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _builder():
    path = Path(__file__).parents[1] / "kaggle/one_line_gpu_pilot_r1/build_pilot.py"
    spec = importlib.util.spec_from_file_location("mixed_gpu_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package(root: Path):
    path = root / "proofs/review.json"
    path.parent.mkdir()
    path.write_bytes(b'{"synthetic":true}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    entry = {"path": "proofs/review.json", "bytes": size, "sha256": digest}
    return {"data": {"proof_files": [entry]}}, {
        "independent_review_path": entry["path"],
        "independent_review_bytes": size,
        "independent_review_sha256": digest,
    }


def test_staging_uses_only_explicit_hash_bound_inventory(tmp_path):
    plan, manifest = _package(tmp_path)
    (tmp_path / "unrelated-private.txt").write_text("not a research artifact")
    files = _builder()._mixed_proof_files(plan, manifest, tmp_path)
    assert list(files) == ["proofs/review.json"]
    (tmp_path / "proofs/review.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash or size"):
        _builder()._mixed_proof_files(plan, manifest, tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "../review.json",
        "/tmp/review.json",
        "proofs/.env",
        "proofs/draft.gguf",
        "proofs/optimizer.pt",
    ],
)
def test_staging_rejects_escape_secrets_and_extra_weights(tmp_path, path):
    plan, manifest = _package(tmp_path)
    plan["data"]["proof_files"][0]["path"] = path
    with pytest.raises(ValueError, match="unsafe"):
        _builder()._mixed_proof_files(plan, manifest, tmp_path)


def test_staging_rejects_symlink_and_missing_review_binding(tmp_path):
    plan, manifest = _package(tmp_path)
    original = tmp_path / "proofs/review.json"
    original.rename(tmp_path / "other.json")
    original.symlink_to(tmp_path / "other.json")
    with pytest.raises(ValueError, match="symlink"):
        _builder()._mixed_proof_files(plan, manifest, tmp_path)
    original.unlink()
    (tmp_path / "other.json").rename(original)
    manifest["independent_review_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="independent review"):
        _builder()._mixed_proof_files(plan, manifest, tmp_path)


def test_fixture_only_worker_rejects_training_claim_and_unlisted_artifacts(tmp_path):
    builder = _builder()
    directory = tmp_path / "inputs"
    directory.mkdir()
    spec = {
        "schema": "single-line-disposable-training-fixture-v2",
        "nonpadding_training_input_tokens": 10592,
    }
    plan = {
        "schema": "one-line-disposable-fixture-plan-v2",
        "quality_evidence": False,
        "accepted_training": 0,
        "training": {"disposable_fixture": spec},
        "budgets": {
            "max_session_seconds": 7200,
            "reserve_seconds": 1200,
            "no_automatic_renewal": True,
        },
    }
    names = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "config.yaml",
        "plan.json",
        "training-fixture.jsonl",
    )
    for name in names:
        (directory / name).write_text(json.dumps(plan) if name == "plan.json" else "fixture")
    session = {
        "fixture_only": True,
        "branch": "prototype/product-r2",
        "commit": "a" * 40,
        "plan_sha256": builder.sha256_file(directory / "plan.json"),
        "config_sha256": builder.sha256_file(directory / "config.yaml"),
        "session_seconds": 7200,
        "reserve_seconds": 1200,
        "disposable_fixture": spec,
    }
    manifest = {
        "schema": "one-line-disposable-fixture-input-v2",
        "branch": session["branch"],
        "commit": session["commit"],
        "plan_sha256": session["plan_sha256"],
        "files": {
            name: {
                "bytes": (directory / name).stat().st_size,
                "sha256": builder.sha256_file(directory / name),
            }
            for name in names
        },
    }
    (directory / "input-manifest.json").write_text(json.dumps(manifest))
    session["input_manifest_sha256"] = builder.sha256_file(directory / "input-manifest.json")
    template = Path(builder.__file__).with_name("run.py").read_text()
    source = template.replace('"__SESSION_LITERAL__"', repr(json.dumps(session)))
    worker = {"__name__": "fixture_test_worker"}
    exec(compile(source, "fixture_test_worker.py", "exec"), worker)
    found, _ = worker["_safe_fixture_input"](directory)
    assert found == directory
    (directory / "extra-model.gguf").write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="unapproved artifact"):
        worker["_safe_fixture_input"](directory)
    (directory / "extra-model.gguf").unlink()
    plan["accepted_training"] = 64
    (directory / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="hash or size"):
        worker["_safe_fixture_input"](directory)


def test_fixture_output_verifies_export_and_per_action_eos(tmp_path):
    builder = _builder()
    training = tmp_path / "disposable-training-fixture"
    training.mkdir()
    checkpoint = training / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    digest = builder.sha256_file(checkpoint)
    counts = {"completed_updates": 32, "skipped_updates": 0, "training_input_tokens": 10592}
    generations = [
        {"gold_action": kind, "terminated_by_eos": True, "valid_action": True, "exact_action": True}
        for kind in ("keep", "replace_line", "insert_before", "delete_line")
        for _ in range(4)
    ]
    observation = {
        "greedy_generation_observations": generations,
        "changed_parameter_elements": 1,
        "response_and_eos_positions_supervised": True,
        "implementation_viability": {"passed": True},
    }
    export = training / "inference-f16"
    export.mkdir()
    weight = export / "model.safetensors"
    weight.write_bytes(b"synthetic-export")
    builder.write_json(
        export / "artifact_manifest.json",
        {"source": {"model_weight_sha256": builder.WEIGHT_SHA256}},
    )
    result = {
        "status": "complete",
        "fingerprint": "test-fixture",
        "examples": 64,
        "cursor": counts,
        "disposable_fixture": observation,
        "identity": {"phase": "fixture", "source": {"model_weight_sha256": builder.WEIGHT_SHA256}},
        "inference_export": {
            "path": "/unavailable/worker/path",
            "files": {
                weight.name: {"bytes": weight.stat().st_size, "sha256": builder.sha256_file(weight)}
            },
        },
    }
    builder.write_json(training / "run_result.json", result)
    builder.write_json(
        training / "latest.json",
        {"checkpoint": checkpoint.name, "sha256": digest, "cursor": counts},
    )
    builder.write_json(
        checkpoint.with_suffix(".pt.complete.json"),
        {"sha256": digest, "fingerprint": "test-fixture"},
    )
    status = {
        "schema": "one-line-instinct-pilot-worker-status-v1",
        "state": "disposable_verified_complete",
        "plan_sha256": "f" * 64,
        "checkpoint_sha256": digest,
        "training_status": "complete",
        "source_weight_sha256": builder.WEIGHT_SHA256,
        "quality_evidence": False,
        "accepted_training": 0,
        "training_input_tokens": 10592,
    }
    builder.write_json(tmp_path / "worker-status.json", status)
    verification = {
        "quality_evidence": False,
        "checkpoint_sha256": digest,
        "token_counts": counts,
        "actual_generation_observations": generations,
    }
    builder.write_json(tmp_path / "disposable-fixture-verification.json", verification)
    checked = builder.verify_output(tmp_path, expected_plan_sha256="f" * 64)
    assert checked["implementation_only"] is True and checked["quality_evidence"] is False
    # A positive summary cannot rescue two missing EOS completions.
    generations[0]["terminated_by_eos"] = False
    generations[1]["terminated_by_eos"] = False
    builder.write_json(training / "run_result.json", result)
    builder.write_json(tmp_path / "disposable-fixture-verification.json", verification)
    with pytest.raises(ValueError, match="per-action EOS"):
        builder.verify_output(tmp_path)


def test_frozen_source_inventory_detects_changes(tmp_path):
    builder = _builder()
    source = tmp_path / "run.py"
    source.write_bytes(b"frozen")
    plan = {"source_files": {"run.py": builder.sha256_file(source)}}
    builder.verify_frozen_sources(plan, tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after freeze"):
        builder.verify_frozen_sources(plan, tmp_path)


def test_fixture_telemetry_preserves_owned_cuda_environment(tmp_path, monkeypatch):
    import os
    import sys

    monkeypatch.setattr(os, "environ", dict(os.environ))

    builder = _builder()
    template = Path(builder.__file__).with_name("run.py").read_text()
    session = {"session_seconds": 7200, "reserve_seconds": 1200, "plan_sha256": "f" * 64}
    source = template.replace('"__SESSION_LITERAL__"', repr(json.dumps(session)))
    worker = {"__name__": "fixture_environment_test"}
    exec(compile(source, "fixture_environment_test.py", "exec"), worker)
    monkeypatch.setitem(worker, "REPO", Path(__file__).parents[1])
    monkeypatch.setitem(worker, "OUT", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    owned_environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/synthetic/cuda/lib64",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TABCOMPLETE_PILOT_STARTED_MONOTONIC": "12345",
    }
    received = {}

    def implementation(dataset, env):
        assert dataset == tmp_path
        received.update(env)
        return 10592

    monkeypatch.setitem(worker, "_run_disposable_training_fixture_impl", implementation)
    assert worker["_run_disposable_training_fixture"](tmp_path, owned_environment) == 10592
    assert owned_environment.items() <= received.items()
    assert received["TABCOMPLETE_CAMPAIGN_ID"] == "tabcomplete-product-r2"
    assert received["TABCOMPLETE_RUN_ID"] == "run-" + "f" * 32
    metadata = tmp_path / "fixture-observability-run.json"
    stored = json.loads(metadata.read_text())
    assert stored["run_id"] == received["TABCOMPLETE_RUN_ID"]
    stored["run_id"] = "run-" + "e" * 32
    metadata.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="different frozen plan"):
        worker["_run_disposable_training_fixture"](tmp_path, owned_environment)
    assert set(owned_environment) == {
        "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "TABCOMPLETE_PILOT_STARTED_MONOTONIC",
    }


def test_storage_budget_counts_reused_hard_links_once(tmp_path):
    import os
    import shutil

    original = tmp_path / "original"
    original.write_bytes(b"synthetic")
    os.link(original, tmp_path / "reused")
    (tmp_path / "alias").symlink_to(original.name)
    assert _builder()._directory_bytes(tmp_path) == len(b"synthetic")
    shutil.copyfile(original, tmp_path / "new-copy")
    assert _builder()._directory_bytes(tmp_path) == 2 * len(b"synthetic")
