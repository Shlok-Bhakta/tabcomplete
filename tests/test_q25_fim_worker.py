from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "kaggle/q25_code_cpt_r2/run_fim.py"


def _load_worker() -> ModuleType:
    source = WORKER_PATH.read_text(encoding="utf-8")
    source = source.replace("__SESSION_JSON__", "{}")
    spec = importlib.util.spec_from_file_location("q25_fim_worker_test_module", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, str(WORKER_PATH), "exec"), module.__dict__)
    return module


def _sha(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


def _cursor(index: int) -> dict[str, int]:
    return {
        "next_example_index": index,
        "completed_updates": index,
        "attempted_updates": index,
        "skipped_updates": 0,
        "training_input_tokens": index * 1024,
        "supervised_target_tokens": index * 64,
        "epoch": 0,
    }


def _resume_identity(module: ModuleType, checkpoint: Path, index: int) -> dict[str, Any]:
    return {
        "fingerprint": _sha(b"stable-fim-run-fingerprint"),
        "checkpoint_sha256": module.sha256_file(checkpoint),
        "cursor": _cursor(index),
    }


def _write_attempt(
    module: ModuleType,
    output: Path,
    *,
    index: int,
    plan_sha256: str,
    input_manifest_sha256: str,
    arm: str,
    initializer: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    training = output / "training"
    training.mkdir(parents=True)
    checkpoint = training / f"checkpoint-{index}.safetensors"
    checkpoint.write_bytes(f"synthetic checkpoint {index}".encode())
    resume_identity = _resume_identity(module, checkpoint, index)
    module._write_json(
        checkpoint.with_suffix(checkpoint.suffix + ".complete.json"),
        {
            "fingerprint": resume_identity["fingerprint"],
            "sha256": resume_identity["checkpoint_sha256"],
        },
    )
    module._write_json(
        training / "latest.json",
        {
            "schema": "q25-fim-latest-checkpoint-v1",
            "path": checkpoint.name,
            "fingerprint": resume_identity["fingerprint"],
            "sha256": resume_identity["checkpoint_sha256"],
            "cursor": resume_identity["cursor"],
        },
    )
    identity = {
        "plan_sha256": plan_sha256,
        "arm": arm,
        "initializer": initializer,
        "corpus_metadata": {"metadata_sha256": _sha(b"fim-metadata")},
        "training_data": {"sha256": _sha(b"fim-training-data")},
    }
    module._write_json(
        training / "run_manifest.json",
        {"fingerprint": resume_identity["fingerprint"], "identity": identity},
    )
    if baseline is not None:
        module._write_json(output / "baseline" / "identity.json", baseline)
        for name, record in baseline["files"].items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = b"first-attempt baseline output: " + name.encode()
            assert len(content) == record["bytes"]
            assert _sha(content) == record["sha256"]
            path.write_bytes(content)
    return checkpoint, resume_identity


def _session(
    *,
    plan_sha256: str,
    input_manifest_sha256: str,
    arm: str,
    resume_source: str | None = None,
    resume_identity: dict[str, Any] | None = None,
    session_seconds: int = 3600,
    external_campaign_tokens: int = 0,
    attempt: int = 1,
) -> dict[str, Any]:
    return {
        "commit": "a" * 40,
        "plan_sha256": plan_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "arm": arm,
        "attempt": attempt,
        "session_seconds": session_seconds,
        "resume_source": resume_source,
        "resume_identity": resume_identity,
        "external_campaign_tokens": external_campaign_tokens,
    }


def _write_bundle(
    module: ModuleType,
    root: Path,
    *,
    external_campaign_tokens: int = 0,
    session_seconds: int = 10_800,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    input_root = root / "input"
    bundle = input_root / "fim-inputs"
    bundle.mkdir(parents=True)
    parent_cpt_sha = _sha(b"frozen parent CPT plan")
    preparation_sha = _sha(b"frozen FIM preparation plan")
    contents = {
        "train.jsonl": b'{"id":1}\n',
        "development.jsonl": b'{"id":2}\n',
        "causal200.jsonl": b'{"id":"causal"}\n',
        "line180.jsonl": b'{"id":"line"}\n',
    }
    metadata = {
        "raw_source_content_emitted": False,
        "sealed_test_accessed": False,
        "preparation_plan_sha256": preparation_sha,
        "parent_cpt_plan_sha256": parent_cpt_sha,
        "tokenizer_id": "Qwen/Qwen2.5-Coder-0.5B",
    }
    contents["corpus_metadata.json"] = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode()
    hashes = {name: _sha(value) for name, value in contents.items()}
    sizes = {name: len(value) for name, value in contents.items()}
    model_content = b"synthetic local initializer weights"
    model_sha = _sha(model_content)
    plan = {
        "schema": "q25-fim-training-plan-v1",
        "gpu_execution_authorized": True,
        "preparation_plan_sha256": preparation_sha,
        "parent_cpt_plan_sha256": parent_cpt_sha,
        "configuration": {
            "budget": {
                "maximum_campaign_input_tokens": 32_000_000,
                "session_seconds": 10_800,
                "new_artifact_bytes_cap": 4 * 1024**3,
                "minimum_free_bytes": 0,
                "paid_compute": False,
                "automatic_renewal_use": False,
            },
            "training": {"max_input_tokens": 1_000_000},
        },
        "data": {
            "corpus_metadata_sha256": hashes["corpus_metadata.json"],
            "train": {"sha256": hashes["train.jsonl"], "bytes": sizes["train.jsonl"]},
            "development": {
                "sha256": hashes["development.jsonl"],
                "bytes": sizes["development.jsonl"],
            },
        },
        "initializers": {
            module.TRAIN_ARM: {
                "kind": "untouched_pretrained",
                "files": {"model.safetensors": {"sha256": model_sha, "bytes": len(model_content)}},
            }
        },
        "evaluation": {
            "fixtures": {
                "causal": {"sha256": hashes["causal200.jsonl"]},
                "line": {"sha256": hashes["line180.jsonl"]},
            }
        },
    }
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    contents["plan.json"] = plan_bytes
    for name, content in contents.items():
        (bundle / name).write_bytes(content)
    manifest = {
        "schema": "q25-fim-input-manifest-v1",
        "files": {
            name: {"bytes": len(content), "sha256": _sha(content)}
            for name, content in contents.items()
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = bundle / "input-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    model_dir = input_root / "base-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(model_content)
    session = _session(
        plan_sha256=_sha(plan_bytes),
        input_manifest_sha256=_sha(manifest_bytes),
        arm=module.TRAIN_ARM,
        session_seconds=session_seconds,
        external_campaign_tokens=external_campaign_tokens,
    )
    return input_root, bundle, session, plan


def test_baseline_is_carried_through_two_successive_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    plan_sha256 = _sha(b"frozen FIM plan")
    input_manifest_sha256 = _sha(b"frozen inputs")
    initializer = {
        "arm": module.TRAIN_ARM,
        "model_revision": "synthetic-pinned-revision",
        "files": {"model.safetensors": {"sha256": _sha(b"weights"), "bytes": 7}},
    }
    initializer_sha256 = module.canonical_sha256(initializer)

    # Attempt 1 has the original before-FIM evaluations plus a model export.
    initial_files: dict[str, dict[str, Any]] = {}
    initial_contents: dict[str, bytes] = {}
    for name in ("before/development/results.json", "before/line/results.json"):
        content = b"first-attempt baseline output: " + name.encode()
        initial_contents[name] = content
        initial_files[name] = {"bytes": len(content), "sha256": _sha(content)}
    baseline = {
        "schema": "q25-fim-baseline-v1",
        "arm": module.TRAIN_ARM,
        "plan_sha256": plan_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "initializer_identity_sha256": initializer_sha256,
        "measured_seconds": 17.25,
        "files": initial_files,
    }
    attempt1_input = tmp_path / "mount-attempt-1"
    attempt1_root = attempt1_input / "previous-attempt-1"
    _, resume1 = _write_attempt(
        module,
        attempt1_root,
        index=1,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        initializer=initializer,
        baseline=baseline,
    )
    export = attempt1_root / "training/inference-f16/model.safetensors"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"do not copy weights")
    session1 = _session(
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        resume_source="owner/fim-arm-attempt-1",
        resume_identity=resume1,
    )
    _, resolved_root1, resolved_baseline1 = module.resolve_verified_resume(
        attempt1_input,
        session1,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        initializer_sha256=initializer_sha256,
        arm=module.TRAIN_ARM,
    )
    assert resolved_root1 == attempt1_root
    assert resolved_baseline1 == baseline

    # Attempt 2 copies only the verified baseline outputs and preserves its identity.
    attempt2_root = tmp_path / "attempt-2-output"
    monkeypatch.setattr(module, "OUT", attempt2_root)
    worker2 = module.Worker(session1)
    worker2.out.mkdir(parents=True)
    worker2._record_baseline(
        initializer_sha256=initializer_sha256,
        measured_seconds=17.25,
        plan={"configuration": {"budget": {"new_artifact_bytes_cap": 2 * 1024**3}}},
        mounted_bytes=0,
        previous_root=resolved_root1,
        inherited=resolved_baseline1,
    )
    assert json.loads((attempt2_root / "baseline/identity.json").read_text()) == baseline
    reference2 = json.loads((attempt2_root / "baseline/reference.json").read_text())
    assert reference2["source"] == "owner/fim-arm-attempt-1"
    assert reference2["source_identity_sha256"] == module.canonical_sha256(baseline)
    for name, content in initial_contents.items():
        assert (attempt2_root / name).read_bytes() == content
    assert not (attempt2_root / "training/inference-f16/model.safetensors").exists()

    # Attempt 2's output becomes attempt 3's sole resume mount. The original baseline
    # must still verify there, and attempt 3 must carry the same files and identity.
    attempt2_input = tmp_path / "mount-attempt-2"
    attempt2_input.mkdir()
    mounted_attempt2 = attempt2_input / "previous-attempt-2"
    attempt2_root.rename(mounted_attempt2)
    checkpoint2, resume2 = _write_attempt(
        module,
        mounted_attempt2,
        index=2,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        initializer=initializer,
        baseline=None,
    )
    session2 = _session(
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        resume_source="owner/fim-arm-attempt-2",
        resume_identity=resume2,
        attempt=2,
    )
    resolved_checkpoint2, resolved_root2, resolved_baseline2 = module.resolve_verified_resume(
        attempt2_input,
        session2,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        initializer_sha256=initializer_sha256,
        arm=module.TRAIN_ARM,
    )
    assert resolved_checkpoint2 == checkpoint2
    assert resolved_root2 == mounted_attempt2
    assert resolved_baseline2 == baseline

    attempt3_root = tmp_path / "attempt-3-output"
    monkeypatch.setattr(module, "OUT", attempt3_root)
    worker3 = module.Worker(session2)
    worker3.out.mkdir(parents=True)
    worker3._record_baseline(
        initializer_sha256=initializer_sha256,
        measured_seconds=17.25,
        plan={"configuration": {"budget": {"new_artifact_bytes_cap": 2 * 1024**3}}},
        mounted_bytes=0,
        previous_root=resolved_root2,
        inherited=resolved_baseline2,
    )
    assert json.loads((attempt3_root / "baseline/identity.json").read_text()) == baseline
    reference3 = json.loads((attempt3_root / "baseline/reference.json").read_text())
    assert reference3["source"] == "owner/fim-arm-attempt-2"
    assert reference3["source_identity_sha256"] == module.canonical_sha256(baseline)
    for name, content in initial_contents.items():
        assert (attempt3_root / name).read_bytes() == content
    assert not (attempt3_root / "training/inference-f16/model.safetensors").exists()


def test_session_validation_rejects_invalid_budget_and_resume_identity() -> None:
    module = _load_worker()
    base = _session(
        plan_sha256=_sha(b"plan"),
        input_manifest_sha256=_sha(b"inputs"),
        arm=module.TRAIN_ARM,
    )
    assert module.validate_session(base) == base

    invalid_sessions = []
    too_long = dict(base, session_seconds=module.MAX_SESSION_SECONDS + 1)
    invalid_sessions.append((too_long, "session_identity_or_budget_invalid"))
    boolean_budget = dict(base, external_campaign_tokens=True)
    invalid_sessions.append((boolean_budget, "session_identity_or_budget_invalid"))
    missing_resume_identity = dict(base, resume_source="owner/fim-attempt")
    invalid_sessions.append((missing_resume_identity, "resume_identity_missing"))
    identity_without_source = dict(base, resume_identity={})
    invalid_sessions.append((identity_without_source, "resume_identity_without_source"))
    invalid_attempt = dict(base, attempt=True)
    invalid_sessions.append((invalid_attempt, "session_identity_or_budget_invalid"))
    missing_attempt = {key: value for key, value in base.items() if key != "attempt"}
    invalid_sessions.append((missing_attempt, "session_missing_required_fields"))

    for invalid, expected_reason in invalid_sessions:
        with pytest.raises(module.WorkerError) as error:
            module.validate_session(invalid)
        assert error.value.reason == expected_reason


def test_frozen_bundle_accepts_bound_files_and_rejects_hash_mismatch(tmp_path: Path) -> None:
    module = _load_worker()
    input_root, bundle, session, plan = _write_bundle(module, tmp_path)
    manifest = json.loads((bundle / "input-manifest.json").read_text())
    matched_root, matched_manifest = module.find_input_manifest(
        input_root, session["input_manifest_sha256"]
    )
    assert matched_root == bundle
    paths, loaded_plan = module.verify_input_bundle(matched_root, matched_manifest, session)
    assert loaded_plan == plan
    assert set(paths) == module.REQUIRED_INPUT_FILES

    (bundle / "train.jsonl").write_bytes(b'{"id":3}\n')
    with pytest.raises(module.WorkerError) as error:
        module.verify_input_bundle(bundle, manifest, session)
    assert error.value.reason == "input_file_hash_or_size_mismatch"


@pytest.mark.parametrize(
    "mutation,expected_reason",
    [
        ("missing_manifest_entry", "input_manifest_file_set_invalid"),
        ("unlisted_file", "input_bundle_contains_unlisted_file"),
        ("symlink", "input_manifest_file_missing_or_unsafe"),
    ],
)
def test_frozen_bundle_rejects_malformed_or_unsafe_inventory(
    tmp_path: Path, mutation: str, expected_reason: str
) -> None:
    module = _load_worker()
    _, bundle, session, _ = _write_bundle(module, tmp_path)
    manifest = json.loads((bundle / "input-manifest.json").read_text())
    if mutation == "missing_manifest_entry":
        manifest["files"].pop("line180.jsonl")
    elif mutation == "unlisted_file":
        (bundle / "extra.jsonl").write_text("{}\n")
    else:
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes((bundle / "train.jsonl").read_bytes())
        (bundle / "train.jsonl").unlink()
        (bundle / "train.jsonl").symlink_to(outside)

    with pytest.raises(module.WorkerError) as error:
        module.verify_input_bundle(bundle, manifest, session)
    assert error.value.reason == expected_reason


def test_run_stage_uses_only_time_left_after_reserved_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    session = _session(
        plan_sha256=_sha(b"plan"),
        input_manifest_sha256=_sha(b"inputs"),
        arm=module.TRAIN_ARM,
    )
    worker = module.Worker(session)
    worker.out = tmp_path / "worker"
    worker.out.mkdir()
    worker.deadline = 112.0
    ticks = itertools.count()
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(monotonic=lambda: 100.0 + next(ticks) * 0.25),
    )
    observed: dict[str, Any] = {}

    def fake_run(_command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert worker.run_stage(["mock-stage"], "mock", reserve_seconds=5) == 0
    assert observed["timeout"] == pytest.approx(7.0)
    assert worker.status["stages"][-1]["exit_code"] == 0


def test_run_stage_fails_closed_when_deadline_reserve_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    session = _session(
        plan_sha256=_sha(b"plan"),
        input_manifest_sha256=_sha(b"inputs"),
        arm=module.TRAIN_ARM,
    )
    worker = module.Worker(session)
    worker.out = tmp_path / "worker"
    worker.out.mkdir()
    worker.deadline = 105.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 100.0))
    calls: list[list[str]] = []

    def capture_forbidden_stage(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", capture_forbidden_stage)

    with pytest.raises(module.WorkerError) as error:
        worker.run_stage(["must-not-run"], "reserved", reserve_seconds=5)
    assert error.value.reason == "session_deadline_reserve_reached"
    assert calls == []


@pytest.mark.parametrize(
    ("session_seconds", "expected_state", "training_expected"),
    [
        (10_800, "partial_checkpoint_preserved", True),
        (2_200, "training_deferred_insufficient_time", False),
    ],
)
def test_execute_persists_training_boundary_and_forwards_global_token_carry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_seconds: int,
    expected_state: str,
    training_expected: bool,
) -> None:
    module = _load_worker()
    carry_tokens = 9_876_543
    input_root, bundle, session, _ = _write_bundle(
        module,
        tmp_path,
        external_campaign_tokens=carry_tokens,
        session_seconds=session_seconds,
    )
    repo = tmp_path / "repo"
    fixture = repo / "data/benchmarks/code_completion_v2.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes((bundle / "causal200.jsonl").read_bytes())
    output = tmp_path / "output/q25_fim"
    monkeypatch.setattr(module, "INPUT_ROOT", input_root)
    monkeypatch.setattr(module, "OUT", output)
    monkeypatch.setattr(module, "REPO", repo)

    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.setenv(name, "")

    torch = ModuleType("torch")
    setattr(torch, "__version__", "mock-cpu-test")  # noqa: B010
    cuda_attribute = "cuda"
    setattr(
        torch,
        cuda_attribute,
        SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4 (mocked, no GPU used)",
            device_count=lambda: 1,
        ),
    )
    tinycomplete = ModuleType("tinycomplete")
    tinycomplete.__path__ = []
    code_cpt = ModuleType("tinycomplete.code_cpt")
    code_cpt.__path__ = []
    q25_fim = ModuleType("tinycomplete.code_cpt.q25_fim")

    def mock_verify_initializer(_model: Path, *, arm: str, entry: dict[str, Any]) -> dict[str, Any]:
        return {"arm": arm, "kind": entry["kind"], "files": entry["files"]}

    setattr(q25_fim, "verify_initializer", mock_verify_initializer)  # noqa: B010
    setattr(tinycomplete, "code_cpt", code_cpt)  # noqa: B010
    setattr(code_cpt, "q25_fim", q25_fim)  # noqa: B010
    for name, module_value in (
        ("torch", torch),
        ("tinycomplete", tinycomplete),
        ("tinycomplete.code_cpt", code_cpt),
        ("tinycomplete.code_cpt.q25_fim", q25_fim),
    ):
        monkeypatch.setitem(sys.modules, name, module_value)

    calls: list[list[str]] = []
    training_started_at_launch: bool | None = None
    startup_status: dict[str, Any] | None = None

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        nonlocal startup_status, training_started_at_launch
        calls.append(command)
        if command[1:3] == ["-m", "pip"]:
            startup_status = json.loads((output / "worker-status.json").read_text())
        if command[:3] == ["git", "-C", str(repo)] and command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=session["commit"] + "\n")
        if "--mode" in command and "--output" in command:
            result_dir = Path(command[command.index("--output") + 1])
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "result.json").write_text("{}\n")
            return SimpleNamespace(returncode=0)
        if "--execute" in command:
            training_status = json.loads((output / "worker-status.json").read_text())
            training_started_at_launch = training_status["training_started"]
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    inserted_source = str(repo / "src")
    try:
        worker = module.Worker(session)
        assert worker.execute() == 0
    finally:
        while inserted_source in sys.path:
            sys.path.remove(inserted_source)

    assert worker.status["state"] == expected_state
    assert worker.status["commit"] == session["commit"]
    assert worker.status["attempt"] == session["attempt"]
    assert worker.status["training_started"] is training_expected
    assert startup_status is not None
    assert startup_status["training_started"] is False
    assert startup_status["commit"] == session["commit"]
    assert startup_status["attempt"] == session["attempt"]
    preflight = next(
        command
        for command in calls
        if "tinycomplete.code_cpt.q25_fim" in command and "--execute" not in command
    )
    assert any(
        stage["name"] == "trainer-preflight" and stage["exit_code"] == 0
        for stage in worker.status["stages"]
    )
    commands_to_check = [preflight]
    if training_expected:
        training = next(command for command in calls if "--execute" in command)
        commands_to_check.append(training)
        assert worker.status["training_exit_code"] == 1
        assert training_started_at_launch is True
    else:
        assert not any("--execute" in command for command in calls)
        assert training_started_at_launch is None
    for command in commands_to_check:
        token_flag = command.index("--external-campaign-tokens")
        assert command[token_flag + 1] == str(carry_tokens)
