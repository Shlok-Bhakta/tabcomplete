"""Budget and artifact identity safeguards before a Sweep GPU allocation."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "kaggle/sweep_comparison_r1/run.py"
SPEC = importlib.util.spec_from_file_location("sweep_kaggle_worker", PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def valid_spec():
    return {
        "schema": "sweep-comparison-kaggle-input-v1",
        "model_revision": worker.MODEL_REVISION,
        "model_sha256": worker.MODEL_SHA256,
        "runtime_revision": worker.RUNTIME_REVISION,
        "session_seconds": 7200,
        "reserve_seconds": 1200,
        "training_enabled": False,
        "campaign_attempt_number": 1,
        "campaign_max_attempts": 3,
        "fifth_attempt_plan_sha256": None,
        "sixth_attempt_plan_sha256": None,
        "commit": "a" * 40,
        "files": {"plan.json": "b" * 64},
        "runner_sha256": "c" * 64,
        "runner_arguments": ["--plan", "{input}/plan.json"],
        "runner_modes": ["download", "quantize", "quality", "next-edit"],
    }


@pytest.mark.parametrize("field,value", [
    ("model_sha256", "a" * 64),
    ("model_revision", "a" * 40),
    ("runtime_revision", "a" * 40),
    ("training_enabled", True),
    ("reserve_seconds", 0),
    ("session_seconds", 7201),
    ("runner_arguments", [None]),
    ("runner_arguments", ["--mode", "download"]),
    ("runner_modes", ["train"]),
    ("runner_sha256", "invalid"),
    ("commit", "invalid"),
    ("files", {"../secret": "b" * 64}),
])
def test_rejects_unapproved_identity_and_input_paths(field, value):
    spec = valid_spec()
    spec[field] = value
    with pytest.raises(ValueError):
        worker.validate_spec(spec)


def test_accepts_exact_authorized_spec():
    worker.validate_spec(valid_spec())


def test_fifth_attempt_accepts_only_reduced_approved_session():
    spec = valid_spec()
    spec.update(
        session_seconds=worker.FIFTH_SESSION_SECONDS,
        campaign_attempt_number=5,
        campaign_max_attempts=5,
        fifth_attempt_plan_sha256="d" * 64,
    )
    spec["files"]["plan.json"] = "d" * 64
    worker.validate_spec(spec)


def test_sixth_attempt_accepts_only_two_hour_session_and_plan_hash():
    spec = valid_spec()
    spec.update(
        campaign_attempt_number=6,
        campaign_max_attempts=6,
        sixth_attempt_plan_sha256="e" * 64,
    )
    spec["files"]["plan.json"] = "e" * 64
    worker.validate_spec(spec)


@pytest.mark.parametrize(("attempt", "session", "max_attempts"), [
    (4, worker.FIFTH_SESSION_SECONDS, 5),
    (5, worker.SESSION_SECONDS, 5),
    (5, worker.FIFTH_SESSION_SECONDS, 4),
    (6, worker.FIFTH_SESSION_SECONDS, 6),
    (6, worker.SESSION_SECONDS, 5),
])
def test_rejects_session_override_without_fifth_attempt_identity(attempt, session, max_attempts):
    spec = valid_spec()
    spec.update(
        campaign_attempt_number=attempt,
        campaign_max_attempts=max_attempts,
        session_seconds=session,
    )
    with pytest.raises(ValueError, match="identity"):
        worker.validate_spec(spec)


def _fifth_plan(spec):
    import hashlib
    import json

    prior = [
        {
            "attempt_number": number,
            "kernel_id": kernel_id,
            "state": "ERROR",
            "conservative_wall_upper_bound_seconds": seconds,
        }
        for number, (kernel_id, seconds) in enumerate(zip(
            (
                "shlokbhakta/tabcomplete-sweep-comparison-r1",
                "shlokbhakta/tabcomplete-sweep-comparison-r1-retry",
                "shlokbhakta/tabcomplete-sweep-comparison-r1-verified",
                "shlokbhakta/tabcomplete-sweep-comparison-r1-final",
            ),
            (1352.229974, 3382.174313, 2330.930933, 2315.877205),
            strict=True,
        ), start=1)
    ]
    plan = {
        "schema": "sweep-comparison-plan-v1",
        "revision": {"number": 2},
        "quota_observation": {"renewal": "2026-10-03T00:00:00"},
        "code": {"runner_sha256": spec["runner_sha256"]},
        "allocation_amendment_v9": {
            "schema": "sweep-allocation-amendment-v9",
            "number": 9,
            "attempt_limits": {
                "maximum_total_attempts": 5,
                "attempts_already_used": 4,
                "additional_attempts_remaining": 1,
                "per_attempt_session_wall_seconds_max": 4800,
                "finalization_reserve_seconds": 1200,
                "maximum_work_seconds": 3600,
                "campaign_aggregate_gpu_session_wall_seconds_max": 14400,
                "fifth_attempt_allowed": True,
            },
            "verified_prior_attempts": prior,
            "prior_wall_upper_bound_seconds": 9381.212425,
            "aggregate_wall_upper_bound_with_fifth_attempt_seconds": 14181.212425,
            "remaining_aggregate_wall_margin_seconds": 218.787575,
        },
    }
    plan["plan_sha256"] = hashlib.sha256((json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()).hexdigest()
    return plan


def test_fifth_attempt_worker_requires_plan_nine_and_matching_budget(tmp_path):
    import json

    spec = valid_spec()
    spec.update(
        session_seconds=worker.FIFTH_SESSION_SECONDS,
        campaign_attempt_number=5,
        campaign_max_attempts=5,
        prior_wall_upper_bound_seconds=9381.212425,
        campaign_wall_cap_seconds=14400,
        fifth_attempt_plan_sha256="0" * 64,
    )
    plan = _fifth_plan(spec)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True, ensure_ascii=False) + "\n")
    plan_hash = worker.digest(plan_path)
    spec["fifth_attempt_plan_sha256"] = plan_hash
    spec["files"]["plan.json"] = plan_hash
    worker.validate_spec(spec)
    worker.validate_fifth_attempt_plan(plan_path, spec)

    changed_limits = json.loads(plan_path.read_text())
    changed_limits["allocation_amendment_v9"]["attempt_limits"]["maximum_work_seconds"] = 4000
    changed_limits["plan_sha256"] = "0" * 64
    changed_path = tmp_path / "changed-plan.json"
    changed_path.write_text(json.dumps(changed_limits, sort_keys=True) + "\n")
    changed_spec = {**spec, "fifth_attempt_plan_sha256": worker.digest(changed_path)}
    changed_spec["files"] = {**spec["files"], "plan.json": worker.digest(changed_path)}
    with pytest.raises(ValueError, match="self-hash"):
        worker.validate_fifth_attempt_plan(changed_path, changed_spec)


def _sixth_plan(spec):
    import hashlib
    import json

    identities = [
        "shlokbhakta/tabcomplete-sweep-comparison-r1",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-retry",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-verified",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-final",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-attempt-5",
    ]
    durations = [1352.229974, 3382.174313, 2330.930933, 2315.877205, 2013.138070]
    prior = [
        {
            "attempt_number": number,
            "kernel_id": kernel_id,
            "state": "ERROR",
            "quota_observed_at": f"2026-09-30T{7 + number:02}:00:00+00:00",
            "terminal_observed_at": f"2026-09-30T{7 + number:02}:30:00+00:00",
            "conservative_wall_upper_bound_seconds": duration,
            "terminal_observation_sha256": f"{number}" * 64,
        }
        for number, (kernel_id, duration) in enumerate(zip(identities, durations, strict=True), 1)
    ]
    plan = {
        "schema": "sweep-comparison-plan-v1",
        "revision": {"number": 2},
        "quota_observation": {"renewal": "2026-10-03T00:00:00"},
        "code": {
            "runner_sha256": spec["runner_sha256"],
            "test_sha256": "f" * 64,
            "native_provider_sha256": "e" * 64,
        },
        "allocation_amendment_v10": {
            "schema": "sweep-allocation-amendment-v10",
            "number": 10,
            "attempt_limits": {
                "maximum_total_attempts": 6,
                "attempts_already_used": 5,
                "additional_attempts_remaining": 1,
                "per_attempt_session_wall_seconds_max": 7200,
                "finalization_reserve_seconds": 1200,
                "maximum_work_seconds": 6000,
                "campaign_aggregate_gpu_session_wall_seconds_max": 21600,
                "sixth_attempt_allowed": True,
            },
            "authorization": {
                "basis": "User explicitly authorized one additional free Kaggle allocation.",
                "additional_model_downloads": False,
                "automatic_renewal_consumption": False,
                "maximum_additional_allocations": 1,
                "paid_compute": False,
                "training": False,
            },
            "quota_gate": {
                "all_relevant_job_statuses_verified": True,
                "automatic_renewal_consumption": False,
                "minimum_remaining_gpu_hours": 2.0,
                "refresh_immediately_before_submission": True,
                "require_no_active_gpu_jobs": True,
                "require_same_renewal_as_campaign": True,
            },
            "fresh_quota_observation_at_freeze": {
                "observed_at": "2026-09-30T14:00:00+00:00",
                "remaining": 42.0,
                "renewal": "2026-10-03T00:00:00",
                "active_jobs": [],
                "all_relevant_job_statuses_verified": True,
                "source": "authenticated kaggle quota --format json and kernel statuses",
            },
            "attempt_005_failure_evidence": {
                "manifest_sha256": "a" * 64,
                "terminal_observation_sha256": "b" * 64,
                "kernel_id": "shlokbhakta/tabcomplete-sweep-comparison-r1-attempt-5",
                "state": "ERROR",
                "predictions": 0,
                "model_quality_assessed": False,
                "conservative_wall_upper_bound_seconds": 2013.138070,
            },
            "session_enforcement": {
                "finalization_reserve_seconds": 1200,
                "kaggle_push_timeout_seconds": 7200,
                "monotonic_worker_work_deadline_seconds": 6000,
                "native_per_request_timeout_seconds_max": 120,
                "setup_compile_evaluation_and_saving_included": True,
                "no_automatic_seventh_allocation": True,
            },
            "verified_prior_attempts": prior,
            "prior_wall_upper_bound_seconds": 11394.350495,
            "aggregate_wall_upper_bound_with_sixth_attempt_seconds": 18594.350495,
            "remaining_aggregate_wall_margin_seconds": 3005.649505,
        },
    }
    plan["plan_sha256"] = hashlib.sha256((json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()).hexdigest()
    return plan


def test_sixth_attempt_worker_requires_plan_ten_and_exact_budget(tmp_path):
    import json

    spec = valid_spec()
    spec.update(
        campaign_attempt_number=6,
        campaign_max_attempts=6,
        prior_wall_upper_bound_seconds=11394.350495,
        campaign_wall_cap_seconds=21600,
        sixth_attempt_plan_sha256="0" * 64,
    )
    plan = _sixth_plan(spec)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True, ensure_ascii=False) + "\n")
    plan_hash = worker.digest(plan_path)
    spec["sixth_attempt_plan_sha256"] = plan_hash
    spec["files"]["plan.json"] = plan_hash
    worker.validate_spec(spec)
    worker.validate_sixth_attempt_plan(plan_path, spec)

    changed = json.loads(plan_path.read_text())
    changed["allocation_amendment_v10"]["attempt_limits"]["maximum_total_attempts"] = 7
    changed["plan_sha256"] = "0" * 64
    changed_path = tmp_path / "changed-plan.json"
    changed_path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    changed_spec = {**spec, "sixth_attempt_plan_sha256": worker.digest(changed_path)}
    changed_spec["files"] = {**spec["files"], "plan.json": worker.digest(changed_path)}
    with pytest.raises(ValueError, match="self-hash"):
        worker.validate_sixth_attempt_plan(changed_path, changed_spec)


def test_configure_session_applies_fifth_work_deadline(monkeypatch):
    monkeypatch.setattr(worker, "START", 100.0)
    monkeypatch.setattr(worker, "SESSION_SECONDS", 7200)
    monkeypatch.setattr(worker, "DEADLINE", 6100.0)
    spec = valid_spec()
    spec.update(
        session_seconds=worker.FIFTH_SESSION_SECONDS,
        campaign_attempt_number=5,
        campaign_max_attempts=5,
        fifth_attempt_plan_sha256="d" * 64,
    )
    spec["files"]["plan.json"] = "d" * 64
    worker.configure_session(spec)
    assert worker.SESSION_SECONDS == 4800
    assert worker.DEADLINE == 3700.0
    assert worker.remaining_seconds(worker.START + 3599) == 1
    with pytest.raises(TimeoutError):
        worker.remaining_seconds(worker.START + 3600)


def test_configure_session_applies_sixth_two_hour_cap_and_reserve(monkeypatch):
    monkeypatch.setattr(worker, "START", 100.0)
    monkeypatch.setattr(worker, "SESSION_SECONDS", 4800)
    monkeypatch.setattr(worker, "DEADLINE", 3700.0)
    spec = valid_spec()
    spec.update(
        campaign_attempt_number=6,
        campaign_max_attempts=6,
        sixth_attempt_plan_sha256="e" * 64,
    )
    spec["files"]["plan.json"] = "e" * 64
    worker.configure_session(spec)
    assert worker.SESSION_SECONDS == 7200
    assert worker.DEADLINE == 6100.0
    assert worker.remaining_seconds(worker.START + 5999) == 1
    with pytest.raises(TimeoutError):
        worker.remaining_seconds(worker.START + 6000)


def test_finalization_reserve_is_not_inference_time():
    assert worker.remaining_seconds(worker.START + 5999) == 1
    with pytest.raises(TimeoutError):
        worker.remaining_seconds(worker.START + 6000)


def test_reexec_preserves_original_session_clock(monkeypatch):
    monkeypatch.setenv("TABCOMPLETE_SWEEP_SESSION_START", "100")
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    assert module.START == 100
    assert module.DEADLINE == 6100


def test_timeout_terminates_owned_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "OUT", tmp_path)
    monkeypatch.setattr(worker, "remaining_seconds", lambda: 0.5)
    pid_file = tmp_path / "child-pid"
    program = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(TimeoutError):
        worker.run([sys.executable, "-c", program, str(pid_file)], "timeout-test")
    assert pid_file.exists()
    child = int(pid_file.read_text())
    for _ in range(50):
        stat = Path(f"/proc/{child}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        os.kill(child, 9)
        pytest.fail("stage timeout left an owned child running")


def test_canonical_q4_staging_checks_identity_and_reuses(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    artifact = source / worker.Q4_FILE
    artifact.write_bytes(b"authorized-test-artifact")
    monkeypatch.setattr(worker, "Q4_BYTES", artifact.stat().st_size)
    monkeypatch.setattr(worker, "Q4_SHA256", worker.digest(artifact))
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(worker, "STORAGE_ROOT", scratch)
    worker.stage_canonical_q4(source, scratch)
    assert (scratch / worker.Q4_FILE).read_bytes() == artifact.read_bytes()
    worker.stage_canonical_q4(source, scratch)
    (scratch / worker.Q4_FILE).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="existing worker"):
        worker.stage_canonical_q4(source, scratch)


def test_bad_canonical_q4_never_stages(tmp_path):
    source = tmp_path / worker.Q4_FILE
    source.write_bytes(b"wrong model")
    scratch = tmp_path / "scratch"
    with pytest.raises(ValueError, match="identity"):
        worker.stage_canonical_q4(tmp_path, scratch)
    assert not scratch.exists()


def test_storage_budget_accounts_existing_files_before_more_work(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "STORAGE_ROOT", tmp_path)
    monkeypatch.setattr(worker, "STORAGE_CAP", 100)
    (tmp_path / "existing").write_bytes(b"x" * 80)
    with pytest.raises(RuntimeError, match="cap"):
        worker.check_storage(21)
    worker.check_storage(20)


def test_existing_cuda_driver_library_is_used_without_installation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"existing driver")
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=f"libcuda.so.1 (libc6,x86-64) => {driver}\n"))
    flags, evidence = worker.cuda_driver_configuration()
    assert flags == [f"-DCUDA_cuda_driver_LIBRARY={driver}"]
    assert evidence["virtual_memory_management"] is True


def test_missing_driver_uses_supported_non_vmm_cuda_path(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    flags, evidence = worker.cuda_driver_configuration()
    assert flags == ["-DGGML_CUDA_NO_VMM=ON"]
    assert evidence["virtual_memory_management"] is False
