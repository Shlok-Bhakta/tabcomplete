"""No real Kaggle calls: reject ambiguous submission state and stale quota."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "kaggle/sweep_comparison_r1/prepare.py"
SPEC = importlib.util.spec_from_file_location("sweep_prepare", PATH)
assert SPEC and SPEC.loader
preparer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparer)


def bundle(tmp_path, *, attempt=1, retry_after=None, prior_attempts=None):
    for directory in ("dataset", "kernel"):
        (tmp_path / directory).mkdir(exist_ok=True)
    prior_attempts = prior_attempts or []
    kernel_id, kernel_title = preparer.kernel_identity(attempt)
    prior_wall = sum(row["conservative_wall_upper_bound_seconds"] for row in prior_attempts)
    files = {
        "plan_sha256": "dataset/plan.json",
        "input_spec_sha256": "dataset/sweep-worker-spec.json",
        "worker_sha256": "kernel/run.py",
    }
    manifest = {"renewal": "2026-10-03T00:00:00"}
    for key, relative in files.items():
        path = tmp_path / relative
        path.write_text("public synthetic fixture")
        manifest[key] = preparer.worker.digest(path)
    diagnostic = tmp_path / "dataset/next_edit_inputs.jsonl"
    diagnostic.write_text("public synthetic fixture")
    spec = {
        "schema": "sweep-comparison-kaggle-input-v1", "commit": "a" * 40,
        "model_revision": preparer.worker.MODEL_REVISION,
        "model_sha256": preparer.worker.MODEL_SHA256,
        "runtime_revision": preparer.worker.RUNTIME_REVISION,
        "session_seconds": preparer._session_seconds_for_attempt(attempt),
        "reserve_seconds": preparer.RESERVE_SECONDS,
        "training_enabled": False, "campaign_attempt_number": attempt,
        "campaign_max_attempts": preparer._campaign_attempt_cap(attempt),
        "prior_wall_upper_bound_seconds": prior_wall,
        "campaign_wall_cap_seconds": preparer._aggregate_wall_cap_for_attempt(attempt),
        "runner_sha256": preparer.worker.digest(
            preparer.ROOT / "scripts/run_sweep_comparison.py"
        ),
        "fifth_attempt_plan_sha256": None,
        "sixth_attempt_plan_sha256": None,
        "runner_arguments": ["--plan", "{input}/plan.json"],
        "runner_modes": ["download", "quantize", "quality", "next-edit"],
        "files": {"plan.json": manifest["plan_sha256"],
                  "next_edit_inputs.jsonl": preparer.worker.digest(diagnostic)},
    }
    spec_path = tmp_path / "dataset/sweep-worker-spec.json"
    spec_path.write_text(json.dumps(spec))
    manifest["input_spec_sha256"] = preparer.worker.digest(spec_path)
    if attempt == 3:
        plan_value = synthetic_plan(prior_attempts, diagnostic)
        plan_path = tmp_path / "dataset/plan.json"
        plan_path.write_text(json.dumps(plan_value, indent=2, sort_keys=True) + "\n")
        manifest["plan_sha256"] = preparer.worker.digest(plan_path)
        spec["files"]["plan.json"] = manifest["plan_sha256"]
        spec["files"]["next_edit_inputs.jsonl"] = preparer.worker.digest(diagnostic)
        spec_path.write_text(json.dumps(spec))
        manifest["input_spec_sha256"] = preparer.worker.digest(spec_path)
    elif attempt == 4:
        source_fixtures = (
            preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
        )
        diagnostic.write_bytes(source_fixtures.read_bytes())
        plan_value = synthetic_plan_v8(prior_attempts, diagnostic)
        plan_path = tmp_path / "dataset/plan.json"
        plan_path.write_text(json.dumps(plan_value, indent=2, sort_keys=True) + "\n")
        manifest["plan_sha256"] = preparer.worker.digest(plan_path)
        spec["files"]["plan.json"] = manifest["plan_sha256"]
        spec["files"]["next_edit_inputs.jsonl"] = preparer.worker.digest(diagnostic)
        spec_path.write_text(json.dumps(spec))
        manifest["input_spec_sha256"] = preparer.worker.digest(spec_path)
    elif attempt == 5:
        source_fixtures = (
            preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
        )
        diagnostic.write_bytes(source_fixtures.read_bytes())
        plan_value = synthetic_plan_v9(prior_attempts, diagnostic)
        plan_path = tmp_path / "dataset/plan.json"
        plan_path.write_text(json.dumps(plan_value, indent=2, sort_keys=True) + "\n")
        manifest["plan_sha256"] = preparer.worker.digest(plan_path)
        spec["fifth_attempt_plan_sha256"] = manifest["plan_sha256"]
        spec["files"]["plan.json"] = manifest["plan_sha256"]
        spec["files"]["next_edit_inputs.jsonl"] = preparer.worker.digest(diagnostic)
        spec_path.write_text(json.dumps(spec))
        manifest["input_spec_sha256"] = preparer.worker.digest(spec_path)
    elif attempt == 6:
        source_fixtures = (
            preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
        )
        diagnostic.write_bytes(source_fixtures.read_bytes())
        plan_value = synthetic_plan_v10(prior_attempts, diagnostic)
        plan_path = tmp_path / "dataset/plan.json"
        plan_path.write_text(json.dumps(plan_value, indent=2, sort_keys=True) + "\n")
        manifest["plan_sha256"] = preparer.worker.digest(plan_path)
        spec["sixth_attempt_plan_sha256"] = manifest["plan_sha256"]
        spec["campaign_wall_cap_seconds"] = preparer.SIXTH_AGGREGATE_WALL_SECONDS
        spec["files"]["plan.json"] = manifest["plan_sha256"]
        spec["files"]["next_edit_inputs.jsonl"] = preparer.worker.digest(diagnostic)
        spec_path.write_text(json.dumps(spec))
        manifest["input_spec_sha256"] = preparer.worker.digest(spec_path)
    manifest.update({
        "kernel_id": kernel_id, "kernel_title": kernel_title, "attempt_number": attempt,
        "maximum_attempts": preparer._campaign_attempt_cap(attempt),
        "aggregate_wall_cap_seconds": preparer._aggregate_wall_cap_for_attempt(attempt),
        "planned_session_seconds": preparer._session_seconds_for_attempt(attempt),
        "retry_after": str(retry_after) if retry_after is not None else None,
        "prior_attempts": prior_attempts,
        "prior_wall_upper_bound_seconds": prior_wall,
        "aggregate_wall_upper_bound_with_this_session_seconds": (
            prior_wall + preparer._session_seconds_for_attempt(attempt)
        ),
    })
    (tmp_path / "bundle-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "kernel/kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": kernel_title,
    }))
    return tmp_path


def synthetic_plan(prior_attempts, fixtures):
    import hashlib

    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    plan = {
        "schema": "sweep-comparison-plan-v1",
        "code": {
            "runner_sha256": preparer.worker.digest(runner),
            "test_sha256": preparer.worker.digest(
                preparer.ROOT / "tests/test_sweep_comparison.py"
            ),
        },
        "comparison": {"next_edit": {
            "fixture_input_sha256": preparer.worker.digest(fixtures),
        }},
        "allocation_amendment_v6": {
            "schema": "sweep-allocation-amendment-v6",
            "attempt_limits": {
                "maximum_total_attempts": 3,
                "attempts_already_used": 2,
                "additional_attempts_remaining": 1,
                "per_attempt_session_wall_seconds_max": preparer.SESSION_SECONDS,
                "campaign_aggregate_gpu_session_wall_seconds_max": (
                    preparer.AGGREGATE_WALL_SECONDS
                ),
                "third_attempt_allowed": True,
            },
            "verified_prior_attempts": prior_attempts,
            "allocation_source_identity": {
                "prepare_sha256": preparer.worker.digest(Path(preparer.__file__)),
                "prepare_test_sha256": preparer.worker.digest(
                    preparer.ROOT / "tests/test_sweep_kaggle_submission.py"
                ),
                "worker_sha256": preparer.worker.digest(
                    Path(preparer.__file__).with_name("run.py")
                ),
                "worker_test_sha256": preparer.worker.digest(
                    preparer.ROOT / "tests/test_sweep_kaggle_worker.py"
                ),
            },
            "aggregate_wall_upper_bound_with_third_attempt_seconds": round(
                sum(row["conservative_wall_upper_bound_seconds"]
                    for row in prior_attempts) + preparer.SESSION_SECONDS, 6
            ),
        },
    }
    canonical = json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"
    plan["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return plan


def synthetic_plan_v8(prior_attempts, fixtures):
    import hashlib

    previous_path = preparer.ROOT / "reports/prototype/sweep_comparison_r1/plan-v7.json"
    previous = json.loads(previous_path.read_text())
    assert previous["comparison"]["next_edit"]["fixture_input_sha256"] == (
        preparer.worker.digest(fixtures)
    )
    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    test = preparer.ROOT / "tests/test_sweep_comparison.py"
    prior_wall = preparer._finite_wall_sum(prior_attempts)
    planned_total = round(prior_wall + preparer.SESSION_SECONDS, 6)
    source_paths = {
        "prepare_sha256": Path(preparer.__file__),
        "prepare_test_sha256": preparer.ROOT / "tests/test_sweep_kaggle_submission.py",
        "worker_sha256": Path(preparer.__file__).with_name("run.py"),
        "worker_test_sha256": preparer.ROOT / "tests/test_sweep_kaggle_worker.py",
    }
    plan = dict(previous)
    plan["code"] = dict(previous["code"])
    plan["code"]["runner_sha256"] = preparer.worker.digest(runner)
    plan["code"]["test_sha256"] = preparer.worker.digest(test)
    plan["allocation_amendment_v8"] = {
        "schema": "sweep-allocation-amendment-v8",
        "number": 8,
        "supersedes_plan_sha256": previous["plan_sha256"],
        "supersedes_plan_file_sha256": preparer.worker.digest(previous_path),
        "attempt_limits": {
            "maximum_total_attempts": 4,
            "attempts_already_used": 3,
            "additional_attempts_remaining": 1,
            "per_attempt_session_wall_seconds_max": preparer.SESSION_SECONDS,
            "campaign_aggregate_gpu_session_wall_seconds_max": (
                preparer.AGGREGATE_WALL_SECONDS
            ),
            "fourth_attempt_allowed": True,
        },
        "verified_prior_attempts": prior_attempts,
        "aggregate_wall_upper_bound_with_fourth_attempt_seconds": planned_total,
        "remaining_aggregate_wall_margin_seconds": round(
            preparer.AGGREGATE_WALL_SECONDS - planned_total, 6
        ),
        "allocation_source_identity": {
            key: preparer.worker.digest(path) for key, path in source_paths.items()
        },
    }
    canonical = json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"
    plan["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return plan


def synthetic_plan_v9(prior_attempts, fixtures):
    import hashlib

    previous_path = preparer.ROOT / "reports/prototype/sweep_comparison_r1/plan-v8.json"
    previous = json.loads(previous_path.read_text())
    assert previous["comparison"]["next_edit"]["fixture_input_sha256"] == (
        preparer.worker.digest(fixtures)
    )
    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    test = preparer.ROOT / "tests/test_sweep_comparison.py"
    prior_wall = preparer._finite_wall_sum(prior_attempts)
    planned_total = round(prior_wall + preparer.FIFTH_SESSION_SECONDS, 6)
    source_paths = {
        "prepare_sha256": Path(preparer.__file__),
        "prepare_test_sha256": preparer.ROOT / "tests/test_sweep_kaggle_submission.py",
        "worker_sha256": Path(preparer.__file__).with_name("run.py"),
        "worker_test_sha256": preparer.ROOT / "tests/test_sweep_kaggle_worker.py",
    }
    plan = dict(previous)
    plan["code"] = dict(previous["code"])
    plan["code"]["runner_sha256"] = preparer.worker.digest(runner)
    plan["code"]["test_sha256"] = preparer.worker.digest(test)
    plan["allocation_amendment_v9"] = {
        "schema": "sweep-allocation-amendment-v9",
        "number": 9,
        "supersedes_plan_sha256": previous["plan_sha256"],
        "supersedes_plan_file_sha256": preparer.worker.digest(previous_path),
        "attempt_limits": {
            "maximum_total_attempts": 5,
            "attempts_already_used": 4,
            "additional_attempts_remaining": 1,
            "per_attempt_session_wall_seconds_max": preparer.FIFTH_SESSION_SECONDS,
            "finalization_reserve_seconds": preparer.RESERVE_SECONDS,
            "maximum_work_seconds": (
                preparer.FIFTH_SESSION_SECONDS - preparer.RESERVE_SECONDS
            ),
            "campaign_aggregate_gpu_session_wall_seconds_max": (
                preparer.AGGREGATE_WALL_SECONDS
            ),
            "fifth_attempt_allowed": True,
        },
        "verified_prior_attempts": prior_attempts,
        "prior_wall_upper_bound_seconds": round(prior_wall, 6),
        "aggregate_wall_upper_bound_with_fifth_attempt_seconds": planned_total,
        "remaining_aggregate_wall_margin_seconds": round(
            preparer.AGGREGATE_WALL_SECONDS - planned_total, 6
        ),
        "allocation_source_identity": {
            key: preparer.worker.digest(path) for key, path in source_paths.items()
        },
    }
    canonical = json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"
    plan["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return plan


def sixth_attempt_evidence():
    plan = json.loads((
        preparer.ROOT / "reports/prototype/sweep_comparison_r1/plan-v9.json"
    ).read_text())
    rows = [dict(row) for row in plan["allocation_amendment_v9"]["verified_prior_attempts"]]
    rows.append({
        "attempt_number": 5,
        "kernel_id": preparer.KERNEL_ID + "-attempt-5",
        "state": "ERROR",
        "quota_observed_at": "2026-09-30T13:03:22.771137+00:00",
        "terminal_observed_at": "2026-09-30T13:36:55.909207+00:00",
        "conservative_wall_upper_bound_seconds": 2013.138070,
        "terminal_observation_sha256": (
            "d2fa5665f4436ae5fd1538089a07e730c045af6b573a15272555a51645fd1dc1"
        ),
    })
    return rows


def synthetic_plan_v10(prior_attempts, fixtures):
    import hashlib

    previous_path = preparer.ROOT / "reports/prototype/sweep_comparison_r1/plan-v9.json"
    previous = json.loads(previous_path.read_text())
    assert previous["comparison"]["next_edit"]["fixture_input_sha256"] == (
        preparer.worker.digest(fixtures)
    )
    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    test = preparer.ROOT / "tests/test_sweep_comparison.py"
    prior_wall = preparer._finite_wall_sum(prior_attempts)
    planned_total = round(prior_wall + preparer.SESSION_SECONDS, 6)
    source_paths = {
        "prepare_sha256": Path(preparer.__file__),
        "prepare_test_sha256": preparer.ROOT / "tests/test_sweep_kaggle_submission.py",
        "worker_sha256": Path(preparer.__file__).with_name("run.py"),
        "worker_test_sha256": preparer.ROOT / "tests/test_sweep_kaggle_worker.py",
        "native_provider_sha256": preparer.ROOT / "scripts/measure_r2_local.py",
    }
    plan = dict(previous)
    plan["code"] = dict(previous["code"])
    plan["code"]["runner_sha256"] = preparer.worker.digest(runner)
    plan["code"]["test_sha256"] = preparer.worker.digest(test)
    plan["code"]["native_provider_sha256"] = preparer.worker.digest(
        preparer.ROOT / "scripts/measure_r2_local.py"
    )
    plan["allocation_amendment_v10"] = {
        "schema": "sweep-allocation-amendment-v10",
        "number": 10,
        "supersedes_plan_sha256": previous["plan_sha256"],
        "supersedes_plan_file_sha256": preparer.worker.digest(previous_path),
        "attempt_limits": {
            "maximum_total_attempts": 6,
            "attempts_already_used": 5,
            "additional_attempts_remaining": 1,
            "per_attempt_session_wall_seconds_max": preparer.SESSION_SECONDS,
            "finalization_reserve_seconds": preparer.RESERVE_SECONDS,
            "maximum_work_seconds": preparer.SESSION_SECONDS - preparer.RESERVE_SECONDS,
            "campaign_aggregate_gpu_session_wall_seconds_max": (
                preparer.SIXTH_AGGREGATE_WALL_SECONDS
            ),
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
            "renewal": previous["quota_observation"]["renewal"],
            "active_jobs": [],
            "all_relevant_job_statuses_verified": True,
            "source": "authenticated kaggle quota --format json and kernel statuses",
        },
        "attempt_005_failure_evidence": {
            "manifest_sha256": preparer.worker.digest(
                preparer.ROOT
                / "reports/prototype/sweep_comparison_r1/setup_failure_005/artifact_manifest.json"
            ),
            "terminal_observation_sha256": preparer.worker.digest(
                preparer.ROOT
                / (
                    "reports/prototype/sweep_comparison_r1/"
                    "setup_failure_005/terminal-observation.json"
                )
            ),
            "kernel_id": preparer.KERNEL_ID + "-attempt-5",
            "state": "ERROR",
            "predictions": 0,
            "model_quality_assessed": False,
            "conservative_wall_upper_bound_seconds": 2013.138070,
        },
        "session_enforcement": {
            "finalization_reserve_seconds": preparer.RESERVE_SECONDS,
            "kaggle_push_timeout_seconds": preparer.SESSION_SECONDS,
            "monotonic_worker_work_deadline_seconds": (
                preparer.SESSION_SECONDS - preparer.RESERVE_SECONDS
            ),
            "native_per_request_timeout_seconds_max": 120,
            "setup_compile_evaluation_and_saving_included": True,
            "no_automatic_seventh_allocation": True,
        },
        "verified_prior_attempts": prior_attempts,
        "prior_wall_upper_bound_seconds": round(prior_wall, 6),
        "aggregate_wall_upper_bound_with_sixth_attempt_seconds": planned_total,
        "remaining_aggregate_wall_margin_seconds": round(
            preparer.SIXTH_AGGREGATE_WALL_SECONDS - planned_total, 6
        ),
        "allocation_source_identity": {
            key: preparer.worker.digest(path) for key, path in source_paths.items()
        },
    }
    canonical = json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"
    plan["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return plan


def rehash_plan(plan):
    import hashlib

    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"
    plan["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


def prior_attempt(path, attempt, *, start, submitted, terminal=None):
    kernel_id, title = preparer.kernel_identity(attempt)
    path.mkdir()
    (path / "kernel").mkdir()
    (path / "kernel/kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": title,
    }))
    parent = None if attempt == 1 else str(path.parent / f"attempt-{attempt - 1}")
    (path / "bundle-manifest.json").write_text(json.dumps({
        "renewal": "2026-10-03T00:00:00", "retry_after": parent,
        "kernel_id": kernel_id, "kernel_title": title,
        "maximum_attempts": preparer._campaign_attempt_cap(attempt),
    }))
    (path / "submission-state.json").write_text(json.dumps({
        "state": "kernel_pushed", "kernel_id": kernel_id,
        "submitted_at": submitted.isoformat(), "quota_before": {
            "observed_at": start.isoformat(), "renewal": "2026-10-03T00:00:00",
            "active_jobs": [], "remaining": 44.5,
        },
    }))
    if terminal is not None:
        (path / "terminal-observation.json").write_text(json.dumps({
            "kernel_id": kernel_id, "state": "ERROR",
            "observed_at": terminal.isoformat(),
            "source": "authenticated kaggle kernels status",
        }))
    return path


@pytest.mark.parametrize("state", ["submission_started", "kernel_pushed"])
@pytest.mark.parametrize("attempt", [1, 3])
def test_prevents_duplicate_or_ambiguous_submission(tmp_path, monkeypatch, state, attempt):
    evidence = []
    retry_after = None
    if attempt == 3:
        evidence = [
            {"conservative_wall_upper_bound_seconds": 600.0},
            {"conservative_wall_upper_bound_seconds": 600.0},
        ]
        retry_after = Path("previous")
    path = bundle(
        tmp_path, attempt=attempt, retry_after=retry_after, prior_attempts=evidence,
    )
    (path / "submission-state.json").write_text(json.dumps({"state": state}))
    monkeypatch.setattr(preparer, "cli", lambda argv: pytest.fail("must not call Kaggle"))
    with pytest.raises(RuntimeError, match="already attempted"):
        preparer.submit(path)


@pytest.mark.parametrize("name", ["dataset/plan.json", "dataset/sweep-worker-spec.json",
                                   "kernel/run.py", "dataset/next_edit_inputs.jsonl"])
def test_rejects_tampering_before_quota_or_allocation(tmp_path, monkeypatch, name):
    path = bundle(tmp_path)
    (path / name).write_text("modified")
    monkeypatch.setattr(preparer, "live_quota", lambda: pytest.fail("must not query quota"))
    with pytest.raises(ValueError, match="modified"):
        preparer.submit(path)


@pytest.mark.parametrize("changes", [
    {"remaining": 3.99}, {"remaining": None}, {"renewal": "2026-10-10T00:00:00"},
    {"active_jobs": [{"reference": "owned/job", "status": "RUNNING"}]},
    {"job_statuses": [{"reference": "owned/job", "status": "unknown"}]},
])
def test_rejects_insufficient_unknown_busy_or_renewed_quota(monkeypatch, changes):
    quota = {
        "remaining": 44.59, "renewal": "2026-10-03T00:00:00", "active_jobs": [],
        "job_statuses": [{"reference": "owned/job", "status": "COMPLETE"}],
    }
    quota.update(changes)
    monkeypatch.setattr(preparer, "live_quota", lambda: quota)
    with pytest.raises(RuntimeError, match="gate failed"):
        preparer.checked_quota("2026-10-03T00:00:00")


def test_records_unknown_push_outcome_before_command_and_refuses_retry(tmp_path, monkeypatch):
    path = bundle(tmp_path)
    monkeypatch.setattr(preparer, "checked_quota", lambda renewal: {"remaining": 44.59})

    def fake_cli(argv):
        if argv[:3] == ["kaggle", "kernels", "push"]:
            state = json.loads((path / "submission-state.json").read_text())
            assert state["state"] == "submission_started"
            raise RuntimeError("simulated ambiguous transport failure")
        if argv[:3] == ["kaggle", "datasets", "status"]:
            return "ready"
        return "[]"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    with pytest.raises(RuntimeError, match="simulated"):
        preparer.submit(path)
    monkeypatch.setattr(preparer, "cli", lambda argv: pytest.fail("no blind second push"))
    with pytest.raises(RuntimeError, match="already attempted"):
        preparer.submit(path)


def test_full_plan_identity_checked_without_gpu_setup(tmp_path):
    import hashlib
    runner = tmp_path / "runner"
    fixtures = tmp_path / "inputs"
    runner.write_text("public source")
    fixtures.write_text("public fixtures")
    plan = {
        "schema": "sweep-comparison-plan-v1",
        "code": {"runner_sha256": preparer.worker.digest(runner)},
        "comparison": {"next_edit": {"fixture_input_sha256": preparer.worker.digest(fixtures)}},
    }
    plan["plan_sha256"] = hashlib.sha256((json.dumps(
        plan, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()).hexdigest()
    preparer.validate_plan(plan, runner, fixtures)
    fixtures.write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        preparer.validate_plan(plan, runner, fixtures)
    plan["schema"] = "changed"
    with pytest.raises(ValueError, match="intact"):
        preparer.validate_plan(plan, runner, fixtures)


@pytest.mark.parametrize("status", ["KernelWorkerStatus.RUNNING", "KernelWorkerStatus.QUEUED"])
def test_retry_rejects_unfinished_initial_allocation(tmp_path, monkeypatch, status):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    path = prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(minutes=10),
        submitted=now - timedelta(minutes=9), terminal=None,
    )
    monkeypatch.setattr(preparer, "cli", lambda argv: status)
    with pytest.raises(RuntimeError, match="terminal ERROR"):
        preparer.validate_failed_attempt(path)


def test_retry_requires_reconciled_push_and_terminal_failure(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    path = prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(minutes=10),
        submitted=now - timedelta(minutes=9), terminal=None,
    )
    state = path / "submission-state.json"
    state_value = json.loads(state.read_text())
    state_value["state"] = "submission_started"
    state.write_text(json.dumps(state_value))
    monkeypatch.setattr(preparer, "cli", lambda argv: pytest.fail("unknown push cannot retry"))
    with pytest.raises(RuntimeError, match="reconciled"):
        preparer.validate_failed_attempt(path)
    state_value["state"] = "kernel_pushed"
    state.write_text(json.dumps(state_value))
    calls = []
    monkeypatch.setattr(
        preparer, "cli", lambda argv: calls.append(argv) or "KernelWorkerStatus.ERROR"
    )
    result = preparer.validate_failed_attempt(path)
    observation = json.loads((path / "terminal-observation.json").read_text())
    assert observation["state"] == "ERROR"
    assert observation["source"] == "authenticated kaggle kernels status"
    assert len(observation["status_output_sha256"]) == 64
    assert calls == [["kaggle", "kernels", "status", preparer.KERNEL_ID]]
    assert result[0]["conservative_wall_upper_bound_seconds"] >= 540


def test_retry_versions_private_dataset_and_preserves_old_version(tmp_path, monkeypatch):
    evidence = [{
        "attempt_number": 1, "kernel_id": preparer.KERNEL_ID, "state": "ERROR",
        "quota_observed_at": "2026-09-30T08:17:13.958025+00:00",
        "terminal_observed_at": "2026-09-30T08:39:46.187999+00:00",
        "conservative_wall_upper_bound_seconds": 1352.229974,
        "terminal_observation_sha256": "c" * 64,
    }]
    path = bundle(tmp_path, attempt=2, retry_after=Path("previous"), prior_attempts=evidence)
    manifest_path = path / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(preparer, "validate_failed_attempt", lambda previous: evidence)
    monkeypatch.setattr(preparer, "checked_quota", lambda renewal: {"remaining": 44.5})
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        if argv[:3] == ["kaggle", "datasets", "list"]:
            return json.dumps([{"ref": preparer.DATASET_ID}])
        if argv[:3] == ["kaggle", "datasets", "status"]:
            return "ready"
        return "[]"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    preparer.submit(path)
    assert any(a[:3] == ["kaggle", "datasets", "version"] for a in calls)
    assert not any("--delete-old-versions" in a or "-d" in a for a in calls)
    state = json.loads((path / "submission-state.json").read_text())
    assert state["state"] == "kernel_pushed"
    assert state["kernel_id"] == preparer.KERNEL_RETRY_ID
    metadata = json.loads((path / "kernel/kernel-metadata.json").read_text())
    assert metadata["title"] == "TabComplete Sweep comparison R1 retry"
    assert re_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]


def test_retry_stops_when_conservative_aggregate_wall_bound_is_too_large(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    path = prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(hours=3),
        submitted=now - timedelta(hours=2, minutes=59), terminal=now - timedelta(minutes=1),
    )
    monkeypatch.setattr(preparer, "cli", lambda argv: "KernelWorkerStatus.ERROR")
    with pytest.raises(RuntimeError, match="aggregate"):
        preparer.validate_failed_attempt(path)


def test_third_attempt_requires_both_prior_authenticated_errors_and_records_second(
    tmp_path, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(minutes=50),
        submitted=now - timedelta(minutes=49), terminal=now - timedelta(minutes=40),
    )
    second = prior_attempt(
        tmp_path / "attempt-2", 2, start=now - timedelta(minutes=30),
        submitted=now - timedelta(minutes=29), terminal=None,
    )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        return "KernelWorkerStatus.ERROR"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    history = preparer.validate_failed_attempt(second)
    assert [item["kernel_id"] for item in history] == [
        preparer.KERNEL_ID, preparer.KERNEL_RETRY_ID,
    ]
    assert calls == [
        ["kaggle", "kernels", "status", preparer.KERNEL_ID],
        ["kaggle", "kernels", "status", preparer.KERNEL_RETRY_ID],
    ]
    terminal = json.loads((second / "terminal-observation.json").read_text())
    assert terminal["kernel_id"] == preparer.KERNEL_RETRY_ID
    assert terminal["state"] == "ERROR"
    assert terminal["source"] == "authenticated kaggle kernels status"
    assert len(terminal["status_output_sha256"]) == 64
    assert len(history) == 2
    assert sum(row["conservative_wall_upper_bound_seconds"] for row in history) < 7200
    verified_id, verified_title = preparer.kernel_identity(3)
    assert verified_id == preparer.KERNEL_VERIFIED_ID
    assert verified_title == "TabComplete Sweep comparison R1 verified"
    assert re_slug(verified_title) == verified_id.split("/", 1)[1]


def test_third_attempt_fails_if_either_previous_kernel_is_not_terminal_error(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(minutes=50),
        submitted=now - timedelta(minutes=49), terminal=now - timedelta(minutes=40),
    )
    second = prior_attempt(
        tmp_path / "attempt-2", 2, start=now - timedelta(minutes=30),
        submitted=now - timedelta(minutes=29), terminal=None,
    )

    def fake_cli(argv):
        if argv[-1] == preparer.KERNEL_RETRY_ID:
            return "KernelWorkerStatus.RUNNING"
        return "KernelWorkerStatus.ERROR"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    with pytest.raises(RuntimeError, match="terminal ERROR"):
        preparer.validate_failed_attempt(second)
    assert not (second / "terminal-observation.json").exists()


def test_third_allocation_budget_counts_both_failures(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(hours=3),
        submitted=now - timedelta(hours=2, minutes=59), terminal=now - timedelta(hours=2),
    )
    second = prior_attempt(
        tmp_path / "attempt-2", 2, start=now - timedelta(hours=2),
        submitted=now - timedelta(hours=1, minutes=59), terminal=now - timedelta(minutes=1),
    )
    monkeypatch.setattr(preparer, "cli", lambda argv: "KernelWorkerStatus.ERROR")
    with pytest.raises(RuntimeError, match="aggregate"):
        preparer.validate_failed_attempt(second)


def test_attempt_cap_has_only_one_sixth_kernel_identity():
    with pytest.raises(RuntimeError, match="six allocations"):
        preparer.kernel_identity(7)
    for attempt in (1, 2, 3, 4, 5, 6):
        kernel_id, title = preparer.kernel_identity(attempt)
        assert re_slug(title) == kernel_id.split("/", 1)[1]


def test_third_submission_uses_verified_slug_and_frozen_budget(tmp_path, monkeypatch):
    evidence = [
        {
            "attempt_number": number, "kernel_id": kernel_id, "state": "ERROR",
            "quota_observed_at": f"2026-09-30T08:{number:02d}:00+00:00",
            "terminal_observed_at": f"2026-09-30T08:{number:02d}:10+00:00",
            "conservative_wall_upper_bound_seconds": 600.0,
            "terminal_observation_sha256": str(number) * 64,
        }
        for number, kernel_id in enumerate(
            (preparer.KERNEL_ID, preparer.KERNEL_RETRY_ID), start=1
        )
    ]
    path = bundle(tmp_path, attempt=3, retry_after=Path("attempt-2"), prior_attempts=evidence)
    monkeypatch.setattr(preparer, "validate_failed_attempt", lambda previous: evidence)
    monkeypatch.setattr(preparer, "checked_quota", lambda renewal: {"remaining": 44.5})
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        if argv[:3] == ["kaggle", "datasets", "list"]:
            return json.dumps([{"ref": preparer.DATASET_ID}])
        if argv[:3] == ["kaggle", "datasets", "status"]:
            return "ready"
        return "[]"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    preparer.submit(path)
    metadata = json.loads((path / "kernel/kernel-metadata.json").read_text())
    assert metadata["id"] == preparer.KERNEL_VERIFIED_ID
    assert metadata["title"] == "TabComplete Sweep comparison R1 verified"
    assert re_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    spec = json.loads((path / "dataset/sweep-worker-spec.json").read_text())
    assert spec["campaign_attempt_number"] == 3
    assert spec["campaign_max_attempts"] == 3
    assert spec["prior_wall_upper_bound_seconds"] == 1200.0
    assert spec["campaign_wall_cap_seconds"] == 14400
    assert any(call[:3] == ["kaggle", "kernels", "push"] for call in calls)


def test_plan_v6_fails_closed_on_missing_or_changed_prior_evidence(tmp_path):
    evidence = [
        {
            "attempt_number": number,
            "kernel_id": kernel_id,
            "state": "ERROR",
            "quota_observed_at": f"2026-09-30T08:{number:02d}:00+00:00",
            "terminal_observed_at": f"2026-09-30T08:{number:02d}:10+00:00",
            "conservative_wall_upper_bound_seconds": 600.0,
            "terminal_observation_sha256": str(number) * 64,
        }
        for number, kernel_id in enumerate(
            (preparer.KERNEL_ID, preparer.KERNEL_RETRY_ID), start=1
        )
    ]
    fixtures = tmp_path / "fixtures.jsonl"
    fixtures.write_text("synthetic fixture")
    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    frozen = synthetic_plan(evidence, fixtures)
    preparer.validate_plan(frozen, runner, fixtures, attempt_number=3, prior_attempts=evidence)

    changed_limits = json.loads(json.dumps(frozen))
    changed_limits["allocation_amendment_v6"]["attempt_limits"]["maximum_total_attempts"] = 4
    changed_evidence = json.loads(json.dumps(frozen))
    changed_evidence["allocation_amendment_v6"]["verified_prior_attempts"][0][
        "terminal_observation_sha256"
    ] = "f" * 64
    changed_aggregate = json.loads(json.dumps(frozen))
    changed_aggregate["allocation_amendment_v6"][
        "aggregate_wall_upper_bound_with_third_attempt_seconds"
    ] = 14399
    changed_source = json.loads(json.dumps(frozen))
    changed_source["allocation_amendment_v6"]["allocation_source_identity"][
        "prepare_sha256"
    ] = "f" * 64
    for changed in (changed_limits, changed_evidence, changed_aggregate, changed_source):
        rehash_plan(changed)
        with pytest.raises(ValueError, match="plan-v6"):
            preparer.validate_plan(
                changed, runner, fixtures, attempt_number=3, prior_attempts=evidence
            )

    with pytest.raises(ValueError, match="two verified earlier"):
        preparer.validate_plan(frozen, runner, fixtures, attempt_number=3, prior_attempts=[])


def fourth_attempt_evidence(walls=(600.0, 600.0, 600.0)):
    return [
        {
            "attempt_number": attempt,
            "kernel_id": preparer.kernel_identity(attempt)[0],
            "state": "ERROR",
            "quota_observed_at": f"2026-09-30T0{attempt}:00:00+00:00",
            "terminal_observed_at": f"2026-09-30T0{attempt}:10:00+00:00",
            "conservative_wall_upper_bound_seconds": wall,
            "terminal_observation_sha256": f"{attempt}" * 64,
        }
        for attempt, wall in enumerate(walls, start=1)
    ]


def fifth_attempt_evidence():
    plan = json.loads((
        preparer.ROOT / "reports/prototype/sweep_comparison_r1/plan-v8.json"
    ).read_text())
    rows = [dict(row) for row in plan["allocation_amendment_v8"]["verified_prior_attempts"]]
    rows.append({
        "attempt_number": 4,
        "kernel_id": preparer.KERNEL_FINAL_ID,
        "state": "ERROR",
        "quota_observed_at": "2026-09-30T11:33:23.479861+00:00",
        "terminal_observed_at": "2026-09-30T12:11:59.357066+00:00",
        "conservative_wall_upper_bound_seconds": 2315.877205,
        "terminal_observation_sha256": (
            "2538e631980ee8302059426e791de9830243eee1808f8948ef33ffb0141336e3"
        ),
    })
    return rows


def test_fourth_attempt_requires_three_authenticated_terminal_errors(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(minutes=50),
        submitted=now - timedelta(minutes=49), terminal=None,
    )
    prior_attempt(
        tmp_path / "attempt-2", 2, start=now - timedelta(minutes=30),
        submitted=now - timedelta(minutes=29), terminal=None,
    )
    third = prior_attempt(
        tmp_path / "attempt-3", 3, start=now - timedelta(minutes=10),
        submitted=now - timedelta(minutes=9), terminal=None,
    )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        return "KernelWorkerStatus.ERROR"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    history = preparer.validate_failed_attempt(third)
    assert [item["attempt_number"] for item in history] == [1, 2, 3]
    assert [item["kernel_id"] for item in history] == [
        preparer.KERNEL_ID, preparer.KERNEL_RETRY_ID, preparer.KERNEL_VERIFIED_ID,
    ]
    assert calls == [
        ["kaggle", "kernels", "status", preparer.KERNEL_ID],
        ["kaggle", "kernels", "status", preparer.KERNEL_RETRY_ID],
        ["kaggle", "kernels", "status", preparer.KERNEL_VERIFIED_ID],
    ]
    assert json.loads((third / "terminal-observation.json").read_text())["state"] == "ERROR"
    final_id, final_title = preparer.kernel_identity(4)
    assert final_id == preparer.KERNEL_FINAL_ID
    assert final_title == "TabComplete Sweep comparison R1 final"
    assert re_slug(final_title) == final_id.split("/", 1)[1]


def test_fourth_attempt_stops_on_any_non_error_prior_status(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(minutes=50),
        submitted=now - timedelta(minutes=49), terminal=None,
    )
    prior_attempt(
        tmp_path / "attempt-2", 2, start=now - timedelta(minutes=30),
        submitted=now - timedelta(minutes=29), terminal=None,
    )
    third = prior_attempt(
        tmp_path / "attempt-3", 3, start=now - timedelta(minutes=10),
        submitted=now - timedelta(minutes=9), terminal=None,
    )

    def fake_cli(argv):
        if argv[-1] == preparer.KERNEL_VERIFIED_ID:
            return "KernelWorkerStatus.RUNNING"
        return "KernelWorkerStatus.ERROR"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    with pytest.raises(RuntimeError, match="terminal ERROR"):
        preparer.validate_failed_attempt(third)
    assert not (third / "terminal-observation.json").exists()


def test_fourth_attempt_fails_closed_when_conservative_budget_does_not_fit(
    tmp_path, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    prior_attempt(
        tmp_path / "attempt-1", 1, start=now - timedelta(hours=7),
        submitted=now - timedelta(hours=6, minutes=59),
        terminal=now - timedelta(hours=5),
    )
    prior_attempt(
        tmp_path / "attempt-2", 2, start=now - timedelta(hours=5),
        submitted=now - timedelta(hours=4, minutes=59),
        terminal=now - timedelta(hours=3),
    )
    third = prior_attempt(
        tmp_path / "attempt-3", 3, start=now - timedelta(hours=3),
        submitted=now - timedelta(hours=2, minutes=59),
        terminal=now - timedelta(hours=1),
    )
    monkeypatch.setattr(preparer, "cli", lambda argv: "KernelWorkerStatus.ERROR")
    with pytest.raises(RuntimeError, match="aggregate"):
        preparer.validate_failed_attempt(third)


def test_fourth_attempt_budget_preserves_actual_remaining_margin():
    evidence = fourth_attempt_evidence((1352.229974, 3382.174313, 2330.930933))
    prior_wall = preparer._finite_wall_sum(evidence)
    planned_total = prior_wall + preparer.SESSION_SECONDS
    assert round(prior_wall, 6) == 7065.335220
    assert round(planned_total, 6) == 14265.335220
    assert round(preparer.AGGREGATE_WALL_SECONDS - planned_total, 6) == 134.664780


def test_fifth_attempt_budget_uses_four_actual_terminal_errors_and_one_hour_work_window():
    evidence = fifth_attempt_evidence()
    prior_wall = preparer._finite_wall_sum(evidence)
    planned_total = prior_wall + preparer.FIFTH_SESSION_SECONDS
    assert round(prior_wall, 6) == 9381.212425
    assert preparer.FIFTH_SESSION_SECONDS == 4800
    assert preparer.RESERVE_SECONDS == 1200
    assert preparer.FIFTH_SESSION_SECONDS - preparer.RESERVE_SECONDS == 3600
    assert round(planned_total, 6) == 14181.212425
    assert round(preparer.AGGREGATE_WALL_SECONDS - planned_total, 6) == 218.787575


def test_sixth_attempt_budget_uses_five_errors_and_one_hour_work_window():
    evidence = sixth_attempt_evidence()
    prior_wall = preparer._finite_wall_sum(evidence)
    planned_total = prior_wall + preparer.SESSION_SECONDS
    assert round(prior_wall, 6) == 11394.350495
    assert preparer.SESSION_SECONDS == 7200
    assert preparer.RESERVE_SECONDS == 1200
    assert preparer.SESSION_SECONDS - preparer.RESERVE_SECONDS == 6000
    assert preparer.SIXTH_AGGREGATE_WALL_SECONDS == 21600
    assert round(planned_total, 6) == 18594.350495
    assert round(preparer.SIXTH_AGGREGATE_WALL_SECONDS - planned_total, 6) == 3005.649505
    assert preparer._campaign_attempt_cap(6) == 6
    assert preparer.kernel_identity(6) == (
        preparer.KERNEL_ID + "-attempt-6",
        "TabComplete Sweep comparison R1 attempt 6",
    )
    assert re_slug(preparer.kernel_identity(6)[1]) == (
        preparer.kernel_identity(6)[0].split("/", 1)[1]
    )


def test_plan_v9_fails_closed_on_fourth_error_chain_or_budget_changes(tmp_path):
    evidence = fifth_attempt_evidence()
    fixtures = preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    frozen = synthetic_plan_v9(evidence, fixtures)
    preparer.validate_plan(frozen, runner, fixtures, attempt_number=5, prior_attempts=evidence)

    changed_limits = json.loads(json.dumps(frozen))
    changed_limits["allocation_amendment_v9"]["attempt_limits"][
        "maximum_work_seconds"
    ] = 4200
    changed_evidence = json.loads(json.dumps(frozen))
    changed_evidence["allocation_amendment_v9"]["verified_prior_attempts"][3][
        "terminal_observation_sha256"
    ] = "f" * 64
    changed_total = json.loads(json.dumps(frozen))
    changed_total["allocation_amendment_v9"][
        "aggregate_wall_upper_bound_with_fifth_attempt_seconds"
    ] += 1
    changed_margin = json.loads(json.dumps(frozen))
    changed_margin["allocation_amendment_v9"][
        "remaining_aggregate_wall_margin_seconds"
    ] -= 1
    changed_source = json.loads(json.dumps(frozen))
    changed_source["allocation_amendment_v9"]["allocation_source_identity"][
        "worker_sha256"
    ] = "f" * 64
    changed_science = json.loads(json.dumps(frozen))
    changed_science["comparison"]["input_budget"] = "changed outside plan-v8"
    for changed in (
        changed_limits, changed_evidence, changed_total, changed_margin,
        changed_source, changed_science,
    ):
        rehash_plan(changed)
        with pytest.raises(ValueError, match="plan-v9"):
            preparer.validate_plan(
                changed, runner, fixtures, attempt_number=5, prior_attempts=evidence
            )

    with pytest.raises(ValueError, match="four verified earlier"):
        preparer.validate_plan(frozen, runner, fixtures, attempt_number=5, prior_attempts=[])


def test_plan_v10_fails_closed_on_chain_budget_source_or_science_changes():
    evidence = sixth_attempt_evidence()
    fixtures = preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    frozen = synthetic_plan_v10(evidence, fixtures)
    preparer.validate_plan(frozen, runner, fixtures, attempt_number=6, prior_attempts=evidence)

    changed_limits = json.loads(json.dumps(frozen))
    changed_limits["allocation_amendment_v10"]["attempt_limits"][
        "maximum_total_attempts"
    ] = 7
    changed_evidence = json.loads(json.dumps(frozen))
    changed_evidence["allocation_amendment_v10"]["verified_prior_attempts"][4][
        "terminal_observation_sha256"
    ] = "f" * 64
    changed_budget = json.loads(json.dumps(frozen))
    changed_budget["allocation_amendment_v10"][
        "aggregate_wall_upper_bound_with_sixth_attempt_seconds"
    ] += 1
    changed_margin = json.loads(json.dumps(frozen))
    changed_margin["allocation_amendment_v10"][
        "remaining_aggregate_wall_margin_seconds"
    ] -= 1
    changed_source = json.loads(json.dumps(frozen))
    changed_source["allocation_amendment_v10"]["allocation_source_identity"][
        "native_provider_sha256"
    ] = "f" * 64
    changed_science = json.loads(json.dumps(frozen))
    changed_science["runtime"]["context_tokens"] = 123
    for changed in (
        changed_limits, changed_evidence, changed_budget, changed_margin,
        changed_source, changed_science,
    ):
        rehash_plan(changed)
        with pytest.raises(ValueError, match="plan-v10"):
            preparer.validate_plan(
                changed, runner, fixtures, attempt_number=6, prior_attempts=evidence
            )

    with pytest.raises(ValueError, match="five verified earlier"):
        preparer.validate_plan(frozen, runner, fixtures, attempt_number=6, prior_attempts=[])


def test_plan_v9_does_not_authorize_sixth_attempt():
    evidence = sixth_attempt_evidence()
    fixtures = preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
    frozen = synthetic_plan_v10(evidence, fixtures)
    del frozen["allocation_amendment_v10"]
    rehash_plan(frozen)
    with pytest.raises(ValueError, match="plan-v10"):
        preparer.validate_plan(
            frozen, preparer.ROOT / "scripts/run_sweep_comparison.py", fixtures,
            attempt_number=6, prior_attempts=evidence,
        )


def test_plan_v8_does_not_authorize_fifth_attempt():
    evidence = fifth_attempt_evidence()
    fixtures = preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
    frozen = synthetic_plan_v9(evidence, fixtures)
    del frozen["allocation_amendment_v9"]
    rehash_plan(frozen)
    with pytest.raises(ValueError, match="plan-v9"):
        preparer.validate_plan(
            frozen, preparer.ROOT / "scripts/run_sweep_comparison.py", fixtures,
            attempt_number=5, prior_attempts=evidence,
        )


def test_plan_v8_fails_closed_on_fourth_attempt_evidence_or_limit_changes(tmp_path):
    evidence = fourth_attempt_evidence()
    fixtures = preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
    runner = preparer.ROOT / "scripts/run_sweep_comparison.py"
    frozen = synthetic_plan_v8(evidence, fixtures)
    preparer.validate_plan(frozen, runner, fixtures, attempt_number=4, prior_attempts=evidence)

    changed_limits = json.loads(json.dumps(frozen))
    changed_limits["allocation_amendment_v8"]["attempt_limits"][
        "maximum_total_attempts"
    ] = 5
    changed_evidence = json.loads(json.dumps(frozen))
    changed_evidence["allocation_amendment_v8"]["verified_prior_attempts"][2][
        "terminal_observation_sha256"
    ] = "f" * 64
    changed_total = json.loads(json.dumps(frozen))
    changed_total["allocation_amendment_v8"][
        "aggregate_wall_upper_bound_with_fourth_attempt_seconds"
    ] += 1
    changed_margin = json.loads(json.dumps(frozen))
    changed_margin["allocation_amendment_v8"][
        "remaining_aggregate_wall_margin_seconds"
    ] -= 1
    changed_source = json.loads(json.dumps(frozen))
    changed_source["allocation_amendment_v8"]["allocation_source_identity"][
        "prepare_sha256"
    ] = "f" * 64
    for changed in (
        changed_limits, changed_evidence, changed_total, changed_margin, changed_source,
    ):
        rehash_plan(changed)
        with pytest.raises(ValueError, match="plan-v8"):
            preparer.validate_plan(
                changed, runner, fixtures, attempt_number=4, prior_attempts=evidence
            )

    with pytest.raises(ValueError, match="three verified earlier"):
        preparer.validate_plan(frozen, runner, fixtures, attempt_number=4, prior_attempts=[])
    changed_cap = json.loads(json.dumps(frozen))
    changed_cap["allocation_amendment_v8"]["attempt_limits"][
        "campaign_aggregate_gpu_session_wall_seconds_max"
    ] += 1
    rehash_plan(changed_cap)
    with pytest.raises(ValueError, match="plan-v8"):
        preparer.validate_plan(
            changed_cap, runner, fixtures, attempt_number=4, prior_attempts=evidence
        )


def test_plan_v7_does_not_authorize_fourth_attempt(tmp_path):
    evidence = fourth_attempt_evidence()
    fixtures = preparer.ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
    frozen = synthetic_plan_v8(evidence, fixtures)
    del frozen["allocation_amendment_v8"]
    rehash_plan(frozen)
    with pytest.raises(ValueError, match="plan-v8"):
        preparer.validate_plan(
            frozen, preparer.ROOT / "scripts/run_sweep_comparison.py", fixtures,
            attempt_number=4, prior_attempts=evidence,
        )


def test_fourth_submission_uses_unique_slug_and_four_hour_campaign_cap(tmp_path, monkeypatch):
    evidence = fourth_attempt_evidence()
    path = bundle(tmp_path, attempt=4, retry_after=Path("attempt-3"), prior_attempts=evidence)
    monkeypatch.setattr(preparer, "validate_failed_attempt", lambda previous: evidence)
    monkeypatch.setattr(preparer, "checked_quota", lambda renewal: {
        "remaining": 44.5, "renewal": renewal, "active_jobs": [],
    })
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        if argv[:3] == ["kaggle", "datasets", "list"]:
            return json.dumps([{"ref": preparer.DATASET_ID}])
        if argv[:3] == ["kaggle", "datasets", "status"]:
            return "ready"
        return "[]"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    preparer.submit(path)
    manifest = json.loads((path / "bundle-manifest.json").read_text())
    spec = json.loads((path / "dataset/sweep-worker-spec.json").read_text())
    metadata = json.loads((path / "kernel/kernel-metadata.json").read_text())
    assert manifest["attempt_number"] == 4
    assert manifest["maximum_attempts"] == 4
    assert manifest["aggregate_wall_upper_bound_with_this_session_seconds"] == 9000.0
    assert spec["campaign_attempt_number"] == 4
    assert spec["campaign_max_attempts"] == 4
    assert spec["prior_wall_upper_bound_seconds"] == 1800.0
    assert spec["campaign_wall_cap_seconds"] == 14400
    assert metadata["id"] == preparer.KERNEL_FINAL_ID
    assert metadata["title"] == "TabComplete Sweep comparison R1 final"
    assert re_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    push = next(call for call in calls if call[:3] == ["kaggle", "kernels", "push"])
    assert push[push.index("--timeout") + 1] == "7200"


def test_fifth_submission_uses_four_error_chain_and_4800_second_timeout(tmp_path, monkeypatch):
    evidence = fifth_attempt_evidence()
    path = bundle(tmp_path, attempt=5, retry_after=Path("attempt-4"), prior_attempts=evidence)
    monkeypatch.setattr(preparer, "validate_failed_attempt", lambda previous: evidence)
    monkeypatch.setattr(preparer, "checked_quota", lambda renewal: {
        "remaining": 43.46, "renewal": renewal, "active_jobs": [],
    })
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        if argv[:3] == ["kaggle", "datasets", "list"]:
            return json.dumps([{"ref": preparer.DATASET_ID}])
        if argv[:3] == ["kaggle", "datasets", "status"]:
            return "ready"
        return "[]"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    preparer.submit(path)
    manifest = json.loads((path / "bundle-manifest.json").read_text())
    spec = json.loads((path / "dataset/sweep-worker-spec.json").read_text())
    metadata = json.loads((path / "kernel/kernel-metadata.json").read_text())
    assert manifest["attempt_number"] == 5
    assert manifest["maximum_attempts"] == 5
    assert round(manifest["aggregate_wall_upper_bound_with_this_session_seconds"], 6) == (
        14181.212425
    )
    assert spec["campaign_attempt_number"] == 5
    assert spec["campaign_max_attempts"] == 5
    assert spec["session_seconds"] == 4800
    assert spec["reserve_seconds"] == 1200
    assert spec["prior_wall_upper_bound_seconds"] == pytest.approx(9381.212425)
    assert spec["fifth_attempt_plan_sha256"] == manifest["plan_sha256"]
    assert metadata["id"] == preparer.KERNEL_ID + "-attempt-5"
    assert metadata["title"] == "TabComplete Sweep comparison R1 attempt 5"
    assert re_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    push = next(call for call in calls if call[:3] == ["kaggle", "kernels", "push"])
    assert push[push.index("--timeout") + 1] == "4800"


def test_sixth_submission_uses_five_error_chain_and_7200_second_timeout(tmp_path, monkeypatch):
    evidence = sixth_attempt_evidence()
    path = bundle(tmp_path, attempt=6, retry_after=Path("attempt-5"), prior_attempts=evidence)
    monkeypatch.setattr(preparer, "validate_failed_attempt", lambda previous: evidence)
    quota_calls = []

    def checked_quota(renewal, minimum_gpu_hours=4.0):
        quota_calls.append((renewal, minimum_gpu_hours))
        return {"remaining": 42.0, "renewal": renewal, "active_jobs": []}

    monkeypatch.setattr(preparer, "checked_quota", checked_quota)
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        if argv[:3] == ["kaggle", "datasets", "list"]:
            return json.dumps([{"ref": preparer.DATASET_ID}])
        if argv[:3] == ["kaggle", "datasets", "status"]:
            return "ready"
        return "[]"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    preparer.submit(path)
    manifest = json.loads((path / "bundle-manifest.json").read_text())
    spec = json.loads((path / "dataset/sweep-worker-spec.json").read_text())
    metadata = json.loads((path / "kernel/kernel-metadata.json").read_text())
    assert manifest["attempt_number"] == 6
    assert manifest["maximum_attempts"] == 6
    assert manifest["aggregate_wall_cap_seconds"] == 21600
    assert round(manifest["aggregate_wall_upper_bound_with_this_session_seconds"], 6) == (
        18594.350495
    )
    assert spec["campaign_attempt_number"] == 6
    assert spec["campaign_max_attempts"] == 6
    assert spec["session_seconds"] == 7200
    assert spec["reserve_seconds"] == 1200
    assert spec["prior_wall_upper_bound_seconds"] == pytest.approx(11394.350495)
    assert spec["campaign_wall_cap_seconds"] == 21600
    assert spec["sixth_attempt_plan_sha256"] == manifest["plan_sha256"]
    assert quota_calls == [
        ("2026-10-03T00:00:00", 2.0),
        ("2026-10-03T00:00:00", 2.0),
    ]
    assert metadata["id"] == preparer.KERNEL_ID + "-attempt-6"
    assert metadata["title"] == "TabComplete Sweep comparison R1 attempt 6"
    assert re_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    push = next(call for call in calls if call[:3] == ["kaggle", "kernels", "push"])
    assert push[push.index("--timeout") + 1] == "7200"


def test_sixth_terminal_observation_accepts_authenticated_watcher_schema(tmp_path, monkeypatch):
    path = tmp_path / "attempt-5"
    path.mkdir()
    observation = {
        "authenticated_status_verified": True,
        "kernel_id": preparer.KERNEL_ID + "-attempt-5",
        "observed_at": "2026-09-30T13:36:55.909207+00:00",
        "query_source": "authenticated kaggle kernels status",
        "status_token": "KernelWorkerStatus.ERROR",
    }
    (path / "terminal-observation.json").write_text(json.dumps(observation))
    monkeypatch.setattr(preparer, "cli", lambda argv: "KernelWorkerStatus.ERROR")
    result = preparer._observe_terminal_error(preparer.KERNEL_ID + "-attempt-5", path)
    assert result == observation


def test_validate_failed_attempt_accepts_watcher_record_for_fifth_terminal_error(
    tmp_path, monkeypatch,
):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    last = None
    for attempt, minutes_ago in enumerate((110, 90, 70, 50, 30), start=1):
        start = now - timedelta(minutes=minutes_ago)
        path = tmp_path / f"attempt-{attempt}"
        last = prior_attempt(
            path, attempt, start=start, submitted=start + timedelta(seconds=1),
            terminal=start + timedelta(minutes=5),
        )
    assert last is not None
    watcher = {
        "authenticated_status_verified": True,
        "kernel_id": preparer.KERNEL_ID + "-attempt-5",
        "observed_at": (now - timedelta(minutes=25)).isoformat(),
        "query_source": "authenticated kaggle kernels status",
        "status_token": "KernelWorkerStatus.ERROR",
    }
    (last / "terminal-observation.json").write_text(json.dumps(watcher))
    monkeypatch.setattr(preparer, "cli", lambda argv: "KernelWorkerStatus.ERROR")
    history = preparer.validate_failed_attempt(last)
    assert len(history) == 5
    assert history[-1]["kernel_id"] == preparer.KERNEL_ID + "-attempt-5"
    assert history[-1]["terminal_observation_sha256"] == preparer.worker.digest(
        last / "terminal-observation.json"
    )
    assert preparer._finite_wall_sum(history) + preparer.SESSION_SECONDS < (
        preparer.SIXTH_AGGREGATE_WALL_SECONDS
    )


def re_slug(value):
    import re

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
