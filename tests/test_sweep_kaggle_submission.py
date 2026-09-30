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


def bundle(tmp_path):
    for directory in ("dataset", "kernel"):
        (tmp_path / directory).mkdir()
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
    diagnostic = tmp_path / "dataset/next_edit_fixtures.json"
    diagnostic.write_text("public synthetic fixture")
    spec = {
        "schema": "sweep-comparison-kaggle-input-v1", "commit": "a" * 40,
        "model_revision": preparer.worker.MODEL_REVISION,
        "model_sha256": preparer.worker.MODEL_SHA256,
        "runtime_revision": preparer.worker.RUNTIME_REVISION,
        "session_seconds": 7200, "reserve_seconds": 1200, "training_enabled": False,
        "runner_sha256": "b" * 64, "runner_arguments": ["--plan", "{input}/plan.json"],
        "runner_modes": ["download", "quantize", "quality", "next-edit"],
        "files": {"plan.json": manifest["plan_sha256"],
                  "next_edit_fixtures.json": preparer.worker.digest(diagnostic)},
    }
    spec_path = tmp_path / "dataset/sweep-worker-spec.json"
    spec_path.write_text(json.dumps(spec))
    manifest["input_spec_sha256"] = preparer.worker.digest(spec_path)
    (tmp_path / "bundle-manifest.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.mark.parametrize("state", ["submission_started", "kernel_pushed"])
def test_prevents_duplicate_or_ambiguous_submission(tmp_path, monkeypatch, state):
    path = bundle(tmp_path)
    (path / "submission-state.json").write_text(json.dumps({"state": state}))
    monkeypatch.setattr(preparer, "cli", lambda argv: pytest.fail("must not call Kaggle"))
    with pytest.raises(RuntimeError, match="already attempted"):
        preparer.submit(path)


@pytest.mark.parametrize("name", ["dataset/plan.json", "dataset/sweep-worker-spec.json",
                                   "kernel/run.py", "dataset/next_edit_fixtures.json"])
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
    (tmp_path / "submission-state.json").write_text(json.dumps({
        "state": "kernel_pushed", "kernel_id": preparer.KERNEL_ID,
    }))
    monkeypatch.setattr(preparer, "cli", lambda argv: status)
    with pytest.raises(RuntimeError, match="terminal ERROR"):
        preparer.validate_failed_attempt(tmp_path)


def test_retry_requires_reconciled_push_and_terminal_failure(tmp_path, monkeypatch):
    state = tmp_path / "submission-state.json"
    state.write_text(json.dumps({"state": "submission_started", "kernel_id": preparer.KERNEL_ID}))
    monkeypatch.setattr(preparer, "cli", lambda argv: pytest.fail("unknown push cannot retry"))
    with pytest.raises(RuntimeError, match="reconciled"):
        preparer.validate_failed_attempt(tmp_path)
    state.write_text(json.dumps({"state": "kernel_pushed", "kernel_id": preparer.KERNEL_ID}))
    from datetime import UTC, datetime, timedelta
    state.write_text(json.dumps({
        "state": "kernel_pushed", "kernel_id": preparer.KERNEL_ID,
        "quota_before": {"observed_at": (datetime.now(UTC) - timedelta(minutes=5)).isoformat()},
    }))
    monkeypatch.setattr(preparer, "cli", lambda argv: "KernelWorkerStatus.ERROR")
    preparer.validate_failed_attempt(tmp_path)
    assert json.loads((tmp_path / "terminal-observation.json").read_text())["state"] == "ERROR"


def test_retry_versions_private_dataset_and_preserves_old_version(tmp_path, monkeypatch):
    path = bundle(tmp_path)
    manifest_path = path / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(kernel_id=preparer.KERNEL_ID + "-retry", retry_after="previous")
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(preparer, "validate_failed_attempt", lambda previous: None)
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
    assert state["kernel_id"] == preparer.KERNEL_ID + "-retry"


def test_retry_stops_when_conservative_aggregate_wall_bound_is_too_large(tmp_path, monkeypatch):
    (tmp_path / "submission-state.json").write_text(json.dumps({
        "state": "kernel_pushed", "kernel_id": preparer.KERNEL_ID,
        "quota_before": {"observed_at": "2026-09-30T00:00:00+00:00"},
    }))
    (tmp_path / "terminal-observation.json").write_text(json.dumps({
        "kernel_id": preparer.KERNEL_ID, "state": "ERROR",
        "observed_at": "2026-09-30T03:00:00+00:00",
    }))
    monkeypatch.setattr(preparer, "cli", lambda argv: "KernelWorkerStatus.ERROR")
    with pytest.raises(RuntimeError, match="aggregate"):
        preparer.validate_failed_attempt(tmp_path)
