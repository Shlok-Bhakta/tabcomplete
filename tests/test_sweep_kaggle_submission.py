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
        return "[]"

    monkeypatch.setattr(preparer, "cli", fake_cli)
    with pytest.raises(RuntimeError, match="simulated"):
        preparer.submit(path)
    monkeypatch.setattr(preparer, "cli", lambda argv: pytest.fail("no blind second push"))
    with pytest.raises(RuntimeError, match="already attempted"):
        preparer.submit(path)
