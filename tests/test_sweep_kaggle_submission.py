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
        "session_seconds": preparer.SESSION_SECONDS, "reserve_seconds": 1200,
        "training_enabled": False, "campaign_attempt_number": attempt,
        "campaign_max_attempts": preparer.MAX_KERNEL_ATTEMPTS,
        "prior_wall_upper_bound_seconds": prior_wall,
        "campaign_wall_cap_seconds": preparer.AGGREGATE_WALL_SECONDS,
        "runner_sha256": "b" * 64, "runner_arguments": ["--plan", "{input}/plan.json"],
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
        spec_path.write_text(json.dumps(spec))
        manifest["input_spec_sha256"] = preparer.worker.digest(spec_path)
    manifest.update({
        "kernel_id": kernel_id, "kernel_title": kernel_title, "attempt_number": attempt,
        "maximum_attempts": preparer.MAX_KERNEL_ATTEMPTS,
        "aggregate_wall_cap_seconds": preparer.AGGREGATE_WALL_SECONDS,
        "planned_session_seconds": preparer.SESSION_SECONDS,
        "retry_after": str(retry_after) if retry_after is not None else None,
        "prior_attempts": prior_attempts,
        "prior_wall_upper_bound_seconds": prior_wall,
        "aggregate_wall_upper_bound_with_this_session_seconds": (
            prior_wall + preparer.SESSION_SECONDS
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
        plan, sort_keys=True, ensure_ascii=False, separators=(",", ":")
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
    parent = None if attempt == 1 else str(path.parent / "attempt-1")
    (path / "bundle-manifest.json").write_text(json.dumps({
        "renewal": "2026-10-03T00:00:00", "retry_after": parent,
        "kernel_id": kernel_id, "kernel_title": title,
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


def test_attempt_cap_has_no_fourth_kernel_identity():
    with pytest.raises(RuntimeError, match="three allocations"):
        preparer.kernel_identity(4)
    for attempt in (1, 2, 3):
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


def re_slug(value):
    import re

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
