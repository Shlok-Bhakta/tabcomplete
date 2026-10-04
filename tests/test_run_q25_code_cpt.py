from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, timedelta
from datetime import datetime as RealDatetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))

import build_pilot  # noqa: E402
import run_q25_code_cpt as campaign  # noqa: E402


def _plan() -> dict[str, Any]:
    return {
        "configuration": {
            "budget": {
                "aggregate_session_seconds": 72_000,
                "automatic_renewal_use": False,
                "conservative_quota_multiplier": 2,
                "maximum_additional_training_input_tokens": 12_000_000,
                "maximum_discarded_replay_input_tokens": 4_000_000,
                "minimum_free_bytes": 2 * 1024**3,
                "new_artifact_bytes_cap": 12 * 1024**3,
                "paid_compute": False,
                "quota_account_gpu_hours_cap": 40,
                "quota_renewal": "2026-10-10T00:00:00",
                "session_seconds": 14_400,
            },
            "training": {
                "effective_batch": 16,
                "max_input_tokens": 8_000_000,
                "sequence_length": 1024,
            },
        }
    }


def _observation(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "active_jobs": [],
        "job_statuses": [],
        "observed_at": "2026-10-03T23:30:00+00:00",
        "remaining": 45.0,
        "renewal": "2026-10-10T00:00:00",
        "source": "authenticated kaggle quota and statuses",
        "units": "Kaggle account GPU-hours",
    }
    value.update(overrides)
    return value


def test_check_quota_accepts_the_frozen_free_window() -> None:
    campaign.check_quota(_plan(), _observation())


@pytest.mark.parametrize(
    "observation",
    [
        _observation(active_jobs=[{"reference": "owner/kernel", "status": "RUNNING"}]),
        _observation(job_statuses=[{"reference": "owner/kernel", "status": "unknown"}]),
        _observation(remaining=7.99),
        _observation(renewal="2026-10-17T00:00:00"),
        _observation(units="session wall-hours"),
        _observation(job_statuses=[{} for _ in range(100)]),
    ],
)
def test_check_quota_rejects_unverifiable_or_insufficient_allocations(
    observation: dict[str, Any],
) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        campaign.check_quota(_plan(), observation)


@pytest.mark.parametrize(
    ("budget_key", "value"),
    [("paid_compute", True), ("automatic_renewal_use", True)],
)
def test_check_quota_rejects_paid_compute_and_automatic_renewal(
    budget_key: str, value: bool
) -> None:
    plan = _plan()
    plan["configuration"]["budget"][budget_key] = value
    with pytest.raises(ValueError):
        campaign.check_quota(plan, _observation())


def _submission_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repository = tmp_path / "repo"
    worker_dir = repository / "kaggle/q25_code_cpt_r2"
    worker_dir.mkdir(parents=True)
    (worker_dir / "run.py").write_text("SESSION = __SESSION_JSON__\n")

    report = repository / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    plan = _plan()
    campaign.save(report / "plan.json", plan)

    artifacts = tmp_path / "artifacts"
    bundle = artifacts / "input-bundle"
    bundle.mkdir(parents=True)
    manifest = bundle / "input-manifest.json"
    campaign.save(manifest, {"schema": "synthetic"})
    campaign.save(
        artifacts / "dataset-submission.json",
        {
            "dataset": campaign.DATASET,
            "input_manifest_sha256": campaign.digest(manifest),
            "state": "verified",
        },
    )

    monkeypatch.setattr(campaign, "ROOT", repository)
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(campaign, "quota", lambda: _observation())
    monkeypatch.setattr(build_pilot, "_csv_refs", lambda _command: set())

    calls: list[tuple[str, ...]] = []
    commit = "f" * 40

    def fake_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        calls.append(call)
        if call[0] == "git" and call[-2:] == ("status", "--porcelain"):
            return ""
        if call[-2:] == ("rev-parse", "HEAD"):
            return commit
        if call[-2:] == ("origin", "refs/heads/research/q25-code-cpt-r2"):
            return f"{commit}\trefs/heads/research/q25-code-cpt-r2"
        return "push accepted"

    monkeypatch.setattr(campaign, "cli", fake_cli)
    return plan, report, artifacts, calls


def test_submit_requires_verified_upload_and_pushes_one_bounded_t4_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, artifacts, calls = _submission_context(tmp_path, monkeypatch)

    job = campaign.submit(plan, attempt=1, resume_source=None)

    push_calls = [call for call in calls if call[:3] == ("kaggle", "kernels", "push")]
    assert len(push_calls) == 1
    push = push_calls[0]
    assert "--timeout" in push
    assert str(plan["configuration"]["budget"]["session_seconds"]) in push
    assert "--accelerator" in push
    assert "NvidiaTeslaT4" in push
    assert not any(call[:2] == ("kaggle", "datasets") for call in calls)
    assert job["conservative_reserved_session_seconds"] == 14_400
    metadata = json.loads((artifacts / "kernel-attempt-1/kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == [campaign.DATASET, campaign.BASE_DATASET]
    assert metadata["kernel_sources"] == []
    assert json.loads((report / "job-1.json").read_text())["status"] == "submitted"


def test_submit_records_title_derived_reference_from_kaggle_push_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, _calls = _submission_context(tmp_path, monkeypatch)
    requested = "shlokbhakta/tc-q25-code-cpt-r2-a1"
    resolved = "shlokbhakta/tabcomplete-q25-code-cpt-r2-attempt-1"
    original_cli = campaign.cli

    def url_returning_cli(*args: str, **kwargs: Any) -> str:
        if args[:3] == ("kaggle", "kernels", "push"):
            return f"Kernel pushed: https://www.kaggle.com/code/{resolved}"
        return original_cli(*args, **kwargs)

    monkeypatch.setattr(campaign, "cli", url_returning_cli)

    job = campaign.submit(plan, attempt=1, resume_source=None)

    saved = json.loads((report / "job-1.json").read_text())
    assert job["requested_reference"] == requested
    assert job["reference"] == resolved
    assert saved["reference"] == resolved


def test_submit_refuses_missing_or_unverified_dataset_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report, artifacts, calls = _submission_context(tmp_path, monkeypatch)
    (artifacts / "dataset-submission.json").unlink()

    with pytest.raises(FileNotFoundError):
        campaign.submit(plan, attempt=1, resume_source=None)

    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in calls)


def test_submit_blocks_if_kernel_reference_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report, _artifacts, calls = _submission_context(tmp_path, monkeypatch)
    monkeypatch.setattr(
        build_pilot,
        "_csv_refs",
        lambda _command: {"shlokbhakta/tc-q25-code-cpt-r2-a1"},
    )

    with pytest.raises(ValueError, match="already exists"):
        campaign.submit(plan, attempt=1, resume_source=None)

    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in calls)


def test_submit_reserves_account_gpu_hours_before_any_remote_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report, _artifacts, calls = _submission_context(tmp_path, monkeypatch)
    plan["configuration"]["budget"]["quota_account_gpu_hours_cap"] = 7.99

    with pytest.raises(RuntimeError, match="account GPU-hour reservation"):
        campaign.submit(plan, attempt=1, resume_source=None)

    assert calls == []


def test_submit_rejects_session_crossing_unchanged_quota_renewal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report, _artifacts, calls = _submission_context(tmp_path, monkeypatch)

    class FixedDatetime:
        @classmethod
        def now(cls, tz: Any = None) -> RealDatetime:
            return RealDatetime(2026, 10, 9, 23, 0, tzinfo=tz or UTC)

        fromisoformat = staticmethod(RealDatetime.fromisoformat)

    monkeypatch.setattr(campaign, "datetime", FixedDatetime)

    with pytest.raises(RuntimeError, match="session deadline would cross"):
        campaign.submit(plan, attempt=1, resume_source=None)

    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in calls)


def test_submit_requires_immediately_prior_verified_complete_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, calls = _submission_context(tmp_path, monkeypatch)
    prior = "shlokbhakta/tc-q25-code-cpt-r2-a1"
    campaign.save(report / "job-1.json", {"reference": prior, "status": "submitted"})

    with pytest.raises(ValueError, match="immediately prior"):
        campaign.submit(plan, attempt=2, resume_source="shlokbhakta/unrelated-kernel")
    assert calls == []

    with pytest.raises(RuntimeError, match="verify the previous complete checkpoint"):
        campaign.submit(plan, attempt=2, resume_source=prior)
    assert calls == []


def test_submit_rejects_resume_while_previous_allocation_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, calls = _submission_context(tmp_path, monkeypatch)
    prior = "shlokbhakta/tc-q25-code-cpt-r2-a1"
    campaign.save(report / "job-1.json", {"reference": prior, "status": "submitted"})
    campaign.save(report / "verified-output-1.json", {"checkpoint_verified": True})
    monkeypatch.setattr(campaign, "cli", lambda *args, **kwargs: "RUNNING")

    with pytest.raises(RuntimeError, match="has not exited"):
        campaign.submit(plan, attempt=2, resume_source=prior)

    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in calls)


def test_submit_passes_verified_external_replay_reservation_to_worker_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, artifacts, _calls = _submission_context(tmp_path, monkeypatch)
    prior = "shlokbhakta/tc-q25-code-cpt-r2-a1"
    campaign.save(report / "job-1.json", {"reference": prior, "status": "submitted"})
    campaign.save(
        report / "verified-output-1.json",
        {
            "checkpoint_verified": True,
            "external_campaign_tokens_for_resume": 123_456,
        },
    )
    original_cli = campaign.cli

    def completed_prior_cli(*args: str, **kwargs: Any) -> str:
        if args[:3] == ("kaggle", "kernels", "status"):
            return "ERROR"
        return original_cli(*args, **kwargs)

    monkeypatch.setattr(campaign, "cli", completed_prior_cli)

    job = campaign.submit(plan, attempt=2, resume_source=prior)

    worker_source = (artifacts / "kernel-attempt-2/run.py").read_text()
    assert job["external_campaign_tokens"] == 123_456
    assert worker_source.count("'external_campaign_tokens': 123456") == 1
    metadata = json.loads((artifacts / "kernel-attempt-2/kernel-metadata.json").read_text())
    assert metadata["kernel_sources"] == [prior]


def test_q25_preflight_caps_8m_tokens_with_a_four_example_final_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch", reason="Q25 module currently imports its torch-backed loss")
    import numpy as np

    import tinycomplete.code_cpt.q25 as q25

    sequence_length = 1024
    train_path = tmp_path / "train.npy"
    development_path = tmp_path / "development.npy"
    np.save(train_path, np.zeros((7813, sequence_length), dtype=np.uint8))
    np.save(development_path, np.zeros((1, sequence_length), dtype=np.uint8))

    training = {
        "sequence_length": sequence_length,
        "effective_batch": 16,
        "microbatch_examples": 1,
        "max_input_tokens": 8_000_000,
        "epochs": 1,
        "learning_rate": 3e-6,
        "checkpoint_every_updates": 64,
        "gradient_checkpointing": True,
        "master_weights": "fp32",
        "compute": "fp16",
        "optimizer": "AdamW8bit",
        "loss": "example_mean_causal_all_source_positions_except_first",
        "cosine_floor_fraction": 0.1,
        "initial_loss_scale": 128,
        "attention": "sdpa",
        "seed": 314159,
    }
    document = {
        "configuration": {
            "model": {
                "id": q25.MODEL_ID,
                "revision": q25.MODEL_REVISION,
                "initializer": "untouched_pretrained",
            },
            "training": training,
            "data": {"sequence_length": sequence_length},
            "budget": {"finalization_reserve_seconds": 1200},
        }
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(document))
    monkeypatch.setattr(
        q25,
        "model_snapshot_identity",
        lambda _model, _document: {"model_id": q25.MODEL_ID, "files": {}},
    )
    example = q25.EncodedExample(
        input_ids=(0,) * sequence_length,
        labels=(0,) * sequence_length,
        prompt_tokens=0,
        response_tokens=sequence_length - 1,
        total_tokens=sequence_length,
        prompt="",
        response="",
    )
    monkeypatch.setattr(
        q25,
        "encode_raw_code_blocks",
        lambda blocks, **_kwargs: (example,) * len(blocks),
    )
    actual_training_batches = q25.training_batches
    observed_sizes: list[int] = []

    def capture_batch_sizes(examples, **kwargs):
        batches = actual_training_batches(examples, **kwargs)
        observed_sizes.extend(map(len, batches))
        return batches

    monkeypatch.setattr(q25, "training_batches", capture_batch_sizes)

    result = q25.run_training(
        model_path=tmp_path / "unused-local-model",
        train_path=train_path,
        development_path=development_path,
        plan_path=plan_path,
        output=tmp_path / "unused-output",
        session_seconds=7200,
        execute=False,
    )

    assert result["training_blocks"] == 7812
    assert result["training_input_tokens"] == 7_999_488
    assert result["updates"] == 489
    assert sorted(observed_sizes) == [4] + [16] * 488
    assert sum(observed_sizes) == 7812


def _collect_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "COMPLETE",
    committed_tokens: int = 1_048_576,
    logged_tokens: list[int] | None = None,
    external_tokens: int = 0,
    training_exit_code: int = 0,
    include_result: bool = True,
    include_updates: bool = True,
    updates_content: str | None = None,
    result_cursor_tokens: int | None = None,
    maximum_tokens: int = 12_000_000,
):
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    plan = _plan()
    plan["configuration"]["budget"]["maximum_additional_training_input_tokens"] = maximum_tokens
    campaign.save(
        report / "job-1.json",
        {
            "reference": "shlokbhakta/tc-q25-code-cpt-r2-a1",
            "external_campaign_tokens": external_tokens,
        },
    )
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=16 * 1024**3),
    )

    fingerprint = "frozen-training-fingerprint"
    checkpoint_name = "resume-step-000064.pt"
    checkpoint_bytes = b"synthetic complete resume state"
    log_values = logged_tokens if logged_tokens is not None else [committed_tokens]
    calls: list[tuple[str, ...]] = []

    def fake_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        calls.append(call)
        if call[:3] == ("kaggle", "kernels", "status"):
            return status
        if call[:3] != ("kaggle", "kernels", "output"):
            raise AssertionError(f"unexpected command: {call[:3]}")
        output = Path(call[call.index("-p") + 1])
        pattern = call[call.index("--file-pattern") + 1]
        training = output / "training"
        training.mkdir(parents=True, exist_ok=True)
        if pattern == r"\.(json|jsonl|log)$":
            pointer = {
                "path": checkpoint_name,
                "fingerprint": fingerprint,
                "cursor": {"training_input_tokens": committed_tokens},
            }
            campaign.save(training / "latest.json", pointer)
            campaign.save(
                training / (checkpoint_name + ".complete.json"),
                {
                    "sha256": campaign.hashlib.sha256(checkpoint_bytes).hexdigest(),
                    "fingerprint": fingerprint,
                },
            )
            if include_result:
                campaign.save(
                    training / "run_result.json",
                    {
                        "status": "deadline_stop",
                        "fingerprint": fingerprint,
                        "cursor": {
                            "training_input_tokens": (
                                committed_tokens
                                if result_cursor_tokens is None
                                else result_cursor_tokens
                            )
                        },
                        "external_campaign_tokens": external_tokens,
                    },
                )
            if include_updates:
                if updates_content is not None:
                    (training / "updates.jsonl").write_text(updates_content, encoding="utf-8")
                else:
                    with (training / "updates.jsonl").open("w", encoding="utf-8") as handle:
                        for value in log_values:
                            handle.write(json.dumps({"cumulative_input_tokens": value}) + "\n")
            campaign.save(
                output / "worker-status.json",
                {"stages": [{"name": "training", "exit_code": training_exit_code}]},
            )
        else:
            (training / checkpoint_name).write_bytes(checkpoint_bytes)
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    return plan, report, artifacts, calls


def test_collect_clean_partial_run_has_no_discarded_or_unlogged_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, _calls = _collect_context(tmp_path, monkeypatch)

    record = campaign.collect(plan, attempt=1)

    assert record["checkpoint_verified"] is True
    assert record["training_status"] == "deadline_stop"
    assert record["discarded_logged_tail_tokens"] == 0
    assert record["unlogged_inflight_input_token_reservation"] == 0
    assert record["external_campaign_tokens_for_resume"] == 0
    assert record["processed_campaign_input_tokens_conservative"] == 1_048_576
    assert json.loads((report / "verified-output-1.json").read_text()) == record


def test_collect_hard_failure_uses_pointer_and_reserves_logged_and_inflight_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = 1_048_576
    logged = committed + 2 * 16 * 1024
    plan, _report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        status="ERROR",
        logged_tokens=[logged],
        training_exit_code=1,
        include_result=False,
    )

    record = campaign.collect(plan, attempt=1)

    assert record["training_status"] == "interrupted"
    assert record["cursor"]["training_input_tokens"] == committed
    assert record["discarded_logged_tail_tokens"] == 2 * 16 * 1024
    assert record["unlogged_inflight_input_token_reservation"] == 16 * 1024
    assert record["external_campaign_tokens_for_resume"] == 3 * 16 * 1024
    assert record["processed_campaign_input_tokens_conservative"] == committed + 3 * 16 * 1024


def test_collect_carries_prior_external_tokens_once_across_a_clean_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    already_external = 1_000_000
    plan, _report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        external_tokens=already_external,
    )

    record = campaign.collect(plan, attempt=1)

    assert record["external_campaign_tokens_for_resume"] == already_external
    assert record["processed_campaign_input_tokens_conservative"] == (1_048_576 + already_external)


def test_collect_rejects_token_reservation_over_campaign_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = 1_048_576
    plan, report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        status="ERROR",
        logged_tokens=[committed + 2 * 16 * 1024],
        external_tokens=100_000,
        training_exit_code=1,
        include_result=False,
        maximum_tokens=committed + 100_000,
    )

    with pytest.raises(RuntimeError, match="processed-token reservation"):
        campaign.collect(plan, attempt=1)

    assert not (report / "verified-output-1.json").exists()


def test_collect_allows_clean_zero_update_checkpoint_without_an_update_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        committed_tokens=0,
        include_updates=False,
    )

    record = campaign.collect(plan, attempt=1)

    assert record["cursor"]["training_input_tokens"] == 0
    assert record["discarded_logged_tail_tokens"] == 0
    assert record["unlogged_inflight_input_token_reservation"] == 0
    assert record["external_campaign_tokens_for_resume"] == 0


def test_collect_counts_valid_tail_and_reserves_batch_after_truncated_final_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = 1_048_576
    logged_tail = 2 * 16 * 1024
    valid_line = json.dumps({"cumulative_input_tokens": committed + logged_tail}) + "\n"
    plan, _report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        status="ERROR",
        committed_tokens=committed,
        updates_content=valid_line + '{"cumulative_input_tokens":',
        training_exit_code=1,
        include_result=False,
    )

    record = campaign.collect(plan, attempt=1)

    assert record["truncated_last_update_record"] is True
    assert record["discarded_logged_tail_tokens"] == logged_tail
    assert record["unlogged_inflight_input_token_reservation"] == 16 * 1024
    assert record["external_campaign_tokens_for_resume"] == logged_tail + 16 * 1024


def test_collect_rejects_malformed_nonfinal_update_log_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid_line = json.dumps({"cumulative_input_tokens": 1_048_576}) + "\n"
    plan, report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        status="ERROR",
        updates_content="{malformed}\n" + valid_line,
        training_exit_code=1,
        include_result=False,
    )

    with pytest.raises(ValueError, match="malformed record"):
        campaign.collect(plan, attempt=1)

    assert not (report / "verified-output-1.json").exists()


def test_collect_rejects_missing_update_log_after_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        status="ERROR",
        include_updates=False,
        training_exit_code=1,
        include_result=False,
    )

    with pytest.raises(ValueError, match="update log is missing"):
        campaign.collect(plan, attempt=1)

    assert not (report / "verified-output-1.json").exists()


def test_collect_rejects_pointer_and_run_result_cursor_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, _calls = _collect_context(
        tmp_path,
        monkeypatch,
        result_cursor_tokens=1_048_577,
    )

    with pytest.raises(ValueError, match="cursors differ"):
        campaign.collect(plan, attempt=1)

    assert not (report / "verified-output-1.json").exists()


def _watch_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan_sha256: str | None = None,
    submitted_at: str = "2099-01-01T00:00:00+00:00",
):
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    plan = _plan()
    plan_path = report / "plan.json"
    campaign.save(plan_path, plan)
    campaign.save(
        report / "job-1.json",
        {
            "reference": "shlokbhakta/tabcomplete-q25-code-cpt-r2-attempt-1",
            "plan_sha256": plan_sha256 or campaign.digest(plan_path),
            "submitted_at": submitted_at,
        },
    )
    monkeypatch.setattr(campaign, "REPORT", report)
    return plan, report


def test_watch_collects_once_after_running_transitions_to_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report = _watch_context(tmp_path, monkeypatch)
    statuses = iter(["RUNNING", "KernelWorkerStatus.COMPLETE"])
    cli_calls: list[tuple[str, ...]] = []
    collect_calls: list[tuple[dict[str, Any], int]] = []
    sleeps: list[float] = []

    def fake_cli(*args: str, **kwargs: Any) -> str:
        cli_calls.append(tuple(args))
        return next(statuses)

    def fake_collect(value: dict[str, Any], attempt: int) -> dict[str, Any]:
        collect_calls.append((value, attempt))
        return {"checkpoint_verified": True, "attempt": attempt}

    monkeypatch.setattr(campaign, "cli", fake_cli)
    monkeypatch.setattr(campaign, "collect", fake_collect)
    monkeypatch.setattr(campaign.time, "sleep", sleeps.append)

    result = campaign.watch(plan, attempt=1, poll_seconds=5)

    assert result == {"checkpoint_verified": True, "attempt": 1}
    assert [call[2] for call in cli_calls] == ["status", "status"]
    assert collect_calls == [(plan, 1)]
    assert sleeps == [5]
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in cli_calls)


def test_watch_recovers_after_one_transient_cli_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report = _watch_context(tmp_path, monkeypatch)
    responses: list[str | Exception] = [
        RuntimeError("offline"),
        "RUNNING",
        "COMPLETE",
    ]
    sleeps: list[float] = []
    collect_calls: list[int] = []

    def fake_cli(*_args: str, **_kwargs: Any) -> str:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def fake_collect(_plan: dict[str, Any], attempt: int) -> dict[str, bool]:
        collect_calls.append(attempt)
        return {"ok": True}

    monkeypatch.setattr(campaign, "cli", fake_cli)
    monkeypatch.setattr(campaign, "collect", fake_collect)
    monkeypatch.setattr(campaign.time, "sleep", sleeps.append)

    assert campaign.watch(plan, attempt=1, poll_seconds=2) == {"ok": True}
    assert json.loads((report / "watch-1.json").read_text())["status"] == "COMPLETE"
    assert sleeps == [2, 2]
    assert collect_calls == [1]


def test_watch_stops_after_three_cli_failures_without_collecting_or_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report = _watch_context(tmp_path, monkeypatch)
    cli_calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []
    collect_calls: list[int] = []

    def failing_cli(*args: str, **_kwargs: Any) -> str:
        cli_calls.append(tuple(args))
        raise RuntimeError("offline")

    monkeypatch.setattr(campaign, "cli", failing_cli)
    monkeypatch.setattr(campaign, "collect", lambda _plan, attempt: collect_calls.append(attempt))
    monkeypatch.setattr(campaign.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="observer lost connectivity"):
        campaign.watch(plan, attempt=1, poll_seconds=1)

    observation = json.loads((report / "watch-1.json").read_text())
    assert observation["consecutive_failures"] == 3
    assert len(cli_calls) == 3
    assert sleeps == [1, 1]
    assert collect_calls == []
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in cli_calls)


def test_watch_rejects_plan_mismatch_and_out_of_range_poll_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report = _watch_context(tmp_path, monkeypatch, plan_sha256="wrong-plan")
    cli_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(campaign, "cli", lambda *args, **_kwargs: cli_calls.append(tuple(args)))

    with pytest.raises(ValueError, match="different plan"):
        campaign.watch(plan, attempt=1)
    assert cli_calls == []

    for poll in (0, 61):
        with pytest.raises(ValueError, match="between one and sixty"):
            campaign.watch(plan, attempt=1, poll_seconds=poll)
    assert cli_calls == []


def test_watch_exits_when_observer_deadline_is_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report = _watch_context(
        tmp_path,
        monkeypatch,
        submitted_at="2000-01-01T00:00:00+00:00",
    )
    cli_calls: list[tuple[str, ...]] = []
    collect_calls: list[int] = []
    sleeps: list[float] = []
    monkeypatch.setattr(campaign, "cli", lambda *args, **_kwargs: cli_calls.append(tuple(args)))
    monkeypatch.setattr(campaign, "collect", lambda _plan, attempt: collect_calls.append(attempt))
    monkeypatch.setattr(campaign.time, "sleep", sleeps.append)

    with pytest.raises(TimeoutError, match="observer deadline"):
        campaign.watch(plan, attempt=1, poll_seconds=1)

    assert cli_calls == []
    assert collect_calls == []
    assert sleeps == []


def _shared_budget_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpt_reservations: tuple[int, ...] = (),
    fim_reservations: tuple[int, ...] = (),
) -> None:
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    campaign.save(
        report / "campaign_budget.json",
        {
            "shared_limits": {
                "aggregate_reserved_session_seconds": 72_000,
                "conservative_account_gpu_hours": 40,
                "minimum_reserved_future_fim_session_seconds": 21_600,
            }
        },
    )
    for index, reservation in enumerate(cpt_reservations, start=1):
        campaign.save(
            report / f"job-{index}.json",
            {"conservative_reserved_session_seconds": reservation},
        )
    for index, reservation in enumerate(fim_reservations, start=1):
        campaign.save(
            report / f"fim-job-{index}.json",
            {"conservative_reserved_session_seconds": reservation},
        )
    monkeypatch.setattr(campaign, "REPORT", report)


def test_shared_allocation_budget_counts_cpt_and_fim_receipts_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shared_budget_context(
        tmp_path,
        monkeypatch,
        cpt_reservations=(36_000,),
        fim_reservations=(21_600,),
    )

    # 10h CPT + 6h FIM + 4h new FIM = exactly 20h / 40 account GPU-hours.
    campaign.check_shared_allocation_budget(14_400, phase="fim")

    with pytest.raises(RuntimeError, match="shared CPT/FIM session reservation"):
        campaign.check_shared_allocation_budget(14_401, phase="fim")


def test_cpt_budget_keeps_the_full_six_hour_reserve_for_two_fim_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shared_budget_context(tmp_path, monkeypatch, cpt_reservations=(36_000,))

    # Existing CPT 10h + new CPT 4h + two 3h FIM arms = the 20h limit.
    campaign.check_shared_allocation_budget(14_400, phase="cpt")

    with pytest.raises(RuntimeError, match="shared CPT/FIM session reservation"):
        campaign.check_shared_allocation_budget(14_401, phase="cpt")


def test_raw_cpt_cannot_restart_after_a_fim_receipt_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shared_budget_context(tmp_path, monkeypatch, fim_reservations=(10_800,))

    with pytest.raises(RuntimeError, match="cannot restart after matched FIM"):
        campaign.check_shared_allocation_budget(14_400, phase="cpt")


def test_submit_rejects_resume_when_raw_cpt_pass_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, calls = _submission_context(tmp_path, monkeypatch)
    prior = "shlokbhakta/tc-q25-code-cpt-r2-a1"
    campaign.save(report / "job-1.json", {"reference": prior, "status": "submitted"})
    campaign.save(
        report / "verified-output-1.json",
        {
            "checkpoint_verified": True,
            "training_status": "complete",
            "external_campaign_tokens_for_resume": 0,
        },
    )

    with pytest.raises(RuntimeError, match="raw-code pass is already complete"):
        campaign.submit(plan, attempt=2, resume_source=prior)

    assert calls == []


def _token_ledger_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    maximum_tokens: int = 32_000_000,
) -> Path:
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    campaign.save(
        report / "campaign_budget.json",
        {
            "shared_limits": {
                "maximum_additional_processed_training_input_tokens": maximum_tokens,
            }
        },
    )
    monkeypatch.setattr(campaign, "REPORT", report)
    return report


def _write_fim_token_attempt(
    report: Path,
    arm: str,
    attempt: int,
    *,
    reservation: int,
    verified_total: int | None = None,
) -> None:
    campaign.save(
        report / f"fim-job-{arm}-{attempt}.json",
        {
            "attempt": attempt,
            "processed_arm_token_reservation": reservation,
        },
    )
    if verified_total is not None:
        campaign.save(
            report / f"fim-verified-{arm}-{attempt}.json",
            {"processed_arm_input_tokens_conservative": verified_total},
        )


def test_shared_token_ledger_uses_latest_cumulative_attempt_per_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _token_ledger_context(tmp_path, monkeypatch)
    for attempt, cumulative in ((1, 9_000_000), (2, 11_500_000)):
        campaign.save(report / f"job-{attempt}.json", {"attempt": attempt})
        campaign.save(
            report / f"verified-output-{attempt}.json",
            {"processed_campaign_input_tokens_conservative": cumulative},
        )
    first_arm, second_arm = campaign.FIM_ARMS
    _write_fim_token_attempt(report, first_arm, 1, reservation=1_500_000, verified_total=1_500_000)
    _write_fim_token_attempt(report, first_arm, 2, reservation=1_750_000, verified_total=1_750_000)
    _write_fim_token_attempt(report, second_arm, 1, reservation=1_600_000, verified_total=1_600_000)

    assert campaign.shared_token_ledger() == {
        "cpt": 11_500_000,
        first_arm: 1_750_000,
        second_arm: 1_600_000,
    }


def test_shared_token_ledger_keeps_full_reservations_for_uncollected_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _token_ledger_context(tmp_path, monkeypatch)
    arm = campaign.FIM_ARMS[0]
    campaign.save(report / "job-1.json", {"attempt": 1})
    _write_fim_token_attempt(report, arm, 1, reservation=2_250_000)

    assert campaign.shared_token_ledger() == {"cpt": 12_000_000, arm: 2_250_000}


def test_shared_token_budget_enforces_the_32m_boundary_across_all_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _token_ledger_context(tmp_path, monkeypatch)
    first_arm, second_arm = campaign.FIM_ARMS
    _write_fim_token_attempt(
        report, first_arm, 1, reservation=20_000_000, verified_total=20_000_000
    )
    _write_fim_token_attempt(
        report, second_arm, 1, reservation=11_999_999, verified_total=11_999_999
    )

    assert campaign.check_shared_token_budget("cpt", 1) == 31_999_999
    with pytest.raises(RuntimeError, match="processed-token reservation exhausted"):
        campaign.check_shared_token_budget("cpt", 2)


def test_shared_token_budget_rejects_negative_request_and_negative_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _token_ledger_context(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="invalid campaign phase or token reservation"):
        campaign.check_shared_token_budget("cpt", -1)

    campaign.save(report / "job-1.json", {"attempt": 1})
    campaign.save(
        report / "verified-output-1.json",
        {"processed_campaign_input_tokens_conservative": -1},
    )
    with pytest.raises(ValueError, match="negative counter"):
        campaign.shared_token_ledger()


def _cpt_export_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    verified_overrides: dict[str, Any] | None = None,
    manifest_overrides: dict[str, Any] | None = None,
    files_override: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path, Path, Path, list[tuple[str, ...]]]:
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    export = artifacts / "output-attempt-1/training/inference-f16"
    export.mkdir(parents=True)
    fingerprint = "frozen-cpt-fingerprint"
    cursor = {"training_input_tokens": 7_872_512, "completed_updates": 481}
    blobs = {"config.json": b"{}", "model.safetensors": b"synthetic weights"}
    file_records = {
        name: {
            "bytes": len(content),
            "sha256": campaign.hashlib.sha256(content).hexdigest(),
        }
        for name, content in blobs.items()
    }
    for name, content in blobs.items():
        (export / name).write_bytes(content)
    manifest: dict[str, Any] = {
        "schema": "q25-cpt-inference-f16-v1",
        "fingerprint": fingerprint,
        "training_cursor": cursor,
        "files": file_records if files_override is None else files_override,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    campaign.save(export / "artifact_manifest.json", manifest)
    verified: dict[str, Any] = {
        "checkpoint_verified": True,
        "training_status": "complete",
        "fingerprint": fingerprint,
        "cursor": cursor,
        "reference": "owner/completed-cpt-kernel",
    }
    if verified_overrides:
        verified.update(verified_overrides)
    campaign.save(report / "verified-output-1.json", verified)
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=16 * 1024**3),
    )
    calls: list[tuple[str, ...]] = []

    def unexpected_cli(*args: str, **_kwargs: Any) -> str:
        calls.append(tuple(args))
        raise AssertionError("unexpected remote retrieval for a fully local export")

    monkeypatch.setattr(campaign, "cli", unexpected_cli)
    return _plan(), report, artifacts, export, calls


def test_collect_cpt_export_accepts_only_the_complete_481_update_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, export, calls = _cpt_export_context(tmp_path, monkeypatch)

    result = campaign.collect_cpt_export(plan, attempt=1)

    assert result == export
    verification = json.loads((report / "cpt-export-verification.json").read_text())
    assert verification["training_cursor"] == {
        "training_input_tokens": 7_872_512,
        "completed_updates": 481,
    }
    assert verification["fingerprint"] == "frozen-cpt-fingerprint"
    assert calls == []


@pytest.mark.parametrize(
    ("verified_overrides", "manifest_overrides", "message"),
    [
        ({"checkpoint_verified": False}, {}, "verified complete CPT pass"),
        ({"training_status": "deadline_stop"}, {}, "verified complete CPT pass"),
        (
            {"cursor": {"training_input_tokens": 7_872_511, "completed_updates": 481}},
            {},
            "frozen completed pass",
        ),
        (
            {},
            {
                "training_cursor": {
                    "training_input_tokens": 7_872_512,
                    "completed_updates": 480,
                }
            },
            "frozen completed pass",
        ),
    ],
)
def test_collect_cpt_export_rejects_incomplete_or_wrong_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_overrides: dict[str, Any],
    manifest_overrides: dict[str, Any],
    message: str,
) -> None:
    plan, report, _artifacts, _export, calls = _cpt_export_context(
        tmp_path,
        monkeypatch,
        verified_overrides=verified_overrides,
        manifest_overrides=manifest_overrides,
    )

    with pytest.raises(ValueError, match=message):
        campaign.collect_cpt_export(plan, attempt=1)

    assert calls == []
    assert not (report / "cpt-export-verification.json").exists()


def test_collect_cpt_export_rejects_unsafe_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _report, _artifacts, _export, calls = _cpt_export_context(
        tmp_path,
        monkeypatch,
        files_override={"../outside.safetensors": {"bytes": 10, "sha256": "0" * 64}},
    )

    with pytest.raises(ValueError, match="unsafe file record"):
        campaign.collect_cpt_export(plan, attempt=1)

    assert calls == []


@pytest.mark.parametrize("limit_kind", ["artifact_cap", "free_space"])
def test_collect_cpt_export_checks_storage_before_retrieving_missing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    plan, _report, _artifacts, _export, calls = _cpt_export_context(
        tmp_path,
        monkeypatch,
        files_override={"missing.safetensors": {"bytes": 100, "sha256": "0" * 64}},
    )
    if limit_kind == "artifact_cap":
        plan["configuration"]["budget"]["new_artifact_bytes_cap"] = 99
        monkeypatch.setattr(campaign, "directory_bytes", lambda _path: 0)
        message = "exceeds storage headroom"
    else:
        monkeypatch.setattr(campaign, "directory_bytes", lambda _path: 0)
        monkeypatch.setattr(
            campaign.shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(free=2 * 1024**3 + 99),
        )
        message = "exceeds storage headroom"

    with pytest.raises(OSError, match=message):
        campaign.collect_cpt_export(plan, attempt=1)

    assert calls == []


def test_collect_cpt_export_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, _export, calls = _cpt_export_context(
        tmp_path,
        monkeypatch,
        files_override={"config.json": {"bytes": 2, "sha256": "0" * 64}},
    )

    with pytest.raises(ValueError, match="file identity differs"):
        campaign.collect_cpt_export(plan, attempt=1)

    assert calls == []
    assert not (report / "cpt-export-verification.json").exists()


def test_freeze_fim_plan_builds_cpu_only_reproducible_matched_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    campaign.save(report / "plan.json", {"schema": "test-cpt-plan"})
    artifacts = tmp_path / "artifacts"
    export = artifacts / "output-attempt-1/training/inference-f16"
    export.mkdir(parents=True)
    weights = b"synthetic completed CPT weights"
    (export / "model.safetensors").write_bytes(weights)
    export_manifest = {
        "schema": "q25-cpt-inference-f16-v1",
        "fingerprint": "completed-cpt-fingerprint",
        "training_cursor": {
            "training_input_tokens": 7_872_512,
            "completed_updates": 481,
        },
        "files": {
            "model.safetensors": {
                "bytes": len(weights),
                "sha256": campaign.hashlib.sha256(weights).hexdigest(),
            }
        },
    }
    campaign.save(export / "artifact_manifest.json", export_manifest)
    cpt_output = artifacts / "output-attempt-1/training"
    cpt_output.mkdir(parents=True, exist_ok=True)
    campaign.save(
        cpt_output / "run_manifest.json",
        {
            "identity": {
                "runtime": {
                    "python": "3.11.0",
                    "torch": "2.14.0+cu128",
                    "transformers": "5.17.0",
                    "bitsandbytes": "0.50.2",
                    "cuda_runtime": "12.8",
                }
            }
        },
    )
    preparation = {
        "planned_training": {
            "maximum_input_tokens_per_arm": 4_194_304,
            "effective_batch": 16,
            "microbatch_examples": 1,
            "epochs": 1,
            "learning_rate": 1e-5,
            "checkpoint_every_updates": 64,
            "compute": "fp16",
            "gradient_checkpointing": True,
            "master_weights": "fp32",
            "optimizer": "AdamW8bit",
            "seed": 314159,
        }
    }
    preparation_path = report / "fim_preparation_plan.json"
    campaign.save(preparation_path, preparation)
    corpus = artifacts / "fim/corpus-r3"
    corpus.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    splits: dict[str, dict[str, Any]] = {}
    for split, tokens in (("train", 1_776_736), ("development", 91_434)):
        name = f"{split}.jsonl"
        content = f"{split} rows\n".encode()
        (corpus / name).write_bytes(content)
        files[name] = {
            "bytes": len(content),
            "sha256": campaign.hashlib.sha256(content).hexdigest(),
        }
        splits[split] = {"file": name, "input_tokens": tokens, "row_count": 1}
    campaign.save(
        corpus / "corpus_metadata.json",
        {
            "preparation_plan_sha256": campaign.digest(preparation_path),
            "files": files,
            "splits": splits,
        },
    )
    campaign.save(
        report / "campaign_budget.json",
        {
            "shared_limits": {
                "new_artifact_bytes_cap_per_machine": 12 * 1024**3,
                "minimum_free_bytes": 2 * 1024**3,
                "quota_renewal": "2026-10-10T00:00:00",
            }
        },
    )
    requirements_path = tmp_path / "fim-requirements.lock"
    requirements_path.write_text("synthetic hashed requirements\n")
    runtime_lock = {
        "expected_versions": {
            "python": "3.11.15",
            "torch": "2.11.0+cu128",
            "transformers": "5.17.0",
            "bitsandbytes": "0.50.2",
            "cuda_runtime": "12.8",
        },
        "requirements_lock": {
            "repo_relative_path": requirements_path.name,
            "sha256": campaign.digest(requirements_path),
        },
        "bootstrap_uv_version": "0.12.3",
        "bootstrap_uv_wheel_sha256": "synthetic-uv-wheel",
    }
    campaign.save(report / "fim_runtime_lock.json", runtime_lock)
    cpt_plan = {
        "configuration": {
            "model": {
                "id": "Qwen/Qwen2.5-Coder-0.5B",
                "revision": "pinned-revision",
            }
        },
        "existing_model_files": {"model.safetensors": {"sha256": "base-model-hash"}},
        "fixtures": {"causal": {"sha256": "fixture-hash"}},
    }
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(campaign, "collect_cpt_export", lambda _plan, _attempt: export)

    plan = campaign.freeze_fim_plan(cpt_plan, attempt=1)

    assert plan["gpu_execution_authorized"] is True
    assert plan["configuration"]["training"]["max_input_tokens"] == 4_194_304
    assert plan["configuration"]["budget"]["session_seconds"] == 10_800
    assert plan["initializers"][campaign.FIM_ARMS[1]]["expected_complete_updates"] == 481
    assert plan["initializers"][campaign.FIM_ARMS[1]]["expected_training_input_tokens"] == (
        7_872_512
    )
    assert plan["evaluation"]["regression"] == "unchanged raw causal 200 and raw line 180"
    assert plan["configuration"]["runtime"]["python"] == "3.11.15"
    assert plan["parent_cpt_runtime_observation"]["python"] == "3.11.0"
    assert plan["evaluation"]["source_syntax"]["selection_gate"] is False
    assert json.loads((report / "fim_training_plan.json").read_text()) == plan
    requirements_path.write_text("modified requirements\n")
    with pytest.raises(ValueError, match="requirements differ"):
        campaign.freeze_fim_plan(cpt_plan, attempt=1)


def _fim_submission_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpt_session_reservation: int = 0,
    cpt_fingerprint: str = "frozen-cpt-fingerprint",
    cpt_cursor: dict[str, int] | None = None,
    active_jobs: list[dict[str, Any]] | None = None,
):
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    manifest_path = artifacts / "fim/input-bundle/input-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    campaign.save(manifest_path, {"schema": "synthetic-fim-inputs"})
    campaign.save(
        artifacts / "fim/dataset-submission.json",
        {
            "state": "verified",
            "input_manifest_sha256": campaign.digest(manifest_path),
        },
    )
    cursor = cpt_cursor or {"training_input_tokens": 7_872_512, "completed_updates": 481}
    plan = {
        "schema": "q25-fim-training-plan-v1",
        "gpu_execution_authorized": True,
        "configuration": {
            "training": {
                "max_input_tokens": 4_194_304,
                "effective_batch": 16,
                "sequence_length": 1024,
            },
            "budget": {
                "session_seconds": 10_800,
                "maximum_discarded_replay_input_tokens": 2_097_152,
                "new_artifact_bytes_cap": 12 * 1024**3,
                "minimum_free_bytes": 2 * 1024**3,
                "paid_compute": False,
                "automatic_renewal_use": False,
                "quota_renewal": "2026-10-10T00:00:00",
                "conservative_quota_multiplier": 2,
                "quota_account_gpu_hours_cap": 40,
            },
        },
        "initializers": {
            campaign.FIM_ARMS[1]: {
                "fingerprint": "frozen-cpt-fingerprint",
                "training_cursor": {"training_input_tokens": 7_872_512, "completed_updates": 481},
            }
        },
        "data": {"train": {"input_tokens": 1_776_736}},
    }
    campaign.save(report / "fim_training_plan.json", plan)
    campaign.save(
        report / "campaign_budget.json",
        {
            "shared_limits": {
                "aggregate_reserved_session_seconds": 72_000,
                "conservative_account_gpu_hours": 40,
                "minimum_reserved_future_fim_session_seconds": 21_600,
                "maximum_additional_processed_training_input_tokens": 32_000_000,
            }
        },
    )
    if cpt_session_reservation:
        campaign.save(
            report / "job-1.json",
            {"conservative_reserved_session_seconds": cpt_session_reservation},
        )
    campaign.save(
        report / "verified-output-1.json",
        {
            "checkpoint_verified": True,
            "training_status": "complete",
            "fingerprint": cpt_fingerprint,
            "cursor": cursor,
            "reference": "owner/completed-cpt-kernel",
        },
    )
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(build_pilot, "_csv_refs", lambda _command: set())
    monkeypatch.setattr(build_pilot, "_job_statuses_verified", lambda _value: True)
    quota_calls: list[dict[str, Any]] = []
    observation = _observation(active_jobs=active_jobs or [])
    monkeypatch.setattr(campaign, "quota", lambda: quota_calls.append(observation) or observation)
    cli_calls: list[tuple[str, ...]] = []
    commit = "c" * 40

    def fake_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        cli_calls.append(call)
        if call[0] == "git" and call[-2:] == ("status", "--porcelain"):
            return ""
        if call[-2:] == ("rev-parse", "HEAD"):
            return commit
        if call[-2:] == ("origin", "refs/heads/research/q25-code-cpt-r2"):
            return f"{commit}\trefs/heads/research/q25-code-cpt-r2"
        if call[:3] == ("kaggle", "kernels", "push"):
            return "Kernel pushed"
        raise AssertionError(f"unexpected remote command: {call[:3]}")

    monkeypatch.setattr(campaign, "cli", fake_cli)
    return plan, report, artifacts, quota_calls, cli_calls


@pytest.mark.parametrize("arm", campaign.FIM_ARMS)
def test_fim_submission_mounts_only_the_selected_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    plan, report, artifacts, _quota_calls, _cli_calls = _fim_submission_context(
        tmp_path, monkeypatch
    )
    plan["initializers"][campaign.FIM_ARMS[1]]["artifact_manifest_sha256"] = "export-manifest"
    campaign.save(report / "fim_training_plan.json", plan)
    campaign.save(
        artifacts / "fim/cpt-initializer-submission.json",
        {"state": "verified", "artifact_manifest_sha256": "export-manifest"},
    )
    campaign.submit_fim(plan, arm=arm, attempt=1, cpt_attempt=1, resume_source=None)
    kernel = artifacts / f"fim/kernel-{arm}-1/kernel-metadata.json"
    metadata = json.loads(kernel.read_text())
    expected_initializer = (
        campaign.BASE_DATASET if arm == campaign.FIM_ARMS[0] else campaign.CPT_INITIALIZER_DATASET
    )
    assert metadata["dataset_sources"] == [campaign.FIM_DATASET, expected_initializer]


@pytest.mark.parametrize("identity_field", ["fingerprint", "cursor"])
def test_submit_fim_rejects_wrong_frozen_cpt_initializer_before_quota_or_push(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    plan, report, _artifacts, quota_calls, cli_calls = _fim_submission_context(
        tmp_path,
        monkeypatch,
        cpt_fingerprint=(
            "different-completed-cpt-fingerprint"
            if identity_field == "fingerprint"
            else "frozen-cpt-fingerprint"
        ),
        cpt_cursor=(
            {"training_input_tokens": 7_872_511, "completed_updates": 481}
            if identity_field == "cursor"
            else None
        ),
    )

    with pytest.raises(ValueError, match="differs from the frozen matched FIM initializer"):
        campaign.submit_fim(
            plan,
            arm=campaign.FIM_ARMS[1],
            attempt=1,
            cpt_attempt=1,
            resume_source=None,
        )

    assert quota_calls == []
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in cli_calls)
    assert not (report / f"fim-job-{campaign.FIM_ARMS[1]}-1.json").exists()


def test_submit_fim_preserves_the_other_arms_three_hour_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, quota_calls, cli_calls = _fim_submission_context(
        tmp_path,
        monkeypatch,
        cpt_session_reservation=50_401,
    )

    with pytest.raises(RuntimeError, match="shared CPT/FIM session reservation exhausted"):
        campaign.submit_fim(
            plan,
            arm=campaign.FIM_ARMS[0],
            attempt=1,
            cpt_attempt=1,
            resume_source=None,
        )

    assert quota_calls == []
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in cli_calls)
    assert not (report / f"fim-job-{campaign.FIM_ARMS[0]}-1.json").exists()


def test_submit_fim_checks_live_quota_before_staging_or_pushing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, _artifacts, quota_calls, cli_calls = _fim_submission_context(
        tmp_path,
        monkeypatch,
        active_jobs=[{"reference": "owner/other-job", "status": "RUNNING"}],
    )

    with pytest.raises(RuntimeError, match="another notebook is active"):
        campaign.submit_fim(
            plan,
            arm=campaign.FIM_ARMS[1],
            attempt=1,
            cpt_attempt=1,
            resume_source=None,
        )

    assert len(quota_calls) == 1
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in cli_calls)
    assert not (report / f"fim-job-{campaign.FIM_ARMS[1]}-1.json").exists()


@pytest.mark.parametrize("identity_mismatch", ["arm", "plan"])
def test_watch_fim_rejects_wrong_job_identity_before_status_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_mismatch: str,
) -> None:
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    plan = {"schema": "q25-fim-training-plan-v1"}
    plan_path = report / "fim_training_plan.json"
    campaign.save(plan_path, plan)
    arm = campaign.FIM_ARMS[0]
    job = {
        "arm": campaign.FIM_ARMS[1] if identity_mismatch == "arm" else arm,
        "plan_sha256": "wrong-plan" if identity_mismatch == "plan" else campaign.digest(plan_path),
        "reference": "owner/fim-kernel",
    }
    campaign.save(report / f"fim-job-{arm}-1.json", job)
    monkeypatch.setattr(campaign, "REPORT", report)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(campaign, "cli", lambda *args, **_kwargs: calls.append(tuple(args)))

    with pytest.raises(ValueError, match="observer identity differs"):
        campaign.watch_fim(plan, arm, attempt=1)

    assert calls == []


def _fim_collect_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str | None = None,
    attempt: int = 2,
    committed_tokens: int = 1_000_000,
    logged_tail_tokens: int = 32_768,
    prior_discarded_tokens: int = 100_000,
    updates_content: str | None = None,
    training_exit_code: int = 1,
):
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    arm = campaign.FIM_ARMS[0] if arm is None else arm
    plan = {
        "schema": "q25-fim-training-plan-v1",
        "configuration": {
            "training": {
                "max_input_tokens": 4_194_304,
                "effective_batch": 16,
                "sequence_length": 1024,
            },
            "budget": {
                "new_artifact_bytes_cap": 12 * 1024**3,
                "minimum_free_bytes": 2 * 1024**3,
                "maximum_discarded_replay_input_tokens": 2_097_152,
            },
        },
        "data": {"train": {"input_tokens": 1_776_736}},
    }
    plan_path = report / "fim_training_plan.json"
    campaign.save(plan_path, plan)
    campaign.save(
        report / "campaign_budget.json",
        {
            "shared_limits": {
                "maximum_additional_processed_training_input_tokens": 32_000_000,
            }
        },
    )
    campaign.save(
        report / "job-1.json",
        {"conservative_reserved_session_seconds": 14_400},
    )
    campaign.save(
        report / "verified-output-1.json",
        {"processed_campaign_input_tokens_conservative": 12_000_000},
    )
    plan_sha = campaign.digest(plan_path)
    if attempt > 1:
        campaign.save(
            report / f"fim-job-{arm}-1.json",
            {
                "attempt": 1,
                "arm": arm,
                "plan_sha256": plan_sha,
                "processed_arm_token_reservation": 6_291_456,
            },
        )
        campaign.save(
            report / f"fim-verified-{arm}-1.json",
            {"processed_arm_input_tokens_conservative": 900_000},
        )
    campaign.save(
        report / f"fim-job-{arm}-{attempt}.json",
        {
            "attempt": attempt,
            "arm": arm,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": "d" * 64,
            "commit": "c" * 40,
            "reference": "owner/fim-kernel",
            "own_discarded_tokens": prior_discarded_tokens,
            "other_phase_tokens": 12_000_000,
            "processed_arm_token_reservation": 6_291_456,
        },
    )
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=16 * 1024**3),
    )
    fingerprint = "frozen-fim-fingerprint"
    cursor = {
        "training_input_tokens": committed_tokens,
        "completed_updates": 64,
        "supervised_target_tokens": committed_tokens // 10,
    }
    checkpoint_name = "resume-step-000064.pt"
    checkpoint_bytes = b"synthetic complete FIM resume state"
    content = updates_content
    if content is None:
        content = (
            json.dumps({"cumulative_input_tokens": committed_tokens + logged_tail_tokens}) + "\n"
        )
    cli_calls: list[tuple[str, ...]] = []

    def fake_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        cli_calls.append(call)
        if call[:3] == ("kaggle", "kernels", "status"):
            return "ERROR"
        if call[:3] != ("kaggle", "kernels", "output"):
            raise AssertionError(f"unexpected command: {call[:3]}")
        output = Path(call[call.index("-p") + 1])
        pattern = call[call.index("--file-pattern") + 1]
        training = output / "training"
        training.mkdir(parents=True, exist_ok=True)
        if pattern == r"\.(json|jsonl|log)$":
            campaign.save(
                training / "latest.json",
                {"path": checkpoint_name, "fingerprint": fingerprint, "cursor": cursor},
            )
            campaign.save(
                training / (checkpoint_name + ".complete.json"),
                {
                    "sha256": campaign.hashlib.sha256(checkpoint_bytes).hexdigest(),
                    "fingerprint": fingerprint,
                },
            )
            campaign.save(
                training / "run_result.json",
                {"status": "deadline_stop", "fingerprint": fingerprint, "cursor": cursor},
            )
            (training / "updates.jsonl").write_text(content, encoding="utf-8")
            campaign.save(
                output / "worker-status.json",
                {"stages": [{"name": "training", "exit_code": training_exit_code}]},
            )
        else:
            (training / checkpoint_name).write_bytes(checkpoint_bytes)
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    return plan, report, cli_calls


def _fim_zero_work_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    identity_mismatch: str | None = None,
    training_started: Any = False,
    evidence: str | None = None,
    missing_status: bool = False,
):
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    arm = campaign.FIM_ARMS[0]
    plan = {
        "schema": "q25-fim-training-plan-v1",
        "configuration": {
            "training": {"effective_batch": 16, "sequence_length": 1024},
            "budget": {
                "new_artifact_bytes_cap": 12 * 1024**3,
                "minimum_free_bytes": 2 * 1024**3,
                "maximum_discarded_replay_input_tokens": 2_097_152,
            },
        },
        "data": {"train": {"input_tokens": 1_776_736}},
    }
    plan_path = report / "fim_training_plan.json"
    campaign.save(plan_path, plan)
    campaign.save(
        report / "campaign_budget.json",
        {"shared_limits": {"maximum_additional_processed_training_input_tokens": 32_000_000}},
    )
    plan_sha = campaign.digest(plan_path)
    input_sha = "d" * 64
    commit = "c" * 40
    job = {
        "reference": "owner/zero-work-fim-kernel",
        "arm": arm,
        "attempt": 1,
        "plan_sha256": plan_sha,
        "input_manifest_sha256": input_sha,
        "commit": commit,
        "resume_source": None,
        "resume_identity": None,
        "own_discarded_tokens": 0,
        "other_phase_tokens": 12_000_000,
        "processed_arm_token_reservation": 6_291_456,
    }
    campaign.save(report / f"fim-job-{arm}-1.json", job)
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=16 * 1024**3),
    )
    worker = {
        "schema": "q25-fim-kaggle-worker-status-v1",
        "state": "failed",
        "commit": commit,
        "attempt": 1,
        "arm": arm,
        "plan_sha256": plan_sha,
        "input_manifest_sha256": input_sha,
        "training_started": training_started,
        "stages": [],
    }
    if identity_mismatch is not None:
        worker[identity_mismatch] = "wrong-identity"
    cli_calls: list[tuple[str, ...]] = []

    def fake_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        cli_calls.append(call)
        if call[:3] == ("kaggle", "kernels", "status"):
            return "ERROR"
        if call[:3] != ("kaggle", "kernels", "output"):
            raise AssertionError(f"unexpected command: {call[:3]}")
        output = Path(call[call.index("-p") + 1])
        if not missing_status:
            if evidence in {"run_result", "updates"}:
                nested_training = output / "q25_fim_r2/training"
                nested_training.mkdir(parents=True)
                path = nested_training / (
                    "run_result.json" if evidence == "run_result" else "updates.jsonl"
                )
                path.write_text("{}\n", encoding="utf-8")
            if evidence == "training_log":
                log_dir = output / "q25_fim_r2/logs"
                log_dir.mkdir(parents=True)
                (log_dir / "training.log").write_text("synthetic\n", encoding="utf-8")
            if evidence == "training_stage":
                worker["stages"] = [{"name": "training", "exit_code": 1}]
            campaign.save(output / "q25_fim_r2/worker-status.json", worker)
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    return plan, report, arm, cli_calls


def test_collect_fim_accepts_only_identity_verified_zero_work_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, arm, cli_calls = _fim_zero_work_context(tmp_path, monkeypatch)

    record = campaign.collect_fim(plan, arm, attempt=1)

    assert record["no_training_executed"] is True
    assert record["training_status"] == "no_training_executed"
    assert record["checkpoint_verified"] is False
    assert record["carried_checkpoint_verified"] is False
    assert record["processed_arm_input_tokens_conservative"] == 0
    assert json.loads((report / f"fim-verified-{arm}-1.json").read_text()) == record
    assert len([call for call in cli_calls if call[:3] == ("kaggle", "kernels", "output")]) == 1


@pytest.mark.parametrize(
    "identity_mismatch", ["commit", "attempt", "arm", "plan_sha256", "input_manifest_sha256"]
)
def test_collect_fim_rejects_zero_work_status_with_wrong_allocation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity_mismatch: str
) -> None:
    plan, report, arm, _calls = _fim_zero_work_context(
        tmp_path, monkeypatch, identity_mismatch=identity_mismatch
    )

    with pytest.raises(ValueError, match="worker identity differs"):
        campaign.collect_fim(plan, arm, attempt=1)

    assert not (report / f"fim-verified-{arm}-1.json").exists()


@pytest.mark.parametrize("training_started", [True, None, 0])
def test_collect_fim_fails_closed_when_zero_work_status_is_not_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_started: Any
) -> None:
    plan, report, arm, _calls = _fim_zero_work_context(
        tmp_path, monkeypatch, training_started=training_started
    )

    with pytest.raises(ValueError, match="training state is unknown"):
        campaign.collect_fim(plan, arm, attempt=1)

    assert not (report / f"fim-verified-{arm}-1.json").exists()


@pytest.mark.parametrize("evidence", ["run_result", "updates", "training_log", "training_stage"])
def test_collect_fim_rejects_training_evidence_even_when_pointer_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str
) -> None:
    plan, report, arm, _calls = _fim_zero_work_context(tmp_path, monkeypatch, evidence=evidence)

    with pytest.raises(ValueError, match="training evidence|training artifacts|training log"):
        campaign.collect_fim(plan, arm, attempt=1)

    assert not (report / f"fim-verified-{arm}-1.json").exists()


def test_collect_fim_rejects_missing_zero_work_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, arm, _calls = _fim_zero_work_context(tmp_path, monkeypatch, missing_status=True)

    with pytest.raises(ValueError, match="exactly one worker status"):
        campaign.collect_fim(plan, arm, attempt=1)

    assert not (report / f"fim-verified-{arm}-1.json").exists()


def _record_zero_work_retry(
    report: Path,
    arm: str,
    *,
    attempt: int,
    reference: str,
    plan_sha: str,
    input_sha: str,
    commit: str,
    resume_source: str | None = None,
    resume_identity: dict[str, Any] | None = None,
    own_discarded: int = 0,
) -> None:
    campaign.save(
        report / f"fim-job-{arm}-{attempt}.json",
        {
            "reference": reference,
            "arm": arm,
            "attempt": attempt,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": input_sha,
            "commit": commit,
            "authorization_lineage_reference": None,
            "resume_source": resume_source,
            "resume_identity": resume_identity,
            "own_discarded_tokens": own_discarded,
            "other_phase_tokens": 0,
            "processed_arm_token_reservation": 6_291_456,
            "conservative_reserved_session_seconds": 10_800,
        },
    )
    cursor = None if resume_identity is None else resume_identity["cursor"]
    committed = 0 if cursor is None else cursor["training_input_tokens"]
    campaign.save(
        report / f"fim-verified-{arm}-{attempt}.json",
        {
            "reference": reference,
            "arm": arm,
            "attempt": attempt,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": input_sha,
            "commit": commit,
            "checkpoint_verified": False,
            "carried_checkpoint_verified": resume_identity is not None,
            "carried_checkpoint_source": resume_source,
            "carried_checkpoint_identity": resume_identity,
            "no_training_executed": True,
            "training_status": "no_training_executed",
            "own_discarded_tokens_for_resume": own_discarded,
            "processed_arm_input_tokens_conservative": committed + own_discarded,
        },
    )


def test_submit_fim_zero_work_retry_authorizes_prior_but_restarts_from_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, artifacts, _quota_calls, _cli_calls = _fim_submission_context(
        tmp_path, monkeypatch
    )
    arm = campaign.FIM_ARMS[0]
    plan_sha = campaign.digest(report / "fim_training_plan.json")
    input_sha = campaign.digest(artifacts / "fim/input-bundle/input-manifest.json")
    prior = "owner/failed-before-training"
    commit = "c" * 40
    _record_zero_work_retry(
        report,
        arm,
        attempt=1,
        reference=prior,
        plan_sha=plan_sha,
        input_sha=input_sha,
        commit=commit,
    )

    job = campaign.submit_fim(plan, arm=arm, attempt=2, cpt_attempt=1, resume_source=prior)

    metadata = json.loads((artifacts / f"fim/kernel-{arm}-2/kernel-metadata.json").read_text())
    session_source = (artifacts / f"fim/kernel-{arm}-2/run.py").read_text()
    assert job["authorization_lineage_reference"] == prior
    assert job["resume_source"] is None
    assert job["resume_identity"] is None
    assert metadata["kernel_sources"] == []
    assert "'authorization_lineage_reference': 'owner/failed-before-training'" in session_source
    assert "'resume_source': None" in session_source


def _retry_plan_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "REPORT", tmp_path)
    old = {
        "plan_revision": 3,
        "configuration": {"training": "unchanged"},
        "evaluation": {"attention_backend": "automatic"},
    }
    campaign.save(tmp_path / "fim_training_plan-r3.json", old)
    previous_sha = campaign.digest(tmp_path / "fim_training_plan-r3.json")
    plan = {
        **old,
        "plan_revision": 4,
        "evaluation": {"attention_backend": "memory-efficient"},
        "restart_after_zero_work": {
            "previous_plan_sha256": previous_sha,
            "previous_plan_file": "fim_training_plan-r3.json",
        },
    }
    campaign.save(tmp_path / "fim_training_plan.json", plan)
    prior = {"plan_sha256": previous_sha, "resume_source": None, "resume_identity": None}
    verified = {
        "no_training_executed": True,
        "training_status": "no_training_executed",
        "processed_arm_input_tokens_conservative": 0,
        "own_discarded_tokens_for_resume": 0,
        "carried_checkpoint_source": None,
        "carried_checkpoint_identity": None,
        "carried_checkpoint_verified": False,
    }
    return plan, prior, verified


def test_submit_fim_changed_evaluator_restarts_only_zero_work(tmp_path, monkeypatch):
    plan, report, artifacts, _quota_calls, _cli_calls = _fim_submission_context(
        tmp_path, monkeypatch
    )
    arm = campaign.FIM_ARMS[0]
    old_sha = campaign.digest(report / "fim_training_plan.json")
    campaign.save(report / "fim_training_plan-r3.json", plan)
    prior = "owner/failed-old-evaluation"
    _record_zero_work_retry(
        report,
        arm,
        attempt=1,
        reference=prior,
        plan_sha=old_sha,
        input_sha=campaign.digest(artifacts / "fim/input-bundle/input-manifest.json"),
        commit="c" * 40,
    )
    plan.update(
        {
            "plan_revision": 4,
            "evaluation": {"attention_backend": "memory-efficient"},
            "restart_after_zero_work": {
                "previous_plan_sha256": old_sha,
                "previous_plan_file": "fim_training_plan-r3.json",
            },
        }
    )
    campaign.save(report / "fim_training_plan.json", plan)
    job = campaign.submit_fim(plan, arm=arm, attempt=2, cpt_attempt=1, resume_source=prior)
    assert job["plan_sha256"] == campaign.digest(report / "fim_training_plan.json")
    assert job["plan_sha256"] != old_sha
    assert job["authorization_lineage_reference"] == prior
    assert job["resume_source"] is None
    assert job["resume_identity"] is None
    source = (artifacts / f"fim/kernel-{arm}-2/run.py").read_text()
    assert "'resume_source': None" in source


def test_fim_evaluation_revision_can_restart_verified_zero_work(tmp_path, monkeypatch):
    plan, prior, verified = _retry_plan_revision(tmp_path, monkeypatch)
    campaign.validate_fim_retry_plan(plan, prior, verified)


@pytest.mark.parametrize(
    "change", ["tokens", "checkpoint", "training", "history", "path", "prompt"]
)
def test_fim_evaluation_revision_cannot_resume_changed_training(tmp_path, monkeypatch, change):
    plan, prior, verified = _retry_plan_revision(tmp_path, monkeypatch)
    if change == "tokens":
        verified["processed_arm_input_tokens_conservative"] = 1
    elif change == "checkpoint":
        prior["resume_source"] = "owner/checkpoint"
    elif change == "training":
        plan["configuration"] = {"training": "changed"}
    elif change == "history":
        (tmp_path / "fim_training_plan-r3.json").write_text("{}")
    elif change == "prompt":
        plan["evaluation"]["prompt"] = "different prompt"
    else:
        plan["restart_after_zero_work"]["previous_plan_file"] = "../outside.json"
    campaign.save(tmp_path / "fim_training_plan.json", plan)
    with pytest.raises(ValueError):
        campaign.validate_fim_retry_plan(plan, prior, verified)


def test_submit_fim_zero_work_retry_inherits_only_the_prior_verified_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, artifacts, _quota_calls, _cli_calls = _fim_submission_context(
        tmp_path, monkeypatch
    )
    arm = campaign.FIM_ARMS[0]
    plan_sha = campaign.digest(report / "fim_training_plan.json")
    input_sha = campaign.digest(artifacts / "fim/input-bundle/input-manifest.json")
    commit = "c" * 40
    source = "owner/verified-checkpoint"
    failed = "owner/failed-before-resume-training"
    cursor = {"training_input_tokens": 500_000, "completed_updates": 32}
    identity = {
        "fingerprint": "frozen-fim-fingerprint",
        "checkpoint_sha256": "e" * 64,
        "cursor": cursor,
    }
    _record_zero_work_retry(
        report,
        arm,
        attempt=1,
        reference=source,
        plan_sha=plan_sha,
        input_sha=input_sha,
        commit=commit,
    )
    campaign.save(
        report / f"fim-verified-{arm}-1.json",
        {
            "reference": source,
            "arm": arm,
            "attempt": 1,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": input_sha,
            "commit": commit,
            "checkpoint_verified": True,
            "training_status": "deadline_stop",
            "fingerprint": identity["fingerprint"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "cursor": cursor,
            "own_discarded_tokens_for_resume": 65_536,
            "processed_arm_input_tokens_conservative": 565_536,
        },
    )
    _record_zero_work_retry(
        report,
        arm,
        attempt=2,
        reference=failed,
        plan_sha=plan_sha,
        input_sha=input_sha,
        commit=commit,
        resume_source=source,
        resume_identity=identity,
        own_discarded=65_536,
    )

    job = campaign.submit_fim(plan, arm=arm, attempt=3, cpt_attempt=1, resume_source=failed)

    metadata = json.loads((artifacts / f"fim/kernel-{arm}-3/kernel-metadata.json").read_text())
    assert job["authorization_lineage_reference"] == failed
    assert job["resume_source"] == source
    assert job["resume_identity"] == identity
    assert job["own_discarded_tokens"] == 65_536
    assert metadata["kernel_sources"] == [source]


def test_submit_fim_resumes_checkpoint_from_its_normal_collection_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, report, artifacts, _quota_calls, _submission_calls = _fim_submission_context(
        tmp_path, monkeypatch
    )
    arm = campaign.FIM_ARMS[0]
    plan_path = report / "fim_training_plan.json"
    campaign.save(plan_path, plan)
    manifest_path = artifacts / "fim/input-bundle/input-manifest.json"
    prior = "owner/collected-partial-fim"
    commit = "c" * 40
    job = {
        "reference": prior,
        "arm": arm,
        "attempt": 1,
        "plan_sha256": campaign.digest(plan_path),
        "input_manifest_sha256": campaign.digest(manifest_path),
        "commit": commit,
        "resume_source": None,
        "resume_identity": None,
        "own_discarded_tokens": 0,
        "other_phase_tokens": 0,
        "processed_arm_token_reservation": 6_291_456,
        "conservative_reserved_session_seconds": 10_800,
    }
    campaign.save(report / f"fim-job-{arm}-1.json", job)
    fingerprint = "frozen-fim-fingerprint"
    cursor = {
        "training_input_tokens": 500_000,
        "completed_updates": 32,
        "supervised_target_tokens": 50_000,
    }
    checkpoint_name = "resume-step-000032.pt"
    checkpoint_bytes = b"synthetic verified FIM checkpoint"
    original_cli = campaign.cli

    def collect_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        if call[:3] == ("kaggle", "kernels", "status"):
            return "ERROR"
        if call[:3] != ("kaggle", "kernels", "output"):
            raise AssertionError(f"unexpected collection command: {call[:3]}")
        output = Path(call[call.index("-p") + 1])
        pattern = call[call.index("--file-pattern") + 1]
        training = output / "training"
        training.mkdir(parents=True, exist_ok=True)
        if pattern == r"\.(json|jsonl|log)$":
            campaign.save(
                training / "latest.json",
                {"path": checkpoint_name, "fingerprint": fingerprint, "cursor": cursor},
            )
            campaign.save(
                training / (checkpoint_name + ".complete.json"),
                {
                    "sha256": campaign.hashlib.sha256(checkpoint_bytes).hexdigest(),
                    "fingerprint": fingerprint,
                },
            )
            campaign.save(
                training / "run_result.json",
                {"status": "deadline_stop", "fingerprint": fingerprint, "cursor": cursor},
            )
            (training / "updates.jsonl").write_text(
                json.dumps({"cumulative_input_tokens": 532_768}) + "\n", encoding="utf-8"
            )
            campaign.save(
                output / "worker-status.json",
                {"stages": [{"name": "training", "exit_code": 124}]},
            )
        else:
            (training / checkpoint_name).write_bytes(checkpoint_bytes)
        return ""

    monkeypatch.setattr(campaign, "cli", collect_cli)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=16 * 1024**3),
    )
    collected = campaign.collect_fim(plan, arm, attempt=1)
    assert collected["checkpoint_verified"] is True
    assert collected["training_status"] == "deadline_stop"
    assert collected["plan_sha256"] == job["plan_sha256"]
    assert collected["input_manifest_sha256"] == job["input_manifest_sha256"]
    assert collected["commit"] == commit

    monkeypatch.setattr(campaign, "cli", original_cli)
    resumed = campaign.submit_fim(plan, arm=arm, attempt=2, cpt_attempt=1, resume_source=prior)

    assert resumed["authorization_lineage_reference"] == prior
    assert resumed["resume_source"] == prior
    assert resumed["resume_identity"] == {
        "fingerprint": fingerprint,
        "checkpoint_sha256": campaign.hashlib.sha256(checkpoint_bytes).hexdigest(),
        "cursor": cursor,
    }
    metadata = json.loads((artifacts / f"fim/kernel-{arm}-2/kernel-metadata.json").read_text())
    assert metadata["kernel_sources"] == [prior]


def _fim_bundle_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, artifact_cap: int
) -> tuple[dict[str, Any], Path]:
    root = tmp_path / "repo"
    report = root / "reports/research/q25_code_cpt_r2"
    artifacts = tmp_path / "artifacts"
    corpus = artifacts / "fim/corpus-r3"
    report.mkdir(parents=True)
    corpus.mkdir(parents=True)
    (root / "data/benchmarks").mkdir(parents=True)
    sources = {
        "train.jsonl": corpus / "train.jsonl",
        "development.jsonl": corpus / "development.jsonl",
        "corpus_metadata.json": corpus / "corpus_metadata.json",
        "causal200.jsonl": root / "data/benchmarks/code_completion_v2.jsonl",
    }
    for name, path in sources.items():
        path.write_text(f"synthetic {name}\n", encoding="utf-8")
    line_source = root / "data/preserved/causal_line_v1-r3.jsonl"
    line_source.parent.mkdir(parents=True)
    line_source.write_text("synthetic preserved line fixture\n", encoding="utf-8")
    plan: dict[str, Any] = {
        "schema": "q25-fim-training-plan-v1",
        "gpu_execution_authorized": True,
        "configuration": {
            "budget": {
                "new_artifact_bytes_cap": artifact_cap,
                "minimum_free_bytes": 1,
            }
        },
        "data": {
            "train": {"sha256": campaign.digest(sources["train.jsonl"])},
            "development": {"sha256": campaign.digest(sources["development.jsonl"])},
            "corpus_metadata_sha256": campaign.digest(sources["corpus_metadata.json"]),
        },
        "evaluation": {
            "fixtures": {
                "causal": {"sha256": campaign.digest(sources["causal200.jsonl"])},
                "line": {"sha256": campaign.digest(line_source)},
            }
        },
    }
    campaign.save(report / "fim_training_plan.json", plan)
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(campaign, "FIM_LINE_SOURCE", line_source)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=64 * 1024**3),
    )
    return plan, artifacts


def test_build_fim_bundle_rejects_unapproved_staging_entry_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, artifacts = _fim_bundle_context(tmp_path, monkeypatch, artifact_cap=12 * 1024**3)
    output = artifacts / "fim/input-bundle"
    output.mkdir(parents=True)
    (output / "unapproved.txt").write_text("synthetic sentinel\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unapproved file"):
        campaign.build_fim_bundle(plan)

    assert sorted(path.name for path in output.iterdir()) == ["unapproved.txt"]


def test_build_fim_bundle_checks_projected_storage_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, artifacts = _fim_bundle_context(tmp_path, monkeypatch, artifact_cap=1)
    output = artifacts / "fim/input-bundle"

    with pytest.raises(OSError, match="staging would exhaust storage headroom"):
        campaign.build_fim_bundle(plan)

    assert not list(output.iterdir())


def test_collect_fim_accounts_for_cumulative_tail_once_across_resume_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = 1_000_000
    logged_tail = 32_768
    prior_discarded = 100_000
    plan, report, cli_calls = _fim_collect_context(
        tmp_path,
        monkeypatch,
        committed_tokens=committed,
        logged_tail_tokens=logged_tail,
        prior_discarded_tokens=prior_discarded,
    )
    arm = campaign.FIM_ARMS[0]

    record = campaign.collect_fim(plan, arm, attempt=2)

    own_discarded = prior_discarded + logged_tail + 16 * 1024
    processed = committed + own_discarded
    assert record["discarded_logged_tail_tokens"] == logged_tail
    assert record["unlogged_inflight_input_token_reservation"] == 16 * 1024
    assert record["own_discarded_tokens_for_resume"] == own_discarded
    assert record["processed_arm_input_tokens_conservative"] == processed
    assert campaign.shared_token_ledger() == {"cpt": 12_000_000, arm: processed}
    assert json.loads((report / f"fim-verified-{arm}-2.json").read_text()) == record
    assert [call[2] for call in cli_calls if call[:3] == ("kaggle", "kernels", "output")] == [
        "output",
        "output",
    ]


def test_collect_fim_rejects_malformed_update_log_without_committing_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm = campaign.FIM_ARMS[0]
    plan, report, _cli_calls = _fim_collect_context(
        tmp_path,
        monkeypatch,
        attempt=1,
        updates_content='{"cumulative_input_tokens":1000000}\n{malformed}\n',
    )

    with pytest.raises(ValueError, match="FIM update log contains a malformed record"):
        campaign.collect_fim(plan, arm, attempt=1)

    assert not (report / f"fim-verified-{arm}-1.json").exists()


def _zero_work_settlement_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    attempts: int = 4,
) -> tuple[Path, Path]:
    report = tmp_path / "reports/research/q25_code_cpt_r2"
    artifacts = tmp_path / "artifacts"
    report.mkdir(parents=True)
    artifacts.mkdir()
    campaign.save(
        report / "campaign_budget.json",
        {
            "shared_limits": {
                "aggregate_reserved_session_seconds": 72_000,
                "conservative_account_gpu_hours": 40,
                "minimum_reserved_future_fim_session_seconds": 21_600,
            }
        },
    )
    campaign.save(
        report / "job-1.json",
        {"conservative_reserved_session_seconds": 40_000},
    )
    arm = campaign.FIM_ARMS[0]
    initializer = {
        "kind": "untouched_pretrained",
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "revision": "frozen-test-revision",
        "files": {"model.safetensors": {"sha256": "a" * 64}},
    }
    plan_paths = {
        3: report / "fim_training_plan-r3.json",
        4: report / "fim_training_plan.json",
    }
    plan_hashes: dict[int, str] = {}
    input_paths = {
        3: artifacts / "fim/history/plan-r3/input-bundle/input-manifest.json",
        4: artifacts / "fim/input-bundle/input-manifest.json",
    }
    input_hashes: dict[int, str] = {}
    input_payloads = {
        "train.jsonl": b'{"prompt":"synthetic train"}\n',
        "development.jsonl": b'{"prompt":"synthetic development"}\n',
        "corpus_metadata.json": b'{"schema":"synthetic corpus"}\n',
        "causal200.jsonl": b'{"case":"synthetic causal"}\n',
        "line180.jsonl": b'{"case":"synthetic line"}\n',
    }
    for revision, manifest_path in input_paths.items():
        bundle = manifest_path.parent
        bundle.mkdir(parents=True)
        records: dict[str, dict[str, Any]] = {}
        for name, payload in input_payloads.items():
            staged = bundle / name
            staged.write_bytes(payload)
            records[name] = {"bytes": len(payload), "sha256": campaign.digest(staged)}
        plan = {
            "schema": "q25-fim-training-plan-v1",
            "gpu_execution_authorized": True,
            "plan_revision": revision,
            "configuration": {"budget": {"session_seconds": 10_800}},
            "initializers": {arm: initializer},
            "data": {
                "train": {
                    "file": "train.jsonl",
                    "bytes": records["train.jsonl"]["bytes"],
                    "sha256": records["train.jsonl"]["sha256"],
                },
                "development": {
                    "file": "development.jsonl",
                    "bytes": records["development.jsonl"]["bytes"],
                    "sha256": records["development.jsonl"]["sha256"],
                },
                "corpus_metadata_sha256": records["corpus_metadata.json"]["sha256"],
            },
            "evaluation": {
                "fixtures": {
                    "causal": records["causal200.jsonl"],
                    "line": records["line180.jsonl"],
                }
            },
        }
        plan_path = plan_paths[revision]
        campaign.save(plan_path, plan)
        plan_hashes[revision] = campaign.digest(plan_path)
        staged_plan = bundle / "plan.json"
        staged_plan.write_bytes(plan_path.read_bytes())
        records["plan.json"] = {
            "bytes": staged_plan.stat().st_size,
            "sha256": campaign.digest(staged_plan),
        }
        campaign.save(
            manifest_path,
            {"files": records, "model_dataset": campaign.BASE_DATASET},
        )
        path = manifest_path
        input_hashes[revision] = campaign.digest(path)

    now = RealDatetime.now(UTC).replace(microsecond=0)
    prior_reference: str | None = None
    watch_gaps = (30.25, 60.1, 450.4, 110.01)
    worker_elapsed = (12.0, 16.0, 400.0, 100.0)
    for attempt in range(1, attempts + 1):
        revision = 3 if attempt <= 3 else 4
        submitted_at = now - timedelta(minutes=60 - 10 * attempt)
        watch_at = submitted_at + timedelta(seconds=watch_gaps[attempt - 1])
        verified_at = watch_at + timedelta(seconds=5)
        reference = f"owner/zero-work-attempt-{attempt}"
        job = {
            "reference": reference,
            "requested_reference": reference,
            "status": "submitted",
            "arm": arm,
            "attempt": attempt,
            "commit": f"{attempt:040x}",
            "plan_sha256": plan_hashes[revision],
            "input_manifest_sha256": input_hashes[revision],
            "session_seconds": 10_800,
            "conservative_reserved_session_seconds": 10_800,
            "submitted_at": submitted_at.isoformat(),
            "authorization_lineage_reference": prior_reference,
            "resume_source": None,
            "resume_identity": None,
            "own_discarded_tokens": 0,
            "other_phase_tokens": 123_456,
            "external_campaign_tokens": 123_456,
        }
        status = f'{reference} has status "KernelWorkerStatus.ERROR"'
        campaign.save(report / f"fim-job-{arm}-{attempt}.json", job)
        campaign.save(
            report / f"fim-watch-{arm}-{attempt}.json",
            {
                "reference": reference,
                "status": status,
                "observed_at": watch_at.isoformat(),
                "automatic_allocation": False,
            },
        )
        campaign.save(
            report / f"fim-verified-{arm}-{attempt}.json",
            {
                "reference": reference,
                "status": status,
                "arm": arm,
                "attempt": attempt,
                "plan_sha256": plan_hashes[revision],
                "input_manifest_sha256": input_hashes[revision],
                "commit": job["commit"],
                "checkpoint_verified": False,
                "carried_checkpoint_verified": False,
                "carried_checkpoint_source": None,
                "carried_checkpoint_identity": None,
                "no_training_executed": True,
                "training_status": "no_training_executed",
                "own_discarded_tokens_for_resume": 0,
                "processed_arm_input_tokens_conservative": 0,
                "other_phase_processed_tokens_at_submission": 123_456,
                "observed_at": verified_at.isoformat(),
            },
        )
        output = artifacts / f"fim/output-{arm}-{attempt}/q25_fim_r2"
        output.mkdir(parents=True)
        campaign.save(
            output / "worker-status.json",
            {
                "schema": "q25-fim-kaggle-worker-status-v1",
                "commit": job["commit"],
                "attempt": attempt,
                "arm": arm,
                "plan_sha256": plan_hashes[revision],
                "input_manifest_sha256": input_hashes[revision],
                "training_started": False,
                "state": "failed",
                "elapsed_seconds": worker_elapsed[attempt - 1],
                "stages": [{"name": "trainer-preflight", "exit_code": 0}],
            },
        )
        prior_reference = reference

    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    return report, artifacts


def test_settle_only_verified_zero_work_attempts_and_use_conservative_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _artifacts = _zero_work_settlement_context(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="shared CPT/FIM session reservation"):
        campaign.check_shared_allocation_budget(21_600, phase="fim")

    result = campaign.settle_fim_zero_work_sessions()

    assert result["settled_count"] == 4
    assert result["already_settled_count"] == 0
    ledger = json.loads((report / campaign.SESSION_SETTLEMENTS).read_text())
    entries = ledger["settlements"]
    assert ledger["schema"] == campaign.SESSION_SETTLEMENT_SCHEMA
    assert [entry["settled_session_seconds"] for entry in entries] == [91, 121, 511, 171]
    assert all(entry["original_reserved_session_seconds"] == 10_800 for entry in entries)
    assert all(
        entry["job_sha256"] and entry["watch_sha256"] and entry["verified_sha256"]
        for entry in entries
    )
    assert all(entry["worker_status_sha256"] for entry in entries)

    campaign.check_shared_allocation_budget(21_600, phase="fim")


@pytest.mark.parametrize(
    "mutation",
    [
        "worker_started",
        "worker_elapsed_exceeds_bound",
        "worker_commit_mismatch",
        "carried_checkpoint",
        "token_exposure",
        "boolean_zero_counter",
        "job_resume_source",
        "reference_mismatch",
        "lineage_mismatch",
        "naive_submission_time",
        "watch_before_submission",
        "receipt_before_watch",
        "boolean_session_reservation",
    ],
)
def test_zero_work_settlement_fails_closed_on_tampered_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    report, artifacts = _zero_work_settlement_context(tmp_path, monkeypatch)
    arm = campaign.FIM_ARMS[0]
    job_path = report / f"fim-job-{arm}-1.json"
    watch_path = report / f"fim-watch-{arm}-1.json"
    verified_path = report / f"fim-verified-{arm}-1.json"
    worker_path = artifacts / f"fim/output-{arm}-1/q25_fim_r2/worker-status.json"
    job = json.loads(job_path.read_text())
    watch = json.loads(watch_path.read_text())
    verified = json.loads(verified_path.read_text())
    worker = json.loads(worker_path.read_text())

    if mutation == "worker_started":
        worker["training_started"] = True
        campaign.save(worker_path, worker)
    elif mutation == "worker_elapsed_exceeds_bound":
        worker["elapsed_seconds"] = 50_000
        campaign.save(worker_path, worker)
    elif mutation == "worker_commit_mismatch":
        worker["commit"] = "e" * 40
        campaign.save(worker_path, worker)
    elif mutation == "carried_checkpoint":
        verified["carried_checkpoint_verified"] = True
        verified["carried_checkpoint_source"] = "owner/old-checkpoint"
        verified["carried_checkpoint_identity"] = {"fingerprint": "bad"}
        campaign.save(verified_path, verified)
    elif mutation == "token_exposure":
        verified["processed_arm_input_tokens_conservative"] = 1
        campaign.save(verified_path, verified)
    elif mutation == "boolean_zero_counter":
        job["own_discarded_tokens"] = False
        campaign.save(job_path, job)
    elif mutation == "job_resume_source":
        job["resume_source"] = "owner/unverified-checkpoint"
        job["resume_identity"] = {"fingerprint": "bad"}
        campaign.save(job_path, job)
    elif mutation == "reference_mismatch":
        watch["reference"] = "owner/other-kernel"
        campaign.save(watch_path, watch)
    elif mutation == "lineage_mismatch":
        later_job_path = report / f"fim-job-{arm}-3.json"
        later_job = json.loads(later_job_path.read_text())
        later_job["authorization_lineage_reference"] = "owner/not-the-prior-attempt"
        campaign.save(later_job_path, later_job)
    elif mutation == "naive_submission_time":
        job["submitted_at"] = job["submitted_at"].split("+")[0]
        campaign.save(job_path, job)
    elif mutation == "watch_before_submission":
        submitted = RealDatetime.fromisoformat(job["submitted_at"])
        watch["observed_at"] = (submitted - timedelta(seconds=1)).isoformat()
        campaign.save(watch_path, watch)
    elif mutation == "receipt_before_watch":
        verified["observed_at"] = (
            RealDatetime.fromisoformat(watch["observed_at"]) - timedelta(seconds=1)
        ).isoformat()
        campaign.save(verified_path, verified)
    elif mutation == "boolean_session_reservation":
        job["session_seconds"] = True
        campaign.save(job_path, job)

    with pytest.raises(ValueError):
        campaign.settle_fim_zero_work_sessions()
    assert not (report / campaign.SESSION_SETTLEMENTS).exists()
    with pytest.raises(RuntimeError, match="shared CPT/FIM session reservation"):
        campaign.check_shared_allocation_budget(21_600, phase="fim")


def test_unknown_running_job_keeps_its_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _artifacts = _zero_work_settlement_context(tmp_path, monkeypatch, attempts=1)
    arm = campaign.FIM_ARMS[0]
    watch_path = report / f"fim-watch-{arm}-1.json"
    watch = json.loads(watch_path.read_text())
    watch["status"] = 'owner/zero-work-attempt-1 has status "KernelWorkerStatus.RUNNING"'
    campaign.save(watch_path, watch)

    result = campaign.settle_fim_zero_work_sessions()

    assert result["settled_count"] == 0
    assert result["skipped_unverified_or_nonterminal_count"] == 1
    with pytest.raises(RuntimeError, match="shared CPT/FIM session reservation"):
        campaign.check_shared_allocation_budget(21_600, phase="fim")


def test_completed_training_keeps_its_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _artifacts = _zero_work_settlement_context(tmp_path, monkeypatch)
    arm = campaign.FIM_ARMS[0]
    verified_path = report / f"fim-verified-{arm}-4.json"
    verified = json.loads(verified_path.read_text())
    verified["no_training_executed"] = False
    verified["training_status"] = "complete"
    verified["checkpoint_verified"] = True
    campaign.save(verified_path, verified)

    result = campaign.settle_fim_zero_work_sessions()

    assert result["settled_count"] == 3
    settlements = campaign._load_validated_session_settlements()
    assert "owner/zero-work-attempt-4" not in settlements
    assert "owner/zero-work-attempt-3" in settlements


def test_settlement_requires_manifest_to_bind_plan_and_frozen_input_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, artifacts = _zero_work_settlement_context(tmp_path, monkeypatch)
    arm = campaign.FIM_ARMS[0]
    manifest_path = artifacts / "fim/history/plan-r3/input-bundle/input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["plan.json"]["sha256"] = "f" * 64
    campaign.save(manifest_path, manifest)
    manifest_sha = campaign.digest(manifest_path)
    for attempt in (1, 2, 3):
        job_path = report / f"fim-job-{arm}-{attempt}.json"
        job = json.loads(job_path.read_text())
        job["input_manifest_sha256"] = manifest_sha
        campaign.save(job_path, job)
        verified_path = report / f"fim-verified-{arm}-{attempt}.json"
        verified = json.loads(verified_path.read_text())
        verified["input_manifest_sha256"] = manifest_sha
        campaign.save(verified_path, verified)
        worker_path = artifacts / f"fim/output-{arm}-{attempt}/q25_fim_r2/worker-status.json"
        worker = json.loads(worker_path.read_text())
        worker["input_manifest_sha256"] = manifest_sha
        campaign.save(worker_path, worker)

    with pytest.raises(ValueError, match="input identity differs"):
        campaign._read_zero_work_evidence(arm, 1)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "resume-step-1.pt",
        "unexpected.safetensors",
        "latest.json",
        "updates.jsonl",
        "run_result.json",
    ],
)
def test_settlement_rejects_training_artifacts_anywhere_in_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_name: str
) -> None:
    _report, artifacts = _zero_work_settlement_context(tmp_path, monkeypatch)
    arm = campaign.FIM_ARMS[0]
    stray = artifacts / f"fim/output-{arm}-1/preflight-cache/{artifact_name}"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"synthetic artifact")

    with pytest.raises(ValueError, match="training artifacts"):
        campaign._read_zero_work_evidence(arm, 1)


@pytest.mark.parametrize("tamper", ["receipt_file", "ledger_charge"])
def test_budget_check_rejects_changed_settlement_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    report, _artifacts = _zero_work_settlement_context(tmp_path, monkeypatch)
    campaign.settle_fim_zero_work_sessions()
    arm = campaign.FIM_ARMS[0]
    if tamper == "receipt_file":
        verified_path = report / f"fim-verified-{arm}-1.json"
        verified = json.loads(verified_path.read_text())
        verified["observed_at"] = (
            RealDatetime.fromisoformat(verified["observed_at"]) + timedelta(seconds=1)
        ).isoformat()
        campaign.save(verified_path, verified)
    else:
        ledger_path = report / campaign.SESSION_SETTLEMENTS
        ledger = json.loads(ledger_path.read_text())
        ledger["settlements"][0]["settled_session_seconds"] = 1
        campaign.save(ledger_path, ledger)

    with pytest.raises(ValueError, match="session settlement"):
        campaign.check_shared_allocation_budget(21_600, phase="fim")


def test_settlement_cli_does_not_freeze_or_allocate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_q25_code_cpt.py", "--settle-sessions"])
    monkeypatch.setattr(campaign, "freeze", lambda _config: pytest.fail("settlement froze a plan"))
    monkeypatch.setattr(
        campaign, "cli", lambda *_args, **_kwargs: pytest.fail("settlement called Kaggle")
    )
    monkeypatch.setattr(
        campaign,
        "settle_fim_zero_work_sessions",
        lambda: {"settled_count": 0, "ledger": "local-ledger"},
    )

    campaign.main()

    assert json.loads(capsys.readouterr().out) == {"ledger": "local-ledger", "settled_count": 0}


def _fim_conversion_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Path]]:
    repository = tmp_path / "repo"
    report = repository / "reports/research/q25_code_cpt_r2"
    report.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    model = tmp_path / "qwen"
    model.mkdir()
    original_config = model / "config.json"
    original_tokenizer = model / "tokenizer.json"
    original_weights = model / "model.safetensors"
    original_config.write_text('{"model_type":"qwen2"}\n')
    original_tokenizer.write_text('{"version":"1.0"}\n')
    original_weights.write_bytes(b"original weights fixture")

    def file_record(path: Path) -> dict[str, Any]:
        return {"bytes": path.stat().st_size, "sha256": campaign.digest(path)}

    arm = "untouched_q25_to_fim"
    attempt = 1
    commit = "a" * 40
    fingerprint = "b" * 64
    input_manifest_sha = "c" * 64
    cursor = {
        "attempted_updates": 256,
        "completed_updates": 256,
        "skipped_updates": 0,
        "training_input_tokens": 1_776_908,
        "supervised_target_tokens": 100_000,
        "next_example_index": 4096,
        "epoch": 1,
    }
    training_plan = {
        "schema": "q25-fim-training-plan-v1",
        "configuration": {"training": {"effective_batch": 16}},
        "data": {
            "train": {
                "row_count": 4096,
                "input_tokens": cursor["training_input_tokens"],
                "target_tokens": cursor["supervised_target_tokens"],
            }
        },
        "initializers": {
            "untouched_q25_to_fim": {
                "kind": "untouched_pretrained",
                "model_id": "Qwen/Qwen2.5-Coder-0.5B",
                "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
                "files": {
                    "config.json": file_record(original_config),
                    "tokenizer.json": file_record(original_tokenizer),
                    "model.safetensors": file_record(original_weights),
                },
            },
            "completed_cpt_q25_to_fim": {
                "kind": "completed_cpt_export",
                "model_id": "Qwen/Qwen2.5-Coder-0.5B",
                "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
                "files": {},
                "artifact_manifest_sha256": "d" * 64,
                "fingerprint": "e" * 64,
                "training_cursor": {"completed_updates": 481},
            },
        },
    }
    plan_path = report / "fim_training_plan.json"
    campaign.save(plan_path, training_plan)
    plan_sha = campaign.digest(plan_path)
    campaign.save(report / "fim_quality_comparison.json", {"paired": "synthetic"})
    quality_sha = campaign.digest(report / "fim_quality_comparison.json")

    output_root = artifacts / f"fim/output-{arm}-{attempt}"
    run_root = output_root / "q25_fim_r2"
    export = output_root / campaign.CONVERSION_EXPORT_DIRECTORY
    export.mkdir(parents=True)
    export_files: dict[str, dict[str, Any]] = {}
    for name, content in {
        "config.json": b'{"model_type":"qwen2"}\n',
        "model.safetensors": b"selected FIM model fixture",
        "tokenizer.json": b'{"version":"1.0"}\n',
    }.items():
        path = export / name
        path.write_bytes(content)
        export_files[name] = file_record(path)
        if name == "model.safetensors":
            path.unlink()
    manifest = {
        "schema": "q25-fim-inference-f16-v1",
        "arm": arm,
        "fingerprint": fingerprint,
        "training_cursor": cursor,
        "files": export_files,
    }
    manifest_path = export / "artifact_manifest.json"
    campaign.save(manifest_path, manifest)
    manifest_sha = campaign.digest(manifest_path)

    selection_path = report / "fim_conversion/selection.json"
    selection_path.parent.mkdir(parents=True)
    campaign.save(
        selection_path,
        {
            "schema": "q25-fim-conversion-selection-v1",
            "status": "selected_complete",
            "training_plan_sha256": plan_sha,
            "selected_arm": arm,
            "source_kernel_reference": "owner/selected-fim-kernel",
            "paired_quality_report_sha256": quality_sha,
            "source_export": {
                "directory": campaign.CONVERSION_EXPORT_DIRECTORY,
                "artifact_manifest_sha256": manifest_sha,
                "fingerprint": fingerprint,
                "training_cursor": cursor,
                "files": export_files,
            },
            "tokenizer": {
                "model_id": "Qwen/Qwen2.5-Coder-0.5B",
                "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
                "sha256": campaign.digest(original_tokenizer),
                "eos_token_id": 151643,
                "fim_marker_ids": {
                    "fim_prefix": 151659,
                    "fim_middle": 151660,
                    "fim_suffix": 151661,
                },
            },
            "model": {
                "architecture": "Qwen2ForCausalLM",
                "model_type": "qwen2",
                "logical_parameter_count": 494_032_768,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0},
            },
        },
    )

    job_path = report / f"fim-job-{arm}-{attempt}.json"
    campaign.save(
        job_path,
        {
            "reference": "owner/selected-fim-kernel",
            "arm": arm,
            "attempt": attempt,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": input_manifest_sha,
            "commit": commit,
        },
    )
    verified_path = report / f"fim-verified-{arm}-{attempt}.json"
    checkpoint = output_root / "q25_fim_r2/training/resume-step-256.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"full cursor checkpoint fixture")
    checkpoint_sha = campaign.digest(checkpoint)
    campaign.save(
        verified_path,
        {
            "reference": "owner/selected-fim-kernel",
            "arm": arm,
            "attempt": attempt,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": input_manifest_sha,
            "commit": commit,
            "training_status": "complete",
            "checkpoint_verified": True,
            "checkpoint_sha256": checkpoint_sha,
            "fingerprint": fingerprint,
            "cursor": cursor,
        },
    )
    training_dir = output_root / "q25_fim_r2/training"
    campaign.save(
        training_dir / "latest.json",
        {"path": checkpoint.name, "fingerprint": fingerprint, "cursor": cursor},
    )
    campaign.save(
        training_dir / f"{checkpoint.name}.complete.json",
        {"version": 1, "sha256": checkpoint_sha, "fingerprint": fingerprint},
    )
    campaign.save(
        output_root / "q25_fim_r2/worker-status.json",
        {
            "schema": "q25-fim-kaggle-worker-status-v1",
            "reference": "owner/selected-fim-kernel",
            "commit": commit,
            "attempt": attempt,
            "arm": arm,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": input_manifest_sha,
            "training_started": True,
            "state": "complete",
            "training": {"status": "complete", "cursor": cursor},
        },
    )
    campaign.save(
        training_dir / "run_result.json",
        {
            "status": "complete",
            "arm": arm,
            "fingerprint": fingerprint,
            "cursor": cursor,
            "logical_training_input_tokens": cursor["training_input_tokens"],
        },
    )
    campaign.save(
        run_root / "baseline/identity.json",
        {
            "schema": "q25-fim-baseline-v1",
            "arm": arm,
            "plan_sha256": plan_sha,
            "input_manifest_sha256": input_manifest_sha,
            "files": {"fixture": "hash"},
        },
    )
    for relative, cases in (
        ("before/development/summary.json", 240),
        ("before/line/summary.json", 180),
        ("after/development/summary.json", 240),
        ("after/line/summary.json", 180),
    ):
        campaign.save(run_root / relative, {"cases": cases, "plan_sha256": plan_sha})
    for relative in (
        "after/development/results.jsonl",
        "after/line/results.jsonl",
        "regression/causal/predictions.jsonl",
        "regression/causal/predictions.jsonl.metadata.json",
        "regression/line/predictions.jsonl",
        "regression/line/predictions.jsonl.metadata.json",
    ):
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    code_sources = {
        "conversion_worker.py": repository / "kaggle/q25_fim_conversion_r1/run.py",
        "src/tinycomplete/__init__.py": repository / "src/tinycomplete/__init__.py",
        "src/tinycomplete/code_cpt/__init__.py": (
            repository / "src/tinycomplete/code_cpt/__init__.py"
        ),
        "src/tinycomplete/code_cpt/q25_fim_conversion.py": (
            repository / "src/tinycomplete/code_cpt/q25_fim_conversion.py"
        ),
        "kaggle/q25_fim_conversion_r1/requirements-conversion.lock": (
            repository / "kaggle/q25_fim_conversion_r1/requirements-conversion.lock"
        ),
    }
    for path in code_sources.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic pinned conversion source\n")
    monkeypatch.setattr(campaign, "ROOT", repository)
    monkeypatch.setattr(campaign, "REPORT", report)
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(campaign, "MODEL", model)
    monkeypatch.setattr(campaign, "CONVERSION_SELECTION", selection_path)
    monkeypatch.setattr(
        campaign, "CONVERSION_QUALITY_REPORT", report / "fim_quality_comparison.json"
    )
    monkeypatch.setattr(campaign, "CONVERSION_PLAN", report / "fim_conversion_plan.json")
    monkeypatch.setattr(
        campaign, "CONVERSION_INPUT_BUNDLE", artifacts / "fim/conversion-input-bundle"
    )
    monkeypatch.setattr(campaign, "CONVERSION_KERNEL_ROOT", artifacts / "fim/conversion-kernel")
    monkeypatch.setattr(campaign, "CONVERSION_JOB_FILE", report / "fim-conversion-job.json")
    monkeypatch.setattr(campaign, "CONVERSION_WATCH_FILE", report / "fim-conversion-watch.json")
    monkeypatch.setattr(campaign, "CONVERSION_HISTORY_ROOT", report / "fim/conversion/history")
    monkeypatch.setattr(
        campaign, "CONVERSION_SUBMISSION_FILE", report / "fim-conversion-dataset-submission.json"
    )
    monkeypatch.setattr(campaign, "CONVERSION_OUTPUT", artifacts / "fim/conversion/output")
    monkeypatch.setattr(campaign, "CONVERSION_FAILED_FILE", report / "fim-conversion-failed-1.json")
    monkeypatch.setattr(campaign, "CONVERSION_QUOTA_FILE", report / "fim-conversion-quota.json")
    monkeypatch.setattr(
        campaign, "CONVERSION_VERIFIED_FILE", report / "fim-conversion-verified.json"
    )
    monkeypatch.setattr(campaign, "_conversion_code_files", lambda: code_sources)
    monkeypatch.setattr(
        campaign.shutil, "disk_usage", lambda _path: SimpleNamespace(free=64 * 1024**3)
    )
    return selection_path, report, artifacts, training_plan, code_sources


def _seed_verified_conversion_failure(
    selection_path: Path, artifacts: Path, report: Path
) -> tuple[dict[str, Any], Path, dict[str, bytes]]:
    plan = campaign.freeze_fim_conversion(selection_path)
    bundle = campaign.build_fim_conversion_bundle(plan)
    input_manifest_sha256 = campaign.digest(bundle / "input-manifest.json")
    submitted_at = "2026-10-04T08:08:55.609237+00:00"
    observed_at = "2026-10-04T08:11:04.083435+00:00"
    prior_reference = campaign._conversion_reference(1)
    commit = "a" * 40
    kernel = campaign.build_fim_conversion_kernel(plan, commit)
    campaign.save(
        campaign.CONVERSION_SUBMISSION_FILE,
        {
            "dataset": campaign.CONVERSION_DATASET,
            "state": "verified",
            "input_manifest_sha256": input_manifest_sha256,
        },
    )
    job = {
        "reference": prior_reference,
        "source_kernel_reference": plan["source"]["kernel_reference"],
        "selected_arm": plan["source"]["arm"],
        "attempt": 1,
        "plan_sha256": campaign.digest(campaign.CONVERSION_PLAN),
        "selection_sha256": plan["selection_sha256"],
        "training_plan_sha256": plan["training_plan_sha256"],
        "input_manifest_sha256": input_manifest_sha256,
        "commit": commit,
        "enable_gpu": False,
        "session_seconds": campaign.CONVERSION_SESSION_SECONDS,
        "conservative_reserved_session_seconds": campaign.CONVERSION_SESSION_SECONDS,
        "account_gpu_hours_reserved": 0,
        "paid_compute": False,
        "automatic_renewal_use": False,
        "status": "submitted",
        "submitted_at": submitted_at,
        "automatic_allocation": False,
    }
    watch = {
        "reference": prior_reference,
        "plan_sha256": campaign.digest(campaign.CONVERSION_PLAN),
        "observed_at": observed_at,
        "automatic_allocation": False,
        "status": f'{prior_reference} has status "KernelWorkerStatus.ERROR"',
    }
    campaign.save(campaign.CONVERSION_JOB_FILE, job)
    campaign.save(campaign.CONVERSION_WATCH_FILE, watch)
    failure_log = artifacts / "fim/conversion-failure-r1/tc-q25-fim-q4-conversion-r1.log"
    failure_log.parent.mkdir(parents=True)
    failure_log.write_text("bounded launcher error fixture\n")
    launcher_sha256 = campaign.digest(kernel / "run.py")
    failure = {
        "schema": "q25-fim-cpu-conversion-failure-v1",
        "reference": prior_reference,
        "attempt": 1,
        "job_sha256": campaign.digest(campaign.CONVERSION_JOB_FILE),
        "watch_sha256": campaign.digest(campaign.CONVERSION_WATCH_FILE),
        "plan_sha256": campaign.digest(campaign.CONVERSION_PLAN),
        "selection_sha256": plan["selection_sha256"],
        "source_kernel_reference": plan["source"]["kernel_reference"],
        "launcher_sha256": launcher_sha256,
        "pulled_launcher_sha256": launcher_sha256,
        "failure_log_sha256": campaign.digest(failure_log),
        "failure_log_file": failure_log.relative_to(artifacts).as_posix(),
        "terminal_status": "ERROR",
        "failure_kind": "launcher_pre_worker",
        "worker_started": False,
        "conversion_started": False,
        "gpu_enabled": False,
        "training_input_tokens": 0,
        "automatic_retry": False,
        "conservative_reserved_session_seconds": campaign.CONVERSION_SESSION_SECONDS,
        "observed_at": observed_at,
        "launcher_failure_observed_seconds": 0.8,
        "remote_inputs_verified": {
            "dataset_sources": [campaign.CONVERSION_DATASET],
            "enable_gpu": False,
            "kernel_sources": [plan["source"]["kernel_reference"]],
        },
        "remote_metadata_sha256": "1" * 64,
        "telemetry_query_state": "no_runs",
        "telemetry_query_sha256": "2" * 64,
    }
    campaign.save(report / "fim-conversion-failed-1.json", failure)
    original = {
        "job": campaign.CONVERSION_JOB_FILE.read_bytes(),
        "watch": campaign.CONVERSION_WATCH_FILE.read_bytes(),
        "failure": (report / "fim-conversion-failed-1.json").read_bytes(),
        "launcher": (kernel / "run.py").read_bytes(),
    }
    return plan, kernel, original


def test_conversion_freeze_binds_manual_arm_quality_receipt_and_cpu_only_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )

    plan = campaign.freeze_fim_conversion(selection_path)

    assert plan["source"]["kernel_reference"] == "owner/selected-fim-kernel"
    assert plan["source"]["paired_quality_report_sha256"] == campaign.digest(
        report / "fim_quality_comparison.json"
    )
    assert plan["execution"]["enable_gpu"] is False
    assert plan["execution"]["session_seconds"] == 10_800
    assert plan["execution"]["finalization_reserve_seconds"] == 1_800


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_seconds", 10_799),
        ("finalization_reserve_seconds", 1_799),
        ("enable_gpu", True),
        ("artifact_bytes_cap", 1),
        ("minimum_free_bytes", 1),
    ],
)
def test_conversion_loader_rejects_plan_with_changed_runtime_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    plan["execution"][field] = value
    campaign.save(campaign.CONVERSION_PLAN, plan)

    with pytest.raises(ValueError, match="frozen CPU conversion plan"):
        campaign._load_conversion_plan()


@pytest.mark.parametrize("tamper", ["quality_report", "cursor", "fingerprint"])
def test_conversion_freeze_rejects_mismatched_manual_selection_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    selection_path, report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    selection = json.loads(selection_path.read_text())
    if tamper == "quality_report":
        selection["paired_quality_report_sha256"] = "f" * 64
    elif tamper == "cursor":
        selection["source_export"]["training_cursor"]["completed_updates"] = 255
    else:
        selection["source_export"]["fingerprint"] = "f" * 64
    campaign.save(selection_path, selection)

    with pytest.raises(ValueError):
        campaign.freeze_fim_conversion(selection_path)

    assert not (report / "fim_conversion_plan.json").exists()


def test_conversion_selection_rejects_duplicate_source_job_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    campaign.save(
        report / "fim-job-untouched_q25_to_fim-2.json",
        {"reference": "owner/selected-fim-kernel", "arm": "untouched_q25_to_fim"},
    )

    with pytest.raises(ValueError, match="exactly one"):
        campaign.freeze_fim_conversion(selection_path)


def test_conversion_input_bundle_contains_only_config_and_selection_not_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    campaign.freeze_fim_conversion(selection_path)

    bundle = campaign.build_fim_conversion_bundle()

    assert {path.name for path in bundle.iterdir()} == {
        "selection.json",
        "training_plan.json",
        "original_config.json",
        "original_tokenizer.json",
        "input-manifest.json",
        "dataset-metadata.json",
    }
    assert not list(bundle.glob("*.safetensors"))
    assert not list(bundle.glob("*.gguf"))


def test_conversion_upload_verifies_kaggle_payloads_and_worker_hashes_all_config_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    bundle = campaign.build_fim_conversion_bundle(plan)
    manifest = json.loads((bundle / "input-manifest.json").read_text())
    payload_names = {
        "selection.json",
        "training_plan.json",
        "original_config.json",
        "original_tokenizer.json",
    }
    assert set(manifest["files"]) == payload_names
    for name, record in manifest["files"].items():
        assert record == {
            "bytes": (bundle / name).stat().st_size,
            "sha256": campaign.digest(bundle / name),
        }

    remote_rows = [
        {"name": name, "size": (bundle / name).stat().st_size}
        for name in sorted((*payload_names, "input-manifest.json"))
    ]
    remote_calls: list[tuple[str, ...]] = []

    def fake_run(args: list[str], *, timeout: int = 90) -> str:
        del timeout
        remote_calls.append(tuple(args))
        return json.dumps(remote_rows)

    monkeypatch.setattr(build_pilot, "_csv_refs", lambda _args: set())
    monkeypatch.setattr(build_pilot, "_run", fake_run)
    monkeypatch.setattr(
        campaign,
        "cli",
        lambda *_args, **_kwargs: "Your private Dataset is being created.",
    )
    monkeypatch.setattr(
        build_pilot.time,
        "sleep",
        lambda _seconds: pytest.fail("published five-file dataset should verify immediately"),
    )

    verification = campaign.upload_fim_conversion_bundle(plan)

    assert verification == {
        "verified_files": 5,
        "remote_files": 5,
        "paths_and_sizes_verified": True,
        "hashes_verified_by_worker": False,
    }
    assert len(remote_calls) == 1
    assert remote_calls[0][:4] == (
        "kaggle",
        "datasets",
        "files",
        campaign.CONVERSION_DATASET,
    )
    assert {row["name"] for row in remote_rows} == payload_names | {"input-manifest.json"}
    assert "dataset-metadata.json" not in {row["name"] for row in remote_rows}

    kernel = campaign.build_fim_conversion_kernel(plan, "f" * 40)
    launcher = (kernel / "run.py").read_text()
    assert 'sha(path) != record.get("sha256")' in launcher
    assert set(json.loads((bundle / "input-manifest.json").read_text())["files"]) == payload_names


def _symlink_hf_model_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    escape_blob_root: bool,
) -> None:
    original_model = campaign.MODEL
    model_cache = tmp_path / "hf-cache/models--Qwen--Qwen2.5-Coder-0.5B"
    blob_root = model_cache / "blobs"
    snapshot = model_cache / "snapshots/revision"
    blob_root.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside-hf-cache"
    outside.mkdir()
    for name in ("config.json", "tokenizer.json"):
        content = (original_model / name).read_bytes()
        target_root = outside if escape_blob_root else blob_root
        target = target_root / f"blob-{name}"
        target.write_bytes(content)
        (snapshot / name).symlink_to(target)
    monkeypatch.setattr(campaign, "MODEL", snapshot)


def test_conversion_bundle_resolves_hf_snapshot_symlinks_only_inside_blob_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    _symlink_hf_model_snapshot(tmp_path, monkeypatch, escape_blob_root=False)
    campaign.freeze_fim_conversion(selection_path)

    bundle = campaign.build_fim_conversion_bundle()

    for name, digest_field in (
        ("original_config.json", "original_q25_config_sha256"),
        ("original_tokenizer.json", "original_q25_tokenizer_sha256"),
    ):
        staged = bundle / name
        assert staged.is_file() and not staged.is_symlink()
        assert campaign.digest(staged) == campaign._load_conversion_plan()[digest_field]


def test_conversion_bundle_rejects_snapshot_symlink_outside_blob_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    _symlink_hf_model_snapshot(tmp_path, monkeypatch, escape_blob_root=True)
    campaign.freeze_fim_conversion(selection_path)

    with pytest.raises(ValueError, match="outside its snapshot"):
        campaign.build_fim_conversion_bundle()


def test_conversion_bundle_fails_closed_on_storage_cap_or_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    campaign.freeze_fim_conversion(selection_path)
    monkeypatch.setattr(
        campaign, "directory_bytes", lambda _root: campaign.CONVERSION_ARTIFACT_CAP_BYTES
    )
    with pytest.raises(OSError, match="artifact cap"):
        campaign.build_fim_conversion_bundle()

    monkeypatch.setattr(campaign, "directory_bytes", lambda _root: 0)
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="headroom"):
        campaign.build_fim_conversion_bundle()


def test_conversion_kernel_is_cpu_only_and_attaches_only_selected_source_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    campaign.freeze_fim_conversion(selection_path)

    kernel = campaign.build_fim_conversion_kernel(campaign._load_conversion_plan(), "f" * 40)

    metadata = json.loads((kernel / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == [campaign.CONVERSION_DATASET]
    assert metadata["kernel_sources"] == ["owner/selected-fim-kernel"]
    assert not any(
        "accelerator" in line.lower() for line in (kernel / "run.py").read_text().splitlines()
    )


@pytest.mark.parametrize("attempt", [0, 3, True])
def test_conversion_attempt_reference_is_bounded(attempt: int) -> None:
    with pytest.raises(ValueError, match="attempts 1 and 2"):
        campaign._conversion_reference(attempt)


def test_conversion_launcher_runs_as_single_file_and_finds_nested_kaggle_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runpy

    selection_path, _report, artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    kernel = campaign.build_fim_conversion_kernel(plan, "f" * 40)
    launcher = (kernel / "run.py").read_text()

    # Kaggle sends only script_body, so the uploaded kernel contains no siblings.
    uploaded_kernel = tmp_path / "uploaded-kernel"
    uploaded_kernel.mkdir()
    (uploaded_kernel / "run.py").write_text(launcher)
    assert [path.name for path in uploaded_kernel.iterdir()] == ["run.py"]

    kaggle_root = tmp_path / "kaggle"
    kaggle_root.mkdir()
    input_root = kaggle_root / "input"
    config_root = input_root / "datasets" / "nested" / "private-config"
    config_root.mkdir(parents=True)
    bundle = campaign.build_fim_conversion_bundle(plan)
    for source in bundle.iterdir():
        if source.name != "dataset-metadata.json":
            (config_root / source.name).write_bytes(source.read_bytes())

    source_root = input_root / "notebooks" / "nested" / "selected-kernel"
    export_root = source_root / campaign.CONVERSION_EXPORT_DIRECTORY
    export_root.mkdir(parents=True)
    source_manifest = (
        artifacts
        / "fim/output-untouched_q25_to_fim-1"
        / campaign.CONVERSION_EXPORT_DIRECTORY
        / "artifact_manifest.json"
    )
    (export_root / "artifact_manifest.json").write_bytes(source_manifest.read_bytes())

    temp_root = kaggle_root / "temp"
    assert not temp_root.exists()
    output_root = tmp_path / "worker-output"
    output_root.mkdir()
    captured: dict[str, Any] = {}

    def fake_run_path(path: str, *, run_name: str) -> dict[str, Any]:
        captured["worker"] = Path(path)
        captured["run_name"] = run_name
        captured["argv"] = list(sys.argv)
        return {}

    monkeypatch.setattr(runpy, "run_path", fake_run_path)
    executable = (
        launcher.replace('Path("/kaggle")', f"Path({str(kaggle_root)!r})")
        .replace('Path("/kaggle/input")', f"Path({str(input_root)!r})")
        .replace('Path("/kaggle/temp")', f"Path({str(temp_root)!r})")
    )
    executable = executable.replace(
        '"/kaggle/working/q25_fim_conversion_r1"', repr(str(output_root))
    ).replace('"/kaggle/temp/q25_fim_conversion_r1"', repr(str(temp_root / "runtime")))
    exec(compile(executable, str(uploaded_kernel / "run.py"), "exec"), {"__name__": "__main__"})

    code_root = temp_root / "q25_fim_conversion_code"
    expected_paths = set(campaign.CONVERSION_EMBEDDED_SOURCE_PATHS)
    assert {
        path.relative_to(code_root).as_posix() for path in code_root.rglob("*") if path.is_file()
    } == expected_paths
    assert temp_root.is_dir()
    assert {relative: campaign.digest(code_root / relative) for relative in expected_paths} == plan[
        "source_code"
    ]
    assert captured["worker"] == code_root / "conversion_worker.py"
    assert captured["run_name"] == "__main__"
    arguments = captured["argv"]
    assert arguments[arguments.index("--input-root") + 1] == str(input_root)
    assert arguments[arguments.index("--selection") + 1] == str(config_root / "selection.json")
    assert arguments[arguments.index("--source-root") + 1] == str(source_root)
    assert arguments[arguments.index("--source-kernel-reference") + 1] == (
        "owner/selected-fim-kernel"
    )


def test_conversion_launcher_reports_bounded_manifest_diagnostic_for_missing_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    kernel = campaign.build_fim_conversion_kernel(plan, "f" * 40)
    launcher = (kernel / "run.py").read_text()

    kaggle_root = tmp_path / "kaggle"
    input_root = kaggle_root / "input"
    input_root.mkdir(parents=True)
    temp_root = kaggle_root / "temp"
    assert not temp_root.exists()
    executable = (
        launcher.replace('Path("/kaggle")', f"Path({str(kaggle_root)!r})")
        .replace('Path("/kaggle/input")', f"Path({str(input_root)!r})")
        .replace('Path("/kaggle/temp")', f"Path({str(temp_root)!r})")
    )

    with pytest.raises(SystemExit, match="stopped before the worker"):
        exec(compile(executable, str(kernel / "run.py"), "exec"), {"__name__": "__main__"})

    lines = capsys.readouterr().err.splitlines()
    diagnostic_lines = [line for line in lines if line.startswith("Q25_FIM_CONVERSION_DIAGNOSTIC ")]
    assert len(diagnostic_lines) == 1
    diagnostic = json.loads(diagnostic_lines[0].split(" ", maxsplit=1)[1])
    assert diagnostic["schema"] == "q25-fim-conversion-launcher-diagnostic-v1"
    assert diagnostic["stage"] == "config_manifest_discovery"
    assert diagnostic["code"] == "config_manifest_missing_or_ambiguous"
    assert diagnostic["temp_root_ready"] is True
    assert diagnostic["manifest_inventory"] == {
        "artifact_manifest_candidates": 0,
        "config_hash_matches": 0,
        "entries_scanned": 0,
        "input_manifest_candidates": 0,
        "manifest_candidates": 0,
        "source_hash_matches": 0,
        "source_path_matches": 0,
    }
    assert len(diagnostic_lines[0].encode("utf-8")) < 2048
    assert str(tmp_path) not in diagnostic_lines[0]


def test_conversion_launcher_rejects_symlinked_temp_root_without_writing_through_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    kernel = campaign.build_fim_conversion_kernel(plan, "f" * 40)
    launcher = (kernel / "run.py").read_text()
    kaggle_root = tmp_path / "kaggle"
    kaggle_root.mkdir()
    outside_temp = tmp_path / "outside-temp"
    outside_temp.mkdir()
    sentinel = outside_temp / "sentinel"
    sentinel.write_text("preserve")
    (kaggle_root / "temp").symlink_to(outside_temp, target_is_directory=True)
    input_root = kaggle_root / "input"
    input_root.mkdir()
    executable = (
        launcher.replace('Path("/kaggle")', f"Path({str(kaggle_root)!r})")
        .replace('Path("/kaggle/input")', f"Path({str(input_root)!r})")
        .replace('Path("/kaggle/temp")', f"Path({str(kaggle_root / 'temp')!r})")
    )

    with pytest.raises(SystemExit, match="stopped before the worker"):
        exec(compile(executable, str(kernel / "run.py"), "exec"), {"__name__": "__main__"})

    diagnostic_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("Q25_FIM_CONVERSION_DIAGNOSTIC ")
    ]
    assert len(diagnostic_lines) == 1
    diagnostic = json.loads(diagnostic_lines[0].split(" ", maxsplit=1)[1])
    assert diagnostic["stage"] == "temp_root"
    assert diagnostic["code"] == "temp_root_symlink"
    assert sentinel.read_text() == "preserve"
    assert list(outside_temp.iterdir()) == [sentinel]


def test_conversion_quota_rejects_gpu_enabled_plan_before_any_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    plan["execution"]["enable_gpu"] = True

    with pytest.raises(ValueError, match="must not enable GPU"):
        campaign._check_conversion_quota(plan, _observation())


def test_conversion_submit_blocks_on_active_job_without_kernel_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    bundle = campaign.build_fim_conversion_bundle(plan)
    campaign.save(
        campaign.CONVERSION_SUBMISSION_FILE,
        {
            "dataset": campaign.CONVERSION_DATASET,
            "state": "verified",
            "input_manifest_sha256": campaign.digest(bundle / "input-manifest.json"),
        },
    )
    campaign.save(
        report / "campaign_budget.json",
        {"shared_limits": {"quota_renewal": "2026-10-10T00:00:00"}},
    )
    calls: list[tuple[str, ...]] = []
    commit = "f" * 40

    def fake_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        calls.append(call)
        if call[-2:] == ("branch", "--show-current"):
            return campaign.CONVERSION_BRANCH
        if call[-2:] == ("status", "--porcelain"):
            return ""
        if call[-2:] == ("rev-parse", "HEAD"):
            return commit
        if call[-2:] == ("origin", f"refs/heads/{campaign.CONVERSION_BRANCH}"):
            return f"{commit}\trefs/heads/{campaign.CONVERSION_BRANCH}"
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    monkeypatch.setattr(
        campaign, "quota", lambda: _observation(active_jobs=[{"status": "RUNNING"}])
    )
    monkeypatch.setattr(campaign, "check_shared_allocation_budget", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="another Kaggle job is active"):
        campaign.submit_fim_conversion(plan)

    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in calls)
    assert not campaign.CONVERSION_JOB_FILE.exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_resume",
        "job_hash",
        "terminal_watch",
        "worker_started",
        "training_tokens",
        "automatic_retry",
        "failure_log",
        "prior_launcher",
        "telemetry_query",
    ],
)
def test_conversion_retry_rejects_unverified_or_nonzero_work_before_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    selection_path, report, artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan, prior_kernel, _original = _seed_verified_conversion_failure(
        selection_path, artifacts, report
    )
    failure_path = report / "fim-conversion-failed-1.json"
    failure = json.loads(failure_path.read_text())
    if tamper == "wrong_resume":
        resume_source = "owner/other-kernel"
    else:
        resume_source = campaign._conversion_reference(1)
    if tamper == "job_hash":
        failure["job_sha256"] = "f" * 64
    elif tamper == "terminal_watch":
        watch = json.loads(campaign.CONVERSION_WATCH_FILE.read_text())
        watch["status"] = "RUNNING"
        campaign.save(campaign.CONVERSION_WATCH_FILE, watch)
        failure["watch_sha256"] = campaign.digest(campaign.CONVERSION_WATCH_FILE)
    elif tamper == "worker_started":
        failure["worker_started"] = True
    elif tamper == "training_tokens":
        failure["training_input_tokens"] = 1
    elif tamper == "automatic_retry":
        failure["automatic_retry"] = True
    elif tamper == "failure_log":
        log_path = artifacts / failure["failure_log_file"]
        log_path.write_text("changed failure output\n")
    elif tamper == "prior_launcher":
        (prior_kernel / "run.py").write_text("changed launcher\n")
    elif tamper == "telemetry_query":
        failure["telemetry_query_state"] = "unknown"
    campaign.save(failure_path, failure)

    calls: list[tuple[str, ...]] = []

    def fake_cli(*args: str, timeout: int = 120) -> str:
        del timeout
        calls.append(tuple(args))
        if args[-2:] == ("branch", "--show-current"):
            return campaign.CONVERSION_BRANCH
        if args[-2:] == ("status", "--porcelain"):
            return ""
        if args[-2:] == ("rev-parse", "HEAD"):
            return "f" * 40
        if args[-2:] == ("origin", f"refs/heads/{campaign.CONVERSION_BRANCH}"):
            return f"{'f' * 40}\trefs/heads/{campaign.CONVERSION_BRANCH}"
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    monkeypatch.setattr(
        campaign, "quota", lambda: pytest.fail("invalid retry evidence reached quota")
    )

    with pytest.raises(ValueError, match="conversion retry"):
        campaign.submit_fim_conversion(plan, attempt=2, resume_source=resume_source)

    assert not campaign.CONVERSION_HISTORY_ROOT.exists()
    assert not campaign._conversion_kernel_root(2).exists()
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in calls)


def test_conversion_retry_archives_attempt_one_and_submits_distinct_cpu_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan, prior_kernel, original = _seed_verified_conversion_failure(
        selection_path, artifacts, report
    )
    commit = "f" * 40
    calls: list[tuple[str, ...]] = []
    budget_calls: list[tuple[int, str]] = []

    def fake_cli(*args: str, timeout: int = 120) -> str:
        del timeout
        call = tuple(args)
        calls.append(call)
        if call[-2:] == ("branch", "--show-current"):
            return campaign.CONVERSION_BRANCH
        if call[-2:] == ("status", "--porcelain"):
            return ""
        if call[-2:] == ("rev-parse", "HEAD"):
            return commit
        if call[-2:] == ("origin", f"refs/heads/{campaign.CONVERSION_BRANCH}"):
            return f"{commit}\trefs/heads/{campaign.CONVERSION_BRANCH}"
        if call[:3] == ("kaggle", "kernels", "status"):
            return "COMPLETE"
        if call[:3] == ("kaggle", "kernels", "push"):
            kernel_root = Path(call[call.index("-p") + 1])
            assert kernel_root == campaign._conversion_kernel_root(2)
            metadata = json.loads((kernel_root / "kernel-metadata.json").read_text())
            assert metadata["id"] == campaign._conversion_reference(2)
            assert metadata["enable_gpu"] is False
            assert metadata["dataset_sources"] == [campaign.CONVERSION_DATASET]
            assert metadata["kernel_sources"] == [plan["source"]["kernel_reference"]]
            return "https://www.kaggle.com/code/shlokbhakta/tc-q25-fim-q4-conversion-r1-a2"
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    monkeypatch.setattr(campaign, "quota", lambda: _observation())
    monkeypatch.setattr(
        campaign,
        "check_shared_allocation_budget",
        lambda seconds, *, phase: budget_calls.append((seconds, phase)),
    )
    monkeypatch.setattr(campaign, "_csv_ref_exists", lambda _reference: False)
    campaign.save(
        report / "campaign_budget.json",
        {"shared_limits": {"quota_renewal": "2026-10-10T00:00:00"}},
    )

    job = campaign.submit_fim_conversion(
        plan, attempt=2, resume_source=campaign._conversion_reference(1)
    )

    archive = campaign.CONVERSION_HISTORY_ROOT / "attempt-1"
    assert {path.name for path in archive.iterdir()} == {
        "failure.json",
        "job.json",
        "run.py",
        "watch.json",
    }
    assert (archive / "job.json").read_bytes() == original["job"]
    assert (archive / "watch.json").read_bytes() == original["watch"]
    assert (archive / "failure.json").read_bytes() == original["failure"]
    assert (archive / "run.py").read_bytes() == original["launcher"]
    assert (prior_kernel / "run.py").read_bytes() == original["launcher"]
    assert job["attempt"] == 2
    assert job["reference"] == campaign._conversion_reference(2)
    assert job["retry_authorization_reference"] == campaign._conversion_reference(1)
    assert job["enable_gpu"] is False
    assert campaign._conversion_kernel_root(2).is_dir()
    assert json.loads(campaign.CONVERSION_JOB_FILE.read_text())["attempt"] == 2
    assert json.loads(campaign.CONVERSION_WATCH_FILE.read_text())["reference"] == job["reference"]
    assert budget_calls == [(campaign.CONVERSION_SESSION_SECONDS, "conversion")]
    assert sum(call[:3] == ("kaggle", "kernels", "push") for call in calls) == 1


def test_conversion_retry_still_requires_a_fresh_idle_quota_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan, _prior_kernel, original = _seed_verified_conversion_failure(
        selection_path, artifacts, report
    )
    campaign.save(
        report / "campaign_budget.json",
        {"shared_limits": {"quota_renewal": "2026-10-10T00:00:00"}},
    )
    commit = "f" * 40
    calls: list[tuple[str, ...]] = []

    def fake_cli(*args: str, timeout: int = 120) -> str:
        del timeout
        calls.append(tuple(args))
        if args[-2:] == ("branch", "--show-current"):
            return campaign.CONVERSION_BRANCH
        if args[-2:] == ("status", "--porcelain"):
            return ""
        if args[-2:] == ("rev-parse", "HEAD"):
            return commit
        if args[-2:] == ("origin", f"refs/heads/{campaign.CONVERSION_BRANCH}"):
            return f"{commit}\trefs/heads/{campaign.CONVERSION_BRANCH}"
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    monkeypatch.setattr(
        campaign,
        "quota",
        lambda: _observation(active_jobs=[{"reference": "owner/other", "status": "RUNNING"}]),
    )
    monkeypatch.setattr(campaign, "check_shared_allocation_budget", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        campaign,
        "_csv_ref_exists",
        lambda _reference: pytest.fail("active job must stop before duplicate lookup"),
    )

    with pytest.raises(RuntimeError, match="another Kaggle job is active"):
        campaign.submit_fim_conversion(
            plan, attempt=2, resume_source=campaign._conversion_reference(1)
        )

    assert campaign.CONVERSION_JOB_FILE.read_bytes() == original["job"]
    assert campaign.CONVERSION_WATCH_FILE.read_bytes() == original["watch"]
    assert not campaign.CONVERSION_HISTORY_ROOT.exists()
    assert not campaign._conversion_kernel_root(2).exists()
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in calls)


def test_shared_budget_counts_archived_and_active_conversion_receipts_once_per_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _selection, report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    job = {
        "reference": campaign._conversion_reference(1),
        "attempt": 1,
        "conservative_reserved_session_seconds": campaign.CONVERSION_SESSION_SECONDS,
        "enable_gpu": False,
    }
    campaign.save(campaign.CONVERSION_JOB_FILE, job)
    archive = campaign.CONVERSION_HISTORY_ROOT / "attempt-1"
    archive.mkdir(parents=True)
    (archive / "job.json").write_bytes(campaign.CONVERSION_JOB_FILE.read_bytes())
    fim_job_path = next(report.glob("fim-job-*.json"))
    fim_job = json.loads(fim_job_path.read_text())
    fim_job["conservative_reserved_session_seconds"] = campaign.CONVERSION_SESSION_SECONDS
    campaign.save(fim_job_path, fim_job)
    campaign.save(
        report / "campaign_budget.json",
        {
            "shared_limits": {
                "aggregate_reserved_session_seconds": 3 * campaign.CONVERSION_SESSION_SECONDS,
                "minimum_reserved_future_fim_session_seconds": 0,
                "conservative_account_gpu_hours": 40,
            }
        },
    )

    campaign.check_shared_allocation_budget(campaign.CONVERSION_SESSION_SECONDS, phase="conversion")

    second_attempt = {
        **job,
        "reference": campaign._conversion_reference(2),
        "attempt": 2,
    }
    campaign.save(campaign.CONVERSION_JOB_FILE, second_attempt)
    campaign.check_shared_allocation_budget(0, phase="conversion")

    limits = json.loads((report / "campaign_budget.json").read_text())
    limits["shared_limits"]["aggregate_reserved_session_seconds"] -= 1
    campaign.save(report / "campaign_budget.json", limits)
    with pytest.raises(RuntimeError, match="session reservation exhausted"):
        campaign.check_shared_allocation_budget(0, phase="conversion")


def test_conversion_cli_has_separate_non_gpu_action_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_q25_code_cpt.py", "--bundle-conversion"])
    monkeypatch.setattr(
        campaign, "freeze", lambda _config: pytest.fail("normal GPU plan was frozen")
    )
    monkeypatch.setattr(campaign, "_load_conversion_plan", lambda: {"schema": "cpu-only"})
    monkeypatch.setattr(
        campaign, "build_fim_conversion_bundle", lambda _plan: Path("/private/config-bundle")
    )

    campaign.main()

    assert json.loads(capsys.readouterr().out) == {"conversion_bundle": "/private/config-bundle"}


def test_conversion_cli_forwards_explicit_retry_lineage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prior_reference = campaign._conversion_reference(1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_q25_code_cpt.py",
            "--execute-conversion",
            "--attempt",
            "2",
            "--resume-source",
            prior_reference,
        ],
    )
    monkeypatch.setattr(campaign, "_load_conversion_plan", lambda: {"schema": "frozen"})
    captured: dict[str, Any] = {}

    def fake_submit(
        _plan: dict[str, Any], *, attempt: int, resume_source: str | None
    ) -> dict[str, Any]:
        captured.update(attempt=attempt, resume_source=resume_source)
        return {"attempt": attempt, "reference": campaign._conversion_reference(attempt)}

    monkeypatch.setattr(campaign, "submit_fim_conversion", fake_submit)

    campaign.main()

    assert captured == {"attempt": 2, "resume_source": prior_reference}
    assert json.loads(capsys.readouterr().out) == {
        "conversion_job": {"attempt": 2, "reference": campaign._conversion_reference(2)}
    }


def test_conversion_revision2_freeze_uses_separate_plan_and_preserves_r1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    original = campaign.freeze_fim_conversion(selection_path)
    original_bytes = campaign.CONVERSION_PLAN.read_bytes()

    with campaign._conversion_revision_scope(2):
        revised = campaign.freeze_fim_conversion(selection_path)
        assert revised["plan_revision"] == 2
        assert campaign.CONVERSION_PLAN == report / "fim_conversion_r2_plan.json"
        assert campaign.CONVERSION_KERNEL_REFERENCE == campaign.CONVERSION_REFERENCE_R2
        assert campaign._conversion_reference(1) == campaign.CONVERSION_REFERENCE_R2
        with pytest.raises(ValueError, match="revision 2"):
            campaign._conversion_reference(2)
        assert revised["selection_sha256"] == original["selection_sha256"]
        assert revised["training_plan_sha256"] == original["training_plan_sha256"]
        assert revised["source"] == original["source"]
        assert revised["source_code"] == original["source_code"]

    assert campaign.CONVERSION_PLAN == report / "fim_conversion_plan.json"
    assert campaign.CONVERSION_PLAN.read_bytes() == original_bytes
    assert (report / "fim_conversion_r2_plan.json").is_file()


def test_conversion_revision2_kernel_has_separate_reference_output_and_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    with campaign._conversion_revision_scope(2):
        plan = campaign.freeze_fim_conversion(selection_path)
        kernel = campaign.build_fim_conversion_kernel(plan, "f" * 40)

        metadata = json.loads((kernel / "kernel-metadata.json").read_text())
        launcher = (kernel / "run.py").read_text()
        assert kernel == artifacts / "fim/conversion-kernel-r2"
        assert metadata["id"] == "shlokbhakta/tc-q25-fim-q4-conversion-r2"
        assert metadata["enable_gpu"] is False
        assert '"/kaggle/working/q25_fim_conversion_r2"' in launcher
        assert '"/kaggle/temp/q25_fim_conversion_r2"' in launcher
        assert '"/kaggle/working/q25_fim_conversion_r1"' not in launcher
        assert campaign.CONVERSION_JOB_FILE == report / "fim-conversion-job-r2.json"
        assert campaign.CONVERSION_WATCH_FILE == report / "fim-conversion-watch-r2.json"
        assert campaign.CONVERSION_OUTPUT == artifacts / "fim/conversion/output-r2"

    assert not (report / "fim-conversion-job.json").exists()
    assert not (artifacts / "fim/conversion-kernel").exists()


def test_conversion_revision2_submit_ignores_but_budgets_preserved_r1_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, report, artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    fim_job_path = next(report.glob("fim-job-*.json"))
    fim_job = json.loads(fim_job_path.read_text())
    fim_job["conservative_reserved_session_seconds"] = 0
    campaign.save(fim_job_path, fim_job)

    r1_reservation = campaign.CONVERSION_SESSION_SECONDS
    r1_job = {
        "reference": campaign.CONVERSION_REFERENCE_R1,
        "attempt": 1,
        "plan_revision": 1,
        "conservative_reserved_session_seconds": r1_reservation,
        "enable_gpu": False,
    }
    r1_job_path = report / "fim-conversion-job.json"
    campaign.save(r1_job_path, r1_job)
    r1_failure_path = report / "fim-conversion-failed-1.json"
    r1_failure_path.write_text("preserved original r1 failure receipt\n")
    r1_history = report / "fim/conversion/history/attempt-1"
    r1_history.mkdir(parents=True)
    (r1_history / "job.json").write_bytes(r1_job_path.read_bytes())
    (r1_history / "failure.json").write_bytes(r1_failure_path.read_bytes())
    (r1_history / "watch.json").write_text("preserved r1 watch receipt\n")
    r1_kernel = artifacts / "fim/conversion-kernel"
    r1_kernel.mkdir(parents=True)
    (r1_kernel / "run.py").write_text("preserved r1 launcher\n")
    preserved_r1_files = {
        path: path.read_bytes()
        for path in (
            r1_job_path,
            r1_failure_path,
            r1_history / "job.json",
            r1_history / "failure.json",
            r1_history / "watch.json",
            r1_kernel / "run.py",
        )
    }

    with campaign._conversion_revision_scope(2):
        plan = campaign.freeze_fim_conversion(selection_path)
        bundle = campaign.build_fim_conversion_bundle(plan)
        input_manifest_sha = campaign.digest(bundle / "input-manifest.json")
        campaign.save(
            campaign.CONVERSION_SUBMISSION_FILE,
            {
                "dataset": campaign.CONVERSION_DATASET,
                "state": "verified",
                "input_manifest_sha256": input_manifest_sha,
            },
        )
        budget = {
            "shared_limits": {
                "aggregate_reserved_session_seconds": r1_reservation * 2,
                "minimum_reserved_future_fim_session_seconds": 0,
                "conservative_account_gpu_hours": 40,
                "quota_renewal": "2026-10-10T00:00:00",
            }
        }
        campaign.save(report / "campaign_budget.json", budget)

        calls: list[tuple[str, ...]] = []
        commit = "f" * 40

        def fake_cli(*args: str, timeout: int = 120) -> str:
            del timeout
            call = tuple(args)
            calls.append(call)
            if call[-2:] == ("branch", "--show-current"):
                return campaign.CONVERSION_BRANCH
            if call[-2:] == ("status", "--porcelain"):
                return ""
            if call[-2:] == ("rev-parse", "HEAD"):
                return commit
            if call[-2:] == (
                "origin",
                f"refs/heads/{campaign.CONVERSION_BRANCH}",
            ):
                return f"{commit}\trefs/heads/{campaign.CONVERSION_BRANCH}"
            if call[:3] == ("kaggle", "kernels", "status"):
                return "COMPLETE"
            if call[:3] == ("kaggle", "kernels", "push"):
                pushed_kernel = Path(call[call.index("-p") + 1])
                assert pushed_kernel == campaign._conversion_kernel_root(1)
                pushed_metadata = json.loads((pushed_kernel / "kernel-metadata.json").read_text())
                assert pushed_metadata["id"] == campaign.CONVERSION_REFERENCE_R2
                return "https://www.kaggle.com/code/shlokbhakta/tc-q25-fim-q4-conversion-r2"
            return ""

        monkeypatch.setattr(campaign, "cli", fake_cli)
        monkeypatch.setattr(campaign, "quota", lambda: _observation())
        monkeypatch.setattr(campaign, "_csv_ref_exists", lambda _reference: False)

        job = campaign.submit_fim_conversion(plan)

        assert job["reference"] == campaign.CONVERSION_REFERENCE_R2
        assert job["plan_revision"] == 2
        assert job["attempt"] == 1
        assert job["retry_authorization_reference"] is None
        assert campaign.CONVERSION_JOB_FILE == report / "fim-conversion-job-r2.json"
        assert campaign.CONVERSION_JOB_FILE.is_file()
        assert (artifacts / "fim/conversion-kernel-r2/run.py").is_file()
        assert len([call for call in calls if call[:3] == ("kaggle", "kernels", "push")]) == 1
        assert all(path.read_bytes() == payload for path, payload in preserved_r1_files.items())


def test_shared_conversion_budget_counts_r1_and_revision2_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _selection, report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    for fim_job in report.glob("fim-job-*.json"):
        fim_job.unlink()
    reservation = campaign.CONVERSION_SESSION_SECONDS
    campaign.save(
        report / "fim-conversion-job.json",
        {
            "reference": campaign.CONVERSION_REFERENCE_R1,
            "attempt": 1,
            "conservative_reserved_session_seconds": reservation,
            "enable_gpu": False,
        },
    )
    archived = report / "fim/conversion/history-r2/attempt-1/job.json"
    campaign.save(
        archived,
        {
            "reference": campaign.CONVERSION_REFERENCE_R2,
            "plan_revision": 2,
            "attempt": 1,
            "conservative_reserved_session_seconds": reservation,
            "enable_gpu": False,
        },
    )
    limits = {
        "shared_limits": {
            "aggregate_reserved_session_seconds": reservation * 3,
            "minimum_reserved_future_fim_session_seconds": 0,
            "conservative_account_gpu_hours": 40,
        }
    }
    campaign.save(report / "campaign_budget.json", limits)

    campaign.check_shared_allocation_budget(reservation, phase="conversion")

    limits["shared_limits"]["aggregate_reserved_session_seconds"] = reservation * 3 - 1
    campaign.save(report / "campaign_budget.json", limits)
    with pytest.raises(RuntimeError, match="session reservation exhausted"):
        campaign.check_shared_allocation_budget(reservation, phase="conversion")


def test_conversion_revision2_cli_flag_is_limited_to_conversion_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_q25_code_cpt.py", "--freeze-conversion", "--conversion-revision2"],
    )
    seen: dict[str, Any] = {}

    def fake_freeze(selection_path: Path) -> dict[str, Any]:
        seen.update(
            selection_path=selection_path,
            revision=campaign.CONVERSION_PLAN_REVISION,
            plan_path=campaign.CONVERSION_PLAN,
        )
        return {"plan_revision": campaign.CONVERSION_PLAN_REVISION}

    monkeypatch.setattr(campaign, "freeze_fim_conversion", fake_freeze)
    campaign.main()
    assert seen == {
        "selection_path": campaign.CONVERSION_SELECTION,
        "revision": 2,
        "plan_path": campaign.CONVERSION_R2_PLAN,
    }
    assert json.loads(capsys.readouterr().out) == {"fim_conversion_plan": {"plan_revision": 2}}

    monkeypatch.setattr(sys, "argv", ["run_q25_code_cpt.py", "--conversion-revision2"])
    with pytest.raises(SystemExit):
        campaign.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_q25_code_cpt.py",
            "--execute-conversion",
            "--conversion-revision2",
            "--attempt",
            "2",
        ],
    )
    with pytest.raises(SystemExit):
        campaign.main()


@pytest.mark.parametrize("tamper", [None, "worker_hash", "storage_cap"])
def test_conversion_collect_retrieves_only_result_manifest_and_selected_q4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str | None
) -> None:
    selection_path, _report, _artifacts, _training_plan, _code_sources = _fim_conversion_context(
        tmp_path, monkeypatch
    )
    plan = campaign.freeze_fim_conversion(selection_path)
    campaign.save(
        campaign.CONVERSION_JOB_FILE,
        {
            "reference": campaign.CONVERSION_KERNEL_REFERENCE,
            "attempt": 1,
            "source_kernel_reference": plan["source"]["kernel_reference"],
            "plan_sha256": campaign.digest(campaign.CONVERSION_PLAN),
            "selection_sha256": plan["selection_sha256"],
            "enable_gpu": False,
            "automatic_allocation": False,
        },
    )
    q4_bytes = b"synthetic q4 artifact"
    q4_sha = hashlib.sha256(q4_bytes).hexdigest()
    q4_name = f"q25-{plan['source']['arm']}-Q4_K_M.gguf"
    result = {
        "schema": "q25-fim-q4-conversion-run-v1",
        "status": "complete",
        "selection_sha256": plan["selection_sha256"],
        "training_plan_sha256": plan["training_plan_sha256"],
        "selected_arm": plan["source"]["arm"],
        "source_export_manifest_sha256": plan["source"]["artifact_manifest_sha256"],
        "source_fingerprint": plan["source"]["fingerprint"],
        "source_kernel_reference": plan["source"]["kernel_reference"],
        "training_cursor": plan["source"]["training_cursor"],
        "runtime": {
            "worker_source_sha256": plan["source_code"]["conversion_worker.py"],
            "contract_source_sha256": plan["source_code"][
                "src/tinycomplete/code_cpt/q25_fim_conversion.py"
            ],
            "requirements_lock_sha256": plan["source_code"][
                "kaggle/q25_fim_conversion_r1/requirements-conversion.lock"
            ],
        },
        "storage": {
            "artifact_cap_bytes": campaign.CONVERSION_ARTIFACT_CAP_BYTES,
            "minimum_free_bytes": campaign.CONVERSION_MINIMUM_FREE_BYTES,
            "final": {
                "current_accounted_bytes": 100,
                "max_artifact_bytes": campaign.CONVERSION_ARTIFACT_CAP_BYTES,
                "minimum_free_bytes": campaign.CONVERSION_MINIMUM_FREE_BYTES,
            },
        },
        "conversion": {"format": "Q4_K_M", "gpu_enabled": False},
        "q4_export": {"file": q4_name, "bytes": len(q4_bytes), "sha256": q4_sha},
    }
    if tamper == "worker_hash":
        result["runtime"]["worker_source_sha256"] = "f" * 64
    elif tamper == "storage_cap":
        result["storage"]["artifact_cap_bytes"] += 1
    calls: list[tuple[str, ...]] = []

    def fake_cli(*args: str, timeout: int = 120) -> str:
        call = tuple(args)
        calls.append(call)
        if call[:3] == ("kaggle", "kernels", "status"):
            return "COMPLETE"
        destination = Path(call[call.index("-p") + 1])
        pattern = call[call.index("--file-pattern") + 1]
        if pattern == r"q25_fim_conversion_r1/conversion\.json$":
            manifest = destination / "q25_fim_conversion_r1/conversion.json"
            campaign.save(manifest, result)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / q4_name).write_bytes(q4_bytes)
        return ""

    monkeypatch.setattr(campaign, "cli", fake_cli)
    monkeypatch.setattr(
        campaign.shutil, "disk_usage", lambda _path: SimpleNamespace(free=64 * 1024**3)
    )

    if tamper is None:
        receipt = campaign.collect_fim_conversion(plan)
        output_patterns = [
            call[call.index("--file-pattern") + 1] for call in calls if "--file-pattern" in call
        ]
        assert output_patterns == [
            r"q25_fim_conversion_r1/conversion\.json$",
            campaign.re.escape(q4_name) + "$",
        ]
        assert receipt["q4_file"] == q4_name
        assert receipt["source_weight_or_checkpoint_retrieval"] is False
    else:
        with pytest.raises(ValueError, match="another input"):
            campaign.collect_fim_conversion(plan)
        output_patterns = [
            call[call.index("--file-pattern") + 1] for call in calls if "--file-pattern" in call
        ]
        assert output_patterns == [r"q25_fim_conversion_r1/conversion\.json$"]
