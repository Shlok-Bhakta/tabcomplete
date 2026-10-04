from __future__ import annotations

import json
import sys
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
    plan["configuration"]["budget"]["maximum_additional_training_input_tokens"] = (
        maximum_tokens
    )
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
    assert record["processed_campaign_input_tokens_conservative"] == (
        1_048_576 + already_external
    )


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
