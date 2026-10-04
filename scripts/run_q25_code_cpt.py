"""Freeze and submit the bounded Qwen2.5 code-pretraining ablation.

Reuses the existing quota reader and private Kaggle dataset/kernel interface.
Preparation is CPU-only; allocation checks current quota immediately before push.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/q25_code_cpt_r2"
ARTIFACTS = Path("/mnt/ssd/tabcomplete-q25-code-cpt-r2")
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    / "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
DATASET = "shlokbhakta/tabcomplete-q25-code-cpt-r2-inputs"
BASE_DATASET = "shlokbhakta/tabcomplete-one-line-instinct-pilot-r1-inputs"
FIM_DATASET = "shlokbhakta/tabcomplete-q25-fim-r2-inputs-r4"
CPT_INITIALIZER_DATASET = "shlokbhakta/tabcomplete-q25-cpt-r2-fim-initializer"
CONVERSION_DATASET = "shlokbhakta/tabcomplete-q25-fim-conversion-r1-inputs"
CONVERSION_DATASET_SLUG = "tabcomplete-q25-fim-conversion-r1-inputs"
CONVERSION_KERNEL_REFERENCE = "shlokbhakta/tc-q25-fim-q4-conversion-r1"
CONVERSION_BRANCH = "research/q25-fim-conversion-orchestration-r1"
CONVERSION_SESSION_SECONDS = 10_800
CONVERSION_FINALIZATION_RESERVE_SECONDS = 1_800
CONVERSION_ARTIFACT_CAP_BYTES = 12 * 1024**3
CONVERSION_MINIMUM_FREE_BYTES = 2 * 1024**3
CONVERSION_EXPORT_DIRECTORY = "q25_fim_r2/training/inference-f16"
CONVERSION_PLAN = REPORT / "fim_conversion_plan.json"
CONVERSION_SELECTION = REPORT / "fim_conversion" / "selection.json"
CONVERSION_QUALITY_REPORT = REPORT / "fim_quality_comparison.json"
CONVERSION_INPUT_BUNDLE = ARTIFACTS / "fim/conversion-input-bundle"
CONVERSION_KERNEL_ROOT = ARTIFACTS / "fim/conversion-kernel"
CONVERSION_JOB_FILE = REPORT / "fim-conversion-job.json"
CONVERSION_SUBMISSION_FILE = REPORT / "fim-conversion-dataset-submission.json"
CONVERSION_OUTPUT = ARTIFACTS / "fim/conversion/output"
FIM_ARMS = ("untouched_q25_to_fim", "completed_cpt_q25_to_fim")
FIM_LINE_SOURCE = Path(
    "/mnt/ssd/tabcomplete-preserved-research/model_data_r2/frozen-corpora/causal_line_v1-r3.jsonl"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cli(*args: str, timeout: int = 120) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}")
    return result.stdout.strip()


def quota() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import live_quota

    return live_quota()


def check_quota(plan: dict[str, Any], observation: dict[str, Any]) -> None:
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _job_statuses_verified

    if observation["active_jobs"] or not _job_statuses_verified(observation):
        raise RuntimeError("another notebook is active or its state is unresolved")
    limits = plan["configuration"]["budget"]
    if limits["paid_compute"] is not False or limits["automatic_renewal_use"] is not False:
        raise ValueError("paid compute and renewed quota consumption are forbidden")
    if observation.get("units") != "Kaggle account GPU-hours" or not observation.get("source"):
        raise ValueError("quota observation lacks units or provenance")
    if len(observation.get("job_statuses", [])) >= 100:
        raise RuntimeError("job listing may be truncated; verify pagination before allocation")
    if observation["renewal"] != limits["quota_renewal"]:
        raise RuntimeError("quota allocation renewed; automatic consumption is forbidden")
    renewal = datetime.fromisoformat(str(observation["renewal"]).replace("Z", "+00:00"))
    if renewal.tzinfo is None:
        renewal = renewal.replace(tzinfo=UTC)
    if datetime.now(UTC) + timedelta(seconds=limits["session_seconds"]) >= renewal:
        raise RuntimeError("session deadline would cross the authorized allocation renewal")
    required = limits["session_seconds"] / 3600 * limits["conservative_quota_multiplier"]
    if observation["remaining"] is None or observation["remaining"] < required:
        raise RuntimeError("insufficient authenticated remaining GPU quota")


def freeze(config: Path) -> dict[str, Any]:
    configuration = yaml.safe_load(config.read_text())
    if configuration["model"]["id"] != "Qwen/Qwen2.5-Coder-0.5B" or (
        configuration["model"]["initializer"] != "untouched_pretrained"
    ):
        raise ValueError("only the untouched approved q25 base is permitted")
    files = {
        name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
        if (path := MODEL / name).exists()
    }
    if files["model.safetensors"]["sha256"] != configuration["model"]["weight_sha256"]:
        raise ValueError("pretrained weight identity differs")
    if files["tokenizer.json"]["sha256"] != configuration["model"]["tokenizer_sha256"]:
        raise ValueError("pretrained tokenizer identity differs")
    pool = Path(configuration["data"]["reuse_pool"])
    sources = {
        language: {
            "bytes": (pool / (language + ".jsonl")).stat().st_size,
            "sha256": digest(pool / (language + ".jsonl")),
            "sidecar_sha256": digest(pool / (language + ".json")),
        }
        for language in configuration["data"]["languages"]
    }
    fixtures = {
        name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for name, path in {
            "causal": ROOT / configuration["evaluation"]["causal_suite"],
            "line": Path(configuration["evaluation"]["line_suite"]),
            "synthetic_edit_probe": ROOT / configuration["evaluation"]["synthetic_edit_probe"],
        }.items()
    }
    for name in ("causal", "line"):
        if fixtures[name]["sha256"] != configuration["evaluation"][name + "_suite_sha256"]:
            raise ValueError("frozen evaluation fixture differs")
    identity = {
        "configuration": configuration,
        "config_sha256": digest(config),
        "existing_model_files": files,
        "source_pool": sources,
        "fixtures": fixtures,
        "controller": {
            "host": platform.node(),
            "python": platform.python_version(),
            "local_inference": "CPU only",
        },
    }
    target = REPORT / "plan.json"
    if target.exists():
        frozen = json.loads(target.read_text())
        if any(frozen[key] != value for key, value in identity.items()):
            raise ValueError("frozen identity changed; create an explicit new plan revision")
        return frozen
    observation = json.loads((ARTIFACTS / "live-quota-jobs.json").read_text())
    check_quota(identity, observation)
    plan = {**identity, "frozen_at": datetime.now(UTC).isoformat(), "quota_at_freeze": observation}
    save(target, plan)
    save(
        REPORT / "download_manifest.json",
        {
            "new_model_weights_downloaded": False,
            "reused_model": files,
            "reused_source_pool": sources,
            "network_source_download_bytes": 0,
            "private_credential_retrieval_bytes": 37,
            "credential_staged_with_training_inputs": False,
            "remote_checked": {
                "rust_first_shard_bytes": 163808740,
                "authenticated_access": "pass",
                "downloaded": False,
            },
        },
    )
    return plan


def directory_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def build_bundle(plan: dict[str, Any]) -> Path:
    output = ARTIFACTS / "input-bundle"
    output.mkdir(exist_ok=True)
    config = plan["configuration"]
    sources = {
        "plan.json": REPORT / "plan.json",
        "train_blocks.npy": ARTIFACTS / "corpus/train_blocks.npy",
        "development_blocks.npy": ARTIFACTS / "corpus/development_blocks.npy",
        "corpus_metadata.json": ARTIFACTS / "corpus/corpus_metadata.json",
        "train_manifest.jsonl": ARTIFACTS / "corpus/train_manifest.jsonl",
        "development_manifest.jsonl": ARTIFACTS / "corpus/development_manifest.jsonl",
        "causal_line_v1-r3.jsonl": Path(config["evaluation"]["line_suite"]),
    }
    allowed = set(sources) | {"input-manifest.json", "dataset-metadata.json"}
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in output.iterdir()):
        raise ValueError("staging directory contains an unapproved artifact")
    projected = sum(
        path.stat().st_size for name, path in sources.items() if not (output / name).exists()
    )
    if directory_bytes(ARTIFACTS) + projected > config["budget"]["new_artifact_bytes_cap"]:
        raise OSError("staging would exceed the new artifact cap")
    if shutil.disk_usage(ARTIFACTS).free < projected + config["budget"]["minimum_free_bytes"]:
        raise OSError("staging would exhaust storage headroom")
    import numpy as np

    train = np.load(sources["train_blocks.npy"], mmap_mode="r", allow_pickle=False)
    development = np.load(sources["development_blocks.npy"], mmap_mode="r", allow_pickle=False)
    for array in (train, development):
        if array.ndim != 2 or array.shape[1] != config["training"]["sequence_length"]:
            raise ValueError("prepared block shape differs from the frozen plan")
        if len(array) == 0 or array.min() < 0 or array.max() >= 151936:
            raise ValueError("prepared token IDs are invalid for the selected q25 vocabulary")
    if train.size > config["training"]["max_input_tokens"]:
        raise ValueError("prepared training input exceeds the frozen token cap")
    record = {}
    for name, path in sources.items():
        target = output / name
        if target.exists() and digest(target) != digest(path):
            raise ValueError("immutable staged input differs")
        if not target.exists():
            shutil.copy2(path, target)
        record[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    save(output / "input-manifest.json", {"files": record, "model_dataset": BASE_DATASET})
    save(
        output / "dataset-metadata.json",
        {
            "id": DATASET,
            "title": "TabComplete Q25 bounded code CPT r2 inputs",
            "licenses": [{"name": "other"}],
        },
    )
    if directory_bytes(ARTIFACTS) > config["budget"]["new_artifact_bytes_cap"]:
        raise OSError("new artifact cap exceeded")
    if shutil.disk_usage(ARTIFACTS).free < config["budget"]["minimum_free_bytes"]:
        raise OSError("storage headroom exhausted")
    return output


def upload_bundle(plan: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _csv_refs, wait_for_remote_inputs

    output = build_bundle(plan)
    marker = ARTIFACTS / "dataset-submission.json"
    expected = json.loads((output / "input-manifest.json").read_text())["files"]
    expected["input-manifest.json"] = {"bytes": (output / "input-manifest.json").stat().st_size}
    if not marker.exists():
        refs = _csv_refs(["kaggle", "datasets", "list", "--mine", "--page-size", "100", "--csv"])
        if DATASET in refs:
            raise ValueError("dataset ID already exists without this campaign's submission receipt")
        response = cli("kaggle", "datasets", "create", "-p", str(output), "-t", timeout=900)
        if "Your private Dataset is being created." not in response:
            raise RuntimeError("Kaggle did not confirm private dataset creation")
        save(
            marker,
            {
                "dataset": DATASET,
                "input_manifest_sha256": digest(output / "input-manifest.json"),
                "state": "created",
            },
        )
    recorded = json.loads(marker.read_text())
    if recorded["input_manifest_sha256"] != digest(output / "input-manifest.json"):
        raise ValueError("uploaded dataset fingerprint differs")
    verification = wait_for_remote_inputs(DATASET, expected)
    save(marker, {**recorded, "state": "verified", "remote_verification": verification})
    return verification


SESSION_SETTLEMENTS = "session_settlements.json"
SESSION_SETTLEMENT_SCHEMA = "q25-session-settlements-v1"
SESSION_SETTLEMENT_MARGIN_SECONDS = 60


def _parse_aware_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"session settlement {field} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"session settlement {field} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"session settlement {field} timestamp has no timezone")
    return parsed.astimezone(UTC)


def _is_terminal_kernel_status(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.search(r"KernelWorkerStatus\.(?:ERROR|COMPLETE)\b", value) is not None
    )


def _is_exact_zero_counter(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _fim_plan_for_hash(plan_sha256: str) -> tuple[Path, dict[str, Any]]:
    candidates = [
        path
        for path in REPORT.glob("fim_training_plan*.json")
        if path.is_file() and not path.is_symlink() and digest(path) == plan_sha256
    ]
    if len(candidates) != 1:
        raise ValueError("session settlement cannot identify one archived FIM plan")
    plan = json.loads(candidates[0].read_text())
    if plan.get("schema") != "q25-fim-training-plan-v1" or (
        plan.get("gpu_execution_authorized") is not True
    ):
        raise ValueError("session settlement plan is not an authorized frozen FIM plan")
    return candidates[0], plan


def _fim_input_manifest_for_hash(input_sha256: str) -> Path:
    candidates = [
        path
        for path in [
            ARTIFACTS / "fim/input-bundle/input-manifest.json",
            *ARTIFACTS.glob("fim/history/*/input-bundle/input-manifest.json"),
        ]
        if path.is_file() and not path.is_symlink() and digest(path) == input_sha256
    ]
    if len(candidates) != 1:
        raise ValueError("session settlement cannot identify one archived FIM input manifest")
    return candidates[0]


def _validate_fim_input_manifest(
    path: Path, *, input_sha256: str, plan_sha256: str, plan: dict[str, Any]
) -> None:
    if digest(path) != input_sha256:
        raise ValueError("session settlement input-manifest fingerprint differs")
    manifest = json.loads(path.read_text())
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if manifest.get("model_dataset") != BASE_DATASET or not isinstance(files, dict):
        raise ValueError("session settlement input manifest identity is invalid")
    training = plan.get("data", {})
    fixtures = plan.get("evaluation", {}).get("fixtures", {})
    expected = {
        "plan.json": {"sha256": plan_sha256},
        "train.jsonl": training.get("train", {}),
        "development.jsonl": training.get("development", {}),
        "corpus_metadata.json": {"sha256": training.get("corpus_metadata_sha256")},
        "causal200.jsonl": fixtures.get("causal", {}),
        "line180.jsonl": fixtures.get("line", {}),
    }
    if set(files) != set(expected):
        raise ValueError("session settlement input manifest file list differs from the frozen plan")
    for name, identity in expected.items():
        record = files.get(name)
        if not isinstance(record, dict):
            raise ValueError("session settlement input manifest file record is invalid")
        expected_sha = identity.get("sha256")
        expected_bytes = identity.get("bytes")
        if name in {"train.jsonl", "development.jsonl"} and identity.get("file") != name:
            raise ValueError("session settlement training filename differs from the frozen plan")
        if (
            not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or record.get("sha256") != expected_sha
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record["bytes"] < 0
            or (expected_bytes is not None and record["bytes"] != expected_bytes)
        ):
            raise ValueError("session settlement input identity differs from the frozen plan")
        staged = path.parent / name
        if (
            not staged.is_file()
            or staged.is_symlink()
            or staged.stat().st_size != record["bytes"]
            or digest(staged) != expected_sha
        ):
            raise ValueError("session settlement staged input file differs from its manifest")
    if digest(path.parent / "plan.json") != plan_sha256:
        raise ValueError("session settlement bundled plan differs from the job plan")


def _read_zero_work_evidence(arm: str, attempt: int) -> dict[str, Any]:
    if (
        arm not in FIM_ARMS
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        raise ValueError("session settlement has an invalid arm or attempt")
    job_path = REPORT / f"fim-job-{arm}-{attempt}.json"
    watch_path = REPORT / f"fim-watch-{arm}-{attempt}.json"
    verified_path = REPORT / f"fim-verified-{arm}-{attempt}.json"
    if not all(
        path.is_file() and not path.is_symlink() for path in (job_path, watch_path, verified_path)
    ):
        raise ValueError("session settlement evidence is incomplete")
    job = json.loads(job_path.read_text())
    watch = json.loads(watch_path.read_text())
    verified = json.loads(verified_path.read_text())
    reference = job.get("reference")
    plan_sha256 = job.get("plan_sha256")
    input_sha256 = job.get("input_manifest_sha256")
    commit = job.get("commit")
    if (
        job.get("status") != "submitted"
        or job.get("arm") != arm
        or job.get("attempt") != attempt
        or not isinstance(reference, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", reference)
        or not isinstance(plan_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", plan_sha256)
        or not isinstance(input_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", input_sha256)
        or not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or any(
            record.get(key) != value
            for record in (watch, verified)
            for key, value in (
                ("reference", reference),
                ("status", watch.get("status")),
            )
        )
        or any(
            verified.get(key) != value
            for key, value in (
                ("arm", arm),
                ("attempt", attempt),
                ("plan_sha256", plan_sha256),
                ("input_manifest_sha256", input_sha256),
                ("commit", commit),
            )
        )
        or not _is_terminal_kernel_status(watch.get("status"))
        or watch.get("automatic_allocation") is not False
    ):
        raise ValueError("session settlement job, watch, and receipt identities differ")
    submitted = _parse_aware_timestamp(job.get("submitted_at"), field="submitted_at")
    observed = _parse_aware_timestamp(watch.get("observed_at"), field="watch observed_at")
    verified_at = _parse_aware_timestamp(verified.get("observed_at"), field="receipt observed_at")
    if observed < submitted or verified_at < observed:
        raise ValueError("session settlement timestamps are out of order")

    plan_path, plan = _fim_plan_for_hash(plan_sha256)
    session_seconds = job.get("session_seconds")
    reserved_seconds = job.get("conservative_reserved_session_seconds")
    if (
        not isinstance(session_seconds, int)
        or isinstance(session_seconds, bool)
        or session_seconds <= 0
        or reserved_seconds != session_seconds
        or session_seconds != plan.get("configuration", {}).get("budget", {}).get("session_seconds")
    ):
        raise ValueError("session settlement job reservation is invalid")

    initializer = plan.get("initializers", {}).get(arm)
    if not isinstance(initializer, dict):
        raise ValueError("session settlement plan lacks the arm initializer")
    input_path = _fim_input_manifest_for_hash(input_sha256)
    _validate_fim_input_manifest(
        input_path,
        input_sha256=input_sha256,
        plan_sha256=plan_sha256,
        plan=plan,
    )
    output = ARTIFACTS / f"fim/output-{arm}-{attempt}"
    worker_paths = list(output.glob("**/worker-status.json"))
    if len(worker_paths) != 1 or worker_paths[0].is_symlink():
        raise ValueError("session settlement requires one preserved worker status")
    worker_path = worker_paths[0]
    worker = json.loads(worker_path.read_text())
    expected_worker_identity = {
        "schema": "q25-fim-kaggle-worker-status-v1",
        "commit": commit,
        "attempt": attempt,
        "arm": arm,
        "plan_sha256": plan_sha256,
        "input_manifest_sha256": input_sha256,
    }
    if not isinstance(worker, dict):
        raise ValueError("session settlement worker status is malformed")
    stages = worker.get("stages")
    if (
        any(worker.get(key) != value for key, value in expected_worker_identity.items())
        or worker.get("training_started") is not False
        or worker.get("state")
        not in {
            "setup",
            "verified_inputs",
            "baseline_evaluation",
            "training_deferred_insufficient_time",
            "failed",
        }
        or not isinstance(stages, list)
        or any(not isinstance(stage, dict) or stage.get("name") == "training" for stage in stages)
        or any(
            key in worker
            for key in ("training", "training_exit_code", "checkpoint_pointer_present")
        )
    ):
        raise ValueError("session settlement worker status does not prove zero training work")
    if output.is_symlink() or any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("session settlement output contains symbolic links")
    training_dirs = list(output.glob("**/training"))
    training_artifact_names = {"latest.json", "updates.jsonl", "run_result.json", "training.log"}
    if any(
        path.is_symlink() or any(child.is_file() or child.is_symlink() for child in path.rglob("*"))
        for path in training_dirs
    ) or any(
        path.is_file()
        and (
            path.name in training_artifact_names
            or (path.name.startswith("resume-step") and path.suffix == ".pt")
            or path.suffix == ".safetensors"
        )
        for path in output.rglob("*")
    ):
        raise ValueError("session settlement output contains training artifacts")

    worker_elapsed = worker.get("elapsed_seconds")
    if (
        not isinstance(worker_elapsed, (int, float))
        or isinstance(worker_elapsed, bool)
        or not math.isfinite(worker_elapsed)
        or worker_elapsed < 0
    ):
        raise ValueError("session settlement worker elapsed time is invalid")
    expected_tokens = job.get("other_phase_tokens")
    discarded = job.get("own_discarded_tokens")
    if (
        not isinstance(expected_tokens, int)
        or isinstance(expected_tokens, bool)
        or expected_tokens < 0
        or not _is_exact_zero_counter(discarded)
        or not isinstance(job.get("external_campaign_tokens"), int)
        or isinstance(job.get("external_campaign_tokens"), bool)
        or job.get("resume_source") is not None
        or job.get("resume_identity") is not None
        or job.get("external_campaign_tokens") != expected_tokens
        or verified.get("no_training_executed") is not True
        or verified.get("training_status") != "no_training_executed"
        or verified.get("checkpoint_verified") is not False
        or verified.get("carried_checkpoint_verified") is not False
        or verified.get("carried_checkpoint_source") is not None
        or verified.get("carried_checkpoint_identity") is not None
        or not _is_exact_zero_counter(verified.get("own_discarded_tokens_for_resume"))
        or not _is_exact_zero_counter(verified.get("processed_arm_input_tokens_conservative"))
        or not isinstance(verified.get("other_phase_processed_tokens_at_submission"), int)
        or isinstance(verified.get("other_phase_processed_tokens_at_submission"), bool)
        or verified.get("other_phase_processed_tokens_at_submission") != expected_tokens
    ):
        raise ValueError("session settlement receipt carries training state or token exposure")

    lineage_reference = job.get("authorization_lineage_reference")
    if attempt == 1:
        if lineage_reference is not None:
            raise ValueError("first FIM attempt has unexpected retry lineage")
    else:
        previous_evidence = _read_zero_work_evidence(arm, attempt - 1)
        if lineage_reference != previous_evidence["reference"]:
            raise ValueError(
                "session settlement retry lineage is not the immediate zero-work attempt"
            )

    elapsed_plus_margin = (observed - submitted).total_seconds() + SESSION_SETTLEMENT_MARGIN_SECONDS
    settled_seconds = math.ceil(elapsed_plus_margin)
    if settled_seconds > session_seconds or math.ceil(worker_elapsed) > settled_seconds:
        raise ValueError("session settlement upper bound does not cover the allocation")
    return {
        "reference": reference,
        "arm": arm,
        "attempt": attempt,
        "plan_sha256": plan_sha256,
        "input_manifest_sha256": input_sha256,
        "commit": commit,
        "initializer_identity_sha256": hashlib.sha256(
            json.dumps(initializer, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "authorization_lineage_reference": lineage_reference,
        "job_file": job_path.name,
        "job_sha256": digest(job_path),
        "watch_file": watch_path.name,
        "watch_sha256": digest(watch_path),
        "verified_file": verified_path.name,
        "verified_sha256": digest(verified_path),
        "worker_status_file": str(worker_path.relative_to(ARTIFACTS)),
        "worker_status_sha256": digest(worker_path),
        "plan_file": plan_path.name,
        "input_manifest_file": str(input_path.relative_to(ARTIFACTS)),
        "submitted_at": job["submitted_at"],
        "terminal_observed_at": watch["observed_at"],
        "receipt_observed_at": verified["observed_at"],
        "worker_elapsed_seconds": worker_elapsed,
        "original_reserved_session_seconds": session_seconds,
        "terminal_observation_margin_seconds": SESSION_SETTLEMENT_MARGIN_SECONDS,
        "settled_session_seconds": settled_seconds,
    }


def _load_validated_session_settlements() -> dict[str, dict[str, Any]]:
    path = REPORT / SESSION_SETTLEMENTS
    if not path.exists():
        return {}
    ledger = json.loads(path.read_text())
    if ledger.get("schema") != SESSION_SETTLEMENT_SCHEMA or not isinstance(
        ledger.get("settlements"), list
    ):
        raise ValueError("session settlement ledger schema is invalid")
    entries: dict[str, dict[str, Any]] = {}
    known_jobs: dict[str, Path] = {}
    for job_path in REPORT.glob("fim-job-*.json"):
        if not job_path.is_file() or job_path.is_symlink():
            raise ValueError("session settlement found an unsafe FIM job receipt")
        reference = json.loads(job_path.read_text()).get("reference")
        if not isinstance(reference, str) or reference in known_jobs:
            raise ValueError("session settlement jobs have missing or duplicate references")
        known_jobs[reference] = job_path
    for entry in ledger["settlements"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("reference"), str):
            raise ValueError("session settlement ledger entry is invalid")
        reference = entry["reference"]
        if reference in entries or reference not in known_jobs:
            raise ValueError("session settlement ledger has a duplicate or unknown job")
        if entry.get("recorded_at") is None:
            raise ValueError("session settlement ledger lacks its recording time")
        recorded_at = _parse_aware_timestamp(entry["recorded_at"], field="recorded_at")
        job_path = known_jobs[reference]
        job = json.loads(job_path.read_text())
        arm, attempt = job.get("arm"), job.get("attempt")
        if not isinstance(arm, str) or not isinstance(attempt, int) or isinstance(attempt, bool):
            raise ValueError("session settlement job identity is invalid")
        expected = _read_zero_work_evidence(arm, attempt)
        if recorded_at < _parse_aware_timestamp(expected["receipt_observed_at"], field="receipt"):
            raise ValueError("session settlement was recorded before the receipt")
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError("session settlement evidence hash or bound differs")
        entries[reference] = entry
    return entries


def settle_fim_zero_work_sessions() -> dict[str, Any]:
    """Record conservative charges for terminal, independently verified zero-work FIM jobs."""
    ledger_path = REPORT / SESSION_SETTLEMENTS
    existing = _load_validated_session_settlements()
    entries = {reference: dict(value) for reference, value in existing.items()}
    skipped = 0
    for job_path in sorted(REPORT.glob("fim-job-*.json")):
        if job_path.is_symlink() or not job_path.is_file():
            raise ValueError("session settlement found an unsafe job receipt")
        job = json.loads(job_path.read_text())
        reference = job.get("reference")
        if reference in entries:
            continue
        arm, attempt = job.get("arm"), job.get("attempt")
        if arm not in FIM_ARMS or not isinstance(attempt, int) or isinstance(attempt, bool):
            skipped += 1
            continue
        watch_path = REPORT / f"fim-watch-{arm}-{attempt}.json"
        verified_path = REPORT / f"fim-verified-{arm}-{attempt}.json"
        if not watch_path.is_file() or not verified_path.is_file():
            skipped += 1
            continue
        watch = json.loads(watch_path.read_text())
        if not _is_terminal_kernel_status(watch.get("status")):
            skipped += 1
            continue
        receipt = json.loads(verified_path.read_text())
        if (
            receipt.get("no_training_executed") is not True
            or receipt.get("training_status") != "no_training_executed"
        ):
            skipped += 1
            continue
        entry = _read_zero_work_evidence(arm, attempt)
        entry["recorded_at"] = datetime.now(UTC).isoformat()
        entries[reference] = entry
    ordered = sorted(
        entries.values(), key=lambda entry: (entry["submitted_at"], entry["reference"])
    )
    save(
        ledger_path,
        {"schema": SESSION_SETTLEMENT_SCHEMA, "settlements": ordered},
    )
    return {
        "settled_count": len(entries) - len(existing),
        "already_settled_count": len(existing),
        "skipped_unverified_or_nonterminal_count": skipped,
        "settled_session_seconds": sum(entry["settled_session_seconds"] for entry in ordered),
        "ledger": str(ledger_path),
    }


def check_shared_allocation_budget(session_seconds: int, *, phase: str) -> None:
    path = REPORT / "campaign_budget.json"
    if not path.exists():
        return
    shared = json.loads(path.read_text())["shared_limits"]
    fim_settlements = _load_validated_session_settlements()
    gpu_jobs = [*REPORT.glob("job-*.json"), *REPORT.glob("fim-job-*.json")]
    wall_reserved = 0
    gpu_reserved = 0
    for job_path in gpu_jobs:
        job = json.loads(job_path.read_text())
        job_reservation = int(job["conservative_reserved_session_seconds"])
        if job_path.name.startswith("fim-job-"):
            settlement = fim_settlements.get(job.get("reference"))
            if settlement is not None:
                if settlement["job_file"] != job_path.name:
                    raise ValueError("session settlement points to a different FIM job")
                job_reservation = int(settlement["settled_session_seconds"])
        wall_reserved += job_reservation
        gpu_reserved += job_reservation
    # A CPU conversion kernel consumes the aggregate Kaggle session window but
    # does not consume account GPU-hours. Keep these units separate.
    for job_path in REPORT.glob("fim-conversion-job*.json"):
        job = json.loads(job_path.read_text())
        reservation = job.get("conservative_reserved_session_seconds")
        if not isinstance(reservation, int) or isinstance(reservation, bool) or reservation <= 0:
            raise ValueError("CPU conversion job has an invalid session reservation")
        if job.get("enable_gpu") is not False:
            raise ValueError("conversion reservation unexpectedly enables a GPU")
        wall_reserved += reservation
    future_reserve = (
        int(shared["minimum_reserved_future_fim_session_seconds"]) if phase == "cpt" else 0
    )
    if (
        wall_reserved + session_seconds + future_reserve
        > shared["aggregate_reserved_session_seconds"]
    ):
        raise RuntimeError("shared CPT/FIM session reservation exhausted")
    new_gpu_seconds = session_seconds if phase in {"cpt", "fim"} else 0
    if (gpu_reserved + new_gpu_seconds) / 3600 * 2 > shared["conservative_account_gpu_hours"]:
        raise RuntimeError("shared CPT/FIM account GPU-hour reservation exhausted")
    if phase == "cpt" and list(REPORT.glob("fim-job-*.json")):
        raise RuntimeError("raw CPT cannot restart after matched FIM adaptation begins")


def collect_cpt_export(plan: dict[str, Any], attempt: int) -> Path:
    """Retrieve the completed CPT initializer without downloading old checkpoints."""
    verified = json.loads((REPORT / f"verified-output-{attempt}.json").read_text())
    if not verified.get("checkpoint_verified") or verified.get("training_status") != "complete":
        raise ValueError("FIM requires a verified complete CPT pass")
    output = ARTIFACTS / f"output-attempt-{attempt}"
    candidates = list(output.glob("**/inference-f16/artifact_manifest.json"))
    if len(candidates) != 1:
        raise ValueError("exactly one completed CPT export manifest is required")
    manifest = json.loads(candidates[0].read_text())
    if (
        manifest.get("schema") != "q25-cpt-inference-f16-v1"
        or manifest.get("fingerprint") != verified["fingerprint"]
        or manifest.get("training_cursor") != verified["cursor"]
        or verified["cursor"]["training_input_tokens"] != 7872512
        or verified["cursor"]["completed_updates"] != 481
    ):
        raise ValueError("CPT export does not represent the frozen completed pass")
    files = manifest["files"]
    export = candidates[0].parent
    missing_bytes = 0
    for name, record in files.items():
        if Path(name).name != name or not isinstance(record.get("bytes"), int):
            raise ValueError("CPT export manifest contains an unsafe file record")
        path = export / name
        if not path.exists():
            missing_bytes += record["bytes"]
    budget = plan["configuration"]["budget"]
    if directory_bytes(ARTIFACTS) + missing_bytes > budget["new_artifact_bytes_cap"] or (
        shutil.disk_usage(ARTIFACTS).free < missing_bytes + budget["minimum_free_bytes"]
    ):
        raise OSError("CPT export retrieval exceeds storage headroom")
    for name, record in files.items():
        path = export / name
        if not path.exists():
            cli(
                "kaggle",
                "kernels",
                "output",
                verified["reference"],
                "-p",
                str(output),
                "-q",
                "--file-pattern",
                r"inference-f16/" + re.escape(name) + "$",
                timeout=900,
            )
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("retrieved CPT export file identity differs")
    save(
        REPORT / "cpt-export-verification.json",
        {
            "reference": verified["reference"],
            "path": str(export),
            "manifest_sha256": digest(candidates[0]),
            "fingerprint": verified["fingerprint"],
            "training_cursor": verified["cursor"],
            "files": files,
            "process_exit_reload_evaluation": "inspect worker candidate evaluation evidence",
        },
    )
    return export


def shared_token_ledger() -> dict[str, int]:
    """Count each cumulative phase once; preserve reservations for uncollected work."""
    cpt_jobs = sorted(REPORT.glob("job-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
    ledger: dict[str, int] = {}
    if cpt_jobs:
        attempt = int(cpt_jobs[-1].stem.split("-")[-1])
        verification = REPORT / f"verified-output-{attempt}.json"
        ledger["cpt"] = (
            int(
                json.loads(verification.read_text())["processed_campaign_input_tokens_conservative"]
            )
            if verification.exists()
            else 12000000
        )
    for arm in FIM_ARMS:
        jobs = sorted(
            REPORT.glob(f"fim-job-{arm}-*.json"), key=lambda p: int(p.stem.split("-")[-1])
        )
        if not jobs:
            continue
        job = json.loads(jobs[-1].read_text())
        verification = REPORT / f"fim-verified-{arm}-{job['attempt']}.json"
        ledger[arm] = (
            int(json.loads(verification.read_text())["processed_arm_input_tokens_conservative"])
            if verification.exists()
            else int(job["processed_arm_token_reservation"])
        )
    if any(value < 0 for value in ledger.values()):
        raise ValueError("campaign token evidence contains a negative counter")
    return ledger


def check_shared_token_budget(phase: str, reservation: int) -> int:
    """Replace only this phase's cumulative reservation; return other-phase carry."""
    if phase not in ("cpt", *FIM_ARMS) or reservation < 0:
        raise ValueError("invalid campaign phase or token reservation")
    limits = json.loads((REPORT / "campaign_budget.json").read_text())["shared_limits"]
    ledger = shared_token_ledger()
    other = sum(value for key, value in ledger.items() if key != phase)
    if other + reservation > limits["maximum_additional_processed_training_input_tokens"]:
        raise RuntimeError("shared CPT/FIM processed-token reservation exhausted")
    return other


def freeze_fim_plan(cpt_plan: dict[str, Any], attempt: int) -> dict[str, Any]:
    """Freeze both matched arms after verifying their actual initializers and runtime."""
    export = collect_cpt_export(cpt_plan, attempt)
    export_manifest = json.loads((export / "artifact_manifest.json").read_text())
    output = ARTIFACTS / f"output-attempt-{attempt}"
    manifests = list(output.glob("**/training/run_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("CPT runtime identity is missing or ambiguous")
    parent_runtime = json.loads(manifests[0].read_text())["identity"]["runtime"]
    runtime_lock_path = REPORT / "fim_runtime_lock.json"
    runtime_lock = json.loads(runtime_lock_path.read_text())
    runtime = runtime_lock["expected_versions"]
    expected_runtime = {
        "python": "3.11.15",
        "torch": "2.11.0+cu128",
        "transformers": "5.17.0",
        "bitsandbytes": "0.50.2",
        "cuda_runtime": "12.8",
    }
    if runtime != expected_runtime:
        raise ValueError("FIM dependency lock differs from the approved Python 3.11 runtime")
    requirements = runtime_lock["requirements_lock"]
    relative_path = Path(requirements["repo_relative_path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("FIM requirements lock must reside inside the repository")
    if digest(ROOT / relative_path) != requirements["sha256"]:
        raise ValueError("FIM hashed requirements differ from the runtime lock")
    corpus = ARTIFACTS / "fim/corpus-r3"
    metadata = json.loads((corpus / "corpus_metadata.json").read_text())
    preparation = json.loads((REPORT / "fim_preparation_plan.json").read_text())
    if metadata["preparation_plan_sha256"] != digest(REPORT / "fim_preparation_plan.json"):
        raise ValueError("FIM preparation plan and corpus identities differ")
    for name, record in metadata["files"].items():
        if Path(name).name != name or digest(corpus / name) != record["sha256"]:
            raise ValueError("prepared FIM corpus file identity differs")
    budget = json.loads((REPORT / "campaign_budget.json").read_text())["shared_limits"]
    training = dict(preparation["planned_training"])
    training["max_input_tokens"] = training.pop("maximum_input_tokens_per_arm")
    training["sequence_length"] = 1024
    training["attention"] = "sdpa"
    training["max_output_bytes"] = 10 * 1024**3
    base = cpt_plan["configuration"]["model"]
    plan = {
        "schema": "q25-fim-training-plan-v1",
        "plan_revision": 4,
        "revision_reason": (
            "Attempt 3 exhausted T4 memory during full-prompt FIM line evaluation. "
            "Transformers enables native GQA, which PyTorch 2.11 supports only through "
            "Flash or math SDPA; T4 cannot use Flash. Revision 4 repeats KV heads "
            "explicitly and requires supported memory-efficient SDPA for evaluation. "
            "Both matched arms rerun all evaluations. Model weights, tokenizer, full "
            "prompts, source fixtures, scoring, training data and trainer are unchanged. "
            "Kernel numerical differences make earlier evaluations historical."
        ),
        "restart_after_zero_work": {
            "previous_plan_sha256": (
                "ce721f444b15346a2dc4bb1ffa3c80204f36a1934a9c040241475c8227c7d1d8"
            ),
            "previous_plan_file": "fim_training_plan-r3.json",
        },
        "gpu_execution_authorized": True,
        "preparation_plan_sha256": digest(REPORT / "fim_preparation_plan.json"),
        "parent_cpt_plan_sha256": digest(REPORT / "plan.json"),
        "campaign_budget_sha256": digest(REPORT / "campaign_budget.json"),
        "parent_cpt_runtime_observation": parent_runtime,
        "runtime_revision_reason": (
            "CPT platform actually used Python 3.13.15; new matched arms pin Python "
            "3.11.15 as required. Both arms share the same locked runtime."
        ),
        "configuration": {
            "training": training,
            "runtime": runtime,
            "runtime_lock": {
                "runtime_lock_sha256": digest(runtime_lock_path),
                "requirements_lock_sha256": requirements["sha256"],
                "bootstrap_uv_version": runtime_lock["bootstrap_uv_version"],
                "bootstrap_uv_wheel_sha256": runtime_lock["bootstrap_uv_wheel_sha256"],
            },
            "budget": {
                "runtime_setup_reserve_seconds": 1800,
                "session_seconds": 10800,
                "finalization_reserve_seconds": 1800,
                "minimum_finalization_reserve_seconds": 1800,
                "maximum_additional_training_input_tokens": 32000000,
                "maximum_campaign_input_tokens": 32000000,
                "maximum_discarded_replay_input_tokens": 2097152,
                "new_artifact_bytes_cap": budget["new_artifact_bytes_cap_per_machine"],
                "minimum_free_bytes": budget["minimum_free_bytes"],
                "paid_compute": False,
                "automatic_renewal_use": False,
                "conservative_quota_multiplier": 2,
                "quota_renewal": budget["quota_renewal"],
            },
        },
        "data": {
            "corpus_metadata_sha256": digest(corpus / "corpus_metadata.json"),
            **{
                split: {
                    **metadata["splits"][split],
                    "bytes": metadata["files"][metadata["splits"][split]["file"]]["bytes"],
                }
                for split in ("train", "development")
            },
        },
        "initializers": {
            FIM_ARMS[0]: {
                "kind": "untouched_pretrained",
                "model_id": base["id"],
                "revision": base["revision"],
                "files": cpt_plan["existing_model_files"],
            },
            FIM_ARMS[1]: {
                "kind": "completed_cpt_export",
                "model_id": base["id"],
                "revision": base["revision"],
                "files": export_manifest["files"],
                "artifact_manifest_sha256": digest(export / "artifact_manifest.json"),
                "fingerprint": export_manifest["fingerprint"],
                "training_cursor": export_manifest["training_cursor"],
                "expected_complete_updates": 481,
                "expected_training_input_tokens": 7872512,
            },
        },
        "evaluation": {
            "attention_backend": "torch-efficient-sdpa-explicit-kv-repeat-v1",
            "source_syntax": {
                "protocol": "q25-fim-source-syntax-v1",
                "purpose": "descriptive original versus verbatim generated full-file parse",
                "selection_gate": False,
                "functional_or_human_intent_evidence": False,
            },
            "development": "240 synthetic source states; EOS-only, greedy, 96 tokens",
            "line": "unchanged 180-case source suite; PSM, newline/EOS, 96 tokens",
            "regression": "unchanged raw causal 200 and raw line 180",
            "fixtures": cpt_plan["fixtures"],
            "selection": (
                "paired development exact/terminated plus general regression; "
                "no automatic promotion"
            ),
            "general_next_edit_quality_established": False,
            "paired_analysis": {
                "primary": "development exact_and_terminated",
                "secondary": ["development exact", "line exact", "raw causal functional pass"],
                "bootstrap_unit": "repository identity",
                "bootstrap_samples": 2000,
                "bootstrap_seed": 271828,
                "report": "paired wins/losses and uncertainty; no equivalence from nonsignificance",
            },
        },
    }
    path = REPORT / "fim_training_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen matched FIM plan differs; create an explicit revision")
    if not path.exists():
        save(path, plan)
    return plan


def build_fim_bundle(plan: dict[str, Any]) -> Path:
    corpus = ARTIFACTS / "fim/corpus-r3"
    output = ARTIFACTS / "fim/input-bundle"
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "plan.json": REPORT / "fim_training_plan.json",
        "train.jsonl": corpus / "train.jsonl",
        "development.jsonl": corpus / "development.jsonl",
        "corpus_metadata.json": corpus / "corpus_metadata.json",
        "causal200.jsonl": ROOT / "data/benchmarks/code_completion_v2.jsonl",
        "line180.jsonl": FIM_LINE_SOURCE,
    }
    if plan.get("schema") != "q25-fim-training-plan-v1" or not plan.get("gpu_execution_authorized"):
        raise ValueError("a frozen full FIM plan is required")
    allowed = set(sources) | {"input-manifest.json", "dataset-metadata.json"}
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in output.iterdir()):
        raise ValueError("FIM bundle staging contains an unapproved file")
    budget = plan["configuration"]["budget"]
    projected = sum(
        path.stat().st_size for name, path in sources.items() if not (output / name).exists()
    )
    if directory_bytes(ARTIFACTS) + projected > budget["new_artifact_bytes_cap"] or (
        shutil.disk_usage(ARTIFACTS).free < projected + budget["minimum_free_bytes"]
    ):
        raise OSError("FIM staging would exhaust storage headroom")
    expected = {
        "train.jsonl": plan["data"]["train"]["sha256"],
        "development.jsonl": plan["data"]["development"]["sha256"],
        "corpus_metadata.json": plan["data"]["corpus_metadata_sha256"],
        "causal200.jsonl": plan["evaluation"]["fixtures"]["causal"]["sha256"],
        "line180.jsonl": plan["evaluation"]["fixtures"]["line"]["sha256"],
    }
    record = {}
    for name, source in sources.items():
        sha = digest(source)
        if name in expected and sha != expected[name]:
            raise ValueError("FIM input differs from frozen plan")
        target = output / name
        if target.exists() and digest(target) != sha:
            raise ValueError("immutable staged FIM input differs")
        if not target.exists():
            shutil.copy2(source, target)
        record[name] = {"sha256": sha, "bytes": target.stat().st_size}
    save(output / "input-manifest.json", {"files": record, "model_dataset": BASE_DATASET})
    save(
        output / "dataset-metadata.json",
        {
            "id": FIM_DATASET,
            "title": "TabComplete Q25 matched FIM r2 inputs",
            "licenses": [{"name": "other"}],
        },
    )
    budget = plan["configuration"]["budget"]
    if directory_bytes(ARTIFACTS) > budget["new_artifact_bytes_cap"] or (
        shutil.disk_usage(ARTIFACTS).free < budget["minimum_free_bytes"]
    ):
        raise OSError("FIM bundle exceeds storage headroom")
    return output


def upload_fim_bundle(plan: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _csv_refs, wait_for_remote_inputs

    output = build_fim_bundle(plan)
    marker = ARTIFACTS / "fim/dataset-submission.json"
    expected = json.loads((output / "input-manifest.json").read_text())["files"]
    expected["input-manifest.json"] = {"bytes": (output / "input-manifest.json").stat().st_size}
    if not marker.exists():
        refs = _csv_refs(["kaggle", "datasets", "list", "--mine", "--page-size", "100", "--csv"])
        if FIM_DATASET in refs:
            raise ValueError("FIM dataset already exists without this campaign's receipt")
        response = cli("kaggle", "datasets", "create", "-p", str(output), "-t", timeout=900)
        if "Your private Dataset is being created." not in response:
            raise RuntimeError("Kaggle did not confirm private FIM dataset creation")
        save(
            marker,
            {
                "dataset": FIM_DATASET,
                "state": "created",
                "input_manifest_sha256": digest(output / "input-manifest.json"),
            },
        )
    recorded = json.loads(marker.read_text())
    if recorded["input_manifest_sha256"] != digest(output / "input-manifest.json"):
        raise ValueError("uploaded FIM dataset fingerprint differs")
    verification = wait_for_remote_inputs(FIM_DATASET, expected)
    save(marker, {**recorded, "state": "verified", "remote_verification": verification})
    return verification


def upload_cpt_initializer(plan: dict[str, Any]) -> dict[str, Any]:
    """Stage only the approved completed export in a private training input."""
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _csv_refs, wait_for_remote_inputs

    entry = plan["initializers"][FIM_ARMS[1]]
    candidates = [
        path
        for path in ARTIFACTS.glob("output-attempt-*/**/inference-f16/artifact_manifest.json")
        if digest(path) == entry["artifact_manifest_sha256"]
    ]
    if len(candidates) != 1:
        raise ValueError("verified completed CPT initializer is unavailable or ambiguous")
    source = candidates[0].parent
    output = ARTIFACTS / "fim/cpt-initializer-bundle"
    output.mkdir(parents=True, exist_ok=True)
    files = {
        **entry["files"],
        "artifact_manifest.json": {
            "sha256": entry["artifact_manifest_sha256"],
            "bytes": candidates[0].stat().st_size,
        },
    }
    for name, record in files.items():
        path = source / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError("CPT initializer source differs from frozen plan")
    marker = ARTIFACTS / "fim/cpt-initializer-submission.json"
    recorded = json.loads(marker.read_text()) if marker.exists() else None
    if recorded is not None and (
        recorded.get("dataset") != CPT_INITIALIZER_DATASET
        or recorded.get("artifact_manifest_sha256") != entry["artifact_manifest_sha256"]
    ):
        raise ValueError("uploaded CPT initializer identity differs")
    # A verified immutable upload needs no second local weight copy on resume.
    if recorded is not None and recorded.get("state") == "verified":
        verification = wait_for_remote_inputs(CPT_INITIALIZER_DATASET, files)
        save(marker, {**recorded, "remote_verification": verification})
        return verification
    allowed = set(files) | {"dataset-metadata.json"}
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in output.iterdir()):
        raise ValueError("CPT initializer staging contains an unapproved file")
    missing = sum(record["bytes"] for name, record in files.items() if not (output / name).exists())
    budget = plan["configuration"]["budget"]
    if directory_bytes(ARTIFACTS) + missing > budget["new_artifact_bytes_cap"] or (
        shutil.disk_usage(ARTIFACTS).free < missing + budget["minimum_free_bytes"]
    ):
        raise OSError("private CPT initializer staging exceeds storage headroom")
    for name, record in files.items():
        if Path(name).name != name or digest(source / name) != record["sha256"]:
            raise ValueError("CPT initializer source differs from frozen plan")
        target = output / name
        if target.exists() and digest(target) != record["sha256"]:
            raise ValueError("immutable staged CPT initializer differs")
        if not target.exists():
            shutil.copy2(source / name, target)
        if target.stat().st_size != record["bytes"]:
            raise ValueError("CPT initializer staged byte count differs")
    save(
        output / "dataset-metadata.json",
        {
            "id": CPT_INITIALIZER_DATASET,
            "title": "TabComplete Q25 completed CPT FIM initializer",
            "licenses": [{"name": "other"}],
        },
    )
    if not marker.exists():
        refs = _csv_refs(["kaggle", "datasets", "list", "--mine", "--page-size", "100", "--csv"])
        if CPT_INITIALIZER_DATASET in refs:
            raise ValueError("CPT initializer dataset exists without this campaign's receipt")
        response = cli("kaggle", "datasets", "create", "-p", str(output), "-t", timeout=900)
        if "Your private Dataset is being created." not in response:
            raise RuntimeError("Kaggle did not confirm private initializer creation")
        save(
            marker,
            {
                "dataset": CPT_INITIALIZER_DATASET,
                "state": "created",
                "artifact_manifest_sha256": entry["artifact_manifest_sha256"],
            },
        )
    recorded = json.loads(marker.read_text())
    if recorded["artifact_manifest_sha256"] != entry["artifact_manifest_sha256"]:
        raise ValueError("uploaded CPT initializer identity differs")
    verification = wait_for_remote_inputs(CPT_INITIALIZER_DATASET, files)
    save(marker, {**recorded, "state": "verified", "remote_verification": verification})
    save(
        REPORT / "cpt-initializer-upload.json",
        {
            "dataset": CPT_INITIALIZER_DATASET,
            "private": True,
            "files": files,
            "purpose": "completed approved q25 artifact for matched FIM; no optimizer states",
            "source_model": "Qwen/Qwen2.5-Coder-0.5B",
            "new_external_model_weights": False,
            "public_weight_publication": False,
        },
    )
    # Only remove our upload staging duplicate, after remote verification. The
    # completed research export and resumable checkpoints remain authoritative.
    staged_weights = output / "model.safetensors"
    record = files["model.safetensors"]
    if (
        staged_weights.is_symlink()
        or staged_weights.stat().st_size != record["bytes"]
        or digest(staged_weights) != record["sha256"]
    ):
        raise ValueError("CPT initializer staging changed before cleanup")
    staged_weights.unlink()
    save(
        marker,
        {
            **json.loads(marker.read_text()),
            "temporary_staged_weight_removed": True,
            "authoritative_export_preserved": True,
        },
    )
    return verification


def validate_fim_retry_plan(
    plan: dict[str, Any], prior: dict[str, Any], verified: dict[str, Any]
) -> None:
    """A changed evaluator may restart only an explicitly verified zero-work arm."""
    current_sha = digest(REPORT / "fim_training_plan.json")
    if prior.get("plan_sha256") == current_sha:
        return
    transition = plan.get("restart_after_zero_work", {})
    name = transition.get("previous_plan_file")
    if (
        verified.get("no_training_executed") is not True
        or verified.get("training_status") != "no_training_executed"
        or verified.get("processed_arm_input_tokens_conservative") != 0
        or verified.get("own_discarded_tokens_for_resume") != 0
        or prior.get("resume_source") is not None
        or prior.get("resume_identity") is not None
        or verified.get("carried_checkpoint_source") is not None
        or verified.get("carried_checkpoint_identity") is not None
        or verified.get("carried_checkpoint_verified") is not False
        or transition.get("previous_plan_sha256") != prior.get("plan_sha256")
        or not isinstance(name, str)
        or Path(name).name != name
        or name in (".", "..")
    ):
        raise ValueError("a changed FIM plan requires an explicit verified zero-work restart")
    archived = REPORT / name
    if not archived.is_file() or digest(archived) != prior["plan_sha256"]:
        raise ValueError("historical FIM plan identity differs")
    old = json.loads(archived.read_text())
    excluded = {"evaluation", "plan_revision", "revision_reason", "restart_after_zero_work"}
    old_evaluation = {
        k: v for k, v in old.get("evaluation", {}).items() if k != "attention_backend"
    }
    new_evaluation = {
        k: v for k, v in plan.get("evaluation", {}).items() if k != "attention_backend"
    }
    if (
        {k: v for k, v in old.items() if k not in excluded}
        != {k: v for k, v in plan.items() if k not in excluded}
        or old_evaluation != new_evaluation
        or int(plan.get("plan_revision", 0)) <= int(old.get("plan_revision", 0))
    ):
        raise ValueError("FIM zero-work revision changed the frozen training identity")


def submit_fim(
    plan: dict[str, Any], *, arm: str, attempt: int, cpt_attempt: int, resume_source: str | None
) -> dict[str, Any]:
    if arm not in FIM_ARMS or plan.get("gpu_execution_authorized") is not True:
        raise ValueError("a frozen authorized FIM arm is required")
    budget = plan["configuration"]["budget"]
    other_arm = next(value for value in FIM_ARMS if value != arm)
    other_outputs = list(REPORT.glob(f"fim-verified-{other_arm}-*.json"))
    other_complete = any(
        json.loads(path.read_text()).get("training_status") == "complete" for path in other_outputs
    )
    future_reserve = 0 if other_complete else int(budget["session_seconds"])
    check_shared_allocation_budget(int(budget["session_seconds"]) + future_reserve, phase="fim")
    reservation = int(plan["configuration"]["training"]["max_input_tokens"])
    reservation += int(budget["maximum_discarded_replay_input_tokens"])
    other_tokens = check_shared_token_budget(arm, reservation)
    existing = list(REPORT.glob(f"fim-job-{arm}-*.json"))
    target = REPORT / f"fim-job-{arm}-{attempt}.json"
    if target.exists() or attempt != len(existing) + 1 or (attempt > 1 and not resume_source):
        raise ValueError("FIM attempts must be new, sequential, and resumable")
    own_discarded = 0
    effective_resume_source = None
    resume_identity = None
    if attempt > 1:
        prior = json.loads((REPORT / f"fim-job-{arm}-{attempt - 1}.json").read_text())
        verified = json.loads((REPORT / f"fim-verified-{arm}-{attempt - 1}.json").read_text())
        if (
            resume_source != prior["reference"]
            or verified.get("reference") != prior["reference"]
            or verified.get("arm") != arm
            or verified.get("attempt") != attempt - 1
            or verified.get("plan_sha256") != prior.get("plan_sha256")
            or verified.get("input_manifest_sha256") != prior.get("input_manifest_sha256")
            or verified.get("commit") != prior.get("commit")
        ):
            raise ValueError("FIM retry authorization lineage differs from its prior receipt")
        validate_fim_retry_plan(plan, prior, verified)
        if verified.get("training_status") == "complete":
            raise ValueError("the frozen FIM pass is already complete")
        if verified.get("no_training_executed") is True:
            if verified.get("training_status") != "no_training_executed":
                raise ValueError("zero-work FIM receipt has an invalid training status")
            effective_resume_source = prior.get("resume_source")
            resume_identity = prior.get("resume_identity")
            if (
                verified.get("carried_checkpoint_source") != effective_resume_source
                or verified.get("carried_checkpoint_identity") != resume_identity
                or verified.get("carried_checkpoint_verified") is not (resume_identity is not None)
            ):
                raise ValueError("zero-work FIM receipt lost its inherited checkpoint identity")
            if effective_resume_source is None:
                if resume_identity is not None:
                    raise ValueError("FIM retry has a checkpoint identity without a source")
            else:
                if not isinstance(resume_identity, dict):
                    raise ValueError("FIM retry lacks its inherited checkpoint identity")
                source_jobs = [
                    json.loads(path.read_text())
                    for path in REPORT.glob(f"fim-job-{arm}-*.json")
                    if path != REPORT / f"fim-job-{arm}-{attempt - 1}.json"
                    and json.loads(path.read_text()).get("reference") == effective_resume_source
                ]
                if len(source_jobs) != 1:
                    raise ValueError("inherited FIM checkpoint source is missing or ambiguous")
                source_job = source_jobs[0]
                source_verified_path = REPORT / f"fim-verified-{arm}-{source_job['attempt']}.json"
                source_verified = json.loads(source_verified_path.read_text())
                expected_identity = {
                    "fingerprint": source_verified.get("fingerprint"),
                    "checkpoint_sha256": source_verified.get("checkpoint_sha256"),
                    "cursor": source_verified.get("cursor"),
                }
                if (
                    not source_verified.get("checkpoint_verified")
                    or source_verified.get("reference") != effective_resume_source
                    or resume_identity != expected_identity
                ):
                    raise ValueError("inherited FIM checkpoint is not verified")
            own_discarded = int(verified["own_discarded_tokens_for_resume"])
        else:
            if not verified.get("checkpoint_verified"):
                raise ValueError(
                    "FIM resume requires a verified checkpoint or verified zero-work receipt"
                )
            effective_resume_source = prior["reference"]
            own_discarded = int(verified["own_discarded_tokens_for_resume"])
            resume_identity = {
                "fingerprint": verified["fingerprint"],
                "checkpoint_sha256": verified["checkpoint_sha256"],
                "cursor": verified["cursor"],
            }
        if own_discarded > budget["maximum_discarded_replay_input_tokens"]:
            raise RuntimeError("FIM discarded-tail reservation exhausted")
    cpt_verified = json.loads((REPORT / f"verified-output-{cpt_attempt}.json").read_text())
    if cpt_verified["training_status"] != "complete" or not cpt_verified["checkpoint_verified"]:
        raise ValueError("matched FIM requires the completed verified CPT initializer")
    initializer = plan["initializers"][FIM_ARMS[1]]
    if (
        cpt_verified["fingerprint"] != initializer["fingerprint"]
        or cpt_verified["cursor"] != initializer["training_cursor"]
    ):
        raise ValueError("CPT allocation differs from the frozen matched FIM initializer")
    if cli("git", "-C", str(ROOT), "status", "--porcelain"):
        raise RuntimeError("commit all FIM campaign code before submitting")
    commit = cli("git", "-C", str(ROOT), "rev-parse", "HEAD")
    remote = cli(
        "git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/research/q25-code-cpt-r2"
    )
    if remote.split()[0] != commit:
        raise RuntimeError("push frozen FIM campaign code before submitting")
    manifest = ARTIFACTS / "fim/input-bundle/input-manifest.json"
    receipt = json.loads((ARTIFACTS / "fim/dataset-submission.json").read_text())
    if receipt.get("state") != "verified" or receipt["input_manifest_sha256"] != digest(manifest):
        raise ValueError("private FIM inputs must be uploaded and verified before allocation")
    reference = f"shlokbhakta/tc-q25-fim-r2-{FIM_ARMS.index(arm)}-a{attempt}"
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _csv_refs

    if reference in _csv_refs(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"]
    ):
        raise ValueError("FIM kernel reference already exists; refusing duplicate allocation")
    observation = quota()
    check_quota(plan, observation)
    kernel = ARTIFACTS / f"fim/kernel-{arm}-{attempt}"
    kernel.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"run.py", "kernel-metadata.json"} for p in kernel.iterdir()):
        raise ValueError("FIM kernel staging contains an unapproved file")
    sources = [effective_resume_source] if effective_resume_source else []
    datasets = [FIM_DATASET]
    if arm == FIM_ARMS[0]:
        datasets.append(BASE_DATASET)
    if arm == FIM_ARMS[1]:
        initializer_receipt = json.loads(
            (ARTIFACTS / "fim/cpt-initializer-submission.json").read_text()
        )
        if initializer_receipt.get("state") != "verified" or (
            initializer_receipt["artifact_manifest_sha256"]
            != initializer["artifact_manifest_sha256"]
        ):
            raise ValueError(
                "private CPT initializer must be uploaded and verified before allocation"
            )
        datasets.append(CPT_INITIALIZER_DATASET)
    session = {
        "commit": commit,
        "plan_sha256": digest(REPORT / "fim_training_plan.json"),
        "input_manifest_sha256": digest(manifest),
        "arm": arm,
        "attempt": attempt,
        "session_seconds": budget["session_seconds"],
        "authorization_lineage_reference": resume_source,
        "resume_source": effective_resume_source,
        "resume_identity": resume_identity,
        "external_campaign_tokens": other_tokens + own_discarded,
        "other_phase_tokens": other_tokens,
        "own_discarded_tokens": own_discarded,
    }
    worker = (ROOT / "kaggle/q25_code_cpt_r2/run_fim.py").read_text()
    (kernel / "run.py").write_text(worker.replace("__SESSION_JSON__", repr(session)))
    save(
        kernel / "kernel-metadata.json",
        {
            "id": reference,
            "title": reference.split("/")[1].replace("-", " "),
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": datasets,
            "kernel_sources": sources,
        },
    )
    save(REPORT / f"fim-quota-{arm}-{attempt}.json", observation)
    job = {
        **session,
        "reference": reference,
        "quota": observation,
        "processed_arm_token_reservation": reservation,
        "conservative_reserved_session_seconds": budget["session_seconds"],
        "status": "submission_pending",
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    save(target, job)
    try:
        response = cli(
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(kernel),
            "--timeout",
            str(budget["session_seconds"]),
            "--accelerator",
            "NvidiaTeslaT4",
            timeout=240,
        )
        urls = re.findall(
            r"https://www\.kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", response
        )
        job.update(
            status="submitted",
            submission_response=response,
            requested_reference=reference,
            reference=urls[-1] if urls else reference,
        )
    except Exception as exc:
        job.update(status="submission_unknown", error_class=type(exc).__name__)
        save(target, job)
        raise
    save(target, job)
    return job


def submit(plan: dict[str, Any], *, attempt: int, resume_source: str | None) -> dict[str, Any]:
    budget = plan["configuration"]["budget"]
    check_shared_allocation_budget(int(budget["session_seconds"]), phase="cpt")
    if (REPORT / "campaign_budget.json").exists():
        check_shared_token_budget("cpt", int(budget["maximum_additional_training_input_tokens"]))
    existing = sorted(REPORT.glob("job-*.json"))
    maximum = len(existing) * budget["session_seconds"] + budget["session_seconds"]
    if maximum > budget["aggregate_session_seconds"]:
        raise RuntimeError("aggregate conservative session reservation exhausted")
    if (
        maximum / 3600 * budget["conservative_quota_multiplier"]
        > (budget["quota_account_gpu_hours_cap"])
    ):
        raise RuntimeError("campaign account GPU-hour reservation exhausted")
    target = REPORT / f"job-{attempt}.json"
    if target.exists():
        raise FileExistsError("attempt already submitted; inspect or resume a new attempt")
    if attempt != len(existing) + 1 or (attempt > 1 and not resume_source):
        raise ValueError("attempts must be sequential and later attempts must resume")
    external_campaign_tokens = 0
    if attempt > 1:
        if resume_source is None:
            raise ValueError("a resumed attempt requires a kernel source")
        prior = json.loads((REPORT / f"job-{attempt - 1}.json").read_text())
        if resume_source != prior["reference"]:
            raise ValueError("resume must use the immediately prior campaign kernel")
        verification = REPORT / f"verified-output-{attempt - 1}.json"
        if not verification.exists() or not json.loads(verification.read_text()).get(
            "checkpoint_verified"
        ):
            raise RuntimeError("verify the previous complete checkpoint before resume")
        evidence = json.loads(verification.read_text())
        if evidence.get("training_status") == "complete":
            raise RuntimeError("the frozen raw-code pass is already complete")
        external_campaign_tokens = int(evidence.get("external_campaign_tokens_for_resume", 0))
        if external_campaign_tokens > budget.get("maximum_discarded_replay_input_tokens", 0):
            raise RuntimeError("discarded-tail replay token reservation exhausted")
        prior_status = cli("kaggle", "kernels", "status", resume_source)
        if not any(x in prior_status for x in ("COMPLETE", "ERROR")):
            raise RuntimeError("previous allocation has not exited")
    if cli("git", "-C", str(ROOT), "status", "--porcelain"):
        raise RuntimeError("commit all campaign code before submitting")
    commit = cli("git", "-C", str(ROOT), "rev-parse", "HEAD")
    remote = cli(
        "git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/research/q25-code-cpt-r2"
    )
    if remote.split()[0] != commit:
        raise RuntimeError("push frozen campaign code before submitting")
    manifest = ARTIFACTS / "input-bundle/input-manifest.json"
    receipt = json.loads((ARTIFACTS / "dataset-submission.json").read_text())
    if receipt.get("state") != "verified" or receipt["input_manifest_sha256"] != digest(manifest):
        raise ValueError("upload and verify immutable private inputs before allocation")
    kernel = ARTIFACTS / f"kernel-attempt-{attempt}"
    reference = f"shlokbhakta/tc-q25-code-cpt-r2-a{attempt}"
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _csv_refs

    if reference in _csv_refs(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"]
    ):
        raise ValueError("kernel ID already exists; refusing duplicate allocation")
    observation = quota()
    check_quota(plan, observation)
    kernel.mkdir(exist_ok=True)
    if any(p.name not in {"run.py", "kernel-metadata.json"} for p in kernel.iterdir()):
        raise ValueError("kernel staging contains an unapproved file")
    session = {
        "commit": commit,
        "plan_sha256": digest(REPORT / "plan.json"),
        "input_manifest_sha256": digest(manifest),
        "attempt": attempt,
        "session_seconds": budget["session_seconds"],
        "resume_source": resume_source,
        "external_campaign_tokens": external_campaign_tokens,
    }
    worker = (ROOT / "kaggle/q25_code_cpt_r2/run.py").read_text()
    (kernel / "run.py").write_text(worker.replace("__SESSION_JSON__", repr(session)))
    save(
        kernel / "kernel-metadata.json",
        {
            "id": reference,
            "title": f"tc q25 code cpt r2 a{attempt}",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [DATASET, BASE_DATASET],
            "kernel_sources": [resume_source] if resume_source else [],
        },
    )
    save(REPORT / f"quota-before-attempt-{attempt}.json", observation)
    job = {
        **session,
        "reference": reference,
        "quota": observation,
        "conservative_reserved_session_seconds": budget["session_seconds"],
        "status": "submission_pending",
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    save(target, job)
    try:
        response = cli(
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(kernel),
            "--timeout",
            str(budget["session_seconds"]),
            "--accelerator",
            "NvidiaTeslaT4",
            timeout=240,
        )
        urls = re.findall(
            r"https://www\.kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", response
        )
        actual_reference = urls[-1] if urls else reference
        job.update(
            status="submitted",
            submission_response=response,
            requested_reference=reference,
            reference=actual_reference,
        )
    except Exception as exc:
        job.update(status="submission_unknown", error_class=type(exc).__name__)
        save(target, job)
        raise
    save(target, job)
    return job


def collect(plan: dict[str, Any], attempt: int) -> dict[str, Any]:
    job = json.loads((REPORT / f"job-{attempt}.json").read_text())
    status = cli("kaggle", "kernels", "status", job["reference"])
    if not any(terminal in status for terminal in ("COMPLETE", "ERROR")):
        return {"status": status, "checkpoint_verified": False}
    output = ARTIFACTS / f"output-attempt-{attempt}"
    output.mkdir(exist_ok=True)
    budget = plan["configuration"]["budget"]
    checkpoint_reservation = 4 * 1024**3
    if directory_bytes(ARTIFACTS) + checkpoint_reservation > budget["new_artifact_bytes_cap"]:
        raise OSError("checkpoint download would exceed the artifact cap")
    if shutil.disk_usage(ARTIFACTS).free < checkpoint_reservation + budget["minimum_free_bytes"]:
        raise OSError("checkpoint download would exhaust storage headroom")
    cli(
        "kaggle",
        "kernels",
        "output",
        job["reference"],
        "-p",
        str(output),
        "-q",
        "--file-pattern",
        r"\.(json|jsonl|log)$",
        timeout=300,
    )
    pointers = list(output.glob("**/training/latest.json"))
    if len(pointers) != 1:
        raise ValueError("worker did not preserve exactly one latest checkpoint pointer")
    pointer = json.loads(pointers[0].read_text())
    name = pointer.get("path") or pointer.get("checkpoint")
    if not isinstance(name, str) or not re.fullmatch(r"resume-step-[0-9]+\.pt", Path(name).name):
        raise ValueError("worker checkpoint filename is invalid")
    name = Path(name).name
    marker_path = pointers[0].parent / (name + ".complete.json")
    marker = json.loads(marker_path.read_text())
    # Only the latest complete state is retrieved; never download obsolete states.
    cli(
        "kaggle",
        "kernels",
        "output",
        job["reference"],
        "-p",
        str(output),
        "-q",
        "--file-pattern",
        re.escape(name) + r"$",
        timeout=900,
    )
    checkpoint = pointers[0].parent / name
    if digest(checkpoint) != marker["sha256"]:
        raise ValueError("downloaded complete checkpoint digest differs")
    result_path = pointers[0].parent / "run_result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {
            "status": "interrupted",
            "fingerprint": pointer["fingerprint"],
            "cursor": pointer["cursor"],
        }
    )
    if result["fingerprint"] != marker["fingerprint"]:
        raise ValueError("checkpoint and scientific result identities differ")
    if (
        result["cursor"]["training_input_tokens"]
        > plan["configuration"]["training"]["max_input_tokens"]
    ):
        raise ValueError("worker exceeded the campaign token cap")
    committed = int(pointer["cursor"]["training_input_tokens"])
    if result["cursor"] != pointer["cursor"]:
        raise ValueError("scientific result and complete checkpoint cursors differ")
    worker_paths = list(output.glob("**/worker-status.json"))
    worker = json.loads(worker_paths[0].read_text()) if len(worker_paths) == 1 else {}
    training_exited_cleanly = any(
        stage.get("name") == "training" and stage.get("exit_code") == 0
        for stage in worker.get("stages", [])
    )
    updates_path = pointers[0].parent / "updates.jsonl"
    updates = []
    truncated_last_record = False
    if updates_path.exists():
        content = updates_path.read_text()
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1 or content.endswith("\n") or training_exited_cleanly:
                    raise ValueError("worker update log contains a malformed record") from None
                truncated_last_record = True
                break
            if not isinstance(row, dict) or not isinstance(row.get("cumulative_input_tokens"), int):
                raise ValueError("worker update log lacks token counters")
            updates.append(row)
    elif committed or not training_exited_cleanly:
        raise ValueError("worker update log is missing; discarded work cannot be bounded")
    recorded = max([committed, *(int(row["cumulative_input_tokens"]) for row in updates)])
    configuration = plan["configuration"]
    unlogged_reservation = (
        0
        if training_exited_cleanly
        else configuration["training"]["effective_batch"]
        * configuration["training"]["sequence_length"]
    )
    discarded_tail = recorded - committed + unlogged_reservation
    external = int(job.get("external_campaign_tokens", 0)) + discarded_tail
    total_processed_reservation = committed + external
    if total_processed_reservation > budget.get(
        "maximum_additional_training_input_tokens", 8000000
    ):
        raise RuntimeError("actual processed-token reservation exceeded the campaign cap")
    if directory_bytes(ARTIFACTS) > budget["new_artifact_bytes_cap"]:
        raise OSError("downloaded research artifacts exceed storage cap")
    record = {
        "reference": job["reference"],
        "status": status,
        "checkpoint_verified": True,
        "checkpoint_sha256": marker["sha256"],
        "fingerprint": marker["fingerprint"],
        "cursor": result["cursor"],
        "training_status": result["status"],
        "discarded_logged_tail_tokens": recorded - committed,
        "truncated_last_update_record": truncated_last_record,
        "unlogged_inflight_input_token_reservation": unlogged_reservation,
        "external_campaign_tokens_for_resume": external,
        "processed_campaign_input_tokens_conservative": total_processed_reservation,
        "checkpoint_path": str(checkpoint),
        "observed_at": datetime.now(UTC).isoformat(),
    }
    save(REPORT / f"verified-output-{attempt}.json", record)
    return record


def watch(plan: dict[str, Any], attempt: int, *, poll_seconds: float = 30) -> dict[str, Any]:
    """Observe an existing allocation and retrieve its terminal output.

    This never allocates compute or retries a submission. Remote training keeps
    its own deadline even if this observer exits or loses connectivity.
    """
    if not 1 <= poll_seconds <= 60:
        raise ValueError("watch interval must be between one and sixty seconds")
    job = json.loads((REPORT / f"job-{attempt}.json").read_text())
    if job["plan_sha256"] != digest(REPORT / "plan.json"):
        raise ValueError("observed allocation belongs to a different plan")
    end = datetime.fromisoformat(job["submitted_at"]).timestamp()
    end += plan["configuration"]["budget"]["session_seconds"] + 900
    failures = 0
    while datetime.now(UTC).timestamp() < end:
        observation: dict[str, Any] = {
            "reference": job["reference"],
            "observed_at": datetime.now(UTC).isoformat(),
            "automatic_allocation": False,
        }
        try:
            status = cli("kaggle", "kernels", "status", job["reference"], timeout=60)
            observation["status"] = status
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            failures += 1
            observation.update(error_class=type(exc).__name__, consecutive_failures=failures)
            save(REPORT / f"watch-{attempt}.json", observation)
            if failures >= 3:
                raise RuntimeError(
                    "remote observer lost connectivity; allocation was not retried"
                ) from None
            time.sleep(poll_seconds)
            continue
        save(REPORT / f"watch-{attempt}.json", observation)
        if any(terminal in status for terminal in ("COMPLETE", "ERROR")):
            return collect(plan, attempt)
        time.sleep(poll_seconds)
    raise TimeoutError("observer deadline reached; inspect the existing allocation before resuming")


def collect_fim(plan: dict[str, Any], arm: str, attempt: int) -> dict[str, Any]:
    if arm not in FIM_ARMS:
        raise ValueError("unknown FIM arm")
    job = json.loads((REPORT / f"fim-job-{arm}-{attempt}.json").read_text())
    if (
        job.get("arm") != arm
        or job.get("attempt") != attempt
        or job.get("plan_sha256") != digest(REPORT / "fim_training_plan.json")
        or not isinstance(job.get("commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", job["commit"])
        or not isinstance(job.get("input_manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", job["input_manifest_sha256"])
    ):
        raise ValueError("FIM allocation belongs to another frozen plan")
    status = cli("kaggle", "kernels", "status", job["reference"])
    if not any(value in status for value in ("COMPLETE", "ERROR")):
        return {"status": status, "checkpoint_verified": False}
    output = ARTIFACTS / f"fim/output-{arm}-{attempt}"
    output.mkdir(parents=True, exist_ok=True)
    budget = plan["configuration"]["budget"]
    reservation = 4 * 1024**3
    if directory_bytes(ARTIFACTS) + reservation > budget["new_artifact_bytes_cap"] or (
        shutil.disk_usage(ARTIFACTS).free < reservation + budget["minimum_free_bytes"]
    ):
        raise OSError("FIM checkpoint retrieval exceeds storage headroom")
    cli(
        "kaggle",
        "kernels",
        "output",
        job["reference"],
        "-p",
        str(output),
        "-q",
        "--file-pattern",
        r"\.(json|jsonl|log)$",
        timeout=300,
    )
    pointers = list(output.glob("**/training/latest.json"))
    if not pointers:
        worker_paths = list(output.glob("**/worker-status.json"))
        if len(worker_paths) != 1:
            raise ValueError("zero-work FIM receipt requires exactly one worker status")
        worker = json.loads(worker_paths[0].read_text())
        expected_worker_identity = {
            "schema": "q25-fim-kaggle-worker-status-v1",
            "commit": job.get("commit"),
            "attempt": attempt,
            "arm": arm,
            "plan_sha256": job.get("plan_sha256"),
            "input_manifest_sha256": job.get("input_manifest_sha256"),
        }
        if not isinstance(worker, dict) or any(
            worker.get(key) != value for key, value in expected_worker_identity.items()
        ):
            raise ValueError("zero-work FIM worker identity differs from its allocation")
        if worker.get("training_started") is not False:
            raise ValueError("FIM training state is unknown; zero-work cannot be verified")
        if worker.get("state") not in {
            "setup",
            "verified_inputs",
            "baseline_evaluation",
            "training_deferred_insufficient_time",
            "failed",
        }:
            raise ValueError("zero-work FIM worker state is inconsistent with no training")
        stages = worker.get("stages")
        if not isinstance(stages, list) or any(
            not isinstance(stage, dict) or stage.get("name") == "training" for stage in stages
        ):
            raise ValueError("zero-work FIM worker status contains training evidence")
        training_dirs = list(output.glob("**/training"))
        if any(
            training_dir.is_symlink()
            or any(path.is_file() or path.is_symlink() for path in training_dir.rglob("*"))
            for training_dir in training_dirs
        ):
            raise ValueError("zero-work FIM output contains training artifacts")
        if any(path.name == "training.log" for path in output.glob("**/*")):
            raise ValueError("zero-work FIM output contains a training log")
        if any(
            key in worker
            for key in ("training", "training_exit_code", "checkpoint_pointer_present")
        ):
            raise ValueError("zero-work FIM worker status contains training evidence")
        resume_source = job.get("resume_source")
        resume_identity = job.get("resume_identity")
        if (resume_source is None) != (resume_identity is None):
            raise ValueError("FIM allocation has an incomplete carried checkpoint identity")
        if resume_identity is None:
            committed = 0
        else:
            cursor = resume_identity.get("cursor")
            if not isinstance(cursor, dict):
                raise ValueError("FIM allocation has an invalid carried checkpoint cursor")
            carried_tokens = cursor.get("training_input_tokens")
            if (
                not isinstance(carried_tokens, int)
                or isinstance(carried_tokens, bool)
                or carried_tokens < 0
                or carried_tokens > plan["data"]["train"]["input_tokens"]
            ):
                raise ValueError("FIM allocation has an invalid carried checkpoint cursor")
            committed = carried_tokens
        own_discarded = job.get("own_discarded_tokens")
        if (
            not isinstance(own_discarded, int)
            or isinstance(own_discarded, bool)
            or own_discarded < 0
        ):
            raise ValueError("FIM allocation has an invalid discarded-token counter")
        if own_discarded > budget["maximum_discarded_replay_input_tokens"]:
            raise RuntimeError("FIM discarded-tail processed token budget exhausted")
        processed = committed + own_discarded
        check_shared_token_budget(arm, processed)
        if directory_bytes(ARTIFACTS) > budget["new_artifact_bytes_cap"]:
            raise OSError("FIM retrieval exceeded the artifact cap")
        record = {
            "reference": job["reference"],
            "status": status,
            "arm": arm,
            "attempt": attempt,
            "plan_sha256": job["plan_sha256"],
            "input_manifest_sha256": job["input_manifest_sha256"],
            "commit": job["commit"],
            "checkpoint_verified": False,
            "carried_checkpoint_verified": resume_identity is not None,
            "carried_checkpoint_source": resume_source,
            "carried_checkpoint_identity": resume_identity,
            "no_training_executed": True,
            "training_status": "no_training_executed",
            "own_discarded_tokens_for_resume": own_discarded,
            "processed_arm_input_tokens_conservative": processed,
            "other_phase_processed_tokens_at_submission": job["other_phase_tokens"],
            "observed_at": datetime.now(UTC).isoformat(),
        }
        save(REPORT / f"fim-verified-{arm}-{attempt}.json", record)
        return record
    if len(pointers) != 1:
        raise ValueError("FIM worker must preserve exactly one complete state pointer")
    pointer = json.loads(pointers[0].read_text())
    name = pointer.get("path")
    if not isinstance(name, str) or not re.fullmatch(r"resume-step-[0-9]+\.pt", name):
        raise ValueError("FIM checkpoint filename is invalid")
    marker = json.loads((pointers[0].parent / (name + ".complete.json")).read_text())
    cli(
        "kaggle",
        "kernels",
        "output",
        job["reference"],
        "-p",
        str(output),
        "-q",
        "--file-pattern",
        re.escape(name) + "$",
        timeout=900,
    )
    checkpoint = pointers[0].parent / name
    if digest(checkpoint) != marker["sha256"] or pointer["fingerprint"] != marker["fingerprint"]:
        raise ValueError("FIM checkpoint hash or identity differs")
    result_path = pointers[0].parent / "run_result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {
            "status": "interrupted",
            "fingerprint": pointer["fingerprint"],
            "cursor": pointer["cursor"],
        }
    )
    if result["fingerprint"] != marker["fingerprint"] or result["cursor"] != pointer["cursor"]:
        raise ValueError("FIM result and complete checkpoint cursors differ")
    committed = int(pointer["cursor"]["training_input_tokens"])
    if committed > plan["data"]["train"]["input_tokens"]:
        raise ValueError("FIM worker exceeded its single ordered pass")
    worker_paths = list(output.glob("**/worker-status.json"))
    worker = json.loads(worker_paths[0].read_text()) if len(worker_paths) == 1 else {}
    clean = any(
        stage.get("name") == "training" and stage.get("exit_code") == 0
        for stage in worker.get("stages", [])
    )
    update_path = pointers[0].parent / "updates.jsonl"
    recorded = committed
    truncated = False
    if update_path.exists():
        content = update_path.read_text()
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1 or content.endswith("\n") or clean:
                    raise ValueError("FIM update log contains a malformed record") from None
                truncated = True
                break
            value = row.get("cumulative_input_tokens") if isinstance(row, dict) else None
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("FIM update log lacks a valid token counter")
            recorded = max(recorded, value)
    elif committed or not clean:
        raise ValueError("FIM update log missing; discarded work cannot be bounded")
    training = plan["configuration"]["training"]
    unlogged = 0 if clean else int(training["effective_batch"]) * int(training["sequence_length"])
    own_discarded = int(job["own_discarded_tokens"]) + recorded - committed + unlogged
    processed = committed + own_discarded
    if own_discarded > budget["maximum_discarded_replay_input_tokens"]:
        raise RuntimeError("FIM discarded-tail processed token budget exhausted")
    check_shared_token_budget(arm, processed)
    if directory_bytes(ARTIFACTS) > budget["new_artifact_bytes_cap"]:
        raise OSError("FIM retrieval exceeded the artifact cap")
    record = {
        "reference": job["reference"],
        "status": status,
        "arm": arm,
        "attempt": attempt,
        "plan_sha256": job["plan_sha256"],
        "input_manifest_sha256": job["input_manifest_sha256"],
        "commit": job["commit"],
        "checkpoint_verified": True,
        "checkpoint_sha256": marker["sha256"],
        "fingerprint": marker["fingerprint"],
        "cursor": pointer["cursor"],
        "training_status": result["status"],
        "checkpoint_path": str(checkpoint),
        "discarded_logged_tail_tokens": recorded - committed,
        "unlogged_inflight_input_token_reservation": unlogged,
        "truncated_last_update_record": truncated,
        "own_discarded_tokens_for_resume": own_discarded,
        "processed_arm_input_tokens_conservative": processed,
        "other_phase_processed_tokens_at_submission": job["other_phase_tokens"],
        "observed_at": datetime.now(UTC).isoformat(),
    }
    save(REPORT / f"fim-verified-{arm}-{attempt}.json", record)
    return record


def collect_fim_export(plan: dict[str, Any], arm: str, attempt: int) -> Path:
    """Retrieve one completed arm for a declared downstream quality comparison."""
    if arm not in FIM_ARMS:
        raise ValueError("unknown FIM arm")
    verified = json.loads((REPORT / f"fim-verified-{arm}-{attempt}.json").read_text())
    if (
        verified.get("arm") != arm
        or not verified.get("checkpoint_verified")
        or verified.get("training_status") != "complete"
        or verified["cursor"]["training_input_tokens"] != plan["data"]["train"]["input_tokens"]
    ):
        raise ValueError("FIM export requires a verified completed matched pass")
    output = ARTIFACTS / f"fim/output-{arm}-{attempt}"
    candidates = list(output.glob("**/inference-f16/artifact_manifest.json"))
    if len(candidates) != 1:
        raise ValueError("exactly one completed FIM export manifest is required")
    manifest = json.loads(candidates[0].read_text())
    if (
        manifest.get("schema") != "q25-fim-inference-f16-v1"
        or manifest.get("arm") != arm
        or manifest.get("fingerprint") != verified["fingerprint"]
        or manifest.get("training_cursor") != verified["cursor"]
    ):
        raise ValueError("FIM export does not represent the verified completed pass")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise ValueError("FIM export lacks a weight inventory")
    export = candidates[0].parent
    missing_bytes = 0
    for name, record in files.items():
        if (
            Path(name).name != name
            or not isinstance(record, dict)
            or type(record.get("bytes")) is not int
            or record["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256")))
            or (export / name).is_symlink()
        ):
            raise ValueError("FIM export contains an unsafe file record")
        if not (export / name).exists():
            missing_bytes += record["bytes"]
    budget = plan["configuration"]["budget"]
    if directory_bytes(ARTIFACTS) + missing_bytes > budget["new_artifact_bytes_cap"] or (
        shutil.disk_usage(ARTIFACTS).free < missing_bytes + budget["minimum_free_bytes"]
    ):
        raise OSError("FIM export retrieval exceeds storage headroom")
    for name, record in files.items():
        path = export / name
        if not path.exists():
            cli(
                "kaggle",
                "kernels",
                "output",
                verified["reference"],
                "-p",
                str(output),
                "-q",
                "--file-pattern",
                r"inference-f16/" + re.escape(name) + "$",
                timeout=900,
            )
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("retrieved FIM export file identity differs")
    save(
        REPORT / f"fim-export-verification-{arm}-{attempt}.json",
        {
            "reference": verified["reference"],
            "path": str(export),
            "manifest_sha256": digest(candidates[0]),
            "fingerprint": verified["fingerprint"],
            "training_cursor": verified["cursor"],
            "files": files,
            "automatic_editor_promotion": False,
        },
    )
    return export


def _conversion_relpath(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a normalized relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} must stay inside its selected root")
    return Path(*relative.parts)


def _conversion_file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("conversion input contains a missing or unsafe file")
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def _resolve_original_model_json(name: str, expected_sha256: str) -> Path:
    candidate = MODEL / name
    if candidate.is_symlink():
        blob_root = MODEL.parent.parent / "blobs"
        if blob_root.is_symlink() or not blob_root.is_dir():
            raise ValueError("pinned Hugging Face snapshot has no regular blob directory")
        allowed_root = blob_root.resolve(strict=True)
    else:
        allowed_root = MODEL.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or not resolved.is_relative_to(allowed_root)
        or digest(resolved) != expected_sha256
    ):
        raise ValueError("pinned original model JSON is outside its snapshot or has changed")
    return resolved


def _conversion_tree_records(root: Path, prefixes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        directory = root / prefix
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("selected FIM run is missing complete evaluation evidence")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("selected FIM evidence contains a symbolic link")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                records[relative] = _conversion_file_record(path)
    return records


def _expected_fim_cursor(training_plan: dict[str, Any]) -> dict[str, int]:
    training = training_plan["configuration"]["training"]
    train = training_plan["data"]["train"]
    row_count = train.get("row_count")
    batch = training.get("effective_batch")
    input_tokens = train.get("input_tokens")
    target_tokens = train.get("target_tokens")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (row_count, batch, input_tokens, target_tokens)
    ):
        raise ValueError("frozen FIM plan has invalid full-pass totals")
    updates = (row_count + batch - 1) // batch
    return {
        "attempted_updates": updates,
        "completed_updates": updates,
        "skipped_updates": 0,
        "training_input_tokens": input_tokens,
        "supervised_target_tokens": target_tokens,
        "next_example_index": row_count,
        "epoch": 1,
    }


def _find_conversion_source_job(
    source_reference: str, selected_arm: str
) -> tuple[Path, dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", source_reference):
        raise ValueError("selection source kernel reference is invalid")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for arm in FIM_ARMS:
        for path in REPORT.glob(f"fim-job-{arm}-*.json"):
            job = json.loads(path.read_text())
            if job.get("reference") == source_reference:
                if arm != selected_arm or job.get("arm") != selected_arm:
                    raise ValueError("selection source reference belongs to another FIM arm")
                matches.append((path, job))
    if len(matches) != 1:
        raise ValueError("selection must identify exactly one completed FIM kernel")
    return matches[0]


def _verify_conversion_source(selection_path: Path) -> dict[str, Any]:
    if selection_path.is_symlink() or not selection_path.is_file():
        raise ValueError("manual conversion selection is missing or unsafe")
    selection_abs = selection_path.resolve(strict=True)
    if not selection_abs.is_relative_to(REPORT.resolve(strict=True)):
        raise ValueError("manual conversion selection must be inside the research report")
    selection = json.loads(selection_abs.read_text())
    training_plan_path = REPORT / "fim_training_plan.json"
    training_plan = json.loads(training_plan_path.read_text())
    plan_sha = digest(training_plan_path)
    selection_sha = digest(selection_abs)
    if (
        selection.get("schema") != "q25-fim-conversion-selection-v1"
        or selection.get("status") != "selected_complete"
        or selection.get("training_plan_sha256") != plan_sha
        or selection.get("selected_arm") not in FIM_ARMS
    ):
        raise ValueError("conversion requires the root-selected complete frozen FIM arm")
    quality_sha = selection.get("paired_quality_report_sha256")
    if (
        not CONVERSION_QUALITY_REPORT.is_file()
        or CONVERSION_QUALITY_REPORT.is_symlink()
        or not re.fullmatch(r"[0-9a-f]{64}", str(quality_sha))
        or digest(CONVERSION_QUALITY_REPORT) != quality_sha
    ):
        raise ValueError("manual selection is not bound to the paired quality report")
    arm = selection["selected_arm"]
    source_ref = selection.get("source_kernel_reference")
    source_export = selection.get("source_export")
    if not isinstance(source_export, dict):
        raise ValueError("selection does not bind an exported FIM initializer")
    if source_export.get("directory") != CONVERSION_EXPORT_DIRECTORY:
        raise ValueError("selection export path differs from the frozen kernel output layout")
    if not re.fullmatch(r"[0-9a-f]{64}", str(source_export.get("artifact_manifest_sha256"))):
        raise ValueError("selection export manifest identity is invalid")
    job_path, job = _find_conversion_source_job(source_ref, arm)
    attempt = job.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("selected FIM attempt identity is invalid")
    verified_path = REPORT / f"fim-verified-{arm}-{attempt}.json"
    verified = json.loads(verified_path.read_text())
    if (
        job.get("plan_sha256") != plan_sha
        or verified.get("reference") != source_ref
        or verified.get("arm") != arm
        or verified.get("attempt") != attempt
        or verified.get("plan_sha256") != plan_sha
        or verified.get("input_manifest_sha256") != job.get("input_manifest_sha256")
        or verified.get("commit") != job.get("commit")
        or verified.get("training_status") != "complete"
        or verified.get("checkpoint_verified") is not True
    ):
        raise ValueError("selected FIM kernel lacks a matching complete verified receipt")
    expected_cursor = _expected_fim_cursor(training_plan)
    if (
        verified.get("cursor") != expected_cursor
        or source_export.get("training_cursor") != expected_cursor
    ):
        raise ValueError("selected FIM kernel did not complete the exact planned pass")
    if (
        verified.get("cursor", {}).get("completed_updates") != 256
        or verified.get("cursor", {}).get("training_input_tokens") != 1_776_908
    ):
        raise ValueError("selected FIM export is not the frozen 256-update pass")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(verified.get("fingerprint")))
        or source_export.get("fingerprint") != verified.get("fingerprint")
        or not re.fullmatch(r"[0-9a-f]{64}", str(verified.get("checkpoint_sha256")))
    ):
        raise ValueError("selected FIM checkpoint/export fingerprint is invalid")

    output_root = ARTIFACTS / f"fim/output-{arm}-{attempt}"
    export = output_root / CONVERSION_EXPORT_DIRECTORY
    manifest_path = export / "artifact_manifest.json"
    if export.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("selected FIM export is missing or unsafe")
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = digest(manifest_path)
    files = manifest.get("files")
    if (
        manifest_sha != source_export["artifact_manifest_sha256"]
        or manifest.get("schema") != "q25-fim-inference-f16-v1"
        or manifest.get("arm") != arm
        or manifest.get("fingerprint") != verified["fingerprint"]
        or manifest.get("training_cursor") != expected_cursor
        or not isinstance(files, dict)
        or source_export.get("files") != files
    ):
        raise ValueError("selected FIM artifact manifest differs from its frozen selection")
    if not isinstance(files, dict) or not files:
        raise ValueError("selected FIM artifact manifest has no model file map")
    required_local_files = {"config.json", "tokenizer.json"}
    for name, record in files.items():
        relative = _conversion_relpath(name, label="selected export filename")
        path = export / relative
        expected_bytes = record.get("bytes") if isinstance(record, dict) else None
        expected_sha = record.get("sha256") if isinstance(record, dict) else None
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha))
        ):
            raise ValueError("selected FIM export file record is invalid")
        if path.is_symlink():
            raise ValueError("selected FIM export contains a symbolic link")
        if path.exists():
            if (
                not path.is_file()
                or path.stat().st_size != expected_bytes
                or digest(path) != expected_sha
            ):
                raise ValueError("locally collected FIM export file differs from its manifest")
        elif name in required_local_files:
            raise ValueError("selected FIM export lacks its local config or tokenizer")
        if name == "model.safetensors" and path.exists():
            raise ValueError("conversion freeze must not retrieve the source F16 weights")
    export_paths = list(export.rglob("*"))
    if any(path.is_symlink() for path in export_paths):
        raise ValueError("selected FIM export contains a symbolic link")
    actual_export_files = {
        path.relative_to(export).as_posix()
        for path in export_paths
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if not required_local_files.issubset(actual_export_files) or not actual_export_files.issubset(
        files
    ):
        raise ValueError("local FIM export files are missing or unmanifested")

    worker_path = output_root / "q25_fim_r2/worker-status.json"
    result_path = output_root / "q25_fim_r2/training/run_result.json"
    pointer_path = output_root / "q25_fim_r2/training/latest.json"
    checkpoint_marker_path: Path | None = None
    worker = json.loads(worker_path.read_text())
    result = json.loads(result_path.read_text())
    pointer = json.loads(pointer_path.read_text())
    checkpoint_name = pointer.get("path")
    if not isinstance(checkpoint_name, str) or not re.fullmatch(
        r"resume-step-[0-9]+\.pt", checkpoint_name
    ):
        raise ValueError("selected full-pass checkpoint pointer is invalid")
    checkpoint = pointer_path.parent / checkpoint_name
    checkpoint_marker_path = pointer_path.parent / f"{checkpoint_name}.complete.json"
    marker = json.loads(checkpoint_marker_path.read_text())
    if (
        worker.get("schema") != "q25-fim-kaggle-worker-status-v1"
        or worker.get("reference") not in (None, source_ref)
        or worker.get("commit") != job.get("commit")
        or worker.get("attempt") != attempt
        or worker.get("arm") != arm
        or worker.get("plan_sha256") != plan_sha
        or worker.get("input_manifest_sha256") != job.get("input_manifest_sha256")
        or worker.get("training_started") is not True
        or worker.get("state") != "complete"
        or worker.get("training", {}).get("status") != "complete"
        or worker.get("training", {}).get("cursor") != expected_cursor
        or result.get("status") != "complete"
        or result.get("arm") != arm
        or result.get("fingerprint") != verified["fingerprint"]
        or result.get("cursor") != expected_cursor
        or pointer.get("fingerprint") != verified["fingerprint"]
        or pointer.get("cursor") != expected_cursor
        or marker.get("version") != 1
        or marker.get("fingerprint") != verified["fingerprint"]
        or marker.get("sha256") != verified.get("checkpoint_sha256")
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or digest(checkpoint) != marker.get("sha256")
    ):
        raise ValueError("selected FIM worker, result, or checkpoint identity is inconsistent")
    if result.get("logical_training_input_tokens") != expected_cursor["training_input_tokens"]:
        raise ValueError("selected FIM result token cursor differs from the full pass")

    run_root = output_root / "q25_fim_r2"
    baseline_path = run_root / "baseline/identity.json"
    baseline = json.loads(baseline_path.read_text())
    if (
        baseline.get("schema") != "q25-fim-baseline-v1"
        or baseline.get("arm") != arm
        or baseline.get("plan_sha256") != plan_sha
        or baseline.get("input_manifest_sha256") != job.get("input_manifest_sha256")
        or not isinstance(baseline.get("files"), dict)
        or not baseline["files"]
    ):
        raise ValueError("selected FIM kernel lacks frozen before-run evaluation evidence")

    required_results = (
        "before/development/summary.json",
        "before/line/summary.json",
        "after/development/results.jsonl",
        "after/development/summary.json",
        "after/line/results.jsonl",
        "after/line/summary.json",
        "regression/causal/predictions.jsonl",
        "regression/causal/predictions.jsonl.metadata.json",
        "regression/line/predictions.jsonl",
        "regression/line/predictions.jsonl.metadata.json",
    )
    evidence_prefixes = ("baseline", "before", "after", "regression")
    evidence_files = _conversion_tree_records(run_root, evidence_prefixes)
    for relative_result in required_results:
        if relative_result not in evidence_files:
            raise ValueError("selected FIM kernel lacks a required paired evaluation result")
    for relative_summary, expected_cases in (
        ("after/development/summary.json", 240),
        ("after/line/summary.json", 180),
        ("before/development/summary.json", 240),
        ("before/line/summary.json", 180),
    ):
        summary = json.loads((run_root / relative_summary).read_text())
        if summary.get("cases") != expected_cases or summary.get("plan_sha256") != plan_sha:
            raise ValueError("selected FIM evaluation result differs from its frozen suite")

    job_sha = digest(job_path)
    verified_sha = digest(verified_path)
    evidence = {
        "job_sha256": job_sha,
        "verified_receipt_sha256": verified_sha,
        "worker_status_sha256": digest(worker_path),
        "run_result_sha256": digest(result_path),
        "checkpoint_pointer_sha256": digest(pointer_path),
        "checkpoint_marker_sha256": digest(checkpoint_marker_path),
        "checkpoint_sha256": digest(checkpoint),
        "export_manifest_sha256": manifest_sha,
        "evaluation_files": evidence_files,
    }
    return {
        "selection": selection,
        "selection_path": selection_abs.relative_to(ROOT).as_posix(),
        "selection_sha256": selection_sha,
        "training_plan": training_plan,
        "training_plan_sha256": plan_sha,
        "source": {
            "arm": arm,
            "attempt": attempt,
            "kernel_reference": source_ref,
            "export_directory": CONVERSION_EXPORT_DIRECTORY,
            "artifact_manifest_sha256": manifest_sha,
            "fingerprint": verified["fingerprint"],
            "training_cursor": expected_cursor,
            "files": files,
            "input_manifest_sha256": job["input_manifest_sha256"],
            "commit": job["commit"],
            "paired_quality_report_sha256": quality_sha,
        },
        "source_evidence": evidence,
        "source_output_root": output_root,
    }


def _conversion_code_files() -> dict[str, Path]:
    return {
        "conversion_worker.py": ROOT / "kaggle/q25_fim_conversion_r1/run.py",
        "src/tinycomplete/__init__.py": ROOT / "src/tinycomplete/__init__.py",
        "src/tinycomplete/code_cpt/__init__.py": ROOT / "src/tinycomplete/code_cpt/__init__.py",
        "src/tinycomplete/code_cpt/q25_fim_conversion.py": ROOT
        / "src/tinycomplete/code_cpt/q25_fim_conversion.py",
        "kaggle/q25_fim_conversion_r1/requirements-conversion.lock": ROOT
        / "kaggle/q25_fim_conversion_r1/requirements-conversion.lock",
    }


def freeze_fim_conversion(selection_path: Path = CONVERSION_SELECTION) -> dict[str, Any]:
    """Freeze only a manual complete-arm selection and its verified provenance."""
    source = _verify_conversion_source(selection_path)
    original_config = MODEL / "config.json"
    original_tokenizer = MODEL / "tokenizer.json"
    if not original_config.is_file() or not original_tokenizer.is_file():
        raise FileNotFoundError("pinned original Qwen config/tokenizer files are unavailable")
    untouched = source["training_plan"]["initializers"]["untouched_q25_to_fim"]
    expected_original = untouched["files"]
    if (
        digest(original_config) != expected_original["config.json"]["sha256"]
        or digest(original_tokenizer) != expected_original["tokenizer.json"]["sha256"]
        or digest(original_tokenizer) != source["selection"]["tokenizer"]["sha256"]
    ):
        raise ValueError("original Qwen config/tokenizer differs from the FIM provenance")
    code_files = _conversion_code_files()
    missing = [name for name, path in code_files.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError("pinned CPU conversion source files are incomplete")
    plan = {
        "schema": "q25-fim-q4-conversion-plan-v1",
        "plan_revision": 1,
        "selection_path": source["selection_path"],
        "selection_sha256": source["selection_sha256"],
        "paired_quality_report_sha256": source["source"]["paired_quality_report_sha256"],
        "training_plan_sha256": source["training_plan_sha256"],
        "source": source["source"],
        "source_evidence": source["source_evidence"],
        "original_q25_config_sha256": digest(original_config),
        "original_q25_tokenizer_sha256": digest(original_tokenizer),
        "source_code": {name: digest(path) for name, path in code_files.items()},
        "execution": {
            "session_seconds": CONVERSION_SESSION_SECONDS,
            "finalization_reserve_seconds": CONVERSION_FINALIZATION_RESERVE_SECONDS,
            "enable_gpu": False,
            "enable_internet": True,
            "paid_compute": False,
            "automatic_renewal_use": False,
            "artifact_bytes_cap": CONVERSION_ARTIFACT_CAP_BYTES,
            "minimum_free_bytes": CONVERSION_MINIMUM_FREE_BYTES,
        },
    }
    if CONVERSION_PLAN.exists():
        frozen = json.loads(CONVERSION_PLAN.read_text())
        if frozen != plan:
            raise ValueError("frozen Q25 FIM conversion plan differs; create a new revision")
        return frozen
    save(CONVERSION_PLAN, plan)
    return plan


def _load_conversion_plan() -> dict[str, Any]:
    plan = json.loads(CONVERSION_PLAN.read_text())
    execution = plan.get("execution", {})
    if (
        plan.get("schema") != "q25-fim-q4-conversion-plan-v1"
        or plan.get("plan_revision") != 1
        or not isinstance(execution, dict)
        or execution.get("session_seconds") != CONVERSION_SESSION_SECONDS
        or execution.get("finalization_reserve_seconds") != CONVERSION_FINALIZATION_RESERVE_SECONDS
        or execution.get("enable_gpu") is not False
        or execution.get("enable_internet") is not True
        or execution.get("paid_compute") is not False
        or execution.get("automatic_renewal_use") is not False
        or execution.get("artifact_bytes_cap") != CONVERSION_ARTIFACT_CAP_BYTES
        or execution.get("minimum_free_bytes") != CONVERSION_MINIMUM_FREE_BYTES
    ):
        raise ValueError("frozen CPU conversion plan is invalid")
    return plan


def build_fim_conversion_bundle(plan: dict[str, Any] | None = None) -> Path:
    plan = _load_conversion_plan() if plan is None else plan
    if plan != _load_conversion_plan():
        raise ValueError("conversion bundle plan differs from its frozen report")
    selection_path = ROOT / plan["selection_path"]
    if digest(selection_path) != plan["selection_sha256"]:
        raise ValueError("manual conversion selection changed after freeze")
    training_plan_path = REPORT / "fim_training_plan.json"
    if digest(training_plan_path) != plan["training_plan_sha256"]:
        raise ValueError("FIM training plan changed after conversion freeze")
    sources = {
        "selection.json": selection_path,
        "training_plan.json": training_plan_path,
        "original_config.json": _resolve_original_model_json(
            "config.json", plan["original_q25_config_sha256"]
        ),
        "original_tokenizer.json": _resolve_original_model_json(
            "tokenizer.json", plan["original_q25_tokenizer_sha256"]
        ),
    }
    allowed = set(sources) | {"input-manifest.json", "dataset-metadata.json"}
    output = CONVERSION_INPUT_BUNDLE
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name not in allowed or path.is_symlink() for path in output.iterdir()):
        raise ValueError("conversion input staging contains an unapproved artifact")
    projected = sum(
        path.stat().st_size for name, path in sources.items() if not (output / name).exists()
    )
    if directory_bytes(ARTIFACTS) + projected > CONVERSION_ARTIFACT_CAP_BYTES:
        raise OSError("conversion input staging exceeds the artifact cap")
    if shutil.disk_usage(ARTIFACTS).free < projected + CONVERSION_MINIMUM_FREE_BYTES:
        raise OSError("conversion input staging would exhaust storage headroom")
    file_records: dict[str, dict[str, Any]] = {}
    for name, source in sources.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError("conversion input source is missing or unsafe")
        target = output / name
        if target.exists():
            if target.is_symlink() or not target.is_file() or digest(target) != digest(source):
                raise ValueError("immutable conversion input differs from its frozen source")
        else:
            shutil.copy2(source, target)
        record = _conversion_file_record(target)
        expected_source_hash = {
            "original_config.json": plan["original_q25_config_sha256"],
            "original_tokenizer.json": plan["original_q25_tokenizer_sha256"],
        }.get(name)
        if expected_source_hash is not None and record["sha256"] != expected_source_hash:
            raise ValueError("staged original Qwen JSON differs from its frozen hash")
        file_records[name] = record
    manifest_value = {
        "schema": "q25-fim-conversion-input-manifest-v1",
        "files": file_records,
        "selection_sha256": plan["selection_sha256"],
        "training_plan_sha256": plan["training_plan_sha256"],
        "selected_arm": plan["source"]["arm"],
        "source_kernel_reference": plan["source"]["kernel_reference"],
        "source_export_directory": plan["source"]["export_directory"],
        "source_export_manifest_sha256": plan["source"]["artifact_manifest_sha256"],
    }
    manifest_path = output / "input-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest_value:
            raise ValueError("immutable conversion input manifest differs")
    else:
        save(manifest_path, manifest_value)
    metadata = {
        "id": CONVERSION_DATASET,
        "title": "TabComplete Q25 selected FIM conversion config r1",
        "licenses": [{"name": "other"}],
    }
    metadata_path = output / "dataset-metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("conversion dataset metadata differs from its frozen identity")
    else:
        save(metadata_path, metadata)
    if directory_bytes(ARTIFACTS) > CONVERSION_ARTIFACT_CAP_BYTES:
        raise OSError("conversion input bundle exceeds the artifact cap")
    if shutil.disk_usage(ARTIFACTS).free < CONVERSION_MINIMUM_FREE_BYTES:
        raise OSError("conversion input bundle exhausted storage headroom")
    return output


def upload_fim_conversion_bundle(plan: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _csv_refs, wait_for_remote_inputs

    output = build_fim_conversion_bundle(plan)
    marker = CONVERSION_SUBMISSION_FILE
    manifest = json.loads((output / "input-manifest.json").read_text())
    expected = dict(manifest["files"])
    expected["input-manifest.json"] = {"bytes": (output / "input-manifest.json").stat().st_size}
    expected["dataset-metadata.json"] = {"bytes": (output / "dataset-metadata.json").stat().st_size}
    if not marker.exists():
        refs = _csv_refs(["kaggle", "datasets", "list", "--mine", "--page-size", "100", "--csv"])
        if CONVERSION_DATASET in refs:
            raise ValueError("conversion dataset exists without this campaign's receipt")
        response = cli(
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(output),
            "-t",
            "--dir-mode",
            "zip",
            timeout=900,
        )
        if "Your private Dataset is being created." not in response:
            raise RuntimeError("Kaggle did not confirm private conversion dataset creation")
        save(
            marker,
            {
                "dataset": CONVERSION_DATASET,
                "state": "created",
                "input_manifest_sha256": digest(output / "input-manifest.json"),
            },
        )
    recorded = json.loads(marker.read_text())
    if recorded.get("dataset") != CONVERSION_DATASET or recorded.get(
        "input_manifest_sha256"
    ) != digest(output / "input-manifest.json"):
        raise ValueError("uploaded conversion dataset differs from its frozen input manifest")
    verification = wait_for_remote_inputs(CONVERSION_DATASET, expected)
    save(marker, {**recorded, "state": "verified", "remote_verification": verification})
    return verification


def _conversion_launcher(session: dict[str, Any]) -> str:
    embedded = repr(json.dumps(session, sort_keys=True, separators=(",", ":")))
    template = r"""from __future__ import annotations
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path, PurePosixPath

SESSION = json.loads(__SESSION_JSON__)
INPUT_ROOT = Path("/kaggle/input")
CODE_ROOT = Path(__file__).resolve().parent

def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def fail():
    raise SystemExit("conversion input identity is missing or ambiguous")

def direct_mounts():
    if INPUT_ROOT.is_symlink() or not INPUT_ROOT.is_dir():
        fail()
    values = []
    for path in INPUT_ROOT.iterdir():
        if path.is_symlink():
            continue
        if path.is_dir():
            values.append(path.resolve(strict=True))
    return values

mounts = direct_mounts()
config_mounts = []
for mount in mounts:
    manifest = mount / "input-manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        continue
    if sha(manifest) == SESSION["input_manifest_sha256"]:
        config_mounts.append(mount)
if len(config_mounts) != 1:
    fail()
config_root = config_mounts[0]
manifest = json.loads((config_root / "input-manifest.json").read_text(encoding="utf-8"))
if (manifest.get("schema") != "q25-fim-conversion-input-manifest-v1"
    or manifest.get("selection_sha256") != SESSION["selection_sha256"]
    or manifest.get("training_plan_sha256") != SESSION["training_plan_sha256"]
    or manifest.get("selected_arm") != SESSION["selected_arm"]
    or manifest.get("source_kernel_reference") != SESSION["source_kernel_reference"]
    or manifest.get("source_export_manifest_sha256") != SESSION["source_export_manifest_sha256"]):
    fail()
files = manifest.get("files")
if not isinstance(files, dict) or set(files) != {
    "selection.json", "training_plan.json", "original_config.json", "original_tokenizer.json"
}:
    fail()
for name, record in files.items():
    relative = PurePosixPath(name)
    path = config_root.joinpath(*relative.parts)
    if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
        or not path.is_file() or path.stat().st_size != record.get("bytes")
        or sha(path) != record.get("sha256")):
        fail()
if sha(config_root / "selection.json") != SESSION["selection_sha256"]:
    fail()
if sha(config_root / "training_plan.json") != SESSION["training_plan_sha256"]:
    fail()
selection = json.loads((config_root / "selection.json").read_text(encoding="utf-8"))
if (selection.get("source_kernel_reference") != SESSION["source_kernel_reference"]
    or selection.get("selected_arm") != SESSION["selected_arm"]
    or selection.get("source_export", {}).get("directory") != SESSION["source_export_directory"]
    or selection.get("source_export", {}).get("artifact_manifest_sha256")
       != SESSION["source_export_manifest_sha256"]):
    fail()
relative_export = PurePosixPath(SESSION["source_export_directory"])
if relative_export.is_absolute() or ".." in relative_export.parts:
    fail()
source_mounts = []
for mount in mounts:
    if mount == config_root:
        continue
    export = mount.joinpath(*relative_export.parts)
    manifest_path = export / "artifact_manifest.json"
    if not manifest_path.exists():
        continue
    if any(path.is_symlink() for path in (mount, export, manifest_path)):
        fail()
    if manifest_path.is_file() and sha(manifest_path) == SESSION["source_export_manifest_sha256"]:
        source_mounts.append(mount)
if len(source_mounts) != 1:
    fail()
source_root = source_mounts[0]
worker = CODE_ROOT / "conversion_worker.py"
if worker.is_symlink() or not worker.is_file():
    fail()
sys.argv = [str(worker),
    "--input-root", str(INPUT_ROOT),
    "--selection", str(config_root / "selection.json"),
    "--training-plan", str(config_root / "training_plan.json"),
    "--expected-selection-sha256", SESSION["selection_sha256"],
    "--code-root", str(CODE_ROOT),
    "--source-root", str(source_root),
    "--source-kernel-reference", SESSION["source_kernel_reference"],
    "--output-root", "/kaggle/working/q25_fim_conversion_r1",
    "--runtime-root", "/kaggle/temp/q25_fim_conversion_r1",
    "--session-seconds", str(SESSION["session_seconds"]),
    "--reserve-seconds", str(SESSION["finalization_reserve_seconds"])]
runpy.run_path(str(worker), run_name="__main__")
"""
    return template.replace("__SESSION_JSON__", embedded)


def build_fim_conversion_kernel(plan: dict[str, Any], commit: str) -> Path:
    bundle = build_fim_conversion_bundle(plan)
    input_manifest_sha = digest(bundle / "input-manifest.json")
    kernel = CONVERSION_KERNEL_ROOT
    kernel.mkdir(parents=True, exist_ok=True)
    code_files = _conversion_code_files()
    allowed_paths = {"run.py", "kernel-metadata.json"} | set(code_files)
    for existing in kernel.rglob("*"):
        if existing.is_symlink():
            raise ValueError("conversion kernel staging contains a symbolic link")
        if existing.is_file() and existing.relative_to(kernel).as_posix() not in allowed_paths:
            raise ValueError("conversion kernel staging contains an unapproved file")
    session = {
        "commit": commit,
        "plan_sha256": digest(CONVERSION_PLAN),
        "input_manifest_sha256": input_manifest_sha,
        "selection_sha256": plan["selection_sha256"],
        "training_plan_sha256": plan["training_plan_sha256"],
        "selected_arm": plan["source"]["arm"],
        "source_kernel_reference": plan["source"]["kernel_reference"],
        "source_export_directory": plan["source"]["export_directory"],
        "source_export_manifest_sha256": plan["source"]["artifact_manifest_sha256"],
        "session_seconds": CONVERSION_SESSION_SECONDS,
        "finalization_reserve_seconds": CONVERSION_FINALIZATION_RESERVE_SECONDS,
    }
    for relative, source in code_files.items():
        target = kernel / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file() or digest(target) != digest(source):
                raise ValueError("staged conversion source differs from the frozen code")
        else:
            shutil.copy2(source, target)
    launcher = _conversion_launcher(session)
    run_path = kernel / "run.py"
    if run_path.exists() and run_path.read_text() != launcher:
        raise ValueError("frozen conversion launcher differs from its input receipt")
    if not run_path.exists():
        run_path.write_text(launcher)
    metadata = {
        "id": CONVERSION_KERNEL_REFERENCE,
        "title": "tc q25 fim q4 conversion r1",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [CONVERSION_DATASET],
        "kernel_sources": [plan["source"]["kernel_reference"]],
    }
    metadata_path = kernel / "kernel-metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("conversion kernel metadata differs from the frozen selection")
    else:
        save(metadata_path, metadata)
    if directory_bytes(ARTIFACTS) > CONVERSION_ARTIFACT_CAP_BYTES:
        raise OSError("conversion kernel bundle exceeds the artifact cap")
    if shutil.disk_usage(ARTIFACTS).free < CONVERSION_MINIMUM_FREE_BYTES:
        raise OSError("conversion kernel bundle exhausted storage headroom")
    return kernel


def _check_conversion_quota(plan: dict[str, Any], observation: dict[str, Any]) -> None:
    execution = plan["execution"]
    if (
        execution.get("enable_gpu") is not False
        or execution.get("paid_compute") is not False
        or execution.get("automatic_renewal_use") is not False
    ):
        raise ValueError("CPU conversion must not enable GPU, paid compute, or renewal use")
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _job_statuses_verified

    if observation.get("active_jobs") or not _job_statuses_verified(observation):
        raise RuntimeError("another Kaggle job is active or its status is unresolved")
    if observation.get("units") != "Kaggle account GPU-hours" or not observation.get("source"):
        raise ValueError("live quota observation lacks units or provenance")
    if len(observation.get("job_statuses", [])) >= 100:
        raise RuntimeError("job listing may be truncated; verify pagination before allocation")
    budget = json.loads((REPORT / "campaign_budget.json").read_text())["shared_limits"]
    if observation.get("renewal") != budget.get("quota_renewal"):
        raise RuntimeError("quota renewal differs from the frozen campaign budget")
    renewal = datetime.fromisoformat(str(observation["renewal"]).replace("Z", "+00:00"))
    if renewal.tzinfo is None:
        renewal = renewal.replace(tzinfo=UTC)
    if datetime.now(UTC) + timedelta(seconds=CONVERSION_SESSION_SECONDS) >= renewal:
        raise RuntimeError("CPU conversion session would cross the authorized quota renewal")
    remaining = observation.get("remaining")
    if not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or remaining < 0:
        raise RuntimeError("live account quota is unavailable or invalid")


def submit_fim_conversion(plan: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = _load_conversion_plan() if plan is None else plan
    if plan != _load_conversion_plan():
        raise ValueError("conversion plan differs from its frozen report")
    if CONVERSION_JOB_FILE.exists():
        raise ValueError("conversion job already has a submission receipt; inspect it instead")
    if cli("git", "-C", str(ROOT), "branch", "--show-current") != CONVERSION_BRANCH:
        raise ValueError("conversion source is not on its frozen branch")
    if cli("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit all conversion source files before submitting")
    commit = cli("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("conversion source commit is invalid")
    remote = cli("git", "-C", str(ROOT), "ls-remote", "origin", f"refs/heads/{CONVERSION_BRANCH}")
    if not remote or remote.split()[0] != commit:
        raise RuntimeError("push the exact frozen conversion source commit before submitting")
    code_files = _conversion_code_files()
    if set(plan["source_code"]) != set(code_files) or any(
        path.is_symlink() or not path.is_file() or digest(path) != plan["source_code"][name]
        for name, path in code_files.items()
    ):
        raise ValueError("conversion source files changed after plan freeze")
    submission = json.loads(CONVERSION_SUBMISSION_FILE.read_text())
    bundle = build_fim_conversion_bundle(plan)
    if (
        submission.get("state") != "verified"
        or submission.get("dataset") != CONVERSION_DATASET
        or submission.get("input_manifest_sha256") != digest(bundle / "input-manifest.json")
    ):
        raise ValueError("private CPU conversion inputs are not uploaded and verified")
    existing = list(REPORT.glob("fim-conversion-job*.json"))
    if existing:
        raise ValueError("conversion allocation already exists; automatic retry is disabled")
    check_shared_allocation_budget(CONVERSION_SESSION_SECONDS, phase="conversion")
    observation = quota()
    _check_conversion_quota(plan, observation)
    if _csv_ref_exists(CONVERSION_KERNEL_REFERENCE):
        raise ValueError(
            "conversion kernel reference already exists; refusing duplicate allocation"
        )
    source_status = cli("kaggle", "kernels", "status", plan["source"]["kernel_reference"])
    if "COMPLETE" not in source_status.upper():
        raise RuntimeError("selected source FIM kernel is not complete")
    kernel = build_fim_conversion_kernel(plan, commit)
    job = {
        "reference": CONVERSION_KERNEL_REFERENCE,
        "source_kernel_reference": plan["source"]["kernel_reference"],
        "selected_arm": plan["source"]["arm"],
        "attempt": 1,
        "plan_sha256": digest(CONVERSION_PLAN),
        "selection_sha256": plan["selection_sha256"],
        "training_plan_sha256": plan["training_plan_sha256"],
        "input_manifest_sha256": digest(bundle / "input-manifest.json"),
        "commit": commit,
        "enable_gpu": False,
        "session_seconds": CONVERSION_SESSION_SECONDS,
        "conservative_reserved_session_seconds": CONVERSION_SESSION_SECONDS,
        "account_gpu_hours_reserved": 0,
        "paid_compute": False,
        "automatic_renewal_use": False,
        "quota": observation,
        "status": "submission_pending",
        "submitted_at": datetime.now(UTC).isoformat(),
        "automatic_allocation": False,
    }
    save(REPORT / "fim-conversion-quota.json", observation)
    save(CONVERSION_JOB_FILE, job)
    try:
        response = cli(
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(kernel),
            "--timeout",
            str(CONVERSION_SESSION_SECONDS),
            timeout=240,
        )
        urls = re.findall(
            r"https://www\.kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", response
        )
        job.update(
            status="submitted",
            submission_response=response,
            requested_reference=CONVERSION_KERNEL_REFERENCE,
            reference=urls[-1] if urls else CONVERSION_KERNEL_REFERENCE,
        )
    except Exception as exc:
        job.update(status="submission_unknown", error_class=type(exc).__name__)
        save(CONVERSION_JOB_FILE, job)
        raise
    save(CONVERSION_JOB_FILE, job)
    return job


def _csv_ref_exists(reference: str) -> bool:
    sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
    from build_pilot import _csv_refs

    return reference in _csv_refs(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"]
    )


def collect_fim_conversion(plan: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = _load_conversion_plan() if plan is None else plan
    if plan != _load_conversion_plan() or not CONVERSION_JOB_FILE.is_file():
        raise ValueError("conversion collection lacks its frozen job receipt")
    job = json.loads(CONVERSION_JOB_FILE.read_text())
    if (
        job.get("plan_sha256") != digest(CONVERSION_PLAN)
        or job.get("selection_sha256") != plan["selection_sha256"]
        or job.get("source_kernel_reference") != plan["source"]["kernel_reference"]
        or job.get("enable_gpu") is not False
        or job.get("automatic_allocation") is not False
    ):
        raise ValueError("conversion job belongs to another plan or source kernel")
    status = cli("kaggle", "kernels", "status", job["reference"])
    if not any(word in status.upper() for word in ("COMPLETE", "ERROR")):
        return {"status": status, "conversion_verified": False}
    if "COMPLETE" not in status.upper():
        raise RuntimeError("CPU conversion kernel did not complete successfully")
    CONVERSION_OUTPUT.mkdir(parents=True, exist_ok=True)
    if CONVERSION_OUTPUT.is_symlink():
        raise ValueError("conversion output directory is unsafe")
    cli(
        "kaggle",
        "kernels",
        "output",
        job["reference"],
        "-p",
        str(CONVERSION_OUTPUT),
        "-q",
        "--file-pattern",
        r"q25_fim_conversion_r1/conversion\.json$",
        timeout=300,
    )
    manifests = list(CONVERSION_OUTPUT.glob("**/conversion.json"))
    if (
        len(manifests) != 1
        or manifests[0].is_symlink()
        or manifests[0].stat().st_size > 2 * 1024**2
    ):
        raise ValueError("conversion worker result manifest is missing or ambiguous")
    result_path = manifests[0]
    result = json.loads(result_path.read_text())
    source_code = plan.get("source_code", {})
    runtime = result.get("runtime")
    storage = result.get("storage")
    final_storage = storage.get("final") if isinstance(storage, dict) else None
    if (
        result.get("schema") != "q25-fim-q4-conversion-run-v1"
        or result.get("status") != "complete"
        or result.get("selection_sha256") != plan["selection_sha256"]
        or result.get("training_plan_sha256") != plan["training_plan_sha256"]
        or result.get("selected_arm") != plan["source"]["arm"]
        or result.get("source_export_manifest_sha256") != plan["source"]["artifact_manifest_sha256"]
        or result.get("source_fingerprint") != plan["source"]["fingerprint"]
        or result.get("source_kernel_reference") != plan["source"]["kernel_reference"]
        or result.get("training_cursor") != plan["source"]["training_cursor"]
        or not isinstance(runtime, dict)
        or runtime.get("worker_source_sha256") != source_code.get("conversion_worker.py")
        or runtime.get("contract_source_sha256")
        != source_code.get("src/tinycomplete/code_cpt/q25_fim_conversion.py")
        or runtime.get("requirements_lock_sha256")
        != source_code.get("kaggle/q25_fim_conversion_r1/requirements-conversion.lock")
        or result.get("conversion", {}).get("format") != "Q4_K_M"
        or result.get("conversion", {}).get("gpu_enabled") is not False
        or not isinstance(storage, dict)
        or storage.get("artifact_cap_bytes") != CONVERSION_ARTIFACT_CAP_BYTES
        or storage.get("minimum_free_bytes") != CONVERSION_MINIMUM_FREE_BYTES
        or not isinstance(final_storage, dict)
        or final_storage.get("max_artifact_bytes") != CONVERSION_ARTIFACT_CAP_BYTES
        or final_storage.get("minimum_free_bytes") != CONVERSION_MINIMUM_FREE_BYTES
        or not isinstance(final_storage.get("current_accounted_bytes"), int)
        or isinstance(final_storage.get("current_accounted_bytes"), bool)
        or final_storage.get("current_accounted_bytes", -1) < 0
        or final_storage["current_accounted_bytes"] > CONVERSION_ARTIFACT_CAP_BYTES
    ):
        raise ValueError("conversion result manifest is incomplete or bound to another input")
    q4 = result.get("q4_export")
    if not isinstance(q4, dict):
        raise ValueError("conversion result does not contain a verified Q4 export")
    expected_name = f"q25-{plan['source']['arm']}-Q4_K_M.gguf"
    if (
        q4.get("file") != expected_name
        or not isinstance(q4.get("bytes"), int)
        or isinstance(q4.get("bytes"), bool)
        or not 0 < q4["bytes"] <= CONVERSION_ARTIFACT_CAP_BYTES
        or not re.fullmatch(r"[0-9a-f]{64}", str(q4.get("sha256")))
    ):
        raise ValueError("conversion Q4 output identity is invalid")
    if "f16_intermediate" in result:
        raise ValueError("conversion worker retained its F16 intermediate")
    existing_bytes = directory_bytes(ARTIFACTS)
    q4_target = CONVERSION_OUTPUT / expected_name
    if not q4_target.exists() and existing_bytes + q4["bytes"] > CONVERSION_ARTIFACT_CAP_BYTES:
        raise OSError("Q4 collection would exceed the artifact cap")
    missing_bytes = 0 if q4_target.exists() else q4["bytes"]
    if shutil.disk_usage(ARTIFACTS).free < missing_bytes + CONVERSION_MINIMUM_FREE_BYTES:
        raise OSError("Q4 collection would exhaust storage headroom")
    cli(
        "kaggle",
        "kernels",
        "output",
        job["reference"],
        "-p",
        str(CONVERSION_OUTPUT),
        "-q",
        "--file-pattern",
        re.escape(expected_name) + "$",
        timeout=900,
    )
    if (
        q4_target.is_symlink()
        or not q4_target.is_file()
        or q4_target.stat().st_size != q4["bytes"]
        or digest(q4_target) != q4["sha256"]
    ):
        raise ValueError("retrieved Q4 artifact differs from the verified conversion result")
    record = {
        "schema": "q25-fim-q4-conversion-receipt-v1",
        "reference": job["reference"],
        "source_kernel_reference": plan["source"]["kernel_reference"],
        "selected_arm": plan["source"]["arm"],
        "conversion_plan_sha256": digest(CONVERSION_PLAN),
        "selection_sha256": plan["selection_sha256"],
        "training_plan_sha256": plan["training_plan_sha256"],
        "source_export_manifest_sha256": plan["source"]["artifact_manifest_sha256"],
        "conversion_manifest_sha256": digest(result_path),
        "q4_file": expected_name,
        "q4_bytes": q4["bytes"],
        "q4_sha256": q4["sha256"],
        "collected_at": datetime.now(UTC).isoformat(),
        "source_weight_or_checkpoint_retrieval": False,
    }
    save(REPORT / "fim-conversion-verified.json", record)
    save(result_path.with_name("conversion-receipt.json"), record)
    return record


def watch_fim_conversion(
    plan: dict[str, Any] | None = None, *, poll_seconds: float = 30
) -> dict[str, Any]:
    plan = _load_conversion_plan() if plan is None else plan
    if not CONVERSION_JOB_FILE.is_file():
        raise FileNotFoundError("conversion observer requires one submitted job receipt")
    job = json.loads(CONVERSION_JOB_FILE.read_text())
    if job.get("plan_sha256") != digest(CONVERSION_PLAN) or job.get("enable_gpu") is not False:
        raise ValueError("conversion observer identity differs from the CPU plan")
    if (
        not isinstance(poll_seconds, (int, float))
        or isinstance(poll_seconds, bool)
        or not 1 <= poll_seconds <= 120
    ):
        raise ValueError("conversion observer poll interval is outside its bound")
    end = datetime.fromisoformat(job["submitted_at"].replace("Z", "+00:00")).timestamp()
    end += CONVERSION_SESSION_SECONDS + 900
    failures = 0
    while datetime.now(UTC).timestamp() < end:
        observation: dict[str, Any] = {
            "reference": job["reference"],
            "plan_sha256": job["plan_sha256"],
            "observed_at": datetime.now(UTC).isoformat(),
            "automatic_allocation": False,
        }
        try:
            status = cli("kaggle", "kernels", "status", job["reference"], timeout=60)
            observation["status"] = status
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            failures += 1
            observation.update(error_class=type(exc).__name__, consecutive_failures=failures)
            save(REPORT / "fim-conversion-watch.json", observation)
            if failures >= 3:
                raise RuntimeError(
                    "conversion observer lost connectivity; no job was retried"
                ) from None
            time.sleep(poll_seconds)
            continue
        save(REPORT / "fim-conversion-watch.json", observation)
        if "COMPLETE" in status.upper():
            return collect_fim_conversion(plan)
        if "ERROR" in status.upper():
            raise RuntimeError("CPU conversion job failed; inspect its existing output")
        time.sleep(poll_seconds)
    raise TimeoutError("conversion observer deadline reached; no new job was allocated")


def watch_fim(plan: dict[str, Any], arm: str, attempt: int) -> dict[str, Any]:
    if arm not in FIM_ARMS:
        raise ValueError("unknown FIM arm")
    job = json.loads((REPORT / f"fim-job-{arm}-{attempt}.json").read_text())
    if job["arm"] != arm or job["plan_sha256"] != digest(REPORT / "fim_training_plan.json"):
        raise ValueError("FIM observer identity differs from the frozen plan")
    end = datetime.fromisoformat(job["submitted_at"]).timestamp() + job["session_seconds"] + 900
    failures = 0
    while datetime.now(UTC).timestamp() < end:
        try:
            status = cli("kaggle", "kernels", "status", job["reference"], timeout=60)
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired):
            failures += 1
            if failures >= 3:
                raise RuntimeError(
                    "FIM observer lost connectivity; no allocation retried"
                ) from None
            time.sleep(30)
            continue
        save(
            REPORT / f"fim-watch-{arm}-{attempt}.json",
            {
                "reference": job["reference"],
                "status": status,
                "observed_at": datetime.now(UTC).isoformat(),
                "automatic_allocation": False,
            },
        )
        if any(value in status for value in ("COMPLETE", "ERROR")):
            return collect_fim(plan, arm, attempt)
        time.sleep(30)
    raise TimeoutError("FIM observer deadline reached; inspect the existing allocation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/research/q25_code_cpt_r2.yaml"
    )
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--phase", choices=("cpt", "fim"), default="cpt")
    parser.add_argument("--freeze-fim", action="store_true")
    parser.add_argument("--freeze-conversion", action="store_true")
    parser.add_argument("--bundle-conversion", action="store_true")
    parser.add_argument("--upload-conversion", action="store_true")
    parser.add_argument("--execute-conversion", action="store_true")
    parser.add_argument("--collect-conversion", action="store_true")
    parser.add_argument("--watch-conversion", action="store_true")
    parser.add_argument("--conversion-selection", type=Path, default=CONVERSION_SELECTION)
    parser.add_argument("--settle-sessions", action="store_true")
    parser.add_argument("--cpt-attempt", type=int, default=1)
    parser.add_argument("--arm", choices=FIM_ARMS, default=FIM_ARMS[0])
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--resume-source")
    args = parser.parse_args()
    result: dict[str, Any]
    conversion_actions = {
        "freeze": args.freeze_conversion,
        "bundle": args.bundle_conversion,
        "upload": args.upload_conversion,
        "execute": args.execute_conversion,
        "collect": args.collect_conversion,
        "watch": args.watch_conversion,
    }
    selected_conversion_actions = [
        name for name, selected in conversion_actions.items() if selected
    ]
    if selected_conversion_actions:
        if (
            len(selected_conversion_actions) != 1
            or args.phase != "cpt"
            or any(
                (
                    args.bundle,
                    args.upload,
                    args.execute,
                    args.collect,
                    args.watch,
                    args.freeze_fim,
                    args.settle_sessions,
                )
            )
        ):
            parser.error("conversion actions must run alone, one at a time")
        action = selected_conversion_actions[0]
        if action == "freeze":
            result = {"fim_conversion_plan": freeze_fim_conversion(args.conversion_selection)}
        else:
            if args.conversion_selection != CONVERSION_SELECTION:
                parser.error("--conversion-selection applies only to --freeze-conversion")
            conversion_plan = _load_conversion_plan()
            if action == "bundle":
                result = {"conversion_bundle": str(build_fim_conversion_bundle(conversion_plan))}
            elif action == "upload":
                result = {"conversion_dataset": upload_fim_conversion_bundle(conversion_plan)}
            elif action == "execute":
                result = {"conversion_job": submit_fim_conversion(conversion_plan)}
            elif action == "collect":
                result = {"conversion_output": collect_fim_conversion(conversion_plan)}
            else:
                result = {"conversion_output": watch_fim_conversion(conversion_plan)}
        print(json.dumps(result, sort_keys=True))
        return
    if args.settle_sessions:
        if any((args.bundle, args.upload, args.execute, args.collect, args.watch, args.freeze_fim)):
            parser.error("--settle-sessions cannot be combined with other campaign actions")
        result = settle_fim_zero_work_sessions()
        print(json.dumps(result, sort_keys=True))
        return
    plan = freeze(args.config)
    result = {"plan_sha256": digest(REPORT / "plan.json"), "frozen": True}
    if args.freeze_fim:
        result["fim_plan"] = freeze_fim_plan(plan, args.cpt_attempt)
    if args.phase == "fim":
        plan = json.loads((REPORT / "fim_training_plan.json").read_text())
        result["plan_sha256"] = digest(REPORT / "fim_training_plan.json")
        if args.bundle:
            result["bundle"] = str(build_fim_bundle(plan))
        if args.upload:
            result["dataset"] = upload_fim_bundle(plan)
            result["cpt_initializer_dataset"] = upload_cpt_initializer(plan)
        if args.execute:
            result["job"] = submit_fim(
                plan,
                arm=args.arm,
                attempt=args.attempt,
                cpt_attempt=args.cpt_attempt,
                resume_source=args.resume_source,
            )
        if args.collect:
            result["output"] = collect_fim(plan, args.arm, args.attempt)
        if args.watch:
            result["output"] = watch_fim(plan, args.arm, args.attempt)
        print(json.dumps(result, sort_keys=True))
        return
    if args.bundle:
        result["bundle"] = str(build_bundle(plan))
    if args.upload:
        result["dataset"] = upload_bundle(plan)
    if args.execute:
        result["job"] = submit(plan, attempt=args.attempt, resume_source=args.resume_source)
    if args.collect:
        result["output"] = collect(plan, args.attempt)
    if args.watch:
        result["output"] = watch(plan, args.attempt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
