"""Freeze and submit the bounded Qwen2.5 code-pretraining ablation.

Reuses the existing quota reader and private Kaggle dataset/kernel interface.
Preparation is CPU-only; allocation checks current quota immediately before push.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
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


def submit(plan: dict[str, Any], *, attempt: int, resume_source: str | None) -> dict[str, Any]:
    budget = plan["configuration"]["budget"]
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
            "title": f"TabComplete Q25 code CPT r2 attempt {attempt}",
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
        job.update(status="submitted", submission_response=response)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/research/q25_code_cpt_r2.yaml"
    )
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--resume-source")
    args = parser.parse_args()
    plan = freeze(args.config)
    result: dict[str, Any] = {"plan_sha256": digest(REPORT / "plan.json"), "frozen": True}
    if args.bundle:
        result["bundle"] = str(build_bundle(plan))
    if args.upload:
        result["dataset"] = upload_bundle(plan)
    if args.execute:
        result["job"] = submit(plan, attempt=args.attempt, resume_source=args.resume_source)
    if args.collect:
        result["output"] = collect(plan, args.attempt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
