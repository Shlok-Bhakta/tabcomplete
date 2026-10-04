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
import time
from datetime import UTC, datetime, timedelta
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
FIM_DATASET = "shlokbhakta/tabcomplete-q25-fim-r2-inputs"
CPT_INITIALIZER_DATASET = "shlokbhakta/tabcomplete-q25-cpt-r2-fim-initializer"
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


def check_shared_allocation_budget(session_seconds: int, *, phase: str) -> None:
    path = REPORT / "campaign_budget.json"
    if not path.exists():
        return
    shared = json.loads(path.read_text())["shared_limits"]
    jobs = [*REPORT.glob("job-*.json"), *REPORT.glob("fim-job-*.json")]
    reserved = sum(
        int(json.loads(job.read_text())["conservative_reserved_session_seconds"]) for job in jobs
    )
    future_reserve = (
        int(shared["minimum_reserved_future_fim_session_seconds"]) if phase == "cpt" else 0
    )
    if reserved + session_seconds + future_reserve > shared["aggregate_reserved_session_seconds"]:
        raise RuntimeError("shared CPT/FIM session reservation exhausted")
    if (reserved + session_seconds) / 3600 * 2 > shared["conservative_account_gpu_hours"]:
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
    runtime = json.loads(manifests[0].read_text())["identity"]["runtime"]
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
        "gpu_execution_authorized": True,
        "preparation_plan_sha256": digest(REPORT / "fim_preparation_plan.json"),
        "parent_cpt_plan_sha256": digest(REPORT / "plan.json"),
        "campaign_budget_sha256": digest(REPORT / "campaign_budget.json"),
        "configuration": {
            "training": training,
            "runtime": {
                key: runtime[key]
                for key in ("python", "torch", "transformers", "bitsandbytes", "cuda_runtime")
            },
            "budget": {
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
            or verified.get("plan_sha256") != digest(REPORT / "fim_training_plan.json")
            or verified.get("input_manifest_sha256") != prior.get("input_manifest_sha256")
            or verified.get("commit") != prior.get("commit")
        ):
            raise ValueError("FIM retry authorization lineage differs from its prior receipt")
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
    datasets = [FIM_DATASET, BASE_DATASET]
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
    parser.add_argument("--cpt-attempt", type=int, default=1)
    parser.add_argument("--arm", choices=FIM_ARMS, default=FIM_ARMS[0])
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--resume-source")
    args = parser.parse_args()
    plan = freeze(args.config)
    result: dict[str, Any] = {"plan_sha256": digest(REPORT / "plan.json"), "frozen": True}
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
