"""Prepare, submit, and verify a private, bounded q25 one-line pilot.

Preparation runs only CPU validation and tokenizer preflight. GPU allocation is
an explicit ``submit`` command; both prepare and submit refresh authenticated
Kaggle quota and reject an active allocation. The main campaign builder and its
20,000-example gate are not modified by this pilot path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tinycomplete.one_line.pilot_data import (
    CONSTRUCTIVE,
    INSTINCT,
    policy_for_schema,
    validate_aggregate_budget,
    validate_constructive_manifest,
    validate_constructive_review,
    validate_constructive_splits,
    validate_pilot_row,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/research/one_line_gpu_pilot_r1"
ARTIFACT_ROOT = ROOT / "artifacts/research/one_line_gpu_pilot_r1"
TRAINER = ROOT / "scripts/train_one_line.py"
ALLOWED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
)
REQUIRED_MODEL_FILES = frozenset({"model.safetensors", "config.json", "tokenizer.json"})
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
MODEL_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
WEIGHT_SHA256 = "aff8914ec707fcaf9e2d4dc97197cded50b1c63e1d3a7a82e56f54d83ea47f80"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
MODEL_CONFIG_SHA256 = "e6bf24d1cf58278dcb4ded7e885b71cb1c56299b34c8046b1b64395d17b1891f"
PLAN_SCHEMA = "one-line-instinct-pilot-plan-v1"
DATA_SCHEMA = "one-line-instinct-pilot-v1"
INPUT_SCHEMA = "one-line-instinct-pilot-input-v1"
SESSION_SECONDS = 120 * 60
RESERVE_SECONDS = 20 * 60
MAX_TRAINING_TOKENS = 2_000_000
MAX_NEW_STORAGE_BYTES = 12 * 1024**3
QUOTA_GPU_HOURS_MULTIPLIER = 2
MIN_TRAIN_ROWS = 128
MAX_TRAIN_ROWS = 1024
MIN_DEV_ROWS = 64
MAX_EVAL_SECONDS = 15 * 60
FIXED_CALIBRATION = (
    "reports/research/one_line_r1/calibration_cases.json",
    "reports/research/one_line_r1/calibration_manifest.json",
)
DATASET_ID = "shlokbhakta/tabcomplete-one-line-instinct-pilot-r1-inputs"
KERNEL_ID = "shlokbhakta/tabcomplete-one-line-instinct-pilot-r1"
HEX_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
HEX_SHA1 = re.compile(r"[a-f0-9]{40}\Z")
KAGGLE_REF = re.compile(r"[a-z0-9_-]+/[a-z0-9_-]+\Z")


def validate_kaggle_refs(dataset_id: str, kernel_id: str) -> None:
    if KAGGLE_REF.fullmatch(dataset_id) is None or KAGGLE_REF.fullmatch(kernel_id) is None:
        raise ValueError("invalid frozen Kaggle references")
    if not 6 <= len(dataset_id.split("/", 1)[1]) <= 50:
        raise ValueError("Kaggle dataset slug must be between 6 and 50 characters")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _run(args: list[str], *, timeout: int = 90) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} command failed (exit {result.returncode})")
    return result.stdout.strip()


def live_quota() -> dict[str, Any]:
    """Read the existing authenticated quota/status helper without writing a ledger."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_one_line_campaign import live_quota as observe  # noqa: PLC0415

    quota = observe()
    _resolve_unknown_kernel_statuses(quota)
    return quota


def _validate_plan(plan: dict[str, Any], config: dict[str, Any]) -> None:
    policy = policy_for_schema(plan.get("data", {}).get("schema"))
    if plan.get("schema") != policy.plan_schema or plan.get("suite_revision") != 3:
        raise ValueError("pilot plan schema or suite revision mismatch")
    if plan.get("branch") != policy.branch:
        raise ValueError("pilot plan branch mismatch")
    if (
        not isinstance(plan.get("base_commit"), str)
        or HEX_SHA1.fullmatch(plan["base_commit"]) is None
    ):
        raise ValueError("pilot plan base commit is invalid")
    if plan.get("config_sha256") is None or not HEX_SHA256.fullmatch(plan["config_sha256"]):
        raise ValueError("pilot plan config hash is invalid")
    student = plan.get("student", {})
    expected_student = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "config_sha256": MODEL_CONFIG_SHA256,
    }
    if any(student.get(key) != value for key, value in expected_student.items()):
        raise ValueError("pilot plan does not pin the approved untouched q25 base")
    if config.get("student", {}).get("initializer") != "untouched_pretrained":
        raise ValueError("pilot config must start from the untouched pretrained model")
    if any(
        config["student"].get(key) != value
        for key, value in expected_student.items()
        if key != "config_sha256"
    ):
        raise ValueError("pilot config model identity differs from the approved q25 base")

    data = plan.get("data", {})
    if (
        data.get("schema") != policy.data_schema
        or data.get("dataset_id") != policy.dataset_id
        or data.get("dataset_license") != policy.dataset_license
        or data.get("source_file_license_status") != policy.file_license_status
        or not isinstance(data.get("dataset_revision"), str)
        or not data["dataset_revision"]
        or data.get("file_groups_disjoint") is not True
        or not MIN_TRAIN_ROWS <= data.get("train_count", 0) <= MAX_TRAIN_ROWS
        or data.get("dev_count", 0) < MIN_DEV_ROWS
    ):
        raise ValueError("pilot source, split, or row-count plan is invalid")
    for key in ("train_sha256", "development_sha256", "manifest_sha256"):
        if not isinstance(data.get(key), str) or HEX_SHA256.fullmatch(data[key]) is None:
            raise ValueError(f"pilot data {key} is invalid")

    training = plan.get("training", {})
    budgets = plan.get("budgets", {})
    if (
        training.get("phase") != "pilot"
        or training.get("epochs") != 1
        or training.get("peak_learning_rate") != 1e-5
        or training.get("planned_nonpadding_input_tokens", MAX_TRAINING_TOKENS + 1)
        > MAX_TRAINING_TOKENS
        or training.get("max_nonpadding_input_tokens") != MAX_TRAINING_TOKENS
    ):
        raise ValueError("pilot training plan exceeds the fixed one-pass/token contract")
    if (
        budgets.get("max_session_seconds") != SESSION_SECONDS
        or budgets.get("reserve_seconds") != RESERVE_SECONDS
        or budgets.get("max_new_storage_bytes") != MAX_NEW_STORAGE_BYTES
        or budgets.get("quota_gpu_hours_multiplier") != QUOTA_GPU_HOURS_MULTIPLIER
        or budgets.get("no_automatic_renewal") is not True
    ):
        raise ValueError("pilot budget contract differs from the fixed campaign limits")
    if policy is CONSTRUCTIVE:
        validate_aggregate_budget(
            budgets,
            planned_tokens=training["planned_nonpadding_input_tokens"],
            session_seconds=SESSION_SECONDS,
        )
    frozen_quota = plan.get("quota_at_freeze", {})
    if (
        not frozen_quota.get("observed_at")
        or not frozen_quota.get("renewal")
        or frozen_quota.get("remaining") is None
        or frozen_quota.get("units") != "Kaggle account GPU-hours"
        or not frozen_quota.get("source")
        or not _job_statuses_verified(frozen_quota)
    ):
        raise ValueError("pilot plan lacks an authenticated quota observation")
    if frozen_quota.get("active_jobs"):
        raise ValueError("pilot was frozen while a GPU job was active")


def _check_live_quota(plan: dict[str, Any], quota: dict[str, Any]) -> None:
    frozen = plan["quota_at_freeze"]
    if quota.get("remaining") is None or quota.get("renewal") is None:
        raise RuntimeError("authenticated Kaggle GPU quota could not be verified")
    if not _job_statuses_verified(quota):
        raise RuntimeError("Kaggle active-job status could not be verified")
    if quota.get("active_jobs"):
        raise RuntimeError("another Kaggle GPU allocation is active")
    if quota.get("renewal") != frozen["renewal"]:
        raise RuntimeError("Kaggle quota renewal changed; automatic renewed use is disabled")
    required = (SESSION_SECONDS / 3600) * QUOTA_GPU_HOURS_MULTIPLIER
    if float(quota["remaining"]) < required:
        raise RuntimeError("insufficient authenticated Kaggle GPU quota for the frozen session")


def _job_statuses_verified(quota: dict[str, Any]) -> bool:
    jobs = quota.get("job_statuses")
    if not isinstance(jobs, list):
        return False
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("status"), str):
            return False
        status = job["status"].strip().casefold()
        if not status:
            return False
        if status == "unknown" and not (
            job.get("not_found_verified") is True
            and job.get("http_status") == 404
            and job.get("verified_at")
            and job.get("verification_source") == "authenticated kaggle kernels status HTTP 404"
        ):
            return False
    return True


def _resolve_unknown_kernel_statuses(quota: dict[str, Any]) -> None:
    """Resolve CLI's `unknown` with an authenticated status lookup; only 404 is absent."""
    jobs = quota.get("job_statuses")
    if not isinstance(jobs, list):
        return
    now = datetime.now(UTC).isoformat()
    for job in jobs:
        if not isinstance(job, dict) or str(job.get("status", "")).strip().casefold() != "unknown":
            continue
        reference = job.get("reference")
        if not isinstance(reference, str) or KAGGLE_REF.fullmatch(reference) is None:
            continue
        result = subprocess.run(
            ["kaggle", "kernels", "status", reference],
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        if result.returncode == 0:
            job["status"] = result.stdout.strip() or "unknown"
            job["verified_at"] = now
            job["verification_source"] = "authenticated kaggle kernels status"
            continue
        response = (result.stdout + "\n" + result.stderr).casefold()
        if "404" in response and "not found" in response:
            job.update(
                {
                    "not_found_verified": True,
                    "http_status": 404,
                    "verified_at": now,
                    "verification_source": "authenticated kaggle kernels status HTTP 404",
                }
            )
    quota["active_jobs"] = [
        job
        for job in jobs
        if isinstance(job, dict)
        and any(word in str(job.get("status", "")).upper() for word in ("RUNNING", "QUEUED"))
    ]


def _git_identity(branch: str, base_commit: str, *, require_remote: bool = True) -> tuple[str, str]:
    current_branch = _run(["git", "-C", str(ROOT), "branch", "--show-current"])
    if current_branch != branch:
        raise ValueError("current branch differs from the frozen pilot branch")
    if _run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"]):
        raise ValueError("commit all pilot inputs and code before bundle preparation")
    commit = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if HEX_SHA1.fullmatch(commit) is None:
        raise ValueError("current Git commit is invalid")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", base_commit, commit],
        capture_output=True,
        check=False,
    )
    if ancestor.returncode:
        raise ValueError("pilot code commit does not descend from its frozen base")
    if require_remote:
        remote = _run(["git", "-C", str(ROOT), "ls-remote", "origin", f"refs/heads/{branch}"])
        if not remote or remote.split()[0] != commit:
            raise ValueError("push the exact clean pilot commit before Kaggle packaging")
    return branch, commit


def _rows(
    path: Path,
    *,
    expected_split: str,
    source_type: str,
    data_schema: str = INSTINCT.data_schema,
) -> list[dict[str, Any]]:
    from tinycomplete.one_line.contract import EditAction, EditState, apply_action  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError("pilot shard contains a row from another split")
            if row.get("source_type") != source_type:
                raise ValueError("pilot shard contains an unapproved source type")
            if not row.get("source_license"):
                raise ValueError("pilot row lacks its declared dataset license field")
            if row.get("validation", {}).get("replay_verified") is not True:
                raise ValueError("pilot row is not replay-verified")
            validate_pilot_row(row, policy_for_schema(data_schema))
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("pilot IDs must be unique nonempty strings")
            ids.add(identifier)
            state = EditState.from_mapping(row["state"])
            action = EditAction(**row["action"])
            if apply_action(state, action) != row["after_source"]:
                raise ValueError("pilot action does not reconstruct its after-state")
            rows.append(row)
    return rows


def _file_group(row: dict[str, Any]) -> str:
    state = row.get("state", {})
    group = (
        row.get("source_group_id")
        or row.get("file_group_id")
        or row.get("provenance", {}).get("file_group_id")
    )
    if not group:
        group = state.get("file_id")
    if not isinstance(group, str) or not group:
        raise ValueError("pilot row lacks a stable file-group identity")
    return group


def validate_inputs(
    *,
    config_path: Path,
    plan_path: Path,
    model_dir: Path,
    train_path: Path,
    development_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    plan = read_json(plan_path)
    policy = policy_for_schema(plan.get("data", {}).get("schema"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_plan(plan, config)
    if sha256_file(config_path) != plan["config_sha256"]:
        raise ValueError("one-line config hash differs from the frozen pilot plan")
    if sha256_file(model_dir / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("untouched q25 weight SHA-256 mismatch")
    if sha256_file(model_dir / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("pinned q25 tokenizer SHA-256 mismatch")
    model_config = read_json(model_dir / "config.json")
    if (
        model_config.get("model_type") != "qwen2"
        or sha256_file(model_dir / "config.json") != MODEL_CONFIG_SHA256
    ):
        raise ValueError("pilot model config identity or architecture mismatch")

    data_plan = plan["data"]
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != data_plan["manifest_sha256"]:
        raise ValueError("pilot data-manifest hash mismatch")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != policy.data_schema
        or manifest.get("dataset_id") != data_plan["dataset_id"]
        or manifest.get("dataset_revision") != data_plan["dataset_revision"]
        or manifest.get("dataset_license") != data_plan["dataset_license"]
        or manifest.get("source_file_license_status") != policy.file_license_status
        or manifest.get("train_sha256") != data_plan["train_sha256"]
        or manifest.get("development_sha256") != data_plan["development_sha256"]
        or manifest.get("train_count") != data_plan["train_count"]
        or manifest.get("dev_count") != data_plan["dev_count"]
        or manifest.get("file_groups_disjoint") is not True
    ):
        raise ValueError("pilot data manifest differs from its frozen plan")
    if sha256_file(train_path) != data_plan["train_sha256"]:
        raise ValueError("pilot training shard hash mismatch")
    if sha256_file(development_path) != data_plan["development_sha256"]:
        raise ValueError("pilot development shard hash mismatch")
    train_rows = _rows(
        train_path,
        expected_split="train",
        source_type=policy.source_type,
        data_schema=policy.data_schema,
    )
    dev_rows = _rows(
        development_path,
        expected_split="development",
        source_type=policy.source_type,
        data_schema=policy.data_schema,
    )
    if not MIN_TRAIN_ROWS <= len(train_rows) <= MAX_TRAIN_ROWS:
        raise ValueError("pilot training row count is outside 128..1024")
    if len(train_rows) != data_plan["train_count"]:
        raise ValueError("pilot training count differs from the frozen plan")
    if len(dev_rows) < MIN_DEV_ROWS or len(dev_rows) != data_plan["dev_count"]:
        raise ValueError("pilot development count differs from the frozen plan")
    train_groups = {_file_group(row) for row in train_rows}
    dev_groups = {_file_group(row) for row in dev_rows}
    if train_groups & dev_groups:
        raise ValueError("pilot train and development file groups overlap")
    if policy == CONSTRUCTIVE:
        # Recompute provenance and normalized-template isolation, rather than
        # trusting only the generator's manifest boolean.
        validate_constructive_splits([*train_rows, *dev_rows])
        validate_constructive_manifest(manifest)
        validate_constructive_review(
            manifest, [*train_rows, *dev_rows], manifest_path.parent / "independent_review.json"
        )
    return {
        "plan": plan,
        "config": config,
        "data_manifest": manifest,
        "train_rows": train_rows,
        "development_rows": dev_rows,
        "plan_sha256": sha256_file(plan_path),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest_sha,
        "train_sha256": data_plan["train_sha256"],
        "development_sha256": data_plan["development_sha256"],
        "model_weight_sha256": WEIGHT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
    }


def inspect_training(
    *,
    config_path: Path,
    plan_path: Path,
    model_dir: Path,
    train_path: Path,
    development_path: Path,
    manifest_path: Path,
    output_parent: Path,
) -> dict[str, Any]:
    """Invoke the existing trainer's CPU-only inspection path and return its counts."""
    plan = read_json(plan_path)
    policy = policy_for_schema(plan.get("data", {}).get("schema"))
    with tempfile.TemporaryDirectory(prefix="tabcomplete-pilot-inspect-") as temporary:
        output = Path(temporary) / "inspect-only-output"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        )
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            sys.executable,
            str(TRAINER),
            "--config",
            str(config_path),
            "--plan",
            str(plan_path),
            "--model",
            str(model_dir),
            "--data",
            str(train_path),
            "--data-sha256",
            sha256_file(train_path),
            "--data-manifest",
            str(manifest_path),
            "--development-data",
            str(development_path),
            "--phase",
            "pilot",
            "--epochs",
            "1",
            "--output",
            str(output),
            "--session-minutes",
            "120",
            "--reserve-minutes",
            "20",
            "--external-campaign-tokens",
            str(plan["budgets"].get("prior_training_input_tokens", 0)),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if result.returncode:
            raise ValueError("existing trainer rejected the CPU pilot preflight")
        try:
            summary = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ValueError(
                "existing trainer returned no parseable CPU preflight summary"
            ) from exc
        if (
            summary.get("examples")
            != len(
                _rows(
                    train_path,
                    expected_split="train",
                    source_type=policy.source_type,
                    data_schema=policy.data_schema,
                )
            )
            or summary.get("identity", {}).get("phase") != "pilot"
            or summary.get("identity", {}).get("source", {}).get("model_weight_sha256")
            != WEIGHT_SHA256
        ):
            raise ValueError("trainer CPU preflight identity differs from the frozen pilot")
        return summary


def _directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def _stage_file(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(strict=True), target)
    return {"bytes": target.stat().st_size, "sha256": sha256_file(target)}


def _record_file_tree(path: Path, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for name in names:
        source = path / name
        if source.is_file():
            items[name] = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
    return items


def prepare_bundle(
    *,
    config_path: Path,
    plan_path: Path,
    model_dir: Path,
    train_path: Path,
    development_path: Path,
    manifest_path: Path,
    output: Path,
    quota: dict[str, Any] | None = None,
    git_identity: tuple[str, str] | None = None,
    inspection: dict[str, Any] | None = None,
    quota_reader: Callable[[], dict[str, Any]] = live_quota,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("pilot bundle output already exists")
    checked = validate_inputs(
        config_path=config_path,
        plan_path=plan_path,
        model_dir=model_dir,
        train_path=train_path,
        development_path=development_path,
        manifest_path=manifest_path,
    )
    plan = checked["plan"]
    policy = policy_for_schema(plan["data"]["schema"])
    if quota is None:
        quota = quota_reader()
    _check_live_quota(plan, quota)
    identity = git_identity or _git_identity(plan["branch"], plan["base_commit"])
    if identity[0] != plan["branch"]:
        raise ValueError("bundle branch differs from the frozen plan")
    if not HEX_SHA1.fullmatch(identity[1]):
        raise ValueError("bundle Git commit is invalid")
    if git_identity is None and sha256_file(ROOT / policy.repository_plan_path) != checked[
        "plan_sha256"
    ]:
        raise ValueError("committed pilot plan differs from the selected staged plan")

    if inspection is None:
        inspection = inspect_training(
            config_path=config_path,
            plan_path=plan_path,
            model_dir=model_dir,
            train_path=train_path,
            development_path=development_path,
            manifest_path=manifest_path,
            output_parent=output.parent,
        )
    training = plan["training"]
    actual_tokens = inspection.get("planned_training_input_tokens")
    if (
        inspection.get("examples") != plan["data"]["train_count"]
        or actual_tokens != training["planned_nonpadding_input_tokens"]
        or not isinstance(actual_tokens, int)
        or actual_tokens > MAX_TRAINING_TOKENS
    ):
        raise ValueError("tokenizer preflight differs from the frozen two-million-token plan")

    calibration_expected = (
        ROOT / FIXED_CALIBRATION[0],
        ROOT / FIXED_CALIBRATION[1],
    )
    teacher = checked["config"]["teacher"]
    for path, expected in zip(
        calibration_expected,
        (teacher["calibration_fixture_sha256"], teacher["calibration_manifest_sha256"]),
        strict=True,
    ):
        if sha256_file(path) != expected:
            raise ValueError("fixed calibration fixture hash differs from the pinned config")

    model_names = tuple(name for name in ALLOWED_MODEL_FILES if (model_dir / name).is_file())
    if not REQUIRED_MODEL_FILES.issubset(model_names):
        raise FileNotFoundError("required pinned q25 files are missing")
    projected = sum((model_dir / name).stat().st_size for name in model_names)
    projected += sum(
        path.stat().st_size
        for path in (train_path, development_path, manifest_path, config_path, plan_path)
    )
    review_path = manifest_path.parent / "independent_review.json"
    if policy is CONSTRUCTIVE:
        projected += review_path.stat().st_size
    if projected > MAX_NEW_STORAGE_BYTES:
        raise ValueError("pilot bundle exceeds the 12 GiB campaign artifact limit")
    output_parent = output.parent
    while not output_parent.exists():
        output_parent = output_parent.parent
    if shutil.disk_usage(output_parent).free < projected + 2 * 1024**3:
        raise OSError("insufficient free space plus 2 GiB headroom for the pilot bundle")
    artifact_root = ARTIFACT_ROOT if policy == INSTINCT else Path("/mnt/ssd/tabcomplete-product-r2")
    existing_campaign_bytes = _directory_bytes(artifact_root)
    if existing_campaign_bytes + projected > MAX_NEW_STORAGE_BYTES:
        raise ValueError("pilot campaign artifacts would exceed the 12 GiB storage limit")

    dataset_id = plan.get("kaggle", {}).get("dataset_id", DATASET_ID)
    kernel_id = plan.get("kaggle", {}).get("kernel_id", KERNEL_ID)
    validate_kaggle_refs(dataset_id, kernel_id)
    temporary = output.with_name(output.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError("incomplete pilot bundle already exists; inspect it first")
    dataset_dir = temporary / "dataset"
    kernel_dir = temporary / "kernel"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    for name in model_names:
        files[name] = _stage_file(model_dir / name, dataset_dir / name)
    for name, source in (
        ("train.jsonl", train_path),
        ("development.jsonl", development_path),
        ("data_manifest.json", manifest_path),
        ("config.yaml", config_path),
        ("plan.json", plan_path),
    ):
        files[name] = _stage_file(source, dataset_dir / name)
    if policy is CONSTRUCTIVE:
        files["independent_review.json"] = _stage_file(
            review_path, dataset_dir / "independent_review.json"
        )
    input_manifest = {
        "schema": INPUT_SCHEMA,
        "plan_sha256": checked["plan_sha256"],
        "config_sha256": checked["config_sha256"],
        "branch": identity[0],
        "commit": identity[1],
        "files": files,
    }
    write_json(dataset_dir / "input-manifest.json", input_manifest)
    write_json(
        dataset_dir / "dataset-metadata.json",
        {
            "title": f"TabComplete {policy.data_schema} (Private)",
            "id": dataset_id,
            "licenses": [{"name": "other"}],
            "description": (
                "Private, bounded research input. Qwen2.5-Coder weights: Apache-2.0. "
                f"Dataset metadata license: {policy.dataset_license}. "
                f"Source-file status: {policy.file_license_status}. "
                "This package does not grant source-file rights. Do not make public."
            ),
        },
    )
    session = {
        "schema": "one-line-instinct-pilot-session-v1",
        "data_schema": policy.data_schema,
        "plan_schema": policy.plan_schema,
        "source_dataset_id": policy.dataset_id,
        "branch": identity[0],
        "commit": identity[1],
        "base_commit": plan["base_commit"],
        "dataset_id": dataset_id,
        "kernel_id": kernel_id,
        "input_manifest_sha256": sha256_file(dataset_dir / "input-manifest.json"),
        "plan_sha256": checked["plan_sha256"],
        "config_sha256": checked["config_sha256"],
        "train_sha256": checked["train_sha256"],
        "development_sha256": checked["development_sha256"],
        "manifest_sha256": checked["manifest_sha256"],
        "model_weight_sha256": checked["model_weight_sha256"],
        "tokenizer_sha256": checked["tokenizer_sha256"],
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_id": plan["student"]["model_id"],
        "model_revision": plan["student"]["revision"],
        "dataset_revision": plan["data"]["dataset_revision"],
        "dataset_license": plan["data"]["dataset_license"],
        "source_file_license_status": plan["data"]["source_file_license_status"],
        "train_count": plan["data"]["train_count"],
        "development_count": plan["data"]["dev_count"],
        "session_seconds": SESSION_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "maximum_training_tokens": MAX_TRAINING_TOKENS,
        "prior_training_input_tokens": plan["budgets"].get("prior_training_input_tokens", 0),
        "prior_session_wall_seconds": plan["budgets"].get("prior_session_wall_seconds", 0),
        "planned_training_tokens": actual_tokens,
        "epochs": 1,
        "phase": "pilot",
        "peak_learning_rate": training["peak_learning_rate"],
        "quota_before_prepare": quota,
    }
    template = (Path(__file__).parent / "run.py").read_text(encoding="utf-8")
    generated = template.replace('"__SESSION_LITERAL__"', repr(json.dumps(session, sort_keys=True)))
    (kernel_dir / "run.py").write_text(generated, encoding="utf-8")
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": kernel_id.split("/", 1)[1],
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [dataset_id],
            "kernel_sources": [],
            "competition_sources": [],
        },
    )
    bundle_manifest = {
        "schema": "one-line-instinct-pilot-bundle-v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "plan_sha256": checked["plan_sha256"],
        "input_manifest_sha256": session["input_manifest_sha256"],
        "branch": identity[0],
        "commit": identity[1],
        "dataset_id": dataset_id,
        "kernel_id": kernel_id,
        "session_seconds": SESSION_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "planned_training_tokens": actual_tokens,
        "train_examples": len(checked["train_rows"]),
        "development_examples": len(checked["development_rows"]),
        "model_weight_sha256": checked["model_weight_sha256"],
        "tokenizer_sha256": checked["tokenizer_sha256"],
        "source_file_license_status": policy.file_license_status,
        "quota_at_prepare": quota,
        "projected_input_bytes": projected,
        "submission_commands": [
            f"kaggle datasets create -p {dataset_dir} -t",
            f"kaggle kernels push -p {kernel_dir} --timeout {SESSION_SECONDS}",
        ],
    }
    write_json(temporary / "bundle-manifest.json", bundle_manifest)
    os.replace(temporary, output)
    return bundle_manifest


def _csv_refs(command: list[str]) -> set[str]:
    output = _run(command, timeout=90)
    rows = csv.DictReader(io.StringIO(output))
    return {str(row.get("ref", "")) for row in rows}


def submit_bundle(
    bundle: Path, *, quota_reader: Callable[[], dict[str, Any]] = live_quota
) -> dict[str, Any]:
    manifest = read_json(bundle / "bundle-manifest.json")
    if manifest.get("schema") != "one-line-instinct-pilot-bundle-v1":
        raise ValueError("unrecognized pilot bundle")
    plan_path = bundle / "dataset/plan.json"
    config_path = bundle / "dataset/config.yaml"
    plan = read_json(plan_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_plan(plan, config)
    if sha256_file(plan_path) != manifest["plan_sha256"]:
        raise ValueError("staged pilot plan changed after preparation")
    if sha256_file(bundle / "dataset/input-manifest.json") != manifest["input_manifest_sha256"]:
        raise ValueError("staged pilot input manifest changed after preparation")
    if manifest["commit"] != _git_identity(plan["branch"], plan["base_commit"])[1]:
        raise ValueError("current pushed pilot code differs from the prepared bundle")
    quota = quota_reader()
    _check_live_quota(plan, quota)

    dataset_id = manifest["dataset_id"]
    kernel_id = manifest["kernel_id"]
    state_path = bundle / "submission-state.json"
    state = read_json(state_path) if state_path.exists() else {"state": "prepared"}
    if state.get("state") == "kernel_pushed":
        raise ValueError("pilot kernel was already submitted; refusing a duplicate allocation")
    datasets = _csv_refs(["kaggle", "datasets", "list", "--mine", "--page-size", "100", "--csv"])
    kernels = _csv_refs(["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"])
    if state.get("state") == "prepared":
        if dataset_id in datasets:
            raise ValueError("pilot dataset ID already exists; reconcile before submitting")
        if kernel_id in kernels:
            raise ValueError("pilot kernel ID already exists; reconcile before submitting")
        _run(["kaggle", "datasets", "create", "-p", str(bundle / "dataset"), "-t"], timeout=900)
        state = {"state": "dataset_created", "dataset_id": dataset_id}
        write_json(state_path, state)

    # Dataset publication is not a GPU allocation. Refresh the quota and job
    # state immediately before the only command that can start the T4 kernel.
    quota = quota_reader()
    _check_live_quota(plan, quota)
    kernels = _csv_refs(["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"])
    if kernel_id in kernels:
        raise ValueError("pilot kernel ID already exists; refusing a duplicate allocation")
    _run(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(bundle / "kernel"),
            "--timeout",
            str(SESSION_SECONDS),
            "--accelerator",
            "NvidiaTeslaT4",
        ],
        timeout=180,
    )
    state = {
        "state": "kernel_pushed",
        "dataset_id": dataset_id,
        "kernel_id": kernel_id,
        "submitted_at": datetime.now(UTC).isoformat(),
        "quota_before_submit": quota,
    }
    write_json(state_path, state)
    return state


def verify_output(path: Path, *, expected_plan_sha256: str | None = None) -> dict[str, Any]:
    status_path = path / "one_line_gpu_pilot_r1/worker-status.json"
    if not status_path.is_file():
        status_path = path / "worker-status.json"
    status = read_json(status_path)
    if status.get("schema") != "one-line-instinct-pilot-worker-status-v1":
        raise ValueError("pilot output status schema mismatch")
    if expected_plan_sha256 and status.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("pilot output plan hash mismatch")
    training = status_path.parent / "training"
    result = read_json(training / "run_result.json")
    latest = read_json(training / "latest.json")
    checkpoint = training / Path(latest["checkpoint"]).name
    marker = read_json(checkpoint.with_suffix(checkpoint.suffix + ".complete.json"))
    checkpoint_sha = sha256_file(checkpoint)
    if (
        checkpoint_sha != latest.get("sha256")
        or checkpoint_sha != marker.get("sha256")
        or result.get("fingerprint") != marker.get("fingerprint")
        or status.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("pilot resumable checkpoint failed hash or identity verification")
    if status.get("training_status") != result.get("status"):
        raise ValueError("pilot worker and trainer terminal statuses disagree")
    worker_state = status.get("state")
    if worker_state == "failed":
        if status.get("failure_stage") != "adapted_evaluation":
            raise ValueError("failed pilot worker did not finish both quality evaluations")
    elif worker_state == "verified_complete":
        parity = status.get("export_tokenizer_equivalence", {})
        if parity.get("development_prompts_with_identical_token_ids") != status.get(
            "development_count"
        ):
            raise ValueError("complete worker lacks export tokenizer parity evidence")
    elif worker_state == "partial_checkpoint_preserved" and result.get("status") == "deadline_stop":
        pass
    else:
        raise ValueError("pilot worker has an unrecognized final state")

    if result.get("status") == "complete":
        export = result.get("inference_export")
        if not isinstance(export, dict):
            raise ValueError("complete pilot output lacks its inference export")
        export_dir = Path(export["path"])
        if not export_dir.is_absolute() or not export_dir.exists():
            export_dir = training / "inference-f16"
        for filename, identity in export.get("files", {}).items():
            artifact = export_dir / filename
            if not artifact.is_file() or artifact.stat().st_size != identity["bytes"]:
                raise ValueError("pilot inference export file size mismatch")
            if sha256_file(artifact) != identity["sha256"]:
                raise ValueError("pilot inference export hash mismatch")
        export_manifest = read_json(export_dir / "artifact_manifest.json")
        source = export_manifest.get("source", {})
        if source.get("model_weight_sha256") != status.get("source_weight_sha256"):
            raise ValueError("pilot inference export source-weight identity mismatch")
        for evaluation_name in ("baseline-evaluation.json", "adapted-evaluation.json"):
            evaluation = read_json(status_path.parent / evaluation_name)
            if evaluation.get("identity", {}).get("development_sha256") != status.get(
                "development_sha256"
            ):
                raise ValueError("pilot evaluation development identity mismatch")
            if "synthetic_calibration" not in evaluation:
                raise ValueError("pilot evaluation lacks the fixed calibration result")
    elif result.get("status") != "deadline_stop":
        raise ValueError("pilot output has an unrecognized terminal training status")
    return {
        "status": result["status"],
        "worker_state": worker_state,
        "worker_failure_type": status.get("failure_type") if worker_state == "failed" else None,
        "artifact_integrity_verified": True,
        "plan_sha256": status["plan_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "training_input_tokens": result.get("cursor", {}).get("training_input_tokens"),
        "complete_export": result.get("status") == "complete",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="CPU-only input checks and private bundle staging")
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/research/one_line_r1.yaml")
    prepare.add_argument("--plan", type=Path, default=REPORT / "plan.json")
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--train", type=Path, required=True)
    prepare.add_argument("--development", type=Path, required=True)
    prepare.add_argument("--data-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    submit = sub.add_parser(
        "submit", help="explicitly upload the private inputs and submit one job"
    )
    submit.add_argument("--bundle", type=Path, required=True)
    verify = sub.add_parser("verify-output", help="verify retrieved checkpoint and export hashes")
    verify.add_argument("path", type=Path)
    verify.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.command == "verify-output":
        result = verify_output(args.path, expected_plan_sha256=args.plan_sha256)
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "submit":
        print(json.dumps(submit_bundle(args.bundle), sort_keys=True))
        return
    record = prepare_bundle(
        config_path=args.config,
        plan_path=args.plan,
        model_dir=args.model,
        train_path=args.train,
        development_path=args.development,
        manifest_path=args.data_manifest,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "state": "prepared_no_upload",
                "bundle": str(args.output),
                "plan_sha256": record["plan_sha256"],
                "train_examples": record["train_examples"],
                "development_examples": record["development_examples"],
                "planned_training_tokens": record["planned_training_tokens"],
                "session_seconds": record["session_seconds"],
                "reserve_seconds": record["reserve_seconds"],
                "dataset_private_by_default": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
