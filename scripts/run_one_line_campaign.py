"""Frozen, budgeted entry point for the one-line edit campaign.

CPU preparation and evidence gates precede any GPU allocation. This process does
not launch a notebook without a frozen, sufficiently diverse public dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/one_line_r1"
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    / "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
SCHEMA = REPORT / "schema.md"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def cli(*args: str, timeout: int = 90) -> str:
    done = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if done.returncode:
        raise RuntimeError(f"{args[0]} exited {done.returncode}")
    return done.stdout


def live_quota() -> dict[str, Any]:
    observed = datetime.now(UTC).isoformat()
    try:
        rows = json.loads(cli("kaggle", "quota", "--format", "json"))
        gpu = next(row for row in rows if row["resource"] == "GPU")
        quota = {
            "used": float(gpu["used"].removesuffix("h")),
            "remaining": float(gpu["remaining"].removesuffix("h")),
            "total": float(gpu["total"].removesuffix("h")),
            "renewal": gpu["refreshAt"],
        }
    except (RuntimeError, ValueError, KeyError, StopIteration):
        quota = {"used": None, "remaining": None, "total": None, "renewal": None}
    try:
        listing = csv.DictReader(
            io.StringIO(cli("kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"))
        )
        jobs = []
        for row in listing:
            try:
                state = cli("kaggle", "kernels", "status", row["ref"]).strip()
            except RuntimeError:
                state = "unknown"
            jobs.append({"reference": row["ref"], "status": state})
    except RuntimeError:
        jobs = [{"reference": "unknown", "status": "unknown"}]
    return {
        **quota,
        "units": "Kaggle account GPU-hours",
        "observed_at": observed,
        "source": "authenticated kaggle quota --format json and kernel statuses",
        "active_jobs": [
            j for j in jobs if any(x in j["status"].upper() for x in ("RUNNING", "QUEUED"))
        ],
        "job_statuses": jobs,
    }


def freeze(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    model = config["student"]
    if (
        model["model_id"] != "Qwen/Qwen2.5-Coder-0.5B"
        or model["initializer"] != "untouched_pretrained"
    ):
        raise ValueError("unapproved student initializer")
    if config["teacher"]["label_route_enabled"]:
        raise ValueError("hosted teacher labels are disabled by current output-use terms")
    if config["suite_revision"] != 3:
        raise ValueError("unexpected suite revision")
    files = {name: MODEL / name for name in ("model.safetensors", "tokenizer.json", "config.json")}
    files_record = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        for name, path in files.items()
    }
    if files_record["model.safetensors"]["sha256"] != model["weight_sha256"]:
        raise ValueError("pretrained weight hash mismatch")
    if files_record["tokenizer.json"]["sha256"] != model["tokenizer_sha256"]:
        raise ValueError("tokenizer hash mismatch")
    environment = {
        "controller_host": platform.node(),
        "controller_platform": platform.platform(),
        "python": platform.python_version(),
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "local_cuda_assumed": False,
    }
    path = REPORT / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if (
            plan["config_sha256"] != digest(config_path)
            or plan["schema_sha256"] != digest(SCHEMA)
            or plan["existing_model_files"] != files_record
            or plan["environment"] != environment
        ):
            raise ValueError("frozen identity changed; register a plan/suite revision")
        return plan
    quota = live_quota()
    plan = {
        "plan_revision": 3,
        "suite_revision": 3,
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_commit": config["source_branch_commit"],
        "config_sha256": digest(config_path),
        "schema_sha256": digest(SCHEMA),
        "existing_model_files": files_record,
        "environment": environment,
        "quota_at_freeze": quota,
        "protocol": config["context"],
        "data_policy": config["data"],
        "training": config["training"],
        "selection": config["selection"],
        "budgets": config["budget"],
        "teacher": config["teacher"],
        "final_test_opened": False,
        "automatic_editor_promotion": False,
    }
    atomic_json(path, plan)
    atomic_json(
        REPORT / "quota_ledger.json",
        {
            "observations": [quota],
            "session_wall_seconds": 0,
            "training_input_tokens": 0,
            "teacher_calls": 0,
        },
    )
    return plan


def budget_check(
    plan: dict[str, Any], *, session_seconds: int = 0, training_tokens: int = 0, new_bytes: int = 0
) -> None:
    limits = plan["budgets"]
    ledger = json.loads((REPORT / "quota_ledger.json").read_text())
    if (
        ledger["session_wall_seconds"] + session_seconds
        > limits["maximum_kaggle_t4x2_session_wall_hours"] * 3600
    ):
        raise RuntimeError("campaign GPU wall-hour cap")
    if (
        ledger["training_input_tokens"] + training_tokens
        > limits["maximum_nonpadding_training_input_tokens"]
    ):
        raise RuntimeError("campaign input-token cap")
    artifact_root = ROOT / "artifacts/research/one_line_r1"
    existing_bytes = sum(
        path.stat().st_size
        for path in artifact_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ) if artifact_root.exists() else 0
    if existing_bytes + new_bytes > limits["maximum_new_persistent_local_research_bytes"]:
        raise RuntimeError("campaign local storage cap")
    if new_bytes > shutil.disk_usage(ROOT).free - 2 * 1024**3:
        raise RuntimeError("insufficient filesystem headroom")
    if session_seconds:
        observed = live_quota()
        ledger["observations"].append(observed)
        atomic_json(REPORT / "quota_ledger.json", ledger)
        if observed["active_jobs"]:
            raise RuntimeError("another GPU notebook is active")
        if observed["renewal"] != plan["quota_at_freeze"]["renewal"]:
            raise RuntimeError("quota window changed; no automatic renewed-allocation use")
        if observed["remaining"] is None:
            if ledger["session_wall_seconds"] or session_seconds > 7200:
                raise RuntimeError("unknown quota allows only one initial two-hour session")
        elif observed["remaining"] * 3600 < session_seconds:
            raise RuntimeError("insufficient live Kaggle quota")
        if session_seconds <= limits["minimum_session_finalization_minutes"] * 60:
            raise RuntimeError("session leaves no checkpoint finalization reserve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/one_line_r1.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = freeze(args.config)
    if args.execute:
        # A serious run requires accepted, grouped data and verified restart evidence.
        # The gate is deliberate: a missing shard cannot turn into a toy training run.
        manifest = REPORT / "data_manifest.json"
        if not manifest.exists():
            raise SystemExit("data gate pending: no validated public edit manifest")
        data = json.loads(manifest.read_text())
        if data.get("accepted_train", 0) < plan["data_policy"]["minimum_main_train"]:
            raise SystemExit(
                "data gate pending: fewer than 20,000 distinct accepted training states"
            )
        budget_check(plan, training_tokens=data["planned_training_input_tokens"])
        raise SystemExit("training submission requires a verified Kaggle job bundle")
    print(json.dumps({"plan": str(REPORT / "plan.json"), "suite_revision": plan["suite_revision"]}))


if __name__ == "__main__":
    main()
