"""Quota-aware local orchestration for the research-r1 Kaggle campaign."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

TOKENS_PER_UPDATE = 32_768
FULL_PILOT_UPDATES = 153
FULL_PILOT_TOKENS = TOKENS_PER_UPDATE * FULL_PILOT_UPDATES
ARM_ORDER = ["C12", "C5", "D12", "D5"]
KERNEL_REF = "shlokbhakta/tabcomplete-code-cpt-campaign-r1"


@dataclass(frozen=True)
class CampaignLimits:
    max_wall_hours: float
    max_gpu_hours: float
    max_training_tokens: int
    reserve_minutes: int
    quota_hours_per_wall_hour: float = 1.0


def affordable_pilot_plan(
    *,
    limits: CampaignLimits,
    quota_remaining_gpu_hours: float,
    campaign_wall_hours_used: float,
    estimated_seconds_per_full_arm: float,
) -> dict[str, Any]:
    """Choose four arms, a matched constant pair, or an equal shorter pair."""
    if limits.quota_hours_per_wall_hour <= 0:
        raise ValueError("quota-hours per wall-hour must be positive")
    campaign_wall_left = max(0.0, limits.max_wall_hours - campaign_wall_hours_used)
    campaign_gpu_wall_left = max(
        0.0, limits.max_gpu_hours / 2.0 - campaign_wall_hours_used
    )
    quota_wall_left = max(
        0.0, quota_remaining_gpu_hours / limits.quota_hours_per_wall_hour
    )
    usable_seconds = max(
        0.0,
        min(campaign_wall_left, campaign_gpu_wall_left, quota_wall_left) * 3600
        - limits.reserve_minutes * 60,
    )
    token_arm_limit = limits.max_training_tokens // FULL_PILOT_TOKENS
    time_arm_limit = int(usable_seconds // estimated_seconds_per_full_arm)
    full_arm_count = min(token_arm_limit, time_arm_limit, len(ARM_ORDER))
    if full_arm_count >= 4:
        arms = ARM_ORDER
        updates = FULL_PILOT_UPDATES
    elif full_arm_count >= 2:
        arms = ARM_ORDER[:2]
        updates = FULL_PILOT_UPDATES
    else:
        arms = ARM_ORDER[:2]
        update_time = estimated_seconds_per_full_arm / FULL_PILOT_UPDATES
        updates = min(
            FULL_PILOT_UPDATES,
            int(usable_seconds // (2 * update_time)),
            limits.max_training_tokens // (2 * TOKENS_PER_UPDATE),
        )
        if updates < 61:
            return {
                "arms": [],
                "updates_per_arm": 0,
                "new_training_tokens": 0,
                "usable_wall_seconds": usable_seconds,
                "reason": "a useful matched pair does not fit",
            }
    return {
        "arms": arms,
        "updates_per_arm": updates,
        "new_training_tokens": len(arms) * updates * TOKENS_PER_UPDATE,
        "usable_wall_seconds": usable_seconds,
        "reason": "full four-arm design" if len(arms) == 4 else "matched constant pair",
    }


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if process.returncode:
        message = (process.stderr or process.stdout).strip()
        raise RuntimeError(f"command failed ({command[:3]}): {message[:500]}")
    return process.stdout


def read_kaggle_quota() -> dict[str, Any]:
    rows = json.loads(_run(["kaggle", "quota", "--format", "json"]))
    gpu = next(row for row in rows if row["resource"] == "GPU")
    return {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "used_gpu_hours": float(gpu["used"].removesuffix("h")),
        "remaining_gpu_hours": float(gpu["remaining"].removesuffix("h")),
        "total_gpu_hours": float(gpu["total"].removesuffix("h")),
        "refresh_at": gpu.get("refreshAt"),
        "source": "authenticated `kaggle quota --format json`",
    }


def load_campaign_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value["runtime"]["input_tokens_per_update"] != TOKENS_PER_UPDATE:
        raise ValueError("campaign update geometry differs from the verified runtime")
    if value["pilot"]["order"] != ARM_ORDER:
        raise ValueError("campaign arm order differs from the frozen design")
    if value["pilot"]["additional_input_tokens"] != FULL_PILOT_TOKENS:
        raise ValueError("campaign pilot token count is not 153 complete updates")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CampaignOrchestrator:
    def __init__(self, repository: Path, config_path: Path):
        self.repository = repository.resolve()
        self.config_path = config_path.resolve()
        self.config = load_campaign_config(self.config_path)
        self.state_dir = self.repository / "artifacts" / "code_cpt" / "research_r1"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def limits(self) -> CampaignLimits:
        raw = self.config["limits"]
        return CampaignLimits(
            max_wall_hours=float(raw["max_t4x2_wall_hours"]),
            max_gpu_hours=float(raw["max_gpu_hours"]),
            max_training_tokens=int(raw["max_new_training_input_tokens"]),
            reserve_minutes=int(raw["finalization_reserve_minutes"]),
            quota_hours_per_wall_hour=float(
                raw["kaggle_quota_hours_per_t4x2_wall_hour"]
            ),
        )

    def validate(self) -> dict[str, Any]:
        branch = _run(["git", "branch", "--show-current"], cwd=self.repository).strip()
        if branch != "stage1/cpt-recipe-r1":
            raise RuntimeError(f"expected stage1/cpt-recipe-r1, found {branch}")
        if self.config["limits"].get("paid_compute") is not False:
            raise ValueError("paid compute must remain disabled")
        quota = read_kaggle_quota()
        plan = affordable_pilot_plan(
            limits=self.limits,
            quota_remaining_gpu_hours=quota["remaining_gpu_hours"],
            campaign_wall_hours_used=0.20,
            estimated_seconds_per_full_arm=4_200,
        )
        if plan["arms"] != ARM_ORDER or plan["updates_per_arm"] != FULL_PILOT_UPDATES:
            raise RuntimeError(
                f"uploaded kernel requires the full design, affordable plan is {plan}"
            )
        record = {"quota_before": quota, "plan": plan}
        (self.state_dir / "submission_plan.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return record

    def submit(self) -> None:
        process = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_REF], text=True, capture_output=True
        )
        status = process.stdout + process.stderr
        if process.returncode == 0 and ("RUNNING" in status or "QUEUED" in status):
            raise RuntimeError(f"campaign kernel is already active: {status.strip()}")
        _run(
            ["kaggle", "kernels", "push", "-p", "kaggle/code_cpt_campaign_r1"],
            cwd=self.repository,
        )

    def poll(self, *, deadline_seconds: float = 10 * 60 * 60) -> str:
        started = time.monotonic()
        while True:
            status = _run(["kaggle", "kernels", "status", KERNEL_REF]).strip()
            (self.state_dir / "last_status.txt").write_text(status + "\n", encoding="utf-8")
            if "COMPLETE" in status:
                return status
            if "ERROR" in status or "CANCEL" in status:
                raise RuntimeError(status)
            if time.monotonic() - started > deadline_seconds:
                raise TimeoutError(
                    "campaign polling deadline reached; Kaggle kernel remains external"
                )
            time.sleep(30)

    def collect(self) -> Path:
        destination = self.state_dir / "kaggle_output"
        destination.mkdir(parents=True, exist_ok=True)
        _run(
            ["kaggle", "kernels", "output", KERNEL_REF, "-p", str(destination), "--quiet"]
        )
        return destination

    def verify(self, destination: Path) -> dict[str, Any]:
        progress_paths = list(destination.rglob("campaign_progress.json"))
        if len(progress_paths) != 1:
            raise RuntimeError(f"expected one campaign progress file, found {progress_paths}")
        progress = json.loads(progress_paths[0].read_text(encoding="utf-8"))
        if progress["status"] != "complete":
            raise RuntimeError(f"campaign artifact is not complete: {progress['status']}")
        if len(progress["completed_arms"]) > 4:
            raise RuntimeError("campaign exceeded the four declared pilot arms")
        total_tokens = sum(row["additional_input_tokens"] for row in progress["completed_arms"])
        if total_tokens > self.limits.max_training_tokens:
            raise RuntimeError("campaign exceeded its token cap")
        verified = []
        for arm in progress["completed_arms"]:
            candidates = list(destination.rglob(f"arms/{arm['name']}/final/model.safetensors"))
            if len(candidates) != 1:
                raise RuntimeError(f"missing final model for {arm['name']}")
            actual = _sha256(candidates[0])
            if actual != arm["model_sha256"]:
                raise RuntimeError(f"model hash mismatch for {arm['name']}")
            verified.append({"arm": arm["name"], "sha256": actual})
        result = {
            "verified_models": verified,
            "total_training_tokens": total_tokens,
            "quota_after": read_kaggle_quota(),
            "extension_eligible": False,
            "extension_reason": (
                "The bounded pilot artifact is valid, but corrected causal functional and "
                "long-context gates must run before an extension."
            ),
        }
        (self.state_dir / "verification.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    def execute(self) -> dict[str, Any]:
        plan = self.validate()
        self.submit()
        self.poll()
        destination = self.collect()
        verification = self.verify(destination)
        result = {"submission": plan, "verification": verification}
        (self.state_dir / "orchestrator_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
