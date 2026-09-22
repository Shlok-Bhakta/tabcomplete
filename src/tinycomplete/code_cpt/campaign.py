"""Quota-aware local orchestration for the research-r1 Kaggle campaign."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tinycomplete.observability.context import subprocess_environment
from tinycomplete.observability.hooks import observed
from tinycomplete.observability.runs import observed_run

TOKENS_PER_UPDATE = 32_768
FULL_PILOT_UPDATES = 153
FULL_PILOT_TOKENS = TOKENS_PER_UPDATE * FULL_PILOT_UPDATES
ARM_ORDER = ["C12", "C5", "D12", "D5"]
KERNEL_REF = "shlokbhakta/tabcomplete-code-cpt-campaign-r1"
EVALUATION_KERNEL_REF = "shlokbhakta/tabcomplete-code-cpt-evaluation-r1"
LONG_CONTEXT_KERNEL_REF = "shlokbhakta/tabcomplete-code-cpt-long-context-r1"


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
    campaign_gpu_wall_left = max(0.0, limits.max_gpu_hours / 2.0 - campaign_wall_hours_used)
    quota_wall_left = max(0.0, quota_remaining_gpu_hours / limits.quota_hours_per_wall_hour)
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


@observed("campaign.phase")
def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    merged_environment = os.environ.copy()
    if environment:
        merged_environment.update(environment)
    # Keep the scientific worker environment; add only owned correlation fields.
    merged_environment.update(subprocess_environment({}))
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        env=merged_environment,
    )
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


def _find_output_file(root: Path, name: str) -> Path:
    """Select the top-level kernel artifact, not a copy inside its cloned repository."""
    matches = list(root.rglob(name))
    if not matches:
        raise RuntimeError(f"missing {name} below {root}")
    depths = {path: len(path.relative_to(root).parts) for path in matches}
    minimum = min(depths.values())
    shallowest = [path for path, depth in depths.items() if depth == minimum]
    if len(shallowest) != 1:
        raise RuntimeError(f"ambiguous top-level {name}: {shallowest}")
    return shallowest[0]


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
            quota_hours_per_wall_hour=float(raw["kaggle_quota_hours_per_t4x2_wall_hour"]),
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
        campaign_status = self.kernel_status(KERNEL_REF)
        reusing_campaign = any(
            state in campaign_status for state in ("COMPLETE", "RUNNING", "QUEUED")
        )
        if not reusing_campaign and (
            plan["arms"] != ARM_ORDER or plan["updates_per_arm"] != FULL_PILOT_UPDATES
        ):
            raise RuntimeError(
                f"uploaded kernel requires the full design, affordable plan is {plan}"
            )
        record = {
            "quota_before": quota,
            "plan": plan,
            "existing_campaign_status": campaign_status,
            "reusing_campaign": reusing_campaign,
        }
        (self.state_dir / "submission_plan.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return record

    def kernel_status(self, reference: str) -> str:
        process = subprocess.run(
            ["kaggle", "kernels", "status", reference], text=True, capture_output=True
        )
        if process.returncode:
            return "UNAVAILABLE"
        return (process.stdout + process.stderr).strip()

    def ensure_submitted(self, reference: str, source_dir: Path) -> str:
        status = self.kernel_status(reference)
        if "COMPLETE" in status:
            return "reused-complete"
        if "RUNNING" in status or "QUEUED" in status:
            return "reused-active"
        _run(
            ["kaggle", "kernels", "push", "-p", str(source_dir)],
            cwd=self.repository,
        )
        return "submitted"

    def submit(self) -> str:
        return self.ensure_submitted(
            KERNEL_REF, self.repository / "kaggle" / "code_cpt_campaign_r1"
        )

    def poll_reference(
        self,
        reference: str,
        *,
        deadline_seconds: float,
        status_name: str,
    ) -> str:
        started = time.monotonic()
        while True:
            status = _run(["kaggle", "kernels", "status", reference]).strip()
            (self.state_dir / status_name).write_text(status + "\n", encoding="utf-8")
            if "COMPLETE" in status:
                return status
            if "ERROR" in status or "CANCEL" in status:
                raise RuntimeError(status)
            if time.monotonic() - started > deadline_seconds:
                raise TimeoutError(
                    "campaign polling deadline reached; Kaggle kernel remains external"
                )
            time.sleep(30)

    def poll(self, *, deadline_seconds: float = 10 * 60 * 60) -> str:
        return self.poll_reference(
            KERNEL_REF,
            deadline_seconds=deadline_seconds,
            status_name="campaign_status.txt",
        )

    def collect_reference(self, reference: str, destination_name: str) -> Path:
        destination = self.state_dir / destination_name
        destination.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "kernels", "output", reference, "-p", str(destination), "--quiet"])
        return destination

    def collect(self) -> Path:
        return self.collect_reference(KERNEL_REF, "kaggle_output")

    def verify(self, destination: Path) -> dict[str, Any]:
        progress_path = _find_output_file(destination, "campaign_progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
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
            if not candidates:
                raise RuntimeError(f"missing final model for {arm['name']}")
            model_path = min(candidates, key=lambda path: len(path.relative_to(destination).parts))
            actual = _sha256(model_path)
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

    def verify_evaluation(self, destination: Path) -> dict[str, Any]:
        progress_path = _find_output_file(destination, "evaluation_progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("status") != "complete":
            raise RuntimeError(f"evaluation artifact is not complete: {progress.get('status')}")
        selection_path = progress_path.parent / "selection.json"
        if not selection_path.exists():
            raise RuntimeError("evaluation artifact has no selection.json")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        predictions = {}
        for row in progress["causal_predictions"]:
            if row["status"] != "complete":
                continue
            path = progress_path.parent / "causal_predictions" / f"{row['model']}.jsonl"
            if not path.exists():
                raise RuntimeError(f"missing causal predictions for {row['model']}")
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
            if count != 200:
                raise RuntimeError(f"expected 200 predictions for {row['model']}, found {count}")
            predictions[row["model"]] = path
        if len(predictions) != 6:
            raise RuntimeError(f"expected six complete causal prediction sets, found {predictions}")
        return {
            "progress_path": str(progress_path),
            "selection_path": str(selection_path),
            "selection": selection,
            "predictions": {label: str(path) for label, path in predictions.items()},
        }

    def run_functional_evaluations(self, evaluation: dict[str, Any]) -> Path:
        output_root = self.state_dir / "causal_functional"
        environment = {"PYTHONPATH": str(self.repository / "src")}
        for label, prediction_path in evaluation["predictions"].items():
            output = output_root / label
            summary = output / "summary.json"
            if summary.exists():
                continue
            output.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    sys.executable,
                    str(self.repository / "scripts" / "evaluate_code_benchmark.py"),
                    "--suite",
                    str(self.repository / "data" / "benchmarks" / "code_completion_v2.jsonl"),
                    "--predictions",
                    prediction_path,
                    "--backend",
                    "container",
                    "--workers",
                    "2",
                    "--output-dir",
                    str(output),
                ],
                cwd=self.repository,
                environment=environment,
            )
        return output_root

    def verify_long_context(self, destination: Path) -> dict[str, Any]:
        progress_path = _find_output_file(destination, "progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("status") not in {"complete", "partial"}:
            raise RuntimeError(f"long-context artifact is not finalized: {progress.get('status')}")
        scores = sorted((progress_path.parent / "results").glob("*.jsonl"))
        if not scores:
            raise RuntimeError("long-context artifact has no completed score files")
        return {
            "progress_path": str(progress_path),
            "status": progress["status"],
            "scores": [str(path) for path in scores],
            "failed": progress.get("failed", []),
            "skipped": progress.get("skipped", []),
        }

    def finalize_long_context(self, verification: dict[str, Any]) -> Path:
        progress_path = Path(verification["progress_path"])
        suite = progress_path.parent / "long_context_v2.jsonl"
        output_root = self.state_dir / "long_context_functional"
        environment = {"PYTHONPATH": str(self.repository / "src")}
        for score_name in verification["scores"]:
            score = Path(score_name)
            output = output_root / score.stem
            summary = output / "summary.json"
            if summary.exists():
                continue
            _run(
                [
                    sys.executable,
                    str(self.repository / "scripts" / "evaluate_long_context_generations.py"),
                    "--suite",
                    str(suite),
                    "--scores",
                    str(score),
                    "--output-dir",
                    str(output),
                    "--workers",
                    "2",
                ],
                cwd=self.repository,
                environment=environment,
            )
        summary = self.state_dir / "long_context_summary.json"
        _run(
            [
                sys.executable,
                str(self.repository / "scripts" / "summarize_long_context_diagnostic.py"),
                "--scores",
                *verification["scores"],
                "--output",
                str(summary),
            ],
            cwd=self.repository,
            environment=environment,
        )
        return summary

    def decide_extension(
        self,
        *,
        campaign: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        selection = evaluation["selection"]
        selected = selection.get("selected_candidate")
        if selected is None:
            return {
                "eligible": False,
                "candidate": None,
                "reason": f"development result is {selection.get('result')}",
            }
        return {
            "eligible": False,
            "candidate": selected,
            "reason": (
                "pilot exports are reloadable inference artifacts but contain no published "
                "optimizer training state; cross-process extension would reset or replay"
            ),
            "verified_campaign_models": campaign["verified_models"],
        }

    def finalize_analysis(
        self,
        *,
        campaign_destination: Path,
        evaluation: dict[str, Any],
        functional_root: Path,
        long_context_summary: Path,
    ) -> Path:
        campaign_progress = _find_output_file(campaign_destination, "campaign_progress.json")
        output = self.state_dir / "plots"
        command = [
            "uv",
            "run",
            "--with",
            "matplotlib",
            "python",
            str(self.repository / "scripts" / "analyze_cpt_campaign.py"),
            "--campaign-root",
            str(campaign_progress.parent),
            "--selection",
            evaluation["selection_path"],
            "--functional-root",
            str(functional_root),
            "--long-context-summary",
            str(long_context_summary),
            "--quota-observations",
            str(
                self.repository / "reports" / "code_cpt" / "research_r1" / "quota_observations.json"
            ),
            "--output-dir",
            str(output),
        ]
        _run(command, cwd=self.repository)
        return output

    @observed_run(lambda self: self.state_dir / "observability-run.json", "campaign")
    def execute(self) -> dict[str, Any]:
        plan = self.validate()
        campaign_submission = self.submit()
        self.poll()
        destination = self.collect()
        verification = self.verify(destination)
        evaluation_submission = self.ensure_submitted(
            EVALUATION_KERNEL_REF,
            self.repository / "kaggle" / "code_cpt_evaluation_r1",
        )
        self.poll_reference(
            EVALUATION_KERNEL_REF,
            deadline_seconds=6 * 60 * 60,
            status_name="evaluation_status.txt",
        )
        evaluation_destination = self.collect_reference(EVALUATION_KERNEL_REF, "evaluation_output")
        evaluation = self.verify_evaluation(evaluation_destination)
        functional_root = self.run_functional_evaluations(evaluation)
        long_context_submission = self.ensure_submitted(
            LONG_CONTEXT_KERNEL_REF,
            self.repository / "kaggle" / "code_cpt_long_context_r1",
        )
        self.poll_reference(
            LONG_CONTEXT_KERNEL_REF,
            deadline_seconds=4 * 60 * 60,
            status_name="long_context_status.txt",
        )
        long_context_destination = self.collect_reference(
            LONG_CONTEXT_KERNEL_REF, "long_context_output"
        )
        long_context = self.verify_long_context(long_context_destination)
        long_context_summary = self.finalize_long_context(long_context)
        extension = self.decide_extension(
            campaign=verification,
            evaluation=evaluation,
        )
        plots = self.finalize_analysis(
            campaign_destination=destination,
            evaluation=evaluation,
            functional_root=functional_root,
            long_context_summary=long_context_summary,
        )
        result = {
            "submission": plan,
            "stages": {
                "campaign": campaign_submission,
                "evaluation": evaluation_submission,
                "long_context": long_context_submission,
            },
            "verification": verification,
            "evaluation": evaluation,
            "long_context": long_context,
            "extension": extension,
            "plots": str(plots),
            "quota_after": read_kaggle_quota(),
        }
        (self.state_dir / "orchestrator_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
