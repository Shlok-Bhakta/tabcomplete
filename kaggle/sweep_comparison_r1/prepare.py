"""Prepare or submit one private Sweep benchmark using existing Kaggle helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
from build_pilot import _job_statuses_verified, live_quota  # noqa: E402

SPEC = importlib.util.spec_from_file_location("sweep_worker", Path(__file__).with_name("run.py"))
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
DATASET_ID = "shlokbhakta/tabcomplete-sweep-comparison-inputs-r1"
KERNEL_ID = "shlokbhakta/tabcomplete-sweep-comparison-r1"
KERNEL_RETRY_ID = KERNEL_ID + "-retry"
KERNEL_VERIFIED_ID = KERNEL_ID + "-verified"
KERNEL_FINAL_ID = KERNEL_ID + "-final"
MAX_KERNEL_ATTEMPTS = 4
LEGACY_MAX_KERNEL_ATTEMPTS = 3
FIFTH_ATTEMPT_MAX_KERNEL_ATTEMPTS = 5
SESSION_SECONDS = 7200
FIFTH_SESSION_SECONDS = 4800
RESERVE_SECONDS = 1200
AGGREGATE_WALL_SECONDS = 14400
LINE_SHA = "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead"


def cli(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, timeout=900)
    if result.returncode:
        raise RuntimeError("Sweep preparation command failed; no automatic retry")
    return result.stdout.strip()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def checked_quota(expected_renewal: str) -> dict:
    value = live_quota()
    if (
        not _job_statuses_verified(value) or value.get("active_jobs")
        or value.get("renewal") != expected_renewal
        or value.get("remaining") is None or float(value["remaining"]) < 4
    ):
        raise RuntimeError("Sweep authenticated quota/job/renewal gate failed")
    return value


def _campaign_attempt_cap(attempt_number: int) -> int:
    if not isinstance(attempt_number, int) or isinstance(attempt_number, bool):
        raise ValueError("Sweep attempt number must be an integer")
    if attempt_number < 1 or attempt_number > FIFTH_ATTEMPT_MAX_KERNEL_ATTEMPTS:
        raise ValueError("Sweep attempt number is outside the five-allocation cap")
    if attempt_number == FIFTH_ATTEMPT_MAX_KERNEL_ATTEMPTS:
        return FIFTH_ATTEMPT_MAX_KERNEL_ATTEMPTS
    if attempt_number == MAX_KERNEL_ATTEMPTS:
        return MAX_KERNEL_ATTEMPTS
    return LEGACY_MAX_KERNEL_ATTEMPTS


def _session_seconds_for_attempt(attempt_number: int) -> int:
    if attempt_number == FIFTH_ATTEMPT_MAX_KERNEL_ATTEMPTS:
        return FIFTH_SESSION_SECONDS
    if 1 <= attempt_number <= MAX_KERNEL_ATTEMPTS:
        return SESSION_SECONDS
    raise ValueError("Sweep allocation attempt number is outside the approved range")


def _finite_wall_sum(prior_attempts: list[dict]) -> float:
    total = 0.0
    for row in prior_attempts:
        value = row.get("conservative_wall_upper_bound_seconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Sweep prior wall-time evidence is invalid")
        seconds = float(value)
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("Sweep prior wall-time evidence is invalid")
        total += seconds
    return total


def _validate_third_attempt_amendment(plan: dict, prior_attempts: list[dict]) -> None:
    amendment = plan.get("allocation_amendment_v6")
    if (
        not isinstance(amendment, dict)
        or amendment.get("schema") != "sweep-allocation-amendment-v6"
    ):
        raise ValueError("third Sweep attempt requires frozen plan-v6 allocation amendment")
    if len(prior_attempts) != 2 or [row.get("attempt_number") for row in prior_attempts] != [1, 2]:
        raise ValueError("third Sweep attempt requires the two verified earlier allocations")
    if any(row.get("state") != "ERROR" for row in prior_attempts):
        raise ValueError(
            "third Sweep attempt requires both earlier allocations to be terminal errors"
        )
    prior_wall = _finite_wall_sum(prior_attempts)
    if prior_wall + SESSION_SECONDS > AGGREGATE_WALL_SECONDS:
        raise ValueError("third Sweep attempt would exceed its aggregate wall-time cap")
    limits = amendment.get("attempt_limits")
    if not isinstance(limits, dict) or limits != {
        "maximum_total_attempts": LEGACY_MAX_KERNEL_ATTEMPTS,
        "attempts_already_used": 2,
        "additional_attempts_remaining": 1,
        "per_attempt_session_wall_seconds_max": SESSION_SECONDS,
        "campaign_aggregate_gpu_session_wall_seconds_max": AGGREGATE_WALL_SECONDS,
        "third_attempt_allowed": True,
    }:
        raise ValueError(
            "frozen plan-v6 allocation limits do not authorize one bounded third attempt"
        )
    recorded = amendment.get("verified_prior_attempts")
    if not isinstance(recorded, list) or len(recorded) != len(prior_attempts):
        raise ValueError("frozen plan-v6 lacks both verified prior-attempt observations")
    keys = (
        "attempt_number", "kernel_id", "state", "quota_observed_at",
        "terminal_observed_at", "conservative_wall_upper_bound_seconds",
        "terminal_observation_sha256",
    )
    for expected, actual in zip(prior_attempts, recorded, strict=True):
        if not isinstance(actual, dict) or any(actual.get(key) != expected[key] for key in keys):
            raise ValueError(
                "frozen plan-v6 prior-attempt evidence differs from authenticated status checks"
            )
    if amendment.get("aggregate_wall_upper_bound_with_third_attempt_seconds") != round(
        prior_wall + SESSION_SECONDS, 6
    ):
        raise ValueError("frozen plan-v6 aggregate wall-time bound is inconsistent")
    source_identity = amendment.get("allocation_source_identity")
    source_paths = {
        "prepare_sha256": Path(__file__),
        "prepare_test_sha256": ROOT / "tests/test_sweep_kaggle_submission.py",
        "worker_sha256": Path(__file__).with_name("run.py"),
        "worker_test_sha256": ROOT / "tests/test_sweep_kaggle_worker.py",
    }
    if not isinstance(source_identity, dict) or any(
        source_identity.get(key) != worker.digest(path)
        for key, path in source_paths.items()
    ):
        raise ValueError("frozen plan-v6 allocation helper source identity changed")


def _validate_fourth_attempt_amendment(plan: dict, prior_attempts: list[dict]) -> None:
    amendment = plan.get("allocation_amendment_v8")
    if (
        not isinstance(amendment, dict)
        or amendment.get("schema") != "sweep-allocation-amendment-v8"
        or amendment.get("number") != 8
    ):
        raise ValueError("fourth Sweep attempt requires frozen plan-v8 allocation amendment")
    if (
        len(prior_attempts) != 3
        or [row.get("attempt_number") for row in prior_attempts] != [1, 2, 3]
    ):
        raise ValueError("fourth Sweep attempt requires three verified earlier allocations")
    expected_ids = [kernel_identity(attempt)[0] for attempt in (1, 2, 3)]
    if [row.get("kernel_id") for row in prior_attempts] != expected_ids:
        raise ValueError("fourth Sweep attempt prior kernel identities are not sequential")
    if any(row.get("state") != "ERROR" for row in prior_attempts):
        raise ValueError(
            "fourth Sweep attempt requires all three allocations to be terminal errors"
        )

    prior_wall = _finite_wall_sum(prior_attempts)
    planned_total = prior_wall + SESSION_SECONDS
    margin = AGGREGATE_WALL_SECONDS - planned_total
    if margin < 0:
        raise ValueError("fourth Sweep attempt would exceed its aggregate wall-time cap")

    limits = amendment.get("attempt_limits")
    if not isinstance(limits, dict) or limits != {
        "maximum_total_attempts": MAX_KERNEL_ATTEMPTS,
        "attempts_already_used": 3,
        "additional_attempts_remaining": 1,
        "per_attempt_session_wall_seconds_max": SESSION_SECONDS,
        "campaign_aggregate_gpu_session_wall_seconds_max": AGGREGATE_WALL_SECONDS,
        "fourth_attempt_allowed": True,
    }:
        raise ValueError(
            "frozen plan-v8 allocation limits do not authorize one bounded fourth attempt"
        )

    recorded = amendment.get("verified_prior_attempts")
    if not isinstance(recorded, list) or len(recorded) != 3:
        raise ValueError("frozen plan-v8 lacks all three verified prior-attempt observations")
    keys = (
        "attempt_number", "kernel_id", "state", "quota_observed_at",
        "terminal_observed_at", "conservative_wall_upper_bound_seconds",
        "terminal_observation_sha256",
    )
    for expected, actual in zip(prior_attempts, recorded, strict=True):
        if not isinstance(actual, dict) or any(actual.get(key) != expected[key] for key in keys):
            raise ValueError(
                "frozen plan-v8 prior-attempt evidence differs from authenticated status checks"
            )

    expected_total = round(planned_total, 6)
    expected_margin = round(margin, 6)
    if (
        amendment.get("aggregate_wall_upper_bound_with_fourth_attempt_seconds")
        != expected_total
        or amendment.get("remaining_aggregate_wall_margin_seconds") != expected_margin
        or expected_margin < 0
    ):
        raise ValueError("frozen plan-v8 aggregate wall-time bound is inconsistent")

    previous_path = ROOT / "reports/prototype/sweep_comparison_r1/plan-v7.json"
    previous_plan = json.loads(previous_path.read_text())
    previous_payload = {key: value for key, value in previous_plan.items()
                        if key != "plan_sha256"}
    previous_digest = hashlib.sha256((json.dumps(
        previous_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()).hexdigest()
    if previous_plan.get("plan_sha256") != previous_digest:
        raise ValueError("frozen plan-v7 comparison identity is invalid")
    if (
        amendment.get("supersedes_plan_sha256") != previous_plan.get("plan_sha256")
        or amendment.get("supersedes_plan_file_sha256") != worker.digest(previous_path)
    ):
        raise ValueError("plan-v8 does not preserve the frozen plan-v7 comparison")
    comparison_payload = {
        key: value for key, value in plan.items()
        if key not in {"plan_sha256", "allocation_amendment_v8"}
    }
    expected_payload = dict(previous_payload)
    expected_code = dict(expected_payload["code"])
    expected_code["runner_sha256"] = plan["code"]["runner_sha256"]
    expected_code["test_sha256"] = plan["code"]["test_sha256"]
    expected_payload["code"] = expected_code
    if comparison_payload != expected_payload:
        raise ValueError("plan-v8 changed frozen plan-v7 scientific inputs")

    source_identity = amendment.get("allocation_source_identity")
    source_paths = {
        "prepare_sha256": Path(__file__),
        "prepare_test_sha256": ROOT / "tests/test_sweep_kaggle_submission.py",
        "worker_sha256": Path(__file__).with_name("run.py"),
        "worker_test_sha256": ROOT / "tests/test_sweep_kaggle_worker.py",
    }
    if not isinstance(source_identity, dict) or any(
        source_identity.get(key) != worker.digest(path)
        for key, path in source_paths.items()
    ):
        raise ValueError("frozen plan-v8 allocation helper source identity changed")


def _validate_fifth_attempt_amendment(plan: dict, prior_attempts: list[dict]) -> None:
    amendment = plan.get("allocation_amendment_v9")
    if (
        not isinstance(amendment, dict)
        or amendment.get("schema") != "sweep-allocation-amendment-v9"
        or amendment.get("number") != 9
    ):
        raise ValueError("fifth Sweep attempt requires frozen plan-v9 allocation amendment")
    if (
        len(prior_attempts) != 4
        or [row.get("attempt_number") for row in prior_attempts] != [1, 2, 3, 4]
    ):
        raise ValueError("fifth Sweep attempt requires four verified earlier allocations")
    expected_ids = [kernel_identity(attempt)[0] for attempt in (1, 2, 3, 4)]
    if [row.get("kernel_id") for row in prior_attempts] != expected_ids:
        raise ValueError("fifth Sweep attempt prior kernel identities are not sequential")
    if any(row.get("state") != "ERROR" for row in prior_attempts):
        raise ValueError(
            "fifth Sweep attempt requires all four allocations to be terminal errors"
        )

    prior_wall = _finite_wall_sum(prior_attempts)
    planned_session = _session_seconds_for_attempt(5)
    planned_total = prior_wall + planned_session
    margin = AGGREGATE_WALL_SECONDS - planned_total
    if margin < 0:
        raise ValueError("fifth Sweep attempt would exceed its aggregate wall-time cap")

    limits = amendment.get("attempt_limits")
    if not isinstance(limits, dict) or limits != {
        "maximum_total_attempts": FIFTH_ATTEMPT_MAX_KERNEL_ATTEMPTS,
        "attempts_already_used": 4,
        "additional_attempts_remaining": 1,
        "per_attempt_session_wall_seconds_max": FIFTH_SESSION_SECONDS,
        "finalization_reserve_seconds": RESERVE_SECONDS,
        "maximum_work_seconds": FIFTH_SESSION_SECONDS - RESERVE_SECONDS,
        "campaign_aggregate_gpu_session_wall_seconds_max": AGGREGATE_WALL_SECONDS,
        "fifth_attempt_allowed": True,
    }:
        raise ValueError(
            "frozen plan-v9 allocation limits do not authorize one bounded fifth attempt"
        )

    recorded = amendment.get("verified_prior_attempts")
    if not isinstance(recorded, list) or len(recorded) != 4:
        raise ValueError("frozen plan-v9 lacks all four verified prior-attempt observations")
    keys = (
        "attempt_number", "kernel_id", "state", "quota_observed_at",
        "terminal_observed_at", "conservative_wall_upper_bound_seconds",
        "terminal_observation_sha256",
    )
    for expected, actual in zip(prior_attempts, recorded, strict=True):
        if not isinstance(actual, dict) or any(actual.get(key) != expected[key] for key in keys):
            raise ValueError(
                "frozen plan-v9 prior-attempt evidence differs from authenticated status checks"
            )

    expected_total = round(planned_total, 6)
    expected_margin = round(margin, 6)
    if (
        amendment.get("prior_wall_upper_bound_seconds") != round(prior_wall, 6)
        or amendment.get("aggregate_wall_upper_bound_with_fifth_attempt_seconds")
        != expected_total
        or amendment.get("remaining_aggregate_wall_margin_seconds") != expected_margin
        or expected_margin < 0
    ):
        raise ValueError("frozen plan-v9 aggregate wall-time bound is inconsistent")

    previous_path = ROOT / "reports/prototype/sweep_comparison_r1/plan-v8.json"
    previous_plan = json.loads(previous_path.read_text())
    previous_payload = {
        key: value for key, value in previous_plan.items()
        if key != "plan_sha256"
    }
    previous_digest = hashlib.sha256((json.dumps(
        previous_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()).hexdigest()
    if previous_plan.get("plan_sha256") != previous_digest:
        raise ValueError("frozen plan-v8 comparison identity is invalid")
    if (
        amendment.get("supersedes_plan_sha256") != previous_plan.get("plan_sha256")
        or amendment.get("supersedes_plan_file_sha256") != worker.digest(previous_path)
    ):
        raise ValueError("plan-v9 does not preserve the frozen plan-v8 comparison")

    comparison_payload = {
        key: value for key, value in plan.items()
        if key not in {"plan_sha256", "allocation_amendment_v8", "allocation_amendment_v9"}
    }
    expected_payload = {
        key: value for key, value in previous_payload.items()
        if key != "allocation_amendment_v8"
    }
    expected_code = dict(expected_payload["code"])
    expected_code["runner_sha256"] = plan["code"]["runner_sha256"]
    expected_code["test_sha256"] = plan["code"]["test_sha256"]
    expected_payload["code"] = expected_code
    if comparison_payload != expected_payload:
        raise ValueError("plan-v9 changed frozen plan-v8 scientific inputs")

    source_identity = amendment.get("allocation_source_identity")
    source_paths = {
        "prepare_sha256": Path(__file__),
        "prepare_test_sha256": ROOT / "tests/test_sweep_kaggle_submission.py",
        "worker_sha256": Path(__file__).with_name("run.py"),
        "worker_test_sha256": ROOT / "tests/test_sweep_kaggle_worker.py",
    }
    if not isinstance(source_identity, dict) or any(
        source_identity.get(key) != worker.digest(path)
        for key, path in source_paths.items()
    ):
        raise ValueError("frozen plan-v9 allocation helper source identity changed")


def validate_plan(
    plan: dict,
    runner: Path,
    fixtures: Path,
    *,
    attempt_number: int | None = None,
    prior_attempts: list[dict] | None = None,
) -> None:
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    digest = hashlib.sha256((json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()).hexdigest()
    if plan.get("schema") != "sweep-comparison-plan-v1" or plan.get("plan_sha256") != digest:
        raise ValueError("expected an intact frozen Sweep comparison plan")
    if (plan["code"]["runner_sha256"] != worker.digest(runner)
            or plan["comparison"]["next_edit"]["fixture_input_sha256"] != worker.digest(fixtures)):
        raise ValueError("Sweep frozen source or fixture identity mismatch")
    if attempt_number in {3, 4, 5} and plan["code"].get("test_sha256") != worker.digest(
        ROOT / "tests/test_sweep_comparison.py"
    ):
        raise ValueError("Sweep plan scoring test identity mismatch")
    if attempt_number == 3:
        _validate_third_attempt_amendment(plan, prior_attempts or [])
    elif attempt_number == 4:
        _validate_fourth_attempt_amendment(plan, prior_attempts or [])
    elif attempt_number == 5:
        _validate_fifth_attempt_amendment(plan, prior_attempts or [])


def kernel_identity(attempt: int) -> tuple[str, str]:
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        raise ValueError("Sweep allocation attempt number must be an integer")
    identities = {
        1: (KERNEL_ID, "TabComplete Sweep comparison R1"),
        2: (KERNEL_RETRY_ID, "TabComplete Sweep comparison R1 retry"),
        3: (KERNEL_VERIFIED_ID, "TabComplete Sweep comparison R1 verified"),
        4: (KERNEL_FINAL_ID, "TabComplete Sweep comparison R1 final"),
        5: (KERNEL_ID + "-attempt-5", "TabComplete Sweep comparison R1 attempt 5"),
    }
    try:
        kernel_id, title = identities[attempt]
    except KeyError as exc:
        raise RuntimeError("Sweep campaign maximum of five allocations reached") from exc
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if slug != kernel_id.partition("/")[2]:
        raise ValueError("Sweep kernel title does not match its unique ID slug")
    return kernel_id, title


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"Sweep {label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"Sweep {label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"Sweep {label} timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def _observe_terminal_error(kernel_id: str, path: Path) -> dict:
    response = cli(["kaggle", "kernels", "status", kernel_id])
    statuses = re.findall(r"KernelWorkerStatus\.([A-Z_]+)", response)
    if statuses != ["ERROR"]:
        raise RuntimeError("prior Sweep Kaggle allocation is not verified terminal ERROR")

    observation_path = path / "terminal-observation.json"
    if observation_path.exists():
        observation = json.loads(observation_path.read_text())
        if (
            observation.get("kernel_id") != kernel_id
            or observation.get("state") != "ERROR"
            or observation.get("source") != "authenticated kaggle kernels status"
        ):
            raise RuntimeError("prior Sweep terminal observation is invalid")
        _timestamp(observation.get("observed_at"), "terminal observation")
    else:
        observation = {
            "schema": "sweep-kernel-terminal-observation-v1",
            "kernel_id": kernel_id,
            "state": "ERROR",
            "status_token": "KernelWorkerStatus.ERROR",
            "observed_at": datetime.now(UTC).isoformat(),
            "source": "authenticated kaggle kernels status",
            "status_output_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        }
        write(observation_path, observation)
    return observation


def validate_failed_attempt(previous: Path) -> list[dict]:
    """Verify every prior allocation and bound aggregate wall time before attempt 2/3."""
    reverse_chain: list[tuple[Path, dict, dict]] = []
    seen: set[Path] = set()
    current = previous.resolve()
    while True:
        if current in seen:
            raise RuntimeError("Sweep prior-attempt bundle chain contains a cycle")
        seen.add(current)
        manifest_path = current / "bundle-manifest.json"
        state_path = current / "submission-state.json"
        if not manifest_path.is_file() or not state_path.is_file():
            raise RuntimeError("Sweep prior-attempt evidence is incomplete")
        manifest = json.loads(manifest_path.read_text())
        state = json.loads(state_path.read_text())
        reverse_chain.append((current, manifest, state))
        parent = manifest.get("retry_after")
        if parent is None:
            break
        if not isinstance(parent, str) or not parent:
            raise RuntimeError("Sweep prior-attempt chain link is invalid")
        current = Path(parent).resolve()

    chain = list(reversed(reverse_chain))
    if not 1 <= len(chain) < FIFTH_ATTEMPT_MAX_KERNEL_ATTEMPTS:
        raise RuntimeError("Sweep prior-attempt count is outside the five-allocation cap")

    summaries: list[dict] = []
    campaign_renewal: str | None = None
    for attempt, (bundle_path, manifest, state) in enumerate(chain, start=1):
        expected_id, expected_title = kernel_identity(attempt)
        if state.get("state") != "kernel_pushed" or state.get("kernel_id") != expected_id:
            raise RuntimeError("Sweep prior allocation is not a reconciled kernel push")
        if manifest.get("kernel_id") not in (None, expected_id):
            raise RuntimeError("Sweep prior bundle kernel identity changed")
        metadata_path = bundle_path / "kernel" / "kernel-metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError("Sweep prior kernel metadata is missing")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("id") != expected_id or metadata.get("title") != expected_title:
            raise RuntimeError("Sweep prior kernel ID/title identity mismatch")

        renewal = manifest.get("renewal")
        quota = state.get("quota_before")
        if not isinstance(renewal, str) or not renewal:
            raise RuntimeError("Sweep prior allocation campaign renewal is missing")
        if campaign_renewal is None:
            campaign_renewal = renewal
        if renewal != campaign_renewal:
            raise RuntimeError("Sweep prior allocations crossed a quota renewal")
        if not isinstance(quota, dict) or quota.get("renewal") != renewal:
            raise RuntimeError("Sweep prior quota observation renewal mismatch")
        if quota.get("active_jobs") != []:
            raise RuntimeError("Sweep prior allocation quota snapshot had active jobs")
        started = _timestamp(quota.get("observed_at"), "prior quota observation")
        submitted = _timestamp(state.get("submitted_at"), "prior submission")
        if submitted < started:
            raise RuntimeError("Sweep prior submission predates its quota observation")

        observation = _observe_terminal_error(expected_id, bundle_path)
        terminal_at = _timestamp(observation.get("observed_at"), "terminal observation")
        if terminal_at < submitted:
            raise RuntimeError("Sweep terminal observation predates its submission")
        upper_bound = (terminal_at - started).total_seconds()
        summaries.append({
            "attempt_number": attempt,
            "kernel_id": expected_id,
            "state": "ERROR",
            "quota_observed_at": quota["observed_at"],
            "terminal_observed_at": observation["observed_at"],
            "conservative_wall_upper_bound_seconds": round(upper_bound, 6),
            "terminal_observation_sha256": worker.digest(bundle_path / "terminal-observation.json"),
        })

    prior_bound = _finite_wall_sum(summaries)
    next_session = _session_seconds_for_attempt(len(chain) + 1)
    if prior_bound + next_session > AGGREGATE_WALL_SECONDS:
        raise RuntimeError("next Sweep allocation exceeds the aggregate wall-time budget")
    return summaries


def prepare(
    plan: Path, line_suite: Path, next_edit_fixtures: Path, output: Path, q4_artifact: Path,
    retry_after: Path | None = None
) -> None:
    if output.exists():
        raise FileExistsError("Sweep bundle already exists; inspect before resuming")
    prior_attempts = validate_failed_attempt(retry_after) if retry_after is not None else []
    attempt_number = len(prior_attempts) + 1
    planned_session = _session_seconds_for_attempt(attempt_number)
    kernel_id, kernel_title = kernel_identity(attempt_number)
    prior_wall_bound = _finite_wall_sum(prior_attempts)
    if prior_wall_bound + planned_session > AGGREGATE_WALL_SECONDS:
        raise RuntimeError("planned Sweep allocation exceeds the aggregate wall-time budget")
    if worker.digest(line_suite) != LINE_SHA:
        raise ValueError("corrected 180-case line fixture identity changed")
    if (q4_artifact.stat().st_size != worker.Q4_BYTES
            or worker.digest(q4_artifact) != worker.Q4_SHA256):
        raise ValueError("canonical Q4 artifact identity mismatch")
    commit = cli(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    runner = ROOT / "scripts/run_sweep_comparison.py"
    committed_runner = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", commit + ":scripts/run_sweep_comparison.py"]
    )
    if committed_runner != runner.read_bytes():
        raise ValueError("Sweep runner must be committed before preparing GPU inputs")
    remote = cli(["git", "-C", str(ROOT), "ls-remote", "origin",
                  "refs/heads/prototype/product-r2"]).split()[0]
    if remote != commit:
        raise ValueError("Sweep worker source commit must be pushed")
    plan_value = json.loads(plan.read_text())
    # The serving runner validates its full model/prompt/runtime/fixture plan.
    # This check also catches selecting an unrelated JSON file accidentally.
    validate_plan(
        plan_value, runner, next_edit_fixtures,
        attempt_number=attempt_number, prior_attempts=prior_attempts,
    )

    temporary = output.with_name(output.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError("incomplete Sweep preparation already exists")
    dataset = temporary / "dataset"
    kernel = temporary / "kernel"
    dataset.mkdir(parents=True, mode=0o700)
    kernel.mkdir(mode=0o700)
    shutil.copyfile(plan, dataset / "plan.json")
    shutil.copyfile(line_suite, dataset / "causal_line_v1-r3.jsonl")
    shutil.copyfile(next_edit_fixtures, dataset / "next_edit_inputs.jsonl")
    # Hardlink avoids a second local weight copy; Kaggle receives a private dataset.
    (dataset / worker.Q4_FILE).hardlink_to(q4_artifact)
    spec = {
        "schema": "sweep-comparison-kaggle-input-v1", "commit": commit,
        "model_revision": worker.MODEL_REVISION, "model_sha256": worker.MODEL_SHA256,
        "runtime_revision": worker.RUNTIME_REVISION, "session_seconds": planned_session,
        "reserve_seconds": RESERVE_SECONDS, "training_enabled": False,
        "campaign_attempt_number": attempt_number,
        "campaign_max_attempts": _campaign_attempt_cap(attempt_number),
        "prior_wall_upper_bound_seconds": prior_wall_bound,
        "fifth_attempt_plan_sha256": worker.digest(plan) if attempt_number == 5 else None,
        "campaign_wall_cap_seconds": AGGREGATE_WALL_SECONDS,
        "runner_sha256": worker.digest(runner),
        "files": {name: worker.digest(dataset / name)
                  for name in ("plan.json", "causal_line_v1-r3.jsonl",
                               "next_edit_inputs.jsonl", worker.Q4_FILE)},
        "runner_arguments": [
            "--plan", "{input}/plan.json", "--runtime", "{runtime}",
            "--artifact-dir", "{scratch}", "--artifact-budget-root", "/kaggle/temp",
            "--output", "{output}/results", "--line-suite", "{input}/causal_line_v1-r3.jsonl",
            "--gpu-layers", "99",
            "--next-edit-fixtures", "{input}/next_edit_inputs.jsonl",
        ],
        "runner_modes": ["download", "quantize", "quality", "next-edit"],
    }
    worker.validate_spec(spec)
    if attempt_number == 5:
        worker.validate_fifth_attempt_plan(dataset / "plan.json", spec)
    write(dataset / "sweep-worker-spec.json", spec)
    write(dataset / "dataset-metadata.json", {
        "id": DATASET_ID, "title": "TabComplete Sweep comparison inputs R1",
        "licenses": [{"name": "other"}],
        "description": "Private benchmark fixtures, frozen metadata and the exact Apache-2.0 "
                       "Sweep Q8-to-Q4 derivative used locally. No personal editor data, "
                       "teacher keys or training examples. Not a public model release.",
    })
    shutil.copyfile(Path(__file__).with_name("run.py"), kernel / "run.py")
    write(kernel / "kernel-metadata.json", {
        "id": kernel_id, "title": kernel_title,
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
    })
    write(temporary / "bundle-manifest.json", {
        "commit": commit, "plan_sha256": worker.digest(plan),
        "input_spec_sha256": worker.digest(dataset / "sweep-worker-spec.json"),
        "worker_sha256": worker.digest(kernel / "run.py"),
        "state": "prepared", "renewal": "2026-10-03T00:00:00",
        "retry_after": str(retry_after) if retry_after is not None else None,
        "kernel_id": kernel_id, "kernel_title": kernel_title,
        "attempt_number": attempt_number,
        "maximum_attempts": _campaign_attempt_cap(attempt_number),
        "aggregate_wall_cap_seconds": AGGREGATE_WALL_SECONDS,
        "planned_session_seconds": planned_session,
        "prior_attempts": prior_attempts,
        "prior_wall_upper_bound_seconds": prior_wall_bound,
        "aggregate_wall_upper_bound_with_this_session_seconds": (
            prior_wall_bound + planned_session
        ),
    })
    temporary.rename(output)


def submit(bundle: Path) -> None:
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    attempt_number = manifest.get("attempt_number")
    kernel_id, kernel_title = kernel_identity(attempt_number)
    if manifest.get("kernel_id") != kernel_id or manifest.get("kernel_title") != kernel_title:
        raise ValueError("Sweep prepared kernel ID/title mismatch")
    state_path = bundle / "submission-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"state": "prepared"}
    if state["state"] in {"submission_started", "kernel_pushed"}:
        raise RuntimeError("Sweep GPU submission already attempted; reconcile, never retry blindly")
    retry_after = manifest.get("retry_after")
    prior_attempts = []
    if retry_after is not None:
        prior_attempts = validate_failed_attempt(Path(retry_after))
    if attempt_number != len(prior_attempts) + 1:
        raise ValueError("Sweep attempt number does not match its verified history")
    if manifest.get("prior_attempts") != prior_attempts:
        raise ValueError("Sweep prior-attempt evidence changed after preparation")
    prior_wall_bound = _finite_wall_sum(prior_attempts)
    campaign_attempt_cap = _campaign_attempt_cap(attempt_number)
    planned_session = _session_seconds_for_attempt(attempt_number)
    planned_seconds = manifest.get("planned_session_seconds", planned_session)
    if (
        not isinstance(planned_seconds, int) or planned_seconds != planned_session
        or isinstance(planned_seconds, bool)
        or prior_wall_bound + planned_seconds > AGGREGATE_WALL_SECONDS
        or manifest.get("maximum_attempts") != campaign_attempt_cap
        or manifest.get("aggregate_wall_cap_seconds") != AGGREGATE_WALL_SECONDS
        or manifest.get("prior_wall_upper_bound_seconds") != prior_wall_bound
        or manifest.get("aggregate_wall_upper_bound_with_this_session_seconds")
        != prior_wall_bound + planned_seconds
    ):
        raise RuntimeError("Sweep campaign allocation or aggregate wall budget is invalid")
    if (
        worker.digest(bundle / "dataset/plan.json") != manifest["plan_sha256"]
        or worker.digest(bundle / "dataset/sweep-worker-spec.json") != manifest["input_spec_sha256"]
        or worker.digest(bundle / "kernel/run.py") != manifest["worker_sha256"]
    ):
        raise ValueError("Sweep staged bundle was modified")
    spec = json.loads((bundle / "dataset/sweep-worker-spec.json").read_text())
    worker.validate_spec(spec)
    if (
        spec.get("campaign_attempt_number") != attempt_number
        or spec.get("campaign_max_attempts") != campaign_attempt_cap
        or spec.get("prior_wall_upper_bound_seconds") != prior_wall_bound
        or spec.get("campaign_wall_cap_seconds") != AGGREGATE_WALL_SECONDS
        or spec.get("session_seconds") != planned_session
        or spec.get("reserve_seconds") != RESERVE_SECONDS
        or spec.get("fifth_attempt_plan_sha256")
        != (manifest["plan_sha256"] if attempt_number == 5 else None)
    ):
        raise ValueError("Sweep worker campaign budget does not match its verified attempt history")
    metadata = json.loads((bundle / "kernel/kernel-metadata.json").read_text())
    if metadata.get("id") != kernel_id or metadata.get("title") != kernel_title:
        raise ValueError("Sweep kernel metadata ID/title does not match its unique slug")
    for name, sha in spec["files"].items():
        if worker.digest(bundle / "dataset" / name) != sha:
            raise ValueError("Sweep staged dataset input was modified")
    if attempt_number in {3, 4, 5}:
        plan_value = json.loads((bundle / "dataset/plan.json").read_text())
        validate_plan(
            plan_value,
            ROOT / "scripts/run_sweep_comparison.py",
            bundle / "dataset/next_edit_inputs.jsonl",
            attempt_number=attempt_number,
            prior_attempts=prior_attempts,
        )
        if attempt_number == 5:
            worker.validate_fifth_attempt_plan(
                bundle / "dataset/plan.json", spec
            )
    quota = checked_quota(manifest["renewal"])
    if state["state"] == "prepared":
        datasets = json.loads(cli(["kaggle", "datasets", "list", "--mine", "--page-size", "100",
                                   "--format", "json"]))
        exists = any(row.get("ref") == DATASET_ID for row in datasets)
        if retry_after is not None:
            if not exists:
                raise RuntimeError("initial private dataset is missing")
            cli(["kaggle", "datasets", "version", "-p", str(bundle / "dataset"), "-t",
                 "-m", f"Preserve prior attempts; stage bounded attempt {attempt_number}"])
        else:
            if exists:
                raise RuntimeError("Sweep dataset already exists; reconcile publication state")
            cli(["kaggle", "datasets", "create", "-p", str(bundle / "dataset"), "-t"])
        state = {"state": "dataset_created"}
        write(state_path, state)
    readiness_deadline = time.monotonic() + 180
    while cli(["kaggle", "datasets", "status", DATASET_ID]).strip().lower() != "ready":
        if time.monotonic() >= readiness_deadline:
            raise RuntimeError("Sweep private dataset is not ready; GPU was not allocated")
        time.sleep(5)
    quota = checked_quota(manifest["renewal"])
    kernels = json.loads(cli(["kaggle", "kernels", "list", "--mine", "--page-size", "100",
                              "--format", "json"]))
    if any(row.get("ref") == kernel_id or row.get("id") == kernel_id for row in kernels):
        raise RuntimeError("Sweep kernel already exists; reconcile instead of duplicating")
    write(state_path, {"state": "submission_started", "quota_before": quota,
                       "submitted_at": datetime.now(UTC).isoformat()})
    cli(["kaggle", "kernels", "push", "-p", str(bundle / "kernel"),
         "--timeout", str(planned_session), "--accelerator", "NvidiaTeslaT4"])
    write(state_path, {"state": "kernel_pushed", "kernel_id": kernel_id,
                       "attempt_number": attempt_number, "quota_before": quota,
                       "submitted_at": datetime.now(UTC).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--line-suite", type=Path)
    parser.add_argument("--next-edit-fixtures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q4-artifact", type=Path)
    parser.add_argument("--retry-after", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit(args.output)
    elif args.plan and args.line_suite and args.next_edit_fixtures and args.q4_artifact:
        prepare(args.plan, args.line_suite, args.next_edit_fixtures, args.output,
                args.q4_artifact, args.retry_after)
    else:
        parser.error("preparation needs --plan, --line-suite, "
                     "--next-edit-fixtures and --q4-artifact")
