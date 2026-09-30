#!/usr/bin/env python3
"""Freeze and score the controller-side Sweep comparison.

This tool consumes only the pinned public/synthetic fixtures and result bundles
retrieved from the private Kaggle job. It never starts inference and never
executes generated source on the controller host.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_causal_line import score_line  # noqa: E402
from run_sweep_comparison import (  # noqa: E402
    CONTROL_OUTPUT_TOKENS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    RUNTIME_REVISION,
    SWEEP_OUTPUT_TOKENS,
    build_sweep_prompt,
    digest_bytes,
    digest_file,
    map_full_file,
    plan_digest,
)

from tinycomplete.eval.code_benchmark import (  # noqa: E402
    BenchmarkCase,
    Prediction,
    _container_runtime,
    evaluate_prediction,
    load_suite,
    summarize_results,
)
from tinycomplete.eval.code_generation import (  # noqa: E402
    build_causal_prompt,
    prediction_metadata_path,
)
from tinycomplete.eval.next_edit_benchmark import NextEditCase, load_next_edit_suite  # noqa: E402
from tinycomplete.observability.bootstrap import current_runtime  # noqa: E402
from tinycomplete.observability.context import RunContext  # noqa: E402
from tinycomplete.observability.runs import run_scope  # noqa: E402
from tinycomplete.observability.spans import operation  # noqa: E402

SCHEMA = "sweep-comparison-scoring-plan-v1"
PLAN_DEFAULT = ROOT / "reports/prototype/sweep_comparison_r1/scoring_plan.json"
CAMPAIGN_PLAN_DEFAULT = ROOT / "reports/prototype/sweep_comparison_r1/plan-v2.json"
STRICT_SUITE_DEFAULT = ROOT / "data/benchmarks/code_completion_v2.jsonl"
NEXT_EDIT_SUITE_DEFAULT = ROOT / "data/benchmarks/next_edit_v2.jsonl"
DIAGNOSTIC_PLAN_DEFAULT = (
    ROOT / "reports/prototype/product_r2/model_diagnostic/diagnostic_plan_v2.json"
)
NEXT_EDIT_INPUTS_DEFAULT = ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
LINE_SUITE_DEFAULT = Path(
    "/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/"
    "kaggle-bundle-r1/dataset/causal_line_v1-r3.jsonl"
)
PRECISIONS = ("q8_0", "q4_k_m")
INTENTS = {
    "clear_functional_intent",
    "no_edit_plausible",
    "ambiguous_requirement",
    "explicit_synthetic_target",
}
TERMINAL_REASONS = {"eos", "word"}
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_REPLICATES = 10_000
MAX_JSONL_BYTES = 256 * 1024**2
MAX_NEXT_EDIT_RESPONSE_BYTES = 2 * 1024**2


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _path_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": digest_file(resolved)}


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path, *, max_bytes: int = MAX_JSONL_BYTES) -> list[dict[str, Any]]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"JSONL input exceeds the bounded size allowance: {path.name}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {number} is not an object: {path.name}")
            rows.append(value)
    return rows


def _jsonl_digest(path: Path, rows: list[dict[str, Any]]) -> str:
    del rows
    return digest_file(path)


def _load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "sweep-comparison-plan-v1":
        raise ValueError("campaign plan schema is unsupported")
    if plan_digest(value) != value.get("plan_sha256"):
        raise ValueError("campaign plan self-hash mismatch")
    return value


def _safe_source_intent(diagnostic: dict[str, Any], fixture_id: str) -> str:
    found = [row for row in diagnostic["fixtures"] if row.get("case_id") == fixture_id]
    if len(found) != 1:
        raise ValueError(f"diagnostic plan does not uniquely identify {fixture_id}")
    intent = found[0].get("target_assessment", {}).get("functional_intent")
    if intent not in INTENTS:
        raise ValueError(f"unknown fixed intent category for {fixture_id}")
    return str(intent)


def _canonical_image_id(raw_id: str) -> str:
    digest = raw_id.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("sandbox image runtime returned an invalid immutable image ID")
    return "sha256:" + digest


def _resolve_images(cases: list[NextEditCase] | list[BenchmarkCase]) -> dict[str, str]:
    runtime = _container_runtime()
    if runtime is None:
        raise RuntimeError("pinned sandbox runtime is unavailable; refusing to freeze scoring")
    tags = sorted(
        {
            case.check.container_image
            for case in cases
            if case.check.container_image
            and (case.check.compile or case.check.test or case.check.run)
        }
    )
    result: dict[str, str] = {}
    for tag in tags:
        proc = subprocess.run(
            [runtime, "image", "inspect", "--format", "{{.Id}}", tag],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"existing sandbox image is unavailable or unpinnable: {tag}")
        image_id = _canonical_image_id(proc.stdout.strip())
        result[tag] = image_id
    if any(
        case.check.container_image is None
        for case in cases
        if case.check.compile or case.check.test or case.check.run
    ):
        raise ValueError("a functional benchmark case has no container image")
    return result


def _sandbox_identity(image_ids: dict[str, str]) -> dict[str, Any]:
    runtime = _container_runtime()
    if runtime is None:
        raise RuntimeError("pinned sandbox runtime is unavailable")
    executable = shutil.which(runtime)
    if executable is None:
        raise RuntimeError("sandbox runtime path disappeared")
    version = subprocess.run(
        [runtime, "--version"], text=True, capture_output=True, timeout=15, check=False
    )
    if version.returncode != 0:
        raise RuntimeError("could not identify the existing sandbox runtime")
    return {
        "runtime": runtime,
        "runtime_path": str(Path(executable).resolve()),
        "runtime_sha256": digest_file(Path(executable).resolve()),
        "runtime_version": version.stdout.strip(),
        "image_ids_by_tag": image_ids,
        "execution_backend": "container",
        "network": "none",
        "pull_policy": "never",
        "resource_flags": [
            "--read-only",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--memory=768m",
            "--cpus=1",
            "--userns=keep-id",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
        ],
    }


def _verify_sandbox(sandbox: dict[str, Any]) -> None:
    runtime = _container_runtime()
    executable = shutil.which(str(sandbox["runtime"]))
    if runtime != sandbox["runtime"] or executable is None:
        raise ValueError("active sandbox runtime differs from the frozen scoring plan")
    runtime_path = Path(executable).resolve()
    if (
        str(runtime_path) != sandbox["runtime_path"]
        or digest_file(runtime_path) != sandbox["runtime_sha256"]
    ):
        raise ValueError("sandbox runtime binary differs from the frozen scoring plan")
    version = subprocess.run(
        [runtime, "--version"], text=True, capture_output=True, timeout=15, check=False
    )
    if version.returncode != 0 or version.stdout.strip() != sandbox["runtime_version"]:
        raise ValueError("sandbox runtime version differs from the frozen scoring plan")
    for tag, expected_id in sandbox["image_ids_by_tag"].items():
        for ref in (tag, expected_id):
            proc = subprocess.run(
                [runtime, "image", "inspect", "--format", "{{.Id}}", ref],
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            actual_id = _canonical_image_id(proc.stdout.strip()) if proc.returncode == 0 else None
            if actual_id != expected_id:
                raise ValueError("pinned sandbox image identity changed or is unavailable")


def _fingerprints() -> dict[str, str]:
    files = {
        "scoring_script": Path(__file__).resolve(),
        "scoring_tests": ROOT / "tests/test_score_sweep_comparison.py",
        "campaign_runner": ROOT / "scripts/run_sweep_comparison.py",
        "causal_line_scorer": ROOT / "scripts/evaluate_causal_line.py",
        "causal_evaluator": ROOT / "src/tinycomplete/eval/code_benchmark.py",
        "causal_prompt_and_predictions": ROOT / "src/tinycomplete/eval/code_generation.py",
        "next_edit_case_contract": ROOT / "src/tinycomplete/eval/next_edit_benchmark.py",
        "next_edit_protocol": ROOT / "src/tinycomplete/eval/next_edit_protocol.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("scoring implementation files are missing: " + ",".join(missing))
    return {name: digest_file(path) for name, path in files.items()}


def _verify_input_hashes(plan: dict[str, Any]) -> None:
    for key, record in plan["inputs"].items():
        path = resolve_recorded_path(record["path"])
        if not path.is_file() or digest_file(path) != record["sha256"]:
            raise ValueError(f"frozen scoring input changed: {key}")
    if _fingerprints() != plan["implementation_sha256"]:
        raise ValueError("scoring implementation differs from the frozen scoring plan")
    _verify_sandbox(plan["sandbox"])


def _validate_inputs(
    campaign: dict[str, Any],
    strict_path: Path,
    line_path: Path,
    next_edit_suite_path: Path,
    diagnostic_path: Path,
    prompt_inputs_path: Path,
) -> tuple[
    list[BenchmarkCase],
    list[dict[str, Any]],
    list[NextEditCase],
    dict[str, Any],
    list[dict[str, Any]],
]:
    strict_cases = load_suite(strict_path)
    line_rows = _read_jsonl(line_path, max_bytes=32 * 1024**2)
    next_edit_cases = load_next_edit_suite(next_edit_suite_path)
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    prompt_rows = _read_jsonl(prompt_inputs_path, max_bytes=64 * 1024**2)

    if len(strict_cases) != 200 or len({case.id for case in strict_cases}) != 200:
        raise ValueError("strict causal suite must retain its 200 unique cases")
    if len(line_rows) != 180 or len({row.get("id") for row in line_rows}) != 180:
        raise ValueError("causal line suite must retain its 180 unique cases")
    if len(next_edit_cases) != 200 or len({case.id for case in next_edit_cases}) != 200:
        raise ValueError("next-edit source suite must retain its 200 unique cases")
    if len(diagnostic.get("fixtures", [])) != 24 or len(prompt_rows) != 24:
        raise ValueError("the frozen next-edit diagnostic and prompt bundle must each have 24 rows")
    if not isinstance(diagnostic.get("plan_sha256"), str):
        raise ValueError("diagnostic source plan lacks its revision identity")

    expected_ids = campaign["comparison"]["next_edit"]["fixture_ids"]
    if [row.get("case_id") for row in prompt_rows] != expected_ids:
        raise ValueError("prompt bundle ordering differs from the frozen campaign plan")
    if [row.get("case_id") for row in diagnostic["fixtures"]] != expected_ids:
        raise ValueError("controller diagnostic ordering differs from the frozen campaign plan")
    source_by_id = {case.id: case for case in next_edit_cases}
    if diagnostic.get("plan_sha256") != campaign["comparison"]["next_edit"]["diagnostic_revision"]:
        raise ValueError("controller labels differ from the pinned diagnostic revision")

    for fixture, prompt_row in zip(diagnostic["fixtures"], prompt_rows, strict=True):
        case_id = fixture["case_id"]
        if case_id != prompt_row.get("case_id"):
            raise ValueError("diagnostic and prompt fixture IDs do not match")
        if not fixture.get("history_reconstructs_current_byte_exactly"):
            raise ValueError(f"controller fixture lacks exact history reconstruction: {case_id}")
        if digest_bytes(prompt_row["current_content"].encode()) != prompt_row.get("current_sha256"):
            raise ValueError(f"prompt current-state hash mismatch: {case_id}")
        if prompt_row.get("prompt_sha256") != digest_bytes(build_sweep_prompt(prompt_row).encode()):
            raise ValueError(f"frozen serving prompt hash mismatch: {case_id}")
        if any(
            key in prompt_row
            for key in ("gold_action", "gold_text", "target_assessment", "expected_after")
        ):
            raise ValueError("model input bundle contains controller-only labels")
        start = prompt_row.get("editable_start_byte")
        end = prompt_row.get("editable_end_byte")
        current = prompt_row["current_content"].encode("utf-8")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start <= end <= len(current)
        ):
            raise ValueError(f"prompt editable byte range is invalid: {case_id}")
        try:
            current[:start].decode("utf-8")
            current[start:end].decode("utf-8")
            current[end:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"prompt editable range splits UTF-8: {case_id}") from exc
        if (start, end) != (
            fixture.get("cursor_byte_offset"),
            fixture.get("region_end_byte_offset")
            if fixture.get("region_end_byte_offset") is not None
            else _line_end_byte(prompt_row["current_content"], start),
        ):
            raise ValueError(f"prompt editable range differs from controller label: {case_id}")
        region = current[start:end].decode("utf-8")
        source_id = fixture.get("source_case_id")
        if source_id is None:
            if fixture.get("target_assessment", {}).get("functional_intent") != (
                "explicit_synthetic_target"
            ):
                raise ValueError("only explicit synthetic targets may lack a source oracle")
            if fixture.get("gold_action") != "replace" or fixture.get("gold_text") != "":
                raise ValueError(
                    "synthetic deletion fixture does not have an explicit empty target"
                )
            continue
        source_case = source_by_id.get(source_id)
        if source_case is None or source_id != case_id:
            raise ValueError(f"diagnostic source-case identity mismatch: {case_id}")
        if (prompt_row["language"], prompt_row["file_path"]) != (
            source_case.language,
            source_case.path,
        ):
            raise ValueError(f"source check language or path differs: {case_id}")
        if region != source_case.region_text:
            raise ValueError(f"editable source region differs from original suite: {case_id}")
        expected_action = "no_edit" if source_case.action == "noop" else "replace"
        if fixture.get("gold_action") != expected_action:
            raise ValueError(f"diagnostic action label differs from source suite: {case_id}")
        if expected_action == "replace" and fixture.get("gold_text") != source_case.expected:
            raise ValueError(f"diagnostic target differs from source suite: {case_id}")
        if expected_action == "no_edit" and fixture.get("gold_text") != "":
            raise ValueError(f"no-edit fixture contains a replacement target: {case_id}")
        if prompt_row.get("context_files") != source_case.context_files:
            raise ValueError(f"source context files differ from the model prompt: {case_id}")
        if fixture["target_assessment"]["functional_intent"] not in INTENTS:
            raise ValueError(f"unknown controller intent category: {case_id}")

    counts = Counter(
        fixture["target_assessment"]["functional_intent"] for fixture in diagnostic["fixtures"]
    )
    expected_counts = campaign["comparison"]["next_edit"].get("target_groups_counts")
    if expected_counts is not None and dict(counts) != expected_counts:
        raise ValueError("diagnostic intent denominators differ from the campaign plan")
    return strict_cases, line_rows, next_edit_cases, diagnostic, prompt_rows


def _line_end_byte(current: str, start: int) -> int:
    raw = current.encode("utf-8")
    end = raw.find(b"\n", start)
    return len(raw) if end < 0 else end


def freeze_plan(args: argparse.Namespace) -> dict[str, Any]:
    campaign = _load_plan(args.campaign_plan)
    expected_paths = campaign["comparison"]["quality_controls"]
    expected_by_key = {
        "strict_suite": expected_paths["strict_causal"]["sha256"],
        "line_suite": expected_paths["causal_line"]["sha256"],
        "next_edit_suite": campaign["comparison"]["next_edit"]["source_suite"]["sha256"],
        "diagnostic_plan": campaign["comparison"]["next_edit"]["diagnostic_plan_sha256"],
        "prompt_inputs": campaign["comparison"]["next_edit"]["fixture_input_sha256"],
    }
    inputs = {
        "campaign_plan": _path_record(args.campaign_plan),
        "strict_suite": _path_record(args.strict_suite),
        "line_suite": _path_record(args.line_suite),
        "next_edit_suite": _path_record(args.next_edit_suite),
        "diagnostic_plan": _path_record(args.diagnostic_plan),
        "prompt_inputs": _path_record(args.prompt_inputs),
    }
    for key, expected_sha in expected_by_key.items():
        if inputs[key]["sha256"] != expected_sha:
            raise ValueError(f"controller input does not match campaign plan: {key}")
    if campaign["code"]["runner_sha256"] != digest_file(ROOT / "scripts/run_sweep_comparison.py"):
        raise ValueError("local campaign runner differs from the frozen campaign plan")
    strict_cases, _line_rows, next_cases, diagnostic, prompt_rows = _validate_inputs(
        campaign,
        args.strict_suite,
        args.line_suite,
        args.next_edit_suite,
        args.diagnostic_plan,
        args.prompt_inputs,
    )
    suite_cases: list[NextEditCase | BenchmarkCase] = [*strict_cases, *next_cases]
    image_ids = _resolve_images(suite_cases)
    sandbox = _sandbox_identity(image_ids)
    intent_groups: dict[str, list[str]] = {name: [] for name in sorted(INTENTS)}
    for fixture in diagnostic["fixtures"]:
        intent = str(fixture["target_assessment"]["functional_intent"])
        intent_groups[intent].append(str(fixture["case_id"]))
    source_oracle_ids = sorted(
        str(fixture["case_id"])
        for fixture in diagnostic["fixtures"]
        if fixture.get("source_case_id") is not None
    )
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "revision": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "campaign": {
            "name": campaign.get("campaign"),
            "path": inputs["campaign_plan"]["path"],
            "raw_file_sha256": inputs["campaign_plan"]["sha256"],
            "embedded_plan_sha256": campaign["plan_sha256"],
            "runner_sha256": campaign["code"]["runner_sha256"],
        },
        "inputs": {key: value for key, value in inputs.items() if key != "campaign_plan"},
        "source_suite": {
            "strict_causal_cases": len(strict_cases),
            "causal_line_cases": 180,
            "next_edit_cases": len(next_cases),
            "controller_diagnostic_cases": len(diagnostic["fixtures"]),
            "prompt_only_rows": len(prompt_rows),
            "source_oracle_case_ids": source_oracle_ids,
        },
        "models": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "q8_0_sha256": campaign["model"]["artifacts"]["q8_0"]["sha256"],
            "q4_k_m_sha256": campaign["model"]["artifacts"]["q4_k_m"]["sha256"],
            "runtime_revision": RUNTIME_REVISION,
        },
        "prompt_and_termination": {
            "strict": (
                "raw causal prompt built by code_generation.build_causal_prompt; no chat template"
            ),
            "line": (
                "exact fixture prompt; score_line restores only runtime-reported word-stop LF/CRLF"
            ),
            "next_edit": (
                "pinned Sweep whole-file prompt and order; no prompt reconstruction for scoring"
            ),
            "terminal_acceptance": (
                "explicit_terminal=true, finish_reason in eos/word, hit_token_cap=false"
            ),
            "response_repair": (
                "none; no trim, fence stripping, parser repair, or multiline truncation"
            ),
            "editable_mapping": (
                "recompute byte-exact prefix/suffix mapping; score only within-region "
                "UTF-8 boundaries"
            ),
            "strict_output_ceiling": CONTROL_OUTPUT_TOKENS,
            "line_output_ceiling": CONTROL_OUTPUT_TOKENS,
            "next_edit_output_ceiling": SWEEP_OUTPUT_TOKENS,
        },
        "scoring": {
            "strict": {
                "protocol": "raw-causal-v1",
                "implementation": (
                    "tinycomplete.eval.code_benchmark.evaluate_prediction/summarize_results"
                ),
                "every_eligible_output_in_exact_denominator": True,
                "context_ineligible_rows": "reported separately and omitted without truncation",
                "execution_backend": "network-none container with frozen local image IDs",
            },
            "line": {
                "protocol": "causal-line-v1",
                "implementation": "scripts/evaluate_causal_line.py:score_line",
                "primary_metrics": ["exact", "syntax"],
                "whitespace_repair": ("none; only registered native word-stop newline restoration"),
            },
            "next_edit": {
                "protocol": "sweep-whole-file-v1 mapped to original marked-region source cases",
                "groups": intent_groups,
                "clear_intent_functional_success": (
                    "clear_functional_intent only; action/reference exact and original source-case "
                    "parse/compile/test pass after applying mapped replacement to the observed "
                    "current state"
                ),
                "no_edit": (
                    "report reference agreement and false-positive edits in the separate "
                    "plausible-no-edit denominator; not certainty about human intent"
                ),
                "ambiguous": (
                    "report reference agreement and objective validity separately; "
                    "exclude from clear-intent success"
                ),
                "synthetic_deletions": (
                    "report exact action/replacement only; no functional oracle"
                ),
                "functional_oracle": (
                    "use original source-suite checks only; generated code runs only "
                    "in pinned network-none containers"
                ),
            },
            "repetitions": {
                "count_per_case_and_precision": 2,
                "consistency": (
                    "report full-response hash and mapped-action agreement across repetitions"
                ),
            },
            "paired_comparison": {
                "direction": "Q4_K_M minus Q8_0",
                "unit": "paired case; next-edit repeats averaged within case before resampling",
                "uncertainty": (
                    f"{BOOTSTRAP_REPLICATES} deterministic case-cluster bootstrap replicates, "
                    f"seed {BOOTSTRAP_SEED}, percentile 95% interval"
                ),
                "interpretation": "paired descriptive evidence only; no equivalence claim",
            },
        },
        "sandbox": sandbox,
        "implementation_sha256": _fingerprints(),
        "output_contract": {
            "raw_outputs_are_not_copied": True,
            "per_case_fields": [
                "case_id",
                "model_precision",
                "response_sha256",
                "terminal_and_mapping_status",
                "scoring_outcomes",
                "sandbox_statuses_without_stdout_or_stderr",
            ],
            "metrics_have_no_case_uuid_path_prompt_response_or_full_model_hash_labels": True,
        },
        "telemetry": {
            "run_scope": True,
            "per_case_eval_and_model_score_spans": True,
            "prompt_or_prediction_content_capture": False,
            "actual_model_output_files_remain_authoritative": True,
        },
    }
    plan["scoring_plan_sha256"] = digest_bytes(
        canonical_json({k: v for k, v in plan.items() if k != "scoring_plan_sha256"})
    )
    return plan


def write_frozen_plan(path: Path, plan: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != plan:
            raise FileExistsError(
                "scoring plan exists with different contents; preserve it and revise"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def load_frozen_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA or plan.get("revision") != 1:
        raise ValueError("unsupported scoring plan schema or revision")
    expected = digest_bytes(
        canonical_json({k: v for k, v in plan.items() if k != "scoring_plan_sha256"})
    )
    if plan.get("scoring_plan_sha256") != expected:
        raise ValueError("scoring plan self-hash mismatch")
    _verify_input_hashes(plan)
    campaign = _load_plan(resolve_recorded_path(plan["campaign"]["path"]))
    if (
        campaign["plan_sha256"] != plan["campaign"]["embedded_plan_sha256"]
        or digest_file(resolve_recorded_path(plan["campaign"]["path"]))
        != plan["campaign"]["raw_file_sha256"]
    ):
        raise ValueError("campaign plan differs from the scoring plan")
    return plan


def _load_predictions(path: Path, expected_ids: set[str]) -> dict[str, Prediction]:
    if not path.is_file():
        raise FileNotFoundError(f"required frozen prediction file is missing: {path.name}")
    if path.stat().st_size > MAX_JSONL_BYTES:
        raise ValueError("prediction file exceeds the scoring size limit")
    rows: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as handle:
        for _number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            prediction = Prediction.model_validate(raw)
            if prediction.case_id not in expected_ids or prediction.case_id in rows:
                raise ValueError(f"prediction IDs are duplicated or unexpected in {path.name}")
            rows[prediction.case_id] = prediction
    if set(rows) != expected_ids:
        raise ValueError(f"prediction coverage differs from frozen eligible set in {path.name}")
    return rows


def _load_token_rows(
    path: Path, prompt_hashes: dict[str, str], cap: int
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in parsed:
            raise ValueError(f"input-token records have a missing or duplicate ID: {path.name}")
        parsed[case_id] = row
    if len(parsed) != len(rows) or set(parsed) != set(prompt_hashes):
        raise ValueError(f"input-token records do not cover the frozen suite: {path.name}")
    for case_id, row in parsed.items():
        if (
            row.get("prompt_sha256") != prompt_hashes[case_id]
            or row.get("output_ceiling") != cap
            or not isinstance(row.get("input_tokens"), int)
            or not isinstance(row.get("context_eligible"), bool)
        ):
            raise ValueError(f"input-token record identity mismatch: {case_id}")
    return parsed


def _verify_metadata(
    path: Path,
    *,
    plan_sha256: str,
    model_sha256: str,
    suite_sha256: str,
    protocol: str,
    max_new_tokens: int,
    eligible_count: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction metadata is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "plan_sha256": plan_sha256,
        "model_revision": model_sha256,
        "suite_sha256": suite_sha256,
        "protocol": protocol,
        "max_new_tokens": max_new_tokens,
        "case_count": eligible_count,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError(f"prediction metadata differs from frozen scoring identity: {path.name}")
    return value


def _verify_generation_details(
    path: Path,
    token_rows: dict[str, dict[str, Any]],
    prompt_hashes: dict[str, str],
) -> None:
    rows = _read_jsonl(path)
    eligible = {case_id for case_id, row in token_rows.items() if row["context_eligible"]}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in by_id:
            raise ValueError(f"generation details have a missing or duplicate ID: {path.name}")
        by_id[case_id] = row
    if len(by_id) != len(rows) or set(by_id) != eligible:
        raise ValueError(f"generation details do not cover eligible cases: {path.name}")
    for case_id, row in by_id.items():
        if (
            row.get("prompt_sha256") != prompt_hashes[case_id]
            or row.get("input_tokens") != token_rows[case_id]["input_tokens"]
        ):
            raise ValueError(f"per-case generation identity mismatch: {case_id}")


def _check_status(result: Any) -> dict[str, Any]:
    return {
        "parse": result.parse.status,
        "compile": result.compile.status,
        "test": result.test.status,
        "compile_configured": result.compile_configured,
        "test_configured": result.test_configured,
        "parse_seconds": result.parse.seconds,
        "compile_seconds": result.compile.seconds,
        "test_seconds": result.test.seconds,
    }


def _functional_valid(result: Any) -> bool:
    return (
        result.parse.status == "pass"
        and (not result.compile_configured or result.compile.status == "pass")
        and (not result.test_configured or result.test.status == "pass")
    )


def _pairwise_bootstrap(
    q8_values: dict[str, float | None], q4_values: dict[str, float | None]
) -> dict[str, Any]:
    keys = sorted(set(q8_values) | set(q4_values))
    pairs: list[tuple[float, float]] = []
    for key in keys:
        q8_value = q8_values.get(key)
        q4_value = q4_values.get(key)
        if q8_value is not None and q4_value is not None:
            pairs.append((float(q8_value), float(q4_value)))
    skipped = len(keys) - len(pairs)
    if not pairs:
        return {"n_paired_cases": 0, "n_skipped": skipped, "q4_minus_q8": None, "ci95": None}
    diffs = [right - left for left, right in pairs]
    mean_diff = sum(diffs) / len(diffs)
    rng = random.Random(BOOTSTRAP_SEED)
    boot_means = sorted(
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    lower = boot_means[math.floor(0.025 * (BOOTSTRAP_REPLICATES - 1))]
    upper = boot_means[math.floor(0.975 * (BOOTSTRAP_REPLICATES - 1))]
    return {
        "n_paired_cases": len(pairs),
        "n_skipped": skipped,
        "q8_case_mean": sum(left for left, _ in pairs) / len(pairs),
        "q4_case_mean": sum(right for _, right in pairs) / len(pairs),
        "q4_minus_q8": mean_diff,
        "ci95": [lower, upper],
        "q4_wins": sum(value > 0 for value in diffs),
        "ties": sum(value == 0 for value in diffs),
        "q8_wins": sum(value < 0 for value in diffs),
        "method": (
            "paired case-cluster percentile bootstrap; repeated observations averaged within case"
        ),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "interpretation": "descriptive uncertainty interval; not an equivalence test",
    }


def _precision_quality(
    *,
    precision: str,
    task: str,
    cases: list[BenchmarkCase],
    predictions: dict[str, Prediction],
    eligible: set[str],
    output_root: Path,
    run_context: RunContext,
    image_ids: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    rows = []
    with tempfile.TemporaryDirectory(prefix="sweep-score-" + precision + "-") as temp_name:
        temp_root = Path(temp_name)
        for index, case in enumerate(cases):
            if case.id not in eligible:
                rows.append(
                    {
                        "case_id": case.id,
                        "precision": precision,
                        "context_eligible": False,
                        "scored": False,
                    }
                )
                continue
            prediction = predictions[case.id]
            check = case.check
            if check.container_image:
                check = check.model_copy(
                    update={"container_image": image_ids[check.container_image]}
                )
            actual_case = case.model_copy(update={"check": check})
            case_context = run_context.for_case(case.id)
            with (
                case_context.activate(),
                operation(
                    "model.score",
                    attributes={
                        "tabcomplete.task": task,
                        "tabcomplete.language": case.language,
                        "tabcomplete.model_alias": "sweep-next-edit-1.5b",
                        "tabcomplete.quantization": precision,
                    },
                ),
            ):
                result = evaluate_prediction(
                    actual_case,
                    prediction,
                    work_root=temp_root / f"case-{index:03d}",
                    execution_backend="container",
                )
            results.append(result)
            rows.append(
                {
                    "case_id": case.id,
                    "precision": precision,
                    "context_eligible": True,
                    "scored": True,
                    "response_sha256": digest_bytes(prediction.completion.encode()),
                    "finish_reason": prediction.finish_reason,
                    "hit_token_cap": prediction.hit_token_cap,
                    "exact_match": result.exact_match,
                    "normalized_exact_match": result.normalized_exact_match,
                    "parse_status": result.parse.status,
                    "compile_status": result.compile.status,
                    "test_status": result.test.status,
                    "functional_valid": _functional_valid(result),
                }
            )
    return rows, summarize_results(results)


def _score_quality_task(
    *,
    plan: dict[str, Any],
    results_root: Path,
    precision: str,
    task: str,
    cases: list[BenchmarkCase],
    suite_path: Path,
    prompt_builder,
    precision_sha: str,
    output_root: Path,
    run_context: RunContext,
    image_ids: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_dir = results_root / "quality" / precision
    token_name, prediction_name, details_name, protocol = (
        ("strict", "strict-predictions.jsonl", "strict-generation-details.jsonl", "raw-causal-v1")
        if task == "strict"
        else ("line", "line-predictions.jsonl", "line-generation-details.jsonl", "causal-line-v1")
    )
    prompts = {case.id: prompt_builder(case) for case in cases}
    prompt_hashes = {case_id: digest_bytes(prompt.encode()) for case_id, prompt in prompts.items()}
    token_rows = _load_token_rows(
        task_dir / f"{token_name}-input-token-counts.jsonl", prompt_hashes, CONTROL_OUTPUT_TOKENS
    )
    eligible = {case_id for case_id, row in token_rows.items() if row["context_eligible"]}
    prediction_path = task_dir / prediction_name
    metadata_path = prediction_metadata_path(prediction_path)
    metadata = _verify_metadata(
        metadata_path,
        plan_sha256=plan["campaign"]["embedded_plan_sha256"],
        model_sha256=precision_sha,
        suite_sha256=digest_file(suite_path),
        protocol=protocol,
        max_new_tokens=CONTROL_OUTPUT_TOKENS,
        eligible_count=len(eligible),
    )
    if metadata.get("provider") != "llama.cpp-native" or metadata.get("model_source") != (
        f"{MODEL_REPOSITORY}:{MODEL_REVISION}"
    ):
        raise ValueError("quality prediction metadata does not identify the frozen model/provider")
    predictions = _load_predictions(prediction_path, eligible)
    _verify_generation_details(task_dir / details_name, token_rows, prompt_hashes)
    rows, summary = _precision_quality(
        precision=precision,
        task=task,
        cases=cases,
        predictions=predictions,
        eligible=eligible,
        output_root=output_root,
        run_context=run_context,
        image_ids=image_ids,
    )
    summary.update(
        {
            "requested_cases": len(cases),
            "context_eligible_cases": len(eligible),
            "context_ineligible_case_ids": sorted(set(token_rows) - eligible),
            "prediction_file_sha256": digest_file(prediction_path),
            "prediction_metadata_sha256": digest_file(metadata_path),
            "token_records_sha256": digest_file(
                task_dir / f"{token_name}-input-token-counts.jsonl"
            ),
            "generation_details_sha256": digest_file(task_dir / details_name),
        }
    )
    return rows, summary


def _adapt_source_case(
    source: NextEditCase,
    prompt_row: dict[str, Any],
    fixture: dict[str, Any],
    image_ids: dict[str, str],
) -> NextEditCase:
    start = int(prompt_row["editable_start_byte"])
    end = int(prompt_row["editable_end_byte"])
    check = source.check
    if check.container_image:
        check = check.model_copy(update={"container_image": image_ids[check.container_image]})
    return NextEditCase(
        id=source.id,
        language=source.language,
        path=source.path,
        current=prompt_row["current_content"],
        region_start=start,
        region_end=end,
        expected=source.expected,
        action=source.action,
        recent_edits=source.recent_edits,
        context_files=source.context_files,
        check=check,
        category=source.category,
        repository_context=source.repository_context,
    )


def _row_terminal_valid(row: dict[str, Any]) -> bool:
    return (
        row.get("explicit_terminal") is True
        and row.get("finish_reason") in TERMINAL_REASONS
        and row.get("hit_token_cap") is False
    )


def _score_next_edit_result(
    *,
    row: dict[str, Any],
    prompt_row: dict[str, Any],
    fixture: dict[str, Any],
    source_case: NextEditCase | None,
    precision: str,
    repetition: int,
    case_index: int,
    temp_root: Path,
    run_context: RunContext,
) -> dict[str, Any]:
    raw_output = row.get("raw_output")
    if row.get("context_eligible") is not True:
        if raw_output is not None or row.get("output_sha256") is not None:
            raise ValueError(
                "context-ineligible next-edit case unexpectedly contains a model output"
            )
        if row.get("editable_range_mapping") != {
            "mapping": "incomplete_or_unterminated_output_not_scored",
            "out_of_range": None,
        }:
            raise ValueError("context-ineligible case has an unexpected output mapping")
        intent = fixture["target_assessment"]["functional_intent"]
        return {
            "case_id": fixture["case_id"],
            "precision": precision,
            "repetition": repetition,
            "intent_group": intent,
            "context_eligible": False,
            "terminal_valid": False,
            "mapping_status": "context_ineligible_not_generated",
            "predicted_action": None,
            "response_sha256": None,
            "exact_reference_match": False,
            "no_edit_agreement": None,
            "false_positive_edit": None,
            "clear_intent_functional_success": (
                False if intent == "clear_functional_intent" else None
            ),
            "objective_patch_valid": None,
            "functional_status": None,
            "functional_oracle_available": source_case is not None,
            "synthetic_deletion_reference_match": None,
            "latency_ms": None,
            "input_tokens": row.get("input_tokens"),
            "output_tokens": 0,
            "observability_ids": row.get("observability_ids"),
        }
    if (
        not isinstance(raw_output, str)
        or len(raw_output.encode("utf-8")) > MAX_NEXT_EDIT_RESPONSE_BYTES
    ):
        raise ValueError("next-edit raw response is missing or exceeds the frozen limit")
    if digest_bytes(raw_output.encode("utf-8")) != row.get("output_sha256"):
        raise ValueError("next-edit raw response hash mismatch")
    terminal_valid = _row_terminal_valid(row)
    expected_mapping = (
        map_full_file(
            prompt_row["current_content"],
            raw_output,
            prompt_row["editable_start_byte"],
            prompt_row["editable_end_byte"],
        )
        if terminal_valid
        else {"mapping": "incomplete_or_unterminated_output_not_scored", "out_of_range": None}
    )
    if row.get("editable_range_mapping") != expected_mapping:
        raise ValueError("stored editable-range mapping differs from the frozen byte mapper")
    mapped = terminal_valid and expected_mapping.get("mapping") == "within_editable_range"
    intent = fixture["target_assessment"]["functional_intent"]
    gold_action = fixture["gold_action"]
    gold_text = fixture["gold_text"]
    predicted_action = expected_mapping.get("action") if mapped else None
    predicted_text = expected_mapping.get("replacement") if mapped else None
    target_text = (
        source_case.region_text
        if source_case is not None and gold_action == "no_edit"
        else gold_text
    )
    reference_exact = (
        bool(mapped and predicted_action == gold_action and predicted_text == target_text)
        if source_case is not None
        else bool(mapped and predicted_action == "replace" and predicted_text == "")
    )
    functional_status: dict[str, Any] | None = None
    functional_valid: bool | None = None
    if mapped and source_case is not None:
        case_context = run_context.for_case(source_case.id)
        completion = str(predicted_text)
        with (
            case_context.activate(),
            operation(
                "model.score",
                attributes={
                    "tabcomplete.task": "sweep-next-edit-mapped-source",
                    "tabcomplete.language": source_case.language,
                    "tabcomplete.model_alias": "sweep-next-edit-1.5b",
                    "tabcomplete.quantization": precision,
                },
            ),
        ):
            result = evaluate_prediction(
                source_case.as_completion_case(),
                Prediction(case_id=source_case.id, completion=completion),
                work_root=temp_root / f"{case_index:03d}-{repetition}",
                execution_backend="container",
            )
        functional_status = _check_status(result)
        functional_valid = _functional_valid(result)
    clear_task_success = None
    if intent == "clear_functional_intent":
        clear_task_success = bool(reference_exact and functional_valid is True)
    no_edit_agreement = (
        bool(mapped and predicted_action == "no_edit") if intent == "no_edit_plausible" else None
    )
    synthetic_delete_exact = (
        bool(mapped and predicted_action == "replace" and predicted_text == "")
        if intent == "explicit_synthetic_target"
        else None
    )
    return {
        "case_id": fixture["case_id"],
        "precision": precision,
        "repetition": repetition,
        "intent_group": intent,
        "context_eligible": True,
        "terminal_valid": terminal_valid,
        "mapping_status": expected_mapping["mapping"],
        "predicted_action": predicted_action,
        "response_sha256": row["output_sha256"],
        "exact_reference_match": reference_exact,
        "no_edit_agreement": no_edit_agreement,
        "false_positive_edit": (
            bool(mapped and predicted_action != "no_edit")
            if intent == "no_edit_plausible"
            else None
        ),
        "clear_intent_functional_success": clear_task_success,
        "objective_patch_valid": functional_valid,
        "functional_status": functional_status,
        "functional_oracle_available": source_case is not None,
        "synthetic_deletion_reference_match": synthetic_delete_exact,
        "latency_ms": row.get("completed_response_ms"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "observability_ids": row.get("observability_ids"),
    }


def _summarize_next_edit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["intent_group"]].append(row)
        cases[row["case_id"]].append(row)
    group_summary = {}
    for intent, group_rows in sorted(groups.items()):
        terminal = [row for row in group_rows if row["terminal_valid"]]
        exact = sum(row["exact_reference_match"] for row in group_rows)
        item: dict[str, Any] = {
            "case_denominator": len({row["case_id"] for row in group_rows}),
            "repetition_denominator": len(group_rows),
            "terminal_valid_count": len(terminal),
            "range_mapped_count": sum(
                row["mapping_status"] == "within_editable_range" for row in group_rows
            ),
            "exact_reference_match_count": exact,
            "exact_reference_match_rate": exact / len(group_rows) if group_rows else None,
        }
        if intent == "clear_functional_intent":
            values = [row["clear_intent_functional_success"] for row in group_rows]
            item["clear_intent_functional_success_count"] = sum(value is True for value in values)
            item["clear_intent_functional_success_denominator"] = len(values)
            item["clear_intent_functional_success_rate"] = (
                sum(value is True for value in values) / len(values) if values else None
            )
            item["objective_checks_unavailable"] = sum(
                row["functional_status"] is not None
                and any(
                    row["functional_status"][name] == "unavailable" for name in ("compile", "test")
                )
                for row in group_rows
            )
        elif intent == "no_edit_plausible":
            item["reference_no_edit_agreement_count"] = sum(
                row["no_edit_agreement"] is True for row in group_rows
            )
            item["false_positive_edits"] = sum(
                row["false_positive_edit"] is True for row in group_rows
            )
            item["interpretation"] = "plausible benchmark no-edit labels, not certain human intent"
        elif intent == "ambiguous_requirement":
            item["interpretation"] = "reference agreement only; excluded from clear-intent success"
        elif intent == "explicit_synthetic_target":
            item["synthetic_deletion_exact_count"] = sum(
                row["synthetic_deletion_reference_match"] is True for row in group_rows
            )
            item["functional_oracle"] = "unavailable; no source functional check"
        group_summary[intent] = item
    repetition_consistency = []
    for case_id, case_rows in sorted(cases.items()):
        by_rep = {row["repetition"]: row for row in case_rows}
        if set(by_rep) == {0, 1}:
            repetition_consistency.append(
                {
                    "case_id": case_id,
                    "response_identical": by_rep[0]["response_sha256"]
                    == by_rep[1]["response_sha256"],
                    "mapped_action_identical": by_rep[0]["predicted_action"]
                    == by_rep[1]["predicted_action"],
                }
            )
    return {
        "requested_case_repetitions": len(rows),
        "terminal_valid_count": sum(row["terminal_valid"] for row in rows),
        "range_mapped_count": sum(row["mapping_status"] == "within_editable_range" for row in rows),
        "intent_groups": group_summary,
        "repetition_consistency": {
            "cases": len(repetition_consistency),
            "identical_response_count": sum(
                row["response_identical"] for row in repetition_consistency
            ),
            "identical_action_count": sum(
                row["mapped_action_identical"] for row in repetition_consistency
            ),
        },
    }


def _case_metric(
    rows: list[dict[str, Any]], metric: str, intent: str | None = None
) -> dict[str, float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if intent is None or row["intent_group"] == intent:
            grouped[row["case_id"]].append(row)
    result = {}
    for case_id, values in grouped.items():
        outcomes = [row.get(metric) for row in values]
        known = [float(value) for value in outcomes if value is not None]
        result[case_id] = sum(known) / len(known) if known else None
    return result


def _score_next_edit(
    *,
    plan: dict[str, Any],
    campaign: dict[str, Any],
    results_root: Path,
    output_root: Path,
    diagnostic: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    next_cases: list[NextEditCase],
    image_ids: dict[str, str],
    run_context: RunContext,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    diagnostic_by_id = {row["case_id"]: row for row in diagnostic["fixtures"]}
    source_by_id = {case.id: case for case in next_cases}
    prompt_by_id = {row["case_id"]: row for row in prompt_rows}
    output: dict[str, list[dict[str, Any]]] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="sweep-next-edit-score-") as temp_name:
        temp_root = Path(temp_name)
        for precision in PRECISIONS:
            model_sha = (
                plan["models"]["q8_0_sha256"]
                if precision == "q8_0"
                else plan["models"]["q4_k_m_sha256"]
            )
            predictions_path = results_root / "next-edit" / precision / "predictions.jsonl"
            if not predictions_path.is_file() or predictions_path.stat().st_size > MAX_JSONL_BYTES:
                raise ValueError("next-edit prediction artifact is missing or oversized")
            row_map: dict[tuple[str, int], dict[str, Any]] = {}
            with predictions_path.open(encoding="utf-8") as handle:
                for _line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row.get("case_id"), row.get("repetition"))
                    if (
                        key[0] not in prompt_by_id
                        or key[1] not in (0, 1)
                        or isinstance(key[1], bool)
                        or key in row_map
                    ):
                        raise ValueError("next-edit result IDs or repetitions are invalid")
                    fixture = diagnostic_by_id[key[0]]
                    prompt = prompt_by_id[key[0]]
                    if (
                        row.get("plan_sha256") != plan["campaign"]["embedded_plan_sha256"]
                        or row.get("model_sha256") != model_sha
                        or row.get("current_sha256") != prompt["current_sha256"]
                        or row.get("original_sha256") != prompt["original_sha256"]
                        or row.get("prompt_sha256") != prompt["prompt_sha256"]
                    ):
                        raise ValueError(f"next-edit result identity mismatch: {key[0]}")
                    ids = row.get("observability_ids")
                    if (
                        not isinstance(ids, dict)
                        or ids.get("case_id") != key[0]
                        or not ids.get("request_id")
                    ):
                        raise ValueError(
                            f"next-edit result is missing its request correlation: {key[0]}"
                        )
                    input_tokens = row.get("input_tokens")
                    if (
                        not isinstance(input_tokens, int)
                        or isinstance(input_tokens, bool)
                        or not isinstance(row.get("context_eligible"), bool)
                    ):
                        raise ValueError("next-edit result has no valid runtime tokenizer count")
                    expected_eligibility = (
                        input_tokens + SWEEP_OUTPUT_TOKENS <= campaign["runtime"]["context_tokens"]
                    )
                    if row["context_eligible"] is not expected_eligibility:
                        raise ValueError(
                            "next-edit context eligibility differs from the frozen limit"
                        )
                    row_map[key] = row
            expected_keys = {
                (case_id, repetition) for case_id in prompt_by_id for repetition in (0, 1)
            }
            if set(row_map) != expected_keys:
                raise ValueError(
                    "next-edit result file is incomplete for the frozen repetition plan"
                )
            scored_rows = []
            for case_index, prompt_row in enumerate(prompt_rows):
                case_id = prompt_row["case_id"]
                fixture = diagnostic_by_id[case_id]
                source = source_by_id.get(fixture.get("source_case_id"))
                adapted = (
                    _adapt_source_case(source, prompt_row, fixture, image_ids)
                    if source is not None
                    else None
                )
                for repetition in (0, 1):
                    row = row_map[(case_id, repetition)]
                    output_row = _score_next_edit_result(
                        row=row,
                        prompt_row=prompt_row,
                        fixture=fixture,
                        source_case=adapted,
                        precision=precision,
                        repetition=repetition,
                        case_index=case_index,
                        temp_root=temp_root,
                        run_context=run_context,
                    )
                    scored_rows.append(output_row)
            output[precision] = scored_rows
            all_rows[precision] = scored_rows
            write_jsonl(output_root / "next-edit" / precision / "results.jsonl", scored_rows)
            (output_root / "next-edit" / precision / "summary.json").parent.mkdir(
                parents=True, exist_ok=True
            )
            (output_root / "next-edit" / precision / "summary.json").write_text(
                json.dumps(_summarize_next_edit(scored_rows), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    q8, q4 = all_rows["q8_0"], all_rows["q4_k_m"]
    paired = {}
    for metric, intent in (
        ("terminal_valid", None),
        ("exact_reference_match", "clear_functional_intent"),
        ("clear_intent_functional_success", "clear_functional_intent"),
        ("no_edit_agreement", "no_edit_plausible"),
        ("exact_reference_match", "ambiguous_requirement"),
        ("synthetic_deletion_reference_match", "explicit_synthetic_target"),
    ):
        q8_values = _case_metric(q8, metric, intent)
        q4_values = _case_metric(q4, metric, intent)
        paired[f"{intent or 'all'}/{metric}"] = _pairwise_bootstrap(q8_values, q4_values)
    summary = {
        "q8_0": _summarize_next_edit(q8),
        "q4_k_m": _summarize_next_edit(q4),
        "paired_q4_minus_q8": paired,
        "interpretation": (
            "synthetic benchmark evidence; ambiguous and no-edit groups are reported separately; "
            "no equivalence or human-quality claim"
        ),
        "raw_result_file_sha256": {
            precision: digest_file(results_root / "next-edit" / precision / "predictions.jsonl")
            for precision in PRECISIONS
        },
    }
    return output, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _paired_quality(
    results_by_precision: dict[str, list[dict[str, Any]]],
    metric: str,
    *,
    eligible_by_precision: dict[str, set[str]],
) -> dict[str, Any]:
    left_rows = {row["case_id"]: row for row in results_by_precision["q8_0"] if row.get("scored")}
    right_rows = {
        row["case_id"]: row for row in results_by_precision["q4_k_m"] if row.get("scored")
    }
    if eligible_by_precision["q8_0"] != eligible_by_precision["q4_k_m"]:
        raise ValueError("Q8 and Q4 prompt eligibility differs; paired comparison is invalid")
    keys = eligible_by_precision["q8_0"]
    values_left: dict[str, float | None] = {key: float(left_rows[key][metric]) for key in keys}
    values_right: dict[str, float | None] = {key: float(right_rows[key][metric]) for key in keys}
    return _pairwise_bootstrap(values_left, values_right)


def _score_line_task(
    *,
    results_root: Path,
    output_root: Path,
    precision: str,
    cases: list[dict[str, Any]],
    plan: dict[str, Any],
    model_sha: str,
    run_context: RunContext,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    task_dir = results_root / "quality" / precision
    path = task_dir / "line-predictions.jsonl"
    suite_path = resolve_recorded_path(plan["inputs"]["line_suite"]["path"])
    prompt_hashes = {row["id"]: digest_bytes(row["prompt"].encode()) for row in cases}
    token_rows = _load_token_rows(
        task_dir / "line-input-token-counts.jsonl", prompt_hashes, CONTROL_OUTPUT_TOKENS
    )
    eligible = {case_id for case_id, row in token_rows.items() if row["context_eligible"]}
    metadata = _verify_metadata(
        prediction_metadata_path(path),
        plan_sha256=plan["campaign"]["embedded_plan_sha256"],
        model_sha256=model_sha,
        suite_sha256=digest_file(suite_path),
        protocol="causal-line-v1",
        max_new_tokens=CONTROL_OUTPUT_TOKENS,
        eligible_count=len(eligible),
    )
    if metadata.get("provider") != "llama.cpp-native" or metadata.get("model_source") != (
        f"{MODEL_REPOSITORY}:{MODEL_REVISION}"
    ):
        raise ValueError("line metadata does not identify the frozen model/provider")
    predictions = _load_predictions(path, eligible)
    _verify_generation_details(
        task_dir / "line-generation-details.jsonl", token_rows, prompt_hashes
    )
    existing = _read_jsonl(task_dir / "causal-line-syntax-results.jsonl")
    existing_by_id = {row.get("case_id"): row for row in existing}
    if len(existing_by_id) != len(existing) or set(existing_by_id) != eligible:
        raise ValueError("line syntax results differ from the frozen eligible set")
    result_rows = []
    exact = syntax = 0
    for case in cases:
        case_id = case["id"]
        if case_id not in eligible:
            result_rows.append({"case_id": case_id, "precision": precision, "scored": False})
            continue
        prediction = predictions[case_id]
        case_context = run_context.for_case(case_id)
        with (
            case_context.activate(),
            operation(
                "model.score",
                attributes={
                    "tabcomplete.task": "causal-line-v1",
                    "tabcomplete.language": case["language"],
                    "tabcomplete.model_alias": "sweep-next-edit-1.5b",
                    "tabcomplete.quantization": precision,
                },
            ),
        ):
            scored = score_line(
                case,
                prediction.completion,
                native_newline_omitted=prediction.finish_reason == "word",
            )
        reference = existing_by_id[case_id]
        for key in ("exact", "syntax", "longest_exact_character_prefix", "source_sha256"):
            if reference.get(key) != scored.get(key):
                raise ValueError(f"stored line scoring differs from frozen rule: {case_id}/{key}")
        exact += bool(scored["exact"])
        syntax += scored["syntax"] == "pass"
        result_rows.append(
            {
                "case_id": case_id,
                "precision": precision,
                "scored": True,
                "response_sha256": digest_bytes(prediction.completion.encode()),
                "finish_reason": prediction.finish_reason,
                "hit_token_cap": prediction.hit_token_cap,
                "exact": scored["exact"],
                "longest_exact_character_prefix": scored["longest_exact_character_prefix"],
                "reference_characters": scored["reference_characters"],
                "returned_characters": scored["returned_characters"],
                "syntax": scored["syntax"],
                "input_tokens": token_rows[case_id]["input_tokens"],
                "latency_seconds": prediction.latency_seconds,
            }
        )
    output_dir = output_root / "line" / precision
    write_jsonl(output_dir / "results.jsonl", result_rows)
    summary = {
        "requested_cases": len(cases),
        "context_eligible_cases": len(eligible),
        "exact_count": exact,
        "exact_rate": exact / len(eligible) if eligible else None,
        "syntax_pass_count": syntax,
        "syntax_pass_rate": syntax / len(eligible) if eligible else None,
        "prediction_file_sha256": digest_file(path),
        "metadata_sha256": digest_file(prediction_metadata_path(path)),
        "runner_scored_results_sha256": digest_file(task_dir / "causal-line-syntax-results.jsonl"),
        "prompt_token_records_sha256": digest_file(task_dir / "line-input-token-counts.jsonl"),
    }
    return result_rows, summary, eligible


def _score_strict_task(
    *,
    results_root: Path,
    output_root: Path,
    precision: str,
    cases: list[BenchmarkCase],
    plan: dict[str, Any],
    model_sha: str,
    run_context: RunContext,
    image_ids: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    task_dir = results_root / "quality" / precision
    path = task_dir / "strict-predictions.jsonl"
    suite_path = resolve_recorded_path(plan["inputs"]["strict_suite"]["path"])
    prompts = {case.id: build_causal_prompt(case) for case in cases}
    prompt_hashes = {case_id: digest_bytes(prompt.encode()) for case_id, prompt in prompts.items()}
    token_rows = _load_token_rows(
        task_dir / "strict-input-token-counts.jsonl", prompt_hashes, CONTROL_OUTPUT_TOKENS
    )
    eligible = {case_id for case_id, row in token_rows.items() if row["context_eligible"]}
    _verify_metadata(
        prediction_metadata_path(path),
        plan_sha256=plan["campaign"]["embedded_plan_sha256"],
        model_sha256=model_sha,
        suite_sha256=digest_file(suite_path),
        protocol="raw-causal-v1",
        max_new_tokens=CONTROL_OUTPUT_TOKENS,
        eligible_count=len(eligible),
    )
    predictions = _load_predictions(path, eligible)
    _verify_generation_details(
        task_dir / "strict-generation-details.jsonl", token_rows, prompt_hashes
    )
    rows, summary = _precision_quality(
        precision=precision,
        task="strict-causal-v1",
        cases=cases,
        predictions=predictions,
        eligible=eligible,
        output_root=output_root,
        run_context=run_context,
        image_ids=image_ids,
    )
    output_dir = output_root / "strict" / precision
    write_jsonl(output_dir / "results.jsonl", rows)
    summary.update(
        {
            "requested_cases": len(cases),
            "context_eligible_cases": len(eligible),
            "context_ineligible_case_ids": sorted(set(token_rows) - eligible),
            "prediction_file_sha256": digest_file(path),
            "metadata_sha256": digest_file(prediction_metadata_path(path)),
            "prompt_token_records_sha256": digest_file(
                task_dir / "strict-input-token-counts.jsonl"
            ),
        }
    )
    return rows, summary, eligible


def _prepare_controller_suites(
    plan: dict[str, Any], diagnostic: dict[str, Any]
) -> tuple[list[BenchmarkCase], list[dict[str, Any]], list[NextEditCase]]:
    strict = load_suite(resolve_recorded_path(plan["inputs"]["strict_suite"]["path"]))
    line = _read_jsonl(
        resolve_recorded_path(plan["inputs"]["line_suite"]["path"]), max_bytes=32 * 1024**2
    )
    source = load_next_edit_suite(resolve_recorded_path(plan["inputs"]["next_edit_suite"]["path"]))
    return strict, line, source


def score_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_frozen_plan(args.plan)
    results_root = args.results_root.resolve()
    final_output_root = args.output.resolve()
    if final_output_root.exists():
        raise FileExistsError("scoring output already exists; preserve it and choose a new attempt")
    if final_output_root == results_root or results_root in final_output_root.parents:
        raise ValueError("scoring output cannot be placed inside raw result artifacts")
    _verify_sandbox(plan["sandbox"])
    campaign = _load_plan(resolve_recorded_path(plan["campaign"]["path"]))
    strict_cases, line_rows, source_cases, diagnostic, prompt_rows = _validate_inputs(
        campaign,
        resolve_recorded_path(plan["inputs"]["strict_suite"]["path"]),
        resolve_recorded_path(plan["inputs"]["line_suite"]["path"]),
        resolve_recorded_path(plan["inputs"]["next_edit_suite"]["path"]),
        resolve_recorded_path(plan["inputs"]["diagnostic_plan"]["path"]),
        resolve_recorded_path(plan["inputs"]["prompt_inputs"]["path"]),
    )
    runtime = current_runtime()
    root_context = RunContext.new(campaign_id="sweep-comparison-score-r1")
    final_output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_root.name}.incomplete-", dir=final_output_root.parent
        )
    )
    output_root = staging_root
    model_hashes = {
        "q8_0": plan["models"]["q8_0_sha256"],
        "q4_k_m": plan["models"]["q4_k_m_sha256"],
    }
    quality_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: {} for task in ("strict", "line")
    }
    quality_summaries: dict[str, dict[str, dict[str, Any]]] = {
        task: {} for task in ("strict", "line")
    }
    eligibility: dict[str, dict[str, set[str]]] = {task: {} for task in ("strict", "line")}
    meta_path = output_root / "observability-run.json"
    try:
        with run_scope(meta_path, "sweep-comparison-controller-score") as scoped_run:
            run_context = scoped_run or root_context
            with operation(
                "campaign.phase",
                attributes={
                    "tabcomplete.phase": "controller_scoring",
                    "tabcomplete.plan.sha256": plan["scoring_plan_sha256"],
                },
            ):
                for precision in PRECISIONS:
                    strict_rows, strict_summary, strict_eligible = _score_strict_task(
                        results_root=results_root,
                        output_root=output_root,
                        precision=precision,
                        cases=strict_cases,
                        plan=plan,
                        model_sha=model_hashes[precision],
                        run_context=run_context,
                        image_ids=plan["sandbox"]["image_ids_by_tag"],
                    )
                    line_rows_scored, line_summary, line_eligible = _score_line_task(
                        results_root=results_root,
                        output_root=output_root,
                        precision=precision,
                        cases=line_rows,
                        plan=plan,
                        model_sha=model_hashes[precision],
                        run_context=run_context,
                    )
                    quality_rows["strict"][precision] = strict_rows
                    quality_rows["line"][precision] = line_rows_scored
                    quality_summaries["strict"][precision] = strict_summary
                    quality_summaries["line"][precision] = line_summary
                    eligibility["strict"][precision] = strict_eligible
                    eligibility["line"][precision] = line_eligible

                strict_pair = {
                    metric: _paired_quality(
                        quality_rows["strict"], metric, eligible_by_precision=eligibility["strict"]
                    )
                    for metric in ("exact_match", "normalized_exact_match", "functional_valid")
                }
                # ``syntax_pass`` is a derived bounded boolean for paired comparison.
                for precision in PRECISIONS:
                    for row in quality_rows["line"][precision]:
                        if row.get("scored"):
                            row["syntax_pass"] = row["syntax"] == "pass"
                line_pair = {
                    metric: _paired_quality(
                        quality_rows["line"], metric, eligible_by_precision=eligibility["line"]
                    )
                    for metric in ("exact", "syntax_pass")
                }
                next_edit_rows, next_edit_summary = _score_next_edit(
                    plan=plan,
                    campaign=campaign,
                    results_root=results_root,
                    output_root=output_root,
                    diagnostic=diagnostic,
                    prompt_rows=prompt_rows,
                    next_cases=source_cases,
                    image_ids=plan["sandbox"]["image_ids_by_tag"],
                    run_context=run_context,
                )
                del next_edit_rows
                summary = {
                    "schema": "sweep-comparison-scoring-results-v1",
                    "scoring_plan_sha256": plan["scoring_plan_sha256"],
                    "campaign_plan_sha256": plan["campaign"]["embedded_plan_sha256"],
                    "results_root": str(results_root),
                    "quality": {
                        task: {
                            precision: {
                                **quality_summaries[task][precision],
                                "case_results_sha256": digest_file(
                                    output_root / task / precision / "results.jsonl"
                                ),
                            }
                            for precision in PRECISIONS
                        }
                        for task in ("strict", "line")
                    },
                    "paired_q4_minus_q8": {"strict": strict_pair, "line": line_pair},
                    "next_edit": next_edit_summary,
                    "limitations": [
                        "The 24 next-edit cases form a small public/synthetic diagnostic slice.",
                        "Ambiguous cases show reference agreement, not universal quality labels.",
                        "No-edit labels are plausible references, not certain human intent.",
                        "Two synthetic deletion cases have no functional source oracle.",
                        "Small paired samples do not support equivalence claims.",
                    ],
                }
                output_root.mkdir(parents=True, exist_ok=True)
                (output_root / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                artifact = {
                    "schema": "sweep-comparison-score-artifact-manifest-v1",
                    "scoring_plan_sha256": plan["scoring_plan_sha256"],
                    "raw_prediction_files": {
                        f"{precision}/{task}": digest_file(
                            results_root
                            / "quality"
                            / precision
                            / f"{'strict' if task == 'strict' else 'line'}-predictions.jsonl"
                        )
                        for precision in PRECISIONS
                        for task in ("strict", "line")
                    }
                    | {
                        f"{precision}/next_edit": digest_file(
                            results_root / "next-edit" / precision / "predictions.jsonl"
                        )
                        for precision in PRECISIONS
                    },
                    "controller_outputs": {},
                }
                for path in sorted(output_root.rglob("*")):
                    if path.is_file() and path.name != "artifact_manifest.json":
                        artifact["controller_outputs"][path.relative_to(output_root).as_posix()] = (
                            digest_file(path)
                        )
                (output_root / "artifact_manifest.json").write_text(
                    json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                staging_root.replace(final_output_root)
                return summary
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        runtime.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-plan", action="store_true")
    modes.add_argument("--score", action="store_true")
    parser.add_argument("--campaign-plan", type=Path, default=CAMPAIGN_PLAN_DEFAULT)
    parser.add_argument("--strict-suite", type=Path, default=STRICT_SUITE_DEFAULT)
    parser.add_argument("--line-suite", type=Path, default=LINE_SUITE_DEFAULT)
    parser.add_argument("--next-edit-suite", type=Path, default=NEXT_EDIT_SUITE_DEFAULT)
    parser.add_argument("--diagnostic-plan", type=Path, default=DIAGNOSTIC_PLAN_DEFAULT)
    parser.add_argument("--prompt-inputs", type=Path, default=NEXT_EDIT_INPUTS_DEFAULT)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.score and (args.results_root is None or args.output is None):
        parser.error("--score requires --results-root and --output")
    return args


def main() -> None:
    args = _parse_args()
    if args.freeze_plan:
        plan = freeze_plan(args)
        write_frozen_plan(args.plan, plan)
        print(
            json.dumps(
                {
                    "state": "scoring_plan_frozen",
                    "path": str(args.plan),
                    "scoring_plan_sha256": plan["scoring_plan_sha256"],
                    "raw_file_sha256": digest_file(args.plan),
                },
                sort_keys=True,
            )
        )
    else:
        summary = score_plan(args)
        print(
            json.dumps(
                {
                    "state": "scored",
                    "scoring_plan_sha256": summary["scoring_plan_sha256"],
                    "summary": str(args.output / "summary.json"),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
