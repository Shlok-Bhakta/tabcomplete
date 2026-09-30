#!/usr/bin/env python3
"""Run the frozen, two-precision Sweep comparison without training or code execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODEL_REPOSITORY = "sweepai/sweep-next-edit-1.5B"
MODEL_REVISION = "409016591c6c1a94f545f22328a85a3516118f34"
MODEL_FILE = "sweep-next-edit-1.5b.q8_0.v2.gguf"
MODEL_BYTES = 1_537_269_856
MODEL_SHA256 = "1321ea5e5d7529e60f9770c6a0b3a965f89542d16cf4ae51bab267f6a88150da"
Q4_SHA256 = "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4"
RUNTIME_REVISION = "f072b103714dfa1eee531f80b24512faf38e3dd2"
STRICT_SHA256 = "ed28739b302e4c0d3f9f45e859e7ccd68a1e8594a62eb7b13ee8425b92a439a4"
LINE_SHA256 = "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead"
NEXT_EDIT_SHA256 = "69c42041f499ac5839195573bfbe691565d013ca132e8e183b5fc3fa61f0a0ff"
DIAGNOSTIC_PLAN_SHA256 = "efa0bf5c245b5edca14df2722b19c6e086e790a3da256656eec1e8db2ed34c34"
ARTIFACT_PREP_PLAN_SHA256 = "b8090328b12d33c29bd81cc263f1ab34df69c98211115952a757aa6edfe6c366"
Q8_DOWNLOAD_MANIFEST_SHA256 = "76efae54e19d0f44256b003aa691fab08602acc6efac5fd9432e24ce54b2cd0a"
Q4_MANIFEST_SHA256 = "856be1ea89966d138d5707a539dc1a252558cbb0390994caa3ece6aad162ea22"
MODEL_CARD_SHA256 = "f1495e987e82ea2497ea711ffd3d9a8db8a6610dce82396839609140bb1fe559"
PUBLISHER_RUNNER_SHA256 = "9eef7df11518d2b3f091aebb07a3f92560d5a2da15711598ecd87cf07b273fc2"
CONTEXT_TOKENS = 8192
CONTROL_OUTPUT_TOKENS = 96
SWEEP_OUTPUT_TOKENS = 512
CAMPAIGN_TOKEN_CEILING = 2_000_000
ARTIFACT_CAP_BYTES = 12 * 1024**3
DISK_RESERVE_BYTES = 2 * 1024**3
EXPECTED_CASE_IDS = (
    "python/stable_11",
    "python/stable_12",
    "python/stable_13",
    "python/100",
    "javascript/500",
    "javascript/501",
    "javascript/502",
    "javascript/509",
    "typescript/0",
    "typescript/1",
    "typescript/2",
    "typescript/3",
    "rust/1",
    "rust/5",
    "rust/6",
    "rust/8",
    "java/5",
    "java/7",
    "java/8",
    "csharp/3",
    "c/stable_12",
    "c/stable_17",
    "synthetic/delete-javascript-debug",
    "synthetic/delete-rust-debug",
)
TERMINAL_OUTCOMES = {"eos", "word"}


def terminal_status(
    stop_type: Any, truncated: Any, output_tokens: int, limit: int
) -> tuple[bool, bool]:
    """Only an observed terminal event counts; an unterminated cap hit stays invalid."""
    explicit_terminal = stop_type in TERMINAL_OUTCOMES and truncated is not True
    hit_token_cap = truncated is True or (output_tokens >= limit and not explicit_terminal)
    return explicit_terminal, hit_token_cap


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file_identity(path: Path, expected_sha256: str, label: str) -> None:
    if digest_file(path) != expected_sha256:
        raise ValueError(f"{label} identity differs from frozen plan")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def plan_digest(plan: dict[str, Any]) -> str:
    return digest_bytes(
        canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def deadline_remaining() -> float:
    raw = os.environ.get("TABCOMPLETE_SWEEP_DEADLINE")
    if not raw:
        return float("inf")
    remaining = float(raw) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Sweep session finalization reserve reached")
    return remaining


def artifact_usage(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def require_storage(root: Path, additional_bytes: int) -> dict[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    current = artifact_usage(root)
    free = shutil.disk_usage(root).free
    if current + additional_bytes > ARTIFACT_CAP_BYTES:
        raise RuntimeError("Sweep artifact cap would be exceeded")
    if free - additional_bytes < DISK_RESERVE_BYTES:
        raise RuntimeError("Sweep filesystem free-space reserve would be violated")
    return {"current_bytes": current, "additional_bytes": additional_bytes, "free_bytes": free}


def _reconstruct_before(current: str, delta: dict[str, Any], case_id: str) -> str:
    raw = current.encode("utf-8")
    has_final_newline = raw.endswith(b"\n")
    lines = raw.split(b"\n")
    if has_final_newline:
        lines.pop()
    start = delta.get("start_row")
    if not isinstance(start, int) or start < 0 or start > len(lines):
        raise ValueError(f"invalid delta row for {case_id}")
    old_text = delta.get("deleted_text")
    new_text = delta.get("inserted_text")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        raise ValueError(f"invalid delta text for {case_id}")
    old = old_text.encode("utf-8").split(b"\n") if old_text else []
    new = new_text.encode("utf-8").split(b"\n") if new_text else []
    if lines[start : start + len(new)] != new:
        raise ValueError(f"delta does not match current source for {case_id}")
    previous = lines[:start] + old + lines[start + len(new) :]
    replayed = previous[:start] + new + previous[start + len(old) :]
    suffix = b"\n" if has_final_newline else b""
    if b"\n".join(replayed) + suffix != raw:
        raise ValueError(f"delta replay is not byte-identical for {case_id}")
    return (b"\n".join(previous) + suffix).decode("utf-8")


def build_sweep_prompt(case: dict[str, Any]) -> str:
    """Match the pinned publisher build_prompt implementation and field order."""
    parts: list[str] = []
    context = case.get("context_files", {})
    if not isinstance(context, dict):
        raise ValueError("context_files must be an object")
    for path, content in context.items():
        parts.extend((f"<|file_sep|>{path}", content))
    diffs = case.get("recent_diffs", [])
    if not isinstance(diffs, list):
        raise ValueError("recent_diffs must be a list")
    for diff in diffs:
        parts.extend(
            (
                f"<|file_sep|>{diff['file_path']}.diff",
                "original:",
                diff["original"],
                "updated:",
                diff["updated"],
            )
        )
    path = case["file_path"]
    parts.extend(
        (
            f"<|file_sep|>original/{path}",
            case["original_content"],
            f"<|file_sep|>current/{path}",
            case["current_content"],
            f"<|file_sep|>updated/{path}",
        )
    )
    return "\n".join(parts)


def build_next_edit_inputs(diagnostic: dict[str, Any], suite_path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suite = {row["id"]: row for row in rows}
    fixtures = diagnostic.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 24:
        raise ValueError("the pinned development diagnostic must contain exactly 24 cases")
    cases: list[dict[str, Any]] = []
    for fixture in fixtures:
        case_id = fixture.get("case_id")
        if case_id not in EXPECTED_CASE_IDS or not fixture.get(
            "history_reconstructs_current_byte_exactly"
        ):
            raise ValueError("diagnostic fixture identity or history evidence is invalid")
        current = fixture.get("current")
        delta = fixture.get("recent_delta")
        path = fixture.get("path")
        if not isinstance(current, str) or not isinstance(delta, dict) or not isinstance(path, str):
            raise ValueError(f"missing prompt state for {case_id}")
        original = _reconstruct_before(current, delta, case_id)
        if digest_bytes(original.encode()) != fixture.get("pre_edit_source_sha256"):
            raise ValueError(f"reconstructed original hash differs for {case_id}")
        if fixture.get("source_case_id") is None:
            context_files: dict[str, str] = {}
        else:
            source_row = suite.get(fixture["source_case_id"])
            if source_row is None:
                raise ValueError(f"source suite lacks selected case {case_id}")
            context_files = dict(source_row.get("context_files") or {})
        current_bytes = current.encode("utf-8")
        start = fixture.get("cursor_byte_offset")
        end = fixture.get("region_end_byte_offset")
        if not isinstance(start, int):
            raise ValueError(f"invalid editable range start for {case_id}")
        if end is None:
            end = current_bytes.find(b"\n", start)
            if end < 0:
                end = len(current_bytes)
        if not isinstance(end, int) or not 0 <= start <= end <= len(current_bytes):
            raise ValueError(f"invalid editable range end for {case_id}")
        for offset in (start, end):
            try:
                current_bytes[:offset].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"editable range is not on a UTF-8 boundary for {case_id}"
                ) from exc
        case = {
            "case_id": case_id,
            "language": fixture["language"],
            "file_path": path,
            "context_files": context_files,
            "recent_diffs": [{"file_path": path, "original": original, "updated": current}],
            "original_content": original,
            "current_content": current,
            "input_semantics": "public synthetic state; original rebuilt from exact recent_delta",
            "current_sha256": digest_bytes(current.encode()),
            "original_sha256": digest_bytes(original.encode()),
            "editable_start_byte": start,
            "editable_end_byte": end,
        }
        prompt = build_sweep_prompt(case)
        case["prompt_sha256"] = digest_bytes(prompt.encode())
        cases.append(case)
    if tuple(case["case_id"] for case in cases) != EXPECTED_CASE_IDS:
        raise ValueError("selected fixture ordering changed")
    return cases


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _runtime_revision(runtime: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()


def freeze_plan(args: argparse.Namespace) -> None:
    if digest_file(args.strict_suite) != STRICT_SHA256:
        raise ValueError("strict causal suite identity mismatch")
    if digest_file(args.line_suite) != LINE_SHA256:
        raise ValueError("line suite identity mismatch")
    if digest_file(args.next_edit_suite) != NEXT_EDIT_SHA256:
        raise ValueError("next-edit source suite identity mismatch")
    if digest_file(args.diagnostic_plan) != DIAGNOSTIC_PLAN_SHA256:
        raise ValueError("diagnostic plan identity mismatch")
    if _runtime_revision(args.runtime) != RUNTIME_REVISION:
        raise ValueError("llama.cpp source revision mismatch")
    artifacts = {
        "q8": args.artifact_dir / MODEL_FILE,
        "q4": args.artifact_dir / "sweep-next-edit-1.5b.q4_k_m.gguf",
    }
    if not artifacts["q8"].is_file() or digest_file(artifacts["q8"]) != MODEL_SHA256:
        raise ValueError("the pinned Q8 artifact is not staged")
    if not artifacts["q4"].is_file() or digest_file(artifacts["q4"]) != Q4_SHA256:
        raise ValueError("the pinned Q4 requantized artifact is not staged")
    diagnostic = json.loads(args.diagnostic_plan.read_text(encoding="utf-8"))
    if (
        diagnostic.get("plan_sha256")
        != "153f2211be603f0c51a9abbe8086ecaaffa6198eda543859d704b512e7bbb03a"
    ):
        raise ValueError("diagnostic plan revision mismatch")
    cases = build_next_edit_inputs(diagnostic, args.next_edit_suite)
    quota = json.loads(args.quota_record.read_text(encoding="utf-8"))
    if len(args.strict_suite.read_text(encoding="utf-8").splitlines()) != 200:
        raise ValueError("strict causal suite count changed")
    if len(args.line_suite.read_text(encoding="utf-8").splitlines()) != 180:
        raise ValueError("causal line suite count changed")
    args.next_edit_fixtures.parent.mkdir(parents=True, exist_ok=True)
    fixture_payload = "".join(
        json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases
    )
    if (
        args.next_edit_fixtures.exists()
        and args.next_edit_fixtures.read_text(encoding="utf-8") != fixture_payload
    ):
        raise FileExistsError(
            "existing next-edit fixture bundle differs; preserve it and revise the plan"
        )
    args.next_edit_fixtures.write_text(fixture_payload, encoding="utf-8")
    assess = [
        fixture["target_assessment"]["functional_intent"] for fixture in diagnostic["fixtures"]
    ]
    plan: dict[str, Any] = {
        "schema": "sweep-comparison-plan-v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "campaign": "sweep-q8-q4-r1",
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "license": "apache-2.0",
            "total_parameters": 1_445_014_016,
            "architecture": "qwen2",
            "eos_token": "<|endoftext|>",
            "publisher_context_tokens": 8192,
            "hub_gguf_context_tokens": 32768,
            "context_tokens_used": CONTEXT_TOKENS,
            "context_discrepancy": "Hub GGUF says 32768; card says 8192; this plan uses 8192.",
            "artifacts": {
                "q8_0": {"file": MODEL_FILE, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256},
                "q4_k_m": {
                    "file": artifacts["q4"].name,
                    "bytes": artifacts["q4"].stat().st_size,
                    "sha256": Q4_SHA256,
                    "source_sha256": MODEL_SHA256,
                    "provenance": "Q8_0 requantized to Q4_K_M, not F16/F32-to-Q4",
                },
            },
        },
        "runtime": {
            "repository_revision": RUNTIME_REVISION,
            "build": (
                "llama.cpp f072b103714dfa1eee531f80b24512faf38e3dd2; "
                "Kaggle CUDA_ARCHITECTURES=75; local compatibility smoke is separate"
            ),
            "one_model_loaded_at_a_time": True,
            "parallel_slots": 1,
            "context_tokens": CONTEXT_TOKENS,
            "batch_tokens": 256,
            "microbatch_tokens": 64,
            "prompt_cache": False,
            "cache_ram_bytes": 0,
            "idle_slot_cache": False,
            "context_shift": False,
            "speculative_decoding": False,
            "gpu_layers": 99,
            "split_mode": "none",
            "main_gpu": 0,
            "kv_cache_precision": "f16",
            "server_flags": [
                "--ctx-size",
                "8192",
                "--batch-size",
                "256",
                "--ubatch-size",
                "64",
                "--parallel",
                "1",
                "--cache-ram",
                "0",
                "--no-cache-idle-slots",
                "--no-context-shift",
                "--cache-type-k",
                "f16",
                "--cache-type-v",
                "f16",
                "--split-mode",
                "none",
                "--main-gpu",
                "0",
                "--n-gpu-layers",
                "99",
            ],
        },
        "comparison": {
            "single_runtime_for_both_precisions": True,
            "quality_controls": {
                "strict_causal": {
                    "path": "data/benchmarks/code_completion_v2.jsonl",
                    "sha256": STRICT_SHA256,
                    "cases": 200,
                    "prompt": "raw source prefix; no chat template",
                    "max_output_tokens": CONTROL_OUTPUT_TOKENS,
                    "purpose": "causal continuation task-mismatch control, not next-edit quality",
                },
                "causal_line": {
                    "path": "causal_line_v1-r3.jsonl",
                    "sha256": LINE_SHA256,
                    "cases": 180,
                    "prompt": "exact fixture prompt, raw causal completion",
                    "max_output_tokens": CONTROL_OUTPUT_TOKENS,
                    "purpose": "line continuation task-mismatch control, not next-edit quality",
                },
            },
            "next_edit": {
                "protocol": (
                    "Pinned Sweep run_model.py build_prompt, whole-file output; no action codec"
                ),
                "run_model_py_sha256": PUBLISHER_RUNNER_SHA256,
                "model_card_sha256": MODEL_CARD_SHA256,
                "source_suite": {
                    "path": "data/benchmarks/next_edit_v2.jsonl",
                    "sha256": NEXT_EDIT_SHA256,
                },
                "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
                "diagnostic_revision": diagnostic["plan_sha256"],
                "fixture_count": 24,
                "fixture_ids": [case["case_id"] for case in cases],
                "fixture_input_path": args.next_edit_fixtures.name,
                "fixture_input_sha256": digest_file(args.next_edit_fixtures),
                "target_groups_counts": {name: assess.count(name) for name in sorted(set(assess))},
                "gold_labels_excluded_from_worker_inputs": True,
                "max_output_tokens": SWEEP_OUTPUT_TOKENS,
                "repetitions": 2,
                "decoding": {"temperature": 0, "stop": ["<|file_sep|>", "</s>"]},
                "input_order": [
                    "context_files",
                    "recent_diffs",
                    "original",
                    "current",
                    "updated header",
                ],
                "history": (
                    "Original/current and the full-file recent diff use only exact reversible "
                    "recent_delta; no gold-derived state"
                ),
                "mapping": (
                    "Require byte-identical text outside the editable range; otherwise mark "
                    "out-of-range without repair"
                ),
            },
            "tokenizer": (
                "Resident llama.cpp /tokenize endpoint, add_special and parse_special; "
                "no second tokenizer stack"
            ),
            "input_budget": (
                "Exact runtime token count plus output ceiling <=8192; otherwise skip without "
                "truncation"
            ),
            "prompt_layout_changes": "none; official Sweep prompt retained",
            "precision_pairing": (
                "Exact Q8_0 and Q4_K_M artifacts use identical prompts, runtime source, flags, "
                "context, cache, output ceilings, and order"
            ),
            "cache": "cache_prompt=false; cache-ram=0; no idle snapshots",
        },
        "budget": {
            "kaggle_session_seconds_max": 7200,
            "finalization_reserve_seconds": 1200,
            "deadline_enforced_by": "TABCOMPLETE_SWEEP_DEADLINE checked per stage and request",
            "active_gpu_jobs": 1,
            "gpu_device": "GPU 0 only; no model split",
            "training": False,
            "additional_training_tokens": 0,
            "paid_compute": False,
            "additional_model_downloads": False,
            "inference_token_ceiling_across_campaign": CAMPAIGN_TOKEN_CEILING,
            "storage_cap_bytes": ARTIFACT_CAP_BYTES,
            "minimum_filesystem_reserve_bytes": DISK_RESERVE_BYTES,
        },
        "quota_observation": {
            "observed_at": quota["observed_at"],
            "remaining": quota["remaining"],
            "total": quota["total"],
            "units": quota["units"],
            "renewal": quota["renewal"],
            "source": quota["source"],
            "active_jobs": quota["active_jobs"],
            "source_sha256": digest_file(args.quota_record),
        },
        "inputs": {
            "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
            "artifact_preparation_plan_sha256": ARTIFACT_PREP_PLAN_SHA256,
            "download_manifest_sha256": Q8_DOWNLOAD_MANIFEST_SHA256,
            "quantization_manifest_sha256": Q4_MANIFEST_SHA256,
            "quota_preparation_sha256": digest_file(args.quota_record),
        },
        "host_preparation": {
            "host": socket.gethostname(),
            "local_cuda": False,
            "local_timed_inference": False,
            "note": "crabcake has no NVIDIA GPU; Q8/Q4 quality comparison uses Kaggle T4",
        },
        "code": {
            "runner_sha256": digest_file(Path(__file__).resolve()),
            "test_sha256": digest_file(args.test_file) if args.test_file.exists() else None,
            "native_provider_sha256": digest_file(
                Path(__file__).resolve().with_name("measure_r2_local.py")
            ),
            "python": "3.11",
            "no_model_generated_code_execution": True,
        },
        "observability": {
            "revision": 2,
            "sdk_versions": {
                "opentelemetry-api": importlib.metadata.version("opentelemetry-api"),
                "opentelemetry-sdk": importlib.metadata.version("opentelemetry-sdk"),
            },
            "required_worker_mode": "offline",
            "required_worker_enabled": True,
            "required_public_synthetic_content_capture": True,
            "required_max_payload_bytes": 8 * 2**20,
            "required_offline_bundle_max_bytes": 128 * 2**20,
            "domain_ids": [
                "campaign_id",
                "run_id",
                "run_attempt_id",
                "case_id",
                "case_attempt_id",
                "request_id",
                "trace_id",
            ],
            "spans": ["run.start", "run.heartbeat", "run.summary", "eval.case", "model.generate"],
            "capture": (
                "ArtifactStore captures public/synthetic requests and raw responses; prompts are "
                "not metric labels."
            ),
        },
    }
    if args.supersedes_plan is not None:
        old_plan = json.loads(args.supersedes_plan.read_text(encoding="utf-8"))
        if old_plan.get("schema") != "sweep-comparison-plan-v1" or plan_digest(
            old_plan
        ) != old_plan.get("plan_sha256"):
            raise ValueError("superseded plan is not a valid frozen comparison plan")
        plan["revision"] = {
            "number": 2,
            "supersedes_plan_sha256": digest_file(args.supersedes_plan),
            "reason": (
                "Adds offline OTel run/case/request correlation and actual ArtifactStore "
                "prompt/response "
                "capture, strict-suite fingerprint verification, cached per-model hashes without "
                "per-case rehashing, and "
                "SDK shutdown; model, prompts, fixtures, decoding, caps, and scoring are unchanged."
            ),
            "scientific_inputs_unchanged": True,
        }
    elif args.plan.name == "plan-v2.json":
        raise ValueError("plan-v2 freeze requires --supersedes-plan")
    plan["plan_sha256"] = plan_digest(plan)
    if args.plan.exists():
        if json.loads(args.plan.read_text(encoding="utf-8")) != plan:
            raise FileExistsError("plan exists with different contents; create a new plan revision")
    else:
        write_json(args.plan, plan)
    print(
        json.dumps(
            {
                "state": "frozen",
                "plan": str(args.plan),
                "plan_sha256": plan["plan_sha256"],
                "fixture_bundle_sha256": plan["comparison"]["next_edit"]["fixture_input_sha256"],
            },
            sort_keys=True,
        )
    )


def load_and_verify_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "sweep-comparison-plan-v1" or plan_digest(plan) != plan.get(
        "plan_sha256"
    ):
        raise ValueError("frozen Sweep plan schema or self-hash mismatch")
    if digest_file(Path(__file__).resolve()) != plan["code"]["runner_sha256"]:
        raise ValueError("runner source differs from frozen plan")
    if "revision" in plan and plan["revision"].get("number") != 2:
        raise ValueError("unsupported Sweep plan revision")
    if "revision" not in plan:
        raise ValueError("Sweep plan predates required observability and strict-input checks")
    verify_file_identity(args.test_file, plan["code"]["test_sha256"], "runner test")
    if "native_provider_sha256" in plan["code"]:
        verify_file_identity(
            Path(__file__).resolve().with_name("measure_r2_local.py"),
            plan["code"]["native_provider_sha256"],
            "native provider transport",
        )
    verify_file_identity(
        args.strict_suite,
        plan["comparison"]["quality_controls"]["strict_causal"]["sha256"],
        "strict causal suite",
    )
    if _runtime_revision(args.runtime) != RUNTIME_REVISION:
        raise ValueError("llama.cpp source revision differs from frozen plan")
    verify_file_identity(args.line_suite, LINE_SHA256, "line fixture")
    verify_file_identity(
        args.next_edit_suite,
        plan["comparison"]["next_edit"]["source_suite"]["sha256"],
        "next-edit source suite",
    )
    if (
        args.next_edit_fixtures
        and digest_file(args.next_edit_fixtures)
        != plan["comparison"]["next_edit"]["fixture_input_sha256"]
    ):
        raise ValueError("next-edit fixture input differs from frozen plan")
    if args.gpu_layers != 99:
        raise ValueError("frozen comparison requires the declared Kaggle GPU 0 backend")
    return plan


def download_q8(args: argparse.Namespace) -> Path:
    target = args.artifact_dir / MODEL_FILE
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == MODEL_BYTES and digest_file(target) == MODEL_SHA256:
            return target
        raise ValueError("existing Q8 artifact does not match the pinned identity")
    estimate = MODEL_BYTES
    require_storage(args.artifact_budget_root, estimate)
    temporary = target.with_suffix(target.suffix + ".part")
    url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
    request = urllib.request.Request(url, headers={"User-Agent": "tabcomplete-sweep-comparison-r1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=min(60, deadline_remaining())) as response,
            temporary.open("wb") as output,
        ):
            if response.headers.get("Content-Length") != str(MODEL_BYTES):
                raise ValueError("Hub response length differs from the pinned model file")
            total = 0
            while block := response.read(8 * 1024 * 1024):
                deadline_remaining()
                total += len(block)
                if total > MODEL_BYTES:
                    raise ValueError("model download exceeded the pinned size")
                output.write(block)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"pinned Sweep download failed: {type(exc).__name__}") from None
    if total != MODEL_BYTES or digest_file(temporary) != MODEL_SHA256:
        raise ValueError("downloaded Sweep Q8 artifact failed size or SHA-256 validation")
    os.replace(temporary, target)
    return target


def quantize_q4(args: argparse.Namespace) -> Path:
    source = args.artifact_dir / MODEL_FILE
    target = args.artifact_dir / "sweep-next-edit-1.5b.q4_k_m.gguf"
    if (
        not source.is_file()
        or source.stat().st_size != MODEL_BYTES
        or digest_file(source) != MODEL_SHA256
    ):
        raise ValueError("pinned Q8 source is missing or corrupt")
    if target.exists():
        if digest_file(target) != Q4_SHA256:
            raise ValueError("existing Q4 artifact differs from the frozen exact conversion")
        return target
    require_storage(args.artifact_budget_root, 1024**3)
    temporary = target.with_suffix(target.suffix + ".part")
    quantizer = args.runtime / "build/bin/llama-quantize"
    if not quantizer.is_file():
        raise FileNotFoundError("pinned llama-quantize binary is missing")
    deadline_remaining()
    result = subprocess.run(
        [str(quantizer), "--allow-requantize", str(source), str(temporary), "Q4_K_M", "4"],
        capture_output=True,
        text=True,
        timeout=max(1.0, deadline_remaining()),
        check=False,
    )
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"pinned Q8-to-Q4 requantization failed with code {result.returncode}")
    if not temporary.is_file() or temporary.stat().st_size > 1024**3:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("requantized Q4 file is missing or exceeds its storage allowance")
    output_hash = digest_file(temporary)
    if output_hash != Q4_SHA256:
        observed = {
            "expected_sha256": Q4_SHA256,
            "actual_sha256": output_hash,
            "bytes": temporary.stat().st_size,
            "source_sha256": MODEL_SHA256,
            "quantizer_revision": RUNTIME_REVISION,
        }
        write_json(args.output / "q4_identity_mismatch.json", observed)
        raise ValueError("requantized Q4 bytes differ from frozen artifact; no inference was run")
    os.replace(temporary, target)
    return target


class Server:
    def __init__(self, args: argparse.Namespace, model: Path):
        self.args = args
        self.model = model
        self.process: subprocess.Popen[str] | None = None
        self.port = self._port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.samples: list[dict[str, int | float]] = []
        self._sample_stop = threading.Event()
        self._sampler: threading.Thread | None = None

    @staticmethod
    def _port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def __enter__(self) -> Server:
        import httpx

        server = self.args.runtime / "build/bin/llama-server"
        self.host_before = _host_snapshot()
        self.gpu_before = _gpu_snapshot()
        log_path = self.args.output / f"server-{self.model.stem}-{time.time_ns()}.log"
        log = log_path.open("wb")
        argv = [
            str(server),
            "--log-verbosity",
            "4",
            "--model",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(CONTEXT_TOKENS),
            "--batch-size",
            "256",
            "--ubatch-size",
            "64",
            "--parallel",
            "1",
            "--threads",
            "4",
            "--threads-batch",
            "4",
            "--cache-ram",
            "0",
            "--no-cache-idle-slots",
            "--no-context-shift",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--split-mode",
            "none",
            "--main-gpu",
            "0",
            "--n-gpu-layers",
            str(self.args.gpu_layers),
            "--no-webui",
        ]
        self.argv = argv
        self.log = log
        self.log_path = log_path
        self.started_at = time.monotonic()
        self.process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, text=True)
        end = min(time.monotonic() + 600, time.monotonic() + deadline_remaining())
        try:
            while time.monotonic() < end:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "llama-server exited during startup; inspect its private log"
                    )
                try:
                    response = httpx.get(self.url + "/health", timeout=2)
                    if response.status_code == 200:
                        self.loaded_at = time.monotonic()
                        self.gpu_after_load = _gpu_snapshot()
                        self.backend_evidence = _runtime_backend_evidence(
                            self.log_path, self.gpu_after_load
                        )
                        write_json(
                            self.log_path.with_suffix(".backend-startup.json"),
                            {
                                "server_pid": self.process.pid,
                                "argv": self.argv,
                                "gpu_before": self.gpu_before,
                                "gpu_after_load": self.gpu_after_load,
                                "backend_evidence": self.backend_evidence,
                            },
                        )
                        if not self.backend_evidence["cuda_device_count"]:
                            raise RuntimeError(
                                "llama-server did not report an available CUDA device"
                            )
                        if not self.backend_evidence["offloaded_layers"]:
                            raise RuntimeError(
                                "llama-server did not confirm any model layers offloaded to CUDA"
                            )
                        self.sample()
                        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
                        self._sampler.start()
                        return self
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            raise TimeoutError("llama-server health endpoint did not become ready")
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.log.close()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._sample_stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2)
        self.sample()
        self.host_after = _host_snapshot()
        self.gpu_after_requests = _gpu_snapshot()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if hasattr(self, "log"):
            self.log.close()

    def tokenize(self, prompt: str) -> int:
        import httpx

        deadline_remaining()
        response = httpx.post(
            self.url + "/tokenize",
            json={
                "content": prompt,
                "add_special": True,
                "parse_special": True,
            },
            timeout=min(60, max(1, deadline_remaining())),
        )
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("runtime tokenizer returned a malformed result")
        return len(tokens)

    def sample(self) -> dict[str, Any]:
        if self.process is None:
            return {}
        sample = _process_memory_snapshot(self.process.pid)
        if sample:
            sample["sampled_at_monotonic_seconds"] = time.monotonic()
            self.samples.append(sample)
        return sample

    def _sample_loop(self) -> None:
        while not self._sample_stop.wait(0.5):
            self.sample()

    def memory_report(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0, "peak": None, "post_request_retained": None}
        keys = (
            "rss_bytes",
            "rss_high_water_bytes",
            "pss_bytes",
            "anonymous_bytes",
            "private_bytes",
            "pss_file_bytes",
            "pss_anonymous_bytes",
            "process_swap_bytes",
        )
        peaks = {key: max(sample.get(key, 0) for sample in self.samples) for key in keys}
        return {
            "sample_period_seconds": 0.5,
            "samples": len(self.samples),
            "peak": peaks,
            "post_request_retained": self.samples[-1],
        }


def _proc_kib_fields(path: Path, wanted: set[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            fields = value.strip().split()
            if fields:
                try:
                    result[key] = int(fields[0]) * (
                        1024 if len(fields) > 1 and fields[1] == "kB" else 1
                    )
                except ValueError:
                    pass
    return result


def _proc_page_fields(path: Path, wanted: set[str]) -> dict[str, int]:
    """Read numeric /proc/vmstat counters, whose values are in pages."""
    result: dict[str, int] = {}
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in wanted:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return result


def _process_memory_snapshot(pid: int) -> dict[str, int | float]:
    process = Path(f"/proc/{pid}")
    status = _proc_kib_fields(process / "status", {"VmRSS", "VmHWM", "VmSwap"})
    maps = _proc_kib_fields(
        process / "smaps_rollup",
        {
            "Pss",
            "Anonymous",
            "Private_Clean",
            "Private_Dirty",
            "Swap",
            "Pss_File",
            "Pss_Anon",
        },
    )
    if not status:
        return {}
    return {
        "rss_bytes": status.get("VmRSS", 0),
        "rss_high_water_bytes": status.get("VmHWM", 0),
        "pss_bytes": maps.get("Pss", 0),
        "anonymous_bytes": maps.get("Anonymous", 0),
        "private_bytes": maps.get("Private_Clean", 0) + maps.get("Private_Dirty", 0),
        "pss_file_bytes": maps.get("Pss_File", 0),
        "pss_anonymous_bytes": maps.get("Pss_Anon", 0),
        "process_swap_bytes": max(status.get("VmSwap", 0), maps.get("Swap", 0)),
    }


def _host_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    mem = _proc_kib_fields(
        Path("/proc/meminfo"), {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    )
    vm = _proc_page_fields(Path("/proc/vmstat"), {"pswpin", "pswpout"})
    result["memory_bytes"] = mem
    result["swap_activity_pages"] = vm
    for name in ("memory", "io"):
        pressure = Path(f"/proc/pressure/{name}")
        result[f"pressure_{name}"] = (
            pressure.read_text().strip() if pressure.exists() else "unavailable"
        )
    return result


def _runtime_backend_evidence(log_path: Path, gpu_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Extract runtime-confirmed CUDA use; requested flags alone are not evidence."""
    import re

    content = log_path.read_text(errors="replace") if log_path.exists() else ""
    device_counts = [
        int(match.group(1))
        for match in re.finditer(r"found\s+(\d+)\s+CUDA devices?", content, re.IGNORECASE)
    ]
    # CUDA's one-time initializer may run before argument parsing configures
    # logging. The pinned runtime also enumerates its actual devices after
    # parsing; use that inventory rather than inferring devices from flags,
    # host buffers, or nvidia-smi alone.
    enumerated_devices = sorted(
        {
            match.group(1)
            for match in re.finditer(
                r"(?:^|\s)common_param:\s+-\s+(CUDA\d+)\s*:\s*\S[^\n]*$",
                content,
                re.MULTILINE,
            )
        }
    )
    inventory_conflict = bool(
        device_counts and max(device_counts) != len(enumerated_devices) and enumerated_devices
    )
    if inventory_conflict:
        device_count, device_count_source = 0, "conflicting_runtime_inventory"
    elif device_counts:
        device_count, device_count_source = max(device_counts), "runtime_cuda_initializer"
    else:
        device_count, device_count_source = len(enumerated_devices), "runtime_device_inventory"
    offloads = [
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(
            r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", content, re.IGNORECASE
        )
    ]
    evidence_lines = [
        line.strip()
        for line in content.splitlines()
        if "CUDA" in line.upper() or "OFFLOADED" in line.upper()
    ][-20:]
    offloaded, total = max(offloads, default=(0, 0))
    return {
        "backend": "CUDA" if device_count > 0 and offloaded > 0 else "unverified",
        "cuda_device_count": device_count,
        "cuda_device_count_source": device_count_source,
        "enumerated_cuda_devices": enumerated_devices,
        "device_inventory_conflict": inventory_conflict,
        "offloaded_layers": offloaded,
        "offloadable_layers": total,
        "nvidia_smi_snapshot": gpu_snapshot,
        "runtime_log_evidence": evidence_lines,
    }


def _swap_activity_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_pages = before.get("swap_activity_pages", {})
    after_pages = after.get("swap_activity_pages", {})
    if not isinstance(before_pages, dict) or not isinstance(after_pages, dict):
        return {}
    return {
        key: after_pages[key] - before_pages[key]
        for key in ("pswpin", "pswpout")
        if isinstance(before_pages.get(key), int) and isinstance(after_pages.get(key), int)
    }


def validate_observability_runtime(
    config: Any, sdk_versions: dict[str, str], plan_observability: dict[str, Any]
) -> dict[str, Any]:
    if (
        not config.enabled
        or config.mode != plan_observability["required_worker_mode"]
        or not config.capture_content
        or config.artifact_max_payload_bytes > plan_observability["required_max_payload_bytes"]
        or config.offline_max_bytes > plan_observability["required_offline_bundle_max_bytes"]
    ):
        raise RuntimeError("Sweep worker observability/capture configuration is not compliant")
    if sdk_versions != plan_observability["sdk_versions"]:
        raise RuntimeError("Sweep worker OpenTelemetry package versions differ from frozen plan")
    return {
        "enabled": config.enabled,
        "mode": config.mode,
        "capture_content": config.capture_content,
        "artifact_max_payload_bytes": config.artifact_max_payload_bytes,
        "offline_max_bytes": config.offline_max_bytes,
        "sdk_versions": sdk_versions,
    }


def _gpu_snapshot() -> dict[str, Any]:
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    return {
        "available": gpu.returncode == 0,
        "devices": gpu.stdout.strip(),
        "compute_processes": processes.stdout.strip()
        if processes.returncode == 0
        else "unavailable",
    }


def _host_info() -> dict[str, Any]:
    memory = _proc_kib_fields(Path("/proc/meminfo"), {"MemTotal", "MemAvailable"})
    cpu_model = None
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    value: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "ram_bytes": memory.get("MemTotal"),
        "ram_available_bytes": memory.get("MemAvailable"),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        value["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        value["nvidia_smi"] = "unavailable"
    return value


def add_token_budget(path: Path, plan_hash: str, entries: list[dict[str, Any]]) -> int:
    """Idempotently charge planned nonpadding inference tokens by model/case."""
    ledger: dict[str, Any] = {"plan_sha256": plan_hash, "entries": {}}
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("plan_sha256") != plan_hash or not isinstance(ledger.get("entries"), dict):
            raise ValueError("campaign inference-token ledger identity mismatch")
    for row in entries:
        key = row["key"]
        previous = ledger["entries"].get(key)
        value = {"input_tokens": row["input_tokens"], "output_ceiling": row["output_ceiling"]}
        if previous is not None and previous != value:
            raise ValueError("resumed inference-token charge changed")
        ledger["entries"][key] = value
    total = sum(
        item["input_tokens"] + item["output_ceiling"] for item in ledger["entries"].values()
    )
    if total > CAMPAIGN_TOKEN_CEILING:
        raise RuntimeError("campaign inference-token ceiling reached before generation")
    write_json(path, ledger)
    return total


def _run_controls(
    args: argparse.Namespace, plan: dict[str, Any], model: Path, precision: str
) -> None:

    from evaluate_causal_line import LineProvider, score_line

    class _NativeLineProvider(LineProvider):
        def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int | None]:
            result = self.generate_detailed(prompt, max_new_tokens)
            return result.text, result.tokens
    from measure_r2_local import NativeProvider

    from tinycomplete.eval.code_benchmark import load_suite
    from tinycomplete.eval.code_generation import (
        build_causal_prompt,
        build_prediction_run_metadata,
        generate_predictions,
    )

    output = args.output / "quality" / precision
    output.mkdir(parents=True, exist_ok=True)
    strict_path = args.strict_suite
    line_rows = _read_jsonl(args.line_suite)
    with Server(args, model) as server:
        native = NativeProvider(server.url, f"sweep-{precision}")
        native.cache = False
        definitions: list[tuple[str, Path, list[Any], Any, int, str]] = []
        strict_cases = load_suite(strict_path)
        definitions.append(
            (
                "raw-causal-v1",
                strict_path,
                strict_cases,
                build_causal_prompt,
                CONTROL_OUTPUT_TOKENS,
                "strict",
            )
        )
        line_cases = [SimpleNamespace(**row) for row in line_rows]
        definitions.append(
            (
                "causal-line-v1",
                args.line_suite,
                line_cases,
                lambda case: case.prompt,
                CONTROL_OUTPUT_TOKENS,
                "line",
            )
        )
        for protocol, suite_path, cases, prompt_builder, cap, name in definitions:
            token_rows = []
            eligible = []
            for case in cases:
                prompt = prompt_builder(case)
                input_count = server.tokenize(prompt)
                row = {
                    "case_id": case.id,
                    "prompt_sha256": digest_bytes(prompt.encode()),
                    "input_tokens": input_count,
                    "output_ceiling": cap,
                    "context_eligible": input_count + cap <= CONTEXT_TOKENS,
                }
                token_rows.append(row)
                if row["context_eligible"]:
                    eligible.append(case)
            token_path = output / f"{name}-input-token-counts.jsonl"
            token_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in token_rows)
            )
            charges = [
                {
                    "key": f"{precision}/{name}/{row['case_id']}",
                    "input_tokens": row["input_tokens"],
                    "output_ceiling": cap,
                }
                for row in token_rows
                if row["context_eligible"]
            ]
            add_token_budget(
                args.output / "campaign-token-ledger.json", plan["plan_sha256"], charges
            )
            metadata = build_prediction_run_metadata(
                suite_path=suite_path,
                case_count=len(eligible),
                provider="llama.cpp-native",
                model_source=f"{MODEL_REPOSITORY}:{MODEL_REVISION}",
                model_revision=digest_file(model),
                max_new_tokens=cap,
                workers=1,
                protocol=protocol,
            )
            metadata.update(
                {
                    "campaign": "sweep-q8-q4-r1",
                    "plan_sha256": plan["plan_sha256"],
                    "precision": precision,
                    "runtime_revision": RUNTIME_REVISION,
                    "context_tokens": CONTEXT_TOKENS,
                    "gpu_layers": args.gpu_layers,
                    "prompt_cache": False,
                    "cache_ram_bytes": 0,
                    "parallel_slots": 1,
                    "excluded_over_context": [
                        row["case_id"] for row in token_rows if not row["context_eligible"]
                    ],
                    "prompt_policy": "raw causal source prompt; no chat template"
                    if name == "strict"
                    else "exact causal line fixture prompt",
                    "note": "Task-mismatch controls; not evidence of trained next-edit quality.",
                }
            )
            wrapper = _CountingNative(native, token_rows, server)
            generation_provider = _NativeLineProvider(wrapper) if name == "line" else wrapper
            predictions = generate_predictions(
                eligible,
                generation_provider,
                output / f"{name}-predictions.jsonl",
                run_metadata=metadata,
                max_new_tokens=cap,
                workers=1,
                prompt_builder=prompt_builder,
            )
            wrapper.flush(output / f"{name}-generation-details.jsonl")
            if name == "line":
                result_rows = []
                by_id = {row["id"]: row for row in line_rows}
                for prediction in predictions:
                    result = score_line(
                        by_id[prediction.case_id],
                        prediction.completion,
                        native_newline_omitted=prediction.finish_reason == "word",
                    )
                    result.update(
                        finish_reason=prediction.finish_reason,
                        hit_token_cap=prediction.hit_token_cap,
                        latency_seconds=prediction.latency_seconds,
                    )
                    result_rows.append(result)
                (output / "causal-line-syntax-results.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows)
                )
    server_meta = {
        "model": precision,
        "model_sha256": digest_file(model),
        "server_argv": server.argv,
        "server_pid": server.process.pid if server.process else None,
        "model_load_seconds": server.loaded_at - server.started_at,
        "server_binary_sha256": digest_file(args.runtime / "build/bin/llama-server"),
        "host": _host_info(),
        "host_before": server.host_before,
        "host_after": server.host_after,
        "swap_activity_delta_pages": _swap_activity_delta(server.host_before, server.host_after),
        "gpu_before": server.gpu_before,
        "gpu_after_load": server.gpu_after_load,
        "gpu_after_requests": server.gpu_after_requests,
        "runtime_backend_evidence": server.backend_evidence,
        "process_memory": server.memory_report(),
    }
    write_json(output / "server-measurement.json", server_meta)


class _CountingNative:
    def __init__(self, native: Any, token_rows: list[dict[str, Any]], server: Server):
        self.native = native
        self.server = server
        self.tokens = {row["prompt_sha256"]: row for row in token_rows}
        self.details: list[dict[str, Any]] = []
        self.stop_first_line = False

    def _generate(self, prompt: str, max_new_tokens: int):
        # The worker deadline already excludes its finalization reserve. Bound
        # this transport to the remaining work rather than reserving it twice.
        self.native.timeout_seconds = min(120.0, deadline_remaining())
        prompt_hash = digest_bytes(prompt.encode())
        token_row = self.tokens.get(prompt_hash)
        if token_row is None or not token_row["context_eligible"]:
            raise ValueError("generation attempted for an unbudgeted prompt")
        result = (
            self.native.generate_line_detailed(prompt, max_new_tokens)
            if self.stop_first_line
            else self.native.generate_detailed(prompt, max_new_tokens)
        )
        self.server.sample()
        self.details.append(
            {
                "case_id": token_row["case_id"],
                "prompt_sha256": prompt_hash,
                "input_tokens": token_row["input_tokens"],
                "server": dict(self.native.last),
            }
        )
        return result

    def generate_detailed(self, prompt: str, max_new_tokens: int):
        self.stop_first_line = False
        return self._generate(prompt, max_new_tokens)

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int | None]:
        result = self.generate_detailed(prompt, max_new_tokens)
        return result.text, result.tokens

    def generate_line_detailed(self, prompt: str, max_new_tokens: int):
        self.stop_first_line = True
        try:
            return self._generate(prompt, max_new_tokens)
        finally:
            self.stop_first_line = False

    def flush(self, path: Path) -> None:
        rows = {row["case_id"]: row for row in _read_jsonl(path)} if path.exists() else {}
        for row in self.details:
            rows[row["case_id"]] = row
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows.values()))


def map_full_file(current: str, updated: str, start: int, end: int) -> dict[str, Any]:
    current_bytes = current.encode("utf-8")
    updated_bytes = updated.encode("utf-8")
    if not 0 <= start <= end <= len(current_bytes):
        return {"mapping": "invalid_range", "out_of_range": True}
    try:
        current_bytes[:start].decode("utf-8")
        current_bytes[start:end].decode("utf-8")
        current_bytes[end:].decode("utf-8")
    except UnicodeDecodeError:
        return {"mapping": "invalid_utf8_boundary", "out_of_range": True}
    prefix = current_bytes[:start]
    suffix = current_bytes[end:]
    if len(updated_bytes) < len(prefix) + len(suffix):
        return {
            "mapping": "out_of_range",
            "out_of_range": True,
            "updated_bytes": len(updated_bytes),
        }
    if not updated_bytes.startswith(prefix) or (suffix and not updated_bytes.endswith(suffix)):
        return {
            "mapping": "out_of_range",
            "out_of_range": True,
            "updated_bytes": len(updated_bytes),
        }
    replacement_end = len(updated_bytes) - len(suffix) if suffix else len(updated_bytes)
    replacement = updated_bytes[start:replacement_end]
    try:
        text = replacement.decode("utf-8")
    except UnicodeDecodeError:
        return {"mapping": "invalid_utf8_boundary", "out_of_range": True}
    action = "no_edit" if updated_bytes == current_bytes else "replace"
    return {
        "mapping": "within_editable_range",
        "out_of_range": False,
        "action": action,
        "replacement": text,
    }


def _generate_sweep_case(
    server: Server,
    prompt: str,
    input_tokens: int,
    *,
    precision: str,
    context: Any,
    case_id: str,
) -> dict[str, Any]:
    import httpx

    from tinycomplete.eval.code_generation import DetailedGeneration
    from tinycomplete.observability.artifacts import ArtifactStore
    from tinycomplete.observability.bootstrap import current_runtime
    from tinycomplete.observability.hooks import model_metrics, usage_metrics
    from tinycomplete.observability.spans import operation

    deadline_remaining()
    if input_tokens + SWEEP_OUTPUT_TOKENS > CONTEXT_TOKENS:
        return {
            "input_tokens": input_tokens,
            "context_eligible": False,
            "reason": "prompt plus 512-token output ceiling exceeds 8192",
        }
    runtime = current_runtime()
    artifact_store = ArtifactStore(
        runtime.config.artifact_root,
        max_payload_bytes=runtime.config.artifact_max_payload_bytes,
        enabled=runtime.config.capture_content,
    )
    input_artifact = artifact_store.capture_text(
        "model-input",
        prompt,
        authorized=runtime.config.capture_content,
        source_ref=f"public-synthetic-fixture:{case_id}",
    )
    labels = {
        "backend": "llama.cpp",
        "model_alias": "sweep-next-edit-1.5b",
        "quantization": precision,
        "device_type": "gpu",
        "suite_version": "next_edit_v2",
        "protocol_version": "sweep-whole-file-v1",
    }
    attributes = {
        "gen_ai.operation.name": "text_completion",
        "gen_ai.provider.name": "llama.cpp",
        "gen_ai.request.model": f"{MODEL_REPOSITORY}:{MODEL_REVISION}",
        "gen_ai.request.max_tokens": SWEEP_OUTPUT_TOKENS,
        "tabcomplete.model.precision": precision,
        "tabcomplete.runtime.revision": RUNTIME_REVISION,
        "tabcomplete.protocol": "sweep-whole-file-v1",
        "gen_ai.usage.input_tokens": input_tokens,
        **input_artifact.attributes("input"),
    }
    request_budget = deadline_remaining()
    if request_budget <= 0:
        raise TimeoutError("finalization reserve reached before Sweep inference")
    started = time.perf_counter()
    with context.activate(), operation("eval.case", attributes={"tabcomplete.task": "next-edit"}):
        with operation("model.generate", attributes=attributes) as span:
            with model_metrics(labels):
                with httpx.stream(
                    "POST",
                    server.url + "/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": SWEEP_OUTPUT_TOKENS,
                        "temperature": 0,
                        "stream": True,
                        "return_tokens": True,
                        "cache_prompt": False,
                        "seed": 1,
                        "id_slot": 0,
                        "stop": ["<|file_sep|>", "</s>"],
                    },
                    timeout=min(300.0, request_budget),
                ) as response:
                    response.raise_for_status()
                    chunks: list[str] = []
                    tokens: list[int] = []
                    first_chunk_ms = None
                    first_token_ms = None
                    final: dict[str, Any] = {}
                    byte_count = 0
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        text_chunk = event.get("content", "")
                        if text_chunk:
                            if first_chunk_ms is None:
                                first_chunk_ms = elapsed_ms
                            chunks.append(text_chunk)
                            byte_count += len(text_chunk.encode("utf-8"))
                            if byte_count > 2 * 1024 * 1024:
                                raise ValueError("Sweep response exceeded the 2 MiB response limit")
                        new_tokens = event.get("tokens", [])
                        if new_tokens and first_token_ms is None:
                            first_token_ms = elapsed_ms
                        tokens.extend(new_tokens)
                        if event.get("stop"):
                            final = event
                            break
                total_ms = (time.perf_counter() - started) * 1000
            text = "".join(chunks)
            stop_type = final.get("stop_type")
            completed, hit_token_cap = terminal_status(
                stop_type, final.get("truncated", False), len(tokens), SWEEP_OUTPUT_TOKENS
            )
            generation = DetailedGeneration(
                text,
                len(tokens),
                stop_type if isinstance(stop_type, str) else None,
                input_tokens=input_tokens,
                usage_source="llama.cpp runtime tokenizer/SSE",
                first_output_ms=first_token_ms,
            )
            span.set_attribute("tabcomplete.timing.total_ms", total_ms)
            span.set_attribute("tabcomplete.timing.kind", "client_end_to_end")
            span.set_attribute("tabcomplete.output.truncated", hit_token_cap)
            span.set_attribute("gen_ai.usage.output_tokens", len(tokens))
            if first_chunk_ms is not None:
                span.set_attribute("tabcomplete.timing.first_chunk_ms", first_chunk_ms)
            if first_token_ms is not None:
                span.set_attribute("tabcomplete.timing.first_actual_token_ms", first_token_ms)
            if isinstance(stop_type, str):
                span.set_attribute("gen_ai.response.finish_reasons", [stop_type])
            usage_metrics(generation, labels)
            output_artifact = artifact_store.capture_text(
                "model-output",
                text,
                authorized=runtime.config.capture_content,
                source_ref=f"public-synthetic-fixture:{case_id}",
            )
            for key, value in output_artifact.attributes("output").items():
                span.set_attribute(key, value)
            span_context = span.get_span_context()
            trace_id = f"{span_context.trace_id:032x}" if span_context.is_valid else None
            observability_ids = {
                "campaign_id": context.campaign_id,
                "run_id": context.run_id,
                "run_attempt_id": context.run_attempt_id,
                "case_id": context.case_id,
                "case_attempt_id": context.case_attempt_id,
                "request_id": context.request_id,
                "trace_id": trace_id,
            }
    return {
        "prompt_sha256": digest_bytes(prompt.encode()),
        "input_tokens": input_tokens,
        "raw_output": text,
        "output_sha256": digest_bytes(text.encode()),
        "output_tokens": len(tokens),
        "finish_reason": stop_type,
        "stopping_word": final.get("stopping_word"),
        "explicit_terminal": bool(completed),
        "hit_token_cap": hit_token_cap,
        "first_chunk_ms": first_chunk_ms,
        "first_actual_token_ms": first_token_ms,
        "completed_response_ms": total_ms,
        "server_timings": final.get("timings", {}),
        "tokens_cached": final.get("tokens_cached"),
        "tokens_evaluated": final.get("tokens_evaluated"),
        "context_eligible": True,
        "observability_ids": observability_ids,
    }


def run_next_edit(
    args: argparse.Namespace,
    plan: dict[str, Any],
    model: Path,
    precision: str,
    *,
    model_sha256: str | None = None,
) -> None:
    from dataclasses import replace

    from tinycomplete.observability.context import RunContext
    from tinycomplete.observability.runs import run_scope

    model_sha256 = model_sha256 or digest_file(model)
    rows = _read_jsonl(args.next_edit_fixtures)
    expected = plan["comparison"]["next_edit"]["fixture_ids"]
    if [row.get("case_id") for row in rows] != expected:
        raise ValueError("next-edit fixture order or identity changed")
    output_dir = args.output / "next-edit" / precision
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    existing = _read_jsonl(predictions_path) if predictions_path.exists() else []
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for result in existing:
        case_id = result.get("case_id")
        repetition = result.get("repetition")
        if (
            not isinstance(case_id, str)
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition not in (0, 1)
            or result.get("plan_sha256") != plan["plan_sha256"]
            or result.get("model_sha256") != model_sha256
        ):
            raise ValueError("existing next-edit prediction output is not resumable")
        key = (case_id, repetition)
        if key in completed:
            raise ValueError("existing next-edit prediction output is not resumable")
        if case_id not in expected:
            raise ValueError("existing next-edit prediction output is not resumable")
        completed[key] = result
    with run_scope(
        output_dir / "observability-run.json", f"sweep-next-edit-{precision}"
    ) as scoped_run:
        run_context = scoped_run or RunContext.new(
            campaign_id=f"campaign-{plan['plan_sha256'][:32]}"
        )
        run_context = replace(
            run_context,
            workload={
                "tabcomplete.protocol": "sweep-whole-file-v1",
                "gen_ai.request.model": f"{MODEL_REPOSITORY}:{MODEL_REVISION}",
                "tabcomplete.model.precision": precision,
                "tabcomplete.runtime.revision": RUNTIME_REVISION,
            },
        )
        with Server(args, model) as server:
            prepared = []
            for row in rows:
                deadline_remaining()
                prompt = build_sweep_prompt(row)
                count = server.tokenize(prompt)
                prepared.append((row, prompt, count, count + SWEEP_OUTPUT_TOKENS <= CONTEXT_TOKENS))
            charges = [
                {
                    "key": f"{precision}/next-edit/{repetition}/{row['case_id']}",
                    "input_tokens": count,
                    "output_ceiling": SWEEP_OUTPUT_TOKENS,
                }
                for repetition in range(2)
                for row, _prompt, count, eligible in prepared
                if eligible
            ]
            add_token_budget(
                args.output / "campaign-token-ledger.json", plan["plan_sha256"], charges
            )
            with predictions_path.open("a", encoding="utf-8") as handle:
                for repetition in range(2):
                    for row, prompt, count, eligible in prepared:
                        key = (row["case_id"], repetition)
                        if key in completed:
                            continue
                        deadline_remaining()
                        if eligible:
                            case_context = run_context.for_case(row["case_id"])
                            result = _generate_sweep_case(
                                server,
                                prompt,
                                count,
                                precision=precision,
                                context=case_context,
                                case_id=row["case_id"],
                            )
                            server.sample()
                        else:
                            result = {
                                "input_tokens": count,
                                "context_eligible": False,
                                "reason": "prompt plus 512-token output ceiling exceeds 8192",
                            }
                        result.update(
                            case_id=row["case_id"],
                            file_path=row["file_path"],
                            current_sha256=row["current_sha256"],
                            original_sha256=row["original_sha256"],
                            repetition=repetition,
                            plan_sha256=plan["plan_sha256"],
                            model_sha256=model_sha256,
                        )
                        if result.get("explicit_terminal"):
                            result["editable_range_mapping"] = map_full_file(
                                row["current_content"],
                                result["raw_output"],
                                row["editable_start_byte"],
                                row["editable_end_byte"],
                            )
                        else:
                            result["editable_range_mapping"] = {
                                "mapping": "incomplete_or_unterminated_output_not_scored",
                                "out_of_range": None,
                            }
                        handle.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed[key] = result
        write_json(
            output_dir / "server-measurement.json",
            {
                "model": precision,
                "model_sha256": model_sha256,
                "server_argv": server.argv,
                "server_binary_sha256": digest_file(args.runtime / "build/bin/llama-server"),
                "model_load_seconds": server.loaded_at - server.started_at,
                "host": _host_info(),
                "host_before": server.host_before,
                "host_after": server.host_after,
                "swap_activity_delta_pages": _swap_activity_delta(
                    server.host_before, server.host_after
                ),
                "gpu_before": server.gpu_before,
                "gpu_after_load": server.gpu_after_load,
                "gpu_after_requests": server.gpu_after_requests,
                "runtime_backend_evidence": server.backend_evidence,
                "process_memory": server.memory_report(),
                "cases": len(completed),
                "repetitions": 2,
            },
        )


def run(args: argparse.Namespace) -> None:
    plan = load_and_verify_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    from tinycomplete.observability.bootstrap import current_runtime

    observed_config = current_runtime().config
    sdk_versions = {
        "opentelemetry-api": importlib.metadata.version("opentelemetry-api"),
        "opentelemetry-sdk": importlib.metadata.version("opentelemetry-sdk"),
    }
    write_json(
        args.output / "observability-runtime.json",
        validate_observability_runtime(observed_config, sdk_versions, plan["observability"]),
    )
    models = (
        ("q8_0", args.artifact_dir / MODEL_FILE),
        ("q4_k_m", args.artifact_dir / "sweep-next-edit-1.5b.q4_k_m.gguf"),
    )
    model_hashes: dict[str, str] = {}
    for precision, model in models:
        if not model.is_file():
            raise FileNotFoundError(f"required Sweep {precision} artifact is missing")
        expected = MODEL_SHA256 if precision == "q8_0" else Q4_SHA256
        model_hash = digest_file(model)
        if model_hash != expected:
            raise ValueError(f"Sweep {precision} artifact hash mismatch")
        model_hashes[precision] = model_hash
    if args.mode == "quality":
        for precision, model in models:
            _run_controls(args, plan, model, precision)
            print(
                json.dumps({"mode": args.mode, "precision": precision, "state": "complete"}),
                flush=True,
            )
    elif args.mode == "next-edit":
        for precision, model in models:
            run_next_edit(args, plan, model, precision, model_sha256=model_hashes[precision])
            print(
                json.dumps({"mode": args.mode, "precision": precision, "state": "complete"}),
                flush=True,
            )
    elif args.mode in {"download", "quantize"}:
        raise AssertionError("artifact modes are dispatched before the comparison plan check")
    else:
        raise ValueError("unknown Sweep campaign mode")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("freeze", "download", "quantize", "quality", "next-edit"), required=True
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--artifact-budget-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--line-suite", type=Path, required=True)
    parser.add_argument("--diagnostic-plan", type=Path)
    parser.add_argument("--supersedes-plan", type=Path)
    parser.add_argument("--next-edit-fixtures", type=Path, required=True)
    parser.add_argument(
        "--strict-suite", type=Path, default=Path("data/benchmarks/code_completion_v2.jsonl")
    )
    parser.add_argument(
        "--next-edit-suite", type=Path, default=Path("data/benchmarks/next_edit_v2.jsonl")
    )
    parser.add_argument(
        "--quota-record",
        type=Path,
        default=Path("reports/prototype/sweep_comparison_r1/quota_preparation.json"),
    )
    parser.add_argument("--test-file", type=Path, default=Path("tests/test_sweep_comparison.py"))
    parser.add_argument("--gpu-layers", type=int, default=99)
    args = parser.parse_args(argv)
    if args.mode == "freeze" and args.diagnostic_plan is None:
        parser.error("--mode freeze requires --diagnostic-plan")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "freeze":
        freeze_plan(args)
    elif args.mode == "download":
        load_and_verify_plan(args)
        path = download_q8(args)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest_file(path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif args.mode == "quantize":
        load_and_verify_plan(args)
        path = quantize_q4(args)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest_file(path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif args.mode in {"quality", "next-edit"}:
        try:
            run(args)
        finally:
            from tinycomplete.observability.bootstrap import current_runtime

            current_runtime().shutdown()
    else:
        raise AssertionError("mode dispatch is incomplete")


if __name__ == "__main__":
    main()
