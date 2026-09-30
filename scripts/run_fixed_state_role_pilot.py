#!/usr/bin/env python3
"""Freeze and run a small fixed-state author/solver/reviewer qualification phase.

This runner consumes only an answer-free public/synthetic state packet. It does
not create states, read oracle files, label human chronology, accept training
rows, or invoke a provider without a matching root-authored release artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinycomplete.observability.context import RunContext
from tinycomplete.observability.runs import run_scope
from tinycomplete.observability.spans import operation
from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION
from tinycomplete.one_line.contract import EditState, apply_action
from tinycomplete.one_line.fixed_state_roles import (
    HISTORY_ORDER,
    MAX_ACTION_TOKENS,
    MAX_CONTEXT_TOKENS,
    MAX_REVIEW_CONTEXT_TOKENS,
    MAX_REVIEW_RESPONSE_TOKENS,
    PROMPT_POLICY_VERSION,
    REVIEW_PROMPT_POLICY_VERSION,
    ROLE_PROTOCOL_SCHEMA,
    SOURCE_TYPE,
    CompletedFixedStateRole,
    build_fixed_state_prompt,
    build_fixed_state_review_prompt,
    build_fixed_state_role_evidence,
    fixed_state_protocol_bindings,
    parse_fixed_state_action,
    verify_fixed_state_role_evidence,
)
from tinycomplete.one_line.pilot_roles import parse_review_response
from tinycomplete.one_line.teacher import (
    AUTHORIZATION_BASIS,
    MAX_CALLS,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    OpenCodeTeacherClient,
    TeacherCompletedResponseError,
    TeacherPolicyError,
    TeacherResponse,
    TeacherResponseValidationError,
    TeacherTransportError,
    TeacherUsageLedger,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DEFAULT = ROOT / "reports/prototype/product_r2/teacher_usage.jsonl"
TOKENIZER_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    / "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
TOKENIZER_JSON_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"

PLAN_SCHEMA = "fixed-state-role-pilot-plan-v1"
RELEASE_SCHEMA = "fixed-state-role-phase-release-v1"
EVENT_SCHEMA = "fixed-state-role-event-v1"
SUMMARY_SCHEMA = "fixed-state-role-pilot-summary-v1"
MAX_CASES = 8
MAX_PHASE_CALLS = 24
MAX_PHASE_WALL_SECONDS = 30 * 60
FINALIZATION_RESERVE_SECONDS = 5 * 60
RESERVE_INPUT_TOKENS = 30_000
RESERVE_OUTPUT_TOKENS = 8_192
CALL_ROLES = ("author", "solver", "reviewer")
ACTOR_SUFFIX = {
    "author": "fixed-state-author-v1",
    "solver": "fixed-state-blind-solver-v1",
    "reviewer": "fixed-state-independent-reviewer-v1",
}
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:/@+-]{1,160}\Z")
_TOKENIZER: Any | None = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _json_unique(payload: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=pairs)


def _private_directory(path: Path, *, create: bool = False) -> None:
    if path.is_symlink():
        raise ValueError("private directory cannot be a symlink")
    if not path.exists():
        if not create:
            raise ValueError("private directory is missing")
        path.mkdir(mode=0o700, parents=False)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("private directory must be owner-only mode 0700")


def _private_file(path: Path, *, required_mode: int = 0o600) -> bytes:
    if path.is_symlink():
        raise ValueError("private file cannot be a symlink")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != required_mode
    ):
        raise ValueError("private file must be an owner-only regular file")
    return path.read_bytes()


def _write_once(path: Path, payload: bytes) -> None:
    _private_directory(path.parent)
    if path.is_symlink():
        raise ValueError("refusing symlink artifact")
    if path.exists():
        if _private_file(path) != payload:
            raise ValueError("existing immutable artifact differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=".fixed-state-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_derived(path: Path, payload: bytes) -> None:
    """Atomically refresh a rebuildable summary; frozen inputs/events stay immutable."""
    _private_directory(path.parent)
    if path.is_symlink():
        raise ValueError("derived output cannot be a symlink")
    if path.exists():
        _private_file(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".summary-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    _private_directory(path.parent)
    if path.is_symlink():
        raise ValueError("append-only event file cannot be a symlink")
    if path.exists():
        _private_file(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(_canonical(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot_ledger(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, int], str]:
    if path.is_symlink():
        raise ValueError("usage ledger cannot be a symlink")
    if not path.exists():
        return {}, {"calls": 0, "input_tokens": 0, "output_tokens": 0}, _sha(b"")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("usage ledger owner or file type is invalid")
    with path.open("r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        raw = handle.read()
    text = raw.decode("utf-8")
    requests = TeacherUsageLedger._read(io.StringIO(text))
    totals = TeacherUsageLedger._totals(requests)
    return (
        requests,
        {
            "calls": totals.calls,
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
        },
        _sha(raw),
    )


def _tokenizer() -> Any:
    global _TOKENIZER
    if _TOKENIZER is None:
        tokenizer_path = TOKENIZER_SNAPSHOT / "tokenizer.json"
        if not tokenizer_path.is_file() or _sha_file(tokenizer_path) != TOKENIZER_JSON_SHA256:
            raise ValueError("pinned local Q25 tokenizer is missing or changed")
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            TOKENIZER_SNAPSHOT, local_files_only=True, trust_remote_code=False
        )
    return _TOKENIZER


def _opencode_identity() -> dict[str, str]:
    binary = shutil.which("opencode")
    if binary is None:
        raise ValueError("installed OpenCode executable is unavailable")
    path = Path(binary).resolve(strict=True)
    result = subprocess.run(
        [str(path), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("OpenCode version probe failed")
    return {
        "binary_sha256": _sha_file(path),
        "version": result.stdout.strip().splitlines()[0][:120],
    }


_SOURCE_FILES = {
    "runner": ROOT / "scripts/run_fixed_state_role_pilot.py",
    "fixed_state_roles": ROOT / "src/tinycomplete/one_line/fixed_state_roles.py",
    "teacher": ROOT / "src/tinycomplete/one_line/teacher.py",
    "context": ROOT / "src/tinycomplete/one_line/context.py",
    "contract": ROOT / "src/tinycomplete/one_line/contract.py",
    "review_parser": ROOT / "src/tinycomplete/one_line/pilot_roles.py",
    "observability_context": ROOT / "src/tinycomplete/observability/context.py",
    "observability_runs": ROOT / "src/tinycomplete/observability/runs.py",
    "observability_spans": ROOT / "src/tinycomplete/observability/spans.py",
}


def _source_hashes() -> dict[str, str]:
    return {key: _sha_file(path) for key, path in _SOURCE_FILES.items()}


def _runtime_identity() -> dict[str, Any]:
    tokenizer_path = TOKENIZER_SNAPSHOT / "tokenizer.json"
    if not tokenizer_path.is_file() or _sha_file(tokenizer_path) != TOKENIZER_JSON_SHA256:
        raise ValueError("pinned local Q25 tokenizer is missing or changed")
    packages: dict[str, str] = {}
    for name in ("httpx", "transformers", "tokenizers", "opentelemetry-api", "opentelemetry-sdk"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "opencode": _opencode_identity(),
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "tokenizer_snapshot_revision": TOKENIZER_SNAPSHOT.name,
        "tokenizer_loading": "local_files_only=true; trust_remote_code=false",
    }


def _validate_state_mapping(value: Any) -> EditState:
    if not isinstance(value, dict) or set(value) != {
        "file_id",
        "filetype",
        "source",
        "target_row",
        "cursor_col",
        "history",
        "relevant",
    }:
        raise ValueError("fixed-state packet state fields are invalid")
    if any(not isinstance(value.get(name), str) for name in ("file_id", "filetype", "source")):
        raise ValueError("fixed-state packet text fields are invalid")
    for name in ("target_row", "cursor_col"):
        if type(value.get(name)) is not int:
            raise ValueError("fixed-state packet coordinate is invalid")
    history = value.get("history")
    relevant = value.get("relevant")
    if not isinstance(history, list) or not isinstance(relevant, list):
        raise ValueError("fixed-state packet history or relevant data is invalid")
    for item in history:
        if not isinstance(item, dict) or set(item) != {"row", "old_text", "new_text"}:
            raise ValueError("fixed-state packet history item is invalid")
        if (
            type(item["row"]) is not int
            or not isinstance(item["old_text"], str)
            or not isinstance(item["new_text"], str)
        ):
            raise ValueError("fixed-state packet history values are invalid")
    if any(not isinstance(item, str) for item in relevant):
        raise ValueError("fixed-state packet relevant context is invalid")
    return EditState.from_mapping(value)


def _read_packet(packet_path: Path, tokenizer: Any) -> tuple[bytes, list[dict[str, Any]]]:
    packet_bytes = _private_file(packet_path)
    rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    seen_seeds: set[str] = set()
    for line in packet_bytes.decode("utf-8").splitlines():
        if not line.strip():
            raise ValueError("fixed-state packet contains a blank line")
        row = _json_unique(line)
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "seed_id",
            "state",
            "prompt",
            "context_sha256",
        }:
            raise ValueError("fixed-state packet must contain only the answer-free row schema")
        for field in ("candidate_id", "seed_id"):
            identity = row.get(field)
            if not isinstance(identity, str) or not _ID.fullmatch(identity):
                raise ValueError("fixed-state packet identifier is invalid")
        if row["candidate_id"] in seen_candidates or row["seed_id"] in seen_seeds:
            raise ValueError("fixed-state packet repeats a candidate or source seed")
        seen_candidates.add(row["candidate_id"])
        seen_seeds.add(row["seed_id"])
        if (
            not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("context_sha256"), str)
            or not _SHA256.fullmatch(row["context_sha256"])
        ):
            raise ValueError("fixed-state packet prompt identity is invalid")
        state = _validate_state_mapping(row["state"])
        prompt = build_fixed_state_prompt(state, tokenizer)
        if row["prompt"] != prompt.user_prompt or row["context_sha256"] != prompt.context_sha256:
            raise ValueError("fixed-state packet context differs from canonical renderer")
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "seed_id": row["seed_id"],
                "state": asdict(state),
                "prompt": prompt.user_prompt,
                "context_sha256": prompt.context_sha256,
            }
        )
    if not 1 <= len(rows) <= MAX_CASES:
        raise ValueError("fixed-state role phase requires one to eight clean packets")
    return packet_bytes, rows


def _request_id(phase_id: str, packet_sha256: str, candidate_id: str, role: str) -> str:
    identity = _sha(_canonical([phase_id, packet_sha256, candidate_id, role]))[:24]
    return f"fixed-state-role-v1-{role}-{identity}"


def _plan_hash(plan_without_hash: Mapping[str, Any]) -> str:
    return _sha(_canonical(dict(plan_without_hash)))


def freeze_plan(
    *,
    packet_path: Path,
    run_dir: Path,
    ledger_path: Path = LEDGER_DEFAULT,
    tokenizer: Any | None = None,
    runtime: dict[str, Any] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze exact answer-free state/prompt identities; makes no provider call."""
    if run_dir.exists() or run_dir.is_symlink():
        raise ValueError("fixed-state plan output directory must be new")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    _private_directory(run_dir.parent)
    _private_directory(run_dir, create=True)
    tokenizer = tokenizer or _tokenizer()
    packet_bytes, rows = _read_packet(packet_path, tokenizer)
    baseline, totals, ledger_sha = _snapshot_ledger(ledger_path)
    planned_calls = len(rows) * len(CALL_ROLES)
    if planned_calls > MAX_PHASE_CALLS:
        raise ValueError("fixed-state plan exceeds the 24-request phase cap")
    if (
        totals["calls"] + planned_calls > MAX_CALLS
        or totals["input_tokens"] + planned_calls * RESERVE_INPUT_TOKENS > MAX_INPUT_TOKENS
        or totals["output_tokens"] + planned_calls * RESERVE_OUTPUT_TOKENS > MAX_OUTPUT_TOKENS
    ):
        raise ValueError("fixed-state phase reservations exceed the locked global ledger caps")

    packet_sha = _sha(packet_bytes)
    phase_id = f"fixed-state-r1-{uuid.uuid4().hex}"
    cases = []
    request_ids: list[str] = []
    for row in rows:
        prompt = build_fixed_state_prompt(row["state"], tokenizer)
        protocol_row = {
            **row,
            "id": row["candidate_id"],
            "source_type": SOURCE_TYPE,
            "history_order": HISTORY_ORDER,
            "human_chronology_observed": False,
            "context_policy": CONTEXT_POLICY_VERSION,
            "history_sha256": prompt.history_sha256,
        }
        protocol = fixed_state_protocol_bindings(protocol_row, tokenizer)
        role_ids = {
            role: _request_id(phase_id, packet_sha, row["candidate_id"], role)
            for role in CALL_ROLES
        }
        request_ids.extend(role_ids[role] for role in CALL_ROLES)
        cases.append(
            {
                "candidate_id": row["candidate_id"],
                "seed_id": row["seed_id"],
                "state_sha256": protocol["state_sha256"],
                "history_sha256": protocol["history_sha256"],
                "source_sha256": protocol["source_sha256"],
                "context_sha256": protocol["context_sha256"],
                "student_prompt_sha256": protocol["student_prompt_sha256"],
                "student_input_tokens": protocol["student_input_tokens"],
                "request_ids": role_ids,
            }
        )
    if len(set(request_ids)) != len(request_ids) or set(request_ids) & set(baseline):
        raise ValueError("fixed-state request IDs are duplicated or already used")

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "phase_id": phase_id,
        "status": "frozen_waiting_for_root_release_no_provider_calls",
        "created_at_unix_ns": time.time_ns(),
        "model_id": MODEL_ID,
        "protocol_schema": ROLE_PROTOCOL_SCHEMA,
        "wire_version": "single-line-edit-v1",
        "source_type": SOURCE_TYPE,
        "history_order": HISTORY_ORDER,
        "human_chronology_observed": False,
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "review_prompt_policy_version": REVIEW_PROMPT_POLICY_VERSION,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_action_supervised_tokens_including_eos": MAX_ACTION_TOKENS,
        "max_reviewer_context_tokens": MAX_REVIEW_CONTEXT_TOKENS,
        "max_reviewer_supervised_tokens_including_eos": MAX_REVIEW_RESPONSE_TOKENS,
        "packet": {
            "input_filename": "inputs.jsonl",
            "sha256": packet_sha,
            "bytes": len(packet_bytes),
            "case_count": len(rows),
            "candidate_ids": [row["candidate_id"] for row in rows],
        },
        "cases": cases,
        "provider": {
            "max_phase_calls": MAX_PHASE_CALLS,
            "planned_calls": planned_calls,
            "role_order_per_case": list(CALL_ROLES),
            "max_wall_seconds": MAX_PHASE_WALL_SECONDS,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "reserve_input_tokens_per_request": RESERVE_INPUT_TOKENS,
            "reserve_output_tokens_per_request": RESERVE_OUTPUT_TOKENS,
            "purpose": {
                "author": "student_label",
                "solver": "student_label",
                "reviewer": "automated_score",
            },
            "source_class": "public",
            "sequential_single_active_request": True,
            "on_transport_or_unknown_usage": "stop_no_retry",
            "on_non_stop_or_invalid_output": "retain_terminal_record_no_parse_repair",
            "automatic_fallback": False,
        },
        "ledger": {
            "path": str(ledger_path.resolve(strict=False)),
            "baseline_sha256": ledger_sha,
            "baseline_request_ids": sorted(baseline),
            "baseline_rows": baseline,
            "baseline_request_ids_sha256": _sha(_canonical(sorted(baseline))),
            "baseline_totals": totals,
            "global_caps": {
                "calls": MAX_CALLS,
                "input_tokens": MAX_INPUT_TOKENS,
                "output_tokens": MAX_OUTPUT_TOKENS,
            },
        },
        "runtime": runtime if runtime is not None else _runtime_identity(),
        "source_hashes": source_hashes if source_hashes is not None else _source_hashes(),
        "qualification_boundary": {
            "provider_roles_are_inferability_evidence_only": True,
            "functional_oracle_status": "not_evaluated_by_role_runner",
            "training_started": False,
            "training_accepted": False,
            "personalization_enabled": False,
            "human_chronology_observed": False,
        },
    }
    plan["plan_sha256"] = _plan_hash(plan)
    _write_once(run_dir / "inputs.jsonl", packet_bytes)
    _write_once(run_dir / "plan.json", _canonical(plan) + b"\n")
    for directory in ("prompts", "responses", "response_receipts", "role_evidence"):
        _private_directory(run_dir / directory, create=True)
    return plan


def _load_frozen_plan(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _private_directory(run_dir)
    plan_bytes = _private_file(run_dir / "plan.json")
    plan = _json_unique(plan_bytes.decode("utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("frozen role plan is not an object")
    claimed = plan.pop("plan_sha256", None)
    if claimed != _plan_hash(plan):
        raise ValueError("frozen role plan self-hash mismatch")
    plan["plan_sha256"] = claimed
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "frozen_waiting_for_root_release_no_provider_calls"
    ):
        raise ValueError("fixed-state plan schema or status is invalid")
    input_bytes = _private_file(run_dir / "inputs.jsonl")
    if _sha(input_bytes) != plan.get("packet", {}).get("sha256"):
        raise ValueError("fixed-state input packet changed after freeze")
    rows = [_json_unique(line) for line in input_bytes.decode("utf-8").splitlines()]
    if len(rows) != plan.get("packet", {}).get("case_count"):
        raise ValueError("fixed-state input packet count changed")
    if plan.get("source_hashes") != _source_hashes():
        raise ValueError("fixed-state runner or protocol code changed after plan freeze")
    return plan, rows


def _validate_case_plan(
    plan: Mapping[str, Any], run_dir: Path, tokenizer: Any
) -> list[dict[str, Any]]:
    packet_bytes, rows = _read_packet(run_dir / "inputs.jsonl", tokenizer)
    if _sha(packet_bytes) != plan["packet"]["sha256"]:
        raise ValueError("fixed-state input packet differs from its frozen identity")
    if len(rows) != len(plan["cases"]):
        raise ValueError("fixed-state plan case count changed")
    for row, case in zip(rows, plan["cases"], strict=True):
        protocol_row = {
            **row,
            "id": row["candidate_id"],
            "source_type": SOURCE_TYPE,
            "history_order": HISTORY_ORDER,
            "human_chronology_observed": False,
            "context_policy": CONTEXT_POLICY_VERSION,
            "history_sha256": build_fixed_state_prompt(row["state"], tokenizer).history_sha256,
        }
        bindings = fixed_state_protocol_bindings(protocol_row, tokenizer)
        if (
            case.get("candidate_id") != row["candidate_id"]
            or case.get("seed_id") != row["seed_id"]
            or any(
                case.get(key) != bindings[binding]
                for key, binding in (
                    ("state_sha256", "state_sha256"),
                    ("history_sha256", "history_sha256"),
                    ("source_sha256", "source_sha256"),
                    ("context_sha256", "context_sha256"),
                    ("student_prompt_sha256", "student_prompt_sha256"),
                    ("student_input_tokens", "student_input_tokens"),
                )
            )
        ):
            raise ValueError("fixed-state case identity differs from frozen prompt and state")
        expected_ids = {
            role: _request_id(
                str(plan["phase_id"]),
                str(plan["packet"]["sha256"]),
                row["candidate_id"],
                role,
            )
            for role in CALL_ROLES
        }
        if case.get("request_ids") != expected_ids:
            raise ValueError("fixed-state request IDs differ from deterministic plan derivation")
    return rows


def _read_release(path: Path, expected_sha256: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("root release SHA-256 is required")
    payload = _private_file(path)
    if _sha(payload) != expected_sha256:
        raise ValueError("root release artifact hash does not match reviewed digest")
    release = _json_unique(payload.decode("utf-8"))
    required = {
        "schema",
        "plan_sha256",
        "baseline_ledger_sha256",
        "baseline_request_ids_sha256",
        "planned_request_ids",
        "max_calls",
        "max_wall_seconds",
        "reserve_input_tokens",
        "reserve_output_tokens",
        "global_caps",
        "released_by",
        "released_at_utc",
    }
    if not isinstance(release, dict) or set(release) != required:
        raise ValueError("root release fields do not match the fixed-state release contract")
    plan_ids = [case["request_ids"][role] for case in plan["cases"] for role in CALL_ROLES]
    if (
        release.get("schema") != RELEASE_SCHEMA
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("baseline_ledger_sha256") != plan["ledger"]["baseline_sha256"]
        or release.get("baseline_request_ids_sha256")
        != plan["ledger"]["baseline_request_ids_sha256"]
        or release.get("planned_request_ids") != plan_ids
        or release.get("max_calls") != len(plan_ids)
        or type(release.get("max_wall_seconds")) is not int
        or not FINALIZATION_RESERVE_SECONDS + 120
        <= release["max_wall_seconds"]
        <= MAX_PHASE_WALL_SECONDS
        or release.get("reserve_input_tokens") != RESERVE_INPUT_TOKENS
        or release.get("reserve_output_tokens") != RESERVE_OUTPUT_TOKENS
        or release.get("global_caps")
        != {
            "calls": MAX_CALLS,
            "input_tokens": MAX_INPUT_TOKENS,
            "output_tokens": MAX_OUTPUT_TOKENS,
        }
        or not isinstance(release.get("released_by"), str)
        or not release["released_by"].strip()
        or not isinstance(release.get("released_at_utc"), str)
    ):
        raise ValueError("root release does not authorize this exact plan and ledger allocation")
    return release


def _read_events(
    run_dir: Path, plan_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return [], {}
    raw = _private_file(path)
    events: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    previous = "0" * 64
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        event = _json_unique(line)
        if not isinstance(event, dict):
            raise ValueError("fixed-state event is not an object")
        claimed = event.pop("event_sha256", None)
        if (
            event.get("schema") != EVENT_SCHEMA
            or event.get("plan_sha256") != plan_sha256
            or event.get("sequence") != index
            or event.get("previous_event_sha256") != previous
            or claimed != _sha(_canonical(event))
        ):
            raise ValueError("fixed-state append-only event chain is invalid")
        event["event_sha256"] = claimed
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or request_id in by_id:
            raise ValueError("fixed-state event request identity is duplicated")
        previous = claimed
        events.append(event)
        by_id[request_id] = event
    return events, by_id


def _append_event(
    run_dir: Path, event: dict[str, Any], previous_events: list[dict[str, Any]]
) -> dict[str, Any]:
    prepared = {
        **event,
        "schema": EVENT_SCHEMA,
        "sequence": len(previous_events),
        "previous_event_sha256": previous_events[-1]["event_sha256"]
        if previous_events
        else "0" * 64,
    }
    prepared["event_sha256"] = _sha(_canonical(prepared))
    _append_jsonl(run_dir / "events.jsonl", prepared)
    previous_events.append(prepared)
    return prepared


def _digest_id(candidate_id: str) -> str:
    return _sha(candidate_id.encode("utf-8"))[:20]


def _prompt_for_role(
    role: str, row: Mapping[str, Any], previous: Mapping[str, Any], tokenizer: Any
) -> tuple[str, str, str, str]:
    state = EditState.from_mapping(row["state"])
    if role in {"author", "solver"}:
        prompt = build_fixed_state_prompt(state, tokenizer)
        return prompt.user_prompt, prompt.system_instruction, "student_label", prompt.prompt_sha256
    author_wire = previous.get("author_wire")
    solver_wire = previous.get("solver_wire")
    if not isinstance(author_wire, str) or not isinstance(solver_wire, str):
        raise ValueError("reviewer requires two complete action wires")
    review = build_fixed_state_review_prompt(state, author_wire, solver_wire, tokenizer)
    return review.user_prompt, review.system_instruction, "automated_score", review.prompt_sha256


def _prompt_path(run_dir: Path, request_id: str) -> Path:
    return run_dir / "prompts" / f"{_sha(request_id.encode())[:32]}.json"


def _response_paths(run_dir: Path, request_id: str) -> tuple[Path, Path]:
    stem = _sha(request_id.encode())[:32]
    return run_dir / "responses" / f"{stem}.txt", run_dir / "response_receipts" / f"{stem}.json"


def _persist_response(run_dir: Path, request_id: str, response: TeacherResponse) -> str:
    response_bytes = response.content.encode("utf-8")
    response_path, receipt_path = _response_paths(run_dir, request_id)
    _write_once(response_path, response_bytes)
    receipt = {
        "schema": "fixed-state-completed-provider-response-v1",
        "request_id": request_id,
        "response_sha256": _sha(response_bytes),
        "model_id": response.model_id,
        "session_id": response.session_id,
        "response_id": response.response_id,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "cached_read_tokens": response.cached_read_tokens,
        "cached_write_tokens": response.cached_write_tokens,
        "total_tokens_reported": response.total_tokens_reported,
        "finish_reason": response.finish_reason,
    }
    receipt_bytes = _canonical(receipt) + b"\n"
    _write_once(receipt_path, receipt_bytes)
    return _sha(receipt_bytes)


def _load_response(run_dir: Path, request_id: str) -> tuple[TeacherResponse, str]:
    response_path, receipt_path = _response_paths(run_dir, request_id)
    content_bytes = _private_file(response_path)
    receipt_bytes = _private_file(receipt_path)
    receipt = _json_unique(receipt_bytes.decode("utf-8"))
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "fixed-state-completed-provider-response-v1"
        or receipt.get("request_id") != request_id
        or receipt.get("response_sha256") != _sha(content_bytes)
        or receipt.get("model_id") != MODEL_ID
    ):
        raise ValueError("persisted provider response receipt is invalid")
    content = content_bytes.decode("utf-8")
    response = TeacherResponse(
        content=content,
        session_id=str(receipt["session_id"]),
        response_id=str(receipt["response_id"]),
        model_id=str(receipt["model_id"]),
        input_tokens=int(receipt["input_tokens"]),
        output_tokens=int(receipt["output_tokens"]),
        reasoning_tokens=int(receipt["reasoning_tokens"]),
        total_tokens_reported=receipt.get("total_tokens_reported"),
        cached_read_tokens=int(receipt["cached_read_tokens"]),
        cached_write_tokens=int(receipt["cached_write_tokens"]),
        finish_reason=receipt.get("finish_reason"),
        cost_usd_reported=None,
    )
    return response, _sha(receipt_bytes)


def _classify_response(
    role: str, response: TeacherResponse, tokenizer: Any
) -> tuple[str, str | None]:
    if response.finish_reason != "stop":
        return "completed_non_stop_invalid", "provider_finish_reason_not_stop"
    try:
        if role in {"author", "solver"}:
            parse_fixed_state_action(response, tokenizer)
        else:
            if len(response.content.encode("utf-8")) > 2048:
                raise ValueError("review response exceeds byte cap")
            if (
                len(tokenizer.encode(response.content, add_special_tokens=False)) + 1
                > MAX_REVIEW_RESPONSE_TOKENS
            ):
                raise ValueError("review response exceeds token cap")
            parse_review_response(response.content)
    except (UnicodeError, ValueError):
        return "completed_invalid_output", "strict_role_parser_rejected_response"
    return ("completed_valid_action" if role != "reviewer" else "completed_valid_review"), None


def _event_for_completed(
    *,
    plan: Mapping[str, Any],
    candidate_id: str,
    role: str,
    request_id: str,
    prompt_sha256: str,
    response: TeacherResponse,
    response_receipt_sha256: str,
    tokenizer: Any,
    recovered: bool = False,
) -> dict[str, Any]:
    status, error = _classify_response(role, response, tokenizer)
    return {
        "plan_sha256": plan["plan_sha256"],
        "candidate_id": candidate_id,
        "candidate_key": _digest_id(candidate_id),
        "role": role,
        "request_id": request_id,
        "actor_id": MODEL_ID + ":" + ACTOR_SUFFIX[role],
        "model_id": response.model_id,
        "session_id": response.session_id,
        "response_id": response.response_id,
        "prompt_sha256": prompt_sha256,
        "response_sha256": _sha(response.content.encode("utf-8")),
        "response_receipt_sha256": response_receipt_sha256,
        "finish_reason": response.finish_reason,
        "provider_input_tokens": response.input_tokens,
        "provider_output_tokens": response.output_tokens,
        "provider_reasoning_tokens": response.reasoning_tokens,
        "status": status,
        "failure_class": error,
        "recovered_from_persisted_receipt": recovered,
        "recorded_at_unix_ns": time.time_ns(),
        "retry_permitted": False,
    }


def _read_attempt_state(run_dir: Path, plan: Mapping[str, Any], max_wall: int) -> dict[str, Any]:
    path = run_dir / "phase_clock.json"
    if path.exists():
        payload = _private_file(path)
        value = _json_unique(payload.decode("utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("plan_sha256") != plan["plan_sha256"]
            or type(value.get("first_started_unix_ns")) is not int
            or value.get("max_wall_seconds") != max_wall
        ):
            raise ValueError("persisted fixed-state phase clock differs from release")
        return value
    value = {
        "schema": "fixed-state-phase-clock-v1",
        "plan_sha256": plan["plan_sha256"],
        "first_started_unix_ns": time.time_ns(),
        "max_wall_seconds": max_wall,
    }
    _write_once(path, _canonical(value) + b"\n")
    return value


def _read_event_log(
    run_dir: Path, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    events, by_id = _read_events(run_dir, str(plan["plan_sha256"]))
    planned = {
        case["request_ids"][role]: (case["candidate_id"], role)
        for case in plan["cases"]
        for role in CALL_ROLES
    }
    for event in events:
        request_id = event["request_id"]
        if request_id not in planned:
            raise ValueError("fixed-state event contains an out-of-plan request")
        candidate_id, role = planned[request_id]
        if event.get("candidate_id") != candidate_id or event.get("role") != role:
            raise ValueError("fixed-state event role or candidate binding changed")
    return events, by_id


def _validate_ledger_progress(
    plan: Mapping[str, Any],
    run_dir: Path,
    events_by_id: Mapping[str, Any],
    ledger_path_override: Path | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, int], set[str]]:
    ledger_path = ledger_path_override or Path(plan["ledger"]["path"])
    current, totals, current_sha = _snapshot_ledger(ledger_path)
    baseline_rows = set(plan["ledger"]["baseline_request_ids"])
    expected_ids = {case["request_ids"][role] for case in plan["cases"] for role in CALL_ROLES}
    current_ids = set(current)
    if not baseline_rows <= current_ids:
        raise ValueError("teacher usage ledger lost a frozen baseline request")
    if not current_ids - baseline_rows <= expected_ids:
        raise ValueError("teacher usage ledger contains a request outside the released phase")
    baseline_set = set(plan["ledger"]["baseline_request_ids"])
    # The frozen rows are retained, allowing append-only additions but no rewriting.
    planned_baseline_map = plan["ledger"].get("baseline_rows")
    if not isinstance(planned_baseline_map, dict) or set(planned_baseline_map) != baseline_set:
        raise ValueError("frozen ledger baseline row map is invalid")
    for request_id, row in planned_baseline_map.items():
        if current.get(request_id) != row:
            raise ValueError("teacher usage ledger changed a frozen baseline reservation")

    new_ids = current_ids - baseline_rows
    if len(new_ids) > len(expected_ids):
        raise ValueError("fixed-state phase exceeded its request ceiling")
    # Each durable event must be charged. A ledger row without an event can be
    # recovered only from a matching persisted completed-response receipt.
    for request_id, event in events_by_id.items():
        if request_id not in current:
            raise ValueError("fixed-state event has no matching ledger reservation")
        row = current[request_id]
        if event.get("status") in {
            "ambiguous_transport_failure",
            "response_unusable_unknown_usage",
        }:
            if "input" in row:
                raise ValueError("unknown-usage event has an unexpected ledger settlement")
        elif event.get("status") in {
            "completed_valid_action",
            "completed_valid_review",
            "completed_invalid_output",
            "completed_non_stop_invalid",
        }:
            if "input" not in row:
                raise ValueError("completed event has an unsettled ledger reservation")
            if (
                event.get("provider_input_tokens") != row["input"]
                or event.get("provider_output_tokens", 0)
                + event.get("provider_reasoning_tokens", 0)
                != row["output"]
            ):
                raise ValueError("completed event usage differs from locked ledger settlement")
        elif event.get("status") == "completed_terminal_failure":
            # Persistence/accounting/overrun failures can leave a full unsettled
            # reservation. The terminal event is sufficient to forbid retry.
            usage = event.get("provider_input_tokens")
            output = event.get("provider_output_tokens")
            reasoning = event.get("provider_reasoning_tokens")
            if usage is not None:
                if "input" in row and (
                    usage != row["input"] or output + reasoning != row["output"]
                ):
                    raise ValueError("terminal response usage differs from ledger settlement")
                if "input" not in row and event.get("ledger_reconciled") is not False:
                    raise ValueError("terminal response usage lacks ledger reconciliation")
    for request_id in new_ids - set(events_by_id):
        row = current[request_id]
        response_path, receipt_path = _response_paths(run_dir, request_id)
        if "input" not in row or not response_path.is_file() or not receipt_path.is_file():
            raise ValueError("unmatched or ambiguous provider reservation; refusing retry")
        response, _receipt_sha = _load_response(run_dir, request_id)
        if (
            response.input_tokens != row["input"]
            or response.output_tokens + response.reasoning_tokens != row["output"]
        ):
            raise ValueError("persisted response usage differs from locked ledger settlement")

    if (
        totals["calls"] > MAX_CALLS
        or totals["input_tokens"] > MAX_INPUT_TOKENS
        or totals["output_tokens"] > MAX_OUTPUT_TOKENS
    ):
        raise ValueError("global teacher usage cap exceeded")
    remaining = expected_ids - current_ids
    if (
        totals["calls"] + len(remaining) > MAX_CALLS
        or totals["input_tokens"] + len(remaining) * RESERVE_INPUT_TOKENS > MAX_INPUT_TOKENS
        or totals["output_tokens"] + len(remaining) * RESERVE_OUTPUT_TOKENS > MAX_OUTPUT_TOKENS
    ):
        raise ValueError("remaining fixed-state phase reservations exceed global caps")
    if not new_ids and current_sha != plan["ledger"]["baseline_sha256"]:
        raise ValueError("usage ledger differs from frozen baseline before first phase request")
    return current, totals, new_ids


def _recovered_event(
    *,
    run_dir: Path,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    role: str,
    request_id: str,
    prompt: str,
    system: str,
    tokenizer: Any,
) -> dict[str, Any]:
    prompt_record = {"system_instruction": system, "user_prompt": prompt}
    prompt_bytes = _canonical(prompt_record)
    stored_prompt = _private_file(_prompt_path(run_dir, request_id))
    if stored_prompt != prompt_bytes:
        raise ValueError("persisted prompt differs from deterministic fixed-state request")
    prompt_hash = _sha(_canonical({"system_instruction": system, "user_prompt": prompt}))
    response, receipt_sha = _load_response(run_dir, request_id)
    event = _event_for_completed(
        plan=plan,
        candidate_id=str(row["candidate_id"]),
        role=role,
        request_id=request_id,
        prompt_sha256=prompt_hash,
        response=response,
        response_receipt_sha256=receipt_sha,
        tokenizer=tokenizer,
        recovered=True,
    )
    return event


def _write_event_and_receipt(
    *,
    run_dir: Path,
    events: list[dict[str, Any]],
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    role: str,
    request_id: str,
    prompt_sha256: str,
    response: TeacherResponse,
    receipt_sha256: str,
    tokenizer: Any,
    recovered: bool = False,
) -> dict[str, Any]:
    event = _event_for_completed(
        plan=plan,
        candidate_id=str(row["candidate_id"]),
        role=role,
        request_id=request_id,
        prompt_sha256=prompt_sha256,
        response=response,
        response_receipt_sha256=receipt_sha256,
        tokenizer=tokenizer,
        recovered=recovered,
    )
    return _append_event(run_dir, event, events)


def _append_failure_event(
    *,
    run_dir: Path,
    events: list[dict[str, Any]],
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    role: str,
    request_id: str,
    prompt_sha256: str,
    status: str,
    failure_class: str,
    failure_stage: str | None,
    terminal_response: TeacherResponse | None = None,
    response_receipt_sha256: str | None = None,
    ledger_reconciled: bool | None = None,
) -> dict[str, Any]:
    response_fields: dict[str, Any] = {}
    if terminal_response is not None:
        response_fields = {
            "response_sha256": _sha(terminal_response.content.encode("utf-8")),
            "response_receipt_sha256": response_receipt_sha256,
            "finish_reason": terminal_response.finish_reason,
            "provider_input_tokens": terminal_response.input_tokens,
            "provider_output_tokens": terminal_response.output_tokens,
            "provider_reasoning_tokens": terminal_response.reasoning_tokens,
            "ledger_reconciled": ledger_reconciled,
        }
    return _append_event(
        run_dir,
        {
            "plan_sha256": plan["plan_sha256"],
            "candidate_id": row["candidate_id"],
            "candidate_key": _digest_id(str(row["candidate_id"])),
            "role": role,
            "request_id": request_id,
            "actor_id": MODEL_ID + ":" + ACTOR_SUFFIX[role],
            "model_id": MODEL_ID,
            "prompt_sha256": prompt_sha256,
            "response_sha256": None,
            "response_receipt_sha256": None,
            "finish_reason": None,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_reasoning_tokens": None,
            **response_fields,
            "status": status,
            "failure_class": failure_class,
            "failure_stage": failure_stage,
            "recovered_from_persisted_receipt": False,
            "recorded_at_unix_ns": time.time_ns(),
            "retry_permitted": False,
        },
        events,
    )


def _episode_row(input_row: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    state = EditState.from_mapping(input_row["state"])
    prompt = build_fixed_state_prompt(state, tokenizer)
    return {
        **input_row,
        "id": input_row["candidate_id"],
        "source_type": SOURCE_TYPE,
        "history_order": HISTORY_ORDER,
        "human_chronology_observed": False,
        "context_policy": CONTEXT_POLICY_VERSION,
        "history_sha256": prompt.history_sha256,
    }


def _role_event_output(run_dir: Path, event: Mapping[str, Any]) -> str:
    if event.get("status") not in {"completed_valid_action", "completed_valid_review"}:
        raise ValueError("role event has no valid complete output")
    response_path, _receipt = _response_paths(run_dir, str(event["request_id"]))
    response = _private_file(response_path).decode("utf-8")
    if _sha(response.encode("utf-8")) != event.get("response_sha256"):
        raise ValueError("role response differs from its append-only event")
    return response


def _role_evidence_for_case(
    *,
    run_dir: Path,
    input_row: Mapping[str, Any],
    role_events: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[dict[str, Any], bytes, Any]:
    author_wire = _role_event_output(run_dir, role_events["author"])
    solver_wire = _role_event_output(run_dir, role_events["solver"])
    author_action = parse_fixed_state_action(
        _response_from_event(run_dir, role_events["author"]), tokenizer
    )[1]
    solver_action = parse_fixed_state_action(
        _response_from_event(run_dir, role_events["solver"]), tokenizer
    )[1]
    if author_action != solver_action:
        raise ValueError("author and blind solver chose different canonical actions")
    state = EditState.from_mapping(input_row["state"])
    candidate = _episode_row(input_row, tokenizer)
    candidate["action"] = asdict(author_action)
    candidate["after_source"] = apply_action(state, author_action)
    reviewer_prompt = build_fixed_state_review_prompt(state, author_wire, solver_wire, tokenizer)
    receipts: dict[str, CompletedFixedStateRole] = {}
    for role in CALL_ROLES:
        event = role_events[role]
        response, _receipt_sha = _load_response(run_dir, str(event["request_id"]))
        expected_prompt_sha = (
            build_fixed_state_prompt(state, tokenizer).prompt_sha256
            if role != "reviewer"
            else reviewer_prompt.prompt_sha256
        )
        receipts[role] = CompletedFixedStateRole(
            actor_id=str(event["actor_id"]),
            request_id=str(event["request_id"]),
            prompt_sha256=expected_prompt_sha,
            response=response,
        )
    payload = build_fixed_state_role_evidence(
        candidate,
        author=receipts["author"],
        solver=receipts["solver"],
        reviewer=receipts["reviewer"],
        tokenizer=tokenizer,
    )
    decision = verify_fixed_state_role_evidence(candidate, payload, tokenizer=tokenizer)
    return candidate, payload, decision


def _response_from_event(run_dir: Path, event: Mapping[str, Any]) -> TeacherResponse:
    response, _receipt_sha = _load_response(run_dir, str(event["request_id"]))
    return response


def _role_completed_as(
    role_events: Mapping[str, dict[str, Any] | None], role: str, expected: str
) -> bool:
    event = role_events[role]
    return event is not None and event.get("status") == expected


def _summary_for_run(
    *,
    run_dir: Path,
    plan: Mapping[str, Any],
    rows: list[dict[str, Any]],
    events_by_id: Mapping[str, dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    evidence_dir = run_dir / "role_evidence"
    _private_directory(evidence_dir, create=True)
    for row in rows:
        candidate_id = row["candidate_id"]
        planned = next(case for case in plan["cases"] if case["candidate_id"] == candidate_id)
        role_events = {role: events_by_id.get(planned["request_ids"][role]) for role in CALL_ROLES}
        status_by_role = {
            role: event.get("status") if event is not None else "not_called"
            for role, event in role_events.items()
        }
        completed_role_events = {
            role: event for role, event in role_events.items() if event is not None
        }
        result: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_key": _digest_id(candidate_id),
            "seed_id": row["seed_id"],
            "role_status": status_by_role,
            "role_evidence_accepted": False,
            "functional_status": "not_evaluated_by_role_runner",
            "training_accepted": False,
        }

        if all(
            _role_completed_as(role_events, role, expected)
            for role, expected in (
                ("author", "completed_valid_action"),
                ("solver", "completed_valid_action"),
                ("reviewer", "completed_valid_review"),
            )
        ):
            try:
                candidate, payload, decision = _role_evidence_for_case(
                    run_dir=run_dir,
                    input_row=row,
                    role_events=completed_role_events,
                    tokenizer=tokenizer,
                )
                role_path = evidence_dir / f"{_digest_id(candidate_id)}.json"
                _write_once(role_path, payload)
                result.update(
                    {
                        "role_evidence_accepted": decision.accepted,
                        "role_evidence_reason": decision.reason,
                        "role_evidence_sha256": _sha(payload),
                        "state_sha256": _sha(_canonical(candidate["state"])),
                        "action_sha256": _sha(_canonical(candidate["action"])),
                        "after_source_sha256": _sha(candidate["after_source"].encode("utf-8")),
                        "reviewer_retained": bool(decision.evidence.get("reviewer_retained")),
                        "author_solver_action_agreement": bool(
                            decision.evidence.get("author_solver_action_agreement")
                        ),
                    }
                )
            except (KeyError, TypeError, UnicodeError, ValueError):
                result["role_evidence_reason"] = "role_evidence_assembly_failed"
        elif status_by_role["reviewer"] == "not_called":
            if all(
                status_by_role[role] == "completed_valid_action" for role in ("author", "solver")
            ):
                try:
                    author_event = role_events["author"]
                    solver_event = role_events["solver"]
                    if author_event is None or solver_event is None:
                        raise ValueError("valid action event missing")
                    author_action = parse_fixed_state_action(
                        _response_from_event(run_dir, author_event), tokenizer
                    )[1]
                    solver_action = parse_fixed_state_action(
                        _response_from_event(run_dir, solver_event), tokenizer
                    )[1]
                    result["role_evidence_reason"] = (
                        "reviewer_skipped_after_action_disagreement"
                        if author_action != solver_action
                        else "reviewer_not_called"
                    )
                except (KeyError, ValueError):
                    result["role_evidence_reason"] = "reviewer_skipped_after_invalid_action"
            else:
                result["role_evidence_reason"] = "reviewer_skipped_without_two_valid_actions"
        cases.append(result)
    return {
        "schema": SUMMARY_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "model_id": MODEL_ID,
        "case_count": len(cases),
        "provider_call_ceiling": plan["provider"]["planned_calls"],
        "provider_calls_made": len(events_by_id),
        "functional_quality_claim": False,
        "training_started": False,
        "training_accepted": False,
        "personalization_enabled": False,
        "cases": cases,
        "updated_at_unix_ns": time.time_ns(),
    }


def _enable_offline_observability(run_dir: Path) -> None:
    os.umask(0o077)
    os.environ["TABCOMPLETE_OBSERVABILITY_ENABLED"] = "1"
    os.environ["TABCOMPLETE_OBSERVABILITY_MODE"] = "offline"
    os.environ["TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE"] = str(run_dir / "telemetry.jsonl")
    os.environ["TABCOMPLETE_OBSERVABILITY_OFFLINE_MAX_BYTES"] = str(4 * 1024 * 1024)
    os.environ["TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT"] = "0"


def execute_plan(
    *,
    run_dir: Path,
    release_path: Path,
    release_sha256: str,
    tokenizer: Any | None = None,
    runtime: dict[str, Any] | None = None,
    ledger_path_override: Path | None = None,
    client_factory: Callable[..., Any] = OpenCodeTeacherClient,
) -> dict[str, Any]:
    """Execute a released phase; tests can inject only local fakes, never auto-call."""
    plan, rows = _load_frozen_plan(run_dir)
    release = _read_release(release_path, release_sha256, plan)
    clock = _read_attempt_state(run_dir, plan, int(release["max_wall_seconds"]))
    phase_deadline = int(clock["first_started_unix_ns"]) / 1_000_000_000 + int(
        release["max_wall_seconds"]
    )
    call_deadline = phase_deadline - FINALIZATION_RESERVE_SECONDS
    if (runtime if runtime is not None else _runtime_identity()) != plan["runtime"]:
        raise ValueError("runtime changed after fixed-state plan freeze")
    if (ledger_path_override or Path(plan["ledger"]["path"])).resolve(strict=False) != Path(
        plan["ledger"]["path"]
    ).resolve(strict=False):
        raise ValueError("execution cannot redirect the frozen usage ledger")
    tokenizer = tokenizer or _tokenizer()
    rows = _validate_case_plan(plan, run_dir, tokenizer)
    events, events_by_id = _read_event_log(run_dir, plan)
    _current_rows, _totals, new_ids = _validate_ledger_progress(
        plan, run_dir, events_by_id, ledger_path_override
    )
    for case in plan["cases"]:
        row = next(item for item in rows if item["candidate_id"] == case["candidate_id"])
        recovered_outputs: dict[str, str] = {}
        for role in ("author", "solver"):
            request_id = case["request_ids"][role]
            existing = events_by_id.get(request_id)
            if request_id in new_ids and existing is None:
                prompt, system, _purpose, _prompt_hash = _prompt_for_role(role, row, {}, tokenizer)
                recovered = _recovered_event(
                    run_dir=run_dir,
                    plan=plan,
                    row=row,
                    role=role,
                    request_id=request_id,
                    prompt=prompt,
                    system=system,
                    tokenizer=tokenizer,
                )
                _append_event(run_dir, recovered, events)
                events_by_id[request_id] = events[-1]
                existing = events[-1]
            if existing is not None and existing.get("status") == "completed_valid_action":
                recovered_outputs[role] = _role_event_output(run_dir, existing)
        reviewer_id = case["request_ids"]["reviewer"]
        if reviewer_id in new_ids and reviewer_id not in events_by_id:
            if set(recovered_outputs) != {"author", "solver"}:
                raise ValueError("reviewer receipt exists without two valid action receipts")
            prompt, system, _purpose, _prompt_hash = _prompt_for_role(
                "reviewer",
                row,
                {
                    "author_wire": recovered_outputs["author"],
                    "solver_wire": recovered_outputs["solver"],
                },
                tokenizer,
            )
            recovered = _recovered_event(
                run_dir=run_dir,
                plan=plan,
                row=row,
                role="reviewer",
                request_id=reviewer_id,
                prompt=prompt,
                system=system,
                tokenizer=tokenizer,
            )
            _append_event(run_dir, recovered, events)
            events_by_id[reviewer_id] = events[-1]
    # An earlier ambiguous/provider failure is terminal for the whole phase.
    if any(
        event.get("status")
        in {
            "ambiguous_transport_failure",
            "response_unusable_unknown_usage",
            "completed_terminal_failure",
        }
        for event in events
    ):
        summary = _summary_for_run(
            run_dir=run_dir, plan=plan, rows=rows, events_by_id=events_by_id, tokenizer=tokenizer
        )
        summary["phase_status"] = "stopped_after_terminal_provider_failure"
        _write_once(run_dir / "summary.json", _canonical(summary) + b"\n")
        return summary

    _enable_offline_observability(run_dir)
    campaign = RunContext.new(campaign_id="fixed-state-role-pilot-v1")
    pending = False
    try:
        with (
            campaign.activate(),
            run_scope(
                run_dir / "observability-run.json", "fixed-state-role-pilot-v1"
            ) as observed_context,
        ):
            correlation = observed_context or campaign
            pending_ids = {
                case["request_ids"][role]
                for case in plan["cases"]
                for role in CALL_ROLES
                if case["request_ids"][role] not in events_by_id
            }
            if not pending_ids:
                summary = _summary_for_run(
                    run_dir=run_dir,
                    plan=plan,
                    rows=rows,
                    events_by_id=events_by_id,
                    tokenizer=tokenizer,
                )
                summary["phase_status"] = (
                    "complete"
                    if len(events_by_id) == plan["provider"]["planned_calls"]
                    else "complete_with_role_skips"
                )
                _replace_derived(run_dir / "summary.json", _canonical(summary) + b"\n")
                return summary

            if time.time() >= call_deadline:
                pending = True
            else:
                ledger = TeacherUsageLedger(ledger_path_override or Path(plan["ledger"]["path"]))
                with client_factory(ledger) as client:
                    for input_row in rows:
                        candidate_id = input_row["candidate_id"]
                        plan_case = next(
                            case for case in plan["cases"] if case["candidate_id"] == candidate_id
                        )
                        role_outputs: dict[str, str] = {}
                        circuit_breaker = False
                        for role in ("author", "solver"):
                            request_id = plan_case["request_ids"][role]
                            existing_event = events_by_id.get(request_id)
                            if existing_event is not None:
                                if existing_event.get("status") == "completed_valid_action":
                                    role_outputs[role] = _role_event_output(run_dir, existing_event)
                                continue
                            if (
                                time.time() >= call_deadline
                                or len(events_by_id) >= release["max_calls"]
                            ):
                                pending = True
                                break
                            prompt, system, purpose, prompt_hash = _prompt_for_role(
                                role, input_row, {}, tokenizer
                            )
                            prompt_record = {"system_instruction": system, "user_prompt": prompt}
                            prompt_bytes = _canonical(prompt_record)
                            _write_once(_prompt_path(run_dir, request_id), prompt_bytes)
                            if prompt_hash != _sha(
                                _canonical({"system_instruction": system, "user_prompt": prompt})
                            ):
                                raise ValueError("fixed-state prompt hash function changed")
                            case_context = correlation.for_case(
                                _digest_id(candidate_id), request_id=request_id
                            )
                            try:
                                with (
                                    case_context.activate(),
                                    operation(
                                        "model.generate",
                                        attributes={
                                            "tabcomplete.phase": role,
                                            "gen_ai.request.model": MODEL_ID,
                                            "rank_role": role,
                                        },
                                    ),
                                ):
                                    response = client.run_role(
                                        request_id=request_id,
                                        prompt=prompt,
                                        purpose=purpose,
                                        source_class="public",
                                        authorization_basis=AUTHORIZATION_BASIS,
                                        reserve_input_tokens=RESERVE_INPUT_TOKENS,
                                        reserve_output_tokens=RESERVE_OUTPUT_TOKENS,
                                        output_schema=None,
                                        system_instruction=system,
                                        persist_completed_response=lambda result, rid=request_id: (
                                            _persist_response(run_dir, rid, result)
                                        ),
                                    )
                                response, receipt_sha = _load_response(run_dir, request_id)
                                event = _write_event_and_receipt(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role=role,
                                    request_id=request_id,
                                    prompt_sha256=prompt_hash,
                                    response=response,
                                    receipt_sha256=receipt_sha,
                                    tokenizer=tokenizer,
                                )
                                events_by_id[request_id] = event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                                if event["status"] == "completed_valid_action":
                                    role_outputs[role] = response.content
                            except TeacherCompletedResponseError as error:
                                event = _append_failure_event(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role=role,
                                    request_id=request_id,
                                    prompt_sha256=prompt_hash,
                                    status="completed_terminal_failure",
                                    failure_class=error.failure_status,
                                    failure_stage=None,
                                    terminal_response=error.response,
                                    response_receipt_sha256=error.evidence_sha256,
                                    ledger_reconciled=error.ledger_reconciled,
                                )
                                events_by_id[request_id] = event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                                circuit_breaker = True
                                break
                            except TeacherResponseValidationError as error:
                                if error.request_id != request_id:
                                    raise RuntimeError(
                                        "response validation request ID mismatch"
                                    ) from None
                                event = _append_failure_event(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role=role,
                                    request_id=request_id,
                                    prompt_sha256=prompt_hash,
                                    status="response_unusable_unknown_usage",
                                    failure_class=error.failure_kind,
                                    failure_stage=error.stage,
                                )
                                events_by_id[request_id] = event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                                circuit_breaker = True
                                break
                            except TeacherTransportError as error:
                                if error.request_id not in {None, request_id}:
                                    raise RuntimeError(
                                        "transport error request ID mismatch"
                                    ) from None
                                event = _append_failure_event(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role=role,
                                    request_id=request_id,
                                    prompt_sha256=prompt_hash,
                                    status="ambiguous_transport_failure",
                                    failure_class=type(error).__name__,
                                    failure_stage=error.stage,
                                )
                                events_by_id[request_id] = event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                                circuit_breaker = True
                                break
                        if circuit_breaker or pending:
                            break

                        if set(role_outputs) != {"author", "solver"}:
                            continue
                        try:
                            author_action = parse_fixed_state_action(
                                _response_from_event(
                                    run_dir, events_by_id[plan_case["request_ids"]["author"]]
                                ),
                                tokenizer,
                            )[1]
                            solver_action = parse_fixed_state_action(
                                _response_from_event(
                                    run_dir, events_by_id[plan_case["request_ids"]["solver"]]
                                ),
                                tokenizer,
                            )[1]
                        except (KeyError, ValueError):
                            continue
                        if author_action != solver_action:
                            continue

                        reviewer_id = plan_case["request_ids"]["reviewer"]
                        reviewer_event = events_by_id.get(reviewer_id)
                        if reviewer_event is None:
                            if (
                                time.time() >= call_deadline
                                or len(events_by_id) >= release["max_calls"]
                            ):
                                pending = True
                                break
                            previous = {
                                "author_wire": role_outputs["author"],
                                "solver_wire": role_outputs["solver"],
                            }
                            prompt, system, purpose, prompt_hash = _prompt_for_role(
                                "reviewer", input_row, previous, tokenizer
                            )
                            _write_once(
                                _prompt_path(run_dir, reviewer_id),
                                _canonical({"system_instruction": system, "user_prompt": prompt}),
                            )
                            case_context = correlation.for_case(
                                _digest_id(candidate_id), request_id=reviewer_id
                            )
                            try:
                                with (
                                    case_context.activate(),
                                    operation(
                                        "model.generate",
                                        attributes={
                                            "tabcomplete.phase": "reviewer",
                                            "gen_ai.request.model": MODEL_ID,
                                            "rank_role": "reviewer",
                                        },
                                    ),
                                ):
                                    client.run_role(
                                        request_id=reviewer_id,
                                        prompt=prompt,
                                        purpose=purpose,
                                        source_class="public",
                                        authorization_basis=AUTHORIZATION_BASIS,
                                        reserve_input_tokens=RESERVE_INPUT_TOKENS,
                                        reserve_output_tokens=RESERVE_OUTPUT_TOKENS,
                                        output_schema=None,
                                        system_instruction=system,
                                        persist_completed_response=lambda result, rid=reviewer_id: (
                                            _persist_response(run_dir, rid, result)
                                        ),
                                    )
                                response, receipt_sha = _load_response(run_dir, reviewer_id)
                                reviewer_event = _write_event_and_receipt(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role="reviewer",
                                    request_id=reviewer_id,
                                    prompt_sha256=prompt_hash,
                                    response=response,
                                    receipt_sha256=receipt_sha,
                                    tokenizer=tokenizer,
                                )
                                events_by_id[reviewer_id] = reviewer_event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                            except TeacherCompletedResponseError as error:
                                reviewer_event = _append_failure_event(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role="reviewer",
                                    request_id=reviewer_id,
                                    prompt_sha256=prompt_hash,
                                    status="completed_terminal_failure",
                                    failure_class=error.failure_status,
                                    failure_stage=None,
                                    terminal_response=error.response,
                                    response_receipt_sha256=error.evidence_sha256,
                                    ledger_reconciled=error.ledger_reconciled,
                                )
                                events_by_id[reviewer_id] = reviewer_event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                                break
                            except TeacherResponseValidationError as error:
                                if error.request_id != reviewer_id:
                                    raise RuntimeError(
                                        "response validation request ID mismatch"
                                    ) from None
                                reviewer_event = _append_failure_event(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role="reviewer",
                                    request_id=reviewer_id,
                                    prompt_sha256=prompt_hash,
                                    status="response_unusable_unknown_usage",
                                    failure_class=error.failure_kind,
                                    failure_stage=error.stage,
                                )
                                events_by_id[reviewer_id] = reviewer_event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                                break
                            except TeacherTransportError as error:
                                if error.request_id not in {None, reviewer_id}:
                                    raise RuntimeError(
                                        "transport error request ID mismatch"
                                    ) from None
                                reviewer_event = _append_failure_event(
                                    run_dir=run_dir,
                                    events=events,
                                    plan=plan,
                                    row=input_row,
                                    role="reviewer",
                                    request_id=reviewer_id,
                                    prompt_sha256=prompt_hash,
                                    status="ambiguous_transport_failure",
                                    failure_class=type(error).__name__,
                                    failure_stage=error.stage,
                                )
                                events_by_id[reviewer_id] = reviewer_event
                                _validate_ledger_progress(
                                    plan, run_dir, events_by_id, ledger_path_override
                                )
                                break
    except (TeacherPolicyError, OSError, ValueError):
        raise

    summary = _summary_for_run(
        run_dir=run_dir, plan=plan, rows=rows, events_by_id=events_by_id, tokenizer=tokenizer
    )
    terminal_failure = any(
        event.get("status")
        in {
            "ambiguous_transport_failure",
            "response_unusable_unknown_usage",
            "completed_terminal_failure",
        }
        for event in events_by_id.values()
    )
    summary["phase_status"] = (
        "stopped_after_terminal_provider_failure"
        if terminal_failure
        else "incomplete_wall_deadline"
        if pending
        else "complete_with_role_skips"
        if len(events_by_id) < plan["provider"]["planned_calls"]
        else "complete"
    )
    summary["phase_deadline_unix_ns"] = int(phase_deadline * 1_000_000_000)
    _replace_derived(run_dir / "summary.json", _canonical(summary) + b"\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, help="owner-only answer-free JSONL packet")
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="new owner-only output directory"
    )
    parser.add_argument("--ledger", type=Path, default=LEDGER_DEFAULT)
    parser.add_argument("--freeze-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--release-sha256")
    args = parser.parse_args(argv)
    if args.freeze_plan == args.execute:
        parser.error("choose exactly one of --freeze-plan or --execute")
    if args.freeze_plan:
        if args.packet is None:
            parser.error("--packet is required with --freeze-plan")
        plan = freeze_plan(packet_path=args.packet, run_dir=args.run_dir, ledger_path=args.ledger)
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "plan_sha256": plan["plan_sha256"],
                    "plan_path": str(args.run_dir / "plan.json"),
                    "case_count": plan["packet"]["case_count"],
                    "planned_calls": plan["provider"]["planned_calls"],
                    "provider_calls": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.release is None or args.release_sha256 is None:
        parser.error("--release and root-provided --release-sha256 are required with --execute")
    summary = execute_plan(
        run_dir=args.run_dir,
        release_path=args.release,
        release_sha256=args.release_sha256,
        ledger_path_override=args.ledger,
    )
    print(
        json.dumps(
            {
                "phase_status": summary["phase_status"],
                "plan_sha256": summary["plan_sha256"],
                "provider_calls_made": summary["provider_calls_made"],
                "case_count": summary["case_count"],
                "functional_quality_claim": False,
                "training_started": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
