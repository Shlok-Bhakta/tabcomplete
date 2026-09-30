#!/usr/bin/env python3
"""Measure the frozen Sweep prompt bundle on crabcake's CPU-only llama.cpp runtime.

This runner is intentionally separate from the Kaggle comparison runner. It never
downloads or converts weights, never connects to the resident editor service, and
requires ``--execute`` after the local plan has been frozen.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "reports/prototype/sweep_comparison_r1/local_replay_plan.json"
DEFAULT_UPSTREAM_PLAN = ROOT / "reports/prototype/sweep_comparison_r1/plan-v2.json"
DEFAULT_FIXTURES = ROOT / "reports/prototype/sweep_comparison_r1/next_edit_inputs.jsonl"
DEFAULT_RUNTIME = Path("/home/crabcake/Projects/tabcomplete/outputs/tools/llama.cpp")
DEFAULT_ARTIFACT_DIR = Path("/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/artifacts")
DEFAULT_OUTPUT_ROOT = Path("/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/local_replay")

MODEL_FILES = {
    "q4_k_m": "sweep-next-edit-1.5b.q4_k_m.gguf",
    "q8_0": "sweep-next-edit-1.5b.q8_0.v2.gguf",
}
MODEL_SHA256 = {
    "q4_k_m": "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4",
    "q8_0": "1321ea5e5d7529e60f9770c6a0b3a965f89542d16cf4ae51bab267f6a88150da",
}
MODEL_BYTES = {"q4_k_m": 883_289_056, "q8_0": 1_537_269_856}
RUNTIME_REVISION = "f072b103714dfa1eee531f80b24512faf38e3dd2"
CONTEXT_TOKENS = 3_072
OUTPUT_TOKENS = 512
SERVER_SETTINGS = {
    "context_tokens": CONTEXT_TOKENS,
    "batch_tokens": 256,
    "microbatch_tokens": 64,
    "parallel_slots": 1,
    "threads": 4,
    "threads_batch": 4,
    "cache_ram_mib": 0,
    "cache_idle_slots": False,
    "context_checkpoints": 0,
    "context_shift": False,
    "weight_repacking": False,
    "kv_cache_k": "f16",
    "kv_cache_v": "f16",
    "speculative_decoding": "none",
    "gpu_layers": 0,
    "warmup": False,
    "log_verbosity": 4,
}
REQUEST_SETTINGS = {
    "cache_prompt": True,
    "n_predict": OUTPUT_TOKENS,
    "temperature": 0,
    "seed": 1,
    "stop": ["<|file_sep|>", "</s>"],
    "return_tokens": True,
    "id_slot": 0,
    "stream": True,
}
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
TRANSITION_PLAN_SCHEMA = "sweep-local-changing-state-plan-v1"
TRANSITION_SUITE_MODE = "changing_editor_state"
PUBLISHER_PROMPT_EXACT_POLICY = "publisher_prompt_exact_v1"
PUBLISHER_HEADER_LF_INPUT_POLICY = "publisher-header-lf-input-v2"
TRANSITION_TYPES = {
    "fresh_file_open",
    "append_chars",
    "near_cursor_replace",
    "explicit_reject_then_divergent_typing",
    "typed_matching_prefix",
    "file_switch",
    "file_switch_return",
    "earlier_edit_invalidation",
}
PROMPT_BUILDER_ARCHIVE_COMMIT = "a681d5c4ba88388742b4217a838d840ed5c257a5"
BASELINE_SUMMARY_SHA256 = "63f5f6de54e64edcb01abd76c2b7dafa86629d9fd7f7c450c07cafad9d662465"
BASELINE_SUMMARY_PATH = (
    "/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/local_replay/"
    "20260930T090103Z-00540772e0/summary.json"
)
CONTEXT_PROVENANCE = "synthetic_repo_tabcomplete_editor_r1"
CONTEXT_RATIONALE = (
    "fixed synthetic adjacent editor modules; per-bucket modules are frozen in plan "
    "and retain their shared ordering"
)
OBSERVABILITY_CAMPAIGN = "sweep-local-replay-crabcake-r1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_RESULT_BYTES = 512 * 1024 * 1024
MAX_WALL_SECONDS = 90 * 60
FINALIZATION_RESERVE_SECONDS = 10 * 60


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def plan_digest(plan: dict[str, Any]) -> str:
    return digest_bytes(
        canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"fixture row {number} is not an object")
        rows.append(value)
    return rows


def _function_source_sha256(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"publisher source must define exactly one {name}")
    segment = ast.get_source_segment(source, matches[0])
    if segment is None:
        raise ValueError(f"could not extract publisher function {name}")
    return digest_bytes(segment.encode("utf-8"))


def validate_trajectory_fixtures(rows: list[dict[str, Any]], replay: dict[str, Any]) -> None:
    """Validate a fixed synthetic state sequence without changing publisher prompts."""
    fixture_order = replay.get("fixture_order")
    if not isinstance(fixture_order, list) or len(fixture_order) != 24:
        raise ValueError("changing-state replay must bind exactly 24 ordered transition states")
    if [row.get("case_id") for row in rows] != fixture_order:
        raise ValueError("changing-state fixture order differs from the frozen plan")
    if len({row.get("case_id") for row in rows}) != 24:
        raise ValueError("changing-state fixture IDs must be unique")
    expected_bucket_counts = {"512": 8, "1024": 8, "2048": 8}
    bucket_counts = Counter(str(row.get("nominal_context_bucket")) for row in rows)
    if dict(bucket_counts) != expected_bucket_counts:
        raise ValueError(
            "changing-state fixtures must contain eight states per nominal context bucket"
        )
    transition_counts = Counter(str(row.get("transition_type")) for row in rows)
    if dict(transition_counts) != replay.get("transition_type_counts"):
        raise ValueError("changing-state transition counts differ from the frozen plan")
    if not set(transition_counts).issubset(TRANSITION_TYPES):
        raise ValueError("changing-state fixture contains an unsupported transition type")
    context_modules_by_bucket = replay.get("context_modules_by_bucket")
    expected_context_hashes = replay.get("context_file_sha256")
    if not isinstance(context_modules_by_bucket, dict) or not isinstance(
        expected_context_hashes, dict
    ):
        raise ValueError("changing-state plan lacks fixed context provenance")

    trajectory_previous: dict[str, dict[str, Any]] = {}
    state_by_id: dict[str, dict[str, Any]] = {}
    closed_trajectories: set[str] = set()
    active_trajectory: str | None = None
    for row in rows:
        trajectory_id = row.get("trajectory_id")
        state_id = row.get("state_id")
        transition_type = row.get("transition_type")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("changing-state fixture lacks a trajectory identity")
        if active_trajectory != trajectory_id:
            if trajectory_id in closed_trajectories:
                raise ValueError("trajectory states must remain contiguous in request order")
            if active_trajectory is not None:
                closed_trajectories.add(active_trajectory)
            active_trajectory = trajectory_id
        if not isinstance(state_id, str) or not state_id or state_id in state_by_id:
            raise ValueError("changing-state state identities must be unique strings")
        if transition_type not in TRANSITION_TYPES:
            raise ValueError("changing-state fixture has an unknown transition type")
        if row.get("synthetic_only") is not True:
            raise ValueError("changing-state editor events must be explicitly marked synthetic")
        file_path = row.get("file_path")
        current = row.get("current_content")
        original = row.get("original_content")
        context_files = row.get("context_files")
        recent_diffs = row.get("recent_diffs")
        if (
            not isinstance(file_path, str)
            or not isinstance(current, str)
            or not isinstance(original, str)
        ):
            raise ValueError(f"changing-state fixture {state_id} has malformed full-file content")
        if not isinstance(context_files, dict) or any(
            not isinstance(path, str) or not isinstance(content, str)
            for path, content in context_files.items()
        ):
            raise ValueError(
                f"changing-state fixture {state_id} has malformed nearby context files"
            )
        bucket = str(row.get("nominal_context_bucket"))
        if list(context_files) != context_modules_by_bucket.get(bucket):
            raise ValueError(
                f"changing-state fixture {state_id} has a different nearby-context module set"
            )
        if (
            row.get("context_provenance") != CONTEXT_PROVENANCE
            or row.get("context_rationale") != CONTEXT_RATIONALE
        ):
            raise ValueError(f"changing-state fixture {state_id} has unpinned context provenance")
        context_hashes = row.get("context_file_sha256")
        expected_hashes = {path: expected_context_hashes.get(path) for path in context_files}
        if not isinstance(context_hashes, dict) or context_hashes != expected_hashes:
            raise ValueError(
                f"changing-state fixture {state_id} context hashes differ from the plan"
            )
        if any(
            digest_bytes(content.encode("utf-8")) != context_hashes.get(path)
            for path, content in context_files.items()
        ):
            raise ValueError(
                f"changing-state fixture {state_id} context bytes differ from their hashes"
            )
        if not isinstance(recent_diffs, list):
            raise ValueError(f"changing-state fixture {state_id} has malformed edit history")
        previous_content: str | None = None
        for diff in recent_diffs:
            if not isinstance(diff, dict) or diff.get("file_path") != file_path:
                raise ValueError(
                    f"changing-state fixture {state_id} has a cross-file history delta"
                )
            old = diff.get("original")
            new = diff.get("updated")
            if not isinstance(old, str) or not isinstance(new, str):
                raise ValueError(f"changing-state fixture {state_id} has malformed history text")
            if previous_content is not None and old != previous_content:
                raise ValueError(f"changing-state fixture {state_id} history is not contiguous")
            previous_content = new
        if recent_diffs and previous_content != current:
            raise ValueError(
                f"changing-state fixture {state_id} history does not reconstruct current source"
            )
        if not recent_diffs and current != original:
            raise ValueError(
                f"unchanged/open fixture {state_id} must use identical original/current source"
            )
        source_bytes = current.encode("utf-8")
        editable_start = row.get("editable_start_byte")
        editable_end = row.get("editable_end_byte")
        cursor_offset = row.get("cursor_byte_offset")
        if (
            not isinstance(editable_start, int)
            or not isinstance(editable_end, int)
            or not isinstance(cursor_offset, int)
        ):
            raise ValueError(
                f"changing-state fixture {state_id} lacks cursor/editable byte positions"
            )
        if not 0 <= editable_start <= editable_end <= len(
            source_bytes
        ) or not 0 <= cursor_offset <= len(source_bytes):
            raise ValueError(
                f"changing-state fixture {state_id} editable byte range is out of bounds"
            )
        for offset in (editable_start, editable_end, cursor_offset):
            try:
                source_bytes[:offset].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"changing-state fixture {state_id} splits a UTF-8 codepoint"
                ) from exc
        expected_start = source_bytes.rfind(b"\n", 0, cursor_offset) + 1
        expected_end = source_bytes.find(b"\n", cursor_offset)
        if expected_end < 0:
            expected_end = len(source_bytes)
        if (editable_start, editable_end) != (expected_start, expected_end):
            raise ValueError(
                f"changing-state fixture {state_id} editable range is not the cursor line"
            )

        previous = trajectory_previous.get(trajectory_id)
        if transition_type == "fresh_file_open":
            if previous is not None or row.get("from_state_id") is not None or recent_diffs:
                raise ValueError(
                    "fresh file opens must begin a trajectory with no preceding edit history"
                )
            if original != current:
                raise ValueError("fresh file opens must start from an unchanged file state")
        elif previous is None:
            raise ValueError("every changing-state trajectory must begin with a fresh file open")
        else:
            if row.get("from_state_id") != previous["state_id"]:
                raise ValueError(
                    f"changing-state fixture {state_id} does not follow its declared predecessor"
                )
            if transition_type in {
                "append_chars",
                "near_cursor_replace",
                "explicit_reject_then_divergent_typing",
                "typed_matching_prefix",
                "earlier_edit_invalidation",
            }:
                if (
                    row.get("file_id") != previous.get("file_id")
                    or file_path != previous["file_path"]
                ):
                    raise ValueError("text-edit transitions must remain in the same file")
                if original != previous["current_content"]:
                    raise ValueError(
                        "edit transition original source differs from its predecessor state"
                    )
                if len(recent_diffs) != len(previous["recent_diffs"]) + 1:
                    raise ValueError("edit transition must append exactly one reconstructed delta")
                if recent_diffs[:-1] != previous["recent_diffs"]:
                    raise ValueError("edit transition dropped or changed earlier history")
                last = recent_diffs[-1]
                if last != {"file_path": file_path, "original": original, "updated": current}:
                    raise ValueError(
                        "edit transition delta does not exactly match its before/after states"
                    )
                event = row.get("synthetic_event")
                if not isinstance(event, dict):
                    raise ValueError("synthetic edit transition lacks event evidence")
                start = event.get("start_byte")
                end = event.get("end_byte")
                cursor = event.get("cursor_byte_offset")
                result_cursor = event.get("result_cursor_byte_offset")
                replacement = event.get("replacement_text")
                original_bytes = original.encode("utf-8")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not isinstance(cursor, int)
                    or not isinstance(result_cursor, int)
                    or not isinstance(replacement, str)
                    or not 0 <= start <= end <= len(original_bytes)
                    or not 0 <= cursor <= len(original_bytes)
                    or not 0 <= result_cursor <= len(current.encode("utf-8"))
                ):
                    raise ValueError("synthetic edit lacks an exact byte-range and cursor record")
                for offset in (start, end, cursor):
                    try:
                        original_bytes[:offset].decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError("synthetic edit range splits a UTF-8 codepoint") from exc
                reconstructed = (
                    original_bytes[:start] + replacement.encode("utf-8") + original_bytes[end:]
                )
                if reconstructed.decode("utf-8") != current:
                    raise ValueError("synthetic byte delta does not reproduce the resulting source")
                current_bytes = current.encode("utf-8")
                try:
                    current_bytes[:result_cursor].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("post-edit cursor splits a UTF-8 codepoint") from exc
                if result_cursor != row.get("cursor_byte_offset"):
                    raise ValueError("fixture cursor does not match its last edit event")
                event_line = original_bytes[:start].count(b"\n")
                edit_cursor_line = original_bytes[:cursor].count(b"\n")
                result_cursor_line = current_bytes[:result_cursor].count(b"\n")
                if transition_type == "append_chars":
                    typed = event.get("typed_text")
                    if not isinstance(typed, str) or not typed or typed != replacement:
                        raise ValueError("append transition lacks exact appended text")
                    if (
                        start != len(original_bytes)
                        or end != start
                        or cursor != start
                        or result_cursor != start + len(replacement.encode("utf-8"))
                    ):
                        raise ValueError("append transition text does not match the source delta")
                elif transition_type in {
                    "explicit_reject_then_divergent_typing",
                    "typed_matching_prefix",
                }:
                    proposal = event.get("stub_proposal_text")
                    typed = event.get("typed_text")
                    if not isinstance(proposal, str) or not isinstance(typed, str) or not typed:
                        raise ValueError(
                            "typed-decision transition lacks its synthetic stub evidence"
                        )
                    if transition_type == "explicit_reject_then_divergent_typing":
                        if (
                            event.get("explicitly_rejected") is not True
                            or proposal.startswith(typed)
                            or typed != replacement
                            or result_cursor != start + len(replacement.encode("utf-8"))
                        ):
                            raise ValueError(
                                "divergent typing control is not an explicit rejected non-prefix"
                            )
                    else:
                        if not (proposal.startswith(typed) and len(typed) < len(proposal)):
                            raise ValueError("typed-match control is not a strict proposal prefix")
                        if not replacement.endswith(typed):
                            raise ValueError(
                                "typed-match text is not the exact inserted source suffix"
                            )
                        if result_cursor != start + len(replacement.encode("utf-8")):
                            raise ValueError(
                                "typed-match post-edit cursor differs from the inserted delta"
                            )
                elif transition_type == "near_cursor_replace":
                    if (
                        event_line != edit_cursor_line
                        or event_line != result_cursor_line
                        or event.get("edited_line_index") != event_line
                        or event.get("cursor_line_index") != result_cursor_line
                        or event.get("distance_lines") != 0
                        or result_cursor != start + len(replacement.encode("utf-8"))
                    ):
                        raise ValueError("near-cursor replacement is not bound to the cursor line")
                elif transition_type == "earlier_edit_invalidation":
                    before_line = event.get("edited_line_index")
                    declared_cursor_line = event.get("cursor_line_index")
                    if not isinstance(before_line, int) or not isinstance(
                        declared_cursor_line, int
                    ):
                        raise ValueError("earlier-edit transition lacks line-position evidence")
                    if (
                        before_line != event_line
                        or declared_cursor_line != result_cursor_line
                        or before_line >= result_cursor_line
                    ):
                        raise ValueError("earlier-edit invalidation must precede the cursor line")
            elif transition_type == "file_switch":
                if row.get("file_id") == previous.get("file_id"):
                    raise ValueError("file-switch transition did not change file identity")
                if original != current or recent_diffs:
                    raise ValueError(
                        "first open of a switched-to file must start from its exact current state"
                    )
            elif transition_type == "file_switch_return":
                target_id = row.get("return_to_state_id")
                if not isinstance(target_id, str):
                    raise ValueError("A-to-B-to-A return lacks a target state ID")
                target = state_by_id.get(target_id)
                if target is None or target.get("file_id") != row.get("file_id"):
                    raise ValueError(
                        "A-to-B-to-A return does not refer to an earlier state of file A"
                    )
                if previous.get("file_id") == row.get("file_id"):
                    raise ValueError("A-to-B-to-A return must follow a different active file")
                for field in (
                    "file_id",
                    "language",
                    "file_path",
                    "original_content",
                    "current_content",
                    "context_files",
                    "context_file_sha256",
                    "context_provenance",
                    "context_rationale",
                    "recent_diffs",
                    "editable_start_byte",
                    "editable_end_byte",
                    "cursor_byte_offset",
                    "nominal_context_bucket",
                    "prompt_sha256",
                ):
                    if row.get(field) != target.get(field):
                        raise ValueError(
                            f"file-switch return changed preserved file state field {field}"
                        )
        trajectory_previous[trajectory_id] = row
        state_by_id[state_id] = row


def validate_input_token_buckets(
    fixtures: list[dict[str, Any]], token_rows: list[dict[str, Any]], replay: dict[str, Any]
) -> None:
    windows = replay.get("input_bucket_windows")
    if not isinstance(windows, dict):
        raise ValueError("changing-state plan lacks frozen actual-token bucket windows")
    tokens_by_case = {row["case_id"]: row["input_tokens"] for row in token_rows}
    prompt_hashes_by_case = {
        row["case_id"]: row.get("effective_input_prompt_sha256") for row in token_rows
    }
    if len(tokens_by_case) != len(fixtures):
        raise ValueError("tokenizer response count differs from changing-state fixture count")
    expected_prompt_hashes = replay.get("effective_input_prompt_sha256_by_case")
    if isinstance(expected_prompt_hashes, dict) and prompt_hashes_by_case != expected_prompt_hashes:
        raise ValueError("effective model input prompt hashes differ from the frozen plan")
    expected_token_counts = replay.get("input_tokens_by_case")
    if isinstance(expected_token_counts, dict) and tokens_by_case != expected_token_counts:
        raise ValueError("actual model input token counts differ from the frozen plan")
    for fixture in fixtures:
        bucket = str(fixture.get("nominal_context_bucket"))
        window = windows.get(bucket)
        actual = tokens_by_case[fixture["case_id"]]
        if (
            not isinstance(window, list)
            or len(window) != 2
            or not isinstance(window[0], int)
            or not isinstance(window[1], int)
            or not window[0] <= actual <= window[1]
        ):
            raise ValueError(
                f"actual GGUF token count for {fixture['case_id']} is outside its frozen bucket"
            )


def validate_plan_policy(plan: dict[str, Any]) -> None:
    replay = plan.get("replay", {})
    suite_mode = replay.get("suite_mode", "fixed_snapshot")
    input_prompt_policy = replay.get("input_prompt_policy", PUBLISHER_PROMPT_EXACT_POLICY)
    if input_prompt_policy not in {
        PUBLISHER_PROMPT_EXACT_POLICY,
        PUBLISHER_HEADER_LF_INPUT_POLICY,
    }:
        raise ValueError("local replay has an unsupported input prompt policy")
    if (
        suite_mode == TRANSITION_SUITE_MODE
        and input_prompt_policy == PUBLISHER_HEADER_LF_INPUT_POLICY
    ):
        prompt_hashes = replay.get("effective_input_prompt_sha256_by_case")
        token_counts = replay.get("input_tokens_by_case")
        if not isinstance(prompt_hashes, dict) or len(prompt_hashes) != 24:
            raise ValueError("header-LF input policy lacks per-case effective prompt hashes")
        if not isinstance(token_counts, dict) or len(token_counts) != 24:
            raise ValueError("header-LF input policy lacks per-case actual token counts")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in prompt_hashes.items()
        ):
            raise ValueError("header-LF prompt hash map is malformed")
        if any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in token_counts.items()
        ):
            raise ValueError("header-LF tokenizer count map is malformed")
    expected_server_settings = dict(SERVER_SETTINGS)
    actual_server_settings = plan.get("runtime", {}).get("server_settings")
    allowed_server_settings: tuple[dict[str, Any], ...]
    if suite_mode == "fixed_snapshot":
        # Historical v1 plans predate the explicit bounded log verbosity flag.
        legacy_server_settings = dict(expected_server_settings)
        legacy_server_settings.pop("log_verbosity")
        allowed_server_settings = (expected_server_settings, legacy_server_settings)
    else:
        allowed_server_settings = (expected_server_settings,)
    if actual_server_settings not in allowed_server_settings:
        raise ValueError("server configuration differs from the frozen local replay plan")
    if replay.get("request_settings") != REQUEST_SETTINGS:
        raise ValueError("request configuration differs from the frozen local replay plan")
    if replay.get("precision_order") != ["q4_k_m", "q8_0"]:
        raise ValueError("local replay must run Q4_K_M then Q8_0, one model at a time")
    if replay.get("repetitions") != 2:
        raise ValueError("local replay must use exactly two repetitions")
    if replay.get("fixture_count") != 24:
        raise ValueError("local replay plan must contain exactly 24 frozen states")
    if suite_mode == "fixed_snapshot":
        if replay.get("fixture_order") != list(EXPECTED_CASE_IDS):
            raise ValueError("local replay fixture order differs from the frozen source states")
        if replay.get("expected_requests") != len(EXPECTED_CASE_IDS) * 2 * 2 * 2:
            raise ValueError("snapshot replay request count must bind both models and cache probes")
        if replay.get("request_order_per_fixture_per_repetition") != [
            "changed_state",
            "immediate_same_prompt_repeat",
        ]:
            raise ValueError("local replay cache-repeat ordering differs from the frozen plan")
    elif suite_mode == TRANSITION_SUITE_MODE:
        if plan.get("schema") != TRANSITION_PLAN_SCHEMA:
            raise ValueError("changing-state replay plan schema is invalid")
        fixture_order = replay.get("fixture_order")
        if not isinstance(fixture_order, list) or len(fixture_order) != 24:
            raise ValueError("changing-state replay must bind 24 ordered transitions")
        if len(set(fixture_order)) != 24:
            raise ValueError("changing-state plan fixture IDs must be unique")
        if replay.get("expected_requests") != 24 * 2 * 2:
            raise ValueError("changing-state replay must run 24 transitions twice on both models")
        if replay.get("request_order_per_fixture_per_repetition") != ["changed_state"]:
            raise ValueError("changing-state replay must not add baseline duplicate cache probes")
        if replay.get("input_bucket_windows") != {
            "512": [384, 768],
            "1024": [900, 1450],
            "2048": [1728, 2304],
        }:
            raise ValueError("changing-state input-token windows differ from the frozen plan")
        if replay.get("input_bucket_case_counts") != {"512": 8, "1024": 8, "2048": 8}:
            raise ValueError("changing-state replay must freeze eight states per token bucket")
    else:
        raise ValueError("local replay plan has an unsupported suite mode")
    limits = plan.get("limits", {})
    if limits.get("finalization_reserve_seconds") != FINALIZATION_RESERVE_SECONDS:
        raise ValueError("local replay finalization reserve must remain ten minutes")
    max_wall = limits.get("max_wall_seconds_including_preflight_and_finalization")
    if (
        not isinstance(max_wall, int)
        or not FINALIZATION_RESERVE_SECONDS < max_wall <= MAX_WALL_SECONDS
    ):
        raise ValueError("local replay wall budget exceeds the remaining bounded campaign cap")
    if limits.get("request_work_deadline_seconds_from_invocation_start") != (
        max_wall - FINALIZATION_RESERVE_SECONDS
    ):
        raise ValueError("local replay request deadline does not reserve finalization time")
    campaign_budget = plan.get("campaign_budget", {})
    if suite_mode == TRANSITION_SUITE_MODE:
        if campaign_budget.get("previous_run_wall_seconds") != 1696.506838:
            raise ValueError("changing-state replay must bind the completed baseline wall usage")
        if campaign_budget.get("previous_run_summary_path") != BASELINE_SUMMARY_PATH:
            raise ValueError("changing-state replay must bind the measured baseline summary")
        if campaign_budget.get("previous_run_summary_sha256") != BASELINE_SUMMARY_SHA256:
            raise ValueError("changing-state replay baseline summary hash differs")
        if campaign_budget.get("campaign_cap_seconds") != MAX_WALL_SECONDS:
            raise ValueError("changing-state replay campaign cap must remain 90 minutes")
        interrupted_run_seconds = campaign_budget.get("interrupted_run_wall_seconds", 0)
        if (
            not isinstance(interrupted_run_seconds, (int, float))
            or isinstance(interrupted_run_seconds, bool)
            or interrupted_run_seconds < 0
        ):
            raise ValueError("changing-state interrupted-run budget must be nonnegative")
        if interrupted_run_seconds > 0 and not isinstance(
            campaign_budget.get("interrupted_attempt"), dict
        ):
            raise ValueError("interrupted inference time must bind its preserved run artifacts")
        plan_validation_seconds = campaign_budget.get("pre_run_plan_validation_wall_seconds", 0)
        if (
            not isinstance(plan_validation_seconds, (int, float))
            or isinstance(plan_validation_seconds, bool)
            or plan_validation_seconds < 0
        ):
            raise ValueError("pre-run plan validation budget must be nonnegative")
        if campaign_budget.get("max_wall_seconds_including_finalization", max_wall) != max_wall:
            raise ValueError("campaign budget max wall value differs from the active plan limit")
        if campaign_budget.get("request_work_deadline_seconds", max_wall - 600) != (
            max_wall - FINALIZATION_RESERVE_SECONDS
        ):
            raise ValueError("campaign budget request deadline differs from the active plan limit")
        tokenizer_preflight_seconds = campaign_budget.get("tokenizer_preflight_wall_seconds")
        if (
            not isinstance(tokenizer_preflight_seconds, (int, float))
            or tokenizer_preflight_seconds <= 0
        ):
            raise ValueError("changing-state replay must account for tokenizer-only preflights")
        remaining_seconds = (
            MAX_WALL_SECONDS
            - campaign_budget["previous_run_wall_seconds"]
            - tokenizer_preflight_seconds
            - interrupted_run_seconds
            - plan_validation_seconds
        )
        if (
            abs(campaign_budget.get("remaining_seconds_before_new_run", -1) - remaining_seconds)
            > 0.001
        ):
            raise ValueError("changing-state replay remaining budget does not reconcile")
        if max_wall != math.floor(remaining_seconds) or (
            campaign_budget["previous_run_wall_seconds"]
            + tokenizer_preflight_seconds
            + interrupted_run_seconds
            + plan_validation_seconds
            + max_wall
            > MAX_WALL_SECONDS
        ):
            raise ValueError("changing-state replay exceeds remaining campaign wall time")
    if limits.get("minimum_available_ram_bytes") != 4 * 1024 * 1024 * 1024:
        raise ValueError("local replay minimum available RAM safety floor differs")
    if limits.get("candidate_predictor_memory_bytes") != int(1.5 * 1024**3):
        raise ValueError("local replay predictor memory classification threshold differs")


def _sweep_runner_module() -> ModuleType:
    path = ROOT / "scripts/run_sweep_comparison.py"
    spec = importlib.util.spec_from_file_location("run_sweep_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen Sweep prompt implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model_input_prompt(
    sweep: ModuleType, fixture: dict[str, Any], policy: str
) -> tuple[str, str]:
    """Build the frozen publisher prompt and apply only the declared input policy."""
    publisher_prompt = sweep.build_sweep_prompt(fixture)
    if digest_bytes(publisher_prompt.encode("utf-8")) != fixture.get("prompt_sha256"):
        raise ValueError(f"publisher prompt hash differs for {fixture.get('case_id')}")
    if policy == PUBLISHER_PROMPT_EXACT_POLICY:
        return publisher_prompt, digest_bytes(publisher_prompt.encode("utf-8"))
    if policy != PUBLISHER_HEADER_LF_INPUT_POLICY:
        raise ValueError("local replay has an unsupported input prompt policy")
    updated_header = f"<|file_sep|>updated/{fixture.get('file_path')}"
    if publisher_prompt.endswith("\n") or not publisher_prompt.endswith(updated_header):
        raise ValueError("publisher prompt does not end at the expected updated-file header")
    model_input = publisher_prompt + "\n"
    return model_input, digest_bytes(model_input.encode("utf-8"))


def load_and_verify_inputs(
    plan_path: Path, upstream_plan_path: Path, fixtures_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") not in {
        "sweep-local-replay-plan-v1",
        TRANSITION_PLAN_SCHEMA,
    } or plan_digest(plan) != plan.get("plan_sha256"):
        raise ValueError("local replay plan schema or self-hash mismatch")
    validate_plan_policy(plan)
    if plan["replay"].get("suite_mode") == TRANSITION_SUITE_MODE:
        baseline_path = Path(plan["campaign_budget"]["previous_run_summary_path"])
        if digest_file(baseline_path) != BASELINE_SUMMARY_SHA256:
            raise ValueError("completed baseline summary no longer matches its frozen budget hash")
        predecessor_path = plan["inputs"].get("predecessor_plan_path")
        predecessor_sha = plan["inputs"].get("predecessor_plan_file_sha256")
        interrupted_attempt = plan["campaign_budget"].get("interrupted_attempt")
        if predecessor_path is not None:
            predecessor_file = ROOT / predecessor_path
            if not predecessor_file.is_file() or digest_file(predecessor_file) != predecessor_sha:
                raise ValueError("previous local replay plan bytes differ from the frozen identity")
            predecessor_plan = json.loads(predecessor_file.read_text(encoding="utf-8"))
            if (
                plan_digest(predecessor_plan) != predecessor_plan.get("plan_sha256")
                or not isinstance(interrupted_attempt, dict)
                or predecessor_plan.get("plan_sha256") != interrupted_attempt.get("plan_sha256")
            ):
                raise ValueError(
                    "interrupted attempt does not reference its frozen predecessor plan"
                )
        superseded_path = plan["inputs"].get("superseded_unrun_plan_path")
        superseded_sha = plan["inputs"].get("superseded_unrun_plan_file_sha256")
        if superseded_path is not None:
            superseded_file = ROOT / superseded_path
            if not superseded_file.is_file() or digest_file(superseded_file) != superseded_sha:
                raise ValueError("superseded unrun plan bytes differ from the frozen identity")
            superseded_plan = json.loads(superseded_file.read_text(encoding="utf-8"))
            if plan_digest(superseded_plan) != superseded_plan.get("plan_sha256"):
                raise ValueError("superseded unrun plan self-hash is invalid")
        if isinstance(interrupted_attempt, dict):
            _verify_interrupted_attempt(interrupted_attempt, plan)
    upstream = json.loads(upstream_plan_path.read_text(encoding="utf-8"))
    if upstream.get("plan_sha256") != plan["inputs"]["upstream_plan_sha256"]:
        raise ValueError("upstream Sweep plan identity differs")
    if digest_file(upstream_plan_path) != plan["inputs"]["upstream_plan_file_sha256"]:
        raise ValueError("upstream Sweep plan bytes differ")
    if digest_file(fixtures_path) != plan["inputs"]["prompt_bundle_sha256"]:
        raise ValueError("frozen prompt-only bundle identity differs")
    sweep = _sweep_runner_module()
    if plan["replay"].get("suite_mode") == TRANSITION_SUITE_MODE:
        inputs = plan["inputs"]
        archived_commit = inputs.get("prompt_builder_source_commit")
        if archived_commit != PROMPT_BUILDER_ARCHIVE_COMMIT:
            raise ValueError("changing-state prompt builder must use the pinned archive commit")
        archived_bytes = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{archived_commit}:scripts/run_sweep_comparison.py",
            ]
        )
        archived_sha = digest_bytes(archived_bytes)
        if (
            archived_sha != inputs.get("prompt_builder_source_sha256")
            or archived_sha != upstream["code"]["runner_sha256"]
        ):
            raise ValueError("pinned publisher source archive differs from the frozen plan")
        archived_source = archived_bytes.decode("utf-8")
        archived_function_sha = _function_source_sha256(archived_source, "build_sweep_prompt")
        current_function_sha = _function_source_sha256(
            (ROOT / "scripts/run_sweep_comparison.py").read_text(encoding="utf-8"),
            "build_sweep_prompt",
        )
        expected_function_sha = inputs.get("prompt_builder_function_sha256")
        if (
            archived_function_sha != expected_function_sha
            or current_function_sha != expected_function_sha
        ):
            raise ValueError(
                "active publisher prompt function differs from the archived implementation"
            )
    else:
        if (
            digest_file(ROOT / "scripts/run_sweep_comparison.py")
            != plan["inputs"]["prompt_builder_source_sha256"]
        ):
            raise ValueError("publisher prompt builder source differs from the frozen plan")
        runner_sha = digest_file(ROOT / "scripts/run_sweep_comparison.py")
        if runner_sha != upstream["code"]["runner_sha256"]:
            raise ValueError("upstream Sweep plan does not bind the serving prompt implementation")
    if digest_file(Path(__file__).resolve()) != plan["code"]["measurement_script_sha256"]:
        raise ValueError("local measurement script differs from the frozen plan")
    if digest_file(ROOT / "tests/test_measure_sweep_local.py") != plan["code"]["test_sha256"]:
        raise ValueError("local measurement tests differ from the frozen plan")
    rows = read_jsonl(fixtures_path)
    if plan["replay"].get("suite_mode") == TRANSITION_SUITE_MODE:
        validate_trajectory_fixtures(rows, plan["replay"])
    elif tuple(row.get("case_id") for row in rows) != EXPECTED_CASE_IDS:
        raise ValueError("prompt bundle case order or identity differs")
    input_prompt_policy = plan["replay"].get("input_prompt_policy", PUBLISHER_PROMPT_EXACT_POLICY)
    effective_prompt_hashes: dict[str, str] = {}
    for row in rows:
        _, effective_prompt_hash = build_model_input_prompt(sweep, row, input_prompt_policy)
        effective_prompt_hashes[row["case_id"]] = effective_prompt_hash
        if digest_bytes(row["current_content"].encode("utf-8")) != row.get("current_sha256"):
            raise ValueError(f"current source hash differs for {row.get('case_id')}")
        if digest_bytes(row["original_content"].encode("utf-8")) != row.get("original_sha256"):
            raise ValueError(f"original source hash differs for {row.get('case_id')}")
    expected_prompt_hashes = plan["replay"].get("effective_input_prompt_sha256_by_case")
    if expected_prompt_hashes is not None and effective_prompt_hashes != expected_prompt_hashes:
        raise ValueError("effective input prompt hashes differ from the frozen plan")
    expected_token_counts = plan["replay"].get("input_tokens_by_case")
    if expected_token_counts is not None and set(expected_token_counts) != set(
        effective_prompt_hashes
    ):
        raise ValueError("actual tokenizer counts do not cover the frozen fixture states")
    return plan, upstream, rows


def _verify_interrupted_attempt(reference: dict[str, Any], plan: dict[str, Any]) -> None:
    """Verify preserved metadata and charge the interrupted invocation wall time."""
    run_dir = Path(reference.get("run_directory", ""))
    artifacts = {
        "metadata": run_dir / "metadata.json",
        "failure": run_dir / "failure.json",
        "q4_predictions": run_dir / "predictions-q4_k_m.jsonl",
    }
    expected_hashes = reference.get("artifact_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("interrupted attempt lacks its artifact hash ledger")
    for name, path in artifacts.items():
        if not path.is_file() or digest_file(path) != expected_hashes.get(name):
            raise ValueError(f"interrupted attempt artifact hash differs: {name}")
    metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
    failure = json.loads(artifacts["failure"].read_text(encoding="utf-8"))
    prior_plan_sha = reference.get("plan_sha256")
    if (
        metadata.get("plan_sha256") != prior_plan_sha
        or failure.get("plan_sha256") != prior_plan_sha
        or prior_plan_sha != reference.get("prior_plan_sha256")
    ):
        raise ValueError("interrupted attempt metadata is bound to another plan")
    if failure.get("error_type") != "KeyboardInterrupt":
        raise ValueError("interrupted attempt termination reason differs from the plan")
    if failure.get("completed_requests") != reference.get("completed_q4_records"):
        raise ValueError("interrupted attempt request count differs from its failure record")
    if failure.get("completed_requests") != 7 or reference.get("q8_request_count") != 0:
        raise ValueError("interrupted attempt is not the recorded seven-request Q4 subset")
    with artifacts["q4_predictions"].open(encoding="utf-8") as stream:
        if sum(1 for line in stream if line.strip()) != 7:
            raise ValueError("interrupted Q4 record count differs from its artifact")
    if (run_dir / "predictions-q8_0.jsonl").exists():
        raise ValueError("interrupted attempt unexpectedly created a Q8 prediction file")
    started = datetime.fromisoformat(
        metadata["invocation_deadline_started_at_utc"].replace("Z", "+00:00")
    )
    failed = datetime.fromisoformat(failure["failure_at_utc"].replace("Z", "+00:00"))
    measured_seconds = (failed - started).total_seconds()
    if abs(measured_seconds - reference.get("wall_seconds", -1)) > 0.001:
        raise ValueError("interrupted attempt wall time differs from preserved timestamps")
    if (
        abs(measured_seconds - plan["campaign_budget"].get("interrupted_run_wall_seconds", -1))
        > 0.001
    ):
        raise ValueError("interrupted attempt wall time is not charged to the campaign budget")


def verify_static_identity(plan: dict[str, Any], runtime: Path) -> dict[str, Any]:
    runtime = runtime.resolve()
    expected_runtime = Path(plan["runtime"]["source_path"]).resolve()
    if runtime != expected_runtime:
        raise ValueError("runtime path differs from frozen local replay plan")
    revision = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RUNTIME_REVISION or revision != plan["runtime"]["source_revision"]:
        raise ValueError("llama.cpp source revision differs from frozen local replay plan")
    binary = runtime / "build/bin/llama-server"
    binary_hash = digest_file(binary)
    if binary_hash != plan["runtime"]["binary_sha256"]:
        raise ValueError("llama-server binary differs from frozen local replay plan")
    expected_model_files = plan["models"]
    for precision in ("q4_k_m", "q8_0"):
        model = Path(expected_model_files[precision]["path"])
        if model.name != MODEL_FILES[precision]:
            raise ValueError(f"unexpected {precision} artifact filename")
        if model.stat().st_size != MODEL_BYTES[precision]:
            raise ValueError(f"{precision} artifact size differs from frozen identity")
        if digest_file(model) != MODEL_SHA256[precision]:
            raise ValueError(f"{precision} artifact hash differs from frozen identity")
        if expected_model_files[precision]["sha256"] != MODEL_SHA256[precision]:
            raise ValueError(f"{precision} plan hash differs from the pinned model identity")
    return {"runtime_revision": revision, "runtime_binary_sha256": binary_hash}


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_server_argv(binary: Path, model: Path, port: int) -> list[str]:
    return [
        str(binary),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
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
        "--ctx-checkpoints",
        "0",
        "--no-context-shift",
        "--no-repack",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--spec-type",
        "none",
        "--n-gpu-layers",
        "0",
        "--no-warmup",
        "--log-verbosity",
        "4",
        "--no-webui",
    ]


def configure_offline_observability(bundle: Path) -> None:
    os.environ["TABCOMPLETE_OBSERVABILITY_ENABLED"] = "1"
    os.environ["TABCOMPLETE_OBSERVABILITY_MODE"] = "offline"
    os.environ["TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE"] = str(bundle)
    os.environ["TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT"] = "0"


async def _capture_stream_async(
    server_url: str,
    prompt: str,
    *,
    timeout_seconds: float,
    deadline_monotonic: float | None,
    started: float,
) -> dict[str, Any]:
    import httpx

    if timeout_seconds <= 0:
        raise TimeoutError("local model request received an expired deadline")
    loop = asyncio.get_running_loop()
    remaining = (
        min(timeout_seconds, max(0.0, deadline_monotonic - time.monotonic()))
        if deadline_monotonic is not None
        else timeout_seconds
    )
    if remaining <= 0:
        raise TimeoutError("local model request received an expired deadline")
    deadline = loop.time() + remaining
    chunks: list[str] = []
    output_ids: list[int] = []
    final: dict[str, Any] = {}
    first_chunk_ms: float | None = None
    first_token_ms: float | None = None
    response_bytes = 0
    time_to_8_output_token_ids_ms: float | None = None
    time_to_16_output_token_ids_ms: float | None = None
    async with asyncio.timeout_at(deadline):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        ) as client:
            async with client.stream(
                "POST",
                server_url + "/completion",
                json={
                    "prompt": prompt,
                    **REQUEST_SETTINGS,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if time.monotonic() >= (deadline_monotonic or float("inf")):
                        raise TimeoutError("local model request exceeded the frozen wall deadline")
                    if not line.startswith("data: "):
                        continue
                    encoded = line[6:]
                    if encoded == "[DONE]":
                        break
                    event = json.loads(encoded)
                    if not isinstance(event, dict):
                        raise ValueError("runtime emitted a non-object SSE event")
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    content = event.get("content", "")
                    if not isinstance(content, str):
                        raise ValueError("runtime emitted malformed completion content")
                    if content:
                        if first_chunk_ms is None:
                            first_chunk_ms = elapsed_ms
                        chunks.append(content)
                        response_bytes += len(content.encode("utf-8"))
                        if response_bytes > MAX_RESPONSE_BYTES:
                            raise ValueError(
                                "runtime response exceeded the frozen response ceiling"
                            )
                    tokens = event.get("tokens", [])
                    if not isinstance(tokens, list) or any(
                        not isinstance(token, int) or isinstance(token, bool) for token in tokens
                    ):
                        raise ValueError("runtime emitted malformed token IDs")
                    if tokens and first_token_ms is None:
                        first_token_ms = elapsed_ms
                    output_ids.extend(tokens)
                    if len(output_ids) >= 8 and time_to_8_output_token_ids_ms is None:
                        time_to_8_output_token_ids_ms = elapsed_ms
                    if len(output_ids) >= 16 and time_to_16_output_token_ids_ms is None:
                        time_to_16_output_token_ids_ms = elapsed_ms
                    if event.get("stop") is True:
                        final = event
                        break
    completed_ms = (time.perf_counter() - started) * 1000
    text = "".join(chunks)
    stop_type = final.get("stop_type")
    truncated = final.get("truncated")
    terminal = stop_type in {"eos", "word"} and truncated is not True
    capped = truncated is True or (len(output_ids) >= OUTPUT_TOKENS and not terminal)
    return {
        "raw_output": text,
        "output_sha256": digest_bytes(text.encode("utf-8")),
        "output_bytes": len(text.encode("utf-8")),
        "output_token_count": len(output_ids),
        "output_token_ids_sha256": digest_bytes(
            json.dumps(output_ids, separators=(",", ":")).encode("ascii")
        ),
        "finish_reason": stop_type,
        "stopping_word": final.get("stopping_word"),
        "actual_terminal_observed": bool(terminal),
        "hit_output_cap": bool(capped),
        "stream_terminal_event_observed": bool(final),
        "first_content_chunk_ms": first_chunk_ms,
        "first_token_ids_ms": first_token_ms,
        "time_to_8_output_token_ids_ms": time_to_8_output_token_ids_ms,
        "time_to_16_output_token_ids_ms": time_to_16_output_token_ids_ms,
        "completed_response_ms": completed_ms,
        "server_timings": final.get("timings", {}),
        "server_tokens_cached": final.get("tokens_cached"),
        "server_tokens_evaluated": final.get("tokens_evaluated"),
        "cache_counter_evidence": (
            "runtime_counters_present"
            if isinstance(final.get("tokens_cached"), int)
            and isinstance(final.get("tokens_evaluated"), int)
            else "runtime_counters_missing"
        ),
        "cache_prompt_requested": True,
        "token_limit": OUTPUT_TOKENS,
        "raw_response_not_repaired": True,
        "response_utf8_sha256": digest_bytes(text.encode("utf-8")),
    }


def _capture_stream(
    server_url: str,
    prompt: str,
    *,
    timeout_seconds: float,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    return asyncio.run(
        _capture_stream_async(
            server_url,
            prompt,
            timeout_seconds=timeout_seconds,
            deadline_monotonic=deadline_monotonic,
            started=started,
        )
    )


def _parse_pressure(path: Path) -> dict[str, Any] | str:
    if not path.exists():
        return "unavailable"
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, values = line.partition(" ")
        row: dict[str, float | int] = {}
        for pair in values.split():
            name, separator, value = pair.partition("=")
            if not separator:
                continue
            try:
                row[name] = int(value) if name == "total" else float(value)
            except ValueError:
                continue
        if key:
            result[key] = row
    return result


def _proc_kib_fields(path: Path, wanted: set[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        key, separator, raw = line.partition(":")
        fields = raw.strip().split()
        if separator and key in wanted and fields:
            try:
                result[key] = int(fields[0]) * (
                    1024 if len(fields) > 1 and fields[1] == "kB" else 1
                )
            except ValueError:
                continue
    return result


def _proc_cpu_ticks() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()
        if fields[0] != "cpu":
            return None
        values = [int(value) for value in fields[1:]]
    except (OSError, ValueError, IndexError):
        return None
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


def _proc_process_info(pid: int) -> dict[str, Any] | None:
    proc = Path(f"/proc/{pid}")
    try:
        stat = (proc / "stat").read_text()
        close = stat.rfind(")")
        if close < 0:
            return None
        comm = stat[stat.find("(") + 1 : close]
        fields = stat[close + 1 :].split()
        parent_pid = int(fields[1])
        clock_ticks = os.sysconf("SC_CLK_TCK")
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        return {
            "pid": pid,
            "ppid": parent_pid,
            "comm": comm,
            "cpu_user_seconds": user_ticks / clock_ticks,
            "cpu_system_seconds": system_ticks / clock_ticks,
            "cpu_seconds": (user_ticks + system_ticks) / clock_ticks,
            "memory": _proc_memory(pid),
            "start_time_ticks": int(fields[19]),
        }
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        return None


_WORKLOAD_NAME_HINTS = (
    "llama",
    "python",
    "uv",
    "pytest",
    "make",
    "ninja",
    "cmake",
    "gcc",
    "cc1",
    "clang",
    "rustc",
    "cargo",
    "go",
    "node",
    "bun",
    "npm",
    "curl",
    "wget",
    "chrom",
    "firefox",
    "nvim",
    "code",
)


def _host_workload_processes(exclude_pids: set[int]) -> list[dict[str, Any]]:
    rows = []
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        pid = int(proc_path.name)
        if pid in exclude_pids:
            continue
        try:
            comm = (proc_path / "comm").read_text(errors="replace").strip()
        except OSError:
            continue
        if not any(hint in comm.lower() for hint in _WORKLOAD_NAME_HINTS):
            continue
        process = _proc_process_info(pid)
        if process is None:
            continue
        rows.append(
            {
                "pid": pid,
                "process_name": comm,
                "start_time_ticks": process["start_time_ticks"],
                "cpu_seconds": process["cpu_seconds"],
                "rss_bytes": process["memory"].get("VmRSS_bytes"),
            }
        )
    return sorted(rows, key=lambda row: row["cpu_seconds"], reverse=True)[:80]


def _workload_cpu_deltas(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    left = {row["pid"]: row for row in before.get("workload_processes", [])}
    right = {row["pid"]: row for row in after.get("workload_processes", [])}
    deltas = []
    for pid in sorted(left.keys() & right.keys()):
        old = left[pid]
        new = right[pid]
        if old["start_time_ticks"] != new["start_time_ticks"]:
            continue
        delta = new["cpu_seconds"] - old["cpu_seconds"]
        if delta >= 0:
            deltas.append(
                {"pid": pid, "process_name": new["process_name"], "cpu_seconds_delta": delta}
            )
    return sorted(deltas, key=lambda row: row["cpu_seconds_delta"], reverse=True)


def _process_tree(root_pid: int) -> list[dict[str, Any]]:
    result = []
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        info = _proc_process_info(pid)
        if info is None:
            continue
        result.append(info)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = [int(child) for child in children_path.read_text().split()]
        except (OSError, ValueError):
            children = []
        pending.extend(children)
    return result


def _system_cpu_percent(
    previous: tuple[int, int] | None,
) -> tuple[float | None, tuple[int, int] | None]:
    current = _proc_cpu_ticks()
    if current is None or previous is None:
        return None, current
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None, current
    return max(0.0, min(100.0, (total_delta - idle_delta) * 100 / total_delta)), current


def _cpu_frequency_mhz() -> list[float]:
    values: list[float] = []
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        key, separator, raw = line.partition(":")
        if separator and key.strip() == "cpu MHz":
            try:
                values.append(float(raw.strip()))
            except ValueError:
                continue
    return values


def _temperature_readings() -> tuple[dict[str, float], dict[str, float]]:
    cpu: dict[str, float] = {}
    other: dict[str, float] = {}

    def save(name: str, value: float) -> None:
        destination = (
            cpu
            if any(
                hint in name.lower()
                for hint in ("cpu", "package", "pkg", "tctl", "tdie", "coretemp")
            )
            else other
        )
        destination[name] = value

    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = int(zone.read_text().strip())
            zone_name = zone.parent / "type"
            name = zone_name.read_text().strip() if zone_name.exists() else zone.parent.name
            save(f"thermal:{name}", raw / 1000)
        except (OSError, ValueError):
            continue
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            hwmon_name = (hwmon / "name").read_text().strip()
        except OSError:
            hwmon_name = hwmon.name
        for input_path in hwmon.glob("temp*_input"):
            try:
                raw = int(input_path.read_text().strip())
                label_path = input_path.with_name(input_path.name.replace("_input", "_label"))
                label = label_path.read_text().strip() if label_path.exists() else input_path.stem
                save(f"hwmon:{hwmon_name}:{label}", raw / 1000)
            except (OSError, ValueError):
                continue
    return cpu, other


def _cpu_temperatures() -> dict[str, float] | None:
    readings, _ = _temperature_readings()
    return readings or None


def _background_model_process(pid: int) -> dict[str, Any] | None:
    try:
        comm = Path(f"/proc/{pid}/comm").read_text(errors="replace").strip()
    except OSError:
        return None
    if "llama" not in comm.lower():
        return None
    process = _proc_process_info(pid)
    if process is None:
        return None
    try:
        command = (
            (Path(f"/proc/{pid}/cmdline").read_bytes())
            .decode("utf-8", errors="replace")
            .split("\0")
        )
    except OSError:
        command = []
    model_value = None
    for index, argument in enumerate(command[:-1]):
        if argument in {"-m", "--model"}:
            model_value = command[index + 1]
            break
    basename = Path(model_value).name if model_value else None
    role = (
        "background_q25_predictor"
        if basename and ("q25" in basename.lower() or "qwen2.5" in basename.lower())
        else "other_llama_server"
    )
    return {
        "pid": pid,
        "process_name": process["comm"],
        "role": role,
        "model_filename": basename,
        "cpu_user_seconds": process["cpu_user_seconds"],
        "cpu_system_seconds": process["cpu_system_seconds"],
        "memory": process["memory"],
    }


def _pressure_total(pressure: Any) -> dict[str, int]:
    if not isinstance(pressure, dict):
        return {}
    totals: dict[str, int] = {}
    for level, values in pressure.items():
        if isinstance(values, dict) and isinstance(values.get("total"), int):
            totals[level] = values["total"]
    return totals


def _proc_memory(pid: int) -> dict[str, int]:
    path = Path(f"/proc/{pid}")
    wanted = {
        "VmRSS",
        "VmHWM",
        "VmSwap",
        "RssAnon",
        "RssFile",
        "Pss",
        "Pss_Anon",
        "Pss_File",
        "Anonymous",
        "Swap",
    }
    values: dict[str, int] = {}
    for filename in ("status", "smaps_rollup"):
        file_path = path / filename
        if not file_path.exists():
            continue
        for line in file_path.read_text(errors="replace").splitlines():
            key, separator, raw = line.partition(":")
            if not separator or key not in wanted:
                continue
            fields = raw.strip().split()
            if fields:
                try:
                    values[key + "_bytes"] = int(fields[0]) * (1024 if len(fields) > 1 else 1)
                except ValueError:
                    pass
    return values


def _proc_pages() -> dict[str, int]:
    wanted = {"pswpin", "pswpout"}
    result: dict[str, int] = {}
    path = Path("/proc/vmstat")
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


def _read_model_process(pid: int) -> dict[str, Any] | None:
    return _background_model_process(pid)


def _find_q25_background_processes(known_pid: int) -> list[dict[str, Any]]:
    candidates: set[int] = {known_pid}
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        pid = int(proc_path.name)
        info = _background_model_process(pid)
        if info and info["role"] == "background_q25_predictor":
            candidates.add(pid)
    found = []
    for pid in sorted(candidates):
        row = _read_model_process(pid)
        if row and row["role"] == "background_q25_predictor":
            found.append(row)
    return found


def _host_snapshot(
    background_pids: Sequence[int], exclude_process_pids: Sequence[int] = ()
) -> dict[str, Any]:
    memory = _proc_kib_fields(
        Path("/proc/meminfo"), {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    )
    total_cpu = _proc_cpu_ticks()
    try:
        load = os.getloadavg()
    except OSError:
        load = None
    swap_memory = {
        "total_bytes": memory.get("SwapTotal", 0),
        "used_bytes": memory.get("SwapTotal", 0) - memory.get("SwapFree", 0),
    }
    processes = [_read_model_process(pid) for pid in background_pids]
    cpu_temperatures, other_temperatures = _temperature_readings()
    excluded = set(background_pids) | set(exclude_process_pids) | {os.getpid()}
    return {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "load_average_1_5_15": list(load) if load is not None else None,
        "cpu_frequency_mhz_per_core": _cpu_frequency_mhz(),
        "cpu_temperature_celsius": cpu_temperatures or None,
        "other_temperature_sensors_celsius": other_temperatures,
        "ram_total_bytes": memory.get("MemTotal"),
        "ram_available_bytes": memory.get("MemAvailable"),
        "swap_occupancy": swap_memory,
        "system_cpu_ticks": total_cpu,
        "swap_activity_pages": _proc_pages(),
        "memory_pressure": _parse_pressure(Path("/proc/pressure/memory")),
        "io_pressure": _parse_pressure(Path("/proc/pressure/io")),
        "background_model_processes": [process for process in processes if process],
        "workload_processes": _host_workload_processes(excluded),
    }


def _cpu_seconds(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return float(row["cpu_user_seconds"]) + float(row["cpu_system_seconds"])


class MemorySampler:
    def __init__(self, pid: int, background_pids: Sequence[int], interval_seconds: float = 0.5):
        self.pid = pid
        self.background_pids = tuple(background_pids)
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.sample_lock = threading.Lock()
        self.previous_cpu_ticks: tuple[int, int] | None = None
        self.thread = threading.Thread(
            target=self._run, name="sweep-local-memory-sampler", daemon=True
        )

    def start(self) -> None:
        self.sample()
        self.thread.start()

    def sample(self) -> int:
        with self.sample_lock:
            tree = _process_tree(self.pid)
            alive = bool(tree and tree[0]["pid"] == self.pid)
            candidate_cpu = sum(float(row["cpu_seconds"]) for row in tree) if tree else None
            memory_keys = (
                "VmRSS_bytes",
                "VmHWM_bytes",
                "Pss_bytes",
                "Anonymous_bytes",
                "Pss_Anon_bytes",
                "Pss_File_bytes",
                "RssAnon_bytes",
                "RssFile_bytes",
                "VmSwap_bytes",
                "Swap_bytes",
            )
            service_memory: dict[str, int] = {}
            for key in memory_keys:
                observed = [row["memory"][key] for row in tree if key in row["memory"]]
                if observed:
                    service_memory[key] = sum(observed)
            cpu_percent, self.previous_cpu_ticks = _system_cpu_percent(self.previous_cpu_ticks)
            backgrounds = [_read_model_process(pid) for pid in self.background_pids]
            sample = {
                "monotonic_seconds": time.monotonic(),
                "candidate_pid_alive": alive,
                "candidate_cpu_seconds": candidate_cpu,
                "system_cpu_percent": cpu_percent,
                "cpu_frequency_mhz_per_core": _cpu_frequency_mhz(),
                "cpu_temperature_celsius": _cpu_temperatures(),
                "candidate_service_memory": service_memory,
                "background_models": [row for row in backgrounds if row],
                "swap_activity_pages": _proc_pages(),
                "memory_pressure": _parse_pressure(Path("/proc/pressure/memory")),
                "io_pressure": _parse_pressure(Path("/proc/pressure/io")),
                "memory_available_bytes": _proc_kib_fields(
                    Path("/proc/meminfo"), {"MemAvailable"}
                ).get("MemAvailable"),
            }
            self.samples.append(sample)
            return len(self.samples) - 1

    def _run(self) -> None:
        while not self.stop.wait(self.interval_seconds):
            self.sample()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()

    def report(self) -> dict[str, Any]:
        if not self.samples:
            return {"sample_count": 0, "peak": None, "post_request_retained": None}
        memory_keys = {key for sample in self.samples for key in sample["candidate_service_memory"]}
        peaks = {
            key: max(
                (sample["candidate_service_memory"].get(key, 0) for sample in self.samples),
                default=0,
            )
            for key in sorted(memory_keys)
        }
        q25_samples: list[tuple[float, float]] = []
        for sample in self.samples:
            for row in sample["background_models"]:
                if row["role"] == "background_q25_predictor":
                    seconds = _cpu_seconds(row)
                    if seconds is not None:
                        q25_samples.append((sample["monotonic_seconds"], seconds))
        q25_cpu_delta = (
            max(0.0, q25_samples[-1][1] - q25_samples[0][1]) if len(q25_samples) >= 2 else None
        )
        cpu_temperature_peaks: dict[str, float] = {}
        cpu_frequency_values: list[float] = []
        for sample in self.samples:
            temperatures = sample.get("cpu_temperature_celsius")
            if isinstance(temperatures, dict):
                for name, value in temperatures.items():
                    cpu_temperature_peaks[name] = max(
                        cpu_temperature_peaks.get(name, float("-inf")), float(value)
                    )
            frequencies = sample.get("cpu_frequency_mhz_per_core")
            if isinstance(frequencies, list):
                cpu_frequency_values.extend(float(value) for value in frequencies)
        return {
            "sample_period_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "peak_candidate_service_bytes": peaks,
            "post_request_retained": self.samples[-1]["candidate_service_memory"],
            "candidate_cpu_seconds_delta": (
                max(
                    0.0,
                    float(self.samples[-1]["candidate_cpu_seconds"])
                    - float(self.samples[0]["candidate_cpu_seconds"]),
                )
                if self.samples[0]["candidate_cpu_seconds"] is not None
                and self.samples[-1]["candidate_cpu_seconds"] is not None
                else None
            ),
            "system_cpu_percent_max": max(
                (
                    float(sample["system_cpu_percent"])
                    for sample in self.samples
                    if sample["system_cpu_percent"] is not None
                ),
                default=None,
            ),
            "system_cpu_percent_mean": _mean(
                [
                    float(sample["system_cpu_percent"])
                    for sample in self.samples
                    if sample["system_cpu_percent"] is not None
                ]
            ),
            "cpu_temperature_peak_celsius": cpu_temperature_peaks or None,
            "cpu_frequency_mhz_min": min(cpu_frequency_values, default=None),
            "cpu_frequency_mhz_mean": _mean(cpu_frequency_values),
            "cpu_frequency_mhz_max": max(cpu_frequency_values, default=None),
            "memory_available_bytes_min": min(
                (
                    int(sample["memory_available_bytes"])
                    for sample in self.samples
                    if isinstance(sample["memory_available_bytes"], int)
                ),
                default=None,
            ),
            "memory_pressure_start": self.samples[0]["memory_pressure"],
            "memory_pressure_end": self.samples[-1]["memory_pressure"],
            "io_pressure_start": self.samples[0]["io_pressure"],
            "io_pressure_end": self.samples[-1]["io_pressure"],
            "swap_activity_start_pages": self.samples[0]["swap_activity_pages"],
            "swap_activity_end_pages": self.samples[-1]["swap_activity_pages"],
            "background_q25_cpu_seconds_delta": q25_cpu_delta,
            "background_q25_compute_observed": (q25_cpu_delta is not None and q25_cpu_delta > 0.01),
        }

    def window_summary(self, start_index: int, end_index: int) -> dict[str, Any]:
        selected = self.samples[max(0, start_index) : min(len(self.samples), end_index + 1)]
        if not selected:
            return {"memory_samples": 0}
        keys = {key for sample in selected for key in sample["candidate_service_memory"]}
        peak = {
            key: max(sample["candidate_service_memory"].get(key, 0) for sample in selected)
            for key in sorted(keys)
        }
        q25_times = []
        for sample in selected:
            for row in sample["background_models"]:
                if row["role"] == "background_q25_predictor":
                    seconds = _cpu_seconds(row)
                    if seconds is not None:
                        q25_times.append((sample["monotonic_seconds"], seconds))
        delta = max(0.0, q25_times[-1][1] - q25_times[0][1]) if len(q25_times) >= 2 else None
        cpu_values = [
            float(sample["system_cpu_percent"])
            for sample in selected
            if sample["system_cpu_percent"] is not None
        ]
        cpu_temperature_peaks: dict[str, float] = {}
        cpu_frequency_values: list[float] = []
        for sample in selected:
            temperatures = sample.get("cpu_temperature_celsius")
            if isinstance(temperatures, dict):
                for name, value in temperatures.items():
                    cpu_temperature_peaks[name] = max(
                        cpu_temperature_peaks.get(name, float("-inf")), float(value)
                    )
            frequencies = sample.get("cpu_frequency_mhz_per_core")
            if isinstance(frequencies, list):
                cpu_frequency_values.extend(float(value) for value in frequencies)
        first_swap = selected[0]["swap_activity_pages"]
        last_swap = selected[-1]["swap_activity_pages"]
        return {
            "memory_samples": len(selected),
            "candidate_peak_memory": peak,
            "candidate_memory_before": selected[0]["candidate_service_memory"],
            "candidate_memory_after": selected[-1]["candidate_service_memory"],
            "q25_cpu_seconds_delta": delta,
            "q25_compute_overlapped": delta is not None and delta > 0.01,
            "system_cpu_percent_max": max(cpu_values, default=None),
            "system_cpu_percent_mean": _mean(cpu_values),
            "cpu_temperature_peak_celsius": cpu_temperature_peaks or None,
            "cpu_frequency_mhz_min": min(cpu_frequency_values, default=None),
            "cpu_frequency_mhz_mean": _mean(cpu_frequency_values),
            "cpu_frequency_mhz_max": max(cpu_frequency_values, default=None),
            "memory_pressure_start": selected[0]["memory_pressure"],
            "memory_pressure_end": selected[-1]["memory_pressure"],
            "swap_activity_delta_pages": {
                key: last_swap.get(key, 0) - first_swap.get(key, 0)
                for key in set(first_swap) | set(last_swap)
            },
            "minimum_available_ram_bytes": min(
                (
                    int(sample["memory_available_bytes"])
                    for sample in selected
                    if isinstance(sample["memory_available_bytes"], int)
                ),
                default=None,
            ),
        }


def _tokenize(
    server_url: str, prompt: str, *, timeout_seconds: float = 60.0
) -> tuple[list[int], float]:
    import httpx

    started = time.perf_counter()
    response = httpx.post(
        server_url + "/tokenize",
        json={"content": prompt, "add_special": True, "parse_special": True},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in tokens
    ):
        raise ValueError("runtime tokenizer endpoint returned invalid token IDs")
    return tokens, (time.perf_counter() - started) * 1000


def _common_token_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    shared = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        shared += 1
    return shared


def _wait_server(process: subprocess.Popen[str], url: str, timeout_seconds: float = 300.0) -> float:
    import httpx

    if timeout_seconds <= 0:
        raise TimeoutError("local replay deadline expired before server health")
    started = time.perf_counter()
    deadline = started + timeout_seconds
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError("local llama-server exited before health")
        try:
            response = httpx.get(url + "/health", timeout=2)
            if response.status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError("local llama-server did not become ready before the fixed deadline")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("local llama-server did not exit after termination") from exc


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def nearest_rank(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["request_kind"], row["precision"])
        groups.setdefault(key, []).append(row)
    summary: dict[str, Any] = {}
    for (kind, precision), group in sorted(groups.items()):
        latency = [float(row["completed_response_ms"]) for row in group]
        cached = [
            int(row["server_tokens_cached"])
            for row in group
            if isinstance(row.get("server_tokens_cached"), int)
        ]
        evaluated = [
            int(row["server_tokens_evaluated"])
            for row in group
            if isinstance(row.get("server_tokens_evaluated"), int)
        ]
        summary[f"{precision}/{kind}"] = {
            "requests": len(group),
            "latency_ms_median": statistics_median(latency),
            "latency_ms_p95_nearest_rank": nearest_rank(latency, 0.95),
            "completed_file_output_count": sum(
                row.get("completed_file_output") is True for row in group
            ),
            "completed_file_output_latency_ms_median": statistics_median(
                [
                    float(row["completed_file_output_latency_ms"])
                    for row in group
                    if isinstance(row.get("completed_file_output_latency_ms"), (int, float))
                ]
            ),
            "completed_file_output_latency_ms_p95_nearest_rank": nearest_rank(
                [
                    float(row["completed_file_output_latency_ms"])
                    for row in group
                    if isinstance(row.get("completed_file_output_latency_ms"), (int, float))
                ],
                0.95,
            ),
            "valid_canonical_action_count": sum(
                row.get("valid_canonical_action") is True for row in group
            ),
            "valid_canonical_action_latency_ms_median": statistics_median(
                [
                    float(row["valid_canonical_action_latency_ms"])
                    for row in group
                    if isinstance(row.get("valid_canonical_action_latency_ms"), (int, float))
                ]
            ),
            "valid_canonical_action_latency_ms_p95_nearest_rank": nearest_rank(
                [
                    float(row["valid_canonical_action_latency_ms"])
                    for row in group
                    if isinstance(row.get("valid_canonical_action_latency_ms"), (int, float))
                ],
                0.95,
            ),
            "actual_terminal_count": sum(bool(row["actual_terminal_observed"]) for row in group),
            "cap_count": sum(bool(row["hit_output_cap"]) for row in group),
            "same_prompt_output_hash_equal_to_primary_count": sum(
                row.get("repeat_output_matches_primary") is True for row in group
            ),
            "backend_tokens_cached_median": statistics_median([float(v) for v in cached]),
            "backend_tokens_evaluated_median": statistics_median([float(v) for v in evaluated]),
            "backend_cache_counters_missing": len(group) - min(len(cached), len(evaluated)),
        }
    return summary


def response_action_measurements(
    result: dict[str, Any], output_file_mapping: dict[str, Any]
) -> dict[str, Any]:
    """Keep terminal full-file latency separate from a range-valid edit action."""
    completed = bool(result.get("actual_terminal_observed") and not result.get("hit_output_cap"))
    action = output_file_mapping.get("action")
    valid = bool(
        completed
        and output_file_mapping.get("mapping") == "within_editable_range"
        and action in {"no_edit", "replace"}
    )
    canonical_action = None
    if valid:
        canonical_action = {"action": action}
        if action == "replace":
            text = output_file_mapping.get("replacement")
            if not isinstance(text, str):
                valid = False
                canonical_action = None
            else:
                canonical_action["text"] = text
    request_ms = result.get("client_request_ms")
    return {
        "completed_file_output": completed,
        "completed_file_output_latency_ms": request_ms if completed else None,
        "valid_canonical_action": valid,
        "canonical_action": canonical_action if valid else None,
        "valid_canonical_action_latency_ms": request_ms if valid else None,
    }


def statistics_median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _server_run(
    *,
    precision: str,
    plan: dict[str, Any],
    runtime: Path,
    fixtures: list[dict[str, Any]],
    run_dir: Path,
    records: list[dict[str, Any]],
    run_context: Any,
    campaign_deadline: float,
) -> dict[str, Any]:

    from tinycomplete.observability.context import RunContext
    from tinycomplete.observability.hooks import model_metrics, usage_metrics
    from tinycomplete.observability.spans import operation

    model_info = plan["models"][precision]
    model = Path(model_info["path"])
    port = _available_port()
    url = f"http://127.0.0.1:{port}"
    binary = runtime / "build/bin/llama-server"
    argv = build_server_argv(binary, model, port)
    log_path = run_dir / f"server-{precision}.log"
    background = _find_q25_background_processes(3343901)
    background_pids = [row["pid"] for row in background]
    before_host = _host_snapshot(background_pids)
    remaining = campaign_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("local replay finalization reserve reached before model startup")
    started = time.perf_counter()
    model_start_wall = datetime.now(UTC).isoformat()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        sampler: MemorySampler | None = None
        try:
            sampler = MemorySampler(process.pid, background_pids)
            sampler.start()
            with operation(
                "model.load",
                attributes={
                    "gen_ai.request.model": "sweep-next-edit-1.5B",
                    "tabcomplete.quantization": model_info["quantization"],
                    "tabcomplete.backend": "CPU",
                },
            ) as load_span:
                _wait_server(process, url, min(300.0, campaign_deadline - time.monotonic()))
                load_seconds = time.perf_counter() - started
                load_span.set_attribute("tabcomplete.timing.total_ms", load_seconds * 1000)
                load_span.set_attribute("tabcomplete.timing.kind", "load_to_health")
            process_memory_at_health = _proc_memory(process.pid)
            token_rows = []
            prompts = []
            sweep = _sweep_runner_module()
            input_prompt_policy = plan["replay"].get(
                "input_prompt_policy", PUBLISHER_PROMPT_EXACT_POLICY
            )
            for fixture in fixtures:
                tokenization_remaining = campaign_deadline - time.monotonic()
                if tokenization_remaining <= 0:
                    raise TimeoutError(
                        "local replay finalization reserve reached during tokenization"
                    )
                prompt, effective_input_prompt_sha256 = build_model_input_prompt(
                    sweep, fixture, input_prompt_policy
                )
                token_ids, tokenize_ms = _tokenize(
                    url, prompt, timeout_seconds=min(60.0, tokenization_remaining)
                )
                input_tokens = len(token_ids)
                if input_tokens + OUTPUT_TOKENS > CONTEXT_TOKENS:
                    raise ValueError(
                        f"frozen case {fixture['case_id']} exceeds input-plus-output context"
                    )
                token_rows.append(
                    {
                        "case_id": fixture["case_id"],
                        "prompt_sha256": fixture["prompt_sha256"],
                        "effective_input_prompt_sha256": effective_input_prompt_sha256,
                        "input_tokens": input_tokens,
                        "token_ids": token_ids,
                        "input_token_ids_sha256": digest_bytes(
                            json.dumps(token_ids, separators=(",", ":")).encode("ascii")
                        ),
                        "tokenization_ms": tokenize_ms,
                    }
                )
                prompts.append(prompt)
            if len(token_rows) != len(fixtures):
                raise ValueError("local replay tokenizer did not cover every frozen source state")
            if plan["replay"].get("suite_mode") == TRANSITION_SUITE_MODE:
                validate_input_token_buckets(fixtures, token_rows, plan["replay"])
            process_memory_before = _proc_memory(process.pid)
            prediction_path = run_dir / f"predictions-{precision}.jsonl"
            primary_output_hashes: dict[tuple[int, str], str] = {}
            request_kinds = plan["replay"]["request_order_per_fixture_per_repetition"]
            repetitions = int(plan["replay"]["repetitions"])
            suite_version = plan["replay"].get("suite_version", "sweep-local-replay-r1")
            previous_request_token_ids: list[int] | None = None
            previous_request_case_id: str | None = None
            with prediction_path.open("x", encoding="utf-8") as output:
                for repetition in range(repetitions):
                    for fixture, prompt, token_row in zip(
                        fixtures, prompts, token_rows, strict=True
                    ):
                        if time.monotonic() >= campaign_deadline:
                            raise TimeoutError("local replay reached its frozen request deadline")
                        input_tokens = token_row["input_tokens"]
                        input_token_ids = token_row["token_ids"]
                        common_prefix_previous_request = (
                            _common_token_prefix(previous_request_token_ids, input_token_ids)
                            if previous_request_token_ids is not None
                            else None
                        )
                        for kind in request_kinds:
                            sample_start = sampler.sample()
                            request_context = (
                                run_context
                                or RunContext.new(
                                    campaign_id=plan.get("observability", {}).get(
                                        "campaign_id", OBSERVABILITY_CAMPAIGN
                                    )
                                )
                            ).for_case(fixture["case_id"])
                            request_started_utc = datetime.now(UTC).isoformat()
                            metrics_labels = {
                                "task": "next_edit",
                                "backend": "llama.cpp-cpu",
                                "model_alias": "sweep-next-edit-1.5B",
                                "quantization": "Q4_K_M" if precision == "q4_k_m" else "Q8_0",
                                "context_size_bucket": "le_4096",
                                "device_type": "cpu",
                            }
                            gen_started = time.perf_counter()
                            request_remaining = campaign_deadline - time.monotonic()
                            if request_remaining <= 0:
                                raise TimeoutError(
                                    "local replay finalization reserve reached before request"
                                )
                            with request_context.activate():
                                with operation(
                                    "request.start",
                                    attributes={
                                        "gen_ai.request.model": "sweep-next-edit-1.5B",
                                        "gen_ai.request.max_tokens": OUTPUT_TOKENS,
                                        "tabcomplete.quantization": metrics_labels["quantization"],
                                        "tabcomplete.cache.condition": kind,
                                    },
                                ):
                                    pass
                                with operation(
                                    "model.generate",
                                    attributes={
                                        "gen_ai.operation.name": "text_completion",
                                        "gen_ai.provider.name": "llama.cpp-native",
                                        "gen_ai.request.model": "sweep-next-edit-1.5B",
                                        "gen_ai.request.max_tokens": OUTPUT_TOKENS,
                                        "tabcomplete.backend": "CPU",
                                        "tabcomplete.quantization": metrics_labels["quantization"],
                                        "tabcomplete.request.context_tokens": input_tokens,
                                        "tabcomplete.cache.condition": kind,
                                        "tabcomplete.suite_version": suite_version,
                                    },
                                ) as span:
                                    with model_metrics(metrics_labels):
                                        result = _capture_stream(
                                            url,
                                            prompt,
                                            timeout_seconds=min(600.0, request_remaining),
                                            deadline_monotonic=campaign_deadline,
                                        )
                                    client_ms = (time.perf_counter() - gen_started) * 1000
                                    result["client_request_ms"] = client_ms
                                    span.set_attribute("tabcomplete.timing.total_ms", client_ms)
                                    span.set_attribute(
                                        "tabcomplete.timing.kind", "client_end_to_end"
                                    )
                                    span.set_attribute(
                                        "tabcomplete.output.truncated", result["hit_output_cap"]
                                    )
                                    span.set_attribute(
                                        "tabcomplete.outcome",
                                        "completed"
                                        if result["actual_terminal_observed"]
                                        else "unterminated",
                                    )
                                    span.set_attribute(
                                        "tabcomplete.output.bytes", result["output_bytes"]
                                    )
                                    if result["finish_reason"] is not None:
                                        span.set_attribute(
                                            "gen_ai.response.finish_reasons",
                                            [result["finish_reason"]],
                                        )
                                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                                    span.set_attribute(
                                        "gen_ai.usage.output_tokens", result["output_token_count"]
                                    )
                                    if result["first_token_ids_ms"] is not None:
                                        span.set_attribute(
                                            "tabcomplete.timing.first_output_ms",
                                            result["first_token_ids_ms"],
                                        )
                                    if isinstance(result["server_tokens_cached"], int):
                                        span.set_attribute(
                                            "tabcomplete.cache.tokens_cached",
                                            result["server_tokens_cached"],
                                        )
                                    if isinstance(result["server_tokens_evaluated"], int):
                                        span.set_attribute(
                                            "tabcomplete.cache.tokens_evaluated",
                                            result["server_tokens_evaluated"],
                                        )
                                    from types import SimpleNamespace

                                    usage_metrics(
                                        SimpleNamespace(
                                            input_tokens=input_tokens,
                                            tokens=result["output_token_count"],
                                            cache_tokens=result["server_tokens_cached"],
                                            output_tokens_known=True,
                                            first_output_ms=result["first_token_ids_ms"],
                                        ),
                                        metrics_labels,
                                    )
                            sample_end = sampler.sample()
                            request_index = len(records)
                            output_file_mapping = (
                                sweep.map_full_file(
                                    fixture["current_content"],
                                    result["raw_output"],
                                    fixture["editable_start_byte"],
                                    fixture["editable_end_byte"],
                                )
                                if result["actual_terminal_observed"]
                                else {
                                    "mapping": "incomplete_or_unterminated_output_not_scored",
                                    "out_of_range": None,
                                }
                            )
                            action_measurements = response_action_measurements(
                                result, output_file_mapping
                            )
                            result.update(
                                schema=(
                                    "sweep-local-changing-state-record-v1"
                                    if plan["replay"].get("suite_mode") == TRANSITION_SUITE_MODE
                                    else "sweep-local-replay-record-v1"
                                ),
                                plan_sha256=plan["plan_sha256"],
                                precision=precision,
                                model_sha256=model_info["sha256"],
                                case_id=fixture["case_id"],
                                state_id=fixture.get("state_id"),
                                trajectory_id=fixture.get("trajectory_id"),
                                transition_type=fixture.get("transition_type"),
                                from_state_id=fixture.get("from_state_id"),
                                nominal_context_bucket=fixture.get("nominal_context_bucket"),
                                repetition=repetition,
                                request_kind=kind,
                                prompt_sha256=fixture["prompt_sha256"],
                                publisher_prompt_sha256=fixture["prompt_sha256"],
                                effective_input_prompt_sha256=token_row[
                                    "effective_input_prompt_sha256"
                                ],
                                input_tokens=input_tokens,
                                input_token_ids_sha256=token_row["input_token_ids_sha256"],
                                previous_request_case_id=previous_request_case_id,
                                common_prefix_tokens_with_previous_request=(
                                    common_prefix_previous_request
                                ),
                                input_plus_output_ceiling=input_tokens + OUTPUT_TOKENS,
                                tokenization_ms=token_row["tokenization_ms"],
                                **action_measurements,
                                editable_range={
                                    "start_byte": fixture["editable_start_byte"],
                                    "end_byte": fixture["editable_end_byte"],
                                },
                                output_file_mapping=output_file_mapping,
                                request_started_at_utc=request_started_utc,
                                measurement_order=request_index,
                                measurement_memory=sampler.window_summary(sample_start, sample_end),
                            )
                            if kind == "changed_state":
                                primary_output_hashes[(repetition, fixture["case_id"])] = result[
                                    "output_sha256"
                                ]
                            elif kind == "immediate_same_prompt_repeat":
                                result["repeat_output_matches_primary"] = (
                                    result["output_sha256"]
                                    == primary_output_hashes[(repetition, fixture["case_id"])]
                                )
                            encoded_result = (
                                json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n"
                            )
                            output.write(encoded_result)
                            output.flush()
                            records.append(result)
                            previous_request_token_ids = input_token_ids
                            previous_request_case_id = fixture["case_id"]
                            if (
                                sum(
                                    len(row.get("raw_output", "").encode("utf-8"))
                                    for row in records
                                )
                                > MAX_TOTAL_RESULT_BYTES
                            ):
                                raise RuntimeError(
                                    "local replay result exceeded its fixed storage cap"
                                )
            sampler.close()
            sampler_report = sampler.report()
            after_host = _host_snapshot(
                [row["pid"] for row in _find_q25_background_processes(3343901)],
                [process.pid],
            )
            server_log = log_path.read_text(errors="replace")
            evidence_lines = [
                line.strip()
                for line in server_log.splitlines()
                if "offload" in line.lower() or "cuda" in line.lower() or "repack" in line.lower()
            ][-30:]
            if "--n-gpu-layers" not in argv or argv[argv.index("--n-gpu-layers") + 1] != "0":
                raise AssertionError("local replay failed its CPU-only backend flag guard")
            return {
                "precision": precision,
                "model_sha256": model_info["sha256"],
                "server_pid": process.pid,
                "server_argv": argv,
                "server_binary_sha256": plan["runtime"]["binary_sha256"],
                "runtime_revision": plan["runtime"]["source_revision"],
                "backend": "CPU requested with --n-gpu-layers 0; verify server log evidence",
                "runtime_log_backend_evidence": evidence_lines,
                "model_load_to_health_seconds": load_seconds,
                "model_started_at_utc": model_start_wall,
                "model_completed_at_utc": datetime.now(UTC).isoformat(),
                "process_memory_at_health": process_memory_at_health,
                "process_memory_before_first_request": process_memory_before,
                "memory_sampler": sampler_report,
                "host_before": before_host,
                "host_after": after_host,
                "background_workload_cpu_deltas": _workload_cpu_deltas(before_host, after_host),
                "swap_activity_delta_pages": {
                    key: after_host["swap_activity_pages"].get(key, 0)
                    - before_host["swap_activity_pages"].get(key, 0)
                    for key in set(before_host["swap_activity_pages"])
                    | set(after_host["swap_activity_pages"])
                },
                "memory_pressure_totals_delta_microseconds": {
                    key: end - start
                    for key, start in _pressure_total(before_host["memory_pressure"]).items()
                    for end in [_pressure_total(after_host["memory_pressure"]).get(key, start)]
                },
                "io_pressure_totals_delta_microseconds": {
                    key: end - start
                    for key, start in _pressure_total(before_host["io_pressure"]).items()
                    for end in [_pressure_total(after_host["io_pressure"]).get(key, start)]
                },
                "server_exit_verified": False,
                "case_count": len(fixtures),
                "repetitions": repetitions,
                "request_count": len(fixtures) * repetitions * len(request_kinds),
                "request_kind_counts": {
                    kind: len(fixtures) * repetitions * request_kinds.count(kind)
                    for kind in set(request_kinds)
                },
                "same_prompt_repeat_requests": (
                    len(fixtures)
                    * repetitions
                    * request_kinds.count("immediate_same_prompt_repeat")
                ),
                "suite_mode": plan["replay"].get("suite_mode", "fixed_snapshot"),
                "quality_evidence": False,
                "process_tree_memory_included": True,
                "background_q25_processes_before": before_host["background_model_processes"],
                "background_q25_processes_after": after_host["background_model_processes"],
            }
        finally:
            if sampler is not None:
                sampler.close()
            _stop_process(process)
            if process.poll() is None:
                raise RuntimeError("candidate llama-server process remains after cleanup")


def execute(args: argparse.Namespace) -> Path:
    invocation_started_at_utc = datetime.now(UTC)
    invocation_started = time.monotonic()
    plan, upstream, fixtures = load_and_verify_inputs(args.plan, args.upstream_plan, args.fixtures)
    runtime = args.runtime.resolve()
    identity = verify_static_identity(plan, runtime)
    if args.execute is not True:
        print(
            json.dumps(
                {
                    "state": "preflight_passed_no_model_loaded",
                    "plan_sha256": plan["plan_sha256"],
                    "upstream_plan_sha256": upstream["plan_sha256"],
                    "fixture_count": len(fixtures),
                    "runtime_identity": identity,
                    "model_files_verified": True,
                    "inference_started": False,
                },
                sort_keys=True,
            )
        )
        return Path()
    max_wall_seconds = min(
        int(plan["limits"]["max_wall_seconds_including_preflight_and_finalization"]),
        MAX_WALL_SECONDS,
    )
    finalization_reserve_seconds = int(plan["limits"]["finalization_reserve_seconds"])
    if finalization_reserve_seconds < 0 or max_wall_seconds <= finalization_reserve_seconds:
        raise ValueError("local replay wall deadline leaves no valid finalization interval")
    campaign_deadline = invocation_started + max_wall_seconds - finalization_reserve_seconds
    if time.monotonic() >= campaign_deadline:
        raise TimeoutError("local replay preflight consumed the bounded inference interval")
    available_ram = _proc_kib_fields(Path("/proc/meminfo"), {"MemAvailable"}).get("MemAvailable", 0)
    if available_ram < int(plan["limits"]["minimum_available_ram_bytes"]):
        raise RuntimeError("available memory is below the frozen preflight safety floor")
    output_root = Path(plan["outputs"]["root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + plan["plan_sha256"][:10]
    run_dir = output_root / run_id
    lock_path = output_root / ".sweep-local-replay.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("another Sweep local replay owns the output lock") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\nrun_dir={run_dir}\n".encode())
        os.close(lock_fd)
        run_dir.mkdir(exist_ok=False)
    except BaseException:
        try:
            os.close(lock_fd)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)
        raise
    configure_offline_observability(run_dir / "observability.jsonl")
    from tinycomplete.observability.runs import run_scope

    records: list[dict[str, Any]] = []
    model_measurements: list[dict[str, Any]] = []
    metadata = {
        "schema": "sweep-local-replay-run-v1",
        "plan_sha256": plan["plan_sha256"],
        "upstream_plan_sha256": upstream["plan_sha256"],
        "prompt_bundle_sha256": plan["inputs"]["prompt_bundle_sha256"],
        "runtime_identity": identity,
        "host": _host_snapshot([row["pid"] for row in _find_q25_background_processes(3343901)]),
        "model_order": plan["replay"]["precision_order"],
        "quality_evidence": False,
        "no_gold_labels_loaded": True,
        "no_editor_service_changed": True,
        "no_model_download_or_conversion": True,
        "observability_capture_content": False,
        "invocation_deadline_started_at_utc": invocation_started_at_utc.isoformat(),
        "max_wall_seconds": max_wall_seconds,
        "finalization_reserve_seconds": finalization_reserve_seconds,
        "start_time_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    try:
        campaign_id = plan.get("observability", {}).get("campaign_id", OBSERVABILITY_CAMPAIGN)
        with run_scope(run_dir / "observability-run.json", campaign_id) as run:
            for precision in plan["replay"]["precision_order"]:
                if precision not in {"q4_k_m", "q8_0"}:
                    raise ValueError("local replay precision order contains an unauthorized value")
                record = _server_run(
                    precision=precision,
                    plan=plan,
                    runtime=runtime,
                    fixtures=fixtures,
                    run_dir=run_dir,
                    records=records,
                    run_context=run,
                    campaign_deadline=campaign_deadline,
                )
                model_measurements.append(record)
                server_pid = record["server_pid"]
                record["server_exit_verified"] = not Path(f"/proc/{server_pid}").exists()
                if not record["server_exit_verified"]:
                    raise RuntimeError("previous candidate model server is still resident")
                _atomic_json(run_dir / f"server-measurement-{precision}.json", record)
        summary = _summarize(records)
        q8_measurement = model_measurements[
            [model["precision"] for model in model_measurements].index("q8_0")
        ]
        q8_memory = q8_measurement["memory_sampler"]["peak_candidate_service_bytes"].get(
            "VmRSS_bytes", 0
        )
        q8_highwater = q8_measurement["memory_sampler"]["peak_candidate_service_bytes"].get(
            "VmHWM_bytes", 0
        )
        q8_peak = max(q8_memory, q8_highwater)
        expected_requests = int(plan["replay"]["expected_requests"])
        report = {
            "schema": (
                "sweep-local-changing-state-summary-v1"
                if plan["replay"].get("suite_mode") == TRANSITION_SUITE_MODE
                else "sweep-local-replay-summary-v1"
            ),
            "suite_mode": plan["replay"].get("suite_mode", "fixed_snapshot"),
            "plan_sha256": plan["plan_sha256"],
            "run_directory": str(run_dir),
            "completed_requests": len(records),
            "expected_requests": expected_requests,
            "same_prompt_repeat_requests": sum(
                model["same_prompt_repeat_requests"] for model in model_measurements
            ),
            "results": summary,
            "models": model_measurements,
            "q8_deployment_classification": (
                "researcher_only_over_1_5_GiB"
                if q8_peak > int(plan["limits"]["candidate_predictor_memory_bytes"])
                else "within_1_5_GiB_measurement_threshold"
            ),
            "q8_peak_rss_bytes": q8_peak,
            "q8_sampled_rss_peak_bytes": q8_memory,
            "q8_service_tree_rss_highwater_sum_bytes": q8_highwater,
            "memory_threshold_bytes": plan["limits"]["candidate_predictor_memory_bytes"],
            "workload_assessment": (
                "competing_q25_compute_observed"
                if any(
                    model["memory_sampler"]["background_q25_compute_observed"] is True
                    for model in model_measurements
                )
                else (
                    "q25_service_resident_without_sampled_compute"
                    if any(model["background_q25_processes_before"] for model in model_measurements)
                    else "q25_service_not_observed; inspect recorded host workload snapshots"
                )
            ),
            "quality_evidence": False,
            "personalization_or_training": False,
            "summary_created_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(run_dir / "summary.json", report)
        print(
            json.dumps(
                {
                    "state": "complete",
                    "plan_sha256": plan["plan_sha256"],
                    "run_directory": str(run_dir),
                    "completed_requests": len(records),
                    "expected_requests": report["expected_requests"],
                    "suite_mode": report["suite_mode"],
                    "q8_deployment_classification": report["q8_deployment_classification"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return run_dir
    except BaseException as exc:
        failure = {
            "schema": "sweep-local-replay-failure-v1",
            "plan_sha256": plan["plan_sha256"],
            "error_type": type(exc).__name__,
            "completed_requests": len(records),
            "failure_at_utc": datetime.now(UTC).isoformat(),
            "quality_evidence": False,
        }
        _atomic_json(run_dir / "failure.json", failure)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--upstream-plan", type=Path, default=DEFAULT_UPSTREAM_PLAN)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--execute", action="store_true", help="run the frozen local CPU replay")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    execute(args)


if __name__ == "__main__":
    main()
