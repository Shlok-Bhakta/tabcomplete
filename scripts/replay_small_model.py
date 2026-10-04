"""Bounded, single-slot native replay on the identified inference host."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import socket
import subprocess
import threading
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx

from tinycomplete.observability.context import current_run_context

FIM_REPLAY_SCHEMA = "q25-fim-native-latency-replay-plan-v1"
FIM_REPLAY_ALIAS = "q25-fim-native-latency"
FIM_REPLAY_TARGET_CONTEXT_TOKENS = (512, 1024, 2048)
FIM_REPLAY_STATES_PER_TARGET = 8
FIM_REPLAY_REPETITIONS = 2
FIM_REPLAY_SESSION_SECONDS = 20 * 60
FIM_REPLAY_RSS_LIMIT_BYTES = 1536 * 1024**2
FIM_REPLAY_SOURCE_FILES = (
    "scripts/replay_small_model.py",
    "scripts/evaluate_q25_fim_native.py",
    "scripts/evaluate_q25_fim.py",
    "scripts/prepare_q25_fim.py",
    "scripts/measure_r2_local.py",
    "scripts/profile_rust_engine.py",
    "src/tinycomplete/data/fim.py",
    "tools/tabcomplete_engine/src/main.rs",
    "tools/tabcomplete_engine/src/fim_v1.rs",
    "tools/tabcomplete_engine/src/context.rs",
    "tools/tabcomplete_engine/Cargo.lock",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proc_memory(pid: int) -> dict[str, int]:
    result = {}
    for source in (f"/proc/{pid}/smaps_rollup", f"/proc/{pid}/status"):
        try:
            for line in Path(source).read_text().splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key in (
                    "Rss",
                    "Pss",
                    "Pss_Anon",
                    "Pss_File",
                    "Anonymous",
                    "Swap",
                    "VmSwap",
                    "VmHWM",
                ):
                    parts = value.strip().split()
                    result[key + ("_status" if source.endswith("status") else "")] = (
                        int(parts[0]) * 1024
                    )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def counters() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    return {
        parts[0]: int(parts[1])
        for line in Path("/proc/vmstat").read_text().splitlines()
        if (parts := line.split()) and parts[0] in wanted
    }


def pressure() -> dict[str, str]:
    return {name: Path(f"/proc/pressure/{name}").read_text().strip() for name in ("memory", "io")}


def source_states() -> list[dict]:
    """Twelve ordered editor states, including a file switch and return."""
    rows = []
    for target, count in ((512, 29), (1024, 58), (2048, 116)):
        base = "".join(f"def f_{i}(value):\n    return value + {i}\n" for i in range(count))
        a = "# file: src/example.py\n" + base
        b = "# file: tests/other.py\n" + base
        edited_a = a.replace("return value + 1", "return value + 2", 1)
        tail = "def compute(value):\n    return "
        if target == 512:
            states = [
                ("fresh_open", "src/example.py", a + tail),
                ("append_chars", "src/example.py", a + tail + "value + "),
                ("replace_near_cursor", "src/example.py", a + tail + "value * "),
                ("reject_then_type_different", "src/example.py", a + tail + "value - "),
            ]
        elif target == 1024:
            states = [
                ("edit_earlier_in_file", "src/example.py", edited_a + tail + "value - "),
                ("switch_to_other_file", "tests/other.py", b + tail + "value * "),
                ("return_to_first_file", "src/example.py", edited_a + tail + "value - "),
                ("append_after_return", "src/example.py", edited_a + tail + "value - 1"),
            ]
        else:
            states = [
                ("fresh_open", "src/example.py", a + tail),
                ("replace_near_cursor", "src/example.py", a + tail + "value * "),
                ("reject_then_type_different", "src/example.py", a + tail + "value - "),
                ("edit_earlier_in_file", "src/example.py", edited_a + tail + "value - "),
            ]
        for index, (operation, file_id, prompt) in enumerate(states):
            rows.append(
                {
                    "id": f"synthetic-{target}-{index}",
                    "source_target": target,
                    "operation": operation,
                    "file_id": file_id,
                    "prompt": prompt,
                    "state_sha256": sha(prompt.encode()),
                }
            )
    assert len(rows) == 12
    return rows


def generate(url: str, prompt: str, *, cache_prompt: bool, max_new_tokens: int = 24) -> dict:
    # Shared provider owns SSE parsing, timing and one bounded model span.
    from measure_r2_local import NativeProvider

    provider = NativeProvider(url, "q25-coder-selected-native")
    provider.cache = cache_prompt
    provider.timeout_seconds = 120
    result = provider.generate_detailed(prompt, max_new_tokens)
    last = provider.last
    return {
        "response": result.text,
        "response_sha256": sha(result.text.encode()),
        "output_token_ids": last["token_ids"],
        "token_arrival_seconds": last["token_arrival_seconds"],
        "completed_seconds": last["total_seconds"],
        "tokens_cached": last["tokens_cached"],
        "tokens_evaluated": last["tokens_evaluated"],
        "stop_type": result.finish_reason,
        "truncated": last["truncated"],
        "generated_tokens": result.tokens,
        "server_timings": last["server_timings"],
    }


def load_frozen_states(path: Path, expected_sha256: str) -> list[dict]:
    if file_sha(path) != expected_sha256:
        raise ValueError("editor replay state fingerprint mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("editor replay states must have unique IDs")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or row.get("state_sha256") != sha(
            row["prompt"].encode()
        ):
            raise ValueError("editor replay prompt hash mismatch")
        if not all(key in row for key in ("operation", "file_id", "source_target")):
            raise ValueError("editor replay state metadata is incomplete")
    return rows


def _jsonl_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError("FIM replay input contains a blank record")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("FIM replay input records must be objects")
        rows.append(value)
    return rows


def _replace_codepoint(text: str, byte_offset: int) -> tuple[str, str, str]:
    raw = text.encode("utf-8")
    if not 0 <= byte_offset < len(raw):
        raise ValueError("FIM replay edit offset is outside source")
    try:
        old = raw[byte_offset:].decode("utf-8")[0]
    except UnicodeError:
        raise ValueError("FIM replay edit offset splits UTF-8") from None
    if old in "\r\n":
        raise ValueError("FIM replay edit would change a physical line boundary")
    replacement = "r" if old == "q" else "q"
    new = raw[:byte_offset] + replacement.encode() + raw[byte_offset + len(old.encode()) :]
    return new.decode("utf-8"), old, replacement


def _window_fim_case(case: dict, target_tokens: int, tokenizer) -> dict | None:
    from evaluate_q25_fim_native import expected_context
    from prepare_q25_fim import _token_ids, _truncate_prefix, _truncate_suffix

    from tinycomplete.data.fim import format_psm

    state = case.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("source"), str):
        return None
    if not isinstance(state.get("cursor_col"), int) or state["cursor_col"] <= 0:
        return None
    try:
        context = expected_context(case, tokenizer)
    except ValueError:
        return None
    raw = state["source"].encode("utf-8")
    cursor_byte = context["model_hole_range"]["start_byte"]
    line_start = context["apply_range"]["start_byte"]
    line_end = context["model_hole_range"]["end_byte"]
    try:
        full_prefix = raw[:cursor_byte].decode("utf-8")
        full_suffix = raw[line_end:].decode("utf-8")
    except UnicodeError:
        return None
    content_budget = target_tokens - 3
    initial_prefix_budget = (content_budget * 2) // 3
    attempts = [0]
    for distance in range(1, 33):
        attempts.extend((-distance, distance))
    best = None
    for adjustment in attempts:
        prefix_budget = initial_prefix_budget + adjustment
        suffix_budget = content_budget - prefix_budget
        if prefix_budget <= 0 or suffix_budget <= 0:
            continue
        try:
            prefix = _truncate_prefix(tokenizer, full_prefix, prefix_budget)
            suffix = _truncate_suffix(tokenizer, full_suffix, suffix_budget)
            prompt = format_psm(prefix, suffix)
            full_context_tokens = len(_token_ids(tokenizer, prompt))
        except Exception:
            continue
        distance_from_target = abs(full_context_tokens - target_tokens)
        if best is None or distance_from_target < best[0]:
            best = (distance_from_target, prefix, suffix)
        if distance_from_target <= 2:
            break
    if best is None or best[0] > 16:
        return None
    _, prefix, suffix = best
    prefix_start = cursor_byte - len(prefix.encode("utf-8"))
    suffix_end = line_end + len(suffix.encode("utf-8"))
    if prefix_start < 0 or suffix_end > len(raw):
        return None
    try:
        window_source = raw[prefix_start:suffix_end].decode("utf-8")
    except UnicodeError:
        return None
    window_state = dict(state)
    window_state["source"] = window_source
    if prefix_start <= line_start:
        window_state["target_row"] = raw[prefix_start:line_start].count(b"\n")
        window_state["cursor_col"] = state["cursor_col"]
    else:
        window_state["target_row"] = 0
        window_state["cursor_col"] = cursor_byte - prefix_start
    window_case = {
        "case_id": case["case_id"],
        "repository": case["repository"],
        "state": window_state,
    }
    try:
        bounded = expected_context(window_case, tokenizer)
        full_context_tokens = len(_token_ids(tokenizer, format_psm(prefix, suffix)))
        bounded_prompt_ids = _token_ids(tokenizer, bounded["prompt"])
    except Exception:
        return None
    if abs(full_context_tokens - target_tokens) > 16:
        return None
    if target_tokens > 899 and (
        bounded["prefix_token_count"] != 640 or bounded["suffix_token_count"] != 256
    ):
        return None
    return {
        "base_case_id": case["case_id"],
        "repository": case["repository"],
        "context_target_tokens": target_tokens,
        "full_context_tokens": full_context_tokens,
        "retained_prompt_tokens": len(bounded_prompt_ids),
        "prefix_context_tokens": bounded["prefix_token_count"],
        "suffix_context_tokens": bounded["suffix_token_count"],
        "prompt_sha256": sha(bounded["prompt"].encode("utf-8")),
        "source_sha256": sha(window_source.encode("utf-8")),
        "state": window_state,
    }


def _mutate_fim_state(row: dict, operation: str, tokenizer) -> dict:
    from evaluate_q25_fim_native import expected_context

    state = dict(row["state"])
    raw = state["source"].encode("utf-8")
    context = expected_context({"state": state}, tokenizer)
    cursor_byte = context["model_hole_range"]["start_byte"]
    line_start = context["apply_range"]["start_byte"]
    line_end = context["model_hole_range"]["end_byte"]
    if operation in {"append_utf8_at_cursor", "divergent_typing"}:
        inserted = "λ" if operation == "append_utf8_at_cursor" else "?"
        raw = raw[:cursor_byte] + inserted.encode("utf-8") + raw[cursor_byte:]
        state["source"] = raw.decode("utf-8")
        state["cursor_col"] += len(inserted.encode("utf-8"))
    elif operation == "near_replace_before_cursor":
        if cursor_byte <= line_start:
            raise ValueError("selected FIM base state has no character before its cursor")
        previous = raw[line_start:cursor_byte].decode("utf-8")[-1]
        replace_at = cursor_byte - len(previous.encode("utf-8"))
        state["source"], old, replacement = _replace_codepoint(state["source"], replace_at)
        state["cursor_col"] += len(replacement.encode()) - len(old.encode())
    elif operation == "earlier_edit":
        edit_limit = line_start if line_start else cursor_byte
        position = next(
            (index for index in range(edit_limit) if raw[index] not in (10, 13, 32, 9)),
            None,
        )
        if position is None:
            raise ValueError("selected FIM base state has no earlier editable character")
        state["source"], old, replacement = _replace_codepoint(state["source"], position)
        if line_start == 0:
            state["cursor_col"] += len(replacement.encode()) - len(old.encode())
    elif operation == "matching_prefix_suffix_change":
        position = next(
            (index for index in range(line_end, len(raw)) if raw[index] not in (10, 13)),
            None,
        )
        if position is None:
            raise ValueError("selected FIM base state has no suffix to change")
        state["source"], _, _ = _replace_codepoint(state["source"], position)
    else:
        raise ValueError("unknown FIM replay transition")
    return {**row, "operation": operation, "state": state}


def build_fim_latency_states(cases: list[dict], tokenizer) -> list[dict]:
    """Freeze 24 public-source transitions in three bounded-context regimes."""
    from evaluate_q25_fim_native import expected_context
    from prepare_q25_fim import _token_ids

    if len(cases) < 2:
        raise ValueError("FIM latency replay requires at least two public development cases")
    ordered = sorted(cases, key=lambda row: (row.get("case_id", ""), row.get("repository", "")))
    rows: list[dict] = []
    for target in FIM_REPLAY_TARGET_CONTEXT_TOKENS:
        candidates = []
        for case in ordered:
            if not isinstance(case.get("case_id"), str) or not isinstance(
                case.get("repository"), str
            ):
                continue
            candidate = _window_fim_case(case, target, tokenizer)
            if candidate is not None:
                candidates.append(candidate)
        if len(candidates) < 2:
            raise ValueError(
                "public cases cannot supply two distinct FIM contexts at a target size"
            )
        base_a = candidates[0]
        base_b = next(
            (
                candidate
                for candidate in candidates[1:]
                if candidate["repository"] != base_a["repository"]
                and candidate["source_sha256"] != base_a["source_sha256"]
            ),
            None,
        )
        if base_b is None:
            raise ValueError("FIM replay A/B transition needs distinct public repositories")
        transitions = [
            ("fresh_open", base_a),
            (
                "append_utf8_at_cursor",
                _mutate_fim_state(base_a, "append_utf8_at_cursor", tokenizer),
            ),
            (
                "near_replace_before_cursor",
                _mutate_fim_state(base_a, "near_replace_before_cursor", tokenizer),
            ),
            ("earlier_edit", _mutate_fim_state(base_a, "earlier_edit", tokenizer)),
            ("divergent_typing", _mutate_fim_state(base_a, "divergent_typing", tokenizer)),
            (
                "matching_prefix_suffix_change",
                _mutate_fim_state(base_a, "matching_prefix_suffix_change", tokenizer),
            ),
            ("switch_A_to_B", base_b),
            ("return_B_to_A", None),
        ]
        # A → B → A is explicit: the return state reproduces the preceding A
        # prompt exactly after a different repository/file has occupied the slot.
        transitions[-1] = ("return_B_to_A", transitions[5][1])
        from tinycomplete.data.fim import format_psm

        for index, (operation, source_row) in enumerate(transitions):
            state = dict(source_row["state"])
            case = {"state": state}
            expected = expected_context(case, tokenizer)
            prompt_ids = _token_ids(tokenizer, expected["prompt"])
            source_bytes = state["source"].encode("utf-8")
            full_prefix = source_bytes[: expected["model_hole_range"]["start_byte"]].decode()
            full_suffix = source_bytes[expected["model_hole_range"]["end_byte"] :].decode()
            full_context_tokens = len(_token_ids(tokenizer, format_psm(full_prefix, full_suffix)))
            row = {
                "id": f"fim-latency-{target}-{index:02d}",
                "operation": operation,
                "context_target_tokens": target,
                "full_context_tokens": full_context_tokens,
                "retained_prompt_tokens": len(prompt_ids),
                "retained_prefix_tokens": expected["prefix_token_count"],
                "retained_suffix_tokens": expected["suffix_token_count"],
                "prefix_range": expected["prefix_range"],
                "suffix_range": expected["suffix_range"],
                "prompt_sha256": sha(expected["prompt"].encode("utf-8")),
                "source_sha256": sha(state["source"].encode("utf-8")),
                "repository": source_row["repository"],
                "base_case_id": source_row["base_case_id"],
                "prompt_token_ids": prompt_ids,
                "state": state,
            }
            if index in (1, 2, 3, 4, 5):
                row["base_case_id"] = base_a["base_case_id"]
            if operation == "return_B_to_A":
                if row["prompt_sha256"] != rows[-2]["prompt_sha256"]:
                    raise ValueError("FIM replay A → B → A return prompt is not identical")
            rows.append(row)
    if len(rows) != 24 or len({row["id"] for row in rows}) != 24:
        raise ValueError("FIM latency replay state count differs from the frozen suite")
    return rows


class LocalProcessSampler:
    """Sample the owned service with the existing 500 ms REMOTE_SAMPLE payload."""

    def __init__(self, pid: int, interval_seconds: float = 0.5):
        from profile_rust_engine import REMOTE_SAMPLE

        if pid <= 1 or interval_seconds != 0.5:
            raise ValueError("FIM sampler requires the owned PID and fixed 500 ms interval")
        self.process = subprocess.Popen(
            ["python3", "-c", REMOTE_SAMPLE, str(pid), "0.500"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        self.first_sample = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        if not self.first_sample.wait(15):
            self.stop()
            raise ValueError("owned FIM service resource sampler did not start")

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.lock:
                self.rows.append(value)
            self.first_sample.set()

    def copy_rows(self) -> list[dict]:
        with self.lock:
            return list(self.rows)

    def stop(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write("stop\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.reader.join(timeout=2)


def _fim_runtime_identity(args, tokenizer, training_plan: dict, conversion: dict) -> dict:
    from evaluate_q25_fim_native import (
        CONTEXT_LAYOUT,
        PROTOCOL,
        TOKENIZER_SHA,
        attest_process,
        load_tokenizer,
        memory_snapshot,
        tokenizer_added_token_inventory,
        validate_native_tokenizer_identity,
    )
    from evaluate_q25_fim_native import (
        sha as native_sha,
    )

    url = args.fim_url
    parsed = urlsplit(url or "")
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("FIM replay requires the configured local owned service")
    if parsed.port is None or args.model is None or args.binary is None or args.conversion is None:
        raise ValueError("FIM replay artifact and service identities are required")
    if (
        args.tokenizer is None
        or args.training_plan is None
        or args.cases is None
        or args.pid is None
    ):
        raise ValueError("FIM replay training and development inputs are required")
    if conversion.get("status") != "complete":
        raise ValueError("FIM replay requires the completed selected Q4 conversion")
    if conversion.get("training_plan_sha256") != file_sha(args.training_plan):
        raise ValueError("selected Q4 conversion training plan differs")
    model_sha = file_sha(args.model)
    binary_sha = file_sha(args.binary)
    if model_sha != conversion.get("q4_export", {}).get("sha256"):
        raise ValueError("selected Q4 file differs from its conversion manifest")
    if training_plan.get("schema") != "q25-fim-training-plan-v1":
        raise ValueError("FIM replay requires the final frozen training plan")
    if file_sha(args.tokenizer / "tokenizer.json") != TOKENIZER_SHA:
        raise ValueError("FIM replay tokenizer differs from the pinned source tokenizer")
    tokenizer = tokenizer or load_tokenizer(args.tokenizer)
    health_response = httpx.get(url + "/health", timeout=10, trust_env=False)
    health_response.raise_for_status()
    health = health_response.json()
    if (
        health.get("status") != "ok"
        or health.get("model_sha256") != model_sha
        or health.get("model_protocol") != PROTOCOL
        or health.get("context_layout") != CONTEXT_LAYOUT
        or health.get("output_tokens") != 96
        or health.get("active_slots") != 1
        or health.get("saved_contexts") != 0
        or health.get("syntax_validation") is not False
        or not isinstance(health.get("runtime_config_hash"), str)
        or len(health["runtime_config_hash"]) != 64
    ):
        raise ValueError("running FIM service differs from its bounded serving contract")
    if health.get("context_size", 0) < 899 + 96 or health.get("input_tokens", 0) < 899 + 96:
        raise ValueError("running FIM service lacks the fixed input and output token reserve")
    process = attest_process(args.pid, binary_sha, parsed.port)
    if native_sha(args.model) != model_sha:
        raise ValueError("selected Q4 model identity could not be read")
    tokenizer_response = httpx.get(url + "/v1/fim-tokenizer", timeout=10, trust_env=False)
    tokenizer_response.raise_for_status()
    tokenizer_profile = validate_native_tokenizer_identity(
        conversion,
        health,
        tokenizer_response.json(),
        tokenizer,
        tokenizer_added_token_inventory(args.tokenizer),
    )
    memory = memory_snapshot(args.pid)
    rss = memory.get("VmRSS_bytes")
    process_memory = proc_memory(args.pid)
    process_rss = process_memory.get("VmRSS_status", process_memory.get("Rss"))
    if (
        not isinstance(rss, int)
        or rss > FIM_REPLAY_RSS_LIMIT_BYTES
        or not isinstance(process_rss, int)
        or process_rss > FIM_REPLAY_RSS_LIMIT_BYTES
    ):
        raise ValueError("owned FIM service exceeds the replay resident-memory safeguard")
    slots_response = httpx.get(url + "/slots", timeout=5, trust_env=False)
    slots_response.raise_for_status()
    if slots_response.json() != [{"id": 0, "is_processing": False}]:
        raise ValueError("owned FIM service must have one idle inference slot")
    runtime_fields = (
        "status",
        "model_sha256",
        "model_protocol",
        "context_layout",
        "context_size",
        "input_tokens",
        "output_tokens",
        "active_slots",
        "saved_contexts",
        "backend",
        "runtime_config_hash",
        "threads",
        "prompt_threads",
        "batch_size",
        "microbatch_size",
        "cache_type",
        "syntax_validation",
        "tokenizer_sha256",
        "tokenizer_contract_sha256",
        "fim_profile",
    )
    if any(key not in health for key in runtime_fields):
        raise ValueError("running FIM service identity is incomplete")
    return {
        "model_sha256": model_sha,
        "binary_sha256": binary_sha,
        "conversion_sha256": file_sha(args.conversion),
        "training_plan_sha256": file_sha(args.training_plan),
        "cases_sha256": file_sha(args.cases),
        "tokenizer_sha256": TOKENIZER_SHA,
        "health": {key: health[key] for key in runtime_fields},
        "tokenizer_profile": tokenizer_profile,
        "process": process,
        "initial_memory": memory,
        "initial_process_memory": process_memory,
    }


def _fim_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: file_sha(root / name) for name in FIM_REPLAY_SOURCE_FILES}


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_fim_state_rows(rows: list[dict], tokenizer) -> None:
    from evaluate_q25_fim_native import expected_context
    from prepare_q25_fim import _token_ids

    from tinycomplete.data.fim import format_psm

    if len(rows) != 24 or len({row.get("id") for row in rows}) != 24:
        raise ValueError("FIM latency fixture must contain exactly 24 unique transitions")
    expected_ids = [
        f"fim-latency-{target}-{index:02d}" for target in (512, 1024, 2048) for index in range(8)
    ]
    if [row.get("id") for row in rows] != expected_ids:
        raise ValueError("FIM latency transition order differs from its fixed sequence")
    for row in rows:
        state = row.get("state")
        if not isinstance(state, dict) or not isinstance(state.get("source"), str):
            raise ValueError("FIM latency state source is invalid")
        case = {"state": state}
        expected = expected_context(case, tokenizer)
        source = state["source"].encode("utf-8")
        full_prefix = source[: expected["model_hole_range"]["start_byte"]].decode("utf-8")
        full_suffix = source[expected["model_hole_range"]["end_byte"] :].decode("utf-8")
        full_context_tokens = len(_token_ids(tokenizer, format_psm(full_prefix, full_suffix)))
        prompt_ids = _token_ids(tokenizer, expected["prompt"])
        expected_values = {
            "full_context_tokens": full_context_tokens,
            "retained_prompt_tokens": len(prompt_ids),
            "retained_prefix_tokens": expected["prefix_token_count"],
            "retained_suffix_tokens": expected["suffix_token_count"],
            "prefix_range": expected["prefix_range"],
            "suffix_range": expected["suffix_range"],
            "prompt_sha256": sha(expected["prompt"].encode("utf-8")),
            "source_sha256": sha(source),
            "prompt_token_ids": prompt_ids,
        }
        if any(row.get(key) != value for key, value in expected_values.items()):
            raise ValueError("FIM latency state differs from its frozen source context")
        if row.get("context_target_tokens") != int(row["id"].split("-")[2]):
            raise ValueError("FIM latency state target size differs from its fixed suite")
        if abs(full_context_tokens - row.get("context_target_tokens", 0)) > 16:
            raise ValueError("FIM latency source context is outside its target size")
        if len(prompt_ids) + 96 > 1024:
            raise ValueError("FIM latency prompt exceeds the fixed inference token reserve")


def freeze_fim_native_replay(args) -> dict:
    from evaluate_q25_fim_native import (
        PROTOCOL,
        TOKENIZER_SHA,
        load_tokenizer,
        read_rows,
    )

    required_paths = (
        args.plan,
        args.states,
        args.output,
        args.training_plan,
        args.cases,
        args.tokenizer,
        args.model,
        args.binary,
        args.conversion,
    )
    if any(path is None for path in required_paths) or args.pid is None or args.fim_url is None:
        raise ValueError("FIM replay freeze requires all frozen artifacts and service identity")
    if args.alias != FIM_REPLAY_ALIAS:
        raise ValueError("FIM replay requires its fixed bounded model alias")
    input_files = (
        args.training_plan,
        args.cases,
        args.model,
        args.binary,
        args.conversion,
    )
    if (
        any(not path.is_file() for path in input_files)
        or not (args.tokenizer / "tokenizer.json").is_file()
    ):
        raise FileNotFoundError("FIM replay freeze input is missing")
    if args.output.is_symlink() or args.plan.is_symlink() or args.states.is_symlink():
        raise ValueError("FIM replay freeze paths cannot be symlinks")
    output_root = args.output.resolve()
    if args.plan.resolve().parent != output_root or args.states.resolve().parent != output_root:
        raise ValueError("FIM replay plan and states must be frozen beside the results")
    if args.plan.exists() or args.states.exists():
        raise FileExistsError("FIM replay freeze preserves existing plan and state files")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("FIM replay output directory must be empty before freeze")
    training_plan = json.loads(args.training_plan.read_text(encoding="utf-8"))
    conversion = json.loads(args.conversion.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.tokenizer)
    cases = read_rows(args.cases)
    if not cases:
        raise ValueError("FIM latency replay needs frozen public development cases")
    from evaluate_q25_fim_native import expected_context

    for case in cases:
        if (
            expected_context(case, tokenizer)["prompt"] != case.get("prompt")
            or tokenizer.encode(case["prompt"], add_special_tokens=False)
            != case.get("prompt_token_ids")
            or sha(case["prompt"].encode("utf-8")) != case.get("context_sha256")
        ):
            raise ValueError("FIM latency source cases do not match their pinned prompts")
    states = build_fim_latency_states(cases, tokenizer)
    _validate_fim_state_rows(states, tokenizer)
    runtime = _fim_runtime_identity(args, tokenizer, training_plan, conversion)
    source_hashes = _fim_source_hashes()
    if args.output.exists() and (args.output / "measurements.jsonl").exists():
        raise FileExistsError("FIM replay output already has measurements")
    _write_jsonl_atomic(args.states, states)
    state_sha256 = file_sha(args.states)
    url = urlsplit(args.fim_url)
    plan = {
        "schema": FIM_REPLAY_SCHEMA,
        "model_alias": FIM_REPLAY_ALIAS,
        "training_plan_sha256": runtime["training_plan_sha256"],
        "cases_sha256": runtime["cases_sha256"],
        "states_sha256": state_sha256,
        "states_count": len(states),
        "tokenizer_sha256": TOKENIZER_SHA,
        "model_sha256": runtime["model_sha256"],
        "binary_sha256": runtime["binary_sha256"],
        "conversion_sha256": runtime["conversion_sha256"],
        "runtime_health": runtime["health"],
        "tokenizer_profile": runtime["tokenizer_profile"],
        "process_identity": runtime["process"],
        "freeze_memory": runtime["initial_memory"],
        "freeze_process_memory": runtime["initial_process_memory"],
        "service_url": f"{url.scheme}://{url.hostname}:{url.port}",
        "source_sha256": source_hashes,
        "target_context_token_counts": list(FIM_REPLAY_TARGET_CONTEXT_TOKENS),
        "retained_context_token_limits": {"prefix": 640, "suffix": 256},
        "states_per_target": FIM_REPLAY_STATES_PER_TARGET,
        "passes": [
            {"repetition": 0, "name": "fresh_cache_off", "requests": 24},
            {"repetition": 1, "name": "fresh_cache_off", "requests": 24},
            {"repetition": 0, "name": "cache_trajectory", "requests": 48},
            {"repetition": 1, "name": "cache_trajectory", "requests": 48},
        ],
        "conditions": {
            "fresh_state_cache_off": 48,
            "changed_editor_state_cache_on": 48,
            "identical_prompt_repeat_cache_on": 48,
        },
        "request_count": 144,
        "deadline_seconds": FIM_REPLAY_SESSION_SECONDS,
        "rss_limit_bytes": FIM_REPLAY_RSS_LIMIT_BYTES,
        "sampling_interval_seconds": 0.5,
        "quality_claim": False,
        "content_recorded": False,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(args.plan, plan)
    args.output.mkdir(parents=True, exist_ok=True)
    return {
        "plan_sha256": file_sha(args.plan),
        "states_sha256": state_sha256,
        "states": len(states),
        "requests": plan["request_count"],
        "model_protocol": PROTOCOL,
        "runtime_config_hash": runtime["health"]["runtime_config_hash"],
    }


def _fim_case_from_state(row: dict, tokenizer) -> tuple[dict, dict]:
    from evaluate_q25_fim_native import expected_context
    from prepare_q25_fim import _token_ids

    state = row["state"]
    expected = expected_context({"state": state}, tokenizer)
    prompt_ids = _token_ids(tokenizer, expected["prompt"])
    if (
        expected["prompt"] == ""
        or prompt_ids != row.get("prompt_token_ids")
        or sha(expected["prompt"].encode("utf-8")) != row.get("prompt_sha256")
    ):
        raise ValueError("FIM replay case differs from the frozen context")
    case = {
        "case_id": row["id"],
        "repository": row["repository"],
        "state": state,
        "prompt": expected["prompt"],
        "prompt_token_ids": prompt_ids,
        "context_sha256": row["prompt_sha256"],
        "target": "",
    }
    return case, expected


def _rss_from_sampler_rows(rows: list[dict]) -> int | None:
    values = []
    for row in rows:
        for value in (
            row.get("process_status", {}).get("VmRSS"),
            row.get("process_memory_bytes", {}).get("Rss"),
        ):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                values.append(value)
    return max(values) if values else None


def _rss_from_proc_memory(snapshot: dict[str, int]) -> int | None:
    values = [
        value
        for key in ("VmRSS_status", "Rss")
        if isinstance((value := snapshot.get(key)), int) and value >= 0
    ]
    return max(values) if values else None


def _percentile_ms(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.5)))
    return ordered[index]


def fim_replay_schedule(rows: list[dict]) -> list[tuple[int, dict, str, bool]]:
    """Return a fresh pass followed by changed-state/repeat cache trajectories."""
    if len(rows) != 24:
        raise ValueError("FIM latency schedule requires exactly 24 frozen states")
    schedule = []
    for repetition in range(FIM_REPLAY_REPETITIONS):
        for row in rows:
            schedule.append((repetition, row, "fresh_state_cache_off", False))
    for repetition in range(FIM_REPLAY_REPETITIONS):
        for row in rows:
            schedule.append((repetition, row, "changed_editor_state_cache_on", True))
            schedule.append((repetition, row, "identical_prompt_repeat_cache_on", True))
    if len(schedule) != 144:
        raise ValueError("FIM latency request schedule differs from its fixed size")
    return schedule


def replay_fim_native(args) -> dict:
    from evaluate_q25_fim_native import (
        TOKENIZER_SHA,
        load_tokenizer,
        memory_snapshot,
        prepare_request,
        score_terminal,
        validate_terminal,
    )
    from evaluate_q25_fim_native import (
        sha as native_sha,
    )
    from measure_r2_local import NativeProvider
    from profile_rust_engine import cgroup_summary

    from tinycomplete.observability.context import RunContext
    from tinycomplete.observability.runs import run_scope
    from tinycomplete.observability.spans import operation

    if (
        any(
            path is None
            for path in (
                args.plan,
                args.states,
                args.output,
                args.training_plan,
                args.cases,
                args.tokenizer,
                args.model,
                args.binary,
                args.conversion,
            )
        )
        or args.pid is None
        or args.fim_url is None
    ):
        raise ValueError("FIM replay requires all frozen inputs and service identity")
    if (
        any(
            not path.is_file()
            for path in (
                args.plan,
                args.states,
                args.training_plan,
                args.cases,
                args.model,
                args.binary,
                args.conversion,
            )
        )
        or not (args.tokenizer / "tokenizer.json").is_file()
    ):
        raise FileNotFoundError("FIM replay input is missing")
    output_root = args.output.resolve()
    if (
        args.plan.resolve().parent != output_root
        or args.states.resolve().parent != output_root
        or any(path.is_symlink() for path in (args.plan, args.states, args.output))
    ):
        raise ValueError(
            "FIM replay inputs and results must share one non-symlink output directory"
        )
    started = time.monotonic()
    deadline = started + FIM_REPLAY_SESSION_SECONDS
    if not args.plan.is_file() or not args.states.is_file():
        raise FileNotFoundError("frozen FIM latency plan or states are missing")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != FIM_REPLAY_SCHEMA:
        raise ValueError("FIM latency replay plan schema differs")
    if args.alias != FIM_REPLAY_ALIAS or plan.get("model_alias") != FIM_REPLAY_ALIAS:
        raise ValueError("FIM latency model alias differs from its frozen plan")
    if args.plan_sha256 is None or file_sha(args.plan) != args.plan_sha256:
        raise ValueError("FIM latency plan differs from its separately frozen SHA-256")
    if (
        plan.get("states_count") != 24
        or plan.get("target_context_token_counts") != [512, 1024, 2048]
        or plan.get("retained_context_token_limits") != {"prefix": 640, "suffix": 256}
        or plan.get("states_per_target") != 8
        or plan.get("passes")
        != [
            {"repetition": 0, "name": "fresh_cache_off", "requests": 24},
            {"repetition": 1, "name": "fresh_cache_off", "requests": 24},
            {"repetition": 0, "name": "cache_trajectory", "requests": 48},
            {"repetition": 1, "name": "cache_trajectory", "requests": 48},
        ]
        or plan.get("conditions")
        != {
            "fresh_state_cache_off": 48,
            "changed_editor_state_cache_on": 48,
            "identical_prompt_repeat_cache_on": 48,
        }
        or plan.get("request_count") != 144
        or plan.get("deadline_seconds") != FIM_REPLAY_SESSION_SECONDS
        or plan.get("rss_limit_bytes") != FIM_REPLAY_RSS_LIMIT_BYTES
        or plan.get("sampling_interval_seconds") != 0.5
        or plan.get("quality_claim") is not False
        or plan.get("content_recorded") is not False
        or plan.get("source_sha256") != _fim_source_hashes()
    ):
        raise ValueError("FIM latency schedule differs from its fixed bounded protocol")
    if args.output.exists() and (args.output / "measurements.jsonl").exists():
        raise FileExistsError("FIM replay results already exist")
    tokenizer = load_tokenizer(args.tokenizer)
    state_rows = _jsonl_rows(args.states)
    if file_sha(args.states) != plan.get("states_sha256") or len(state_rows) != 24:
        raise ValueError("FIM latency state fixture differs from its frozen identity")
    _validate_fim_state_rows(state_rows, tokenizer)
    training_plan = json.loads(args.training_plan.read_text(encoding="utf-8"))
    conversion = json.loads(args.conversion.read_text(encoding="utf-8"))
    runtime = _fim_runtime_identity(args, tokenizer, training_plan, conversion)
    expected_identities = {
        "training_plan_sha256": runtime["training_plan_sha256"],
        "cases_sha256": runtime["cases_sha256"],
        "tokenizer_sha256": TOKENIZER_SHA,
        "model_sha256": runtime["model_sha256"],
        "binary_sha256": runtime["binary_sha256"],
        "conversion_sha256": runtime["conversion_sha256"],
        "runtime_health": runtime["health"],
        "tokenizer_profile": runtime["tokenizer_profile"],
        "process_identity": runtime["process"],
    }
    if any(plan.get(key) != value for key, value in expected_identities.items()):
        raise ValueError("FIM latency runtime or source differs from the frozen plan")
    expected_url = urlsplit(args.fim_url)
    if (
        plan.get("service_url")
        != f"{expected_url.scheme}://{expected_url.hostname}:{expected_url.port}"
    ):
        raise ValueError("FIM latency service URL differs from its frozen plan")
    if args.output.exists() and any(
        path.name not in {args.plan.name, args.states.name} for path in args.output.iterdir()
    ):
        raise FileExistsError("FIM replay output directory contains unrelated artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    measurements_path = args.output / "measurements.jsonl"
    summary_path = args.output / "summary.json"
    if measurements_path.exists() or summary_path.exists():
        raise FileExistsError("FIM replay result files already exist")
    plan_hash = file_sha(args.plan)
    _write_json_atomic(
        args.output / "metadata.json",
        {
            "schema": "q25-fim-native-latency-replay-run-v1",
            "plan_sha256": plan_hash,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "process_identity": runtime["process"],
            "quality_claim": False,
            "content_recorded": False,
        },
    )

    before_vmstat = counters()
    before_pressure = pressure()
    provider = NativeProvider(args.fim_url, args.alias)
    provider.timeout_seconds = min(60, FIM_REPLAY_SESSION_SECONDS)
    sampler = LocalProcessSampler(args.pid, 0.5)
    records: list[dict] = []
    schedule = fim_replay_schedule(state_rows)
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0), trust_env=False) as context_client:
            with run_scope(args.output / "observability-run.json", "q25-fim-native-latency") as run:
                with operation(
                    "campaign.phase", attributes={"tabcomplete.phase": "native_fim_latency_replay"}
                ):
                    for repetition, row, condition, cached in schedule:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("FIM latency replay reached its 20-minute deadline")
                        case, expected = _fim_case_from_state(row, tokenizer)
                        slots = httpx.get(args.fim_url + "/slots", timeout=5, trust_env=False)
                        slots.raise_for_status()
                        if slots.json() != [{"id": 0, "is_processing": False}]:
                            raise ValueError("owned FIM inference slot was occupied before request")
                        provider.cache = cached
                        remaining_seconds = deadline - time.monotonic()
                        if remaining_seconds <= 5:
                            raise TimeoutError("FIM latency replay reached its 20-minute deadline")
                        provider.timeout_seconds = min(60, remaining_seconds - 2)
                        request_context = (run or RunContext.new()).for_case(
                            f"{row['id']}/{repetition}/{condition}"
                        )
                        with (
                            request_context.activate(),
                            operation(
                                "eval.case",
                                attributes={"tabcomplete.case_kind": "synthetic_fim_latency"},
                            ),
                        ):
                            context_started = time.perf_counter()
                            process_memory_before = proc_memory(args.pid)
                            memory_before = memory_snapshot(args.pid)
                            rss_before = _rss_from_proc_memory(process_memory_before)
                            memory_rss_before = memory_before.get("VmRSS_bytes")
                            if (
                                rss_before is None
                                or rss_before > FIM_REPLAY_RSS_LIMIT_BYTES
                                or not isinstance(memory_rss_before, int)
                                or memory_rss_before > FIM_REPLAY_RSS_LIMIT_BYTES
                            ):
                                raise ValueError("owned FIM service exceeded the 1.5 GiB RSS limit")
                            prepared = prepare_request(
                                context_client,
                                args.fim_url,
                                case,
                                request_context.request_id,
                                runtime["health"],
                                tokenizer,
                            )
                            context_ms = (time.perf_counter() - context_started) * 1000
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "FIM latency replay reached its 20-minute deadline"
                                )
                            if prepared["prompt"] != expected["prompt"] or prepared[
                                "prompt_tokens"
                            ] + 96 > min(
                                runtime["health"]["input_tokens"],
                                runtime["health"]["context_size"],
                            ):
                                raise ValueError("served FIM context violates its frozen budget")
                            binding = {
                                "request_id": request_context.request_id,
                                "context_hash": prepared["context_hash"],
                                "completion_mode": "remaining_logical_line_after_utf8_cursor",
                            }
                            provider.editor_request_binding = binding
                            provider.repository_identity = row["repository"]
                            request_started = time.perf_counter()
                            with patch.dict(os.environ, {"NO_PROXY": "*", "no_proxy": "*"}):
                                result = provider.generate_detailed(prepared["prompt"], 96)
                            completion_ms = (time.perf_counter() - request_started) * 1000
                            if any(
                                not math.isfinite(value) or value < 0
                                for value in (context_ms, completion_ms)
                            ):
                                raise ValueError("native FIM client timings are invalid")
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "FIM latency replay reached its 20-minute deadline"
                                )
                            terminal = provider.last.get("terminal_event")
                            token_ids = provider.last.get("token_ids")
                            if (
                                not isinstance(terminal, dict)
                                or not validate_terminal(terminal, runtime["health"], binding)
                                or terminal.get("sampled_token_ids") != token_ids
                            ):
                                raise ValueError("native FIM terminal identity is stale or invalid")
                            with operation("model.score"):
                                scored = score_terminal(case, terminal, result.text, tokenizer)
                            timing = provider.last.get("server_timings")
                            if not isinstance(timing, dict):
                                raise ValueError("native FIM request timings are missing")
                            cache_n = timing.get("cache_n")
                            prompt_n = timing.get("prompt_n")
                            predicted_n = timing.get("predicted_n")
                            if (
                                not all(
                                    isinstance(value, int)
                                    and not isinstance(value, bool)
                                    and value >= 0
                                    for value in (cache_n, prompt_n, predicted_n)
                                )
                                or cache_n + prompt_n != prepared["prompt_tokens"]
                                or predicted_n != len(token_ids)
                                or result.tokens != predicted_n
                                or (not cached and cache_n != 0)
                            ):
                                raise ValueError(
                                    "native FIM observed token counts are inconsistent"
                                )
                            timing_values = [
                                timing.get(key) for key in ("prompt_ms", "predicted_ms", "total_ms")
                            ]
                            if any(
                                not isinstance(value, (int, float))
                                or isinstance(value, bool)
                                or not math.isfinite(float(value))
                                or value < 0
                                for value in timing_values
                            ):
                                raise ValueError("native FIM server timings are invalid")
                            token_arrivals = provider.last.get("token_arrival_seconds", {})
                            first_token_seconds = token_arrivals.get("1")
                            if first_token_seconds is not None and (
                                not isinstance(first_token_seconds, (int, float))
                                or isinstance(first_token_seconds, bool)
                                or not math.isfinite(float(first_token_seconds))
                                or first_token_seconds < 0
                            ):
                                raise ValueError("native FIM first-token timing is invalid")
                            first_token_ms = (
                                first_token_seconds * 1000
                                if first_token_seconds is not None
                                else None
                            )
                            memory = memory_snapshot(args.pid)
                            rss = memory.get("VmRSS_bytes")
                            process_memory = proc_memory(args.pid)
                            process_rss = _rss_from_proc_memory(process_memory)
                            if (
                                not isinstance(rss, int)
                                or rss > FIM_REPLAY_RSS_LIMIT_BYTES
                                or process_rss is None
                                or process_rss > FIM_REPLAY_RSS_LIMIT_BYTES
                            ):
                                raise ValueError("owned FIM service exceeded the 1.5 GiB RSS limit")
                            sampler_rows = sampler.copy_rows()
                            sampled_rss = _rss_from_sampler_rows(sampler_rows)
                            if sampled_rss is not None and sampled_rss > FIM_REPLAY_RSS_LIMIT_BYTES:
                                raise ValueError(
                                    "sampled FIM service exceeded the 1.5 GiB RSS limit"
                                )
                            row_result = {
                                "state_id": row["id"],
                                "operation": row["operation"],
                                "source_sha256": row["source_sha256"],
                                "repository_identity": row["repository"],
                                "context_target_tokens": row["context_target_tokens"],
                                "full_context_tokens": row["full_context_tokens"],
                                "retained_prompt_tokens": prepared["prompt_tokens"],
                                "retained_prefix_tokens": prepared["prefix_token_count"],
                                "retained_suffix_tokens": prepared["suffix_token_count"],
                                "prefix_range": prepared["prefix_range"],
                                "suffix_range": prepared["suffix_range"],
                                "prompt_sha256": row["prompt_sha256"],
                                "repetition": repetition,
                                "condition": condition,
                                "cache_requested": cached,
                                "actual_cache_n": cache_n,
                                "recomputed_prompt_n": prompt_n,
                                "generated_body_tokens": predicted_n,
                                "stop_type": terminal["stop_type"],
                                "terminal_token_id": terminal.get("terminal_token_id"),
                                "quality_error_code": scored["quality_error_code"],
                                "terminated": scored["terminated"],
                                "canonical_action_sha256": (
                                    native_sha(
                                        json.dumps(
                                            scored["canonical_action"],
                                            sort_keys=True,
                                            separators=(",", ":"),
                                        ).encode("utf-8")
                                    )
                                    if scored["canonical_action"] is not None
                                    else None
                                ),
                                "output_token_ids_sha256": native_sha(
                                    json.dumps(token_ids, separators=(",", ":")).encode()
                                ),
                                "response_sha256": native_sha(result.text.encode("utf-8")),
                                "context_construction_ms": round(context_ms, 3),
                                "client_completion_ms": round(completion_ms, 3),
                                "first_observed_token_ms": (
                                    round(first_token_ms, 3) if first_token_ms is not None else None
                                ),
                                "backend_prompt_ms": timing.get("prompt_ms"),
                                "backend_decode_ms": timing.get("predicted_ms"),
                                "backend_total_ms": timing.get("total_ms"),
                                "memory_before": memory_before,
                                "memory_after": memory,
                                "process_memory_before": process_memory_before,
                                "process_memory_after": process_memory,
                                "request_id": request_context.request_id,
                            }
                        with measurements_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row_result, sort_keys=True) + "\n")
                        records.append(row_result)
    finally:
        sampler.stop()
    resource_samples = sampler.copy_rows()
    if not resource_samples:
        raise ValueError("FIM latency replay produced no 500 ms process samples")
    if any(sample.get("pid") != args.pid for sample in resource_samples):
        raise ValueError("FIM resource samples include a different process")
    if _rss_from_sampler_rows(resource_samples) is not None and (
        _rss_from_sampler_rows(resource_samples) > FIM_REPLAY_RSS_LIMIT_BYTES
    ):
        raise ValueError("sampled FIM service exceeded the 1.5 GiB RSS limit")
    _write_jsonl_atomic(args.output / "resource-samples.jsonl", resource_samples)
    after_vmstat = counters()
    after_pressure = pressure()
    expected_conditions = {
        "fresh_state_cache_off": 48,
        "changed_editor_state_cache_on": 48,
        "identical_prompt_repeat_cache_on": 48,
    }
    condition_summaries = {}
    for condition, expected_count in expected_conditions.items():
        subset = [row for row in records if row["condition"] == condition]
        if len(subset) != expected_count:
            raise ValueError("FIM latency request count differs from its frozen schedule")
        durations = [float(row["client_completion_ms"]) for row in subset]
        first_token_durations = [
            float(row["first_observed_token_ms"])
            for row in subset
            if row["first_observed_token_ms"] is not None
        ]
        condition_summaries[condition] = {
            "count": len(subset),
            "client_completion_ms_median": _percentile_ms(durations, 0.5),
            "client_completion_ms_p95": _percentile_ms(durations, 0.95),
            "first_observed_token_ms_median": _percentile_ms(first_token_durations, 0.5),
            "first_observed_token_ms_p95": _percentile_ms(first_token_durations, 0.95),
            "backend_prompt_ms_median": _percentile_ms(
                [float(row["backend_prompt_ms"]) for row in subset], 0.5
            ),
            "backend_decode_ms_median": _percentile_ms(
                [float(row["backend_decode_ms"]) for row in subset], 0.5
            ),
            "actual_cache_tokens": sum(row["actual_cache_n"] for row in subset),
            "recomputed_prompt_tokens": sum(row["recomputed_prompt_n"] for row in subset),
            "actual_eos": sum(row["stop_type"] == "eos" for row in subset),
            "canonical_valid": sum(row["quality_error_code"] is None for row in subset),
        }
    elapsed_seconds = time.monotonic() - started
    if elapsed_seconds > FIM_REPLAY_SESSION_SECONDS:
        raise TimeoutError("FIM latency replay exceeded its 20-minute deadline")
    summary = {
        "schema": "q25-fim-native-latency-replay-result-v1",
        "plan_sha256": plan_hash,
        "requests": len(records),
        "conditions": condition_summaries,
        "process_identity": runtime["process"],
        "runtime_config_hash": runtime["health"]["runtime_config_hash"],
        "resource_summary": cgroup_summary(resource_samples),
        "resource_sample_count": len(resource_samples),
        "vmstat_delta": {
            key: after_vmstat[key] - before_vmstat.get(key, after_vmstat[key])
            for key in after_vmstat
        },
        "pressure_before": before_pressure,
        "pressure_after": after_pressure,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "deadline_seconds": FIM_REPLAY_SESSION_SECONDS,
        "rss_limit_bytes": FIM_REPLAY_RSS_LIMIT_BYTES,
        "quality_claim": False,
        "content_recorded": False,
        "observed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19094)
    parser.add_argument("--diagnostic-saved-idle-cache", action="store_true")
    parser.add_argument("--states", type=Path, help="optional frozen trained-format editor states")
    parser.add_argument("--states-sha256")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--rust-url", help="owned resident Rust engine, never an external API")
    parser.add_argument("--ssh-host", help="configured inference host for process measurements")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--setting", help="frozen plan setting identity")
    parser.add_argument("--fim-native-stage", choices=("freeze", "replay"))
    parser.add_argument("--fim-url", help="owned localhost native FIM service")
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--conversion", type=Path)
    parser.add_argument("--pid", type=int, help="PID of the owned native service")
    parser.add_argument("--plan-sha256", help="independent SHA-256 from the freeze result")
    args = parser.parse_args()
    if args.fim_native_stage:
        if args.rust_url or args.ssh_host or args.runtime or args.setting:
            parser.error("native FIM replay cannot be combined with legacy replay options")
        try:
            result = (
                freeze_fim_native_replay(args)
                if args.fim_native_stage == "freeze"
                else replay_fim_native(args)
            )
        except (FileNotFoundError, FileExistsError, TimeoutError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, sort_keys=True))
        return
    if any(
        value is not None
        for value in (
            args.fim_url,
            args.training_plan,
            args.cases,
            args.tokenizer,
            args.binary,
            args.conversion,
            args.pid,
        )
    ):
        parser.error("native FIM arguments require --fim-native-stage")
    if args.rust_url:
        measure_rust(args)
        return
    if args.model is None or args.runtime is None:
        parser.error("legacy replay requires --model and --runtime")
    if (args.states is None) != (args.states_sha256 is None):
        raise ValueError("editor replay requires both states and their fingerprint")
    if not 1 <= args.max_new_tokens <= 96:
        raise ValueError("editor replay output cap must be within 1..96")
    states = load_frozen_states(args.states, args.states_sha256) if args.states else source_states()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "measurements.jsonl").exists():
        raise FileExistsError(
            "replay output already exists; use a new suite or condition directory"
        )
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if (
        subprocess.check_output(
            ["git", "-C", str(args.runtime), "rev-parse", "HEAD"], text=True
        ).strip()
        != "f072b103714dfa1eee531f80b24512faf38e3dd2"
    ):
        raise ValueError("unregistered llama.cpp revision")
    memory_baseline = pressure()
    vm_before = counters()
    command = [
        str(args.runtime / "build/bin/llama-server"),
        "-m",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-t",
        "4",
        "-tb",
        "4",
        "-ngl",
        "0",
        "-c",
        "2304",
        "-b",
        "256",
        "-ub",
        "64",
        "-np",
        "1",
        "--cache-ram",
        "128",
        "--ctx-checkpoints",
        "32",
        "--cache-idle-slots" if args.diagnostic_saved_idle_cache else "--no-cache-idle-slots",
        "--no-warmup",
        "--perf",
    ]
    # Active context checkpoints remain at 32. Saved idle slots are optional;
    # compare them separately because lowering checkpoint count loses reuse.
    log = (args.output / "server.log").open("w")
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    sampler_stop = threading.Event()
    memory_samples = []

    def sample() -> None:
        while not sampler_stop.wait(0.05):
            memory_samples.append(proc_memory(process.pid))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{args.port}"
    records: list[dict] = []
    try:
        for _ in range(900):
            if process.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError("model load exceeded 180 seconds")
        load_seconds = time.perf_counter() - start
        previous_ids: list[int] = []
        for state in states:
            tokens_response = httpx.post(
                url + "/tokenize",
                timeout=30,
                json={"content": state["prompt"], "add_special": True, "parse_special": True},
            )
            tokens_response.raise_for_status()
            input_ids = tokens_response.json()["tokens"]
            if len(input_ids) + args.max_new_tokens > 2304:
                raise ValueError("source exceeds frozen context budget")
            common = 0
            for left, right in zip(previous_ids, input_ids, strict=False):
                if left != right:
                    break
                common += 1
            previous_ids = input_ids
            for repetition in range(2):
                before = proc_memory(process.pid)
                run = current_run_context()
                case_scope = (
                    run.for_case(state["id"] + f"/rep-{repetition}").activate()
                    if run
                    else nullcontext()
                )
                with case_scope:
                    result = generate(
                        url, state["prompt"], cache_prompt=True, max_new_tokens=args.max_new_tokens
                    )
                after = proc_memory(process.pid)
                records.append(
                    {
                        "state_id": state["id"],
                        "operation": state["operation"],
                        "file_id": state["file_id"],
                        "state_sha256": state["state_sha256"],
                        "source_target": state["source_target"],
                        "input_tokens": len(input_ids),
                        "token_prefix_with_previous_state": common,
                        "repetition": repetition,
                        "process_cold_request": len(records) == 0,
                        "before_memory": before,
                        "after_memory": after,
                        **result,
                    }
                )
                with (args.output / "measurements.jsonl").open("a") as handle:
                    handle.write(json.dumps(records[-1]) + "\n")
        vm_after = counters()
        peak = {
            key: max((row.get(key, 0) for row in memory_samples), default=0)
            for key in {key for row in memory_samples for key in row}
        }
        summary = {
            "alias": args.alias,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": subprocess.check_output(["lscpu"], text=True),
            "ram": Path("/proc/meminfo").read_text(),
            "runtime_revision": "f072b103714dfa1eee531f80b24512faf38e3dd2",
            "model_sha256": file_sha(args.model),
            "model_bytes": args.model.stat().st_size,
            "threads": 4,
            "slots": 1,
            "context_tokens": 2304,
            "cache_ram_mib": 128,
            "context_checkpoints": 32,
            "batch_tokens": 256,
            "microbatch_tokens": 64,
            "saved_idle_slots": args.diagnostic_saved_idle_cache,
            "model_load_seconds": load_seconds,
            "peak_memory": peak,
            "post_request_memory": proc_memory(process.pid),
            "pressure_before": memory_baseline,
            "pressure_after": pressure(),
            "vmstat_delta": {key: vm_after[key] - vm_before[key] for key in vm_before},
            "requests": len(records),
            "states_sha256": args.states_sha256,
            "output_token_limit": args.max_new_tokens,
            "observed_at": datetime.now(UTC).isoformat(),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        sampler_stop.set()
        thread.join(timeout=2)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def measure_rust(args) -> None:
    """Replay exact editor states through the resident context and inference worker."""
    from measure_r2_local import NativeProvider

    from tinycomplete.observability.runs import run_scope

    if not args.plan or not args.states or not args.ssh_host or not args.setting:
        raise ValueError("Rust replay needs a frozen plan, states, setting and configured SSH host")
    if args.ssh_host != "thinkpad" or args.rust_url != "http://127.0.0.1:19194":
        raise ValueError("this campaign allows only the configured ThinkPad tunnel")
    plan = json.loads(args.plan.read_text())
    root = Path(__file__).resolve().parents[1]
    for source, expected in plan["source_files"].items():
        if file_sha(root / source) != expected:
            raise ValueError("producer source fingerprint differs: " + source)
    if file_sha(args.states) != plan["states_sha256"]:
        raise ValueError("editor state fingerprint differs")
    setting = next(s for s in plan["settings"] if s["id"] == args.setting)
    if setting["alias"] != args.alias:
        raise ValueError("model setting differs")
    if args.output.exists():
        raise FileExistsError("preserve old results; use a new result directory")
    args.output.mkdir(parents=True)
    health = httpx.get(args.rust_url + "/health", timeout=20).json()
    for key in ("threads", "prompt_threads", "input_tokens", "cache_type"):
        if health[key] != setting[key]:
            raise ValueError("running setting differs: " + key)
    if health["model_sha256"] != plan["models"][args.alias]["sha256"]:
        raise ValueError("running model differs")
    binary_check = subprocess.check_output(
        ["ssh", args.ssh_host, "sha256sum " + shlex.quote(plan["binary"])], text=True
    ).split()[0]
    if binary_check != plan["binary_sha256"]:
        raise ValueError("native executable differs from plan")
    service_unit = plan.get("service_unit", "tabcomplete-engine-smoke")
    if service_unit not in ("tabcomplete-engine-smoke", "tabcomplete-engine"):
        raise ValueError("unapproved inference service unit")
    if plan.get("context_owned_by_runtime"):
        if health.get("context_layout") != setting.get("context_layout"):
            raise ValueError("resident context layout differs")
    # Sampling runs on the inference host. Controller memory is never substituted.
    remote_code = """import json,pathlib,subprocess,socket
pid=subprocess.check_output(["systemctl","--user","show",SERVICE_UNIT,
 "--property=MainPID","--value"],text=True).strip()
p=pathlib.Path("/proc")/pid
result={"host":socket.gethostname(),"pid":int(pid),
 "memory":(p/"smaps_rollup").read_text(),"status":(p/"status").read_text(),
 "pressure":{n:pathlib.Path("/proc/pressure/"+n).read_text() for n in ["memory","io"]},
 "vmstat":{a[0]:int(a[1]) for l in pathlib.Path("/proc/vmstat").read_text().splitlines()
  if (a:=l.split())[0] in ["pswpin","pswpout","pgmajfault"]},
 "frequency_khz":{str(q):q.read_text().strip()
  for q in pathlib.Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq")},
 "temperatures":{str(q):q.read_text().strip()
  for q in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp")}}
print(json.dumps(result))
"""

    remote_code = remote_code.replace("SERVICE_UNIT", repr(service_unit))

    def snapshot():
        return json.loads(
            subprocess.check_output(
                ["ssh", args.ssh_host, "python3 -c " + shlex.quote(remote_code)], text=True
            )
        )

    before = snapshot()
    if before["host"] != plan["inference_host"]:
        raise ValueError("inference host identity differs")
    provider = NativeProvider(args.rust_url, args.alias)
    provider.timeout_seconds = 90
    provider.cache = True
    rows = [json.loads(line) for line in args.states.read_text().splitlines()]
    quality_path = args.plan.parent / plan["quality_states_file"]
    if file_sha(quality_path) != plan["quality_states_sha256"]:
        raise ValueError("development fixture fingerprint differs")
    quality_rows = [json.loads(line) for line in quality_path.read_text().splitlines()]
    deadline = time.monotonic() + plan["max_setting_seconds"]
    results = []
    continuation = plan.get("continuation")
    if continuation:
        prior_path = Path(continuation["measurements_file"])
        if file_sha(prior_path) != continuation["measurements_sha256"]:
            raise ValueError("continuation result fingerprint differs")
        prior_plan = Path(continuation["plan_file"])
        if file_sha(prior_plan) != continuation["plan_sha256"]:
            raise ValueError("continuation plan fingerprint differs")
        previous = json.loads(prior_plan.read_text())
        for key in ("binary_sha256", "models", "states_sha256", "quality_states_sha256"):
            if previous[key] != plan[key]:
                raise ValueError("continuation identity differs: " + key)
        old_setting = next(s for s in previous["settings"] if s["id"] == continuation["setting_id"])
        if {k: v for k, v in old_setting.items() if k != "id"} != {
            k: v for k, v in setting.items() if k != "id"
        }:
            raise ValueError("continuation runtime settings differ")
        if health["runtime_config_hash"] != continuation["runtime_config_hash"]:
            raise ValueError("continuation resident runtime differs")
        results = [json.loads(line) for line in prior_path.read_text().splitlines()]
        if any(r.get("model_sha256") != health["model_sha256"] for r in results):
            raise ValueError("continuation model differs")
    completed = replay_completed_keys(results)
    allowed = {
        (row["id"], repetition, condition)
        for row in rows
        for repetition in (0, 1)
        for condition in ("changed_editor_state", "identical_prompt_repeat")
    } | {(row["id"], 2, "development_check") for row in quality_rows}
    if not completed <= allowed:
        raise ValueError("continuation contains an unknown fixture request")
    with run_scope(args.output / "observability-run.json", "rust-editor-replay") as run:
        for repetition in range(3):
            states = (
                rows
                if repetition < 2
                else [
                    {
                        "id": row["id"],
                        "operation": "development_" + row["expected_action"]["kind"],
                        "source_bucket": None,
                        "expected_action": row["expected_action"],
                        "request": {
                            "state": row["state"],
                            "buffers": [],
                            "repository_identity": "synthetic:rust-editor-development",
                        },
                    }
                    for row in quality_rows
                ]
            )
            for state in states:
                conditions = (
                    ("changed_editor_state", "identical_prompt_repeat")
                    if repetition < 2
                    else ("development_check",)
                )
                if all((state["id"], repetition, c) in completed for c in conditions):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("frozen inference deadline reached")
                request = state["request"]
                started = time.perf_counter()
                response = httpx.post(
                    args.rust_url + "/v1/editor/context", json=request, timeout=20
                )
                if response.status_code == 422 and repetition == 2:
                    result = {
                        "id": state["id"],
                        "condition": "development_check",
                        "expected_action": state["expected_action"],
                        "canonical_action": None,
                        "stop_type": None,
                        "context_rejected": True,
                    }
                    results.append(result)
                    with (args.output / "measurements.jsonl").open("a") as handle:
                        handle.write(json.dumps(result) + "\n")
                    continue
                response.raise_for_status()
                prepared = response.json()
                if setting.get("context_layout") == "cursor-last-v1" and not plan.get(
                    "context_owned_by_runtime"
                ):
                    prepared["prompt"] = cursor_last_prompt(prepared["prompt"])
                    tokenized = httpx.post(
                        args.rust_url + "/tokenize",
                        timeout=20,
                        json={"content": prepared["prompt"], "add_special": True},
                    )
                    tokenized.raise_for_status()
                    prepared["prompt_tokens"] = len(tokenized.json()["tokens"])
                    if prepared["prompt_tokens"] > health["input_tokens"]:
                        raise ValueError("same-information layout exceeds input budget")
                    prepared["context_hash"] = sha(prepared["prompt"].encode())
                    prepared["context_policy_version"] = "single-line-cursor-last-context-v1"
                construction_ms = (time.perf_counter() - started) * 1000
                if sha(prepared["prompt"].encode()) != prepared["context_hash"]:
                    raise ValueError("context hash mismatch")
                provider.editor_window = prepared.get("window")
                provider.repository_identity = request["repository_identity"]
                for condition in conditions:
                    if (state["id"], repetition, condition) in completed:
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("frozen inference deadline reached")
                    provider.timeout_seconds = min(90, remaining)
                    case_id = f"{state['id']}/{repetition}/{condition}"
                    context = run.for_case(case_id) if run else nullcontext()
                    with context.activate() if run else context:
                        provider.generate_detailed(prepared["prompt"], health["output_tokens"])
                    result = {
                        "id": state["id"],
                        "operation": state["operation"],
                        "source_bucket": state["source_bucket"],
                        "repetition": repetition,
                        "condition": condition,
                        "context_construction_ms": construction_ms,
                        "actual_input_tokens": prepared["prompt_tokens"],
                        "context_policy_version": prepared["context_policy_version"],
                        "prompt_sha256": prepared["context_hash"],
                        **provider.last,
                    }
                    if repetition == 2:
                        result["expected_action"] = state["expected_action"]
                    if run:
                        result["run_id"] = run.run_id
                    results.append(result)
                    with (args.output / "measurements.jsonl").open("a") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "plan_sha256": file_sha(args.plan),
        "setting": setting,
        "identity": health,
        "controller_host": socket.gethostname(),
        "transport": "owned SSH tunnel",
        "before": before,
        "after": snapshot(),
        "requests": len(results),
        "conditions": {},
        "quality_claim": False,
        "continuation": continuation,
        "memory_window": "current invocation snapshots; VmHWM includes earlier resident requests",
    }
    for condition in ("changed_editor_state", "identical_prompt_repeat"):
        values = sorted(r["total_seconds"] * 1000 for r in results if r["condition"] == condition)
        subset = [r for r in results if r["condition"] == condition]
        summary["conditions"][condition] = {
            "count": len(values),
            "median_ms": values[len(values) // 2],
            "p95_ms": values[min(len(values) - 1, (95 * len(values) + 99) // 100 - 1)],
            "actual_eos": sum(r["stop_type"] == "eos" for r in subset),
            "canonical_valid": sum(r["canonical_action"] is not None for r in subset),
            "cache_tokens": sum(r["server_timings"].get("cache_n", 0) for r in subset),
            "recomputed_tokens": sum(r["server_timings"].get("prompt_n", 0) for r in subset),
        }
    quality = [r for r in results if r["condition"] == "development_check"]

    def exact(row):
        action = row["canonical_action"]
        expected = row["expected_action"]
        return bool(
            action is not None
            and action["kind"] == expected["kind"]
            and action.get("text") == expected.get("text")
        )

    summary["development"] = {
        "count": len(quality),
        "canonical_valid": sum(r["canonical_action"] is not None for r in quality),
        "actual_eos": sum(r["stop_type"] == "eos" for r in quality),
        "exact_actions": sum(exact(r) for r in quality),
        "by_action": {
            kind: {
                "count": sum(r["expected_action"]["kind"] == kind for r in quality),
                "exact": sum(exact(r) for r in quality if r["expected_action"]["kind"] == kind),
            }
            for kind in ("keep", "replace_line", "insert_before", "delete_line")
        },
        "no_edit_false_changes": sum(
            r["expected_action"]["kind"] == "keep"
            and r["canonical_action"] is not None
            and r["canonical_action"]["kind"] != "keep"
            for r in quality
        ),
        "scope": (
            "24 preselected existing development states; synthetic cases are codec "
            "regressions, not human-edit quality"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def replay_completed_keys(rows: list[dict]) -> set[tuple[str, int, str]]:
    """Only exact recorded request identities can be skipped in a declared continuation."""
    keys: set[tuple[str, int, str]] = set()
    for row in rows:
        condition = row["condition"]
        repetition = row.get("repetition", 2 if condition == "development_check" else -1)
        if condition not in {
            "changed_editor_state",
            "identical_prompt_repeat",
            "development_check",
        }:
            raise ValueError("unknown recorded replay condition")
        if repetition not in (0, 1, 2) or (repetition == 2) != (condition == "development_check"):
            raise ValueError("invalid recorded replay repetition")
        key = (row["id"], repetition, condition)
        if key in keys:
            raise ValueError("duplicate recorded replay request")
        keys.add(key)
    return keys


def cursor_last_prompt(prompt: str) -> str:
    """One declared same-information layout experiment; never trim source/markers."""
    lines = prompt.split("\n")
    cursor = [line for line in lines if line.startswith("Cursor byte column: ")]
    if len(cursor) != 1 or lines[-1] != "Action:":
        raise ValueError("unexpected control prompt layout")
    lines.remove(cursor[0])
    lines.insert(len(lines) - 1, cursor[0])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
