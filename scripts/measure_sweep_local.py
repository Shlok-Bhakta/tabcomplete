#!/usr/bin/env python3
"""Measure the frozen Sweep prompt bundle on crabcake's CPU-only llama.cpp runtime.

This runner is intentionally separate from the Kaggle comparison runner. It never
downloads or converts weights, never connects to the resident editor service, and
requires ``--execute`` after the local plan has been frozen.
"""

from __future__ import annotations

import argparse
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


def validate_plan_policy(plan: dict[str, Any]) -> None:
    if plan.get("runtime", {}).get("server_settings") != SERVER_SETTINGS:
        raise ValueError("server configuration differs from the frozen local replay plan")
    replay = plan.get("replay", {})
    if replay.get("request_settings") != REQUEST_SETTINGS:
        raise ValueError("request configuration differs from the frozen local replay plan")
    if replay.get("precision_order") != ["q4_k_m", "q8_0"]:
        raise ValueError("local replay must run Q4_K_M then Q8_0, one model at a time")
    if replay.get("repetitions") != 2:
        raise ValueError("local replay must use exactly two repetitions")
    if replay.get("fixture_count") != len(EXPECTED_CASE_IDS):
        raise ValueError("local replay plan must contain the frozen 24 source states")
    if replay.get("fixture_order") != list(EXPECTED_CASE_IDS):
        raise ValueError("local replay fixture order differs from the frozen source states")
    if replay.get("expected_requests") != len(EXPECTED_CASE_IDS) * 2 * 2 * 2:
        raise ValueError("local replay request count must bind both precisions and cache repeats")
    if replay.get("request_order_per_fixture_per_repetition") != [
        "changed_state",
        "immediate_same_prompt_repeat",
    ]:
        raise ValueError("local replay cache-repeat ordering differs from the frozen plan")
    limits = plan.get("limits", {})
    if limits.get("max_wall_seconds_including_preflight_and_finalization") != MAX_WALL_SECONDS:
        raise ValueError("local replay wall budget differs from the 90-minute campaign cap")
    if limits.get("finalization_reserve_seconds") != FINALIZATION_RESERVE_SECONDS:
        raise ValueError("local replay finalization reserve must remain ten minutes")
    if limits.get("request_work_deadline_seconds_from_invocation_start") != (
        MAX_WALL_SECONDS - FINALIZATION_RESERVE_SECONDS
    ):
        raise ValueError("local replay request deadline does not reserve finalization time")
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


def load_and_verify_inputs(
    plan_path: Path, upstream_plan_path: Path, fixtures_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "sweep-local-replay-plan-v1" or plan_digest(plan) != plan.get(
        "plan_sha256"
    ):
        raise ValueError("local replay plan schema or self-hash mismatch")
    validate_plan_policy(plan)
    upstream = json.loads(upstream_plan_path.read_text(encoding="utf-8"))
    if upstream.get("plan_sha256") != plan["inputs"]["upstream_plan_sha256"]:
        raise ValueError("upstream Sweep plan identity differs")
    if digest_file(upstream_plan_path) != plan["inputs"]["upstream_plan_file_sha256"]:
        raise ValueError("upstream Sweep plan bytes differ")
    if digest_file(fixtures_path) != plan["inputs"]["prompt_bundle_sha256"]:
        raise ValueError("frozen prompt-only bundle identity differs")
    sweep = _sweep_runner_module()
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
    if tuple(row.get("case_id") for row in rows) != EXPECTED_CASE_IDS:
        raise ValueError("prompt bundle case order or identity differs")
    for row in rows:
        prompt = sweep.build_sweep_prompt(row)
        if digest_bytes(prompt.encode("utf-8")) != row.get("prompt_sha256"):
            raise ValueError(f"publisher prompt hash differs for {row.get('case_id')}")
        if digest_bytes(row["current_content"].encode("utf-8")) != row.get("current_sha256"):
            raise ValueError(f"current source hash differs for {row.get('case_id')}")
        if digest_bytes(row["original_content"].encode("utf-8")) != row.get("original_sha256"):
            raise ValueError(f"original source hash differs for {row.get('case_id')}")
    return plan, upstream, rows


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


def _tokenize(server_url: str, prompt: str, *, timeout_seconds: float = 60.0) -> tuple[int, float]:
    import httpx

    started = time.perf_counter()
    response = httpx.post(
        server_url + "/tokenize",
        json={"content": prompt, "add_special": True, "parse_special": True},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
        raise ValueError("runtime tokenizer endpoint returned invalid token IDs")
    return len(tokens), (time.perf_counter() - started) * 1000


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
            for fixture in fixtures:
                tokenization_remaining = campaign_deadline - time.monotonic()
                if tokenization_remaining <= 0:
                    raise TimeoutError(
                        "local replay finalization reserve reached during tokenization"
                    )
                prompt = sweep.build_sweep_prompt(fixture)
                input_tokens, tokenize_ms = _tokenize(
                    url, prompt, timeout_seconds=min(60.0, tokenization_remaining)
                )
                if input_tokens + OUTPUT_TOKENS > CONTEXT_TOKENS:
                    raise ValueError(
                        f"frozen case {fixture['case_id']} exceeds input-plus-output context"
                    )
                token_rows.append(
                    {
                        "case_id": fixture["case_id"],
                        "prompt_sha256": fixture["prompt_sha256"],
                        "input_tokens": input_tokens,
                        "tokenization_ms": tokenize_ms,
                    }
                )
                prompts.append(prompt)
            if len(token_rows) != 24:
                raise ValueError("local replay requires all 24 frozen source states")
            process_memory_before = _proc_memory(process.pid)
            prediction_path = run_dir / f"predictions-{precision}.jsonl"
            primary_output_hashes: dict[tuple[int, str], str] = {}
            with prediction_path.open("x", encoding="utf-8") as output:
                for repetition in range(2):
                    for fixture, prompt, token_row in zip(
                        fixtures, prompts, token_rows, strict=True
                    ):
                        if time.monotonic() >= campaign_deadline:
                            raise TimeoutError("local replay reached its frozen four-hour cap")
                        input_tokens = token_row["input_tokens"]
                        for kind in ("changed_state", "immediate_same_prompt_repeat"):
                            sample_start = sampler.sample()
                            request_context = (
                                run_context or RunContext.new(campaign_id=OBSERVABILITY_CAMPAIGN)
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
                                        "tabcomplete.suite_version": "sweep-local-replay-r1",
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
                            result.update(
                                schema="sweep-local-replay-record-v1",
                                plan_sha256=plan["plan_sha256"],
                                precision=precision,
                                model_sha256=model_info["sha256"],
                                case_id=fixture["case_id"],
                                repetition=repetition,
                                request_kind=kind,
                                prompt_sha256=fixture["prompt_sha256"],
                                input_tokens=input_tokens,
                                input_plus_output_ceiling=input_tokens + OUTPUT_TOKENS,
                                tokenization_ms=token_row["tokenization_ms"],
                                editable_range={
                                    "start_byte": fixture["editable_start_byte"],
                                    "end_byte": fixture["editable_end_byte"],
                                },
                                output_file_mapping=(
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
                                ),
                                request_started_at_utc=request_started_utc,
                                measurement_order=request_index,
                                measurement_memory=sampler.window_summary(sample_start, sample_end),
                            )
                            if kind == "changed_state":
                                primary_output_hashes[(repetition, fixture["case_id"])] = result[
                                    "output_sha256"
                                ]
                            else:
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
                "repetitions": 2,
                "primary_and_cache_repeat_requests": len(fixtures) * 2 * 2,
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
        with run_scope(run_dir / "observability-run.json", OBSERVABILITY_CAMPAIGN) as run:
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
        report = {
            "schema": "sweep-local-replay-summary-v1",
            "plan_sha256": plan["plan_sha256"],
            "run_directory": str(run_dir),
            "completed_requests": len(records),
            "expected_requests": 24 * 2 * 2 * 2,
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
