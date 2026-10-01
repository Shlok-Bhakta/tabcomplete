"""Bounded, single-slot native replay on the identified inference host."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import threading
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tinycomplete.observability.context import current_run_context


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19094)
    parser.add_argument("--diagnostic-saved-idle-cache", action="store_true")
    parser.add_argument("--states", type=Path, help="optional frozen trained-format editor states")
    parser.add_argument("--states-sha256")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
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
                    if run else nullcontext()
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


if __name__ == "__main__":
    main()
