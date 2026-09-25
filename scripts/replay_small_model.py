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
from datetime import UTC, datetime
from pathlib import Path

import httpx


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
    """Twelve exact synthetic editor states; same bytes for every tokenizer."""
    rows = []
    for target, count in ((512, 29), (1024, 58), (2048, 116)):
        base = "".join(f"def f_{i}(value):\n    return value + {i}\n" for i in range(count))
        states = [
            ("fresh_open", base + "def compute(value):\n    return "),
            ("append_chars", base + "def compute(value):\n    return value + "),
            ("reject_then_type_different", base + "def compute(value):\n    return value - "),
            (
                "earlier_edit_switch_return",
                base.replace("return value + 1", "return value + 2", 1)
                + "def compute(value):\n    return value - ",
            ),
        ]
        for index, (operation, prompt) in enumerate(states):
            rows.append(
                {
                    "id": f"synthetic-{target}-{index}",
                    "source_target": target,
                    "operation": operation,
                    "prompt": prompt,
                    "state_sha256": sha(prompt.encode()),
                }
            )
    return rows


def generate(url: str, prompt: str, *, cache_prompt: bool) -> dict:
    start = time.perf_counter()
    pieces, tokens, times, final = [], [], {}, {}
    with httpx.stream(
        "POST",
        url + "/completion",
        timeout=120,
        json={
            "prompt": prompt,
            "n_predict": 24,
            "temperature": 0,
            "stream": True,
            "return_tokens": True,
            "cache_prompt": cache_prompt,
            "seed": 928173,
            "id_slot": 0,
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("stop"):
                final = event
                continue
            pieces.append(event.get("content", ""))
            for token in event.get("tokens", []):
                tokens.append(token)
                if len(tokens) in (1, 8, 16):
                    times[str(len(tokens))] = time.perf_counter() - start
    return {
        "response": "".join(pieces),
        "response_sha256": sha("".join(pieces).encode()),
        "output_token_ids": tokens,
        "token_arrival_seconds": times,
        "completed_seconds": time.perf_counter() - start,
        "tokens_cached": final.get("tokens_cached"),
        "tokens_evaluated": final.get("tokens_evaluated"),
        "stop_type": final.get("stop_type"),
        "server_timings": final.get("timings", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19094)
    parser.add_argument("--diagnostic-no-saved-cache", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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
        "2" if args.diagnostic_no_saved_cache else "32",
        "--no-cache-idle-slots",
        "--no-warmup",
        "--perf",
    ]
    # Keep the minimum tested checkpoint count that retains active sequence reuse.
    # Zero or one checkpoints, or cache-ram=0, removes it in this runtime.
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
        for state in source_states():
            tokens_response = httpx.post(
                url + "/tokenize",
                timeout=30,
                json={"content": state["prompt"], "add_special": True, "parse_special": True},
            )
            tokens_response.raise_for_status()
            input_ids = tokens_response.json()["tokens"]
            if len(input_ids) + 24 > 2304:
                raise ValueError("source exceeds frozen context budget")
            common = 0
            for left, right in zip(previous_ids, input_ids, strict=False):
                if left != right:
                    break
                common += 1
            previous_ids = input_ids
            for repetition in range(2):
                before = proc_memory(process.pid)
                result = generate(
                    url, state["prompt"], cache_prompt=not args.diagnostic_no_saved_cache
                )
                after = proc_memory(process.pid)
                records.append(
                    {
                        "state_id": state["id"],
                        "operation": state["operation"],
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
            "context_checkpoints": 2 if args.diagnostic_no_saved_cache else 32,
            "batch_tokens": 256,
            "microbatch_tokens": 64,
            "diagnostic_no_saved_cache": args.diagnostic_no_saved_cache,
            "model_load_seconds": load_seconds,
            "peak_memory": peak,
            "post_request_memory": proc_memory(process.pid),
            "pressure_before": memory_baseline,
            "pressure_after": pressure(),
            "vmstat_delta": {key: vm_after[key] - vm_before[key] for key in vm_before},
            "requests": len(records),
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
