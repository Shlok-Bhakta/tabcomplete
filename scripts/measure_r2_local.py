"""Isolated llama.cpp CPU inference; exact source prompts and token-arrival evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psutil

from tinycomplete.eval.code_generation import DetailedGeneration, returned_first_line
from tinycomplete.observability.context import RunContext
from tinycomplete.observability.hooks import observed_generation
from tinycomplete.observability.runs import run_scope


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_unused_port(port):
    with socket.socket() as check:
        # TIME_WAIT is not an active service. An active listener still refuses
        # this bind, so no process occupying the experiment port is replaced.
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", port))


def wait_between_pairs(pause_file, acknowledgement, *, poll_seconds=0.2):
    """Pause only outside measured requests so CPU judging cannot contaminate timings."""
    if not pause_file.exists():
        return 0.0
    started = time.monotonic()
    save(acknowledgement, {"pid": os.getpid(), "state": "paused_between_request_pairs"})
    try:
        while pause_file.exists():
            time.sleep(poll_seconds)
    finally:
        acknowledgement.unlink(missing_ok=True)
    return time.monotonic() - started


def verify_context_budget(alias, input_tokens, output_tokens=32):
    declared = {"q35-p12": 262144, "q25-coder": 32768, "granite-h350": 32768}[alias]
    if input_tokens + output_tokens > min(declared, 40960):
        raise ValueError("source prompt plus output exceeds declared/runtime context")


def prepare_prompts(suite, destination):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", revision="8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
    )
    cases = [json.loads(line) for line in suite.read_text().splitlines()]
    rows = []
    for bucket in (2048, 8192, 32768):
        for index in range(20):
            source = ""
            cursor = index
            # Public held-out source, cyclic concatenation only for runtime measurement.
            # This is not a correctness or dependency-context benchmark.
            while len(tokenizer.encode(source, add_special_tokens=False)) < bucket - 64:
                case = cases[cursor % len(cases)]
                source += case["source_before"] + case["reference"] + case["source_after"] + "\n"
                cursor += 1
            ids = tokenizer.encode(source, add_special_tokens=False)[: bucket - 64]
            prompt = tokenizer.decode(ids, skip_special_tokens=False)
            rows.append(
                {
                    "case_id": f"runtime-{bucket}-{index:02d}",
                    "bucket": bucket,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "reference_token_count": len(
                        tokenizer.encode(prompt, add_special_tokens=False)
                    ),
                    "source_fixture": "concatenated public held-out files; latency only",
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class NativeProvider:
    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name
        self.cache = False
        self.stop_first_line = False
        self.last = {}

    def generate_line_detailed(self, prompt, max_new_tokens):
        self.stop_first_line = True
        try:
            return self.generate_detailed(prompt, max_new_tokens)
        finally:
            self.stop_first_line = False

    @observed_generation
    def generate_detailed(self, prompt, max_new_tokens):
        started = time.perf_counter()
        tokens, text, arrivals = [], "", {}
        line_at = None
        final = {}
        with httpx.stream(
            "POST",
            self.url + "/completion",
            timeout=1800,
            json={
                "prompt": prompt,
                "n_predict": max_new_tokens,
                "temperature": 0,
                "stream": True,
                "return_tokens": True,
                "cache_prompt": self.cache,
                "seed": 928173,
                "id_slot": 0,
                "stop": ["\n"] if self.stop_first_line else [],
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                elapsed = time.perf_counter() - started
                new = event.get("tokens", [])
                if not event.get("stop"):
                    tokens.extend(new)
                    if new:
                        for count in (1, 8, 16, 32):
                            if len(tokens) == count:
                                arrivals.setdefault(str(count), elapsed)
                    text += event.get("content", "")
                    if line_at is None and "\n" in text:
                        line_at = elapsed
                else:
                    final = event
        total = time.perf_counter() - started
        timings = final.get("timings", {})
        self.last = {
            "raw_response": text,
            "returned_text": returned_first_line(text),
            "token_ids": tokens,
            "token_arrival_seconds": arrivals,
            "token_arrival_definition": (
                "client receipt of identified output token, not kernel timestamp"
            ),
            "returned_line_seconds": line_at
            if line_at is not None
            else (total if final.get("stop_type") in ("eos", "word") else None),
            "total_seconds": total,
            "server_timings": timings,
            "tokens_cached": final.get("tokens_cached"),
            "tokens_evaluated": final.get("tokens_evaluated"),
            "stop_type": final.get("stop_type"),
            "stopping_word": final.get("stopping_word"),
            "server_stop_requested": self.stop_first_line,
            "truncated": final.get("truncated"),
            "cache_requested": self.cache,
        }
        return DetailedGeneration(
            text,
            timings.get("predicted_n", len(tokens)),
            final.get("stop_type"),
            input_tokens=timings.get("prompt_n"),
            usage_source="llama.cpp",
            first_output_ms=(arrivals["1"] * 1000 if "1" in arrivals else None),
        )


def measure(args):
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "-C", str(args.runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "f072b103714dfa1eee531f80b24512faf38e3dd2":
        raise ValueError("runtime revision differs from preregistration")
    verify_unused_port(args.port)
    metadata = {
        "model_alias": args.alias,
        "model_sha256": sha(args.model),
        "runtime_sha": revision,
        "prompts_sha256": sha(args.prompts),
        "threads": 4,
        "precision": "Q4_K_M",
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "ram_bytes": psutil.virtual_memory().total,
        "concurrency": 1,
        "cold_disk": False,
        "power_state": "not controlled",
        "other_processes": [
            {"name": p.info["name"], "pid": p.info["pid"]}
            for p in psutil.process_iter(["name", "pid"])
            if any(x in (p.info["name"] or "").lower() for x in ("llama", "python", "ollama"))
        ],
    }
    target = args.output / "measurements.jsonl"
    completed = set()
    previous_attempts = []
    if target.exists():
        if not args.resume:
            raise ValueError(
                "measurement output exists; explicit --resume and matching hashes required"
            )
        saved = json.loads((args.output / "metadata.json").read_text())
        for key in (
            "model_alias",
            "model_sha256",
            "runtime_sha",
            "prompts_sha256",
            "threads",
            "precision",
            "platform",
            "host",
            "ram_bytes",
            "concurrency",
        ):
            if saved[key] != metadata[key]:
                raise ValueError("measurement resume fingerprint differs: " + key)
        for line in target.read_text().splitlines():
            row = json.loads(line)
            pair = (row["case_id"], row["repetition"])
            if pair in completed:
                raise ValueError("duplicate completed measurement")
            completed.add(pair)
        previous_attempts = saved.get(
            "process_attempts",
            [
                {
                    "attempt_id": "initial-legacy",
                    "model_load_to_health_seconds": saved["model_load_to_health_seconds"],
                    "started_at": None,
                    "note": "Original records predate per-attempt timestamps; wall time unknown",
                }
            ],
        )
    attempt = {"attempt_id": str(uuid.uuid4()), "started_at": datetime.now(UTC).isoformat()}
    log = (args.output / ("server-" + attempt["attempt_id"] + ".log")).open("w")
    started = time.perf_counter()
    process = subprocess.Popen(
        [
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
            "40960",
            "-np",
            "1",
            "--no-warmup",
            "--perf",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    stopped = threading.Event()
    peak = [0]

    def sample():
        while not stopped.wait(0.02):
            try:
                peak[0] = max(peak[0], psutil.Process(process.pid).memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    url = f"http://127.0.0.1:{args.port}"
    try:
        while time.perf_counter() - started < 180:
            if process.poll() is not None:
                raise RuntimeError("isolated server failed; inspect server.log")
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError("model load exceeded 180 seconds")
        metadata["model_load_to_health_seconds"] = time.perf_counter() - started
        attempt["model_load_to_health_seconds"] = metadata["model_load_to_health_seconds"]
        metadata["process_attempts"] = [*previous_attempts, attempt]
        metadata["status"] = "running"
        save(args.output / "metadata.json", metadata)
        provider = NativeProvider(url, args.alias)
        rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]
        first = True
        with run_scope(args.output / "observability-run.json", "local-inference") as run:
            for row in rows:
                if args.bucket and row["bucket"] != args.bucket:
                    continue
                if any((row["case_id"], rep) not in completed for rep in range(2)):
                    paused = wait_between_pairs(
                        args.output.parent / "pause-requested",
                        args.output.parent / "paused.json",
                    )
                    if paused:
                        attempt.setdefault("between_pair_pauses", []).append(
                            {"before_case": row["case_id"], "seconds": paused}
                        )
                        save(args.output / "metadata.json", metadata)
                    tokenized = httpx.post(
                        url + "/tokenize",
                        json={"content": row["prompt"], "add_special": True, "parse_special": True},
                        timeout=30,
                    )
                    tokenized.raise_for_status()
                    verify_context_budget(args.alias, len(tokenized.json()["tokens"]))
                had_pending = False
                for repetition in range(2):
                    if (row["case_id"], repetition) in completed:
                        continue
                    had_pending = True
                    context = (
                        run or RunContext.new(campaign_id="tabcomplete-model-data-r2")
                    ).for_case(row["case_id"])
                    observed_at = datetime.now(UTC).isoformat()
                    with context.activate():
                        provider.generate_detailed(row["prompt"], 32)
                    result = {k: v for k, v in row.items() if k != "prompt"}
                    result.update(
                        provider.last,
                        repetition=repetition,
                        model_alias=args.alias,
                        process_cold_first=first,
                        peak_resident_bytes=peak[0],
                        process_attempt_id=attempt["attempt_id"],
                        observed_at=observed_at,
                    )
                    with target.open("a") as handle:
                        handle.write(json.dumps(result) + "\n")
                    first = False
                    completed.add((row["case_id"], repetition))
                if not had_pending:
                    continue
                print(
                    json.dumps(
                        {
                            "alias": args.alias,
                            "case": row["case_id"],
                            "seconds": provider.last["total_seconds"],
                        }
                    ),
                    flush=True,
                )
        metadata.update(
            peak_resident_bytes=peak[0],
            status="complete" if len(completed) == len(rows) * 2 else "partial_grid",
            completed_measurements=len(completed),
            planned_measurements=len(rows) * 2,
            measurement_sha256=sha(target),
        )
        save(args.output / "metadata.json", metadata)
    finally:
        stopped.set()
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        sampler.join(timeout=1)
        log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-suite", type=Path)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--alias")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=19091)
    parser.add_argument("--bucket", type=int, choices=[2048, 8192, 32768])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_suite:
        if args.prompts.exists():
            raise ValueError("prompt fixture already exists")
        prepare_prompts(args.prepare_suite, args.prompts)
    else:
        measure(args)


if __name__ == "__main__":
    main()
