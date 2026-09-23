"""Isolated llama.cpp CPU inference; exact source prompts and token-arrival evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import psutil

from tinycomplete.eval.code_generation import DetailedGeneration
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
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


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
        self.last = {}

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
            "returned_text": text.split("\n", 1)[0],
            "token_ids": tokens,
            "token_arrival_seconds": arrivals,
            "token_arrival_definition": (
                "client receipt of identified output token, not kernel timestamp"
            ),
            "returned_line_seconds": line_at
            if line_at is not None
            else (total if final.get("stop_type") == "eos" else None),
            "total_seconds": total,
            "server_timings": timings,
            "tokens_cached": final.get("tokens_cached"),
            "tokens_evaluated": final.get("tokens_evaluated"),
            "stop_type": final.get("stop_type"),
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
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
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
    if target.exists():
        raise ValueError("measurement output exists; do not mix process-cold attempts")
    log = (args.output / "server.log").open("w")
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
        save(args.output / "metadata.json", metadata)
        provider = NativeProvider(url, args.alias)
        rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]
        first = True
        with run_scope(args.output / "observability-run.json", "local-inference") as run:
            for row in rows:
                if args.bucket and row["bucket"] != args.bucket:
                    continue
                for repetition in range(2):
                    context = (
                        run or RunContext.new(campaign_id="tabcomplete-model-data-r2")
                    ).for_case(row["case_id"])
                    with context.activate():
                        provider.generate_detailed(row["prompt"], 32)
                    result = {k: v for k, v in row.items() if k != "prompt"}
                    result.update(
                        provider.last,
                        repetition=repetition,
                        model_alias=args.alias,
                        process_cold_first=first,
                        peak_resident_bytes=peak[0],
                    )
                    with target.open("a") as handle:
                        handle.write(json.dumps(result) + "\n")
                    first = False
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
            peak_resident_bytes=peak[0], status="complete", measurement_sha256=sha(target)
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
    args = parser.parse_args()
    if args.prepare_suite:
        if args.prompts.exists():
            raise ValueError("prompt fixture already exists")
        prepare_prompts(args.prepare_suite, args.prompts)
    else:
        measure(args)


if __name__ == "__main__":
    main()
