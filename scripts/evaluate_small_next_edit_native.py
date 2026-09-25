"""Evaluate a frozen compact next-edit split with the pinned local Q4 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from replay_small_model import counters, pressure, proc_memory

from tinycomplete.eval.compact_next_edit import decode
from tinycomplete.eval.next_edit_protocol import apply_next_edit_action

RUNTIME_REVISION = "f072b103714dfa1eee531f80b24512faf38e3dd2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def request(url: str, prompt: str) -> dict:
    pieces: list[str] = []
    tokens: list[int] = []
    final: dict = {}
    started = time.perf_counter()
    with httpx.stream(
        "POST",
        url + "/completion",
        timeout=120,
        json={
            "prompt": prompt,
            "n_predict": 96,
            "temperature": 0,
            "seed": 928173,
            "stream": True,
            "return_tokens": True,
            "cache_prompt": True,
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
            else:
                pieces.append(event.get("content", ""))
                tokens.extend(event.get("tokens", []))
    return {
        "text": "".join(pieces),
        "tokens": len(tokens),
        "stop_type": final.get("stop_type"),
        "timings": final.get("timings", {}),
        "latency_seconds": time.perf_counter() - started,
    }


def score(row: dict, response: dict) -> dict:
    eos = response["stop_type"] == "eos"
    parsed = decode(
        response["text"],
        finish_reason="eos" if eos else "length",
        generated_tokens=response["tokens"],
    )
    predicted = parsed.action
    correct = False
    predicted_after = None
    if predicted is not None:
        try:
            predicted_after = apply_next_edit_action(
                row["current"], row["region_start"], row["region_end"], predicted
            )
            correct = predicted_after == row["after"]
        except ValueError:
            pass
    return {
        "id": row["id"],
        "gold_action": row["action"],
        "predicted_action": predicted.action if predicted else None,
        "predicted_text": predicted.text if predicted and predicted.action == "replace" else None,
        "raw_output": response["text"],
        "parse_status": parsed.status,
        "terminated": eos,
        "exact_after_state": correct,
        "false_positive": row["action"] == "no_edit"
        and predicted is not None
        and predicted.action == "replace",
        "unnecessary_change": row["action"] == "no_edit"
        and predicted_after is not None
        and predicted_after != row["current"],
        "reverses_latest_edit": predicted_after is not None
        and row["history_before"] != row["current"]
        and predicted_after == row["history_before"],
        "truncated": response["stop_type"] == "limit",
        "output_tokens": response["tokens"],
        "output_bytes": len(response["text"].encode()),
        "latency_seconds": response["latency_seconds"],
        "stop_type": response["stop_type"],
        "timings": response["timings"],
    }


def summarize(records: list[dict]) -> dict:
    by_action: dict[str, dict[str, int]] = {}
    for row in records:
        bucket = by_action.setdefault(
            row["gold_action"],
            {"total": 0, "valid": 0, "terminated": 0, "exact_after_state": 0, "false_positive": 0},
        )
        bucket["total"] += 1
        bucket["valid"] += row["parse_status"] == "ok"
        bucket["terminated"] += row["terminated"]
        bucket["exact_after_state"] += row["exact_after_state"]
        bucket["false_positive"] += row["false_positive"]
    edits = [row for row in records if row["gold_action"] != "no_edit"]
    latencies = sorted(row["latency_seconds"] for row in records)
    output_tokens = sorted(row["output_tokens"] for row in records)
    def percentile(values: list[float] | list[int], p: float) -> float | None:
        if not values:
            return None
        return float(values[min(len(values) - 1, math.ceil(len(values) * p) - 1)])
    return {
        "cases": len(records),
        "valid": sum(row["parse_status"] == "ok" for row in records),
        "terminated": sum(row["terminated"] for row in records),
        "edit_required": len(edits),
        "edit_required_exact_after_state": sum(row["exact_after_state"] for row in edits),
        "no_edit_total": by_action.get("no_edit", {}).get("total", 0),
        "no_edit_false_positive": by_action.get("no_edit", {}).get("false_positive", 0),
        "unnecessary_changes": sum(row["unnecessary_change"] for row in records),
        "reversals_of_latest_edit": sum(row["reverses_latest_edit"] for row in records),
        "truncations": sum(row["truncated"] for row in records),
        "output_tokens_mean": statistics.mean(output_tokens) if output_tokens else None,
        "output_tokens_p95": percentile(output_tokens, 0.95),
        "completed_action_median_seconds": statistics.median(latencies) if latencies else None,
        "completed_action_p95_seconds": percentile(latencies, 0.95),
        "by_action": by_action,
        "score_kind": "synthetic exact after-state; no user-acceptance claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19095)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evaluation output already exists")
    if sha(args.model) != args.model_sha256 or sha(args.data) != args.data_sha256:
        raise ValueError("model or split identity mismatch")
    revision = subprocess.check_output(
        ["git", "-C", str(args.runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RUNTIME_REVISION:
        raise ValueError("unregistered runtime revision")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    args.output.mkdir(parents=True)
    command = [
        str(args.runtime / "build/bin/llama-server"), "-m", str(args.model),
        "--host", "127.0.0.1", "--port", str(args.port), "-t", "4", "-tb", "4",
        "-ngl", "0", "-c", "2304", "-b", "256", "-ub", "64", "-np", "1",
        "--cache-ram", "128", "--ctx-checkpoints", "32", "--no-cache-idle-slots",
        "--no-warmup", "--perf",
    ]
    log = (args.output / "server.log").open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{args.port}"
    records = []
    memory = []
    vm_before = counters()
    pressure_before = pressure()
    try:
        for _ in range(900):
            if process.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    break
            except (httpx.HTTPError, OSError):
                pass
            time.sleep(0.1)
        else:
            raise TimeoutError("server not ready")
        with (args.output / "predictions.jsonl").open("w") as handle:
            for row in rows:
                response = request(url, row["prompt"])
                record = score(row, response)
                records.append(record)
                memory.append(proc_memory(process.pid))
                handle.write(json.dumps(record) + "\n")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
    report = {
        "observed_at": datetime.now(UTC).isoformat(),
        "model_sha256": args.model_sha256,
        "data_sha256": args.data_sha256,
        "runtime_revision": revision,
        "precision": "Q4_K_M",
        "summary": summarize(records),
        "peak_sampled_rss_bytes": max((row.get("Rss", 0) for row in memory), default=0),
        "post_request_memory": memory[-1] if memory else {},
        "swap_and_major_fault_delta": {
            key: counters().get(key, 0) - value for key, value in vm_before.items()
        },
        "pressure_before": pressure_before,
        "pressure_after": pressure(),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
