"""Bounded, matched completion diagnostic with existing Qwen Q4 weights.

This is an evaluation, never a training data builder or editor installer.
Reuse the native provider, fixed line scorer and local memory sampler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from evaluate_causal_line import score_line
from measure_r2_local import NativeProvider
from measure_sweep_local import (
    MemorySampler,
    _available_port,
    _stop_process,
    _wait_server,
    build_server_argv,
)

from tinycomplete.data.fim import format_psm
from tinycomplete.observability.bootstrap import current_runtime
from tinycomplete.observability.runs import run_scope

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "7947f482e7d645aa8d3a5f863e4ba5c24ebe17d9602cb40e1cddb60e37b65c30"
MODEL_BYTES = 397_807_328
RUNTIME_SHA = "064d67d7cc3d2a3fbbed39e6f1b1e779b481a01c23f55902ac1e5599ced1180c"
SUITE_SHA = "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead"
MAX_REQUESTS = 72
MAX_SECONDS = 900


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prompt(case: dict[str, Any], policy: str) -> str:
    if policy == "raw_prefix":
        return str(case["source_before"])
    if policy == "fim_psm":
        return format_psm(str(case["source_before"]), str(case["source_after"]))
    raise ValueError("unknown prompt policy")


def selected_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    selected = []
    for row in rows:
        if counts[row["language"]] < 2:
            selected.append(row)
            counts[row["language"]] += 1
    if len(selected) != 18 or len(counts) != 9:
        raise ValueError("fixed two-per-language diagnostic requires nine languages")
    return selected


def identities(args: argparse.Namespace) -> dict[str, Any]:
    if digest(args.model) != MODEL_SHA or args.model.stat().st_size != MODEL_BYTES:
        raise ValueError("existing untouched Qwen artifact identity mismatch")
    if digest(args.binary) != RUNTIME_SHA or digest(args.suite) != SUITE_SHA:
        raise ValueError("runtime or fixed suite identity mismatch")
    return {
        "model_sha256": MODEL_SHA,
        "model_bytes": MODEL_BYTES,
        "binary_sha256": RUNTIME_SHA,
        "suite_sha256": SUITE_SHA,
        "code": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/evaluate_causal_line.py",
                ROOT / "scripts/measure_r2_local.py",
                ROOT / "scripts/measure_sweep_local.py",
                ROOT / "src/tinycomplete/data/fim.py",
            )
        },
        "host": platform.node(),
        "backend": "CPU-only llama.cpp f072b103714dfa1eee531f80b24512faf38e3dd2",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "binary", "suite", "plan", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = selected_cases([json.loads(line) for line in args.suite.read_text().splitlines()])
    plan = {
        "schema": "q25-matched-fim-line-diagnostic-v1",
        **identities(args),
        "cases": [row["id"] for row in rows],
        "prompts": {
            policy: {
                row["id"]: hashlib.sha256(prompt(row, policy).encode()).hexdigest() for row in rows
            }
            for policy in ("raw_prefix", "fim_psm")
        },
        "repetitions": 2,
        "max_requests": MAX_REQUESTS,
        "max_wall_seconds": MAX_SECONDS,
        "max_output_tokens": 96,
        "input_budget": 2976,
        "context_tokens": 3072,
        "decoding": "greedy seed 928173; native newline stop or EOS; no cache",
        "scoring": "existing score_line; only native omitted-LF restoration",
        "selection": "first two cases per language in unchanged source fixture order",
        "order": "repetition, case, raw prefix then FIM; same resident model",
        "overflow": "record skipped before generation; never truncate either prompt",
        "task": "line completion, not next-edit or human acceptance calibration",
        "runtime_argv": build_server_argv(args.binary, args.model, 0),
        "source_identity": "existing R2 verified-local-models q25 source_checkpoint_modified=false",
    }
    if not args.execute:
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open("x") as stream:
            json.dump(plan, stream, sort_keys=True, indent=2)
        print(json.dumps({"frozen": True, "plan_sha256": digest(args.plan)}))
        return
    if json.loads(args.plan.read_text()) != plan:
        raise ValueError("frozen diagnostic identity changed")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(args.output, 0o700)
    os.environ["TABCOMPLETE_OBSERVABILITY_MODE"] = "offline"
    os.environ["TABCOMPLETE_OBSERVABILITY_ENABLED"] = "1"
    os.environ["TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT"] = "0"
    os.environ["TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE"] = str(args.output / "telemetry.jsonl")
    deadline = time.monotonic() + MAX_SECONDS
    port = _available_port()
    url = f"http://127.0.0.1:{port}"
    records = []
    with (args.output / "server.log").open("wb") as log:
        process = subprocess.Popen(
            build_server_argv(args.binary, args.model, port),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        sampler = MemorySampler(process.pid, [])
        sampler.start()
        try:
            load_seconds = _wait_server(process, url, timeout_seconds=60)
            provider = NativeProvider(url, "q25-untouched-q4-fim-diagnostic")
            provider.cache = False
            with run_scope(args.output / "run.json", "q25-fim-line-diagnostic") as run:
                with (args.output / "predictions.jsonl").open("x") as output:
                    for repetition in range(2):
                        for case in rows:
                            for policy in ("raw_prefix", "fim_psm"):
                                if time.monotonic() >= deadline - 30:
                                    raise TimeoutError("diagnostic finalization reserve reached")
                                text = prompt(case, policy)
                                response = httpx.post(
                                    url + "/tokenize",
                                    timeout=10,
                                    json={
                                        "content": text,
                                        "add_special": True,
                                        "parse_special": True,
                                    },
                                )
                                response.raise_for_status()
                                tokens = response.json()["tokens"]
                                result = {
                                    "case_id": case["id"],
                                    "policy": policy,
                                    "repetition": repetition,
                                    "input_tokens": len(tokens),
                                }
                                if len(tokens) > plan["input_budget"]:
                                    result["status"] = "input_budget_skipped"
                                else:
                                    provider.timeout_seconds = min(
                                        60, deadline - time.monotonic() - 30
                                    )
                                    with run.for_case(
                                        f"{case['id']}/{policy}/{repetition}"
                                    ).activate():
                                        generated = provider.generate_line_detailed(text, 96)
                                        result.update(
                                            score_line(
                                                case,
                                                generated.text,
                                                native_newline_omitted=generated.finish_reason
                                                == "word",
                                            )
                                        )
                                result.update(status="completed", native=provider.last)
                                result["explicit_termination"] = generated.finish_reason in (
                                    "eos",
                                    "word",
                                )
                                records.append(result)
                                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                                output.flush()
            summary = {
                "plan_sha256": digest(args.plan),
                "load_seconds": load_seconds,
                "requests": len(records),
                "memory": sampler.report(),
                "policies": {
                    policy: {
                        "completed": sum(
                            r["status"] == "completed" for r in records if r["policy"] == policy
                        ),
                        "exact": sum(
                            r.get("exact", False) for r in records if r["policy"] == policy
                        ),
                    }
                    for policy in ("raw_prefix", "fim_psm")
                },
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"requests": len(records), "policies": summary["policies"]}))
        finally:
            sampler.close()
            _stop_process(process)
            current_runtime().shutdown()


if __name__ == "__main__":
    main()
