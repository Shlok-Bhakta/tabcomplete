"""Staged teacher generation: fake by default, paid only with authorization.

Stages: A (10 requests, pipeline validation) -> B (50 states x candidates,
>=90% schema success required) -> C (up to paid-example cap / budget).
Every candidate (accepted or rejected with reasons) is stored as JSONL.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from ..data.static_edits import make_next_edit
from .base import EditableRegion, TeacherRequest, provider_from_name
from .budget import Budget, estimate_cost_usd
from .validate import validate_response

__all__ = ["STAGE_SIZES", "build_request", "run_generation"]

STAGE_SIZES = {"A": 10, "B": 50}

# Conservative per-1k-token estimates (PEAK public pricing + SAFETY_MARGIN
# applied in estimate_cost_usd) used when a provider reports no exact cost.
# Source: https://api-docs.deepseek.com/quick_start/pricing (2026-09-17).
ESTIMATED_PRICING_PER_1K = {
    "deepseek": (0.0003, 0.0012),  # flash peak: $0.30/1M in, $1.20/1M out
    "openrouter": (0.010, 0.030),
    "fake": (0.0, 0.0),
}

FIXTURE_ARITH = (
    '"""Synthetic fixture: arithmetic helpers."""\nimport os\n\n\n'
    "def add(a, b):\n    return a + b\n"
)
FIXTURE_GREET = (
    '"""Synthetic fixture: greetings."""\nfrom pathlib import Path\n\n\n'
    'def greet(name):\n    if name:\n        return "hi " + name\n    return "hi"\n'
)
FIXTURE_STATS = (
    '"""Synthetic fixture: running statistics."""\nfrom dataclasses import dataclass, field\n\n\n'
    "@dataclass\nclass Running:\n    total: float = 0.0\n    count: int = 0\n\n"
    "    def push(self, value):\n        self.total += value\n        self.count += 1\n\n"
    "    def mean(self):\n        if not self.count:\n            return 0.0\n"
    "        return self.total / self.count\n"
)
FIXTURE_RETRY = (
    '"""Synthetic fixture: retry helper."""\nimport time\n\n\n'
    "def fetch_with_retry(call, attempts=3, delay=1.0):\n"
    "    last = None\n    for i in range(attempts):\n"
    "        try:\n            return call()\n"
    "        except IOError as exc:\n            last = exc\n"
    "            time.sleep(delay * (i + 1))\n"
    "    raise last\n"
)
FIXTURE_TABLE = (
    '"""Synthetic fixture: report table."""\n\n\n'
    "def render_table(rows):\n"
    '    lines = ["| name | value |"]\n'
    "    for name, value in rows:\n"
    '        lines.append(f"| {name} | {value} |")\n'
    '    return "\\n".join(lines)\n'
)
FIXTURE_CONFIG = (
    '"""Synthetic fixture: config merge."""\n\n\n'
    "DEFAULTS = {'timeout': 30, 'retries': 3, 'verbose': False}\n\n\n"
    "def merge_config(overrides):\n"
    "    merged = dict(DEFAULTS)\n"
    "    for key, value in overrides.items():\n"
    "        if key in merged:\n"
    "            merged[key] = value\n"
    "    return merged\n"
)
FIXTURE_CHUNK = (
    '"""Synthetic fixture: batching."""\nfrom collections.abc import Iterator\n\n\n'
    "def batched(items, size):\n"
    "    batch = []\n    for item in items:\n"
    "        batch.append(item)\n"
    "        if len(batch) >= size:\n"
    "            yield batch\n"
    "            batch = []\n"
    "    if batch:\n"
    "        yield batch\n"
)
FIXTURE_PARSE = (
    '"""Synthetic fixture: line parsing."""\n\n\n'
    "def parse_kv(line):\n"
    '    key, sep, value = line.partition("=")\n'
    "    if not sep:\n"
    "        raise ValueError(f'bad line: {line!r}')\n"
    "    return key.strip(), value.strip()\n"
)
FIXTURE_CACHE = (
    '"""Synthetic fixture: tiny memo cache."""\n\n\n'
    "def memoize(fn):\n"
    "    cache = {}\n"
    "    def wrapper(*args):\n"
    "        if args not in cache:\n"
    "            cache[args] = fn(*args)\n"
    "        return cache[args]\n"
    "    return wrapper\n"
)
FIXTURE_FILES = (
    '"""Synthetic fixture: file helpers."""\nimport os\n\n\n'
    "def latest(path):\n"
    "    entries = sorted(os.listdir(path))\n"
    "    if not entries:\n"
    "        return None\n"
    "    return os.path.join(path, entries[-1])\n"
)

FIXTURE_SOURCES: tuple[str, ...] = (
    FIXTURE_ARITH,
    FIXTURE_GREET,
    FIXTURE_STATS,
    FIXTURE_RETRY,
    FIXTURE_TABLE,
    FIXTURE_CONFIG,
    FIXTURE_CHUNK,
    FIXTURE_PARSE,
    FIXTURE_CACHE,
    FIXTURE_FILES,
)


def build_request(source: str, seed: int, num_candidates: int, state_id: str) -> TeacherRequest:
    example = make_next_edit(source, seed)
    region_text = example.input_text.split("[[EDIT]]", 1)[1].split("[[/EDIT]]", 1)[0]
    body = example.input_text.split("<file", 1)[1].split(">", 1)[1].rsplit("</file>", 1)[0]
    assert body.startswith("\n") and body.endswith("\n")
    marked = body[1:-1]
    file_text = marked.replace("[[EDIT]]", "").replace("[[/EDIT]]", "")
    raw = file_text.encode("utf-8")
    assert raw[example.region_start : example.region_end].decode() == region_text
    return TeacherRequest(
        state_id=state_id,
        serialized_state=example.input_text,
        region=EditableRegion(
            path=example.source_path or "fixture.py",
            start=example.region_start,
            end=example.region_end,
            text=region_text,
        ),
        repo_context="",
        recent_edits=example.recent_edits,
        num_candidates=num_candidates,
        language=example.language,
        file_text=file_text,
    )


async def _predict_one(provider, request: TeacherRequest) -> dict:
    base = {
        "state_id": request.state_id,
        "serialized_state": request.serialized_state,
        "region": request.region.model_dump(),
        "language": request.language,
        "num_candidates": request.num_candidates,
    }
    try:
        response = await provider.predict(request)
    except Exception as exc:  # pipeline continues; failure recorded
        return {
            **base,
            "ok": False,
            "error": str(exc)[:300],
            "accepted": [],
            "rejected": [],
        }
    accepted, rejected = validate_response(response.candidates, request)
    return {
        **base,
        "ok": True,
        "provider": response.provider,
        "model": response.model,
        "request_id": response.request_id,
        "raw_sha256": response.raw_sha256,
        "prompt_tokens": response.prompt_tokens,
        "output_tokens": response.output_tokens,
        "reported_cost_usd": response.reported_cost_usd,
        "accepted": accepted,
        "rejected": rejected,
    }


def run_generation(
    provider_name: str = "fake",
    max_states: int = 10,
    seed: int = 0,
    out_path: str = "",
    num_candidates: int = 3,
    stage_override: str = "",
    **provider_kwargs,
) -> dict:
    """Run staged generation synchronously. Returns a summary dict."""
    budget = Budget.from_env()
    if provider_name != "fake" and not budget.paid_enabled():
        raise RuntimeError("paid generation not enabled (ALLOW_PAID_SYNTHETIC=1 required)")
    provider = provider_from_name(provider_name, **provider_kwargs)
    total = max(0, max_states)
    # Stage schedule: A=10, then B up to 50, then C remainder.
    schedule = (
        ["A"] * min(total, STAGE_SIZES["A"])
        + ["B"] * min(max(0, total - STAGE_SIZES["A"]), STAGE_SIZES["B"])
        + ["C"] * max(0, total - STAGE_SIZES["A"] - STAGE_SIZES["B"])
    )
    if stage_override:
        schedule = [stage_override] * total
    results: list[dict] = []

    async def _run() -> None:
        avg_cost = 0.01  # conservative per-state ceiling until measured
        for k, stage in enumerate(schedule):
            if provider_name != "fake" and not budget.can_spend(avg_cost, 1):
                results.append(
                    {"state_id": "budget-stop", "ok": False, "error": "budget exhausted"}
                )
                break
            source = FIXTURE_SOURCES[(seed + k) % len(FIXTURE_SOURCES)]
            request = build_request(source, seed * 100003 + k, num_candidates, f"s{seed}-{k}")
            record = await _predict_one(provider, request)
            record["stage"] = stage
            results.append(record)
            if provider_name != "fake" and record.get("ok"):
                cost = record.get("reported_cost_usd")
                if cost is None:
                    price_in, price_out = ESTIMATED_PRICING_PER_1K.get(provider_name, (0.01, 0.03))
                    cost = estimate_cost_usd(
                        record.get("prompt_tokens") or 0,
                        record.get("output_tokens") or 0,
                        price_in,
                        price_out,
                    )
                    record["estimated_cost_usd"] = cost
                budget.record(cost, 1)
                done = [r for r in results if r.get("ok")]
                costs = [
                    r.get("estimated_cost_usd") or r.get("reported_cost_usd") or 0.0 for r in done
                ]
                if costs:
                    avg_cost = 2.0 * sum(costs) / len(costs)
            if stage == "A" and k == STAGE_SIZES["A"] - 1 and total > STAGE_SIZES["A"]:
                ok_rate = sum(1 for r in results if r["ok"]) / len(results)
                if ok_rate < 1.0:
                    raise RuntimeError(f"stage A unhealthy: ok_rate={ok_rate}")
            if stage == "B" and len([r for r in results if r["stage"] == "B"]) == STAGE_SIZES["B"]:
                schema_ok = sum(1 for r in results if r["ok"] and r["accepted"]) / STAGE_SIZES["B"]
                if schema_ok < 0.9:
                    raise RuntimeError(f"stage B unhealthy: schema_ok={schema_ok}")

    asyncio.run(_run())
    accepted = sum(len(r["accepted"]) for r in results)
    rejected = sum(len(r["rejected"]) for r in results)
    summary = {
        "provider": provider_name,
        "states": len(results),
        "accepted": accepted,
        "rejected": rejected,
        "spent_usd": budget.state.spent_usd if provider_name != "fake" else 0.0,
        "timestamp": time.time(),
    }
    if out_path:
        directory = os.path.dirname(out_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        with open(out_path.replace(".jsonl", ".summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    return summary
