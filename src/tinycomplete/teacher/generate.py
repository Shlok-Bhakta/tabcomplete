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
from .budget import Budget
from .validate import apply_replacement, validate_candidate

__all__ = ["STAGE_SIZES", "build_request", "run_generation"]

STAGE_SIZES = {"A": 10, "B": 50}

FIXTURE_ARITH = (
    '"""Synthetic fixture: arithmetic helpers."""\nimport os\n\n\n'
    "def add(a, b):\n    return a + b\n"
)
FIXTURE_GREET = (
    '"""Synthetic fixture: greetings."""\nfrom pathlib import Path\n\n\n'
    'def greet(name):\n    if name:\n        return "hi " + name\n    return "hi"\n'
)

FIXTURE_SOURCES: tuple[str, ...] = (FIXTURE_ARITH, FIXTURE_GREET)


def build_request(source: str, seed: int, num_candidates: int, state_id: str) -> TeacherRequest:
    example = make_next_edit(source, seed)
    region_text = example.input_text.split("[[EDIT]]", 1)[1].split("[[/EDIT]]", 1)[0]
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
    accepted: list[dict] = []
    rejected: list[dict] = []
    for cand in response.candidates:
        result = validate_candidate(
            cand,
            region_text=request.region.text,
            full_text=request.serialized_state,
            region_start=request.region.start,
            region_end=request.region.end,
            language=request.language,
        )
        record = {
            "action": cand.action,
            "replacement": cand.replacement,
            "reasons": list(result.reasons),
        }
        (accepted if result.ok else rejected).append(record)
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
) -> dict:
    """Run staged generation synchronously. Returns a summary dict."""
    budget = Budget.from_env()
    if provider_name != "fake" and not budget.paid_enabled():
        raise RuntimeError("paid generation not enabled (ALLOW_PAID_SYNTHETIC=1 required)")
    provider = provider_from_name(provider_name)
    total = max(0, max_states)
    # Stage schedule: A=10, then B up to 50, then C remainder.
    schedule = (
        ["A"] * min(total, STAGE_SIZES["A"])
        + ["B"] * min(max(0, total - STAGE_SIZES["A"]), STAGE_SIZES["B"])
        + ["C"] * max(0, total - STAGE_SIZES["A"] - STAGE_SIZES["B"])
    )
    results: list[dict] = []

    async def _run() -> None:
        for k, stage in enumerate(schedule):
            source = FIXTURE_SOURCES[(seed + k) % len(FIXTURE_SOURCES)]
            request = build_request(source, seed * 100003 + k, num_candidates, f"s{seed}-{k}")
            record = await _predict_one(provider, request)
            record["stage"] = stage
            results.append(record)
            if provider_name != "fake":
                cost = record.get("reported_cost_usd") or 0.0
                budget.record(cost, 1)
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
    _ = apply_replacement  # re-exported for pipeline consumers
    return summary
