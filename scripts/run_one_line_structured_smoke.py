"""One-call, plan-bound synthetic OpenCode structured-output access smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from tinycomplete.one_line.teacher import (
    AUTHORIZATION_BASIS,
    MODEL_ID,
    OpenCodeTeacherClient,
    TeacherUsageLedger,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "reports/research/one_line_r1/plan-v8.json"
SPEC_REL = "configs/research/one_line_structured_smoke_v1.json"
SPEC_SHA = "3ed3ee4906b7ff4eeda82d6f07a20ac05f3add7087841c50b7051b872b3aa4ef"
PROMPT_SHA = "f2a58002ea51c3b30669f07b9241ef511a6d9cc1e3d3ef620ed01274526d77bc"
SCHEMA_SHA = "682f6ca6d81c70e16c3c800dce2f565ca0690dd449a342c5bb9c645feaf1f6d5"
LEDGER = ROOT / "reports/research/one_line_r1/teacher_usage.jsonl"
OUTPUT = ROOT / "reports/research/one_line_r1/structured_smoke_plan8.json"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def verify_frozen(plan_sha256: str, *, plan_path: Path = PLAN, root: Path = ROOT) -> dict:
    if re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise ValueError("plan SHA-256 must be supplied")
    if sha_bytes(plan_path.read_bytes()) != plan_sha256:
        raise ValueError("plan-8 hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_revision") != 8:
        raise ValueError("structured smoke requires plan revision 8")
    teacher = plan.get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError("plan teacher policy missing")
    if teacher.get("structured_smoke_spec_path") != SPEC_REL:
        raise ValueError("structured smoke spec path mismatch")
    if teacher.get("structured_smoke_spec_sha256") != SPEC_SHA:
        raise ValueError("structured smoke spec hash mismatch")
    spec_path = root / SPEC_REL
    if sha_bytes(spec_path.read_bytes()) != SPEC_SHA:
        raise ValueError("frozen structured smoke spec changed")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if sha_bytes(spec["prompt"].encode()) != PROMPT_SHA:
        raise ValueError("frozen synthetic prompt changed")
    if sha_bytes(canonical_bytes(spec["schema"])) != SCHEMA_SHA:
        raise ValueError("frozen structured schema changed")
    if (
        spec.get("version") != "one-line-structured-smoke-v1"
        or spec.get("source_class") != "synthetic"
        or spec.get("model") != MODEL_ID
        or spec.get("purpose") != "calibration"
        or spec.get("request_id") != "structured8-synthetic-001"
        or spec.get("maximum_calls") != 1
        or spec.get("allow_text_fallback") is not False
        or spec.get("allow_other_model") is not False
    ):
        raise ValueError("structured smoke policy changed")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = verify_frozen(args.plan_sha256)
    ledger = TeacherUsageLedger(LEDGER)
    with ledger.path.open("a+", encoding="utf-8") as handle:
        existing = ledger._read(handle)
    if spec["request_id"] in existing:
        raise ValueError("structured smoke request already reserved; no retry")
    if OUTPUT.exists():
        raise ValueError("structured smoke output already exists")
    if not args.execute:
        print(json.dumps({"ready": True, "request_id": spec["request_id"]}))
        return
    with OpenCodeTeacherClient(ledger, structured_output_only=True) as client:
        response = client.run_role(
            request_id=spec["request_id"],
            prompt=spec["prompt"],
            purpose="calibration",
            source_class="synthetic",
            authorization_basis=AUTHORIZATION_BASIS,
            reserve_input_tokens=10_000,
            reserve_output_tokens=8_192,
            output_schema=spec["schema"],
        )
    output = json.loads(response.content)
    valid = output == spec["expected"] and response.finish_reason == "stop"
    result = {
        "plan_sha256": args.plan_sha256,
        "spec_sha256": SPEC_SHA,
        "prompt_sha256": PROMPT_SHA,
        "schema_sha256": SCHEMA_SHA,
        "request_id": spec["request_id"],
        "session_id": response.session_id,
        "response_id": response.response_id,
        "model_id": response.model_id,
        "finish_reason": response.finish_reason,
        "content_sha256": sha_bytes(response.content.encode()),
        "content": output,
        "expected_match": valid,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "generated_tokens_including_reasoning": response.output_tokens + response.reasoning_tokens,
        "reported_cost_usd": response.cost_usd_reported,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"request_id": spec["request_id"], "expected_match": valid}))


if __name__ == "__main__":
    main()
