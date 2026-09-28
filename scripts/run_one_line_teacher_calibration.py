"""Run one resumable, frozen plan-7 Muse calibration on fresh synthetic cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from tinycomplete.one_line.calibration_fresh32 import (
    fresh32_prompts,
    load_fresh32_cases,
    score_fresh32_predictions,
)
from tinycomplete.one_line.contract import decode_action
from tinycomplete.one_line.evaluate import Prediction
from tinycomplete.one_line.teacher import (
    AUTHORIZATION_BASIS,
    OpenCodeTeacherClient,
    TeacherCandidateError,
    TeacherResponse,
    TeacherUsageLedger,
    extract_final_action_block_v7,
)

ROOT = Path(__file__).resolve().parents[1]
REL = Path("reports/research/one_line_r1")
PLAN_SHA = "144e807dbaae426225f954730cc7a42933d2db3c2789190777e3fbc479decb02"
PROTOCOL_SHA = "4f2b6245fd941c05e4e09d63683632611e5f6605b1b92fe7b7fb6ec222ea98a7"
FIXTURE_SHA = "7ba1b86df983fe30009a10746ba343e38963b034e5300dc77dc4aaa2df634177"
MANIFEST_SHA = "66f61346ba32b618e7814b024e2b63a3a46deaab14e13c957fe6841ac8113689"
TOKENIZER_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    / "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
RAW_PATH = ROOT / "artifacts/research/one_line_r1/teacher_calibration_plan7_fresh32.jsonl"
REPORT_PATH = ROOT / REL / "teacher_calibration_pass_v3_fresh32.json"
LEDGER_PATH = ROOT / REL / "teacher_usage.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen() -> dict:
    paths = {
        "plan": (ROOT / REL / "plan-v7.json", PLAN_SHA),
        "protocol": (ROOT / "configs/research/one_line_teacher_protocol_v3.json", PROTOCOL_SHA),
        "fixture": (ROOT / REL / "calibration_fresh32_cases.json", FIXTURE_SHA),
        "manifest": (ROOT / REL / "calibration_fresh32_manifest.json", MANIFEST_SHA),
        "tokenizer": (MODEL_SNAPSHOT / "tokenizer.json", TOKENIZER_SHA),
    }
    for name, (path, expected) in paths.items():
        if sha(path) != expected:
            raise ValueError(f"frozen {name} hash mismatch")
    plan = json.loads(paths["plan"][0].read_text())
    protocol = json.loads(paths["protocol"][0].read_text())
    if plan["plan_revision"] != 7 or protocol["version"] != "one-line-teacher-protocol-v3":
        raise ValueError("frozen plan or protocol version mismatch")
    if plan["teacher"]["fresh32_fixture_sha256"] != FIXTURE_SHA:
        raise ValueError("plan fixture identity mismatch")
    if plan["teacher"]["fresh32_manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("plan manifest identity mismatch")
    if plan["teacher"]["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("plan protocol identity mismatch")
    return protocol


def completed_rows(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        case_id = row.get("case_id")
        if case_id not in expected_ids or case_id in rows:
            raise ValueError("raw calibration artifact has unknown or duplicate case")
        if row.get("plan_sha256") != PLAN_SHA or row.get("protocol_sha256") != PROTOCOL_SHA:
            raise ValueError("raw calibration artifact frozen identity mismatch")
        if sha_text(row["raw_content"]) != row["raw_output_sha256"]:
            raise ValueError("raw calibration artifact content hash mismatch")
        rows[case_id] = row
    return rows


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_row(
    case_id: str, prompt: str, prompt_tokens: int, response: TeacherResponse, tokenizer: object
) -> dict:
    content = response.content
    complete = response.finish_reason == "stop"
    raw_count = len(tokenizer.encode(content, add_special_tokens=False)) + 1  # type: ignore[attr-defined]
    raw = decode_action(content, terminated=complete, generated_tokens=raw_count, max_tokens=64)
    try:
        extracted = extract_final_action_block_v7(
            content, provider_complete=complete, tokenizer=tokenizer
        )
        wire = extracted.wire
        q25_tokens = extracted.wire_plus_q25_eos_tokens
        error = None
    except TeacherCandidateError as exc:
        wire = None
        q25_tokens = None
        error = str(exc)
    return {
        "case_id": case_id,
        "request_id": f"cal7-{case_id}",
        "plan_sha256": PLAN_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "prompt_sha256": sha_text(prompt),
        "prompt_q25_tokens": prompt_tokens,
        "raw_content": content,
        "raw_output_sha256": sha_text(content),
        "raw_output_utf8_bytes": len(content.encode("utf-8")),
        "raw_wire_status": raw.status,
        "provider_finish": response.finish_reason,
        "provider_message_complete": complete,
        "extracted_status": "ok" if wire is not None else "invalid",
        "extracted_error": error,
        "extracted_wire": wire,
        "extracted_wire_plus_q25_eos_tokens": q25_tokens,
        "model_id": response.model_id,
        "session_id": response.session_id,
        "response_id": response.response_id,
        "input_tokens": response.input_tokens,
        "visible_output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "generated_output_tokens": response.output_tokens + response.reasoning_tokens,
        "total_tokens_reported": response.total_tokens_reported,
        "cache_read_tokens": response.cached_read_tokens,
        "cache_write_tokens": response.cached_write_tokens,
        "cost_usd_reported": response.cost_usd_reported,
    }


def report(cases: list, rows: dict[str, dict]) -> dict:
    predictions = [
        Prediction(
            case_id=case.id,
            wire=rows[case.id]["extracted_wire"] or "",
            terminated=rows[case.id]["extracted_wire"] is not None,
            generated_tokens=rows[case.id]["extracted_wire_plus_q25_eos_tokens"],
        )
        for case in cases
    ]
    scored = score_fresh32_predictions(cases, predictions)
    summary = scored["summary"]
    gate = {
        "valid_fraction": summary["valid_action"] / 32 >= 0.95,
        "edit_success_fraction": summary["edit_success_rate"] is not None
        and summary["edit_success_rate"] >= 0.5,
        "keep_false_edit_fraction": summary["false_positive_edit_rate"] is not None
        and summary["false_positive_edit_rate"] <= 0.1,
        "utility": summary["utility"] >= 1,
    }
    return {
        "plan_sha256": PLAN_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "fixture_sha256": FIXTURE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "raw_artifact": str(RAW_PATH.relative_to(ROOT)),
        "raw_artifact_sha256": sha(RAW_PATH),
        "case_usage_and_hashes": [
            {k: v for k, v in rows[case.id].items() if k != "raw_content"} for case in cases
        ],
        "score": scored,
        "gate": gate,
        "pilot_allowed": all(gate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol = verify_frozen()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    cases = load_fresh32_cases(
        ROOT / REL / "calibration_fresh32_cases.json",
        ROOT / REL / "calibration_fresh32_manifest.json",
    )
    prompts = fresh32_prompts(cases, tokenizer)
    rows = completed_rows(RAW_PATH, {case.id for case in cases})
    prompt_by_id = {item["case_id"]: item for item in prompts}
    for case_id, row in rows.items():
        if row["prompt_sha256"] != sha_text(prompt_by_id[case_id]["prompt"]):
            raise ValueError("resumed prompt identity mismatch")
        if row["prompt_q25_tokens"] != prompt_by_id[case_id]["input_tokens"]:
            raise ValueError("resumed prompt token count mismatch")
    if not args.execute:
        print(json.dumps({"validated_cases": len(cases), "completed_cases": len(rows)}))
        return
    ledger = TeacherUsageLedger(LEDGER_PATH)
    # Any reserved request without a durable output is ambiguous. Never retry it.
    with ledger.path.open("a+", encoding="utf-8") as handle:
        ledger_requests = ledger._read(handle)
    for case_id, row in rows.items():
        request = ledger_requests.get(f"cal7-{case_id}")
        if request is None or "input" not in request:
            raise ValueError("completed row has no settled ledger reservation")
        if row["request_id"] != f"cal7-{case_id}":
            raise ValueError("completed row request identity mismatch")
    if len(rows) < len(cases):
        with OpenCodeTeacherClient(ledger) as client:
            for item in prompts:
                case_id = item["case_id"]
                if case_id in rows:
                    continue
                request_id = f"cal7-{case_id}"
                if request_id in ledger_requests:
                    raise ValueError("ambiguous prior provider call; no automatic retry")
                started = time.perf_counter()
                response = client.run_role(
                    request_id=request_id,
                    prompt=item["prompt"],
                    purpose="calibration",
                    source_class="synthetic",
                    authorization_basis=AUTHORIZATION_BASIS,
                    reserve_input_tokens=10_000,
                    reserve_output_tokens=8_192,
                    system_instruction=protocol["calibration"]["fixed_instruction"],
                )
                row = make_row(case_id, item["prompt"], item["input_tokens"], response, tokenizer)
                row["elapsed_seconds"] = time.perf_counter() - started
                append_row(RAW_PATH, row)
                rows[case_id] = row
                ledger_requests[request_id] = {"input": response.input_tokens}
                print(json.dumps({"case_id": case_id, "status": row["extracted_status"]}))
    result = report(cases, rows)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(rows), "gate": result["gate"]}))


if __name__ == "__main__":
    main()
