"""Continue the frozen Muse author pilot on untouched sources only.

The first seven plan-10 rows remain authoritative. Its eighth provider response
exceeded the input reservation and is accounted for, but never becomes a row
or a retry. This runner starts at source index eight.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from run_one_line_author_pilot_v2 import (
    BUILDER_REL,
    BUILDER_SHA,
    LEDGER,
    MODEL_ID,
    MODEL_SNAPSHOT,
    PROTOCOL_REL,
    PROTOCOL_SHA,
    ROOT,
    SOURCE_REL,
    SOURCE_SHA,
    TOKENIZER_SHA,
    append_row,
    evaluate_response,
    load_inputs,
    sha_bytes,
    sha_file,
    source_preflight,
)
from run_one_line_author_pilot_v2 import (
    RAW as PREVIOUS_RAW,
)
from run_one_line_author_pilot_v2 import (
    request_id as previous_request_id,
)

from tinycomplete.one_line.teacher import (
    AUTHORIZATION_BASIS,
    OpenCodeTeacherClient,
    TeacherUsageLedger,
)

REPORT = ROOT / "reports/research/one_line_r1"
PLAN = REPORT / "plan.json"
PLAN_SHA = "fb0a76a94d67b1a07a877ad650599da6679eafcb4162c6ac3ab6a9e1b9ca1860"
SPEC_REL = "configs/research/one_line_author_continuation_v1.json"
SPEC_SHA = "4de9a435032e7385a2ad8156d0e10d6f94df338292a577c5b85b2f19cd819e57"
RAW = ROOT / "artifacts/research/one_line_r1/author_pilot_plan11.jsonl"
RESULT = REPORT / "author_pilot_plan11.json"
COMBINED = REPORT / "author_pilot_combined_plan11.json"


def request_id(source_id: str) -> str:
    return "author11-public-" + sha_bytes(source_id.encode())[:24]


def verify_frozen() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if len(PLAN_SHA) != 64 or len(SPEC_SHA) != 64:
        raise ValueError("plan-11 continuation is not frozen")
    for path, digest in (
        (PLAN, PLAN_SHA),
        (ROOT / SPEC_REL, SPEC_SHA),
        (ROOT / PROTOCOL_REL, PROTOCOL_SHA),
        (ROOT / BUILDER_REL, BUILDER_SHA),
        (ROOT / SOURCE_REL, SOURCE_SHA),
        (MODEL_SNAPSHOT / "tokenizer.json", TOKENIZER_SHA),
    ):
        if sha_file(path) != digest:
            raise ValueError("frozen continuation identity changed")
    plan = json.loads(PLAN.read_text())
    spec = json.loads((ROOT / SPEC_REL).read_text())
    teacher = plan["teacher"]
    if (
        plan["plan_revision"] != 11
        or teacher["author_continuation_path"] != SPEC_REL
        or teacher["author_continuation_sha256"] != SPEC_SHA
        or spec["version"] != "one-line-author-continuation-v1"
        or spec["model"] != MODEL_ID
        or spec["author_protocol_path"] != PROTOCOL_REL
        or spec["author_protocol_sha256"] != PROTOCOL_SHA
        or spec["prompt_builder_file"] != BUILDER_REL
        or spec["prompt_builder_file_sha256"] != BUILDER_SHA
        or spec["source_artifact"] != SOURCE_REL
        or spec["source_sha256"] != SOURCE_SHA
        or spec["previous_plan_sha256"]
        != "938dbdc96ad025c3a8404b2d6bcb2532d33bb2bb2c6194be3622cb87dc8eb94b"
        or spec["previous_settled_artifact"] != str(PREVIOUS_RAW["public"].relative_to(ROOT))
        or spec["previous_settled_source_count"] != 7
        or spec["ignored_overrun_artifact"]
        != "artifacts/research/one_line_r1/author_pilot_plan10_unsettled_failure.json"
        or spec["ignored_source_index_zero_based"] != 7
        or spec["remaining_source_start_index_zero_based"] != 8
        or spec["remaining_source_count"] != 92
        or spec["reserve_input_tokens_per_call"] != 30_000
        or spec["reserve_output_tokens_per_call_including_reasoning"] != 8_192
        or spec["maximum_new_calls"] != 92
        or spec["one_call_per_source_across_plans"] is not True
        or spec["no_retry_on_ambiguous_failure"] is not True
        or spec["ignored_response_reuse"] is not False
        or spec["accepted_training_before_independent_review"] != 0
    ):
        raise ValueError("frozen continuation policy mismatch")
    sources = load_inputs("public")
    previous_path = ROOT / spec["previous_settled_artifact"]
    ignored_path = ROOT / spec["ignored_overrun_artifact"]
    if (
        sha_file(previous_path) != spec["previous_settled_artifact_sha256"]
        or sha_file(ignored_path) != spec["ignored_overrun_artifact_sha256"]
    ):
        raise ValueError("historical author artifacts changed")
    previous = [json.loads(line) for line in previous_path.read_text().splitlines()]
    if len(previous) != 7:
        raise ValueError("historical settled source count changed")
    for index, row in enumerate(previous):
        source = sources[index]
        if (
            row["source_id"] != source["id"]
            or row["request_id"] != previous_request_id("public", source["id"])
            or row["prompt_sha256"] != sha_bytes(source_preflight(source, synthetic=False).encode())
            or row["accepted_training"] is not False
        ):
            raise ValueError("historical settled source identity changed")
    ignored = json.loads(ignored_path.read_text())
    if (
        ignored["request_id"] != spec["ignored_request_id"]
        or ignored["request_id"] != previous_request_id("public", sources[7]["id"])
        or spec["ignored_source_id"] != sources[7]["id"]
        or ignored["accepted_training"] is not False
        or ignored["raw_output_sha256"] != sha_bytes(ignored["raw_content"].encode())
    ):
        raise ValueError("ignored overrun identity changed")
    ledger = TeacherUsageLedger(LEDGER)
    with ledger.path.open("a+", encoding="utf-8") as handle:
        entries = ledger._read(handle)
    for row in previous:
        usage = entries.get(row["request_id"])
        if (
            usage is None
            or usage.get("input") != row["input_tokens"]
            or usage.get("output") != row["generated_tokens"]
        ):
            raise ValueError("historical settled row lacks exact ledger usage")
    overrun = entries.get(ignored["request_id"])
    if (
        overrun is None
        or overrun.get("overrun") != 1
        or overrun.get("input") != ignored["tokens"]["input"]
        or overrun.get("output") != ignored["tokens"]["output"] + ignored["tokens"]["reasoning"]
    ):
        raise ValueError("ignored overrun lacks exact ledger usage")
    return spec, json.loads((ROOT / PROTOCOL_REL).read_text()), sources


def completed_rows(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {row["id"]: row for row in sources[8:]}
    completed: dict[str, dict[str, Any]] = {}
    if not RAW.exists():
        return completed
    for line in RAW.read_text().splitlines():
        row = json.loads(line)
        source_id = row.get("source_id")
        if source_id not in expected or source_id in completed:
            raise ValueError("continuation artifact has unknown or duplicate source")
        source = expected[source_id]
        if (
            row.get("plan_sha256") != PLAN_SHA
            or row.get("spec_sha256") != SPEC_SHA
            or row.get("request_id") != request_id(source_id)
            or row.get("prompt_sha256")
            != sha_bytes(source_preflight(source, synthetic=False).encode())
            or row.get("raw_output_sha256") != sha_bytes(row["raw_content"].encode())
            or row.get("accepted_training") is not False
        ):
            raise ValueError("continuation artifact identity changed")
        completed[source_id] = row
    return completed


def summarize(rows: dict[str, dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows.values())
    result = {
        "plan_sha256": PLAN_SHA,
        "spec_sha256": SPEC_SHA,
        "phase": "public_continuation",
        "expected_new_calls": 92,
        "completed_new_calls": len(rows),
        "remaining_new_calls": 92 - len(rows),
        "counts": dict(counts),
        "accepted_training": 0,
        "raw_artifact": str(RAW.relative_to(ROOT)),
        "raw_artifact_sha256": sha_file(RAW),
        "reported_input_tokens": sum(row["input_tokens"] for row in rows.values()),
        "reported_generated_tokens": sum(row["generated_tokens"] for row in rows.values()),
        "reported_cost_usd": sum(row["reported_cost_usd"] or 0 for row in rows.values()),
    }
    if len(rows) == 92:
        combined = {
            "plan_sha256": PLAN_SHA,
            "spec_sha256": SPEC_SHA,
            "source_sha256": SOURCE_SHA,
            "protocol_sha256": PROTOCOL_SHA,
            "durable_rows": 99,
            "plan10_durable_rows": 7,
            "plan11_durable_rows": 92,
            "failed_source_id": sources[7]["id"],
            "plan10_raw_artifact": str(PREVIOUS_RAW["public"].relative_to(ROOT)),
            "plan10_raw_artifact_sha256": sha_file(PREVIOUS_RAW["public"]),
            "plan11_raw_artifact": str(RAW.relative_to(ROOT)),
            "plan11_raw_artifact_sha256": sha_file(RAW),
            "accepted_training": 0,
        }
        COMBINED.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    return result


def run(*, execute: bool) -> dict[str, Any]:
    spec, protocol, sources = verify_frozen()
    completed = completed_rows(sources)
    if not execute:
        return {
            "phase": "public_continuation",
            "preflight_sources": 92,
            "completed": len(completed),
            "skipped_historical": 8,
        }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    ledger = TeacherUsageLedger(LEDGER)
    with ledger.path.open("a+", encoding="utf-8") as handle:
        ledger_rows = ledger._read(handle)
    for source_id, row in completed.items():
        entry = ledger_rows.get(request_id(source_id))
        if (
            entry is None
            or entry.get("input") != row["input_tokens"]
            or entry.get("output") != row["generated_tokens"]
        ):
            raise ValueError("continuation output lacks exact settled ledger usage")
    if len(completed) < 92:
        with OpenCodeTeacherClient(ledger) as client:
            for source in sources[8:]:
                source_id = source["id"]
                if source_id in completed:
                    continue
                rid = request_id(source_id)
                if rid in ledger_rows:
                    raise ValueError("ambiguous continuation request; no automatic retry")
                prompt = source_preflight(source, synthetic=False)
                response = client.run_role(
                    request_id=rid,
                    prompt=prompt,
                    purpose="student_label",
                    source_class="public",
                    authorization_basis=AUTHORIZATION_BASIS,
                    reserve_input_tokens=spec["reserve_input_tokens_per_call"],
                    reserve_output_tokens=spec[
                        "reserve_output_tokens_per_call_including_reasoning"
                    ],
                    system_instruction=protocol["system_instruction"],
                )
                status, rejection, candidate = evaluate_response(
                    "public", source, response.content, response.finish_reason, tokenizer
                )
                if candidate is not None:
                    candidate["provenance"]["author_actor_id"] = response.model_id
                    candidate["provenance"]["author_session_id"] = response.session_id
                    candidate["provenance"]["author_response_id"] = response.response_id
                row = {
                    "plan_sha256": PLAN_SHA,
                    "spec_sha256": SPEC_SHA,
                    "source_id": source_id,
                    "source_sha256": source["authoring_metadata"]["source_sha256"],
                    "request_id": rid,
                    "prompt_sha256": sha_bytes(prompt.encode()),
                    "raw_content": response.content,
                    "raw_output_sha256": sha_bytes(response.content.encode()),
                    "raw_output_utf8_bytes": len(response.content.encode()),
                    "provider_finish": response.finish_reason,
                    "status": status,
                    "rejection_reason": rejection,
                    "candidate": candidate,
                    "accepted_training": False,
                    "session_id": response.session_id,
                    "response_id": response.response_id,
                    "model_id": response.model_id,
                    "input_tokens": response.input_tokens,
                    "visible_output_tokens": response.output_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                    "generated_tokens": response.output_tokens + response.reasoning_tokens,
                    "reported_cost_usd": response.cost_usd_reported,
                    "file_license_scope_unverified_without_notice": source[
                        "authoring_metadata"
                    ].get("file_license_scope_unverified_without_notice"),
                }
                append_row(RAW, row)
                completed[source_id] = row
                ledger_rows[rid] = {
                    "input": response.input_tokens,
                    "output": response.output_tokens + response.reasoning_tokens,
                }
                progress = summarize(completed, sources)
                RESULT.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"source_id": source_id, "status": status}), flush=True)
    result = summarize(completed, sources)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(execute=args.execute), sort_keys=True))


if __name__ == "__main__":
    main()
