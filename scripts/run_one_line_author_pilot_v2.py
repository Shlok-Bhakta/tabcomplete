"""Frozen plan-10 zero-based Muse author pilot; candidates stay unaccepted."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tinycomplete.one_line.data import parse_author_response
from tinycomplete.one_line.pilot_roles import build_author_prompt_v2
from tinycomplete.one_line.teacher import (
    AUTHORIZATION_BASIS,
    MODEL_ID,
    OpenCodeTeacherClient,
    TeacherCandidateError,
    TeacherUsageLedger,
    assert_opencode_request_allowed,
    extract_author_candidate_block,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/one_line_r1"
PLAN = REPORT / "plan-v10.json"
PLAN_SHA = "938dbdc96ad025c3a8404b2d6bcb2532d33bb2bb2c6194be3622cb87dc8eb94b"
PROTOCOL_REL = "configs/research/one_line_author_protocol_v2.json"
PROTOCOL_SHA = "0ef8ad1633e899bba9982090ad396c450fa0da59c168541181c1d89aa867412e"
BUILDER_REL = "src/tinycomplete/one_line/pilot_roles.py"
BUILDER_SHA = "5991a4315ded56b45162698f1e7bf60fbae7abd93d5716fd43e7196febdd7041"
SOURCE_REL = "artifacts/research/one_line_r1/public_source_authoring_100.jsonl"
SOURCE_SHA = "f1320155723d450454484999ebb3605614e251c223e1801f7e9808d1d3bde933"
SMOKE_REL = "reports/research/one_line_r1/author_indexed_smoke4.json"
SMOKE_SHA = "1801b47ec859e7257c0bb78a239161160052b7a793b40b5f7ee5101c5b1d71a2"
LEDGER = REPORT / "teacher_usage.jsonl"
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    / "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
TOKENIZER_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
RAW = {
    "smoke": ROOT / "artifacts/research/one_line_r1/author_indexed_smoke_plan10.jsonl",
    "public": ROOT / "artifacts/research/one_line_r1/author_pilot_plan10.jsonl",
}
RESULT = {
    "smoke": REPORT / "author_indexed_smoke_plan10.json",
    "public": REPORT / "author_pilot_plan10.json",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def verify_frozen() -> dict[str, Any]:
    expected = (
        (PLAN, PLAN_SHA),
        (ROOT / PROTOCOL_REL, PROTOCOL_SHA),
        (ROOT / BUILDER_REL, BUILDER_SHA),
        (ROOT / SOURCE_REL, SOURCE_SHA),
        (ROOT / SMOKE_REL, SMOKE_SHA),
        (MODEL_SNAPSHOT / "tokenizer.json", TOKENIZER_SHA),
    )
    for path, digest in expected:
        if sha_file(path) != digest:
            raise ValueError("frozen author plan, protocol, source, smoke, or tokenizer changed")
    plan = json.loads(PLAN.read_text())
    protocol = json.loads((ROOT / PROTOCOL_REL).read_text())
    teacher = plan["teacher"]
    if plan["plan_revision"] != 10 or teacher["author_indexed_protocol_path"] != PROTOCOL_REL:
        raise ValueError("plan-10 author protocol identity mismatch")
    if teacher["author_indexed_protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("plan-10 author protocol hash mismatch")
    if teacher["authoring_source_sha256"] != SOURCE_SHA:
        raise ValueError("plan-10 author source hash mismatch")
    if (
        protocol["version"] != "one-line-author-text-v2"
        or protocol["model"] != MODEL_ID
        or protocol["prompt_builder"] != "build_author_prompt_v2"
        or protocol["prompt_builder_file"] != BUILDER_REL
        or protocol["prompt_builder_file_sha256"] != BUILDER_SHA
        or protocol["source_artifact"] != SOURCE_REL
        or protocol["source_sha256"] != SOURCE_SHA
        or protocol["smoke_artifact"] != SMOKE_REL
        or protocol["smoke_sha256"] != SMOKE_SHA
        or protocol["terminal_open"] != "<AUTHOR_CANDIDATE>"
        or protocol["terminal_close"] != "</AUTHOR_CANDIDATE>"
        or protocol["maximum_json_utf8_bytes"] != 8192
        or protocol["require_provider_finish_stop"] is not True
        or protocol["one_call_per_seed"] is not True
        or protocol["allow_repo_mit_without_file_notice_for_authoring"] is not True
        or protocol["training_acceptance_requires_separate_file_scope_review"] is not True
        or protocol["synthetic_smoke"]["cases"] != 4
        or protocol["synthetic_smoke"]["maximum_calls"] != 4
        or protocol["public_pilot"]["candidates"] != 100
        or protocol["public_pilot"]["maximum_calls"] != 100
        or protocol["public_pilot"]["accepted_labels_before_independent_review"] != 0
    ):
        raise ValueError("frozen author protocol policy mismatch")
    return protocol


def source_preflight(row: dict[str, Any], *, synthetic: bool) -> str:
    metadata = row["authoring_metadata"]
    seed = row["student_state_seed"]
    source = seed["source"]
    if not isinstance(source, str) or sha_bytes(source.encode()) != metadata["source_sha256"]:
        raise ValueError("source bytes differ from pinned source hash")
    if metadata["source_license"] != "MIT" or metadata["source_provenance_verified"] is not True:
        raise ValueError("source license or provenance is unverified")
    if not re.fullmatch(r"[a-f0-9]{40}", metadata["source_revision"]):
        raise ValueError("source revision is not immutable")
    if not re.fullmatch(r"[a-f0-9]{64}", metadata["license_sha256"]):
        raise ValueError("license artifact hash is invalid")
    notice = metadata.get("file_spdx_notice")
    if notice is not None and notice != "MIT":
        raise ValueError("file notice conflicts with pinned MIT repository license")
    if not synthetic:
        if not metadata.get("license_path") or not metadata.get("source_url", "").startswith(
            "https://github.com/"
        ):
            raise ValueError("public source provenance URL or license path missing")
        if metadata["source_revision"] not in metadata["source_url"]:
            raise ValueError("public source URL does not pin revision")
        if metadata.get("file_license_scope_unverified_without_notice") not in (True, False):
            raise ValueError("file-level license scope flag missing")
    prompt = build_author_prompt_v2(row)
    assert_opencode_request_allowed(
        model_id=MODEL_ID,
        purpose="calibration" if synthetic else "student_label",
        source_class="synthetic" if synthetic else "public",
        prompt=prompt,
        authorization_basis=AUTHORIZATION_BASIS,
    )
    return prompt


def load_inputs(phase: str) -> list[dict[str, Any]]:
    if phase == "smoke":
        rows = json.loads((ROOT / SMOKE_REL).read_text())
    else:
        rows = [json.loads(line) for line in (ROOT / SOURCE_REL).read_text().splitlines()]
    expected = 4 if phase == "smoke" else 100
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError("author input count or source IDs changed")
    protocol = json.loads((ROOT / PROTOCOL_REL).read_text())
    for row in rows:
        prompt = source_preflight(row, synthetic=phase == "smoke")
        if phase == "smoke":
            language = row["id"].rsplit("/", 1)[-1]
            if sha_bytes(prompt.encode()) != protocol["smoke_prompt_sha256_by_language"].get(
                language
            ):
                raise ValueError("golden indexed smoke prompt changed")
    return rows


def request_id(phase: str, source_id: str) -> str:
    return "author10-" + phase + "-" + sha_bytes(source_id.encode())[:24]


def completed_rows(path: Path, phase: str, sources: list[dict[str, Any]]) -> dict[str, dict]:
    expected = {row["id"]: row for row in sources}
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        source_id = row.get("source_id")
        if source_id not in expected or source_id in completed:
            raise ValueError("author artifact has unknown or duplicate source")
        if row.get("plan_sha256") != PLAN_SHA or row.get("protocol_sha256") != PROTOCOL_SHA:
            raise ValueError("author artifact frozen identity changed")
        if row.get("request_id") != request_id(phase, source_id):
            raise ValueError("author artifact request identity changed")
        prompt = source_preflight(expected[source_id], synthetic=phase == "smoke")
        if row.get("prompt_sha256") != sha_bytes(prompt.encode()):
            raise ValueError("author artifact prompt identity changed")
        if row.get("raw_output_sha256") != sha_bytes(row["raw_content"].encode()):
            raise ValueError("author artifact raw hash changed")
        completed[source_id] = row
    return completed


def format_shape(value: dict[str, object]) -> bool:
    if set(value) != {"prior_edit", "target_row", "action", "intent_evidence", "objective"}:
        return False
    prior, action, objective = value["prior_edit"], value["action"], value["objective"]
    return (
        isinstance(prior, dict)
        and set(prior) == {"row", "old_text", "new_text"}
        and type(prior["row"]) is int
        and isinstance(prior["old_text"], str)
        and isinstance(prior["new_text"], str)
        and isinstance(action, dict)
        and set(action) == {"kind", "text"}
        and action["kind"] in ("N", "D", "R", "I")
        and (isinstance(action["text"], str) or action["text"] is None)
        and isinstance(objective, dict)
        and set(objective) == {"kind", "description", "checks"}
        and isinstance(objective["kind"], str)
        and isinstance(objective["description"], str)
        and isinstance(objective["checks"], list)
        and type(value["target_row"]) is int
        and isinstance(value["intent_evidence"], str)
    )


def evaluate_response(
    phase: str, source: dict[str, Any], content: str, finish: str | None, tokenizer: object
) -> tuple[str, str | None, dict[str, Any] | None]:
    try:
        block = extract_author_candidate_block(content, provider_complete=finish == "stop")
        if not format_shape(block.value):
            raise TeacherCandidateError("author JSON shape invalid")
        parsed_source = source
        if phase == "smoke":
            # The synthetic fixture omits aliases because it has no public repository.
            # Fill only that provenance field for the shared replay validator.
            parsed_source = {
                **source,
                "authoring_metadata": {
                    **source["authoring_metadata"],
                    "source_aliases": [source["authoring_metadata"]["source_repo"]],
                },
            }
        candidate = parse_author_response(block.json_text, parsed_source, tokenizer)
        if candidate["validation"]["accepted_training"] is not False:
            raise ValueError("author candidate was unexpectedly promoted")
        return "candidate_preflight", None, candidate
    except (TeacherCandidateError, ValueError) as exc:
        return "rejected", str(exc), None


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize(phase: str, rows: dict[str, dict], protocol: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows.values())
    valid = counts["candidate_preflight"]
    required = protocol["synthetic_smoke"]["minimum_structurally_valid"]
    result = {
        "phase": phase,
        "plan_sha256": PLAN_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "source_sha256": SMOKE_SHA if phase == "smoke" else SOURCE_SHA,
        "raw_artifact_sha256": sha_file(RAW[phase]),
        "raw_artifact": str(RAW[phase].relative_to(ROOT)),
        "counts": dict(counts),
        "completed_calls": len(rows),
        "structurally_valid_or_candidate_preflight": valid,
        "accepted_training": 0,
        "reported_input_tokens": sum(row["input_tokens"] for row in rows.values()),
        "reported_generated_tokens": sum(row["generated_tokens"] for row in rows.values()),
        "reported_cost_usd": sum(row["reported_cost_usd"] or 0 for row in rows.values()),
    }
    if phase == "smoke":
        result["format_gate_passed"] = len(rows) == 4 and valid >= required
    return result


def assert_smoke_gate() -> None:
    if not RESULT["smoke"].exists():
        raise ValueError("synthetic author format smoke has not completed")
    smoke = json.loads(RESULT["smoke"].read_text())
    if (
        smoke.get("plan_sha256") != PLAN_SHA
        or smoke.get("protocol_sha256") != PROTOCOL_SHA
        or smoke.get("format_gate_passed") is not True
        or smoke.get("completed_calls") != 4
        or smoke.get("raw_artifact_sha256") != sha_file(RAW["smoke"])
    ):
        raise ValueError("synthetic author format gate is not verified")


def run(phase: str, *, execute: bool) -> dict[str, Any]:
    protocol = verify_frozen()
    sources = load_inputs(phase)
    completed = completed_rows(RAW[phase], phase, sources)
    if phase == "public" and execute:
        assert_smoke_gate()
    if not execute:
        return {"phase": phase, "preflight_sources": len(sources), "completed": len(completed)}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    ledger = TeacherUsageLedger(LEDGER)
    with ledger.path.open("a+", encoding="utf-8") as handle:
        ledger_rows = ledger._read(handle)
    for source_id in completed:
        entry = ledger_rows.get(request_id(phase, source_id))
        if entry is None or "input" not in entry:
            raise ValueError("author output lacks settled ledger entry")
    if len(completed) < len(sources):
        with OpenCodeTeacherClient(ledger) as client:
            for source in sources:
                source_id = source["id"]
                if source_id in completed:
                    continue
                rid = request_id(phase, source_id)
                if rid in ledger_rows:
                    raise ValueError("ambiguous author request; no automatic retry")
                prompt = source_preflight(source, synthetic=phase == "smoke")
                response = client.run_role(
                    request_id=rid,
                    prompt=prompt,
                    purpose="calibration" if phase == "smoke" else "student_label",
                    source_class="synthetic" if phase == "smoke" else "public",
                    authorization_basis=AUTHORIZATION_BASIS,
                    reserve_input_tokens=10_000,
                    reserve_output_tokens=8_192,
                    system_instruction=protocol["system_instruction"],
                )
                status, rejection, candidate = evaluate_response(
                    phase, source, response.content, response.finish_reason, tokenizer
                )
                if candidate is not None:
                    candidate["provenance"]["author_actor_id"] = response.model_id
                    candidate["provenance"]["author_session_id"] = response.session_id
                    candidate["provenance"]["author_response_id"] = response.response_id
                row = {
                    "plan_sha256": PLAN_SHA,
                    "protocol_sha256": PROTOCOL_SHA,
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
                append_row(RAW[phase], row)
                completed[source_id] = row
                ledger_rows[rid] = {"input": response.input_tokens}
                print(json.dumps({"source_id": source_id, "status": status}), flush=True)
    summary = summarize(phase, completed, protocol)
    RESULT[phase].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "public"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(args.phase, execute=args.execute)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
