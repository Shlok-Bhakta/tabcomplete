"""Audit the completed plan-10/11 public author pilot without accepting labels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import run_one_line_author_pilot_v2 as pilot

from tinycomplete.one_line.context import serialize_state_bounded
from tinycomplete.one_line.contract import EditAction, EditState, apply_action, encode_action
from tinycomplete.one_line.data import connected_groups, replay_replacement_history

SAMPLE_SIZE = 12
SAMPLE_SALT = b"plan10-human-audit-v1\0"
COMBINED_SUMMARY = pilot.REPORT / "author_pilot_combined_plan11.json"
AUDIT = pilot.REPORT / "author_pilot_combined_plan11_audit.json"
BLIND = pilot.REPORT / "author_pilot_combined_plan11_blind_sample.jsonl"
PARTIAL_INCIDENT = pilot.REPORT / "author_pilot_plan10_usage_incident.json"
PARTIAL_AUDIT = pilot.REPORT / "author_pilot_plan10_partial7_audit.json"
PARTIAL_BLIND = pilot.REPORT / "author_pilot_plan10_partial7_blind.jsonl"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_source_ids(source_ids: list[str], limit: int = SAMPLE_SIZE) -> list[str]:
    """Rank all eligible IDs by a fixed SHA key, with lexical tie breaking."""
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate eligible source ID")
    return sorted(
        source_ids,
        key=lambda source_id: (sha_bytes(SAMPLE_SALT + source_id.encode("utf-8")), source_id),
    )[:limit]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def _artifact_path(relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("combined summary has no raw artifact path")
    root = pilot.ROOT.resolve()
    path = (root / relative).resolve()
    allowed = root / "artifacts/research/one_line_r1"
    if path.parent != allowed or path.suffix != ".jsonl":
        raise ValueError("combined summary raw artifact path is outside the pilot directory")
    return path


def _read_artifact(
    path: Path,
    *,
    expected_hash: object,
    expected_plan_sha: str,
    sources: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(expected_hash, str) or pilot.sha_file(path) != expected_hash:
        raise ValueError("public raw artifact differs from completed combined summary")
    rows: dict[str, dict[str, Any]] = {}
    request_ids: set[str] = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        source_id = record.get("source_id")
        if source_id not in sources or source_id in rows:
            raise ValueError("unknown or duplicate source ID in public raw artifact")
        if (
            record.get("plan_sha256") != expected_plan_sha
            or record.get("protocol_sha256") != pilot.PROTOCOL_SHA
        ):
            raise ValueError("public raw row has a different frozen plan or protocol")
        prompt = pilot.source_preflight(sources[source_id], synthetic=False)
        if record.get("prompt_sha256") != sha_bytes(prompt.encode("utf-8")):
            raise ValueError("public raw row has a different author prompt")
        if record.get("source_sha256") != sources[source_id]["authoring_metadata"]["source_sha256"]:
            raise ValueError("public raw row has a different public source")
        content = record.get("raw_content")
        if not isinstance(content, str) or record.get("raw_output_sha256") != sha_bytes(
            content.encode("utf-8")
        ):
            raise ValueError("public raw output hash mismatch")
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise ValueError("missing or duplicate public request ID")
        if record.get("accepted_training") is not False:
            raise ValueError("raw public row was unexpectedly accepted for training")
        rows[source_id] = record
        request_ids.add(request_id)
    return rows


def audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # The combined summary is written after 99 durable rows and one censored
    # request. Refuse to open either append-only raw artifact until it exists.
    if not COMBINED_SUMMARY.exists():
        raise ValueError("combined pilot summary absent; do not inspect partial results")
    completed = json.loads(COMBINED_SUMMARY.read_text())
    if completed.get("durable_rows") != 99 or completed.get("accepted_training") != 0:
        raise ValueError("combined pilot incomplete or accepted labels unexpectedly changed")
    if (
        completed.get("protocol_sha256") != pilot.PROTOCOL_SHA
        or completed.get("source_sha256") != pilot.SOURCE_SHA
    ):
        raise ValueError("combined pilot has a different frozen protocol or source")
    frozen = (
        (pilot.ROOT / pilot.PROTOCOL_REL, pilot.PROTOCOL_SHA),
        (pilot.ROOT / pilot.BUILDER_REL, pilot.BUILDER_SHA),
        (pilot.ROOT / pilot.SOURCE_REL, pilot.SOURCE_SHA),
        (pilot.ROOT / pilot.SMOKE_REL, pilot.SMOKE_SHA),
        (pilot.MODEL_SNAPSHOT / "tokenizer.json", pilot.TOKENIZER_SHA),
    )
    if any(pilot.sha_file(path) != digest for path, digest in frozen):
        raise ValueError("frozen protocol, source, smoke, builder, or tokenizer changed")
    plan11_sha = pilot.sha_file(pilot.PLAN)
    if json.loads(pilot.PLAN.read_text()).get("plan_revision") != 11:
        raise ValueError("current plan is not revision 11")
    sources = pilot.load_inputs("public")
    source_by_id = {row["id"]: row for row in sources}
    plan10_path = _artifact_path(completed.get("plan10_raw_artifact"))
    plan11_path = _artifact_path(completed.get("plan11_raw_artifact"))
    if plan10_path != pilot.RAW["public"].resolve() or plan10_path == plan11_path:
        raise ValueError("combined summary does not identify distinct plan-10/11 artifacts")
    plan10_rows = _read_artifact(
        plan10_path,
        expected_hash=completed.get("plan10_raw_artifact_sha256"),
        expected_plan_sha=pilot.PLAN_SHA,
        sources=source_by_id,
    )
    plan11_rows = _read_artifact(
        plan11_path,
        expected_hash=completed.get("plan11_raw_artifact_sha256"),
        expected_plan_sha=plan11_sha,
        sources=source_by_id,
    )
    failed_id = completed.get("failed_source_id")
    raw = {**plan10_rows, **plan11_rows}
    if (
        len(plan10_rows) != 7
        or len(plan11_rows) != 92
        or len(raw) != 99
        or set(plan10_rows) & set(plan11_rows)
        or not isinstance(failed_id, str)
        or set(raw) | {failed_id} != set(source_by_id)
        or failed_id in raw
    ):
        raise ValueError("combined result does not partition the frozen 100 seeds")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pilot.MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    candidates: list[dict[str, Any]] = []
    candidate_by_source: dict[str, dict[str, Any]] = {}
    replay_count = 0
    input_tokens: list[int] = []
    target_tokens: list[int] = []
    input_omitted_rows: list[int] = []
    for source_id, record in raw.items():
        candidate = record.get("candidate")
        if record["status"] != "candidate_preflight":
            if candidate is not None:
                raise ValueError("rejected row contains a candidate")
            continue
        if not isinstance(candidate, dict):
            raise ValueError("preflight row lacks candidate")
        validation = candidate["validation"]
        if validation.get("accepted_training") is not False:
            raise ValueError("candidate was unexpectedly accepted")
        metadata = source_by_id[source_id]["authoring_metadata"]
        if (
            candidate["provenance"]["source_id"] != source_id
            or candidate["source_repo"] != metadata["source_repo"]
            or candidate["source_revision"] != metadata["source_revision"]
        ):
            raise ValueError("candidate source identity differs from frozen seed")
        state = EditState.from_mapping(candidate["state"])
        action = EditAction(**candidate["action"])
        source = source_by_id[source_id]["student_state_seed"]
        reconstructed = replay_replacement_history(
            source["source"],
            state.history,
            file_id=state.file_id,
            filetype=state.filetype,
        )
        if (
            reconstructed != state.source
            or apply_action(state, action) != candidate["after_source"]
        ):
            raise ValueError("candidate replay mismatch")
        context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
        if context.input_tokens is None:
            raise ValueError("candidate context token count is unavailable")
        response_tokens = len(tokenizer.encode(encode_action(action), add_special_tokens=False)) + 1
        if (
            context.input_tokens != validation["input_tokens"]
            or context.included_history != 1
            or response_tokens != validation["response_tokens_including_eos"]
            or response_tokens > 64
            or context.input_tokens + response_tokens > 2048
        ):
            raise ValueError("candidate context or token budget mismatch")
        if any(
            validation.get(flag) is not True
            for flag in (
                "history_replays_to_state",
                "apply_reconstructs_after",
                "history_visible_in_prompt",
            )
        ):
            raise ValueError("candidate validation flags disagree with replay")
        if (
            validation.get("reviewer_verified") is not False
            or validation.get("blind_solver_verified") is not False
        ):
            raise ValueError("unreviewed candidate carries a review claim")
        replay_count += 1
        input_tokens.append(context.input_tokens)
        target_tokens.append(response_tokens)
        input_omitted_rows.append(context.omitted_rows)
        candidates.append(candidate)
        candidate_by_source[source_id] = candidate

    selected = selected_source_ids(list(candidate_by_source))
    blind: list[dict[str, Any]] = []
    sample_index: list[dict[str, str]] = []
    for source_id in selected:
        candidate = candidate_by_source[source_id]
        context = serialize_state_bounded(
            EditState.from_mapping(candidate["state"]), tokenizer, max_input_tokens=1024
        )
        prompt_hash = sha_bytes(context.text.encode("utf-8"))
        blind.append(
            {
                "source_id": source_id,
                "candidate_id": candidate["id"],
                "student_prompt": context.text,
                "student_prompt_sha256": prompt_hash,
            }
        )
        sample_index.append(
            {
                "source_id": source_id,
                "candidate_id": candidate["id"],
                "selection_sha256": sha_bytes(SAMPLE_SALT + source_id.encode("utf-8")),
                "student_prompt_sha256": prompt_hash,
            }
        )

    action_counts = Counter(candidate["action"]["kind"] for candidate in candidates)
    repo_counts = Counter(candidate["source_repo"] for candidate in candidates)
    mechanism_counts = Counter(candidate["mechanism"] for candidate in candidates)
    objective_counts = Counter(
        candidate["provenance"]["objective"]["kind"] for candidate in candidates
    )
    groups = connected_groups(candidates)
    result: dict[str, Any] = {
        "plan10_sha256": pilot.PLAN_SHA,
        "plan11_sha256": plan11_sha,
        "protocol_sha256": pilot.PROTOCOL_SHA,
        "source_sha256": pilot.SOURCE_SHA,
        "plan10_raw_artifact_sha256": completed["plan10_raw_artifact_sha256"],
        "plan11_raw_artifact_sha256": completed["plan11_raw_artifact_sha256"],
        "durable_rows": 99,
        "failed_source_id": failed_id,
        "censored_failed_requests": 1,
        "accepted_training": 0,
        "status_counts": dict(sorted(Counter(row["status"] for row in raw.values()).items())),
        "candidate_replay_verified": replay_count,
        "candidate_input_tokens": _distribution(input_tokens),
        "candidate_target_tokens_including_eos": _distribution(target_tokens),
        "candidate_total_nonpadding_tokens": sum(input_tokens) + sum(target_tokens),
        "candidate_omitted_source_rows": _distribution(input_omitted_rows),
        "provider_reported_input_tokens": _distribution(
            [row["input_tokens"] for row in raw.values()]
        ),
        "provider_reported_generated_tokens": _distribution(
            [row["generated_tokens"] for row in raw.values()]
        ),
        "action_counts": {
            kind: action_counts[kind]
            for kind in ("keep", "delete_line", "replace_line", "insert_before")
        },
        "source_repo_counts": dict(sorted(repo_counts.items())),
        "source_seed_repo_counts": dict(
            sorted(Counter(row["authoring_metadata"]["source_repo"] for row in sources).items())
        ),
        "source_group_count": len(groups),
        "largest_connected_group": max((len(group) for group in groups), default=0),
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "author_objective_kind_counts_unreviewed": dict(sorted(objective_counts.items())),
        "sample_rule": (
            "candidate_preflight rows ranked by SHA256("
            "plan10-human-audit-v1\\0 + UTF-8 source_id); first 12"
        ),
        "sample_index": sample_index,
        "role_caveat": (
            "Same Muse model may author, solve, and review in separate sessions. "
            "Actor IDs show role isolation, not independent model families or human review."
        ),
        "smoke_caveat": "The 3/4 synthetic smoke was a format gate. Public rows remain candidates.",
    }
    return result, blind


def audit_partial() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Review seven settled rows; the eighth overrun remains censored."""
    incident = json.loads(PARTIAL_INCIDENT.read_text())
    if (
        incident.get("plan_revision") != 10
        or incident.get("plan_sha256") != pilot.PLAN_SHA
        or incident.get("durable_candidates") != 7
        or incident.get("failed_completed_attempts") != 1
        or incident.get("failed_status") != "settled_overrun_not_candidate"
        or incident.get("accepted_training") != 0
        or incident.get("durable_raw_artifact") != str(pilot.RAW["public"].relative_to(pilot.ROOT))
    ):
        raise ValueError("plan-10 partial incident identity or denominator changed")
    frozen = (
        (pilot.ROOT / pilot.PROTOCOL_REL, pilot.PROTOCOL_SHA),
        (pilot.ROOT / pilot.BUILDER_REL, pilot.BUILDER_SHA),
        (pilot.ROOT / pilot.SOURCE_REL, pilot.SOURCE_SHA),
        (pilot.ROOT / pilot.SMOKE_REL, pilot.SMOKE_SHA),
        (pilot.MODEL_SNAPSHOT / "tokenizer.json", pilot.TOKENIZER_SHA),
    )
    if any(pilot.sha_file(path) != digest for path, digest in frozen):
        raise ValueError("frozen plan-10 author inputs changed")
    sources = pilot.load_inputs("public")
    source_by_id = {row["id"]: row for row in sources}
    failed_id = incident.get("failed_source_id")
    if failed_id != sources[7]["id"] or failed_id in {row["id"] for row in sources[:7]}:
        raise ValueError("excluded overrun source is not the eighth frozen seed")
    if (
        incident.get("failed_request_id") != pilot.request_id("public", failed_id)
        or incident.get("ledger_overrun_recorded") is not True
    ):
        raise ValueError("excluded overrun request or ledger evidence is inconsistent")
    evidence_rel = incident.get("failed_evidence_artifact")
    if not isinstance(evidence_rel, str):
        raise ValueError("excluded overrun evidence path is missing")
    evidence_path = (pilot.ROOT / evidence_rel).resolve()
    if (
        evidence_path.parent != (pilot.ROOT / "artifacts/research/one_line_r1").resolve()
        or evidence_path.suffix != ".json"
        or pilot.sha_file(evidence_path) != incident.get("failed_evidence_sha256")
    ):
        raise ValueError("excluded overrun evidence hash mismatch")
    raw = _read_artifact(
        pilot.RAW["public"],
        expected_hash=incident.get("durable_raw_sha256"),
        expected_plan_sha=pilot.PLAN_SHA,
        sources=source_by_id,
    )
    if set(raw) != {row["id"] for row in sources[:7]} or failed_id in raw:
        raise ValueError("partial raw rows are not exactly the first seven settled seeds")
    if set(pilot.completed_rows(pilot.RAW["public"], "public", sources)) != set(raw):
        raise ValueError("plan-10 runner and audit disagree on settled row identity")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pilot.MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    candidate_by_source: dict[str, dict[str, Any]] = {}
    context_by_source: dict[str, str] = {}
    input_tokens: list[int] = []
    target_tokens: list[int] = []
    python_ast_parse_before_after = 0
    for source_id, record in raw.items():
        candidate = record.get("candidate")
        if record.get("status") != "candidate_preflight" or not isinstance(candidate, dict):
            raise ValueError("one of the seven durable rows is not a preflight candidate")
        validation = candidate["validation"]
        if (
            validation.get("accepted_training") is not False
            or validation.get("reviewer_verified") is not False
            or validation.get("objective_verified") is not False
            or validation.get("blind_solver_verified") is not False
        ):
            raise ValueError("partial candidate carries an unsupported validation claim")
        metadata = source_by_id[source_id]["authoring_metadata"]
        if (
            candidate["provenance"]["source_id"] != source_id
            or candidate["source_repo"] != metadata["source_repo"]
            or candidate["source_revision"] != metadata["source_revision"]
        ):
            raise ValueError("partial candidate source identity mismatch")
        state = EditState.from_mapping(candidate["state"])
        action = EditAction(**candidate["action"])
        before = replay_replacement_history(
            source_by_id[source_id]["student_state_seed"]["source"],
            state.history,
            file_id=state.file_id,
            filetype=state.filetype,
        )
        after = apply_action(state, action)
        if before != state.source or after != candidate["after_source"]:
            raise ValueError("partial candidate history or action replay mismatch")
        if state.filetype == "python":
            try:
                ast.parse(before)
                ast.parse(after)
            except SyntaxError:
                pass
            else:
                python_ast_parse_before_after += 1
        if (
            sha_bytes(before.encode("utf-8")) != validation["source_before_sha256"]
            or sha_bytes(after.encode("utf-8")) != validation["source_after_sha256"]
        ):
            raise ValueError("partial candidate before/after hash mismatch")
        context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
        if context.input_tokens is None:
            raise ValueError("partial candidate context token count is unavailable")
        target_count = len(tokenizer.encode(encode_action(action), add_special_tokens=False)) + 1
        if (
            context.input_tokens != validation["input_tokens"]
            or context.included_history != 1
            or target_count != validation["response_tokens_including_eos"]
            or target_count > 64
            or context.input_tokens + target_count > 2048
        ):
            raise ValueError("partial candidate student token budget mismatch")
        candidate_by_source[source_id] = candidate
        context_by_source[source_id] = context.text
        input_tokens.append(context.input_tokens)
        target_tokens.append(target_count)

    ordered = selected_source_ids(list(candidate_by_source))
    blind = [
        {
            "source_id": source_id,
            "candidate_id": candidate_by_source[source_id]["id"],
            "student_prompt": context_by_source[source_id],
            "student_prompt_sha256": sha_bytes(context_by_source[source_id].encode("utf-8")),
        }
        for source_id in ordered
    ]
    candidates = list(candidate_by_source.values())
    result: dict[str, Any] = {
        "scope": "partial_first_seven_only_not_full_pilot",
        "plan10_sha256": pilot.PLAN_SHA,
        "protocol_sha256": pilot.PROTOCOL_SHA,
        "source_sha256": pilot.SOURCE_SHA,
        "raw_artifact_sha256": incident["durable_raw_sha256"],
        "usage_incident_sha256": pilot.sha_file(PARTIAL_INCIDENT),
        "excluded_overrun_evidence_sha256": incident["failed_evidence_sha256"],
        "settled_candidate_rows": 7,
        "excluded_overrun_source_id": failed_id,
        "excluded_overrun_request_id": incident["failed_request_id"],
        "accepted_training": 0,
        "mechanical_replay_verified": len(candidate_by_source),
        "python311_ast_parse_before_after": python_ast_parse_before_after,
        "student_input_tokens": _distribution(input_tokens),
        "target_tokens_including_eos": _distribution(target_tokens),
        "action_counts": dict(sorted(Counter(c["action"]["kind"] for c in candidates).items())),
        "source_repo_counts": dict(sorted(Counter(c["source_repo"] for c in candidates).items())),
        "mechanism_counts_unreviewed": dict(
            sorted(Counter(c["mechanism"] for c in candidates).items())
        ),
        "sample_rule": (
            "all seven candidate rows, fixed SHA ordering with plan10-human-audit-v1 salt"
        ),
        "sample_source_ids": ordered,
        "warning": (
            "This is a partial candidate audit. The usage-overrun attempt is censored. "
            "No student labels have been accepted."
        ),
    }
    return result, blind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial", action="store_true", help="review the seven settled plan-10 rows"
    )
    args = parser.parse_args()
    result, blind = audit_partial() if args.partial else audit()
    audit_path, blind_path = (PARTIAL_AUDIT, PARTIAL_BLIND) if args.partial else (AUDIT, BLIND)
    _atomic_json(audit_path, result)
    _atomic_jsonl(blind_path, blind)
    print(
        json.dumps(
            {
                "durable_rows": (
                    result["settled_candidate_rows"] if args.partial else result["durable_rows"]
                ),
                "censored_failed_requests": (
                    1 if args.partial else result["censored_failed_requests"]
                ),
                "accepted_training": result["accepted_training"],
                "audit": str(audit_path),
                "blind_sample": str(blind_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
