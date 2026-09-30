"""CPU-only gates for public-source one-line candidate labels.

Author prose is never an objective. A separately reviewed, hash-pinned fixture
must evaluate the gold state and controls before this checker can accept a row.
The caller must freeze the manifest digest and verify reviewer independence;
neither claim can be inferred from an author's response.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from tinycomplete.eval.code_benchmark import (
    BenchmarkCase,
    CheckSpec,
    Prediction,
    evaluate_prediction,
)
from tinycomplete.one_line.context import serialize_state_bounded
from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    PhysicalLine,
    apply_action,
    decode_action,
    encode_action,
    physical_lines,
)
from tinycomplete.one_line.data import (
    replay_replacement_history,
    sha256_bytes,
    validate_splits,
)
from tinycomplete.one_line.pilot_roles import (
    build_blind_solver_prompt,
    build_reviewer_prompt,
    parse_review_response,
)

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_REVISION = re.compile(r"[a-f0-9]{40,64}\Z")
_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"})
_WRONG_TEXTS = ("__tabcomplete_wrong_edit_control_1__", "__tabcomplete_wrong_edit_control_2__")
_SESSION = re.compile(r"[A-Za-z0-9_.:-]{8,128}\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{8,160}\Z")
_RESPONSE_ID = re.compile(r"[A-Za-z0-9_.:-]{8,160}\Z")
_ROLE_EVIDENCE_V8 = "one-line-role-evidence-frozen-v8-v1"
_FROZEN_V8_PACKET = {
    "plan_file_sha256": "de41567f02fee65725d29a7984307b5bf5a843165b3a03ebcb3814b1c4f4ae49",
    "plan_canonical_sha256": "abead2f6e6106db75e8793a172921354ad4f9499b0fa902a2f5d692828b9f15d",
    "artifact_manifest_sha256": "5b8e313ce4b240b012f5cdcc0a865ee54a42e1f99e26801f95018669a7c46ebe",
    "inputs_sha256": "937357690bbfc930ab1a353bb688c80a825d55d6a8ff7a5a3c28e428ac62c2de",
    "oracle_fixtures_sha256": "3fc96642f21280f6952867944c83cfdc4bc558d096215c23536b27d4e196a57c",
    "preflight_result_sha256": "12815e64e374d7bf9f5d970ee94c7b0f080c186e8bc425bfaa677e7209ff8eea",
    "license_scope_audit_sha256": (
        "c6ba8efb36aee1b9800d16fd02043c30f56396c164884d4b0e2a3d39c0aa3f92"
    ),
    "qualification_sha256": "aad4eb4b6cbb9c61b495695e308443f5b399736b3f7600847760655eeec518dc",
    "qualification_events_sha256": (
        "9d8ec61a01228f48b23240638226453dba668f43789db08923c08aa53a9aa984"
    ),
    "role_event_log_hashes": {
        "author": "6d74b5f2dba2bbf523fb728e4870ad883d8f6bdf7d2bcfc165667a82939e8968",
        "solver": "1bcde98f06f0d74bd800e633d2a28c345017c4d44bd2e83bab99e85c8e1d4f58",
        "reviewer": "13e0ccb077a22ecb85a2b0c81912c5e69e2a5aa92110bab897f69b93eb33bce7",
    },
}
_FROZEN_V8_CASES = {
    "synthetic-a": {
        "input_row_sha256": "f0ab66bfe4f68dea46e5e6eeb4ab1341c45bdb26486acac88beddef956258302",
        "source_parent_sha256": "6bcdc4567f658e6bf9c02f759f769ce82d589e71e527fc4742755fb88707705b",
        "source_seed_sha256": "292b2d47f60d3f9cd96d29f9e93b7f9cbd309f8b3fc42e9cd1342de978b928ca",
        "source_transform_sha256": (
            "dcbd2d99c2a18ac7e1a16b6c0c847df3ecbbb6874d7e09640fce63d30b841e70"
        ),
        "prompt_sha256": {
            "author": "f3fb1c5c1590752e2d65f99b530e9cf09c23305ae7578b4ff645837248561f17",
            "solver": "204693ca882b61a3abe97914c8091320b1631dcfd2165134d3d8730dd701a512",
        },
        "request_ids": {
            "author": "two-seed-source-grounded-v2-author-3b0a7ca3a01569f4af07",
            "solver": "two-seed-source-grounded-v2-solver-3b0a7ca3a01569f4af07",
            "reviewer": "two-seed-source-grounded-v2-reviewer-3b0a7ca3a01569f4af07",
        },
    },
    "synthetic-b": {
        "input_row_sha256": "67775d8c563189b041cc51eab801de71918aa7a4d464d4d51f20dcf9430e8722",
        "source_parent_sha256": "85dcf3906393521abadfe7ef06b710e8a7ded2cc33664891036a25ec75bd64d5",
        "source_seed_sha256": "85dcf3906393521abadfe7ef06b710e8a7ded2cc33664891036a25ec75bd64d5",
        "source_transform_sha256": (
            "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"
        ),
        "prompt_sha256": {
            "author": "80750e1ddcc9deb260446bcc3e3cf48bc8248ea19eae4e229a93ab504b71fbae",
            "solver": "01dd85437c6d2412f2ec8dad158b0fbe7e9edd8c29285e82125b768278c49dda",
        },
        "request_ids": {
            "author": "two-seed-source-grounded-v2-author-11ae3038c1fe627502b6",
            "solver": "two-seed-source-grounded-v2-solver-11ae3038c1fe627502b6",
            "reviewer": "two-seed-source-grounded-v2-reviewer-11ae3038c1fe627502b6",
        },
    },
}


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    evidence: dict[str, Any]


def _reject(reason: str, **evidence: Any) -> AcceptanceDecision:
    return AcceptanceDecision(False, reason, evidence)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate objective manifest key")
        result[key] = value
    return result


def _objective_record(
    manifest_bytes: bytes,
    expected_sha256: str,
    candidate_id: str,
    source_id: str,
    author_actor_id: str,
) -> Mapping[str, Any]:
    if len(manifest_bytes) > 256_000 or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("objective manifest is unbounded or unpinned")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256:
        raise ValueError("objective manifest hash mismatch")
    try:
        raw = json.loads(manifest_bytes, object_pairs_hook=_strict_pairs)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("objective manifest is not strict JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"schema", "reviewer_id", "entries"}:
        raise ValueError("objective manifest shape is invalid")
    if raw["schema"] != "one-line-independent-objective-v1":
        raise ValueError("objective manifest schema is invalid")
    reviewer = raw["reviewer_id"]
    if not isinstance(reviewer, str) or not reviewer or reviewer == author_actor_id:
        raise ValueError("objective reviewer is absent or equals author")
    entries = raw["entries"]
    if not isinstance(entries, list) or len(entries) > 1000:
        raise ValueError("objective entries are unbounded")
    matches = [
        row for row in entries if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1 or matches[0].get("source_id") != source_id:
        raise ValueError("candidate lacks one independent objective record")
    return cast(Mapping[str, Any], matches[0])


def _split_record(
    manifest_bytes: bytes,
    expected_sha256: str,
    candidate_id: str,
    source_repo: str,
) -> Mapping[str, Any]:
    if len(manifest_bytes) > 256_000 or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("split manifest is unbounded or unpinned")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256:
        raise ValueError("split manifest hash mismatch")
    try:
        raw = json.loads(manifest_bytes, object_pairs_hook=_strict_pairs)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("split manifest is not strict JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"schema", "status", "assignments"}:
        raise ValueError("split manifest shape is invalid")
    if raw["schema"] != "one-line-accepted-split-v1" or raw["status"] != "frozen":
        raise ValueError("split manifest is not frozen")
    assignments = raw["assignments"]
    if not isinstance(assignments, list) or len(assignments) > 1000:
        raise ValueError("split assignments are unbounded")
    matches = [
        row
        for row in assignments
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1 or matches[0].get("source_repo") != source_repo:
        raise ValueError("candidate lacks one split assignment")
    return cast(Mapping[str, Any], matches[0])


def _parse_pinned_artifact(data: bytes | None, digest: str | None, *, label: str) -> Any:
    if data is None or digest is None or len(data) > 256_000 or not _SHA256.fullmatch(digest):
        raise ValueError(f"{label} artifact is absent or unbounded")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{label} artifact hash mismatch")
    try:
        return json.loads(data, object_pairs_hook=_strict_pairs)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"{label} artifact is not strict JSON") from error


def _license_review_verified(
    data: bytes | None,
    digest: str | None,
    *,
    source_row: Mapping[str, Any],
    author_actor_id: str,
) -> bool:
    raw = _parse_pinned_artifact(data, digest, label="file-license review")
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "source_id",
        "source_repo",
        "source_revision",
        "source_sha256",
        "source_path",
        "license_spdx",
        "license_sha256",
        "reviewer_id",
        "finding",
        "evidence",
    }:
        return False
    metadata = source_row["authoring_metadata"]
    expected = {
        "schema": "one-line-file-license-review-v1",
        "source_id": source_row["id"],
        "source_repo": metadata["source_repo"],
        "source_revision": metadata["source_revision"],
        "source_sha256": metadata["source_sha256"],
        "source_path": metadata["source_path"],
        "license_spdx": metadata["source_license"],
        "license_sha256": metadata["license_sha256"],
        "finding": "license_covers_file",
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        return False
    reviewer, evidence = raw["reviewer_id"], raw["evidence"]
    return (
        isinstance(reviewer, str)
        and bool(reviewer)
        and reviewer != author_actor_id
        and isinstance(evidence, str)
        and 20 <= len(evidence) <= 1000
    )


def _source_seed_verified(source_row: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    """Verify either exact source bytes or a single explicitly pinned task transform."""
    seed = source_row.get("student_state_seed")
    if not isinstance(seed, Mapping) or not isinstance(seed.get("source"), str):
        return False
    seed_bytes = seed["source"].encode("utf-8")
    seed_sha256 = sha256_bytes(seed_bytes)
    transform = source_row.get("synthetic_source_transform")
    parent = source_row.get("public_source_parent")
    metadata_seed_sha256 = metadata.get("synthetic_seed_sha256")
    metadata_transform_sha256 = metadata.get("synthetic_source_transform_sha256")
    if metadata_seed_sha256 is not None and metadata_seed_sha256 != seed_sha256:
        return False
    if transform is None:
        if seed_sha256 != metadata.get("source_sha256"):
            return False
        if parent is None:
            return metadata_transform_sha256 is None
        if not isinstance(parent, str) or parent.encode("utf-8") != seed_bytes:
            return False
        if sha256_bytes(parent.encode("utf-8")) != metadata.get("source_sha256"):
            return False
        if metadata_transform_sha256 is not None:
            none_digest = sha256_bytes(
                json.dumps(None, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            return metadata_transform_sha256 == none_digest
        return True
    if not isinstance(transform, Mapping) or not isinstance(parent, str):
        return False
    transform_digest = sha256_bytes(
        json.dumps(
            dict(transform), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    if (
        metadata_transform_sha256 is not None and metadata_transform_sha256 != transform_digest
    ) or (
        transform.get("schema") == "one-line-public-source-transform-v2"
        and metadata_transform_sha256 != transform_digest
    ):
        return False
    if transform.get("schema") == "one-line-public-source-transform-v2":
        expected_v2_keys = {
            "schema",
            "kind",
            "parent_source_sha256",
            "seed_source_sha256",
            "edits",
            "signature_prestate_row",
            "privacy_redaction_rows",
            "plan_file_sha256",
            "input_bundle_sha256",
            "upstream_declared_transform",
        }
        edits = transform.get("edits")
        declared = transform.get("upstream_declared_transform")
        if (
            set(transform) != expected_v2_keys
            or transform.get("kind")
            != "replace_explicit_physical_lines_for_synthetic_public_source"
            or not isinstance(edits, list)
            or not edits
            or len(edits) > 32
            or type(transform.get("signature_prestate_row")) is not int
            or not isinstance(transform.get("privacy_redaction_rows"), list)
            or not isinstance(declared, Mapping)
            or set(declared)
            != {
                "kind",
                "public_snapshot_line_sha256",
                "purpose",
                "row",
                "synthetic_author_line_sha256",
            }
            or transform.get("parent_source_sha256") != metadata.get("source_sha256")
            or sha256_bytes(parent.encode("utf-8")) != metadata.get("source_sha256")
            or transform.get("seed_source_sha256") != seed_sha256
            or not _SHA256.fullmatch(str(transform.get("plan_file_sha256", "")))
            or not _SHA256.fullmatch(str(transform.get("input_bundle_sha256", "")))
        ):
            return False
        try:
            lines = list(physical_lines(parent.encode("utf-8")))
            previous_row = -1
            categories: dict[int, str] = {}
            for edit in edits:
                if not isinstance(edit, Mapping) or set(edit) != {
                    "row",
                    "old_line",
                    "new_line",
                    "old_line_sha256",
                    "new_line_sha256",
                    "classification",
                }:
                    return False
                row = edit.get("row")
                old_line, new_line = edit.get("old_line"), edit.get("new_line")
                if (
                    type(row) is not int
                    or row <= previous_row
                    or row < 0
                    or row >= len(lines)
                    or not isinstance(old_line, str)
                    or not isinstance(new_line, str)
                    or any(char in old_line + new_line for char in "\r\n")
                ):
                    return False
                previous_row = row
                classification = edit.get("classification")
                if classification not in {"synthetic_prestate_signature", "privacy_redaction"}:
                    return False
                categories[row] = classification
                line = lines[row]
                if line.content.decode("utf-8") != old_line:
                    return False
                if sha256_bytes(old_line.encode("utf-8")) != edit.get("old_line_sha256"):
                    return False
                if sha256_bytes(new_line.encode("utf-8")) != edit.get("new_line_sha256"):
                    return False
                lines[row] = PhysicalLine(new_line.encode("utf-8"), line.terminator)
            signature_row = transform["signature_prestate_row"]
            redaction_rows = transform["privacy_redaction_rows"]
            signature_edits = {
                edit["row"]: edit
                for edit in edits
                if isinstance(edit, Mapping)
                and edit.get("classification") == "synthetic_prestate_signature"
            }
            signature_edit = signature_edits.get(signature_row)
            if (
                set(signature_edits) != {signature_row}
                or categories.get(signature_row) != "synthetic_prestate_signature"
                or any(type(row) is not int for row in redaction_rows)
                or redaction_rows != sorted(set(redaction_rows))
                or any(categories.get(row) != "privacy_redaction" for row in redaction_rows)
                or {
                    row
                    for row, classification in categories.items()
                    if classification == "privacy_redaction"
                }
                != set(redaction_rows)
                or declared.get("row") != signature_row
                or declared.get("kind") != "synthetic_single_line_prestate_from_public_source"
                or declared.get("purpose")
                != "make the declared synthetic history replay to pinned source bytes"
                or not isinstance(signature_edit, Mapping)
                or declared.get("public_snapshot_line_sha256")
                != signature_edit.get("old_line_sha256")
                or declared.get("synthetic_author_line_sha256")
                != signature_edit.get("new_line_sha256")
            ):
                return False
            rebuilt = b"".join(line.raw for line in lines)
            return rebuilt == seed_bytes
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
            return False
    expected_keys = {
        "schema",
        "kind",
        "parent_source_sha256",
        "seed_source_sha256",
        "row",
        "old_line",
        "new_line",
        "old_line_sha256",
        "new_line_sha256",
        "plan_file_sha256",
        "input_bundle_sha256",
    }
    if (
        set(transform) != expected_keys
        or transform.get("schema") != "one-line-public-source-transform-v1"
    ):
        return False
    if transform.get("kind") != "replace_single_physical_line_for_synthetic_history":
        return False
    if (
        not _SHA256.fullmatch(str(transform.get("plan_file_sha256", "")))
        or not _SHA256.fullmatch(str(transform.get("input_bundle_sha256", "")))
        or sha256_bytes(parent.encode("utf-8")) != metadata.get("source_sha256")
        or transform.get("parent_source_sha256") != metadata.get("source_sha256")
        or seed_sha256 != transform.get("seed_source_sha256")
    ):
        return False
    row = transform.get("row")
    old_line, new_line = transform.get("old_line"), transform.get("new_line")
    if (
        type(row) is not int
        or row < 0
        or not isinstance(old_line, str)
        or not isinstance(new_line, str)
    ):
        return False
    if any(char in value for value in (old_line, new_line) for char in "\r\n"):
        return False
    if sha256_bytes(old_line.encode("utf-8")) != transform.get("old_line_sha256") or sha256_bytes(
        new_line.encode("utf-8")
    ) != transform.get("new_line_sha256"):
        return False
    try:
        lines = list(physical_lines(parent.encode("utf-8")))
        if row >= len(lines) or lines[row].content.decode("utf-8") != old_line:
            return False
        lines[row] = PhysicalLine(new_line.encode("utf-8"), lines[row].terminator)
        rebuilt = b"".join(line.raw for line in lines)
        return rebuilt == seed_bytes
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
        return False


def _role_evidence_verified(
    data: bytes | None,
    digest: str | None,
    *,
    candidate: Mapping[str, Any],
    state: EditState,
    tokenizer: Any,
    author_actor_id: str,
) -> bool:
    raw = _parse_pinned_artifact(data, digest, label="role evidence")
    if isinstance(raw, dict) and raw.get("schema") == _ROLE_EVIDENCE_V8:
        return _frozen_v8_role_evidence_verified(
            raw,
            candidate=candidate,
            state=state,
            tokenizer=tokenizer,
            author_actor_id=author_actor_id,
        )
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "candidate_id",
        "author",
        "solver",
        "reviewer",
    }:
        return False
    if raw["schema"] != "one-line-role-evidence-v1" or raw["candidate_id"] != candidate["id"]:
        return False
    author, solver, reviewer = raw["author"], raw["solver"], raw["reviewer"]
    if not all(isinstance(part, dict) for part in (author, solver, reviewer)):
        return False
    if (
        set(author) != {"actor_id", "session_id", "response_sha256"}
        or set(solver)
        != {
            "actor_id",
            "session_id",
            "prompt_sha256",
            "wire",
            "wire_sha256",
            "terminated",
            "generated_tokens",
        }
        or set(reviewer)
        != {
            "actor_id",
            "session_id",
            "prompt_sha256",
            "verdict_json",
            "verdict_sha256",
        }
    ):
        return False
    actors = [author["actor_id"], solver["actor_id"], reviewer["actor_id"]]
    sessions = [author["session_id"], solver["session_id"], reviewer["session_id"]]
    if (
        not all(isinstance(actor, str) and actor for actor in actors)
        or len(set(actors)) != 3
        or not all(isinstance(session, str) and _SESSION.fullmatch(session) for session in sessions)
        or len(set(sessions)) != 3
        or author["actor_id"] != author_actor_id
        or author["response_sha256"] != candidate["provenance"]["author_response_sha256"]
    ):
        return False
    wire = solver["wire"]
    if not isinstance(wire, str) or len(wire.encode("utf-8")) > 512:
        return False
    if solver["wire_sha256"] != sha256_bytes(wire.encode("utf-8")):
        return False
    if solver["prompt_sha256"] != sha256_bytes(
        build_blind_solver_prompt(candidate, tokenizer).encode("utf-8")
    ):
        return False
    if type(solver["terminated"]) is not bool or type(solver["generated_tokens"]) is not int:
        return False
    actual_tokens = len(tokenizer.encode(wire, add_special_tokens=False)) + 1
    if actual_tokens != solver["generated_tokens"] or actual_tokens > 64:
        return False
    decoded = decode_action(
        wire, terminated=solver["terminated"], generated_tokens=solver["generated_tokens"]
    )
    if decoded.status != "ok" or decoded.action is None:
        return False
    try:
        if apply_action(state, decoded.action) != candidate["after_source"]:
            return False
    except ValueError:
        return False
    verdict_json = reviewer["verdict_json"]
    if not isinstance(verdict_json, str) or len(verdict_json.encode("utf-8")) > 2048:
        return False
    if reviewer["verdict_sha256"] != sha256_bytes(verdict_json.encode("utf-8")):
        return False
    if reviewer["prompt_sha256"] != sha256_bytes(
        build_reviewer_prompt(candidate, wire).encode("utf-8")
    ):
        return False
    verdict = parse_review_response(verdict_json)
    return verdict.retain and not verdict.ambiguous


def _frozen_v8_role_evidence_verified(
    raw: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    state: EditState,
    tokenizer: Any,
    author_actor_id: str,
) -> bool:
    """Verify the explicitly versioned six-call packet role evidence.

    Unlike v1, this protocol uses frozen request IDs, provider session IDs,
    terminal action blocks, and a reviewer prompt materialized after the
    solver reply. All three exact prompts and completed response bodies are
    carried in the package and tied back to the captured event logs.
    """
    if set(raw) != {"schema", "candidate_id", "packet", "author", "solver", "reviewer"}:
        return False
    if raw.get("candidate_id") != candidate.get("id"):
        return False
    packet = raw.get("packet")
    if not isinstance(packet, Mapping) or set(packet) != {
        "schema",
        "case_id",
        "model_id",
        "plan_file_sha256",
        "plan_canonical_sha256",
        "artifact_manifest_sha256",
        "inputs_sha256",
        "oracle_fixtures_sha256",
        "preflight_result_sha256",
        "license_scope_audit_sha256",
        "qualification_sha256",
        "qualification_events_sha256",
        "role_event_log_hashes",
        "role_prompt_hashes",
        "role_request_ids",
        "source_parent_sha256",
        "input_row_sha256",
        "source_seed_sha256",
        "source_transform_sha256",
    }:
        return False
    if packet.get("schema") != "two-seed-v8-role-packet-binding-v1":
        return False
    case_id = packet.get("case_id")
    frozen_case = _FROZEN_V8_CASES.get(str(case_id))
    if frozen_case is None:
        return False
    if any(packet.get(key) != value for key, value in _FROZEN_V8_PACKET.items()):
        return False
    for key, value in packet.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
            return False
    for key in ("role_event_log_hashes", "role_prompt_hashes"):
        hashes = packet.get(key)
        if (
            not isinstance(hashes, Mapping)
            or set(hashes) != {"author", "solver", "reviewer"}
            or any(
                not isinstance(value, str) or not _SHA256.fullmatch(value)
                for value in hashes.values()
            )
        ):
            return False
    planned_request_ids = packet.get("role_request_ids")
    if (
        not isinstance(planned_request_ids, Mapping)
        or set(planned_request_ids) != {"author", "solver", "reviewer"}
        or any(
            not isinstance(value, str) or not _REQUEST_ID.fullmatch(value)
            for value in planned_request_ids.values()
        )
    ):
        return False
    role_prompt_hashes = packet.get("role_prompt_hashes")
    if not isinstance(role_prompt_hashes, Mapping):
        return False
    if (
        packet.get("input_row_sha256") != frozen_case["input_row_sha256"]
        or packet.get("source_parent_sha256") != frozen_case["source_parent_sha256"]
        or packet.get("source_seed_sha256") != frozen_case["source_seed_sha256"]
        or packet.get("source_transform_sha256") != frozen_case["source_transform_sha256"]
        or packet.get("role_request_ids") != frozen_case["request_ids"]
        or {key: role_prompt_hashes.get(key) for key in ("author", "solver")}
        != frozen_case["prompt_sha256"]
    ):
        return False
    model_id = packet.get("model_id")
    if not isinstance(case_id, str) or not case_id or not isinstance(model_id, str) or not model_id:
        return False
    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("frozen_v8_packet_binding") != dict(
        packet
    ):
        return False
    if (
        provenance.get("source_id") != case_id
        or provenance.get("source_sha256") != packet.get("source_parent_sha256")
        or provenance.get("source_only_input_row_sha256") != packet.get("input_row_sha256")
        or provenance.get("source_seed_sha256") != packet.get("source_seed_sha256")
        or provenance.get("source_transform_sha256") != packet.get("source_transform_sha256")
    ):
        return False
    raw_roles = tuple(raw.get(name) for name in ("author", "solver", "reviewer"))
    if not all(isinstance(role, Mapping) for role in raw_roles):
        return False
    roles = cast(tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], raw_roles)
    author, solver, reviewer = roles
    required_role_keys = {
        "actor_id",
        "session_id",
        "request_id",
        "response_id",
        "model_id",
        "prompt_text",
        "prompt_sha256",
        "response_text",
        "response_sha256",
        "finish_reason",
        "event_log",
        "response_evidence_json",
    }
    if any(
        set(role)
        - required_role_keys
        - {"wire", "generated_tokens", "terminated", "verdict_json", "verdict_sha256"}
        for role in roles
    ):
        return False
    if any(not required_role_keys <= set(role) for role in roles):
        return False
    role_names = ("author", "solver", "reviewer")
    actor_ids: list[str] = []
    session_ids: list[str] = []
    request_ids: list[str] = []
    response_ids: list[str] = []
    decoded_solver = None
    for role_name, role in zip(role_names, roles, strict=True):
        actor_id = role.get("actor_id")
        session_id = role.get("session_id")
        request_id = role.get("request_id")
        response_id = role.get("response_id")
        if (
            not isinstance(actor_id, str)
            or actor_id != f"{model_id}:{role_name}"
            or not isinstance(session_id, str)
            or not _SESSION.fullmatch(session_id)
            or not isinstance(request_id, str)
            or not _REQUEST_ID.fullmatch(request_id)
            or not isinstance(response_id, str)
            or not _RESPONSE_ID.fullmatch(response_id)
            or role.get("model_id") != model_id
            or role.get("finish_reason") != "stop"
            or request_id != planned_request_ids.get(role_name)
            or role.get("prompt_sha256") != packet["role_prompt_hashes"].get(role_name)
        ):
            return False
        actor_ids.append(actor_id)
        session_ids.append(session_id)
        request_ids.append(request_id)
        response_ids.append(response_id)
        prompt_text, response_text = role.get("prompt_text"), role.get("response_text")
        if not isinstance(prompt_text, str) or not isinstance(response_text, str):
            return False
        if (
            len(prompt_text.encode("utf-8")) > 32_000
            or len(response_text.encode("utf-8")) > 16_384
            or role.get("prompt_sha256") != sha256_bytes(prompt_text.encode("utf-8"))
            or role.get("response_sha256") != sha256_bytes(response_text.encode("utf-8"))
        ):
            return False
        event_log = role.get("event_log")
        evidence_json = role.get("response_evidence_json")
        if (
            not isinstance(event_log, str)
            or not isinstance(evidence_json, str)
            or len(event_log.encode("utf-8")) > 16_384
            or len(evidence_json.encode("utf-8")) > 4096
            or sha256_bytes(event_log.encode("utf-8")) != packet["role_event_log_hashes"][role_name]
        ):
            return False
        try:
            event_rows = [json.loads(line) for line in event_log.splitlines() if line]
            events = [
                event
                for event in event_rows
                if isinstance(event, dict) and event.get("request_id") == request_id
            ]
            response_evidence = json.loads(evidence_json, object_pairs_hook=_strict_pairs)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
        if len(events) != 1 or not isinstance(response_evidence, dict):
            return False
        event = events[0]
        expected_event_fields = {
            "role": role_name,
            "source_id": case_id,
            "request_id": request_id,
            "session_id": session_id,
            "response_id": response_id,
            "model_id": model_id,
            "prompt_sha256": role["prompt_sha256"],
            "output_sha256": role["response_sha256"],
            "response_evidence_sha256": sha256_bytes(evidence_json.encode("utf-8")),
            "finish_reason": "stop",
            "output_available": True,
            "secret_suspected": False,
        }
        if any(event.get(key) != value for key, value in expected_event_fields.items()):
            return False
        expected_response_fields = {
            "schema": "opencode-completed-response-evidence-v1",
            "request_id": request_id,
            "session_id": session_id,
            "response_id": response_id,
            "model_id": model_id,
            "response_sha256": role["response_sha256"],
            "finish_reason": "stop",
        }
        if any(
            response_evidence.get(key) != value for key, value in expected_response_fields.items()
        ):
            return False
        if (
            type(response_evidence.get("input_tokens")) is not int
            or type(response_evidence.get("output_tokens")) is not int
            or type(response_evidence.get("reasoning_tokens")) is not int
            or response_evidence["input_tokens"] < 0
            or response_evidence["output_tokens"] < 0
            or response_evidence["reasoning_tokens"] < 0
        ):
            return False
        if role_name == "solver":
            wire = role.get("wire")
            generated_tokens = role.get("generated_tokens")
            if (
                not isinstance(wire, str)
                or type(generated_tokens) is not int
                or role.get("terminated") is not True
                or wire not in response_text
                or prompt_text != build_blind_solver_prompt(candidate, tokenizer)
            ):
                return False
            decoded = decode_action(
                wire, terminated=True, generated_tokens=generated_tokens, max_tokens=64
            )
            if decoded.status != "ok" or decoded.action is None:
                return False
            if generated_tokens != len(tokenizer.encode(wire, add_special_tokens=False)) + 1:
                return False
            if decoded.action != EditAction(**candidate["action"]):
                return False
            decoded_solver = decoded
        elif role_name == "author":
            if role.get("actor_id") != author_actor_id:
                return False
            # Bind the raw author JSON to the candidate fields; the provider's
            # wrapper may contain prose, so compare only the strict block.
            try:
                from tinycomplete.one_line.teacher import extract_author_candidate_block

                block = extract_author_candidate_block(response_text, provider_complete=True)
                author_json = json.loads(block.json_text, object_pairs_hook=_strict_pairs)
            except (ImportError, ValueError, TypeError, json.JSONDecodeError):
                return False
            if sha256_bytes(block.json_text.encode("utf-8")) != candidate.get("provenance", {}).get(
                "author_response_sha256"
            ):
                return False
            state_raw = candidate["state"]
            history = state_raw.get("history")
            if not isinstance(history, (list, tuple)) or len(history) != 1:
                return False
            prior = author_json.get("prior_edit")
            if (
                not isinstance(prior, dict)
                or prior != asdict(state.history[0])
                or author_json.get("target_row") != state.target_row
                or author_json.get("intent_evidence")
                != candidate.get("provenance", {}).get("intent_evidence")
                or author_json.get("objective") != candidate.get("provenance", {}).get("objective")
            ):
                return False
            action_raw = author_json.get("action")
            if not isinstance(action_raw, dict):
                return False
            raw_action_kind = action_raw.get("kind")
            if not isinstance(raw_action_kind, str):
                return False
            action_kind = {
                "N": "keep",
                "D": "delete_line",
                "R": "replace_line",
                "I": "insert_before",
            }.get(raw_action_kind)
            if action_kind != candidate["action"].get("kind"):
                return False
            if action_kind in {"replace_line", "insert_before"}:
                if action_raw.get("text") != candidate["action"].get("text"):
                    return False
            elif action_raw.get("text") not in (None, ""):
                return False
        else:
            verdict_json = role.get("verdict_json")
            if (
                not isinstance(verdict_json, str)
                or response_text != verdict_json
                or role.get("verdict_sha256") != sha256_bytes(verdict_json.encode("utf-8"))
            ):
                return False
            try:
                prompt_body = role["prompt_text"].split("\n", 1)[1]
                review_input = json.loads(prompt_body, object_pairs_hook=_strict_pairs)
            except (IndexError, json.JSONDecodeError, TypeError, ValueError):
                return False
            action = EditAction(**candidate["action"])
            author_wire = encode_action(action)
            if (
                review_input.get("file_id") != state.file_id
                or review_input.get("filetype") != state.filetype
                or review_input.get("source") != state.source
                or review_input.get("target_row") != state.target_row
                or review_input.get("cursor_col") != state.cursor_col
                or review_input.get("history") != [asdict(edit) for edit in state.history]
                or review_input.get("author_action_wire") != author_wire
                or review_input.get("blind_solver_wire") != solver["wire"]
                or review_input.get("objective") != candidate.get("provenance", {}).get("objective")
            ):
                return False
            # The frozen v8 reviewer prompt is the legacy reviewed-state JSON
            # with an explicit strict-output suffix; verify that exact builder.
            expected_prompt = build_reviewer_prompt(candidate, str(solver["wire"])).replace(
                "Return exactly the requested JSON schema.",
                "Return exactly one raw JSON object with exactly these keys: retain (boolean), "
                "ambiguous (boolean), reason (string, 1 to 500 characters). Do not use a code "
                "fence, markdown, extra fields, tools, or text outside the JSON object.",
            )
            if role["prompt_text"] != expected_prompt:
                return False
            verdict = parse_review_response(verdict_json)
            if not verdict.retain or verdict.ambiguous:
                return False
    return (
        len(set(actor_ids)) == 3
        and len(set(session_ids)) == 3
        and len(set(request_ids)) == 3
        and len(set(response_ids)) == 3
        and decoded_solver is not None
        and actor_ids[0] == author_actor_id
        and decoded_solver.action == EditAction(**candidate["action"])
        and apply_action(state, decoded_solver.action) == candidate["after_source"]
    )


def _text_count_check(source: str, record: Mapping[str, Any]) -> bool:
    if set(record) - {
        "candidate_id",
        "source_id",
        "kind",
        "counts",
    } or not {"candidate_id", "source_id", "kind", "counts"} <= set(record):
        raise ValueError("text objective fields are invalid")
    counts = record["counts"]
    if not isinstance(counts, list) or not 1 <= len(counts) <= 8:
        raise ValueError("text objective has no bounded assertions")
    for assertion in counts:
        if not isinstance(assertion, dict) or set(assertion) != {"text", "equals"}:
            raise ValueError("text assertion shape is invalid")
        fragment, expected = assertion["text"], assertion["equals"]
        if not isinstance(fragment, str) or not 1 <= len(fragment) <= 256:
            raise ValueError("text assertion fragment is invalid")
        if type(expected) is not int or expected < 0 or expected > 1000:
            raise ValueError("text assertion count is invalid")
        if source.count(fragment) != expected:
            return False
    return True


def _sandbox_check(source: str, record: Mapping[str, Any], filetype: str) -> bool | None:
    """Run a reviewed fixture in the existing no-network OCI evaluator.

    `None` means the checker was unavailable or timed out, never a pass.
    """
    if set(record) - {
        "candidate_id",
        "source_id",
        "kind",
        "path",
        "check",
    } or not {"candidate_id", "source_id", "kind", "path", "check"} <= set(record):
        raise ValueError("sandbox objective fields are invalid")
    check = CheckSpec.model_validate(record["check"])
    if not check.container_image or not check.test or check.timeout_seconds > 10:
        raise ValueError("sandbox objective needs a pinned image and short test")
    if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", check.container_image):
        raise ValueError("sandbox objective image is not pinned by digest")
    case = BenchmarkCase(
        id=str(record["candidate_id"]),
        language=cast(Any, filetype),
        path=str(record["path"]),
        prefix="",
        expected="fixture-only",
        check=check,
    )
    with tempfile.TemporaryDirectory(prefix="tabcomplete-objective-") as directory:
        result = evaluate_prediction(
            case,
            Prediction(case_id=case.id, completion=source),
            work_root=Path(directory),
            execution_backend="container",
        )
    statuses = (result.parse.status, result.compile.status, result.test.status)
    if "fail" in statuses:
        return False
    if result.parse.status == "pass" and result.test.status == "pass":
        return True
    return None


def _check_source(source: str, record: Mapping[str, Any], filetype: str) -> bool | None:
    kind = record.get("kind")
    if kind == "text_counts":
        return _text_count_check(source, record)
    if kind == "sandbox_test":
        return _sandbox_check(source, record, filetype)
    raise ValueError("unsupported independent objective kind")


def _wrong_edits(state: EditState, gold_after: str) -> list[str]:
    actions = [
        EditAction("replace_line", _WRONG_TEXTS[0]),
        EditAction("delete_line"),
        EditAction("insert_before", _WRONG_TEXTS[1]),
    ]
    results: list[str] = []
    for action in actions:
        try:
            after = apply_action(state, action)
        except ValueError:
            continue
        if after not in {state.source, gold_after} and after not in results:
            results.append(after)
    return results[:2]


def check_candidate_acceptance(
    candidate: Mapping[str, Any],
    source_row: Mapping[str, Any],
    tokenizer: Any,
    *,
    objective_manifest: bytes | None,
    expected_objective_manifest_sha256: str | None,
    split_manifest: bytes | None,
    expected_split_manifest_sha256: str | None,
    role_evidence: bytes | None = None,
    expected_role_evidence_sha256: str | None = None,
    file_license_review: bytes | None = None,
    expected_file_license_review_sha256: str | None = None,
    existing_rows: Sequence[Mapping[str, Any]] = (),
    reserved_repositories: frozenset[str] = frozenset(),
) -> AcceptanceDecision:
    """Accept only when provenance, split, tokens, and independent controls pass.

    The caller supplies the frozen manifest hashes from an approved plan, not
    from candidate or teacher output. No check strings from the author execute.
    """
    if objective_manifest is None or expected_objective_manifest_sha256 is None:
        return _reject("independent_objective_absent")
    if split_manifest is None or expected_split_manifest_sha256 is None:
        return _reject("split_manifest_unpinned")
    try:
        metadata = source_row["authoring_metadata"]
        seed = source_row["student_state_seed"]
        provenance = candidate["provenance"]
        validation = candidate["validation"]
        state = EditState.from_mapping(candidate["state"])
        action = EditAction(**candidate["action"])
        source_id = str(source_row["id"])
        author_actor_id = provenance.get("author_actor_id")
        if not isinstance(author_actor_id, str) or not author_actor_id:
            return _reject("author_identity_absent")
        if (
            provenance["source_id"] != source_id
            or candidate["source_type"] != "muse_author_public_candidate"
        ):
            return _reject("source_identity_mismatch")
        if (
            candidate["source_repo"] != metadata["source_repo"]
            or candidate["source_revision"] != metadata["source_revision"]
        ):
            return _reject("source_identity_mismatch")
        if not _REVISION.fullmatch(str(metadata["source_revision"])):
            return _reject("source_revision_unpinned")
        if metadata["source_license"] not in _LICENSES or not metadata.get(
            "source_provenance_verified"
        ):
            return _reject("source_license_unverified")
        if not _SHA256.fullmatch(str(metadata["license_sha256"])):
            return _reject("source_license_unverified")
        if not _source_seed_verified(source_row, metadata):
            return _reject("source_hash_mismatch")
        if (
            provenance["source_sha256"] != metadata["source_sha256"]
            or provenance["source_license_sha256"] != metadata["license_sha256"]
        ):
            return _reject("source_hash_mismatch")
        if "synthetic_seed_sha256" in metadata and (
            provenance.get("source_seed_sha256") != metadata["synthetic_seed_sha256"]
            or provenance.get("source_transform_sha256")
            != metadata.get("synthetic_source_transform_sha256")
            or provenance.get("source_transform") != source_row.get("synthetic_source_transform")
        ):
            return _reject("synthetic_source_transform_unbound")
        if state.file_id != seed["file_id"] or state.filetype != seed["filetype"]:
            return _reject("state_identity_mismatch")
        if (
            replay_replacement_history(
                str(seed["source"]),
                state.history,
                file_id=state.file_id,
                filetype=state.filetype,
            )
            != state.source
        ):
            return _reject("history_replay_failed")
        if action.kind == "keep":
            return _reject("no_edit_has_no_observed_decision")
        if apply_action(state, action) != candidate["after_source"]:
            return _reject("target_replay_failed")
        if candidate["after_source"] == state.source:
            return _reject("target_is_unchanged")
        if validation.get("source_before_sha256") != sha256_bytes(
            state.source.encode()
        ) or validation.get("source_after_sha256") != sha256_bytes(
            str(candidate["after_source"]).encode()
        ):
            return _reject("candidate_state_hash_mismatch")
        if (
            validation.get("history_replays_to_state") is not True
            or validation.get("history_visible_in_prompt") is not True
        ):
            return _reject("history_not_verified")
        if (
            candidate.get("split") not in {"train", "development"}
            or candidate.get("split_manifest_sha256") != expected_split_manifest_sha256
        ):
            return _reject("split_unassigned_or_mismatched")
        assignment = _split_record(
            split_manifest,
            expected_split_manifest_sha256,
            str(candidate["id"]),
            str(candidate["source_repo"]),
        )
        if assignment.get("split") != candidate["split"] or set(assignment) != {
            "candidate_id",
            "source_repo",
            "split",
        }:
            return _reject("split_unassigned_or_mismatched")
        aliases = {str(candidate["source_repo"]).casefold()}
        aliases.update(str(value).casefold() for value in candidate.get("source_aliases", ()))
        if aliases & {value.casefold() for value in reserved_repositories}:
            return _reject("reserved_repository_overlap")
        validate_splits([*existing_rows, candidate])
        context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
        if context.input_tokens is None or context.included_history != len(state.history):
            return _reject("bounded_context_omits_history")
        response_tokens = len(tokenizer.encode(encode_action(action), add_special_tokens=False)) + 1
        if response_tokens > 64 or context.input_tokens + response_tokens > 2048:
            return _reject("token_budget_exceeded")
        if (
            validation.get("input_tokens") != context.input_tokens
            or validation.get("response_tokens_including_eos") != response_tokens
        ):
            return _reject("token_counts_changed")
        if not _SHA256.fullmatch(str(provenance.get("author_response_sha256", ""))):
            return _reject("author_response_unpinned")
        record = _objective_record(
            objective_manifest,
            expected_objective_manifest_sha256,
            str(candidate["id"]),
            source_id,
            author_actor_id,
        )
        if metadata.get("file_license_scope_unverified_without_notice"):
            if file_license_review is None or expected_file_license_review_sha256 is None:
                return _reject("file_license_scope_unreviewed")
            if not _license_review_verified(
                file_license_review,
                expected_file_license_review_sha256,
                source_row=source_row,
                author_actor_id=author_actor_id,
            ):
                return _reject("file_license_scope_unreviewed")
        gold = _check_source(str(candidate["after_source"]), record, state.filetype)
        unchanged = _check_source(state.source, record, state.filetype)
        controls = _wrong_edits(state, str(candidate["after_source"]))
        if len(controls) < 2:
            return _reject("wrong_edit_controls_unavailable")
        wrong_results = [_check_source(control, record, state.filetype) for control in controls]
        evidence = {
            "objective_manifest_sha256": expected_objective_manifest_sha256,
            "objective_kind": record["kind"],
            "gold": gold,
            "unchanged": unchanged,
            "wrong_edits": wrong_results,
            "input_tokens": context.input_tokens,
            "response_tokens_including_eos": response_tokens,
        }
        if (
            gold is not True
            or unchanged is not False
            or any(value is not False for value in wrong_results)
        ):
            return _reject("independent_objective_unproven", **evidence)
        if (
            validation.get("blind_solver_verified") is not True
            or validation.get("reviewer_verified") is not True
        ):
            return _reject("independent_roles_unverified", **evidence)
        if role_evidence is None or expected_role_evidence_sha256 is None:
            return _reject("independent_role_evidence_absent", **evidence)
        if not _role_evidence_verified(
            role_evidence,
            expected_role_evidence_sha256,
            candidate=candidate,
            state=state,
            tokenizer=tokenizer,
            author_actor_id=author_actor_id,
        ):
            return _reject("independent_role_evidence_invalid", **evidence)
        evidence["role_evidence_sha256"] = expected_role_evidence_sha256
        if metadata.get("file_license_scope_unverified_without_notice"):
            evidence["file_license_review_sha256"] = expected_file_license_review_sha256
        return AcceptanceDecision(True, "objective_and_controls_passed", evidence)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        # Do not print candidate data or exception text; either may include code.
        return _reject("invalid_candidate_or_fixture", error_type=type(error).__name__)
