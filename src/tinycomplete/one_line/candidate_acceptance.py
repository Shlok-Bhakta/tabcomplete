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
from dataclasses import dataclass
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
    apply_action,
    decode_action,
    encode_action,
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
        if sha256_bytes(str(seed["source"]).encode("utf-8")) != metadata["source_sha256"]:
            return _reject("source_hash_mismatch")
        if (
            provenance["source_sha256"] != metadata["source_sha256"]
            or provenance["source_license_sha256"] != metadata["license_sha256"]
        ):
            return _reject("source_hash_mismatch")
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
