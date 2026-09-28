"""CPU-only acceptance gates with synthetic public-source fixtures."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tinycomplete.one_line.candidate_acceptance import check_candidate_acceptance
from tinycomplete.one_line.contract import EditAction, encode_action
from tinycomplete.one_line.data import parse_author_response
from tinycomplete.one_line.pilot_roles import build_blind_solver_prompt, build_reviewer_prompt


class _ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode()) + ([0] if add_special_tokens else [])


def _source_row() -> dict[str, Any]:
    source = "def compute(value):\n    total = value + 1\n    return total\n"
    return {
        "id": "public-source/fixed",
        "student_state_seed": {"file_id": "file_fixed.py", "filetype": "python", "source": source},
        "authoring_metadata": {
            "source_repo": "public/example",
            "source_aliases": ["public/example"],
            "source_revision": "a" * 40,
            "source_path": "compute.py",
            "source_license": "MIT",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "license_sha256": "b" * 64,
            "authoring_focus": "call site consistency",
            "source_provenance_verified": True,
            "file_license_scope_unverified_without_notice": True,
        },
    }


def _candidate(source: dict[str, Any], *, kind: str = "R") -> dict[str, Any]:
    response = {
        "prior_edit": {
            "row": 1,
            "old_text": "    total = value + 1",
            "new_text": "    result = value + 1",
        },
        "target_row": 2,
        "action": {"kind": kind, "text": "    return result" if kind == "R" else None},
        "intent_evidence": "The return should use the value named by the prior edit.",
        "objective": {"kind": "author_claim", "description": "Return uses result."},
    }
    candidate = parse_author_response(json.dumps(response), source, _ByteTokenizer())
    candidate["provenance"]["author_actor_id"] = "author-model"
    candidate["validation"]["blind_solver_verified"] = True
    candidate["validation"]["reviewer_verified"] = True
    candidate["split"] = "train"
    candidate["split_manifest_sha256"] = hashlib.sha256(_split_manifest(candidate)).hexdigest()
    return candidate


def _split_manifest(candidate: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "schema": "one-line-accepted-split-v1",
            "status": "frozen",
            "assignments": [
                {
                    "candidate_id": candidate["id"],
                    "source_repo": candidate["source_repo"],
                    "split": candidate["split"],
                }
            ],
        },
        sort_keys=True,
    ).encode()


def _manifest(
    candidate: dict[str, Any],
    *,
    counts: list[dict[str, Any]] | None = None,
    reviewer: str = "independent-reviewer",
) -> bytes:
    entry: dict[str, Any] = {
        "candidate_id": candidate["id"],
        "source_id": candidate["provenance"]["source_id"],
        "kind": "text_counts",
        "counts": counts
        if counts is not None
        else [
            {"text": "return result", "equals": 1},
            {"text": "return total", "equals": 0},
        ],
    }
    return json.dumps(
        {
            "schema": "one-line-independent-objective-v1",
            "reviewer_id": reviewer,
            "entries": [entry],
        },
        sort_keys=True,
    ).encode()


def _license_review(source: dict[str, Any]) -> bytes:
    metadata = source["authoring_metadata"]
    return json.dumps(
        {
            "schema": "one-line-file-license-review-v1",
            "source_id": source["id"],
            "source_repo": metadata["source_repo"],
            "source_revision": metadata["source_revision"],
            "source_sha256": metadata["source_sha256"],
            "source_path": metadata["source_path"],
            "license_spdx": metadata["source_license"],
            "license_sha256": metadata["license_sha256"],
            "reviewer_id": "license-reviewer",
            "finding": "license_covers_file",
            "evidence": "Synthetic unit fixture with source and repository license checked.",
        },
        sort_keys=True,
    ).encode()


def _role_evidence(candidate: dict[str, Any]) -> bytes:
    tokenizer = _ByteTokenizer()
    wire = encode_action(EditAction(**candidate["action"]))
    verdict_json = json.dumps({"retain": True, "ambiguous": False, "reason": "Synthetic review."})
    return json.dumps(
        {
            "schema": "one-line-role-evidence-v1",
            "candidate_id": candidate["id"],
            "author": {
                "actor_id": "author-model",
                "session_id": "session-author-001",
                "response_sha256": candidate["provenance"]["author_response_sha256"],
            },
            "solver": {
                "actor_id": "blind-solver-model",
                "session_id": "session-solver-002",
                "prompt_sha256": hashlib.sha256(
                    build_blind_solver_prompt(candidate, tokenizer).encode()
                ).hexdigest(),
                "wire": wire,
                "wire_sha256": hashlib.sha256(wire.encode()).hexdigest(),
                "terminated": True,
                "generated_tokens": len(tokenizer.encode(wire, add_special_tokens=False)) + 1,
            },
            "reviewer": {
                "actor_id": "reviewer-model",
                "session_id": "session-reviewer-003",
                "prompt_sha256": hashlib.sha256(
                    build_reviewer_prompt(candidate, wire).encode()
                ).hexdigest(),
                "verdict_json": verdict_json,
                "verdict_sha256": hashlib.sha256(verdict_json.encode()).hexdigest(),
            },
        },
        sort_keys=True,
    ).encode()


def _check(
    candidate: dict[str, Any],
    source: dict[str, Any],
    manifest: bytes | None,
    split_manifest: bytes | None = None,
    **kwargs: Any,
):
    if split_manifest is None:
        split_manifest = _split_manifest(candidate)
    role_evidence = kwargs.pop("role_evidence", _role_evidence(candidate))
    file_license_review = kwargs.pop("file_license_review", _license_review(source))
    return check_candidate_acceptance(
        candidate,
        source,
        _ByteTokenizer(),
        objective_manifest=manifest,
        expected_objective_manifest_sha256=(
            hashlib.sha256(manifest).hexdigest() if manifest is not None else None
        ),
        split_manifest=split_manifest,
        expected_split_manifest_sha256=hashlib.sha256(split_manifest).hexdigest(),
        role_evidence=role_evidence,
        expected_role_evidence_sha256=(
            hashlib.sha256(role_evidence).hexdigest() if role_evidence is not None else None
        ),
        file_license_review=file_license_review,
        expected_file_license_review_sha256=(
            hashlib.sha256(file_license_review).hexdigest()
            if file_license_review is not None
            else None
        ),
        **kwargs,
    )


def test_independent_objective_and_controls_can_accept() -> None:
    source = _source_row()
    candidate = _candidate(source)
    decision = _check(candidate, source, _manifest(candidate))
    assert decision.accepted is True
    assert decision.evidence["gold"] is True
    assert decision.evidence["unchanged"] is False
    assert decision.evidence["wrong_edits"] == [False, False]
    assert candidate["validation"]["accepted_training"] is False


def test_author_objective_alone_never_accepts() -> None:
    source = _source_row()
    candidate = _candidate(source)
    assert _check(candidate, source, None).reason == "independent_objective_absent"


def test_wrong_edit_and_unchanged_controls_reject_weak_objectives() -> None:
    source = _source_row()
    candidate = _candidate(source)
    weak = _manifest(candidate, counts=[{"text": "return total", "equals": 0}])
    decision = _check(candidate, source, weak)
    assert decision.accepted is False
    assert decision.reason == "independent_objective_unproven"
    assert decision.evidence["gold"] is True
    assert decision.evidence["unchanged"] is False
    assert True in decision.evidence["wrong_edits"]


def test_license_and_split_evidence_are_required() -> None:
    source = _source_row()
    candidate = _candidate(source)
    assert _check(candidate, source, _manifest(candidate), file_license_review=None).reason == (
        "file_license_scope_unreviewed"
    )
    candidate["split_manifest_sha256"] = "e" * 64
    assert (
        _check(candidate, source, _manifest(candidate)).reason == "split_unassigned_or_mismatched"
    )


def test_reserved_group_and_token_drift_are_rejected() -> None:
    source = _source_row()
    candidate = _candidate(source)
    manifest = _manifest(candidate)
    assert (
        _check(
            candidate, source, manifest, reserved_repositories=frozenset({"PUBLIC/EXAMPLE"})
        ).reason
        == "reserved_repository_overlap"
    )
    candidate["validation"]["response_tokens_including_eos"] += 1
    assert _check(candidate, source, manifest).reason == "token_counts_changed"


def test_unobserved_keep_is_not_promoted() -> None:
    source = _source_row()
    candidate = _candidate(source, kind="N")
    assert _check(candidate, source, _manifest(candidate)).reason == (
        "no_edit_has_no_observed_decision"
    )


def test_author_cannot_pose_as_independent_reviewer() -> None:
    source = _source_row()
    candidate = _candidate(source)
    decision = _check(candidate, source, _manifest(candidate, reviewer="author-model"))
    assert decision.accepted is False
    assert decision.reason == "invalid_candidate_or_fixture"


def test_tampered_gold_fails_reconstruction() -> None:
    source = _source_row()
    candidate = _candidate(source)
    manifest = _manifest(candidate)
    candidate["after_source"] += "# altered\n"
    assert _check(candidate, source, manifest).reason == "target_replay_failed"


def test_forged_boolean_flags_do_not_replace_role_evidence() -> None:
    source = _source_row()
    candidate = _candidate(source)
    assert candidate["validation"]["blind_solver_verified"] is True
    assert candidate["validation"]["reviewer_verified"] is True
    assert _check(candidate, source, _manifest(candidate), role_evidence=None).reason == (
        "independent_role_evidence_absent"
    )
    candidate["validation"]["reviewer_verified"] = False
    assert _check(candidate, source, _manifest(candidate)).reason == "independent_roles_unverified"


def test_forged_role_hashes_and_actor_identity_fail() -> None:
    source = _source_row()
    candidate = _candidate(source)
    raw = json.loads(_role_evidence(candidate))
    raw["solver"]["wire_sha256"] = "f" * 64
    assert _check(
        candidate, source, _manifest(candidate), role_evidence=json.dumps(raw).encode()
    ).reason == ("independent_role_evidence_invalid")
    raw = json.loads(_role_evidence(candidate))
    raw["reviewer"]["actor_id"] = raw["author"]["actor_id"]
    assert _check(
        candidate, source, _manifest(candidate), role_evidence=json.dumps(raw).encode()
    ).reason == ("independent_role_evidence_invalid")


def test_license_review_must_bind_exact_source_and_license() -> None:
    source = _source_row()
    candidate = _candidate(source)
    raw = json.loads(_license_review(source))
    raw["source_sha256"] = "0" * 64
    assert _check(
        candidate, source, _manifest(candidate), file_license_review=json.dumps(raw).encode()
    ).reason == ("file_license_scope_unreviewed")
