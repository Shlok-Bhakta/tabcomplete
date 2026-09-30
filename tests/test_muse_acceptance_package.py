from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tinycomplete.one_line import candidate_acceptance
from tinycomplete.one_line.contract import EditAction, encode_action
from tinycomplete.one_line.data import parse_author_response
from tinycomplete.one_line.muse_acceptance_package import (
    _digest,
    _solver_action,
    export_muse_acceptance_package,
    objective_sandbox_path,
    verify_muse_candidate_row,
    verify_role_evidence,
)
from tinycomplete.one_line.pilot_roles import build_blind_solver_prompt, build_reviewer_prompt


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8")) + ([0] if add_special_tokens else [])


def test_objective_uses_frozen_synthetic_sandbox_paths() -> None:
    assert objective_sandbox_path("python") == "solution.py"
    assert objective_sandbox_path("typescript") == "tooltip-view.ts"
    with pytest.raises(ValueError, match="unsupported fixture language"):
        objective_sandbox_path("rust")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_row() -> dict[str, Any]:
    source = "def compute(value):\n    total = value + 1\n    return total\n"
    return {
        "id": "public-source/fixed",
        "student_state_seed": {
            "file_id": "file_fixed.py",
            "filetype": "python",
            "source": source,
        },
        "authoring_metadata": {
            "source_repo": "public/example",
            "source_aliases": ["public/example"],
            "source_revision": "a" * 40,
            "source_path": "compute.py",
            "source_license": "MIT",
            "source_sha256": _sha(source.encode()),
            "license_sha256": "b" * 64,
            "authoring_focus": "call site consistency",
            "source_provenance_verified": True,
            "file_license_scope_unverified_without_notice": True,
        },
    }


def _candidate(source: dict[str, Any]) -> dict[str, Any]:
    response = {
        "prior_edit": {
            "row": 1,
            "old_text": "    total = value + 1",
            "new_text": "    result = value + 1",
        },
        "target_row": 2,
        "action": {"kind": "R", "text": "    return result"},
        "intent_evidence": "Return the value introduced by the visible rename.",
        "objective": {
            "kind": "source_consistency",
            "description": "Return the new variable.",
            "checks": ["The old return expression is absent."],
        },
    }
    candidate = parse_author_response(json.dumps(response), source, ByteTokenizer())
    candidate["provenance"]["author_actor_id"] = "author-model"
    candidate["validation"]["blind_solver_verified"] = True
    candidate["validation"]["reviewer_verified"] = True
    candidate["split"] = "train"
    candidate["source_group_id"] = "public/example"
    candidate["task_family_id"] = candidate["generator_family"]
    return candidate


def _split(candidate: dict[str, Any]) -> bytes:
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
        separators=(",", ":"),
    ).encode()


def _objective(candidate: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "schema": "one-line-independent-objective-v1",
            "reviewer_id": "objective-reviewer",
            "entries": [
                {
                    "candidate_id": candidate["id"],
                    "source_id": candidate["provenance"]["source_id"],
                    "kind": "text_counts",
                    "counts": [
                        {"text": "return result", "equals": 1},
                        {"text": "return total", "equals": 0},
                    ],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
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
            "evidence": "The pinned public file and repository license were checked.",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _role(candidate: dict[str, Any]) -> bytes:
    tokenizer = ByteTokenizer()
    wire = encode_action(EditAction(**candidate["action"]))
    verdict = json.dumps(
        {"retain": True, "ambiguous": False, "reason": "Visible rename supports the edit."},
        sort_keys=True,
        separators=(",", ":"),
    )
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
                "prompt_sha256": _sha(build_blind_solver_prompt(candidate, tokenizer).encode()),
                "wire": wire,
                "wire_sha256": _sha(wire.encode()),
                "terminated": True,
                "generated_tokens": len(tokenizer.encode(wire, add_special_tokens=False)) + 1,
            },
            "reviewer": {
                "actor_id": "reviewer-model",
                "session_id": "session-reviewer-003",
                "prompt_sha256": _sha(build_reviewer_prompt(candidate, wire).encode()),
                "verdict_json": verdict,
                "verdict_sha256": _sha(verdict.encode()),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _training_row(candidate: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    metadata = source["authoring_metadata"]
    return {
        **candidate,
        "source_path": metadata["source_path"],
        "source_license_sha256": metadata["license_sha256"],
        "source_sha256": metadata["source_sha256"],
    }


def _export(tmp_path: Path):
    source = _source_row()
    candidate = _candidate(source)
    objective = _objective(candidate)
    split = _split(candidate)
    candidate["split_manifest_sha256"] = _sha(split)
    role = _role(candidate)
    license_review = _license_review(source)
    result = export_muse_acceptance_package(
        package_root=tmp_path / "packages",
        candidate=candidate,
        source_row=source,
        tokenizer=ByteTokenizer(),
        objective_manifest=objective,
        split_manifest=split,
        role_evidence=role,
        file_license_review=license_review,
    )
    assert result.accepted is True
    assert result.acceptance_package_ref is not None
    row = _training_row(candidate, source)
    row["acceptance_package_ref"] = result.acceptance_package_ref
    return row, result, tmp_path / "packages"


def test_export_and_portable_verification_bind_functional_solver(tmp_path: Path) -> None:
    row, result, package_root = _export(tmp_path)
    decision = verify_muse_candidate_row(row, package_root=package_root, tokenizer=ByteTokenizer())
    assert decision.accepted is True
    assert decision.evidence["solver_functional_status"] == "pass"
    assert (
        decision.evidence["package_manifest_sha256"]
        == result.acceptance_package_ref["manifest_sha256"]
    )
    assert row["validation"]["accepted_training"] is False
    package = package_root / result.acceptance_package_ref["root"]
    assert os.stat(package).st_mode & 0o077 == 0


def test_training_verifier_is_portable_and_does_not_rerun_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, _result, package_root = _export(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("portable reader must not rerun sandbox qualification")

    import tinycomplete.one_line.muse_acceptance_package as package_module

    monkeypatch.setattr(package_module, "check_candidate_acceptance", forbidden)
    monkeypatch.setattr(package_module, "_check_source", forbidden)
    assert verify_muse_candidate_row(
        row, package_root=package_root, tokenizer=ByteTokenizer()
    ).accepted


def test_manifest_corruption_and_row_drift_reject(tmp_path: Path) -> None:
    row, result, package_root = _export(tmp_path)
    package = package_root / result.acceptance_package_ref["root"]
    with (package / "solver_functional.json").open("ab") as stream:
        stream.write(b" ")
    assert not verify_muse_candidate_row(
        row, package_root=package_root, tokenizer=ByteTokenizer()
    ).accepted

    row2, _result2, package_root2 = _export(tmp_path / "second")
    row2["action"] = {"kind": "replace_line", "text": "    return value"}
    assert not verify_muse_candidate_row(
        row2, package_root=package_root2, tokenizer=ByteTokenizer()
    ).accepted


def test_rehashed_package_cannot_claim_failed_solver_function_pass(tmp_path: Path) -> None:
    row, result, package_root = _export(tmp_path)
    package = package_root / result.acceptance_package_ref["root"]
    functional = json.loads((package / "solver_functional.json").read_bytes())
    functional["functional_status"] = "fail"
    functional_bytes = json.dumps(
        functional, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    (package / "solver_functional.json").write_bytes(functional_bytes)
    acceptance = json.loads((package / "acceptance_result.json").read_bytes())
    acceptance["solver_functional_sha256"] = _sha(functional_bytes)
    acceptance_bytes = json.dumps(
        acceptance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    (package / "acceptance_result.json").write_bytes(acceptance_bytes)
    manifest = json.loads((package / "manifest.json").read_bytes())
    for filename, payload in (
        ("solver_functional.json", functional_bytes),
        ("acceptance_result.json", acceptance_bytes),
    ):
        manifest["files"][filename] = {"sha256": _sha(payload), "bytes": len(payload)}
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    (package / "manifest.json").write_bytes(manifest_bytes)
    row["acceptance_package_ref"]["manifest_sha256"] = _sha(manifest_bytes)
    decision = verify_muse_candidate_row(row, package_root=package_root, tokenizer=ByteTokenizer())
    assert not decision.accepted


def test_cpu_export_rejects_solver_functional_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tinycomplete.one_line.muse_acceptance_package as package_module

    source = _source_row()
    candidate = _candidate(source)
    split = _split(candidate)
    candidate["split_manifest_sha256"] = _sha(split)
    monkeypatch.setattr(package_module, "_check_source", lambda *_args: False)
    result = package_module.export_muse_acceptance_package(
        package_root=tmp_path / "packages",
        candidate=candidate,
        source_row=source,
        tokenizer=ByteTokenizer(),
        objective_manifest=_objective(candidate),
        split_manifest=split,
        role_evidence=_role(candidate),
        file_license_review=_license_review(source),
    )
    assert result.accepted is False
    assert result.reason == "blind_solver_functional_objective_failed"


def test_wrong_solver_action_is_rejected_even_if_final_state_would_match(
    tmp_path: Path,
) -> None:
    source = _source_row()
    candidate = _candidate(source)
    role = json.loads(_role(candidate))
    role["solver"]["wire"] = encode_action(EditAction("replace_line", "    return value"))
    role["solver"]["wire_sha256"] = _sha(role["solver"]["wire"].encode())
    role["solver"]["generated_tokens"] = (
        len(ByteTokenizer().encode(role["solver"]["wire"], add_special_tokens=False)) + 1
    )
    tampered = json.dumps(role, sort_keys=True, separators=(",", ":")).encode()
    decision = verify_role_evidence(candidate, tampered, tokenizer=ByteTokenizer())
    assert decision.accepted is False
    assert decision.reason == "role_solver_action_mismatch"


def test_package_reference_rejects_parent_traversal(tmp_path: Path) -> None:
    row, _result, package_root = _export(tmp_path)
    row["acceptance_package_ref"]["root"] = "../escape"
    decision = verify_muse_candidate_row(row, package_root=package_root, tokenizer=ByteTokenizer())
    assert not decision.accepted


def test_explicit_synthetic_source_transform_reconstructs_exact_bytes() -> None:
    parent = "value = 1\r\nreturn value\n"
    seed = "value = 2\r\nreturn value\n"
    transform = {
        "schema": "one-line-public-source-transform-v2",
        "kind": "replace_explicit_physical_lines_for_synthetic_public_source",
        "parent_source_sha256": _sha(parent.encode()),
        "seed_source_sha256": _sha(seed.encode()),
        "edits": [
            {
                "row": 0,
                "old_line": "value = 1",
                "new_line": "value = 2",
                "old_line_sha256": _sha(b"value = 1"),
                "new_line_sha256": _sha(b"value = 2"),
                "classification": "synthetic_prestate_signature",
            }
        ],
        "signature_prestate_row": 0,
        "privacy_redaction_rows": [],
        "plan_file_sha256": "a" * 64,
        "input_bundle_sha256": "b" * 64,
        "upstream_declared_transform": {
            "kind": "synthetic_single_line_prestate_from_public_source",
            "public_snapshot_line_sha256": _sha(b"value = 1"),
            "purpose": "make the declared synthetic history replay to pinned source bytes",
            "row": 0,
            "synthetic_author_line_sha256": _sha(b"value = 2"),
        },
    }
    source = {
        "student_state_seed": {"source": seed},
        "public_source_parent": parent,
        "synthetic_source_transform": transform,
    }
    metadata = {
        "source_sha256": _sha(parent.encode()),
        "synthetic_seed_sha256": _sha(seed.encode()),
        "synthetic_source_transform_sha256": _digest(transform),
    }
    assert candidate_acceptance._source_seed_verified(source, metadata)

    tampered = dict(transform, new_line="value = 3")
    assert not candidate_acceptance._source_seed_verified(
        dict(source, synthetic_source_transform=tampered), metadata
    )
    assert not candidate_acceptance._source_seed_verified(
        source, dict(metadata, synthetic_source_transform_sha256="c" * 64)
    )

    mismatched_declaration = dict(
        transform,
        upstream_declared_transform=dict(
            transform["upstream_declared_transform"],
            public_snapshot_line_sha256="c" * 64,
        ),
    )
    assert not candidate_acceptance._source_seed_verified(
        dict(source, synthetic_source_transform=mismatched_declaration),
        dict(metadata, synthetic_source_transform_sha256=_digest(mismatched_declaration)),
    )

    parent_with_extra_signature = "value = 1\r\nreturn value\n"
    seed_with_extra_signature = "value = 2\r\nreturn other\n"
    extra_signature = dict(transform)
    extra_signature["parent_source_sha256"] = _sha(parent_with_extra_signature.encode())
    extra_signature["seed_source_sha256"] = _sha(seed_with_extra_signature.encode())
    extra_signature["edits"] = [
        *transform["edits"],
        {
            "row": 1,
            "old_line": "return value",
            "new_line": "return other",
            "old_line_sha256": _sha(b"return value"),
            "new_line_sha256": _sha(b"return other"),
            "classification": "synthetic_prestate_signature",
        },
    ]
    assert not candidate_acceptance._source_seed_verified(
        {
            "student_state_seed": {"source": seed_with_extra_signature},
            "public_source_parent": parent_with_extra_signature,
            "synthetic_source_transform": extra_signature,
        },
        {
            "source_sha256": _sha(parent_with_extra_signature.encode()),
            "synthetic_seed_sha256": _sha(seed_with_extra_signature.encode()),
            "synthetic_source_transform_sha256": _digest(extra_signature),
        },
    )


def test_exact_public_source_with_explicit_null_transform_is_bound() -> None:
    text = "def f():\n    return 1\n"
    source = {
        "student_state_seed": {"source": text},
        "public_source_parent": text,
    }
    metadata = {
        "source_sha256": _sha(text.encode()),
        "synthetic_seed_sha256": _sha(text.encode()),
        "synthetic_source_transform_sha256": _digest(None),
    }
    assert candidate_acceptance._source_seed_verified(source, metadata)
    assert not candidate_acceptance._source_seed_verified(
        source, dict(metadata, synthetic_source_transform_sha256="c" * 64)
    )


def test_prequalification_parses_nonmatching_solver_action_before_objective() -> None:
    candidate = _candidate(_source_row())
    alternative = EditAction("insert_before", "    # independent solver action")
    wire = encode_action(alternative)
    role = json.loads(_role(candidate))
    role["solver"]["wire"] = wire
    role["solver"]["wire_sha256"] = _sha(wire.encode())
    role["solver"]["generated_tokens"] = len(wire.encode()) + 1
    role_bytes = json.dumps(role, sort_keys=True, separators=(",", ":")).encode()

    _wire, actual_action, after = _solver_action(
        role_bytes, candidate, ByteTokenizer(), require_candidate_match=False
    )
    assert actual_action == alternative
    assert after != candidate["after_source"]
    with pytest.raises(ValueError, match="differs from the canonical"):
        _solver_action(role_bytes, candidate, ByteTokenizer())
