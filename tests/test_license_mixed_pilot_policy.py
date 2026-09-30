from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION, serialize_state_bounded
from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    RecentEdit,
    apply_action,
    encode_action,
)
from tinycomplete.one_line.fixed_state_role_receipts import (
    export_fixed_state_role_execution_proof,
)
from tinycomplete.one_line.fixed_state_roles import (
    build_fixed_state_prompt,
    build_fixed_state_review_prompt,
    fixed_state_protocol_bindings,
)
from tinycomplete.one_line.pilot_data import (
    LICENSE_MIXED,
    LICENSE_MIXED_HISTORY,
    REVIEWED_PUBLIC_HISTORY_SOURCE,
    _canonical_bytes,
    _runtime_sha256,
    _verified_oracle_record,
    license_mixed_artifact_root,
    license_mixed_row_bindings,
    validate_license_mixed_manifest,
    validate_license_mixed_review,
    validate_license_mixed_row,
    validate_license_mixed_source_artifacts,
    validate_license_mixed_splits,
)
from tinycomplete.one_line.pilot_roles import (
    build_blind_solver_prompt,
    build_reviewer_prompt,
)
from tinycomplete.one_line.teacher import MODEL_ID, TeacherResponse, TeacherUsageLedger

_TRAINER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "train_one_line.py"),
    run_name="tabcomplete_train_one_line",
)
load_training_rows = _TRAINER["load_training_rows"]


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8")) + ([0] if add_special_tokens else [])


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _put(root: Path, relative: str, payload: bytes) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": _sha(payload), "bytes": len(payload)}


def _source_package(root: Path) -> dict[str, Any]:
    source = b"value = 1\n"
    license_bytes = b"MIT License\n"
    source_ref = _put(root, "artifacts/source.py", source)
    license_ref = _put(root, "artifacts/LICENSE", license_bytes)
    scope = {
        "schema": "exact-parent-license-scope-v2",
        "results": [
            {
                "candidate_id": "public/sequence-1",
                "status": "verified_path_scope",
                "repository": "public/example",
                "parent_commit": "a" * 40,
                "source_path": "src/example.py",
                "source_sha256": _sha(source),
                "source_group_id": "public/example",
                "source_header_spdx": ["MIT"],
                "source_header_spdx_ambiguous": False,
                "source_header_spdx_expression_count": 0,
                "path_scope": {
                    "license_path": "LICENSE",
                    "git_blob_sha": "c" * 40,
                    "sha256": _sha(license_bytes),
                    "scope": "root",
                    "spdx": ["MIT"],
                    "spdx_ambiguous": False,
                    "spdx_expression_count": 0,
                },
                "root_license": {
                    "path": "LICENSE",
                    "sha256": _sha(license_bytes),
                    "git_blob_sha": "c" * 40,
                    "root_spdx": ["MIT"],
                },
                "additional_license_references": [],
                "reuse_dep5_references": [],
            }
        ],
    }
    scope_ref = _put(root, "artifacts/license-scope.json", _canonical_bytes(scope))
    transform = {
        "kind": "exact_parent_file",
        "start_line_1based": 1,
        "end_line_1based_inclusive": 1,
        "parent_source_sha256": _sha(source),
        "selected_source_sha256": _sha(source),
    }
    state = EditState(
        file_id="src/example.py",
        filetype="python",
        source=source.decode(),
        target_row=0,
        cursor_col=0,
        relevant=("Change value to two",),
    )
    action = EditAction("replace_line", "value = 2")
    history_sha = _sha(_canonical_bytes([]))
    return {
        "id": "public/sequence-1",
        "candidate_id": "public/sequence-1",
        "split": "train",
        "source_type": "synthetic_public_source_task",
        "source_group_id": "public/example",
        "session_or_commit": "a" * 40,
        "task_family_id": "public_source_synthetic_contract_v1",
        "template_id": "template-a",
        "state": asdict(state),
        "action": asdict(action),
        "after_source": apply_action(state, action),
        "source_license": "MIT",
        "human_chronology_observed": False,
        "visible_request_location": "state.relevant[0]",
        "history_sha256": history_sha,
        "authoring_metadata": {
            "source_repo": "public/example",
            "source_revision": "a" * 40,
            "source_tree_sha": "b" * 40,
            "source_path": "src/example.py",
            "source_sha256": _sha(source),
            "source_artifact_path": source_ref["path"],
            "source_artifact_sha256": source_ref["sha256"],
            "source_artifact_bytes": source_ref["bytes"],
            "source_license": "MIT",
            "path_license": "MIT",
            "path_license_sha256": _sha(license_bytes),
            "path_license_git_blob_sha": "c" * 40,
            "license_path": "LICENSE",
            "license_scope_status": "verified_path_scope",
            "path_license_artifact_path": license_ref["path"],
            "path_license_artifact_sha256": license_ref["sha256"],
            "path_license_artifact_bytes": license_ref["bytes"],
            "license_scope_artifact_path": scope_ref["path"],
            "license_scope_sha256": scope_ref["sha256"],
            "license_scope_artifact_bytes": scope_ref["bytes"],
            "selected_source_sha256": _sha(source),
            "transform": transform,
            "transform_sha256": _sha(_canonical_bytes(transform)),
        },
    }


def _license_manifest(*, train_count: int = 128, dev_count: int = 64) -> dict[str, Any]:
    return {
        "schema": LICENSE_MIXED.data_schema,
        "dataset_id": LICENSE_MIXED.dataset_id,
        "dataset_license": LICENSE_MIXED.dataset_license,
        "source_file_license_status": LICENSE_MIXED.file_license_status,
        "train_sha256": "a" * 64,
        "development_sha256": "b" * 64,
        "independent_review_sha256": "c" * 64,
        "independent_review_path": "independent_review.json",
        "independent_review_bytes": 1,
        "train_count": train_count,
        "dev_count": dev_count,
        "file_groups_disjoint": True,
        "candidate_split_counts": {"train": train_count, "development": dev_count},
        "artifact_root": ".",
    }


def _keep_oracle_fixture(tmp_path: Path) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, str], tuple[dict[str, Any], str]],
]:
    state = EditState(
        file_id="synthetic.py",
        filetype="python",
        source="value = 1\n",
        target_row=0,
        cursor_col=0,
        relevant=("Keep the current value",),
    )
    keep = EditAction("keep")
    bad = EditAction("replace_line", "value = 2")
    row: dict[str, Any] = {
        "candidate_id": "synthetic/keep-1",
        "id": "synthetic/keep-1",
        "split": "train",
        "source_type": "synthetic_public_source_task",
        "source_group_id": "public/example",
        "task_family_id": "public_source_synthetic_contract_v1",
        "template_id": "visible-request-n",
        "seed_id": "seed-1",
        "state": asdict(state),
        "action": asdict(keep),
        "after_source": apply_action(state, keep),
        "history_sha256": _sha(_canonical_bytes([])),
        "human_chronology_observed": False,
        "visible_request_location": "state.relevant[0]",
    }
    context = serialize_state_bounded(state, ByteTokenizer(), max_input_tokens=1024)
    row["prompt"] = context.text
    row["context_sha256"] = _sha(context.text.encode())
    objective = {
        "container_image": "registry.example/python@sha256:" + "d" * 64,
        "compile": ["python -m py_compile synthetic.py"],
        "test": ["python -m unittest"],
    }
    fixture_task = {
        "seed_id": "seed-1",
        "action": asdict(keep),
        "wrong_action": asdict(bad),
        "oracle": objective,
    }
    fixture_document = {"seeds": [fixture_task]}
    fixture_payload = _canonical_bytes(fixture_document)
    fixture_ref = _put(tmp_path, "artifacts/fixture.json", fixture_payload)
    fixture_sha = _sha(_canonical_bytes(objective))
    variants: dict[str, dict[str, Any]] = {
        "gold": {
            "action": keep,
            "expected": True,
            "parse": "pass",
            "compile": "pass",
            "test": "pass",
            "functional": "pass",
        },
        "behavior_breaking": {
            "action": bad,
            "expected": False,
            "parse": "pass",
            "compile": "pass",
            "test": "fail",
            "functional": "fail",
        },
    }
    diagnostics: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    record_variants: dict[str, dict[str, Any]] = {}
    for index, (name, variant) in enumerate(variants.items(), start=1):
        applied = apply_action(state, variant["action"])
        action_sha = _sha(_canonical_bytes(asdict(variant["action"])))
        after_sha = _sha(applied.encode())
        work_sha = _sha(("working-tree-" + name).encode())
        case_ids = {
            "case_id": "case-" + name,
            "case_attempt_id": "attempt-" + name,
            "request_id": "request-" + name,
        }
        checks = {
            "parse": (variant["parse"], 0),
            "compile": (variant["compile"], 0),
            "test": (variant["test"], 0 if variant["test"] == "pass" else 1),
        }
        diagnostic = {
            "candidate_id": row["candidate_id"],
            "variant": name,
            **case_ids,
            "parse_status": variant["parse"],
            "compile_status": variant["compile"],
            "test_status": variant["test"],
            "compile_configured": True,
            "test_configured": True,
            "working_tree_sha256": work_sha,
            "checks": {
                check_name: {
                    "status": status,
                    "returncode": returncode,
                    "elapsed_seconds": float(index),
                    "stdout": "",
                    "stderr": "",
                }
                for check_name, (status, returncode) in checks.items()
            },
        }
        diagnostic_sha = _sha(_canonical_bytes(diagnostic))
        diagnostics[(row["candidate_id"], name)] = (diagnostic, diagnostic_sha)
        record_variants[name] = {
            "action_sha256": action_sha,
            "after_source_sha256": after_sha,
            "functional_expected": variant["expected"],
            "functional_status": variant["functional"],
            "parse_status": variant["parse"],
            "compile_status": variant["compile"],
            "test_status": variant["test"],
            "working_tree_sha256": work_sha,
            **case_ids,
            "diagnostic_sha256": diagnostic_sha,
        }
    record = {
        "candidate_id": row["candidate_id"],
        "seed_id": row["seed_id"],
        "split": row["split"],
        "source_group_id": row["source_group_id"],
        "task_family_id": row["task_family_id"],
        "template_id": row["template_id"],
        "source_type": row["source_type"],
        "state_sha256": license_mixed_row_bindings(row)["state_sha256"],
        "action_sha256": license_mixed_row_bindings(row)["action_sha256"],
        "context_sha256": row["context_sha256"],
        "history_sha256": row["history_sha256"],
        "fixture_sha256": fixture_sha,
        "fixture_artifact_path": fixture_ref["path"],
        "fixture_artifact_sha256": fixture_ref["sha256"],
        "fixture_artifact_bytes": fixture_ref["bytes"],
        "evaluator_sha256": "e" * 64,
        "runtime_sha256": _runtime_sha256({"oracle": objective}),
        "execution_backend": "container",
        "network_access": "none",
        "human_chronology_observed": False,
        "variants": record_variants,
    }
    return row, record, objective, fixture_task, diagnostics


def test_source_and_license_artifacts_are_hash_bound_and_relocatable(tmp_path: Path) -> None:
    original = tmp_path / "first-root"
    original.mkdir()
    row = _source_package(original)
    manifest = {"artifact_root": "."}
    first = validate_license_mixed_source_artifacts(row, package_root=original)
    assert license_mixed_artifact_root(manifest, original) == original.resolve()
    relocated = tmp_path / "relocated-root"
    shutil.copytree(original, relocated)
    assert license_mixed_artifact_root(manifest, relocated) == relocated.resolve()
    assert validate_license_mixed_source_artifacts(row, package_root=relocated) == first

    mismatch = json.loads(json.dumps(row))
    mismatch["authoring_metadata"]["path_license"] = "Apache-2.0"
    with pytest.raises(ValueError, match="license identity"):
        validate_license_mixed_source_artifacts(mismatch, package_root=relocated)

    absolute = json.loads(json.dumps(row))
    absolute["authoring_metadata"]["source_artifact_path"] = str(
        original / "artifacts/source.py"
    )
    with pytest.raises(ValueError, match="package-relative"):
        validate_license_mixed_source_artifacts(absolute, package_root=relocated)


def test_manifest_keeps_frozen_train_and_development_floors() -> None:
    validate_license_mixed_manifest(_license_manifest())
    with pytest.raises(ValueError, match="floors"):
        validate_license_mixed_manifest(_license_manifest(train_count=127))
    with pytest.raises(ValueError, match="floors"):
        validate_license_mixed_manifest(_license_manifest(dev_count=63))


def test_fixed_history_policy_is_separate_from_license_mixed_v1(tmp_path: Path) -> None:
    parent = b"value = 1\n"
    parent_sha = _sha(parent)
    source_ref = _put(tmp_path, "artifacts/source.py", parent)
    license_ref = _put(tmp_path, "artifacts/LICENSE", b"MIT License\n")
    state = EditState(
        file_id="src/example.py",
        filetype="python",
        source="value = 2\n",
        target_row=0,
        cursor_col=0,
        history=(RecentEdit(0, "value = 1", "value = 2"),),
        relevant=(),
    )
    action = EditAction("replace_line", "value = 3")
    transform = {
        "kind": "exact_parent_file",
        "start_line_1based": 1,
        "end_line_1based_inclusive": 1,
        "parent_source_sha256": parent_sha,
        "selected_source_sha256": parent_sha,
    }
    scope = {
        "schema": "exact-parent-license-scope-v2",
        "results": [
            {
                "candidate_id": "public/history-seed-1",
                "status": "verified_path_scope",
                "repository": "public/example",
                "parent_commit": "a" * 40,
                "source_path": "src/example.py",
                "source_sha256": parent_sha,
                "source_group_id": "public/example@parent",
                "source_header_spdx": ["MIT"],
                "source_header_spdx_ambiguous": False,
                "source_header_spdx_expression_count": 0,
                "path_scope": {
                    "license_path": "LICENSE",
                    "git_blob_sha": "c" * 40,
                    "sha256": license_ref["sha256"],
                    "scope": "root",
                    "spdx": ["MIT"],
                    "spdx_ambiguous": False,
                    "spdx_expression_count": 0,
                },
                "root_license": {
                    "path": "LICENSE",
                    "sha256": license_ref["sha256"],
                    "git_blob_sha": "c" * 40,
                    "root_spdx": ["MIT"],
                },
                "additional_license_references": [],
                "reuse_dep5_references": [],
            }
        ],
    }
    scope_ref = _put(tmp_path, "artifacts/scope.json", _canonical_bytes(scope))
    state_context = serialize_state_bounded(state, ByteTokenizer(), max_input_tokens=1024)
    history_sha = _sha(_canonical_bytes([asdict(edit) for edit in state.history]))
    row = {
        "id": "public/history-seed-1",
        "candidate_id": "public/history-seed-1",
        "split": "train",
        "source_type": REVIEWED_PUBLIC_HISTORY_SOURCE,
        "source_group_id": "public/example@parent",
        "session_or_commit": "a" * 40,
        "task_family_id": "history-return-value-v1",
        "template_id": "return-value-history-1",
        "seed_id": "seed-1",
        "state": asdict(state),
        "action": asdict(action),
        "after_source": apply_action(state, action),
        "source_license": "MIT",
        "history_order": "synthetic_fixed_before_provider",
        "human_chronology_observed": False,
        "history_sha256": history_sha,
        "context_policy": CONTEXT_POLICY_VERSION,
        "prompt": state_context.text,
        "context_sha256": _sha(state_context.text.encode()),
        "authoring_metadata": {
            "source_repo": "public/example",
            "source_revision": "a" * 40,
            "source_tree_sha": "b" * 40,
            "source_path": "src/example.py",
            "source_sha256": parent_sha,
            "source_artifact_path": source_ref["path"],
            "source_artifact_sha256": source_ref["sha256"],
            "source_artifact_bytes": source_ref["bytes"],
            "source_license": "MIT",
            "path_license": "MIT",
            "path_license_sha256": license_ref["sha256"],
            "path_license_git_blob_sha": "c" * 40,
            "license_path": "LICENSE",
            "license_scope_status": "verified_path_scope",
            "path_license_artifact_path": license_ref["path"],
            "path_license_artifact_sha256": license_ref["sha256"],
            "path_license_artifact_bytes": license_ref["bytes"],
            "license_scope_artifact_path": scope_ref["path"],
            "license_scope_sha256": scope_ref["sha256"],
            "license_scope_artifact_bytes": scope_ref["bytes"],
            "selected_source_sha256": parent_sha,
            "transform": transform,
            "transform_sha256": _sha(_canonical_bytes(transform)),
        },
    }
    validate_license_mixed_row(
        row, package_root=tmp_path, policy=LICENSE_MIXED_HISTORY
    )
    with pytest.raises(ValueError, match="unapproved source type"):
        validate_license_mixed_row(row, package_root=tmp_path, policy=LICENSE_MIXED)
    no_history = json.loads(json.dumps(row))
    no_history["history_order"] = "observed"
    with pytest.raises(ValueError, match="fixed-history row declaration"):
        validate_license_mixed_row(
            no_history, package_root=tmp_path, policy=LICENSE_MIXED_HISTORY
        )
    request_cue = json.loads(json.dumps(row))
    request_cue["state"]["relevant"] = ["replace value with three"]
    with pytest.raises(ValueError, match="fixed-history row declaration"):
        validate_license_mixed_row(
            request_cue, package_root=tmp_path, policy=LICENSE_MIXED_HISTORY
        )


@pytest.fixture(scope="module")
def _fixed_history_review_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any], list[Mapping[str, Any]]]:
    controller = tmp_path_factory.mktemp("fixed-history-review") / "controller-package"
    manifest, rows = _full_review_package(controller, policy=LICENSE_MIXED_HISTORY)
    return controller, manifest, rows


def test_fixed_history_review_revalidates_portable_role_and_oracle_receipts(
    tmp_path: Path,
    _fixed_history_review_package: tuple[Path, dict[str, Any], list[Mapping[str, Any]]],
) -> None:
    controller, frozen_manifest, rows = _fixed_history_review_package
    manifest = json.loads(json.dumps(frozen_manifest))
    staged = tmp_path / "relocated-package"
    shutil.copytree(controller, staged)
    result = validate_license_mixed_review(
        manifest,
        rows,
        tokenizer=ByteTokenizer(),
        package_root=staged,
        policy=LICENSE_MIXED_HISTORY,
    )
    assert result["verified_rows"] == 192
    assert result["quality_evidence"] is False
    action_counts = {
        kind: sum(EditAction(**row["action"]).kind == kind for row in rows)
        for kind in ("keep", "replace_line", "insert_before", "delete_line")
    }
    assert action_counts == {
        "keep": 48,
        "replace_line": 48,
        "insert_before": 48,
        "delete_line": 48,
    }

    task_document = json.loads(
        (staged / "artifacts/objective-fixtures.json").read_bytes()
    )
    task_by_seed = {task["seed_id"]: task for task in task_document["seeds"]}
    representative_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        kind = EditAction(**row["action"]).kind
        if row["split"] == "train" and kind not in representative_rows:
            representative_rows[kind] = row

    def execute(row: Mapping[str, Any], source: str) -> Any:
        kind = EditAction(**row["action"]).kind
        suffix = int(str(row["candidate_id"]).rsplit("-", 1)[1])
        namespace: dict[str, Any] = {}
        exec(compile(source, "<policy-test-fixture>", "exec"), namespace)
        if kind == "keep":
            return namespace["answer"]
        if kind == "replace_line":
            return namespace[f"double_{suffix}"](3)
        if kind == "insert_before":
            return namespace[f"result_{suffix}"]
        if kind == "delete_line":
            return namespace[f"answer_{suffix}"]()
        raise AssertionError("unexpected fixed-history action kind")

    expected_outputs = {
        "replace_line": (6, 3, 9),
        "insert_before": (4, "NameError", "NameError"),
        "delete_line": (45, "NotImplementedError", 0),
    }
    for action_kind, row in representative_rows.items():
        state = EditState.from_mapping(row["state"])
        action = EditAction(**row["action"])
        task = task_by_seed[row["seed_id"]]
        wrong_action = EditAction(**task["wrong_action"])
        variants = {
            "before": state.source,
            "gold": apply_action(state, action),
            "behavior_breaking": apply_action(state, wrong_action),
        }
        for source in variants.values():
            compile(source, "<policy-test-fixture>", "exec")
        results: dict[str, Any] = {}
        for name, source in variants.items():
            try:
                results[name] = execute(row, source)
            except Exception as error:
                results[name] = type(error).__name__
        if action_kind == "keep":
            assert results["before"] == results["gold"]
            assert results["behavior_breaking"] != results["gold"]
        else:
            expected_gold, expected_before, expected_wrong = expected_outputs[action_kind]
            assert results == {
                "before": expected_before,
                "gold": expected_gold,
                "behavior_breaking": expected_wrong,
            }

    review_path = staged / manifest["independent_review_path"]
    review = json.loads(review_path.read_bytes())
    role_entry = review["rows"][0]
    unproved_rows = json.loads(json.dumps(rows))
    del unproved_rows[0]["role_execution_proof_ref"]
    with pytest.raises(ValueError, match="runner proof is unpinned"):
        validate_license_mixed_review(
            manifest,
            unproved_rows,
            tokenizer=ByteTokenizer(),
            package_root=staged,
            policy=LICENSE_MIXED_HISTORY,
        )

    role_payload = b"{}"
    role_path = staged / role_entry["role_evidence_artifact_path"]
    role_path.write_bytes(role_payload)
    with pytest.raises(ValueError, match="runner proof failed portable verification"):
        validate_license_mixed_review(
            manifest,
            rows,
            tokenizer=ByteTokenizer(),
            package_root=staged,
            policy=LICENSE_MIXED_HISTORY,
        )


def test_fixed_history_review_rejects_rebound_controls_and_retargeted_edits(
    tmp_path: Path,
    _fixed_history_review_package: tuple[Path, dict[str, Any], list[Mapping[str, Any]]],
) -> None:
    controller, frozen_manifest, rows = _fixed_history_review_package
    manifest = json.loads(json.dumps(frozen_manifest))
    staged = tmp_path / "staged-package"
    shutil.copytree(controller, staged)
    replacement = next(
        row for row in rows if EditAction(**row["action"]).kind == "replace_line"
    )
    candidate_id = str(replacement["candidate_id"])

    nonfailing_root = tmp_path / "nonfailing-control"
    shutil.copytree(staged, nonfailing_root)
    nonfailing_manifest = json.loads(json.dumps(manifest))
    _rewrite_history_oracle_variant(
        nonfailing_root,
        nonfailing_manifest,
        candidate_id=candidate_id,
        variant_name="behavior_breaking",
        variant_updates={},
        test_status="pass",
    )
    with pytest.raises(ValueError, match="behavior-breaking control did not fail functionally"):
        validate_license_mixed_review(
            nonfailing_manifest,
            rows,
            tokenizer=ByteTokenizer(),
            package_root=nonfailing_root,
            policy=LICENSE_MIXED_HISTORY,
        )

    rebound_root = tmp_path / "control-rebound-to-gold"
    shutil.copytree(staged, rebound_root)
    rebound_manifest = json.loads(json.dumps(manifest))
    oracle_rows = [
        json.loads(line)
        for line in (rebound_root / rebound_manifest["oracle_results_path"])
        .read_bytes()
        .splitlines()
    ]
    oracle = next(item for item in oracle_rows if item["candidate_id"] == candidate_id)
    gold = oracle["variants"]["gold"]
    _rewrite_history_oracle_variant(
        rebound_root,
        rebound_manifest,
        candidate_id=candidate_id,
        variant_name="behavior_breaking",
        variant_updates={
            "action_sha256": gold["action_sha256"],
            "after_source_sha256": gold["after_source_sha256"],
        },
    )
    with pytest.raises(ValueError, match="behavior-breaking control did not fail functionally"):
        validate_license_mixed_review(
            rebound_manifest,
            rows,
            tokenizer=ByteTokenizer(),
            package_root=rebound_root,
            policy=LICENSE_MIXED_HISTORY,
        )

    retargeted_rows = json.loads(json.dumps(rows))
    retargeted = next(item for item in retargeted_rows if item["candidate_id"] == candidate_id)
    state = EditState.from_mapping(retargeted["state"])
    moved_state = EditState(
        file_id=state.file_id,
        filetype=state.filetype,
        source=state.source,
        target_row=0,
        cursor_col=state.cursor_col,
        history=state.history,
        relevant=state.relevant,
    )
    retargeted["state"] = asdict(moved_state)
    retargeted["after_source"] = apply_action(
        moved_state, EditAction(**retargeted["action"])
    )
    context = serialize_state_bounded(moved_state, ByteTokenizer(), max_input_tokens=1024)
    retargeted["prompt"] = context.text
    retargeted["context_sha256"] = _sha(context.text.encode("utf-8"))
    with pytest.raises(ValueError, match="differs from independently reviewed bindings"):
        validate_license_mixed_review(
            manifest,
            retargeted_rows,
            tokenizer=ByteTokenizer(),
            package_root=staged,
            policy=LICENSE_MIXED_HISTORY,
        )


def test_split_audit_rejects_repository_and_task_family_leakage(tmp_path: Path) -> None:
    first = _source_package(tmp_path)
    second = json.loads(json.dumps(first))
    second["candidate_id"] = second["id"] = "public/sequence-2"
    second["split"] = "development"
    second["state"]["source"] = "value = 3\n"
    second["state"]["target_row"] = 0
    second["authoring_metadata"]["source_repo"] = "public/example"
    second["source_group_id"] = "public/example"
    with pytest.raises(ValueError, match="crosses splits"):
        validate_license_mixed_splits([first, second])


def test_split_audit_rejects_duplicate_input_hidden_by_file_and_repo_aliases(
    tmp_path: Path,
) -> None:
    first = _source_package(tmp_path)
    duplicate = json.loads(json.dumps(first))
    duplicate["candidate_id"] = duplicate["id"] = "public/sequence-alias"
    duplicate["state"]["file_id"] = "renamed.py"
    duplicate["source_group_id"] = "public/alias-repo"
    duplicate["session_or_commit"] = "d" * 40
    duplicate["task_family_id"] = "another-task-family"
    duplicate["template_id"] = "another-template"
    duplicate["authoring_metadata"]["source_repo"] = "public/alias-repo"
    duplicate["authoring_metadata"]["source_revision"] = "d" * 40
    with pytest.raises(ValueError, match="duplicate normalized LICENSE-MIXED model input"):
        validate_license_mixed_splits([first, duplicate])

    distinct = json.loads(json.dumps(duplicate))
    distinct["candidate_id"] = distinct["id"] = "public/sequence-distinct"
    distinct["state"]["relevant"] = ["Keep the separate visible request"]
    assert validate_license_mixed_splits([first, distinct])["rows"] == 2


def test_muse_source_a_can_have_empty_relevant_but_needs_acceptance_package(
    tmp_path: Path,
) -> None:
    state = EditState(
        file_id="public.py",
        filetype="python",
        source="value = 1\n",
        target_row=0,
        cursor_col=0,
        relevant=(),
    )
    action = EditAction("replace_line", "value = 2")
    row = {
        "candidate_id": "muse/source-a",
        "id": "muse/source-a",
        "source_type": "muse_author_public_candidate",
        "source_license": "MIT",
        "split": "train",
        "human_chronology_observed": False,
        "state": asdict(state),
        "action": asdict(action),
        "after_source": apply_action(state, action),
        "acceptance_package_ref": {
            "schema": "one-line-muse-acceptance-package-v1",
            "root": "muse/source-a",
            "candidate_id": "muse/source-a",
            "manifest_sha256": "a" * 64,
            "candidate_sha256": "b" * 64,
            "source_row_sha256": "c" * 64,
        },
    }
    validate_license_mixed_row(row, package_root=tmp_path)
    v2_row = json.loads(json.dumps(row))
    v2_row["acceptance_package_ref"]["schema"] = "one-line-muse-acceptance-package-v2"
    validate_license_mixed_row(v2_row, package_root=tmp_path)
    unsupported = json.loads(json.dumps(row))
    unsupported["acceptance_package_ref"]["schema"] = "one-line-muse-acceptance-package-v3"
    with pytest.raises(ValueError, match="acceptance package"):
        validate_license_mixed_row(unsupported, package_root=tmp_path)
    context = serialize_state_bounded(state, ByteTokenizer(), max_input_tokens=1024)
    row["prompt"] = context.text
    row["context_sha256"] = _sha(context.text.encode())
    from tinycomplete.one_line.pilot_data import _context_action_bindings

    assert _context_action_bindings(row, ByteTokenizer())["context_includes_request"] is False

    missing = dict(row)
    del missing["acceptance_package_ref"]
    with pytest.raises(ValueError, match="acceptance package"):
        validate_license_mixed_row(missing, package_root=tmp_path)

    raw_teacher = dict(row, source_type="opencode_raw_teacher_output")
    with pytest.raises(ValueError, match="unapproved source type"):
        validate_license_mixed_row(raw_teacher, package_root=tmp_path)


def test_keep_n_requires_positive_gold_and_failing_behavior_control(tmp_path: Path) -> None:
    row, record, objective, fixture_task, diagnostics = _keep_oracle_fixture(tmp_path)
    fixture_path = tmp_path / "artifacts/fixture.json"
    manifest = {"oracle_evaluator_sha256": "e" * 64}
    fixture = {"oracle": objective}
    kwargs = {
        "manifest": manifest,
        "artifact_root": tmp_path,
        "expected_fixture_sha256": record["fixture_sha256"],
        "expected_fixture_artifact": {
            "path": record["fixture_artifact_path"],
            "sha256": record["fixture_artifact_sha256"],
            "bytes": record["fixture_artifact_bytes"],
        },
        "expected_runtime_sha256": record["runtime_sha256"],
        "fixture_task": fixture_task,
        "fixture_objective": fixture["oracle"],
        "diagnostics": diagnostics,
    }
    assert fixture_path.is_file()
    assert _verified_oracle_record(row, record, **kwargs)["fixture_sha256"] == record[
        "fixture_sha256"
    ]

    no_gold = json.loads(json.dumps(record))
    no_gold_diagnostics = dict(diagnostics)
    gold_diagnostic = json.loads(json.dumps(diagnostics[(row["candidate_id"], "gold")][0]))
    gold_diagnostic["test_status"] = "fail"
    gold_diagnostic["checks"]["test"]["status"] = "fail"
    gold_diagnostic["checks"]["test"]["returncode"] = 1
    gold_diag_sha = _sha(_canonical_bytes(gold_diagnostic))
    no_gold_diagnostics[(row["candidate_id"], "gold")] = (gold_diagnostic, gold_diag_sha)
    no_gold["variants"]["gold"].update(
        {"test_status": "fail", "functional_status": "fail", "diagnostic_sha256": gold_diag_sha}
    )
    with pytest.raises(ValueError, match="positive action"):
        _verified_oracle_record(row, no_gold, **{**kwargs, "diagnostics": no_gold_diagnostics})

    no_bad_control = json.loads(json.dumps(record))
    no_bad_diagnostics = dict(diagnostics)
    bad_diagnostic = json.loads(
        json.dumps(diagnostics[(row["candidate_id"], "behavior_breaking")][0])
    )
    bad_diagnostic["test_status"] = "pass"
    bad_diagnostic["checks"]["test"]["status"] = "pass"
    bad_diagnostic["checks"]["test"]["returncode"] = 0
    bad_diag_sha = _sha(_canonical_bytes(bad_diagnostic))
    no_bad_diagnostics[(row["candidate_id"], "behavior_breaking")] = (
        bad_diagnostic,
        bad_diag_sha,
    )
    no_bad_control["variants"]["behavior_breaking"].update(
        {"test_status": "pass", "functional_status": "pass", "diagnostic_sha256": bad_diag_sha}
    )
    with pytest.raises(ValueError, match="behavior-breaking control"):
        _verified_oracle_record(
            row, no_bad_control, **{**kwargs, "diagnostics": no_bad_diagnostics}
        )

    no_visible_request = json.loads(json.dumps(row))
    no_visible_request["visible_request_location"] = None
    with pytest.raises(ValueError, match="visible explicit request"):
        _verified_oracle_record(no_visible_request, record, **kwargs)


def test_trainer_rejects_too_small_train_and_development_shards(tmp_path: Path) -> None:
    def rows(count: int, split: str) -> list[dict[str, Any]]:
        output = []
        for index in range(count):
            source = f"value = {index}\n"
            state = EditState(
                file_id=f"file-{index}.py",
                filetype="python",
                source=source,
                target_row=0,
                cursor_col=0,
                relevant=(),
            )
            action = EditAction("replace_line", f"value = {index + 1}")
            output.append(
                {
                    "candidate_id": f"muse/candidate-{split}-{index}",
                    "id": f"muse/candidate-{split}-{index}",
                    "split": split,
                    "source_type": "muse_author_public_candidate",
                    "source_license": "MIT",
                    "human_chronology_observed": False,
                    "state": asdict(state),
                    "action": asdict(action),
                    "after_source": apply_action(state, action),
                    "acceptance_package_ref": {
                        "schema": "one-line-muse-acceptance-package-v1",
                        "root": f"muse/{index}",
                        "candidate_id": f"muse/candidate-{split}-{index}",
                        "manifest_sha256": "a" * 64,
                        "candidate_sha256": "b" * 64,
                        "source_row_sha256": "c" * 64,
                    },
                }
            )
        return output

    train_path = tmp_path / "train.jsonl"
    train_payload = b"\n".join(_canonical_bytes(row) for row in rows(127, "train")) + b"\n"
    train_path.write_bytes(train_payload)
    with pytest.raises(ValueError, match="128 to 1,024"):
        load_training_rows(
            train_path,
            _sha(train_payload),
            phase="pilot",
            minimum_main_train=20_000,
            pilot_schema=LICENSE_MIXED.data_schema,
            package_root=tmp_path,
        )

    dev_path = tmp_path / "development.jsonl"
    dev_payload = b"\n".join(_canonical_bytes(row) for row in rows(63, "development")) + b"\n"
    dev_path.write_bytes(dev_payload)
    with pytest.raises(ValueError, match="at least 64"):
        load_training_rows(
            dev_path,
            _sha(dev_payload),
            phase="pilot",
            minimum_main_train=20_000,
            pilot_schema=LICENSE_MIXED.data_schema,
            expected_split="development",
            package_root=tmp_path,
        )


def _fixed_state_roles(row: Mapping[str, Any], tokenizer: Any, index: int) -> bytes:
    state = EditState.from_mapping(row["state"])
    action = EditAction(**row["action"])
    student = build_fixed_state_prompt(state, tokenizer)
    wire = encode_action(action)
    reviewer_prompt = build_fixed_state_review_prompt(state, wire, wire, tokenizer)
    verdict = _canonical_bytes(
        {"retain": True, "ambiguous": False, "reason": "The actions match the visible history."}
    ).decode("utf-8")
    common = {
        "model_id": MODEL_ID,
        "finish_reason": "stop",
        "provider_input_tokens": 100,
        "provider_output_tokens": 10,
        "provider_reasoning_tokens": 0,
    }
    wire_sha = _sha(wire.encode("utf-8"))
    reviewer_sha = _sha(verdict.encode("utf-8"))

    def student_role(name: str) -> dict[str, Any]:
        return {
            **common,
            "actor_id": f"actor-{name}-{index:03d}",
            "session_id": f"session-{name}-{index:03d}",
            "request_id": f"request-{name}-{index:03d}",
            "response_id": f"response-{name}-{index:03d}",
            "prompt_sha256": student.prompt_sha256,
            "response_sha256": wire_sha,
            "wire": wire,
            "wire_sha256": wire_sha,
            "q25_supervised_tokens_including_eos": (
                len(tokenizer.encode(wire, add_special_tokens=False)) + 1
            ),
        }

    record = {
        "schema": "one-line-fixed-state-roles-v1",
        "candidate_id": row["candidate_id"],
        "protocol": fixed_state_protocol_bindings(row, tokenizer),
        "author": student_role("author"),
        "solver": student_role("solver"),
        "reviewer": {
            **common,
            "actor_id": f"actor-review-{index:03d}",
            "session_id": f"session-review-{index:03d}",
            "request_id": f"request-review-{index:03d}",
            "response_id": f"response-review-{index:03d}",
            "prompt_sha256": reviewer_prompt.prompt_sha256,
            "response_sha256": reviewer_sha,
            "verdict_json": verdict,
            "verdict_sha256": reviewer_sha,
            "q25_supervised_tokens_including_eos": (
                len(tokenizer.encode(verdict, add_special_tokens=False)) + 1
            ),
        },
    }
    return _canonical_bytes(record)


def _run_fixed_state_role_batch(
    candidates: list[Mapping[str, Any]],
    *,
    work_root: Path,
    package_root: Path,
    tokenizer: Any,
    batch_index: int,
) -> dict[str, dict[str, Any]]:
    """Create real-format portable receipts with the local runner and a fake client."""
    runner_path = Path(__file__).parents[1] / "scripts/run_fixed_state_role_pilot.py"
    runner_spec = importlib.util.spec_from_file_location(
        f"fixed_state_policy_test_runner_{batch_index}", runner_path
    )
    assert runner_spec is not None and runner_spec.loader is not None
    runner = importlib.util.module_from_spec(runner_spec)
    sys.modules[runner_spec.name] = runner
    runner_spec.loader.exec_module(runner)

    phase_dir = work_root / f"phase-{batch_index:03d}"
    phase_dir.mkdir(mode=0o700)
    packet_path = phase_dir / "inputs.jsonl"
    packet_rows = [
        {
            "candidate_id": row["candidate_id"],
            "seed_id": row["seed_id"],
            "state": row["state"],
            "prompt": row["prompt"],
            "context_sha256": row["context_sha256"],
        }
        for row in candidates
    ]
    packet_path.write_bytes(
        b"\n".join(_canonical_bytes(row) for row in packet_rows) + b"\n"
    )
    packet_path.chmod(0o600)
    run_dir = phase_dir / "run"
    ledger_path = phase_dir / "usage.jsonl"
    runtime = {"fixture": "policy-test-local-fake"}
    plan = runner.freeze_plan(
        packet_path=packet_path,
        run_dir=run_dir,
        ledger_path=ledger_path,
        tokenizer=tokenizer,
        runtime=runtime,
    )
    planned_ids = [
        case["request_ids"][role]
        for case in plan["cases"]
        for role in runner.CALL_ROLES
    ]
    release = {
        "schema": runner.RELEASE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "baseline_ledger_sha256": plan["ledger"]["baseline_sha256"],
        "baseline_request_ids_sha256": plan["ledger"]["baseline_request_ids_sha256"],
        "planned_request_ids": planned_ids,
        "max_calls": len(planned_ids),
        "max_wall_seconds": runner.MAX_PHASE_WALL_SECONDS,
        "reserve_input_tokens": runner.RESERVE_INPUT_TOKENS,
        "reserve_output_tokens": runner.RESERVE_OUTPUT_TOKENS,
        "global_caps": {
            "calls": runner.MAX_CALLS,
            "input_tokens": runner.MAX_INPUT_TOKENS,
            "output_tokens": runner.MAX_OUTPUT_TOKENS,
        },
        "released_by": "test-only-local-fixture",
        "released_at_utc": "2026-09-30T00:00:00Z",
    }
    release_bytes = runner._canonical(release) + b"\n"
    release_path = phase_dir / "root-release.json"
    release_path.write_bytes(release_bytes)
    release_path.chmod(0o600)
    rows_by_request = {
        case["request_ids"][role]: next(
            row for row in candidates if row["candidate_id"] == case["candidate_id"]
        )
        for case in plan["cases"]
        for role in runner.CALL_ROLES
    }

    class FakeClient:
        def __init__(self, ledger: TeacherUsageLedger) -> None:
            self.ledger = ledger
            self.calls = 0

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def run_role(self, **kwargs: Any) -> TeacherResponse:
            self.calls += 1
            request_id = kwargs["request_id"]
            self.ledger.reserve(
                request_id,
                input_tokens=kwargs["reserve_input_tokens"],
                max_output_tokens=kwargs["reserve_output_tokens"],
            )
            if kwargs["purpose"] == "automated_score":
                text = json.dumps(
                    {"retain": True, "ambiguous": False, "reason": "Local fixture."},
                    separators=(",", ":"),
                )
            else:
                candidate = rows_by_request[request_id]
                text = encode_action(EditAction(**candidate["action"]))
            response = TeacherResponse(
                content=text,
                session_id=f"policy-test-session-{batch_index}-{self.calls}",
                response_id=f"policy-test-response-{batch_index}-{self.calls}",
                model_id=MODEL_ID,
                input_tokens=10,
                output_tokens=len(text.encode("utf-8")),
                reasoning_tokens=0,
                total_tokens_reported=None,
                cached_read_tokens=0,
                cached_write_tokens=0,
                finish_reason="stop",
                cost_usd_reported=None,
            )
            kwargs["persist_completed_response"](response)
            self.ledger.settle(
                request_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens + response.reasoning_tokens,
            )
            return response

    from unittest.mock import patch

    with (
        patch.object(runner, "run_scope", lambda *_args: nullcontext(None)),
        patch.object(runner, "_enable_offline_observability", lambda *_args: None),
        patch.object(runner, "operation", lambda *_args, **_kwargs: nullcontext(None)),
    ):
        summary = runner.execute_plan(
            run_dir=run_dir,
            release_path=release_path,
            release_sha256=_sha(release_bytes),
            tokenizer=tokenizer,
            runtime=runtime,
            ledger_path_override=ledger_path,
            client_factory=FakeClient,
        )
    assert summary["phase_status"] == "complete"

    proofs = {}
    for candidate in candidates:
        proof_ref = export_fixed_state_role_execution_proof(
            candidate,
            run_dir=run_dir,
            release_path=release_path,
            release_sha256=_sha(release_bytes),
            ledger_path=ledger_path,
            package_root=package_root,
            tokenizer=tokenizer,
        )
        proofs[str(candidate["candidate_id"])] = proof_ref
    return proofs


def _fixed_state_role_proofs(
    candidates: list[Mapping[str, Any]],
    *,
    work_root: Path,
    package_root: Path,
    tokenizer: Any,
) -> dict[str, dict[str, Any]]:
    work_root.mkdir(mode=0o700)
    package_root.chmod(0o700)
    proofs: dict[str, dict[str, Any]] = {}
    for batch_index, offset in enumerate(range(0, len(candidates), 8)):
        proofs.update(
            _run_fixed_state_role_batch(
                candidates[offset : offset + 8],
                work_root=work_root,
                package_root=package_root,
                tokenizer=tokenizer,
                batch_index=batch_index,
            )
        )
    return proofs


def _full_review_package(
    root: Path,
    *,
    tokenizer: Any | None = None,
    policy: Any = LICENSE_MIXED,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """Build explicit test-only proofs for the entire frozen 128/64 review path."""
    from tinycomplete.eval.code_benchmark import evaluate_prediction

    root.mkdir(parents=True)
    history_policy = policy is LICENSE_MIXED_HISTORY
    tokenizer = tokenizer or ByteTokenizer()
    license_payload = b"MIT License\nSynthetic policy-test fixture only.\n"
    license_ref = _put(root, "artifacts/LICENSE", license_payload)
    license_sha = license_ref["sha256"]
    license_blob = "f" * 40
    objective = {
        "container_image": "python:3.11-slim@sha256:" + "d" * 64,
        "compile": ["python -m py_compile src/value.py"],
        "test": ["python -m unittest"],
    }
    evaluator_sha = _sha(Path(evaluate_prediction.__code__.co_filename).read_bytes())
    rows: list[dict[str, Any]] = []
    source_scope_rows: list[dict[str, Any]] = []
    source_audit_rows: list[dict[str, Any]] = []
    provenance_candidates: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []
    row_artifacts: list[dict[str, Any]] = []

    for split, count, split_offset in (("train", 128, 0), ("development", 64, 128)):
        for local_index in range(count):
            index = split_offset + local_index
            candidate_id = f"synthetic-policy-test/{split}-{local_index:03d}"
            seed_id = f"seed-{index:03d}"
            history: tuple[RecentEdit, ...] = ()
            repo = f"public/policy-fixture-{index:03d}"
            commit = f"{index + 1:040x}"
            tree_sha = f"{index + 1001:040x}"
            source_path = "src/value.py"
            if history_policy and index % 4 == 0:
                operators = ("+", "-", "*", "//", "%", "|", "&", "^")

                def expression(
                    pattern: int, operators: tuple[str, ...] = operators
                ) -> str:
                    digits = []
                    for _ in range(3):
                        digits.append(operators[pattern % len(operators)])
                        pattern //= len(operators)
                    return "answer = 1 " + " ".join(
                        token
                        for pair in zip(digits, ("2", "3", "4"), strict=True)
                        for token in pair
                    )

                old_source = expression((index - 1) % 512)
                state_line = expression(index)
                if split == "development":
                    old_source = old_source.replace("answer =", "answer: int =")
                    state_line = state_line.replace("answer =", "answer: int =")
                state_source = state_line + "\n"
                source_payload = (old_source + "\n").encode()
                history = (RecentEdit(0, old_source, state_line),)
                relevant: tuple[str, ...] = ()
                target_row = 0
                action = EditAction("keep")
                wrong_action = EditAction("replace_line", "answer = -1")
            elif history_policy and index % 4 == 1:
                function_name = f"double_{index}"
                if split == "train":
                    source_payload = (
                        f"def {function_name}(value):\n    return value * 2\n"
                    ).encode()
                    state_source = f"def {function_name}(value):\n    return value\n"
                    target_row = 1
                    old_line = "    return value * 2"
                    new_line = "    return value"
                else:
                    source_payload = (
                        f"class Box_{index}:\n"
                        "    def double(self, value):\n"
                        "        return value * 2\n"
                    ).encode()
                    state_source = (
                        f"class Box_{index}:\n"
                        "    def double(self, value):\n"
                        "        return value\n"
                    )
                    target_row = 2
                    old_line = "        return value * 2"
                    new_line = "        return value"
                history = (RecentEdit(target_row, old_line, new_line),)
                relevant = ()
                action = EditAction("replace_line", old_line)
                wrong_action = EditAction("replace_line", old_line.replace("2", "3"))
            elif history_policy and index % 4 == 2:
                result_name = f"result_{index}"
                input_value = index + 1
                if split == "train":
                    expression_line = f"{result_name} = ceil({input_value}.2)"
                else:
                    expression_line = (
                        f"def calculate_{index}():\n    return ceil({input_value}.2)"
                    )
                source_payload = f"from math import ceil\n{expression_line}\n".encode()
                state_source = f"# import removed\n{expression_line}\n"
                history = (RecentEdit(0, "from math import ceil", "# import removed"),)
                relevant = ()
                target_row = 0
                action = EditAction("insert_before", "from math import ceil")
                wrong_action = EditAction("insert_before", "from math import floor")
            elif history_policy:
                function_name = f"answer_{index}"
                answer = index + 42
                if split == "train":
                    source_payload = (
                        f"def {function_name}():\n"
                        f"    return {answer}\n"
                        f"    return {answer}\n"
                    ).encode()
                    state_source = (
                        f"def {function_name}():\n"
                        "    raise NotImplementedError()\n"
                        f"    return {answer}\n"
                    )
                    target_row = 1
                    old_line = f"    return {answer}"
                    new_line = "    raise NotImplementedError()"
                    wrong_line = "    return 0"
                else:
                    source_payload = (
                        f"class Answer_{index}:\n"
                        "    def value(self):\n"
                        f"        return {answer}\n"
                        f"        return {answer}\n"
                    ).encode()
                    state_source = (
                        f"class Answer_{index}:\n"
                        "    def value(self):\n"
                        "        raise NotImplementedError()\n"
                        f"        return {answer}\n"
                    )
                    target_row = 2
                    old_line = f"        return {answer}"
                    new_line = "        raise NotImplementedError()"
                    wrong_line = "        return 0"
                history = (RecentEdit(target_row, old_line, new_line),)
                relevant = ()
                action = EditAction("delete_line")
                wrong_action = EditAction("replace_line", wrong_line)
            else:
                values = ",".join("1" for _ in range(local_index + 1))
                prefix = "value = " if split == "train" else "value: list[int] = "
                source_payload = f"{prefix}[{values}]\n".encode()
                state_source = source_payload.decode("utf-8")
                history = ()
                relevant = ("Keep the current public value unchanged.",)
                target_row = 0
                action = EditAction("keep")
                wrong_action = EditAction("replace_line", "value = []")
            source_sha = _sha(source_payload)
            source_ref = _put(
                root,
                f"artifacts/sources/{split}-{local_index:03d}.py",
                source_payload,
            )
            state = EditState(
                file_id=source_path,
                filetype="python",
                source=state_source,
                target_row=target_row,
                cursor_col=0,
                history=history,
                relevant=relevant,
            )
            after_source = apply_action(state, action)
            request = state.relevant[0] if state.relevant else None
            history_sha = _sha(
                _canonical_bytes([asdict(edit) for edit in state.history])
            )
            transform = {
                "kind": "exact_parent_file",
                "start_line_1based": 1,
                "end_line_1based_inclusive": len(source_payload.splitlines()),
                "parent_source_sha256": source_sha,
                "selected_source_sha256": source_sha,
            }
            metadata: dict[str, Any] = {
                "source_repo": repo,
                "source_revision": commit,
                "source_tree_sha": tree_sha,
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_artifact_path": source_ref["path"],
                "source_artifact_sha256": source_ref["sha256"],
                "source_artifact_bytes": source_ref["bytes"],
                "source_license": "MIT",
                "path_license": "MIT",
                "path_license_spdx": ["MIT"],
                "path_license_scope": "root",
                "path_license_sha256": license_sha,
                "path_license_git_blob_sha": license_blob,
                "license_path": "LICENSE",
                "license_scope_status": "verified_path_scope",
                "path_license_artifact_path": license_ref["path"],
                "path_license_artifact_sha256": license_ref["sha256"],
                "path_license_artifact_bytes": license_ref["bytes"],
                "selected_source_sha256": source_sha,
                "transform": transform,
                "transform_sha256": _sha(_canonical_bytes(transform)),
            }
            context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
            row: dict[str, Any] = {
                "id": candidate_id,
                "candidate_id": candidate_id,
                "split": split,
                "source_type": (
                    REVIEWED_PUBLIC_HISTORY_SOURCE
                    if history_policy
                    else "synthetic_public_source_task"
                ),
                "source_group_id": f"group-{index:03d}",
                "session_or_commit": commit,
                "task_family_id": f"contract-family-{index:03d}",
                "template_id": (
                    f"fixed-history-{action.kind}-{index:03d}"
                    if history_policy
                    else f"visible-keep-{index:03d}"
                ),
                "seed_id": seed_id,
                "state": asdict(state),
                "action": asdict(action),
                "after_source": after_source,
                "source_license": "MIT",
                "human_chronology_observed": False,
                "history_sha256": history_sha,
                "prompt": context.text,
                "context_sha256": _sha(context.text.encode("utf-8")),
                "authoring_metadata": metadata,
            }
            if history_policy:
                row["history_order"] = "synthetic_fixed_before_provider"
                row["context_policy"] = CONTEXT_POLICY_VERSION
            else:
                row["visible_request_location"] = "state.relevant[0]"
                row["provenance"] = {
                    "author_actor_id": f"author-test-{index:03d}",
                    "author_response_sha256": _sha(f"author-response-{index}".encode()),
                    "objective": {
                        "kind": "visible_keep_contract",
                        "description": "Retain the visible value under the explicit request.",
                        "checks": ["The value remains unchanged."],
                    },
                }
            fixture_task: dict[str, Any] = {
                "seed_id": seed_id,
                "candidate_id": candidate_id,
                "action": asdict(action),
                "wrong_action": asdict(wrong_action),
                "oracle": objective,
            }
            if not history_policy:
                assert request is not None
                fixture_task["student_request"] = request
            fixture_sha = _sha(_canonical_bytes(objective))
            row["objective_fixture_binding"] = {"sha256": fixture_sha}
            bindings = license_mixed_row_bindings(row)
            scope_row = {
                "candidate_id": candidate_id,
                "status": "verified_path_scope",
                "repository": repo,
                "parent_commit": commit,
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_group_id": row["source_group_id"],
                "source_header_spdx": ["MIT"],
                "source_header_spdx_ambiguous": False,
                "source_header_spdx_expression_count": 0,
                "path_scope": {
                    "license_path": "LICENSE",
                    "git_blob_sha": license_blob,
                    "sha256": license_sha,
                    "scope": "root",
                    "spdx": ["MIT"],
                    "spdx_ambiguous": False,
                    "spdx_expression_count": 0,
                },
                "root_license": {
                    "path": "LICENSE",
                    "sha256": license_sha,
                    "git_blob_sha": license_blob,
                    "root_spdx": ["MIT"],
                },
                "additional_license_references": [],
                "reuse_dep5_references": [],
            }
            source_scope_rows.append(scope_row)
            source_audit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "source_and_license_verified_for_human_review",
                    "repository": repo,
                    "parent_commit": commit,
                    "file_path": source_path,
                    "source_group_id": row["source_group_id"],
                    "source_pair": {"parent_sha256": source_sha},
                    "parent_license": {
                        "tree_sha": tree_sha,
                        "root_license": {"path": "LICENSE", "spdx": "MIT"},
                    },
                }
            )
            provenance_candidate = {
                "candidate_id": candidate_id,
                "repository": repo,
                "source_revision": commit,
                "source_tree_sha": tree_sha,
                "source_path": source_path,
                "parent_source_sha256": source_sha,
                "selected_source_sha256": source_sha,
                "transform_sha256": metadata["transform_sha256"],
                "source_group_id": row["source_group_id"],
                "path_license": "MIT",
                "path_license_spdx": ["MIT"],
                "path_license_scope": "root",
                "path_license_git_blob_sha": license_blob,
                "path_license_sha256": license_sha,
                "history_sha256": history_sha,
                "seed_id": seed_id,
                "state_sha256": bindings["state_sha256"],
                "context_sha256": row["context_sha256"],
                "objective_fixture_sha256": fixture_sha,
                "source_artifact": source_ref,
                "path_license_artifact": license_ref,
            }
            if history_policy:
                provenance_candidate["history_order"] = row["history_order"]
            provenance_candidates.append(provenance_candidate)
            seed_rows.append(
                {
                    "candidate_id": candidate_id,
                    "seed_id": seed_id,
                    **(
                        {"history_order": row["history_order"], "action": asdict(action)}
                        if history_policy
                        else {"student_request": request}
                    ),
                }
            )
            fixture_rows.append(fixture_task)
            rows.append(row)
            row_artifacts.append(
                {
                    "row": row,
                    "fixture_task": fixture_task,
                    "source_ref": source_ref,
                    "provenance_candidate": provenance_candidate,
                }
            )

    scope_payload = _canonical_bytes(
        {"schema": "exact-parent-license-scope-v2", "results": source_scope_rows}
    )
    scope_ref = _put(root, "artifacts/license-scope.json", scope_payload)
    for artifact in row_artifacts:
        row = artifact["row"]
        metadata = row["authoring_metadata"]
        metadata.update(
            {
                "license_scope_artifact_path": scope_ref["path"],
                "license_scope_sha256": scope_ref["sha256"],
                "license_scope_artifact_bytes": scope_ref["bytes"],
            }
        )
        provenance_candidate = artifact["provenance_candidate"]
        provenance_candidate["license_scope_artifact"] = scope_ref
        provenance_candidate["transform_sha256"] = metadata["transform_sha256"]

    spec_ref = _put(root, "artifacts/source-task-spec.json", _canonical_bytes({"seeds": seed_rows}))
    fixtures_payload = _canonical_bytes({"seeds": fixture_rows})
    fixtures_ref = _put(root, "artifacts/objective-fixtures.json", fixtures_payload)
    diagnostics: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    review_entries: list[dict[str, Any]] = []
    role_proofs = (
        _fixed_state_role_proofs(
            [artifact["row"] for artifact in row_artifacts],
            work_root=root.parent / f"{root.name}-role-runner-work",
            package_root=root,
            tokenizer=tokenizer,
        )
        if history_policy
        else {}
    )

    for index, artifact in enumerate(row_artifacts):
        row = artifact["row"]
        candidate_id = row["candidate_id"]
        fixture_task = artifact["fixture_task"]
        state = EditState.from_mapping(row["state"])
        action = EditAction(**row["action"])
        bindings = license_mixed_row_bindings(row)
        variants: dict[str, dict[str, Any]] = {}
        variant_specs = [("gold", action, True, "pass", "pass")]
        if action.kind != "keep":
            variant_specs.append(("before", EditAction("keep"), False, "fail", "fail"))
        variant_specs.append(
            (
                "behavior_breaking",
                EditAction(**fixture_task["wrong_action"]),
                False,
                "fail",
                "fail",
            )
        )
        for name, variant_action, expected, test_status, functional_status in variant_specs:
            after_source = apply_action(state, variant_action)
            diagnostic = {
                "candidate_id": candidate_id,
                "variant": name,
                "case_id": f"test-case-{index:03d}-{name}",
                "case_attempt_id": f"test-attempt-{index:03d}-{name}",
                "request_id": f"test-request-{index:03d}-{name}",
                "parse_status": "pass",
                "compile_status": "pass",
                "test_status": test_status,
                "compile_configured": True,
                "test_configured": True,
                "working_tree_sha256": _sha(f"working-tree-{index}-{name}".encode()),
                "checks": {
                    "parse": {
                        "status": "pass",
                        "returncode": 0,
                        "elapsed_seconds": 0.01,
                        "stdout": "",
                        "stderr": "",
                    },
                    "compile": {
                        "status": "pass",
                        "returncode": 0,
                        "elapsed_seconds": 0.02,
                        "stdout": "",
                        "stderr": "",
                    },
                    "test": {
                        "status": test_status,
                        "returncode": 0 if test_status == "pass" else 1,
                        "elapsed_seconds": 0.03,
                        "stdout": "",
                        "stderr": "",
                    },
                },
            }
            diagnostic_sha = _sha(_canonical_bytes(diagnostic))
            diagnostics.append(diagnostic)
            variants[name] = {
                "action_sha256": _sha(_canonical_bytes(asdict(variant_action))),
                "after_source_sha256": _sha(after_source.encode("utf-8")),
                "functional_expected": expected,
                "functional_status": functional_status,
                "parse_status": "pass",
                "compile_status": "pass",
                "test_status": test_status,
                "working_tree_sha256": diagnostic["working_tree_sha256"],
                "case_id": diagnostic["case_id"],
                "case_attempt_id": diagnostic["case_attempt_id"],
                "request_id": diagnostic["request_id"],
                "diagnostic_sha256": diagnostic_sha,
            }
        fixture_sha = _sha(_canonical_bytes(objective))
        runtime_sha = _runtime_sha256(fixture_task)
        result = {
            "candidate_id": candidate_id,
            "seed_id": row["seed_id"],
            "split": row["split"],
            "source_group_id": row["source_group_id"],
            "task_family_id": row["task_family_id"],
            "template_id": row["template_id"],
            "source_type": row["source_type"],
            "state_sha256": bindings["state_sha256"],
            "action_sha256": bindings["action_sha256"],
            "context_sha256": row["context_sha256"],
            "history_sha256": bindings["history_sha256"],
            "fixture_sha256": fixture_sha,
            "fixture_artifact_path": fixtures_ref["path"],
            "fixture_artifact_sha256": fixtures_ref["sha256"],
            "fixture_artifact_bytes": fixtures_ref["bytes"],
            "evaluator_sha256": evaluator_sha,
            "runtime_sha256": runtime_sha,
            "execution_backend": "container",
            "network_access": "none",
            "human_chronology_observed": False,
            "variants": variants,
        }
        oracle_records.append(result)
        oracle_sha = _sha(_canonical_bytes(result))
        if history_policy:
            role_proof_ref = role_proofs[candidate_id]
            row["role_execution_proof_ref"] = role_proof_ref
            role_ref = {
                "path": (
                    f"{role_proof_ref['root']}/{role_proof_ref['role_evidence_path']}"
                ),
                "sha256": role_proof_ref["role_evidence_sha256"],
                "bytes": role_proof_ref["role_evidence_bytes"],
            }
        else:
            role_candidate = dict(row)
            author_id = row["provenance"]["author_actor_id"]
            role_wire = encode_action(action)
            verdict = _canonical_bytes(
                {
                    "retain": True,
                    "ambiguous": False,
                    "reason": "The explicit request supports the keep action.",
                }
            ).decode("utf-8")
            role = {
                "schema": "one-line-role-evidence-v1",
                "candidate_id": candidate_id,
                "author": {
                    "actor_id": author_id,
                    "session_id": f"author-session-{index:03d}",
                    "response_sha256": row["provenance"]["author_response_sha256"],
                },
                "solver": {
                    "actor_id": f"blind-solver-{index:03d}",
                    "session_id": f"solver-session-{index:03d}",
                    "prompt_sha256": _sha(
                        build_blind_solver_prompt(role_candidate, tokenizer).encode("utf-8")
                    ),
                    "wire": role_wire,
                    "wire_sha256": _sha(role_wire.encode("utf-8")),
                    "terminated": True,
                    "generated_tokens": len(
                        tokenizer.encode(role_wire, add_special_tokens=False)
                    )
                    + 1,
                },
                "reviewer": {
                    "actor_id": f"reviewer-{index:03d}",
                    "session_id": f"review-session-{index:03d}",
                    "prompt_sha256": _sha(
                        build_reviewer_prompt(role_candidate, role_wire).encode("utf-8")
                    ),
                    "verdict_json": verdict,
                    "verdict_sha256": _sha(verdict.encode("utf-8")),
                },
            }
            role_payload = _canonical_bytes(role)
            role_ref = _put(root, f"artifacts/roles/role-{index:03d}.json", role_payload)
        row_artifacts[index]["role_ref"] = role_ref
        review_entries.append(
            {
                "candidate_id": candidate_id,
                "split": row["split"],
                "source_type": row["source_type"],
                "source_group_id": row["source_group_id"],
                "source_repo": row["authoring_metadata"]["source_repo"],
                "source_revision": row["authoring_metadata"]["source_revision"],
                "source_tree_sha": row["authoring_metadata"]["source_tree_sha"],
                "source_path": row["authoring_metadata"]["source_path"],
                "source_sha256": row["authoring_metadata"]["source_sha256"],
                "source_license": row["source_license"],
                "source_license_sha256": row["authoring_metadata"]["path_license_sha256"],
                "path_license_sha256": row["authoring_metadata"]["path_license_sha256"],
                "license_scope_sha256": row["authoring_metadata"]["license_scope_sha256"],
                "transform_sha256": row["authoring_metadata"]["transform_sha256"],
                "session_or_commit": row["session_or_commit"],
                "task_family_id": row["task_family_id"],
                "template_id": row["template_id"],
                "state_sha256": bindings["state_sha256"],
                "action_sha256": bindings["action_sha256"],
                "after_source_sha256": bindings["after_source_sha256"],
                "history_sha256": bindings["history_sha256"],
                "near_duplicate_sha256": bindings["near_duplicate_sha256"],
                "context_sha256": row["context_sha256"],
                "human_chronology_observed": False,
                **({"history_order": row["history_order"]} if history_policy else {}),
                "accepted_training": True,
                "inferability_reviewed": True,
                "objective_verified": True,
                "wrong_action_controls_rejected": True,
                "history_leakage_check": True,
                "independent_reviewer_id": "synthetic-review-fixture",
                "oracle_result_sha256": oracle_sha,
                "fixture_sha256": fixture_sha,
                "evaluator_sha256": evaluator_sha,
                "runtime_sha256": runtime_sha,
                "role_evidence_artifact_path": role_ref["path"],
                "role_evidence_sha256": role_ref["sha256"],
                "role_evidence_bytes": role_ref["bytes"],
                **(
                    {"role_execution_proof_ref": role_proof_ref}
                    if history_policy
                    else {}
                ),
            }
        )

    results_payload = b"\n".join(_canonical_bytes(record) for record in oracle_records) + b"\n"
    diagnostics_payload = b"\n".join(_canonical_bytes(item) for item in diagnostics) + b"\n"
    oracle_ref = _put(root, "artifacts/oracle-results.jsonl", results_payload)
    diagnostics_ref = _put(root, "artifacts/oracle-diagnostics.jsonl", diagnostics_payload)
    source_rows_payload = b"\n".join(_canonical_bytes(item) for item in source_audit_rows) + b"\n"
    source_rows_ref = _put(
        root, "artifacts/source-verification-candidates.jsonl", source_rows_payload
    )
    source_manifest = {
        "schema": "commit-sequence-source-verification-result-v1",
        "status": "complete",
        "accepted_training": 0,
        "chronology_observed": False,
        "candidate_results_sha256": source_rows_ref["sha256"],
    }
    source_manifest_ref = _put(
        root,
        "artifacts/source-verification-manifest.json",
        _canonical_bytes(source_manifest),
    )
    provenance = {
        "schema": "private-public-source-provenance-manifest-v2",
        "training_ready": False,
        "accepted_training_count": 0,
        "source_verification_manifest": source_manifest_ref,
        "source_verification_candidate_rows": source_rows_ref,
        "source_task_spec": spec_ref,
        "private_objective_fixtures": fixtures_ref,
        "candidates": provenance_candidates,
    }
    provenance_ref = _put(
        root,
        "artifacts/source-provenance.json",
        _canonical_bytes(provenance),
    )
    review = {
        "schema": (
            "one-line-license-mixed-history-independent-review-v2"
            if history_policy
            else "one-line-license-mixed-independent-review-v1"
        ),
        "full_split_audit_complete": True,
        "independent_reviewer_id": "synthetic-review-fixture",
        "rows": review_entries,
    }
    review_ref = _put(root, "artifacts/independent-review.json", _canonical_bytes(review))
    train_rows = [row for row in rows if row["split"] == "train"]
    development_rows = [row for row in rows if row["split"] == "development"]
    manifest: dict[str, Any] = {
        "schema": policy.data_schema,
        "dataset_id": policy.dataset_id,
        "dataset_license": policy.dataset_license,
        "source_file_license_status": policy.file_license_status,
        "train_sha256": _sha(_canonical_bytes(train_rows)),
        "development_sha256": _sha(_canonical_bytes(development_rows)),
        "independent_review_sha256": review_ref["sha256"],
        "independent_review_path": review_ref["path"],
        "independent_review_bytes": review_ref["bytes"],
        "train_count": len(train_rows),
        "dev_count": len(development_rows),
        "file_groups_disjoint": True,
        "candidate_split_counts": {"train": len(train_rows), "development": len(development_rows)},
        "artifact_root": ".",
        "oracle_results_path": oracle_ref["path"],
        "oracle_results_sha256": oracle_ref["sha256"],
        "oracle_results_bytes": oracle_ref["bytes"],
        "oracle_evaluator_sha256": evaluator_sha,
        "oracle_diagnostics_path": diagnostics_ref["path"],
        "oracle_diagnostics_sha256": diagnostics_ref["sha256"],
        "oracle_diagnostics_bytes": diagnostics_ref["bytes"],
        "provenance_manifest_path": provenance_ref["path"],
        "provenance_manifest_sha256": provenance_ref["sha256"],
        "provenance_manifest_bytes": provenance_ref["bytes"],
    }
    return manifest, cast(list[Mapping[str, Any]], rows)


def _rewrite_review(root: Path, manifest: dict[str, Any], review: dict[str, Any]) -> None:
    payload = _canonical_bytes(review)
    (root / manifest["independent_review_path"]).write_bytes(payload)
    manifest["independent_review_sha256"] = _sha(payload)
    manifest["independent_review_bytes"] = len(payload)


def _rewrite_history_oracle_variant(
    root: Path,
    manifest: dict[str, Any],
    *,
    candidate_id: str,
    variant_name: str,
    variant_updates: Mapping[str, Any],
    test_status: str | None = None,
) -> None:
    result_path = root / manifest["oracle_results_path"]
    result_rows = [json.loads(line) for line in result_path.read_bytes().splitlines()]
    result = next(item for item in result_rows if item["candidate_id"] == candidate_id)
    variant = result["variants"][variant_name]
    variant.update(variant_updates)

    diagnostic_path = root / manifest["oracle_diagnostics_path"]
    diagnostic_rows = [json.loads(line) for line in diagnostic_path.read_bytes().splitlines()]
    diagnostic = next(
        item
        for item in diagnostic_rows
        if item["candidate_id"] == candidate_id and item["variant"] == variant_name
    )
    if test_status is not None:
        diagnostic["test_status"] = test_status
        diagnostic["checks"]["test"]["status"] = test_status
        diagnostic["checks"]["test"]["returncode"] = 0 if test_status == "pass" else 1
        variant["test_status"] = test_status
        variant["functional_status"] = "pass" if test_status == "pass" else "fail"
        variant["diagnostic_sha256"] = _sha(_canonical_bytes(diagnostic))
        diagnostic_payload = b"\n".join(_canonical_bytes(item) for item in diagnostic_rows) + b"\n"
        diagnostic_path.write_bytes(diagnostic_payload)
        manifest["oracle_diagnostics_sha256"] = _sha(diagnostic_payload)
        manifest["oracle_diagnostics_bytes"] = len(diagnostic_payload)

    result_payload = b"\n".join(_canonical_bytes(item) for item in result_rows) + b"\n"
    result_path.write_bytes(result_payload)
    manifest["oracle_results_sha256"] = _sha(result_payload)
    manifest["oracle_results_bytes"] = len(result_payload)
    review = json.loads((root / manifest["independent_review_path"]).read_bytes())
    review_entry = next(item for item in review["rows"] if item["candidate_id"] == candidate_id)
    review_entry["oracle_result_sha256"] = _sha(_canonical_bytes(result))
    _rewrite_review(root, manifest, review)


def test_full_review_accepts_relocated_package_and_rejects_unproved_flags(
    tmp_path: Path,
) -> None:
    manifest, rows = _full_review_package(tmp_path / "controller-package")
    encoded_inputs = json.dumps({"manifest": manifest, "rows": rows}, sort_keys=True)
    assert str(tmp_path / "controller-package") not in encoded_inputs
    staged = tmp_path / "staged-package"
    shutil.copytree(tmp_path / "controller-package", staged)
    verified = validate_license_mixed_review(
        manifest, rows, tokenizer=ByteTokenizer(), package_root=staged
    )
    assert verified["verified_rows"] == 192
    assert verified["quality_evidence"] is False

    missing_role_root = tmp_path / "forged-review-missing-role"
    shutil.copytree(staged, missing_role_root)
    missing_role_manifest = json.loads(json.dumps(manifest))
    review_path = missing_role_root / missing_role_manifest["independent_review_path"]
    review = json.loads(review_path.read_bytes())
    review["rows"][0]["role_evidence_artifact_path"] = None
    review["rows"][0]["role_evidence_sha256"] = None
    review["rows"][0]["role_evidence_bytes"] = None
    _rewrite_review(missing_role_root, missing_role_manifest, review)
    with pytest.raises(ValueError, match="role evidence is unpinned"):
        validate_license_mixed_review(
            missing_role_manifest,
            rows,
            tokenizer=ByteTokenizer(),
            package_root=missing_role_root,
        )

    bad_control_root = tmp_path / "forged-review-passing-control"
    shutil.copytree(staged, bad_control_root)
    bad_control_manifest = json.loads(json.dumps(manifest))
    result_path = bad_control_root / bad_control_manifest["oracle_results_path"]
    result_rows = [json.loads(line) for line in result_path.read_bytes().splitlines()]
    target_id = rows[0]["candidate_id"]
    target_result = next(record for record in result_rows if record["candidate_id"] == target_id)
    target_variant = target_result["variants"]["behavior_breaking"]
    diagnostic_path = bad_control_root / bad_control_manifest["oracle_diagnostics_path"]
    diagnostic_rows = [json.loads(line) for line in diagnostic_path.read_bytes().splitlines()]
    target_diagnostic = next(
        item
        for item in diagnostic_rows
        if item["candidate_id"] == target_id and item["variant"] == "behavior_breaking"
    )
    target_diagnostic["test_status"] = "pass"
    target_diagnostic["checks"]["test"]["status"] = "pass"
    target_diagnostic["checks"]["test"]["returncode"] = 0
    target_variant["test_status"] = "pass"
    target_variant["functional_status"] = "pass"
    target_variant["diagnostic_sha256"] = _sha(_canonical_bytes(target_diagnostic))
    result_payload = b"\n".join(_canonical_bytes(record) for record in result_rows) + b"\n"
    diagnostic_payload = b"\n".join(_canonical_bytes(item) for item in diagnostic_rows) + b"\n"
    result_path.write_bytes(result_payload)
    diagnostic_path.write_bytes(diagnostic_payload)
    bad_control_manifest.update(
        {
            "oracle_results_sha256": _sha(result_payload),
            "oracle_results_bytes": len(result_payload),
            "oracle_diagnostics_sha256": _sha(diagnostic_payload),
            "oracle_diagnostics_bytes": len(diagnostic_payload),
        }
    )
    review_path = bad_control_root / bad_control_manifest["independent_review_path"]
    review = json.loads(review_path.read_bytes())
    review_entry = next(item for item in review["rows"] if item["candidate_id"] == target_id)
    review_entry["oracle_result_sha256"] = _sha(_canonical_bytes(target_result))
    _rewrite_review(bad_control_root, bad_control_manifest, review)
    with pytest.raises(ValueError, match="behavior-breaking control did not fail functionally"):
        validate_license_mixed_review(
            bad_control_manifest,
            rows,
            tokenizer=ByteTokenizer(),
            package_root=bad_control_root,
        )


def test_full_review_rechecks_relocated_muse_v2_package_without_flag_bypass(
    tmp_path: Path,
) -> None:
    accepted_rows_path = Path(
        os.environ.get(
            "TABCOMPLETE_MUSE_V2_ACCEPTED_ROWS",
            "/mnt/ssd/tabcomplete-product-r2/muse-acceptance-v1/packages-v6/accepted_rows.jsonl",
        )
    )
    if not accepted_rows_path.is_file():
        pytest.skip("frozen local Muse v2 acceptance package is not staged")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B-Base", local_files_only=True, trust_remote_code=True
        )
    except (ImportError, OSError, ValueError):
        pytest.skip("the pinned Qwen tokenizer is not present in the local cache")

    muse_rows = [json.loads(line) for line in accepted_rows_path.read_bytes().splitlines()]
    assert len(muse_rows) == 1
    muse_row = muse_rows[0]
    assert muse_row["source_type"] == "muse_author_public_candidate"
    assert muse_row["acceptance_package_ref"]["schema"] == (
        "one-line-muse-acceptance-package-v2"
    )
    source_package_root = Path(
        os.environ.get(
            "TABCOMPLETE_MUSE_V2_PACKAGE_ROOT", str(accepted_rows_path.parent)
        )
    )
    source_package = source_package_root / muse_row["acceptance_package_ref"]["root"]
    assert source_package.is_dir()

    controller_root = tmp_path / "controller-root"
    manifest, synthetic_rows = _full_review_package(controller_root, tokenizer=tokenizer)
    package_destination = controller_root / muse_row["acceptance_package_ref"]["root"]
    package_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_package, package_destination)

    muse_bindings = license_mixed_row_bindings(muse_row)
    review_path = controller_root / str(manifest["independent_review_path"])
    review = json.loads(review_path.read_bytes())
    reviewer_id = review["independent_reviewer_id"]
    muse_metadata = muse_row.get("authoring_metadata", {})
    assert isinstance(muse_metadata, Mapping)
    review_entry = {
        "candidate_id": muse_row["candidate_id"],
        "split": muse_row["split"],
        "source_type": muse_row["source_type"],
        "source_group_id": muse_row["source_group_id"],
        "source_repo": muse_row["source_repo"],
        "source_revision": muse_row["source_revision"],
        "source_tree_sha": muse_row.get("source_tree_sha"),
        "source_path": muse_row["source_path"],
        "source_sha256": muse_row["source_sha256"],
        "source_license": muse_row["source_license"],
        "source_license_sha256": muse_row["source_license_sha256"],
        "session_or_commit": muse_row["session_or_commit"],
        "task_family_id": muse_row["task_family_id"],
        "template_id": muse_row["template_id"],
        **muse_bindings,
        "context_sha256": muse_row["context_sha256"],
        "human_chronology_observed": False,
        # These are explicit test-only review fixtures for exercising the whole
        # policy path; this test does not create a trainable shard or claim
        # training acceptance for the private Muse row.
        "accepted_training": True,
        "inferability_reviewed": True,
        "objective_verified": True,
        "wrong_action_controls_rejected": True,
        "history_leakage_check": True,
        "independent_reviewer_id": reviewer_id,
        "acceptance_package_ref": muse_row["acceptance_package_ref"],
    }
    review["rows"].append(review_entry)
    _rewrite_review(controller_root, manifest, review)

    rows = [*synthetic_rows, muse_row]
    train_rows = [row for row in rows if row["split"] == "train"]
    development_rows = [row for row in rows if row["split"] == "development"]
    manifest.update(
        {
            "train_sha256": _sha(_canonical_bytes(train_rows)),
            "development_sha256": _sha(_canonical_bytes(development_rows)),
            "train_count": len(train_rows),
            "dev_count": len(development_rows),
            "candidate_split_counts": {
                "train": len(train_rows),
                "development": len(development_rows),
            },
        }
    )
    _rewrite_review(controller_root, manifest, review)

    staged_root = tmp_path / "relocated-root"
    shutil.copytree(controller_root, staged_root)
    assert str(controller_root) not in json.dumps(
        {"manifest": manifest, "rows": rows}, sort_keys=True
    )
    verified = validate_license_mixed_review(
        manifest, rows, tokenizer=tokenizer, package_root=staged_root
    )
    assert verified["verified_rows"] == 193
    assert verified["quality_evidence"] is False

    forged_root = tmp_path / "relocated-with-missing-role-proof"
    shutil.copytree(staged_root, forged_root)
    role_evidence = (
        forged_root
        / muse_row["acceptance_package_ref"]["root"]
        / "role_evidence.json"
    )
    role_evidence.unlink()
    with pytest.raises(ValueError, match="Muse acceptance package is not valid"):
        validate_license_mixed_review(
            manifest, rows, tokenizer=tokenizer, package_root=forged_root
        )
