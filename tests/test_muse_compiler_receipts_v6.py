from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from tinycomplete.eval.code_benchmark import evaluate_prediction
from tinycomplete.one_line.contract import EditAction, EditState, RecentEdit, apply_action
from tinycomplete.one_line.muse_acceptance_package import (
    COMPILER_CONTROLS_SCHEMA,
    COMPILER_PLAN_SCHEMA,
    _canonical,
    _compiler_controls_payload,
    _digest,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture() -> tuple[bytes, bytes, dict[str, Any], bytes, bytes, bytes, bytes, bytes]:
    source = "def compute(value):\n    total = value + 1\n    return total\n"
    state = EditState(
        file_id="synthetic/a.py",
        filetype="python",
        source=source,
        target_row=2,
        cursor_col=4,
        history=(RecentEdit(row=1, old_text="total", new_text="total"),),
    )
    action = EditAction(kind="replace_line", text="    return value + 1")
    candidate = {
        "id": "muse-author/synthetic-compiler-case",
        "state": asdict(state),
        "action": asdict(action),
        "after_source": apply_action(state, action),
        "source_repo": "public/example",
        "provenance": {
            "source_id": "synthetic-a",
            "author_actor_id": "Muse:author",
        },
    }
    source_row = {
        "id": "synthetic/a.py",
        "student_state_seed": {
            "file_id": state.file_id,
            "filetype": state.filetype,
            "source": source,
        },
        "authoring_metadata": {
            "source_repo": "public/example",
            "source_sha256": _sha(source.encode()),
            "license_sha256": "c" * 64,
        },
    }
    image = "python@sha256:" + "a" * 64
    objective = {
        "schema": "one-line-independent-objective-v1",
        "reviewer_id": "independent-objective",
        "entries": [
            {
                "candidate_id": candidate["id"],
                "source_id": "synthetic-a",
                "kind": "sandbox_test",
                "path": "solution.py",
                "check": {
                    "container_image": image,
                    "test": ["python", "oracle.py"],
                    "files": {"oracle.py": "assert True"},
                    "timeout_seconds": 10.0,
                },
            }
        ],
    }
    objective_bytes = _canonical(objective)
    split_bytes = _canonical({"schema": "test-split"})
    license_bytes = _canonical({"schema": "test-license"})
    role = {"schema": "one-line-role-evidence-frozen-v8-v1", "packet": {"case_id": "synthetic-a"}}
    role_bytes = _canonical(role)
    wrong = {"name": "no_edit", "action": {"kind": "keep"}}
    controls = [
        {
            "case_id": "synthetic-a",
            "case_action_id": "synthetic-a:gold",
            "action_sha256": _digest(asdict(action)),
            "complete_source_sha256": _sha(candidate["after_source"].encode()),
        },
        {
            "case_id": "synthetic-a",
            "case_action_id": "synthetic-a:wrong:no_edit",
            "action_sha256": _digest({"kind": "keep"}),
            "complete_source_sha256": _sha(source.encode()),
        },
    ]
    preflight = {"cases": controls}
    fixture = {
        "case_id": "synthetic-a",
        "gold_action": asdict(action),
        "wrong_controls": [wrong],
    }
    packet = role["packet"]
    semantic_bytes = _canonical(
        {
            "schema": "one-line-frozen-v8-semantic-controls-v1",
            "candidate_id": candidate["id"],
            "packet_binding": packet,
            "oracle_fixtures_jsonl": json.dumps(fixture, sort_keys=True) + "\n",
            "preflight_result_json": json.dumps(preflight, sort_keys=True),
            "selected_control_case_ids": [row["case_action_id"] for row in controls],
        }
    )
    return (
        _canonical(candidate),
        _canonical(source_row),
        candidate,
        objective_bytes,
        split_bytes,
        role_bytes,
        license_bytes,
        semantic_bytes,
    )


def _bound_records() -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]:
    (
        candidate_bytes,
        source_bytes,
        candidate,
        objective_bytes,
        split_bytes,
        role_bytes,
        license_bytes,
        semantic_bytes,
    ) = _fixture()
    candidate = json.loads(candidate_bytes)
    source = json.loads(source_bytes)
    semantic = json.loads(semantic_bytes)
    preflight = json.loads(semantic["preflight_result_json"])
    objective = json.loads(objective_bytes)
    image = objective["entries"][0]["check"]["container_image"]
    runtime = {
        "kind": "sandbox_container",
        "identity": image,
        "identity_sha256": _sha(image.encode()),
    }
    packet = semantic["packet_binding"]
    identities = {
        "candidate_sha256": _sha(candidate_bytes),
        "source_row_sha256": _sha(source_bytes),
        "objective_manifest_sha256": _sha(objective_bytes),
        "split_manifest_sha256": _sha(split_bytes),
        "role_evidence_sha256": _sha(role_bytes),
        "file_license_review_sha256": _sha(license_bytes),
        "semantic_control_sha256": _sha(semantic_bytes),
        "source_sha256": source["authoring_metadata"]["source_sha256"],
        "license_sha256": source["authoring_metadata"]["license_sha256"],
    }
    controls = []
    for index, row in enumerate(preflight["cases"]):
        controls.append(
            {
                "case_action_id": row["case_action_id"],
                "action_sha256": row["action_sha256"],
                "canonical_action_sha256": _digest(
                    asdict(
                        EditAction(
                            **(
                                {"kind": "replace_line", "text": "    return value + 1"}
                                if row["case_action_id"].endswith(":gold")
                                else {"kind": "keep"}
                            )
                        )
                    )
                ),
                "after_source_sha256": row["complete_source_sha256"],
                "expected_test_status": "pass" if index == 0 else "fail",
            }
        )
    code_sha256 = {
        "qualification_runner": _sha(
            (
                Path(__file__).resolve().parents[1]
                / "scripts/qualify_frozen_v8_muse_compiler_v6.py"
            ).read_bytes()
        ),
        "package_verifier": _sha(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("src/tinycomplete/one_line/muse_acceptance_package.py")
            .read_bytes()
        ),
        "candidate_acceptance": _sha(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("src/tinycomplete/one_line/candidate_acceptance.py")
            .read_bytes()
        ),
        "code_benchmark": _sha(Path(evaluate_prediction.__code__.co_filename).read_bytes()),
    }
    compiler = {
        "language": "python",
        "command": ["python", "-m", "py_compile", "solution.py"],
        "target_path": "solution.py",
        "runtime": runtime,
    }
    plan_body = {
        "schema": COMPILER_PLAN_SCHEMA,
        "status": "frozen_before_compile",
        "candidate_id": candidate["id"],
        "packet_binding": packet,
        "identities": identities,
        "compiler": compiler,
        "controls": controls,
        "evaluator_source_sha256": code_sha256["code_benchmark"],
        "code_sha256": code_sha256,
    }
    plan = {**plan_body, "plan_sha256": _digest(plan_body)}
    plan_bytes = _canonical(plan)
    receipts = []
    for control in controls:
        receipts.append(
            {
                **control,
                "parse_status": "pass",
                "compile_configured": True,
                "compile_status": "pass",
                "compile_returncode": 0,
                "test_configured": True,
                "test_status": control["expected_test_status"],
                "test_returncode": 0 if control["expected_test_status"] == "pass" else 1,
                "compile_stdout_sha256": "d" * 64,
                "compile_stderr_sha256": "e" * 64,
                "test_stdout_sha256": "f" * 64,
                "test_stderr_sha256": "1" * 64,
                "working_tree_sha256": "2" * 64,
                "runtime": runtime,
                "evaluator_source_sha256": code_sha256["code_benchmark"],
                "observability": {
                    "campaign_id": "campaign-test",
                    "run_id": "run-test",
                    "run_attempt_id": "attempt-test",
                    "case_id": control["case_action_id"],
                    "case_attempt_id": "case-attempt-test",
                    "request_id": "request-test-" + str(len(receipts)),
                },
            }
        )
    result_body = {
        "schema": COMPILER_CONTROLS_SCHEMA,
        "candidate_id": candidate["id"],
        "plan_file_sha256": _sha(plan_bytes),
        "plan_sha256": plan["plan_sha256"],
        "packet_binding": packet,
        "identities": identities,
        "compiler": compiler,
        "evaluator_source_sha256": code_sha256["code_benchmark"],
        "controls": receipts,
        "provider_calls": 0,
        "training_started": False,
    }
    result = {**result_body, "result_sha256": _digest(result_body)}
    result_bytes = _canonical(result)
    return (
        plan,
        result,
        (
            plan_bytes,
            result_bytes,
            candidate,
            source,
            objective_bytes,
            split_bytes,
            role_bytes,
            license_bytes,
            semantic_bytes,
        ),
    )


def test_v6_compile_receipts_bind_configured_pass_and_semantic_failures() -> None:
    _plan, _result, args = _bound_records()
    (
        plan_bytes,
        result_bytes,
        candidate,
        source,
        objective,
        split,
        role,
        license_review,
        semantic,
    ) = args
    returned = _compiler_controls_payload(
        plan_bytes,
        result_bytes,
        candidate=candidate,
        source_row=source,
        objective_bytes=objective,
        split_bytes=split,
        role_bytes=role,
        license_review_bytes=license_review,
        semantic_bytes=semantic,
    )
    assert returned[0] == plan_bytes
    assert returned[1] == result_bytes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("compile_status", "fail"),
        ("compile_configured", False),
        ("test_status", "pass"),
        ("compile_returncode", 1),
    ],
)
def test_v6_rejects_failed_or_unconfigured_compile_and_controls(field: str, value: Any) -> None:
    _plan, result, args = _bound_records()
    (
        plan_bytes,
        _result_bytes,
        candidate,
        source,
        objective,
        split,
        role,
        license_review,
        semantic,
    ) = args
    row = result["controls"][1 if field == "test_status" else 0]
    row[field] = value
    result_body = dict(result)
    result_body.pop("result_sha256")
    result["result_sha256"] = _digest(result_body)
    with pytest.raises(ValueError):
        _compiler_controls_payload(
            plan_bytes,
            _canonical(result),
            candidate=candidate,
            source_row=source,
            objective_bytes=objective,
            split_bytes=split,
            role_bytes=role,
            license_review_bytes=license_review,
            semantic_bytes=semantic,
        )
