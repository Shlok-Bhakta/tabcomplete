"""Portable evidence packages for accepted public-source Muse candidates.

CPU export performs the independent objective and blind-solver checks. The
training-side reader verifies only the sealed bytes, bindings, and recorded
outcomes; it never invokes a provider, compiler, OCI runtime, or sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from tinycomplete.eval.code_benchmark import evaluate_prediction
from tinycomplete.one_line import candidate_acceptance
from tinycomplete.one_line.candidate_acceptance import (
    AcceptanceDecision,
    _check_source,
    _objective_record,
    _role_evidence_verified,
    check_candidate_acceptance,
)
from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION, serialize_state_bounded
from tinycomplete.one_line.contract import EditAction, EditState, apply_action, decode_action
from tinycomplete.one_line.data import replay_replacement_history, validate_splits
from tinycomplete.one_line.pilot_roles import build_blind_solver_prompt

PACKAGE_SCHEMA = "one-line-muse-acceptance-package-v1"
PACKAGE_SCHEMA_V2 = "one-line-muse-acceptance-package-v2"
ACCEPTANCE_SCHEMA = "one-line-muse-acceptance-result-v1"
ACCEPTANCE_SCHEMA_V2 = "one-line-muse-acceptance-result-v2"
SOLVER_FUNCTIONAL_SCHEMA = "one-line-solver-functional-oracle-v1"
COMPILER_PLAN_SCHEMA = "frozen-v8-muse-compiler-qualification-plan-v6"
COMPILER_CONTROLS_SCHEMA = "one-line-frozen-v8-compiler-controls-v1"
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_FILE_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_PACKAGE_FILES = frozenset(
    {
        "candidate.json",
        "source_row.json",
        "objective_manifest.json",
        "split_manifest.json",
        "role_evidence.json",
        "file_license_review.json",
        "acceptance_result.json",
        "solver_functional.json",
        "semantic_controls.json",
    }
)
_PACKAGE_FILES_V2 = _PACKAGE_FILES | frozenset(
    {"compiler_qualification_plan.json", "compiler_controls.json"}
)


def objective_sandbox_path(filetype: str) -> str:
    """Return the fixture target path used by the frozen local oracle runner."""
    paths = {"python": "solution.py", "typescript": "tooltip-view.ts"}
    try:
        return paths[filetype]
    except KeyError:
        raise ValueError("unsupported fixture language for objective sandbox") from None


@dataclass(frozen=True)
class PackageExportResult:
    accepted: bool
    reason: str
    acceptance_package_ref: dict[str, str] | None
    evidence: dict[str, Any]
    candidate_row: dict[str, Any] | None = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("duplicate package JSON key")
        parsed[key] = value
    return parsed


def _json_bytes(payload: bytes, *, label: str, maximum: int = MAX_ARTIFACT_BYTES) -> Any:
    if not payload or len(payload) > maximum:
        raise ValueError(f"{label} artifact is absent or unbounded")
    try:
        return json.loads(payload, object_pairs_hook=_strict_pairs)
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError(f"{label} artifact is not strict JSON") from None


def _checked_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} artifact is not a regular file")
    status = path.stat()
    if status.st_size > MAX_ARTIFACT_BYTES or stat.S_IMODE(status.st_mode) & 0o077:
        raise ValueError(f"{label} artifact is not private or exceeds its limit")
    payload = path.read_bytes()
    if len(payload) != status.st_size:
        raise ValueError(f"{label} artifact changed while reading")
    return payload


def _ensure_private_directory(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise ValueError("package directory cannot be a symlink")
    if not path.exists():
        if not create:
            raise ValueError("package directory is absent")
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    status = path.stat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError("package directory is not owner-only")


def _load_package_file(directory: Path, filename: str, manifest: Mapping[str, Any]) -> bytes:
    files = manifest.get("files")
    schema = manifest.get("schema")
    inventory = (
        _PACKAGE_FILES
        if schema == PACKAGE_SCHEMA
        else (_PACKAGE_FILES_V2 if schema == PACKAGE_SCHEMA_V2 else frozenset())
    )
    if not isinstance(files, dict) or set(files) != inventory:
        raise ValueError("package file inventory is incomplete")
    descriptor = files.get(filename)
    if not isinstance(descriptor, dict) or set(descriptor) != {"sha256", "bytes"}:
        raise ValueError("package file descriptor is invalid")
    expected_hash, expected_size = descriptor["sha256"], descriptor["bytes"]
    if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
        raise ValueError("package file hash is invalid")
    if type(expected_size) is not int or not 0 < expected_size <= MAX_ARTIFACT_BYTES:
        raise ValueError("package file size is invalid")
    payload = _checked_file(directory / filename, label=filename)
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("package file bytes do not match the manifest")
    return payload


def _compiler_controls_payload(
    plan_bytes: bytes | None,
    controls_bytes: bytes | None,
    *,
    candidate: Mapping[str, Any],
    source_row: Mapping[str, Any],
    objective_bytes: bytes,
    split_bytes: bytes,
    role_bytes: bytes,
    license_review_bytes: bytes,
    semantic_bytes: bytes,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Validate the frozen v6 compile/test receipts without executing code."""
    if plan_bytes is None or controls_bytes is None:
        raise ValueError("frozen v8 roles require compiler qualification receipts")
    plan = _json_bytes(plan_bytes, label="compiler qualification plan")
    result = _json_bytes(controls_bytes, label="compiler controls")
    role = _json_bytes(role_bytes, label="role evidence")
    semantic = _json_bytes(semantic_bytes, label="semantic controls")
    if not all(isinstance(value, dict) for value in (plan, result, role, semantic)):
        raise ValueError("compiler qualification records must be JSON objects")
    claimed_plan = plan.pop("plan_sha256", None)
    if claimed_plan != _digest(plan):
        raise ValueError("compiler qualification plan self-hash mismatch")
    plan["plan_sha256"] = claimed_plan
    if plan.get("schema") != COMPILER_PLAN_SCHEMA or plan.get("status") != "frozen_before_compile":
        raise ValueError("compiler qualification plan is not frozen v6")
    if plan.get("candidate_id") != candidate.get("id"):
        raise ValueError("compiler qualification candidate identity mismatch")
    packet = role.get("packet")
    if not isinstance(packet, dict) or plan.get("packet_binding") != packet:
        raise ValueError("compiler qualification is not bound to the frozen role packet")
    source_metadata = _source_fields(source_row)
    expected_identities = {
        "candidate_sha256": hashlib.sha256(_canonical(dict(candidate))).hexdigest(),
        "source_row_sha256": hashlib.sha256(_canonical(dict(source_row))).hexdigest(),
        "objective_manifest_sha256": hashlib.sha256(objective_bytes).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "role_evidence_sha256": hashlib.sha256(role_bytes).hexdigest(),
        "file_license_review_sha256": hashlib.sha256(license_review_bytes).hexdigest(),
        "semantic_control_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
        "source_sha256": source_metadata.get("source_sha256"),
        "license_sha256": source_metadata.get("license_sha256"),
    }
    if plan.get("identities") != expected_identities:
        raise ValueError("compiler qualification candidate inputs changed")
    objective = _objective_record(
        objective_bytes,
        hashlib.sha256(objective_bytes).hexdigest(),
        str(candidate["id"]),
        str(candidate["provenance"]["source_id"]),
        str(candidate["provenance"]["author_actor_id"]),
    )
    runtime = _runtime_identity(objective)
    target = objective_sandbox_path(str(candidate["state"]["filetype"]))
    expected_command = ["python", "-m", "py_compile", target]
    compiler = plan.get("compiler")
    if compiler != {
        "language": "python",
        "command": expected_command,
        "target_path": target,
        "runtime": runtime,
    }:
        raise ValueError("compiler qualification does not use the pinned Python sandbox")
    code_hashes = plan.get("code_sha256")
    if (
        not isinstance(code_hashes, dict)
        or set(code_hashes)
        != {
            "qualification_runner",
            "package_verifier",
            "candidate_acceptance",
            "code_benchmark",
        }
        or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in code_hashes.values()
        )
    ):
        raise ValueError("compiler qualification source identities are malformed")
    repo_root = Path(__file__).resolve().parents[3]
    current_code = {
        "package_verifier": Path(__file__).resolve().read_bytes(),
        "candidate_acceptance": Path(candidate_acceptance.__file__).resolve().read_bytes(),
        "code_benchmark": Path(evaluate_prediction.__code__.co_filename).resolve().read_bytes(),
        "qualification_runner": (
            repo_root / "scripts/qualify_frozen_v8_muse_compiler_v6.py"
        ).read_bytes(),
    }
    if any(
        hashlib.sha256(current_code[name]).hexdigest() != code_hashes[name] for name in code_hashes
    ):
        raise ValueError("compiler qualification source code changed")
    if plan.get("evaluator_source_sha256") != code_hashes["code_benchmark"]:
        raise ValueError("compiler qualification evaluator identity is inconsistent")
    if semantic.get("schema") != "one-line-frozen-v8-semantic-controls-v1":
        raise ValueError("compiler qualification needs frozen v8 semantic controls")
    preflight = json.loads(semantic["preflight_result_json"], object_pairs_hook=_strict_pairs)
    fixture_lines = [
        json.loads(line, object_pairs_hook=_strict_pairs)
        for line in semantic["oracle_fixtures_jsonl"].splitlines()
        if line
    ]
    case_id = packet.get("case_id")
    matching_fixtures = [row for row in fixture_lines if row.get("case_id") == case_id]
    if len(matching_fixtures) != 1 or not isinstance(preflight, dict):
        raise ValueError("compiler qualification source fixture is missing")
    fixture = matching_fixtures[0]
    result_rows = [
        row
        for row in preflight.get("cases", [])
        if isinstance(row, dict) and row.get("case_id") == case_id
    ]
    controls = plan.get("controls")
    receipts = result.get("controls")
    if not isinstance(controls, list) or not isinstance(receipts, list):
        raise ValueError("compiler control rows are absent")
    expected_ids = semantic.get("selected_control_case_ids")
    if (
        not isinstance(expected_ids, list)
        or not expected_ids
        or any(not isinstance(value, str) for value in expected_ids)
    ):
        raise ValueError("compiler qualification control IDs are malformed")
    selected = [row for row in result_rows if row.get("case_action_id") in expected_ids]
    if (
        [row.get("case_action_id") for row in selected] != expected_ids
        or len(controls) != len(selected)
        or len(receipts) != len(selected)
    ):
        raise ValueError("compiler controls do not cover the frozen semantic controls")
    expected_plan_controls: list[dict[str, Any]] = []
    wrong_by_id = {
        f"{case_id}:wrong:{row['name']}": row for row in fixture.get("wrong_controls", [])
    }
    gold = fixture.get("gold_action")
    for index, prior in enumerate(selected):
        action_id = prior.get("case_action_id")
        expected_pass = index == 0
        if expected_pass:
            action = gold
        else:
            wrong = wrong_by_id.get(str(action_id))
            action = wrong.get("action") if isinstance(wrong, dict) else None
        if not isinstance(action, dict):
            raise ValueError("compiler control action does not match the frozen fixture")
        parsed_action = EditAction(**action)
        action_sha = _digest(action)
        canonical_action_sha = _digest(asdict(parsed_action))
        after_sha = prior.get("complete_source_sha256")
        if (
            prior.get("action_sha256") != action_sha
            or not isinstance(after_sha, str)
            or not _SHA256.fullmatch(after_sha)
        ):
            raise ValueError("compiler control action/source identity differs from preflight")
        expected_plan_controls.append(
            {
                "case_action_id": action_id,
                "action_sha256": action_sha,
                "canonical_action_sha256": canonical_action_sha,
                "after_source_sha256": after_sha,
                "expected_test_status": "pass" if expected_pass else "fail",
            }
        )
    if controls != expected_plan_controls:
        raise ValueError("compiler plan controls do not bind the frozen semantic actions")
    result_self_hash = result.pop("result_sha256", None)
    if result_self_hash != _digest(result):
        raise ValueError("compiler control receipt self-hash mismatch")
    result["result_sha256"] = result_self_hash
    expected_result_identity = {
        "schema": COMPILER_CONTROLS_SCHEMA,
        "candidate_id": candidate["id"],
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_sha256": claimed_plan,
        "packet_binding": packet,
        "identities": expected_identities,
        "compiler": compiler,
        "evaluator_source_sha256": plan.get("evaluator_source_sha256"),
        "controls": [],
        "provider_calls": 0,
        "training_started": False,
    }
    if any(
        result.get(key) != value
        for key, value in expected_result_identity.items()
        if key != "controls"
    ):
        raise ValueError("compiler receipt identity differs from the qualification plan")
    sanitized_receipts: list[dict[str, Any]] = []
    for expected, receipt, planned in zip(controls, receipts, expected_plan_controls, strict=True):
        if not isinstance(receipt, dict):
            raise ValueError("compiler receipt row is malformed")
        receipt_fields = {
            "case_action_id": planned["case_action_id"],
            "action_sha256": planned["action_sha256"],
            "canonical_action_sha256": planned["canonical_action_sha256"],
            "after_source_sha256": planned["after_source_sha256"],
            "expected_test_status": planned["expected_test_status"],
            "parse_status": "pass",
            "compile_configured": True,
            "compile_status": "pass",
            "compile_returncode": 0,
            "test_configured": True,
            "test_status": planned["expected_test_status"],
            "runtime": runtime,
            "evaluator_source_sha256": plan.get("evaluator_source_sha256"),
        }
        if any(receipt.get(key) != value for key, value in receipt_fields.items()):
            raise ValueError("a frozen compile/test control did not meet its expected outcome")
        allowed_receipt_fields = set(receipt_fields) | {
            "test_returncode",
            "compile_stdout_sha256",
            "compile_stderr_sha256",
            "test_stdout_sha256",
            "test_stderr_sha256",
            "working_tree_sha256",
            "observability",
        }
        if set(receipt) != allowed_receipt_fields:
            raise ValueError("compiler receipt row contains unapproved fields")
        for key in (
            "compile_stdout_sha256",
            "compile_stderr_sha256",
            "test_stdout_sha256",
            "test_stderr_sha256",
            "working_tree_sha256",
        ):
            if not isinstance(receipt.get(key), str) or not _SHA256.fullmatch(receipt[key]):
                raise ValueError("compiler receipt output hash is invalid")
        observation = receipt.get("observability")
        if (
            not isinstance(observation, dict)
            or set(observation)
            != {
                "campaign_id",
                "run_id",
                "run_attempt_id",
                "case_id",
                "case_attempt_id",
                "request_id",
            }
            or any(not isinstance(value, str) or not value for value in observation.values())
        ):
            raise ValueError("compiler receipt is missing its run/case/request identity")
        test_returncode = receipt.get("test_returncode")
        if expected["expected_test_status"] == "pass":
            if test_returncode != 0:
                raise ValueError("gold objective control did not exit successfully")
        elif type(test_returncode) is not int or test_returncode == 0:
            raise ValueError("semantic wrong action did not fail the objective test")
        if not isinstance(receipt.get("working_tree_sha256"), str) or not _SHA256.fullmatch(
            receipt["working_tree_sha256"]
        ):
            raise ValueError("compiler receipt working-tree hash is invalid")
        sanitized_receipts.append(dict(receipt))
    expected_result_identity["controls"] = sanitized_receipts
    check_result = dict(result)
    check_result.pop("result_sha256", None)
    if check_result != expected_result_identity:
        raise ValueError("compiler receipts contain unexpected or missing fields")
    return plan_bytes, controls_bytes, {"plan": plan, "result": result}


def _source_fields(source_row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = source_row.get("authoring_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("source row has no pinned authoring metadata")
    return metadata


def _same_training_identity(
    row: Mapping[str, Any], candidate: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    metadata = _source_fields(source)
    if not candidate_acceptance._source_seed_verified(source, metadata):
        raise ValueError("accepted source seed is not bound to its public parent")
    if row.get("id") != candidate.get("id"):
        raise ValueError("training row candidate identity mismatch")
    if (
        row.get("source_type") != "muse_author_public_candidate"
        or candidate.get("source_type") != "muse_author_public_candidate"
    ):
        raise ValueError("training row source type is not Muse")
    for key in ("state", "action", "after_source"):
        if key not in row or _canonical(row[key]) != _canonical(candidate.get(key)):
            raise ValueError("training row model target differs from its accepted candidate")
    expected = {
        "source_repo": metadata.get("source_repo"),
        "source_revision": metadata.get("source_revision"),
        "source_path": metadata.get("source_path"),
        "source_license": metadata.get("source_license"),
        "source_license_sha256": metadata.get("license_sha256"),
        "source_sha256": metadata.get("source_sha256"),
    }
    for key, value in expected.items():
        if value is not None and row.get(key) != value:
            raise ValueError("training row source provenance mismatch")
    if row.get("source_repo") != candidate.get("source_repo"):
        raise ValueError("training row repository mismatch")
    if row.get("source_revision") != candidate.get("source_revision"):
        raise ValueError("training row revision mismatch")
    if candidate.get("split") != row.get("split"):
        raise ValueError("training row split mismatch")
    for key in ("source_group_id", "session_or_commit", "template_id"):
        value = row.get(key)
        if not isinstance(value, str) or not value or value != candidate.get(key):
            raise ValueError("training row grouping identity mismatch")
    candidate_family = candidate.get("task_family_id", candidate.get("generator_family"))
    row_family = row.get("task_family_id", row.get("generator_family"))
    if (
        not isinstance(candidate_family, str)
        or not candidate_family
        or row_family != candidate_family
    ):
        raise ValueError("training row task-family identity mismatch")
    if row.get("generator_family") is not None and row.get("generator_family") != candidate.get(
        "generator_family"
    ):
        raise ValueError("training row generator-family identity mismatch")
    state = EditState.from_mapping(row["state"])
    action = EditAction(**row["action"])
    if (
        replay_replacement_history(
            str(source["student_state_seed"]["source"]),
            state.history,
            file_id=state.file_id,
            filetype=state.filetype,
        )
        != state.source
    ):
        raise ValueError("accepted training history does not reconstruct the state")
    if apply_action(state, action) != row["after_source"] or row["after_source"] == state.source:
        raise ValueError("accepted training action does not reconstruct an edit")


def _solver_action(
    role_bytes: bytes,
    candidate: Mapping[str, Any],
    tokenizer: Any,
    *,
    require_candidate_match: bool = True,
) -> tuple[str, EditAction, str]:
    role = _json_bytes(role_bytes, label="role evidence")
    if not isinstance(role, dict) or not isinstance(role.get("solver"), dict):
        raise ValueError("solver role evidence is absent")
    solver = role["solver"]
    wire = solver.get("wire")
    if not isinstance(wire, str):
        raise ValueError("solver wire is absent")
    if solver.get("terminated") is not True or type(solver.get("generated_tokens")) is not int:
        raise ValueError("solver completion did not terminate explicitly")
    decoded = decode_action(
        wire,
        terminated=True,
        generated_tokens=solver["generated_tokens"],
    )
    if decoded.status != "ok" or decoded.action is None:
        raise ValueError("solver action is invalid")
    state = EditState.from_mapping(candidate["state"])
    expected_action = EditAction(**candidate["action"])
    if require_candidate_match and decoded.action != expected_action:
        raise ValueError("blind solver action differs from the canonical candidate action")
    actual_tokens = len(tokenizer.encode(wire, add_special_tokens=False)) + 1
    if actual_tokens != solver["generated_tokens"]:
        raise ValueError("solver token count changed")
    result = apply_action(state, decoded.action)
    if require_candidate_match and result != candidate["after_source"]:
        raise ValueError("blind solver action does not reproduce the accepted result")
    return wire, decoded.action, result


def _semantic_controls_payload(
    payload: bytes | None,
    *,
    expected_digest: str | None,
    candidate: Mapping[str, Any],
    role_bytes: bytes,
) -> tuple[bytes, dict[str, Any]]:
    if payload is None:
        legacy = {"schema": "one-line-semantic-controls-not-applicable-v1"}
        return _canonical(legacy), legacy
    if expected_digest is None or not _SHA256.fullmatch(expected_digest):
        raise ValueError("semantic control evidence is not pinned")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError("semantic control evidence hash mismatch")
    controls = _json_bytes(payload, label="semantic controls")
    role = _json_bytes(role_bytes, label="role evidence")
    if not isinstance(controls, dict) or not isinstance(role, dict):
        raise ValueError("semantic control evidence schema is invalid")
    if controls == {"schema": "one-line-semantic-controls-not-applicable-v1"}:
        if role.get("schema") == candidate_acceptance._ROLE_EVIDENCE_V8:
            raise ValueError("frozen v8 roles require semantic control receipts")
        return payload, controls
    if controls.get("schema") != "one-line-frozen-v8-semantic-controls-v1":
        raise ValueError("semantic control evidence schema is invalid")
    if role.get("schema") != candidate_acceptance._ROLE_EVIDENCE_V8:
        raise ValueError("semantic controls require frozen v8 roles")
    if controls.get("candidate_id") != candidate.get("id"):
        raise ValueError("semantic control candidate identity mismatch")
    if controls.get("packet_binding") != role.get("packet"):
        raise ValueError("semantic controls are not bound to the role packet")
    expected_keys = {
        "schema",
        "candidate_id",
        "packet_binding",
        "oracle_fixtures_jsonl",
        "preflight_result_json",
        "selected_case_fixture_sha256",
        "selected_control_case_ids",
    }
    if set(controls) != expected_keys:
        raise ValueError("semantic control evidence fields are invalid")
    fixture_bytes = controls.get("oracle_fixtures_jsonl")
    preflight_bytes = controls.get("preflight_result_json")
    if not isinstance(fixture_bytes, str) or not isinstance(preflight_bytes, str):
        raise ValueError("semantic control source artifacts are absent")
    if (
        hashlib.sha256(fixture_bytes.encode("utf-8")).hexdigest()
        != controls["packet_binding"]["oracle_fixtures_sha256"]
        or hashlib.sha256(preflight_bytes.encode("utf-8")).hexdigest()
        != controls["packet_binding"]["preflight_result_sha256"]
    ):
        raise ValueError("semantic control source artifact hashes differ")
    fixtures = [
        json.loads(line, object_pairs_hook=_strict_pairs)
        for line in fixture_bytes.splitlines()
        if line
    ]
    fixture_rows = [
        row
        for row in fixtures
        if isinstance(row, dict) and row.get("case_id") == controls["packet_binding"]["case_id"]
    ]
    if len(fixture_rows) != 1:
        raise ValueError("semantic control fixture is not unique")
    fixture = fixture_rows[0]
    fixture_sha = hashlib.sha256(_canonical(fixture)).hexdigest()
    if controls.get("selected_case_fixture_sha256") != fixture_sha:
        raise ValueError("semantic control fixture hash mismatch")
    state = EditState.from_mapping(candidate["state"])
    if _digest(asdict(state)) != fixture.get("state_sha256"):
        raise ValueError("semantic control state differs from the frozen fixture")
    preflight = json.loads(preflight_bytes, object_pairs_hook=_strict_pairs)
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "two-seed-oracle-preflight-v1"
        or preflight.get("status") != "complete"
        or preflight.get("provider_calls") != 0
        or preflight.get("training_started") is not False
        or preflight.get("plan_sha256") != controls["packet_binding"]["plan_canonical_sha256"]
        or preflight.get("input_sha256") != controls["packet_binding"]["inputs_sha256"]
        or preflight.get("oracle_sha256") != controls["packet_binding"]["oracle_fixtures_sha256"]
    ):
        raise ValueError("semantic control preflight identity is invalid")
    claimed = preflight.pop("result_sha256", None)
    if claimed != hashlib.sha256(_canonical(preflight)).hexdigest():
        raise ValueError("semantic control preflight self-hash mismatch")
    preflight["result_sha256"] = claimed
    result_rows = preflight.get("cases")
    if not isinstance(result_rows, list):
        raise ValueError("semantic control result rows are absent")
    selected_ids = controls.get("selected_control_case_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("semantic control selected IDs are absent")
    selected = [
        row
        for row in result_rows
        if isinstance(row, dict) and row.get("case_id") == fixture["case_id"]
    ]
    if [row.get("case_action_id") for row in selected] != selected_ids:
        raise ValueError("semantic control rows are not in frozen order")
    expected_action_ids = [
        f"{fixture['case_id']}:gold",
        *[
            f"{fixture['case_id']}:wrong:{control['name']}"
            for control in fixture.get("wrong_controls", [])
        ],
    ]
    if selected_ids != expected_action_ids or len(selected) < 3:
        raise ValueError("semantic controls do not cover all frozen positive and wrong cases")
    gold = fixture.get("gold_action")
    if not isinstance(gold, dict) or candidate.get("action") != gold:
        raise ValueError("candidate action differs from the frozen semantic gold action")
    if (
        selected[0].get("functional_status") != "pass"
        or selected[0].get("functional_expected") != "pass"
        or selected[0].get("action_sha256") != _digest(asdict(EditAction(**gold)))
        or hashlib.sha256(apply_action(state, EditAction(**gold)).encode("utf-8")).hexdigest()
        != selected[0].get("complete_source_sha256")
    ):
        raise ValueError("frozen semantic gold control did not pass")
    for row in selected[1:]:
        if (
            row.get("functional_expected") != "fail"
            or row.get("functional_status") != "fail"
            or row.get("parse_status") != "pass"
            or row.get("test_status") != "fail"
        ):
            raise ValueError(
                "a parse-valid frozen semantic wrong control did not fail the objective"
            )
    fixture_wrong = fixture.get("wrong_controls")
    if not isinstance(fixture_wrong, list) or len(fixture_wrong) != len(selected) - 1:
        raise ValueError("semantic wrong-control inventory is incomplete")
    for result_row, fixture_control in zip(selected[1:], fixture_wrong, strict=True):
        if result_row.get("action_sha256") != fixture_control.get(
            "action_sha256"
        ) or result_row.get("complete_source_sha256") == selected[0].get("complete_source_sha256"):
            raise ValueError("frozen semantic wrong-control identity mismatch")
        action_raw = fixture_control.get("action")
        if not isinstance(action_raw, dict):
            raise ValueError("frozen semantic wrong action is malformed")
        wrong_after = apply_action(state, EditAction(**action_raw))
        if hashlib.sha256(wrong_after.encode("utf-8")).hexdigest() != result_row.get(
            "complete_source_sha256"
        ):
            raise ValueError("frozen semantic wrong action does not reconstruct its receipt")
    return payload, controls


def verify_role_evidence(
    candidate: Mapping[str, Any],
    role_evidence: bytes,
    *,
    tokenizer: Any,
    author_actor_id: str | None = None,
) -> AcceptanceDecision:
    """Verify author/solver/reviewer wire evidence without executing code.

    This is also usable for independently authored synthetic public-source
    rows: it establishes a blind solver action and nonambiguous reviewer
    agreement, while leaving functional behavior to the separately pinned
    objective fixture.
    """
    try:
        provenance = candidate.get("provenance")
        if not isinstance(provenance, Mapping):
            return AcceptanceDecision(False, "role_provenance_absent", {})
        author = author_actor_id or provenance.get("author_actor_id")
        if not isinstance(author, str) or not author:
            return AcceptanceDecision(False, "role_author_identity_absent", {})
        state = EditState.from_mapping(candidate["state"])
        role = _json_bytes(role_evidence, label="role evidence")
        if not isinstance(role, dict) or not isinstance(role.get("solver"), dict):
            return AcceptanceDecision(False, "role_evidence_invalid", {})
        decoded = decode_action(
            str(role["solver"].get("wire", "")),
            terminated=role["solver"].get("terminated") is True,
            generated_tokens=role["solver"].get("generated_tokens"),
        )
        expected_action = EditAction(**candidate["action"])
        if decoded.status != "ok" or decoded.action != expected_action:
            return AcceptanceDecision(False, "role_solver_action_mismatch", {})
        if not _role_evidence_verified(
            role_evidence,
            hashlib.sha256(role_evidence).hexdigest(),
            candidate=candidate,
            state=state,
            tokenizer=tokenizer,
            author_actor_id=author,
        ):
            return AcceptanceDecision(False, "role_evidence_invalid", {})
        return AcceptanceDecision(
            True,
            "role_evidence_verified",
            {
                "candidate_id": candidate.get("id"),
                "role_evidence_sha256": hashlib.sha256(role_evidence).hexdigest(),
                "context_sha256": hashlib.sha256(
                    build_blind_solver_prompt(candidate, tokenizer).encode("utf-8")
                ).hexdigest(),
                "functional_status": "not_evaluated_by_role_verifier",
            },
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return AcceptanceDecision(False, "role_evidence_invalid", {})


def _runtime_identity(objective: Mapping[str, Any]) -> dict[str, str]:
    kind = objective.get("kind")
    if kind == "sandbox_test":
        check = objective.get("check")
        if not isinstance(check, Mapping):
            raise ValueError("sandbox objective has no pinned check")
        image = check.get("container_image")
        if not isinstance(image, str) or not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image):
            raise ValueError("sandbox runtime is not digest pinned")
        return {
            "kind": "sandbox_container",
            "identity": image,
            "identity_sha256": hashlib.sha256(image.encode()).hexdigest(),
        }
    if kind == "text_counts":
        return {
            "kind": "deterministic_text_counts",
            "identity": "candidate-acceptance-text-counts-v1",
            "identity_sha256": hashlib.sha256(b"candidate-acceptance-text-counts-v1").hexdigest(),
        }
    raise ValueError("unsupported objective runtime")


def export_muse_acceptance_package(
    *,
    package_root: Path,
    candidate: Mapping[str, Any],
    source_row: Mapping[str, Any],
    tokenizer: Any,
    objective_manifest: bytes,
    split_manifest: bytes,
    role_evidence: bytes,
    file_license_review: bytes,
    semantic_control_evidence: bytes | None = None,
    expected_semantic_control_sha256: str | None = None,
    compiler_qualification_plan: bytes | None = None,
    compiler_control_evidence: bytes | None = None,
    existing_rows: Sequence[Mapping[str, Any]] = (),
    reserved_repositories: frozenset[str] = frozenset(),
) -> PackageExportResult:
    """Run the CPU acceptance gates and write an owner-only content package.

    The solver's actual action is evaluated against the same frozen objective
    fixture as the author action. A package is written only if both actions
    functionally pass and all no-op/wrong-action controls fail.
    """
    package_candidate = dict(candidate)
    source_metadata = _source_fields(source_row)
    repo = package_candidate.get("source_repo")
    if not isinstance(repo, str) or repo != source_metadata.get("source_repo"):
        return PackageExportResult(False, "source_repository_identity_invalid", None, {})
    if package_candidate.get("source_group_id") is None:
        package_candidate["source_group_id"] = "repo:" + repo.casefold()
    if package_candidate.get("task_family_id") is None:
        package_candidate["task_family_id"] = package_candidate.get("generator_family")
    candidate = package_candidate
    author_actor = candidate.get("provenance", {}).get("author_actor_id")
    for field in ("source_group_id", "session_or_commit", "generator_family", "template_id"):
        if not isinstance(candidate.get(field), str) or not candidate[field]:
            return PackageExportResult(False, "candidate_group_identity_incomplete", None, {})
    split_sha = hashlib.sha256(split_manifest).hexdigest()
    candidate_data = _canonical(dict(candidate))
    source_data = _canonical(dict(source_row))
    try:
        record = _objective_record(
            objective_manifest,
            hashlib.sha256(objective_manifest).hexdigest(),
            str(candidate["id"]),
            str(candidate["provenance"]["source_id"]),
            str(author_actor),
        )
        if record.get("kind") not in {"sandbox_test", "text_counts"}:
            raise ValueError("objective kind is not executable by the pinned checker")
        wire, solver_action, solver_after = _solver_action(
            role_evidence, candidate, tokenizer, require_candidate_match=False
        )
        functional_pass = _check_source(solver_after, record, str(candidate["state"]["filetype"]))
        if functional_pass is not True:
            return PackageExportResult(
                False,
                "blind_solver_functional_objective_failed",
                None,
                {
                    "solver_functional_status": "fail"
                    if functional_pass is False
                    else "unavailable",
                    "solver_action_sha256": _digest(asdict(solver_action)),
                },
            )
        if solver_action != EditAction(**candidate["action"]):
            return PackageExportResult(
                False,
                "blind_solver_action_differs_from_author_action",
                None,
                {"solver_functional_status": "pass"},
            )
        semantic_payload, _semantic_record = _semantic_controls_payload(
            semantic_control_evidence,
            expected_digest=expected_semantic_control_sha256,
            candidate=candidate,
            role_bytes=role_evidence,
        )
        role_record = _json_bytes(role_evidence, label="role evidence")
        if (
            isinstance(role_record, dict)
            and role_record.get("schema") == candidate_acceptance._ROLE_EVIDENCE_V8
        ):
            _compiler_controls_payload(
                compiler_qualification_plan,
                compiler_control_evidence,
                candidate=candidate,
                source_row=source_row,
                objective_bytes=objective_manifest,
                split_bytes=split_manifest,
                role_bytes=role_evidence,
                license_review_bytes=file_license_review,
                semantic_bytes=semantic_payload,
            )
            package_schema = PACKAGE_SCHEMA_V2
            acceptance_schema = ACCEPTANCE_SCHEMA_V2
        else:
            if compiler_qualification_plan is not None or compiler_control_evidence is not None:
                raise ValueError("compiler receipts are reserved for frozen v8 role evidence")
            package_schema = PACKAGE_SCHEMA
            acceptance_schema = ACCEPTANCE_SCHEMA
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return PackageExportResult(False, "frozen_objective_or_semantic_evidence_invalid", None, {})

    decision = check_candidate_acceptance(
        candidate,
        source_row,
        tokenizer,
        objective_manifest=objective_manifest,
        expected_objective_manifest_sha256=hashlib.sha256(objective_manifest).hexdigest(),
        split_manifest=split_manifest,
        expected_split_manifest_sha256=split_sha,
        role_evidence=role_evidence,
        expected_role_evidence_sha256=hashlib.sha256(role_evidence).hexdigest(),
        file_license_review=file_license_review,
        expected_file_license_review_sha256=hashlib.sha256(file_license_review).hexdigest(),
        existing_rows=existing_rows,
        reserved_repositories=reserved_repositories,
    )
    if not decision.accepted:
        return PackageExportResult(False, decision.reason, None, decision.evidence)
    decision.evidence["semantic_control_sha256"] = hashlib.sha256(semantic_payload).hexdigest()
    try:
        if not candidate_acceptance._license_review_verified(
            file_license_review,
            hashlib.sha256(file_license_review).hexdigest(),
            source_row=source_row,
            author_actor_id=str(author_actor),
        ):
            return PackageExportResult(False, "file_license_review_invalid", None, {})
    except (AttributeError, KeyError, TypeError, ValueError):
        return PackageExportResult(False, "file_license_review_invalid", None, {})

    try:
        state = EditState.from_mapping(candidate["state"])
        context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
        if context.input_tokens is None or context.text != build_blind_solver_prompt(
            candidate, tokenizer
        ):
            raise ValueError("solver input context changed during qualification")
        evaluator_file = Path(evaluate_prediction.__code__.co_filename).resolve()
        acceptance_file = Path(candidate_acceptance.__file__).resolve()
        oracle_record_sha = hashlib.sha256(_canonical(dict(record))).hexdigest()
        functional = {
            "schema": SOLVER_FUNCTIONAL_SCHEMA,
            "candidate_id": candidate["id"],
            "source_id": candidate["provenance"]["source_id"],
            "candidate_sha256": hashlib.sha256(candidate_data).hexdigest(),
            "source_row_sha256": hashlib.sha256(source_data).hexdigest(),
            "state_sha256": _digest(asdict(state)),
            "action_sha256": _digest(asdict(solver_action)),
            "solver_wire_sha256": hashlib.sha256(wire.encode("utf-8")).hexdigest(),
            "solver_after_source_sha256": hashlib.sha256(solver_after.encode("utf-8")).hexdigest(),
            "context_sha256": hashlib.sha256(context.text.encode("utf-8")).hexdigest(),
            "context_policy": CONTEXT_POLICY_VERSION,
            "objective_manifest_sha256": hashlib.sha256(objective_manifest).hexdigest(),
            "objective_fixture_sha256": oracle_record_sha,
            "evaluator_source_sha256": hashlib.sha256(evaluator_file.read_bytes()).hexdigest(),
            "acceptance_source_sha256": hashlib.sha256(acceptance_file.read_bytes()).hexdigest(),
            "runtime": _runtime_identity(record),
            "functional_status": "pass",
            "outcome": {"gold": True, "termination": True},
        }
        if package_schema == PACKAGE_SCHEMA_V2:
            assert compiler_qualification_plan is not None and compiler_control_evidence is not None
            functional["compiler_qualification_plan_sha256"] = hashlib.sha256(
                compiler_qualification_plan
            ).hexdigest()
            functional["compiler_control_evidence_sha256"] = hashlib.sha256(
                compiler_control_evidence
            ).hexdigest()
            functional["compiler_status"] = "pass"
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return PackageExportResult(False, "solver_functional_evidence_invalid", None, {})

    acceptance = {
        "schema": acceptance_schema,
        "candidate_id": candidate["id"],
        "source_id": candidate["provenance"]["source_id"],
        "candidate_sha256": hashlib.sha256(candidate_data).hexdigest(),
        "source_row_sha256": hashlib.sha256(source_data).hexdigest(),
        "objective_manifest_sha256": hashlib.sha256(objective_manifest).hexdigest(),
        "split_manifest_sha256": split_sha,
        "role_evidence_sha256": hashlib.sha256(role_evidence).hexdigest(),
        "file_license_review_sha256": hashlib.sha256(file_license_review).hexdigest(),
        "solver_functional_sha256": hashlib.sha256(_canonical(functional)).hexdigest(),
        "semantic_control_sha256": hashlib.sha256(semantic_payload).hexdigest(),
        "decision": "accepted",
        "reason": decision.reason,
        "evidence": decision.evidence,
    }
    file_payloads = {
        "candidate.json": candidate_data,
        "source_row.json": source_data,
        "objective_manifest.json": objective_manifest,
        "split_manifest.json": split_manifest,
        "role_evidence.json": role_evidence,
        "file_license_review.json": file_license_review,
        "acceptance_result.json": _canonical(acceptance),
        "solver_functional.json": _canonical(functional),
        "semantic_controls.json": semantic_payload,
    }
    if package_schema == PACKAGE_SCHEMA_V2:
        assert compiler_qualification_plan is not None and compiler_control_evidence is not None
        file_payloads["compiler_qualification_plan.json"] = compiler_qualification_plan
        file_payloads["compiler_controls.json"] = compiler_control_evidence
        acceptance["compiler_qualification_plan_sha256"] = hashlib.sha256(
            compiler_qualification_plan
        ).hexdigest()
        acceptance["compiler_control_evidence_sha256"] = hashlib.sha256(
            compiler_control_evidence
        ).hexdigest()
        acceptance["compiler_status"] = "pass"
        file_payloads["acceptance_result.json"] = _canonical(acceptance)
    for _name, payload in file_payloads.items():
        if not payload or len(payload) > MAX_ARTIFACT_BYTES:
            return PackageExportResult(False, "acceptance_artifact_unbounded", None, {})

    package_base = package_root / "muse"
    if not package_base.exists():
        _ensure_private_directory(package_root, create=True)
        _ensure_private_directory(package_base, create=True)
    else:
        _ensure_private_directory(package_root, create=False)
        _ensure_private_directory(package_base, create=False)
    directory_name = hashlib.sha256(str(candidate["id"]).encode("utf-8")).hexdigest()[:32]
    relative_root = f"muse/{directory_name}"
    destination = package_root / relative_root
    if destination.exists() or destination.is_symlink():
        return PackageExportResult(False, "acceptance_package_already_exists", None, {})

    stage = Path(tempfile.mkdtemp(prefix=".muse-package-", dir=package_base))
    os.chmod(stage, 0o700)
    try:
        file_manifest: dict[str, dict[str, Any]] = {}
        for filename, payload in file_payloads.items():
            target = stage / filename
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            file_manifest[filename] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        package_manifest = {
            "schema": package_schema,
            "candidate_id": candidate["id"],
            "source_id": candidate["provenance"]["source_id"],
            "candidate_sha256": hashlib.sha256(candidate_data).hexdigest(),
            "source_row_sha256": hashlib.sha256(source_data).hexdigest(),
            "objective_manifest_sha256": hashlib.sha256(objective_manifest).hexdigest(),
            "split_manifest_sha256": split_sha,
            "evaluator_source_sha256": functional["evaluator_source_sha256"],
            "acceptance_source_sha256": functional["acceptance_source_sha256"],
            "runtime": functional["runtime"],
            "semantic_control_sha256": hashlib.sha256(semantic_payload).hexdigest(),
            "files": file_manifest,
        }
        if package_schema == PACKAGE_SCHEMA_V2:
            package_manifest["compiler_qualification_plan_sha256"] = hashlib.sha256(
                compiler_qualification_plan or b""
            ).hexdigest()
            package_manifest["compiler_control_evidence_sha256"] = hashlib.sha256(
                compiler_control_evidence or b""
            ).hexdigest()
            package_manifest["compiler_status"] = "pass"
        manifest_data = _canonical(package_manifest)
        manifest_path = stage / "manifest.json"
        descriptor = os.open(
            manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(manifest_data)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(stage, destination)
    finally:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
    reference = {
        "schema": package_schema,
        "root": relative_root,
        "candidate_id": str(candidate["id"]),
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_data).hexdigest(),
        "source_row_sha256": hashlib.sha256(source_data).hexdigest(),
    }
    return PackageExportResult(
        True, decision.reason, reference, acceptance["evidence"], dict(candidate)
    )


def _read_reference(row: Mapping[str, Any]) -> Mapping[str, str]:
    reference = row.get("acceptance_package_ref")
    expected_fields = {
        "schema",
        "root",
        "candidate_id",
        "manifest_sha256",
        "candidate_sha256",
        "source_row_sha256",
    }
    if not isinstance(reference, Mapping) or set(reference) != expected_fields:
        raise ValueError("Muse acceptance package reference is invalid")
    if reference.get("schema") not in {PACKAGE_SCHEMA, PACKAGE_SCHEMA_V2} or reference.get(
        "candidate_id"
    ) != row.get("id"):
        raise ValueError("Muse acceptance package identity mismatch")
    for key in ("manifest_sha256", "candidate_sha256", "source_row_sha256"):
        if not isinstance(reference.get(key), str) or not _SHA256.fullmatch(reference[key]):
            raise ValueError("Muse acceptance package hash reference is invalid")
    relative = PurePosixPath(str(reference.get("root", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Muse acceptance package path is invalid")
    return cast(Mapping[str, str], reference)


def verify_muse_candidate_row(
    row: Mapping[str, Any],
    *,
    package_root: Path,
    tokenizer: Any,
    existing_rows: Sequence[Mapping[str, Any]] = (),
    reserved_repositories: frozenset[str] = frozenset(),
) -> AcceptanceDecision:
    """Verify a frozen CPU-qualified package without rerunning its sandbox.

    This is the trainer-side path. It parses and cross-checks all pinned records,
    reconstructs the state/action/context, and validates the recorded oracle
    outcomes and evaluator/runtime identities. It performs no compilation,
    subprocess, network, provider, or OCI work.
    """
    try:
        if row.get("source_type") != "muse_author_public_candidate":
            return AcceptanceDecision(False, "source_type_not_muse", {})
        reference = _read_reference(row)
        base = package_root.resolve(strict=True)
        package_directory = base.joinpath(*PurePosixPath(reference["root"]).parts)
        current = base
        for part in PurePosixPath(reference["root"]).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("package path contains a symlink")
        resolved = package_directory.resolve(strict=True)
        if not resolved.is_relative_to(base):
            raise ValueError("package path escapes the configured package root")
        _ensure_private_directory(resolved, create=False)
        manifest_path = resolved / "manifest.json"
        manifest_bytes = _checked_file(manifest_path, label="package manifest")
        if hashlib.sha256(manifest_bytes).hexdigest() != reference["manifest_sha256"]:
            raise ValueError("package manifest hash mismatch")
        manifest = _json_bytes(manifest_bytes, label="package manifest")
        if not isinstance(manifest, dict):
            raise ValueError("package manifest is not an object")
        manifest_schema = manifest.get("schema")
        expected_manifest_fields = {
            "schema",
            "candidate_id",
            "source_id",
            "candidate_sha256",
            "source_row_sha256",
            "objective_manifest_sha256",
            "split_manifest_sha256",
            "evaluator_source_sha256",
            "acceptance_source_sha256",
            "runtime",
            "semantic_control_sha256",
            "files",
        }
        if manifest_schema == PACKAGE_SCHEMA_V2:
            expected_manifest_fields |= {
                "compiler_qualification_plan_sha256",
                "compiler_control_evidence_sha256",
                "compiler_status",
            }
        if (
            set(manifest) != expected_manifest_fields
            or manifest_schema not in {PACKAGE_SCHEMA, PACKAGE_SCHEMA_V2}
            or manifest.get("candidate_id") != row.get("id")
        ):
            raise ValueError("package manifest identity is invalid")
        payload_inventory = (
            _PACKAGE_FILES if manifest_schema == PACKAGE_SCHEMA else _PACKAGE_FILES_V2
        )
        if set(manifest.get("files", {})) != payload_inventory:
            raise ValueError("package file inventory does not match its schema")
        payloads = {
            name: _load_package_file(resolved, name, manifest) for name in payload_inventory
        }
        if (
            hashlib.sha256(payloads["candidate.json"]).hexdigest() != reference["candidate_sha256"]
            or hashlib.sha256(payloads["source_row.json"]).hexdigest()
            != reference["source_row_sha256"]
            or manifest.get("candidate_sha256") != reference["candidate_sha256"]
            or manifest.get("source_row_sha256") != reference["source_row_sha256"]
        ):
            raise ValueError("package candidate/source binding mismatch")
        candidate = _json_bytes(payloads["candidate.json"], label="candidate")
        source = _json_bytes(payloads["source_row.json"], label="source row")
        objective = _json_bytes(payloads["objective_manifest.json"], label="objective manifest")
        split = _json_bytes(payloads["split_manifest.json"], label="split manifest")
        role = _json_bytes(payloads["role_evidence.json"], label="role evidence")
        license_review = _json_bytes(
            payloads["file_license_review.json"], label="file-license review"
        )
        accepted = _json_bytes(payloads["acceptance_result.json"], label="acceptance result")
        functional = _json_bytes(payloads["solver_functional.json"], label="solver outcome")
        if not all(
            isinstance(value, dict)
            for value in (
                candidate,
                source,
                objective,
                split,
                role,
                license_review,
                accepted,
                functional,
            )
        ):
            raise ValueError("package contains a non-object JSON record")
        _same_training_identity(row, candidate, source)
        state = EditState.from_mapping(candidate["state"])
        actor = candidate.get("provenance", {}).get("author_actor_id")
        objective_bytes, split_bytes = (
            payloads["objective_manifest.json"],
            payloads["split_manifest.json"],
        )
        role_bytes, license_bytes = (
            payloads["role_evidence.json"],
            payloads["file_license_review.json"],
        )
        objective_sha = hashlib.sha256(objective_bytes).hexdigest()
        split_sha = hashlib.sha256(split_bytes).hexdigest()
        role_sha = hashlib.sha256(role_bytes).hexdigest()
        license_sha = hashlib.sha256(license_bytes).hexdigest()
        semantic_sha = hashlib.sha256(payloads["semantic_controls.json"]).hexdigest()
        compiler_plan_sha: str | None = None
        compiler_controls_sha: str | None = None
        if manifest_schema == PACKAGE_SCHEMA_V2:
            compiler_plan_sha = hashlib.sha256(
                payloads["compiler_qualification_plan.json"]
            ).hexdigest()
            compiler_controls_sha = hashlib.sha256(payloads["compiler_controls.json"]).hexdigest()
            _compiler_controls_payload(
                payloads["compiler_qualification_plan.json"],
                payloads["compiler_controls.json"],
                candidate=candidate,
                source_row=source,
                objective_bytes=payloads["objective_manifest.json"],
                split_bytes=payloads["split_manifest.json"],
                role_bytes=role_bytes,
                license_review_bytes=license_bytes,
                semantic_bytes=payloads["semantic_controls.json"],
            )
            if (
                manifest.get("compiler_qualification_plan_sha256") != compiler_plan_sha
                or manifest.get("compiler_control_evidence_sha256") != compiler_controls_sha
                or manifest.get("compiler_status") != "pass"
            ):
                raise ValueError("package compiler receipt hash or status mismatch")
        elif role.get("schema") == candidate_acceptance._ROLE_EVIDENCE_V8:
            raise ValueError("frozen v8 Muse package is missing required compiler receipts")
        if (
            objective_sha != manifest.get("objective_manifest_sha256")
            or split_sha != manifest.get("split_manifest_sha256")
            or manifest.get("evaluator_source_sha256") != functional.get("evaluator_source_sha256")
            or manifest.get("acceptance_source_sha256")
            != functional.get("acceptance_source_sha256")
            or manifest.get("runtime") != functional.get("runtime")
            or manifest.get("semantic_control_sha256") != semantic_sha
            or hashlib.sha256(_canonical(functional)).hexdigest()
            != accepted.get("solver_functional_sha256")
        ):
            raise ValueError("acceptance evidence identity mismatch")
        record = _objective_record(
            objective_bytes,
            objective_sha,
            str(candidate["id"]),
            str(candidate["provenance"]["source_id"]),
            str(actor),
        )
        if not candidate_acceptance._license_review_verified(
            license_bytes,
            license_sha,
            source_row=source,
            author_actor_id=str(actor),
        ):
            raise ValueError("file-license acceptance evidence is invalid")
        if not _role_evidence_verified(
            role_bytes,
            role_sha,
            candidate=candidate,
            state=state,
            tokenizer=tokenizer,
            author_actor_id=str(actor),
        ):
            raise ValueError("independent author/solver/reviewer evidence is invalid")
        semantic_payload, semantic_record = _semantic_controls_payload(
            payloads["semantic_controls.json"],
            expected_digest=semantic_sha,
            candidate=candidate,
            role_bytes=role_bytes,
        )
        if hashlib.sha256(semantic_payload).hexdigest() != semantic_sha:
            raise ValueError("semantic control hash changed during verification")
        expected_acceptance_schema = (
            ACCEPTANCE_SCHEMA_V2 if manifest_schema == PACKAGE_SCHEMA_V2 else ACCEPTANCE_SCHEMA
        )
        if not isinstance(accepted, dict) or accepted.get("schema") != expected_acceptance_schema:
            raise ValueError("acceptance result schema is invalid")
        if (
            accepted.get("candidate_id") != candidate["id"]
            or accepted.get("source_id") != candidate["provenance"]["source_id"]
            or accepted.get("candidate_sha256") != reference["candidate_sha256"]
            or accepted.get("source_row_sha256") != reference["source_row_sha256"]
            or accepted.get("objective_manifest_sha256") != objective_sha
            or accepted.get("split_manifest_sha256") != split_sha
            or accepted.get("role_evidence_sha256") != role_sha
            or accepted.get("file_license_review_sha256") != license_sha
            or accepted.get("semantic_control_sha256") != semantic_sha
            or accepted.get("decision") != "accepted"
            or accepted.get("reason") != "objective_and_controls_passed"
        ):
            raise ValueError("CPU acceptance receipt does not match the package")
        if manifest_schema == PACKAGE_SCHEMA_V2 and (
            accepted.get("compiler_qualification_plan_sha256") != compiler_plan_sha
            or accepted.get("compiler_control_evidence_sha256") != compiler_controls_sha
            or accepted.get("compiler_status") != "pass"
        ):
            raise ValueError("CPU acceptance receipt omits compiler receipts")
        evidence = accepted.get("evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("objective_manifest_sha256") != objective_sha
            or evidence.get("gold") is not True
            or evidence.get("unchanged") is not False
            or evidence.get("wrong_edits") != [False, False]
            or evidence.get("role_evidence_sha256") != role_sha
            or evidence.get("semantic_control_sha256") != semantic_sha
        ):
            raise ValueError("recorded objective/control outcomes are not accepted")
        if not isinstance(functional, dict) or functional.get("schema") != SOLVER_FUNCTIONAL_SCHEMA:
            raise ValueError("blind solver functional result schema is invalid")
        wire, action, after = _solver_action(role_bytes, candidate, tokenizer)
        runtime = _runtime_identity(record)
        evaluator_file = Path(evaluate_prediction.__code__.co_filename).resolve()
        acceptance_file = Path(candidate_acceptance.__file__).resolve()
        expected_functional = {
            "candidate_id": candidate["id"],
            "source_id": candidate["provenance"]["source_id"],
            "candidate_sha256": reference["candidate_sha256"],
            "source_row_sha256": reference["source_row_sha256"],
            "state_sha256": _digest(asdict(state)),
            "action_sha256": _digest(asdict(action)),
            "solver_wire_sha256": hashlib.sha256(wire.encode("utf-8")).hexdigest(),
            "solver_after_source_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
            "objective_manifest_sha256": objective_sha,
            "objective_fixture_sha256": hashlib.sha256(_canonical(dict(record))).hexdigest(),
            "evaluator_source_sha256": hashlib.sha256(evaluator_file.read_bytes()).hexdigest(),
            "acceptance_source_sha256": hashlib.sha256(acceptance_file.read_bytes()).hexdigest(),
            "runtime": runtime,
            "functional_status": "pass",
            "outcome": {"gold": True, "termination": True},
        }
        if manifest_schema == PACKAGE_SCHEMA_V2:
            expected_functional.update(
                {
                    "compiler_qualification_plan_sha256": compiler_plan_sha,
                    "compiler_control_evidence_sha256": compiler_controls_sha,
                    "compiler_status": "pass",
                }
            )
        context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
        expected_functional["context_sha256"] = hashlib.sha256(context.text.encode()).hexdigest()
        expected_functional["context_policy"] = CONTEXT_POLICY_VERSION
        if any(functional.get(key) != value for key, value in expected_functional.items()):
            raise ValueError("blind solver functional result is not bound to this action/runtime")
        if (
            accepted.get("solver_functional_sha256")
            != hashlib.sha256(_canonical(functional)).hexdigest()
        ):
            raise ValueError("blind solver functional result hash mismatch")
        if row.get("split") not in {"train", "development"}:
            raise ValueError("Muse row is not assigned to an allowed split")
        assignment = candidate_acceptance._split_record(
            split_bytes, split_sha, str(candidate["id"]), str(candidate["source_repo"])
        )
        if assignment.get("split") != row.get("split") or candidate.get("split") != row.get(
            "split"
        ):
            raise ValueError("Muse row split assignment mismatch")
        aliases = {str(row.get("source_repo", "")).casefold()}
        aliases.update(str(value).casefold() for value in row.get("source_aliases", ()))
        if aliases & {value.casefold() for value in reserved_repositories}:
            raise ValueError("Muse row overlaps a reserved repository")
        validate_splits([*existing_rows, candidate])
        return AcceptanceDecision(
            True,
            "frozen_muse_acceptance_package_verified",
            {
                "package_manifest_sha256": reference["manifest_sha256"],
                "candidate_sha256": reference["candidate_sha256"],
                "source_row_sha256": reference["source_row_sha256"],
                "objective_manifest_sha256": objective_sha,
                "solver_functional_status": "pass",
                "solver_functional_sha256": hashlib.sha256(_canonical(functional)).hexdigest(),
                "semantic_control_sha256": semantic_sha,
                "runtime": runtime,
                "split": row["split"],
                "compiler_status": "pass"
                if manifest_schema == PACKAGE_SCHEMA_V2
                else "not_required",
            },
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return AcceptanceDecision(False, "frozen_muse_acceptance_package_invalid", {})
