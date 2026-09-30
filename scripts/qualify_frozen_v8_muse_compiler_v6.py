"""Add pinned Python compile receipts to the already accepted frozen-v8 case A.

This is a CPU-only, no-provider qualification. It reuses the v5 accepted public
source candidate and the exact frozen v8 objective, solver, and semantic
controls. The v5 records remain untouched; v6 writes a separate package set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tinycomplete.eval.code_benchmark import (  # noqa: E402
    BenchmarkCase,
    CheckSpec,
    Prediction,
    evaluate_prediction,
)
from tinycomplete.observability.context import RunContext, current_run_context  # noqa: E402
from tinycomplete.observability.runs import run_scope  # noqa: E402
from tinycomplete.one_line import candidate_acceptance, muse_acceptance_package  # noqa: E402
from tinycomplete.one_line.contract import EditAction, EditState, apply_action  # noqa: E402

LEGACY_SCRIPT = ROOT / "scripts/export_frozen_v8_muse_acceptance.py"
PRIVATE_ROOT = Path("/mnt/ssd/tabcomplete-product-r2/muse-acceptance-v1")
PACKET_ROOT = Path("/mnt/ssd/tabcomplete-product-r2/next-pilot-frozen-v8")
TOKENIZER = Path(
    "/home/crabcake/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/"
    "snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
TOKENIZER_JSON_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
PLAN_PATH = PRIVATE_ROOT / "qualification-plan-v6.json"
RESULT_PATH = PRIVATE_ROOT / "qualification-result-v6.json"
PARTIAL_PATH = PRIVATE_ROOT / "qualification-partial-v6.json"
PACKAGE_ROOT = PRIVATE_ROOT / "packages-v6"
ROWS_PATH = PACKAGE_ROOT / "accepted_rows.jsonl"
ROWS_MANIFEST_PATH = PACKAGE_ROOT / "accepted_rows_manifest.json"
OBSERVABILITY_PATH = PRIVATE_ROOT / "observability-v6.json"
V5_PLAN_FILE_SHA256 = "ecc516ea4a7508383b2c354995c16f1fc2023eecc30ad4653b6d7ccea3a9a816"
V5_PLAN_SHA256 = "91757f476869319cc29f1b108ffc59327cae1f1de8ed9af0ae394ff04f691ef3"
V5_RESULT_FILE_SHA256 = "8e9f99740dbcdd6eb0e64442179cee460aa6374d8d4993abca9d3ddf0a51b844"
V5_RESULT_SHA256 = "b663b33feee8a617d74bcab2e3bd708ae725a7b8dbd8bd9c82987752c30f34c0"
V5_ROWS_SHA256 = "b45a0b26b2741a77859dfcefb71012f0170a0020895fe44db7af84dcec9c37f3"
V5_ROWS_MANIFEST_SHA256 = "b074262daeb032e8d0706c0216b24f017b66e1f48bd3ce5281815a941b4fefe5"
V5_PACKAGE_MANIFEST_SHA256 = "77d5bb807b036158eed96b4a0112a76356b3546b4dd1dd037f4408902d23af36"
V5_CANDIDATE_ID = "muse-author/37039315e2c6ac46b9880c2e"
MAX_SECONDS = 900
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _read_private(path: Path, *, limit: int = 16 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("qualification input is not a regular file")
    status = path.stat()
    if status.st_size > limit or status.st_mode & 0o077:
        raise ValueError("qualification input is not owner-only or is oversized")
    payload = path.read_bytes()
    if len(payload) != status.st_size:
        raise ValueError("qualification input changed while reading")
    return payload


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("frozen_v8_muse_v5", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ValueError("frozen v5 exporter is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v5_material() -> dict[str, Any]:
    plan_bytes = _read_private(PRIVATE_ROOT / "qualification-plan-v5.json")
    result_bytes = _read_private(PRIVATE_ROOT / "qualification-result-v5.json")
    rows_bytes = _read_private(PRIVATE_ROOT / "packages-v5/accepted_rows.jsonl")
    rows_manifest_bytes = _read_private(PRIVATE_ROOT / "packages-v5/accepted_rows_manifest.json")
    if (
        _sha(plan_bytes) != V5_PLAN_FILE_SHA256
        or _sha(result_bytes) != V5_RESULT_FILE_SHA256
        or _sha(rows_bytes) != V5_ROWS_SHA256
        or _sha(rows_manifest_bytes) != V5_ROWS_MANIFEST_SHA256
    ):
        raise ValueError("historical v5 qualification bytes changed")
    plan = json.loads(plan_bytes)
    result = json.loads(result_bytes)
    rows_manifest = json.loads(rows_manifest_bytes)
    if (
        plan.get("plan_sha256") != V5_PLAN_SHA256
        or result.get("result_sha256") != V5_RESULT_SHA256
        or result.get("plan_sha256") != V5_PLAN_SHA256
        or rows_manifest.get("schema") != "one-line-muse-accepted-rows-manifest-v1"
        or rows_manifest.get("accepted_candidate_ids") != [V5_CANDIDATE_ID]
        or rows_manifest.get("accepted_rows_sha256") != V5_ROWS_SHA256
    ):
        raise ValueError("historical v5 qualification identity is invalid")
    rows = [json.loads(line) for line in rows_bytes.splitlines() if line]
    if len(rows) != 1 or rows[0].get("id") != V5_CANDIDATE_ID:
        raise ValueError("historical v5 accepted row is missing or ambiguous")
    row = rows[0]
    reference = row.get("acceptance_package_ref")
    if (
        not isinstance(reference, dict)
        or reference.get("manifest_sha256") != V5_PACKAGE_MANIFEST_SHA256
    ):
        raise ValueError("historical v5 accepted package reference changed")
    package_directory = PRIVATE_ROOT / "packages-v5" / reference["root"]
    manifest_bytes = _read_private(package_directory / "manifest.json")
    if _sha(manifest_bytes) != V5_PACKAGE_MANIFEST_SHA256:
        raise ValueError("historical v5 package manifest changed")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != muse_acceptance_package.PACKAGE_SCHEMA:
        raise ValueError("historical package is not v1")
    payloads = {
        name: muse_acceptance_package._load_package_file(package_directory, name, manifest)
        for name in muse_acceptance_package._PACKAGE_FILES
    }
    candidate = json.loads(payloads["candidate.json"])
    source_row = json.loads(payloads["source_row.json"])
    if (
        candidate.get("id") != V5_CANDIDATE_ID
        or _sha(payloads["candidate.json"]) != reference.get("candidate_sha256")
        or _sha(payloads["source_row.json"]) != reference.get("source_row_sha256")
    ):
        raise ValueError("historical accepted candidate bytes changed")
    role = json.loads(payloads["role_evidence.json"])
    packet = role.get("packet")
    if not isinstance(packet, dict) or packet.get("case_id") != "synthetic-a":
        raise ValueError("accepted candidate is not the frozen v8 source case A")
    if row.get("acceptance_package_ref") != reference:
        raise ValueError("accepted row package reference mismatch")
    return {
        "v5_plan_bytes": plan_bytes,
        "v5_result_bytes": result_bytes,
        "v5_rows_bytes": rows_bytes,
        "v5_rows_manifest_bytes": rows_manifest_bytes,
        "row": row,
        "reference": reference,
        "package_directory": package_directory,
        "package_manifest_bytes": manifest_bytes,
        "candidate": candidate,
        "source_row": source_row,
        "payloads": payloads,
        "role": role,
        "packet": packet,
    }


def _candidate_identities(material: dict[str, Any]) -> dict[str, str | None]:
    payloads = material["payloads"]
    source_metadata = material["source_row"]["authoring_metadata"]
    return {
        "candidate_sha256": _sha(payloads["candidate.json"]),
        "source_row_sha256": _sha(payloads["source_row.json"]),
        "objective_manifest_sha256": _sha(payloads["objective_manifest.json"]),
        "split_manifest_sha256": _sha(payloads["split_manifest.json"]),
        "role_evidence_sha256": _sha(payloads["role_evidence.json"]),
        "file_license_review_sha256": _sha(payloads["file_license_review.json"]),
        "semantic_control_sha256": _sha(payloads["semantic_controls.json"]),
        "source_sha256": source_metadata.get("source_sha256"),
        "license_sha256": source_metadata.get("license_sha256"),
    }


def _controls_from_frozen_evidence(material: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = material["candidate"]
    payloads = material["payloads"]
    semantic = json.loads(payloads["semantic_controls.json"])
    prior = json.loads(semantic["preflight_result_json"])
    fixture_rows = [
        json.loads(line) for line in semantic["oracle_fixtures_jsonl"].splitlines() if line
    ]
    case_id = material["packet"]["case_id"]
    fixtures = [row for row in fixture_rows if row.get("case_id") == case_id]
    if len(fixtures) != 1:
        raise ValueError("frozen v8 semantic fixture is not unique")
    fixture = fixtures[0]
    expected_ids = semantic.get("selected_control_case_ids")
    prior_rows = [
        row
        for row in prior.get("cases", [])
        if isinstance(row, dict) and row.get("case_action_id") in expected_ids
    ]
    if [row.get("case_action_id") for row in prior_rows] != expected_ids:
        raise ValueError("frozen semantic control order changed")
    wrong = {f"{case_id}:wrong:{row['name']}": row for row in fixture.get("wrong_controls", [])}
    state = EditState.from_mapping(candidate["state"])
    output: list[dict[str, Any]] = []
    for index, prior_row in enumerate(prior_rows):
        action_id = prior_row["case_action_id"]
        action_raw = (
            fixture.get("gold_action") if index == 0 else wrong.get(action_id, {}).get("action")
        )
        if not isinstance(action_raw, dict):
            raise ValueError("frozen control action is absent")
        action = EditAction(**action_raw)
        after = apply_action(state, action)
        after_sha = _sha(after.encode("utf-8"))
        action_sha = muse_acceptance_package._digest(action_raw)
        canonical_action_sha = muse_acceptance_package._digest(
            __import__("dataclasses").asdict(action)
        )
        if (
            prior_row.get("action_sha256") != action_sha
            or prior_row.get("complete_source_sha256") != after_sha
        ):
            raise ValueError("frozen action no longer reconstructs the prior oracle source")
        output.append(
            {
                "case_action_id": action_id,
                "action_sha256": action_sha,
                "canonical_action_sha256": canonical_action_sha,
                "after_source_sha256": after_sha,
                "expected_test_status": "pass" if index == 0 else "fail",
                "_action": action,
                "_after_source": after,
            }
        )
    if candidate.get("action") != fixture.get("gold_action"):
        raise ValueError("accepted candidate action differs from the frozen v8 gold action")
    return output


def _code_hashes() -> dict[str, str]:
    files = {
        "qualification_runner": Path(__file__).resolve(),
        "package_verifier": Path(muse_acceptance_package.__file__).resolve(),
        "candidate_acceptance": Path(candidate_acceptance.__file__).resolve(),
        "code_benchmark": Path(evaluate_prediction.__code__.co_filename).resolve(),
    }
    return {name: _sha(path.read_bytes()) for name, path in files.items()}


def _qualify_inputs(tokenizer: Any | None) -> dict[str, Any]:
    material = _v5_material()
    payloads = material["payloads"]
    plan_v5 = json.loads(material["v5_plan_bytes"])
    result_v5 = json.loads(material["v5_result_bytes"])
    if (
        plan_v5.get("plan_sha256") != V5_PLAN_SHA256
        or result_v5.get("result_sha256") != V5_RESULT_SHA256
        or result_v5.get("status") != "complete"
        or result_v5.get("training_started") is not False
    ):
        raise ValueError("historical v5 outcome is not complete and nontraining")
    role_bytes = payloads["role_evidence.json"]
    state = EditState.from_mapping(material["candidate"]["state"])
    if tokenizer is not None:
        if not candidate_acceptance._role_evidence_verified(
            role_bytes,
            _sha(role_bytes),
            candidate=material["candidate"],
            state=state,
            tokenizer=tokenizer,
            author_actor_id=material["candidate"]["provenance"]["author_actor_id"],
        ):
            raise ValueError("historical v8 role evidence no longer verifies")
    semantic, _semantic_record = muse_acceptance_package._semantic_controls_payload(
        payloads["semantic_controls.json"],
        expected_digest=_sha(payloads["semantic_controls.json"]),
        candidate=material["candidate"],
        role_bytes=role_bytes,
    )
    controls = _controls_from_frozen_evidence(material)
    objective = json.loads(payloads["objective_manifest.json"])
    record = candidate_acceptance._objective_record(
        payloads["objective_manifest.json"],
        _sha(payloads["objective_manifest.json"]),
        material["candidate"]["id"],
        material["candidate"]["provenance"]["source_id"],
        material["candidate"]["provenance"]["author_actor_id"],
    )
    compiler = {
        "language": "python",
        "command": ["python", "-m", "py_compile", "solution.py"],
        "target_path": "solution.py",
        "runtime": muse_acceptance_package._runtime_identity(record),
    }
    del objective
    return {
        "material": material,
        "semantic": semantic,
        "controls": controls,
        "compiler": compiler,
        "identities": _candidate_identities(material),
    }


def _freeze_plan() -> dict[str, Any]:
    if any(path.exists() or path.is_symlink() for path in (PLAN_PATH, RESULT_PATH, PACKAGE_ROOT)):
        raise ValueError("v6 qualification outputs already exist; refusing overwrite")
    tok_file = TOKENIZER / "tokenizer.json"
    resolved = tok_file.resolve(strict=True)
    if _sha(resolved.read_bytes()) != TOKENIZER_JSON_SHA256:
        raise ValueError("pinned local tokenizer identity changed")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    inputs = _qualify_inputs(tokenizer)
    material = inputs["material"]
    controls = [
        {
            key: row[key]
            for key in (
                "case_action_id",
                "action_sha256",
                "canonical_action_sha256",
                "after_source_sha256",
                "expected_test_status",
            )
        }
        for row in inputs["controls"]
    ]
    evaluator_hash = _code_hashes()["code_benchmark"]
    body = {
        "schema": muse_acceptance_package.COMPILER_PLAN_SCHEMA,
        "revision": 6,
        "status": "frozen_before_compile",
        "purpose": "bind configured Python bytecode compilation to v8 source-semantic controls",
        "provider_calls": 0,
        "training_started": False,
        "candidate_id": material["candidate"]["id"],
        "case_id": material["packet"]["case_id"],
        "packet_binding": material["packet"],
        "historical_v5": {
            "plan_file_sha256": _sha(material["v5_plan_bytes"]),
            "plan_sha256": V5_PLAN_SHA256,
            "result_file_sha256": _sha(material["v5_result_bytes"]),
            "result_sha256": V5_RESULT_SHA256,
            "accepted_rows_sha256": _sha(material["v5_rows_bytes"]),
            "accepted_rows_manifest_sha256": _sha(material["v5_rows_manifest_bytes"]),
            "package_manifest_sha256": _sha(material["package_manifest_bytes"]),
        },
        "identities": inputs["identities"],
        "semantic_control_case_ids": [row["case_action_id"] for row in controls],
        "controls": controls,
        "compiler": inputs["compiler"],
        "evaluator_source_sha256": evaluator_hash,
        "code_sha256": _code_hashes(),
        "tokenizer": {
            "path": "pinned local Qwen2.5-Coder tokenizer",
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        },
        "limits": {
            "max_wall_seconds": MAX_SECONDS,
            "compiles": len(controls),
            "provider_calls": 0,
            "training_tokens": 0,
        },
        "outputs": {
            "result_path": str(RESULT_PATH),
            "package_root": str(PACKAGE_ROOT),
            "rows_path": str(ROWS_PATH),
            "rows_manifest_path": str(ROWS_MANIFEST_PATH),
        },
    }
    doc = {**body, "plan_sha256": muse_acceptance_package._digest(body)}
    _write_once(
        PLAN_PATH, json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    )
    return doc


def _verify_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    plan_bytes = _read_private(PLAN_PATH)
    plan = json.loads(plan_bytes)
    claimed = plan.pop("plan_sha256", None)
    if claimed != muse_acceptance_package._digest(plan):
        raise ValueError("v6 plan self-hash mismatch")
    plan["plan_sha256"] = claimed
    if (
        plan.get("schema") != muse_acceptance_package.COMPILER_PLAN_SCHEMA
        or plan.get("status") != "frozen_before_compile"
    ):
        raise ValueError("v6 plan is not in its frozen pre-output state")
    if plan.get("code_sha256") != _code_hashes():
        raise ValueError("v6 qualification code changed after freeze")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    actual = _qualify_inputs(tokenizer)
    material = actual["material"]
    controls = [
        {
            key: row[key]
            for key in (
                "case_action_id",
                "action_sha256",
                "canonical_action_sha256",
                "after_source_sha256",
                "expected_test_status",
            )
        }
        for row in actual["controls"]
    ]
    if (
        plan.get("candidate_id") != material["candidate"]["id"]
        or plan.get("packet_binding") != material["packet"]
        or plan.get("identities") != actual["identities"]
        or plan.get("controls") != controls
        or plan.get("compiler") != actual["compiler"]
        or plan.get("evaluator_source_sha256") != _code_hashes()["code_benchmark"]
    ):
        raise ValueError("v6 frozen inputs differ from the historical candidate evidence")
    return plan, actual


def _compiler_receipt(plan: dict[str, Any], actual: dict[str, Any], run_id: str) -> dict[str, Any]:
    material = actual["material"]
    objective_bytes = material["payloads"]["objective_manifest.json"]
    record = candidate_acceptance._objective_record(
        objective_bytes,
        _sha(objective_bytes),
        material["candidate"]["id"],
        material["candidate"]["provenance"]["source_id"],
        material["candidate"]["provenance"]["author_actor_id"],
    )
    check_raw = record["check"]
    old_check = CheckSpec.model_validate(check_raw)
    check = CheckSpec(
        compile=plan["compiler"]["command"],
        test=old_check.test,
        files=old_check.files,
        timeout_seconds=old_check.timeout_seconds,
        container_image=old_check.container_image,
    )
    receipts: list[dict[str, Any]] = []
    started = time.monotonic()
    context = actual.get("run_context") or RunContext.new(campaign_id="frozen-v8-muse-compiler-v6")
    for control in actual["controls"]:
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("v6 compiler qualification reached its wall limit")
        action_id = control["case_action_id"]
        case = BenchmarkCase(
            id=action_id,
            language="python",
            path=plan["compiler"]["target_path"],
            prefix=control["_after_source"],
            expected=control["_after_source"],
            check=check,
            category="frozen-v8-semantic-control",
        )
        case_context = context.for_case(action_id)
        with (
            case_context.activate(),
            tempfile.TemporaryDirectory(prefix="tc-muse-compile-v6-") as directory,
        ):
            try:
                result = evaluate_prediction(
                    case,
                    Prediction(case_id=action_id, completion=""),
                    work_root=Path(directory),
                    execution_backend="container",
                )
                current_context = current_run_context()
                run_identity = {
                    "campaign_id": current_context.campaign_id if current_context else None,
                    "run_id": current_context.run_id if current_context else run_id,
                    "run_attempt_id": current_context.run_attempt_id if current_context else None,
                    "case_id": current_context.case_id if current_context else action_id,
                    "case_attempt_id": current_context.case_attempt_id if current_context else None,
                    "request_id": current_context.request_id if current_context else None,
                }
                receipt = {
                    "case_action_id": action_id,
                    "action_sha256": control["action_sha256"],
                    "canonical_action_sha256": control["canonical_action_sha256"],
                    "after_source_sha256": control["after_source_sha256"],
                    "expected_test_status": control["expected_test_status"],
                    "parse_status": result.parse.status,
                    "compile_configured": result.compile_configured,
                    "compile_status": result.compile.status,
                    "compile_returncode": result.compile.returncode,
                    "test_configured": result.test_configured,
                    "test_status": result.test.status,
                    "test_returncode": result.test.returncode,
                    "compile_stdout_sha256": _sha(result.compile.stdout.encode("utf-8")),
                    "compile_stderr_sha256": _sha(result.compile.stderr.encode("utf-8")),
                    "test_stdout_sha256": _sha(result.test.stdout.encode("utf-8")),
                    "test_stderr_sha256": _sha(result.test.stderr.encode("utf-8")),
                    "working_tree_sha256": result.working_tree_sha256,
                    "runtime": plan["compiler"]["runtime"],
                    "evaluator_source_sha256": plan["evaluator_source_sha256"],
                    "observability": run_identity,
                }
            except Exception as error:
                receipt = {
                    "case_action_id": action_id,
                    "action_sha256": control["action_sha256"],
                    "canonical_action_sha256": control["canonical_action_sha256"],
                    "after_source_sha256": control["after_source_sha256"],
                    "expected_test_status": control["expected_test_status"],
                    "parse_status": "error",
                    "compile_configured": True,
                    "compile_status": "error",
                    "compile_returncode": None,
                    "test_configured": True,
                    "test_status": "not_run",
                    "test_returncode": None,
                    "failure_type": type(error).__name__,
                    "runtime": plan["compiler"]["runtime"],
                    "evaluator_source_sha256": plan["evaluator_source_sha256"],
                    "observability": {"run_id": run_id, "case_id": action_id},
                }
            receipts.append(receipt)
            _write_partial(plan, receipts, run_id)
    evidence: dict[str, Any] = {
        "schema": muse_acceptance_package.COMPILER_CONTROLS_SCHEMA,
        "candidate_id": material["candidate"]["id"],
        "plan_file_sha256": _sha(_read_private(PLAN_PATH)),
        "plan_sha256": plan["plan_sha256"],
        "packet_binding": plan["packet_binding"],
        "identities": plan["identities"],
        "compiler": plan["compiler"],
        "evaluator_source_sha256": plan["evaluator_source_sha256"],
        "controls": receipts,
        "provider_calls": 0,
        "training_started": False,
    }
    evidence["result_sha256"] = muse_acceptance_package._digest(evidence)
    return evidence


def _write_partial(plan: dict[str, Any], receipts: list[dict[str, Any]], run_id: str) -> None:
    partial = {
        "schema": "frozen-v8-muse-compiler-partial-v6",
        "plan_sha256": plan["plan_sha256"],
        "run_id": run_id,
        "completed_controls": len(receipts),
        "controls": receipts,
        "provider_calls": 0,
        "training_started": False,
    }
    data = json.dumps(partial, sort_keys=True, ensure_ascii=False, indent=2).encode() + b"\n"
    if PARTIAL_PATH.exists():
        temp = PARTIAL_PATH.with_suffix(".tmp")
        temp.write_bytes(data)
        os.chmod(temp, 0o600)
        os.replace(temp, PARTIAL_PATH)
    else:
        _write_once(PARTIAL_PATH, data)


def _run_execute() -> dict[str, Any]:
    if any(
        path.exists() or path.is_symlink()
        for path in (RESULT_PATH, PACKAGE_ROOT, ROWS_PATH, ROWS_MANIFEST_PATH)
    ):
        raise ValueError("v6 outputs already exist; refusing rerun")
    plan, actual = _verify_plan()
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    from tinycomplete.observability.context import current_run_context as active_context

    with run_scope(OBSERVABILITY_PATH, "frozen-v8-muse-compiler-qualification-v6") as run:
        context = (
            run or active_context() or RunContext.new(campaign_id="frozen-v8-muse-compiler-v6")
        )
        actual["run_context"] = context
        compiler_receipts = _compiler_receipt(plan, actual, context.run_id)
    if time.monotonic() - start > MAX_SECONDS:
        raise TimeoutError("v6 compiler qualification exceeded its 15-minute budget")
    material = actual["material"]
    values = {
        "package_root": PACKAGE_ROOT,
        "candidate": material["candidate"],
        "source_row": material["source_row"],
        "tokenizer": tokenizer,
        "objective_manifest": material["payloads"]["objective_manifest.json"],
        "split_manifest": material["payloads"]["split_manifest.json"],
        "role_evidence": material["payloads"]["role_evidence.json"],
        "file_license_review": material["payloads"]["file_license_review.json"],
        "semantic_control_evidence": material["payloads"]["semantic_controls.json"],
        "expected_semantic_control_sha256": _sha(material["payloads"]["semantic_controls.json"]),
        "compiler_qualification_plan": _read_private(PLAN_PATH),
        "compiler_control_evidence": _canonical(compiler_receipts),
    }
    outcome = muse_acceptance_package.export_muse_acceptance_package(**values)
    case_record = {
        "case_id": material["packet"]["case_id"],
        "candidate_id": material["candidate"]["id"],
        "decision": "accepted" if outcome.accepted else "rejected",
        "reason": outcome.reason,
        "provider_calls": 0,
        "training_started": False,
        "compiler_receipt_sha256": _sha(_canonical(compiler_receipts)),
        "compile_statuses": [row.get("compile_status") for row in compiler_receipts["controls"]],
        "test_statuses": [row.get("test_status") for row in compiler_receipts["controls"]],
    }
    accepted_rows: list[dict[str, Any]] = []
    if outcome.accepted:
        if outcome.candidate_row is None or outcome.acceptance_package_ref is None:
            raise ValueError("v6 exporter accepted without a portable row")
        row = dict(material["row"])
        row.update(outcome.candidate_row)
        row["candidate_id"] = row["id"]
        row["acceptance_package_ref"] = outcome.acceptance_package_ref
        row["validation"]["accepted_training"] = False
        row["human_chronology_observed"] = False
        portable = muse_acceptance_package.verify_muse_candidate_row(
            row, package_root=PACKAGE_ROOT, tokenizer=tokenizer
        )
        case_record["portable_verification"] = {
            "accepted": portable.accepted,
            "reason": portable.reason,
            "compiler_status": portable.evidence.get("compiler_status"),
        }
        if not portable.accepted:
            raise ValueError("v6 portable training-side verification failed")
        accepted_rows.append(row)
        _write_rows(accepted_rows, plan)
        relocated = _verify_relocated(accepted_rows, tokenizer, plan)
        case_record["relocation_verification"] = relocated
    else:
        case_record["portable_verification"] = {"accepted": False, "reason": outcome.reason}
    result = {
        "schema": "frozen-v8-muse-compiler-qualification-result-v6",
        "plan_sha256": plan["plan_sha256"],
        "status": "complete"
        if all(row.get("compile_status") == "pass" for row in compiler_receipts["controls"])
        else "compile_failed",
        "elapsed_seconds": time.monotonic() - start,
        "provider_calls": 0,
        "training_started": False,
        "compiler_controls_sha256": _sha(_canonical(compiler_receipts)),
        "compiler_control_summary": case_record["compile_statuses"],
        "case": case_record,
        "accepted_row_count": len(accepted_rows),
        "accepted_rows_sha256": _sha(_canonical(accepted_rows)) if accepted_rows else None,
    }
    result["result_sha256"] = muse_acceptance_package._digest(result)
    _write_once(
        RESULT_PATH,
        json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2).encode() + b"\n",
    )
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
    return result


def _write_rows(rows: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    package_dir = PACKAGE_ROOT
    package_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(package_dir, 0o700)
    rows_data = b"".join(_canonical(row) + b"\n" for row in rows)
    _write_once(ROWS_PATH, rows_data)
    body = {
        "schema": "one-line-muse-accepted-rows-manifest-v2",
        "plan_sha256": plan["plan_sha256"],
        "accepted_candidate_ids": [row["id"] for row in rows],
        "accepted_row_count": len(rows),
        "accepted_rows_sha256": _sha(rows_data),
        "packages": [row["acceptance_package_ref"] for row in rows],
        "provider_calls": 0,
        "training_started": False,
    }
    body["manifest_sha256"] = muse_acceptance_package._digest(body)
    _write_once(
        ROWS_MANIFEST_PATH,
        json.dumps(body, sort_keys=True, ensure_ascii=False, indent=2).encode() + b"\n",
    )


def _verify_relocated(
    rows: list[dict[str, Any]], tokenizer: Any, plan: dict[str, Any]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="muse-v6-relocation-") as temporary:
        root = Path(temporary) / "relocated"
        source_root = PACKAGE_ROOT
        package_ref = rows[0]["acceptance_package_ref"]
        source_package = source_root / package_ref["root"]
        relative_package = Path(package_ref["root"])
        target_package = root / relative_package
        target_package.mkdir(mode=0o700, parents=True)
        for source in source_package.iterdir():
            target = target_package / source.name
            shutil.copyfile(source, target)
            os.chmod(target, 0o600)
        relocated_row = dict(rows[0])
        relocated = muse_acceptance_package.verify_muse_candidate_row(
            relocated_row, package_root=root, tokenizer=tokenizer
        )
        if not relocated.accepted:
            raise ValueError("relocated v6 package verification failed")
        return {
            "status": "verified",
            "reason": relocated.reason,
            "compiler_status": relocated.evidence.get("compiler_status"),
            "package_manifest_sha256": package_ref["manifest_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "absolute_original_root_embedded": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = _freeze_plan() if args.freeze else _run_execute()
        print(
            json.dumps(
                {
                    "status": result.get("status", "frozen"),
                    "plan_sha256": result.get("plan_sha256"),
                    "candidate_id": result.get("candidate_id")
                    or result.get("case", {}).get("candidate_id"),
                    "accepted_row_count": result.get("accepted_row_count"),
                }
            )
        )
        return 0
    except Exception as error:
        # Keep private paths, raw code, provider text, and exception bodies out of output.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
