"""Qualify the two already-frozen v8 public-source cases on local CPU.

This entry point has two phases: `--freeze` records all packet/code/runtime
identities without evaluating a candidate; `--execute` verifies that frozen
plan and then runs the existing no-network sandbox acceptance path. It makes
no provider calls and never changes the deployed model or editor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tinycomplete.one_line import candidate_acceptance  # noqa: E402
from tinycomplete.one_line.candidate_acceptance import _role_evidence_verified  # noqa: E402
from tinycomplete.one_line.contract import EditState, physical_lines  # noqa: E402
from tinycomplete.one_line.data import (  # noqa: E402
    parse_author_response,
    replay_replacement_history,
)
from tinycomplete.one_line.muse_acceptance_package import (  # noqa: E402
    _canonical,
    _digest,
    _semantic_controls_payload,
    export_muse_acceptance_package,
    objective_sandbox_path,
    verify_muse_candidate_row,
)
from tinycomplete.one_line.pilot_roles import parse_review_response  # noqa: E402
from tinycomplete.one_line.teacher import (  # noqa: E402
    extract_author_candidate_block,
    extract_final_action_block_v7,
)

PACKET = Path("/mnt/ssd/tabcomplete-product-r2/next-pilot-frozen-v8")
LICENSE_AUDIT = Path(
    "/mnt/ssd/tabcomplete-product-r2/next-pilot-license-scope-audit-v1/audit-v2.json"
)
SOURCE_PAIRS = Path(
    "/mnt/ssd/tabcomplete-product-r2/commitpackft/source-verification-v1/source_pairs"
)
TOKENIZER = Path(
    "/home/crabcake/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/"
    "snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
TOKENIZER_JSON_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
PRIVATE_ROOT = Path("/mnt/ssd/tabcomplete-product-r2/muse-acceptance-v1")
PLAN_PATH = PRIVATE_ROOT / "qualification-plan-v5.json"
RESULT_PATH = PRIVATE_ROOT / "qualification-result-v5.json"
PACKAGE_ROOT = PRIVATE_ROOT / "packages-v5"
ROWS_PATH = PACKAGE_ROOT / "accepted_rows.jsonl"
ROWS_MANIFEST_PATH = PACKAGE_ROOT / "accepted_rows_manifest.json"
CASE_SPLITS = {"synthetic-a": "train", "synthetic-b": "development"}
_SHA = re.compile(r"[a-f0-9]{64}\Z")
PREVIOUS_PLAN_SHA256 = "08a5173bcbaac6dadca7dcb56840f1690788749301d82b0f4fce8d21136f51c0"
PREVIOUS_RESULT_SHA256 = "c3b2e10a4893b244e3443336c458deb4f4a1385b7fdbe2b7d0167328c2db6243"
PREVIOUS_PLAN_FILE_SHA256 = "73e45787becc9f7f86fbd5875ea2d3c2fa4a25adb4c1de60a77524938a95a533"
PREVIOUS_RESULT_FILE_SHA256 = "dd8a512177d3338caa2ab44d6c5bf291a5e0f3e9c4ca331adadcee2ecf9c8ec1"
FAILED_V2_PLAN_SHA256 = "584f343f935f8719ed9101ff5b259761acaf168cef191f3092403a8b89e998a0"
FAILED_V2_PLAN_FILE_SHA256 = "4017f347a10ffb191221fee8c489cf6f869b4d698162aaf2a057ee3c1c2b32dd"
SUCCESSFUL_V3_PLAN_SHA256 = "cd85103fc07b5c74b8af137c366b6e39353207cd5ec951b3f8eb98c47a359f62"
SUCCESSFUL_V3_PLAN_FILE_SHA256 = "8b70eac37d6e08b167117509112d52891b03635951a47d8c26f3bda11d090c9f"
SUCCESSFUL_V3_RESULT_SHA256 = "b0eb645b4477701bf432d109e60d2f18913b1df2c5dcd85a57c6c2f81cba1307"
SUCCESSFUL_V3_RESULT_FILE_SHA256 = (
    "129b058f1d19873c07839c0f9bcf0b7ec939ed13b3e7fc0005d2fb0b89d034eb"
)
SUCCESSFUL_V4_PLAN_SHA256 = "7f4506df66b417c836235b8ae60d6f41f1c380d7c3a21a7f34af3df76db94a88"
SUCCESSFUL_V4_PLAN_FILE_SHA256 = "969537c2383da19ed80b6185379aff5fb94a5f5601f4029717420f659964ce56"
SUCCESSFUL_V4_RESULT_SHA256 = "bc04b285890cef7c8a971f33188b70cc3ec3e2780c448ee539c4e2761cd4445c"
SUCCESSFUL_V4_RESULT_FILE_SHA256 = (
    "e5ff6d4fefe6ef78e3ec0b2f916f5a34505d99c7f71e1edd125017460b67ddfb"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_private(path: Path, *, max_bytes: int = 8 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen input is not a regular file")
    info = path.stat()
    if info.st_size > max_bytes or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("frozen input is not owner-only or exceeds its limit")
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise ValueError("frozen input changed while reading")
    return data


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_json(path: Path) -> Any:
    return json.loads(_read_private(path).decode("utf-8"))


def _verify_self_hashed_document(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_identity_sha256: str,
    identity_key: str,
    label: str,
) -> dict[str, Any]:
    """Check serialized-file identity separately from canonical document identity."""
    payload = _read_private(path)
    if _sha(payload) != expected_file_sha256:
        raise ValueError(f"{label} file hash mismatch")
    try:
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=candidate_acceptance._strict_pairs
        )
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError):
        raise ValueError(f"{label} is not strict JSON") from None
    if not isinstance(document, dict):
        raise ValueError(f"{label} shape is invalid")
    claimed_identity = document.pop(identity_key, None)
    if (
        claimed_identity != expected_identity_sha256
        or _digest(document) != expected_identity_sha256
    ):
        raise ValueError(f"{label} canonical identity mismatch")
    document[identity_key] = claimed_identity
    return document


def _objective_entry(
    *, candidate_id: str, source_id: str, filetype: str, fixture: dict[str, Any]
) -> dict[str, Any]:
    """Build an objective entry using the same target paths as oracle preflight."""
    oracle_filename = "oracle.py" if filetype == "python" else "oracle.js"
    return {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "kind": "sandbox_test",
        "path": objective_sandbox_path(filetype),
        "check": {
            "container_image": fixture["runtime_image"],
            "test": fixture["runtime_command"],
            "files": {oracle_filename: fixture["test_source"]},
            "timeout_seconds": 10.0,
        },
    }


def _file_hashes() -> dict[str, str]:
    files = (
        "scripts/export_frozen_v8_muse_acceptance.py",
        "src/tinycomplete/one_line/muse_acceptance_package.py",
        "src/tinycomplete/one_line/candidate_acceptance.py",
        "src/tinycomplete/one_line/contract.py",
        "src/tinycomplete/one_line/data.py",
        "src/tinycomplete/one_line/pilot_roles.py",
        "src/tinycomplete/one_line/teacher.py",
        "src/tinycomplete/eval/code_benchmark.py",
        "tests/test_muse_acceptance_package.py",
        "tests/test_frozen_v8_muse_qualification.py",
    )
    return {name: _sha((ROOT / name).read_bytes()) for name in files}


def _tokenizer_hash() -> str:
    tokenizer_json = TOKENIZER / "tokenizer.json"
    try:
        target = tokenizer_json.resolve(strict=True)
        blob_root = (TOKENIZER.parents[1] / "blobs").resolve(strict=True)
    except OSError:
        raise ValueError("pinned local tokenizer is unavailable") from None
    if (
        not target.is_file()
        or not target.is_relative_to(blob_root)
        or target.stat().st_size > 32_000_000
    ):
        raise ValueError("pinned local tokenizer is unavailable")
    return _sha(target.read_bytes())


def _packet_hashes() -> dict[str, Any]:
    paths = {
        "plan_file_sha256": PACKET / "plan.json",
        "artifact_manifest_sha256": PACKET / "artifact_manifest.json",
        "inputs_sha256": PACKET / "source_only_inputs.jsonl",
        "oracle_fixtures_sha256": PACKET / "oracle_fixtures.jsonl",
        "preflight_result_sha256": PACKET / "oracle_preflight_result.json",
        "license_scope_audit_sha256": LICENSE_AUDIT,
        "qualification_sha256": PACKET / "execution-v8/qualification.json",
        "qualification_events_sha256": PACKET / "execution-v8/qualification_events.jsonl",
    }
    output: dict[str, Any] = {key: _sha(_read_private(path)) for key, path in paths.items()}
    output["plan_canonical_sha256"] = _load_json(PACKET / "plan.json")["plan_sha256"]
    output["role_event_log_hashes"] = {
        role: _sha(_read_private(PACKET / f"execution-v8/{role}_events.jsonl"))
        for role in ("author", "solver", "reviewer")
    }
    return output


def _no_output_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    """Check frozen role/source bindings without executing candidate code."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    checks = []
    expected_role_acceptance = {"synthetic-a": True, "synthetic-b": False}
    expected_reviewer = {
        "synthetic-a": (True, False),
        "synthetic-b": (False, True),
    }
    for case_id in ("synthetic-a", "synthetic-b"):
        values = _case_inputs(case_id, tokenizer, plan)
        candidate = values["candidate"]
        state = EditState.from_mapping(candidate["state"])
        role_bytes = values["role_evidence"]
        role = json.loads(role_bytes)
        verifier_accepts = _role_evidence_verified(
            role_bytes,
            _sha(role_bytes),
            candidate=candidate,
            state=state,
            tokenizer=tokenizer,
            author_actor_id=candidate["provenance"]["author_actor_id"],
        )
        reviewer = parse_review_response(role["reviewer"]["verdict_json"])
        expected_acceptance = expected_role_acceptance[case_id]
        if verifier_accepts is not expected_acceptance:
            raise ValueError("frozen source-role preflight outcome differs")
        if (reviewer.retain, reviewer.ambiguous) != expected_reviewer[case_id]:
            raise ValueError("frozen reviewer verdict differs from the approved case outcome")
        control_payload = json.loads(values["semantic_control_evidence"])
        case_fixture = next(
            json.loads(line)
            for line in control_payload["oracle_fixtures_jsonl"].splitlines()
            if line and json.loads(line).get("case_id") == case_id
        )
        control_candidate = dict(candidate, action=case_fixture["gold_action"])
        controls, _record = _semantic_controls_payload(
            values["semantic_control_evidence"],
            expected_digest=_sha(values["semantic_control_evidence"]),
            candidate=control_candidate,
            role_bytes=role_bytes,
        )
        action = json.loads(role_bytes)["solver"]["wire"]
        transform = candidate["provenance"].get("source_transform")
        checks.append(
            {
                "case_id": case_id,
                "candidate_sha256": _sha(_canonical(candidate)),
                "state_sha256": _digest(asdict(state)),
                "source_row_sha256": _sha(_canonical(values["source_row"])),
                "role_evidence_sha256": _sha(role_bytes),
                "semantic_control_sha256": _sha(controls),
                "role_verifier_accepted": verifier_accepts,
                "reviewer_retain": reviewer.retain,
                "reviewer_ambiguous": reviewer.ambiguous,
                "candidate_action_matches_frozen_gold": candidate["action"]
                == case_fixture["gold_action"],
                "solver_wire_sha256": _sha(action.encode("utf-8")),
                "source_seed_sha256": candidate["provenance"]["source_seed_sha256"],
                "source_transform_sha256": candidate["provenance"]["source_transform_sha256"],
                "source_transform_rows": [
                    {
                        "row": edit["row"],
                        "classification": edit["classification"],
                        "old_line_sha256": edit["old_line_sha256"],
                        "new_line_sha256": edit["new_line_sha256"],
                    }
                    for edit in (transform or {}).get("edits", [])
                ],
                "sandbox_executed": False,
            }
        )
    return {
        "schema": "frozen-v8-role-source-preflight-v1",
        "status": "complete",
        "provider_calls": 0,
        "training_started": False,
        "sandbox_executed": False,
        "cases": checks,
    }


def _freeze_plan() -> dict[str, Any]:
    if any(
        path.exists() or path.is_symlink()
        for path in (PLAN_PATH, RESULT_PATH, ROWS_PATH, ROWS_MANIFEST_PATH, PACKAGE_ROOT)
    ):
        raise ValueError("qualification plan or result already exists")
    packet_plan = _load_json(PACKET / "plan.json")
    artifact_manifest = _load_json(PACKET / "artifact_manifest.json")
    if packet_plan.get("status") != "frozen_before_provider_calls":
        raise ValueError("upstream v8 plan is not frozen")
    if packet_plan.get("plan_sha256") != _digest(
        {key: value for key, value in packet_plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("upstream v8 plan self-hash is invalid")
    if artifact_manifest.get("plan_file_sha256") != _packet_hashes()["plan_file_sha256"]:
        raise ValueError("upstream artifact manifest does not pin its plan")
    if artifact_manifest.get("plan_canonical_sha256") != packet_plan["plan_sha256"]:
        raise ValueError("upstream artifact manifest canonical plan identity differs")
    tokenizer_hash = _tokenizer_hash()
    if tokenizer_hash != TOKENIZER_JSON_SHA256:
        raise ValueError("local tokenizer hash differs from the approved identity")
    actual_packet_hashes = _packet_hashes()
    expected_packet_hashes = {
        key: value
        for key, value in candidate_acceptance._FROZEN_V8_PACKET.items()
        if key != "role_event_log_hashes"
    }
    expected_packet_hashes["role_event_log_hashes"] = candidate_acceptance._FROZEN_V8_PACKET[
        "role_event_log_hashes"
    ]
    if actual_packet_hashes != expected_packet_hashes:
        raise ValueError("v8 packet hashes differ from the pinned campaign identity")
    doc = {
        "schema": "frozen-v8-muse-cpu-qualification-plan-v5",
        "status": "frozen_before_cpu_acceptance",
        "revision": 5,
        "supersedes": {
            "plan_sha256": PREVIOUS_PLAN_SHA256,
            "plan_file_sha256": PREVIOUS_PLAN_FILE_SHA256,
            "result_sha256": PREVIOUS_RESULT_SHA256,
            "result_file_sha256": PREVIOUS_RESULT_FILE_SHA256,
            "failed_plan_sha256": FAILED_V2_PLAN_SHA256,
            "failed_plan_file_sha256": FAILED_V2_PLAN_FILE_SHA256,
            "reason": (
                "v1 sandbox path mismatch; v2 verifier compared canonical hashes as file hashes"
            ),
        },
        "earlier_successful_revision": {
            "plan_sha256": SUCCESSFUL_V3_PLAN_SHA256,
            "plan_file_sha256": SUCCESSFUL_V3_PLAN_FILE_SHA256,
            "result_sha256": SUCCESSFUL_V3_RESULT_SHA256,
            "result_file_sha256": SUCCESSFUL_V3_RESULT_FILE_SHA256,
            "reason": "adds direct hash-domain and sandbox target-path regression coverage",
        },
        "previous_successful_revision": {
            "plan_sha256": SUCCESSFUL_V4_PLAN_SHA256,
            "plan_file_sha256": SUCCESSFUL_V4_PLAN_FILE_SHA256,
            "result_sha256": SUCCESSFUL_V4_RESULT_SHA256,
            "result_file_sha256": SUCCESSFUL_V4_RESULT_FILE_SHA256,
            "reason": "persists the verified canonical accepted row and package manifest",
        },
        "row_export": {
            "schema": "one-line-muse-accepted-rows-manifest-v1",
            "rows_path": str(ROWS_PATH),
            "manifest_path": str(ROWS_MANIFEST_PATH),
            "package_root": str(PACKAGE_ROOT),
            "selection": "objective+role+semantic+portable-verified rows only",
            "maximum_rows": 2,
        },
        "packet_root": str(PACKET),
        "source_scope_audit": str(LICENSE_AUDIT),
        "source_pair_root": str(SOURCE_PAIRS),
        "package_root": str(PACKAGE_ROOT),
        "cases": CASE_SPLITS,
        "provider_calls": 0,
        "training_started": False,
        "cpu_check_budget_seconds": 900,
        "tokenizer": {"path": str(TOKENIZER), "tokenizer_json_sha256": tokenizer_hash},
        "upstream": {
            "plan_canonical_sha256": packet_plan["plan_sha256"],
            **actual_packet_hashes,
        },
        "code_sha256": _file_hashes(),
        "sandbox_policy": {
            "backend": "existing code_benchmark container backend",
            "network": "disabled by existing sandbox",
            "target_path_policy": {
                "python": "solution.py",
                "typescript": "tooltip-view.ts",
                "source_repository_path_remains_provenance_only": True,
            },
            "new_weights": False,
            "models_loaded": 0,
            "expected_qualification": {
                "synthetic-a": "solver_functional_pass_and_reviewer_retain",
                "synthetic-b": "solver_functional_fail_and_reviewer_ambiguous",
            },
        },
    }
    doc["source_role_preflight"] = _no_output_preflight(doc)
    doc["source_provenance_reconciliation"] = {
        "case_id": "synthetic-a",
        "upstream_annotation": "incomplete_parent_to_author_input_transform",
        "upstream_plan_file_sha256": actual_packet_hashes["plan_file_sha256"],
        "parent_source_sha256": candidate_acceptance._FROZEN_V8_CASES["synthetic-a"][
            "source_parent_sha256"
        ],
        "author_input_row_sha256": candidate_acceptance._FROZEN_V8_CASES["synthetic-a"][
            "input_row_sha256"
        ],
        "author_seed_sha256": candidate_acceptance._FROZEN_V8_CASES["synthetic-a"][
            "source_seed_sha256"
        ],
        "parent_to_seed_transform_sha256": candidate_acceptance._FROZEN_V8_CASES["synthetic-a"][
            "source_transform_sha256"
        ],
        "declared_synthetic_prestate_row": 4,
        "additional_privacy_redaction_rows": [5, 16],
        "exact_byte_reconstruction_verified": True,
        "upstream_plan_inputs_and_prompts_unchanged": True,
        "claim_of_human_chronology": False,
    }
    doc["plan_sha256"] = _digest(doc)
    _write_once(
        PLAN_PATH, (json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    )
    return doc


def _verify_plan() -> dict[str, Any]:
    plan_bytes = _read_private(PLAN_PATH)
    plan = json.loads(plan_bytes)
    claimed = plan.pop("plan_sha256", None)
    if claimed != _digest(plan):
        raise ValueError("qualification plan self-hash mismatch")
    plan["plan_sha256"] = claimed
    packet_plan = _load_json(PACKET / "plan.json")
    artifact_manifest = _load_json(PACKET / "artifact_manifest.json")
    actual = _packet_hashes()
    expected_upstream = {
        "plan_canonical_sha256": packet_plan.get("plan_sha256"),
        **actual,
    }
    preflight_cases = plan.get("source_role_preflight", {}).get("cases", [])
    preflight_a = next(
        (row for row in preflight_cases if row.get("case_id") == "synthetic-a"), None
    )
    reconciliation = plan.get("source_provenance_reconciliation")
    _verify_self_hashed_document(
        PRIVATE_ROOT / "qualification-plan.json",
        expected_file_sha256=PREVIOUS_PLAN_FILE_SHA256,
        expected_identity_sha256=PREVIOUS_PLAN_SHA256,
        identity_key="plan_sha256",
        label="superseded v1 plan",
    )
    _verify_self_hashed_document(
        PRIVATE_ROOT / "qualification-result.json",
        expected_file_sha256=PREVIOUS_RESULT_FILE_SHA256,
        expected_identity_sha256=PREVIOUS_RESULT_SHA256,
        identity_key="result_sha256",
        label="superseded v1 result",
    )
    _verify_self_hashed_document(
        PRIVATE_ROOT / "qualification-plan-v2.json",
        expected_file_sha256=FAILED_V2_PLAN_FILE_SHA256,
        expected_identity_sha256=FAILED_V2_PLAN_SHA256,
        identity_key="plan_sha256",
        label="superseded v2 plan",
    )
    successful_v3_plan = _verify_self_hashed_document(
        PRIVATE_ROOT / "qualification-plan-v3.json",
        expected_file_sha256=SUCCESSFUL_V3_PLAN_FILE_SHA256,
        expected_identity_sha256=SUCCESSFUL_V3_PLAN_SHA256,
        identity_key="plan_sha256",
        label="superseded successful v3 plan",
    )
    _verify_self_hashed_document(
        PRIVATE_ROOT / "qualification-result-v3.json",
        expected_file_sha256=SUCCESSFUL_V3_RESULT_FILE_SHA256,
        expected_identity_sha256=SUCCESSFUL_V3_RESULT_SHA256,
        identity_key="result_sha256",
        label="superseded successful v3 result",
    )
    successful_v4_plan = _verify_self_hashed_document(
        PRIVATE_ROOT / "qualification-plan-v4.json",
        expected_file_sha256=SUCCESSFUL_V4_PLAN_FILE_SHA256,
        expected_identity_sha256=SUCCESSFUL_V4_PLAN_SHA256,
        identity_key="plan_sha256",
        label="superseded successful v4 plan",
    )
    _verify_self_hashed_document(
        PRIVATE_ROOT / "qualification-result-v4.json",
        expected_file_sha256=SUCCESSFUL_V4_RESULT_FILE_SHA256,
        expected_identity_sha256=SUCCESSFUL_V4_RESULT_SHA256,
        identity_key="result_sha256",
        label="superseded successful v4 result",
    )
    if (
        plan.get("schema") != "frozen-v8-muse-cpu-qualification-plan-v5"
        or plan.get("revision") != 5
        or plan.get("supersedes", {}).get("plan_sha256") != PREVIOUS_PLAN_SHA256
        or plan.get("supersedes", {}).get("plan_file_sha256") != PREVIOUS_PLAN_FILE_SHA256
        or plan.get("supersedes", {}).get("result_sha256") != PREVIOUS_RESULT_SHA256
        or plan.get("supersedes", {}).get("result_file_sha256") != PREVIOUS_RESULT_FILE_SHA256
        or plan.get("supersedes", {}).get("failed_plan_sha256") != FAILED_V2_PLAN_SHA256
        or plan.get("supersedes", {}).get("failed_plan_file_sha256") != FAILED_V2_PLAN_FILE_SHA256
        or plan.get("earlier_successful_revision", {}).get("plan_sha256")
        != SUCCESSFUL_V3_PLAN_SHA256
        or plan.get("earlier_successful_revision", {}).get("plan_file_sha256")
        != SUCCESSFUL_V3_PLAN_FILE_SHA256
        or plan.get("earlier_successful_revision", {}).get("result_sha256")
        != SUCCESSFUL_V3_RESULT_SHA256
        or plan.get("earlier_successful_revision", {}).get("result_file_sha256")
        != SUCCESSFUL_V3_RESULT_FILE_SHA256
        or successful_v3_plan.get("schema") != "frozen-v8-muse-cpu-qualification-plan-v3"
        or plan.get("previous_successful_revision", {}).get("plan_sha256")
        != SUCCESSFUL_V4_PLAN_SHA256
        or plan.get("previous_successful_revision", {}).get("plan_file_sha256")
        != SUCCESSFUL_V4_PLAN_FILE_SHA256
        or plan.get("previous_successful_revision", {}).get("result_sha256")
        != SUCCESSFUL_V4_RESULT_SHA256
        or plan.get("previous_successful_revision", {}).get("result_file_sha256")
        != SUCCESSFUL_V4_RESULT_FILE_SHA256
        or successful_v4_plan.get("schema") != "frozen-v8-muse-cpu-qualification-plan-v4"
        or plan.get("row_export")
        != {
            "schema": "one-line-muse-accepted-rows-manifest-v1",
            "rows_path": str(ROWS_PATH),
            "manifest_path": str(ROWS_MANIFEST_PATH),
            "package_root": str(PACKAGE_ROOT),
            "selection": "objective+role+semantic+portable-verified rows only",
            "maximum_rows": 2,
        }
        or (PRIVATE_ROOT / "qualification-result-v2.json").exists()
        or plan.get("status") != "frozen_before_cpu_acceptance"
        or plan.get("upstream") != expected_upstream
        or plan.get("code_sha256") != _file_hashes()
        or plan.get("tokenizer", {}).get("tokenizer_json_sha256") != _tokenizer_hash()
        or artifact_manifest.get("plan_canonical_sha256") != packet_plan.get("plan_sha256")
        or plan.get("source_role_preflight") != _no_output_preflight(plan)
        or plan.get("sandbox_policy", {}).get("target_path_policy")
        != {
            "python": "solution.py",
            "typescript": "tooltip-view.ts",
            "source_repository_path_remains_provenance_only": True,
        }
        or not isinstance(preflight_a, dict)
        or not isinstance(reconciliation, dict)
        or reconciliation.get("author_seed_sha256") != preflight_a.get("source_seed_sha256")
        or reconciliation.get("parent_to_seed_transform_sha256")
        != preflight_a.get("source_transform_sha256")
        or [
            (row.get("row"), row.get("classification"))
            for row in preflight_a.get("source_transform_rows", [])
        ]
        != [
            (4, "synthetic_prestate_signature"),
            (5, "privacy_redaction"),
            (16, "privacy_redaction"),
        ]
    ):
        raise ValueError("frozen qualification input identity changed")
    return plan


def _find_content(root: Path, digest: str, *, suffix: str) -> tuple[Path, bytes]:
    found = []
    for path in root.glob(f"*.{suffix}"):
        data = _read_private(path, max_bytes=32_000)
        if _sha(data) == digest:
            found.append((path, data))
    if len(found) != 1:
        raise ValueError("frozen role artifact is missing or ambiguous")
    return found[0]


def _source_parent(case: dict[str, Any]) -> str:
    matches = list(SOURCE_PAIRS.glob(f"*/parent-{case['parent_source_sha256']}.src"))
    if len(matches) != 1:
        raise ValueError("exact public parent source is missing or ambiguous")
    data = _read_private(matches[0], max_bytes=32_000)
    if _sha(data) != case["parent_source_sha256"]:
        raise ValueError("public parent source bytes do not match the v8 plan")
    return data.decode("utf-8")


def _source_transform(
    *, parent: str, seed: str, case: dict[str, Any], plan_file_sha256: str, inputs_sha256: str
) -> dict[str, Any] | None:
    spec = case.get("author_source_transform")
    if spec is None:
        if parent != seed:
            raise ValueError("unplanned synthetic source transformation")
        return None
    if (
        not isinstance(spec, dict)
        or spec.get("kind") != "synthetic_single_line_prestate_from_public_source"
    ):
        raise ValueError("unsupported frozen synthetic source transform")
    parent_lines = list(physical_lines(parent.encode("utf-8")))
    seed_lines = list(physical_lines(seed.encode("utf-8")))
    row = spec.get("row")
    if type(row) is not int or row >= len(parent_lines) or len(parent_lines) != len(seed_lines):
        raise ValueError("synthetic public-source transform has invalid line layout")

    edits: list[dict[str, Any]] = []
    for index, (old, new) in enumerate(zip(parent_lines, seed_lines, strict=True)):
        if old.terminator != new.terminator:
            raise ValueError("synthetic source transform changes line endings")
        old_text, new_text = old.content.decode("utf-8"), new.content.decode("utf-8")
        if old_text == new_text:
            continue
        if index == row:
            classification = "synthetic_prestate_signature"
        elif "<redacted-" in new_text and ">" in new_text:
            classification = "privacy_redaction"
        else:
            raise ValueError("unclassified public-source input transformation")
        edits.append(
            {
                "row": index,
                "old_line": old_text,
                "new_line": new_text,
                "old_line_sha256": _sha(old_text.encode("utf-8")),
                "new_line_sha256": _sha(new_text.encode("utf-8")),
                "classification": classification,
            }
        )
    declared = next((edit for edit in edits if edit["row"] == row), None)
    if declared is None or (
        declared["old_line_sha256"] != spec.get("public_snapshot_line_sha256")
        or declared["new_line_sha256"] != spec.get("synthetic_author_line_sha256")
    ):
        raise ValueError("synthetic source transform line hash differs from frozen plan")
    return {
        "schema": "one-line-public-source-transform-v2",
        "kind": "replace_explicit_physical_lines_for_synthetic_public_source",
        "parent_source_sha256": _sha(parent.encode()),
        "seed_source_sha256": _sha(seed.encode()),
        "edits": edits,
        "signature_prestate_row": row,
        "privacy_redaction_rows": [
            edit["row"] for edit in edits if edit["classification"] == "privacy_redaction"
        ],
        "plan_file_sha256": plan_file_sha256,
        "input_bundle_sha256": inputs_sha256,
        "upstream_declared_transform": spec,
    }


def _role_evidence(
    *,
    packet_root: Path,
    case_id: str,
    candidate: dict[str, Any],
    tokenizer: Any,
    source_parent_sha256: str,
    input_row_sha256: str,
    source_seed_sha256: str,
    transform_sha256: str,
) -> bytes:
    plan_bytes = _read_private(packet_root / "plan.json")
    plan = json.loads(plan_bytes)
    manifest_bytes = _read_private(packet_root / "artifact_manifest.json")
    json.loads(manifest_bytes)
    inputs_bytes = _read_private(packet_root / "source_only_inputs.jsonl")
    fixture_bytes = _read_private(packet_root / "oracle_fixtures.jsonl")
    preflight_bytes = _read_private(packet_root / "oracle_preflight_result.json")
    qualification_bytes = _read_private(packet_root / "execution-v8/qualification.json")
    qualification_events = _read_private(packet_root / "execution-v8/qualification_events.jsonl")
    audit_bytes = _read_private(LICENSE_AUDIT)
    model_id = str(plan["provider"]["model_id"])
    role_logs = {
        role: _read_private(packet_root / f"execution-v8/{role}_events.jsonl").decode("utf-8")
        for role in ("author", "solver", "reviewer")
    }
    event_rows = {
        role: [json.loads(line) for line in raw.splitlines() if line]
        for role, raw in role_logs.items()
    }
    request_ids = {
        role: plan["provider"]["request_ids"][f"{case_id}:{role}"]
        for role in ("author", "solver", "reviewer")
    }
    chosen: dict[str, dict[str, Any]] = {}
    prompts: dict[str, str] = {}
    responses: dict[str, str] = {}
    response_evidence: dict[str, str] = {}
    for role in ("author", "solver", "reviewer"):
        matches = [
            row
            for row in event_rows[role]
            if row.get("request_id") == request_ids[role] and row.get("source_id") == case_id
        ]
        if len(matches) != 1:
            raise ValueError("frozen role event is not unique")
        event = matches[0]
        if (
            event.get("role") != role
            or event.get("finish_reason") != "stop"
            or not event.get("output_available")
        ):
            raise ValueError("frozen role event did not complete successfully")
        _prompt_path, prompt_bytes = _find_content(
            packet_root / "execution-v8/prompts", event["prompt_sha256"], suffix="txt"
        )
        _response_path, response_bytes = _find_content(
            packet_root / "execution-v8/responses", event["output_sha256"], suffix="txt"
        )
        evidence_path = next(
            path
            for path in (packet_root / "execution-v8/completed_response_evidence").glob("*.json")
            if _sha(_read_private(path)) == event["response_evidence_sha256"]
        )
        evidence_bytes = _read_private(evidence_path)
        chosen[role] = event
        prompts[role] = prompt_bytes.decode("utf-8")
        responses[role] = response_bytes.decode("utf-8")
        response_evidence[role] = evidence_bytes.decode("utf-8")
        if role in {"author", "solver"}:
            declared = plan["prompts"][case_id][role]["sha256"]
            if declared != event["prompt_sha256"]:
                raise ValueError("author or solver prompt differs from the frozen plan")
    role_prompt_hashes = {role: chosen[role]["prompt_sha256"] for role in chosen}
    role_event_log_hashes = {role: _sha(role_logs[role].encode("utf-8")) for role in role_logs}
    packet_binding = {
        "schema": "two-seed-v8-role-packet-binding-v1",
        "case_id": case_id,
        "model_id": model_id,
        "plan_file_sha256": _sha(plan_bytes),
        "plan_canonical_sha256": plan["plan_sha256"],
        "artifact_manifest_sha256": _sha(manifest_bytes),
        "inputs_sha256": _sha(inputs_bytes),
        "oracle_fixtures_sha256": _sha(fixture_bytes),
        "preflight_result_sha256": _sha(preflight_bytes),
        "license_scope_audit_sha256": _sha(audit_bytes),
        "qualification_sha256": _sha(qualification_bytes),
        "qualification_events_sha256": _sha(qualification_events),
        "role_event_log_hashes": role_event_log_hashes,
        "role_prompt_hashes": role_prompt_hashes,
        "role_request_ids": request_ids,
        "source_parent_sha256": source_parent_sha256,
        "input_row_sha256": input_row_sha256,
        "source_seed_sha256": source_seed_sha256,
        "source_transform_sha256": transform_sha256,
    }
    candidate["provenance"]["frozen_v8_packet_binding"] = packet_binding
    roles: dict[str, Any] = {}
    for role in ("author", "solver", "reviewer"):
        event = chosen[role]
        item: dict[str, Any] = {
            "actor_id": f"{model_id}:{role}",
            "session_id": event["session_id"],
            "request_id": event["request_id"],
            "response_id": event["response_id"],
            "model_id": model_id,
            "prompt_text": prompts[role],
            "prompt_sha256": event["prompt_sha256"],
            "response_text": responses[role],
            "response_sha256": event["output_sha256"],
            "finish_reason": event["finish_reason"],
            "event_log": role_logs[role],
            "response_evidence_json": response_evidence[role],
        }
        if role == "solver":
            extracted = extract_final_action_block_v7(
                responses[role], provider_complete=True, tokenizer=tokenizer
            )
            item.update(
                {
                    "wire": extracted.wire,
                    "generated_tokens": extracted.wire_plus_q25_eos_tokens,
                    "terminated": True,
                }
            )
        elif role == "reviewer":
            verdict = parse_review_response(responses[role])
            item.update(
                {
                    "verdict_json": responses[role],
                    "verdict_sha256": _sha(responses[role].encode("utf-8")),
                    "retain": verdict.retain,
                    "ambiguous": verdict.ambiguous,
                }
            )
        roles[role] = item
    # The generic role schema remains strict; reviewer booleans are parsed
    # from the raw verdict and checked by the canonical parser, not extra keys.
    roles["reviewer"].pop("retain")
    roles["reviewer"].pop("ambiguous")
    payload = {
        "schema": "one-line-role-evidence-frozen-v8-v1",
        "candidate_id": candidate["id"],
        "packet": packet_binding,
        **roles,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _case_inputs(case_id: str, tokenizer: Any, plan: dict[str, Any]) -> dict[str, Any]:
    packet_plan = _load_json(PACKET / "plan.json")
    input_bytes = _read_private(PACKET / "source_only_inputs.jsonl")
    input_rows = [
        (json.loads(line.decode("utf-8")), line)
        for line in input_bytes.splitlines(keepends=True)
        if line.strip()
    ]
    fixture_rows = [
        json.loads(line)
        for line in _read_private(PACKET / "oracle_fixtures.jsonl").decode().splitlines()
        if line
    ]
    inputs = [row for row in input_rows if row[0].get("case_id") == case_id]
    fixtures = [row for row in fixture_rows if row.get("case_id") == case_id]
    if len(inputs) != 1 or len(fixtures) != 1:
        raise ValueError("frozen source or objective row is not unique")
    input_row, input_row_bytes = inputs[0]
    fixture = fixtures[0]
    case = packet_plan["cases"][case_id]
    audit_bytes = _read_private(LICENSE_AUDIT)
    audit = json.loads(audit_bytes.decode("utf-8"))
    audit_rows = [row for row in audit["cases"] if row.get("case_id") == case_id]
    if len(audit_rows) != 1:
        raise ValueError("pinned file-license scope audit row is absent")
    audit_row = audit_rows[0]
    parent_source = _source_parent(
        {
            "parent_source_sha256": case["parent_source_sha256"],
        }
    )
    seed_source = input_row["author_source_text"]
    transform = _source_transform(
        parent=parent_source,
        seed=seed_source,
        case=case,
        plan_file_sha256=plan["upstream"]["plan_file_sha256"],
        inputs_sha256=plan["upstream"]["inputs_sha256"],
    )
    source_sha256 = _sha(parent_source.encode("utf-8"))
    seed_sha256 = _sha(seed_source.encode("utf-8"))
    if (
        audit_row.get("source_raw_sha256") != source_sha256
        or audit_row.get("root_license_raw_sha256") != case["root_license_sha256"]
        or audit_row.get("source_path") != case["path"]
        or audit_row.get("repository") != case["repo"]
        or audit_row.get("tree_recursive_complete") is not True
        or audit_row.get("expected_source_sha256_match") is not True
    ):
        raise ValueError("file-license scope audit does not bind to this public source")
    transform_sha = _digest(transform)
    md = {
        "source_repo": case["repo"],
        "source_aliases": [case["repo"]],
        "source_revision": case["parent"],
        "source_path": case["path"],
        "source_url": f"https://github.com/{case['repo']}/blob/{case['parent']}/{case['path']}",
        "source_sha256": source_sha256,
        "source_license": "MIT",
        "license_path": audit_row["root_license_path"],
        "license_sha256": audit_row["root_license_raw_sha256"],
        "source_provenance_verified": True,
        "file_license_scope_unverified_without_notice": True,
        "authoring_focus": None,
        "synthetic_seed_sha256": seed_sha256,
        "synthetic_source_transform_sha256": transform_sha,
    }
    source_row: dict[str, Any] = {
        "id": case_id,
        "student_state_seed": {
            "file_id": input_row["state"]["file_id"],
            "filetype": input_row["state"]["filetype"],
            "source": seed_source,
        },
        "authoring_metadata": md,
        "public_source_parent": parent_source,
    }
    if transform is not None:
        source_row["synthetic_source_transform"] = transform
    role_author_event = next(
        row
        for row in (
            json.loads(line)
            for line in _read_private(PACKET / "execution-v8/author_events.jsonl")
            .decode()
            .splitlines()
            if line
        )
        if row.get("source_id") == case_id
    )
    response_path, author_response_bytes = _find_content(
        PACKET / "execution-v8/responses", role_author_event["output_sha256"], suffix="txt"
    )
    del response_path
    author_response = author_response_bytes.decode("utf-8")
    extracted = extract_author_candidate_block(author_response, provider_complete=True)
    # parse_author_response expects the source hash to describe its seed; keep
    # this transient parse row separate from the immutable public-parent row.
    parse_row = json.loads(json.dumps(source_row))
    parse_row["authoring_metadata"]["source_sha256"] = seed_sha256
    candidate = parse_author_response(extracted.json_text, parse_row, tokenizer)
    state = EditState.from_mapping(candidate["state"])
    expected_state = EditState.from_mapping(input_row["state"])
    candidate["state"] = asdict(
        state.__class__(
            file_id=state.file_id,
            filetype=state.filetype,
            source=state.source,
            target_row=state.target_row,
            cursor_col=state.cursor_col,
            history=state.history,
            relevant=expected_state.relevant,
        )
    )
    state = EditState.from_mapping(candidate["state"])
    if (
        state != expected_state
        or replay_replacement_history(
            seed_source, state.history, file_id=state.file_id, filetype=state.filetype
        )
        != state.source
        or state.source != input_row["state"]["source"]
        or state.target_row != case["target_row"]
    ):
        raise ValueError("frozen author state/history does not reconstruct the source input")
    candidate["source_repo"] = case["repo"]
    candidate["source_revision"] = case["parent"]
    candidate["source_license"] = "MIT"
    candidate["source_aliases"] = [case["repo"]]
    candidate["source_path"] = case["path"]
    candidate["provenance"]["source_sha256"] = source_sha256
    candidate["provenance"]["source_license_sha256"] = audit_row["root_license_raw_sha256"]
    candidate["provenance"]["source_only_input_row_sha256"] = _sha(input_row_bytes)
    candidate["provenance"]["source_seed_sha256"] = seed_sha256
    candidate["provenance"]["source_transform_sha256"] = transform_sha
    candidate["provenance"]["author_actor_id"] = f"{packet_plan['provider']['model_id']}:author"
    candidate["provenance"]["human_edit_order_observed"] = False
    candidate["validation"]["human_chronology_observed"] = False
    candidate["validation"]["history_replays_to_state"] = True
    candidate["validation"]["history_visible_in_prompt"] = True
    candidate["validation"]["blind_solver_verified"] = True
    candidate["validation"]["reviewer_verified"] = True
    candidate["split"] = CASE_SPLITS[case_id]
    candidate["split_manifest_sha256"] = "0" * 64
    split_data = json.dumps(
        {
            "schema": "one-line-accepted-split-v1",
            "status": "frozen",
            "assignments": [
                {
                    "candidate_id": candidate["id"],
                    "source_repo": case["repo"],
                    "split": CASE_SPLITS[case_id],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    candidate["split_manifest_sha256"] = _sha(split_data)
    candidate["source_group_id"] = f"repo:{case['repo'].casefold()}"
    candidate["task_family_id"] = f"frozen-v8-public-source/{case['repo']}"
    candidate["template_id"] = f"frozen-v8/{case_id}"
    candidate["generator_family"] = "muse-frozen-v8-public-source"
    candidate["session_or_commit"] = case["parent"]
    candidate["provenance"]["source_id"] = case_id
    candidate["provenance"]["source_transform"] = transform
    objective_manifest = json.dumps(
        {
            "schema": "one-line-independent-objective-v1",
            "reviewer_id": "pinned-v8-sandbox-objective",
            "entries": [
                _objective_entry(
                    candidate_id=candidate["id"],
                    source_id=case_id,
                    filetype=state.filetype,
                    fixture=fixture,
                )
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    role_bytes = _role_evidence(
        packet_root=PACKET,
        case_id=case_id,
        candidate=candidate,
        tokenizer=tokenizer,
        source_parent_sha256=source_sha256,
        input_row_sha256=_sha(input_row_bytes),
        source_seed_sha256=seed_sha256,
        transform_sha256=transform_sha,
    )
    license_review = json.dumps(
        {
            "schema": "one-line-file-license-review-v1",
            "source_id": case_id,
            "source_repo": case["repo"],
            "source_revision": case["parent"],
            "source_sha256": source_sha256,
            "source_path": case["path"],
            "license_spdx": "MIT",
            "license_sha256": audit_row["root_license_raw_sha256"],
            "reviewer_id": "pinned-public-source-scope-audit-v2",
            "finding": "license_covers_file",
            "evidence": audit_row["scope_assessment"] + "; audit sha256=" + _sha(audit_bytes),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    preflight_text = _read_private(PACKET / "oracle_preflight_result.json").decode("utf-8")
    fixture_text = _read_private(PACKET / "oracle_fixtures.jsonl").decode("utf-8")
    preflight = json.loads(preflight_text)
    selected_ids = [
        row["case_action_id"] for row in preflight["cases"] if row["case_id"] == case_id
    ]
    role_payload = json.loads(role_bytes)
    semantic_controls = json.dumps(
        {
            "schema": "one-line-frozen-v8-semantic-controls-v1",
            "candidate_id": candidate["id"],
            "packet_binding": role_payload["packet"],
            "oracle_fixtures_jsonl": fixture_text,
            "preflight_result_json": preflight_text,
            "selected_case_fixture_sha256": _digest(fixture),
            "selected_control_case_ids": selected_ids,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return {
        "candidate": candidate,
        "source_row": source_row,
        "objective_manifest": objective_manifest,
        "split_manifest": split_data,
        "role_evidence": role_bytes,
        "file_license_review": license_review,
        "semantic_control_evidence": semantic_controls,
    }


def _execute() -> dict[str, Any]:
    plan = _verify_plan()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    result_cases = []
    accepted_rows: list[dict[str, Any]] = []
    start_ns = __import__("time").time_ns()
    for case_id in ("synthetic-a", "synthetic-b"):
        values = _case_inputs(case_id, tokenizer, plan)
        outcome = export_muse_acceptance_package(
            package_root=PACKAGE_ROOT,
            tokenizer=tokenizer,
            expected_semantic_control_sha256=_sha(values["semantic_control_evidence"]),
            **values,
        )
        record: dict[str, Any] = {
            "case_id": case_id,
            "candidate_id": values["candidate"]["id"],
            "decision": "accepted" if outcome.accepted else "rejected",
            "reason": outcome.reason,
            "evidence": outcome.evidence,
        }
        if outcome.accepted:
            assert outcome.acceptance_package_ref is not None and outcome.candidate_row is not None
            metadata = values["source_row"]["authoring_metadata"]
            row = {
                **outcome.candidate_row,
                "candidate_id": outcome.candidate_row["id"],
                "source_path": metadata["source_path"],
                "source_license_sha256": metadata["license_sha256"],
                "source_sha256": metadata["source_sha256"],
                "source_license": metadata["source_license"],
                "human_chronology_observed": False,
                "history_sha256": _digest(values["candidate"]["state"]["history"]),
                "context_sha256": _sha(
                    __import__(
                        "tinycomplete.one_line.context", fromlist=["serialize_state_bounded"]
                    )
                    .serialize_state_bounded(
                        EditState.from_mapping(values["candidate"]["state"]),
                        tokenizer,
                        max_input_tokens=1024,
                    )
                    .text.encode()
                ),
                "acceptance_package_ref": outcome.acceptance_package_ref,
            }
            portable = verify_muse_candidate_row(
                row,
                package_root=PACKAGE_ROOT,
                tokenizer=tokenizer,
                existing_rows=accepted_rows,
            )
            record["portable_verification"] = {
                "accepted": portable.accepted,
                "reason": portable.reason,
                "evidence": portable.evidence,
            }
            if not portable.accepted:
                raise ValueError("portable training-side verification failed after CPU export")
            accepted_rows.append(row)
            record["row_sha256"] = _sha(_canonical(row))
            record["package_manifest_sha256"] = outcome.acceptance_package_ref["manifest_sha256"]
            record["semantic_control_sha256"] = hashlib.sha256(
                values["semantic_control_evidence"]
            ).hexdigest()
        result_cases.append(record)
    elapsed = (__import__("time").time_ns() - start_ns) / 1_000_000_000
    if elapsed > 900:
        raise ValueError("CPU qualification exceeded its frozen 15-minute budget")
    result = {
        "schema": "frozen-v8-muse-cpu-qualification-result-v5",
        "plan_sha256": plan["plan_sha256"],
        "status": "complete",
        "elapsed_seconds": elapsed,
        "provider_calls": 0,
        "training_started": False,
        "cases": result_cases,
        "_accepted_rows_for_export": accepted_rows,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.freeze:
            plan = _freeze_plan()
            print(json.dumps({"status": plan["status"], "plan_sha256": plan["plan_sha256"]}))
            return 0
        if any(
            path.exists() or path.is_symlink()
            for path in (RESULT_PATH, ROWS_PATH, ROWS_MANIFEST_PATH)
        ):
            raise ValueError("qualification result already exists")
        result = _execute()
        accepted_rows = result.pop("_accepted_rows_for_export")
        rows_payload = b"".join(_canonical(row) + b"\n" for row in accepted_rows)
        row_packages = [
            {
                "candidate_id": row["id"],
                "root": row["acceptance_package_ref"]["root"],
                "manifest_sha256": row["acceptance_package_ref"]["manifest_sha256"],
                "candidate_sha256": row["acceptance_package_ref"]["candidate_sha256"],
                "source_row_sha256": row["acceptance_package_ref"]["source_row_sha256"],
            }
            for row in accepted_rows
        ]
        row_manifest: dict[str, Any] = {
            "schema": "one-line-muse-accepted-rows-manifest-v1",
            "plan_sha256": result["plan_sha256"],
            "accepted_rows_sha256": _sha(rows_payload),
            "accepted_row_count": len(accepted_rows),
            "accepted_candidate_ids": [row["id"] for row in accepted_rows],
            "packages": row_packages,
            "provider_calls": 0,
            "training_started": False,
        }
        row_manifest["manifest_sha256"] = _digest(row_manifest)
        row_manifest_payload = _canonical(row_manifest) + b"\n"
        _write_once(ROWS_PATH, rows_payload)
        _write_once(ROWS_MANIFEST_PATH, row_manifest_payload)
        result.update(
            {
                "accepted_row_count": len(accepted_rows),
                "accepted_candidate_ids": [row["id"] for row in accepted_rows],
                "accepted_rows_sha256": _sha(rows_payload),
                "accepted_rows_manifest_sha256": row_manifest["manifest_sha256"],
                "accepted_rows_manifest_file_sha256": _sha(row_manifest_payload),
            }
        )
        result["result_sha256"] = _digest(result)
        _write_once(
            RESULT_PATH,
            (json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(),
        )
        summary = [
            {"case_id": row["case_id"], "decision": row["decision"], "reason": row["reason"]}
            for row in result["cases"]
        ]
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "cases": summary,
                    "result_sha256": result["result_sha256"],
                }
            )
        )
        return 0
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        print("Frozen v8 CPU qualification failed: frozen input or acceptance check invalid")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
