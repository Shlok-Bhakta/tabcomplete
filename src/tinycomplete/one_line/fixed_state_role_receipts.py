"""Portable verification and export for fixed-state role-run receipts.

The exporter reads an already completed local role run and copies only the
selected candidate's role receipts plus metadata-only events. It never exports
the global usage ledger. The verifier is provider-, ledger-, and sandbox-free.

These artifacts establish consistency with the locally audited runner and
root-release files. They are not a cryptographic signature from the provider.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from tinycomplete.one_line.contract import EditState
from tinycomplete.one_line.teacher import MODEL_ID

if TYPE_CHECKING:
    from tinycomplete.one_line.muse_acceptance_package import AcceptanceDecision

ROLE_PROOF_SCHEMA = "fixed-state-role-execution-proof-v1"
ROLE_PROOF_REF_SCHEMA = "fixed-state-role-execution-proof-ref-v1"
PLAN_PROJECTION_SCHEMA = "fixed-state-role-plan-projection-v1"
RECEIPT_SCHEMA = "fixed-state-completed-provider-response-v1"
PLAN_SCHEMA = "fixed-state-role-pilot-plan-v1"
RELEASE_SCHEMA = "fixed-state-role-phase-release-v1"
EVENT_SCHEMA = "fixed-state-role-event-v1"
MAX_PROOF_FILE_BYTES = 512 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:/@+-]{1,160}\Z")
_ROLES = ("author", "solver", "reviewer")
_ACTOR_SUFFIX = {
    "author": "fixed-state-author-v1",
    "solver": "fixed-state-blind-solver-v1",
    "reviewer": "fixed-state-independent-reviewer-v1",
}
_FILES = {
    "root_release": "root_release.json",
    "plan_projection": "plan_projection.json",
    "event_chain": "events.jsonl",
    "role_evidence": "role_evidence.json",
    "author_receipt": "receipts/author.json",
    "solver_receipt": "receipts/solver.json",
    "reviewer_receipt": "receipts/reviewer.json",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_unique(payload: bytes, *, maximum: int = MAX_PROOF_FILE_BYTES) -> Any:
    if not payload or len(payload) > maximum:
        raise ValueError("fixed-state proof JSON is absent or unbounded")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("fixed-state proof JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("fixed-state proof JSON is malformed") from None


def _digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _safe_relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("fixed-state proof path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("fixed-state proof path escapes its package")
    return path


def _read_relative(root: Path, relative: object, *, maximum: int) -> bytes:
    parts = _safe_relative(relative)
    if root.is_symlink():
        raise ValueError("fixed-state proof package root is a symlink")
    current = root
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("fixed-state proof artifact path contains a symlink")
    info = current.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError("fixed-state proof artifact is not a bounded regular file")
    payload = current.read_bytes()
    if len(payload) != info.st_size:
        raise ValueError("fixed-state proof artifact changed while reading")
    return payload


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ValueError("fixed-state proof output cannot be a symlink")
    if path.exists():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("existing fixed-state proof output is not private")
        if path.read_bytes() != payload:
            raise ValueError("existing immutable fixed-state proof differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=".role-proof-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _private_output_directory(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise ValueError("fixed-state proof output directory cannot be a symlink")
    if not path.exists():
        if not create:
            raise ValueError("fixed-state proof output directory is absent")
        path.mkdir(mode=0o700)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("fixed-state proof output directory must be owner-only")


def _load_runner():
    script = Path(__file__).resolve().parents[3] / "scripts/run_fixed_state_role_pilot.py"
    spec = importlib.util.spec_from_file_location("fixed_state_role_pilot_receipt_export", script)
    if spec is None or spec.loader is None:
        raise ValueError("fixed-state role runner is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Remove global ledger path and rows while retaining release-checkable facts."""
    ledger = plan.get("ledger")
    packet = plan.get("packet")
    provider = plan.get("provider")
    cases = plan.get("cases")
    if (
        not isinstance(ledger, Mapping)
        or not isinstance(packet, Mapping)
        or not isinstance(provider, Mapping)
        or not isinstance(cases, list)
    ):
        raise ValueError("fixed-state plan is incomplete")
    projected_ledger = {
        "baseline_sha256": ledger.get("baseline_sha256"),
        "baseline_request_ids_sha256": ledger.get("baseline_request_ids_sha256"),
        "baseline_totals": ledger.get("baseline_totals"),
        "global_caps": ledger.get("global_caps"),
    }
    required_top = (
        "phase_id",
        "model_id",
        "protocol_schema",
        "wire_version",
        "source_type",
        "history_order",
        "human_chronology_observed",
        "context_policy_version",
        "prompt_policy_version",
        "review_prompt_policy_version",
        "max_context_tokens",
        "max_action_supervised_tokens_including_eos",
        "max_reviewer_context_tokens",
        "max_reviewer_supervised_tokens_including_eos",
        "runtime",
        "source_hashes",
    )
    projection: dict[str, Any] = {
        "schema": PLAN_PROJECTION_SCHEMA,
        "original_plan_sha256": plan.get("plan_sha256"),
        **{key: plan.get(key) for key in required_top},
        "packet": {
            "sha256": packet.get("sha256"),
            "bytes": packet.get("bytes"),
            "case_count": packet.get("case_count"),
            "candidate_ids": packet.get("candidate_ids"),
        },
        "cases": [
            {
                key: case.get(key)
                for key in (
                    "candidate_id",
                    "seed_id",
                    "state_sha256",
                    "history_sha256",
                    "source_sha256",
                    "context_sha256",
                    "student_prompt_sha256",
                    "student_input_tokens",
                    "request_ids",
                )
            }
            for case in cases
            if isinstance(case, Mapping)
        ],
        "provider": {
            key: provider.get(key)
            for key in (
                "max_phase_calls",
                "planned_calls",
                "role_order_per_case",
                "max_wall_seconds",
                "finalization_reserve_seconds",
                "reserve_input_tokens_per_request",
                "reserve_output_tokens_per_request",
                "source_class",
                "sequential_single_active_request",
                "on_transport_or_unknown_usage",
                "on_non_stop_or_invalid_output",
                "automatic_fallback",
            )
        },
        "ledger_summary": projected_ledger,
        "qualification_boundary": plan.get("qualification_boundary"),
    }
    projected_provider = projection.get("provider")
    if not isinstance(projected_provider, Mapping):
        raise ValueError("fixed-state projected provider policy is invalid")
    if (
        not _digest(projection["original_plan_sha256"])
        or projected_provider.get("automatic_fallback") is not False
        or projected_provider.get("source_class") != "public"
        or projection.get("qualification_boundary")
        != {
            "provider_roles_are_inferability_evidence_only": True,
            "functional_oracle_status": "not_evaluated_by_role_runner",
            "training_started": False,
            "training_accepted": False,
            "personalization_enabled": False,
            "human_chronology_observed": False,
        }
    ):
        raise ValueError("fixed-state plan projection violates the role-only boundary")
    return projection


def _validate_release_projection(release: Mapping[str, Any], projection: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema",
        "plan_sha256",
        "baseline_ledger_sha256",
        "baseline_request_ids_sha256",
        "planned_request_ids",
        "max_calls",
        "max_wall_seconds",
        "reserve_input_tokens",
        "reserve_output_tokens",
        "global_caps",
        "released_by",
        "released_at_utc",
    }
    if set(release) != expected_fields:
        raise ValueError("fixed-state root release fields changed")
    cases = projection["cases"]
    ids = [case["request_ids"][role] for case in cases for role in _ROLES]
    provider = projection["provider"]
    ledger = projection["ledger_summary"]
    if (
        release.get("schema") != RELEASE_SCHEMA
        or release.get("plan_sha256") != projection.get("original_plan_sha256")
        or release.get("baseline_ledger_sha256") != ledger.get("baseline_sha256")
        or release.get("baseline_request_ids_sha256")
        != ledger.get("baseline_request_ids_sha256")
        or release.get("planned_request_ids") != ids
        or release.get("max_calls") != provider.get("planned_calls")
        or type(release.get("max_wall_seconds")) is not int
        or release.get("max_wall_seconds", 0)
        < provider.get("finalization_reserve_seconds", 0) + 120
        or release.get("max_wall_seconds", 0) > provider.get("max_wall_seconds", 0)
        or release.get("reserve_input_tokens")
        != provider.get("reserve_input_tokens_per_request")
        or release.get("reserve_output_tokens")
        != provider.get("reserve_output_tokens_per_request")
        or release.get("global_caps") != ledger.get("global_caps")
        or not isinstance(release.get("released_by"), str)
        or not isinstance(release.get("released_at_utc"), str)
    ):
        raise ValueError("fixed-state root release differs from projected role plan")


def _validate_plan_projection(projection: Mapping[str, Any]) -> None:
    expected_top = {
        "schema",
        "original_plan_sha256",
        "phase_id",
        "model_id",
        "protocol_schema",
        "wire_version",
        "source_type",
        "history_order",
        "human_chronology_observed",
        "context_policy_version",
        "prompt_policy_version",
        "review_prompt_policy_version",
        "max_context_tokens",
        "max_action_supervised_tokens_including_eos",
        "max_reviewer_context_tokens",
        "max_reviewer_supervised_tokens_including_eos",
        "runtime",
        "source_hashes",
        "packet",
        "cases",
        "provider",
        "ledger_summary",
        "qualification_boundary",
    }
    if set(projection) != expected_top:
        raise ValueError("fixed-state sanitized plan projection fields changed")
    if (
        projection.get("model_id") != MODEL_ID
        or projection.get("protocol_schema") != "one-line-fixed-state-roles-v1"
        or projection.get("wire_version") != "single-line-edit-v1"
        or projection.get("source_type") != "reviewed_public_history_candidate"
        or projection.get("history_order") != "synthetic_fixed_before_provider"
        or projection.get("human_chronology_observed") is not False
        or not isinstance(projection.get("phase_id"), str)
        or not _ID.fullmatch(str(projection.get("phase_id")))
    ):
        raise ValueError("fixed-state sanitized plan protocol identity is invalid")
    packet = projection.get("packet")
    provider = projection.get("provider")
    ledger = projection.get("ledger_summary")
    cases = projection.get("cases")
    if (
        not isinstance(packet, Mapping)
        or set(packet) != {"sha256", "bytes", "case_count", "candidate_ids"}
        or not _digest(packet.get("sha256"))
        or type(packet.get("bytes")) is not int
        or type(packet.get("case_count")) is not int
        or not isinstance(packet.get("candidate_ids"), list)
        or not isinstance(provider, Mapping)
        or not isinstance(ledger, Mapping)
        or not isinstance(cases, list)
        or len(cases) != packet.get("case_count")
        or len(cases) != len(packet.get("candidate_ids", []))
        or not 1 <= len(cases) <= 8
    ):
        raise ValueError("fixed-state sanitized plan packet/provider fields are invalid")
    expected_provider = {
        "max_phase_calls",
        "planned_calls",
        "role_order_per_case",
        "max_wall_seconds",
        "finalization_reserve_seconds",
        "reserve_input_tokens_per_request",
        "reserve_output_tokens_per_request",
        "source_class",
        "sequential_single_active_request",
        "on_transport_or_unknown_usage",
        "on_non_stop_or_invalid_output",
        "automatic_fallback",
    }
    if set(provider) != expected_provider or (
        provider.get("planned_calls") != len(cases) * 3
        or provider.get("max_phase_calls") != 24
        or provider.get("role_order_per_case") != list(_ROLES)
        or provider.get("source_class") != "public"
        or provider.get("sequential_single_active_request") is not True
        or provider.get("on_transport_or_unknown_usage") != "stop_no_retry"
        or provider.get("automatic_fallback") is not False
    ):
        raise ValueError("fixed-state sanitized provider policy is invalid")
    if set(ledger) != {
        "baseline_sha256",
        "baseline_request_ids_sha256",
        "baseline_totals",
        "global_caps",
    } or not _digest(ledger.get("baseline_sha256")) or not _digest(
        ledger.get("baseline_request_ids_sha256")
    ):
        raise ValueError("fixed-state sanitized ledger summary is invalid")
    case_ids: set[str] = set()
    seeds: set[str] = set()
    planned_requests: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or set(case) != {
            "candidate_id",
            "seed_id",
            "state_sha256",
            "history_sha256",
            "source_sha256",
            "context_sha256",
            "student_prompt_sha256",
            "student_input_tokens",
            "request_ids",
        }:
            raise ValueError("fixed-state sanitized case fields are invalid")
        candidate_id = case.get("candidate_id")
        seed_id = case.get("seed_id")
        if (
            not isinstance(candidate_id, str)
            or not _ID.fullmatch(candidate_id)
            or not isinstance(seed_id, str)
            or not _ID.fullmatch(seed_id)
            or candidate_id in case_ids
            or seed_id in seeds
            or packet["candidate_ids"][index] != candidate_id
            or any(
                not _digest(case.get(key))
                for key in (
                    "state_sha256",
                    "history_sha256",
                    "source_sha256",
                    "context_sha256",
                    "student_prompt_sha256",
                )
            )
            or type(case.get("student_input_tokens")) is not int
            or not isinstance(case.get("request_ids"), Mapping)
            or set(case["request_ids"]) != set(_ROLES)
        ):
            raise ValueError("fixed-state sanitized case identity is invalid")
        case_ids.add(candidate_id)
        seeds.add(seed_id)
        for role in _ROLES:
            request_id = case["request_ids"][role]
            identity = _sha(
                _canonical(
                    [projection["phase_id"], packet["sha256"], candidate_id, role]
                )
            )[:24]
            expected_id = f"fixed-state-role-v1-{role}-{identity}"
            if request_id != expected_id or request_id in planned_requests:
                raise ValueError("fixed-state planned role request ID is invalid")
            planned_requests.add(request_id)
    if (
        case_ids != set(packet["candidate_ids"])
        or len(planned_requests) != provider["planned_calls"]
    ):
        raise ValueError("fixed-state sanitized plan candidate/request inventory differs")


def _read_event_chain(
    payload: bytes, plan_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(payload) > MAX_PROOF_FILE_BYTES or not payload.endswith(b"\n"):
        raise ValueError("fixed-state event chain is absent or unbounded")
    events: list[dict[str, Any]] = []
    by_request: dict[str, dict[str, Any]] = {}
    previous = "0" * 64
    for sequence, raw_line in enumerate(payload.splitlines()):
        record = _json_unique(raw_line)
        if not isinstance(record, dict):
            raise ValueError("fixed-state event chain has a non-object record")
        event_hash = record.pop("event_sha256", None)
        request_id = record.get("request_id")
        if (
            record.get("schema") != EVENT_SCHEMA
            or record.get("plan_sha256") != plan_sha256
            or record.get("sequence") != sequence
            or record.get("previous_event_sha256") != previous
            or event_hash != _sha(_canonical(record))
            or not isinstance(request_id, str)
            or request_id in by_request
        ):
            raise ValueError("fixed-state event chain is invalid or duplicated")
        record["event_sha256"] = event_hash
        events.append(record)
        by_request[request_id] = record
        previous = event_hash
    return events, by_request


def _expected_prompt(
    candidate: Mapping[str, Any], role: str, evidence: Mapping[str, Any], tokenizer: Any
):
    from tinycomplete.one_line.fixed_state_roles import (
        build_fixed_state_prompt,
        build_fixed_state_review_prompt,
    )

    state = EditState.from_mapping(candidate["state"])
    if role in {"author", "solver"}:
        prompt = build_fixed_state_prompt(state, tokenizer)
        return prompt.system_instruction, prompt.user_prompt, prompt.prompt_sha256
    review = build_fixed_state_review_prompt(
        state,
        str(evidence["author"]["wire"]),
        str(evidence["solver"]["wire"]),
        tokenizer,
    )
    return review.system_instruction, review.user_prompt, review.prompt_sha256


def _manifest_file_map(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_FILES.values()):
        raise ValueError("fixed-state proof file inventory is incomplete")
    for path, entry in files.items():
        _safe_relative(path)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "bytes"}
            or not _digest(entry.get("sha256"))
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 1
            or entry["bytes"] > MAX_PROOF_FILE_BYTES
        ):
            raise ValueError("fixed-state proof file descriptor is invalid")
    return files


def verify_fixed_state_role_execution_proof(
    candidate: Mapping[str, Any],
    proof_ref: Mapping[str, Any],
    *,
    package_root: Path,
    tokenizer: Any,
) -> AcceptanceDecision:
    """Check portable run artifacts without reading a ledger or invoking tools."""
    from tinycomplete.one_line.fixed_state_roles import (
        SOURCE_TYPE,
        fixed_state_protocol_bindings,
        verify_fixed_state_role_evidence,
    )
    from tinycomplete.one_line.muse_acceptance_package import AcceptanceDecision

    try:
        expected_ref_fields = {
            "schema",
            "root",
            "candidate_id",
            "manifest_path",
            "manifest_sha256",
            "manifest_bytes",
            "role_evidence_path",
            "role_evidence_sha256",
            "role_evidence_bytes",
        }
        if not isinstance(proof_ref, Mapping) or set(proof_ref) != expected_ref_fields:
            raise ValueError("fixed-state run proof reference fields are invalid")
        if (
            proof_ref.get("schema") != ROLE_PROOF_REF_SCHEMA
            or proof_ref.get("candidate_id") != candidate.get("candidate_id", candidate.get("id"))
            or proof_ref.get("manifest_path") != "proof_manifest.json"
            or proof_ref.get("role_evidence_path") != _FILES["role_evidence"]
            or not _digest(proof_ref.get("manifest_sha256"))
            or not _digest(proof_ref.get("role_evidence_sha256"))
            or type(proof_ref.get("manifest_bytes")) is not int
            or type(proof_ref.get("role_evidence_bytes")) is not int
        ):
            raise ValueError("fixed-state run proof reference identity is invalid")
        proof_root = _safe_relative(proof_ref.get("root"))
        manifest_bytes = _read_relative(
            package_root,
            str(proof_root / "proof_manifest.json"),
            maximum=MAX_MANIFEST_BYTES,
        )
        if (
            len(manifest_bytes) != proof_ref["manifest_bytes"]
            or _sha(manifest_bytes) != proof_ref["manifest_sha256"]
        ):
            raise ValueError("fixed-state proof manifest hash or length mismatch")
        manifest = _json_unique(manifest_bytes, maximum=MAX_MANIFEST_BYTES)
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema",
            "attestation_limit",
            "candidate_id",
            "candidate_bindings",
            "plan_sha256",
            "root_release_sha256",
            "runner_source_hashes",
            "proof_verifier_sha256",
            "files",
            "calls",
        }:
            raise ValueError("fixed-state proof manifest schema is invalid")
        if (
            manifest.get("schema") != ROLE_PROOF_SCHEMA
            or manifest.get("attestation_limit")
            != "locally_audited_runner_artifacts_not_provider_signed"
            or manifest.get("candidate_id") != proof_ref.get("candidate_id")
            or not _digest(manifest.get("plan_sha256"))
            or not _digest(manifest.get("root_release_sha256"))
            or not _digest(manifest.get("proof_verifier_sha256"))
        ):
            raise ValueError("fixed-state proof manifest identity is invalid")
        files = _manifest_file_map(manifest)

        def read_inventory_file(relative: str) -> bytes:
            payload = _read_relative(
                package_root / Path(*proof_root.parts), relative, maximum=MAX_PROOF_FILE_BYTES
            )
            entry = files[relative]
            if len(payload) != entry["bytes"] or _sha(payload) != entry["sha256"]:
                raise ValueError("fixed-state proof artifact digest mismatch")
            return payload

        release_bytes = read_inventory_file(_FILES["root_release"])
        projection_bytes = read_inventory_file(_FILES["plan_projection"])
        event_bytes = read_inventory_file(_FILES["event_chain"])
        evidence_bytes = read_inventory_file(_FILES["role_evidence"])
        if (
            len(evidence_bytes) != proof_ref["role_evidence_bytes"]
            or _sha(evidence_bytes) != proof_ref["role_evidence_sha256"]
        ):
            raise ValueError("fixed-state role evidence reference mismatch")
        projection = _json_unique(projection_bytes)
        release = _json_unique(release_bytes)
        if (
            not isinstance(projection, dict)
            or projection.get("schema") != PLAN_PROJECTION_SCHEMA
            or projection.get("original_plan_sha256") != manifest.get("plan_sha256")
            or _sha(release_bytes) != manifest.get("root_release_sha256")
            or manifest.get("runner_source_hashes") != projection.get("source_hashes")
        ):
            raise ValueError("fixed-state plan projection or root release mismatch")
        _validate_plan_projection(projection)
        _validate_release_projection(release, projection)
        if manifest.get("plan_sha256") != release.get("plan_sha256"):
            raise ValueError("fixed-state proof is not bound to the released plan")

        candidate_id = str(candidate.get("candidate_id", candidate.get("id", "")))
        if not _ID.fullmatch(candidate_id) or candidate.get("source_type") != SOURCE_TYPE:
            raise ValueError("fixed-state candidate identity or source type is invalid")
        state = EditState.from_mapping(candidate["state"])
        prompt = __import__(
            "tinycomplete.one_line.fixed_state_roles", fromlist=["build_fixed_state_prompt"]
        ).build_fixed_state_prompt(state, tokenizer)
        candidate_protocol = fixed_state_protocol_bindings(candidate, tokenizer)
        bindings = manifest.get("candidate_bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != {
            "candidate_id",
            "seed_id",
            "state_sha256",
            "history_sha256",
            "source_sha256",
            "context_sha256",
            "student_prompt_sha256",
            "student_input_tokens",
        }:
            raise ValueError("fixed-state candidate proof bindings are invalid")
        if (
            bindings.get("candidate_id") != candidate_id
            or bindings.get("seed_id") != candidate.get("seed_id")
            or bindings.get("state_sha256") != candidate_protocol["state_sha256"]
            or bindings.get("history_sha256") != candidate_protocol["history_sha256"]
            or bindings.get("source_sha256") != candidate_protocol["source_sha256"]
            or bindings.get("context_sha256") != prompt.context_sha256
            or bindings.get("student_prompt_sha256") != prompt.prompt_sha256
            or bindings.get("student_input_tokens") != prompt.input_tokens
        ):
            raise ValueError("fixed-state proof is bound to a different candidate state")

        cases = projection.get("cases")
        if not isinstance(cases, list):
            raise ValueError("fixed-state projected plan has no case list")
        selected = [
            case
            for case in cases
            if isinstance(case, Mapping) and case.get("candidate_id") == candidate_id
        ]
        if len(selected) != 1:
            raise ValueError("fixed-state candidate is absent or duplicated in the plan")
        plan_case = selected[0]
        if any(
            plan_case.get(key) != bindings.get(binding)
            for key, binding in (
                ("state_sha256", "state_sha256"),
                ("history_sha256", "history_sha256"),
                ("source_sha256", "source_sha256"),
                ("context_sha256", "context_sha256"),
                ("student_prompt_sha256", "student_prompt_sha256"),
                ("student_input_tokens", "student_input_tokens"),
            )
        ):
            raise ValueError("fixed-state plan case does not match candidate binding")
        request_ids = plan_case.get("request_ids")
        if not isinstance(request_ids, Mapping) or set(request_ids) != set(_ROLES):
            raise ValueError("fixed-state planned request identities are invalid")

        event_rows, events_by_request = _read_event_chain(event_bytes, str(manifest["plan_sha256"]))
        planned = {
            case["request_ids"][role]: (case["candidate_id"], role)
            for case in cases
            for role in _ROLES
        }
        if any(
            event.get("request_id") not in planned
            or (event.get("candidate_id"), event.get("role")) != planned[event["request_id"]]
            for event in event_rows
        ):
            raise ValueError("fixed-state event chain contains an out-of-plan event")

        role_evidence = _json_unique(evidence_bytes)
        if not isinstance(role_evidence, dict) or role_evidence.get("candidate_id") != candidate_id:
            raise ValueError("fixed-state role evidence candidate ID differs")
        calls = manifest.get("calls")
        if not isinstance(calls, list) or len(calls) != len(_ROLES):
            raise ValueError("fixed-state proof must contain exactly three role calls")
        call_by_role = {entry.get("role"): entry for entry in calls if isinstance(entry, Mapping)}
        if set(call_by_role) != set(_ROLES):
            raise ValueError("fixed-state proof call roles are missing or duplicated")

        action_events: list[dict[str, Any]] = []
        for role in _ROLES:
            call = call_by_role[role]
            if set(call) != {
                "role",
                "request_id",
                "actor_id",
                "model_id",
                "session_id",
                "response_id",
                "prompt_sha256",
                "prompt_file_sha256",
                "response_sha256",
                "response_receipt_sha256",
                "event_sequence",
                "event_sha256",
                "finish_reason",
                "receipt",
                "settled_usage",
            }:
                raise ValueError("fixed-state per-role call inventory fields changed")
            request_id = request_ids.get(role)
            if not isinstance(request_id, str):
                raise ValueError("fixed-state planned request ID is invalid")
            event = events_by_request.get(request_id)
            if (
                event is None
                or event.get("candidate_id") != candidate_id
                or event.get("role") != role
                or event.get("status")
                != ("completed_valid_review" if role == "reviewer" else "completed_valid_action")
            ):
                raise ValueError("fixed-state candidate lacks a completed role event")
            if (
                call.get("role") != role
                or call.get("request_id") != request_id
                or call.get("event_sequence") != event.get("sequence")
                or call.get("event_sha256") != event.get("event_sha256")
                or call.get("actor_id") != event.get("actor_id")
                or call.get("model_id") != event.get("model_id")
                or call.get("session_id") != event.get("session_id")
                or call.get("response_id") != event.get("response_id")
                or call.get("prompt_sha256") != event.get("prompt_sha256")
                or call.get("response_sha256") != event.get("response_sha256")
                or call.get("response_receipt_sha256") != event.get("response_receipt_sha256")
                or event.get("actor_id") != f"{MODEL_ID}:{_ACTOR_SUFFIX[role]}"
                or event.get("model_id") != MODEL_ID
                or event.get("request_id") != request_id
                or event.get("retry_permitted") is not False
            ):
                raise ValueError("fixed-state call record differs from its event")

            receipt_path = _FILES[f"{role}_receipt"]
            receipt_bytes = read_inventory_file(receipt_path)
            receipt = _json_unique(receipt_bytes)
            if not isinstance(receipt, Mapping) or set(receipt) != {
                "schema",
                "request_id",
                "response_sha256",
                "model_id",
                "session_id",
                "response_id",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cached_read_tokens",
                "cached_write_tokens",
                "total_tokens_reported",
                "finish_reason",
            }:
                raise ValueError("fixed-state persisted response receipt fields are invalid")
            proof_receipt_descriptor = call.get("receipt")
            if not isinstance(proof_receipt_descriptor, Mapping) or proof_receipt_descriptor != {
                "path": receipt_path,
                "sha256": _sha(receipt_bytes),
                "bytes": len(receipt_bytes),
            }:
                raise ValueError("fixed-state response receipt descriptor mismatch")
            if (
                receipt.get("schema") != RECEIPT_SCHEMA
                or receipt.get("request_id") != request_id
                or receipt.get("response_sha256") != event.get("response_sha256")
                or receipt.get("model_id") != event.get("model_id")
                or receipt.get("session_id") != event.get("session_id")
                or receipt.get("response_id") != event.get("response_id")
                or receipt.get("finish_reason") != "stop"
                or receipt.get("response_sha256") != role_evidence[role].get("response_sha256")
            ):
                raise ValueError("fixed-state receipt differs from event or role evidence")
            system, user_prompt, expected_prompt_sha = _expected_prompt(
                candidate, role, role_evidence, tokenizer
            )
            prompt_file = _canonical(
                {"system_instruction": system, "user_prompt": user_prompt}
            )
            if (
                call.get("prompt_sha256") != expected_prompt_sha
                or call.get("prompt_file_sha256") != _sha(prompt_file)
                or event.get("prompt_sha256") != expected_prompt_sha
                or call.get("finish_reason") != receipt.get("finish_reason")
            ):
                raise ValueError("fixed-state request prompt is not reproducible")

            usage = call.get("settled_usage")
            if not isinstance(usage, Mapping) or set(usage) != {
                "reserved_input_tokens",
                "reserved_output_tokens",
                "actual_input_tokens",
                "actual_output_tokens_including_reasoning",
                "overrun",
                "corrected",
            }:
                raise ValueError("fixed-state per-request usage projection is invalid")
            reserve_input = projection["provider"]["reserve_input_tokens_per_request"]
            reserve_output = projection["provider"]["reserve_output_tokens_per_request"]
            actual_input = receipt.get("input_tokens")
            actual_output = receipt.get("output_tokens")
            reasoning = receipt.get("reasoning_tokens")
            if (
                type(actual_input) is not int
                or type(actual_output) is not int
                or type(reasoning) is not int
            ):
                raise ValueError("fixed-state response usage values are invalid")
            if (
                usage.get("reserved_input_tokens") != reserve_input
                or usage.get("reserved_output_tokens") != reserve_output
                or usage.get("actual_input_tokens") != actual_input
                or usage.get("actual_output_tokens_including_reasoning")
                != actual_output + reasoning
                or usage.get("overrun") is not False
                or type(usage.get("corrected")) is not bool
                or actual_input < 1
                or actual_output < 1
                or reasoning < 0
                or actual_input > reserve_input
                or actual_output + reasoning > reserve_output
                or event.get("provider_input_tokens") != actual_input
                or event.get("provider_output_tokens") != actual_output
                or event.get("provider_reasoning_tokens") != reasoning
            ):
                raise ValueError("fixed-state settled usage does not reconcile")
            action_events.append(dict(event))

        if not (
            action_events[0]["sequence"] < action_events[1]["sequence"]
            < action_events[2]["sequence"]
        ):
            raise ValueError("fixed-state roles are not ordered author/solver/reviewer")
        role_decision = verify_fixed_state_role_evidence(
            candidate, evidence_bytes, tokenizer=tokenizer
        )
        if not role_decision.accepted:
            raise ValueError("fixed-state role evidence did not pass its independent verifier")
        return AcceptanceDecision(
            True,
            "fixed_state_role_execution_proof_verified",
            {
                "candidate_id": candidate_id,
                "proof_manifest_sha256": _sha(manifest_bytes),
                "plan_sha256": manifest["plan_sha256"],
                "root_release_sha256": manifest["root_release_sha256"],
                "event_chain_sha256": files[_FILES["event_chain"]]["sha256"],
                "role_evidence_sha256": _sha(evidence_bytes),
                "request_ids": [call_by_role[role]["request_id"] for role in _ROLES],
                "functional_status": "not_evaluated_by_role_runner",
                "provider_signature": False,
            },
        )
    except (AttributeError, KeyError, TypeError, UnicodeError, ValueError, OSError):
        return AcceptanceDecision(False, "fixed_state_role_execution_proof_invalid", {})


def export_fixed_state_role_execution_proof(
    candidate: Mapping[str, Any],
    *,
    run_dir: Path,
    release_path: Path,
    release_sha256: str,
    ledger_path: Path,
    package_root: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    """Locally validate and export one candidate's already completed role run.

    The private global ledger is read and validated here, but only the selected
    candidate's three settled usage rows are included in the portable bundle.
    """
    from tinycomplete.one_line.fixed_state_roles import verify_fixed_state_role_evidence

    runner = _load_runner()
    plan, input_rows = runner._load_frozen_plan(run_dir)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("fixed-state run plan schema is unsupported")
    release = runner._read_release(release_path, release_sha256, plan)
    input_rows = runner._validate_case_plan(plan, run_dir, tokenizer)
    events, events_by_id = runner._read_event_log(run_dir, plan)
    current_rows, _totals, _new_ids = runner._validate_ledger_progress(
        plan, run_dir, events_by_id, ledger_path
    )
    candidate_id = candidate.get("candidate_id", candidate.get("id"))
    if not isinstance(candidate_id, str) or not _ID.fullmatch(candidate_id):
        raise ValueError("candidate ID is invalid")
    matches = [row for row in input_rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError("candidate is not uniquely present in the frozen input packet")
    input_row = matches[0]
    if (
        candidate.get("seed_id") != input_row.get("seed_id")
        or candidate.get("state") != input_row.get("state")
        or candidate.get("context_sha256") != input_row.get("context_sha256")
        or candidate.get("prompt") != input_row.get("prompt")
    ):
        raise ValueError("candidate row differs from the answer-free frozen role input")
    if candidate.get("source_type") != "reviewed_public_history_candidate":
        raise ValueError("fixed-state role proof only accepts reviewed public-history candidates")

    plan_case = next(case for case in plan["cases"] if case["candidate_id"] == candidate_id)
    role_events: dict[str, dict[str, Any]] = {}
    for role in _ROLES:
        event = events_by_id.get(plan_case["request_ids"][role])
        expected_status = (
            "completed_valid_review" if role == "reviewer" else "completed_valid_action"
        )
        if event is None or event.get("status") != expected_status:
            raise ValueError("selected candidate does not have three completed valid roles")
        role_events[role] = event
    reconstructed, evidence_bytes, decision = runner._role_evidence_for_case(
        run_dir=run_dir,
        input_row=input_row,
        role_events=role_events,
        tokenizer=tokenizer,
    )
    evidence_path = run_dir / "role_evidence" / f"{runner._digest_id(candidate_id)}.json"
    stored_evidence = runner._private_file(evidence_path)
    if stored_evidence != evidence_bytes:
        raise ValueError("persisted role evidence differs from completed response receipts")
    if not decision.accepted:
        raise ValueError("independent role evidence was rejected")
    if candidate.get("action") != reconstructed.get("action") or candidate.get(
        "after_source"
    ) != reconstructed.get("after_source"):
        raise ValueError("candidate label differs from completed author and solver actions")
    if not verify_fixed_state_role_evidence(
        candidate, evidence_bytes, tokenizer=tokenizer
    ).accepted:
        raise ValueError("role evidence failed candidate-bound portable verification")

    from tinycomplete.one_line.teacher import _SECRET

    secret_pattern = _SECRET
    if secret_pattern.search(evidence_bytes.decode("utf-8")):
        raise ValueError("role evidence was quarantined by secret screening")

    projection = _projection(plan)
    release_bytes = runner._private_file(release_path)
    if _sha(release_bytes) != release_sha256:
        raise ValueError("root release changed during proof export")
    _validate_release_projection(release, projection)
    event_bytes = runner._private_file(run_dir / "events.jsonl")
    if len(event_bytes) > MAX_PROOF_FILE_BYTES:
        raise ValueError("fixed-state phase event chain exceeds export bound")
    _read_event_chain(event_bytes, str(plan["plan_sha256"]))

    proof_parent = package_root / "fixed_state_role_proofs"
    _private_output_directory(package_root, create=False)
    _private_output_directory(proof_parent, create=True)
    safe_id = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:24]
    proof_dir = proof_parent / safe_id
    _private_output_directory(proof_dir, create=True)
    receipts_dir = proof_dir / "receipts"
    _private_output_directory(receipts_dir, create=True)

    artifacts: dict[str, bytes] = {
        _FILES["root_release"]: release_bytes,
        _FILES["plan_projection"]: _canonical(projection) + b"\n",
        _FILES["event_chain"]: event_bytes,
        _FILES["role_evidence"]: evidence_bytes,
    }
    call_records: list[dict[str, Any]] = []
    for role in _ROLES:
        event = role_events[role]
        request_id = str(event["request_id"])
        response_path, receipt_path = runner._response_paths(run_dir, request_id)
        response_bytes = runner._private_file(response_path)
        receipt_bytes = runner._private_file(receipt_path)
        receipt = runner._json_unique(receipt_bytes.decode("utf-8"))
        if not isinstance(receipt, Mapping):
            raise ValueError("persisted completed response receipt is malformed")
        if secret_pattern.search(response_bytes.decode("utf-8")):
            raise ValueError("a completed role response was quarantined by secret screening")
        prompt_path = runner._prompt_path(run_dir, request_id)
        prompt_bytes = runner._private_file(prompt_path)
        prompt = runner._json_unique(prompt_bytes.decode("utf-8"))
        expected_system, expected_user, expected_prompt_sha = _expected_prompt(
            candidate, role, _json_unique(evidence_bytes), tokenizer
        )
        if prompt_bytes != _canonical(
            {"system_instruction": expected_system, "user_prompt": expected_user}
        ):
            raise ValueError("persisted role prompt differs from canonical state renderer")
        usage = current_rows.get(request_id)
        if not isinstance(usage, Mapping) or "input" not in usage or "output" not in usage:
            raise ValueError("selected role request has no settled usage in the local ledger")
        if (
            usage.get("overrun")
            or usage.get("input") != receipt.get("input_tokens")
            or usage.get("output")
            != receipt.get("output_tokens", -1) + receipt.get("reasoning_tokens", -1)
            or receipt.get("request_id") != request_id
            or receipt.get("response_sha256") != _sha(response_bytes)
            or event.get("response_sha256") != _sha(response_bytes)
            or event.get("response_receipt_sha256") != _sha(receipt_bytes)
            or event.get("prompt_sha256") != expected_prompt_sha
        ):
            raise ValueError("role event, response receipt, and settled usage disagree")

        receipt_rel = _FILES[f"{role}_receipt"]
        artifacts[receipt_rel] = receipt_bytes
        call_records.append(
            {
                "role": role,
                "request_id": request_id,
                "actor_id": event["actor_id"],
                "model_id": event["model_id"],
                "session_id": event["session_id"],
                "response_id": event["response_id"],
                "prompt_sha256": expected_prompt_sha,
                "prompt_file_sha256": _sha(prompt_bytes),
                "response_sha256": _sha(response_bytes),
                "response_receipt_sha256": _sha(receipt_bytes),
                "event_sequence": event["sequence"],
                "event_sha256": event["event_sha256"],
                "finish_reason": event["finish_reason"],
                "receipt": {
                    "path": receipt_rel,
                    "sha256": _sha(receipt_bytes),
                    "bytes": len(receipt_bytes),
                },
                "settled_usage": {
                    "reserved_input_tokens": usage["reserved_input"],
                    "reserved_output_tokens": usage["reserved_output"],
                    "actual_input_tokens": usage["input"],
                    "actual_output_tokens_including_reasoning": usage["output"],
                    "overrun": bool(usage.get("overrun", 0)),
                    "corrected": bool(usage.get("corrected", 0)),
                },
            }
        )

    files: dict[str, dict[str, Any]] = {}
    for relative, payload in artifacts.items():
        if len(payload) > MAX_PROOF_FILE_BYTES:
            raise ValueError("fixed-state role proof artifact exceeds export bound")
        target = proof_dir / Path(*PurePosixPath(relative).parts)
        if not target.parent.exists():
            _private_output_directory(target.parent, create=True)
        _write_once(target, payload)
        files[relative] = {"sha256": _sha(payload), "bytes": len(payload)}

    # Bind the exact training-state identity without copying another prompt body.
    from tinycomplete.one_line.fixed_state_roles import build_fixed_state_prompt

    prompt = build_fixed_state_prompt(EditState.from_mapping(candidate["state"]), tokenizer)
    bindings = {
        "candidate_id": candidate_id,
        "seed_id": candidate.get("seed_id"),
        "state_sha256": prompt.state_sha256,
        "history_sha256": prompt.history_sha256,
        "source_sha256": prompt.source_sha256,
        "context_sha256": prompt.context_sha256,
        "student_prompt_sha256": prompt.prompt_sha256,
        "student_input_tokens": prompt.input_tokens,
    }
    manifest = {
        "schema": ROLE_PROOF_SCHEMA,
        "attestation_limit": "locally_audited_runner_artifacts_not_provider_signed",
        "candidate_id": candidate_id,
        "candidate_bindings": bindings,
        "plan_sha256": plan["plan_sha256"],
        "root_release_sha256": release_sha256,
        "runner_source_hashes": plan["source_hashes"],
        "proof_verifier_sha256": _sha(Path(__file__).read_bytes()),
        "files": files,
        "calls": call_records,
    }
    manifest_bytes = _canonical(manifest) + b"\n"
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("fixed-state proof manifest exceeds its export bound")
    _write_once(proof_dir / "proof_manifest.json", manifest_bytes)
    relative_root = proof_dir.relative_to(package_root).as_posix()
    return {
        "schema": ROLE_PROOF_REF_SCHEMA,
        "root": relative_root,
        "candidate_id": candidate_id,
        "manifest_path": "proof_manifest.json",
        "manifest_sha256": _sha(manifest_bytes),
        "manifest_bytes": len(manifest_bytes),
        "role_evidence_path": _FILES["role_evidence"],
        "role_evidence_sha256": _sha(evidence_bytes),
        "role_evidence_bytes": len(evidence_bytes),
    }
