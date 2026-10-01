"""Portable proof and split checks for the narrow source/functional pilot."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .public_prefix_pilot import (
    PUBLIC_PREFIX_SOURCE_TYPE,
    PUBLIC_SYNTHETIC_PILOT_SCHEMA,
    canonical_sha256,
    sha256_bytes,
    validate_public_prefix_row,
)
from .synthetic_functional_mix import (
    SYNTHETIC_FUNCTIONAL_SOURCE_TYPE,
    validate_synthetic_candidate,
    validate_synthetic_receipt,
)


def read_proof(root: Path, descriptor: Mapping[str, Any]) -> bytes:
    name = descriptor.get("path")
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("unsafe pilot proof path")
    p = root
    for part in Path(name).parts:
        p = p / part
        if p.is_symlink():
            raise ValueError("pilot proof contains a symlink")
    payload = p.read_bytes()
    if len(payload) != descriptor.get("bytes") or sha256_bytes(payload) != descriptor.get("sha256"):
        raise ValueError("pilot proof hash or size mismatch")
    return payload


def validate_implementation_evidence(payload: bytes, descriptor: Mapping[str, Any]) -> None:
    if sha256_bytes(payload) != descriptor.get("sha256"):
        raise ValueError("implementation evidence hash mismatch")
    record = json.loads(payload)
    viability = record["implementation_viability"]
    integrity = record["artifact_integrity"]
    if (
        integrity.get("plan_sha256") != descriptor.get("fixture_plan_sha256")
        or integrity.get("status") != "complete"
        or record.get("accepted_training") != 0
        or record.get("token_counts", {}).get("completed_updates") != 32
        or record.get("token_counts", {}).get("skipped_updates") != 0
        or viability.get("passed") is not True
        or viability.get("quality_evidence") is not False
        or viability.get("total_observations") != 16
        or any(
            v.get("exact_actions") != 4 or v.get("terminated_by_eos") != 4
            for v in viability["per_action"].values()
        )
    ):
        raise ValueError("implementation check did not pass its original gates")


def validate_row(row: Mapping[str, Any], *, package_root: Path) -> None:
    if (
        row.get("schema") != PUBLIC_SYNTHETIC_PILOT_SCHEMA
        or row.get("accepted_training") is not True
        or row.get("human_chronology_observed") is not False
        or row.get("validation", {}).get("replay_verified") is not True
    ):
        raise ValueError("pilot row policy identity mismatch")
    if row.get("source_type") == PUBLIC_PREFIX_SOURCE_TYPE:
        validate_public_prefix_row(row, package_root=package_root)
    elif row.get("source_type") == SYNTHETIC_FUNCTIONAL_SOURCE_TYPE:
        validate_synthetic_candidate(dict(row))
        if sha256_bytes((package_root / "proofs/generator_source.py").read_bytes()) != row.get(
            "source_revision"
        ):
            raise ValueError("synthetic generator source identity mismatch")
        raw = read_proof(
            package_root,
            {
                "path": row.get("receipt_path"),
                "sha256": row.get("receipt_sha256"),
                "bytes": row.get("receipt_bytes"),
            },
        )
        record = json.loads(raw)
        receipt = row["synthetic_objective_receipt"]
        validate_synthetic_receipt(dict(row), evaluator_sha256=receipt["evaluator_sha256"])
        if (
            record.get("id") != row.get("id")
            or record.get("qualified") is not True
            or record.get("receipt") != receipt
        ):
            raise ValueError("synthetic receipt does not bind its actual diagnostics")
        for name in ("gold", "wrong"):
            result = record["results"][name]
            for check in ("parse", "compile", "test"):
                if result[check]["status"] != receipt[name + "_" + check]:
                    raise ValueError("synthetic check status and receipt disagree")
    else:
        raise ValueError("unapproved pilot source type")


def validate_manifest(
    manifest: Mapping[str, Any], rows: list[dict[str, Any]], *, package_root: Path
) -> dict[str, Any]:
    if (
        manifest.get("schema") != PUBLIC_SYNTHETIC_PILOT_SCHEMA
        or manifest.get("training_ready") is not True
    ):
        raise ValueError("source/functional pilot is not ready")
    for descriptor in manifest["proof_files"]:
        read_proof(package_root, descriptor)
    states: set[str] = set()
    ids: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    actions: dict[str, Counter[str]] = {"train": Counter(), "development": Counter()}
    for row in rows:
        validate_row(row, package_root=package_root)
        split = row["split"]
        if split not in actions:
            raise ValueError("reserved split entered the pilot")
        identifier = row["id"]
        normalized_state = dict(row["state"])
        normalized_state.pop("file_id")
        state_key = canonical_sha256(normalized_state)
        if identifier in ids or state_key in states:
            raise ValueError("duplicate model state or case")
        ids.add(identifier)
        states.add(state_key)
        fields = ["source_group_id", "task_family_id", "template_id", "near_duplicate_sha256"]
        if row["source_type"] == PUBLIC_PREFIX_SOURCE_TYPE:
            fields += ["source_repo", "session_or_commit"]
        for field in fields:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("pilot split grouping identity missing")
            key = (field, value.casefold() if field == "source_repo" else value)
            if groups.setdefault(key, split) != split:
                raise ValueError("pilot group crosses splits: " + field)
        counts[split] += 1
        actions[split][row["action"]["kind"]] += 1
        if (
            row["source_type"] == SYNTHETIC_FUNCTIONAL_SOURCE_TYPE
            and row["synthetic_objective_receipt"]["evaluator_sha256"]
            != manifest["oracle_evaluator_sha256"]
        ):
            raise ValueError("synthetic evaluator identity differs from manifest")
    if (
        counts["train"] != manifest["train_count"]
        or counts["development"] != manifest["dev_count"]
        or counts["train"] < 128
        or counts["development"] < 64
    ):
        raise ValueError("pilot split counts mismatch or below floor")
    for split in actions:
        if set(actions[split]) != {"keep", "insert_before", "delete_line", "replace_line"}:
            raise ValueError("pilot requires all four actions in each split")
    return {
        "split_counts": dict(counts),
        "action_counts": {k: dict(v) for k, v in actions.items()},
        "cross_split_group_overlap": 0,
    }
