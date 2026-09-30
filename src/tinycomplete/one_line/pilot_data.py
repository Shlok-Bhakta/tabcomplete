"""Allowed data contracts for the existing bounded one-line training pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contract import EditAction, EditState
from .data import near_duplicate_key


@dataclass(frozen=True)
class PilotDataPolicy:
    data_schema: str
    plan_schema: str
    branch: str
    dataset_id: str
    dataset_license: str
    file_license_status: str
    source_type: str
    repository_plan_path: str


INSTINCT = PilotDataPolicy(
    "one-line-instinct-pilot-v1",
    "one-line-instinct-pilot-plan-v1",
    "research/one-line-gpu-pilot-r1",
    "continuedev/instinct-data",
    "Apache-2.0",
    "unverified",
    "continue_instinct_observed",
    "reports/research/one_line_gpu_pilot_r1/plan.json",
)
CONSTRUCTIVE = PilotDataPolicy(
    "one-line-constructive-pilot-v1",
    "one-line-constructive-pilot-plan-v1",
    "prototype/product-r2",
    "synthetic/tabcomplete-constructive-r1",
    "MIT",
    "own_synthetic_source",
    "synthetic_constructive_objective",
    "reports/prototype/product_r2/constructive_pilot_plan.json",
)


def policy_for_schema(schema: object) -> PilotDataPolicy:
    for policy in (INSTINCT, CONSTRUCTIVE):
        if schema == policy.data_schema:
            return policy
    raise ValueError("unapproved bounded-pilot data schema")


def validate_aggregate_budget(
    budgets: Mapping[str, Any],
    *,
    planned_tokens: int,
    session_seconds: int,
    external_campaign_tokens: int | None = None,
) -> None:
    """Count previous failed sessions as well as completed training exposure."""
    prior_tokens = budgets.get("prior_training_input_tokens")
    prior_seconds = budgets.get("prior_session_wall_seconds")
    if (
        type(prior_tokens) is not int
        or type(prior_seconds) is not int
        or prior_tokens < 0
        or prior_seconds < 0
        or prior_tokens + planned_tokens > 100_000_000
        or prior_seconds + session_seconds > 24 * 3600
        or (external_campaign_tokens is not None and external_campaign_tokens != prior_tokens)
    ):
        raise ValueError("constructive pilot lacks valid aggregate campaign accounting")


def validate_constructive_splits(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Hold out actual authored families within one declared synthetic source.

    Public repository isolation remains in data.validate_splits. This diagnostic
    is authored in a single repository and does not claim repository diversity.
    """
    ids: set[str] = set()
    states: set[str] = set()
    grouped: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    sources: set[str] = set()
    for row in rows:
        identifier = row.get("id")
        split = row.get("split")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("duplicate or missing constructive row id")
        if split not in {"train", "development", "test_new_mechanism"}:
            raise ValueError("unknown constructive split")
        ids.add(identifier)
        counts[split] += 1
        source = row.get("source_repo")
        if not isinstance(source, str) or not source:
            raise ValueError("constructive source identity missing")
        sources.add(source)
        state_hash = hashlib.sha256(
            json.dumps(
                asdict(EditState.from_mapping(row["state"])), sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        if state_hash in states:
            raise ValueError("duplicate constructive model input state")
        states.add(state_hash)
        keys = [("near_duplicate", near_duplicate_key(row))]
        for field in ("source_group_id", "session_or_commit", "generator_family", "template_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("constructive grouping identity missing: " + field)
            keys.append((field, value))
        for key in keys:
            prior_split = grouped.setdefault(key, split)
            if prior_split != split:
                raise ValueError("constructive family or near duplicate crosses splits")
    if len(sources) != 1:
        raise ValueError("constructive pilot must declare its one authored source")
    return {
        "rows": len(rows),
        "splits": dict(sorted(counts.items())),
        "authored_source_count": len(sources),
        "repository_diversity_claimed": False,
        "grouping_policy": "constructive-family-split-v1",
    }


def validate_constructive_manifest(manifest: Mapping[str, Any]) -> None:
    """Require the pinned evidence produced by independent candidate review."""
    for key, expected in (
        ("schema", CONSTRUCTIVE.data_schema),
        ("dataset_id", CONSTRUCTIVE.dataset_id),
        ("dataset_license", CONSTRUCTIVE.dataset_license),
        ("source_file_license_status", CONSTRUCTIVE.file_license_status),
    ):
        if manifest.get(key) != expected:
            raise ValueError("constructive pilot manifest identity mismatch: " + key)
    for key in ("generator_sha256", "oracle_sha256", "independent_review_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get(key, ""))):
            raise ValueError("constructive pilot manifest lacks pinned evidence: " + key)


def constructive_row_bindings(row: Mapping[str, Any]) -> dict[str, str]:
    """Bind review to the actual canonical input and supervised action."""
    values = {
        "state_sha256": asdict(EditState.from_mapping(row["state"])),
        "action_sha256": asdict(EditAction(**row["action"])),
    }
    return {
        key: hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        for key, value in values.items()
    }


def validate_constructive_review(
    manifest: Mapping[str, Any], rows: list[Mapping[str, Any]], review_path: Path
) -> None:
    """Require the independent review artifact, not self-certified row flags."""
    payload = review_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("independent_review_sha256"):
        raise ValueError("constructive independent review artifact hash mismatch")
    review = json.loads(payload)
    if (
        not isinstance(review, dict)
        or review.get("schema") != "one-line-constructive-independent-review-v1"
        or review.get("generator_sha256") != manifest["generator_sha256"]
        or review.get("oracle_sha256") != manifest["oracle_sha256"]
        or review.get("full_split_audit_complete") is not True
        or not isinstance(review.get("independent_reviewer_id"), str)
        or not review["independent_reviewer_id"]
        or not isinstance(review.get("rows"), list)
    ):
        raise ValueError("constructive independent review identity or audit is incomplete")
    entries: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], str] = {}
    audited_counts: Counter[str] = Counter()
    group_fields = ("source_group_id", "session_or_commit", "generator_family", "template_id")
    proof_flags = (
        "accepted_training",
        "inferability_reviewed",
        "objective_verified",
        "wrong_action_controls_rejected",
        "history_leakage_check",
    )
    for entry in review["rows"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid constructive review row")
        identifier = entry.get("id")
        split = entry.get("split")
        if not isinstance(identifier, str) or not identifier or identifier in entries:
            raise ValueError("duplicate or missing constructive review row id")
        if split not in {"train", "development", "test_new_mechanism"}:
            raise ValueError("invalid constructive review split")
        entries[identifier] = entry
        audited_counts[split] += 1
        for field in ("state_sha256", "action_sha256", "near_duplicate_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", str(entry.get(field, ""))):
                raise ValueError("constructive review is missing a bound hash: " + field)
        for field in (*group_fields, "near_duplicate_sha256"):
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("constructive review grouping identity missing: " + field)
            previous = grouped.setdefault((field, value), split)
            if previous != split:
                raise ValueError("full constructive review audit crosses splits")
    if (
        dict(audited_counts) != manifest.get("candidate_split_counts")
        or audited_counts["test_new_mechanism"] < 1
    ):
        raise ValueError("constructive review does not cover the complete reserved split audit")
    approved = {
        key
        for key, entry in entries.items()
        if entry["split"] in {"train", "development"} and entry.get("accepted_training") is True
    }
    if approved != {row["id"] for row in rows}:
        raise ValueError("constructive shard IDs differ from independent approved review")
    for row in rows:
        entry = entries[row["id"]]
        expected = {
            **constructive_row_bindings(row),
            **{field: row[field] for field in group_fields},
            "near_duplicate_sha256": near_duplicate_key(row),
            "split": row["split"],
        }
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError("constructive row differs from independently reviewed state/action")
        if row.get("independent_reviewer_id") != review["independent_reviewer_id"] or any(
            entry.get(flag) is not True for flag in proof_flags
        ):
            raise ValueError("constructive row lacks actual independent approval")


def validate_pilot_row(row: Mapping[str, Any], policy: PilotDataPolicy) -> None:
    if row.get("source_type") != policy.source_type:
        raise ValueError("pilot shard contains an unapproved source type")
    validation = row.get("validation", {})
    if not isinstance(validation, Mapping) or validation.get("replay_verified") is not True:
        raise ValueError("pilot row is not replay-verified")
    if policy == INSTINCT:
        return
    if row.get("source_license") != policy.dataset_license:
        raise ValueError("constructive pilot source license mismatch")
    for key in (
        "history_replays_to_state",
        "apply_reconstructs_after",
        "objective_verified",
        "inferability_reviewed",
        "accepted_training",
        "wrong_action_controls_rejected",
        "history_leakage_check",
        "after_objective_passes",
    ):
        if validation.get(key) is not True:
            raise ValueError("constructive pilot row lacks independent validation: " + key)
    for key in ("source_group_id", "template_id", "independent_reviewer_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError("constructive pilot identity is missing: " + key)
    if not re.fullmatch(r"[a-f0-9]{64}", str(row.get("oracle_sha256", ""))):
        raise ValueError("constructive pilot oracle is not hash-pinned")
    action = row.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("constructive pilot action is missing")
    before_passes = validation.get("before_objective_passes")
    if not isinstance(before_passes, bool) or before_passes != (action.get("kind") == "keep"):
        raise ValueError("constructive edit/keep objective control disagrees with the action")
