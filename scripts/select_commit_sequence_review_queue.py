"""Select a small, source-only verification queue from frozen CommitPackFT rows.

This consumes the earlier public commit-sequence candidate artifact. It skips
reserved splits by their serialized split token before JSON decoding, keeps the
original train/development assignments, and emits metadata only. It does not
establish inferability, license scope, observed chronology, or training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tinycomplete.one_line.contract import EditAction, EditState, apply_action, physical_lines
from tinycomplete.one_line.data import connected_groups

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/prototype/product_r2"
SOURCE_PLAN = REPORT / "commit_source_download_plan.json"
SOURCE_MANIFEST = REPORT / "commit_source_download_manifest.json"
SEQUENCE_PLAN = REPORT / "commit_sequence_review_plan.json"
SEQUENCE_RESULT = REPORT / "commit_sequence_review_result.json"
POOL = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft/prepared/candidates.jsonl")
OUTPUT_DIR = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft/authoring-queue-v4")
PLAN_PATH = REPORT / "commit_sequence_authoring_queue_plan_v4.json"
SELECTOR = Path(__file__).resolve()
SEQUENCE_MODULE = ROOT / "src/tinycomplete/one_line/commit_sequences.py"
GROUPING_MODULE = ROOT / "src/tinycomplete/one_line/data.py"
PRIOR_BUILDER = ROOT / "scripts/build_commit_sequence_pilot.py"
TESTS = ROOT / "tests/test_commit_sequence_review_queue.py"

SCHEMA = "commit-sequence-source-review-queue-v4"
SEED = "commit-sequence-source-review-queue-v4"
CAPS = {"max_rows": 128, "train_rows": 96, "development_rows": 32}
ALLOWED_SPLITS = {"train", "development"}
RESERVED_SPLITS = ("test_new_repo", "test_new_mechanism")
LANGUAGE_ORDER = ("python", "typescript", "go", "rust")
ACTION_ORDER = ("replace_line", "insert_before", "delete_line")

_DECLARATION = {
    "python": re.compile(r"\b(?:async\s+)?(?:def|class|Protocol)\s+[A-Za-z_]\w*"),
    "typescript": re.compile(
        r"\b(?:function|class|interface|type|enum)\s+[A-Za-z_$][\w$]*"
    ),
    "go": re.compile(r"\b(?:func|type)\s+[A-Za-z_]\w*"),
    "rust": re.compile(r"\b(?:fn|struct|enum|trait|impl|type)\s+[A-Za-z_]\w*"),
}
_TYPE_MARKER = re.compile(r"(?:->|::|\b(?:string|number|boolean|bool|int|str|u\d+|i\d+)\b)")
_DOC_OR_COMMENT = re.compile(r"^\s*(?:#|//|/\*|\*|'''|\"\"\")")
_GUARD = re.compile(
    r"\b(?:if|else|match|switch|assert|raise|throw|return|Result|Option|error)\b"
)
_RESERVED_SPLIT_FIELD = re.compile(
    rb'"split"\s*:\s*"(?:test_new_repo|test_new_mechanism)"'
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _hash_path(path: Path) -> str:
    return sha256_file(path)[0]


def _input_identity() -> dict[str, Any]:
    source_plan = json.loads(SOURCE_PLAN.read_text())
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    parent_result = json.loads(SEQUENCE_RESULT.read_text())
    pool_digest, pool_size = sha256_file(POOL)
    if pool_digest != parent_result.get("candidate_sha256"):
        raise ValueError("pinned candidate artifact hash differs from its result manifest")
    if pool_size != parent_result.get("candidate_bytes"):
        raise ValueError("pinned candidate artifact size differs from its result manifest")
    if source_plan.get("revision") != source_manifest.get("dataset_revision"):
        raise ValueError("download plan and source manifest revisions disagree")
    return {
        "schema": SCHEMA,
        "selector_sha256": _hash_path(SELECTOR),
        "test_sha256": _hash_path(TESTS),
        "prior_builder_sha256": _hash_path(PRIOR_BUILDER),
        "reconstruction_module_sha256": _hash_path(SEQUENCE_MODULE),
        "grouping_module_sha256": _hash_path(GROUPING_MODULE),
        "source_download_plan_sha256": _hash_path(SOURCE_PLAN),
        "source_manifest_sha256": _hash_path(SOURCE_MANIFEST),
        "parent_sequence_plan_sha256": _hash_path(SEQUENCE_PLAN),
        "parent_sequence_result_sha256": _hash_path(SEQUENCE_RESULT),
        "parent_split_audit": parent_result["split_audit"],
        "dataset": source_plan["dataset"],
        "dataset_revision": source_manifest["dataset_revision"],
        "source_files": [
            {
                "path": row["path"],
                "size": row["size"],
                "sha256": row["sha256"],
            }
            for row in source_manifest["files"]
        ],
        "candidate_pool_sha256": pool_digest,
        "candidate_pool_bytes": pool_size,
        "candidate_pool_path": str(POOL),
        "seed": SEED,
        "caps": CAPS,
        "eligible_splits": sorted(ALLOWED_SPLITS),
        "reserved_splits_skipped_before_json_decode": list(RESERVED_SPLITS),
        "group_policy": (
            "one row per connected component over source repo aliases, commit, family, "
            "template and normalized near-duplicate key"
        ),
        "selection_policy": (
            "split quotas 96 train/32 development; round-robin language/action strata; "
            "within stratum rank visible contract/context signals then seeded candidate id"
        ),
        "candidate_use": "source/license and inferability review only",
        "human_chronology_claimed": False,
        "candidate_training_accepted": False,
        "teacher_or_provider_calls": 0,
        "new_source_downloads": 0,
    }


def _read_eligible_rows(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with path.open("rb") as stream:
        for raw in stream:
            if _RESERVED_SPLIT_FIELD.search(raw):
                counts["reserved_rows_skipped_before_json_decode"] += 1
                continue
            if not raw.strip():
                continue
            record = json.loads(raw)
            split = record.get("split")
            if split not in ALLOWED_SPLITS:
                raise ValueError("candidate row has an unknown or unreserved split")
            rows.append(record)
            counts[f"input_{split}"] += 1
    return rows, counts


def _visible_context_signals(row: Mapping[str, Any]) -> dict[str, Any]:
    state = row["state"]
    lines = physical_lines(str(state["source"]).encode("utf-8"))
    target = int(state["target_row"])
    action_kind = str(row["action"]["kind"])
    max_target = len(lines) if action_kind == "insert_before" else len(lines) - 1
    if target < 0 or target > max_target:
        raise ValueError("candidate target row is outside its pre-edit source")
    language = str(state["filetype"])
    if language not in _DECLARATION:
        raise ValueError("candidate language is unsupported by the review selector")
    window_start = max(0, target - 16)
    window_end = min(len(lines), target + 5)
    window = [line.content.decode("utf-8") for line in lines[window_start:window_end]]
    prior_window = [line.content.decode("utf-8") for line in lines[window_start:target]]
    history_count = len(state.get("history", ()))
    declaration_visible = any(_DECLARATION[language].search(line) for line in prior_window)
    typed_contract_visible = declaration_visible and any(
        _TYPE_MARKER.search(line) for line in prior_window
    )
    documentation_visible = any(_DOC_OR_COMMENT.search(line) for line in prior_window[-8:])
    guard_or_error_visible = any(_GUARD.search(line) for line in window)
    score = (
        4 * int(typed_contract_visible)
        + 2 * int(documentation_visible)
        + int(guard_or_error_visible)
        + min(history_count, 4)
    )
    return {
        "prior_edit_count": history_count,
        "declaration_visible": declaration_visible,
        "typed_contract_visible": typed_contract_visible,
        "documentation_or_comment_visible": documentation_visible,
        "guard_or_error_context_visible": guard_or_error_visible,
        "selection_priority": score,
    }


def _stable_tie(seed: str, row_id: str) -> str:
    return sha256_bytes((seed + row_id).encode("utf-8"))


def _candidate_summary(row: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    state = row["state"]
    action = row["action"]
    provenance = row["provenance"]
    state_object = EditState.from_mapping(state)
    action_object = EditAction(**action)
    if apply_action(state_object, action_object) != row["after_source"]:
        raise ValueError("candidate action differs from canonical byte-exact replay")
    state_bytes = json.dumps(state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    action_bytes = json.dumps(action, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "candidate_id": row["id"],
        "source_group_id": group_id,
        "split": row["split"],
        "language": state["filetype"],
        "action_kind": action["kind"],
        "source_repo": row["source_repo"],
        "source_aliases": list(row.get("source_aliases", ())),
        "source_revision": row["source_revision"],
        "source_license_claim": row["source_license"],
        "file_path": state["file_id"],
        "public_commit_url": provenance["public_url"],
        "source_record_sha256": provenance["source_record_sha256"],
        "source_before_sha256": provenance["history_origin_sha256"],
        "committed_child_file_sha256": provenance["committed_child_sha256"],
        "pre_state_sha256": sha256_bytes(state_bytes),
        "pre_source_sha256": sha256_bytes(state["source"].encode("utf-8")),
        "post_source_sha256": sha256_bytes(row["after_source"].encode("utf-8")),
        "action_sha256": sha256_bytes(action_bytes),
        "prompt_sha256": sha256_bytes(str(row["prompt"]).encode("utf-8")),
        "target_row": int(state["target_row"]),
        "context_signals": _visible_context_signals(row),
        "source_file_license_status": "unverified_at_exact_revision",
        "human_edit_order_observed": False,
        "history_order": "synthetic ascending file order from reconstructed commit diff",
        "inferability_reviewed": False,
        "objective_verified": False,
        "accepted_training": False,
    }


def _group_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = connected_groups(rows)
    representatives: list[dict[str, Any]] = []
    for indices in groups:
        members = [rows[index] for index in indices]
        splits = {str(row["split"]) for row in members}
        if len(splits) != 1 or not splits <= ALLOWED_SPLITS:
            raise ValueError("connected source group crosses or enters a reserved split")
        group_material = sorted(str(row["id"]) for row in members)
        group_id = sha256_bytes(json.dumps(group_material, separators=(",", ":")).encode())
        summarized = [
            (_candidate_summary(row, group_id), row)
            for row in members
        ]
        summarized.sort(
            key=lambda pair: (
                -int(pair[0]["context_signals"]["selection_priority"]),
                _stable_tie(SEED, str(pair[0]["candidate_id"])),
            )
        )
        selected, row = summarized[0]
        selected["selection_context_priority"] = selected["context_signals"][
            "selection_priority"
        ]
        selected["context_signals"].pop("selection_priority")
        selected["stratum"] = f"{row['state']['filetype']}/{row['action']['kind']}"
        representatives.append(selected)
    return representatives


def _round_robin(representatives: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        buckets[f"{row['language']}/{row['action_kind']}"].append(row)
    for items in buckets.values():
        items.sort(
            key=lambda row: (
                -int(row["selection_context_priority"]),
                _stable_tie(SEED, str(row["candidate_id"])),
            )
        )
    bucket_order = [
        f"{language}/{action}"
        for language in LANGUAGE_ORDER
        for action in ACTION_ORDER
        if f"{language}/{action}" in buckets
    ]
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in bucket_order}
    while len(selected) < limit:
        progressed = False
        for key in bucket_order:
            offset = offsets[key]
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset])
                offsets[key] += 1
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return selected


def select_review_queue(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    representatives = _group_representatives(rows)
    by_split = {
        split: [row for row in representatives if row["split"] == split]
        for split in sorted(ALLOWED_SPLITS)
    }
    chosen: list[dict[str, Any]] = []
    for split, limit in (("train", CAPS["train_rows"]), ("development", CAPS["development_rows"])):
        chosen.extend(_round_robin(by_split[split], limit))
    if len(chosen) > CAPS["max_rows"]:
        raise AssertionError("review queue exceeded its frozen row cap")
    group_ids = [str(row["source_group_id"]) for row in chosen]
    candidate_ids = [str(row["candidate_id"]) for row in chosen]
    if len(set(group_ids)) != len(group_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("review queue contains duplicate source groups or candidates")
    summary = {
        "selected_rows": len(chosen),
        "unique_source_groups": len(set(group_ids)),
        "candidate_ids_unique": len(set(candidate_ids)) == len(candidate_ids),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in chosen).items())),
        "language_counts": dict(sorted(Counter(str(row["language"]) for row in chosen).items())),
        "action_counts": dict(sorted(Counter(str(row["action_kind"]) for row in chosen).items())),
        "context_signal_counts": {
            key: sum(bool(row["context_signals"][key]) for row in chosen)
            for key in (
                "declaration_visible",
                "typed_contract_visible",
                "documentation_or_comment_visible",
                "guard_or_error_context_visible",
            )
        },
        "all_training_accepted_false": all(not row["accepted_training"] for row in chosen),
        "all_file_license_scope_unverified": all(
            row["source_file_license_status"] == "unverified_at_exact_revision" for row in chosen
        ),
        "all_human_edit_order_unobserved": all(
            not row["human_edit_order_observed"] for row in chosen
        ),
    }
    return chosen, summary


def _write_plan() -> dict[str, Any]:
    if PLAN_PATH.exists() or OUTPUT_DIR.exists():
        raise FileExistsError("review queue plan or output already exists")
    identity = _input_identity()
    plan = dict(identity)
    plan["plan_identity_sha256"] = _canonical_hash(identity)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _load_frozen_plan() -> dict[str, Any]:
    if not PLAN_PATH.is_file():
        raise ValueError("freeze the review queue plan before selecting candidates")
    plan = json.loads(PLAN_PATH.read_text())
    identity = {key: value for key, value in plan.items() if key != "plan_identity_sha256"}
    if plan.get("plan_identity_sha256") != _canonical_hash(identity):
        raise ValueError("frozen review queue plan identity is invalid")
    current = _input_identity()
    if current != identity:
        raise ValueError("source, candidate, selector or split fingerprint changed")
    return plan


def _execute() -> dict[str, Any]:
    plan = _load_frozen_plan()
    if OUTPUT_DIR.exists():
        raise FileExistsError("review queue output already exists")
    os.umask(0o077)
    rows, input_counts = _read_eligible_rows(POOL)
    chosen, summary = select_review_queue(rows)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen
    ).encode("utf-8")
    output_manifest = {
        "schema": "commit-sequence-source-review-queue-result-v1",
        "plan_file_sha256": _hash_path(PLAN_PATH),
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "candidate_pool_sha256": plan["candidate_pool_sha256"],
        "candidate_pool_bytes": plan["candidate_pool_bytes"],
        "private_index_sha256": sha256_bytes(payload),
        "private_index_bytes": len(payload),
        "input_counts": dict(sorted(input_counts.items())),
        "summary": summary,
        "reserved_splits_used": [],
        "raw_source_bodies_exported": False,
        "target_text_exported": False,
        "teacher_or_provider_calls": 0,
        "new_source_downloads": 0,
        "accepted_training": 0,
        "quality_evidence": False,
    }
    OUTPUT_DIR.mkdir(parents=True, mode=0o700)
    os.chmod(OUTPUT_DIR, 0o700)
    (OUTPUT_DIR / "review_index.jsonl").write_bytes(payload)
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    return output_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-plan", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = _write_plan() if args.freeze_plan else _execute()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
