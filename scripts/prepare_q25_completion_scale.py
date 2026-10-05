"""Prepare a pinned, CPU-only matched repetition-versus-new-state corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_q25_fim as fim  # noqa: E402

PLAN_PATH = REPO_ROOT / "reports/research/q25_completion_scale_r1/preparation_plan-r2.json"
EXPECTED_PLAN_SHA256 = "be920b0e4e598ae35ad5e5cca85f1a2bc220d0446e2a8f8ba170689230befa2d"
DEFAULT_ARTIFACT_ROOT = Path("/mnt/ssd/tabcomplete-q25-completion-scale-r1")
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_ROOT / "corpus-r3"
PREVIOUS_TRAIN_PATH = Path("/mnt/ssd/tabcomplete-q25-code-cpt-r2/fim/corpus-r3/train.jsonl")
PREVIOUS_DEV_PATH = Path("/mnt/ssd/tabcomplete-q25-code-cpt-r2/fim/corpus-r3/development.jsonl")
PREVIOUS_TRAIN_SHA256 = "341f2d54da2d3c64299c18a918049ded75235ed375137012df71a9ce737fd690"
PREVIOUS_DEV_SHA256 = "43c56d113a819256c7e175ef1f863b9f622a8119f4d3d01b24d455a010fed4ac"
SCHEMA = "q25-completion-scale-corpus-v1"
LANGUAGES = ("python", "rust", "typescript", "go")
INPUT_EDGES = (128, 256, 512, 768, 1024)
TARGET_EDGES = (8, 16, 32, 64, 96)


class PreparationError(RuntimeError):
    """A sanitized preparation error for stable CLI output."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise PreparationError("file_read_failed") from None
    return digest.hexdigest()


def _read_json(path: Path, reason: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise PreparationError(reason) from None


def _read_jsonl_bytes(payload: bytes, reason: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeError):
        raise PreparationError(reason) from None
    if any(not isinstance(row, dict) for row in rows):
        raise PreparationError(reason)
    return rows


def _load_plan(path: Path) -> dict[str, Any]:
    if _sha256_file(path) != EXPECTED_PLAN_SHA256:
        raise PreparationError("scale_plan_hash_mismatch")
    plan = _read_json(path, "scale_plan_invalid")
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "q25-completion-scale-preparation-plan-v1"
        or plan.get("plan_revision") != 2
        or plan.get("branch") != "research/q25-completion-scale-r1"
    ):
        raise PreparationError("scale_plan_identity_mismatch")
    budget = plan.get("budget")
    data = plan.get("data")
    sources = plan.get("sources")
    previous = plan.get("previous_training")
    previous_dev = plan.get("previous_development")
    if not isinstance(budget, dict):
        raise PreparationError("scale_plan_invalid")
    if not isinstance(data, dict):
        raise PreparationError("scale_plan_invalid")
    if not isinstance(sources, dict):
        raise PreparationError("scale_plan_invalid")
    if not isinstance(previous, dict):
        raise PreparationError("scale_plan_invalid")
    if not isinstance(previous_dev, dict):
        raise PreparationError("scale_plan_invalid")
    if (
        budget.get("cpu_preparation_only") is not True
        or budget.get("gpu_allocation_authorized_by_this_plan") is not False
        or budget.get("paid_compute") is not False
        or budget.get("automatic_renewal_use") is not False
        or data.get("new_train_states") != 4096
        or data.get("new_development_states") != 512
        or data.get("variants_per_document_max") != 2
        or data.get("new_development_seed") != 271828
        or data.get("seed") != 314159
        or plan.get("parent_cpt_plan_sha256") != fim.PARENT_CPT_PLAN_SHA256
        or previous.get("sha256") != PREVIOUS_TRAIN_SHA256
        or previous_dev.get("sha256") != PREVIOUS_DEV_SHA256
    ):
        raise PreparationError("scale_plan_policy_mismatch")
    return plan


def _load_previous(
    plan: dict[str, Any],
) -> tuple[bytes, bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    previous = plan["previous_training"]
    previous_dev = plan["previous_development"]
    train_path = Path(previous["path"])
    dev_path = Path(previous_dev["path"])
    try:
        train_bytes = train_path.read_bytes()
        dev_bytes = dev_path.read_bytes()
    except OSError:
        raise PreparationError("previous_prepared_corpus_missing") from None
    if _sha256_bytes(train_bytes) != PREVIOUS_TRAIN_SHA256:
        raise PreparationError("previous_train_hash_mismatch")
    if _sha256_bytes(dev_bytes) != PREVIOUS_DEV_SHA256:
        raise PreparationError("previous_development_hash_mismatch")
    train_rows = _read_jsonl_bytes(train_bytes, "previous_train_invalid")
    dev_rows = _read_jsonl_bytes(dev_bytes, "previous_development_invalid")
    if len(train_rows) != 4096 or len(dev_rows) != 240:
        raise PreparationError("previous_corpus_row_count_mismatch")
    _validate_rows(train_rows, split="train", expected_ids=range(4096))
    _validate_rows(dev_rows, split="development", expected_ids=range(4096, 4336))
    return train_bytes, dev_bytes, train_rows, dev_rows


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    expected_ids: Iterable[int] | None = None,
    require_group_id: bool = False,
) -> None:
    seen: set[int] = set()
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id in seen:
            raise PreparationError("serialized_row_id_invalid")
        seen.add(row_id)
        ids = row.get("input_ids")
        prompt_tokens = row.get("prompt_tokens")
        target_tokens = row.get("target_tokens")
        total_tokens = row.get("total_tokens")
        if (
            row.get("split") != split
            or row.get("prompt_format") != "psm"
            or not isinstance(ids, list)
            or any(isinstance(token, bool) or not isinstance(token, int) for token in ids)
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or prompt_tokens < 1
            or target_tokens < 1
            or prompt_tokens + target_tokens != total_tokens
            or len(ids) != total_tokens
            or total_tokens > 1024
        ):
            raise PreparationError("serialized_row_tokens_invalid")
        for key in (
            "source_content_sha256",
            "repository_identity_sha256",
            "prompt_sha256",
            "target_sha256",
        ):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise PreparationError("serialized_row_provenance_invalid")
        aliases = row.get("repository_alias_sha256")
        if (
            not isinstance(aliases, list)
            or not aliases
            or row["repository_identity_sha256"] not in aliases
        ):
            raise PreparationError("serialized_row_aliases_invalid")
        group_id = row.get("repository_group_sha256")
        if require_group_id and group_id is None:
            raise PreparationError("serialized_row_group_id_missing")
        if group_id is not None and (
            not isinstance(group_id, str)
            or len(group_id) != 64
            or any(character not in "0123456789abcdef" for character in group_id)
        ):
            raise PreparationError("serialized_row_group_id_invalid")
    if expected_ids is not None and seen != set(expected_ids):
        raise PreparationError("serialized_row_id_range_invalid")


def _state_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    start = row.get("region_start")
    end = row.get("region_end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise PreparationError("serialized_row_region_invalid")
    return (
        str(row["source_content_sha256"]),
        str(row["mode"]),
        start,
        end,
        str(row["target_sha256"]),
    )


def _state_id(row: dict[str, Any]) -> str:
    value = json.dumps(_state_key(row), separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(value)


def _aliases(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(alias) for row in rows for alias in row["repository_alias_sha256"]}


def _components(documents: list[Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return transitive alias components and map each source hash to a group ID."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        first, second = sorted((root_left, root_right))
        parent[second] = first

    aliases_by_content: dict[str, tuple[str, ...]] = {}
    for document in documents:
        aliases = tuple(sorted(set(document.repository_alias_sha256)))
        if not aliases or document.repository_identity_sha256 not in aliases:
            raise PreparationError("source_alias_group_invalid")
        aliases_by_content[document.content_sha256] = aliases
        for alias in aliases:
            find(alias)
        for alias in aliases[1:]:
            union(aliases[0], alias)

    grouped_aliases: dict[str, set[str]] = defaultdict(set)
    grouped_contents: dict[str, list[str]] = defaultdict(list)
    for content_hash, aliases in aliases_by_content.items():
        root = find(aliases[0])
        grouped_contents[root].append(content_hash)
        grouped_aliases[root].update(aliases)
    records: list[dict[str, Any]] = []
    content_to_group: dict[str, str] = {}
    for root, component_aliases in grouped_aliases.items():
        sorted_aliases = sorted(component_aliases)
        group_id = _sha256_bytes("\0".join(sorted_aliases).encode("ascii"))
        records.append(
            {
                "group_id": group_id,
                "aliases": sorted_aliases,
                "documents": sorted(grouped_contents[root]),
            }
        )
        for content_hash in grouped_contents[root]:
            content_to_group[content_hash] = group_id
    records.sort(key=lambda group: group["group_id"])
    if len(content_to_group) != len(documents):
        raise PreparationError("duplicate_source_content_in_components")
    return records, content_to_group


def _group_order(seed: int, aliases: list[str]) -> bytes:
    payload = json.dumps(
        {"seed": seed, "aliases": sorted(aliases)}, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).digest()


def _ordered_state(row: dict[str, Any], seed: int, split: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{split}\0{row['source_content_sha256']}\0{row['variant']}".encode("ascii")
    ).digest()


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {language: total * weights[language] for language in LANGUAGES}
    quotas = {language: int(value) for language, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        LANGUAGES,
        key=lambda language: (raw[language] - quotas[language], language),
        reverse=True,
    )
    for language in order[:remainder]:
        quotas[language] += 1
    return quotas


def _select_new_development(
    candidates: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    content_to_group: dict[str, str],
    *,
    previous_aliases: set[str],
    weights: dict[str, float],
    requested: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quotas = _largest_remainder(weights, requested)
    ineligible: Counter[str] = Counter()
    eligible_groups = []
    for group in groups:
        if previous_aliases.intersection(group["aliases"]):
            ineligible["overlaps_previous_training_or_development"] += 1
            continue
        eligible_groups.append(group)
    eligible_groups.sort(
        key=lambda group: (_group_order(seed, group["aliases"]), group["group_id"])
    )
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        group_id = content_to_group.get(row["source_content_sha256"])
        if group_id is None:
            raise PreparationError("development_source_group_missing")
        rows_by_group[group_id].append(row)
    for rows in rows_by_group.values():
        rows.sort(key=lambda row: (row["_order"], row["source_content_sha256"], row["variant"]))

    selected: list[dict[str, Any]] = []
    selected_groups: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for group in eligible_groups:
        if sum(counts.values()) >= requested:
            break
        group_selected = []
        for language in LANGUAGES:
            remaining = quotas[language] - counts[language]
            if remaining <= 0:
                continue
            language_rows = [
                row
                for row in rows_by_group.get(group["group_id"], [])
                if row["language"] == language
            ]
            take = language_rows[:remaining]
            group_selected.extend(take)
            counts[language] += len(take)
        if group_selected:
            selected.extend(group_selected)
            selected_groups.append(group)

    selected_group_order = {
        group["group_id"]: _group_order(seed, group["aliases"]) for group in selected_groups
    }
    selected.sort(
        key=lambda row: (
            selected_group_order[content_to_group[row["source_content_sha256"]]],
            row["_order"],
            row["source_content_sha256"],
            row["variant"],
        )
    )
    selected_aliases = {alias for group in selected_groups for alias in group["aliases"]}
    if previous_aliases.intersection(selected_aliases):
        raise PreparationError("development_group_overlap")
    audit = {
        "requested_states": requested,
        "actual_states": len(selected),
        "quotas": quotas,
        "counts_by_language": dict(sorted(counts.items())),
        "shortages_by_language": {
            language: quotas[language] - counts[language]
            for language in LANGUAGES
            if quotas[language] > counts[language]
        },
        "eligible_group_count": len(eligible_groups),
        "excluded_group_count_by_reason": dict(sorted(ineligible.items())),
        "selected_group_ids": [group["group_id"] for group in selected_groups],
        "selected_group_order_sha256": _sha256_bytes(
            "\n".join(group["group_id"] for group in selected_groups).encode("ascii")
        ),
        "selected_alias_count": len(selected_aliases),
        "selected_aliases_sha256": _sha256_bytes(
            "\n".join(sorted(selected_aliases)).encode("ascii")
        ),
    }
    return selected, selected_groups, audit


def _input_bin(value: int) -> int:
    for index, edge in enumerate(INPUT_EDGES):
        if value <= edge:
            return index
    raise PreparationError("input_tokens_outside_frozen_bins")


def _target_bin(value: int) -> int:
    for index, edge in enumerate(TARGET_EDGES):
        if value <= edge:
            return index
    raise PreparationError("target_tokens_outside_frozen_bins")


def _bin_key(row: dict[str, Any]) -> tuple[int, int]:
    return _input_bin(int(row["total_tokens"])), _target_bin(int(row["target_tokens"]))


def _select_matching_new_train(
    candidates: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_ids = {_state_key(row) for row in previous_rows}
    previous_per_document = Counter(row["source_content_sha256"] for row in previous_rows)
    candidate_ids: set[tuple[str, str, int, int, str]] = set()
    candidate_rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in candidates:
        key = _state_key(row)
        if key in previous_ids:
            rejected["exact_previous_state"] += 1
            continue
        if previous_per_document[row["source_content_sha256"]] >= 2:
            rejected["previous_document_variant_cap"] += 1
            continue
        if key in candidate_ids:
            rejected["duplicate_source_state"] += 1
            continue
        candidate_ids.add(key)
        row["_scale_order"] = _ordered_state(row, seed, "scaled-train")
        candidate_rows.append(row)
    rows_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_document[row["source_content_sha256"]].append(row)
    allowed_candidates: list[dict[str, Any]] = []
    for content_hash, rows in rows_by_document.items():
        rows.sort(key=lambda row: (row["_scale_order"], row["variant"]))
        remaining_variants = max(0, 2 - previous_per_document[content_hash])
        allowed_candidates.extend(rows[:remaining_variants])
        if len(rows) > remaining_variants:
            rejected["combined_document_variant_cap"] += len(rows) - remaining_variants
    by_language_bucket: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {
        language: defaultdict(list) for language in LANGUAGES
    }
    for row in allowed_candidates:
        by_language_bucket[str(row["language"])][_bin_key(row)].append(row)
    for bucket_map in by_language_bucket.values():
        for rows in bucket_map.values():
            rows.sort(
                key=lambda row: (row["_scale_order"], row["source_content_sha256"], row["variant"])
            )

    wanted: dict[str, Counter[tuple[int, int]]] = {language: Counter() for language in LANGUAGES}
    for row in previous_rows:
        wanted[str(row["language"])][_bin_key(row)] += 1
    selected: list[dict[str, Any]] = []
    actual: dict[str, Counter[tuple[int, int]]] = {language: Counter() for language in LANGUAGES}
    remaining_candidates: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {}
    filled: dict[str, Counter[tuple[int, int]]] = {language: Counter() for language in LANGUAGES}
    for language in LANGUAGES:
        remaining_candidates[language] = {
            bucket: rows[:] for bucket, rows in by_language_bucket[language].items()
        }
        for bucket in sorted(wanted[language]):
            rows = by_language_bucket[language].get(bucket, [])
            take = min(wanted[language][bucket], len(rows))
            selected.extend(rows[:take])
            actual[language][bucket] += take
            filled[language][bucket] += take
            remaining_candidates[language][bucket] = rows[take:]

    # Fill target-bin shortages from nearest same-language bins, after all exact bins.
    for language in LANGUAGES:
        buckets = sorted(wanted[language])
        for target_bucket in buckets:
            shortage = wanted[language][target_bucket] - filled[language][target_bucket]
            if shortage <= 0:
                continue
            donor_buckets = sorted(
                by_language_bucket[language],
                key=lambda bucket: (
                    abs(bucket[0] - target_bucket[0]) + abs(bucket[1] - target_bucket[1]),
                    bucket,
                ),
            )
            for donor_bucket in donor_buckets:
                if shortage <= 0:
                    break
                donor = remaining_candidates[language].get(donor_bucket, [])
                if not donor:
                    continue
                take = min(shortage, len(donor))
                picked = donor[:take]
                selected.extend(picked)
                filled[language][target_bucket] += take
                remaining_candidates[language][donor_bucket] = donor[take:]
                shortage -= take

    # The exact-match reservation and fallback share candidates; dedupe selected keys.
    unique_selected: list[dict[str, Any]] = []
    seen_selected: set[tuple[str, str, int, int, str]] = set()
    for row in selected:
        key = _state_key(row)
        if key not in seen_selected:
            seen_selected.add(key)
            unique_selected.append(row)
    unique_selected.sort(
        key=lambda row: (row["_scale_order"], row["source_content_sha256"], row["variant"])
    )
    # A fallback candidate may have been drawn from a bin already selected exactly.
    # Recompute truthful achieved bins from the actual selected states.
    actual = {language: Counter() for language in LANGUAGES}
    for row in unique_selected:
        actual[str(row["language"])][_bin_key(row)] += 1
    desired_rows = sum(sum(hist.values()) for hist in wanted.values())
    if len(unique_selected) > desired_rows or len(unique_selected) > count:
        raise PreparationError("new_training_selection_exceeded_cap")
    wanted_json = {
        language: {f"input_{i}_target_{t}": n for (i, t), n in sorted(hist.items())}
        for language, hist in wanted.items()
    }
    actual_json = {
        language: {f"input_{i}_target_{t}": n for (i, t), n in sorted(hist.items())}
        for language, hist in actual.items()
    }
    shortage_json = {
        language: {
            f"input_{i}_target_{t}": wanted[language][(i, t)] - filled[language][(i, t)]
            for i, t in sorted(wanted[language])
            if wanted[language][(i, t)] > filled[language][(i, t)]
        }
        for language in LANGUAGES
    }
    shortage_json = {language: values for language, values in shortage_json.items() if values}
    audit = {
        "requested_states": count,
        "actual_states": len(unique_selected),
        "counts_by_language": dict(
            sorted(Counter(row["language"] for row in unique_selected).items())
        ),
        "target_histogram_by_language": wanted_json,
        "actual_histogram_by_language": actual_json,
        "shortages_by_language_and_bins": shortage_json,
        "candidate_rejections": dict(sorted(rejected.items())),
        "candidate_states_before_filtering": len(candidates),
        "candidate_states_after_previous_and_variant_filters": len(allowed_candidates),
    }
    return unique_selected, audit


def _generate_states(
    documents: list[Any], tokenizer: Any, *, plan: dict[str, Any], seed: int, output_split: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    data = plan["data"]
    marker_map: dict[str, int] = {}
    for name, marker in zip(fim.MARKER_NAMES, fim.MARKERS, strict=True):
        ids = fim._token_ids(tokenizer, marker)
        if len(ids) != 1:
            raise PreparationError("fim_marker_not_single_token")
        marker_map[name] = ids[0]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    marker_ids: tuple[int, int, int] = (
        marker_map["fim_prefix"],
        marker_map["fim_suffix"],
        marker_map["fim_middle"],
    )
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise PreparationError("tokenizer_eos_missing")
    if len(set(marker_ids)) != 3 or eos_token_id in marker_ids:
        raise PreparationError("tokenizer_special_token_identity_invalid")
    states, rejected = fim._generate_states(
        documents,
        tokenizer,
        seed=seed,
        prefix_limit=int(data["prefix_tokens_max"]),
        suffix_limit=int(data["suffix_tokens_max"]),
        target_limit=int(data["target_tokens_including_eos_max"]),
        total_limit=int(data["max_total_tokens"]),
        eos_token_id=eos_token_id,
        marker_ids=marker_ids,
    )
    for row in states:
        row["split"] = output_split
        row["_order"] = _ordered_state(row, seed, output_split)
    states.sort(key=lambda row: (row["_order"], row["source_content_sha256"], row["variant"]))
    return states, rejected


def _row_bytes(row: dict[str, Any], row_id: int, split: str) -> bytes:
    public = fim._public_row(row, row_id)
    public["split"] = split
    if "repository_group_sha256" in row:
        public["repository_group_sha256"] = row["repository_group_sha256"]
    try:
        return (
            json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
            + b"\n"
        )
    except (TypeError, ValueError):
        raise PreparationError("corpus_serialization_failed") from None


def _encoded_new_rows(rows: list[dict[str, Any]], start_id: int, split: str) -> bytes:
    return b"".join(_row_bytes(row, start_id + index, split) for index, row in enumerate(rows))


def _file_record(payload: bytes) -> dict[str, Any]:
    return {"sha256": _sha256_bytes(payload), "bytes": len(payload)}


def _split_record(filename: str, payload: bytes, rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = _read_jsonl_bytes(payload, "serialized_split_invalid")
    return {
        "file": filename,
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "row_count": len(parsed),
        "input_tokens": sum(int(row["total_tokens"]) for row in parsed),
        "target_tokens": sum(int(row["target_tokens"]) for row in parsed),
    }


def _tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _safe_output_path(output_dir: Path, artifact_root: Path) -> None:
    root = artifact_root.resolve()
    output = output_dir.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        raise PreparationError("output_path_outside_artifact_root") from None
    if output == root or output.exists() or output_dir.is_symlink():
        raise PreparationError("output_directory_already_exists_or_invalid")


def _build_metadata(
    *,
    plan: dict[str, Any],
    inputs: dict[str, Any],
    marker_ids: dict[str, int],
    eos_token_id: int,
    files: dict[str, bytes],
    rows_by_file: dict[str, list[dict[str, Any]]],
    selection: dict[str, Any],
    source_script_sha256: str,
) -> dict[str, Any]:
    split_names = {
        "repeat_train": "repeat_train.jsonl",
        "scaled_train": "scaled_train.jsonl",
        "development_new": "development_new.jsonl",
        "development_previous": "development_previous.jsonl",
    }
    file_records = {name: _file_record(payload) for name, payload in sorted(files.items())}
    split_records = {
        split: _split_record(filename, files[filename], rows_by_file[filename])
        for split, filename in split_names.items()
    }
    all_data_bytes = sum(len(payload) for payload in files.values())
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "preparation_plan_sha256": EXPECTED_PLAN_SHA256,
        "preparation_plan_revision": plan["plan_revision"],
        "parent_cpt_plan_sha256": plan["parent_cpt_plan_sha256"],
        "parent_preparation_plan": plan["parent_preparation_plan"],
        "tokenizer_id": fim.TOKENIZER_ID,
        "tokenizer_revision": fim.TOKENIZER_REVISION,
        "tokenizer_sha256": plan["model"]["tokenizer_sha256"],
        "eos_token_id": eos_token_id,
        "fim_marker_ids": marker_ids,
        "source_script_sha256": source_script_sha256,
        "files": file_records,
        "splits": split_records,
        "input_provenance": {
            "dataset_id": fim.DATASET_ID,
            "dataset_revision": plan["sources"]["revision"],
            "raw_pool": inputs["pool_path"],
            "pool_files": inputs["pool_hashes"],
            "train_manifest": inputs["train_manifest"],
            "development_manifest": inputs["development_manifest"],
            "benchmark_exclusion_sha256": inputs["benchmark_exclusion_sha256"],
            "benchmark_exclusion_count": inputs["benchmark_exclusion_count"],
            "allowed_licenses": inputs["allowed_licenses"],
            "previous_train_sha256": PREVIOUS_TRAIN_SHA256,
            "previous_development_sha256": PREVIOUS_DEV_SHA256,
        },
        "selection": selection,
        "data_files_bytes": all_data_bytes,
        "training_input_tokens_include_prompt_target_and_eos": True,
        "development_input_tokens_include_prompt_target_and_eos": True,
        "sealed_test_accessed": False,
        "raw_source_content_emitted": False,
        "gpu_allocation_authorized_by_preparation": False,
    }


def _write_corpus(
    output_dir: Path,
    artifact_root: Path,
    plan: dict[str, Any],
    files: dict[str, bytes],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    _safe_output_path(output_dir, artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(artifact_root).free < int(plan["budget"]["minimum_free_bytes"]):
        raise PreparationError("minimum_free_space_not_available")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    existing_root_bytes = _tree_bytes(artifact_root)
    stage = Path(tempfile.mkdtemp(prefix=".q25-completion-scale-", dir=output_dir.parent))
    try:
        for filename, payload in files.items():
            (stage / filename).write_bytes(payload)
        metadata_payload = (
            json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8")
            + b"\n"
        )
        (stage / "corpus_metadata.json").write_bytes(metadata_payload)
        stage_bytes = _tree_bytes(stage)
        if existing_root_bytes + stage_bytes > int(plan["budget"]["new_artifact_bytes_cap"]):
            raise PreparationError("new_artifact_bytes_cap_exceeded")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, output_dir)
        return {**metadata, "artifact_bytes": stage_bytes}
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _verify_existing(
    output_dir: Path,
    artifact_root: Path,
    plan: dict[str, Any],
    inputs: dict[str, Any],
    previous_train_bytes: bytes,
    previous_dev_bytes: bytes,
    train_documents: list[Any],
) -> dict[str, Any]:
    try:
        output_dir.resolve().relative_to(artifact_root.resolve())
    except ValueError:
        raise PreparationError("output_path_outside_artifact_root") from None
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise PreparationError("existing_corpus_missing_or_invalid")
    metadata = _read_json(output_dir / "corpus_metadata.json", "existing_metadata_invalid")
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != SCHEMA
        or metadata.get("preparation_plan_sha256") != EXPECTED_PLAN_SHA256
        or metadata.get("parent_cpt_plan_sha256") != plan["parent_cpt_plan_sha256"]
        or metadata.get("tokenizer_sha256") != plan["model"]["tokenizer_sha256"]
        or metadata.get("source_script_sha256") != _sha256_file(Path(__file__))
        or metadata.get("sealed_test_accessed") is not False
        or metadata.get("raw_source_content_emitted") is not False
        or metadata.get("gpu_allocation_authorized_by_preparation") is not False
    ):
        raise PreparationError("existing_corpus_identity_mismatch")
    provenance = metadata.get("input_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("pool_files") != inputs["pool_hashes"]
        or provenance.get("train_manifest") != inputs["train_manifest"]
        or provenance.get("development_manifest") != inputs["development_manifest"]
        or provenance.get("benchmark_exclusion_sha256") != inputs["benchmark_exclusion_sha256"]
    ):
        raise PreparationError("existing_source_identity_mismatch")
    expected_names = {
        "repeat_train.jsonl",
        "scaled_train.jsonl",
        "development_new.jsonl",
        "development_previous.jsonl",
    }
    if {path.name for path in output_dir.iterdir()} != expected_names | {"corpus_metadata.json"}:
        raise PreparationError("existing_corpus_file_set_mismatch")
    files = metadata.get("files")
    splits = metadata.get("splits")
    if (
        not isinstance(files, dict)
        or set(files) != expected_names
        or not isinstance(splits, dict)
        or set(splits)
        != {"repeat_train", "scaled_train", "development_new", "development_previous"}
    ):
        raise PreparationError("existing_corpus_ledger_invalid")
    parsed_files: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(expected_names):
        path = output_dir / name
        record = files.get(name)
        if not isinstance(record, dict) or path.stat().st_size != record.get("bytes"):
            raise PreparationError("existing_corpus_file_size_mismatch")
        if _sha256_file(path) != record.get("sha256"):
            raise PreparationError("existing_corpus_file_hash_mismatch")
        payload = path.read_bytes()
        parsed_files[name] = _read_jsonl_bytes(payload, "existing_corpus_jsonl_invalid")
    if (output_dir / "repeat_train.jsonl").read_bytes() != previous_train_bytes:
        raise PreparationError("existing_repeat_train_changed")
    if (output_dir / "development_previous.jsonl").read_bytes() != previous_dev_bytes:
        raise PreparationError("existing_previous_development_changed")
    if not (output_dir / "scaled_train.jsonl").read_bytes().startswith(previous_train_bytes):
        raise PreparationError("existing_scaled_train_prefix_changed")
    _validate_rows(parsed_files["repeat_train.jsonl"], split="train", expected_ids=range(4096))
    _validate_rows(
        parsed_files["scaled_train.jsonl"],
        split="train",
        expected_ids=range(len(parsed_files["scaled_train.jsonl"])),
    )
    _validate_rows(
        parsed_files["scaled_train.jsonl"][4096:],
        split="train",
        expected_ids=range(4096, len(parsed_files["scaled_train.jsonl"])),
        require_group_id=True,
    )
    _validate_rows(
        parsed_files["development_new.jsonl"],
        split="development",
        require_group_id=True,
    )
    _validate_rows(
        parsed_files["development_previous.jsonl"],
        split="development",
        expected_ids=range(4096, 4336),
    )
    if [row["id"] for row in parsed_files["development_new.jsonl"]] != list(
        range(8192, 8192 + len(parsed_files["development_new.jsonl"]))
    ):
        raise PreparationError("existing_new_development_ids_invalid")
    selection = metadata.get("selection")
    if not isinstance(selection, dict):
        raise PreparationError("existing_selection_ledger_invalid")
    new_training_record = selection.get("new_training")
    new_development_record = selection.get("new_development")
    if not isinstance(new_training_record, dict) or not isinstance(new_development_record, dict):
        raise PreparationError("existing_selection_ledger_invalid")
    actual_new_train = new_training_record.get("actual_states")
    if isinstance(actual_new_train, bool) or not isinstance(actual_new_train, int):
        raise PreparationError("existing_selection_ledger_invalid")
    if len(parsed_files["scaled_train.jsonl"]) != 4096 + actual_new_train:
        raise PreparationError("existing_scaled_train_count_mismatch")
    scaled_keys = [_state_key(row) for row in parsed_files["scaled_train.jsonl"]]
    if len(scaled_keys) != len(set(scaled_keys)):
        raise PreparationError("existing_scaled_train_state_duplicate")
    dev_keys = [_state_key(row) for row in parsed_files["development_new.jsonl"]]
    if len(dev_keys) != len(set(dev_keys)) or set(dev_keys).intersection(scaled_keys):
        raise PreparationError("existing_development_state_identity_invalid")
    split_aliases = {name: _aliases(rows) for name, rows in parsed_files.items()}
    if split_aliases["scaled_train.jsonl"].intersection(split_aliases["development_new.jsonl"]):
        raise PreparationError("existing_train_development_group_overlap")
    components, content_to_group = _components(train_documents)
    del components
    previous_reserved_aliases = _aliases(
        parsed_files["repeat_train.jsonl"] + parsed_files["development_previous.jsonl"]
    )
    previous_dev_aliases = _aliases(parsed_files["development_previous.jsonl"])
    new_train_rows = parsed_files["scaled_train.jsonl"][4096:]
    new_dev_rows = parsed_files["development_new.jsonl"]
    for row in new_train_rows + new_dev_rows:
        if content_to_group.get(row["source_content_sha256"]) != row.get("repository_group_sha256"):
            raise PreparationError("existing_row_group_identity_mismatch")
    if any(
        previous_reserved_aliases.intersection(row["repository_alias_sha256"])
        for row in new_dev_rows
    ):
        raise PreparationError("existing_development_previous_alias_overlap")
    if any(
        previous_dev_aliases.intersection(row["repository_alias_sha256"]) for row in new_train_rows
    ):
        raise PreparationError("existing_training_previous_development_alias_overlap")
    selected_dev_groups = {row["repository_group_sha256"] for row in new_dev_rows}
    if selected_dev_groups != set(new_development_record.get("selected_group_ids", [])):
        raise PreparationError("existing_development_group_ledger_mismatch")
    if selected_dev_groups.intersection(row["repository_group_sha256"] for row in new_train_rows):
        raise PreparationError("existing_train_development_group_overlap")
    for _split, record in splits.items():
        filename = record.get("file")
        if filename not in expected_names:
            raise PreparationError("existing_split_file_invalid")
        rows = parsed_files[filename]
        if (
            record.get("sha256") != files[filename]["sha256"]
            or record.get("bytes") != files[filename]["bytes"]
            or record.get("row_count") != len(rows)
            or record.get("input_tokens") != sum(row["total_tokens"] for row in rows)
            or record.get("target_tokens") != sum(row["target_tokens"] for row in rows)
        ):
            raise PreparationError("existing_split_totals_mismatch")
    total_exposure_tokens = 2 * sum(
        row["total_tokens"] for row in parsed_files["repeat_train.jsonl"]
    ) + sum(row["total_tokens"] for row in parsed_files["scaled_train.jsonl"])
    if total_exposure_tokens != selection.get(
        "total_campaign_exposure_input_tokens"
    ) or total_exposure_tokens > int(plan["budget"]["maximum_campaign_input_tokens"]):
        raise PreparationError("existing_campaign_token_budget_mismatch")
    if _tree_bytes(artifact_root) > int(plan["budget"]["new_artifact_bytes_cap"]):
        raise PreparationError("new_artifact_bytes_cap_exceeded")
    return metadata


def prepare(
    *,
    plan_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    artifact_root: Path,
    verify_existing: bool,
) -> dict[str, Any]:
    plan = _load_plan(plan_path)
    parent_plan_path = fim.PLAN_PATH
    _, inputs, train_documents, _ = fim._load_pinned_inputs(parent_plan_path)
    previous_train_bytes, previous_dev_bytes, previous_train_rows, previous_dev_rows = (
        _load_previous(plan)
    )
    if verify_existing:
        return _verify_existing(
            output_dir,
            artifact_root,
            plan,
            inputs,
            previous_train_bytes,
            previous_dev_bytes,
            train_documents,
        )
    _safe_output_path(output_dir, artifact_root)
    tokenizer = fim._load_local_tokenizer(tokenizer_path, str(plan["model"]["tokenizer_sha256"]))
    data = plan["data"]
    weights = {language: float(data["language_weights"][language]) for language in LANGUAGES}
    if set(weights) != set(LANGUAGES) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise PreparationError("language_weight_plan_invalid")

    components, content_to_group = _components(train_documents)
    old_train_aliases = _aliases(previous_train_rows)
    old_development_aliases = _aliases(previous_dev_rows)
    old_aliases = old_train_aliases | old_development_aliases
    eligible_group_ids = {
        group["group_id"] for group in components if not old_aliases.intersection(group["aliases"])
    }
    eligible_dev_documents = [
        document
        for document in train_documents
        if content_to_group[document.content_sha256] in eligible_group_ids
    ]
    development_candidates, development_rejected = _generate_states(
        eligible_dev_documents,
        tokenizer,
        plan=plan,
        seed=int(data["new_development_seed"]),
        output_split="development",
    )
    new_dev_internal, selected_groups, development_audit = _select_new_development(
        development_candidates,
        components,
        content_to_group,
        previous_aliases=old_aliases,
        weights=weights,
        requested=int(data["new_development_states"]),
        seed=int(data["new_development_seed"]),
    )
    heldout_group_ids = {group["group_id"] for group in selected_groups}
    heldout_aliases = {alias for group in selected_groups for alias in group["aliases"]}
    prior_development_group_ids = {
        group["group_id"]
        for group in components
        if old_development_aliases.intersection(group["aliases"])
    }
    eligible_train_documents = [
        document
        for document in train_documents
        if content_to_group[document.content_sha256] not in heldout_group_ids
        and content_to_group[document.content_sha256] not in prior_development_group_ids
    ]
    train_candidates, train_rejected = _generate_states(
        eligible_train_documents,
        tokenizer,
        plan=plan,
        seed=int(data["seed"]),
        output_split="train",
    )
    new_train_internal, train_audit = _select_matching_new_train(
        train_candidates,
        previous_train_rows,
        count=int(data["new_train_states"]),
        seed=314160,
    )
    if len(new_train_internal) > int(data["new_train_states"]):
        raise PreparationError("new_training_count_exceeded")
    if heldout_aliases.intersection(
        _aliases([fim._public_row(row, 0) for row in new_train_internal])
    ):
        raise PreparationError("selected_train_development_alias_overlap")
    if old_development_aliases.intersection(
        _aliases([fim._public_row(row, 0) for row in new_train_internal])
    ):
        raise PreparationError("selected_train_previous_development_alias_overlap")
    new_train_keys = {_state_key(row) for row in new_train_internal}
    if new_train_keys.intersection({_state_key(row) for row in previous_train_rows}):
        raise PreparationError("selected_train_previous_state_overlap")
    if len({_state_key(row) for row in new_train_internal}) != len(new_train_internal):
        raise PreparationError("selected_new_training_state_duplicate")
    if len({_state_key(row) for row in new_dev_internal}) != len(new_dev_internal):
        raise PreparationError("selected_new_development_state_duplicate")

    old_train_state_ids = {_state_id(row) for row in previous_train_rows}
    new_train_state_ids = [_state_id(row) for row in new_train_internal]
    new_dev_state_ids = [_state_id(row) for row in new_dev_internal]
    combined_train_ids = old_train_state_ids.union(new_train_state_ids)
    if len(combined_train_ids) != 4096 + len(new_train_internal):
        raise PreparationError("scaled_training_state_identity_duplicate")
    if set(new_dev_state_ids).intersection(combined_train_ids):
        raise PreparationError("train_development_state_identity_overlap")

    for row in new_train_internal:
        row["repository_group_sha256"] = content_to_group[row["source_content_sha256"]]
    for row in new_dev_internal:
        row["repository_group_sha256"] = content_to_group[row["source_content_sha256"]]
    new_train_bytes = _encoded_new_rows(new_train_internal, 4096, "train")
    new_dev_bytes = _encoded_new_rows(
        new_dev_internal, int(data["development_id_offset"]), "development"
    )
    files = {
        "repeat_train.jsonl": previous_train_bytes,
        "scaled_train.jsonl": previous_train_bytes + new_train_bytes,
        "development_new.jsonl": new_dev_bytes,
        "development_previous.jsonl": previous_dev_bytes,
    }
    parsed_files = {
        name: _read_jsonl_bytes(payload, "generated_corpus_invalid")
        for name, payload in files.items()
    }
    _validate_rows(parsed_files["repeat_train.jsonl"], split="train", expected_ids=range(4096))
    _validate_rows(
        parsed_files["scaled_train.jsonl"],
        split="train",
        expected_ids=range(len(parsed_files["scaled_train.jsonl"])),
    )
    _validate_rows(
        parsed_files["scaled_train.jsonl"][4096:],
        split="train",
        expected_ids=range(4096, len(parsed_files["scaled_train.jsonl"])),
        require_group_id=True,
    )
    _validate_rows(
        parsed_files["development_new.jsonl"],
        split="development",
        require_group_id=True,
    )
    _validate_rows(
        parsed_files["development_previous.jsonl"],
        split="development",
        expected_ids=range(4096, 4336),
    )
    if [row["id"] for row in parsed_files["development_new.jsonl"]] != list(
        range(
            int(data["development_id_offset"]),
            int(data["development_id_offset"]) + len(new_dev_internal),
        )
    ):
        raise PreparationError("generated_development_ids_invalid")
    previous_training_input_tokens = sum(row["total_tokens"] for row in previous_train_rows)
    new_training_input_tokens = sum(row["total_tokens"] for row in new_train_internal)
    repeat_exposure_tokens = 2 * previous_training_input_tokens
    scaled_exposure_tokens = previous_training_input_tokens + new_training_input_tokens
    total_campaign_exposure_tokens = repeat_exposure_tokens + scaled_exposure_tokens
    if total_campaign_exposure_tokens > int(plan["budget"]["maximum_campaign_input_tokens"]):
        raise PreparationError("campaign_training_input_token_cap_exceeded")

    marker_ids = {
        name: fim._token_ids(tokenizer, marker)[0]
        for name, marker in zip(fim.MARKER_NAMES, fim.MARKERS, strict=True)
    }
    eos_token_id = int(tokenizer.eos_token_id)
    selection = {
        "previous_training": {
            "states": len(previous_train_rows),
            "counts_by_language": dict(
                sorted(Counter(row["language"] for row in previous_train_rows).items())
            ),
            "state_ids_sha256": _sha256_bytes(
                "\n".join(sorted(old_train_state_ids)).encode("ascii")
            ),
        },
        "new_training": train_audit,
        "new_development": development_audit,
        "new_development_rejected_states": dict(sorted(development_rejected.items())),
        "new_training_rejected_states": dict(sorted(train_rejected.items())),
        "new_training_excluded_previous_development_groups": len(prior_development_group_ids),
        "new_training_input_tokens": new_training_input_tokens,
        "previous_training_input_tokens": previous_training_input_tokens,
        "repeat_arm_exposure_input_tokens": repeat_exposure_tokens,
        "scaled_arm_exposure_input_tokens": scaled_exposure_tokens,
        "total_campaign_exposure_input_tokens": total_campaign_exposure_tokens,
        "maximum_campaign_input_tokens": int(plan["budget"]["maximum_campaign_input_tokens"]),
        "new_training_states_sha256": _sha256_bytes(
            "\n".join(sorted(new_train_state_ids)).encode("ascii")
        ),
        "new_development_states_sha256": _sha256_bytes(
            "\n".join(sorted(new_dev_state_ids)).encode("ascii")
        ),
        "combined_training_distinct_states": len(combined_train_ids),
        "all_new_development_groups_disjoint_from_previous_training_and_development": True,
        "all_new_training_groups_disjoint_from_new_development": True,
        "transitive_repository_alias_components": True,
        "development_group_order": "sha256(compact-json({seed, sorted aliases}))",
        "training_state_selection_order_seed": 314160,
        "variant_limit_per_document": 2,
        "new_development_feeder_split": "train",
        "old_development_secondary_only": True,
    }
    metadata = _build_metadata(
        plan=plan,
        inputs=inputs,
        marker_ids=marker_ids,
        eos_token_id=eos_token_id,
        files=files,
        rows_by_file=parsed_files,
        selection=selection,
        source_script_sha256=_sha256_file(Path(__file__)),
    )
    return _write_corpus(output_dir, artifact_root, plan, files, metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=fim.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        metadata = prepare(
            plan_path=args.plan,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            artifact_root=args.artifact_root,
            verify_existing=args.verify_existing,
        )
        summary = {
            "status": metadata["status"],
            "output_dir": str(args.output_dir),
            "plan_sha256": metadata["preparation_plan_sha256"],
            "repeat_train_rows": metadata["splits"]["repeat_train"]["row_count"],
            "scaled_train_rows": metadata["splits"]["scaled_train"]["row_count"],
            "new_development_rows": metadata["splits"]["development_new"]["row_count"],
            "corpus_sha256": _sha256_bytes(
                "\0".join(
                    f"{name}:{record['sha256']}"
                    for name, record in sorted(metadata["files"].items())
                ).encode("ascii")
            ),
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except PreparationError as exc:
        print(json.dumps({"status": "FAIL", "reason": exc.reason}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "FAIL", "reason": "unexpected_internal_error"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
