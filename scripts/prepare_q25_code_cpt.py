"""Prepare a pinned, local-only Qwen2.5 code-CPT corpus from the frozen pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tinycomplete.code_cpt.data import (
    BlockPacker,
    RepoSplit,
    SourceFilter,
    allocate_blocks,
    repository_identity,
)
from tinycomplete.code_cpt.prepare import DATASET_ID, research_split_for_bucket

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PLAN_SHA256 = "1aaff2ee4f4e0adeeafb87e616f99e657f7ff0be4fa438f40ecbb1150aeed1f9"
DATASET_REVISION = "17cad72c886a2858e08d4c349a00d6466f54df63"
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
MODEL_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
LANGUAGE_ORDER = ("python", "rust", "typescript", "go")
DEFAULT_OUTPUT_DIR = Path("/mnt/ssd/tabcomplete-q25-code-cpt-r2/corpus")
DEFAULT_PREVIOUS_ROOT = Path("/mnt/ssd/tabcomplete-preserved-research/model_data_r2")
REPOSITORY_KEYS = {
    "repository",
    "repository_name",
    "repo",
    "repo_name",
    "max_stars_repo_name",
    "max_forks_repo_name",
    "max_issues_repo_name",
}
REPOSITORY_HASH = re.compile(r"^[0-9a-fA-F]{64}$")


class PreparationError(RuntimeError):
    """A sanitized, stable preparation failure."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


@dataclass(frozen=True)
class Candidate:
    language: str
    source_file: Path
    offset: int
    repository: str
    aliases: tuple[str, ...]
    path: str
    content_sha256: str
    licenses: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise PreparationError("file_read_failed") from None
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_plan(plan_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise PreparationError("plan_hash_mismatch")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise PreparationError("plan_invalid") from None
    if sha256_file(config_path) != plan.get("config_sha256"):
        raise PreparationError("config_hash_mismatch")
    configuration = plan.get("configuration")
    if not isinstance(configuration, dict):
        raise PreparationError("plan_configuration_missing")
    data = configuration.get("data")
    model = configuration.get("model")
    if (
        not isinstance(data, dict)
        or not isinstance(model, dict)
        or configuration.get("schema") != "q25-code-cpt-plan-v1"
        or data.get("source") != DATASET_ID
        or data.get("revision") != DATASET_REVISION
        or model.get("id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or data.get("test_access") is not False
    ):
        raise PreparationError("plan_identity_mismatch")
    if tuple(data.get("languages", {})) != LANGUAGE_ORDER:
        weights = data.get("languages", {})
        if set(weights) != set(LANGUAGE_ORDER):
            raise PreparationError("language_plan_mismatch")
    return plan, configuration


def load_local_tokenizer(tokenizer_path: Path, plan: dict[str, Any]):
    """Load only the pinned local tokenizer files; never resolve network credentials."""
    model_files = plan.get("existing_model_files", {})
    expected_files = {
        name: model_files.get(name)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
    }
    for name, expected in expected_files.items():
        path = tokenizer_path / name
        if not isinstance(expected, dict) or not path.is_file():
            raise PreparationError("local_tokenizer_file_missing")
        try:
            if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get(
                "sha256"
            ):
                raise PreparationError("local_tokenizer_hash_mismatch")
        except OSError:
            raise PreparationError("local_tokenizer_file_unreadable") from None
    if (
        plan["configuration"]["model"].get("tokenizer_sha256")
        != expected_files["tokenizer.json"]["sha256"]
    ):
        raise PreparationError("tokenizer_plan_hash_mismatch")
    if tokenizer_path.name != MODEL_REVISION:
        raise PreparationError("local_tokenizer_revision_mismatch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception:
        raise PreparationError("local_tokenizer_load_failed") from None
    if tokenizer.eos_token_id is None:
        raise PreparationError("tokenizer_eos_missing")
    return tokenizer


def _read_repo_refs(
    value: Any, names: set[str], repository_hashes: set[str], content_hashes: set[str]
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.lower()
            if (
                normalized_key == "content_sha256"
                and isinstance(child, str)
                and REPOSITORY_HASH.fullmatch(child)
            ):
                content_hashes.add(child.lower())
                continue
            if normalized_key in {"repository_identity_sha256", "repository_hash"}:
                values = child if isinstance(child, list) else [child]
                repository_hashes.update(
                    item.lower()
                    for item in values
                    if isinstance(item, str) and REPOSITORY_HASH.fullmatch(item)
                )
            elif normalized_key == "repository_alias_sha256":
                values = child if isinstance(child, list) else [child]
                repository_hashes.update(
                    item.lower()
                    for item in values
                    if isinstance(item, str) and REPOSITORY_HASH.fullmatch(item)
                )
            elif normalized_key in REPOSITORY_KEYS:
                values = child if isinstance(child, list) else [child]
                for item in values:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    item = item.strip()
                    if REPOSITORY_HASH.fullmatch(item):
                        repository_hashes.add(item.lower())
                    else:
                        names.add(item)
            else:
                _read_repo_refs(child, names, repository_hashes, content_hashes)
    elif isinstance(value, list):
        for child in value:
            _read_repo_refs(child, names, repository_hashes, content_hashes)


def _read_document(path: Path) -> Any:
    try:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise PreparationError("reservation_manifest_invalid") from None


def collect_reserved_repositories(
    repo_root: Path,
    previous_root: Path,
    excluded_repository_path: Path,
    excluded_repository_sha256: str,
) -> tuple[set[str], set[str], set[str]]:
    """Collect identity manifests only; benchmark labels and source fixtures stay sealed."""
    names: set[str] = set()
    hashes: set[str] = set()
    content_hashes: set[str] = set()
    stage1 = previous_root / "stage1-corpus/code_cpt_corpus"
    old_ids = stage1 / "validation_repositories.json"
    if old_ids.is_file():
        value = _read_document(old_ids)
        if isinstance(value, dict):
            names.update(str(name) for name in value)
        elif isinstance(value, list):
            names.update(item for item in value if isinstance(item, str))
    old_manifest = previous_root / "previous-corpus/code_cpt_research_r1/corpus"
    for name in ("development_manifest.jsonl", "test_manifest.jsonl"):
        path = old_manifest / name
        if path.is_file():
            _read_repo_refs(_read_document(path), names, hashes, content_hashes)
    if (stage1 / "validation_records.jsonl").is_file():
        _read_repo_refs(
            _read_document(stage1 / "validation_records.jsonl"), names, hashes, content_hashes
        )

    if (
        not excluded_repository_path.is_file()
        or sha256_file(excluded_repository_path) != excluded_repository_sha256
    ):
        raise PreparationError("benchmark_exclusion_hash_mismatch")
    explicit = _read_document(excluded_repository_path)
    if not isinstance(explicit, list) or any(not isinstance(item, str) for item in explicit):
        raise PreparationError("benchmark_exclusion_manifest_invalid")
    names.update(item.strip() for item in explicit if item.strip())

    return names, hashes, content_hashes


def _normalize_repository(value: str) -> str:
    return value.strip()


def _repo_ids(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    primary = row.get("repository")
    if not isinstance(primary, str) or not primary.strip():
        raise ValueError("missing")
    # The frozen pool is normalized to `repository`; adapt it to the checked
    # Stage-1 helper's canonical Stack field without changing the identity.
    identity = repository_identity({"max_stars_repo_name": primary})
    raw_aliases = row.get("repository_aliases", [])
    if raw_aliases is None:
        raw_aliases = []
    if isinstance(raw_aliases, str):
        raw_aliases = [raw_aliases]
    if not isinstance(raw_aliases, list) or any(not isinstance(v, str) for v in raw_aliases):
        raise ValueError("invalid_aliases")
    aliases = tuple(
        sorted({identity, *(_normalize_repository(v) for v in raw_aliases if v.strip())})
    )
    return identity, aliases


def _license_list(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("licenses")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return ()
    return tuple(
        sorted({value.strip() for value in values if isinstance(value, str) and value.strip()})
    )


def _bucket_split(bucket: int, development_range: tuple[int, int]) -> str:
    legacy = research_split_for_bucket(bucket)
    if legacy in {"excluded_stage1_validation", "development", "test"}:
        return "reserved"
    if development_range[0] <= bucket < development_range[1]:
        return "development"
    return "train"


def _verify_pool(
    source_pool: Path,
    languages: tuple[str, ...],
    source_manifest: dict[str, Any],
) -> dict[str, tuple[Path, Path]]:
    files: dict[str, tuple[Path, Path]] = {}
    expected_by_language = source_manifest.get("source_pool")
    if not isinstance(expected_by_language, dict):
        raise PreparationError("source_manifest_missing")
    for language in languages:
        expected = expected_by_language.get(language)
        data_path = source_pool / f"{language}.jsonl"
        sidecar_path = source_pool / f"{language}.json"
        if not isinstance(expected, dict) or not data_path.is_file() or not sidecar_path.is_file():
            raise PreparationError("pinned_source_file_missing")
        if (
            data_path.stat().st_size != expected.get("bytes")
            or sha256_file(data_path) != expected.get("sha256")
            or sha256_file(sidecar_path) != expected.get("sidecar_sha256")
        ):
            raise PreparationError("pinned_source_hash_mismatch")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            raise PreparationError("source_sidecar_invalid") from None
        if (
            sidecar.get("source_revision") != DATASET_REVISION
            or sidecar.get("frozen_before_training") is not True
            or sidecar.get("sha256") != expected.get("sha256")
        ):
            raise PreparationError("source_sidecar_identity_mismatch")
        files[language] = (data_path, sidecar_path)
    return files


def _read_source_line(handle, candidate: Candidate) -> str:
    try:
        handle.seek(candidate.offset)
        row = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise PreparationError("pinned_source_changed_during_read") from None
    content = row.get("content") if isinstance(row, dict) else None
    if not isinstance(content, str) or _sha256_text(content) != candidate.content_sha256:
        raise PreparationError("pinned_source_changed_during_read")
    return content


def _manifest_record(
    candidate: Candidate,
    tokenizer_tokens: int,
    blocks_added: int,
    split: str,
) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "language": candidate.language,
        "split": split,
        "repository": candidate.repository,
        "repository_identity_sha256": _sha256_text(candidate.repository),
        "repository_alias_sha256": sorted(_sha256_text(alias) for alias in candidate.aliases),
        "repo_bucket": RepoSplit().bucket(candidate.repository),
        "path": candidate.path,
        "content_sha256": candidate.content_sha256,
        "licenses": list(candidate.licenses),
        "tokenizer_tokens": tokenizer_tokens,
        "blocks_added": blocks_added,
    }


def _hashable_external_repo(name: str) -> bool:
    return bool(name and name.strip())


def prepare_corpus(
    *,
    source_pool: Path,
    output_dir: Path,
    plan: dict[str, Any],
    configuration: dict[str, Any],
    tokenizer,
    reserved_repositories: set[str] | None = None,
    reserved_repository_hashes: set[str] | None = None,
    reserved_content_hashes: set[str] | None = None,
    benchmark_reserved_repositories: set[str] | None = None,
    benchmark_exclusion_path: Path | None = None,
    benchmark_exclusion_sha256: str | None = None,
    maximum_output_bytes: int | None = None,
) -> dict[str, Any]:
    data_config = configuration["data"]
    weights = {language: float(weight) for language, weight in data_config["languages"].items()}
    languages = tuple(language for language in LANGUAGE_ORDER if language in weights)
    if set(weights) != set(LANGUAGE_ORDER) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise PreparationError("language_plan_mismatch")
    block_size = int(data_config["sequence_length"])
    if block_size != 1024 or tokenizer.eos_token_id is None:
        raise PreparationError("sequence_or_eos_mismatch")
    development_range = tuple(data_config["new_q25_development_buckets"])
    legacy_reserved_range = tuple(data_config["original_reserved_buckets"])
    train_range = tuple(data_config["train_buckets"])
    if (
        legacy_reserved_range != (0, 30)
        or train_range != (30, 970)
        or development_range != (970, 1000)
    ):
        raise PreparationError("repository_split_plan_mismatch")
    if data_config.get("test_access") is not False:
        raise PreparationError("sealed_test_policy_mismatch")
    expected_exclusion_path = data_config.get("excluded_repositories_path")
    expected_exclusion_sha256 = data_config.get("excluded_repositories_sha256")
    if (
        benchmark_exclusion_path is None
        or benchmark_exclusion_sha256 != expected_exclusion_sha256
        or not benchmark_exclusion_path.is_file()
        or sha256_file(benchmark_exclusion_path) != expected_exclusion_sha256
        or not isinstance(expected_exclusion_path, str)
        or not isinstance(expected_exclusion_sha256, str)
    ):
        raise PreparationError("benchmark_exclusion_plan_mismatch")
    if benchmark_exclusion_path.name != Path(expected_exclusion_path).name:
        raise PreparationError("benchmark_exclusion_path_mismatch")
    source_files = _verify_pool(source_pool, languages, plan)
    split = RepoSplit(bucket_count=1000)
    source_filter = SourceFilter()
    allowed_licenses = set(data_config["allowed_licenses"])
    reserved_names = {
        _normalize_repository(name).casefold() for name in (reserved_repositories or set())
    }
    reserved_hashes = {value.lower() for value in (reserved_repository_hashes or set())}
    historical_content_hashes = {value.lower() for value in (reserved_content_hashes or set())}
    benchmark_names = {
        _normalize_repository(name).casefold()
        for name in (benchmark_reserved_repositories or set())
    }
    explicit_reserved_roots: set[str] = set()
    invalid_license_roots: set[str] = set()
    repo_buckets: dict[str, int] = {}
    union = UnionFind()
    row_refs: list[tuple[str, tuple[str, ...], str]] = []
    candidates: list[Candidate] = []
    rejected: Counter[str] = Counter()
    license_rows: Counter[str] = Counter()
    raw_rows: Counter[str] = Counter()
    benchmark_overlap_rows: Counter[str] = Counter()

    for language in languages:
        data_path, _ = source_files[language]
        try:
            handle = data_path.open("rb")
        except OSError:
            raise PreparationError("pinned_source_unreadable") from None
        with handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                raw_rows[language] += 1
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    raise PreparationError("pinned_source_jsonl_invalid") from None
                if not isinstance(row, dict) or row.get("language") != language:
                    rejected["invalid_record_language"] += 1
                    continue
                content = row.get("content")
                if not isinstance(content, str):
                    rejected["missing_content"] += 1
                    continue
                actual_content_hash = _sha256_text(content)
                claimed_content_hash = row.get("content_sha256")
                if claimed_content_hash != actual_content_hash:
                    raise PreparationError("source_content_hash_mismatch")
                try:
                    repository, aliases = _repo_ids(row)
                except ValueError:
                    rejected["invalid_repository_identity"] += 1
                    row_refs.append((actual_content_hash, (), language))
                    continue
                for alias in aliases:
                    union.add(alias)
                    repo_buckets[alias] = split.bucket(alias)
                for alias in aliases[1:]:
                    union.union(aliases[0], alias)
                row_refs.append((actual_content_hash, aliases, language))
                claimed_bucket = row.get("bucket")
                primary_bucket = split.bucket(repository)
                if not isinstance(claimed_bucket, int) or claimed_bucket != primary_bucket:
                    raise PreparationError("source_repository_bucket_mismatch")

                licenses = _license_list(row)
                if not licenses or any(
                    license_id not in allowed_licenses for license_id in licenses
                ):
                    for alias in aliases:
                        invalid_license_roots.add(alias)
                else:
                    license_rows[language] += 1
                if any(alias.casefold() in reserved_names for alias in aliases) or any(
                    _sha256_text(alias).lower() in reserved_hashes for alias in aliases
                ):
                    explicit_reserved_roots.update(aliases)
                if any(alias.casefold() in benchmark_names for alias in aliases):
                    benchmark_overlap_rows[language] += 1
                    explicit_reserved_roots.update(aliases)
                path_value = row.get("path")
                path = path_value if isinstance(path_value, str) else ""
                checked = source_filter.check(content, path)
                if not checked.accepted:
                    rejected[f"source_filter:{checked.reason}"] += 1
                    continue
                if not licenses or any(
                    license_id not in allowed_licenses for license_id in licenses
                ):
                    rejected["license_not_allowlisted"] += 1
                    continue
                candidates.append(
                    Candidate(
                        language=language,
                        source_file=data_path,
                        offset=offset,
                        repository=repository,
                        aliases=aliases,
                        path=path,
                        content_sha256=actual_content_hash,
                        licenses=licenses,
                    )
                )

    def root_for_any(aliases: tuple[str, ...]) -> str:
        return union.find(aliases[0]) if aliases else ""

    component_has_reserved: set[str] = {root_for_any((name,)) for name in explicit_reserved_roots}
    component_has_bad_license: set[str] = {root_for_any((name,)) for name in invalid_license_roots}
    component_split: dict[str, str] = {}
    for repository in list(union.parent):
        root = union.find(repository)
        bucket = repo_buckets[repository]
        assignment = _bucket_split(bucket, development_range)
        previous = component_split.get(root)
        if previous == "reserved" or assignment == "reserved":
            component_split[root] = "reserved"
        elif previous == "development" or assignment == "development":
            component_split[root] = "development"
        else:
            component_split[root] = "train"
    component_has_reserved.update(
        root for root, assignment in component_split.items() if assignment == "reserved"
    )

    all_reserved_content_hashes = set(historical_content_hashes)
    development_content_hashes: set[str] = set()
    for content_hash, aliases, _language in row_refs:
        if not aliases:
            all_reserved_content_hashes.add(content_hash)
            continue
        root = root_for_any(aliases)
        if root in component_has_reserved:
            all_reserved_content_hashes.add(content_hash)
        elif component_split.get(root) == "development":
            development_content_hashes.add(content_hash)

    per_language: dict[str, dict[str, list[Candidate]]] = {
        "train": {language: [] for language in languages},
        "development": {language: [] for language in languages},
    }
    seed = int(configuration["training"]["seed"])
    for candidate in candidates:
        root = root_for_any(candidate.aliases)
        if root in component_has_reserved:
            rejected["reserved_repository"] += 1
            continue
        if root in component_has_bad_license:
            rejected["repository_license_scope_invalid"] += 1
            continue
        if candidate.content_sha256 in all_reserved_content_hashes:
            rejected["reserved_content_hash"] += 1
            continue
        assignment = component_split.get(root, "reserved")
        if assignment == "development":
            split_name = "development"
        elif assignment == "train":
            if candidate.content_sha256 in development_content_hashes:
                rejected["development_content_duplicate"] += 1
                continue
            split_name = "train"
        else:
            rejected["reserved_repository"] += 1
            continue
        per_language[split_name][candidate.language].append(candidate)

    for split_name in per_language:
        for _language, values in per_language[split_name].items():
            values.sort(
                key=lambda item: (
                    hashlib.sha256(
                        f"{seed}\0{split_name}\0{item.content_sha256}\0{item.repository}".encode()
                    ).digest(),
                    item.source_file.name,
                    item.offset,
                )
            )

    requested_train = int(data_config["train_tokens"])
    requested_development = int(data_config["development_tokens"])
    train_blocks_requested = requested_train // block_size
    development_blocks_requested = requested_development // block_size
    train_targets = allocate_blocks(weights, train_blocks_requested)
    development_targets = allocate_blocks(weights, development_blocks_requested)

    handles: dict[Path, Any] = {}
    training_input_budget = int(configuration["budget"]["maximum_additional_training_input_tokens"])
    training_input_tokens = 0
    try:

        def build_split(split_name: str, targets: dict[str, int]):
            nonlocal training_input_tokens
            result: dict[str, list[np.ndarray]] = {}
            manifests: list[dict[str, Any]] = []
            actual_by_language: dict[str, int] = {}
            license_counts: dict[str, Counter[str]] = {}
            seen_content: set[str] = set()
            stats_by_language: dict[str, dict[str, int]] = {}
            for language in languages:
                packer = BlockPacker(block_size, tokenizer.eos_token_id)
                language_blocks: list[np.ndarray] = []
                language_manifest: list[dict[str, Any]] = []
                rejected_documents = 0
                for candidate in per_language[split_name][language]:
                    if len(language_blocks) >= targets[language]:
                        break
                    if candidate.content_sha256 in seen_content:
                        rejected["duplicate_content"] += 1
                        continue
                    seen_content.add(candidate.content_sha256)
                    handle = handles.get(candidate.source_file)
                    if handle is None:
                        try:
                            handle = candidate.source_file.open("rb")
                        except OSError:
                            raise PreparationError("pinned_source_unreadable") from None
                        handles[candidate.source_file] = handle
                    content = _read_source_line(handle, candidate)
                    try:
                        token_ids = tokenizer.encode(content, add_special_tokens=False)
                    except Exception:
                        raise PreparationError("tokenization_failed") from None
                    if not token_ids:
                        rejected["empty_tokenization"] += 1
                        rejected_documents += 1
                        continue
                    additional_input = len(token_ids) + int(bool(packer.documents))
                    if (
                        split_name == "train"
                        and training_input_tokens + additional_input > training_input_budget
                    ):
                        rejected["training_input_budget"] += 1
                        continue
                    if split_name == "train":
                        training_input_tokens += additional_input
                    blocks = packer.add_document(token_ids)
                    remaining = targets[language] - len(language_blocks)
                    stored = blocks[:remaining]
                    language_blocks.extend(np.asarray(block, dtype=np.uint32) for block in stored)
                    language_manifest.append(
                        _manifest_record(candidate, len(token_ids), len(stored), split_name)
                    )
                    for license_id in candidate.licenses:
                        license_counts.setdefault(language, Counter())[license_id] += 1
                    if len(language_blocks) >= targets[language]:
                        break
                partial_remainder = packer.discard_remainder()
                result[language] = language_blocks
                actual_by_language[language] = len(language_blocks)
                manifests.extend(language_manifest)
                stats_by_language[language] = {
                    "candidate_documents": len(per_language[split_name][language]),
                    "manifest_documents": len(language_manifest),
                    "input_tokens": packer.source_tokens,
                    "boundary_tokens": packer.boundary_tokens,
                    "discarded_remainder_tokens": partial_remainder,
                    "discarded_quota_tokens": max(
                        0, packer.emitted_tokens - len(language_blocks) * block_size
                    ),
                    "blocks": len(language_blocks),
                    "rejected_empty_tokenization": rejected_documents,
                }
            return result, manifests, actual_by_language, license_counts, stats_by_language

        dev_result, dev_manifest, dev_actual, dev_license_counts, dev_stats = build_split(
            "development", development_targets
        )
        train_result, train_manifest, train_actual, train_license_counts, train_stats = build_split(
            "train", train_targets
        )
    finally:
        for handle in handles.values():
            handle.close()

    train_actual_blocks = sum(train_actual.values())
    development_actual_blocks = sum(dev_actual.values())
    benchmark_hashes = {_sha256_text(name) for name in benchmark_names}
    selected_manifest = train_manifest + dev_manifest
    selected_benchmark_overlap = sum(
        1
        for record in selected_manifest
        if record["repository_identity_sha256"] in benchmark_hashes
        or benchmark_hashes.intersection(record["repository_alias_sha256"])
    )
    if selected_benchmark_overlap:
        raise PreparationError("benchmark_repository_selected")
    if train_actual_blocks == 0:
        raise PreparationError("training_corpus_empty")
    if development_actual_blocks == 0:
        raise PreparationError("development_corpus_empty")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise PreparationError("output_path_already_exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir.parent).free
    min_free = int(configuration["budget"].get("minimum_free_bytes", 0))
    if free_bytes < min_free:
        raise PreparationError("insufficient_free_storage")
    cap = int(
        maximum_output_bytes
        if maximum_output_bytes is not None
        else configuration["budget"]["new_artifact_bytes_cap"]
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        language_order = list(languages)
        train_schedule = [
            language for language in language_order for _ in range(train_actual[language])
        ]
        random.Random(seed).shuffle(train_schedule)
        train_array = np.lib.format.open_memmap(
            stage / "train_blocks.npy",
            mode="w+",
            dtype=np.uint32,
            shape=(train_actual_blocks, block_size),
        )
        train_language_ids = np.lib.format.open_memmap(
            stage / "train_languages.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(train_actual_blocks,),
        )
        positions: Counter[str] = Counter()
        for index, language in enumerate(train_schedule):
            train_array[index] = train_result[language][positions[language]]
            train_language_ids[index] = language_order.index(language)
            positions[language] += 1
        train_array.flush()
        train_language_ids.flush()

        development_schedule = [
            language for language in language_order for _ in range(dev_actual[language])
        ]
        random.Random(seed + 1).shuffle(development_schedule)
        development_array = np.lib.format.open_memmap(
            stage / "development_blocks.npy",
            mode="w+",
            dtype=np.uint32,
            shape=(development_actual_blocks, block_size),
        )
        positions = Counter()
        for index, language in enumerate(development_schedule):
            development_array[index] = dev_result[language][positions[language]]
            positions[language] += 1
        development_array.flush()

        for name, records in (
            ("train_manifest.jsonl", train_manifest),
            ("development_manifest.jsonl", dev_manifest),
        ):
            try:
                with (stage / name).open("w", encoding="utf-8") as manifest_handle:
                    for record in records:
                        manifest_handle.write(json.dumps(record, sort_keys=True) + "\n")
            except OSError:
                raise PreparationError("manifest_write_failed") from None

        output_files: dict[str, dict[str, Any]] = {}
        for output_path in sorted(stage.iterdir()):
            if output_path.is_file():
                output_files[output_path.name] = {
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                }
        output_bytes = sum(item["bytes"] for item in output_files.values())
        if output_bytes > cap:
            raise PreparationError("new_artifact_cap_exceeded")

        source_file_metadata = {}
        for language, (source_path, sidecar_path) in source_files.items():
            source_hash = sha256_file(source_path)
            sidecar_hash = sha256_file(sidecar_path)
            expected_source = plan.get("source_pool", {}).get(language, {})
            if source_hash != expected_source.get("sha256") or sidecar_hash != expected_source.get(
                "sidecar_sha256"
            ):
                raise PreparationError("pinned_source_changed_during_run")
            source_file_metadata[language] = {
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": source_hash,
                "sidecar_sha256": sidecar_hash,
            }
        metadata = {
            "schema_version": 1,
            "campaign": "q25_code_cpt_r2",
            "status": (
                "PASS"
                if train_actual_blocks == train_blocks_requested
                and development_actual_blocks == development_blocks_requested
                else "PARTIAL"
            ),
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_pool": str(source_pool),
            "source_files": source_file_metadata,
            "plan_sha256": EXPECTED_PLAN_SHA256,
            "config_sha256": plan.get("config_sha256"),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_sha256": configuration["model"]["tokenizer_sha256"],
            "sequence_length": block_size,
            "eos_token_id": int(tokenizer.eos_token_id),
            "language_order": language_order,
            "language_weights": weights,
            "requested_train_tokens": requested_train,
            "target_train_tokens_packed": train_blocks_requested * block_size,
            "actual_train_tokens": train_actual_blocks * block_size,
            "maximum_additional_training_input_tokens": training_input_budget,
            "actual_training_input_tokens": training_input_tokens,
            "train_blocks_by_language": train_actual,
            "development_requested_tokens": requested_development,
            "development_actual_tokens": development_actual_blocks * block_size,
            "development_complete": development_actual_blocks == development_blocks_requested,
            "development_blocks_by_language": dev_actual,
            "split": {
                "hash": "sha256(repository_identity)[:8] big-endian modulo 1000",
                "historical_reserved_buckets": [0, 30],
                "train_buckets": [30, 970],
                "new_development_buckets": [970, 1000],
                "test_access": False,
                "alias_policy": (
                    "any reserved alias excludes; any development alias assigns development"
                ),
                "benchmark_reserved_repository_count": len(benchmark_names),
                "benchmark_reserved_hash_count": len(reserved_hashes),
                "benchmark_exclusion_entry_count": len(benchmark_names),
                "benchmark_overlap_source_rows_by_language": dict(benchmark_overlap_rows),
                "benchmark_overlap_selected_rows_after_exclusion": selected_benchmark_overlap,
                "benchmark_exclusion_path": str(benchmark_exclusion_path)
                if benchmark_exclusion_path
                else None,
                "benchmark_exclusion_sha256": benchmark_exclusion_sha256,
            },
            "licenses": {
                "allowed_spdx": sorted(allowed_licenses),
                "train_counts_by_language": {
                    lang: dict(sorted(counts.items()))
                    for lang, counts in train_license_counts.items()
                },
                "development_counts_by_language": {
                    lang: dict(sorted(counts.items()))
                    for lang, counts in dev_license_counts.items()
                },
                "repository_level_all_claims_must_be_allowlisted": True,
            },
            "rejections": dict(sorted(rejected.items())),
            "raw_rows_by_language": dict(raw_rows),
            "allowlisted_rows_by_language": dict(license_rows),
            "train_feeder": train_stats,
            "development_feeder": dev_stats,
            "output_files": output_files,
            "data_files_bytes": output_bytes,
            "output_bytes": output_bytes,
            "output_byte_cap": cap,
            "maximum_discarded_replay_input_tokens": int(
                configuration["budget"]["maximum_discarded_replay_input_tokens"]
            ),
            "corpus_sha256": hashlib.sha256(
                "".join(
                    f"{name}:{item['bytes']}:{item['sha256']}\n"
                    for name, item in sorted(output_files.items())
                ).encode("utf-8")
            ).hexdigest(),
        }
        metadata_path = stage / "corpus_metadata.json"
        try:
            for _ in range(3):
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                final_size = sum(path.stat().st_size for path in stage.iterdir() if path.is_file())
                if metadata.get("output_bytes") == final_size:
                    break
                metadata["output_bytes"] = final_size
        except OSError:
            raise PreparationError("metadata_write_failed") from None
        final_size = sum(path.stat().st_size for path in stage.iterdir() if path.is_file())
        if final_size > cap:
            raise PreparationError("new_artifact_cap_exceeded")
        os.replace(stage, output_dir)
        return metadata
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _default_tokenizer_path() -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
        / MODEL_REVISION
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=REPO_ROOT / "reports/research/q25_code_cpt_r2/plan.json"
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/research/q25_code_cpt_r2.yaml"
    )
    parser.add_argument("--source-pool", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=_default_tokenizer_path())
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")
    os.environ["RAYON_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if args.cpu_threads == 1 else "true"
    try:
        plan, configuration = load_plan(args.plan, args.config)
        pool_path = args.source_pool or Path(configuration["data"]["reuse_pool"])
        data_config = configuration["data"]
        exclusion_value = data_config.get("excluded_repositories_path")
        exclusion_hash = data_config.get("excluded_repositories_sha256")
        if not isinstance(exclusion_value, str) or not isinstance(exclusion_hash, str):
            raise PreparationError("benchmark_exclusion_plan_missing")
        exclusion_path = Path(exclusion_value)
        if not exclusion_path.is_absolute():
            exclusion_path = REPO_ROOT / exclusion_path
        if sha256_file(exclusion_path) != exclusion_hash:
            raise PreparationError("benchmark_exclusion_hash_mismatch")
        exclusion_data = _read_document(exclusion_path)
        if not isinstance(exclusion_data, list) or any(
            not isinstance(item, str) for item in exclusion_data
        ):
            raise PreparationError("benchmark_exclusion_manifest_invalid")
        benchmark_names = {item.strip() for item in exclusion_data if item.strip()}
        reserved_names, reserved_hashes, reserved_content_hashes = collect_reserved_repositories(
            REPO_ROOT, DEFAULT_PREVIOUS_ROOT, exclusion_path, exclusion_hash
        )
        tokenizer = load_local_tokenizer(args.tokenizer_path, plan)
        metadata = prepare_corpus(
            source_pool=pool_path,
            output_dir=args.output_dir,
            plan=plan,
            configuration=configuration,
            tokenizer=tokenizer,
            reserved_repositories=reserved_names,
            reserved_repository_hashes=reserved_hashes,
            reserved_content_hashes=reserved_content_hashes,
            benchmark_reserved_repositories=benchmark_names,
            benchmark_exclusion_path=exclusion_path,
            benchmark_exclusion_sha256=exclusion_hash,
        )
        print(
            json.dumps(
                {
                    "status": metadata["status"],
                    "output_dir": str(args.output_dir),
                    "train_tokens": metadata["actual_train_tokens"],
                    "development_tokens": metadata["development_actual_tokens"],
                    "corpus_sha256": metadata["corpus_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except PreparationError as exc:
        print(json.dumps({"status": "FAIL", "reason": exc.reason}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "FAIL", "reason": "unexpected_internal_error"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
