"""Prepare a pinned, local-only Qwen2.5 PSM line-completion corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prepare_q25_code_cpt import _license_list, _repo_ids

from tinycomplete.code_cpt.data import RepoSplit
from tinycomplete.data.fim import format_psm
from tinycomplete.data.schema import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "reports/research/q25_code_cpt_r2/fim_preparation_plan.json"
EXPECTED_PLAN_SHA256 = "460770a6bab77eeecd7b31eb2c8da278c5c29cea98cc872d8fbd36b22a6dc6de"
PARENT_CPT_PLAN_SHA256 = "1aaff2ee4f4e0adeeafb87e616f99e657f7ff0be4fa438f40ecbb1150aeed1f9"
EXCLUSION_PATH = REPO_ROOT / "reports/research/q25_code_cpt_r2/excluded_benchmark_repositories.json"
TOKENIZER_ID = "Qwen/Qwen2.5-Coder-0.5B"
TOKENIZER_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
DATASET_ID = "bigcode/the-stack-dedup"
DEFAULT_TOKENIZER_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    / TOKENIZER_REVISION
)
DEFAULT_OUTPUT_DIR = Path("/mnt/ssd/tabcomplete-q25-code-cpt-r2/fim/corpus-r3")
LANGUAGES = ("python", "rust", "typescript", "go")
MARKERS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)
MARKER_NAMES = ("fim_prefix", "fim_suffix", "fim_middle")
HEX_SHA256_LENGTH = 64


class PreparationError(RuntimeError):
    """A sanitized preparation failure suitable for a stable CLI message."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _RejectedState(Exception):
    """A deterministic candidate rejection with an audit reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Document:
    split: str
    language: str
    content: str
    content_sha256: str
    repository_identity_sha256: str
    repository_alias_sha256: tuple[str, ...]
    path: str
    licenses: tuple[str, ...]
    dataset_id: str
    dataset_revision: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
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


def _read_jsonl(path: Path, reason: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PreparationError(reason)
                rows.append(value)
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise PreparationError(reason) from None
    return rows


def _require_sha(value: Any, reason: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != HEX_SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise PreparationError(reason)
    return value.lower()


def _manifest_docs(
    rows: list[dict[str, Any]], split: str, allowed_licenses: set[str], revision: str
) -> list[dict[str, Any]]:
    expected_dataset = DATASET_ID
    docs: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    for row in rows:
        if (
            row.get("split") != split
            or row.get("dataset_id") != expected_dataset
            or row.get("dataset_revision") != revision
            or row.get("language") not in LANGUAGES
        ):
            raise PreparationError("feeder_manifest_identity_mismatch")
        content_hash = _require_sha(row.get("content_sha256"), "feeder_content_hash_invalid")
        repository_hash = _require_sha(
            row.get("repository_identity_sha256"), "feeder_repository_hash_invalid"
        )
        raw_aliases = row.get("repository_alias_sha256")
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise PreparationError("feeder_repository_aliases_invalid")
        aliases = tuple(
            sorted({_require_sha(item, "feeder_repository_hash_invalid") for item in raw_aliases})
        )
        if repository_hash not in aliases:
            raise PreparationError("feeder_primary_repository_missing_from_aliases")
        licenses = _license_list(row)
        if not licenses or not set(licenses).issubset(allowed_licenses):
            raise PreparationError("feeder_license_not_allowlisted")
        path = row.get("path")
        bucket = row.get("repo_bucket")
        if (
            not isinstance(path, str)
            or isinstance(bucket, bool)
            or not isinstance(bucket, int)
            or not 0 <= bucket < 1000
        ):
            raise PreparationError("feeder_provenance_invalid")
        if content_hash in seen_content:
            raise PreparationError("duplicate_content_in_feeder_manifest")
        seen_content.add(content_hash)
        docs.append(
            {
                "split": split,
                "language": row["language"],
                "content_sha256": content_hash,
                "repository_identity_sha256": repository_hash,
                "repository_alias_sha256": aliases,
                "path": path,
                "licenses": licenses,
                "repo_bucket": bucket,
                "dataset_id": expected_dataset,
                "dataset_revision": revision,
            }
        )
    return docs


def _load_pinned_inputs(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Document], list[Document]]:
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise PreparationError("preparation_plan_hash_mismatch")
    plan = _read_json(plan_path, "preparation_plan_invalid")
    if plan.get("schema") != "q25-fim-cpu-preparation-plan-v1" or plan.get("plan_revision") != 3:
        raise PreparationError("preparation_plan_identity_mismatch")
    if plan.get("gpu_allocation_authorized_by_this_plan") is not False:
        raise PreparationError("preparation_plan_gpu_policy_mismatch")
    data = plan.get("data")
    sources = plan.get("sources")
    budget = plan.get("budget")
    model = plan.get("model")
    if not all(isinstance(value, dict) for value in (data, sources, budget, model)):
        raise PreparationError("preparation_plan_invalid")
    if (
        model.get("id") != TOKENIZER_ID
        or model.get("revision") != TOKENIZER_REVISION
        or data.get("tokenizer_revision") != TOKENIZER_REVISION
        or model.get("tokenizer_sha256") != data.get("tokenizer_sha256")
        or sources.get("dataset") != DATASET_ID
    ):
        raise PreparationError("preparation_plan_model_or_source_mismatch")

    parent_cpt_plan = REPO_ROOT / "reports/research/q25_code_cpt_r2/plan.json"
    if sha256_file(parent_cpt_plan) != PARENT_CPT_PLAN_SHA256:
        raise PreparationError("parent_cpt_plan_hash_mismatch")
    parent_plan = _read_json(parent_cpt_plan, "parent_cpt_plan_invalid")
    allowed_licenses = set(
        parent_plan.get("configuration", {}).get("data", {}).get("allowed_licenses", [])
    )
    if not allowed_licenses:
        raise PreparationError("parent_cpt_license_policy_missing")

    exclusion_hash = _require_sha(
        sources.get("benchmark_exclusion_sha256"), "benchmark_exclusion_hash_invalid"
    )
    if sha256_file(EXCLUSION_PATH) != exclusion_hash:
        raise PreparationError("benchmark_exclusion_hash_mismatch")
    exclusion_value = _read_json(EXCLUSION_PATH, "benchmark_exclusion_manifest_invalid")
    if not isinstance(exclusion_value, list) or any(
        not isinstance(item, str) for item in exclusion_value
    ):
        raise PreparationError("benchmark_exclusion_manifest_invalid")
    excluded_names = {item.strip().casefold() for item in exclusion_value if item.strip()}

    pool_path = Path(sources.get("raw_pool", ""))
    if not pool_path.is_absolute() or not pool_path.is_dir():
        raise PreparationError("pinned_source_pool_missing")
    pool_hashes = sources.get("pool")
    if not isinstance(pool_hashes, dict) or set(pool_hashes) != set(LANGUAGES):
        raise PreparationError("pinned_source_pool_plan_invalid")
    for language in LANGUAGES:
        expected = pool_hashes[language]
        data_path = pool_path / f"{language}.jsonl"
        sidecar_path = pool_path / f"{language}.json"
        if (
            not isinstance(expected, dict)
            or not data_path.is_file()
            or not sidecar_path.is_file()
            or data_path.stat().st_size != expected.get("bytes")
            or sha256_file(data_path) != expected.get("sha256")
            or sha256_file(sidecar_path) != expected.get("sidecar_sha256")
        ):
            raise PreparationError("pinned_source_pool_hash_mismatch")
        sidecar = _read_json(sidecar_path, "pinned_source_sidecar_invalid")
        if (
            sidecar.get("source_revision") != sources.get("revision")
            or sidecar.get("frozen_before_training") is not True
            or sidecar.get("sha256") != expected.get("sha256")
        ):
            raise PreparationError("pinned_source_sidecar_identity_mismatch")

    manifests = sources.get("train_manifest"), sources.get("development_manifest")
    parsed_manifests: dict[str, list[dict[str, Any]]] = {}
    for split, source in zip(("train", "development"), manifests, strict=True):
        if not isinstance(source, dict):
            raise PreparationError("feeder_manifest_plan_invalid")
        manifest_path = Path(source.get("path", ""))
        expected_hash = _require_sha(source.get("sha256"), "feeder_manifest_hash_invalid")
        if not manifest_path.is_absolute() or sha256_file(manifest_path) != expected_hash:
            raise PreparationError("feeder_manifest_hash_mismatch")
        parsed_manifests[split] = _manifest_docs(
            _read_jsonl(manifest_path, "feeder_manifest_invalid"),
            split,
            allowed_licenses,
            str(sources.get("revision")),
        )

    train_docs = parsed_manifests["train"]
    development_docs = parsed_manifests["development"]
    train_content = {row["content_sha256"] for row in train_docs}
    development_content = {row["content_sha256"] for row in development_docs}
    if train_content & development_content:
        raise PreparationError("train_development_content_overlap")
    train_repositories = {alias for row in train_docs for alias in row["repository_alias_sha256"]}
    development_repositories = {
        alias for row in development_docs for alias in row["repository_alias_sha256"]
    }
    if train_repositories & development_repositories:
        raise PreparationError("train_development_repository_overlap")

    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _split, docs in parsed_manifests.items():
        for doc in docs:
            key = (
                doc["language"],
                doc["content_sha256"],
                doc["repository_identity_sha256"],
                doc["path"],
            )
            if key in lookup:
                raise PreparationError("duplicate_feeder_document_identity")
            lookup[key] = doc

    found: dict[tuple[str, str, str, str], Document] = {}
    repo_split = RepoSplit(bucket_count=1000)
    for language in LANGUAGES:
        pool_file = pool_path / f"{language}.jsonl"
        try:
            handle = pool_file.open("r", encoding="utf-8")
        except OSError:
            raise PreparationError("pinned_source_pool_unreadable") from None
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    raise PreparationError("pinned_source_pool_invalid") from None
                if not isinstance(raw, dict) or raw.get("language") != language:
                    continue
                content = raw.get("content")
                if not isinstance(content, str):
                    continue
                claimed_content = raw.get("content_sha256")
                if not isinstance(claimed_content, str):
                    continue
                repository, aliases = _repo_ids(raw)
                repository_hash = sha256_text(repository)
                raw_path = raw.get("path")
                path = raw_path if isinstance(raw_path, str) else ""
                key = (language, claimed_content.lower(), repository_hash, path)
                expected = lookup.get(key)
                if expected is None:
                    continue
                if sha256_text(content) != expected["content_sha256"]:
                    raise PreparationError("selected_source_content_hash_mismatch")
                alias_hashes = tuple(sorted(sha256_text(alias) for alias in aliases))
                licenses = _license_list(raw)
                claimed_bucket = raw.get("bucket")
                actual_bucket = repo_split.bucket(repository)
                if (
                    alias_hashes != expected["repository_alias_sha256"]
                    or licenses != expected["licenses"]
                    or actual_bucket != expected["repo_bucket"]
                    or (isinstance(claimed_bucket, int) and claimed_bucket != actual_bucket)
                ):
                    raise PreparationError("selected_source_provenance_mismatch")
                if any(alias.casefold() in excluded_names for alias in aliases):
                    raise PreparationError("benchmark_repository_in_feeder_manifest")
                if key in found:
                    raise PreparationError("duplicate_selected_source_record")
                source_path = expected.get("path")
                if not isinstance(source_path, str):
                    raise PreparationError("selected_source_path_invalid")
                found[key] = Document(
                    split=expected["split"],
                    language=language,
                    content=content,
                    content_sha256=expected["content_sha256"],
                    repository_identity_sha256=repository_hash,
                    repository_alias_sha256=alias_hashes,
                    path=source_path,
                    licenses=licenses,
                    dataset_id=DATASET_ID,
                    dataset_revision=str(sources["revision"]),
                )
    if set(found) != set(lookup):
        raise PreparationError("feeder_source_document_missing")
    train = [found[key] for key, row in lookup.items() if row["split"] == "train"]
    development = [found[key] for key, row in lookup.items() if row["split"] == "development"]
    # Preserve the active plan and source inputs for the reproducible metadata record.
    inputs = {
        "allowed_licenses": sorted(allowed_licenses),
        "benchmark_exclusion_sha256": exclusion_hash,
        "benchmark_exclusion_count": len(excluded_names),
        "parent_cpt_plan_sha256": PARENT_CPT_PLAN_SHA256,
        "pool_path": str(pool_path),
        "pool_hashes": pool_hashes,
        "train_manifest": manifests[0],
        "development_manifest": manifests[1],
    }
    return plan, inputs, train, development


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        raise PreparationError("tokenization_failed") from None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        raise PreparationError("tokenizer_decode_failed") from None
    if not isinstance(decoded, str):
        raise PreparationError("tokenizer_decode_failed")
    return decoded


def _offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except Exception:
        raise PreparationError("tokenizer_offsets_unavailable") from None
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    token_ids = [int(value) for value in ids]
    char_offsets = [(int(start), int(end)) for start, end in offsets]
    if len(token_ids) != len(char_offsets) or any(
        start < 0 or end < start or end > len(text) for start, end in char_offsets
    ):
        raise PreparationError("tokenizer_offsets_invalid")
    return token_ids, char_offsets


def _truncate_prefix(tokenizer: Any, context: str, limit: int) -> str:
    candidate = context
    for _ in range(len(context) + 1):
        token_ids, offsets = _offsets(tokenizer, candidate)
        if len(token_ids) <= limit:
            return candidate
        index = max(0, len(token_ids) - limit)
        cut = offsets[index][0] if offsets else 0
        if cut <= 0:
            cut = next((start for start, end in offsets if end > start and start > 0), 0)
        if cut <= 0:
            cut = 1
        candidate = candidate[cut:]
    raise PreparationError("prefix_context_crop_failed")


def _truncate_suffix(tokenizer: Any, context: str, limit: int) -> str:
    candidate = context
    for _ in range(len(context) + 1):
        token_ids, offsets = _offsets(tokenizer, candidate)
        if len(token_ids) <= limit:
            return candidate
        index = max(0, min(limit - 1, len(offsets) - 1))
        cut = offsets[index][1] if offsets else 0
        if cut >= len(candidate):
            prior = [end for start, end in offsets if end > start and end < len(candidate)]
            cut = prior[-1] if prior else len(candidate) - 1
        if cut <= 0:
            cut = max(0, len(candidate) - 1)
        candidate = candidate[:cut]
    raise PreparationError("suffix_context_crop_failed")


def _line_spans(source: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in source.splitlines(keepends=True):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    if offset < len(source):
        spans.append((offset, len(source)))
    return spans


def _line_body_end(line: str) -> int:
    if line.endswith("\r\n"):
        return len(line) - 2
    if line.endswith(("\n", "\r")):
        return len(line) - 1
    return len(line)


def _digest_order(seed: int, split: str, content_sha256: str, variant: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{split}\0{content_sha256}\0{variant}".encode()).digest()


def _mode_for_variant(content_sha256: str, variant: int) -> str:
    content_digest = hashlib.sha256(content_sha256.encode("ascii")).digest()
    mode_index = (content_digest[0] + variant) % 2
    return ("whole_logical_line", "remaining_logical_line_after_utf8_cursor")[mode_index]


def _chosen_line(
    source: str, content_sha256: str, variant: int, seed: int
) -> tuple[int, int] | None:
    spans = _line_spans(source)
    nonblank: list[tuple[int, int]] = []
    for start, end in spans:
        line = source[start:end]
        if line[: _line_body_end(line)].strip():
            nonblank.append((start, end))
    if not nonblank:
        return None
    digest = hashlib.sha256(f"{seed}\0{content_sha256}\0{variant}\0line".encode()).digest()
    return nonblank[int.from_bytes(digest[:8], "big") % len(nonblank)]


def _make_state(
    document: Document,
    variant: int,
    tokenizer: Any,
    *,
    seed: int,
    prefix_limit: int,
    suffix_limit: int,
    target_limit: int,
    total_limit: int,
    eos_token_id: int,
    marker_ids: tuple[int, int, int],
) -> dict[str, Any] | None:
    source = document.content
    span = _chosen_line(source, document.content_sha256, variant, seed)
    if span is None:
        return None
    line_start, line_end = span
    line = source[line_start:line_end]
    mode = _mode_for_variant(document.content_sha256, variant)
    if mode == "whole_logical_line":
        region_start_char, region_end_char = line_start, line_end
    else:
        body_end = _line_body_end(line)
        body = line[:body_end]
        if not body:
            return None
        cursor_digest = hashlib.sha256(
            f"{seed}\0{document.content_sha256}\0{variant}\0utf8-cursor".encode()
        ).digest()
        cursor_char = int.from_bytes(cursor_digest[:8], "big") % len(body)
        region_start_char = line_start + cursor_char
        region_end_char = line_end

    target = source[region_start_char:region_end_char]
    if not target or not target.strip():
        return None
    target_ids = _token_ids(tokenizer, target)
    if len(target_ids) + 1 > target_limit:
        return None
    if any(token_id in set(marker_ids) or token_id == eos_token_id for token_id in target_ids):
        return None

    prefix = _truncate_prefix(tokenizer, source[:region_start_char], prefix_limit)
    suffix = _truncate_suffix(tokenizer, source[region_end_char:], suffix_limit)
    prefix_ids = _token_ids(tokenizer, prefix)
    suffix_ids = _token_ids(tokenizer, suffix)
    if (
        len(prefix_ids) > prefix_limit
        or len(suffix_ids) > suffix_limit
        or any(token_id in set(marker_ids) or token_id == eos_token_id for token_id in prefix_ids)
        or any(token_id in set(marker_ids) or token_id == eos_token_id for token_id in suffix_ids)
    ):
        return None

    prompt = format_psm(prefix, suffix)
    prompt_ids = _token_ids(tokenizer, prompt)
    if (
        len(prompt_ids) + len(target_ids) + 1 > total_limit
        or prompt_ids.count(marker_ids[0]) != 1
        or prompt_ids.count(marker_ids[1]) != 1
        or prompt_ids.count(marker_ids[2]) != 1
    ):
        return None
    marker_positions = [prompt_ids.index(marker_id) for marker_id in marker_ids]
    if marker_positions != sorted(marker_positions):
        return None

    decoded_prompt = _decode_tokens(tokenizer, prompt_ids)
    decoded_target = _decode_tokens(tokenizer, target_ids)
    if decoded_prompt != prompt or decoded_target != target:
        raise _RejectedState("token_roundtrip_mismatch")

    region_start = len(source[:region_start_char].encode("utf-8"))
    region_end = len(source[:region_end_char].encode("utf-8"))
    if (
        source[:region_start_char] + target + source[region_end_char:] != source
        or len(source[:region_start_char].encode("utf-8")) != region_start
        or len(source[:region_end_char].encode("utf-8")) != region_end
    ):
        raise PreparationError("source_region_roundtrip_failed")
    input_ids = prompt_ids + target_ids + [eos_token_id]
    return {
        "split": document.split,
        "language": document.language,
        "mode": mode,
        "prompt_format": "psm",
        "variant": variant,
        "source_content_sha256": document.content_sha256,
        "repository_identity_sha256": document.repository_identity_sha256,
        "repository_alias_sha256": list(document.repository_alias_sha256),
        "source_path": document.path,
        "dataset_id": document.dataset_id,
        "dataset_revision": document.dataset_revision,
        "licenses": list(document.licenses),
        "region_start": region_start,
        "region_end": region_end,
        "prompt_sha256": sha256_text(prompt),
        "target_sha256": sha256_text(target),
        "input_ids": input_ids,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids) + 1,
        "total_tokens": len(input_ids),
        "_order": _digest_order(seed, document.split, document.content_sha256, variant),
        "_state_identity": (mode, region_start, region_end, sha256_text(target)),
    }


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


def _generate_states(
    documents: Iterable[Document],
    tokenizer: Any,
    *,
    seed: int,
    prefix_limit: int,
    suffix_limit: int,
    target_limit: int,
    total_limit: int,
    eos_token_id: int,
    marker_ids: tuple[int, int, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    states: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for document in documents:
        for variant in (0, 1):
            try:
                state = _make_state(
                    document,
                    variant,
                    tokenizer,
                    seed=seed,
                    prefix_limit=prefix_limit,
                    suffix_limit=suffix_limit,
                    target_limit=target_limit,
                    total_limit=total_limit,
                    eos_token_id=eos_token_id,
                    marker_ids=marker_ids,
                )
            except _RejectedState as exc:
                rejected[exc.reason] += 1
                continue
            if state is None:
                rejected["invalid_or_overlength_state"] += 1
                continue
            states.append(state)
    states.sort(key=lambda row: (row["_order"], row["source_content_sha256"], row["variant"]))
    distinct: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, int, int, str]]] = set()
    for row in states:
        key = (
            row["split"],
            row["source_content_sha256"],
            row["_state_identity"],
        )
        if key in seen:
            rejected["duplicate_variant_state"] += 1
            continue
        seen.add(key)
        distinct.append(row)
    return distinct, rejected


def _public_row(row: dict[str, Any], row_id: int) -> dict[str, Any]:
    fields = (
        "split",
        "language",
        "mode",
        "prompt_format",
        "variant",
        "source_content_sha256",
        "repository_identity_sha256",
        "repository_alias_sha256",
        "source_path",
        "dataset_id",
        "dataset_revision",
        "licenses",
        "region_start",
        "region_end",
        "prompt_sha256",
        "target_sha256",
        "input_ids",
        "prompt_tokens",
        "target_tokens",
        "total_tokens",
    )
    public = {field: row[field] for field in fields}
    public["id"] = row_id
    return public


def _select_states(
    train_states: list[dict[str, Any]],
    development_states: list[dict[str, Any]],
    weights: dict[str, float],
    train_cap: int,
    development_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    quotas = _largest_remainder(weights, train_cap)
    train_selected: list[dict[str, Any]] = []
    for language in LANGUAGES:
        candidates = [row for row in train_states if row["language"] == language]
        candidates.sort(
            key=lambda row: (row["_order"], row["source_content_sha256"], row["variant"])
        )
        train_selected.extend(candidates[: quotas[language]])
    train_selected.sort(
        key=lambda row: (row["_order"], row["source_content_sha256"], row["variant"])
    )
    development_states.sort(
        key=lambda row: (row["_order"], row["source_content_sha256"], row["variant"])
    )
    development_selected = development_states[:development_cap]
    return train_selected, development_selected, quotas


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[str, int]:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                )
                handle.write("\n")
    except OSError:
        raise PreparationError("corpus_write_failed") from None
    return sha256_file(path), path.stat().st_size


def prepare_examples(
    train_documents: list[Document],
    development_documents: list[Document],
    tokenizer: Any,
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = plan["data"]
    weights = {language: float(data["language_weights"][language]) for language in LANGUAGES}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise PreparationError("language_weight_plan_invalid")
    marker_map: dict[str, int] = {}
    for name, marker in zip(MARKER_NAMES, MARKERS, strict=True):
        ids = _token_ids(tokenizer, marker)
        if len(ids) != 1:
            raise PreparationError("fim_marker_not_single_token")
        marker_map[name] = ids[0]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise PreparationError("tokenizer_eos_missing")
    marker_ids: tuple[int, int, int] = (
        marker_map[MARKER_NAMES[0]],
        marker_map[MARKER_NAMES[1]],
        marker_map[MARKER_NAMES[2]],
    )
    if eos_token_id in marker_ids or len(set(marker_ids)) != 3:
        raise PreparationError("tokenizer_special_token_identity_invalid")

    train_states, train_rejected = _generate_states(
        train_documents,
        tokenizer,
        seed=int(data["seed"]),
        prefix_limit=int(data["prefix_tokens_max"]),
        suffix_limit=int(data["suffix_tokens_max"]),
        target_limit=int(data["target_tokens_including_eos_max"]),
        total_limit=int(data["max_total_tokens"]),
        eos_token_id=eos_token_id,
        marker_ids=marker_ids,
    )
    development_states, development_rejected = _generate_states(
        development_documents,
        tokenizer,
        seed=int(data["seed"]),
        prefix_limit=int(data["prefix_tokens_max"]),
        suffix_limit=int(data["suffix_tokens_max"]),
        target_limit=int(data["target_tokens_including_eos_max"]),
        total_limit=int(data["max_total_tokens"]),
        eos_token_id=eos_token_id,
        marker_ids=marker_ids,
    )
    train_selected, development_selected, quotas = _select_states(
        train_states,
        development_states,
        weights,
        int(data["train_states_max"]),
        int(data["development_states_max"]),
    )
    train_rows = [_public_row(row, row_id) for row_id, row in enumerate(train_selected)]
    development_rows = [
        _public_row(row, len(train_rows) + row_id)
        for row_id, row in enumerate(development_selected)
    ]
    audit = {
        "requested_train_states": int(data["train_states_max"]),
        "actual_train_states": len(train_rows),
        "requested_development_states": int(data["development_states_max"]),
        "actual_development_states": len(development_rows),
        "training_quotas": quotas,
        "training_counts_by_language": dict(Counter(row["language"] for row in train_rows)),
        "development_counts_by_language": dict(
            Counter(row["language"] for row in development_rows)
        ),
        "train_rejected_states": dict(sorted(train_rejected.items())),
        "development_rejected_states": dict(sorted(development_rejected.items())),
        "train_manifest_documents": len(train_documents),
        "development_manifest_documents": len(development_documents),
        "training_variant_limit_per_document": 2,
        "development_variant_limit_per_document": 2,
    }
    return (
        train_rows,
        development_rows,
        {"audit": audit, "marker_ids": marker_map, "eos_token_id": eos_token_id},
    )


def _load_local_tokenizer(tokenizer_path: Path, expected_sha256: str):
    if tokenizer_path.name != TOKENIZER_REVISION:
        raise PreparationError("local_tokenizer_revision_mismatch")
    tokenizer_json = tokenizer_path / "tokenizer.json"
    if not tokenizer_json.is_file() or sha256_file(tokenizer_json) != expected_sha256:
        raise PreparationError("local_tokenizer_hash_mismatch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    if not getattr(tokenizer, "is_fast", False):
        raise PreparationError("local_fast_tokenizer_required")
    if tokenizer.eos_token_id is None:
        raise PreparationError("tokenizer_eos_missing")
    return tokenizer


def _validate_serialized_rows(
    rows: list[dict[str, Any]], split: str, max_total: int, eos_token_id: int
) -> None:
    seen_ids: set[int] = set()
    seen_variants: Counter[str] = Counter()
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id in seen_ids:
            raise PreparationError("serialized_row_id_invalid")
        seen_ids.add(row_id)
        if row.get("split") != split or row.get("prompt_format") != "psm":
            raise PreparationError("serialized_row_split_or_format_invalid")
        prompt_tokens = row.get("prompt_tokens")
        target_tokens = row.get("target_tokens")
        total_tokens = row.get("total_tokens")
        ids = row.get("input_ids")
        if (
            not isinstance(ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or prompt_tokens + target_tokens != total_tokens
            or total_tokens != len(ids)
            or total_tokens > max_total
            or target_tokens < 1
            or not ids
            or ids[-1] != eos_token_id
        ):
            raise PreparationError("serialized_row_token_counts_invalid")
        seen_variants[row["source_content_sha256"]] += 1
        if seen_variants[row["source_content_sha256"]] > 2:
            raise PreparationError("serialized_document_variant_limit_exceeded")


def write_corpus(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    inputs: dict[str, Any],
    audit: dict[str, Any],
    tokenizer_sha256: str,
    tokenizer_revision: str,
    marker_ids: dict[str, int],
    eos_token_id: int,
) -> dict[str, Any]:
    data = plan["data"]
    _validate_serialized_rows(train_rows, "train", int(data["max_total_tokens"]), eos_token_id)
    _validate_serialized_rows(
        development_rows, "development", int(data["max_total_tokens"]), eos_token_id
    )
    if not train_rows or not development_rows:
        raise PreparationError("empty_prepared_split")
    if output_dir.exists():
        raise PreparationError("output_directory_already_exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < int(plan["budget"]["minimum_free_bytes"]):
        raise PreparationError("minimum_free_space_not_available")
    stage = Path(tempfile.mkdtemp(prefix=".q25-fim-corpus-", dir=output_dir.parent))
    try:
        train_hash, train_bytes = _write_jsonl(stage / "train.jsonl", train_rows)
        development_hash, development_bytes = _write_jsonl(
            stage / "development.jsonl", development_rows
        )
        file_records = {
            "train.jsonl": {"sha256": train_hash, "bytes": train_bytes},
            "development.jsonl": {"sha256": development_hash, "bytes": development_bytes},
        }
        split_records = {
            "train": {
                "file": "train.jsonl",
                "sha256": train_hash,
                "row_count": len(train_rows),
                "input_tokens": sum(row["total_tokens"] for row in train_rows),
                "target_tokens": sum(row["target_tokens"] for row in train_rows),
            },
            "development": {
                "file": "development.jsonl",
                "sha256": development_hash,
                "row_count": len(development_rows),
                "input_tokens": sum(row["total_tokens"] for row in development_rows),
                "target_tokens": sum(row["target_tokens"] for row in development_rows),
            },
        }
        metadata: dict[str, Any] = {
            "schema": "q25-fim-prepared-corpus-v1",
            "status": "PASS",
            "preparation_plan_sha256": EXPECTED_PLAN_SHA256,
            "parent_cpt_plan_sha256": inputs["parent_cpt_plan_sha256"],
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_sha256": tokenizer_sha256,
            "eos_token_id": eos_token_id,
            "fim_marker_ids": marker_ids,
            "files": file_records,
            "splits": split_records,
            "input_provenance": {
                "dataset_id": DATASET_ID,
                "dataset_revision": plan["sources"]["revision"],
                "pool_path": inputs["pool_path"],
                "pool_files": inputs["pool_hashes"],
                "train_manifest": inputs["train_manifest"],
                "development_manifest": inputs["development_manifest"],
                "benchmark_exclusion_sha256": inputs["benchmark_exclusion_sha256"],
                "benchmark_exclusion_count": inputs["benchmark_exclusion_count"],
                "allowed_licenses": inputs["allowed_licenses"],
            },
            "selection": audit,
            "train_license_counts": _license_counts(train_rows),
            "development_license_counts": _license_counts(development_rows),
            "training_input_tokens_include_prompt_target_and_eos": True,
            "development_input_tokens_include_prompt_target_and_eos": True,
            "sealed_test_accessed": False,
            "raw_source_content_emitted": False,
            "gpu_allocation_authorized_by_preparation": False,
        }
        artifact_bytes = train_bytes + development_bytes
        if artifact_bytes > int(plan["budget"]["new_artifact_bytes_cap"]):
            raise PreparationError("new_artifact_bytes_cap_exceeded")
        metadata["data_files_bytes"] = artifact_bytes
        metadata["corpus_sha256"] = sha256_bytes(
            f"train.jsonl\0{train_hash}\0development.jsonl\0{development_hash}".encode()
        )
        metadata_path = stage / "corpus_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        final_size = sum(path.stat().st_size for path in stage.iterdir() if path.is_file())
        if final_size > int(plan["budget"]["new_artifact_bytes_cap"]):
            raise PreparationError("new_artifact_bytes_cap_exceeded")
        os.replace(stage, output_dir)
        metadata["artifact_bytes"] = final_size
        return metadata
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _license_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for license_id in row["licenses"]:
            counts[row["language"]][license_id] += 1
    return {language: dict(sorted(counter.items())) for language, counter in sorted(counts.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        plan, inputs, train_documents, development_documents = _load_pinned_inputs(args.plan)
        tokenizer_sha256 = str(plan["data"]["tokenizer_sha256"])
        tokenizer = _load_local_tokenizer(args.tokenizer_path, tokenizer_sha256)
        train_rows, development_rows, prepared = prepare_examples(
            train_documents, development_documents, tokenizer, plan
        )
        metadata = write_corpus(
            args.output_dir,
            train_rows,
            development_rows,
            plan,
            inputs,
            prepared["audit"],
            tokenizer_sha256,
            TOKENIZER_REVISION,
            prepared["marker_ids"],
            prepared["eos_token_id"],
        )
        print(
            json.dumps(
                {
                    "status": metadata["status"],
                    "output_dir": str(args.output_dir),
                    "train_rows": metadata["splits"]["train"]["row_count"],
                    "development_rows": metadata["splits"]["development"]["row_count"],
                    "corpus_sha256": metadata["corpus_sha256"],
                    "artifact_bytes": metadata["artifact_bytes"],
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
    raise SystemExit(main())
