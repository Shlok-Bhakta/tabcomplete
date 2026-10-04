"""Local-only, resumable Qwen2.5 FIM adaptation.

The preparation worker freezes source-group-split examples and token IDs. This
module only validates those local artifacts and runs the same full-weight
training primitives as raw CPT, with response-only FIM labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tinycomplete.code_cpt import q25
from tinycomplete.data.schema import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from tinycomplete.observability.runs import run_scope
from tinycomplete.observability.spans import operation
from tinycomplete.one_line.train import (
    IGNORE_INDEX,
    CosineUpdateSchedule,
    EncodedExample,
    TrainingCursor,
    batch_order_sha256,
    selected_position_causal_loss,
    train_encoded,
)

MODEL_ID = q25.MODEL_ID
MODEL_REVISION = q25.MODEL_REVISION
TOKENIZER_SHA256 = q25.TOKENIZER_SHA256
CONFIG_SHA256 = q25.CONFIG_SHA256
WEIGHT_SHA256 = q25.MODEL_WEIGHT_SHA256
PARAMETER_COUNT = q25.PARAMETER_COUNT
VOCAB_SIZE = 151_936
TRAIN_ARM = "untouched_q25_to_fim"
CPT_ARM = "completed_cpt_q25_to_fim"
ARMS = (TRAIN_ARM, CPT_ARM)
MAX_SEQUENCE_LENGTH = 1024
MAX_TRAIN_EXAMPLES = 4096
MAX_DEVELOPMENT_EXAMPLES = 512
MAX_TARGET_TOKENS = 96
MAX_CAMPAIGN_INPUT_TOKENS = 32_000_000
CPT_EXPECTED_COMPLETE_UPDATES = 481
CPT_EXPECTED_TRAINING_INPUT_TOKENS = 7_872_512
Q25_CONFIG_SEMANTICS: dict[str, Any] = {
    "architectures": ["Qwen2ForCausalLM"],
    "bos_token_id": 151_643,
    "eos_token_id": 151_643,
    "hidden_act": "silu",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "max_window_layers": 24,
    "model_type": "qwen2",
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "vocab_size": VOCAB_SIZE,
}
FIM_SCHEMA = "q25-fim-response-only-resume-v1"
FIM_EXPORT_SCHEMA = "q25-fim-inference-f16-v1"
LATEST_SCHEMA = "q25-fim-latest-checkpoint-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise ValueError("FIM campaign JSON input is invalid") from None
    if not isinstance(value, dict):
        raise ValueError("FIM campaign JSON input must be an object")
    return value


def _configuration(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("configuration", document)
    if not isinstance(value, dict):
        raise ValueError("FIM training configuration must be an object")
    return value


def _training_config(document: dict[str, Any]) -> dict[str, Any]:
    configuration = _configuration(document)
    value = configuration.get("training", document.get("planned_training"))
    if not isinstance(value, dict):
        raise ValueError("FIM training plan is missing its training configuration")
    max_input = value.get(
        "max_input_tokens",
        value.get(
            "maximum_input_tokens_per_arm", value.get("maximum_additional_training_input_tokens")
        ),
    )
    normalized = dict(value)
    if max_input is not None:
        normalized["max_input_tokens"] = int(max_input)
    if "sequence_length" not in normalized:
        normalized["sequence_length"] = MAX_SEQUENCE_LENGTH
    return normalized


def validate_training_configuration(value: dict[str, Any]) -> dict[str, Any]:
    """Reject objective or schedule drift from the frozen paired FIM plan."""
    expected: dict[str, Any] = {
        "epochs": 1,
        "sequence_length": MAX_SEQUENCE_LENGTH,
        "effective_batch": 16,
        "microbatch_examples": 1,
        "checkpoint_every_updates": 64,
        "seed": 314159,
        "attention": "sdpa",
        "gradient_checkpointing": True,
        "master_weights": "fp32",
        "compute": "fp16",
        "optimizer": "AdamW8bit",
        "objective": "example_mean_response_only_FIM_target_and_EOS",
        "learning_rate": 1e-5,
        "warmup_fraction": 0.03,
        "cosine_floor_fraction": 0.1,
        "gradient_clip": 1.0,
        "weight_decay": 0.01,
        "initial_loss_scale": 128,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"FIM training setting differs from the frozen plan: {key}")
    max_input_tokens = value.get("max_input_tokens")
    if (
        not isinstance(max_input_tokens, int)
        or isinstance(max_input_tokens, bool)
        or not 1 <= max_input_tokens <= MAX_TRAIN_EXAMPLES * MAX_SEQUENCE_LENGTH
    ):
        raise ValueError("FIM per-arm input-token budget is invalid")
    return value


def _split_manifest_record(metadata: dict[str, Any], split: str) -> dict[str, Any]:
    splits = metadata.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get(split), dict):
        raise ValueError(f"FIM corpus manifest is missing the {split} split")
    return splits[split]


def _row_source_keys(row: dict[str, Any]) -> set[str]:
    content_hash = row.get("source_content_sha256")
    repository_hash = row.get("repository_identity_sha256")
    aliases = row.get("repository_alias_sha256")
    if not _is_sha256(content_hash) or not _is_sha256(repository_hash):
        raise ValueError("FIM row has invalid source-group hashes")
    if not isinstance(aliases, list) or any(not _is_sha256(alias) for alias in aliases):
        raise ValueError("FIM row has invalid repository alias hashes")
    return {
        f"content:{content_hash}",
        f"repository:{repository_hash}",
        *(f"repository:{x}" for x in aliases),
    }


def load_fim_examples(
    path: Path,
    *,
    split: str,
    eos_token_id: int,
    fim_marker_ids: dict[str, int],
    max_total_tokens: int = MAX_SEQUENCE_LENGTH,
    vocab_size: int = VOCAB_SIZE,
) -> tuple[EncodedExample, ...]:
    """Load prepared prompt+target+EOS IDs and derive prompt-masked labels."""
    if split not in ("train", "development"):
        raise ValueError("unknown FIM split")
    if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool) or eos_token_id < 0:
        raise ValueError("FIM tokenizer EOS ID is invalid")
    marker_keys = ("fim_prefix", "fim_suffix", "fim_middle")
    if any(
        not isinstance(fim_marker_ids.get(key), int)
        or isinstance(fim_marker_ids.get(key), bool)
        or not 0 <= int(fim_marker_ids[key]) < vocab_size
        for key in marker_keys
    ):
        raise ValueError("FIM tokenizer marker IDs are invalid")

    examples: list[EncodedExample] = []
    source_groups: set[str] = set()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError:
        raise FileNotFoundError("FIM encoded-example file is missing") from None
    with stream:
        for line in stream:
            if not line.strip():
                raise ValueError("FIM encoded-example file contains a blank row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError("FIM encoded-example row is invalid JSON") from None
            if not isinstance(row, dict):
                raise ValueError("FIM encoded-example row must be an object")
            identifier = row.get("id")
            expected_identifier = (0 if split == "train" else MAX_TRAIN_EXAMPLES) + len(examples)
            if (
                not isinstance(identifier, int)
                or isinstance(identifier, bool)
                or identifier != expected_identifier
            ):
                raise ValueError("FIM example IDs must be nonempty and unique")
            if row.get("split") != split or row.get("mode") not in (
                "whole_logical_line",
                "remaining_logical_line_after_utf8_cursor",
            ):
                raise ValueError("FIM row split or mode differs from the frozen data contract")
            if row.get("prompt_format") != "psm":
                raise ValueError("FIM row does not use the frozen PSM prompt format")
            for field in ("language", "source_path", "dataset_id", "dataset_revision"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"FIM row is missing required provenance: {field}")
            if not isinstance(row.get("licenses"), list) or any(
                not isinstance(item, str) or not item for item in row["licenses"]
            ):
                raise ValueError("FIM row has invalid license provenance")
            for field in ("prompt_sha256", "target_sha256"):
                if not _is_sha256(row.get(field)):
                    raise ValueError("FIM row has an invalid prompt or target digest")
            region_start = row.get("region_start")
            region_end = row.get("region_end")
            variant = row.get("variant")
            if (
                not isinstance(region_start, int)
                or isinstance(region_start, bool)
                or not isinstance(region_end, int)
                or isinstance(region_end, bool)
                or region_start < 0
                or region_end <= region_start
                or not isinstance(variant, int)
                or isinstance(variant, bool)
                or variant not in (0, 1)
            ):
                raise ValueError("FIM row has invalid source-region provenance")
            input_ids = row.get("input_ids")
            prompt_tokens = row.get("prompt_tokens")
            target_tokens = row.get("target_tokens")
            total_tokens = row.get("total_tokens")
            if (
                not isinstance(input_ids, list)
                or any(
                    not isinstance(token, int)
                    or isinstance(token, bool)
                    or token < 0
                    or token >= vocab_size
                    for token in input_ids
                )
                or not isinstance(prompt_tokens, int)
                or isinstance(prompt_tokens, bool)
                or not isinstance(target_tokens, int)
                or isinstance(target_tokens, bool)
                or not isinstance(total_tokens, int)
                or isinstance(total_tokens, bool)
            ):
                raise ValueError("FIM row contains invalid token IDs or lengths")
            if (
                prompt_tokens < 3
                or target_tokens < 2
                or target_tokens > MAX_TARGET_TOKENS
                or total_tokens != len(input_ids)
                or prompt_tokens + target_tokens != total_tokens
                or total_tokens > max_total_tokens
                or input_ids[-1] != eos_token_id
            ):
                raise ValueError("FIM row length, EOS, or no-truncation invariant failed")
            prompt_ids = input_ids[:prompt_tokens]
            if (
                prompt_ids[0] != fim_marker_ids["fim_prefix"]
                or prompt_ids[-1] != fim_marker_ids["fim_middle"]
                or fim_marker_ids["fim_suffix"] not in prompt_ids[1:-1]
            ):
                raise ValueError("FIM row does not contain the ordered PSM prompt markers")
            groups = _row_source_keys(row)
            source_groups.update(groups)
            examples.append(
                EncodedExample(
                    input_ids=tuple(input_ids),
                    labels=tuple([IGNORE_INDEX] * prompt_tokens + input_ids[prompt_tokens:]),
                    prompt_tokens=prompt_tokens,
                    response_tokens=target_tokens,
                    total_tokens=total_tokens,
                    prompt="",
                    response="",
                )
            )
    if not examples:
        raise ValueError(f"FIM {split} split is empty")
    limit = MAX_TRAIN_EXAMPLES if split == "train" else MAX_DEVELOPMENT_EXAMPLES
    if len(examples) > limit:
        raise ValueError(f"FIM {split} split exceeds its frozen example cap")
    return tuple(examples)


def load_fim_corpus(
    *,
    train_path: Path,
    development_path: Path,
    metadata_path: Path,
    plan: dict[str, Any],
) -> tuple[tuple[EncodedExample, ...], tuple[EncodedExample, ...], dict[str, Any]]:
    metadata = _read_json(metadata_path)
    preparation_hash = plan.get("preparation_plan_sha256")
    if (
        not _is_sha256(preparation_hash)
        or metadata.get("preparation_plan_sha256") != preparation_hash
    ):
        raise ValueError("FIM corpus was not prepared from the frozen preparation plan")
    if (
        metadata.get("tokenizer_id") != MODEL_ID
        or metadata.get("tokenizer_revision") != MODEL_REVISION
        or metadata.get("tokenizer_sha256") != TOKENIZER_SHA256
    ):
        raise ValueError("FIM corpus tokenizer identity differs from the pinned Q25 tokenizer")
    if metadata.get("schema") != "q25-fim-prepared-corpus-v1":
        raise ValueError("FIM corpus manifest has an unknown schema")
    if metadata.get("raw_source_content_emitted") is not False:
        raise ValueError("FIM corpus must not emit raw source content")
    if metadata.get("sealed_test_accessed") is not False:
        raise ValueError("FIM corpus must not access sealed test fixtures")
    data_plan = _configuration(plan).get("data", plan.get("data", {}))
    if not isinstance(data_plan, dict):
        raise ValueError("FIM full training plan has no data identity")
    parent_cpt_hash = plan.get("parent_cpt_plan_sha256")
    if not _is_sha256(parent_cpt_hash) or metadata.get("parent_cpt_plan_sha256") != parent_cpt_hash:
        raise ValueError("FIM corpus parent CPT identity differs from the training plan")
    eos_token_id = metadata.get("eos_token_id")
    marker_ids = metadata.get("fim_marker_ids")
    if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool):
        raise ValueError("FIM corpus is missing its tokenizer EOS ID")
    if not isinstance(marker_ids, dict):
        raise ValueError("FIM corpus is missing marker-token identity")

    paths = {"train": train_path, "development": development_path}
    rows_by_split: dict[str, tuple[EncodedExample, ...]] = {}
    split_identity: dict[str, dict[str, Any]] = {}
    all_train_groups: set[str] = set()
    all_development_groups: set[str] = set()
    for split, path in paths.items():
        record = _split_manifest_record(metadata, split)
        file_name = record.get("file")
        expected_sha = record.get("sha256")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or file_name != path.name
            or not _is_sha256(expected_sha)
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"FIM {split} data file differs from its frozen corpus manifest")
        files = metadata.get("files")
        if not isinstance(files, dict):
            raise ValueError("FIM corpus manifest has no file ledger")
        file_record = files.get(file_name)
        if not isinstance(file_record, dict):
            raise ValueError(f"FIM {split} file is absent from the corpus file ledger")
        if (
            file_record.get("sha256") != expected_sha
            or file_record.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"FIM {split} file differs from the corpus file ledger")
        examples = load_fim_examples(
            path,
            split=split,
            eos_token_id=eos_token_id,
            fim_marker_ids=marker_ids,
        )
        row_count = record.get("row_count")
        input_tokens = record.get("input_tokens")
        target_tokens = record.get("target_tokens")
        actual_input = sum(item.total_tokens for item in examples)
        actual_targets = sum(item.response_tokens for item in examples)
        if (
            row_count != len(examples)
            or input_tokens != actual_input
            or target_tokens != actual_targets
        ):
            raise ValueError(f"FIM {split} row or token totals differ from its frozen manifest")
        rows_by_split[split] = examples
        split_identity[split] = {
            "sha256": expected_sha,
            "rows": row_count,
            "input_tokens": actual_input,
            "target_tokens": actual_targets,
        }
        groups: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                groups.update(_row_source_keys(row))
        if split == "train":
            all_train_groups = groups
        else:
            all_development_groups = groups
    if all_train_groups & all_development_groups:
        raise ValueError("FIM train and development source groups overlap")

    metadata_sha = data_plan.get("corpus_metadata_sha256") or data_plan.get("metadata_sha256")
    if metadata_sha is not None and sha256_file(metadata_path) != metadata_sha:
        raise ValueError("FIM corpus metadata hash differs from the frozen training plan")
    if plan.get("schema") == "q25-fim-training-plan-v1":
        if not _is_sha256(metadata_sha) or sha256_file(metadata_path) != metadata_sha:
            raise ValueError("full FIM plan must bind the corpus metadata SHA-256")
        for split in paths:
            expected_record = data_plan.get(split)
            if expected_record is None and isinstance(data_plan.get("splits"), dict):
                expected_record = data_plan["splits"].get(split)
            if not isinstance(expected_record, dict):
                raise ValueError(f"full FIM plan is missing its {split} data identity")
            actual = split_identity[split]
            required_fields = ("sha256", "bytes", "row_count", "input_tokens", "target_tokens")
            if any(field not in expected_record for field in required_fields):
                raise ValueError(f"full FIM plan has incomplete {split} data identity")
            expected_values = {
                "sha256": actual["sha256"],
                "bytes": paths[split].stat().st_size,
                "row_count": actual["rows"],
                "input_tokens": actual["input_tokens"],
                "target_tokens": actual["target_tokens"],
            }
            if any(expected_record[field] != value for field, value in expected_values.items()):
                raise ValueError(f"FIM {split} data differs from the frozen training plan")
    result_identity = {
        "metadata_sha256": sha256_file(metadata_path),
        "preparation_plan_sha256": preparation_hash,
        "tokenizer_revision": metadata["tokenizer_revision"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "eos_token_id": eos_token_id,
        "fim_marker_ids": marker_ids,
        "splits": split_identity,
    }
    return rows_by_split["train"], rows_by_split["development"], result_identity


def validate_fim_tokenizer(
    tokenizer: Any,
    metadata: dict[str, Any],
    examples: tuple[EncodedExample, ...] = (),
) -> None:
    """Check frozen FIM IDs and exact round-trips for the prepared token corpus."""
    if tokenizer.eos_token_id != metadata.get("eos_token_id"):
        raise ValueError("loaded Q25 EOS ID differs from the FIM corpus")
    for token, key in (
        (FIM_PREFIX, "fim_prefix"),
        (FIM_SUFFIX, "fim_suffix"),
        (FIM_MIDDLE, "fim_middle"),
    ):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        token_id = metadata["fim_marker_ids"][key]
        if encoded != [token_id] or tokenizer.convert_tokens_to_ids(token) != token_id:
            raise ValueError("loaded Q25 tokenizer does not preserve a frozen FIM marker ID")
    for index, example in enumerate(examples):
        decoded = tokenizer.decode(
            list(example.input_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        encoded = tokenizer.encode(decoded, add_special_tokens=False)
        if tuple(encoded) != example.input_ids:
            raise ValueError(f"loaded tokenizer changes prepared FIM token IDs at row {index}")


def validate_local_tokenizer_corpus(
    model_path: Path,
    metadata: dict[str, Any],
    examples: tuple[EncodedExample, ...],
) -> None:
    """Load only the local tokenizer and verify every frozen training/dev row."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("the pinned local Transformers tokenizer is unavailable") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    validate_fim_tokenizer(tokenizer, metadata, examples)


def training_batches(
    examples: tuple[EncodedExample, ...], *, effective_batch: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    return q25.training_batches(examples, effective_batch=effective_batch, seed=seed)


def _initializer_kind(arm: str, entry: dict[str, Any]) -> str:
    if arm not in ARMS:
        raise ValueError("unknown Q25 FIM arm")
    kind = entry.get("kind", entry.get("initializer"))
    expected = "untouched_pretrained" if arm == TRAIN_ARM else "completed_cpt_export"
    if kind != expected:
        raise ValueError("FIM arm initializer kind differs from its frozen identity")
    return expected


def verify_initializer(
    model_path: Path,
    *,
    arm: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Verify either the original snapshot or a completed local CPT export."""
    kind = _initializer_kind(arm, entry)
    if not model_path.is_dir():
        raise FileNotFoundError("local Q25 initializer directory is missing")
    if (
        entry.get("model_id", MODEL_ID) != MODEL_ID
        or entry.get("revision", MODEL_REVISION) != MODEL_REVISION
    ):
        raise ValueError("FIM initializer plan names a different Q25 model revision")
    files: dict[str, dict[str, Any]] = {}
    if kind == "untouched_pretrained":
        required = {
            "config.json": CONFIG_SHA256,
            "model.safetensors": WEIGHT_SHA256,
            "tokenizer.json": TOKENIZER_SHA256,
        }
        expected_files = entry.get("files", {})
        if not isinstance(expected_files, dict):
            raise ValueError("untouched initializer plan has no file identities")
        for name, pinned_digest in required.items():
            path = model_path / name
            if not path.is_file():
                raise FileNotFoundError("untouched Q25 snapshot is missing a required file")
            digest = sha256_file(path)
            record = expected_files.get(name, {})
            expected_digest = record.get("sha256") if isinstance(record, dict) else record
            expected_bytes = record.get("bytes") if isinstance(record, dict) else None
            if (
                digest != pinned_digest
                or expected_digest != digest
                or expected_bytes != path.stat().st_size
            ):
                raise ValueError("untouched Q25 file differs from its frozen identity")
            files[name] = {"bytes": path.stat().st_size, "sha256": digest}
        _verify_q25_config(model_path / "config.json")
        initializer_identity: dict[str, Any] = {
            "kind": kind,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "files": files,
        }
    else:
        manifest_path = model_path / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("completed CPT export manifest is missing")
        manifest_sha = sha256_file(manifest_path)
        expected_manifest_sha = entry.get("artifact_manifest_sha256")
        if not _is_sha256(expected_manifest_sha) or manifest_sha != expected_manifest_sha:
            raise ValueError("CPT export manifest differs from the frozen FIM initializer")
        manifest = _read_json(manifest_path)
        if (
            manifest.get("schema") != "q25-cpt-inference-f16-v1"
            or manifest.get("base_model") != MODEL_ID
            or manifest.get("base_revision") != MODEL_REVISION
            or manifest.get("fingerprint") != entry.get("fingerprint")
            or manifest.get("training_cursor") != entry.get("training_cursor")
        ):
            raise ValueError("CPT export is not the completed initializer frozen for FIM")
        cursor = manifest.get("training_cursor")
        if not isinstance(cursor, dict) or cursor.get("attempted_updates") != cursor.get(
            "completed_updates"
        ):
            raise ValueError("CPT initializer contains skipped or incomplete updates")
        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, dict) or not manifest_files:
            raise ValueError("CPT export manifest contains no model files")
        expected_files = entry.get("files")
        if not isinstance(expected_files, dict) or set(expected_files) != set(manifest_files):
            raise ValueError("CPT initializer plan has invalid file identities")
        for name, record in manifest_files.items():
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("CPT manifest contains an unsafe relative filename")
            path = model_path / name
            if not path.is_file() or not isinstance(record, dict):
                raise FileNotFoundError("CPT export is missing a manifested file")
            digest = sha256_file(path)
            expected = expected_files.get(name)
            expected_digest = expected.get("sha256") if isinstance(expected, dict) else expected
            expected_bytes = expected.get("bytes") if isinstance(expected, dict) else None
            if (
                digest != record.get("sha256")
                or path.stat().st_size != record.get("bytes")
                or expected_digest != digest
                or expected_bytes != path.stat().st_size
            ):
                raise ValueError("CPT exported file differs from its frozen manifest")
            files[name] = {"bytes": path.stat().st_size, "sha256": digest}
        _verify_q25_config(model_path / "config.json")
        expected_updates = entry.get("expected_complete_updates")
        expected_tokens = entry.get("expected_training_input_tokens")
        if (
            expected_updates != CPT_EXPECTED_COMPLETE_UPDATES
            or expected_tokens != CPT_EXPECTED_TRAINING_INPUT_TOKENS
        ):
            raise ValueError("CPT initializer plan does not bind the frozen complete-run totals")
        if (
            cursor.get("attempted_updates") != expected_updates
            or cursor.get("completed_updates") != expected_updates
            or cursor.get("skipped_updates") != 0
            or cursor.get("training_input_tokens") != expected_tokens
        ):
            raise ValueError("CPT initializer is not the frozen complete training run")
        disk_files = {
            path.relative_to(model_path).as_posix()
            for path in model_path.rglob("*")
            if path.is_file()
        }
        if disk_files != set(manifest_files) | {"artifact_manifest.json"}:
            raise ValueError("CPT export contains files outside its frozen artifact manifest")
        initializer_identity = {
            "kind": kind,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifact_manifest_sha256": manifest_sha,
            "fingerprint": manifest["fingerprint"],
            "training_cursor": manifest["training_cursor"],
            "files": files,
        }
    return initializer_identity


def _verify_q25_config(config_path: Path) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise ValueError("Q25 initializer config is invalid") from None
    normalized_fields = {"rope_theta", "sliding_window"}
    if not isinstance(config, dict) or any(
        config.get(key) != expected
        for key, expected in Q25_CONFIG_SEMANTICS.items()
        if key not in normalized_fields
    ):
        raise ValueError("Q25 initializer config differs from pinned architecture semantics")
    # Transformers 5 migrates legacy RoPE fields and clears an inactive window.
    # Compare their meaning while retaining the exact serialized file hashes.
    rope = config.get("rope_parameters")
    if rope is None:
        rope = {"rope_type": "default", "rope_theta": config.get("rope_theta")}
    if (
        rope != {"rope_type": "default", "rope_theta": 1_000_000.0}
        or config.get("rope_scaling") is not None
        or config.get("sliding_window") not in (None, 32768)
        or ("layer_types" in config and config["layer_types"] != ["full_attention"] * 24)
    ):
        raise ValueError("Q25 initializer config differs from pinned architecture semantics")
    return config


def runtime_identity(device: Any) -> dict[str, Any]:
    identity = q25.runtime_identity(device)
    identity["source_sha256"]["q25_fim_training"] = sha256_file(Path(__file__))
    return identity


def _check_runtime_plan(expected: Any, actual: dict[str, Any]) -> None:
    if not isinstance(expected, dict):
        raise ValueError("full FIM training plan is missing runtime identity")
    versions = expected.get("expected_versions", expected)
    if not isinstance(versions, dict):
        raise ValueError("full FIM training plan has invalid runtime versions")
    required = ("python", "torch", "transformers", "bitsandbytes", "cuda_runtime")
    if any(key not in versions for key in required):
        raise ValueError("full FIM training plan omits pinned runtime versions")
    for key in required:
        if versions[key] != actual.get(key):
            raise ValueError(f"Q25 FIM runtime differs from its frozen plan: {key}")


def save_fim_training_cursor(
    output: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    fingerprint: str,
    cursor: TrainingCursor,
    output_cap_bytes: int,
    total_cap_bytes: int,
    input_artifact_bytes: int,
    minimum_free_bytes: int,
    source_weight_bytes: int,
) -> Path:
    """Reuse the full checkpoint format and tag its atomic pointer as FIM."""
    checkpoint = q25.save_training_cursor(
        output,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        fingerprint=fingerprint,
        cursor=cursor,
        output_cap_bytes=output_cap_bytes,
        total_cap_bytes=total_cap_bytes,
        input_artifact_bytes=input_artifact_bytes,
        minimum_free_bytes=minimum_free_bytes,
        source_weight_bytes=checkpoint_weight_basis(model, source_weight_bytes),
    )
    pointer_path = output / "latest.json"
    pointer = _read_json(pointer_path)
    if pointer.get("fingerprint") != fingerprint or pointer.get("path") != checkpoint.name:
        raise ValueError("FIM latest checkpoint pointer does not match its committed state")
    if pointer.get("schema") != LATEST_SCHEMA:
        pointer["schema"] = LATEST_SCHEMA
        q25.atomic_json(pointer_path, pointer)
    return checkpoint


def checkpoint_weight_basis(model: Any, serialized_weight_bytes: int) -> int:
    """Bound the BF16 basis by unique parameters, excluding duplicated tied tensors.

    The parent estimator reserves four times this basis for FP32 masters and
    optimizer state, and independently checks actual model/optimizer tensor bytes.
    The CPT export serializes tied embeddings twice. Its file size must still be
    counted as input storage, but it does not double resumable training state.
    """
    logical_bf16_bytes = sum(
        parameter.numel() * 2 for parameter in model.parameters() if parameter.is_floating_point()
    )
    if serialized_weight_bytes <= 0 or logical_bf16_bytes <= 0:
        raise ValueError("checkpoint weight basis must be positive")
    return min(serialized_weight_bytes, logical_bf16_bytes)


def _budget(document: dict[str, Any]) -> dict[str, Any]:
    configuration = _configuration(document)
    value = configuration.get("budget", document.get("budget"))
    if not isinstance(value, dict):
        raise ValueError("full FIM training plan is missing its budget")
    return value


def _plan_data_record(document: dict[str, Any], split: str) -> dict[str, Any]:
    configuration = _configuration(document)
    data = configuration.get("data", document.get("data"))
    if not isinstance(data, dict):
        raise ValueError("full FIM training plan is missing data identity")
    record = data.get(split)
    if record is None and isinstance(data.get("splits"), dict):
        record = data["splits"].get(split)
    if not isinstance(record, dict):
        raise ValueError(f"full FIM training plan is missing {split} identity")
    return record


def evaluate_fim_nll(
    model: Any,
    examples: tuple[EncodedExample, ...],
    *,
    device: Any,
    pad_token_id: int,
) -> dict[str, Any]:
    """Measure equal-example response-token loss on the fixed held-out FIM set."""
    import torch

    from tinycomplete.one_line.train import collate_examples

    if not examples:
        raise ValueError("FIM validation set is empty")
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    try:
        with operation("model.score"):
            with torch.inference_mode():
                for start in range(0, len(examples), 1):
                    batch = examples[start : start + 1]
                    tensors = {
                        key: value.to(device)
                        for key, value in collate_examples(batch, pad_token_id=pad_token_id).items()
                    }
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        loss = selected_position_causal_loss(model, **tensors)
                    total_loss += loss.detach().float() * len(batch)
                    count += len(batch)
                    del tensors, loss
    finally:
        model.train(was_training)
    value = float((total_loss / count).item())
    from tinycomplete.observability.metrics import record_metric

    record_metric(
        "tabcomplete.training.validation_nll",
        value,
        {"rank_role": "authoritative", "task": "fim_response"},
        kind="gauge",
    )
    return {
        "metric": "fim_response_token_nll_equal_example",
        "nll": value,
        "examples": count,
        "input_tokens": sum(item.total_tokens for item in examples),
        "target_tokens_including_eos": sum(item.response_tokens for item in examples),
        "max_sequence_tokens": max(item.total_tokens for item in examples),
        "tokenizer_model": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
    }


def _input_artifact_bytes(
    *,
    model_path: Path,
    train_path: Path,
    development_path: Path,
    metadata_path: Path,
    plan_path: Path,
    resume_path: Path | None,
    output: Path,
    mounted_artifact_bytes: int = 0,
) -> int:
    if mounted_artifact_bytes < 0:
        raise ValueError("mounted artifact bytes cannot be negative")
    model_bytes = q25.directory_bytes(model_path)
    resume_bytes = 0
    if resume_path is not None and not resume_path.resolve().is_relative_to(output.resolve()):
        marker_path = resume_path.with_suffix(resume_path.suffix + ".complete.json")
        if not marker_path.is_file():
            raise ValueError("resume checkpoint is missing its committed completion marker")
        resume_bytes = resume_path.stat().st_size + marker_path.stat().st_size
    return (
        model_bytes
        + train_path.stat().st_size
        + development_path.stat().st_size
        + metadata_path.stat().st_size
        + plan_path.stat().st_size
        + resume_bytes
        + mounted_artifact_bytes
    )


def _restore_first_evaluation(output: Path, resume_path: Path | None, fingerprint: str) -> None:
    if resume_path is None:
        return
    previous = resume_path.parent
    previous_manifest_path = previous / "run_manifest.json"
    try:
        manifest = _read_json(previous_manifest_path)
    except (OSError, ValueError):
        return
    if manifest.get("fingerprint") != fingerprint:
        return
    before_path = previous / "fim-development-before.json"
    target_path = output / before_path.name
    if not target_path.exists() and before_path.is_file():
        try:
            before = _read_json(before_path)
        except ValueError:
            before = {}
        if before.get("fingerprint") == fingerprint:
            q25.atomic_json(target_path, before)
    previous_run = previous / "observability-run.json"
    target_run = output / previous_run.name
    if not target_run.exists() and previous_run.is_file():
        try:
            metadata = _read_json(previous_run)
        except ValueError:
            metadata = {}
        if set(metadata) >= {"campaign_id", "run_id"}:
            q25.atomic_json(target_run, metadata)


def save_fim_inference_export(
    model: Any,
    tokenizer: Any,
    destination: Path,
    *,
    output_root: Path,
    output_cap_bytes: int,
    total_cap_bytes: int,
    input_artifact_bytes: int,
    minimum_free_bytes: int,
    fingerprint: str,
    cursor: TrainingCursor,
    arm: str,
    initializer_identity: dict[str, Any],
) -> dict[str, Any]:
    """Atomically export FP16 weights with a FIM-specific parent manifest."""
    import torch

    backup = destination.with_name(destination.name + ".previous")
    if not destination.exists() and backup.exists():
        os.replace(backup, destination)
    if destination.exists():
        manifest_path = destination / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError("FIM export path already exists without its manifest")
        manifest = _read_json(manifest_path)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("existing FIM export belongs to another training run")
        if manifest.get("training_cursor") == asdict(cursor):
            if backup.exists():
                shutil.rmtree(backup)
            return {"path": str(destination), "files": manifest.get("files", {})}
    source_weight_bytes = int(initializer_identity["files"]["model.safetensors"]["bytes"])
    predicted = source_weight_bytes * 2 + 64 * 1024**2
    q25._storage_preflight(
        output=output_root,
        predicted_write_bytes=predicted,
        output_cap_bytes=output_cap_bytes,
        total_cap_bytes=total_cap_bytes,
        input_artifact_bytes=input_artifact_bytes,
        minimum_free_bytes=minimum_free_bytes,
    )
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with operation("checkpoint.save"):
            state = {
                name: (
                    tensor.detach().to(device="cpu", dtype=torch.float16)
                    if tensor.is_floating_point()
                    else tensor.detach().cpu()
                )
                for name, tensor in model.state_dict().items()
            }
            model.save_pretrained(
                temporary,
                state_dict=state,
                safe_serialization=True,
                max_shard_size="2GB",
            )
            tokenizer.save_pretrained(temporary)
            files = {
                path.relative_to(temporary).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            }
            q25.atomic_json(
                temporary / "artifact_manifest.json",
                {
                    "schema": FIM_EXPORT_SCHEMA,
                    "fingerprint": fingerprint,
                    "arm": arm,
                    "base_model": MODEL_ID,
                    "base_revision": MODEL_REVISION,
                    "base_snapshot_provenance": {
                        "config_sha256": CONFIG_SHA256,
                        "tokenizer_sha256": TOKENIZER_SHA256,
                        "model_weights_sha256": WEIGHT_SHA256,
                    },
                    "initializer": initializer_identity,
                    "training_cursor": asdict(cursor),
                    "precision": "F16 export from FP32 master parameters",
                    "files": files,
                },
            )
        actual = q25.directory_bytes(output_root)
        if actual > output_cap_bytes or input_artifact_bytes + actual > total_cap_bytes:
            raise OSError("FIM inference export exceeds its declared storage cap")
        if destination.exists():
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
        os.replace(temporary, destination)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"path": str(destination), "files": files}


def run_training(
    *,
    model_path: Path,
    train_path: Path,
    development_path: Path,
    metadata_path: Path,
    plan_path: Path,
    arm: str,
    output: Path,
    session_seconds: float,
    reserve_seconds: float | None = None,
    resume: Path | None = None,
    external_campaign_tokens: int = 0,
    mounted_artifact_bytes: int = 0,
    execute: bool = False,
    invocation_started: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic() if invocation_started is None else invocation_started
    if session_seconds <= 0:
        raise ValueError("session duration must be positive")
    if external_campaign_tokens < 0 or mounted_artifact_bytes < 0:
        raise ValueError("external campaign tokens and mounted artifact bytes cannot be negative")
    deadline = started + session_seconds
    document = _read_json(plan_path)
    if arm not in ARMS:
        raise ValueError("unknown Q25 FIM training arm")
    training = validate_training_configuration(_training_config(document))
    budget = _budget(document)
    if execute and document.get("gpu_execution_authorized") is not True:
        raise PermissionError("this plan does not authorize GPU execution")
    if document.get("schema") != "q25-fim-training-plan-v1":
        if execute:
            raise ValueError("CPU preparation plan cannot authorize Q25 FIM training")
    plan_session = budget.get("session_seconds")
    if plan_session is not None and session_seconds > float(plan_session):
        raise ValueError("session duration exceeds the frozen FIM plan")
    minimum_reserve = max(
        1800.0,
        float(budget.get("minimum_finalization_reserve_seconds", 1800)),
    )
    reserve = (
        float(budget.get("finalization_reserve_seconds", minimum_reserve))
        if reserve_seconds is None
        else float(reserve_seconds)
    )
    if reserve < minimum_reserve:
        raise ValueError("FIM finalization reserve is below the frozen minimum")
    campaign_cap = int(
        budget.get(
            "maximum_campaign_input_tokens",
            budget.get("max_campaign_input_tokens", MAX_CAMPAIGN_INPUT_TOKENS),
        )
    )
    if not 1 <= campaign_cap <= MAX_CAMPAIGN_INPUT_TOKENS:
        raise ValueError("global campaign input-token cap is outside the frozen maximum")

    initializer_entries = document.get("initializers")
    if not isinstance(initializer_entries, dict) or not isinstance(
        initializer_entries.get(arm), dict
    ):
        raise ValueError("FIM plan does not identify the selected initializer")
    initializer_identity = verify_initializer(model_path, arm=arm, entry=initializer_entries[arm])
    examples, development_examples, data_identity = load_fim_corpus(
        train_path=train_path,
        development_path=development_path,
        metadata_path=metadata_path,
        plan=document,
    )
    corpus_metadata = _read_json(metadata_path)
    validate_local_tokenizer_corpus(model_path, corpus_metadata, examples + development_examples)
    input_tokens = sum(item.total_tokens for item in examples)
    target_tokens = sum(item.response_tokens for item in examples)
    if input_tokens > int(training["max_input_tokens"]):
        raise ValueError("prepared FIM examples exceed the frozen per-arm token budget")
    actual_planned_campaign_tokens = q25.validate_campaign_token_budget(
        logical_training_tokens=input_tokens,
        external_campaign_tokens=external_campaign_tokens,
        maximum_additional_tokens=campaign_cap,
    )
    batches = training_batches(
        examples,
        effective_batch=int(training["effective_batch"]),
        seed=int(training["seed"]),
    )
    if len(batches) != (len(examples) + int(training["effective_batch"]) - 1) // int(
        training["effective_batch"]
    ):
        raise ValueError("FIM batch schedule is not a single pass")
    plan_sha = sha256_file(plan_path)
    if not execute:
        return {
            "status": "preflight",
            "arm": arm,
            "initializer": initializer_identity,
            "plan_sha256": plan_sha,
            "data": data_identity,
            "training_examples": len(examples),
            "development_examples": len(development_examples),
            "training_input_tokens": input_tokens,
            "external_campaign_tokens": external_campaign_tokens,
            "actual_planned_campaign_tokens": actual_planned_campaign_tokens,
            "maximum_campaign_input_tokens": campaign_cap,
            "maximum_additional_training_input_tokens": campaign_cap,
            "supervised_target_tokens_including_eos": target_tokens,
            "effective_batch": int(training["effective_batch"]),
            "microbatch_examples": int(training["microbatch_examples"]),
            "updates": len(batches),
            "batch_order_sha256": batch_order_sha256(batches),
        }

    import torch

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]))
    random.seed(int(training["seed"]))
    torch.cuda.manual_seed_all(int(training["seed"]))
    device = torch.device("cuda:0")
    runtime = runtime_identity(device)
    runtime_training = q25._runtime_training(training)
    runtime["training"] = runtime_training
    runtime_config = _configuration(document).get("runtime", document.get("runtime"))
    _check_runtime_plan(runtime_config, runtime)
    identity = {
        "schema": FIM_SCHEMA,
        "plan_sha256": plan_sha,
        "arm": arm,
        "initializer": initializer_identity,
        "training_data": {
            **data_identity["splits"]["train"],
            "batch_order_sha256": batch_order_sha256(batches),
            "batch_count": len(batches),
        },
        "development_data": data_identity["splits"]["development"],
        "corpus_metadata": data_identity,
        "training": training,
        "runtime": runtime,
    }
    fingerprint = canonical_sha256(identity)
    output_cap_bytes = min(
        int(training.get("max_output_bytes", 12 * 1024**3)),
        int(budget.get("output_bytes_cap", budget.get("new_artifact_bytes_cap", 12 * 1024**3))),
    )
    total_cap_bytes = int(budget.get("new_artifact_bytes_cap", 12 * 1024**3))
    minimum_free_bytes = int(budget.get("minimum_free_bytes", 2 * 1024**3))
    resume_path = q25.resolve_resume_path(resume) if resume is not None else None
    input_artifact_bytes = _input_artifact_bytes(
        model_path=model_path,
        train_path=train_path,
        development_path=development_path,
        metadata_path=metadata_path,
        plan_path=plan_path,
        resume_path=resume_path,
        output=output,
        mounted_artifact_bytes=mounted_artifact_bytes,
    )
    if output.exists() and resume is None and any(output.iterdir()):
        raise FileExistsError("existing FIM output requires an exact-resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    run_manifest = {"fingerprint": fingerprint, "identity": identity}
    if manifest_path.exists():
        if _read_json(manifest_path).get("fingerprint") != fingerprint:
            raise ValueError("FIM model, data, runtime, or code identity changed on resume")
    else:
        q25.atomic_json(manifest_path, run_manifest)
    _restore_first_evaluation(output, resume_path, fingerprint)

    with run_scope(output / "observability-run.json", "q25-fim"):
        torch.cuda.reset_peak_memory_stats(device)
        with operation("model.load"):
            model, tokenizer = q25._load_model(model_path)
        if model.config.vocab_size != VOCAB_SIZE:
            raise ValueError("loaded Q25 vocabulary differs from the pinned model")
        validate_fim_tokenizer(tokenizer, corpus_metadata)
        if training["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        model.to(device)
        optimizer, optimizer_name = q25._make_optimizer(model, training)
        scheduler = CosineUpdateSchedule(
            optimizer,
            peak_lr=float(training["learning_rate"]),
            total_updates=len(batches),
            warmup_fraction=float(training["warmup_fraction"]),
            floor_fraction=float(training["cosine_floor_fraction"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda", init_scale=float(training["initial_loss_scale"]), growth_interval=2_000
        )
        before_path = output / "fim-development-before.json"
        if before_path.exists():
            before = _read_json(before_path)
            if before.get("fingerprint") != fingerprint:
                raise ValueError("stored FIM validation baseline belongs to another run")
        else:
            before = evaluate_fim_nll(
                model,
                development_examples,
                device=device,
                pad_token_id=int(tokenizer.eos_token_id),
            )
            before["fingerprint"] = fingerprint
            q25.atomic_json(before_path, before)

        cursor = TrainingCursor()
        if resume_path is not None:
            cursor = q25.load_training_cursor(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                fingerprint=fingerprint,
            )
        if external_campaign_tokens + cursor.training_input_tokens > campaign_cap:
            raise ValueError(
                "external campaign carry plus resume cursor exceeds the global token cap"
            )
        latest_path: Path | None = None
        latest_json = output / "latest.json"
        if latest_json.exists():
            latest_path = q25.resolve_resume_path(latest_json)
        max_checkpoint_save_seconds = 0.0

        def save_cursor(value: TrainingCursor) -> None:
            nonlocal latest_path, max_checkpoint_save_seconds
            checkpoint_started = time.monotonic()
            try:
                latest_path = save_fim_training_cursor(
                    output,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    fingerprint=fingerprint,
                    cursor=value,
                    output_cap_bytes=output_cap_bytes,
                    total_cap_bytes=total_cap_bytes,
                    input_artifact_bytes=input_artifact_bytes,
                    minimum_free_bytes=minimum_free_bytes,
                    source_weight_bytes=int(
                        initializer_identity["files"]["model.safetensors"]["bytes"]
                    ),
                )
            finally:
                max_checkpoint_save_seconds = max(
                    max_checkpoint_save_seconds, time.monotonic() - checkpoint_started
                )

        def dynamic_reserve_seconds() -> float:
            return max(reserve, 2 * max_checkpoint_save_seconds + 60)

        # A complete cursor zero protects the run if its first update fails.
        save_cursor(cursor)

        successful_updates = cursor.completed_updates

        def log_update(record: dict[str, int | float | bool]) -> None:
            nonlocal successful_updates
            campaign_tokens = external_campaign_tokens + int(record["cumulative_input_tokens"])
            if campaign_tokens > campaign_cap:
                raise ValueError("FIM update would exceed the global campaign token cap")
            successful_updates += int(bool(record["applied"]))
            with (output / "updates.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            from tinycomplete.observability.hooks import training_progress

            training_progress(
                {
                    "optimizer_step": successful_updates,
                    "training_tokens": int(record["cumulative_input_tokens"]),
                    "loss": float(record["mean_example_loss"]),
                    "gradient_norm": float(record["gradient_norm"]),
                    "learning_rate": float(record["learning_rate"]),
                    "event": "update" if record["applied"] else "loss_scale_overflow",
                }
            )

        result = train_encoded(
            model,
            examples,
            batches,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            pad_token_id=int(tokenizer.eos_token_id),
            cursor=cursor,
            microbatch_examples=int(training["microbatch_examples"]),
            max_input_tokens=campaign_cap,
            external_campaign_tokens=external_campaign_tokens,
            deadline_monotonic=deadline,
            finalization_reserve_seconds=dynamic_reserve_seconds,
            checkpoint_every_updates=int(training["checkpoint_every_updates"]),
            on_checkpoint=save_cursor,
            on_update=log_update,
        )
        summary: dict[str, Any] = {
            "schema": "q25-fim-response-only-run-v1",
            "status": result.status,
            "arm": arm,
            "fingerprint": fingerprint,
            "identity": identity,
            "cursor": asdict(result.cursor),
            "logical_training_input_tokens": result.cursor.training_input_tokens,
            "external_campaign_tokens": external_campaign_tokens,
            "actual_campaign_input_tokens": (
                external_campaign_tokens + result.cursor.training_input_tokens
            ),
            "actual_planned_campaign_tokens": actual_planned_campaign_tokens,
            "maximum_campaign_input_tokens": campaign_cap,
            "maximum_additional_training_input_tokens": campaign_cap,
            "maximum_per_arm_training_input_tokens": int(training["max_input_tokens"]),
            "supervised_target_tokens_including_eos": result.cursor.supervised_target_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "session_seconds": session_seconds,
            "finalization_reserve_seconds": dynamic_reserve_seconds(),
            "max_checkpoint_save_seconds": max_checkpoint_save_seconds,
            "training_updates": len(result.update_records),
            "optimizer": optimizer_name,
            "training_memory": {
                "scope": "model load, initial development loss, updates and checkpointing",
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cuda_retained_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_retained_reserved_bytes": torch.cuda.memory_reserved(device),
                "driver_and_other_process_memory_included": False,
                "inference_service_memory": False,
            },
            "development_before": before,
            "latest_checkpoint": latest_path.name if latest_path else None,
            "storage": {
                "input_artifact_bytes_including_resume": input_artifact_bytes,
                "mounted_artifact_bytes": mounted_artifact_bytes,
                "output_bytes": q25.directory_bytes(output),
                "output_cap_bytes": output_cap_bytes,
                "aggregate_cap_bytes": total_cap_bytes,
                "checkpoint_estimate_bytes": q25.estimate_checkpoint_bytes(
                    model,
                    optimizer,
                    checkpoint_weight_basis(
                        model,
                        int(initializer_identity["files"]["model.safetensors"]["bytes"]),
                    ),
                ),
            },
        }
        q25.atomic_json(output / "run_result.json", summary)
        if result.cursor.attempted_updates:
            after = evaluate_fim_nll(
                model,
                development_examples,
                device=device,
                pad_token_id=int(tokenizer.eos_token_id),
            )
            after["fingerprint"] = fingerprint
            q25.atomic_json(output / "fim-development-after.json", after)
            summary["development_after"] = after
            # The worker evaluates exports only after a complete pass. Keep a
            # partial run's resumable state without an unused duplicate export.
            if result.status == "complete":
                summary["inference_export"] = save_fim_inference_export(
                    model,
                    tokenizer,
                    output / "inference-f16",
                    output_root=output,
                    output_cap_bytes=output_cap_bytes,
                    total_cap_bytes=total_cap_bytes,
                    input_artifact_bytes=input_artifact_bytes,
                    minimum_free_bytes=minimum_free_bytes,
                    fingerprint=fingerprint,
                    cursor=result.cursor,
                    arm=arm,
                    initializer_identity=initializer_identity,
                )
            summary["storage"]["output_bytes"] = q25.directory_bytes(output)
        q25.atomic_json(output / "run_result.json", summary)
        return summary


def main() -> None:
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--data-metadata", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-seconds", type=float, required=True)
    parser.add_argument("--reserve-seconds", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--external-campaign-tokens", type=int, default=0)
    parser.add_argument("--mounted-artifact-bytes", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_training(
        model_path=args.model,
        train_path=args.train,
        development_path=args.development,
        metadata_path=args.data_metadata,
        plan_path=args.plan,
        arm=args.arm,
        output=args.output,
        session_seconds=args.session_seconds,
        reserve_seconds=args.reserve_seconds,
        resume=args.resume,
        external_campaign_tokens=args.external_campaign_tokens,
        mounted_artifact_bytes=args.mounted_artifact_bytes,
        execute=args.execute,
        invocation_started=invocation_started,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
