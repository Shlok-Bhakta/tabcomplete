"""Strict selection and artifact checks for private Q25 FIM Q4 conversion.

This module intentionally uses only the Python standard library. It can reject
an unselected, partial, or mismatched model before a conversion environment or
large intermediate is created.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeGuard

SELECTION_SCHEMA = "q25-fim-conversion-selection-v1"
PLAN_SCHEMA = "q25-fim-training-plan-v1"
EXPORT_SCHEMA = "q25-fim-inference-f16-v1"
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
MODEL_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
# SHA-256 of the pinned tokenizer backend serialized by Transformers 5.17.0
# and tokenizers 0.23.2 from the raw tokenizer.json bound by TOKENIZER_SHA256.
# Canonicalization only converts legacy merge strings to pair arrays and
# defaults omitted ignore_merges to false; all pipeline flags remain exact.
TOKENIZER_SEMANTICS_SHA256 = "d45a2f780107c285e830ea2645d4c7e5e1e03bb6fd975d35487f1a3b1197ad7d"
PARAMETER_COUNT = 494_032_768
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 896
NUM_HIDDEN_LAYERS = 24
QWEN_GGUF_PHYSICAL_PARAMETER_COUNT = PARAMETER_COUNT + VOCAB_SIZE * HIDDEN_SIZE
EOS_TOKEN_ID = 151_643
FIM_MARKER_IDS = {
    "fim_prefix": 151_659,
    "fim_middle": 151_660,
    "fim_suffix": 151_661,
}
FIM_MARKER_TEXT = {
    "fim_prefix": "<|fim_prefix|>",
    "fim_middle": "<|fim_middle|>",
    "fim_suffix": "<|fim_suffix|>",
}
QWEN_CONFIG_SEMANTICS = {
    "architectures": ["Qwen2ForCausalLM"],
    "bos_token_id": EOS_TOKEN_ID,
    "eos_token_id": EOS_TOKEN_ID,
    "hidden_act": "silu",
    "hidden_size": HIDDEN_SIZE,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "max_window_layers": 24,
    "model_type": "qwen2",
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "vocab_size": VOCAB_SIZE,
}
LLAMA_CPP_REVISION = "f072b103714dfa1eee531f80b24512faf38e3dd2"
MAX_ARTIFACT_BYTES = 12 * 1024**3
MINIMUM_FREE_BYTES = 2 * 1024**3
GIB = 1024**3


class ConversionContractError(ValueError):
    """An input does not match the selected FIM export contract."""


@dataclass(frozen=True)
class VerifiedConversionInput:
    selection: dict[str, Any]
    selection_sha256: str
    plan: dict[str, Any]
    plan_sha256: str
    export_manifest: dict[str, Any]
    export_manifest_sha256: str
    source_export: Path
    source_files: dict[str, dict[str, Any]]
    expected_cursor: dict[str, int]
    config: dict[str, Any]
    safetensors: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ConversionContractError(f"{label} is not valid JSON") from None
    if not isinstance(value, dict):
        raise ConversionContractError(f"{label} must be a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ConversionContractError(f"{label} must be a normalized relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ConversionContractError(f"{label} must stay inside the mounted input root")
    if path.as_posix() != value:
        raise ConversionContractError(f"{label} must be a normalized relative path")
    return path


def _path_under(root: Path, value: Any, label: str) -> Path:
    relative = _safe_relative(value, label)
    root_resolved = root.resolve(strict=True)
    path = root.joinpath(*relative.parts)
    current = root_resolved
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ConversionContractError(f"{label} must not contain symbolic links")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise FileNotFoundError(f"{label} is missing from the mounted input bundle") from None
    if not resolved.is_relative_to(root_resolved):
        raise ConversionContractError(f"{label} escapes the mounted input root")
    return resolved


def _safe_file_tree(root: Path) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise ConversionContractError("selected FIM export must be a regular directory")
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ConversionContractError("selected FIM export contains a symbolic link")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _record_map(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ConversionContractError(f"{label} must be a nonempty file map")
    result: dict[str, dict[str, Any]] = {}
    for name, record in value.items():
        _safe_relative(name, f"{label} filename")
        if not isinstance(record, dict):
            raise ConversionContractError(f"{label} entries must contain byte and hash fields")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not _is_sha256(digest)
        ):
            raise ConversionContractError(f"{label} entry is missing a valid byte count or SHA-256")
        result[name] = {"bytes": size, "sha256": digest}
    return result


def _file_sha256(value: Any, name: str) -> Any:
    if not isinstance(value, dict):
        return None
    files = value.get("files")
    if not isinstance(files, dict):
        return None
    record = files.get(name)
    return record.get("sha256") if isinstance(record, dict) else None


def _positive_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _initializer_identity_matches(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    kind = expected.get("kind")
    if kind == "untouched_pretrained":
        identity_fields = {"kind", "model_id", "revision", "files"}
    elif kind == "completed_cpt_export":
        identity_fields = {
            "kind",
            "model_id",
            "revision",
            "files",
            "artifact_manifest_sha256",
            "fingerprint",
            "training_cursor",
        }
    else:
        return False
    return set(actual) == identity_fields and all(
        name in expected and actual[name] == expected[name] for name in identity_fields
    )


def _verify_fim_training_plan(plan: dict[str, Any]) -> tuple[str, dict[str, int]]:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("gpu_execution_authorized") is not True:
        raise ConversionContractError("training plan is not the frozen executable Q25 FIM plan")
    config = plan.get("configuration")
    if not isinstance(config, dict):
        raise ConversionContractError("training plan has no configuration object")
    training = config.get("training")
    data = plan.get("data")
    if not isinstance(training, dict) or not isinstance(data, dict):
        raise ConversionContractError("training plan is missing training or data metadata")
    expected_training = {
        "attention": "sdpa",
        "checkpoint_every_updates": 64,
        "compute": "fp16",
        "cosine_floor_fraction": 0.1,
        "epochs": 1,
        "gradient_checkpointing": True,
        "gradient_clip": 1.0,
        "sequence_length": 1024,
        "effective_batch": 16,
        "initial_loss_scale": 128,
        "learning_rate": 1e-5,
        "master_weights": "fp32",
        "max_input_tokens": 4 * 1024**2,
        "max_output_bytes": 10 * 1024**3,
        "microbatch_examples": 1,
        "objective": "example_mean_response_only_FIM_target_and_EOS",
        "optimizer": "AdamW8bit",
        "same_examples_and_order": True,
        "seed": 314159,
        "warmup_fraction": 0.03,
        "weight_decay": 0.01,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ConversionContractError("training plan is not the frozen one-pass FIM schedule")
    train = data.get("train")
    if not isinstance(train, dict):
        raise ConversionContractError("training plan has no frozen training split")
    row_count = train.get("row_count")
    input_tokens = train.get("input_tokens")
    target_tokens = train.get("target_tokens")
    batch_size = training["effective_batch"]
    if (
        not _positive_int(row_count)
        or not _positive_int(input_tokens)
        or not _positive_int(target_tokens)
        or not _positive_int(batch_size)
    ):
        raise ConversionContractError("training split totals are invalid")
    if target_tokens > input_tokens or input_tokens > training["max_input_tokens"]:
        raise ConversionContractError("training split exceeds its frozen input-token cap")
    arms = training.get("arms")
    initializers = plan.get("initializers")
    if (
        not isinstance(arms, list)
        or any(not isinstance(arm, str) or not arm for arm in arms)
        or len(set(arms)) != len(arms)
        or set(arms) != {"untouched_q25_to_fim", "completed_cpt_q25_to_fim"}
        or not isinstance(initializers, dict)
        or set(initializers) != set(arms)
    ):
        raise ConversionContractError("training plan has no paired FIM arm identities")
    expected_cursor = {
        "attempted_updates": math.ceil(row_count / batch_size),
        "completed_updates": math.ceil(row_count / batch_size),
        "skipped_updates": 0,
        "training_input_tokens": input_tokens,
        "supervised_target_tokens": target_tokens,
        "next_example_index": row_count,
        "epoch": 1,
    }
    return "|".join(arms), expected_cursor


def _verify_config(config: dict[str, Any]) -> None:
    if any(config.get(key) != value for key, value in QWEN_CONFIG_SEMANTICS.items()):
        raise ConversionContractError("selected export config differs from Qwen2.5-Coder semantics")
    rope = config.get("rope_parameters")
    if rope is None:
        rope = {"rope_type": "default", "rope_theta": config.get("rope_theta")}
    if (
        not isinstance(rope, dict)
        or rope.get("rope_type") != "default"
        or rope.get("rope_theta") != 1_000_000.0
        or config.get("rope_scaling") is not None
        or config.get("sliding_window") not in (None, 32768)
        or (
            "layer_types" in config
            and config["layer_types"] != ["full_attention"] * NUM_HIDDEN_LAYERS
        )
    ):
        raise ConversionContractError("selected export does not preserve the pinned RoPE semantics")


def _safetensors_layout(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ConversionContractError("selected safetensors header is truncated")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > 128 * 1024**2:
                raise ConversionContractError("selected safetensors header length is invalid")
            header_raw = handle.read(header_length)
            header = json.loads(header_raw)
    except (OSError, json.JSONDecodeError, UnicodeError, struct.error):
        raise ConversionContractError("selected safetensors header is invalid") from None
    if not isinstance(header, dict):
        raise ConversionContractError("selected safetensors header must be an object")
    data_start = 8 + header_length
    file_size = path.stat().st_size
    tensors: dict[str, dict[str, Any]] = {}
    total_elements = 0
    spans: list[tuple[int, int, str]] = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ConversionContractError("selected safetensors tensor entry is invalid")
        dtype = record.get("dtype")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (
            dtype != "F16"
            or not isinstance(shape, list)
            or not shape
            or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) for v in offsets)
        ):
            raise ConversionContractError(
                "selected safetensors contains a non-F16 or malformed tensor"
            )
        start, end = offsets
        elements = math.prod(shape)
        if start < 0 or end < start or data_start + end > file_size or end - start != elements * 2:
            raise ConversionContractError("selected safetensors tensor offsets are invalid")
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "start": start,
            "end": end,
            "elements": elements,
        }
        total_elements += elements
        spans.append((start, end, name))
    spans.sort()
    for previous, current in zip(spans, spans[1:], strict=False):
        if current[0] < previous[1] and (current[0], current[1]) != (previous[0], previous[1]):
            raise ConversionContractError("selected safetensors tensor data ranges overlap")

    embedding = tensors.get("model.embed_tokens.weight")
    head = tensors.get("lm_head.weight")
    if embedding is None:
        raise ConversionContractError("selected safetensors has no Qwen2 token embedding")
    expected_shape = [VOCAB_SIZE, HIDDEN_SIZE]
    if embedding["shape"] != expected_shape:
        raise ConversionContractError("selected embedding shape differs from Qwen2.5-Coder")
    logical_elements = total_elements
    if head is not None:
        if head["shape"] != expected_shape:
            raise ConversionContractError("selected tied output head shape is invalid")
        if not _file_ranges_equal(
            path,
            data_start + embedding["start"],
            data_start + head["start"],
            embedding["end"] - embedding["start"],
        ):
            raise ConversionContractError("selected tied embedding and output-head bytes differ")
        logical_elements -= head["elements"]
    if logical_elements != PARAMETER_COUNT:
        raise ConversionContractError("selected safetensors logical parameter count is incorrect")
    return {
        "physical_parameter_count": total_elements,
        "logical_parameter_count": logical_elements,
        "tensor_count": len(tensors),
        "tied_output_head_present": head is not None,
    }


def _file_ranges_equal(path: Path, first: int, second: int, length: int) -> bool:
    if length < 0:
        return False
    with path.open("rb") as handle:
        offset = 0
        while offset < length:
            amount = min(8 * 1024**2, length - offset)
            handle.seek(first + offset)
            left = handle.read(amount)
            handle.seek(second + offset)
            right = handle.read(amount)
            if len(left) != amount or left != right:
                return False
            offset += amount
    return True


def verify_selection_bundle(
    *,
    input_root: Path,
    selection_path: Path,
    training_plan_path: Path,
    expected_selection_sha256: str,
    source_root: Path | None = None,
) -> VerifiedConversionInput:
    """Verify a root-selected FIM export under the input or selected-kernel root."""
    root = input_root.resolve(strict=True)
    export_root = root
    if source_root is not None:
        candidate_root = Path(os.path.abspath(source_root))
        if not candidate_root.is_relative_to(root):
            raise ConversionContractError(
                "selected export root must be inside the mounted input root"
            )
        current = root
        for part in candidate_root.relative_to(root).parts:
            current = current / part
            if current.is_symlink():
                raise ConversionContractError(
                    "selected export root must not contain symbolic links"
                )
        if not candidate_root.is_dir():
            raise ConversionContractError(
                "selected export root must be a regular mounted directory"
            )
        export_root = candidate_root.resolve(strict=True)
        if not export_root.is_relative_to(root):
            raise ConversionContractError(
                "selected export root must be inside the mounted input root"
            )
    if selection_path.is_symlink() or training_plan_path.is_symlink():
        raise ConversionContractError("selection and training plan must not be symbolic links")
    selection_file = selection_path.resolve(strict=True)
    plan_file = training_plan_path.resolve(strict=True)
    if not selection_file.is_relative_to(root) or not plan_file.is_relative_to(root):
        raise ConversionContractError(
            "selection and training plan must be inside the mounted input root"
        )
    if not _is_sha256(expected_selection_sha256):
        raise ConversionContractError("expected selection SHA-256 is required")
    selection_sha = sha256_file(selection_file)
    if selection_sha != expected_selection_sha256:
        raise ConversionContractError("selection document does not match its frozen SHA-256")
    selection = _read_json(selection_file, "selection document")
    plan = _read_json(plan_file, "FIM training plan")
    plan_sha = sha256_file(plan_file)
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("status") != "selected_complete"
    ):
        raise ConversionContractError("conversion requires a complete root-selected FIM arm")
    if selection.get("training_plan_sha256") != plan_sha:
        raise ConversionContractError("selected FIM arm is bound to a different training plan")

    allowed_arms, expected_cursor = _verify_fim_training_plan(plan)
    arm = selection.get("selected_arm")
    if not isinstance(arm, str) or arm not in allowed_arms.split("|"):
        raise ConversionContractError("selection names an unknown FIM arm")
    initializers = plan.get("initializers")
    if not isinstance(initializers, dict) or arm not in initializers:
        raise ConversionContractError("selected FIM arm is absent from the training plan")

    tokenizer = selection.get("tokenizer")
    expected_tokenizer = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "sha256": TOKENIZER_SHA256,
        "eos_token_id": EOS_TOKEN_ID,
        "fim_marker_ids": FIM_MARKER_IDS,
    }
    if tokenizer != expected_tokenizer:
        raise ConversionContractError(
            "selection tokenizer identity or FIM token IDs differ from Qwen2.5"
        )
    model = selection.get("model")
    expected_model = {
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "logical_parameter_count": PARAMETER_COUNT,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0},
    }
    if model != expected_model:
        raise ConversionContractError("selection model architecture or RoPE identity is invalid")

    source = selection.get("source_export")
    if not isinstance(source, dict):
        raise ConversionContractError("selection has no source FIM export binding")
    source_path = _path_under(export_root, source.get("directory"), "source export directory")
    export_manifest_path = source_path / "artifact_manifest.json"
    expected_manifest_sha = source.get("artifact_manifest_sha256")
    if (
        not _is_sha256(expected_manifest_sha)
        or not export_manifest_path.is_file()
        or sha256_file(export_manifest_path) != expected_manifest_sha
    ):
        raise ConversionContractError(
            "selected FIM export manifest hash differs from the selection"
        )
    export_manifest = _read_json(export_manifest_path, "FIM export manifest")
    initializers = plan.get("initializers")
    if not isinstance(initializers, dict):
        raise ConversionContractError("training plan has no frozen initializers")
    untouched_initializer = initializers.get("untouched_q25_to_fim")
    selected_initializer = initializers.get(arm)
    base_provenance = export_manifest.get("base_snapshot_provenance")
    if (
        export_manifest.get("schema") != EXPORT_SCHEMA
        or export_manifest.get("arm") != arm
        or export_manifest.get("base_model") != MODEL_ID
        or export_manifest.get("base_revision") != MODEL_REVISION
        or export_manifest.get("fingerprint") != source.get("fingerprint")
        or not _is_sha256(source.get("fingerprint"))
        or export_manifest.get("training_cursor") != expected_cursor
        or source.get("training_cursor") != expected_cursor
        or not _initializer_identity_matches(
            export_manifest.get("initializer"), selected_initializer
        )
        or not isinstance(untouched_initializer, dict)
        or not isinstance(base_provenance, dict)
        or base_provenance.get("config_sha256")
        != _file_sha256(untouched_initializer, "config.json")
        or base_provenance.get("model_weights_sha256")
        != _file_sha256(untouched_initializer, "model.safetensors")
        or base_provenance.get("tokenizer_sha256")
        != _file_sha256(untouched_initializer, "tokenizer.json")
    ):
        raise ConversionContractError("selected FIM export is partial or mismatched")
    if base_provenance.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ConversionContractError(
            "selected FIM export lost its immutable Qwen tokenizer provenance"
        )
    files = _record_map(export_manifest.get("files"), "FIM export manifest")
    selected_files = _record_map(source.get("files"), "selection export binding")
    if selected_files != files:
        raise ConversionContractError("selection file map differs from the FIM export manifest")
    if _safe_file_tree(source_path) != set(files) | {"artifact_manifest.json"}:
        raise ConversionContractError("selected FIM export contains unmanifested files")
    for name, record in files.items():
        file_path = source_path.joinpath(*PurePosixPath(name).parts)
        if not file_path.is_file() or file_path.stat().st_size != record["bytes"]:
            raise ConversionContractError("selected FIM export file size differs from its manifest")
        if sha256_file(file_path) != record["sha256"]:
            raise ConversionContractError("selected FIM export file hash differs from its manifest")

    config_path = source_path / "config.json"
    weights_path = source_path / "model.safetensors"
    if "config.json" not in files or "model.safetensors" not in files:
        raise ConversionContractError("selected export lacks the local Qwen config or weights")
    config = _read_json(config_path, "selected Qwen config")
    _verify_config(config)
    safetensors = _safetensors_layout(weights_path)
    return VerifiedConversionInput(
        selection=selection,
        selection_sha256=selection_sha,
        plan=plan,
        plan_sha256=plan_sha,
        export_manifest=export_manifest,
        export_manifest_sha256=expected_manifest_sha,
        source_export=source_path,
        source_files=files,
        expected_cursor=expected_cursor,
        config=config,
        safetensors=safetensors,
    )


def verify_local_hf_tokenizer(
    source_export: Path, tokenizer_identity: dict[str, Any]
) -> dict[str, Any]:
    """Check saved tokenizer semantics without resolving anything from a hub."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            source_export,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception:
        raise ConversionContractError(
            "selected local tokenizer could not be loaded offline"
        ) from None
    return validate_tokenizer_semantics(
        tokenizer,
        tokenizer_identity,
        tokenizer_json_path=source_export / "tokenizer.json",
    )


def _canonical_tokenizer_semantics_sha256(value: Any) -> str:
    """Hash a tokenizer mapping after only pinned serialization normalization."""
    if not isinstance(value, dict):
        raise ConversionContractError("selected tokenizer mapping must be a JSON object")
    model = value.get("model")
    if not isinstance(model, dict):
        raise ConversionContractError("selected tokenizer.json has no model mapping")
    merges = model.get("merges")
    if not isinstance(merges, list):
        raise ConversionContractError("selected tokenizer.json has invalid BPE merges")
    normalized_merges: list[list[str]] = []
    for merge in merges:
        if isinstance(merge, str):
            left, separator, right = merge.partition(" ")
            if not separator or not left or not right:
                raise ConversionContractError("selected tokenizer.json has invalid BPE merges")
            normalized_merges.append([left, right])
        elif (
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(token, str) and token for token in merge)
        ):
            normalized_merges.append([merge[0], merge[1]])
        else:
            raise ConversionContractError("selected tokenizer.json has invalid BPE merges")
    normalized_model = dict(model)
    normalized_model["merges"] = normalized_merges
    if "ignore_merges" not in normalized_model:
        normalized_model["ignore_merges"] = False
    value["model"] = normalized_model
    try:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeError):
        raise ConversionContractError("selected tokenizer.json is not canonicalizable") from None
    return hashlib.sha256(canonical).hexdigest()


def _tokenizer_json_semantics_sha256(path: Path) -> str:
    """Hash the complete saved tokenizer mapping after pinned normalization."""
    return _canonical_tokenizer_semantics_sha256(_read_json(path, "selected tokenizer.json"))


def validate_tokenizer_semantics(
    tokenizer: Any,
    tokenizer_identity: dict[str, Any],
    *,
    tokenizer_json_path: Path,
) -> dict[str, Any]:
    tokenizer_semantics_sha256 = _tokenizer_json_semantics_sha256(tokenizer_json_path)
    if tokenizer_semantics_sha256 != TOKENIZER_SEMANTICS_SHA256:
        raise ConversionContractError(
            "selected tokenizer.json mapping differs from the frozen Qwen tokenizer"
        )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serializer = getattr(backend, "to_str", None)
    if not callable(serializer):
        raise ConversionContractError("selected tokenizer has no serializable backend mapping")
    try:
        backend_mapping = json.loads(serializer())
    except (TypeError, UnicodeError, json.JSONDecodeError):
        raise ConversionContractError("selected tokenizer backend mapping is invalid") from None
    if _canonical_tokenizer_semantics_sha256(backend_mapping) != tokenizer_semantics_sha256:
        raise ConversionContractError(
            "selected tokenizer backend differs from its saved tokenizer.json mapping"
        )
    if tokenizer.eos_token_id != EOS_TOKEN_ID:
        raise ConversionContractError("selected tokenizer EOS ID differs from the frozen Qwen ID")
    checks: dict[str, Any] = {
        "eos_token_id": int(tokenizer.eos_token_id),
        "tokenizer_json_semantics_sha256": tokenizer_semantics_sha256,
    }
    for name, token_id in FIM_MARKER_IDS.items():
        token = FIM_MARKER_TEXT[name]
        if tokenizer.convert_tokens_to_ids(token) != token_id:
            raise ConversionContractError(
                "selected tokenizer FIM marker ID differs from the frozen ID"
            )
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ConversionContractError(
                "selected tokenizer does not preserve a FIM marker as one token"
            )
        added = getattr(tokenizer, "added_tokens_decoder", {}).get(token_id)
        if (
            added is None
            or getattr(added, "content", None) != token
            or getattr(added, "special", None) is not False
        ):
            raise ConversionContractError(
                "selected tokenizer does not preserve frozen FIM added-token metadata"
            )
        checks[name] = token_id
    if tokenizer_identity.get("sha256") != TOKENIZER_SHA256:
        raise ConversionContractError(
            "selected tokenizer base identity differs from the frozen Qwen hash"
        )
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "sha256": TOKENIZER_SHA256, **checks}


def _field_value(field: Any) -> Any:
    return field.contents()


def read_gguf_inspection(
    gguf_path: Path, llama_cpp_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read selected metadata and tensor shapes through the pinned llama.cpp reader."""
    package_root = llama_cpp_root / "gguf-py"
    if not (package_root / "gguf" / "gguf_reader.py").is_file():
        raise ConversionContractError("pinned llama.cpp GGUF reader is missing")
    import sys

    sys.path.insert(0, str(package_root))
    try:
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path, "r")
        metadata = {key: _field_value(field) for key, field in reader.fields.items()}
        tensors = [
            {
                "name": item.name,
                "elements": int(item.n_elements),
                "shape": [int(d) for d in item.shape],
            }
            for item in reader.tensors
        ]
    except Exception:
        raise ConversionContractError(
            "converted GGUF could not be read by the pinned parser"
        ) from None
    finally:
        sys.path.remove(str(package_root))
    return metadata, tensors


def validate_qwen_gguf(
    metadata: dict[str, Any],
    tensors: Iterable[dict[str, Any]],
    *,
    expected_file_type: int,
    precision_name: str,
) -> dict[str, Any]:
    """Reject GGUFs with wrong precision, RoPE, tokenizer controls, or shapes."""
    expected_fields = {
        "general.architecture": "qwen2",
        "qwen2.context_length": 32768,
        "qwen2.embedding_length": HIDDEN_SIZE,
        "qwen2.feed_forward_length": 4864,
        "qwen2.block_count": 24,
        "qwen2.attention.head_count": 14,
        "qwen2.attention.head_count_kv": 2,
        "qwen2.rope.freq_base": 1_000_000.0,
        "tokenizer.ggml.eos_token_id": EOS_TOKEN_ID,
        "general.file_type": expected_file_type,
    }
    if any(metadata.get(key) != value for key, value in expected_fields.items()):
        raise ConversionContractError(
            "Q4 GGUF metadata differs from the selected Qwen2/FIM identity"
        )
    token_list = metadata.get("tokenizer.ggml.tokens")
    token_types = metadata.get("tokenizer.ggml.token_type")
    if not isinstance(token_list, list) or len(token_list) != VOCAB_SIZE:
        raise ConversionContractError("Q4 GGUF vocabulary size differs from the selected tokenizer")
    if not isinstance(token_types, list) or len(token_types) != VOCAB_SIZE:
        raise ConversionContractError("Q4 GGUF token-type array is missing or misaligned")
    if token_list[EOS_TOKEN_ID] != "<|endoftext|>":
        raise ConversionContractError("Q4 GGUF EOS vocabulary entry differs from Qwen2.5")
    for name, token_id in FIM_MARKER_IDS.items():
        if token_list[token_id] != FIM_MARKER_TEXT[name] or token_types[token_id] != 3:
            raise ConversionContractError("Q4 GGUF FIM control token string or type is incorrect")
        metadata_id_key = {
            "fim_prefix": "tokenizer.ggml.fim_pre_token_id",
            "fim_middle": "tokenizer.ggml.fim_mid_token_id",
            "fim_suffix": "tokenizer.ggml.fim_suf_token_id",
        }[name]
        if metadata_id_key in metadata and metadata[metadata_id_key] != token_id:
            raise ConversionContractError("Q4 GGUF FIM token metadata ID is incorrect")
    tensor_list = list(tensors)
    by_name = {item.get("name"): item for item in tensor_list}
    embedding = by_name.get("token_embd.weight")
    output = by_name.get("output.weight")
    tied_elements = VOCAB_SIZE * HIDDEN_SIZE
    if embedding is None or embedding.get("elements") != tied_elements:
        raise ConversionContractError("Q4 GGUF token embedding shape is incorrect")
    if output is not None and output.get("elements") != tied_elements:
        raise ConversionContractError("Q4 GGUF tied output-head shape is incorrect")
    logical_parameters = sum(int(item["elements"]) for item in tensor_list)
    if output is not None:
        logical_parameters -= tied_elements
    if logical_parameters != PARAMETER_COUNT:
        raise ConversionContractError(
            "Q4 GGUF tensor inventory has the wrong logical parameter count"
        )
    return {
        "architecture": "qwen2",
        "file_type": precision_name,
        "rope_theta": 1_000_000.0,
        "eos_token_id": EOS_TOKEN_ID,
        "fim_marker_ids": FIM_MARKER_IDS,
        "physical_parameter_count": sum(int(item["elements"]) for item in tensor_list),
        "logical_parameter_count": logical_parameters,
        "tensor_count": len(tensor_list),
    }


def validate_q4_gguf(metadata: dict[str, Any], tensors: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return validate_qwen_gguf(
        metadata,
        tensors,
        expected_file_type=15,
        precision_name="Q4_K_M",
    )


def projected_q4_output_bytes(
    *,
    f16_gguf_bytes: int,
    metadata_allowance_bytes: int = 128 * 1024**2,
) -> int:
    if (
        not _positive_int(f16_gguf_bytes)
        or not isinstance(metadata_allowance_bytes, int)
        or isinstance(metadata_allowance_bytes, bool)
        or metadata_allowance_bytes < 0
    ):
        raise ConversionContractError("Q4 output size estimate inputs are invalid")
    return f16_gguf_bytes + metadata_allowance_bytes


def projected_f16_gguf_bytes(
    *,
    source_safetensors_bytes: int,
    physical_parameter_count: int,
    metadata_allowance_bytes: int = 64 * 1024**2,
) -> int:
    if (
        not _positive_int(source_safetensors_bytes)
        or not _positive_int(physical_parameter_count)
        or physical_parameter_count < PARAMETER_COUNT
        or metadata_allowance_bytes < 0
    ):
        raise ConversionContractError("F16 output size estimate inputs are invalid")
    return max(source_safetensors_bytes, physical_parameter_count * 2) + metadata_allowance_bytes


def accounted_bytes(roots: Iterable[Path]) -> int:
    """Count unique file paths across input, runtime, and output trees."""
    seen: set[Path] = set()
    total = 0
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths: Iterable[Path] = (root,)
        else:
            paths = (path for path in root.rglob("*") if path.is_file())
        for path in paths:
            try:
                identity = path.resolve(strict=True)
                if identity in seen:
                    continue
                seen.add(identity)
                total += identity.stat().st_size
            except OSError:
                raise ConversionContractError("unable to account for conversion storage") from None
    return total


def preflight_storage(
    *,
    roots: Iterable[Path],
    additional_peak_bytes: int,
    free_bytes: int,
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, int]:
    """Require both the 12 GiB aggregate cap and independent free-space reserve."""
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (additional_peak_bytes, free_bytes, max_artifact_bytes, minimum_free_bytes)
    ):
        raise ConversionContractError("storage budget values must be nonnegative byte counts")
    current = accounted_bytes(roots)
    projected_peak = current + additional_peak_bytes
    if projected_peak > max_artifact_bytes:
        raise ConversionContractError("conversion peak would exceed the aggregate artifact cap")
    if free_bytes < additional_peak_bytes + minimum_free_bytes:
        raise ConversionContractError("conversion peak would violate the free-space reserve")
    return {
        "current_accounted_bytes": current,
        "projected_peak_bytes": projected_peak,
        "additional_peak_bytes": additional_peak_bytes,
        "minimum_free_bytes": minimum_free_bytes,
        "max_artifact_bytes": max_artifact_bytes,
    }


def projected_conversion_peak_bytes(
    *,
    source_safetensors_bytes: int,
    physical_parameter_count: int,
    converter_log_bytes: int = 64 * 1024**2,
) -> int:
    """Conservative temporary F16 GGUF + Q4 + log allowance."""
    if source_safetensors_bytes <= 0 or converter_log_bytes < 0:
        raise ConversionContractError("conversion size estimate inputs are invalid")
    f16_gguf = projected_f16_gguf_bytes(
        source_safetensors_bytes=source_safetensors_bytes,
        physical_parameter_count=physical_parameter_count,
    )
    q4_gguf = projected_q4_output_bytes(
        f16_gguf_bytes=f16_gguf,
    )
    return f16_gguf + q4_gguf + converter_log_bytes
