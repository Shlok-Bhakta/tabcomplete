"""Local-only Qwen2.5 raw-code continued pretraining for the Kaggle worker.

This module consumes frozen token-ID blocks. It does not fetch models or data;
the controller owns those artifacts, the campaign plan, and GPU allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tinycomplete.observability.runs import run_scope
from tinycomplete.observability.spans import operation
from tinycomplete.one_line.train import (
    CosineUpdateSchedule,
    EncodedExample,
    TrainingCursor,
    batch_order_sha256,
    bucketed_batches,
    load_resume_checkpoint,
    save_resume_checkpoint,
    train_encoded,
)

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
MODEL_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
MODEL_WEIGHT_SHA256 = "aff8914ec707fcaf9e2d4dc97197cded50b1c63e1d3a7a82e56f54d83ea47f80"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
CONFIG_SHA256 = "e6bf24d1cf58278dcb4ded7e885b71cb1c56299b34c8046b1b64395d17b1891f"
PARAMETER_COUNT = 494_032_768
MAX_CAMPAIGN_INPUT_TOKENS = 12_000_000
SOURCE_NLL_BLOCKS = 128
CHECKPOINT_PREFIX = "resume-step-"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def plan_configuration(document: dict[str, Any]) -> dict[str, Any]:
    configuration = document.get("configuration", document)
    if not isinstance(configuration, dict):
        raise ValueError("plan configuration must be an object")
    return configuration


def model_snapshot_identity(model_path: Path, document: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen original Q25 files without consulting the model hub."""
    configuration = plan_configuration(document)
    model_plan = configuration.get("model", {})
    if (
        model_plan.get("id") != MODEL_ID
        or model_plan.get("revision") != MODEL_REVISION
        or model_plan.get("initializer") != "untouched_pretrained"
    ):
        raise ValueError("plan does not identify the approved untouched Q25 initializer")

    paths = {
        "config.json": model_path / "config.json",
        "model.safetensors": model_path / "model.safetensors",
        "tokenizer.json": model_path / "tokenizer.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("local Q25 snapshot is missing a required model file")

    expected = document.get("existing_model_files", {})
    actual: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        digest = sha256_file(path)
        expected_record = expected.get(name, {})
        expected_digest = expected_record.get("sha256")
        if expected_digest and digest != expected_digest:
            raise ValueError(f"local Q25 {name} hash differs from the frozen input manifest")
        actual[name] = {"bytes": path.stat().st_size, "sha256": digest}

    if actual["config.json"]["sha256"] != CONFIG_SHA256:
        raise ValueError("local Q25 config does not match the pinned revision")
    if actual["model.safetensors"]["sha256"] != MODEL_WEIGHT_SHA256:
        raise ValueError("local Q25 weights do not match the pinned revision")
    if actual["tokenizer.json"]["sha256"] != TOKENIZER_SHA256:
        raise ValueError("local Q25 tokenizer does not match the pinned revision")

    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2":
        raise ValueError("local model architecture is not Qwen2")
    if "Qwen2ForCausalLM" not in config.get("architectures", []):
        raise ValueError("local model does not declare Qwen2ForCausalLM")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "initializer": "untouched_pretrained",
        "parameter_count": PARAMETER_COUNT,
        "files": actual,
    }


def load_token_blocks(path: Path, *, sequence_length: int) -> np.ndarray:
    blocks = np.load(path, mmap_mode="r", allow_pickle=False)
    if blocks.ndim != 2 or blocks.shape[1] != sequence_length:
        raise ValueError("token block array must have shape [blocks, sequence_length]")
    if blocks.shape[0] == 0 or blocks.dtype.kind not in "iu":
        raise ValueError("token block array must contain integer token IDs")
    if blocks.dtype.kind == "i" and int(blocks.min()) < 0:
        raise ValueError("token block array contains a negative token ID")
    return blocks


def encode_raw_code_blocks(
    blocks: np.ndarray, *, sequence_length: int, vocab_size: int
) -> tuple[EncodedExample, ...]:
    """Encode each fixed raw-code block with every causal source token active."""
    if blocks.ndim != 2 or blocks.shape[1] != sequence_length or len(blocks) == 0:
        raise ValueError("raw-code blocks must be a nonempty fixed-width matrix")
    if blocks.dtype.kind not in "iu":
        raise ValueError("raw-code blocks must contain integer token IDs")
    if blocks.dtype.kind == "i" and int(blocks.min()) < 0:
        raise ValueError("raw-code blocks contain a negative token ID")
    if int(blocks.max()) >= vocab_size:
        raise ValueError("raw-code block contains a token outside the model vocabulary")
    examples = []
    for row in blocks:
        tokens = tuple(int(token) for token in row)
        examples.append(
            EncodedExample(
                input_ids=tokens,
                labels=tokens,
                prompt_tokens=0,
                response_tokens=sequence_length - 1,
                total_tokens=sequence_length,
                prompt="",
                response="",
            )
        )
    return tuple(examples)


def training_batches(
    examples: tuple[EncodedExample, ...], *, effective_batch: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    return bucketed_batches(
        examples, epochs=1, effective_batch=effective_batch, seed=seed
    )


def validate_campaign_token_budget(
    *, logical_training_tokens: int, external_campaign_tokens: int, maximum_additional_tokens: int
) -> int:
    if min(logical_training_tokens, external_campaign_tokens) < 0 or maximum_additional_tokens < 1:
        raise ValueError("campaign token counts and cap must be nonnegative")
    actual = logical_training_tokens + external_campaign_tokens
    if actual > maximum_additional_tokens:
        raise ValueError("external replay tokens plus logical training exceed the campaign cap")
    return actual


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_identity(device: Any) -> dict[str, Any]:
    import torch
    import transformers

    from tinycomplete.code_cpt import eval as causal_eval

    def module_path(module: Any) -> Path:
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            raise RuntimeError("cannot fingerprint a training source module")
        return Path(filename)

    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        capability = list(torch.cuda.get_device_capability(device))
        properties = torch.cuda.get_device_properties(device)
        device_memory = int(properties.total_memory)
        multiprocessors = int(properties.multi_processor_count)
    else:
        name = "cpu"
        capability = None
        device_memory = None
        multiprocessors = None
    source_files = {
        "q25_training": Path(__file__),
        "one_line_training": module_path(sys.modules[train_encoded.__module__]),
        "causal_loss": module_path(causal_eval),
    }
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "bitsandbytes": _version("bitsandbytes"),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": name,
        "device_capability": capability,
        "device_memory_bytes": device_memory,
        "device_multiprocessors": multiprocessors,
        "world_size": 1,
        "source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
    }


def training_identity(
    *,
    document: dict[str, Any],
    plan_sha256: str,
    model_identity: dict[str, Any],
    train_path: Path,
    development_path: Path,
    train_blocks: np.ndarray,
    development_blocks: np.ndarray,
    batches: tuple[tuple[int, ...], ...],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    configuration = plan_configuration(document)
    training = configuration["training"]
    sequence_length = int(training["sequence_length"])
    training_count = min(
        len(train_blocks), int(training["max_input_tokens"]) // sequence_length
    )
    selected_examples = training_count * sequence_length
    return {
        "schema": "q25-raw-code-cpt-resume-v1",
        "plan_sha256": plan_sha256,
        "model": model_identity,
        "training_data": {
            "sha256": sha256_file(train_path),
            "shape": list(train_blocks.shape),
            "dtype": str(train_blocks.dtype),
            "selected_blocks": training_count,
            "selected_input_tokens": selected_examples,
            "batch_order_sha256": batch_order_sha256(batches),
            "batch_count": len(batches),
        },
        "development_data": {
            "sha256": sha256_file(development_path),
            "shape": list(development_blocks.shape),
            "dtype": str(development_blocks.dtype),
            "scored_blocks": min(len(development_blocks), SOURCE_NLL_BLOCKS),
        },
        "training": training,
        "runtime": runtime,
    }


def estimate_checkpoint_bytes(model: Any, optimizer: Any, source_weight_bytes: int) -> int:
    """Conservatively estimate one atomic model+AdamW8bit checkpoint write."""
    import torch

    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.is_floating_point()
    )
    optimizer_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                optimizer_bytes += value.numel() * value.element_size()
    # The pinned initializer is BF16 on disk; four times that size covers FP32
    # masters plus 8-bit first and second moments and leaves serialization slack.
    return max(model_bytes + optimizer_bytes, source_weight_bytes * 4) + 64 * 1024**2


def _storage_preflight(
    *,
    output: Path,
    predicted_write_bytes: int,
    output_cap_bytes: int,
    total_cap_bytes: int,
    input_artifact_bytes: int,
    minimum_free_bytes: int,
) -> None:
    current_output_bytes = directory_bytes(output)
    if current_output_bytes + predicted_write_bytes > output_cap_bytes:
        raise OSError("Q25-owned output would exceed its declared storage cap")
    if input_artifact_bytes + current_output_bytes + predicted_write_bytes > total_cap_bytes:
        raise OSError("Q25 inputs and output would exceed the aggregate artifact cap")
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < predicted_write_bytes + minimum_free_bytes:
        raise OSError("insufficient free space plus the required checkpoint headroom")


def _checkpoint_from_pointer(pointer_path: Path) -> Path:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    value = pointer.get("path") or pointer.get("checkpoint")
    if not isinstance(value, str) or Path(value).name != value:
        raise ValueError("latest checkpoint pointer is invalid")
    checkpoint = pointer_path.parent / value
    if not checkpoint.is_file():
        raise FileNotFoundError("latest checkpoint file is missing")
    return checkpoint


def resolve_resume_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "latest.json"
    if path.name == "latest.json":
        return _checkpoint_from_pointer(path)
    if not path.is_file():
        raise FileNotFoundError("resume checkpoint does not exist")
    return path


def load_training_cursor(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    fingerprint: str,
) -> TrainingCursor:
    checkpoint = resolve_resume_path(path)
    with operation("checkpoint.load"):
        state = load_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprint=fingerprint,
        )
    return TrainingCursor(**state)


def save_training_cursor(
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
    """Commit one update-boundary checkpoint before replacing the latest pointer."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{CHECKPOINT_PREFIX}{cursor.attempted_updates:06d}.pt"
    marker_path = checkpoint.with_suffix(checkpoint.suffix + ".complete.json")
    latest_path = output / "latest.json"
    old_checkpoint: Path | None = None
    if latest_path.exists():
        old_checkpoint = resolve_resume_path(latest_path)
        old_pointer = json.loads(latest_path.read_text(encoding="utf-8"))
        if (
            old_pointer.get("fingerprint") == fingerprint
            and old_pointer.get("cursor") == asdict(cursor)
        ):
            return old_checkpoint
    recovered = False
    if checkpoint.exists() or marker_path.exists():
        if not marker_path.exists() or not checkpoint.exists():
            checkpoint.unlink(missing_ok=True)
            marker_path.unlink(missing_ok=True)
        else:
            with operation("checkpoint.load"):
                recovered_state = load_resume_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    expected_fingerprint=fingerprint,
                )
            if recovered_state != asdict(cursor):
                raise ValueError("orphan checkpoint cursor differs from the requested update")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            recovered = True

    if not recovered:
        predicted = estimate_checkpoint_bytes(model, optimizer, source_weight_bytes)
        _storage_preflight(
            output=output,
            predicted_write_bytes=predicted,
            output_cap_bytes=output_cap_bytes,
            total_cap_bytes=total_cap_bytes,
            input_artifact_bytes=input_artifact_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        with operation(
            "checkpoint.save",
            attributes={
                "tabcomplete.training.attempted_updates": cursor.attempted_updates,
                "tabcomplete.training.input_tokens": cursor.training_input_tokens,
            },
        ):
            marker = save_resume_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                fingerprint=fingerprint,
                **asdict(cursor),
            )
        actual_output_bytes = directory_bytes(output)
        if (
            actual_output_bytes > output_cap_bytes
            or input_artifact_bytes + actual_output_bytes > total_cap_bytes
        ):
            checkpoint.unlink(missing_ok=True)
            marker_path.unlink(missing_ok=True)
            raise OSError("completed resume checkpoint exceeds the declared storage cap")
    pointer = {
        "schema": "q25-cpt-latest-checkpoint-v1",
        "path": checkpoint.name,
        "checkpoint": checkpoint.name,
        "sha256": marker["sha256"],
        "fingerprint": fingerprint,
        "cursor": asdict(cursor),
    }
    atomic_json(latest_path, pointer)
    if (
        old_checkpoint is not None
        and old_checkpoint.parent == output
        and old_checkpoint != checkpoint
    ):
        old_checkpoint.unlink(missing_ok=True)
        old_checkpoint.with_suffix(old_checkpoint.suffix + ".complete.json").unlink(
            missing_ok=True
        )
    for candidate in output.glob(f"{CHECKPOINT_PREFIX}*.pt"):
        if candidate != checkpoint and candidate != old_checkpoint:
            candidate.unlink(missing_ok=True)
            candidate.with_suffix(candidate.suffix + ".complete.json").unlink(
                missing_ok=True
            )
    actual_output_bytes = directory_bytes(output)
    if (
        actual_output_bytes > output_cap_bytes
        or input_artifact_bytes + actual_output_bytes > total_cap_bytes
    ):
        raise OSError("committed resume checkpoint exceeds the declared storage cap")
    return checkpoint


def evaluate_source_nll(
    model: Any,
    blocks: np.ndarray,
    *,
    device: Any,
    max_blocks: int = SOURCE_NLL_BLOCKS,
) -> dict[str, Any]:
    """Score exact next-token NLL on a fixed prefix of the same-tokenizer blocks."""
    import torch

    from tinycomplete.code_cpt.eval import causal_nll_from_logits

    if blocks.ndim != 2 or len(blocks) == 0 or max_blocks < 1:
        raise ValueError("development token blocks are empty or malformed")
    selected = min(len(blocks), max_blocks)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    try:
        with operation("model.score"):
            with torch.inference_mode():
                for index in range(selected):
                    ids = torch.as_tensor(
                        np.array(blocks[index : index + 1], dtype=np.int64, copy=True),
                        device=device,
                    )
                    precision = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with precision:
                        output = model(input_ids=ids, use_cache=False)
                    nll, count = causal_nll_from_logits(output.logits, ids)
                    total_nll += float(nll.detach().float().item())
                    total_tokens += count
                    del output, ids, nll
    finally:
        model.train(was_training)
    nll_value = total_nll / float(total_tokens)
    result = {
        "metric": "causal_next_token_nll",
        "nll": nll_value,
        "nll_sum": total_nll,
        "scored_tokens": total_tokens,
        "blocks": selected,
        "sequence_length": int(blocks.shape[1]),
        "tokenizer_model": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
    }
    from tinycomplete.observability.metrics import record_metric

    record_metric(
        "tabcomplete.training.validation_nll",
        nll_value,
        {"rank_role": "authoritative"},
        kind="gauge",
    )
    return result


def save_inference_export(
    model: Any,
    tokenizer: Any,
    destination: Path,
    *,
    output_root: Path,
    output_cap_bytes: int,
    total_cap_bytes: int,
    input_artifact_bytes: int,
    minimum_free_bytes: int,
    source_weight_bytes: int,
    fingerprint: str,
    cursor: TrainingCursor,
) -> dict[str, Any]:
    """Atomically export standard F16 HF weights for evaluation and later SFT."""
    import torch

    backup = destination.with_name(destination.name + ".previous")
    if not destination.exists() and backup.exists():
        os.replace(backup, destination)
    if destination.exists():
        manifest_path = destination / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError("inference export path already exists without a manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("existing inference export belongs to a different training run")
        if manifest.get("training_cursor") == asdict(cursor):
            if backup.exists():
                shutil.rmtree(backup)
            return {"path": str(destination), "files": manifest.get("files", {})}
    predicted = source_weight_bytes * 2 + 64 * 1024**2
    _storage_preflight(
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
            atomic_json(
                temporary / "artifact_manifest.json",
                {
                    "schema": "q25-cpt-inference-f16-v1",
                    "fingerprint": fingerprint,
                    "base_model": MODEL_ID,
                    "base_revision": MODEL_REVISION,
                    "training_cursor": asdict(cursor),
                    "precision": "F16 export from FP32 master parameters",
                    "files": files,
                },
            )
        actual_output_bytes = directory_bytes(output_root)
        if (
            actual_output_bytes > output_cap_bytes
            or input_artifact_bytes + actual_output_bytes > total_cap_bytes
        ):
            raise OSError("inference export exceeds the declared storage cap")
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


def _load_model(model_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if model.config.model_type != "qwen2":
        raise ValueError("loaded model architecture is not Qwen2")
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("loaded Q25 parameter count differs from the frozen revision")
    model.float()
    if any(
        parameter.dtype != torch.float32
        for parameter in model.parameters()
        if parameter.is_floating_point()
    ):
        raise TypeError("FP32 master parameters are required")
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    if not isinstance(tokenizer.eos_token_id, int):
        raise ValueError("pinned Q25 tokenizer has no EOS token")
    return model, tokenizer


def _make_optimizer(model: Any, training: dict[str, Any]) -> tuple[Any, str]:
    try:
        from bitsandbytes.optim import AdamW8bit
    except (ImportError, OSError) as exc:
        raise RuntimeError("the pinned bitsandbytes AdamW8bit optimizer is unavailable") from exc
    optimizer = AdamW8bit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    return optimizer, "bitsandbytes.AdamW8bit"


def validate_training_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    training = configuration.get("training")
    if not isinstance(training, dict):
        raise ValueError("plan training configuration is missing")
    expected = {
        "epochs": 1,
        "master_weights": "fp32",
        "compute": "fp16",
        "optimizer": "AdamW8bit",
        "attention": "sdpa",
        "loss": "example_mean_causal_all_source_positions_except_first",
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("plan differs from the frozen raw-code CPT objective or runtime")
    sequence_length = int(training["sequence_length"])
    if sequence_length not in (1024, 2048):
        raise ValueError("Q25 raw-code sequence length must be 1024 or 2048")
    if int(training["max_input_tokens"]) > MAX_CAMPAIGN_INPUT_TOKENS:
        raise ValueError("Q25 raw-code token budget exceeds the campaign cap")
    if int(training["effective_batch"]) < 1 or int(training["microbatch_examples"]) < 1:
        raise ValueError("Q25 batch sizes must be positive")
    if int(training["microbatch_examples"]) > int(training["effective_batch"]):
        raise ValueError("Q25 microbatch cannot exceed its effective batch")
    if int(training["checkpoint_every_updates"]) < 1:
        raise ValueError("periodic complete-checkpoint saving is required")
    if float(training["learning_rate"]) <= 0 or not 0 < float(
        training["cosine_floor_fraction"]
    ) <= 1:
        raise ValueError("invalid Q25 learning-rate schedule")
    if float(training.get("initial_loss_scale", 128)) != 128.0:
        raise ValueError("Q25 T4 training requires the frozen initial loss scale of 128")
    return training


def _runtime_training(run_config: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("execution requires an explicitly allocated CUDA worker")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Q25 CPT worker requires exactly one active GPU")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        raise RuntimeError("Q25 CPT worker requires a free Kaggle T4")
    return {
        "initial_loss_scale": float(run_config["initial_loss_scale"]),
        "gradient_checkpointing": bool(run_config["gradient_checkpointing"]),
        "compute": "fp16",
        "master_weights": "fp32",
        "attention": "sdpa",
    }


def run_training(
    *,
    model_path: Path,
    train_path: Path,
    development_path: Path,
    plan_path: Path,
    output: Path,
    session_seconds: float,
    reserve_seconds: float | None = None,
    resume: Path | None = None,
    external_campaign_tokens: int = 0,
    execute: bool = False,
    invocation_started: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic() if invocation_started is None else invocation_started
    if session_seconds <= 0:
        raise ValueError("session-seconds must be positive")
    deadline = started + session_seconds
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    configuration = plan_configuration(document)
    training = validate_training_configuration(configuration)
    budget = configuration.get("budget", {})
    if external_campaign_tokens < 0:
        raise ValueError("external campaign token count cannot be negative")
    reserve = (
        float(budget.get("finalization_reserve_seconds", 1200))
        if reserve_seconds is None
        else float(reserve_seconds)
    )
    if reserve < max(1200, float(budget.get("finalization_reserve_seconds", 1200))):
        raise ValueError("Q25 finalization reserve is below the frozen minimum")
    if session_seconds > float(budget.get("session_seconds", session_seconds)):
        raise ValueError("Q25 session duration exceeds the frozen maximum")
    if int(training["sequence_length"]) != int(configuration["data"]["sequence_length"]):
        raise ValueError("training and frozen data sequence lengths differ")

    model_identity = model_snapshot_identity(model_path, document)
    train_blocks = load_token_blocks(
        train_path, sequence_length=int(training["sequence_length"])
    )
    development_blocks = load_token_blocks(
        development_path, sequence_length=int(training["sequence_length"])
    )
    selected_count = min(
        len(train_blocks), int(training["max_input_tokens"]) // int(training["sequence_length"])
    )
    selected_blocks = train_blocks[:selected_count]
    examples = encode_raw_code_blocks(
        selected_blocks,
        sequence_length=int(training["sequence_length"]),
        vocab_size=151_936,
    )
    batches = training_batches(
        examples,
        effective_batch=int(training["effective_batch"]),
        seed=int(training["seed"]),
    )
    logical_training_tokens = sum(example.total_tokens for example in examples)
    max_additional_tokens = int(
        budget.get("maximum_additional_training_input_tokens", training["max_input_tokens"])
    )
    actual_planned_campaign_tokens = validate_campaign_token_budget(
        logical_training_tokens=logical_training_tokens,
        external_campaign_tokens=external_campaign_tokens,
        maximum_additional_tokens=max_additional_tokens,
    )
    plan_sha = sha256_file(plan_path)
    if not execute:
        return {
            "status": "preflight",
            "model": model_identity,
            "plan_sha256": plan_sha,
            "train_file_sha256": sha256_file(train_path),
            "development_file_sha256": sha256_file(development_path),
            "training_blocks": len(examples),
            "training_input_tokens": sum(example.total_tokens for example in examples),
            "causal_target_tokens": sum(example.response_tokens for example in examples),
            "external_campaign_tokens": external_campaign_tokens,
            "actual_planned_campaign_tokens": actual_planned_campaign_tokens,
            "maximum_additional_training_input_tokens": max_additional_tokens,
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
    hardware_runtime = runtime_identity(device)
    runtime = {
        **hardware_runtime,
        **_runtime_training(training),
    }
    identity = training_identity(
        document=document,
        plan_sha256=plan_sha,
        model_identity=model_identity,
        train_path=train_path,
        development_path=development_path,
        train_blocks=train_blocks,
        development_blocks=development_blocks,
        batches=batches,
        runtime=runtime,
    )
    fingerprint = canonical_sha256(identity)
    output_cap_bytes = int(training["max_output_bytes"])
    total_cap_bytes = int(budget.get("new_artifact_bytes_cap", 12 * 1024**3))
    minimum_free_bytes = int(budget.get("minimum_free_bytes", 2 * 1024**3))
    resume_path = resolve_resume_path(resume) if resume is not None else None
    resume_is_output_owned = (
        resume_path is not None
        and resume_path.resolve().is_relative_to(output.resolve())
    )
    resume_bytes = (
        resume_path.stat().st_size
        if resume_path is not None and not resume_is_output_owned
        else 0
    )
    input_artifact_bytes = (
        sum(int(item["bytes"]) for item in model_identity["files"].values())
        + train_path.stat().st_size
        + development_path.stat().st_size
        + resume_bytes
    )
    if output.exists() and resume is None and any(output.iterdir()):
        raise FileExistsError("existing Q25 output requires an exact-resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest = {"fingerprint": fingerprint, "identity": identity}
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("fingerprint") != fingerprint:
            raise ValueError("Q25 runtime, model, or data identity changed on resume")
    else:
        atomic_json(manifest_path, manifest)

    if resume_path is not None:
        previous_output = resume_path.parent
        previous_manifest_path = previous_output / "run_manifest.json"
        try:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_manifest = {}
        if previous_manifest.get("fingerprint") == fingerprint:
            previous_before_path = previous_output / "development-before.json"
            before_path = output / "development-before.json"
            if not before_path.exists():
                try:
                    previous_before = json.loads(previous_before_path.read_text(encoding="utf-8"))
                    if previous_before.get("fingerprint") == fingerprint:
                        atomic_json(before_path, previous_before)
                except (OSError, ValueError):
                    pass
            observability_path = output / "observability-run.json"
            previous_observability = previous_output / "observability-run.json"
            if not observability_path.exists():
                try:
                    metadata = json.loads(previous_observability.read_text(encoding="utf-8"))
                    if set(metadata) >= {"campaign_id", "run_id"}:
                        atomic_json(observability_path, metadata)
                except (OSError, ValueError):
                    pass

    with run_scope(output / "observability-run.json", "q25-code-cpt"):
        with operation("model.load"):
            model, tokenizer = _load_model(model_path)
        if model.config.vocab_size != 151_936:
            raise ValueError("loaded Q25 model vocabulary differs from the pinned tokenizer")
        if training["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        model.to(device)
        optimizer, optimizer_name = _make_optimizer(model, training)
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
        before_path = output / "development-before.json"
        if before_path.exists():
            before = json.loads(before_path.read_text(encoding="utf-8"))
            if before.get("fingerprint") != fingerprint:
                raise ValueError("stored development baseline belongs to another run")
        else:
            before = evaluate_source_nll(
                model, development_blocks, device=device, max_blocks=SOURCE_NLL_BLOCKS
            )
            before["fingerprint"] = fingerprint
            atomic_json(before_path, before)

        cursor = TrainingCursor()
        if resume_path is not None:
            cursor = load_training_cursor(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                fingerprint=fingerprint,
            )

        latest_path: Path | None = None
        latest_json = output / "latest.json"
        if latest_json.exists():
            latest_path = resolve_resume_path(latest_json)

        max_checkpoint_save_seconds = 0.0

        def save_cursor(value: TrainingCursor) -> None:
            nonlocal latest_path
            nonlocal max_checkpoint_save_seconds
            checkpoint_started = time.monotonic()
            try:
                latest_path = save_training_cursor(
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
                        model_identity["files"]["model.safetensors"]["bytes"]
                    ),
                )
            finally:
                max_checkpoint_save_seconds = max(
                    max_checkpoint_save_seconds, time.monotonic() - checkpoint_started
                )

        def dynamic_reserve_seconds() -> float:
            return max(reserve, 2 * max_checkpoint_save_seconds + 60)

        # A committed update-boundary state exists even if the first full
        # optimizer update later fails or the worker is interrupted.
        save_cursor(cursor)

        successful_updates = cursor.completed_updates

        def log_update(record: dict[str, int | float | bool]) -> None:
            nonlocal successful_updates
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
            max_input_tokens=max_additional_tokens,
            external_campaign_tokens=external_campaign_tokens,
            deadline_monotonic=deadline,
            finalization_reserve_seconds=dynamic_reserve_seconds,
            checkpoint_every_updates=int(training["checkpoint_every_updates"]),
            on_checkpoint=save_cursor,
            on_update=log_update,
        )
        summary: dict[str, Any] = {
            "schema": "q25-raw-code-cpt-run-v1",
            "status": result.status,
            "fingerprint": fingerprint,
            "identity": identity,
            "cursor": asdict(result.cursor),
            "logical_training_input_tokens": result.cursor.training_input_tokens,
            "external_campaign_tokens": external_campaign_tokens,
            "actual_campaign_input_tokens": (
                external_campaign_tokens + result.cursor.training_input_tokens
            ),
            "maximum_additional_training_input_tokens": max_additional_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "session_seconds": session_seconds,
            "finalization_reserve_seconds": dynamic_reserve_seconds(),
            "max_checkpoint_save_seconds": max_checkpoint_save_seconds,
            "training_updates": len(result.update_records),
            "optimizer": optimizer_name,
            "development_before": before,
            "latest_checkpoint": latest_path.name if latest_path else None,
            "storage": {
                "input_artifact_bytes_including_resume": input_artifact_bytes,
                "output_bytes": directory_bytes(output),
                "output_cap_bytes": output_cap_bytes,
                "aggregate_cap_bytes": total_cap_bytes,
                "checkpoint_estimate_bytes": estimate_checkpoint_bytes(
                    model,
                    optimizer,
                    int(model_identity["files"]["model.safetensors"]["bytes"]),
                ),
            },
        }
        atomic_json(output / "run_result.json", summary)

        if result.cursor.attempted_updates:
            after = evaluate_source_nll(
                model, development_blocks, device=device, max_blocks=SOURCE_NLL_BLOCKS
            )
            after["fingerprint"] = fingerprint
            atomic_json(output / "development-after.json", after)
            summary["development_after"] = after
            export = save_inference_export(
                model,
                tokenizer,
                output / "inference-f16",
                output_root=output,
                output_cap_bytes=output_cap_bytes,
                total_cap_bytes=total_cap_bytes,
                input_artifact_bytes=input_artifact_bytes,
                minimum_free_bytes=minimum_free_bytes,
                source_weight_bytes=int(model_identity["files"]["model.safetensors"]["bytes"]),
                fingerprint=fingerprint,
                cursor=result.cursor,
            )
            summary["inference_export"] = export
            summary["storage"]["output_bytes"] = directory_bytes(output)
        atomic_json(output / "run_result.json", summary)
        return summary


def main() -> None:
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-seconds", type=float, required=True)
    parser.add_argument("--reserve-seconds", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--external-campaign-tokens", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_training(
        model_path=args.model,
        train_path=args.train,
        development_path=args.development,
        plan_path=args.plan,
        output=args.output,
        session_seconds=args.session_seconds,
        reserve_seconds=args.reserve_seconds,
        resume=args.resume,
        external_campaign_tokens=args.external_campaign_tokens,
        execute=args.execute,
        invocation_started=invocation_started,
    )
    if args.execute:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "cursor": result["cursor"],
                    "result": str(args.output / "run_result.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
