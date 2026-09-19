"""Exact-token full-weight causal training for Stage 1."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tinycomplete.code_cpt.prepare import CORE_LANGUAGES, MODEL_ID, MODEL_REVISION


def distributed_block_indices(length: int, rank: int, world_size: int) -> list[int]:
    """Return an equal-size, non-overlapping block partition for one rank."""
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed rank/world_size")
    usable = length - length % world_size
    return list(range(rank, usable, world_size))


def milestones_crossed(previous: int, current: int, milestones: Iterable[int]) -> list[int]:
    return [milestone for milestone in milestones if previous < milestone <= current]


def bounded_optimizer_steps(
    *,
    remaining_tokens: int,
    tokens_per_update: int,
    available_blocks: int,
    blocks_per_update: int,
) -> int:
    requested = math.ceil(max(0, remaining_tokens) / tokens_per_update)
    available = available_blocks // blocks_per_update
    return min(requested, available)


def language_mix_for_prefix(
    language_path: Path, language_order: list[str], blocks: int, block_size: int
) -> dict[str, dict[str, float] | dict[str, int]]:
    ids = np.load(language_path, mmap_mode="r")[:blocks]
    counts = {
        language: int(np.count_nonzero(ids == index))
        for index, language in enumerate(language_order)
    }
    token_counts = {language: count * block_size for language, count in counts.items()}
    total = sum(token_counts.values())
    percentages = {
        language: 100.0 * count / total if total else 0.0
        for language, count in token_counts.items()
    }
    return {"token_counts": token_counts, "percentages": percentages}


def is_broad_deterioration(current: dict, baseline: dict) -> bool:
    languages = list(CORE_LANGUAGES)
    regressions = sum(
        current[language]["nll"] > baseline[language]["nll"] * 1.01
        for language in languages
    )
    code_worse = current["overall_code"]["nll"] > baseline["overall_code"]["nll"] * 1.01
    general_collapse = current["general"]["nll"] > baseline["general"]["nll"] * 1.25
    return bool((code_worse and regressions >= 5) or general_collapse)


@dataclass
class TrainingCounters:
    world_size: int = 1
    training_tokens: int = 0
    microsteps: int = 0
    optimizer_steps: int = 0
    data_wait_seconds: float = 0.0

    def record_microstep(self, local_nonpadding_tokens: int, data_wait_seconds: float) -> None:
        self.training_tokens += local_nonpadding_tokens * self.world_size
        self.microsteps += 1
        self.data_wait_seconds += data_wait_seconds

    def record_optimizer_step(self) -> None:
        self.optimizer_steps += 1


class PackedBlocksDataset:
    def __init__(self, path: Path, start_block: int = 0, block_count: int | None = None):
        self.blocks = np.load(path, mmap_mode="r")
        if self.blocks.ndim != 2:
            raise ValueError("packed block array must have shape [blocks, sequence]")
        if start_block < 0 or start_block >= len(self.blocks):
            raise ValueError("start_block is outside the packed corpus")
        self.start_block = start_block
        available = len(self.blocks) - start_block
        self.block_count = available if block_count is None else min(block_count, available)

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int):
        import torch

        if index < 0 or index >= self.block_count:
            raise IndexError(index)
        values = np.array(self.blocks[self.start_block + index], dtype=np.int64, copy=True)
        ids = torch.from_numpy(values)
        return {"input_ids": ids, "labels": ids.clone()}


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def resolve_optional_hf_token() -> tuple[str | None, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token, "environment"
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token, "kaggle_secret"
    except Exception:
        pass
    return None, "anonymous_public_model"


def _load_model_and_tokenizer(token: str | None, checkpoint: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = checkpoint or MODEL_ID
    revision = None if checkpoint else MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=revision, token=token, trust_remote_code=False
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        token=token,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # The source config advertises BF16 and some Transformers versions honor
    # that metadata despite the requested dtype. T4 GradScaler cannot unscale
    # BF16 gradients, so keep explicit FP32 master parameters and allow the
    # distributed mixed-precision policy to cast only computation to FP16.
    model.float()
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise TypeError(f"expected FP32 master parameters, got {floating_dtypes}")
    model.config.use_cache = False
    return model, tokenizer


def _optimizer(model, name: str, learning_rate: float, weight_decay: float):
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if name == "adamw_torch":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=torch.cuda.is_available(),
        )
    if name == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit

        return AdamW8bit(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {name}")


def _constant_with_warmup(optimizer, warmup_steps: int):
    import torch

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate_micro(
    model, micro_dir: Path, device, *, batch_size: int = 1, accelerator=None
) -> dict:
    import torch

    model.eval()
    result = {}
    names = [*CORE_LANGUAGES, "general"]
    rank = accelerator.process_index if accelerator is not None else 0
    world_size = accelerator.num_processes if accelerator is not None else 1
    with torch.inference_mode():
        for name in names:
            blocks = np.load(micro_dir / f"{name}.npy", mmap_mode="r")
            total_nll = 0.0
            token_count = 0
            local_indices = list(range(rank, len(blocks), world_size))
            for offset in range(0, len(local_indices), batch_size):
                indices = local_indices[offset : offset + batch_size]
                batch = torch.from_numpy(
                    np.array(blocks[indices], dtype=np.int64, copy=True)
                ).to(device)
                precision_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with precision_context:
                    output = model(input_ids=batch, labels=batch, use_cache=False)
                scored = batch.numel() - batch.shape[0]
                total_nll += float(output.loss.float().item()) * scored
                token_count += scored
                del output, batch
            if accelerator is not None:
                totals = torch.tensor(
                    [total_nll, token_count], device=device, dtype=torch.float64
                )
                totals = accelerator.reduce(totals, reduction="sum")
                total_nll, token_count = float(totals[0].item()), int(totals[1].item())
            result[name] = {"nll": total_nll / token_count, "tokens": token_count}
    code_nll_sum = sum(result[name]["nll"] * result[name]["tokens"] for name in CORE_LANGUAGES)
    code_tokens = sum(result[name]["tokens"] for name in CORE_LANGUAGES)
    result["overall_code"] = {"nll": code_nll_sum / code_tokens, "tokens": code_tokens}
    model.train()
    return result


def extract_mtp_from_snapshot(snapshot: Path, destination: Path) -> dict:
    """Copy ignored native MTP tensors from a downloaded snapshot into a sidecar."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = snapshot / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    mtp_keys = sorted(key for key in weight_map if key.startswith("mtp."))
    tensors = {}
    by_shard: dict[str, list[str]] = {}
    for key in mtp_keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    for shard, keys in by_shard.items():
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    destination.mkdir(parents=True, exist_ok=True)
    sidecar = destination / "mtp-original.safetensors"
    save_file(tensors, sidecar)
    manifest = {
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "trained": False,
        "tensor_count": len(tensors),
        "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
        "bytes": sidecar.stat().st_size,
        "note": (
            "Transformers Qwen3_5ForCausalLM ignores these tensors; "
            "preserved for later MTP work."
        ),
    }
    _json_write(destination / "mtp-manifest.json", manifest)
    return manifest


def extract_mtp_sidecar(destination: Path, token: str | None) -> dict:
    """Download the pinned source if needed, then preserve its untrained MTP tensors."""
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    )
    return extract_mtp_from_snapshot(snapshot, destination)


def save_snapshot(
    accelerator, model, tokenizer, destination: Path, metadata: dict, token: str | None
) -> None:
    import torch

    accelerator.wait_for_everyone()
    disk_probe = destination
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    if shutil.disk_usage(disk_probe).free < 3 * 2**30:
        raise OSError("less than 3 GiB free before model snapshot")
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        snapshot_state = {
            name: (
                tensor.detach().to(device="cpu", dtype=torch.float16)
                if tensor.is_floating_point()
                else tensor.detach().cpu()
            )
            for name, tensor in state_dict.items()
        }
        unwrapped.save_pretrained(
            destination,
            state_dict=snapshot_state,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(destination)
        extract_mtp_sidecar(destination, token)
        _json_write(destination / "training_metadata.json", metadata)
    accelerator.wait_for_everyone()


@dataclass
class RunConfig:
    corpus_dir: Path
    output_dir: Path
    learning_rate: float
    microbatch: int = 1
    gradient_accumulation: int = 16
    optimizer: str = "adamw_8bit"
    gradient_checkpointing: bool = True
    weight_decay: float = 0.01
    warmup_steps: int = 5
    max_tokens: int = 524_288
    start_block: int = 0
    workers: int = 1
    prefetch_factor: int = 2
    seed: int = 271828
    max_grad_norm: float = 1.0
    deadline_seconds: float = 0.0
    save_final: bool = False
    save_resume: bool = False
    eval_final: bool = False
    baseline_path: Path | None = None
    resume_from: Path | None = None
    milestones: tuple[int, ...] = ()
    distributed_mode: str = "ddp"


def run_training(config: RunConfig) -> dict:
    import torch
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from accelerate.utils import GradientAccumulationPlugin
    from torch.utils.data import DataLoader

    fsdp_plugin = None
    if config.distributed_mode == "fsdp":
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            MixedPrecision,
            ShardingStrategy,
            StateDictType,
        )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["Qwen3_5DecoderLayer"],
            mixed_precision_policy=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ),
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            ),
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
        )
    elif config.distributed_mode != "ddp":
        raise ValueError(f"unsupported distributed mode: {config.distributed_mode}")
    accumulation_plugin = GradientAccumulationPlugin(
        num_steps=config.gradient_accumulation,
        # FSDP no_sync retains full, unsharded gradients until the optimizer
        # boundary. Synchronize each microbatch to keep memory truly sharded.
        sync_each_batch=config.distributed_mode == "fsdp",
    )
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_plugin=accumulation_plugin,
        fsdp_plugin=fsdp_plugin,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    token, credential_source = resolve_optional_hf_token()
    model, tokenizer = _load_model_and_tokenizer(token)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    resume_metadata = {}
    if config.resume_from is not None:
        resume_metadata = json.loads(
            (config.resume_from / "resume_metadata.json").read_text(encoding="utf-8")
        )
    initial_tokens = int(resume_metadata.get("training_tokens", 0))
    initial_microsteps = int(resume_metadata.get("microsteps", 0))
    initial_optimizer_steps = int(resume_metadata.get("optimizer_steps", 0))
    initial_data_wait = float(resume_metadata.get("data_wait_seconds", 0.0))
    start_block = int(resume_metadata.get("next_block", config.start_block))
    block_path = config.corpus_dir / "train_blocks.npy"
    block_shape = np.load(block_path, mmap_mode="r").shape
    corpus_metadata = json.loads(
        (config.corpus_dir / "corpus_metadata.json").read_text(encoding="utf-8")
    )
    tokens_per_microstep = config.microbatch * block_shape[1] * accelerator.num_processes
    tokens_per_update = tokens_per_microstep * config.gradient_accumulation
    remaining_tokens = max(0, config.max_tokens - initial_tokens)
    blocks_per_update = (
        config.gradient_accumulation * config.microbatch * accelerator.num_processes
    )
    available_blocks = block_shape[0] - start_block
    additional_optimizer_steps = bounded_optimizer_steps(
        remaining_tokens=remaining_tokens,
        tokens_per_update=tokens_per_update,
        available_blocks=available_blocks,
        blocks_per_update=blocks_per_update,
    )
    if additional_optimizer_steps == 0:
        raise ValueError("no complete optimizer update remains in the token budget/corpus")
    blocks_needed = (
        additional_optimizer_steps * blocks_per_update
    )
    dataset = PackedBlocksDataset(block_path, start_block, blocks_needed)
    loader_options: dict[str, Any] = {
        "batch_size": config.microbatch,
        "shuffle": False,
        "drop_last": True,
        "num_workers": config.workers,
        "pin_memory": True,
    }
    if config.workers:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=config.prefetch_factor,
        )
    loader: Any = DataLoader(dataset, **loader_options)  # type: ignore[arg-type,var-annotated]
    optimizer = _optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = _constant_with_warmup(optimizer, config.warmup_steps)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.register_for_checkpointing(scheduler)
    if config.resume_from is not None:
        accelerator.load_state(config.resume_from)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        **asdict(config),
        "corpus_dir": str(config.corpus_dir),
        "output_dir": str(config.output_dir),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "precision": "fp16 autocast with dynamic loss scaling",
        "master_parameter_dtype": "float32",
        "world_size": accelerator.num_processes,
        "tokens_per_update": tokens_per_update,
        "credential_source": credential_source,
        "milestones": list(config.milestones),
        "baseline_path": str(config.baseline_path) if config.baseline_path else None,
        "resume_from": str(config.resume_from) if config.resume_from else None,
    }
    if accelerator.is_main_process:
        _json_write(config.output_dir / "run_config.json", run_metadata)
    counters = TrainingCounters(
        world_size=accelerator.num_processes,
        training_tokens=initial_tokens,
        microsteps=initial_microsteps,
        optimizer_steps=initial_optimizer_steps,
        data_wait_seconds=initial_data_wait,
    )
    target_optimizer_steps = initial_optimizer_steps + additional_optimizer_steps
    losses = []
    grad_norms = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accelerator.wait_for_everyone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    non_training_seconds = 0.0
    steady_start_time = None
    steady_start_tokens = initial_tokens
    steady_non_training_seconds = 0.0
    iterator = iter(loader)
    stop_reason = "max_tokens"
    milestone_checked_tokens = counters.training_tokens
    stop_requested = False
    baseline_metrics = None
    if config.baseline_path is not None:
        baseline_metrics = json.loads(config.baseline_path.read_text(encoding="utf-8"))["metrics"]
    while counters.optimizer_steps < target_optimizer_steps:
        batch_released = False
        end_after_step = False
        wait_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            stop_reason = "corpus_exhausted"
            break
        data_wait = time.perf_counter() - wait_start
        local_tokens = int(batch["labels"].ne(-100).sum().item())
        with accelerator.accumulate(model):
            with accelerator.autocast():
                output = model(**batch, use_cache=False)
                loss = output.loss
            finite = accelerator.reduce(torch.isfinite(loss).float(), reduction="min")
            if not bool(finite.item()):
                raise FloatingPointError("non-finite training loss")
            accelerator.backward(loss)
            grad_norm = None
            grad_norm_value = None
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                assert grad_norm is not None
                finite_grad = accelerator.reduce(torch.isfinite(grad_norm).float(), reduction="min")
                if not bool(finite_grad.item()):
                    raise FloatingPointError("non-finite gradient norm")
                grad_norm_value = float(grad_norm.item())
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        counters.record_microstep(local_tokens, data_wait)
        if accelerator.sync_gradients:
            assert grad_norm_value is not None
            counters.record_optimizer_step()
            reduced_loss = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            loss_entry = {
                "optimizer_step": counters.optimizer_steps,
                "microsteps": counters.microsteps,
                "training_tokens": counters.training_tokens,
                "loss": reduced_loss,
                "gradient_norm": grad_norm_value,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            losses.append(loss_entry)
            grad_norms.append(grad_norm_value)
            if accelerator.is_main_process:
                _append_jsonl(config.output_dir / "train_log.jsonl", loss_entry)
            if steady_start_time is None:
                steady_start_time = time.perf_counter()
                steady_start_tokens = counters.training_tokens
            del output, loss, batch
            batch_released = True
            for milestone in milestones_crossed(
                milestone_checked_tokens, counters.training_tokens, config.milestones
            ):
                pause_start = time.perf_counter()
                metadata = {**run_metadata, **asdict(counters), "milestone": milestone}
                destination = config.output_dir / "snapshots" / f"tokens-{milestone:09d}"
                save_snapshot(accelerator, model, tokenizer, destination, metadata, token)
                evaluation = evaluate_micro(
                    model,
                    config.corpus_dir / "micro",
                    accelerator.device,
                    accelerator=accelerator,
                )
                if accelerator.is_main_process:
                    _json_write(destination / "micro_eval.json", evaluation)
                    stop_requested = bool(
                        baseline_metrics
                        and is_broad_deterioration(evaluation, baseline_metrics)
                    )
                stop_tensor = torch.tensor(
                    int(stop_requested), device=accelerator.device, dtype=torch.int32
                )
                stop_tensor = accelerator.reduce(stop_tensor, reduction="max")
                stop_requested = bool(stop_tensor.item())
                accelerator.wait_for_everyone()
                pause_seconds = time.perf_counter() - pause_start
                non_training_seconds += pause_seconds
                if steady_start_time is not None:
                    steady_non_training_seconds += pause_seconds
            milestone_checked_tokens = counters.training_tokens
            if stop_requested:
                stop_reason = "broad_validation_deterioration"
                end_after_step = True
            elif config.deadline_seconds and time.perf_counter() - start >= config.deadline_seconds:
                stop_reason = "wall_deadline"
                end_after_step = True
        if not batch_released:
            del output, loss, batch
        if end_after_step:
            break
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    session_tokens = counters.training_tokens - initial_tokens
    session_data_wait = counters.data_wait_seconds - initial_data_wait
    steady_wall_seconds = (
        time.perf_counter() - steady_start_time - steady_non_training_seconds
        if steady_start_time is not None
        else 0.0
    )
    steady_tokens = counters.training_tokens - steady_start_tokens
    peak_vram = torch.cuda.max_memory_allocated() / 2**30
    rank_stats = accelerator.gather(
        torch.tensor(
            [
                wall_seconds,
                non_training_seconds,
                session_data_wait,
                steady_wall_seconds,
                peak_vram,
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )
    ).view(-1, 5)
    wall_seconds = float(rank_stats[:, 0].max().item())
    non_training_seconds = float(rank_stats[:, 1].max().item())
    session_data_wait = float(rank_stats[:, 2].mean().item())
    steady_wall_seconds = float(rank_stats[:, 3].max().item())
    peak_vram_by_gpu = [float(value) for value in rank_stats[:, 4].tolist()]
    peak_vram = max(peak_vram_by_gpu)
    training_wall_seconds = wall_seconds - non_training_seconds
    summary = {
        **run_metadata,
        **asdict(counters),
        "wall_seconds": wall_seconds,
        "non_training_checkpoint_eval_seconds": non_training_seconds,
        "training_wall_seconds": training_wall_seconds,
        "session_training_tokens": session_tokens,
        "tokens_per_second": session_tokens / training_wall_seconds,
        "steady_state_tokens_per_second": (
            steady_tokens / steady_wall_seconds
            if steady_wall_seconds > 0 and steady_tokens
            else None
        ),
        "data_wait_percent": 100.0 * session_data_wait / training_wall_seconds,
        "peak_vram_gb": peak_vram,
        "peak_vram_by_gpu_gb": peak_vram_by_gpu,
        "stop_reason": stop_reason,
        "nan_or_inf": False,
        "loss_history": losses,
        "gradient_norm_history": grad_norms,
        "actual_language_mix": language_mix_for_prefix(
            config.corpus_dir / "train_languages.npy",
            corpus_metadata["language_order"],
            counters.training_tokens // block_shape[1],
            block_shape[1],
        ),
        "prepared_blocks_initial": len(dataset),
        "prepared_seconds_ahead_at_measured_rate": (
            len(dataset) * block_shape[1] / (session_tokens / training_wall_seconds)
        ),
    }
    if config.save_final:
        save_snapshot(
            accelerator,
            model,
            tokenizer,
            config.output_dir / "final",
            summary,
            token,
        )
        evaluation = evaluate_micro(
            model,
            config.corpus_dir / "micro",
            accelerator.device,
            accelerator=accelerator,
        )
        if accelerator.is_main_process:
            _json_write(config.output_dir / "final" / "micro_eval.json", evaluation)
        accelerator.wait_for_everyone()
    elif config.eval_final:
        evaluation = evaluate_micro(
            model,
            config.corpus_dir / "micro",
            accelerator.device,
            accelerator=accelerator,
        )
        if accelerator.is_main_process:
            _json_write(config.output_dir / "micro_eval.json", evaluation)
        accelerator.wait_for_everyone()
    if config.save_resume:
        required_free = 10 * 2**30 if config.optimizer == "adamw_torch" else 5 * 2**30
        free = shutil.disk_usage(config.output_dir).free
        if free >= required_free:
            accelerator.save_state(config.output_dir / "resume-latest", safe_serialization=True)
            if accelerator.is_main_process:
                resume = {
                    **asdict(counters),
                    "next_block": start_block + session_tokens // block_shape[1],
                    "tokens_per_update": tokens_per_update,
                    "model_revision": MODEL_REVISION,
                }
                _json_write(
                    config.output_dir / "resume-latest" / "resume_metadata.json", resume
                )
            summary["resume_state_saved"] = True
        else:
            summary["resume_state_saved"] = False
            summary["resume_state_skip_reason"] = (
                f"only {free / 2**30:.2f} GiB free; "
                f"required {required_free / 2**30:.0f} GiB"
            )
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _json_write(config.output_dir / "summary.json", summary)
    accelerator.wait_for_everyone()
    return summary


def run_baseline(corpus_dir: Path, output_path: Path) -> dict:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("baseline evaluation requires CUDA")
    token, credential_source = resolve_optional_hf_token()
    model, _ = _load_model_and_tokenizer(token)
    model.to(device="cuda")
    metrics = evaluate_micro(model, corpus_dir / "micro", torch.device("cuda"))
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": "fp16 autocast over fp32 master weights",
        "seed": 271828,
        "credential_source": credential_source,
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
        "metrics": metrics,
    }
    _json_write(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--corpus-dir", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--corpus-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--learning-rate", type=float, required=True)
    train.add_argument("--microbatch", type=int, default=1)
    train.add_argument("--gradient-accumulation", type=int, default=16)
    train.add_argument("--optimizer", choices=("adamw_torch", "adamw_8bit"), default="adamw_8bit")
    train.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument("--max-tokens", type=int, default=524_288)
    train.add_argument("--start-block", type=int, default=0)
    train.add_argument("--workers", type=int, default=1)
    train.add_argument("--prefetch-factor", type=int, default=2)
    train.add_argument("--warmup-steps", type=int, default=5)
    train.add_argument("--deadline-seconds", type=float, default=0)
    train.add_argument("--save-final", action="store_true")
    train.add_argument("--save-resume", action="store_true")
    train.add_argument("--eval-final", action="store_true")
    train.add_argument("--baseline-path", type=Path)
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--milestones", type=int, nargs="*", default=[])
    train.add_argument("--distributed-mode", choices=("ddp", "fsdp"), default="ddp")
    args = parser.parse_args()
    if args.command == "baseline":
        print(json.dumps(run_baseline(args.corpus_dir, args.output), indent=2, sort_keys=True))
        return
    config = RunConfig(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        microbatch=args.microbatch,
        gradient_accumulation=args.gradient_accumulation,
        optimizer=args.optimizer,
        gradient_checkpointing=args.gradient_checkpointing,
        max_tokens=args.max_tokens,
        start_block=args.start_block,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        warmup_steps=args.warmup_steps,
        deadline_seconds=args.deadline_seconds,
        save_final=args.save_final,
        save_resume=args.save_resume,
        eval_final=args.eval_final,
        baseline_path=args.baseline_path,
        resume_from=args.resume_from,
        milestones=tuple(args.milestones),
        distributed_mode=args.distributed_mode,
    )
    summary = run_training(config)
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
