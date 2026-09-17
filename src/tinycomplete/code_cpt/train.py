"""Exact-token full-weight causal training for Stage 1."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Iterable
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


def evaluate_micro(model, micro_dir: Path, device, *, batch_size: int = 1) -> dict:
    import torch

    model.eval()
    result = {}
    names = [*CORE_LANGUAGES, "general"]
    with torch.inference_mode():
        for name in names:
            blocks = np.load(micro_dir / f"{name}.npy", mmap_mode="r")
            total_nll = 0.0
            token_count = 0
            for start in range(0, len(blocks), batch_size):
                batch = torch.from_numpy(
                    np.array(blocks[start : start + batch_size], dtype=np.int64, copy=True)
                ).to(device)
                output = model(input_ids=batch, labels=batch, use_cache=False)
                scored = batch.numel() - batch.shape[0]
                total_nll += float(output.loss.float().item()) * scored
                token_count += scored
                del output, batch
            result[name] = {"nll": total_nll / token_count, "tokens": token_count}
    code_nll_sum = sum(result[name]["nll"] * result[name]["tokens"] for name in CORE_LANGUAGES)
    code_tokens = sum(result[name]["tokens"] for name in CORE_LANGUAGES)
    result["overall_code"] = {"nll": code_nll_sum / code_tokens, "tokens": code_tokens}
    model.train()
    return result


def extract_mtp_sidecar(destination: Path, token: str | None) -> dict:
    """Copy the ignored native MTP tensors into a sidecar without training them."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    )
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


def save_snapshot(
    accelerator, model, tokenizer, destination: Path, metadata: dict, token: str | None
) -> None:
    import torch

    accelerator.wait_for_everyone()
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
    milestones: tuple[int, ...] = ()


def run_training(config: RunConfig) -> dict:
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=config.gradient_accumulation,
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
    block_path = config.corpus_dir / "train_blocks.npy"
    block_shape = np.load(block_path, mmap_mode="r").shape
    tokens_per_microstep = config.microbatch * block_shape[1] * accelerator.num_processes
    tokens_per_update = tokens_per_microstep * config.gradient_accumulation
    max_optimizer_steps = max(1, math.ceil(config.max_tokens / tokens_per_update))
    blocks_needed = (
        max_optimizer_steps
        * config.gradient_accumulation
        * config.microbatch
        * accelerator.num_processes
    )
    dataset = PackedBlocksDataset(block_path, config.start_block, blocks_needed)
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
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        **asdict(config),
        "corpus_dir": str(config.corpus_dir),
        "output_dir": str(config.output_dir),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "precision": "fp16 autocast with dynamic loss scaling",
        "world_size": accelerator.num_processes,
        "tokens_per_update": tokens_per_update,
        "credential_source": credential_source,
        "milestones": list(config.milestones),
    }
    if accelerator.is_main_process:
        _json_write(config.output_dir / "run_config.json", run_metadata)
    counters = TrainingCounters(world_size=accelerator.num_processes)
    losses = []
    grad_norms = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accelerator.wait_for_everyone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    iterator = iter(loader)
    stop_reason = "max_tokens"
    milestone_checked_tokens = 0
    while counters.optimizer_steps < max_optimizer_steps:
        if config.deadline_seconds and time.perf_counter() - start >= config.deadline_seconds:
            stop_reason = "wall_deadline"
            break
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
            for milestone in milestones_crossed(
                milestone_checked_tokens, counters.training_tokens, config.milestones
            ):
                metadata = {**run_metadata, **asdict(counters), "milestone": milestone}
                destination = config.output_dir / "snapshots" / f"tokens-{milestone:09d}"
                save_snapshot(accelerator, model, tokenizer, destination, metadata, token)
                if accelerator.is_main_process:
                    evaluation = evaluate_micro(
                        accelerator.unwrap_model(model),
                        config.corpus_dir / "micro",
                        accelerator.device,
                    )
                    _json_write(destination / "micro_eval.json", evaluation)
                accelerator.wait_for_everyone()
            milestone_checked_tokens = counters.training_tokens
        del output, loss, batch
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() / 2**30
    summary = {
        **run_metadata,
        **asdict(counters),
        "wall_seconds": wall_seconds,
        "tokens_per_second": counters.training_tokens / wall_seconds,
        "data_wait_percent": 100.0 * counters.data_wait_seconds / wall_seconds,
        "peak_vram_gb": peak_vram,
        "stop_reason": stop_reason,
        "nan_or_inf": False,
        "loss_history": losses,
        "gradient_norm_history": grad_norms,
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
        if accelerator.is_main_process:
            evaluation = evaluate_micro(
                accelerator.unwrap_model(model), config.corpus_dir / "micro", accelerator.device
            )
            _json_write(config.output_dir / "final" / "micro_eval.json", evaluation)
        accelerator.wait_for_everyone()
    elif config.eval_final:
        if accelerator.is_main_process:
            evaluation = evaluate_micro(
                accelerator.unwrap_model(model), config.corpus_dir / "micro", accelerator.device
            )
            _json_write(config.output_dir / "micro_eval.json", evaluation)
        accelerator.wait_for_everyone()
    if config.save_resume:
        accelerator.save_state(config.output_dir / "resume-latest", safe_serialization=True)
    if accelerator.is_main_process:
        _json_write(config.output_dir / "summary.json", summary)
    accelerator.wait_for_everyone()
    return summary


def run_baseline(corpus_dir: Path, output_path: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("baseline evaluation requires CUDA")
    token, credential_source = resolve_optional_hf_token()
    model, _ = _load_model_and_tokenizer(token)
    model.to(device="cuda", dtype=torch.float16)
    metrics = evaluate_micro(model, corpus_dir / "micro", torch.device("cuda"))
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": "fp16",
        "seed": 271828,
        "credential_source": credential_source,
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
    train.add_argument("--milestones", type=int, nargs="*", default=[])
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
        milestones=tuple(args.milestones),
    )
    summary = run_training(config)
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
