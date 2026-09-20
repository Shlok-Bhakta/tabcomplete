"""T4x2 training-throughput benchmark worker for Stage 1.

Each candidate runs in its own fresh ``torch.distributed.run`` subprocess so
that FSDP state, ``torch.compile`` caches, Qwen kernel backend selection, and
CUDA allocator state cannot leak between measurements.

The worker preserves the production training problem exactly:

* sequence length comes from the packed corpus (2048)
* full-weight training (every parameter requires grad)
* full vocabulary causal LM objective
* same packed ``train_blocks.npy`` token ordering from ``start_block``
* same seed and model revision as production

It writes one machine-readable result JSON per candidate, including failed
candidates (OOM / import failure / NaN / GradScaler failure / ...).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tinycomplete.code_cpt.prepare import MODEL_ID, MODEL_REVISION
from tinycomplete.code_cpt.train import (
    PackedBlocksDataset,
    _constant_with_warmup,
    _load_model_and_tokenizer,
    _optimizer,
    bounded_optimizer_steps,
)

BASELINE_TOKENS_PER_SECOND = 810.0
BASELINE_UPDATE_TOKENS = 32_768
STAGE_A_TOKENS = 98_304  # 3 x 32,768 complete optimizer updates
STAGE_B_TOKENS = 524_288  # 16 x 32,768 complete optimizer updates


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return None


def _spec_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def probe_kernel_packages() -> dict[str, Any]:
    """Record which optional optimized kernels are importable and their versions."""
    return {
        "fla_available": _spec_available("fla"),
        "fla_version": _package_version("flash-linear-attention"),
        "causal_conv1d_available": _spec_available("causal_conv1d"),
        "causal_conv1d_version": _package_version("causal-conv1d"),
        "liger_available": _spec_available("liger_kernel"),
        "liger_version": _package_version("liger-kernel"),
        "triton_available": _spec_available("triton"),
        "triton_version": _package_version("triton"),
        "bitsandbytes_version": _package_version("bitsandbytes"),
    }


def probe_gdn_backends(model) -> dict[str, Any]:
    """Inspect the live model for the actual Gated DeltaNet backend selection.

    Transformers 5.x resolves ``fla`` / ``causal-conv1d`` kernels at import
    time via ``use_kernel_func_from_hub_with_fallback``: when the package is
    missing the wrapper falls back to the pure-PyTorch reference. The most
    reliable local signal is therefore package availability plus a live-module
    inspection of one real ``Qwen3_5GatedDeltaNet`` layer.
    """
    info: dict[str, Any] = {}
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as q35

        info["module_chunk_delta_rule"] = repr(getattr(q35, "chunk_gated_delta_rule", None))[:300]
        info["module_recurrent_delta_rule"] = repr(
            getattr(q35, "fused_recurrent_gated_delta_rule", None)
        )[:300]
        info["module_causal_conv"] = repr(getattr(q35, "causal_conv1d_fn", None))[:300]
        # Closure probe: the wrapper closes over ``implementation`` and
        # ``torch_function``. When they are identical the torch fallback runs.
        for label in (
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
        ):
            fn = getattr(q35, label, None)
            closure: dict[str, str] = {}
            try:
                freevars = getattr(fn, "__code__", None)
                cells = getattr(fn, "__closure__", None) or ()
                names = list(freevars.co_freevars) if freevars else []
                for var_name, cell in zip(names, cells, strict=False):
                    try:
                        value = cell.cell_contents
                    except Exception:
                        continue
                    closure[var_name] = repr(value)[:200]
            except Exception:
                pass
            info[f"closure_{label}"] = closure
    except Exception as exc:
        info["probe_error"] = f"{type(exc).__name__}: {exc}"
    gdn = next(
        (
            module
            for module in model.modules()
            if module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
        ),
        None,
    )
    if gdn is None:
        info["gdn_layer_found"] = False
    else:
        info["gdn_layer_found"] = True
        info["gdn_class"] = type(gdn).__name__
        info["gdn_norm"] = type(getattr(gdn, "norm", None)).__name__
        info["gdn_layer_type"] = getattr(gdn, "layer_type", None)
    info["attn_implementation"] = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
    return info


def collect_environment() -> dict[str, Any]:
    import torch
    import transformers

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    try:
        import accelerate

        env["accelerate"] = accelerate.__version__
    except Exception:
        env["accelerate"] = None
    env["numpy"] = np.__version__
    env.update(probe_kernel_packages())
    return env


def resolve_fused_ce(mode: str):
    """Resolve a fused/chunked full-vocabulary linear-CE implementation.

    ``mode`` is one of ``none`` / ``liger`` / ``auto``. ``auto`` prefers Liger
    when importable and otherwise falls back to stock CE so the candidate can
    still run (recorded in JSON).
    """
    if mode in ("none", "stock"):
        return None, "stock"
    errors: dict[str, str] = {}
    if mode in ("liger", "auto"):
        for path in (
            "liger_kernel.chunked_loss:LigerFusedLinearCrossEntropyLoss",
            "liger_kernel.transformers:LigerFusedLinearCrossEntropyLoss",
        ):
            module_name, attr = path.split(":")
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attr)(), "liger"
            except Exception as exc:
                errors[path] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if mode == "liger":
        raise ImportError(f"liger fused CE unavailable: {errors}")
    return None, "stock"


@dataclass
class BenchConfig:
    corpus_dir: Path
    output: Path
    name: str = "candidate"
    learning_rate: float = 3e-6
    max_tokens: int = STAGE_A_TOKENS
    start_block: int = 0
    seed: int = 271828
    microbatch: int = 1
    gradient_accumulation: int = 8
    optimizer: str = "adamw_8bit"
    gradient_checkpointing: bool = True
    fsdp_strategy: str = "FULL_SHARD"
    attn_implementation: str = "sdpa"
    fused_ce: str = "none"
    torch_compile: bool = False
    sync_cleanup: bool = False
    workers: int = 1
    prefetch_factor: int = 2
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    force_torch_fallback: bool = False
    profile_steps: int = 0


def _block_torch_fallback_targets() -> None:
    """Hide ``fla``/``causal_conv1d`` so Transformers resolves torch fallback.

    Used for the controlled baseline-vs-fast-path A/B without pip churn.
    Must run before ``transformers`` Qwen3.5 modeling is imported.
    """
    for blocked in ("fla", "causal_conv1d"):
        sys.modules.pop(blocked, None)

        class _Blocker:
            def __init__(self, target: str) -> None:
                self.target = target

            def find_spec(  # type: ignore[no-untyped-def]
                self, name, path=None, target=None
            ):
                if name == self.target or name.startswith(self.target + "."):
                    raise ImportError(f"bench blocked import: {name}")
                return None

        sys.meta_path.insert(0, _Blocker(blocked))  # type: ignore[arg-type]
        sys.modules[blocked] = None  # type: ignore[assignment]


def run_bench(config: BenchConfig) -> dict[str, Any]:
    import torch
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from accelerate.utils import GradientAccumulationPlugin, GradScalerKwargs
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.utils.data import DataLoader

    if config.force_torch_fallback:
        _block_torch_fallback_targets()

    proc_start = time.perf_counter()
    compile_seconds = 0.0
    status = "pass"
    error_type: str | None = None
    error_message: str | None = None

    fsdp_strategy = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    }[config.fsdp_strategy]

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=fsdp_strategy,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Qwen3_5DecoderLayer"],
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        state_dict_type="FULL_STATE_DICT",
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )
    accumulation_plugin = GradientAccumulationPlugin(
        num_steps=config.gradient_accumulation,
        sync_each_batch=True,
    )
    scaler_kwargs = GradScalerKwargs(init_scale=256.0, growth_interval=2_000)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_plugin=accumulation_plugin,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[scaler_kwargs],
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    from tinycomplete.code_cpt.train import resolve_optional_hf_token

    token, _ = resolve_optional_hf_token()
    model, _tokenizer = _load_model_and_tokenizer(token)
    if config.attn_implementation:
        model.config._attn_implementation = config.attn_implementation
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    # Correctness gate: every parameter must stay trainable.
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise RuntimeError(f"frozen parameters violate full-weight rule: {frozen[:5]}")

    backend_probe = probe_gdn_backends(model)
    environment = collect_environment()

    fused_ce_loss, fused_ce_resolved = resolve_fused_ce(config.fused_ce)
    ce_parity_abs_diff: float | None = None

    block_path = config.corpus_dir / "train_blocks.npy"
    block_shape = np.load(block_path, mmap_mode="r").shape
    sequence_length = int(block_shape[1])
    tokens_per_microstep = config.microbatch * sequence_length * accelerator.num_processes
    tokens_per_update = tokens_per_microstep * config.gradient_accumulation
    blocks_per_update = config.gradient_accumulation * config.microbatch * accelerator.num_processes
    available_blocks = int(block_shape[0]) - config.start_block
    steps = bounded_optimizer_steps(
        remaining_tokens=config.max_tokens,
        tokens_per_update=tokens_per_update,
        available_blocks=available_blocks,
        blocks_per_update=blocks_per_update,
    )
    if steps == 0:
        raise ValueError("token budget does not cover one complete optimizer update")
    training_tokens_target = steps * tokens_per_update
    dataset = PackedBlocksDataset(block_path, config.start_block, steps * blocks_per_update)
    loader_options: dict[str, Any] = {
        "batch_size": config.microbatch,
        "shuffle": False,
        "drop_last": True,
        "num_workers": config.workers,
        "pin_memory": True,
    }
    if config.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=config.prefetch_factor)
    loader: Any = DataLoader(dataset, **loader_options)  # type: ignore[arg-type]
    optimizer = _optimizer(model, config.optimizer, config.learning_rate, weight_decay=0.01)
    scheduler = _constant_with_warmup(optimizer, config.warmup_steps)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if config.torch_compile:
        compile_start = time.perf_counter()
        model = torch.compile(model)  # type: ignore[assignment]
        compile_seconds = time.perf_counter() - compile_start

    # Weight-update probe: sample one full-precision scalar before training.
    with torch.no_grad():
        probe_before = None
        for param in accelerator.unwrap_model(model).parameters():
            probe_before = float(param.detach().float().flatten()[0].item())
            break

    config.output.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    data_wait_seconds = 0.0
    forward_seconds = 0.0
    backward_seconds = 0.0
    optimizer_seconds = 0.0
    first_update_seconds: float | None = None
    optimizer_steps = 0
    microsteps = 0
    training_tokens = 0
    losses: list[float] = []
    grad_norms: list[float] = []
    loss_scale_overflows = 0
    nan_or_inf = False
    loss_start: float | None = None
    loss_end: float | None = None

    iterator = iter(loader)
    try:
        while optimizer_steps < steps:
            wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                status = "error"
                error_type = "CorpusExhausted"
                error_message = "corpus exhausted before token target"
                break
            data_wait_seconds += time.perf_counter() - wait_start
            if config.sync_cleanup:
                local_tokens = config.microbatch * sequence_length
            else:
                local_tokens = int(batch["labels"].ne(-100).sum().item())

            t0 = time.perf_counter()
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    if fused_ce_loss is None:
                        output = model(**batch, use_cache=False)
                        loss = output.loss
                    else:
                        out = model(input_ids=batch["input_ids"], use_cache=False)
                        logits = out.logits.float()
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = batch["input_ids"][:, 1:].contiguous()
                        # Exact packed blocks: every position is a real target.
                        ce = fused_ce_loss(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        loss = ce
                        if ce_parity_abs_diff is None:
                            with torch.no_grad():
                                ref = torch.nn.functional.cross_entropy(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1),
                                )
                            ce_parity_abs_diff = float(abs(float(ce) - float(ref)))
                        output = out
                forward_seconds += time.perf_counter() - t0
                if not config.sync_cleanup:
                    finite = accelerator.reduce(torch.isfinite(loss).float(), reduction="min")
                    if not bool(finite.item()):
                        raise FloatingPointError("non-finite training loss")
                t1 = time.perf_counter()
                accelerator.backward(loss)
                backward_seconds += time.perf_counter() - t1
                grad_norm_value = None
                if accelerator.sync_gradients:
                    t2 = time.perf_counter()
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    assert grad_norm is not None
                    finite_grad = None
                    if not config.sync_cleanup:
                        finite_grad = accelerator.reduce(
                            torch.isfinite(grad_norm).float(), reduction="min"
                        )
                    grad_norm_value = float(grad_norm.item())
                    optimizer.step()
                    optimizer_seconds += time.perf_counter() - t2
                    skipped = bool(accelerator.optimizer_step_was_skipped)
                    if skipped:
                        loss_scale_overflows += 1
                        if loss_scale_overflows >= 8:
                            raise FloatingPointError("persistent FP16 gradient overflow")
                        optimizer.zero_grad(set_to_none=True)
                        microsteps += 1
                        training_tokens += local_tokens * accelerator.num_processes
                        del output, loss, batch
                        continue
                    if config.sync_cleanup:
                        grad_ok = bool(torch.isfinite(grad_norm).item())
                    else:
                        assert finite_grad is not None
                        grad_ok = bool(finite_grad.item())
                    if not grad_ok:
                        raise FloatingPointError("non-finite gradient escaped dynamic loss scaling")
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            microsteps += 1
            training_tokens += local_tokens * accelerator.num_processes
            if accelerator.sync_gradients:
                assert grad_norm_value is not None
                optimizer_steps += 1
                reduced = accelerator.reduce(loss.detach().float(), reduction="mean")
                loss_value = float(reduced.item())
                losses.append(loss_value)
                grad_norms.append(grad_norm_value)
                if loss_start is None:
                    loss_start = loss_value
                loss_end = loss_value
                if first_update_seconds is None:
                    first_update_seconds = time.perf_counter() - train_start
            del output, loss, batch
    except torch.OutOfMemoryError as exc:
        status = "oom"
        error_type = "CUDA_OOM"
        error_message = str(exc)[:500]
        nan_or_inf = False
    except FloatingPointError as exc:
        status = "error"
        error_type = "NumericalFailure"
        error_message = f"{type(exc).__name__}: {exc}"[:500]
        nan_or_inf = True
    except Exception as exc:  # noqa: BLE001 - failures are data
        status = "error"
        error_type = type(exc).__name__
        error_message = f"{type(exc).__name__}: {exc}"[:500]
        error_message += f" | {traceback.format_exc(limit=3)[-400:]}"

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - proc_start
    training_seconds = time.perf_counter() - train_start
    total_tokens_all_ranks = training_tokens
    # training_tokens already counts all ranks (local * world_size).
    tokens_per_second = total_tokens_all_ranks / training_seconds if training_seconds > 0 else 0.0
    if first_update_seconds and optimizer_steps > 1:
        steady_tokens = total_tokens_all_ranks - tokens_per_update
        steady_seconds = training_seconds - first_update_seconds
        steady_tps = steady_tokens / steady_seconds if steady_seconds > 0 else 0.0
    elif training_seconds > 0:
        steady_tokens = total_tokens_all_ranks
        steady_seconds = training_seconds
        steady_tps = tokens_per_second
    else:
        steady_tokens, steady_seconds, steady_tps = 0, 0.0, 0.0

    peak_allocated = peak_reserved = 0.0
    peak_allocated_by_gpu: list[float] = []
    peak_reserved_by_gpu: list[float] = []
    if torch.cuda.is_available():
        stats = accelerator.gather(
            torch.tensor(
                [
                    float(torch.cuda.max_memory_allocated() / 2**30),
                    float(torch.cuda.max_memory_reserved() / 2**30),
                    wall_seconds,
                    training_seconds,
                    data_wait_seconds,
                ],
                device=accelerator.device,
                dtype=torch.float64,
            )
        ).view(-1, 5)
        peak_allocated_by_gpu = [float(v) for v in stats[:, 0].tolist()]
        peak_reserved_by_gpu = [float(v) for v in stats[:, 1].tolist()]
        wall_seconds = float(stats[:, 2].max().item())
        training_seconds = float(stats[:, 3].max().item())
        data_wait_seconds = float(stats[:, 4].mean().item())
        peak_allocated = max(peak_allocated_by_gpu)
        peak_reserved = max(peak_reserved_by_gpu)

    probe_after = probe_before
    weights_updated = False
    try:
        with torch.no_grad():
            for param in accelerator.unwrap_model(model).parameters():
                probe_after = float(param.detach().float().flatten()[0].item())
                break
        weights_updated = (
            probe_before is not None
            and probe_after is not None
            and probe_before != probe_after
            and optimizer_steps > 0
        )
    except Exception:
        weights_updated = optimizer_steps > 0 and not nan_or_inf

    tokens_verified = total_tokens_all_ranks == optimizer_steps * tokens_per_update
    result: dict[str, Any] = {
        "name": config.name,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "training_tokens": total_tokens_all_ranks,
        "training_tokens_target": training_tokens_target,
        "tokens_verified": tokens_verified,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_target": steps,
        "microsteps": microsteps,
        "wall_seconds": wall_seconds,
        "training_seconds": training_seconds,
        "compile_seconds": compile_seconds,
        "steady_state_seconds": steady_seconds,
        "steady_state_tokens": steady_tokens,
        "tokens_per_second": tokens_per_second,
        "steady_state_tokens_per_second": steady_tps,
        "speedup_vs_baseline": (steady_tps / BASELINE_TOKENS_PER_SECOND if steady_tps else 0.0),
        "peak_allocated_vram_gib": peak_allocated_by_gpu,
        "peak_reserved_vram_gib": peak_reserved_by_gpu,
        "peak_allocated_vram_gib_max": peak_allocated,
        "peak_reserved_vram_gib_max": peak_reserved,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "gradient_norm_mean": (float(sum(grad_norms) / len(grad_norms)) if grad_norms else None),
        "gradient_norm_max": float(max(grad_norms)) if grad_norms else None,
        "loss_scale_overflows": loss_scale_overflows,
        "nan_or_inf": nan_or_inf,
        "data_wait_percent": (
            100.0 * data_wait_seconds / training_seconds if training_seconds else 0.0
        ),
        "data_wait_seconds": data_wait_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "gdn_backend": ("fast" if environment.get("fla_available") else "torch_fallback"),
        "causal_conv_backend": (
            "causal_conv1d" if environment.get("causal_conv1d_available") else "torch_fallback"
        ),
        "attention_backend": backend_probe.get("attn_implementation"),
        "fused_cross_entropy": fused_ce_resolved,
        "ce_parity_abs_diff": ce_parity_abs_diff,
        "gradient_checkpointing": config.gradient_checkpointing,
        "fsdp_strategy": config.fsdp_strategy,
        "optimizer": config.optimizer,
        "microbatch_per_gpu": config.microbatch,
        "gradient_accumulation": config.gradient_accumulation,
        "sequence_length": sequence_length,
        "tokens_per_update": tokens_per_update,
        "torch_compile": config.torch_compile,
        "sync_cleanup": config.sync_cleanup,
        "force_torch_fallback": config.force_torch_fallback,
        "weights_updated": weights_updated,
        "world_size": accelerator.num_processes,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "backend_probe": backend_probe,
        "environment": environment,
        "bench_config": {
            **asdict(config),
            "corpus_dir": str(config.corpus_dir),
            "output": str(config.output),
        },
    }
    if accelerator.is_main_process:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench")
    bench.add_argument("--corpus-dir", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--name", type=str, default="candidate")
    bench.add_argument("--learning-rate", type=float, default=3e-6)
    bench.add_argument("--max-tokens", type=int, default=STAGE_A_TOKENS)
    bench.add_argument("--start-block", type=int, default=0)
    bench.add_argument("--seed", type=int, default=271828)
    bench.add_argument("--microbatch", type=int, default=1)
    bench.add_argument("--gradient-accumulation", type=int, default=8)
    bench.add_argument("--optimizer", default="adamw_8bit")
    bench.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    bench.add_argument(
        "--fsdp-strategy", default="FULL_SHARD", choices=("FULL_SHARD", "SHARD_GRAD_OP")
    )
    bench.add_argument("--attn-implementation", default="sdpa")
    bench.add_argument("--fused-ce", default="none", choices=("none", "liger", "auto"))
    bench.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument("--sync-cleanup", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument("--workers", type=int, default=1)
    bench.add_argument("--prefetch-factor", type=int, default=2)
    bench.add_argument("--warmup-steps", type=int, default=5)
    bench.add_argument(
        "--force-torch-fallback", action=argparse.BooleanOptionalAction, default=False
    )
    bench.add_argument("--profile-steps", type=int, default=0)
    probe = sub.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    return parser


def run_probe(output: Path) -> dict[str, Any]:
    from tinycomplete.code_cpt.train import resolve_optional_hf_token

    token, _ = resolve_optional_hf_token()
    model, _ = _load_model_and_tokenizer(token)
    result = {
        "environment": collect_environment(),
        "backend_probe": probe_gdn_backends(model),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "probe":
        run_probe(args.output)
        return
    config = BenchConfig(
        corpus_dir=args.corpus_dir,
        output=args.output,
        name=args.name,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        start_block=args.start_block,
        seed=args.seed,
        microbatch=args.microbatch,
        gradient_accumulation=args.gradient_accumulation,
        optimizer=args.optimizer,
        gradient_checkpointing=args.gradient_checkpointing,
        fsdp_strategy=args.fsdp_strategy,
        attn_implementation=args.attn_implementation,
        fused_ce=args.fused_ce,
        torch_compile=args.torch_compile,
        sync_cleanup=args.sync_cleanup,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        warmup_steps=args.warmup_steps,
        force_torch_fallback=args.force_torch_fallback,
        profile_steps=args.profile_steps,
    )
    try:
        result = run_bench(config)
    except Exception as exc:  # noqa: BLE001 - failures are data
        result = {
            "name": config.name,
            "status": "oom" if "out of memory" in str(exc).lower() else "error",
            "error_type": type(exc).__name__,
            "error_message": f"{type(exc).__name__}: {exc}"[:500],
            "training_tokens": 0,
            "optimizer_steps": 0,
            "bench_config": {
                **asdict(config),
                "corpus_dir": str(config.corpus_dir),
                "output": str(config.output),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
