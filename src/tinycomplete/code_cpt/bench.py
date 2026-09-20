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
from dataclasses import asdict, dataclass, field
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
# Deterministic first-update loss of every FLA+stock candidate at seed 271828
# (identical token ordering). Parity anchor for gate-fusion experiments.
FLA_STOCK_LOSS_START = 1.3077329397201538


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
    try:
        env["nccl_version"] = ".".join(str(v) for v in torch.cuda.nccl.version())
    except Exception:
        env["nccl_version"] = None
    try:
        if torch.cuda.device_count() >= 2:
            env["gpu_p2p"] = bool(
                torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)
            )
        else:
            env["gpu_p2p"] = None
    except Exception:
        env["gpu_p2p"] = None
    return env


def resolve_fused_ce(mode: str):
    """Resolve a memory-saving full-vocabulary linear-CE implementation.

    Modes: ``none``/``stock`` (full-logits stock CE), ``chunked``
    (sequence-chunked stock ops through the FSDP-managed lm_head — exact,
    never materializes full logits), ``liger`` (true fused linear CE over
    decoder hidden states plus the lm_head weight), ``auto`` (liger when
    importable, else chunked). All modes preserve the exact full-vocabulary
    causal LM objective; the worker records a loss-parity check.
    """
    if mode in ("none", "stock"):
        return None, "stock"
    if mode == "chunked":
        return None, "chunked"
    errors: dict[str, str] = {}
    if mode in ("liger", "auto"):
        for path in (
            "liger_kernel.transformers:LigerFusedLinearCrossEntropyLoss",
            "liger_kernel.chunked_loss:LigerFusedLinearCrossEntropyLoss",
        ):
            module_name, attr = path.split(":")
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attr)(), "liger"
            except Exception as exc:
                errors[path] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if mode == "liger":
        raise ImportError(f"liger fused CE unavailable: {errors}")
    return None, "chunked"


def _summoned_decoder_hidden(model, batch):
    """Run the decoder stack with all FSDP params summoned.

    Calling the decoder submodule directly does NOT fire the outer FSDP
    unit's all-gather hook, so outer-unit params (embed_tokens) stay sharded.
    Callers must already hold ``summon_full_params`` for this to work.
    """
    decoder = getattr(model, "model", None)
    if decoder is None:
        raise RuntimeError("causal model exposes no .model decoder stack")
    return decoder(input_ids=batch["input_ids"], use_cache=False)


def _check_full_weight(weight, label: str) -> None:
    if weight is None or weight.dim() != 2:
        raise RuntimeError(
            f"summon failed: {label} weight dim "
            f"{None if weight is None else weight.dim()}, expected 2-D full params"
        )


def _chunked_fused_step(accelerator, model, batch, *, chunk=1024, checkpointing=False):
    """One exact full-vocab causal CE microstep without full logits.

    Holds ``summon_full_params`` across the decoder forward and the
    sequence-chunked lm_head loop (per-chunk forward/backward, chunk logits
    freed immediately). The full model is transiently materialized; gradients
    are applied inside the context. Requires gradient checkpointing OFF:
    per-chunk backward with retain_graph is incompatible with checkpoint
    recompute. Returns ``(mean_loss_value, backward_seconds, hidden_out)``.
    """
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    backward_seconds = 0.0
    if checkpointing:
        raise RuntimeError(
            "chunked CE requires gradient checkpointing OFF: per-chunk "
            "backward with retain_graph is incompatible with checkpoint "
            "recompute (shape mismatch in recompute_fn)"
        )
    with FSDP.summon_full_params(model, recurse=True, writeback=False):
        hidden_out = _summoned_decoder_hidden(model, batch)
        hidden = hidden_out.last_hidden_state
        shift_h = hidden[:, :-1, :].contiguous()
        shift_l = batch["input_ids"][:, 1:].contiguous()
        lm_head = model.lm_head
        _check_full_weight(getattr(lm_head, "weight", None), "lm_head")
        denom = shift_l.numel()
        spans = list(range(0, shift_h.size(1), chunk))
        running: torch.Tensor | None = None
        for index, start in enumerate(spans):
            logits = lm_head(shift_h[:, start : start + chunk, :])
            part = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_l[:, start : start + chunk].reshape(-1),
                reduction="sum",
            )
            running = part.detach() if running is None else running + part.detach()
            tick = time.perf_counter()
            accelerator.backward(part / denom, retain_graph=index < len(spans) - 1)
            backward_seconds += time.perf_counter() - tick
            del logits, part
    if running is None:
        raise RuntimeError("empty hidden states for chunked CE")
    return running / denom, backward_seconds, hidden_out


def _liger_fused_step(accelerator, model, liger_loss, batch):
    """One true-fused-linear-CE microstep over hidden states + lm_head weight.

    Same summon discipline as :func:`_chunked_fused_step`. Returns
    ``(loss, backward_seconds, hidden_out)``.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    tick = time.perf_counter()
    with FSDP.summon_full_params(model, recurse=True, writeback=False):
        hidden_out = _summoned_decoder_hidden(model, batch)
        hidden = hidden_out.last_hidden_state
        shift_h = hidden[:, :-1, :].contiguous()
        shift_l = batch["input_ids"][:, 1:].contiguous()
        embeddings = model.get_output_embeddings()
        weight = getattr(embeddings, "weight", None)
        _check_full_weight(weight, "output-embedding")
        loss = liger_loss(weight, shift_h.reshape(-1, shift_h.size(-1)), shift_l.reshape(-1))
        accelerator.backward(loss)
    return loss, time.perf_counter() - tick, hidden_out


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
    # --- Round-2 knobs ---
    compile_placement: str = "post_fsdp"  # post_fsdp | pre_fsdp
    compile_mode: str = "default"  # default | reduce-overhead | max-autotune | ...
    compile_dynamic: bool = True
    compile_fullgraph: bool = False
    inductor_options: dict[str, Any] = field(default_factory=dict)
    fsdp_backward_prefetch: str = "default"  # default | BACKWARD_PRE | BACKWARD_POST
    fsdp_forward_prefetch: bool = False
    limit_all_gathers: bool = True
    fsdp_group_layers: int = 1  # adjacent decoder layers per FSDP unit
    grad_sync_every: int = 1  # 1 = every microstep; 8 = accumulation boundary only
    liger_rmsnorm: bool = False
    liger_swiglu: bool = False
    fla_gate_fusion: int = 0  # 0 off | 2 gate in-kernel | 3 gate+beta-sigmoid in-kernel


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


def _validated_inductor_options(options: dict[str, Any]) -> dict[str, Any]:
    """Validate Inductor option names against the installed PyTorch.

    Never invent option names: unknown keys raise with the valid name list.
    """
    import torch._inductor

    valid = set(torch._inductor.list_options())
    unknown = [key for key in options if key not in valid]
    if unknown:
        sample = sorted(valid)[:40]
        raise ValueError(f"unknown inductor options {unknown}; e.g. {sample}")
    return dict(options)


def _dynamo_counters_snapshot() -> dict[str, Any]:
    """Best-effort TorchDynamo diagnostics (graph breaks, recompiles)."""
    try:
        from torch._dynamo.utils import counters

        graph_breaks: dict[Any, Any] = dict(counters.get("graph_break", {}) or {})
        stats: dict[Any, Any] = dict(counters.get("stats", {}) or {})
        return {
            "graph_break_count": int(sum(graph_breaks.values())) if graph_breaks else 0,
            "graph_break_reasons": dict(list(graph_breaks.items())[:10]),
            "unique_graphs": stats.get("unique_graphs"),
            "graph_calls": stats.get("graph_calls"),
            "stats_keys": sorted(str(k) for k in stats)[:20],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _allocator_snapshot() -> dict[str, Any]:
    """Allocator headroom/fragmentation counters (no profiler needed)."""
    import torch

    if not torch.cuda.is_available():
        return {}
    try:
        stats = torch.cuda.memory_stats()
        get = lambda k: int(stats.get(k, 0))  # noqa: E731
        gib = 2**30
        return {
            "allocated_gib": get("allocated_bytes.all.current") / gib,
            "reserved_gib": get("reserved_bytes.all.current") / gib,
            "inactive_split_gib": get("inactive_split_bytes.all.current") / gib,
            "num_alloc_retries": get("num_alloc_retries"),
            "num_ooms": get("num_ooms"),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _compile_model(model, config: BenchConfig):
    """Apply torch.compile with the candidate's mode/options.

    Returns ``(model, compile_seconds, dynamo_info)``.
    """
    import torch

    options = _validated_inductor_options(dict(config.inductor_options or {}))
    kwargs: dict[str, Any] = {
        "mode": config.compile_mode,
        "dynamic": config.compile_dynamic,
        "fullgraph": config.compile_fullgraph,
    }
    if options:
        kwargs["options"] = options
    start = time.perf_counter()
    compiled = torch.compile(model, **kwargs)
    seconds = time.perf_counter() - start
    return compiled, seconds, _dynamo_counters_snapshot()


def _apply_liger_non_ce(model, config: BenchConfig) -> dict[str, Any]:
    """Apply the official Liger Qwen3.5 patch WITHOUT fused linear CE.

    Only RMSNorm and/or SwiGLU triton kernels per candidate flags. Stock CE
    and the FSDP model forward are untouched.
    """
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

    apply_liger_kernel_to_qwen3_5(
        rms_norm=bool(config.liger_rmsnorm),
        swiglu=bool(config.liger_swiglu),
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        model=model,
    )
    return {"rmsnorm": bool(config.liger_rmsnorm), "swiglu": bool(config.liger_swiglu)}


def _fla_gate_fusion_forward(self, hidden_states, cache_params=None, attention_mask=None):
    """Transformers 5.5.0 Qwen3_5GatedDeltaNet.forward with in-kernel gating.

    Byte-faithful copy of the 5.5.0 body except the ``beta``/``g`` pointwise
    math, which moves into the FLA kernel when ``_bench_gate_fusion_level`` is
    2 (gate) or 3 (gate + beta sigmoid). Mathematical operation preserved.
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_mask_to_padding_states,
    )

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state(self.layer_idx)
        and seq_len == 1
    )
    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states
    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    level = int(getattr(self, "_bench_gate_fusion_level", 0))
    extra: dict[str, Any] = {}
    if level >= 2:
        g = a
        extra.update(use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias)
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if level >= 3:
        beta = b
        extra.update(use_beta_sigmoid_in_kernel=True)
    else:
        beta = b.sigmoid()
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            **extra,
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            **extra,
        )
    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def _set_gate_fusion(model, level: int | None) -> int:
    """Patch/unpatch every GatedDeltaNet instance forward. Returns count."""
    import types

    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3_5GatedDeltaNet":
            continue
        if level is None:
            original = getattr(module, "_bench_orig_forward", None)
            if original is not None:
                module.forward = original
                del module._bench_orig_forward
                if hasattr(module, "_bench_gate_fusion_level"):
                    del module._bench_gate_fusion_level
        else:
            if "_bench_orig_forward" not in module.__dict__:
                module._bench_orig_forward = module.forward
            module._bench_gate_fusion_level = int(level)
            module.forward = types.MethodType(_fla_gate_fusion_forward, module)
        count += 1
    return count


def _apply_wrap_groups(model, group_size: int) -> dict[str, Any]:
    """Regroup adjacent decoder layers into coarser FSDP units.

    Replaces ``model.model.layers`` (24 singles) with ``24/group_size``
    transparent containers forwarding all args, so forward math is exactly
    identical. Verifies the state-dict key remap is a bijection with the
    original keys (recoverable for production save conversion).
    """
    import torch.nn as nn

    class LayerGroup(nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            for layer in self.layers:
                hidden_states = layer(hidden_states, *args, **kwargs)
            return hidden_states

    stack = model.model.layers
    total = len(stack)
    if group_size <= 1:
        return {"group_layers": 1, "fsdp_units": total, "key_remap_verified": True}
    if total % group_size:
        raise ValueError(f"{total} layers not divisible by group size {group_size}")
    original_keys = {k for k in model.state_dict()}
    groups = []
    for start in range(0, total, group_size):
        group = LayerGroup()
        group.layers = nn.ModuleList(list(stack[start : start + group_size]))
        groups.append(group)
    model.model.layers = nn.ModuleList(groups)

    def remap(key: str) -> str:
        # model.layers.{g}.layers.{j}.{rest} -> model.layers.{g*n+j}.{rest}
        parts = key.split(".")
        li = parts.index("layers")
        group_index = int(parts[li + 1])
        inner_index = int(parts[li + 3])
        flat = group_index * group_size + inner_index
        return ".".join([*parts[:li], "layers", str(flat), *parts[li + 4 :]])

    remapped = {remap(k) for k in model.state_dict()}
    if remapped != original_keys:
        raise RuntimeError("FSDP group key remap is not a bijection with original keys")
    return {
        "group_layers": group_size,
        "fsdp_units": len(groups),
        "key_remap_verified": True,
        "original_key_count": len(original_keys),
    }


def run_bench(config: BenchConfig) -> dict[str, Any]:
    import torch
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from accelerate.utils import GradientAccumulationPlugin, GradScalerKwargs
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.utils.data import DataLoader

    if config.fused_ce not in ("none", "stock"):
        # PROVEN INCOMPATIBLE (Stage-A v6/v7): under FSDP1 full-weight
        # training, decoder-direct calls bypass the outer unit's all-gather
        # hook, and summon_full_params + backward-inside leaves full-shaped
        # .grads that trip "Cannot writeback when the gradient shape changes"
        # on the next FSDP forward. Fail before model load. Non-stock CE
        # needs FSDP2/DTensor (untried).
        raise RuntimeError(
            "non-stock CE is incompatible with FSDP1 full-weight training "
            "(summon grad writeback failure); use stock CE"
        )
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
    if config.fsdp_backward_prefetch not in ("default", "BACKWARD_PRE", "BACKWARD_POST"):
        raise ValueError(f"bad fsdp_backward_prefetch: {config.fsdp_backward_prefetch}")
    if config.grad_sync_every not in (1, 8):
        raise ValueError("grad_sync_every must be 1 or 8 to preserve update geometry")
    if config.fsdp_group_layers not in (1, 2, 3, 4):
        raise ValueError(f"bad fsdp_group_layers: {config.fsdp_group_layers}")
    if config.fla_gate_fusion not in (0, 2, 3):
        raise ValueError(f"bad fla_gate_fusion: {config.fla_gate_fusion}")
    if config.compile_placement not in ("post_fsdp", "pre_fsdp"):
        raise ValueError(f"bad compile_placement: {config.compile_placement}")

    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    wrap_targets = ["LayerGroup"] if config.fsdp_group_layers > 1 else ["Qwen3_5DecoderLayer"]
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=fsdp_strategy,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=wrap_targets,
        backward_prefetch=prefetch_map.get(config.fsdp_backward_prefetch),
        forward_prefetch=config.fsdp_forward_prefetch,
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
        limit_all_gathers=config.limit_all_gathers,
    )
    accumulation_plugin = GradientAccumulationPlugin(
        num_steps=config.gradient_accumulation,
        sync_each_batch=config.grad_sync_every == 1,
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
    wrap_info = _apply_wrap_groups(model, config.fsdp_group_layers)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    liger_info: dict[str, Any] = {"rmsnorm": False, "swiglu": False}
    if config.liger_rmsnorm or config.liger_swiglu:
        liger_info = _apply_liger_non_ce(model, config)
    gate_fusion_layers = 0
    if config.fla_gate_fusion:
        if config.force_torch_fallback:
            raise RuntimeError("fla gate fusion needs the FLA fast path, not torch fallback")
        gate_fusion_layers = _set_gate_fusion(model, config.fla_gate_fusion)
        if gate_fusion_layers == 0:
            raise RuntimeError("no Qwen3_5GatedDeltaNet layers found for gate fusion")

    # Correctness gate: every parameter must stay trainable.
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise RuntimeError(f"frozen parameters violate full-weight rule: {frozen[:5]}")

    backend_probe = probe_gdn_backends(model)
    environment = collect_environment()

    fused_ce_loss, fused_ce_mode = resolve_fused_ce(config.fused_ce)
    ce_parity_abs_diff: float | None = None
    ce_fallback_note: str | None = None
    assert fused_ce_mode == "stock"  # non-stock rejected at entry

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
    dynamo_info: dict[str, Any] = {}
    if config.torch_compile and config.compile_placement == "pre_fsdp":
        model, compile_seconds, dynamo_info = _compile_model(model, config)
    else:
        compile_seconds = 0.0
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if config.torch_compile and config.compile_placement == "post_fsdp":
        model, compile_seconds, dynamo_info = _compile_model(model, config)

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

    profiler = None
    profile_top_cuda_ops: str | None = None
    profile_error: str | None = None
    if config.profile_steps > 0:
        try:
            from torch.profiler import ProfilerActivity, profile

            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
            )
            profiler.__enter__()
        except Exception as exc:
            profile_error = f"{type(exc).__name__}: {exc}"[:300]
            profiler = None

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
                    pre_backwarded = False
                    inner_backward_seconds = 0.0
                    if fused_ce_mode == "stock":
                        output = model(**batch, use_cache=False)
                        loss = output.loss
                    else:
                        # Bypass the lm_head: decoder hidden states feed a
                        # chunked/fused full-vocabulary causal CE, so full
                        # logits are never materialized. Everything runs
                        # inside summon_full_params (see helpers): calling the
                        # decoder submodule directly would not fire the outer
                        # FSDP unit's all-gather hook.
                        if config.torch_compile:
                            raise RuntimeError(
                                "non-stock CE crossed with torch.compile is "
                                "untested: summon/decoder-direct bypasses the "
                                "compiled graph"
                            )
                        if fused_ce_mode == "liger":
                            try:
                                loss, inner_backward_seconds, output = _liger_fused_step(
                                    accelerator, model, fused_ce_loss, batch
                                )
                            except Exception as exc:
                                if config.fused_ce != "auto":
                                    raise
                                ce_fallback_note = (
                                    f"liger_failed_then_chunked: {type(exc).__name__}"
                                )
                                fused_ce_mode = "chunked"
                                loss, inner_backward_seconds, output = _chunked_fused_step(
                                    accelerator,
                                    model,
                                    batch,
                                    checkpointing=config.gradient_checkpointing,
                                )
                            pre_backwarded = True
                        else:
                            loss, inner_backward_seconds, output = _chunked_fused_step(
                                accelerator,
                                model,
                                batch,
                                checkpointing=config.gradient_checkpointing,
                            )
                            pre_backwarded = True
                        if ce_parity_abs_diff is None:
                            with torch.no_grad():
                                ref_out = model(**batch, use_cache=False)
                                ref_loss = ref_out.loss.detach().float()
                            ce_parity_abs_diff = float(
                                abs(loss.detach().float().item() - ref_loss.item())
                            )
                            del ref_out
                forward_seconds += time.perf_counter() - t0 - inner_backward_seconds
                if not config.sync_cleanup:
                    finite = accelerator.reduce(torch.isfinite(loss).float(), reduction="min")
                    if not bool(finite.item()):
                        raise FloatingPointError("non-finite training loss")
                t1 = time.perf_counter()
                if not pre_backwarded:
                    accelerator.backward(loss)
                backward_seconds += time.perf_counter() - t1 + inner_backward_seconds
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
                if profiler is not None and optimizer_steps >= config.profile_steps:
                    try:
                        profiler.__exit__(None, None, None)
                        table = profiler.key_averages().table(
                            sort_by="cuda_time_total", row_limit=20
                        )
                        profile_top_cuda_ops = table[:6000]
                    except Exception as exc:
                        profile_error = f"{type(exc).__name__}: {exc}"[:300]
                    profiler = None
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
        # Full traceback: truncated tails hid the raising bench.py frame twice.
        error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"[:8000]

    if profiler is not None:
        try:
            profiler.__exit__(None, None, None)
            table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            profile_top_cuda_ops = table[:6000]
        except Exception as exc:
            profile_error = f"{type(exc).__name__}: {exc}"[:300]
        profiler = None

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
    dynamo_final = _dynamo_counters_snapshot()
    allocator_final = _allocator_snapshot()
    nccl_env = {
        key: os.environ.get(key)
        for key in (
            "NCCL_PROTO",
            "NCCL_ALGO",
            "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS",
            "TORCH_NCCL_HIGH_PRIORITY",
            "NCCL_DEBUG",
        )
    }
    # Deterministic parity anchor: every FLA+stock candidate with seed 271828
    # sees identical first batches, so loss_start must equal the control value
    # below. Used to validate gate-fusion math without disturbing compile.
    fla_parity_abs_diff = None
    if config.fla_gate_fusion and loss_start is not None:
        fla_parity_abs_diff = abs(loss_start - FLA_STOCK_LOSS_START)
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
        "fused_cross_entropy": fused_ce_mode,
        "ce_fallback_note": ce_fallback_note,
        "profile_top_cuda_ops": profile_top_cuda_ops,
        "profile_error": profile_error,
        "ce_parity_abs_diff": ce_parity_abs_diff,
        "gradient_checkpointing": config.gradient_checkpointing,
        "fsdp_strategy": config.fsdp_strategy,
        "optimizer": config.optimizer,
        "microbatch_per_gpu": config.microbatch,
        "gradient_accumulation": config.gradient_accumulation,
        "sequence_length": sequence_length,
        "tokens_per_update": tokens_per_update,
        "torch_compile": config.torch_compile,
        "compile_placement": config.compile_placement if config.torch_compile else None,
        "compile_mode": config.compile_mode if config.torch_compile else None,
        "compile_dynamic": config.compile_dynamic if config.torch_compile else None,
        "compile_fullgraph": config.compile_fullgraph,
        "inductor_options": dict(config.inductor_options or {}),
        "graph_break_count": dynamo_final.get("graph_break_count"),
        "graph_break_reasons": dynamo_final.get("graph_break_reasons"),
        "unique_graphs": dynamo_final.get("unique_graphs"),
        "recompile_graph_calls": dynamo_final.get("graph_calls"),
        "fsdp_version": 1,
        "fsdp_backward_prefetch": config.fsdp_backward_prefetch,
        "fsdp_forward_prefetch": config.fsdp_forward_prefetch,
        "limit_all_gathers": config.limit_all_gathers,
        "fsdp_group_layers": config.fsdp_group_layers,
        "fsdp_units": wrap_info.get("fsdp_units"),
        "key_remap_verified": wrap_info.get("key_remap_verified"),
        "reshard_after_forward": config.fsdp_strategy == "FULL_SHARD",
        "grad_sync_every": config.grad_sync_every,
        "nccl_version": environment.get("nccl_version"),
        "nccl_algo": nccl_env.get("NCCL_ALGO") or "default",
        "nccl_proto": nccl_env.get("NCCL_PROTO") or "default",
        "nccl_min_channels": nccl_env.get("NCCL_MIN_NCHANNELS"),
        "nccl_high_priority": nccl_env.get("TORCH_NCCL_HIGH_PRIORITY"),
        "gpu_p2p": environment.get("gpu_p2p"),
        "allgather_calls": None,
        "allgather_cuda_seconds": None,
        "reduce_scatter_calls": None,
        "reduce_scatter_cuda_seconds": None,
        "liger_rmsnorm": liger_info.get("rmsnorm", False),
        "liger_swiglu": liger_info.get("swiglu", False),
        "liger_fused_ce": False,
        "fla_gate_fused": config.fla_gate_fusion >= 2,
        "fla_beta_sigmoid_fused": config.fla_gate_fusion >= 3,
        "fla_fusion_layers": gate_fusion_layers,
        "fla_parity_abs_diff": fla_parity_abs_diff,
        "allocator": allocator_final,
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
    bench.add_argument("--fused-ce", default="none", choices=("none", "liger", "auto", "chunked"))
    bench.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument("--sync-cleanup", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument("--workers", type=int, default=1)
    bench.add_argument("--prefetch-factor", type=int, default=2)
    bench.add_argument("--warmup-steps", type=int, default=5)
    bench.add_argument(
        "--force-torch-fallback", action=argparse.BooleanOptionalAction, default=False
    )
    bench.add_argument("--profile-steps", type=int, default=0)
    bench.add_argument(
        "--compile-placement", default="post_fsdp", choices=("post_fsdp", "pre_fsdp")
    )
    bench.add_argument("--compile-mode", default="default")
    bench.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    bench.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument(
        "--inductor-option",
        action="append",
        default=[],
        help="key=value pair passed to torch.compile options (repeatable)",
    )
    bench.add_argument(
        "--fsdp-backward-prefetch",
        default="default",
        choices=("default", "BACKWARD_PRE", "BACKWARD_POST"),
    )
    bench.add_argument(
        "--fsdp-forward-prefetch", action=argparse.BooleanOptionalAction, default=False
    )
    bench.add_argument("--limit-all-gathers", action=argparse.BooleanOptionalAction, default=True)
    bench.add_argument("--fsdp-group-layers", type=int, default=1)
    bench.add_argument("--grad-sync-every", type=int, default=1)
    bench.add_argument("--liger-rmsnorm", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument("--liger-swiglu", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument("--fla-gate-fusion", type=int, default=0, choices=(0, 2, 3))
    probe = sub.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    topo = sub.add_parser("topo")
    topo.add_argument("--output", type=Path, required=True)
    compiler_probe = sub.add_parser("compiler_probe")
    compiler_probe.add_argument("--output", type=Path, required=True)
    nccl_bench = sub.add_parser("nccl_bench")
    nccl_bench.add_argument("--output", type=Path, required=True)
    nccl_bench.add_argument("--iters", type=int, default=20)
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


def run_topo(output: Path) -> dict[str, Any]:
    """Single-process GPU/PCIe/NCCL topology probe (Step 0-B)."""
    import torch

    result: dict[str, Any] = {"environment": collect_environment()}
    for command in (["nvidia-smi", "topo", "-m"], ["nvidia-smi", "-q"]):
        try:
            proc = __import__("subprocess").run(command, text=True, capture_output=True, timeout=60)
            result[" ".join(command[1:])] = (proc.stdout + proc.stderr)[-6000:]
        except Exception as exc:
            result[" ".join(command[1:])] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        import torch.distributed as dist

        result["dist_available"] = dist.is_available()
        result["nccl_built"] = dist.is_nccl_available()
    except Exception as exc:
        result["dist_probe_error"] = f"{type(exc).__name__}: {exc}"[:200]
    # Discover whether torch supports a high-priority NCCL stream knob.
    try:
        import pathlib

        torch_src = pathlib.Path(torch.__file__).parent
        hits = []
        for path in list(torch_src.rglob("*.py"))[:4000]:
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            if "HIGH_PRIORITY" in text and "nccl" in text.lower():
                hits.append(str(path.relative_to(torch_src)))
                if len(hits) >= 5:
                    break
        result["high_priority_knob_files"] = hits
    except Exception as exc:
        result["high_priority_probe_error"] = f"{type(exc).__name__}: {exc}"[:200]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def run_compiler_probe(output: Path) -> dict[str, Any]:
    """List installed compiler modes/options without compiling a model."""
    import torch

    result: dict[str, Any] = {"environment": collect_environment()}
    try:
        modes = torch._inductor.list_mode_options()
        result["inductor_modes"] = sorted(str(m) for m in modes)[:20]
    except Exception as exc:
        result["inductor_modes_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        options = torch._inductor.list_options()
        result["inductor_option_names"] = sorted(str(o) for o in options)
    except Exception as exc:
        result["inductor_options_error"] = f"{type(exc).__name__}: {exc}"[:200]
    result["dynamo_counters"] = _dynamo_counters_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def run_nccl_bench(output: Path, iters: int = 20) -> dict[str, Any]:
    """Two-process NCCL all_gather/reduce_scatter latency microbenchmark.

    Must run under ``torch.distributed.run --nproc_per_node=2``. Tensor sizes
    cover representative FSDP unit payloads (decoder layer ~60 MB fp16,
    lm_head ~500 MB fp16).
    """
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2 or not torch.cuda.is_available():
        raise RuntimeError("nccl_bench requires exactly 2 CUDA devices")
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    sizes = {
        "8MB": 4_194_304,
        "62MB_layer": 32_505_856,
        "256MB": 134_217_728,
        "508MB_lmhead": 266_338_304,
    }
    measurements: dict[str, Any] = {}
    for label, numel in sizes.items():
        for collective in ("all_gather", "reduce_scatter"):
            try:
                if collective == "all_gather":
                    src = torch.randn(numel // 2, dtype=torch.float16, device=device)
                    dst = torch.empty(numel, dtype=torch.float16, device=device)

                    def fn(s=src, d=dst):
                        dist.all_gather_into_tensor(d, s)

                    out_numel = numel
                else:
                    src = torch.randn(numel, dtype=torch.float16, device=device)
                    dst = torch.empty(numel // 2, dtype=torch.float16, device=device)

                    def fn(s=src, d=dst):
                        dist.reduce_scatter_tensor(d, s)

                    out_numel = numel // 2
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize()
                seconds = (time.perf_counter() - start) / iters
                gib = out_numel * 2 / 2**30
                measurements[f"{collective}/{label}"] = {
                    "ms_per_op": seconds * 1000.0,
                    "effective_gbps": gib / seconds,
                }
            except Exception as exc:
                measurements[f"{collective}/{label}"] = {
                    "error": f"{type(exc).__name__}: {exc}"[:200]
                }
    result = {
        "rank": rank,
        "world_size": world,
        "iters": iters,
        "environment": collect_environment(),
        "measurements": measurements,
    }
    dist.barrier()
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    dist.destroy_process_group()
    return result


def _parse_inductor_options(pairs: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"inductor option must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        value: Any = raw
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        options[key.strip()] = value
    return options


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "probe":
        run_probe(args.output)
        return
    if args.command == "topo":
        run_topo(args.output)
        return
    if args.command == "compiler_probe":
        run_compiler_probe(args.output)
        return
    if args.command == "nccl_bench":
        run_nccl_bench(args.output, args.iters)
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
        compile_placement=args.compile_placement,
        compile_mode=args.compile_mode,
        compile_dynamic=args.compile_dynamic,
        compile_fullgraph=args.compile_fullgraph,
        inductor_options=_parse_inductor_options(args.inductor_option),
        fsdp_backward_prefetch=args.fsdp_backward_prefetch,
        fsdp_forward_prefetch=args.fsdp_forward_prefetch,
        limit_all_gathers=args.limit_all_gathers,
        fsdp_group_layers=args.fsdp_group_layers,
        grad_sync_every=args.grad_sync_every,
        liger_rmsnorm=args.liger_rmsnorm,
        liger_swiglu=args.liger_swiglu,
        fla_gate_fusion=args.fla_gate_fusion,
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
