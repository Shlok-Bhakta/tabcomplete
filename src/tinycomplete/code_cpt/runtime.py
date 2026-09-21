"""Shared verified T4x2 FSDP runtime construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionRuntime:
    gradient_accumulation: int = 8
    backward_prefetch: str = "BACKWARD_PRE"
    forward_prefetch: bool = True
    compile_mode: str = "default"
    compile_dynamic: bool = False
    initial_loss_scale: float = 256.0


def build_production_accelerator(config: ProductionRuntime):
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

    prefetch = {"BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE}.get(config.backward_prefetch)
    if prefetch is None:
        raise ValueError(f"unsupported production backward prefetch: {config.backward_prefetch}")
    plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Qwen3_5DecoderLayer"],
        backward_prefetch=prefetch,
        forward_prefetch=config.forward_prefetch,
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
    accumulation = GradientAccumulationPlugin(
        num_steps=config.gradient_accumulation,
        sync_each_batch=True,
    )
    scaler = GradScalerKwargs(
        init_scale=config.initial_loss_scale,
        growth_interval=2_000,
    )
    return Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_plugin=accumulation,
        fsdp_plugin=plugin,
        kwargs_handlers=[scaler],
    )


def compile_production_model(model, config: ProductionRuntime):
    import torch

    return torch.compile(
        model,
        mode=config.compile_mode,
        dynamic=config.compile_dynamic,
        fullgraph=False,
    )
