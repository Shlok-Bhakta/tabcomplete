"""Compare two same-layout distributed training checkpoints semantically."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _local_tensor(value):
    import torch

    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        return to_local()
    local_shards = getattr(value, "local_shards", None)
    if callable(local_shards):
        shards = local_shards()
        if len(shards) != 1:
            raise ValueError(f"expected one local shard, got {len(shards)}")
        return shards[0].tensor
    if isinstance(value, torch.Tensor):
        return value
    return None


def flatten_tensor_state(value: Any, prefix: str = "") -> dict[str, Any]:
    """Return every tensor leaf under a stable slash-separated key."""
    tensor = _local_tensor(value)
    if tensor is not None:
        return {prefix: tensor}
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_tensor_state(value[key], child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}/{index}" if prefix else str(index)
            result.update(flatten_tensor_state(item, child))
        return result
    return {}


def compare_tensor_states(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Compare complete tensor inventories and report exact and numeric differences."""
    import torch

    if set(first) != set(second):
        missing = sorted(set(first) - set(second))
        added = sorted(set(second) - set(first))
        raise ValueError(f"tensor inventory differs; missing={missing[:5]}, added={added[:5]}")
    different_tensors = 0
    different_elements = 0
    total_elements = 0
    max_abs_difference = 0.0
    sum_abs_difference = 0.0
    for key in sorted(first):
        left = first[key].detach().to(device="cpu")
        right = second[key].detach().to(device="cpu")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(
                f"tensor metadata differs for {key}: "
                f"{tuple(left.shape)}/{left.dtype} vs {tuple(right.shape)}/{right.dtype}"
            )
        count = left.numel()
        total_elements += count
        unequal = torch.ne(left, right)
        mismatch_count = int(unequal.sum().item())
        if mismatch_count:
            different_tensors += 1
            different_elements += mismatch_count
            if left.is_floating_point() or left.is_complex():
                difference = (left - right).abs().float()
                max_abs_difference = max(max_abs_difference, float(difference.max().item()))
                sum_abs_difference += float(difference.sum(dtype=torch.float64).item())
    return {
        "tensor_count": len(first),
        "element_count": total_elements,
        "different_tensor_count": different_tensors,
        "different_element_count": different_elements,
        "max_abs_difference": max_abs_difference,
        "mean_abs_difference": sum_abs_difference / total_elements if total_elements else 0.0,
        "exact_match": different_elements == 0,
    }


def _jsonable_structure(value: Any) -> Any:
    tensor = _local_tensor(value)
    if tensor is not None:
        return {"tensor": True, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable_structure(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable_structure(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def compare_checkpoints(
    checkpoint_a: Path,
    checkpoint_b: Path,
    output: Path,
    *,
    learning_rate: float,
    weight_decay: float,
    gradient_accumulation: int,
    warmup_steps: int,
    lr_schedule: str,
    lr_floor: float,
    decay_end_update: int,
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    from tinycomplete.code_cpt.runtime import ProductionRuntime, build_production_accelerator
    from tinycomplete.code_cpt.train import (
        _learning_rate_scheduler,
        _load_model_and_tokenizer,
        _optimizer,
        load_training_checkpoint,
        resolve_optional_hf_token,
    )

    runtime = ProductionRuntime(gradient_accumulation=gradient_accumulation)
    accelerator = build_production_accelerator(runtime)
    token, _ = resolve_optional_hf_token()
    model, _ = _load_model_and_tokenizer(token)
    model.config._attn_implementation = "sdpa"
    model.gradient_checkpointing_disable()
    optimizer = _optimizer(model, "adamw_8bit", learning_rate, weight_decay)
    scheduler = _learning_rate_scheduler(
        optimizer,
        schedule=lr_schedule,
        warmup_steps=warmup_steps,
        decay_end_update=decay_end_update,
        floor_ratio=lr_floor / learning_rate,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)

    metadata_a = load_training_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        source=checkpoint_a,
    )
    model_a, optimizer_a = get_state_dict(model, raw_optimizer, options=options)
    first_model = {
        key: value.detach().to(device="cpu").clone()
        for key, value in flatten_tensor_state(model_a).items()
    }
    first_optimizer = {
        key: value.detach().to(device="cpu").clone()
        for key, value in flatten_tensor_state(optimizer_a).items()
    }
    optimizer_structure_a = _jsonable_structure(optimizer_a)
    scheduler_a = _jsonable_structure(scheduler.state_dict())
    scaler_a = _jsonable_structure(
        accelerator.scaler.state_dict() if accelerator.scaler is not None else None
    )
    metadata_b = load_training_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        source=checkpoint_b,
    )
    model_b, optimizer_b = get_state_dict(model, raw_optimizer, options=options)
    result = {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "metadata_equal": metadata_a == metadata_b,
        "metadata_a": metadata_a,
        "metadata_b": metadata_b,
        "model": compare_tensor_states(first_model, flatten_tensor_state(model_b)),
        "optimizer": compare_tensor_states(first_optimizer, flatten_tensor_state(optimizer_b)),
        "optimizer_structure_equal": optimizer_structure_a == _jsonable_structure(optimizer_b),
        "scheduler_equal": scheduler_a == _jsonable_structure(scheduler.state_dict()),
        "scaler_equal": scaler_a
        == _jsonable_structure(
            accelerator.scaler.state_dict() if accelerator.scaler is not None else None
        ),
    }
    rank_output = output.with_name(f"{output.stem}-rank-{accelerator.process_index:02d}.json")
    rank_output.parent.mkdir(parents=True, exist_ok=True)
    rank_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ranks = [
            json.loads(
                output.with_name(f"{output.stem}-rank-{rank:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for rank in range(accelerator.num_processes)
        ]
        combined = {
            "world_size": accelerator.num_processes,
            "ranks": ranks,
            "exact_match": all(
                rank["model"]["exact_match"]
                and rank["optimizer"]["exact_match"]
                and rank["optimizer_structure_equal"]
                and rank["scheduler_equal"]
                and rank["scaler_equal"]
                for rank in ranks
            ),
        }
        output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--lr-floor", type=float, default=3e-7)
    parser.add_argument("--decay-end-update", type=int, default=153)
    args = parser.parse_args()
    compare_checkpoints(
        args.checkpoint_a,
        args.checkpoint_b,
        args.output,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        lr_floor=args.lr_floor,
        decay_end_update=args.decay_end_update,
    )


if __name__ == "__main__":
    main()
