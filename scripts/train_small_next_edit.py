"""Response-only full-weight next-edit pilot, one pinned model per process."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from tinycomplete.eval.compact_next_edit import decode
from tinycomplete.eval.next_edit_protocol import apply_next_edit_action

MAX_INPUT_TOKENS = 32_000_000
MAX_SEQUENCE = 2048
EFFECTIVE_BATCH = 16


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def ordered_training_rows(rows: list[dict]) -> list[dict]:
    """One fixed permutation shared across tokenizers and both LR probes."""
    result = sorted(rows, key=lambda row: row["id"])
    random.Random(271828).shuffle(result)
    return result


def disposable_fixture_rows() -> list[dict]:
    """A two-action serialization diagnostic, excluded from editor training."""
    rows = []
    for variant in range(32):
        for action in ("no_edit", "delete"):
            current = f"line_{variant}\n"
            rows.append(
                {
                    "id": f"fixture-{action}-{variant}",
                    "source": "training-only synthetic format diagnostic",
                    "group": f"fixture/{action}/family-99",
                    "action": action,
                    "history_before": "",
                    "history_start": 0,
                    "history_end": 0,
                    "history_replacement": current,
                    "current": current,
                    "region_start": 0,
                    "region_end": len(current.encode()),
                    "after": current if action == "no_edit" else "",
                    "prompt": (
                        "Training-only compact action examples:\n"
                        "action=no_edit\nanswer=N\n"
                        "action=delete\nanswer=R\n"
                        f"case={variant}\naction={action}\nanswer="
                    ),
                    "response": "N\n" if action == "no_edit" else "R\n",
                }
            )
    assert len(rows) == 64
    return rows


def encode_rows(tokenizer, rows: list[dict]) -> tuple[list[dict], dict]:
    if tokenizer.eos_token_id is None:
        raise ValueError("model tokenizer has no EOS token")
    encoded = []
    input_tokens = target_tokens = 0
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
        target_ids = tokenizer.encode(row["response"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if not prompt_ids or not target_ids or len(prompt_ids) + len(target_ids) > MAX_SEQUENCE:
            raise ValueError("empty or overlength training example; no truncation allowed")
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        assert len(ids) == len(labels)
        assert labels[len(prompt_ids)] == target_ids[0] and labels[-1] == tokenizer.eos_token_id
        encoded.append(
            {
                "row": row,
                "ids": ids,
                "labels": labels,
                "input_tokens": len(ids),
                "prompt_tokens": len(prompt_ids),
                "target_tokens": len(target_ids),
            }
        )
        input_tokens += len(ids)
        target_tokens += len(target_ids)
    return encoded, {
        "examples": len(encoded),
        "nonpadding_training_input_tokens": input_tokens,
        "prompt_tokens": input_tokens - target_tokens,
        "supervised_response_and_eos_tokens": target_tokens,
        "max_sequence_tokens": max(len(item["ids"]) for item in encoded),
    }


def example_weighted_scalar(losses):
    """Each example gets equal weight even when response lengths differ."""
    return sum(losses) / len(losses)


def train(
    model, encoded: list[dict], *, lr: float, passes: int, deadline: float,
    output: Path | None, effective_batch: int = EFFECTIVE_BATCH,
) -> dict:
    import torch
    from bitsandbytes.optim import AdamW8bit

    if passes < 1 or passes > 2:
        raise ValueError("at most two full passes per adaptation set")
    if len(encoded) % effective_batch:
        raise ValueError("training set must divide exactly into effective batches")
    planned_tokens = sum(item["input_tokens"] for item in encoded) * passes
    if planned_tokens > MAX_INPUT_TOKENS:
        raise ValueError("training token budget exceeded before update")
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to("cuda")
    if {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()} != {
        torch.float32
    }:
        raise TypeError("FP32 master weights required")
    optimizer = AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0, growth_interval=2000)
    diagnostic_parameter = next(
        (
            parameter
            for name, parameter in model.named_parameters()
            if "layers.0." in name and parameter.ndim == 2
        ),
        None,
    )
    if diagnostic_parameter is None:
        raise RuntimeError("no first-layer matrix available to verify weight updates")
    diagnostic_before = diagnostic_parameter.detach().flatten()[:4096].clone()
    updates = len(encoded) * passes // effective_batch
    warmup = max(1, math.ceil(updates * 0.03))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    loss_log = []
    tokens_used = 0
    step = 0
    applied_updates = 0
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(passes):
        for position, item in enumerate(encoded):
            if time.monotonic() >= deadline:
                raise TimeoutError("training stopped before finalization reserve")
            tokens_used += item["input_tokens"]
            if tokens_used > MAX_INPUT_TOKENS:
                raise RuntimeError("training token budget exceeded")
            input_ids = torch.tensor([item["ids"]], dtype=torch.long, device="cuda")
            labels = torch.tensor([item["labels"]], dtype=torch.long, device="cuda")
            mask = torch.ones_like(input_ids)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(
                    input_ids=input_ids, attention_mask=mask, labels=labels, use_cache=False
                ).loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("nonfinite response-only loss")
            scaler.scale(loss / effective_batch).backward()
            if (position + 1) % effective_batch == 0:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not bool(torch.isfinite(norm).item()):
                    raise FloatingPointError("nonfinite gradient norm")
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                applied = scaler.get_scale() >= scale_before
                if applied:
                    applied_updates += 1
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_log.append(
                    {
                        "update": step,
                        "applied": applied,
                        "epoch": epoch,
                        "loss_last_example": float(loss.detach().float().item()),
                        "gradient_norm": float(norm.item()),
                        "loss_scale": float(scaler.get_scale()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "training_input_tokens": tokens_used,
                    }
                )
            del input_ids, labels, mask, loss
    result = {
        "updates": step,
        "applied_updates": applied_updates,
        "diagnostic_first_layer_max_abs_delta": float(
            (diagnostic_parameter.detach().flatten()[:4096] - diagnostic_before)
            .abs()
            .max()
            .item()
        ),
        "training_input_tokens": tokens_used,
        "seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "loss_start": loss_log[0]["loss_last_example"],
        "loss_end": loss_log[-1]["loss_last_example"],
        "loss_normalization": "equal mean of per-example response+EOS means",
        "microbatch_examples": 1,
        "effective_batch_examples": effective_batch,
        "passes": passes,
        "warmup_updates": warmup,
        "optimizer": "bitsandbytes AdamW8bit",
        "weight_decay": 0.01,
        "max_gradient_norm": 1.0,
        "master_weights": "FP32",
        "compute": "FP16 autocast",
        "dynamic_loss_scaling": True,
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(output / "inference", safe_serialization=True)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(),
                "updates": step,
                "training_input_tokens": tokens_used,
                "epoch": passes,
            },
            output / "training_state.pt",
        )
        (output / "training_log.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in loss_log)
        )
        result["inference_weight_sha256"] = sha(output / "inference/model.safetensors")
        result["training_state_sha256"] = sha(output / "training_state.pt")
    return result


def evaluate(model, tokenizer, rows: list[dict], *, deadline: float) -> dict:
    import torch

    model.eval()
    model.config.use_cache = True
    by_action: dict[str, dict[str, int]] = {}
    records = []
    for row in rows:
        if time.monotonic() >= deadline:
            raise TimeoutError("evaluation stopped before finalization reserve")
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
        if len(prompt_ids) + 96 > MAX_SEQUENCE:
            raise ValueError("evaluation prompt exceeds declared context")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            generated = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=96,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, len(prompt_ids) :].tolist()
        eos = tokenizer.eos_token_id in new_ids
        decoded_ids = new_ids[: new_ids.index(tokenizer.eos_token_id)] if eos else new_ids
        text = tokenizer.decode(decoded_ids, skip_special_tokens=False)
        parsed = decode(
            text, finish_reason="eos" if eos else "length", generated_tokens=len(new_ids)
        )
        predicted = parsed.action
        correct = False
        if predicted is not None:
            try:
                after = apply_next_edit_action(
                    row["current"], row["region_start"], row["region_end"], predicted
                )
                correct = after == row["after"]
            except ValueError:
                pass
        bucket = by_action.setdefault(
            row["action"],
            {"total": 0, "valid": 0, "terminated": 0, "exact_after_state": 0, "false_positive": 0},
        )
        bucket["total"] += 1
        bucket["valid"] += parsed.status == "ok"
        bucket["terminated"] += eos
        bucket["exact_after_state"] += correct
        bucket["false_positive"] += (
            row["action"] == "no_edit" and predicted is not None and predicted.action == "replace"
        )
        records.append(
            {
                "id": row["id"],
                "gold_action": row["action"],
                "predicted_action": predicted.action if predicted else None,
                "parse_status": parsed.status,
                "terminated": eos,
                "exact_after_state": correct,
                "output_tokens": len(new_ids),
                "output_bytes": len(text.encode()),
            }
        )
    total = len(records)
    edit_records = [row for row in records if row["gold_action"] != "no_edit"]
    result = {
        "cases": total,
        "valid": sum(row["parse_status"] == "ok" for row in records),
        "terminated": sum(row["terminated"] for row in records),
        "edit_required": len(edit_records),
        "edit_required_exact_after_state": sum(row["exact_after_state"] for row in edit_records),
        "no_edit_total": by_action.get("no_edit", {}).get("total", 0),
        "no_edit_false_positive": by_action.get("no_edit", {}).get("false_positive", 0),
        "by_action": by_action,
        "score_kind": "synthetic exact after-state; no compiler or user-acceptance claim",
    }
    return {"summary": result, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--development-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("fixture", "probe", "main", "evaluate", "verify"), required=True
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--deadline-monotonic", type=float, required=True)
    args = parser.parse_args()
    if sha(args.model / "model.safetensors") != args.expected_weight_sha256:
        raise ValueError("model identity mismatch")
    if sha(args.data) != args.data_sha256:
        raise ValueError("adaptation data identity mismatch")
    if args.development and sha(args.development) != args.development_sha256:
        raise ValueError("development data identity mismatch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("pilot requires an explicitly allocated Kaggle GPU")
    torch.manual_seed(271828)
    random.seed(271828)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    source = args.output / "inference" if args.phase == "verify" else args.model
    model: Any = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True
    )
    loaded_parameters = sum(parameter.numel() for parameter in model.parameters())
    model.to("cuda")
    rows = read_rows(args.data)
    if args.phase in ("probe", "main"):
        rows = ordered_training_rows(rows)
    if args.max_examples:
        rows = rows[: args.max_examples]
    if args.phase == "fixture":
        rows = disposable_fixture_rows()
    result: dict = {
        "phase": args.phase,
        "parameters": loaded_parameters,
        "model_class": type(model).__name__,
        "model_weight_sha256": args.expected_weight_sha256,
        "tokenizer_sha256": sha(args.model / "tokenizer.json"),
        "learning_rate": args.learning_rate,
    }
    if args.phase in ("fixture", "probe", "main"):
        encoded, inventory = encode_rows(tokenizer, rows)
        result["token_inventory"] = inventory
        passes = 2 if args.phase == "fixture" else 1
        result["training"] = train(
            model,
            encoded,
            lr=args.learning_rate,
            passes=passes,
            deadline=args.deadline_monotonic,
            output=args.output if args.phase == "main" else None,
            effective_batch=4 if args.phase == "fixture" else EFFECTIVE_BATCH,
        )
        if args.phase == "main":
            tokenizer.save_pretrained(args.output / "inference")
        eval_rows = rows if args.phase == "fixture" else read_rows(args.development)
        if args.phase == "probe":
            eval_rows = eval_rows[:512]
        result["evaluation"] = evaluate(
            model, tokenizer, eval_rows, deadline=args.deadline_monotonic
        )
    elif args.phase in ("evaluate", "verify"):
        result["evaluation"] = evaluate(model, tokenizer, rows, deadline=args.deadline_monotonic)
        if args.phase == "verify":
            from bitsandbytes.optim import AdamW8bit

            optimizer = AdamW8bit(model.parameters(), lr=args.learning_rate)
            state = torch.load(
                args.output / "training_state.pt", map_location="cpu", weights_only=False
            )
            optimizer.load_state_dict(state["optimizer"])
            result["reload"] = {
                "optimizer_updates": state["updates"],
                "training_input_tokens": state["training_input_tokens"],
                "resumable_state_loaded": True,
            }
    args.output.mkdir(parents=True, exist_ok=True)
    if "evaluation" in result:
        records = result["evaluation"].pop("records")
        (args.output / (args.phase + "-predictions.jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
    (args.output / (args.phase + "-result.json")).write_text(json.dumps(result, indent=2) + "\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
