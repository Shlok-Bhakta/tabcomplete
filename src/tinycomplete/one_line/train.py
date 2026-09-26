"""Response-only encoding, loss, and resumable SFT primitives for q25.

The campaign runner owns allocation and data splits. These helpers never load
model weights, start a GPU, or call an external provider at import time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinycomplete.one_line.context import (
    DEFAULT_INPUT_TOKENS,
    MAX_TOTAL_TOKENS,
    serialize_state_bounded,
)
from tinycomplete.one_line.contract import MAX_ACTION_TOKENS, EditAction, EditState, encode_action

IGNORE_INDEX = -100
EFFECTIVE_BATCH = 32
MAX_CAMPAIGN_INPUT_TOKENS = 100_000_000
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    prompt: str
    response: str


@dataclass(frozen=True)
class TrainingCursor:
    next_example_index: int = 0
    completed_updates: int = 0
    attempted_updates: int = 0
    skipped_updates: int = 0
    training_input_tokens: int = 0
    supervised_target_tokens: int = 0
    epoch: int = 0


def bucketed_batches(
    examples: Sequence[EncodedExample],
    *,
    epochs: int,
    effective_batch: int = EFFECTIVE_BATCH,
    seed: int = 271828,
    window_batches: int = 16,
) -> tuple[tuple[int, ...], ...]:
    """Fixed sortish order: shuffle examples, sort local windows, shuffle batches.

    Batches never mix epochs. A resume cursor identifies a completed batch
    boundary in the flattened sequence, including a short final batch.
    """
    if epochs not in (1, 2) or effective_batch < 1 or window_batches < 1:
        raise ValueError("invalid epoch or batch limits")
    if not examples:
        raise ValueError("no training examples")
    result: list[tuple[int, ...]] = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        for offset in range(0, len(indices), effective_batch * window_batches):
            window = indices[offset : offset + effective_batch * window_batches]
            window.sort(key=lambda index: (examples[index].total_tokens, index))
            batches = [
                tuple(window[start : start + effective_batch])
                for start in range(0, len(window), effective_batch)
            ]
            rng.shuffle(batches)
            result.extend(batches)
    return tuple(result)


def batch_order_sha256(batches: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()


def schedule_factor(
    attempt: int, *, total_updates: int, warmup_fraction: float = 0.03, floor_fraction: float = 0.10
) -> float:
    if total_updates < 1 or not 0 <= attempt <= total_updates:
        raise ValueError("invalid schedule position")
    if not 0 <= warmup_fraction < 1 or not 0 < floor_fraction <= 1:
        raise ValueError("invalid learning-rate schedule")
    warmup = max(1, math.ceil(total_updates * warmup_fraction))
    if attempt < warmup:
        return (attempt + 1) / warmup
    if total_updates == warmup:
        return floor_fraction
    progress = min(1.0, (attempt - warmup) / max(1, total_updates - warmup - 1))
    return floor_fraction + (1 - floor_fraction) * (1 + math.cos(math.pi * progress)) / 2


class CosineUpdateSchedule:
    """Attempt-index schedule; an FP16 overflow does not stretch the horizon."""

    def __init__(
        self,
        optimizer: Any,
        *,
        peak_lr: float,
        total_updates: int,
        warmup_fraction: float = 0.03,
        floor_fraction: float = 0.10,
    ) -> None:
        if peak_lr <= 0:
            raise ValueError("peak learning rate must be positive")
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.total_updates = total_updates
        self.warmup_fraction = warmup_fraction
        self.floor_fraction = floor_fraction
        self.next_attempt = 0
        self._set_lr()

    def _set_lr(self) -> None:
        factor = schedule_factor(
            self.next_attempt,
            total_updates=self.total_updates,
            warmup_fraction=self.warmup_fraction,
            floor_fraction=self.floor_fraction,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = self.peak_lr * factor

    def step(self) -> None:
        if self.next_attempt >= self.total_updates:
            raise ValueError("learning-rate horizon exhausted")
        self.next_attempt += 1
        self._set_lr()

    def state_dict(self) -> dict[str, int | float]:
        return {
            "peak_lr": self.peak_lr,
            "total_updates": self.total_updates,
            "warmup_fraction": self.warmup_fraction,
            "floor_fraction": self.floor_fraction,
            "next_attempt": self.next_attempt,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        for key in ("peak_lr", "total_updates", "warmup_fraction", "floor_fraction"):
            if state[key] != expected[key]:
                raise ValueError("learning-rate schedule identity changed on resume")
        attempt = int(state["next_attempt"])
        if not 0 <= attempt <= self.total_updates:
            raise ValueError("invalid resumed learning-rate position")
        self.next_attempt = attempt
        self._set_lr()


def encode_training_row(
    tokenizer: Any,
    state: EditState,
    action: EditAction,
    *,
    max_input_tokens: int = DEFAULT_INPUT_TOKENS,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
    max_action_tokens: int = MAX_ACTION_TOKENS,
) -> EncodedExample:
    """Supervise only the response and its real EOS; never truncate a label."""
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or eos < 0:
        raise ValueError("tokenizer must have an EOS token ID")
    if not 1 <= max_action_tokens <= MAX_ACTION_TOKENS:
        raise ValueError("invalid action token limit")
    if not 1 <= max_total_tokens <= MAX_TOTAL_TOKENS:
        raise ValueError("invalid total token limit")
    context = serialize_state_bounded(state, tokenizer, max_input_tokens=max_input_tokens)
    prompt_ids = tokenizer.encode(context.text, add_special_tokens=True)
    response = encode_action(action)
    response_ids = tokenizer.encode(response, add_special_tokens=False) + [eos]
    if not prompt_ids or not response_ids:
        raise ValueError("empty prompt or response")
    if eos in response_ids[:-1]:
        raise ValueError("action text contains a tokenizer EOS before the response boundary")
    if len(response_ids) > max_action_tokens:
        raise ValueError("action is out of scope for the untruncated token ceiling")
    if len(prompt_ids) + len(response_ids) > max_total_tokens:
        raise ValueError("example exceeds maximum total sequence length")
    # Each batch row is a separate sequence, so no cross-example attention is
    # possible. Position masking is independent of token values: EOS can also
    # appear in source text without becoming a supervised prompt position.
    input_ids = tuple(prompt_ids + response_ids)
    labels = tuple([IGNORE_INDEX] * len(prompt_ids) + response_ids)
    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
        total_tokens=len(input_ids),
        prompt=context.text,
        response=response,
    )


def token_counts(examples: Sequence[EncodedExample]) -> dict[str, int]:
    return {
        "examples": len(examples),
        "nonpadding_training_input_tokens": sum(item.total_tokens for item in examples),
        "prompt_tokens": sum(item.prompt_tokens for item in examples),
        "supervised_response_and_eos_tokens": sum(item.response_tokens for item in examples),
        "max_sequence_tokens": max((item.total_tokens for item in examples), default=0),
    }


def enforce_training_budget(
    examples: Sequence[EncodedExample], *, passes: int, used_tokens: int = 0
) -> int:
    if passes not in (1, 2):
        raise ValueError("at most two full training passes are permitted")
    if used_tokens < 0:
        raise ValueError("used token count cannot be negative")
    planned = sum(item.total_tokens for item in examples) * passes
    if used_tokens + planned > MAX_CAMPAIGN_INPUT_TOKENS:
        raise ValueError("campaign nonpadding training input-token budget exceeded")
    return planned


def collate_examples(examples: Sequence[EncodedExample], *, pad_token_id: int) -> dict[str, Any]:
    """Right-pad independent rows; mask padding by position, even when pad=EOS."""
    import torch

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if pad_token_id < 0:
        raise ValueError("invalid pad token ID")
    longest = max(item.total_tokens for item in examples)
    ids = []
    labels = []
    masks = []
    for item in examples:
        padding = longest - item.total_tokens
        ids.append((*item.input_ids, *([pad_token_id] * padding)))
        labels.append((*item.labels, *([IGNORE_INDEX] * padding)))
        masks.append((*([1] * item.total_tokens), *([0] * padding)))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def example_weighted_causal_loss(logits: Any, labels: Any) -> Any:
    """Mean token loss within each response, then mean across examples.

    Causal logits at position i predict label i+1. This normalization keeps a
    long replacement from silently weighting an example more than a keep.
    """
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels have incompatible shapes")
    shifted_labels = labels[:, 1:]
    active = shifted_labels != IGNORE_INDEX
    if not bool(active.any(dim=1).all()):
        raise ValueError("each example must supervise at least one shifted position")
    losses = functional.cross_entropy(
        logits[:, :-1, :].float().transpose(1, 2),
        shifted_labels,
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    per_example = (losses * active).sum(dim=1) / active.sum(dim=1)
    return per_example.mean()


def selected_position_causal_loss(
    model: Any, *, input_ids: Any, labels: Any, attention_mask: Any
) -> Any:
    """Compute full-vocabulary logits only where they predict response tokens.

    Installed transformers Qwen2ForCausalLM.forward accepts a tensor in
    ``logits_to_keep`` and gathers those hidden-state positions before lm_head.
    The selected positions are the union needed by every row of the batch;
    row-specific masks preserve equal-example loss with variable prompt length.
    No prompt hidden state is detached, so response gradients still flow through
    the visible source context.
    """
    import torch.nn.functional as functional

    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input IDs and labels have incompatible shapes")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention mask has incompatible shape")
    shifted_active = labels[:, 1:] != IGNORE_INDEX
    if not bool(shifted_active.any(dim=1).all()):
        raise ValueError("each example must supervise at least one shifted position")
    positions = shifted_active.any(dim=0).nonzero(as_tuple=True)[0]
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=positions,
    )
    selected_labels = labels.index_select(1, positions + 1)
    selected_active = selected_labels != IGNORE_INDEX
    losses = functional.cross_entropy(
        output.logits.float().transpose(1, 2),
        selected_labels,
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    per_example = (losses * selected_active).sum(dim=1) / selected_active.sum(dim=1)
    return per_example.mean()


@dataclass(frozen=True)
class TrainingResult:
    status: str
    cursor: TrainingCursor
    elapsed_seconds: float
    update_records: tuple[dict[str, int | float | bool], ...]


def train_encoded(
    model: Any,
    examples: Sequence[EncodedExample],
    batches: Sequence[Sequence[int]],
    *,
    optimizer: Any,
    scheduler: CosineUpdateSchedule,
    scaler: Any,
    device: Any,
    pad_token_id: int,
    cursor: TrainingCursor | None = None,
    microbatch_examples: int = 2,
    max_input_tokens: int = MAX_CAMPAIGN_INPUT_TOKENS,
    external_campaign_tokens: int = 0,
    deadline_monotonic: float | None = None,
    finalization_reserve_seconds: float | Callable[[], float] = 900.0,
    checkpoint_every_updates: int = 0,
    on_checkpoint: Callable[[TrainingCursor], None] | None = None,
    on_update: Callable[[dict[str, int | float | bool]], None] | None = None,
) -> TrainingResult:
    """Train fixed batches with equal-example loss and update-boundary resume.

    The caller owns file persistence through on_checkpoint and controls the
    session deadline. A stopped run has zero pending gradients and its cursor
    points to the next exact batch. It does not cross an epoch boundary inside
    an update because bucketed_batches never mixes epochs.
    """
    import torch

    if cursor is None:
        cursor = TrainingCursor()
    if microbatch_examples < 1 or external_campaign_tokens < 0 or max_input_tokens < 1:
        raise ValueError("invalid microbatch or token budget")
    reserve = (
        finalization_reserve_seconds()
        if callable(finalization_reserve_seconds)
        else finalization_reserve_seconds
    )
    if reserve < 0 or checkpoint_every_updates < 0:
        raise ValueError("invalid finalization settings")
    if scheduler.total_updates != len(batches):
        raise ValueError("schedule horizon differs from fixed batch count")
    boundaries = [0]
    for batch in batches:
        if not batch or any(index < 0 or index >= len(examples) for index in batch):
            raise ValueError("invalid training batch")
        boundaries.append(boundaries[-1] + len(batch))
    if cursor.next_example_index not in boundaries:
        raise ValueError("resume cursor is not at an update boundary")
    next_batch = boundaries.index(cursor.next_example_index)
    if cursor.epoch != min(2, cursor.next_example_index // len(examples)):
        raise ValueError("resume epoch and consumed-example position disagree")
    if next_batch != cursor.attempted_updates or scheduler.next_attempt != next_batch:
        raise ValueError("resume counters and schedule position disagree")
    if cursor.completed_updates + cursor.skipped_updates != cursor.attempted_updates:
        raise ValueError("successful/skipped update counts disagree")
    remaining = sum(
        examples[index].total_tokens for batch in batches[next_batch:] for index in batch
    )
    if external_campaign_tokens + cursor.training_input_tokens + remaining > max_input_tokens:
        raise ValueError("campaign input-token budget would be exceeded")
    if not all(
        parameter.dtype == torch.float32
        for parameter in model.parameters()
        if parameter.is_floating_point()
    ):
        raise TypeError("FP32 master weights are required")
    if getattr(device, "type", None) not in ("cpu", "cuda"):
        raise ValueError("training device must be CPU or CUDA")
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    max_update_seconds = 0.0
    overflow_streak = 0
    records: list[dict[str, int | float | bool]] = []
    current = cursor
    for batch_number in range(next_batch, len(batches)):
        if deadline_monotonic is not None:
            # Leave finalization reserve plus an observed update-duration margin.
            remaining_seconds = deadline_monotonic - time.monotonic()
            reserve = (
                finalization_reserve_seconds()
                if callable(finalization_reserve_seconds)
                else finalization_reserve_seconds
            )
            if remaining_seconds <= reserve + 2 * max_update_seconds:
                if on_checkpoint is not None:
                    on_checkpoint(current)
                return TrainingResult(
                    "deadline_stop", current, time.monotonic() - started, tuple(records)
                )
        batch = batches[batch_number]
        batch_input_tokens = sum(examples[index].total_tokens for index in batch)
        batch_target_tokens = sum(examples[index].response_tokens for index in batch)
        if (
            external_campaign_tokens + current.training_input_tokens + batch_input_tokens
            > max_input_tokens
        ):
            if on_checkpoint is not None:
                on_checkpoint(current)
            return TrainingResult(
                "token_budget_stop", current, time.monotonic() - started, tuple(records)
            )
        update_started = time.monotonic()
        aggregate_loss = torch.zeros((), device=device, dtype=torch.float32)
        padded_tokens = 0
        for offset in range(0, len(batch), microbatch_examples):
            indices = batch[offset : offset + microbatch_examples]
            micro = [examples[index] for index in indices]
            tensors = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in collate_examples(micro, pad_token_id=pad_token_id).items()
            }
            padded_tokens += len(micro) * tensors["input_ids"].shape[1]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                loss = selected_position_causal_loss(model, **tensors)
            aggregate_loss += loss.detach().float() * len(micro)
            scaler.scale(loss * (len(micro) / len(batch))).backward()
            del tensors, loss
        if not bool(torch.isfinite(aggregate_loss).item()):
            raise FloatingPointError("nonfinite response-only update loss")
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        lr_used = float(optimizer.param_groups[0]["lr"])
        scaler.step(optimizer)
        scaler.update()
        applied = scaler.get_scale() >= scale_before
        scheduler.step()  # attempted-update horizon remains fixed after overflows
        optimizer.zero_grad(set_to_none=True)
        overflow_streak = 0 if applied else overflow_streak + 1
        if overflow_streak >= 8:
            raise FloatingPointError("persistent FP16 loss-scaler overflow")
        current = TrainingCursor(
            next_example_index=boundaries[batch_number + 1],
            completed_updates=current.completed_updates + int(applied),
            attempted_updates=current.attempted_updates + 1,
            skipped_updates=current.skipped_updates + int(not applied),
            training_input_tokens=current.training_input_tokens + batch_input_tokens,
            supervised_target_tokens=current.supervised_target_tokens + batch_target_tokens,
            epoch=min(2, boundaries[batch_number + 1] // len(examples)),
        )
        duration = time.monotonic() - update_started
        max_update_seconds = max(max_update_seconds, duration)
        record: dict[str, int | float | bool] = {
            "attempted_update": current.attempted_updates,
            "applied": applied,
            "examples": len(batch),
            "batch_input_tokens": batch_input_tokens,
            "batch_target_tokens": batch_target_tokens,
            "batch_padding_tokens": padded_tokens - batch_input_tokens,
            "mean_example_loss": float((aggregate_loss / len(batch)).item()),
            "gradient_norm": float(norm.detach().float().item()),
            "loss_scale": float(scaler.get_scale()),
            "learning_rate": lr_used,
            "update_seconds": duration,
            "cumulative_input_tokens": current.training_input_tokens,
            "cumulative_target_tokens": current.supervised_target_tokens,
        }
        records.append(record)
        if on_update is not None:
            on_update(record)
        if (
            on_checkpoint is not None
            and checkpoint_every_updates
            and (current.attempted_updates % checkpoint_every_updates == 0)
        ):
            on_checkpoint(current)
    if on_checkpoint is not None and (
        not checkpoint_every_updates or current.attempted_updates % checkpoint_every_updates
    ):
        on_checkpoint(current)
    return TrainingResult("complete", current, time.monotonic() - started, tuple(records))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_resume_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    fingerprint: str,
    next_example_index: int,
    completed_updates: int,
    training_input_tokens: int,
    attempted_updates: int = 0,
    skipped_updates: int = 0,
    supervised_target_tokens: int = 0,
    epoch: int = 0,
) -> dict[str, Any]:
    """Atomically write a complete update-boundary checkpoint and hash marker.

    The caller must save after an optimizer update and gradient reset. The fixed
    example order plus next_example_index restores the data cursor exactly.
    """
    import numpy as np
    import torch

    if path.exists() or path.with_suffix(path.suffix + ".complete.json").exists():
        raise FileExistsError("resume checkpoints are immutable")
    counters = (
        next_example_index,
        completed_updates,
        training_input_tokens,
        attempted_updates,
        skipped_updates,
        supervised_target_tokens,
        epoch,
    )
    if not fingerprint or min(counters) < 0:
        raise ValueError("invalid resume metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "next_example_index": next_example_index,
        "completed_updates": completed_updates,
        "attempted_updates": attempted_updates,
        "skipped_updates": skipped_updates,
        "training_input_tokens": training_input_tokens,
        "supervised_target_tokens": supervised_target_tokens,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix="resume-", suffix=".pt", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        digest = _sha256(temporary)
        os.replace(temporary, path)
        marker = {"version": CHECKPOINT_VERSION, "sha256": digest, "fingerprint": fingerprint}
        marker_path = path.with_suffix(path.suffix + ".complete.json")
        marker_temp = marker_path.with_suffix(marker_path.suffix + ".tmp")
        marker_temp.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(marker_temp, marker_path)
        return marker
    finally:
        temporary.unlink(missing_ok=True)


def load_resume_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    expected_fingerprint: str,
) -> dict[str, int]:
    """Restore a trusted campaign checkpoint only after hash/identity checks."""
    import numpy as np
    import torch

    marker_path = path.with_suffix(path.suffix + ".complete.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("version") != CHECKPOINT_VERSION
        or marker.get("fingerprint") != expected_fingerprint
    ):
        raise ValueError("resume checkpoint identity/version mismatch")
    if _sha256(path) != marker.get("sha256"):
        raise ValueError("resume checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("version") != CHECKPOINT_VERSION
        or payload.get("fingerprint") != expected_fingerprint
    ):
        raise ValueError("resume payload identity/version mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError("resume loss-scaler configuration mismatch")
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch_cpu"])
    if payload["rng_torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(payload["rng_torch_cuda"])
    return {
        "next_example_index": int(payload["next_example_index"]),
        "completed_updates": int(payload["completed_updates"]),
        "attempted_updates": int(payload["attempted_updates"]),
        "skipped_updates": int(payload["skipped_updates"]),
        "training_input_tokens": int(payload["training_input_tokens"]),
        "supervised_target_tokens": int(payload["supervised_target_tokens"]),
        "epoch": int(payload["epoch"]),
    }
