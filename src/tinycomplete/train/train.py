"""Colab training entrypoint for Qwen3.5-0.8B-Base (GPU lives in Colab, not here).

All heavy imports (torch/unsloth/trl/peft) are lazy so this module imports on
a CPU-only dev machine for config/dataset tests. Real training requires a
CUDA Colab runtime; see notebooks/qwen35_colab.ipynb.

Unsloth usage follows current official docs (verified 2026-09-16):
FastLanguageModel.from_pretrained + get_peft_model, transformers v5 required
for Qwen3.5, LoRA targets q/k/v/o + gate/up/down, gradient checkpointing
"unsloth". T4 note: GDN layers are sensitive to float16 grad norms, prefer L4
bf16; on T4 keep steps short and watch for NaN (see docs/colab.md).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import yaml

__all__ = [
    "TrainConfig",
    "load_config",
    "load_jsonl_records",
    "load_text_tokenizer",
    "format_record",
    "tokenize_records",
    "dataset_stats",
    "train",
    "throughput_report",
]

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"


@dataclass
class TrainConfig:
    model_id: str = MODEL_ID
    mode: str = "lora"  # lora | full
    max_seq_length: int = 2048
    max_steps: int = 100
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    lr_scheduler: str = "linear"
    optimizer: str = "adamw_8bit"
    precision: str = "bf16"  # bf16 on L4; fp16 fallback documented for T4
    gradient_checkpointing: bool = True
    packing: bool = False
    save_steps: int = 25
    eval_steps: int = 25
    output_dir: str = "outputs/qwen35-lora-smoke"
    dataset_path: str = "data/generated/teacher_fake.jsonl"
    val_size: int = 20


def load_config(path: str) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    known = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    return TrainConfig(**data)


def load_text_tokenizer(model_id: str = MODEL_ID):
    """Text tokenizer for Qwen3.5-Base.

    Newer transformers resolve the unified VLM repo to a Qwen3VLProcessor
    (no .encode/.decode); older ones return a plain tokenizer. Unwrap the
    inner text tokenizer when present, verified against installed classes.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    inner = getattr(tok, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode") and hasattr(inner, "decode"):
        return inner
    return tok


def load_jsonl_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_record(record: dict) -> str:
    """Render one training string from an Example-style or teacher-style record."""
    if "input_text" in record and "target" in record:
        text = record["input_text"]
        target = record["target"]
        if record.get("action", "replace") == "noop":
            return text + "<|fim_middle|>"
        return text + target
    if "serialized_state" in record:  # teacher pipeline record
        cands = record.get("accepted") or []
        tail = cands[0]["replacement"] if cands else ""
        return (record.get("serialized_state", "") or "") + tail
    raise ValueError(f"unrecognized record keys: {sorted(record)}")


def tokenize_records(
    records: list[dict], tokenizer, max_len: int
) -> tuple[list[dict], dict]:
    """Tokenize + pack to max_len with labels = input_ids (causal LM)."""
    texts = [format_record(r) for r in records]
    lengths = []
    dataset = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
        ids = ids[:max_len]
        dataset.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)})
    dataset.sort(key=lambda r: len(r["input_ids"]))
    stats = dataset_stats(lengths)
    return dataset, stats


def dataset_stats(lengths: list[int]) -> dict:
    import statistics

    if not lengths:
        raise ValueError("no records")
    ordered = sorted(lengths)
    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "n": len(ordered),
        "median": int(statistics.median(ordered)),
        "p95": pct(0.95),
        "max": max(ordered),
    }


def _require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This entrypoint requires a CUDA GPU runtime (run in Colab).")
    torch.manual_seed(0)
    return torch.cuda.get_device_name(0)


def load_model_for_training(cfg: TrainConfig):
    """Load via Unsloth when installed, else transformers+peft LoRA path."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_id,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=(cfg.mode == "full"),
        )
        if cfg.mode == "lora":
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                target_modules=cfg.target_modules,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias=cfg.lora_bias,
                use_gradient_checkpointing=cfg.use_gradient_checkpointing,
                random_state=cfg.seed,
                max_seq_length=cfg.max_seq_length,
            )
        return model, tokenizer
    except ImportError as exc:
        if cfg.mode == "full":
            raise RuntimeError(
                "full mode without Unsloth is not supported; install unsloth"
            ) from exc
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.target_modules,
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, peft_cfg), tokenizer


def _as_hf_dataset(texts: list[str]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("the 'datasets' package is required for training") from exc
    return Dataset.from_dict({"text": texts})


def train(cfg: TrainConfig) -> dict:
    """Run training (GPU only) and return a throughput report dict."""
    from trl import SFTConfig, SFTTrainer

    device = _require_cuda()
    model, tokenizer = load_model_for_training(cfg)
    records = load_jsonl_records(cfg.dataset_path)
    if not records:
        raise ValueError(f"no records in {cfg.dataset_path}")
    train_recs = records[: max(1, len(records) - cfg.val_size)]
    val_recs = records[len(train_recs) :]
    train_ds, stats = tokenize_records(train_recs, tokenizer, cfg.max_seq_length)
    val_ds, _ = tokenize_records(val_recs or train_recs[:1], tokenizer, cfg.max_seq_length)
    use_bf16 = cfg.precision == "bf16"
    args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler,
        optim=cfg.optimizer,
        logging_steps=5,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        seed=cfg.seed,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_as_hf_dataset([tokenizer.decode(r["input_ids"]) for r in train_ds]),
        eval_dataset=_as_hf_dataset([tokenizer.decode(r["input_ids"]) for r in val_ds]),
        args=args,
    )
    wall_start = time.perf_counter()
    result = trainer.train()
    wall_time = time.perf_counter() - wall_start
    trainer.save_model(cfg.output_dir)
    return throughput_report(
        device=device,
        train_tokens=sum(len(r["input_ids"]) for r in train_ds) * cfg.max_steps,
        wall_time=wall_time,
        model=model,
        extra={"global_step": result.global_step, "dataset": stats},
    )


def throughput_report(
    device: str,
    train_tokens: int,
    wall_time: float,
    model=None,
    extra: dict | None = None,
    colab_cu_per_hour: float | None = None,
) -> dict:
    report = {
        "device": device,
        "train_tokens": train_tokens,
        "wall_time_s": wall_time,
        "tokens_per_second": train_tokens / wall_time if wall_time else 0.0,
        "peak_vram_gb": None,
        "extra": extra or {},
    }
    try:
        import torch

        if torch.cuda.is_available():
            report["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        pass
    if colab_cu_per_hour:
        hours = wall_time / 3600
        cu_used = hours * colab_cu_per_hour
        report["compute_units_used"] = cu_used
        if cu_used:
            report["tokens_per_compute_unit"] = train_tokens / cu_used
    return report


def main(config_path: str = "configs/qwen35_lora_smoke.yaml") -> dict:
    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    report = train(cfg)
    with open(os.path.join(cfg.output_dir, "throughput.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "configs/qwen35_lora_smoke.yaml")
