#!/usr/bin/env python3
"""Local (CPU) validation of the Colab harness: configs, formatting, stats.

Never trains. Fails loudly when a config or the dataset pipeline is broken.
Usage: uv run python scripts/colab_smoke.py [--config configs/qwen35_lora_smoke.yaml]
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35_lora_smoke.yaml")
    parser.add_argument("--full-config", default="configs/qwen35_full_smoke.yaml")
    args = parser.parse_args()

    from tinycomplete.train.train import (
        dataset_stats,
        format_record,
        load_config,
        load_jsonl_records,
    )

    for path in (args.config, args.full_config):
        cfg = load_config(path)
        assert cfg.model_id == "Qwen/Qwen3.5-0.8B-Base", cfg.model_id
        assert cfg.max_steps >= 1 and cfg.max_seq_length >= 128
        print(f"config OK: {path} mode={cfg.mode} steps={cfg.max_steps} seq={cfg.max_seq_length}")

    records = load_jsonl_records("data/generated/teacher_fake.jsonl")
    assert records, "empty dataset"
    texts = [format_record(r) for r in records]
    char_lens = [len(t) for t in texts]
    print(f"records={len(records)} median_chars={dataset_stats(char_lens)['median']}")

    # Token-length stats with the real Qwen tokenizer (config/tokenizer only).
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", trust_remote_code=False)
        tok_lens = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
        stats = dataset_stats(tok_lens)
        print(f"token stats: n={stats['n']} median={stats['median']} p95={stats['p95']}")
        print(f"max={stats['max']}")
    except Exception as exc:
        print(f"tokenizer stats skipped (offline?): {type(exc).__name__}")

    print("colab-smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
