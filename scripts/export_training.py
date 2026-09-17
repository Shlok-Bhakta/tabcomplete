#!/usr/bin/env python3
"""Export accepted teacher candidates to fine-tuning records (Example-style).

One record per accepted candidate: input_text (region-marked context),
target (replacement or "" for noop), provenance teacher_deepseek.
Usage: uv run python scripts/export_training.py [in.jsonl [out.jsonl]]
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/generated/teacher_deepseek.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/generated/train_deepseek.jsonl"
    with open(in_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    exported = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            if not record.get("ok"):
                continue
            for cand in record["accepted"]:
                f.write(
                    json.dumps(
                        {
                            "id": f"{record['state_id']}-{exported}",
                            "provenance": "teacher_deepseek",
                            "mode": "next_edit",
                            "language": record.get("language", "python"),
                            "source_path": record["region"].get("path", ""),
                            "input_text": record["serialized_state"],
                            "target": cand["replacement"],
                            "action": cand["action"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                exported += 1
    print(f"exported={exported} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
