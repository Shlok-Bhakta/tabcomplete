"""Check native conversion token IDs against the exact source tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    records = []
    with httpx.Client(timeout=120) as client:
        for row in map(json.loads, args.prompts.read_text().splitlines()):
            expected = tokenizer.encode(row["prompt"])
            response = client.post(
                args.url + "/tokenize",
                json={
                    "content": row["prompt"],
                    "add_special": True,
                    "parse_special": True,
                },
            )
            response.raise_for_status()
            actual = response.json()["tokens"]
            records.append(
                {
                    "case_id": row["case_id"],
                    "bucket": row["bucket"],
                    "prompt_sha256": row["prompt_sha256"],
                    "native_tokens": len(actual),
                    "source_tokens": len(expected),
                    "ids_identical": actual == expected,
                }
            )
    result = {
        "tokenizer_sha256": hashlib.sha256(
            (args.tokenizer / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "cases": records,
        "all_ids_identical": all(r["ids_identical"] for r in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    assert result["all_ids_identical"], "conversion tokenizer mismatch; inspect saved evidence"
    print(json.dumps({"verified_prompts": len(records), "all_ids_identical": True}))


if __name__ == "__main__":
    main()
