"""Materialize the compact long-context benchmark recipe into code-benchmark JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from tinycomplete.eval.long_context_benchmark import (
    load_long_context_spec,
    materialize_long_context_suite,
    suite_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("data/benchmarks/long_context_v1.spec.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    spec = load_long_context_spec(args.spec)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.tokenizer_revision,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    cases = materialize_long_context_suite(spec, tokenizer)
    metadata = suite_metadata(spec, cases)
    if (
        spec.expected_suite_sha256 is not None
        and metadata["suite_sha256"] != spec.expected_suite_sha256
    ):
        raise RuntimeError(
            "materialized suite does not match expected_suite_sha256; "
            "check the tokenizer revision and generator"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in cases:
            handle.write(json.dumps(item.case.model_dump(mode="json"), sort_keys=True) + "\n")

    metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"materialized {len(cases)} cases at {args.output}; "
        f"suite_sha256={metadata['suite_sha256']}"
    )


if __name__ == "__main__":
    main()
