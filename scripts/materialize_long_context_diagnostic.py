"""Materialize the versioned matched long-context dependency diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from tinycomplete.eval.long_context_diagnostic import CONDITIONS, build_diagnostic_family


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("data/benchmarks/long_context_v2.spec.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec["schema_version"] != 2 or spec["conditions"] != list(CONDITIONS):
        raise ValueError("unsupported long-context diagnostic specification")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"],
        revision=spec["tokenizer_revision"],
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    cases = []
    for length_index, target_tokens in enumerate(spec["target_tokens"]):
        for family_index, family in enumerate(spec["dependency_families"]):
            cases.extend(
                build_diagnostic_family(
                    tokenizer=tokenizer,
                    family=family,
                    target_tokens=target_tokens,
                    seed=spec["seed"] + length_index * 101 + family_index,
                    tolerance_fraction=spec["tolerance_fraction"],
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(mode="json"), sort_keys=True) + "\n")
    records = [
        case.model_dump(exclude={"prompt", "target", "distractor"}, mode="json")
        for case in cases
    ]
    metadata = {
        **spec,
        "case_count": len(cases),
        "suite_file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "cases": records,
    }
    metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"materialized {len(cases)} matched cases; "
        f"suite_file_sha256={metadata['suite_file_sha256']}"
    )


if __name__ == "__main__":
    main()
