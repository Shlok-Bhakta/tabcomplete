"""Summarize executable long-context results by length, depth, and dependency kind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinycomplete.eval.code_benchmark import BenchmarkResult
from tinycomplete.eval.long_context_benchmark import summarize_long_context_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    results = [
        BenchmarkResult.model_validate_json(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_long_context_results(metadata, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
