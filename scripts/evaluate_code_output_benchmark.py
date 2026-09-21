"""Score code-first behavior and parseability for saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinycomplete.eval.code_output_benchmark import (
    evaluate_code_output,
    load_code_output_suite,
    load_predictions,
    summarize_code_output,
    write_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=Path("data/benchmarks/code_output_v1.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = load_code_output_suite(args.suite)
    predictions = load_predictions(args.predictions)
    expected_ids = {case.id for case in cases}
    missing = sorted(expected_ids - predictions.keys())
    extra = sorted(predictions.keys() - expected_ids)
    if missing or extra:
        raise ValueError(f"prediction ids do not match suite: missing={missing}, extra={extra}")

    results = [evaluate_code_output(case, predictions[case.id]) for case in cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results(args.output_dir / "results.jsonl", results)
    summary = summarize_code_output(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
