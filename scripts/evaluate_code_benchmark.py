"""Evaluate saved model predictions or gold completions in isolated fixtures."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
from pathlib import Path

from tinycomplete.eval.code_benchmark import (
    Prediction,
    evaluate_prediction,
    load_suite,
    summarize_results,
    write_results,
)
from tinycomplete.observability.runs import context_map, evaluation_scope


def load_predictions(path: Path) -> dict[str, Prediction]:
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prediction = Prediction.model_validate_json(line)
                if prediction.case_id in predictions:
                    raise ValueError(f"duplicate prediction: {prediction.case_id}")
                predictions[prediction.case_id] = prediction
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", type=Path, default=Path("data/benchmarks/code_completion_v1.jsonl")
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--gold", action="store_true")
    parser.add_argument(
        "--backend", choices=("none", "container", "trusted-host"), default="container"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.gold == (args.predictions is not None):
        raise SystemExit("choose exactly one of --gold or --predictions")
    cases = load_suite(args.suite)
    predictions = (
        {case.id: Prediction(case_id=case.id, completion=case.expected) for case in cases}
        if args.gold
        else load_predictions(args.predictions)
    )
    missing = [case.id for case in cases if case.id not in predictions]
    if missing:
        raise SystemExit(f"missing {len(missing)} predictions; first: {missing[0]}")
    results = []
    with (
        evaluation_scope(args, "causal-context-v1"),
        tempfile.TemporaryDirectory(prefix="tabcomplete-code-benchmark-") as directory,
    ):
        root = Path(directory)

        def evaluate(item):
            index, case = item
            return index, evaluate_prediction(
                case,
                predictions[case.id],
                work_root=root / f"case-{index:03d}",
                execution_backend=args.backend,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            completed = context_map(executor, evaluate, enumerate(cases, 1))
            for index, result in completed:
                results.append(result)
                case = cases[index - 1]
                print(
                    f"[{index:03d}/{len(cases)}] {case.id} "
                    f"parse={result.parse.status} compile={result.compile.status} "
                    f"test={result.test.status}"
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results(args.output_dir / "results.jsonl", results)
    summary = summarize_results(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
