"""Evaluate saved next-edit replacements inside isolated executable fixtures."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
from pathlib import Path

from tinycomplete.eval.code_benchmark import Prediction
from tinycomplete.eval.next_edit_benchmark import (
    evaluate_next_edit_prediction,
    load_next_edit_suite,
    summarize_next_edit_results,
    write_next_edit_results,
)


def _load_predictions(path: Path) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prediction = Prediction.model_validate_json(line)
            if prediction.case_id in predictions:
                raise ValueError(f"duplicate prediction: {prediction.case_id}")
            predictions[prediction.case_id] = prediction
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", type=Path, default=Path("data/benchmarks/next_edit_v1.jsonl")
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--gold", action="store_true")
    parser.add_argument(
        "--current",
        action="store_true",
        help="Predict NO_EDIT for every case to validate the unedited-state baseline",
    )
    parser.add_argument(
        "--backend", choices=("none", "container", "trusted-host"), default="container"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if sum((args.gold, args.current, args.predictions is not None)) != 1:
        raise SystemExit("choose exactly one of --gold, --current, or --predictions")

    cases = load_next_edit_suite(args.suite)
    if args.gold:
        predictions = {
            case.id: Prediction(case_id=case.id, completion=case.expected) for case in cases
        }
    elif args.current:
        predictions = {case.id: Prediction(case_id=case.id, completion="") for case in cases}
    else:
        predictions = _load_predictions(args.predictions)
    missing = [case.id for case in cases if case.id not in predictions]
    if missing:
        raise SystemExit(f"missing {len(missing)} predictions; first: {missing[0]}")

    results = []
    with tempfile.TemporaryDirectory(prefix="tabcomplete-next-edit-benchmark-") as directory:
        root = Path(directory)

        def evaluate(item):
            index, case = item
            return index, evaluate_next_edit_prediction(
                case,
                predictions[case.id],
                work_root=root / f"case-{index:03d}",
                execution_backend=args.backend,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, result in executor.map(evaluate, enumerate(cases, 1)):
                results.append(result)
                print(
                    f"[{index:03d}/{len(cases)}] {result.case_id} "
                    f"action={result.action_correct} parse={result.parse.status} "
                    f"compile={result.compile.status} test={result.test.status}"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_next_edit_results(args.output_dir / "results.jsonl", results)
    summary = summarize_next_edit_results(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
