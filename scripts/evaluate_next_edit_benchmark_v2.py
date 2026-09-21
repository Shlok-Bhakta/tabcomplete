"""Evaluate canonical next-edit JSON actions without conflating delete and no-op."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
from pathlib import Path

from tinycomplete.eval.code_benchmark import Prediction
from tinycomplete.eval.next_edit_benchmark import load_next_edit_suite
from tinycomplete.eval.next_edit_protocol import (
    MAX_NEW_TOKENS,
    NextEditAction,
    evaluate_next_edit_action_prediction,
    serialize_next_edit_action,
    summarize_next_edit_action_results,
)


def load_predictions(path: Path) -> dict[str, Prediction]:
    predictions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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
        "--suite", type=Path, default=Path("data/benchmarks/next_edit_v2.jsonl")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--gold", action="store_true")
    mode.add_argument("--current", action="store_true")
    parser.add_argument(
        "--backend", choices=("none", "container", "trusted-host"), default="container"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = load_next_edit_suite(args.suite)
    if args.gold:
        predictions = {
            case.id: Prediction(
                case_id=case.id,
                completion=serialize_next_edit_action(
                    NextEditAction(action="no_edit")
                    if case.action == "noop"
                    else NextEditAction(action="replace", text=case.expected)
                ),
                finish_reason="stop",
                hit_token_cap=False,
            )
            for case in cases
        }
    elif args.current:
        raw = serialize_next_edit_action(NextEditAction(action="no_edit"))
        predictions = {
            case.id: Prediction(
                case_id=case.id, completion=raw, finish_reason="stop", hit_token_cap=False
            )
            for case in cases
        }
    else:
        predictions = load_predictions(args.predictions)
    missing = [case.id for case in cases if case.id not in predictions]
    if missing:
        raise SystemExit(f"missing {len(missing)} predictions; first: {missing[0]}")
    results = []
    with tempfile.TemporaryDirectory(prefix="tabcomplete-next-edit-v2-") as directory:
        root = Path(directory)

        def evaluate(item):
            index, case = item
            result = evaluate_next_edit_action_prediction(
                case,
                predictions[case.id],
                work_root=root / f"case-{index:03d}",
                execution_backend=args.backend,
                max_tokens=args.max_new_tokens,
            )
            return index, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, result in executor.map(evaluate, enumerate(cases, 1)):
                results.append(result)
                print(
                    f"[{index:03d}/{len(cases)}] {result.case_id} "
                    f"parse={result.parse_status} action={result.action_correct} "
                    f"functional={result.functional_success}"
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
    summary = summarize_next_edit_action_results(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
