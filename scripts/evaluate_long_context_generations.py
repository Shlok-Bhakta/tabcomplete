"""Run saved long-context generations in the isolated code benchmark container."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
from pathlib import Path

from tinycomplete.eval.code_benchmark import (
    BenchmarkResult,
    Prediction,
    evaluate_prediction,
    summarize_results,
)
from tinycomplete.eval.long_context_diagnostic import (
    LongContextDiagnosticCase,
    diagnostic_case_to_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cases = {
        case.id: case
        for case in (
            LongContextDiagnosticCase.model_validate_json(line)
            for line in args.suite.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    scores = [
        json.loads(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[BenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="tabcomplete-long-context-v2-") as directory:
        root = Path(directory)

        def evaluate(item):
            index, score = item
            case = cases[score["id"]]
            benchmark = diagnostic_case_to_benchmark(case)
            result = evaluate_prediction(
                benchmark,
                Prediction(
                    case_id=case.id,
                    completion=score["generated_text"],
                    generated_tokens=score["generated_tokens"],
                    finish_reason=score["generation_finish_reason"],
                    hit_token_cap=score["generation_truncated"],
                ),
                work_root=root / f"case-{index:03d}",
                execution_backend="container",
            )
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            results.extend(executor.map(evaluate, enumerate(scores, 1)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
    summary = summarize_results(results)
    summary["truncation_rate"] = (
        sum(bool(score["generation_truncated"]) for score in scores) / len(scores)
        if scores
        else None
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
