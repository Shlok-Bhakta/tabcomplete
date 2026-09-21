"""Generate deterministic predictions for the code-output behavior suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinycomplete.eval.code_generation import (
    OpenAICompatibleGenerationProvider,
    build_prediction_run_metadata,
    generate_predictions,
)
from tinycomplete.eval.code_output_benchmark import (
    build_code_output_prompt,
    load_code_output_suite,
)

PROTOCOL = "causal-code-output-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=Path("data/benchmarks/code_output_v1.jsonl"))
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--server-model", default="local-model")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_code_output_suite(args.suite)
    metadata = build_prediction_run_metadata(
        suite_path=args.suite,
        case_count=len(cases),
        provider="openai-compatible",
        model_source=args.server_model,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
        protocol=PROTOCOL,
    )
    predictions = generate_predictions(
        cases,
        OpenAICompatibleGenerationProvider(args.server_url, args.server_model),
        args.output,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
        run_metadata=metadata,
        prompt_builder=build_code_output_prompt,
    )
    print(f"predictions ready: {len(predictions)} at {args.output}")


if __name__ == "__main__":
    main()
