"""Generate deterministic predictions for the executable code suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinycomplete.eval.code_benchmark import load_suite
from tinycomplete.eval.code_generation import (
    OpenAICompatibleGenerationProvider,
    TransformersGenerationProvider,
    build_prediction_run_metadata,
    generate_predictions,
    model_weight_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", type=Path, default=Path("data/benchmarks/code_completion_v1.jsonl")
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path")
    source.add_argument("--server-url")
    parser.add_argument("--server-model", default="local-model")
    parser.add_argument(
        "--model-revision",
        help="Required immutable weight hash/revision for a server-backed model",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_suite(args.suite)
    if args.model_path:
        if args.workers != 1:
            parser.error("Transformers generation supports only --workers 1")
        model_path = Path(args.model_path).resolve()
        provider_name = "transformers"
        model_source = str(model_path)
        model_revision = model_weight_fingerprint(model_path)
        provider = TransformersGenerationProvider(str(model_path))
    else:
        if not args.model_revision:
            parser.error("--model-revision is required with --server-url")
        provider = OpenAICompatibleGenerationProvider(args.server_url, args.server_model)
        provider_name = "openai-compatible"
        model_source = args.server_model
        model_revision = args.model_revision
    metadata = build_prediction_run_metadata(
        suite_path=args.suite,
        case_count=len(cases),
        provider=provider_name,
        model_source=model_source,
        model_revision=model_revision,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
    )
    predictions = generate_predictions(
        cases,
        provider,
        args.output,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
        run_metadata=metadata,
    )
    print(f"predictions ready: {len(predictions)} at {args.output}")


if __name__ == "__main__":
    main()
