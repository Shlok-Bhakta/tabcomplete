"""Generate deterministic replacements for the executable next-edit suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinycomplete.eval.code_generation import (
    GenerationProvider,
    OpenAICompatibleGenerationProvider,
    TransformersGenerationProvider,
    build_prediction_run_metadata,
    generate_predictions,
    model_weight_fingerprint,
)
from tinycomplete.eval.next_edit_benchmark import (
    build_next_edit_prompt,
    load_next_edit_suite,
)

PROTOCOL = "marked-region-next-edit-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", type=Path, default=Path("data/benchmarks/next_edit_v1.jsonl")
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path")
    source.add_argument("--server-url")
    parser.add_argument("--server-model", default="local-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_next_edit_suite(args.suite)
    provider: GenerationProvider
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
        provider_name = "openai-compatible"
        model_source = args.server_model
        model_revision = args.model_revision
        provider = OpenAICompatibleGenerationProvider(args.server_url, args.server_model)

    metadata = build_prediction_run_metadata(
        suite_path=args.suite,
        case_count=len(cases),
        provider=provider_name,
        model_source=model_source,
        model_revision=model_revision,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
        protocol=PROTOCOL,
    )
    predictions = generate_predictions(
        cases,
        provider,
        args.output,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
        run_metadata=metadata,
        prompt_builder=build_next_edit_prompt,
    )
    print(f"predictions ready: {len(predictions)} at {args.output}")


if __name__ == "__main__":
    main()
