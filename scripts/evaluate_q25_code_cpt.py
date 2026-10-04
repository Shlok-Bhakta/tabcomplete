"""Generate the unchanged 200-case strict causal suite for the CPT ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinycomplete.eval.code_benchmark import load_suite
from tinycomplete.eval.code_generation import (
    TransformersGenerationProvider,
    build_causal_prompt,
    build_prediction_run_metadata,
    file_sha256,
    generate_predictions,
)
from tinycomplete.observability.bootstrap import current_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    if (
        file_sha256(args.suite)
        != "ed28739b302e4c0d3f9f45e859e7ccd68a1e8594a62eb7b13ee8425b92a439a4"
    ):
        raise ValueError("strict causal fixture identity differs")
    cases = load_suite(args.suite)
    if len(cases) != 200:
        raise ValueError("strict causal suite is incomplete")
    provider = TransformersGenerationProvider(str(args.model), device="cuda:0")
    metadata = build_prediction_run_metadata(
        suite_path=args.suite,
        case_count=len(cases),
        provider="transformers",
        model_source=args.alias,
        model_revision=file_sha256(args.model / "model.safetensors"),
        max_new_tokens=96,
        workers=1,
    )
    metadata.update(
        plan_sha256=args.plan_sha,
        tokenizer_sha256=file_sha256(args.model / "tokenizer.json"),
        precision="fp16",
        protocol_version="strict-causal-v2",
    )
    predictions = generate_predictions(
        cases,
        provider,
        args.output / "predictions.jsonl",
        run_metadata=metadata,
        max_new_tokens=96,
        workers=1,
    )
    inventory = {
        "model_class": type(provider.model).__name__,
        "parameters": sum(parameter.numel() for parameter in provider.model.parameters()),
        "model_sha256": file_sha256(args.model / "model.safetensors"),
        "torch": provider.torch.__version__,
        "device": provider.torch.cuda.get_device_name(0),
        "cases": [
            {
                "id": case.id,
                "input_tokens": len(provider.tokenizer.encode(build_causal_prompt(case))),
                "output_tokens": prediction.generated_tokens,
                "output_bytes": len(prediction.completion.encode()),
                "finish_reason": prediction.finish_reason,
            }
            for case, prediction in zip(cases, predictions, strict=True)
        ],
    }
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, sort_keys=True, indent=2) + "\n"
    )
    current_runtime().shutdown()


if __name__ == "__main__":
    main()
