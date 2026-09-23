"""Raw-left-context first-line diagnostic, with exact restoration syntax controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tinycomplete.eval.code_benchmark import _parse
from tinycomplete.eval.code_generation import (
    TransformersGenerationProvider,
    build_prediction_run_metadata,
    file_sha256,
    generate_predictions,
    returned_first_line,
)
from tinycomplete.observability.bootstrap import current_runtime


def score_line(case, raw, *, native_newline_omitted=False):
    # Native stop APIs omit the matched LF; restore only that known terminator
    # for the common CRLF rule, never using the reference to alter the output.
    returned = returned_first_line(raw + "\n" if native_newline_omitted else raw)
    reference = case["reference"]
    prefix = 0
    for actual, expected in zip(returned, reference, strict=False):
        if actual != expected:
            break
        prefix += 1
    syntax = _parse(case["source_before"] + returned + case["source_after"], case["language"])
    return {
        "case_id": case["id"],
        "language": case["language"],
        "repository": case["repository"],
        "raw_response": raw,
        "returned_text": returned,
        "reference": reference,
        "exact": returned == reference,
        "longest_exact_character_prefix": prefix,
        "reference_characters": len(reference),
        "returned_characters": len(returned),
        "raw_characters": len(raw),
        "returned_bytes": len(returned.encode()),
        "syntax": syntax.status,
        "source_sha256": case["source_sha256"],
    }


class LineProvider:
    def __init__(self, shared):
        self.shared = shared

    def generate_detailed(self, prompt, max_new_tokens):
        return self.shared.generate_line_detailed(prompt, max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.suite.read_text().splitlines()]
    for case in rows:
        original = case["source_before"] + case["reference"] + case["source_after"]
        assert hashlib.sha256(original.encode()).hexdigest() == case["source_sha256"]
        assert _parse(original, case["language"]).status == "pass"
    provider = TransformersGenerationProvider(str(args.model), device=args.device)
    metadata = build_prediction_run_metadata(
        suite_path=args.suite,
        case_count=len(rows),
        provider="transformers-first-newline",
        model_source=args.alias,
        model_revision=file_sha256(args.model / "model.safetensors"),
        max_new_tokens=96,
        workers=1,
        protocol="causal_line_v1",
    )
    metadata.update(
        campaign_id="tabcomplete-model-data-r2",
        tokenizer_sha256=file_sha256(args.model / "tokenizer.json"),
        stopping="incremental first token containing newline or EOS; ceiling 96",
        plan_sha256=file_sha256(Path("reports/research/model_data_r2/preregistered_plan.json")),
        model_inventory={
            "class": type(provider.model).__name__,
            "parameters": sum(parameter.numel() for parameter in provider.model.parameters()),
            "tokenizer_class": type(provider.tokenizer).__name__,
            "tokenizer_length": len(provider.tokenizer),
            "config_vocabulary": provider.model.config.vocab_size,
            "precision": str(next(provider.model.parameters()).dtype),
            "device": str(provider.device),
            "torch": provider.torch.__version__,
        },
    )
    cases = [SimpleNamespace(**case) for case in rows]
    predictions = generate_predictions(
        cases,
        LineProvider(provider),
        args.output / "predictions.jsonl",
        run_metadata=metadata,
        max_new_tokens=96,
        workers=1,
        prompt_builder=lambda c: c.prompt,
    )
    records = []
    for case, prediction in zip(rows, predictions, strict=True):
        row = score_line(case, prediction.completion)
        row.update(
            latency_seconds=prediction.latency_seconds,
            returned_line_latency_seconds=prediction.latency_seconds,
            latency_definition="incremental stopping observed through completed generate call",
            input_tokens=len(provider.tokenizer.encode(case["prompt"])),
            output_tokens=prediction.generated_tokens,
            finish_reason=prediction.finish_reason,
            fallback_cap=prediction.finish_reason == "length",
        )
        records.append(row)
    with (args.output / "results.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "total": len(records),
                "exact": sum(r["exact"] for r in records),
                "syntax_pass": sum(r["syntax"] == "pass" for r in records),
                "suite_sha256": metadata["suite_sha256"],
                "model": args.alias,
                "protocol": "causal_line_v1",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    current_runtime().shutdown()


if __name__ == "__main__":
    main()
