"""Evaluate campaign-owned Q4 conversions through an isolated native server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from evaluate_causal_line import LineProvider, score_line
from measure_r2_local import NativeProvider

from tinycomplete.eval.code_benchmark import load_suite
from tinycomplete.eval.code_generation import (
    build_prediction_run_metadata,
    file_sha256,
    generate_predictions,
)
from tinycomplete.observability.bootstrap import current_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--line-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--precision", choices=("Q4_K_M", "F16"), default="Q4_K_M")
    parser.add_argument("--line-only", action="store_true")
    parser.add_argument("--campaign-id", default="tabcomplete-model-data-r2")
    parser.add_argument(
        "--plan", type=Path, default=Path("reports/research/model_data_r2/preregistered_plan.json")
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=2304)
    parser.add_argument("--suite-revision", default="r2-original")
    args = parser.parse_args()
    httpx.get(args.url + "/health", timeout=10).raise_for_status()
    provider = NativeProvider(args.url, args.alias)
    model_hash = file_sha256(args.model)
    strict = Path("data/benchmarks/code_completion_v2.jsonl")
    for suite, protocol in ((strict, "causal-context-v1"), (args.line_suite, "causal_line_v1")):
        line_mode = protocol == "causal_line_v1"
        if args.line_only and not line_mode:
            continue
        raw_cases = [json.loads(line) for line in suite.read_text().splitlines()]
        cases = [SimpleNamespace(**row) for row in raw_cases] if line_mode else load_suite(suite)
        metadata = build_prediction_run_metadata(
            suite_path=suite,
            case_count=len(cases),
            provider="llama.cpp-native",
            model_source=args.alias,
            model_revision=model_hash,
            max_new_tokens=96,
            workers=1,
            protocol=protocol,
        )
        metadata.update(
            campaign_id=args.campaign_id,
            precision=args.precision,
            runtime_revision="f072b103714dfa1eee531f80b24512faf38e3dd2",
            hardware=args.hardware,
            threads=args.threads,
            runtime_context_tokens=args.context_tokens,
            quality_suite_revision=args.suite_revision,
            concurrency=1,
            stopping="native first newline or EOS, ceiling 96"
            if line_mode
            else "EOS or ceiling 96",
            raw_response_definition="exact native API text; native stop string is omitted",
            plan_sha256=file_sha256(args.plan),
        )
        kwargs = {"prompt_builder": lambda case: case.prompt} if line_mode else {}
        predictions = generate_predictions(
            cases,
            LineProvider(provider) if line_mode else provider,
            args.output / protocol / "predictions.jsonl",
            run_metadata=metadata,
            max_new_tokens=96,
            workers=1,
            **kwargs,
        )
        if line_mode:
            records = []
            for case, prediction in zip(raw_cases, predictions, strict=True):
                row = score_line(
                    case,
                    prediction.completion,
                    native_newline_omitted=prediction.finish_reason == "word",
                )
                row.update(
                    latency_seconds=prediction.latency_seconds,
                    returned_line_latency_seconds=prediction.latency_seconds,
                    output_tokens=prediction.generated_tokens,
                    finish_reason=prediction.finish_reason,
                    fallback_cap=prediction.hit_token_cap,
                    latency_definition="native incremental stop; complete response receipt",
                )
                records.append(row)
            target = args.output / protocol
            (target / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
            (target / "summary.json").write_text(
                json.dumps(
                    {
                        "total": len(records),
                        "exact": sum(row["exact"] for row in records),
                        "syntax_pass": sum(row["syntax"] == "pass" for row in records),
                        "provider_errors": 0,
                        "metadata": metadata,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(
            json.dumps({"model": args.alias, "protocol": protocol, "cases": len(predictions)}),
            flush=True,
        )
    current_runtime().shutdown()


if __name__ == "__main__":
    main()
