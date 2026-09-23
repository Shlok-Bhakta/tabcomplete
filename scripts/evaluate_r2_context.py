"""Verify efficient attention, persist dependency scores, then attempt generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tinycomplete.eval.code_generation import TransformersGenerationProvider, file_sha256
from tinycomplete.eval.efficient_attention import explicit_kv_efficient_sdpa
from tinycomplete.eval.long_context_diagnostic import score_target_continuation
from tinycomplete.observability.bootstrap import current_runtime
from tinycomplete.observability.context import current_run_context
from tinycomplete.observability.runs import run_scope


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=2100)
    args = parser.parse_args()
    import torch

    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    provider = TransformersGenerationProvider(str(args.model), device="cuda:0")
    model, tokenizer = provider.model, provider.tokenizer
    weights = sorted(args.model.glob("model*.safetensors"))
    assert len(weights) == 1, "unexpected immutable candidate weight layout"
    inventory = {
        "model_sha256": file_sha256(weights[0]),
        "weight_filename": weights[0].name,
        "tokenizer_sha256": file_sha256(args.model / "tokenizer.json"),
        "suite_sha256": file_sha256(args.suite),
        "model_class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    save(args.output / "model-inventory.json", inventory)
    model.set_attn_implementation("sdpa")
    cases = [json.loads(line) for line in args.suite.read_text().splitlines()]
    selected = list(cases)
    control = next(
        c
        for c in selected
        if c["requested_context_tokens"] == 2048 and c["condition"] == "long_near"
    )
    ordinary = score_target_continuation(
        model, tokenizer, control["prompt"], control["target"], torch.device("cuda")
    )
    ordinary_text = provider.generate_detailed(control["prompt"], 12).text
    with explicit_kv_efficient_sdpa() as observed:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            efficient = score_target_continuation(
                model, tokenizer, control["prompt"], control["target"], torch.device("cuda")
            )
        efficient_text = provider.generate_detailed(control["prompt"], 12).text
        kernels = [
            event.key for event in profiler.key_averages() if "scaled_dot_product" in event.key
        ]
        verification = {
            "stock": ordinary,
            "efficient": efficient,
            "target_mean_absolute_difference": abs(
                ordinary["target_nll_mean"] - efficient["target_nll_mean"]
            ),
            "short_greedy_identical": ordinary_text == efficient_text,
            "profiler_operators": kernels,
            "qkv": observed,
            **inventory,
        }
        save(args.output / "parity.json", verification)
        if (
            verification["target_mean_absolute_difference"] > 0.002
            or not verification["short_greedy_identical"]
            or not any("efficient_attention" in name for name in kernels)
        ):
            raise RuntimeError("efficient backend parity/dispatch gate failed")
        # Order is registered: short controls, then genuine long contexts. Do not
        # classify a short_control in a 32k family as a completed 32k inference.
        selected.sort(
            key=lambda c: (
                {2048: 0, 32000: 1, 4096: 2, 8192: 3, 16384: 4}[c["requested_context_tokens"]],
                c["condition"] != "long_far",
                c["id"],
            )
        )
        scored = []
        with run_scope(args.output / "observability-run.json", "long-context"):
            for case in selected:
                if time.monotonic() - started > args.seconds - 180:
                    break
                actual_tokens = len(tokenizer.encode(case["prompt"], add_special_tokens=False))
                limit = getattr(model.config, "max_position_embeddings", 262144)
                if actual_tokens + 96 > limit:
                    save(
                        args.output / (case["id"].replace("/", "_") + "-excluded.json"),
                        {
                            "case_id": case["id"],
                            "actual_tokens": actual_tokens,
                            "limit": limit,
                            "reason": "model context limit including output",
                        },
                    )
                    continue
                context = current_run_context().for_case(case["id"])
                with context.activate():
                    correct = score_target_continuation(
                        model, tokenizer, case["prompt"], case["target"], torch.device("cuda")
                    )
                    distractor = score_target_continuation(
                        model, tokenizer, case["prompt"], case["distractor"], torch.device("cuda")
                    )
                row = {k: v for k, v in case.items() if k != "prompt"}
                row.update(
                    model=args.alias,
                    actual_prompt_tokens=actual_tokens,
                    correct=correct,
                    distractor_score=distractor,
                    correct_preferred=correct["target_nll_mean"] < distractor["target_nll_mean"],
                )
                with (args.output / "scores.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                scored.append((case, context))
            # Scoring is durable before any strict generation, including all prior cases.
            for case, context in scored:
                if time.monotonic() - started > args.seconds - 30:
                    break
                with context.activate():
                    result = provider.generate_detailed(case["prompt"], 96)
                row = {
                    "case_id": case["id"],
                    "raw_response": result.text,
                    "exact": result.text == case["target"],
                    "tokens": result.tokens,
                    "finish_reason": result.finish_reason,
                }
                with (args.output / "generation.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
        save(args.output / "dispatch.json", observed)
        save(
            args.output / "completion.json",
            {
                "scored_cases": len(scored),
                "planned_cases": len(selected),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    current_runtime().shutdown()


if __name__ == "__main__":
    main()
