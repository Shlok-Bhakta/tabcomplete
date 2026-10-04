"""Strict FIM completion diagnostics through the existing model and run helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluate_causal_line import score_line

from tinycomplete.data.fim import format_psm
from tinycomplete.eval.code_generation import (
    DetailedGeneration,
    TransformersGenerationProvider,
    build_prediction_run_metadata,
    file_sha256,
    generate_predictions,
)
from tinycomplete.observability.bootstrap import current_runtime

LINE_SUITE_SHA = "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead"


def paired_development(
    first: list[dict[str, Any]], second: list[dict[str, Any]], *, metric: str
) -> dict[str, Any]:
    """Case-weighted paired difference, resampling whole repository groups."""
    import numpy as np

    if metric not in {"exact", "exact_and_terminated"}:
        raise ValueError("unsupported registered development metric")
    left = {row["case_id"]: row for row in first}
    right = {row["case_id"]: row for row in second}
    if not left or len(left) != len(first) or len(right) != len(second) or set(left) != set(right):
        raise ValueError("paired development identities are empty, duplicate or different")
    groups: dict[str, list[int]] = {}
    wins, losses, shared = [], [], []
    for case_id in sorted(left):
        a, b = left[case_id], right[case_id]
        if (
            not isinstance(a.get("repository"), str)
            or not a["repository"]
            or any(a.get(key) != b.get(key) for key in ("repository", "context_sha256"))
            or not a.get("context_sha256")
            or type(a.get(metric)) is not bool
            or type(b.get(metric)) is not bool
        ):
            raise ValueError("paired development evidence or boolean outcome differs")
        delta = int(b[metric]) - int(a[metric])
        groups.setdefault(a["repository"], []).append(delta)
        if delta > 0:
            wins.append(case_id)
        elif delta < 0:
            losses.append(case_id)
        elif a[metric]:
            shared.append(case_id)
    ordered = [groups[key] for key in sorted(groups)]
    sums = np.array([sum(group) for group in ordered])
    counts = np.array([len(group) for group in ordered])
    rng = np.random.default_rng(271828)
    samples = []
    for _ in range(2000):
        selected = rng.integers(0, len(ordered), len(ordered))
        samples.append(float(sums[selected].sum() / counts[selected].sum()))
    return {
        "metric": metric,
        "cases": len(left),
        "repository_groups": len(groups),
        "first_successes": len(shared) + len(losses),
        "second_successes": len(shared) + len(wins),
        "wins": wins,
        "losses": losses,
        "shared_successes": shared,
        "difference_second_minus_first": float(sums.sum() / counts.sum()),
        "paired_repository_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "bootstrap_samples": 2000,
        "bootstrap_seed": 271828,
        "bootstrap_unit": "repository identity",
        "interpretation": (
            "Synthetic completion. A small or inconclusive difference is not equivalence."
        ),
    }


def development_case(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    ids = row["input_ids"]
    offset = row["prompt_tokens"]
    if ids[-1] != tokenizer.eos_token_id:
        raise ValueError("prepared FIM target lacks its declared EOS")
    prompt = tokenizer.decode(
        ids[:offset], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    target = tokenizer.decode(
        ids[offset:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"] or (
        hashlib.sha256(target.encode()).hexdigest() != row["target_sha256"]
    ):
        raise ValueError("prepared token IDs do not reconstruct the exact FIM strings")
    if tokenizer.encode(prompt, add_special_tokens=False) != ids[:offset]:
        raise ValueError("served tokenizer changes prepared FIM input token IDs")
    return {
        **row,
        "id": f"fim-development-{row['id']}",
        "prompt": prompt,
        "target": target,
        "repository": row["repository_identity_sha256"],
    }


def decoded_completion(
    tokenizer: Any, token_ids: list[int], *, ceiling: int, newline_stop: bool
) -> tuple[str, str, dict[str, Any]]:
    """Remove only an observed terminal EOS; retain malformed control text."""
    eos = tokenizer.eos_token_id
    ended_by_eos = bool(token_ids) and token_ids[-1] == eos
    content_ids = token_ids[:-1] if ended_by_eos else token_ids
    known_ids = set(tokenizer.get_vocab().values())
    unknown = [value for value in content_ids if value not in known_ids]
    unexpected = [value for value in content_ids if value in tokenizer.all_special_ids]
    text = tokenizer.decode(
        content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if unknown or unexpected:
        reason = "invalid_control_or_vocabulary"
    elif ended_by_eos:
        reason = "eos"
    elif newline_stop and "\n" in text:
        reason = "newline"
    elif len(token_ids) >= ceiling:
        reason = "length"
    else:
        reason = "unverified_stop"
    return (
        text,
        reason,
        {
            "output_token_ids": token_ids,
            "ended_by_eos": ended_by_eos,
            "unexpected_special_token_ids": unexpected,
            "unknown_token_ids": unknown,
            "reached_token_ceiling": len(token_ids) >= ceiling,
            "truncated": reason == "length",
        },
    )


class StrictFimProvider(TransformersGenerationProvider):
    def __init__(self, model_path: str, *, newline_stop: bool, evidence_path: Path) -> None:
        super().__init__(model_path, device="cuda:0")
        self.newline_stop = newline_stop
        self.evidence: dict[str, dict[str, Any]] = {}
        self.evidence_path = evidence_path
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        if evidence_path.exists():
            for line in evidence_path.read_text().splitlines():
                row = json.loads(line)
                self.evidence[row["context_sha256"]] = row

    def _generate(
        self, prompt: str, max_new_tokens: int, *, stop_first_line: bool
    ) -> DetailedGeneration:
        inputs = {
            name: value.to(self.device)
            for name, value in self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).items()
        }
        prefix_length = int(inputs["input_ids"].shape[1])
        extra: dict[str, Any] = {}
        if self.newline_stop:
            from transformers import StoppingCriteria, StoppingCriteriaList

            tokenizer = self.tokenizer

            class FirstNewline(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    return "\n" in tokenizer.decode(
                        input_ids[0, prefix_length:],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )

            extra["stopping_criteria"] = StoppingCriteriaList([FirstNewline()])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **extra,
            )
        ids = generated[0, prefix_length:].tolist()
        text, reason, evidence = decoded_completion(
            self.tokenizer, ids, ceiling=max_new_tokens, newline_stop=self.newline_stop
        )
        key = hashlib.sha256(prompt.encode()).hexdigest()
        evidence.update(
            context_sha256=key,
            completion_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        self.evidence[key] = evidence
        with self.evidence_path.open("a") as handle:
            handle.write(json.dumps(evidence, sort_keys=True) + "\n")
        return DetailedGeneration(
            text, len(ids), reason, input_tokens=prefix_length, usage_source="tokenizer"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--mode", choices=("development", "line"), required=True)
    args = parser.parse_args()
    if not args.model.is_dir() or not (args.model / "model.safetensors").is_file():
        raise ValueError("FIM evaluation requires existing local weights; downloads are forbidden")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != "q25-fim-training-plan-v1":
        raise ValueError("a frozen FIM training/evaluation plan is required")
    expected = (
        plan["data"]["development"]["sha256"] if args.mode == "development" else LINE_SUITE_SHA
    )
    if file_sha256(args.input) != expected:
        raise ValueError("FIM evaluation input identity differs")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.mode == "line" and len(rows) != 180:
        raise ValueError("the unchanged line suite must contain 180 cases")
    if args.mode == "development" and len(rows) != plan["data"]["development"]["row_count"]:
        raise ValueError("FIM development state count differs")
    provider = StrictFimProvider(
        str(args.model),
        newline_stop=args.mode == "line",
        evidence_path=args.output / "token_evidence.jsonl",
    )
    if (
        type(provider.model).__name__ != "Qwen2ForCausalLM"
        or sum(parameter.numel() for parameter in provider.model.parameters()) != 494032768
    ):
        raise ValueError("FIM evaluation loaded a different model architecture or size")
    if args.mode == "development":
        rows = [development_case(row, provider.tokenizer) for row in rows]
    prepared = []
    for row in rows:
        item = dict(row)
        item["prompt"] = (
            format_psm(row["source_before"], row["source_after"])
            if args.mode == "line"
            else row["prompt"]
        )
        prepared.append(SimpleNamespace(**item))
    metadata = build_prediction_run_metadata(
        suite_path=args.input,
        case_count=len(rows),
        provider="transformers-strict-fim",
        model_source=args.alias,
        model_revision=file_sha256(args.model / "model.safetensors"),
        max_new_tokens=96,
        workers=1,
        protocol=f"q25-fim-{args.mode}-v1",
    )
    metadata.update(
        plan_sha256=file_sha256(args.plan),
        tokenizer_sha256=file_sha256(args.model / "tokenizer.json"),
        precision="fp16",
        runtime_revision=str(provider.torch.__version__),
        model_inventory={
            "class": type(provider.model).__name__,
            "parameters": sum(p.numel() for p in provider.model.parameters()),
            "device": provider.torch.cuda.get_device_name(0),
        },
        stopping="observed EOS only" if args.mode == "development" else "registered newline or EOS",
        control_tokens="retained and invalidated; only terminal EOS removed",
        task="synthetic FIM completion, not observed next-edit intent",
    )
    predictions = generate_predictions(
        prepared,
        provider,
        args.output / "predictions.jsonl",
        run_metadata=metadata,
        max_new_tokens=96,
        workers=1,
        prompt_builder=lambda case: case.prompt,
    )
    records = []
    for row, case, prediction in zip(rows, prepared, predictions, strict=True):
        record = (
            score_line(row, prediction.completion)
            if args.mode == "line"
            else {
                "case_id": row["id"],
                "repository": row["repository"],
                "language": row["language"],
                "exact": prediction.completion == row["target"],
                "raw_response": prediction.completion,
                "syntax": "not_measured",
            }
        )
        context_sha = hashlib.sha256(case.prompt.encode()).hexdigest()
        evidence = provider.evidence[context_sha]
        if (
            evidence["completion_sha256"]
            != hashlib.sha256(prediction.completion.encode()).hexdigest()
        ):
            raise ValueError("persisted token evidence and prediction text differ")
        terminated = evidence["ended_by_eos"] or prediction.finish_reason == "newline"
        valid_tokens = (
            not evidence["unexpected_special_token_ids"] and not evidence["unknown_token_ids"]
        )
        record.update(
            terminated=terminated,
            valid_output_tokens=valid_tokens,
            exact_and_terminated=bool(record["exact"]) and terminated and valid_tokens,
            finish_reason=prediction.finish_reason,
            input_tokens=len(provider.tokenizer.encode(case.prompt, add_special_tokens=False)),
            output_tokens=prediction.generated_tokens,
            latency_seconds=prediction.latency_seconds,
            context_sha256=context_sha,
        )
        record.update(evidence)
        records.append(record)
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    summary = {
        "cases": len(records),
        "exact": sum(row["exact"] for row in records),
        "exact_and_terminated": sum(row["exact_and_terminated"] for row in records),
        "terminated": sum(row["terminated"] for row in records),
        "valid_output_tokens": sum(row["valid_output_tokens"] for row in records),
        "empty_outputs": sum(not row["raw_response"] for row in records),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in records)),
        "syntax_pass": sum(row["syntax"] == "pass" for row in records),
        "syntax_denominator": len(records) if args.mode == "line" else 0,
        "quality_caveat": (
            "Synthetic source completion; no general next-edit or human acceptance claim."
        ),
        "plan_sha256": metadata["plan_sha256"],
        "model_sha256": metadata["model_revision"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    current_runtime().shutdown()


if __name__ == "__main__":
    main()
