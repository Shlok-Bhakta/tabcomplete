"""Score the v2 long-context diagnostic with target-only logits and strict generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tinycomplete.eval.long_context_diagnostic import (
    LongContextDiagnosticCase,
    score_target_continuation,
    verify_selected_scoring,
)


def model_hash(model_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_path.glob("*.safetensors")):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 2**20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, nargs="*")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 96:
        parser.error("--max-new-tokens must be between 1 and 96")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path, trust_remote_code=False
    )
    tokenizer_source = (args.tokenizer_path or args.model_path).resolve()
    tokenizer_sha256 = hashlib.sha256(
        (tokenizer_source / "tokenizer.json").read_bytes()
    ).hexdigest()
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=False,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = model.to("cuda")
    model.eval()
    cases = [
        LongContextDiagnosticCase.model_validate_json(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.context_tokens:
        allowed = set(args.context_tokens)
        cases = [case for case in cases if case.requested_context_tokens in allowed]
    if not cases:
        raise ValueError("no diagnostic cases selected")
    short = min(cases, key=lambda case: case.prompt_tokens)
    scorer_check = verify_selected_scoring(
        model, tokenizer, short.prompt, short.target, torch.device("cuda")
    )
    if scorer_check["absolute_difference"] > 1e-4:
        raise RuntimeError(f"selected-logit scorer differs from full logits: {scorer_check}")
    weight_sha256 = model_hash(args.model_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            correct = score_target_continuation(
                model, tokenizer, case.prompt, case.target, torch.device("cuda")
            )
            distractor = score_target_continuation(
                model, tokenizer, case.prompt, case.distractor, torch.device("cuda")
            )
            prompt_ids = tokenizer.encode(case.prompt, add_special_tokens=False)
            inputs = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            continuation = generated[0, len(prompt_ids) :]
            generated_tokens = int(continuation.numel())
            text = tokenizer.decode(continuation, skip_special_tokens=True)
            record = {
                **case.model_dump(exclude={"prompt"}),
                "model_label": args.model_label,
                "model_sha256": weight_sha256,
                "tokenizer_source": str(tokenizer_source),
                "tokenizer_sha256": tokenizer_sha256,
                "scorer_verification": scorer_check,
                "correct": correct,
                "distractor_score": distractor,
                "correct_preferred": (
                    correct["target_nll_mean"] < distractor["target_nll_mean"]
                ),
                "generated_text": text,
                "generated_tokens": generated_tokens,
                "generation_exact": text == case.target,
                "generation_finish_reason": (
                    "length" if generated_tokens >= args.max_new_tokens else "eos_or_stop"
                ),
                "generation_truncated": generated_tokens >= args.max_new_tokens,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{index:03d}/{len(cases)}] {case.id} "
                f"preferred={record['correct_preferred']} exact={record['generation_exact']}"
            )


if __name__ == "__main__":
    main()
