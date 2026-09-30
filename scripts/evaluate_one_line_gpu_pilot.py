"""Evaluate a frozen pilot development shard with exact compact-action decoding.

Edit-required and no-edit denominators remain separate. Exact target agreement
is not a functional oracle; independently checked synthetic behavior is evaluated
separately after retrieving the predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinycomplete.one_line.context import serialize_state_bounded
from tinycomplete.one_line.contract import EditAction, EditState, apply_action, decode_action
from tinycomplete.one_line.evaluate import (
    Prediction,
    calibration_prompts,
    load_calibration_cases,
    score_calibration_predictions,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resident_bytes() -> int:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def action_outcome_counts(observations: list[dict[str, Any]]) -> dict[str, Any]:
    edit = [row for row in observations if row["gold_action"] != "keep"]
    keep = [row for row in observations if row["gold_action"] == "keep"]
    return {
        "edit_required_cases": len(edit),
        "edit_required_exact": sum(bool(row["exact_after"]) for row in edit),
        "edit_required_predicted_keep": sum(row["predicted_action"] == "keep" for row in edit),
        "no_edit_cases": len(keep),
        "no_edit_recalled": sum(row["predicted_action"] == "keep" for row in keep),
        "no_edit_false_positive_changes": sum(row.get("source_changed") is True for row in keep),
        "no_edit_invalid": sum(not row["valid_action"] for row in keep),
        "gold_action_breakdown": {
            kind: {
                "cases": sum(row["gold_action"] == kind for row in observations),
                "exact": sum(
                    row["gold_action"] == kind and bool(row["exact_after"]) for row in observations
                ),
            }
            for kind in ("keep", "replace_line", "insert_before", "delete_line")
        },
        "functional_success": None,
        "functional_status": "requires independent objective evaluation",
    }


def load_rows(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise ValueError("development shard SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate development IDs")
    for row in rows:
        if row["split"] != "development":
            raise ValueError("evaluation input contains non-development row")
        state = EditState.from_mapping(row["state"])
        action = EditAction(**row["action"])
        if apply_action(state, action) != row["after_source"]:
            raise ValueError("development action does not replay")
    return rows


def evaluate(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    model_weight_sha256: str,
    source_weight_sha256: str,
    data_sha256: str,
    calibration_fixture: Path | None = None,
    calibration_manifest: Path | None = None,
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("pilot evaluation requires the allocated CUDA session")
    weights = model_dir / "model.safetensors"
    if sha256_file(weights) != model_weight_sha256:
        raise ValueError("evaluation model weight hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(torch.device("cuda:0"))
    model.eval()
    torch.cuda.reset_peak_memory_stats(0)
    observations: list[dict[str, Any]] = []
    for row in rows:
        state = EditState.from_mapping(row["state"])
        context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
        ids = tokenizer.encode(context.text, add_special_tokens=True, return_tensors="pt").to(
            "cuda:0"
        )
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                ids,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize(0)
        elapsed_ms = (time.perf_counter() - started) * 1000
        generated = output[0, ids.shape[1] :].tolist()
        terminated = bool(generated and generated[-1] == tokenizer.eos_token_id)
        wire_ids = generated[:-1] if terminated else generated
        wire = tokenizer.decode(
            wire_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not isinstance(wire, str):
            raise ValueError("tokenizer decoded a non-text action")
        decoded = decode_action(wire, terminated=terminated, generated_tokens=len(generated))
        after: str | None = None
        if decoded.action is not None:
            try:
                after = apply_action(state, decoded.action)
            except ValueError:
                pass
        observations.append(
            {
                "id": row["id"],
                "state_sha256": hashlib.sha256(
                    json.dumps(row["state"], sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                "source_sha256": hashlib.sha256(state.source.encode("utf-8")).hexdigest(),
                "context_sha256": hashlib.sha256(context.text.encode("utf-8")).hexdigest(),
                "gold_action": row["action"]["kind"],
                "predicted_action": (
                    decoded.action.kind
                    if after is not None and decoded.action is not None
                    else None
                ),
                "valid_action": after is not None,
                "exact_after": after == row["after_source"] if after is not None else False,
                "source_changed": after != state.source if after is not None else None,
                "canonical_action": (
                    asdict(decoded.action)
                    if after is not None and decoded.action is not None
                    else None
                ),
                "terminated_by_eos": terminated,
                "generated_tokens": len(generated),
                "input_tokens": context.input_tokens,
                "output_bytes": len(wire.encode("utf-8")),
                "elapsed_ms": elapsed_ms,
                "wire": wire,
            }
        )
    actions = Counter(item["predicted_action"] or "invalid" for item in observations)
    result = {
        "identity": {
            "model_weight_sha256": model_weight_sha256,
            "source_weight_sha256": source_weight_sha256,
            "development_sha256": data_sha256,
            "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "precision": "F16 compute",
        },
        "cases": len(observations),
        "valid_actions": sum(bool(item["valid_action"]) for item in observations),
        "explicit_termination": sum(bool(item["terminated_by_eos"]) for item in observations),
        "exact_after": sum(bool(item["exact_after"]) for item in observations),
        "actions": dict(actions),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "retained_cuda_allocated_bytes": torch.cuda.memory_allocated(0),
        "retained_process_rss_bytes": resident_bytes(),
        "observations": observations,
        "task_outcomes": action_outcome_counts(observations),
    }
    if calibration_fixture is not None and calibration_manifest is not None:
        cases = load_calibration_cases(calibration_fixture, calibration_manifest)
        predictions: list[Prediction] = []
        for item in calibration_prompts(cases, tokenizer):
            ids = tokenizer.encode(item["prompt"], add_special_tokens=True, return_tensors="pt").to(
                "cuda:0"
            )
            with torch.inference_mode():
                output = model.generate(
                    ids,
                    max_new_tokens=64,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            generated = output[0, ids.shape[1] :].tolist()
            terminated = bool(generated and generated[-1] == tokenizer.eos_token_id)
            wire = tokenizer.decode(
                generated[:-1] if terminated else generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(wire, str):
                raise ValueError("tokenizer decoded a non-text calibration action")
            predictions.append(
                Prediction(item["case_id"], wire, terminated, generated_tokens=len(generated))
            )
        result["synthetic_calibration"] = score_calibration_predictions(cases, predictions)
        result["synthetic_calibration_fixture_sha256"] = sha256_file(calibration_fixture)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-weight-sha256", required=True)
    parser.add_argument("--source-weight-sha256", required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-sha256", required=True)
    parser.add_argument("--calibration-fixture", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evaluation output already exists")
    if (args.calibration_fixture is None) != (args.calibration_manifest is None):
        raise ValueError("calibration fixture and manifest must be supplied together")
    rows = load_rows(args.development, args.development_sha256)
    result = evaluate(
        args.model,
        rows,
        model_weight_sha256=args.model_weight_sha256,
        source_weight_sha256=args.source_weight_sha256,
        data_sha256=args.development_sha256,
        calibration_fixture=args.calibration_fixture,
        calibration_manifest=args.calibration_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "cases": result["cases"],
                "valid_actions": result["valid_actions"],
                "exact_after": result["exact_after"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
