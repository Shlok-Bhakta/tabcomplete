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
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tinycomplete.observability.context import current_run_context
from tinycomplete.observability.runs import run_scope
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


def native_observation(
    row: dict[str, Any], context: Any, generated: Any, backend: dict[str, Any]
) -> dict[str, Any]:
    state = EditState.from_mapping(row["state"])
    tokens = generated.tokens
    if type(tokens) is not int or tokens < 0:
        raise ValueError("native runtime did not report output-token count")
    terminated = generated.finish_reason == "eos" and backend.get("truncated") is False
    decoded = decode_action(generated.text, terminated=terminated, generated_tokens=tokens)
    after = None
    if decoded.action is not None:
        try:
            after = apply_action(state, decoded.action)
        except ValueError:
            pass
    return {
        "id": row["id"],
        "state_sha256": hashlib.sha256(
            json.dumps(row["state"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(state.source.encode()).hexdigest(),
        "context_sha256": hashlib.sha256(context.text.encode()).hexdigest(),
        "gold_action": row["action"]["kind"],
        "predicted_action": decoded.action.kind if after is not None and decoded.action else None,
        "valid_action": after is not None,
        "exact_after": after == row["after_source"] if after is not None else False,
        "source_changed": after != state.source if after is not None else None,
        "canonical_action": (
            asdict(decoded.action) if after is not None and decoded.action else None
        ),
        "terminated_by_eos": terminated,
        "generated_tokens": tokens,
        "input_tokens": context.input_tokens,
        "output_bytes": len(generated.text.encode()),
        "elapsed_ms": backend["total_seconds"] * 1000,
        "wire": generated.text,
        "backend": backend,
    }


def evaluate_native(
    model_file: Path,
    rows: list[dict[str, Any]],
    *,
    server_url: str,
    server_pid: int,
    model_weight_sha256: str,
    source_weight_sha256: str,
    data_sha256: str,
    weight_precision: str = "Q4_K_M",
    calibration_fixture: Path | None = None,
    calibration_manifest: Path | None = None,
) -> dict[str, Any]:
    """Use the existing native provider and runtime tokenizer, with exact EOS evidence."""
    import httpx

    if weight_precision not in {"Q4_K_M", "Q5_K_M"}:
        raise ValueError("native weight precision must match an authorized conversion")
    parsed = urlsplit(server_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("native pilot evaluation requires the configured loopback service")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("native endpoint must not contain credentials or query parameters")
    if sha256_file(model_file) != model_weight_sha256:
        raise ValueError("native model weight identity differs from the frozen plan")
    proc = Path(f"/proc/{server_pid}")
    command = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    if any(arg in {"--api-key", "--api-key-file"} for arg in command):
        raise ValueError("native process configuration must not expose authentication arguments")
    for flag, value in (("-ngl", "0"), ("-np", "1")):
        indices = [i for i, arg in enumerate(command) if arg == flag]
        if (len(indices) != 1 or indices[0] + 1 >= len(command)
                or command[indices[0] + 1] != value):
            raise ValueError("native pilot requires the declared CPU backend and one slot")
    model_args = [i for i, value in enumerate(command) if value in {"-m", "--model"}]
    if (len(model_args) != 1 or model_args[0] + 1 >= len(command)
            or Path(command[model_args[0] + 1]).resolve() != model_file.resolve()):
        raise ValueError("running native process uses a different model")
    binary = proc.joinpath("exe").resolve(strict=True)
    if binary.name != "llama-server":
        raise ValueError("native evaluator requires the pinned llama-server process")
    from measure_r2_local import NativeProvider
    from replay_small_model import proc_memory

    provider = NativeProvider(server_url, "q25-coder-adapted-Q4_K_M")
    provider.cache = False
    provider.timeout_seconds = 120
    memory_before = proc_memory(server_pid)
    observations = []
    with httpx.Client(base_url=server_url, timeout=10) as client:
        class RuntimeTokenizer:
            def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
                response = client.post("/tokenize", json={
                    "content": text, "add_special": add_special_tokens,
                    "parse_special": True,
                })
                response.raise_for_status()
                tokens = response.json().get("tokens")
                if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                    raise ValueError("runtime tokenizer returned malformed token IDs")
                return tokens

        tokenizer = RuntimeTokenizer()
        for row in rows:
            if proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0") != command:
                raise ValueError("native process identity changed during evaluation")
            started = time.perf_counter()
            context = serialize_state_bounded(EditState.from_mapping(row["state"]), tokenizer)
            construction_ms = (time.perf_counter() - started) * 1000
            run = current_run_context()
            with run.for_case(row["id"]).activate() if run else nullcontext():
                generated = provider.generate_detailed(context.text, 64)
            if proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0") != command:
                raise ValueError("native process identity changed during generation")
            observed = native_observation(row, context, generated, dict(provider.last))
            observed["context_construction_ms"] = construction_ms
            observed["post_request_memory"] = proc_memory(server_pid)
            observations.append(observed)
        calibration = None
        if calibration_fixture is not None and calibration_manifest is not None:
            cases = load_calibration_cases(calibration_fixture, calibration_manifest)
            predictions = []
            for item in calibration_prompts(cases, tokenizer):
                run = current_run_context()
                with run.for_case(item["case_id"]).activate() if run else nullcontext():
                    generated = provider.generate_detailed(item["prompt"], 64)
                terminated = generated.finish_reason == "eos" and provider.last.get(
                    "truncated"
                ) is False
                predictions.append(Prediction(
                    item["case_id"], generated.text, terminated, generated_tokens=generated.tokens
                ))
            calibration = score_calibration_predictions(cases, predictions)
    result = {
        "identity": {
            "model_weight_sha256": model_weight_sha256,
            "source_weight_sha256": source_weight_sha256,
            "development_sha256": data_sha256,
            "runtime_binary_sha256": sha256_file(binary),
            "runtime_command": command,
            "server_pid": server_pid,
            "precision": weight_precision + " weights, CPU native runtime",
            "tokenizer": "resident llama-server /tokenize, add_special=true",
            "context_policy_version": "single-line-context-v2",
            "evaluator_sha256": sha256_file(Path(__file__)),
            "cache_prompt": False,
        },
        "cases": len(observations),
        "valid_actions": sum(bool(item["valid_action"]) for item in observations),
        "explicit_termination": sum(bool(item["terminated_by_eos"]) for item in observations),
        "exact_after": sum(bool(item["exact_after"]) for item in observations),
        "actions": dict(Counter(item["predicted_action"] or "invalid" for item in observations)),
        "task_outcomes": action_outcome_counts(observations),
        "memory_before": memory_before,
        "memory_after": proc_memory(server_pid),
        "observations": observations,
    }
    if calibration is not None:
        assert calibration_fixture is not None
        result["synthetic_calibration"] = calibration
        result["synthetic_calibration_fixture_sha256"] = sha256_file(calibration_fixture)
    return result


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
    parser.add_argument("--server-url", help="optional resident native loopback backend")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--precision", choices=("Q4_K_M", "Q5_K_M"), default="Q4_K_M")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evaluation output already exists")
    if (args.calibration_fixture is None) != (args.calibration_manifest is None):
        raise ValueError("calibration fixture and manifest must be supplied together")
    rows = load_rows(args.development, args.development_sha256)
    if (args.server_url is None) != (args.server_pid is None):
        raise ValueError("native backend requires both endpoint and process identity")
    evaluator = evaluate_native if args.server_url is not None else evaluate
    native_args = {"server_url": args.server_url, "server_pid": args.server_pid,
                   "weight_precision": args.precision} if (
        args.server_url is not None
    ) else {}
    with run_scope(args.output.with_suffix(".observability.json"), "one-line-development"):
        result = evaluator(
            args.model,
            rows,
            model_weight_sha256=args.model_weight_sha256,
            source_weight_sha256=args.source_weight_sha256,
            data_sha256=args.development_sha256,
            calibration_fixture=args.calibration_fixture,
            calibration_manifest=args.calibration_manifest,
            **native_args,
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
