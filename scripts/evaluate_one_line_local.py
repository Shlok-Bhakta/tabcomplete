"""Run the frozen one-line development calibration on a verified local q25 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from tinycomplete.one_line.evaluate import (
    Prediction,
    calibration_prompts,
    load_calibration_cases,
    score_calibration_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    / "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
)
WEIGHT_SHA256 = "aff8914ec707fcaf9e2d4dc97197cded50b1c63e1d3a7a82e56f54d83ea47f80"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resident_bytes() -> int:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "reports/research/one_line_r1/calibration_cases.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "reports/research/one_line_r1/calibration_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/research/one_line_r1/evaluations/untouched_q25_calibration.json",
    )
    args = parser.parse_args()
    if sha256_file(args.model / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("untouched q25 weight hash mismatch")
    if sha256_file(args.model / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("q25 tokenizer hash mismatch")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    cases = load_calibration_cases(args.fixture, args.manifest)
    prompts = calibration_prompts(cases, tokenizer)
    loaded_at = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - loaded_at
    peak_rss = resident_bytes()
    predictions: list[Prediction] = []
    observations = []
    for item in prompts:
        case_id = item["case_id"]
        prompt = item["prompt"]
        if not isinstance(case_id, str) or not isinstance(prompt, str):
            raise ValueError("invalid calibration prompt")
        ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
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
        seconds = time.perf_counter() - started
        generated = output[0, ids.shape[1] :].tolist()
        terminated = bool(generated and generated[-1] == tokenizer.eos_token_id)
        wire_ids = generated[:-1] if terminated else generated
        wire = tokenizer.decode(
            wire_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not isinstance(wire, str):
            raise ValueError("tokenizer returned non-text calibration output")
        predictions.append(Prediction(case_id, wire, terminated, generated_tokens=len(generated)))
        observations.append(
            {
                "case_id": case_id,
                "input_tokens": item["input_tokens"],
                "generated_tokens": len(generated),
                "terminated_by_eos": terminated,
                "wire": wire,
                "elapsed_seconds": seconds,
                "resident_bytes_after": resident_bytes(),
            }
        )
        peak_rss = max(peak_rss, observations[-1]["resident_bytes_after"])
    scores = score_calibration_predictions(cases, predictions)
    result = {
        "identity": {
            "model": "Qwen/Qwen2.5-Coder-0.5B",
            "revision": "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
            "weight_sha256": WEIGHT_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "fixture_sha256": sha256_file(args.fixture),
            "manifest_sha256": sha256_file(args.manifest),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "backend": "CPU",
            "threads": 2,
            "precision": "FP32",
        },
        "load_seconds": load_seconds,
        "peak_observed_rss_bytes": peak_rss,
        "retained_rss_bytes": resident_bytes(),
        "observations": observations,
        "scores": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "summary": scores["summary"]}))


if __name__ == "__main__":
    main()
