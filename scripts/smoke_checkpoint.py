"""Load a downloaded checkpoint and run one deterministic, non-executed prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinycomplete.eval.code_generation import (
    TransformersGenerationProvider,
    model_weight_fingerprint,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    provider = TransformersGenerationProvider(str(args.model_path), device=args.device)
    prompt = "def add(a: int, b: int) -> int:\n    "
    generation = provider.generate_detailed(prompt, max_new_tokens=8)
    result = {
        "schema_version": 1,
        "label": args.label,
        "model_path": str(args.model_path.resolve()),
        "model_file_sha256": sha256_file(args.model_path / "model.safetensors"),
        "model_weight_fingerprint": model_weight_fingerprint(args.model_path),
        "device": args.device,
        "prompt": prompt,
        "completion": generation.text,
        "generated_tokens": generation.tokens,
        "finish_reason": generation.finish_reason,
        "loaded_and_generated": True,
        "generated_code_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
