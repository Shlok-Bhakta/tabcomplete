"""Install the FIM-only attention policy before running a frozen raw regression."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from tinycomplete.eval.q25_fim_attention import (
    ATTENTION_BACKEND,
    install_q25_fim_attention,
)

_POLICY_METADATA = {
    "attention_backend": ATTENTION_BACKEND,
    "key_value_head_expansion": "explicit-repeat",
    "attention_heads": {"query": 14, "key_value": 2, "head_dim": 64},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("causal", "line"), required=True)
    parser.add_argument("--attention-backend", required=True)
    args, forwarded = parser.parse_known_args()
    if args.attention_backend != ATTENTION_BACKEND:
        parser.error("the FIM regression attention backend differs from the frozen policy")
    try:
        output_index = forwarded.index("--output")
        output_directory = Path(forwarded[output_index + 1])
        plan_sha_index = forwarded.index("--plan-sha")
        plan_sha = forwarded[plan_sha_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("raw regression plan or output identity is missing") from exc
    if len(plan_sha) != 64 or any(character not in "0123456789abcdef" for character in plan_sha):
        raise ValueError("raw regression plan identity is invalid")
    metadata_path = output_directory / "predictions.jsonl.metadata.json"
    if metadata_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in _POLICY_METADATA.items():
            if key in saved and saved[key] != value:
                raise ValueError("existing raw regression metadata uses another attention policy")
            saved.pop(key, None)
        metadata_path.write_text(
            json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    script_name = "evaluate_q25_code_cpt.py" if args.kind == "causal" else "evaluate_causal_line.py"
    script_path = Path(__file__).resolve().with_name(script_name)
    if not script_path.is_file():
        raise FileNotFoundError("frozen FIM regression evaluator is unavailable")
    install_q25_fim_attention(ATTENTION_BACKEND)
    sys.argv = [str(script_path), *forwarded]
    runpy.run_path(str(script_path), run_name="__main__")
    if not metadata_path.is_file():
        raise ValueError("raw regression metadata was not produced")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("plan_sha256") != plan_sha:
        raise ValueError("raw regression output plan identity differs")
    metadata.update(_POLICY_METADATA)
    serialized = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    metadata_path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
