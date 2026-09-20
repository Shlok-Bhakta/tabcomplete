"""Index a causal Qwen3.5 checkpoint and its preserved MTP sidecar for GGUF export."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinycomplete.eval.gguf_export import prepare_gguf_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare_gguf_source(args.checkpoint, args.output)
    print(
        f"indexed {result['tensor_count']} tensors ({result['total_size']} source bytes); "
        f"tied output deduplicated: {result['tied_output_deduplicated']}"
    )


if __name__ == "__main__":
    main()
