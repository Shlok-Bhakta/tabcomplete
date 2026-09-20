"""Prepare a Qwen3.5 causal checkpoint and preserved MTP sidecar for GGUF export."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def prepare_gguf_source(checkpoint: Path, output: Path) -> dict:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_weights = [checkpoint / "model.safetensors", checkpoint / "mtp-original.safetensors"]
    missing = [path.name for path in source_weights if not path.is_file()]
    if missing:
        raise ValueError(f"checkpoint is missing required weights: {', '.join(missing)}")

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    excluded_tensors: set[str] = set()
    tied_output_deduplicated = False
    if config.get("tie_word_embeddings"):
        with safe_open(source_weights[0], framework="pt") as handle:
            keys = set(handle.keys())
            output_name = "lm_head.weight"
            embedding_name = "model.language_model.embed_tokens.weight"
            if output_name in keys and embedding_name in keys:
                output_weight = handle.get_tensor(output_name)
                embedding_weight = handle.get_tensor(embedding_name)
                if not torch.equal(output_weight, embedding_weight):
                    raise ValueError("tied output and embedding tensors are not identical")
                excluded_tensors.add(output_name)
                tied_output_deduplicated = True

    metadata_files = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    for name in metadata_files:
        source = checkpoint / name
        if source.exists():
            (output / name).symlink_to(source)

    weight_map: dict[str, str] = {}
    total_size = 0
    for index, source in enumerate(source_weights, 1):
        local_name = f"model-{index:05d}-of-00002.safetensors"
        local_path = output / local_name
        if index == 1 and tied_output_deduplicated:
            with safe_open(source, framework="pt") as handle:
                tensors = {
                    name: handle.get_tensor(name)
                    for name in handle.keys()
                    if name not in excluded_tensors
                }
                save_file(tensors, local_path, metadata=handle.metadata())
        else:
            local_path.symlink_to(source)
        total_size += local_path.stat().st_size
        with safe_open(local_path, framework="pt") as handle:
            for tensor_name in handle.keys():
                if tensor_name in weight_map:
                    raise ValueError(f"duplicate tensor in checkpoint: {tensor_name}")
                weight_map[tensor_name] = local_name

    index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "tensor_count": len(weight_map),
        "total_size": total_size,
        "tied_output_deduplicated": tied_output_deduplicated,
    }
