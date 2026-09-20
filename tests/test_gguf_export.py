"""GGUF export preparation keeps the preserved Qwen3.5 MTP tensors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tinycomplete.eval.gguf_export import prepare_gguf_source


def test_prepare_source_indexes_main_and_mtp_weights(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"mtp_num_hidden_layers": 1, "tie_word_embeddings": true}\n'
    )
    save_file(
        {
            "lm_head.weight": torch.ones(2),
            "model.language_model.embed_tokens.weight": torch.ones(2),
            "model.layer.weight": torch.ones(2),
        },
        checkpoint / "model.safetensors",
    )
    save_file({"mtp.fc.weight": torch.ones(3)}, checkpoint / "mtp-original.safetensors")

    output = tmp_path / "gguf-source"
    result = prepare_gguf_source(checkpoint, output)
    index = json.loads((output / "model.safetensors.index.json").read_text())

    assert result["tensor_count"] == 3
    assert result["tied_output_deduplicated"] is True
    assert index["weight_map"] == {
        "model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.layer.weight": "model-00001-of-00002.safetensors",
        "mtp.fc.weight": "model-00002-of-00002.safetensors",
    }
    assert (output / "config.json").is_symlink()
    assert not (output / "model-00001-of-00002.safetensors").is_symlink()


def test_prepare_source_refuses_nonempty_output(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user data\n")
    with pytest.raises(ValueError, match="not empty"):
        prepare_gguf_source(tmp_path / "checkpoint", output)


def test_prepare_source_refuses_diverged_tied_weights(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"tie_word_embeddings": true}\n')
    save_file(
        {
            "lm_head.weight": torch.zeros(2),
            "model.language_model.embed_tokens.weight": torch.ones(2),
        },
        checkpoint / "model.safetensors",
    )
    save_file({"mtp.fc.weight": torch.ones(1)}, checkpoint / "mtp-original.safetensors")
    with pytest.raises(ValueError, match="not identical"):
        prepare_gguf_source(checkpoint, tmp_path / "output")
