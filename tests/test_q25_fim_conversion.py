from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any, cast

import pytest

from tinycomplete.code_cpt import q25_fim_conversion as conversion


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversion, "MODEL_ID", "Qwen/test-coder")
    monkeypatch.setattr(conversion, "MODEL_REVISION", "rev-test")
    monkeypatch.setattr(conversion, "TOKENIZER_SHA256", "a" * 64)
    monkeypatch.setattr(conversion, "PARAMETER_COUNT", 10)
    monkeypatch.setattr(conversion, "VOCAB_SIZE", 4)
    monkeypatch.setattr(conversion, "HIDDEN_SIZE", 2)
    monkeypatch.setattr(conversion, "NUM_HIDDEN_LAYERS", 2)
    monkeypatch.setattr(conversion, "EOS_TOKEN_ID", 3)
    monkeypatch.setattr(
        conversion,
        "FIM_MARKER_IDS",
        {"fim_prefix": 0, "fim_middle": 1, "fim_suffix": 2},
    )
    monkeypatch.setattr(
        conversion,
        "FIM_MARKER_TEXT",
        {
            "fim_prefix": "<|test_prefix|>",
            "fim_middle": "<|test_middle|>",
            "fim_suffix": "<|test_suffix|>",
        },
    )
    monkeypatch.setattr(
        conversion,
        "QWEN_CONFIG_SEMANTICS",
        {
            "architectures": ["Qwen2ForCausalLM"],
            "bos_token_id": 3,
            "eos_token_id": 3,
            "hidden_act": "silu",
            "hidden_size": 2,
            "intermediate_size": 3,
            "max_position_embeddings": 16,
            "max_window_layers": 2,
            "model_type": "qwen2",
            "num_attention_heads": 1,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "use_sliding_window": False,
            "vocab_size": 4,
        },
    )


def _write_safetensors(path: Path, *, tied: bool = True) -> None:
    embedding = b"\x01\x00" * 8
    head = embedding if tied else b"\x02\x00" * 8
    tail = b"\x03\x00" * 2
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [4, 2],
            "data_offsets": [0, 16],
        },
        "lm_head.weight": {"dtype": "F16", "shape": [4, 2], "data_offsets": [16, 32]},
        "model.norm.weight": {"dtype": "F16", "shape": [2], "data_offsets": [32, 36]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    _write(path, struct.pack("<Q", len(raw)) + raw + embedding + head + tail)


def _tiny_bundle(root: Path, *, arm: str = "untouched_q25_to_fim") -> tuple[Path, Path, Path, str]:
    export = root / "selected" / "inference-f16"
    _write_safetensors(export / "model.safetensors")
    _write(
        export / "config.json",
        json.dumps(
            {
                **conversion.QWEN_CONFIG_SEMANTICS,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0},
                "sliding_window": None,
                "layer_types": ["full_attention"] * 2,
            }
        ).encode(),
    )
    _write(export / "tokenizer.json", b"local tokenizer bytes")
    _write(export / "tokenizer_config.json", b"local tokenizer config")
    files = {
        path.relative_to(export).as_posix(): {"bytes": path.stat().st_size, "sha256": _hash(path)}
        for path in sorted(export.rglob("*"))
        if path.is_file()
    }
    cursor = {
        "attempted_updates": 1,
        "completed_updates": 1,
        "skipped_updates": 0,
        "training_input_tokens": 8,
        "supervised_target_tokens": 3,
        "next_example_index": 2,
        "epoch": 1,
    }
    manifest = {
        "schema": conversion.EXPORT_SCHEMA,
        "fingerprint": "b" * 64,
        "arm": arm,
        "base_model": conversion.MODEL_ID,
        "base_revision": conversion.MODEL_REVISION,
        "base_snapshot_provenance": {"tokenizer_sha256": conversion.TOKENIZER_SHA256},
        "training_cursor": cursor,
        "files": files,
    }
    _write(export / "artifact_manifest.json", json.dumps(manifest, sort_keys=True).encode())

    initializer_files = {
        "config.json": {
            "bytes": (export / "config.json").stat().st_size,
            "sha256": _hash(export / "config.json"),
        },
        "model.safetensors": {
            "bytes": (export / "model.safetensors").stat().st_size,
            "sha256": _hash(export / "model.safetensors"),
        },
        "tokenizer.json": {"bytes": 23, "sha256": conversion.TOKENIZER_SHA256},
    }
    parent_cursor = {
        "attempted_updates": 1,
        "completed_updates": 1,
        "skipped_updates": 0,
        "training_input_tokens": 5,
        "supervised_target_tokens": 4,
        "next_example_index": 1,
        "epoch": 1,
    }
    initializer_map = {
        "untouched_q25_to_fim": {
            "kind": "untouched_pretrained",
            "model_id": conversion.MODEL_ID,
            "revision": conversion.MODEL_REVISION,
            "files": initializer_files,
        },
        "completed_cpt_q25_to_fim": {
            "kind": "completed_cpt_export",
            "model_id": conversion.MODEL_ID,
            "revision": conversion.MODEL_REVISION,
            "artifact_manifest_sha256": "c" * 64,
            "fingerprint": "d" * 64,
            "training_cursor": parent_cursor,
            "expected_complete_updates": 1,
            "expected_training_input_tokens": 5,
            "files": initializer_files,
        },
    }
    training_plan = {
        "schema": conversion.PLAN_SCHEMA,
        "gpu_execution_authorized": True,
        "configuration": {
            "training": {
                "attention": "sdpa",
                "checkpoint_every_updates": 64,
                "compute": "fp16",
                "cosine_floor_fraction": 0.1,
                "epochs": 1,
                "gradient_checkpointing": True,
                "gradient_clip": 1.0,
                "sequence_length": 1024,
                "effective_batch": 16,
                "initial_loss_scale": 128,
                "learning_rate": 1e-5,
                "master_weights": "fp32",
                "max_input_tokens": 4 * 1024**2,
                "max_output_bytes": 10 * 1024**3,
                "microbatch_examples": 1,
                "objective": "example_mean_response_only_FIM_target_and_EOS",
                "optimizer": "AdamW8bit",
                "same_examples_and_order": True,
                "seed": 314159,
                "warmup_fraction": 0.03,
                "weight_decay": 0.01,
                "arms": ["untouched_q25_to_fim", "completed_cpt_q25_to_fim"],
            }
        },
        "data": {"train": {"row_count": 2, "input_tokens": 8, "target_tokens": 3}},
        "initializers": initializer_map,
    }
    manifest["base_snapshot_provenance"] = {
        "config_sha256": _hash(export / "config.json"),
        "model_weights_sha256": _hash(export / "model.safetensors"),
        "tokenizer_sha256": conversion.TOKENIZER_SHA256,
    }
    selected_initializer = cast(dict[str, Any], initializer_map[arm])
    identity_fields = {
        "kind",
        "model_id",
        "revision",
        "files",
        "artifact_manifest_sha256",
        "fingerprint",
        "training_cursor",
    }
    manifest["initializer"] = {
        name: value for name, value in selected_initializer.items() if name in identity_fields
    }
    _write(export / "artifact_manifest.json", json.dumps(manifest, sort_keys=True).encode())

    plan_path = root / "training_plan.json"
    _write(plan_path, json.dumps(training_plan, sort_keys=True).encode())
    selection = {
        "schema": conversion.SELECTION_SCHEMA,
        "status": "selected_complete",
        "selected_arm": arm,
        "training_plan_sha256": _hash(plan_path),
        "tokenizer": {
            "model_id": conversion.MODEL_ID,
            "revision": conversion.MODEL_REVISION,
            "sha256": conversion.TOKENIZER_SHA256,
            "eos_token_id": conversion.EOS_TOKEN_ID,
            "fim_marker_ids": conversion.FIM_MARKER_IDS,
        },
        "model": {
            "architecture": "Qwen2ForCausalLM",
            "model_type": "qwen2",
            "logical_parameter_count": conversion.PARAMETER_COUNT,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0},
        },
        "source_export": {
            "directory": "selected/inference-f16",
            "artifact_manifest_sha256": _hash(export / "artifact_manifest.json"),
            "fingerprint": manifest["fingerprint"],
            "training_cursor": cursor,
            "files": files,
        },
    }
    selection_path = root / "selection.json"
    _write(selection_path, json.dumps(selection, sort_keys=True).encode())
    return selection_path, plan_path, export, _hash(selection_path)


def _move_export_to_selected_kernel_root(
    root: Path, selection_path: Path, export: Path
) -> tuple[Path, Path]:
    source_root = root / "selected-kernel"
    source_export = source_root / "q25_fim_r2" / "training" / "inference-f16"
    source_export.parent.mkdir(parents=True)
    shutil.move(str(export), str(source_export))
    selection = json.loads(selection_path.read_text())
    selection["source_export"]["directory"] = "q25_fim_r2/training/inference-f16"
    _write(selection_path, json.dumps(selection, sort_keys=True).encode())
    return source_root, source_export


def test_complete_selection_binds_plan_export_cursor_config_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, selection_sha = _tiny_bundle(tmp_path)

    verified = conversion.verify_selection_bundle(
        input_root=tmp_path,
        selection_path=selection_path,
        training_plan_path=plan_path,
        expected_selection_sha256=selection_sha,
    )

    assert verified.source_export == export
    assert verified.expected_cursor["completed_updates"] == 1
    assert verified.safetensors["logical_parameter_count"] == 10


def test_selection_can_bind_kernel_export_from_a_separate_mounted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, _ = _tiny_bundle(tmp_path)
    source_root, source_export = _move_export_to_selected_kernel_root(
        tmp_path, selection_path, export
    )

    verified = conversion.verify_selection_bundle(
        input_root=tmp_path,
        selection_path=selection_path,
        training_plan_path=plan_path,
        expected_selection_sha256=_hash(selection_path),
        source_root=source_root,
    )

    assert verified.source_export == source_export


def test_selection_rejects_source_roots_outside_or_linked_into_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, _ = _tiny_bundle(tmp_path)
    source_root, _ = _move_export_to_selected_kernel_root(tmp_path, selection_path, export)
    external_root = tmp_path.parent / f"{tmp_path.name}-external"
    external_root.mkdir()

    with pytest.raises(conversion.ConversionContractError, match="inside the mounted input root"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
            source_root=external_root,
        )

    linked_root = tmp_path / "kernel-source-link"
    linked_root.symlink_to(source_root, target_is_directory=True)
    with pytest.raises(conversion.ConversionContractError, match="symbolic links"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
            source_root=linked_root,
        )

    linked_parent = tmp_path / "mounted-parent-link"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(conversion.ConversionContractError, match="symbolic links"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
            source_root=linked_parent / "selected-kernel",
        )


def test_selection_rejects_symlinked_export_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, _ = _tiny_bundle(tmp_path)
    source_root, source_export = _move_export_to_selected_kernel_root(
        tmp_path, selection_path, export
    )
    real_export = source_root / "actual-export"
    source_export.rename(real_export)
    source_export.symlink_to(real_export, target_is_directory=True)

    with pytest.raises(conversion.ConversionContractError, match="symbolic links"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
            source_root=source_root,
        )


@pytest.mark.parametrize(
    "identity",
    [
        {
            "kind": "untouched_pretrained",
            "model_id": "Qwen/test-coder",
            "revision": "rev-test",
            "files": {"tokenizer.json": {"bytes": 10, "sha256": "a" * 64}},
        },
        {
            "kind": "completed_cpt_export",
            "model_id": "Qwen/test-coder",
            "revision": "rev-test",
            "files": {"tokenizer.json": {"bytes": 10, "sha256": "a" * 64}},
            "artifact_manifest_sha256": "b" * 64,
            "fingerprint": "c" * 64,
            "training_cursor": {"completed_updates": 481},
        },
    ],
    ids=["untouched", "completed-cpt"],
)
def test_initializer_identity_requires_all_parent_fields_and_values(
    identity: dict[str, object],
) -> None:
    expected = {**identity}
    if identity["kind"] == "completed_cpt_export":
        expected["expected_complete_updates"] = 481
        expected["expected_training_input_tokens"] = 7_872_512

    assert conversion._initializer_identity_matches(identity, expected)
    assert not conversion._initializer_identity_matches({}, expected)

    missing_files = {key: value for key, value in identity.items() if key != "files"}
    assert not conversion._initializer_identity_matches(missing_files, expected)

    extra_field = {**identity, "unbound": "value"}
    assert not conversion._initializer_identity_matches(extra_field, expected)

    mismatched_value = {**identity, "model_id": "Qwen/other-model"}
    assert not conversion._initializer_identity_matches(mismatched_value, expected)


def test_completed_cpt_initializer_is_bound_to_frozen_plan_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, selection_sha = _tiny_bundle(
        tmp_path, arm="completed_cpt_q25_to_fim"
    )

    verified = conversion.verify_selection_bundle(
        input_root=tmp_path,
        selection_path=selection_path,
        training_plan_path=plan_path,
        expected_selection_sha256=selection_sha,
    )
    assert verified.selection["selected_arm"] == "completed_cpt_q25_to_fim"

    manifest_path = export / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["initializer"]["fingerprint"] = "e" * 64
    _write(manifest_path, json.dumps(manifest, sort_keys=True).encode())
    selection = json.loads(selection_path.read_text())
    selection["source_export"]["artifact_manifest_sha256"] = _hash(manifest_path)
    _write(selection_path, json.dumps(selection, sort_keys=True).encode())
    with pytest.raises(conversion.ConversionContractError, match="partial or mismatched"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
        )


def test_conversion_requires_the_frozen_training_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, _, _ = _tiny_bundle(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["configuration"]["training"]["learning_rate"] = 2e-5
    _write(plan_path, json.dumps(plan, sort_keys=True).encode())
    selection = json.loads(selection_path.read_text())
    selection["training_plan_sha256"] = _hash(plan_path)
    _write(selection_path, json.dumps(selection, sort_keys=True).encode())

    with pytest.raises(conversion.ConversionContractError, match="frozen one-pass FIM schedule"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
        )


def test_partial_cursor_is_not_a_conversion_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, _ = _tiny_bundle(tmp_path)
    manifest_path = export / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    partial = dict(manifest["training_cursor"], completed_updates=0, skipped_updates=1)
    manifest["training_cursor"] = partial
    _write(manifest_path, json.dumps(manifest, sort_keys=True).encode())
    selection = json.loads(selection_path.read_text())
    selection["source_export"]["artifact_manifest_sha256"] = _hash(manifest_path)
    selection["source_export"]["training_cursor"] = partial
    _write(selection_path, json.dumps(selection, sort_keys=True).encode())

    with pytest.raises(conversion.ConversionContractError, match="partial or mismatched"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
        )


def test_bundle_rejects_changed_plan_or_export_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, selection_sha = _tiny_bundle(tmp_path)
    _write(export / "tokenizer.json", b"changed")
    with pytest.raises(conversion.ConversionContractError, match="file size|file hash"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=selection_sha,
        )

    selection_path, plan_path, _, selection_sha = _tiny_bundle(tmp_path / "second")
    _write(plan_path, b"{}")
    with pytest.raises(conversion.ConversionContractError, match="training plan"):
        conversion.verify_selection_bundle(
            input_root=tmp_path / "second",
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=selection_sha,
        )


def test_bundle_rejects_rope_semantic_drift_and_unmanifested_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    selection_path, plan_path, export, _ = _tiny_bundle(tmp_path)
    config_path = export / "config.json"
    config = json.loads(config_path.read_text())
    config["rope_parameters"]["rope_theta"] = 10_000.0
    _write(config_path, json.dumps(config).encode())
    manifest_path = export / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["config.json"] = {
        "bytes": config_path.stat().st_size,
        "sha256": _hash(config_path),
    }
    _write(manifest_path, json.dumps(manifest, sort_keys=True).encode())
    selection_document = json.loads(selection_path.read_text())
    selection_document["source_export"]["artifact_manifest_sha256"] = _hash(manifest_path)
    selection_document["source_export"]["files"] = manifest["files"]
    _write(selection_path, json.dumps(selection_document, sort_keys=True).encode())
    with pytest.raises(conversion.ConversionContractError, match="RoPE"):
        conversion.verify_selection_bundle(
            input_root=tmp_path,
            selection_path=selection_path,
            training_plan_path=plan_path,
            expected_selection_sha256=_hash(selection_path),
        )

    selection, plan, export, _ = _tiny_bundle(tmp_path / "reset")
    (export / "unbound.txt").write_text("extra")
    with pytest.raises(conversion.ConversionContractError, match="unmanifested"):
        conversion.verify_selection_bundle(
            input_root=tmp_path / "reset",
            selection_path=selection,
            training_plan_path=plan,
            expected_selection_sha256=_hash(selection),
        )


def test_safetensors_requires_tied_embedding_bytes_to_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, tied=False)

    with pytest.raises(conversion.ConversionContractError, match="bytes differ"):
        conversion._safetensors_layout(path)


class _Added:
    def __init__(self, content: str, special: bool = False):
        self.content = content
        self.special = special


class _Tokenizer:
    eos_token_id = 3
    added_tokens_decoder = {
        0: _Added("<|test_prefix|>", special=False),
        1: _Added("<|test_middle|>", special=False),
        2: _Added("<|test_suffix|>", special=False),
    }

    class Backend:
        def __init__(self, value: dict[str, object]) -> None:
            self.value = value

        def to_str(self) -> str:
            return json.dumps(self.value, ensure_ascii=False)

    def __init__(self, backend_value: dict[str, object] | None = None) -> None:
        self.backend_tokenizer = self.Backend(backend_value or _tiny_tokenizer_json())

    def convert_tokens_to_ids(self, text: str) -> int:
        return {
            "<|test_prefix|>": 0,
            "<|test_middle|>": 1,
            "<|test_suffix|>": 2,
        }[text]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.convert_tokens_to_ids(text)]


def _tiny_tokenizer_json() -> dict[str, object]:
    token_by_id = [
        conversion.FIM_MARKER_TEXT["fim_prefix"],
        conversion.FIM_MARKER_TEXT["fim_middle"],
        conversion.FIM_MARKER_TEXT["fim_suffix"],
        "<|test_eos|>",
        "a",
        "b",
        "ab",
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": token_id == conversion.EOS_TOKEN_ID,
            }
            for token_id, token in enumerate(token_by_id[:4])
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "vocab": {token: token_id for token_id, token in enumerate(token_by_id)},
            "merges": ["a b"],
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
        },
    }


def _write_tiny_tokenizer_json(path: Path, value: dict[str, object]) -> None:
    _write(path, json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def test_tokenizer_semantics_require_fim_ids_and_frozen_added_token_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tiny_constants(monkeypatch)
    tokenizer_json = tmp_path / "tokenizer.json"
    _write_tiny_tokenizer_json(tokenizer_json, _tiny_tokenizer_json())
    monkeypatch.setattr(
        conversion,
        "TOKENIZER_SEMANTICS_SHA256",
        conversion._tokenizer_json_semantics_sha256(tokenizer_json),
    )
    identity = {"sha256": conversion.TOKENIZER_SHA256}
    report = conversion.validate_tokenizer_semantics(
        _Tokenizer(), identity, tokenizer_json_path=tokenizer_json
    )
    assert report["fim_middle"] == 1
    assert report["tokenizer_json_semantics_sha256"] == conversion.TOKENIZER_SEMANTICS_SHA256

    monkeypatch.setitem(_Tokenizer.added_tokens_decoder, 1, _Added("<|test_middle|>", special=True))
    with pytest.raises(conversion.ConversionContractError, match="frozen FIM added-token metadata"):
        conversion.validate_tokenizer_semantics(
            _Tokenizer(), identity, tokenizer_json_path=tokenizer_json
        )


def test_tokenizer_semantics_accept_only_merge_serialization_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    tokenizer_json = tmp_path / "tokenizer.json"
    normalized_backend = _tiny_tokenizer_json()
    normalized_backend["pre_tokenizer"]["trim_offsets"] = True  # type: ignore[index]
    normalized_backend["decoder"] = {  # type: ignore[assignment]
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": True,
        "use_regex": True,
    }
    _write_tiny_tokenizer_json(tokenizer_json, normalized_backend)
    expected_semantics = conversion._tokenizer_json_semantics_sha256(tokenizer_json)
    monkeypatch.setattr(conversion, "TOKENIZER_SEMANTICS_SHA256", expected_semantics)

    saved_variant = json.loads(json.dumps(normalized_backend))
    saved_variant["model"]["merges"] = [["a", "b"]]  # type: ignore[index]
    saved_variant["model"]["ignore_merges"] = False  # type: ignore[index]
    _write_tiny_tokenizer_json(tokenizer_json, saved_variant)
    report = conversion.validate_tokenizer_semantics(
        _Tokenizer(saved_variant),
        {"sha256": conversion.TOKENIZER_SHA256},
        tokenizer_json_path=tokenizer_json,
    )
    assert report["tokenizer_json_semantics_sha256"] == expected_semantics

    raw_original = _tiny_tokenizer_json()
    _write_tiny_tokenizer_json(tokenizer_json, raw_original)
    with pytest.raises(conversion.ConversionContractError, match="mapping differs"):
        conversion.validate_tokenizer_semantics(
            _Tokenizer(raw_original),
            {"sha256": conversion.TOKENIZER_SHA256},
            tokenizer_json_path=tokenizer_json,
        )


def test_tokenizer_backend_must_match_frozen_saved_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_constants(monkeypatch)
    tokenizer_json = tmp_path / "tokenizer.json"
    saved = _tiny_tokenizer_json()
    _write_tiny_tokenizer_json(tokenizer_json, saved)
    monkeypatch.setattr(
        conversion,
        "TOKENIZER_SEMANTICS_SHA256",
        conversion._tokenizer_json_semantics_sha256(tokenizer_json),
    )
    changed_backend = json.loads(json.dumps(saved))
    changed_backend["decoder"]["use_regex"] = True  # type: ignore[index]
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        tokenizer.backend_tokenizer,
        "to_str",
        lambda: json.dumps(changed_backend, ensure_ascii=False),
    )

    with pytest.raises(conversion.ConversionContractError, match="backend differs"):
        conversion.validate_tokenizer_semantics(
            tokenizer,
            {"sha256": conversion.TOKENIZER_SHA256},
            tokenizer_json_path=tokenizer_json,
        )


@pytest.mark.parametrize(
    "change",
    [
        "vocab",
        "added_token",
        "normalizer",
        "pre_tokenizer_regex",
        "pre_tokenizer_offsets",
        "decoder_prefix",
        "decoder_regex",
        "decoder_offsets",
        "merge_order",
    ],
)
def test_tokenizer_semantics_reject_mapping_and_pipeline_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _tiny_constants(monkeypatch)
    tokenizer_json = tmp_path / "tokenizer.json"
    baseline = _tiny_tokenizer_json()
    _write_tiny_tokenizer_json(tokenizer_json, baseline)
    monkeypatch.setattr(
        conversion,
        "TOKENIZER_SEMANTICS_SHA256",
        conversion._tokenizer_json_semantics_sha256(tokenizer_json),
    )
    changed = json.loads(json.dumps(baseline))
    if change == "vocab":
        changed["model"]["vocab"]["a"] = 7  # type: ignore[index]
    elif change == "added_token":
        changed["added_tokens"][1]["special"] = True  # type: ignore[index]
    elif change == "normalizer":
        changed["normalizer"] = {"type": "NFD"}
    elif change == "pre_tokenizer_regex":
        changed["pre_tokenizer"]["use_regex"] = True  # type: ignore[index]
    elif change == "pre_tokenizer_offsets":
        changed["pre_tokenizer"]["trim_offsets"] = True  # type: ignore[index]
    elif change == "decoder_prefix":
        changed["decoder"]["add_prefix_space"] = True  # type: ignore[index]
    elif change == "decoder_regex":
        changed["decoder"]["use_regex"] = True  # type: ignore[index]
    elif change == "decoder_offsets":
        changed["decoder"]["trim_offsets"] = True  # type: ignore[index]
    else:
        changed["model"]["merges"] = ["b a"]  # type: ignore[index]
    _write_tiny_tokenizer_json(tokenizer_json, changed)

    with pytest.raises(conversion.ConversionContractError, match="mapping differs"):
        conversion.validate_tokenizer_semantics(
            _Tokenizer(),
            {"sha256": conversion.TOKENIZER_SHA256},
            tokenizer_json_path=tokenizer_json,
        )


def _q4_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    tokens = [""] * conversion.VOCAB_SIZE
    token_types = [1] * conversion.VOCAB_SIZE
    tokens[conversion.EOS_TOKEN_ID] = "<|endoftext|>"
    for name, token_id in conversion.FIM_MARKER_IDS.items():
        tokens[token_id] = conversion.FIM_MARKER_TEXT[name]
        token_types[token_id] = 3
    metadata: dict[str, object] = {
        "general.architecture": "qwen2",
        "qwen2.context_length": 32768,
        "qwen2.embedding_length": conversion.HIDDEN_SIZE,
        "qwen2.feed_forward_length": 4864,
        "qwen2.block_count": 24,
        "qwen2.attention.head_count": 14,
        "qwen2.attention.head_count_kv": 2,
        "qwen2.rope.freq_base": 1_000_000.0,
        "tokenizer.ggml.eos_token_id": conversion.EOS_TOKEN_ID,
        "general.file_type": 15,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": token_types,
    }
    tied = conversion.VOCAB_SIZE * conversion.HIDDEN_SIZE
    other = conversion.PARAMETER_COUNT - tied
    tensors: list[dict[str, object]] = [
        {
            "name": "token_embd.weight",
            "elements": tied,
            "shape": [conversion.HIDDEN_SIZE, conversion.VOCAB_SIZE],
        },
        {
            "name": "output.weight",
            "elements": tied,
            "shape": [conversion.HIDDEN_SIZE, conversion.VOCAB_SIZE],
        },
        {"name": "other", "elements": other, "shape": [other]},
    ]
    return metadata, tensors


def test_q4_validation_checks_rope_precision_parameters_and_fim_controls() -> None:
    metadata, tensors = _q4_fixture()
    report = conversion.validate_q4_gguf(metadata, tensors)
    assert report["file_type"] == "Q4_K_M"
    assert report["logical_parameter_count"] == conversion.PARAMETER_COUNT
    assert report["physical_parameter_count"] == (
        conversion.PARAMETER_COUNT + conversion.VOCAB_SIZE * conversion.HIDDEN_SIZE
    )

    metadata["qwen2.rope.freq_base"] = 10_000.0
    with pytest.raises(conversion.ConversionContractError, match="metadata"):
        conversion.validate_q4_gguf(metadata, tensors)

    metadata, tensors = _q4_fixture()
    metadata["tokenizer.ggml.tokens"][conversion.FIM_MARKER_IDS["fim_prefix"]] = "<bad>"  # type: ignore[index]
    with pytest.raises(conversion.ConversionContractError, match="FIM control"):
        conversion.validate_q4_gguf(metadata, tensors)


def test_storage_preflight_counts_input_runtime_and_intermediate_peak(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    runtime_root = tmp_path / "runtime"
    output_root = tmp_path / "output"
    _write(input_root / "weights.bin", b"12345")
    _write(runtime_root / "quantizer", b"123")
    _write(output_root / "manifest.json", b"12")

    report = conversion.preflight_storage(
        roots=[input_root, runtime_root, output_root],
        additional_peak_bytes=10,
        free_bytes=13,
        max_artifact_bytes=20,
        minimum_free_bytes=3,
    )
    assert report["current_accounted_bytes"] == 10
    assert report["projected_peak_bytes"] == 20

    with pytest.raises(conversion.ConversionContractError, match="free-space reserve"):
        conversion.preflight_storage(
            roots=[input_root],
            additional_peak_bytes=10,
            free_bytes=12,
            max_artifact_bytes=100,
            minimum_free_bytes=3,
        )
    with pytest.raises(conversion.ConversionContractError, match="artifact cap"):
        conversion.preflight_storage(
            roots=[input_root],
            additional_peak_bytes=10,
            free_bytes=100,
            max_artifact_bytes=14,
            minimum_free_bytes=0,
        )


def test_conversion_peak_estimate_keeps_f16_and_q4_until_verification() -> None:
    source_safetensors_bytes = 1_260_367_152
    physical = conversion.QWEN_GGUF_PHYSICAL_PARAMETER_COUNT
    f16_upper_bound = conversion.projected_f16_gguf_bytes(
        source_safetensors_bytes=source_safetensors_bytes,
        physical_parameter_count=physical,
    )
    q4_bound = conversion.projected_q4_output_bytes(f16_gguf_bytes=f16_upper_bound)
    assert q4_bound == f16_upper_bound + 128 * 1024**2
    assert q4_bound > 491_399_808

    estimate = conversion.projected_conversion_peak_bytes(
        source_safetensors_bytes=source_safetensors_bytes,
        physical_parameter_count=physical,
    )
    assert estimate == f16_upper_bound + q4_bound + 64 * 1024**2
    assert estimate < 2 * conversion.MAX_ARTIFACT_BYTES
