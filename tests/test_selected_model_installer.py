from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/install_small_model_lazyvim.py"
SPEC = importlib.util.spec_from_file_location("selected_model_installer_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)

ORIGINAL = """return {{
    dir = "/existing/tools/trajectory_collector/nvim",
    config = function()
      require("tabcomplete_trajectory").setup({
        server_url = "http://127.0.0.1:8787",
      })
      vim.keymap.set("i", "<Tab>", existing_completion)
    end,
}}
"""


def test_installer_explicitly_selects_codec_and_preserves_completion_mapping() -> None:
    result = installer.render_config(
        ORIGINAL,
        model_revision="a" * 64,
        model_alias="q25-adapted",
        runtime_config_hash="b" * 64,
        experimental_automatic=True,
        protocol_version="single-line-edit-v1",
    )
    assert 'protocol_version = "single-line-edit-v1"' in result
    assert 'mode = "automatic", experimental_auto_opt_in = true' in result
    assert "automatic_quality_validated = false" in result
    assert "automatic_personalization_enabled = false" in result
    assert 'vim.keymap.set("i", "<Tab>", existing_completion)' in result
    assert 'server_url = "http://127.0.0.1:8787"' in result
    assert (
        installer.render_config(
            result,
            model_revision="a" * 64,
            model_alias="q25-adapted",
            runtime_config_hash="b" * 64,
            experimental_automatic=True,
            protocol_version="single-line-edit-v1",
        )
        == result
    )


def test_installer_rejects_unrecognized_codec_before_writing() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        installer.render_config(
            ORIGINAL,
            model_revision="a" * 64,
            model_alias="q25-adapted",
            runtime_config_hash="b" * 64,
            experimental_automatic=True,
            protocol_version="unknown",
        )


def test_installer_records_q5_identity_and_rejects_unknown_precision() -> None:
    result = installer.render_config(
        ORIGINAL,
        model_revision="a" * 64,
        model_alias="q25-adapted",
        runtime_config_hash="b" * 64,
        experimental_automatic=True,
        protocol_version="single-line-edit-v1",
        precision="Q5_K_M",
    )
    assert 'precision = "Q5_K_M"' in result
    with pytest.raises(ValueError, match="precision"):
        installer.render_config(
            ORIGINAL,
            model_revision="a" * 64,
            model_alias="q25-adapted",
            runtime_config_hash="b" * 64,
            experimental_automatic=True,
            precision="unknown",
        )


def test_installer_pins_actual_selected_size_for_conversion_dry_run(tmp_path: Path) -> None:
    config = tmp_path / "plugin.lua"
    config.write_text(ORIGINAL)
    model = tmp_path / "synthetic.gguf"
    model.write_bytes(b"synthetic installation test, no model")
    runtime = tmp_path / "llama-server"
    runtime.write_bytes(b"synthetic executable identity")
    result = installer.install(
        config,
        tmp_path / "service",
        model,
        runtime,
        installer.sha(model),
        "synthetic",
        dry_run=True,
        experimental_automatic=True,
        expected_bytes=model.stat().st_size,
        protocol_version="single-line-edit-v1",
    )
    assert result["model_bytes"] == model.stat().st_size
    assert result["protocol_version"] == "single-line-edit-v1"
    assert config.read_text() == ORIGINAL
    assert not (tmp_path / "service").exists()
    with pytest.raises(ValueError, match="size mismatch"):
        installer.install(
            config,
            tmp_path / "service",
            model,
            runtime,
            installer.sha(model),
            "synthetic",
            dry_run=True,
        )


def test_failed_service_restart_restores_config_and_restarts_previous_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from types import SimpleNamespace

    config = tmp_path / "plugin.lua"
    config.write_text(ORIGINAL)
    unit = tmp_path / "tabcomplete-predictor.service"
    unit.write_text("previous stable unit\n")
    model = tmp_path / "synthetic.gguf"
    model.write_bytes(b"synthetic only")
    runtime = tmp_path / "llama-server"
    runtime.write_bytes(b"synthetic executable")
    calls = []

    def invoke(command, **_kwargs):
        calls.append(command)
        if command[2] == "restart" and len([c for c in calls if c[2] == "restart"]) == 1:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(installer.subprocess, "run", invoke)
    with pytest.raises(RuntimeError, match="restored configuration"):
        installer.install(
            config,
            unit,
            model,
            runtime,
            installer.sha(model),
            "synthetic",
            dry_run=False,
            expected_bytes=model.stat().st_size,
            protocol_version="single-line-edit-v1",
            experimental_automatic=True,
        )
    assert config.read_text() == ORIGINAL
    assert unit.read_text() == "previous stable unit\n"
    assert len([c for c in calls if c[2] == "restart"]) == 2


def test_native_fim_profile_binds_artifact_and_all_added_controls(tmp_path, monkeypatch):
    import json

    from tinycomplete.code_cpt import q25_fim_conversion as contract

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(
        json.dumps(
            {
                "model": {"vocab": {"code": 17}},
                "added_tokens": [
                    {"id": 151643, "content": "<|endoftext|>", "special": True},
                    {"id": 151659, "content": "<|fim_prefix|>", "special": False},
                    {"id": 151660, "content": "<|fim_middle|>", "special": False},
                    {"id": 151661, "content": "<|fim_suffix|>", "special": False},
                ],
            }
        )
    )
    monkeypatch.setattr(contract, "TOKENIZER_SHA256", installer.sha(tokenizer))
    model = tmp_path / "synthetic.gguf"
    model.write_bytes(b"synthetic profile fixture, not model weights")
    conversion = tmp_path / "conversion.json"
    value = {
        "schema": "q25-fim-q4-conversion-run-v1",
        "status": "complete",
        "source_export_manifest_sha256": "a" * 64,
        "conversion": {"format": "Q4_K_M", "gpu_enabled": False},
        "tokenizer": {
            "model_id": contract.MODEL_ID,
            "revision": contract.MODEL_REVISION,
            "sha256": installer.sha(tokenizer),
            "eos_token_id": contract.EOS_TOKEN_ID,
            "fim_marker_ids": contract.FIM_MARKER_IDS,
        },
        "q4_export": {"bytes": model.stat().st_size, "sha256": installer.sha(model)},
    }
    conversion.write_text(json.dumps(value))
    output = tmp_path / "profile"
    result = installer.prepare_native_fim_profile(model, tokenizer, conversion, output)
    assert result["activated"] is False
    assert result["automatic_personalization_enabled"] is False
    registry = json.loads((output / "registry.json").read_text())["q25-fim"]
    profile = registry["fim_profile"]
    assert profile["artifact_manifest_sha256"] == value["source_export_manifest_sha256"]
    assert profile["tokenizer"]["tokenizer_vocab_ids"] == [17, 151643, 151659, 151660, 151661]
    assert len(profile["tokenizer"]["special_tokens"]) == 4
    editor = json.loads((output / "editor-model.json").read_text())
    assert "tokenizer_vocab_ids" not in editor["fim_profile"]["tokenizer"]
    embedded = json.loads((output / "embedded-profile.json").read_text())
    assert embedded == {
        "schema": "tabcomplete-q25-fim-embedded-profile-v1",
        "model_sha256": installer.sha(model),
        "serving_profile": profile,
    }
    assert result["embedded_profile"] == str(output / "embedded-profile.json")
    assert installer.prepare_native_fim_profile(model, tokenizer, conversion, output) == result
    (output / "registry.json").write_text("different identity")
    with pytest.raises(ValueError, match="another identity"):
        installer.prepare_native_fim_profile(model, tokenizer, conversion, output)
    value["q4_export"]["sha256"] = "0" * 64
    conversion.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="verified selected"):
        installer.prepare_native_fim_profile(model, tokenizer, conversion, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
