from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
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


def _native_fim_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
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
    model = tmp_path / "selected.gguf"
    model.write_bytes(b"synthetic test artifact; not model weights")
    runtime = tmp_path / "tabcomplete-q25-fim"
    runtime.write_bytes(b"synthetic executable; no model is loaded")
    runtime.chmod(0o700)
    conversion = tmp_path / "conversion.json"
    conversion.write_text(
        json.dumps(
            {
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
        )
    )
    profile_dir = tmp_path / "profile"
    installer.prepare_native_fim_profile(model, tokenizer, conversion, profile_dir)
    return {
        "model": model,
        "runtime": runtime,
        "editor_model": profile_dir / "editor-model.json",
        "full_profile": json.loads((profile_dir / "registry.json").read_text())["q25-fim"][
            "fim_profile"
        ],
        "spec": json.loads((profile_dir / "editor-model.json").read_text()),
    }


def _native_health(inputs: dict, runtime_hash: str = "d" * 64) -> dict:
    spec = inputs["spec"]
    tokenizer = spec["fim_profile"]["tokenizer"]
    return {
        "status": "ok",
        "alias": "q25-fim",
        "model_sha256": spec["model_sha256"],
        "model_protocol": "q25-fim-line-completion-v1",
        "model_embedded": True,
        "model_switch_supported": False,
        "model_storage": "executable-mmap",
        "backend": "llama.cpp CPU via Rust",
        "context_layout": "q25-fim-psm-bounded-v2",
        "context_size": 2304,
        "input_tokens": 1024,
        "output_tokens": 96,
        "threads": 4,
        "prompt_threads": 4,
        "batch_size": 256,
        "microbatch_size": 64,
        "cache_type": "f16",
        "syntax_validation": False,
        "saved_contexts": 0,
        "active_slots": 1,
        "completion_mode": "remaining_logical_line_after_utf8_cursor",
        "fim_profile": spec["fim_profile"],
        "tokenizer_id": tokenizer["tokenizer_id"],
        "tokenizer_revision": tokenizer["tokenizer_revision"],
        "tokenizer_sha256": tokenizer["tokenizer_sha256"],
        "tokenizer_contract_sha256": tokenizer["tokenizer_contract_sha256"],
        "tokenizer_vocab_size": tokenizer["tokenizer_vocab_size"],
        "tokenizer_vocab_ids_sha256": tokenizer["tokenizer_vocab_ids_sha256"],
        "runtime_config_hash": runtime_hash,
    }


def _native_fim_installer(config: Path, unit: Path, inputs: dict, *, dry_run: bool):
    return installer.install_native_fim(
        config,
        unit,
        inputs["model"],
        inputs["runtime"],
        inputs["editor_model"],
        installer.sha(inputs["model"]),
        installer.sha(inputs["runtime"]),
        inputs["model"].stat().st_size,
        dry_run=dry_run,
    )


def test_native_fim_dry_run_returns_concrete_plan_without_activation(tmp_path, monkeypatch):
    inputs = _native_fim_inputs(tmp_path, monkeypatch)
    config = tmp_path / "plugin.lua"
    config.write_text(ORIGINAL)
    unit = tmp_path / "tabcomplete-predictor.service"
    result = _native_fim_installer(config, unit, inputs, dry_run=True)
    assert result["dry_run"] and not result["activated"]
    assert result["model_sha256"] == installer.sha(inputs["model"])
    assert result["runtime_sha256"] == installer.sha(inputs["runtime"])
    assert result["editor_model_sha256"] == installer.sha(inputs["editor_model"])
    assert result["runtime_config_hash"] is None
    assert "verified /health" in result["runtime_config_hash_source"]
    assert result["memory_max_bytes"] == 1500 * 1024 * 1024
    assert "--host 127.0.0.1 --port 19093" in result["unit_content"]
    assert "--threads 4 --prompt-threads 4 --context-size 2304" in result["unit_content"]
    assert "--input-tokens 1024 --batch-size 256 --microbatch-size 64" in result["unit_content"]
    assert "--output-tokens 96 --cache-type f16 --syntax-validation false" in result["unit_content"]
    assert "MemoryMax=1500M" in result["unit_content"]
    assert "MemorySwapMax=0" in result["unit_content"]
    assert str(inputs["model"]) not in result["unit_content"]
    assert "--model " not in result["unit_content"]
    assert '"tokenizer_vocab_ids"=' not in result["candidate_setup"]
    assert 'backend = "rust-editor-v1"' in result["candidate_setup"]
    assert 'mode = "automatic"' in result["candidate_setup"]
    assert "automatic_quality_validated = false" in result["candidate_setup"]
    assert "automatic_personalization_enabled = false" in result["candidate_setup"]
    assert "tabcomplete-q25-fim-mode.json" in result["candidate_setup"]
    assert 'accept_key = "<M-l>"' in result["candidate_setup"]
    assert 'predict_key = "<M-p>"' in result["candidate_setup"]
    assert config.read_text() == ORIGINAL
    assert not unit.exists()


def test_native_fim_cli_dry_run_is_explicit_and_json(tmp_path, monkeypatch):
    from io import StringIO

    inputs = _native_fim_inputs(tmp_path, monkeypatch)
    config = tmp_path / "plugin.lua"
    config.write_text(ORIGINAL)
    unit = tmp_path / "tabcomplete-predictor.service"
    argv = [
        str(SCRIPT),
        "--install-native-fim",
        "--model",
        str(inputs["model"]),
        "--model-sha256",
        installer.sha(inputs["model"]),
        "--model-bytes",
        str(inputs["model"].stat().st_size),
        "--runtime",
        str(inputs["runtime"]),
        "--runtime-sha256",
        installer.sha(inputs["runtime"]),
        "--editor-model-spec",
        str(inputs["editor_model"]),
        "--config",
        str(config),
        "--unit",
        str(unit),
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdout", output_stream)
    installer.main()
    output = json.loads(output_stream.getvalue())
    assert output["dry_run"] is True
    assert output["model_sha256"] == installer.sha(inputs["model"])
    assert output["unit_sha256"] == hashlib.sha256(output["unit_content"].encode()).hexdigest()
    assert not unit.exists()


def test_native_fim_success_verifies_health_process_slots_and_commits_actual_runtime_hash(
    tmp_path, monkeypatch
):
    inputs = _native_fim_inputs(tmp_path, monkeypatch)
    config = tmp_path / "plugin.lua"
    config.write_text(ORIGINAL)
    unit = tmp_path / "tabcomplete-predictor.service"
    unit.write_text("old predictor unit\n")
    process_calls = []

    def systemctl(args, *, check=True):
        process_calls.append(args)
        if args[0] == "is-enabled":
            return subprocess.CompletedProcess(args, 0, "enabled\n", "")
        if args[0] == "is-active":
            return subprocess.CompletedProcess(args, 0, "active\n", "")
        if args[0] == "show":
            return subprocess.CompletedProcess(
                args,
                0,
                "MainPID=123\nMemoryCurrent=500000000\nMemoryMax=1572864000\n",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(installer, "_systemctl", systemctl)
    monkeypatch.setattr(
        installer,
        "_local_json",
        lambda path: {
            "/health": _native_health(inputs),
            "/slots": [{"id": 0, "is_processing": False}],
        }[path],
    )
    monkeypatch.setattr(installer, "_process_executable", lambda _pid: inputs["runtime"])
    result = _native_fim_installer(config, unit, inputs, dry_run=False)
    assert result["activated"] is True
    assert result["runtime_config_hash"] == "d" * 64
    assert result["runtime_config_hash"] != result["unit_sha256"]
    assert f'runtime_config_hash = "{"d" * 64}"' in config.read_text()
    assert 'model = "q25-fim"' in config.read_text()
    assert "automatic_quality_validated = false" in config.read_text()
    assert 'server_url = "http://127.0.0.1:8787"' in config.read_text()
    assert 'vim.keymap.set("i", "<Tab>", existing_completion)' in config.read_text()
    assert f"ExecStart={inputs['runtime']}" in unit.read_text()
    assert installer.sha(unit) == result["unit_sha256"]
    assert len([call for call in process_calls if call[0] == "restart"]) == 1
    assert result["config_backup"]
    assert result["unit_backup"]


def test_native_fim_refuses_nix_owned_configuration_and_unit(tmp_path, monkeypatch):
    inputs = _native_fim_inputs(tmp_path, monkeypatch)
    target = tmp_path / "nix-generated.lua"
    target.write_text(ORIGINAL)
    config = tmp_path / "plugin.lua"
    config.symlink_to(target)
    unit = tmp_path / "tabcomplete-predictor.service"
    with pytest.raises(RuntimeError, match="Nix/Home Manager owns"):
        _native_fim_installer(config, unit, inputs, dry_run=True)
    assert target.read_text() == ORIGINAL
    assert not unit.exists()


@pytest.mark.parametrize("failure", ["launch", "health", "process_hash", "memory"])
def test_native_fim_failure_restores_config_and_unit(tmp_path, monkeypatch, failure):
    inputs = _native_fim_inputs(tmp_path, monkeypatch)
    config = tmp_path / "plugin.lua"
    config.write_text(ORIGINAL)
    unit = tmp_path / "tabcomplete-predictor.service"
    unit.write_text("previous stable unit\n")
    original_systemctl = []
    restart_count = 0

    def systemctl(args, *, check=True):
        nonlocal restart_count
        original_systemctl.append(args)
        if args[0] == "is-enabled":
            return subprocess.CompletedProcess(args, 0, "enabled\n", "")
        if args[0] == "is-active":
            return subprocess.CompletedProcess(args, 0, "active\n", "")
        if args[0] == "show":
            memory = installer.FIM_MEMORY_MAX + 1 if failure == "memory" else 500000000
            return subprocess.CompletedProcess(
                args,
                0,
                f"MainPID=123\nMemoryCurrent={memory}\nMemoryMax={installer.FIM_MEMORY_MAX}\n",
                "",
            )
        if args[0] == "restart":
            restart_count += 1
            if failure == "launch" and restart_count == 1:
                raise RuntimeError("untrusted failure detail")
        return subprocess.CompletedProcess(args, 0, "", "")

    health = _native_health(inputs)
    if failure == "health":
        health["model_sha256"] = "0" * 64
    monkeypatch.setattr(installer, "_systemctl", systemctl)
    monkeypatch.setattr(
        installer,
        "_local_json",
        lambda path: {
            "/health": health,
            "/slots": [{"id": 0, "is_processing": False}],
        }[path],
    )
    wrong_runtime = inputs["runtime"].with_name("wrong-executable")
    wrong_runtime.write_bytes(b"wrong process identity")
    process_path = wrong_runtime if failure == "process_hash" else inputs["runtime"]
    monkeypatch.setattr(installer, "_process_executable", lambda _pid: process_path)
    with pytest.raises(RuntimeError, match="restored") as error:
        _native_fim_installer(config, unit, inputs, dry_run=False)
    assert "untrusted failure detail" not in str(error.value)
    assert config.read_text() == ORIGINAL
    assert unit.read_text() == "previous stable unit\n"
    assert restart_count >= 1
    assert [call for call in original_systemctl if call[0] == "restart"]


def test_native_fim_rejects_hash_mismatch_before_writes(tmp_path, monkeypatch):
    inputs = _native_fim_inputs(tmp_path, monkeypatch)
    config = tmp_path / "plugin.lua"
    config.write_text(ORIGINAL)
    unit = tmp_path / "tabcomplete-predictor.service"
    with pytest.raises(ValueError, match="identity mismatch"):
        installer.install_native_fim(
            config,
            unit,
            inputs["model"],
            inputs["runtime"],
            inputs["editor_model"],
            "0" * 64,
            installer.sha(inputs["runtime"]),
            inputs["model"].stat().st_size,
            dry_run=True,
        )
    assert config.read_text() == ORIGINAL
    assert not unit.exists()


def test_native_fim_config_loads_in_headless_neovim_and_isolates_persisted_mode(
    tmp_path, monkeypatch
):
    inputs = _native_fim_inputs(tmp_path, monkeypatch)
    config_text, _ = installer.render_native_fim_config(ORIGINAL, inputs["spec"], "d" * 64)
    config = tmp_path / "tabcomplete-trajectory.lua"
    config.write_text(config_text)
    script = tmp_path / "verify.lua"
    script.write_text(
        """
        local spec = dofile(vim.env.TC_CONFIG)
        assert(type(spec) == 'table' and type(spec[1]) == 'table')
        local plugin = spec[1]
        vim.opt.rtp:prepend(plugin.dir)
        local state_dir = vim.fn.stdpath('state')
        vim.fn.mkdir(state_dir, 'p')
        if vim.env.TC_PHASE == 'first' then
          local old = assert(io.open(state_dir .. '/tabcomplete-predictor-mode.json', 'w'))
          old:write('{\"mode\":\"off\",\"experimental_auto_opt_in\":true}')
          old:close()
        end
        _G.existing_completion = function() end
        vim.system = function(_, _, callback)
          if callback then callback({ code = 1, stdout = '', stderr = '' }) end
          return {
            kill = function() end,
            wait = function() return { code = 1, stdout = '', stderr = '' } end,
          }
        end
        plugin.config()
        local predict = require('tabcomplete_trajectory.predict')
        local status = predict.status()
        assert(status.backend == 'rust-editor-v1')
        assert(status.model == 'q25-fim')
        assert(status.protocol_version == 'q25-fim-line-completion-v1')
        assert(status.automatic_normal_mode == true)
        if vim.env.TC_PHASE == 'first' then
          assert(status.mode == 'automatic', 'legacy model mode masked the explicit FIM opt-in')
          predict.set_mode('off')
          local saved = assert(io.open(state_dir .. '/tabcomplete-q25-fim-mode.json', 'r'))
          local value = saved:read('*a')
          saved:close()
          assert(value:find('off', 1, true), 'FIM mode was not persisted separately')
        else
          assert(status.mode == 'off', 'explicit FIM off mode did not survive restart')
        end
        assert(vim.fn.maparg('<Tab>', 'i') ~= '', 'existing Tab completion mapping was lost')
        assert(vim.fn.maparg('<M-p>', 'i') ~= '', 'manual prediction mapping is missing')
        assert(vim.fn.maparg('<M-l>', 'i') ~= '', 'accept mapping is missing')
        vim.cmd('qa!')
        """
    )
    state_home = tmp_path / "state-home"
    env = {
        **os.environ,
        "XDG_STATE_HOME": str(state_home),
        "XDG_DATA_HOME": str(tmp_path / "data-home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config-home"),
        "TC_CONFIG": str(config),
    }
    for phase in ("first", "restart"):
        subprocess.run(
            ["nvim", "--headless", "-u", "NONE", "-l", str(script)],
            env={**env, "TC_PHASE": phase},
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
