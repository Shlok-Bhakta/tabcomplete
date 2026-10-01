from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install_small_model_lazyvim.py'
SPEC = importlib.util.spec_from_file_location('selected_model_installer_test', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)

ORIGINAL = '''return {{
    dir = "/existing/tools/trajectory_collector/nvim",
    config = function()
      require("tabcomplete_trajectory").setup({
        server_url = "http://127.0.0.1:8787",
      })
      vim.keymap.set("i", "<Tab>", existing_completion)
    end,
}}
'''


def test_installer_explicitly_selects_codec_and_preserves_completion_mapping() -> None:
    result = installer.render_config(ORIGINAL, model_revision='a' * 64,
                                     model_alias='q25-adapted', runtime_config_hash='b' * 64,
                                     experimental_automatic=True,
                                     protocol_version='single-line-edit-v1')
    assert 'protocol_version = "single-line-edit-v1"' in result
    assert 'mode = "automatic", experimental_auto_opt_in = true' in result
    assert 'automatic_quality_validated = false' in result
    assert 'automatic_personalization_enabled = false' in result
    assert 'vim.keymap.set("i", "<Tab>", existing_completion)' in result
    assert 'server_url = "http://127.0.0.1:8787"' in result
    assert installer.render_config(result, model_revision='a' * 64,
                                   model_alias='q25-adapted', runtime_config_hash='b' * 64,
                                   experimental_automatic=True,
                                   protocol_version='single-line-edit-v1') == result


def test_installer_rejects_unrecognized_codec_before_writing() -> None:
    with pytest.raises(ValueError, match='unsupported'):
        installer.render_config(ORIGINAL, model_revision='a' * 64, model_alias='q25-adapted',
                                runtime_config_hash='b' * 64, experimental_automatic=True,
                                protocol_version='unknown')


def test_installer_records_q5_identity_and_rejects_unknown_precision() -> None:
    result = installer.render_config(ORIGINAL, model_revision='a' * 64,
                                     model_alias='q25-adapted', runtime_config_hash='b' * 64,
                                     experimental_automatic=True,
                                     protocol_version='single-line-edit-v1', precision='Q5_K_M')
    assert 'precision = "Q5_K_M"' in result
    with pytest.raises(ValueError, match='precision'):
        installer.render_config(ORIGINAL, model_revision='a' * 64,
                                model_alias='q25-adapted', runtime_config_hash='b' * 64,
                                experimental_automatic=True, precision='unknown')


def test_installer_pins_actual_selected_size_for_conversion_dry_run(tmp_path: Path) -> None:
    config = tmp_path / 'plugin.lua'
    config.write_text(ORIGINAL)
    model = tmp_path / 'synthetic.gguf'
    model.write_bytes(b'synthetic installation test, no model')
    runtime = tmp_path / 'llama-server'
    runtime.write_bytes(b'synthetic executable identity')
    result = installer.install(config, tmp_path / 'service', model, runtime, installer.sha(model),
                               'synthetic', dry_run=True, experimental_automatic=True,
                               expected_bytes=model.stat().st_size,
                               protocol_version='single-line-edit-v1')
    assert result['model_bytes'] == model.stat().st_size
    assert result['protocol_version'] == 'single-line-edit-v1'
    assert config.read_text() == ORIGINAL
    assert not (tmp_path / 'service').exists()
    with pytest.raises(ValueError, match='size mismatch'):
        installer.install(config, tmp_path / 'service', model, runtime, installer.sha(model),
                          'synthetic', dry_run=True)


def test_failed_service_restart_restores_config_and_restarts_previous_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from types import SimpleNamespace

    config = tmp_path / 'plugin.lua'
    config.write_text(ORIGINAL)
    unit = tmp_path / 'tabcomplete-predictor.service'
    unit.write_text('previous stable unit\n')
    model = tmp_path / 'synthetic.gguf'
    model.write_bytes(b'synthetic only')
    runtime = tmp_path / 'llama-server'
    runtime.write_bytes(b'synthetic executable')
    calls = []

    def invoke(command, **_kwargs):
        calls.append(command)
        if command[2] == 'restart' and len([c for c in calls if c[2] == 'restart']) == 1:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(installer.subprocess, 'run', invoke)
    with pytest.raises(RuntimeError, match='restored configuration'):
        installer.install(config, unit, model, runtime, installer.sha(model), 'synthetic',
                          dry_run=False, expected_bytes=model.stat().st_size,
                          protocol_version='single-line-edit-v1', experimental_automatic=True)
    assert config.read_text() == ORIGINAL
    assert unit.read_text() == 'previous stable unit\n'
    assert len([c for c in calls if c[2] == 'restart']) == 2
