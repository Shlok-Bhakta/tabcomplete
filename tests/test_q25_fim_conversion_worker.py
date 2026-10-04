from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKER_PATH = Path(__file__).parents[1] / "kaggle" / "q25_fim_conversion_r1" / "run.py"
SPEC = importlib.util.spec_from_file_location("q25_fim_conversion_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def test_cpu_runtime_probe_is_valid_python_without_importing_torch() -> None:
    compile(WORKER._cpu_inventory_script(), "cpu_runtime_probe.py", "exec")


def test_preflight_failure_records_stage_without_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WORKER, "PREFLIGHT_STAGE", "selected_export_identity")
    output = tmp_path / "failure.json"
    WORKER._record_preflight_failure(output, ValueError("secret-value-must-not-escape"))
    record = json.loads(output.read_text())
    assert record["stage"] == "selected_export_identity"
    assert record["exception_class"] == "ValueError"
    assert record["exception_text_recorded"] is False
    assert record["training_input_tokens"] == 0
    assert record["gpu_enabled"] is False
    assert "secret-value" not in output.read_text()


def test_preflight_failure_does_not_follow_output_symlink(tmp_path: Path) -> None:
    target = tmp_path / "preserved"
    target.write_text("original")
    output = tmp_path / "failure.json"
    output.symlink_to(target)
    WORKER._record_preflight_failure(output, RuntimeError("untrusted"))
    assert target.read_text() == "original"


def test_source_kernel_reference_is_a_safe_owner_and_slug_pair() -> None:
    assert WORKER._valid_source_kernel_reference("shlokbhakta/q25-fim-selected")
    assert not WORKER._valid_source_kernel_reference("shlokbhakta")
    assert not WORKER._valid_source_kernel_reference("owner/kernel/extra")
    assert not WORKER._valid_source_kernel_reference("owner/..")
    assert not WORKER._valid_source_kernel_reference("owner/private token")


def test_cli_requires_and_preserves_selected_source_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WORKER.sys,
        "argv",
        [
            "run.py",
            "--input-root",
            "/kaggle/input",
            "--selection",
            "/kaggle/input/config/selection.json",
            "--training-plan",
            "/kaggle/input/config/training-plan.json",
            "--expected-selection-sha256",
            "a" * 64,
            "--source-root",
            "/kaggle/input/selected-kernel",
            "--source-kernel-reference",
            "shlokbhakta/q25-fim-selected",
            "--code-root",
            "/tmp/code",
            "--session-seconds",
            "3600",
        ],
    )

    args = WORKER._parse_args()

    assert args.source_root == Path("/kaggle/input/selected-kernel")
    assert args.source_kernel_reference == "shlokbhakta/q25-fim-selected"


def test_conversion_runtime_lock_is_hashed_and_cpu_only() -> None:
    lock = WORKER_PATH.with_name("requirements-conversion.lock")
    contents = lock.read_text(encoding="utf-8")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()

    assert digest == WORKER.REQUIREMENTS_LOCK_SHA256
    assert "torch==2.11.0+cpu" in contents
    assert "pyyaml==6.0.3" in contents
    assert "requests==2.34.2" in contents
    assert "tqdm==4.70.1" in contents
    assert "nvidia-" not in contents
    assert "triton==" not in contents
    assert "bitsandbytes==" not in contents


def test_worker_environment_ignores_host_credentials_and_cuda(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "synthetic-test-token")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://invalid.example/with-token")
    monkeypatch.setenv("PIP_INDEX_URL", "https://invalid.example/with-token")
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'credential.helper=synthetic-helper'")
    monkeypatch.setenv("GIT_SSH_COMMAND", "synthetic-credential-wrapper")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/synthetic/agent.sock")
    monkeypatch.setenv("HOME", "/host/user-home")

    runtime_root = tmp_path / "runtime"
    environment = WORKER._safe_environment(runtime_root=runtime_root)

    assert "HF_TOKEN" not in environment
    assert "UV_EXTRA_INDEX_URL" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert "PYTHONPATH" not in environment
    assert "GIT_CONFIG_PARAMETERS" not in environment
    assert "GIT_SSH_COMMAND" not in environment
    assert "SSH_AUTH_SOCK" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HOME"] == "/host/user-home"
    assert environment["HF_HOME"] == str(runtime_root / "home-cache" / "huggingface")
    assert environment["XDG_CACHE_HOME"] == str(runtime_root / "home-cache")
    assert environment["XDG_CONFIG_HOME"] == str(runtime_root / "xdg-config")
    assert environment["XDG_DATA_HOME"] == str(runtime_root / "xdg-data")
    assert environment["XDG_STATE_HOME"] == str(runtime_root / "xdg-state")
    assert environment["UV_NO_CONFIG"] == "1"
    assert environment["PIP_CONFIG_FILE"] == WORKER.os.devnull
    assert environment["GIT_CONFIG_GLOBAL"] == WORKER.os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"


def test_io_roots_reject_mounted_tree_writes_and_nested_output_roots(
    tmp_path: Path,
) -> None:
    mounted = tmp_path / "input"
    code = mounted / "source"
    code.mkdir(parents=True)
    working = tmp_path / "working"
    temporary = tmp_path / "temporary"

    output, runtime = WORKER._validate_io_roots(
        input_root=mounted,
        code_root=code,
        output_root=working / "result",
        runtime_root=temporary / "runtime",
    )
    assert output == working / "result"
    assert runtime == temporary / "runtime"

    with pytest.raises(ValueError, match="overlaps a mounted input"):
        WORKER._validate_io_roots(
            input_root=mounted,
            code_root=code,
            output_root=mounted / "result",
            runtime_root=temporary / "runtime",
        )
    with pytest.raises(ValueError, match="separate trees"):
        WORKER._validate_io_roots(
            input_root=mounted,
            code_root=code,
            output_root=working / "result",
            runtime_root=working / "result" / "runtime",
        )


def test_free_space_preflight_uses_the_least_available_write_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "working" / "result"
    runtime = tmp_path / "temp" / "runtime"
    values = iter((100, 50))
    monkeypatch.setattr(
        WORKER.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=next(values)),
    )

    assert WORKER._minimum_free_bytes(output, runtime) == 50
