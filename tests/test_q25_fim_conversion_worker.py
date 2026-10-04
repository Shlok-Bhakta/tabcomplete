from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKER_PATH = Path(__file__).parents[1] / "kaggle" / "q25_fim_conversion_r1" / "run.py"
SPEC = importlib.util.spec_from_file_location("q25_fim_conversion_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def _write_uv_wheel(
    path: Path, *, entries: tuple[tuple[str, bytes, int], ...] | None = None
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if entries is None:
        entries = (
            (
                WORKER.UV_WHEEL_BINARY_MEMBER,
                b"\x7fELFsynthetic-pinned-uv-binary",
                stat.S_IFREG | 0o755,
            ),
        )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, data, mode in entries:
            info = zipfile.ZipInfo(filename)
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cpu_runtime_probe_is_valid_python_without_importing_torch() -> None:
    compile(WORKER._cpu_inventory_script(), "cpu_runtime_probe.py", "exec")


def test_pinned_uv_extraction_uses_only_the_wheel_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / WORKER.UV_WHEEL_NAME
    wheel_sha = _write_uv_wheel(wheel)
    monkeypatch.setattr(WORKER, "UV_WHEEL_SHA256", wheel_sha)

    executable = WORKER._extract_pinned_uv(wheel, tmp_path / "bootstrap")

    assert executable == tmp_path / "bootstrap" / "bin" / "uv"
    assert executable.read_bytes() == b"\x7fELFsynthetic-pinned-uv-binary"
    assert stat.S_IMODE(executable.stat().st_mode) == 0o755


@pytest.mark.parametrize(
    "entries",
    [
        (),
        (
            (
                WORKER.UV_WHEEL_BINARY_MEMBER,
                b"\x7fELFone",
                stat.S_IFREG | 0o755,
            ),
            (
                WORKER.UV_WHEEL_BINARY_MEMBER,
                b"\x7fELFtwo",
                stat.S_IFREG | 0o755,
            ),
        ),
        (
            (
                WORKER.UV_WHEEL_BINARY_MEMBER,
                b"\x7fELFsymlink",
                stat.S_IFLNK | 0o777,
            ),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_pinned_uv_extraction_rejects_missing_duplicate_or_link_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entries: tuple[tuple[str, bytes, int], ...],
) -> None:
    wheel = tmp_path / WORKER.UV_WHEEL_NAME
    monkeypatch.setattr(WORKER, "UV_WHEEL_SHA256", _write_uv_wheel(wheel, entries=entries))
    bootstrap = tmp_path / "bootstrap"

    with pytest.raises(RuntimeError):
        WORKER._extract_pinned_uv(wheel, bootstrap)

    assert not (bootstrap / "bin" / "uv").exists()


def test_pinned_uv_extraction_checks_hash_and_wheel_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / WORKER.UV_WHEEL_NAME
    _write_uv_wheel(wheel)
    monkeypatch.setattr(WORKER, "UV_WHEEL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="identity"):
        WORKER._extract_pinned_uv(wheel, tmp_path / "bad-hash")

    renamed = tmp_path / "untrusted.whl"
    renamed.write_bytes(wheel.read_bytes())
    monkeypatch.setattr(WORKER, "UV_WHEEL_SHA256", hashlib.sha256(renamed.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        WORKER._extract_pinned_uv(renamed, tmp_path / "bad-name")


@pytest.mark.parametrize(
    ("version", "accepted"),
    [
        ("uv 0.12.3", True),
        ("uv 0.12.3 (x86_64-unknown-linux-gnu)", True),
        ("uv 0.12.30", False),
        ("uv 0.12.3a1", False),
        ("uv 0.12.3 unpinned-build", False),
    ],
)
def test_managed_uv_runs_extracted_binary_and_checks_exact_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    accepted: bool,
) -> None:
    runtime_root = tmp_path / "runtime"
    wheel = runtime_root / "uv-bootstrap" / "wheel" / WORKER.UV_WHEEL_NAME
    monkeypatch.setattr(WORKER, "UV_WHEEL_SHA256", _write_uv_wheel(wheel))
    calls: list[tuple[list[str], str]] = []

    def fake_run(command: list[str], *, label: str, **_: object) -> str:
        calls.append((command, label))
        assert label == "uv-version"
        return version

    monkeypatch.setattr(WORKER, "_run", fake_run)
    monkeypatch.setattr(
        WORKER.shutil,
        "which",
        lambda *_: pytest.fail("uv must never fall back to a host executable"),
    )

    if accepted:
        uv_path, binary_sha = WORKER._managed_uv(
            runtime_root, log_root=tmp_path / "logs", deadline=10**12
        )
        assert calls == [([str(uv_path), "--version"], "uv-version")]
        assert uv_path == runtime_root / "uv-bootstrap" / "bin" / "uv"
        assert binary_sha == hashlib.sha256(uv_path.read_bytes()).hexdigest()
    else:
        with pytest.raises(RuntimeError, match="version differs"):
            WORKER._managed_uv(runtime_root, log_root=tmp_path / "logs", deadline=10**12)


def test_managed_uv_rejects_symlinked_bootstrap_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    bootstrap_target = tmp_path / "external-bootstrap"
    runtime_root.mkdir()
    bootstrap_target.mkdir()
    (runtime_root / "uv-bootstrap").symlink_to(bootstrap_target, target_is_directory=True)
    monkeypatch.setattr(
        WORKER,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("must reject before downloading a wheel"),
    )

    with pytest.raises(RuntimeError, match="bootstrap directory is unsafe"):
        WORKER._managed_uv(runtime_root, log_root=tmp_path / "logs", deadline=10**12)

    assert list(bootstrap_target.iterdir()) == []


def test_managed_python_install_and_discovery_share_one_install_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    managed_python = runtime_root / "managed-python"
    venv_python = runtime_root / "python311" / "bin" / "python"
    managed_interpreter = managed_python / "cpython-3.11.15" / "bin" / "python3.11"
    inventory = {
        "python": "3.11.15",
        "torch": "2.11.0+cpu",
        "torch_cuda": None,
        "cuda_available": False,
        "cuda_devices": 0,
        "numpy": "2.4.6",
        "transformers": "5.17.0",
        "tokenizers": "0.23.2",
        "nvidia_packages": [],
    }
    calls: list[tuple[list[str], str, dict[str, str]]] = []

    def fake_run(
        command: list[str],
        *,
        label: str,
        environment: dict[str, str],
        **_: object,
    ) -> str:
        calls.append((command, label, environment))
        if label == "python-find":
            return str(managed_interpreter)
        if label == "cpu-runtime-check":
            return json.dumps(inventory)
        return ""

    monkeypatch.setattr(WORKER, "_run", fake_run)

    resolved_python, reported_inventory = WORKER._install_python_runtime(
        uv_path=tmp_path / "uv",
        runtime_root=runtime_root,
        lock_path=tmp_path / "requirements.lock",
        log_root=tmp_path / "logs",
        deadline=10**12,
    )

    install_command, install_label, install_env = next(
        call for call in calls if call[1] == "python-install"
    )
    find_command, find_label, find_env = next(call for call in calls if call[1] == "python-find")
    assert install_label == "python-install"
    assert find_label == "python-find"
    assert install_command[install_command.index("--install-dir") + 1] == str(managed_python)
    assert install_env["UV_PYTHON_INSTALL_DIR"] == find_env["UV_PYTHON_INSTALL_DIR"]
    assert find_env["UV_PYTHON_INSTALL_DIR"] == str(managed_python)
    assert find_command[-2:] == ["--managed-python", "--no-project"]
    assert resolved_python == venv_python
    assert reported_inventory == inventory


def test_subprocess_launch_failure_log_omits_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_executable(*_: object, **__: object) -> None:
        raise FileNotFoundError("private-path-must-not-escape")

    monkeypatch.setattr(WORKER.subprocess, "run", missing_executable)

    with pytest.raises(RuntimeError, match="could not be started"):
        WORKER._run(
            ["/synthetic/private-path"],
            label="uv-version",
            log_root=tmp_path / "logs",
            environment={},
            deadline=10**12,
        )

    log = (tmp_path / "logs" / "uv-version.log").read_text(encoding="utf-8")
    assert log == "subprocess launch failed: FileNotFoundError\n"
    assert "private-path" not in log


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
