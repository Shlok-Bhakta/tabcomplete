"""Bounded CPU-only conversion of one root-selected Q25 FIM export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LLAMA_CPP_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_REVISION = "f072b103714dfa1eee531f80b24512faf38e3dd2"
UV_VERSION = "0.12.3"
UV_WHEEL_NAME = "uv-0.12.3-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_WHEEL_SHA256 = "1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1"
REQUIREMENTS_LOCK_SHA256 = "88a05eef53ef5d3ff95c06d3d2e69266afdc59be032ef2bb414f7555f8de4e58"
MAX_SESSION_SECONDS = 10_800
MINIMUM_RESERVE_SECONDS = 1_800
RUNTIME_SETUP_PEAK_BYTES = 4 * 1024**3
MAX_ARTIFACT_BYTES = 12 * 1024**3
MINIMUM_FREE_BYTES = 2 * 1024**3
CHECKOUT_FILES = (
    "convert_hf_to_gguf.py",
    "conversion/base.py",
    "conversion/qwen.py",
    "gguf-py/gguf/constants.py",
    "gguf-py/gguf/gguf_reader.py",
    "gguf-py/gguf/vocab.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_source_kernel_reference(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(
        part not in {"", ".", ".."}
        and all(
            character.isascii() and (character.isalnum() or character in "._-")
            for character in part
        )
        for part in parts
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _contract_module(code_root: Path):
    src = code_root / "src"
    module = src / "tinycomplete" / "code_cpt" / "q25_fim_conversion.py"
    if not module.is_file():
        raise ValueError("conversion contract source is missing from the pinned code checkout")
    sys.path.insert(0, str(src))
    from tinycomplete.code_cpt import q25_fim_conversion

    return q25_fim_conversion


def _safe_environment(*, runtime_root: Path, python_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    secret_like = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
    }
    for name in tuple(env):
        lower_name = name.lower()
        unsafe_configuration = (
            name.startswith(("UV_INDEX", "PIP_INDEX", "PIP_EXTRA_INDEX"))
            or name
            in {
                "UV_CONFIG_FILE",
                "UV_EXTRA_INDEX_URL",
                "UV_FIND_LINKS",
                "UV_PYPI_URL",
                "PIP_CONFIG_FILE",
                "PIP_TRUSTED_HOST",
                "PIP_NO_INDEX",
                "PYTHONHOME",
                "PYTHONPATH",
                "CMAKE_PREFIX_PATH",
                "CUDA_HOME",
                "CUDA_PATH",
                "CUDACXX",
                "LD_LIBRARY_PATH",
                "GIT_ASKPASS",
                "GIT_CONFIG_PARAMETERS",
                "SSH_ASKPASS",
                "SSH_AUTH_SOCK",
                "GIT_SSH",
                "GIT_SSH_COMMAND",
                "GIT_CONFIG_COUNT",
            }
            or name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
            or lower_name.endswith("_proxy")
        )
        if (
            name in secret_like
            or name.endswith(("_API_KEY", "_ACCESS_TOKEN", "_SECRET", "_TOKEN", "_PASSWORD"))
            or unsafe_configuration
        ):
            env.pop(name, None)
    temp = runtime_root / "tmp"
    home_cache = runtime_root / "home-cache"
    xdg_config = runtime_root / "xdg-config"
    xdg_data = runtime_root / "xdg-data"
    xdg_state = runtime_root / "xdg-state"
    uv_cache = runtime_root / "uv-cache"
    for directory in (temp, home_cache, xdg_config, xdg_data, xdg_state, uv_cache):
        directory.mkdir(parents=True, exist_ok=True)
    path_prefix = str(python_path.parent) if python_path is not None else ""
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(home_cache / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_CACHE_DIR": str(home_cache / "pip"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TEMP": str(temp),
            "TMP": str(temp),
            "TMPDIR": str(temp),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_CACHE_DIR": str(uv_cache),
            "UV_COMPILE_BYTECODE": "0",
            "UV_NO_CACHE": "1",
            "UV_NO_CONFIG": "1",
            "UV_PYTHON_INSTALL_DIR": str(runtime_root / "managed-python"),
            "XDG_CACHE_HOME": str(home_cache),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_DATA_HOME": str(xdg_data),
            "XDG_STATE_HOME": str(xdg_state),
        }
    )
    if path_prefix:
        env["PATH"] = path_prefix + os.pathsep + env.get("PATH", "")
    return env


def _run(
    command: list[str],
    *,
    label: str,
    log_root: Path,
    environment: dict[str, str],
    deadline: float,
) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("conversion deadline reserve reached")
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        (log_root / f"{label}.log").write_text(stdout + stderr, encoding="utf-8")
        raise TimeoutError(f"{label} ran out of the conversion session") from None
    log = log_root / f"{label}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed; see the private conversion log")
    return result.stdout.strip()


def _require_empty_or_missing(path: Path, label: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{label} already contains files; use a fresh attempt directory")


def _validate_io_roots(
    *,
    input_root: Path,
    code_root: Path,
    output_root: Path,
    runtime_root: Path,
) -> tuple[Path, Path]:
    protected = (input_root.resolve(strict=True), code_root.resolve(strict=True))
    if output_root.is_symlink() or runtime_root.is_symlink():
        raise ValueError("conversion output and runtime roots must not be symbolic links")
    output = output_root.resolve()
    runtime = runtime_root.resolve()

    def overlaps(first: Path, second: Path) -> bool:
        return first == second or first.is_relative_to(second) or second.is_relative_to(first)

    if overlaps(output, runtime):
        raise ValueError("conversion output and runtime roots must be separate trees")
    for candidate, label in ((output, "output"), (runtime, "runtime")):
        if any(overlaps(candidate, root) for root in protected):
            raise ValueError(f"conversion {label} root overlaps a mounted input or code tree")
    return output, runtime


def _minimum_free_bytes(*write_roots: Path) -> int:
    if not write_roots:
        raise ValueError("at least one conversion write root is required")
    return min(shutil.disk_usage(root.parent).free for root in write_roots)


def _cpu_inventory_script() -> str:
    return "\n".join(
        (
            "import importlib.metadata as metadata",
            "import json, platform, torch, numpy, transformers, tokenizers",
            "names = {",
            "    dist.metadata['Name'].lower().replace('_', '-')",
            "    for dist in metadata.distributions()",
            "}",
            "inventory = {",
            "    'python': platform.python_version(),",
            "    'torch': torch.__version__,",
            "    'torch_cuda': torch.version.cuda,",
            "    'cuda_available': torch.cuda.is_available(),",
            "    'cuda_devices': torch.cuda.device_count(),",
            "    'numpy': numpy.__version__,",
            "    'transformers': transformers.__version__,",
            "    'tokenizers': tokenizers.__version__,",
            "    'nvidia_packages': sorted(",
            "        name for name in names",
            "        if name.startswith('nvidia-') or name == 'bitsandbytes'",
            "    ),",
            "}",
            "print(json.dumps(inventory, sort_keys=True))",
        )
    )


def _managed_uv(runtime_root: Path, *, log_root: Path, deadline: float) -> tuple[Path, str]:
    bootstrap = runtime_root / "uv-bootstrap"
    wheel_dir = bootstrap / "wheel"
    install_prefix = bootstrap / "prefix"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / UV_WHEEL_NAME
    base_env = _safe_environment(runtime_root=runtime_root)
    if not wheel_path.exists():
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "--dest",
                str(wheel_dir),
                f"uv=={UV_VERSION}",
            ],
            label="bootstrap-uv-download",
            log_root=log_root,
            environment=base_env,
            deadline=deadline,
        )
    if not wheel_path.is_file() or sha256_file(wheel_path) != UV_WHEEL_SHA256:
        raise RuntimeError("pinned uv bootstrap wheel identity failed")
    bin_dir = install_prefix / "bin"
    uv_path = bin_dir / "uv"
    if not uv_path.is_file():
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--prefix",
                str(install_prefix),
                str(wheel_path),
            ],
            label="bootstrap-uv-install",
            log_root=log_root,
            environment=base_env,
            deadline=deadline,
        )
    version = _run(
        [str(uv_path), "--version"],
        label="uv-version",
        log_root=log_root,
        environment=base_env,
        deadline=deadline,
    )
    if not version.startswith(f"uv {UV_VERSION} "):
        raise RuntimeError("uv bootstrap version differs from the pinned toolchain")
    return uv_path, sha256_file(uv_path)


def _install_python_runtime(
    *,
    uv_path: Path,
    runtime_root: Path,
    lock_path: Path,
    log_root: Path,
    deadline: float,
) -> tuple[Path, dict[str, Any]]:
    env = _safe_environment(runtime_root=runtime_root)
    python_install = runtime_root / "managed-python"
    python_env = runtime_root / "python311"
    if python_env.exists():
        raise FileExistsError(
            "CPU conversion environment already exists; use a fresh runtime directory"
        )
    _run(
        [
            str(uv_path),
            "python",
            "install",
            "3.11.15",
            "--install-dir",
            str(python_install),
            "--no-cache",
        ],
        label="python-install",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    )
    managed_env = {**env, "UV_MANAGED_PYTHON": "1", "UV_PYTHON_DOWNLOADS": "never"}
    python = _run(
        [str(uv_path), "python", "find", "3.11.15", "--managed-python", "--no-project"],
        label="python-find",
        log_root=log_root,
        environment=managed_env,
        deadline=deadline,
    ).splitlines()[-1]
    python_path = Path(python)
    _run(
        [str(uv_path), "venv", "--python", str(python_path), "--no-project", str(python_env)],
        label="venv-create",
        log_root=log_root,
        environment=managed_env,
        deadline=deadline,
    )
    venv_python = python_env / "bin" / "python"
    _run(
        [
            str(uv_path),
            "pip",
            "sync",
            "--python",
            str(venv_python),
            "--require-hashes",
            "--no-cache",
            "--index-strategy",
            "unsafe-best-match",
            str(lock_path),
        ],
        label="cpu-dependencies-install",
        log_root=log_root,
        environment=managed_env,
        deadline=deadline,
    )
    inventory_script = _cpu_inventory_script()
    raw_inventory = _run(
        [str(venv_python), "-c", inventory_script],
        label="cpu-runtime-check",
        log_root=log_root,
        environment=managed_env,
        deadline=deadline,
    )
    try:
        runtime = json.loads(raw_inventory.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError("CPU conversion runtime did not report a valid inventory") from None
    if (
        runtime.get("python") != "3.11.15"
        or runtime.get("torch") != "2.11.0+cpu"
        or runtime.get("torch_cuda") is not None
        or runtime.get("cuda_available") is not False
        or runtime.get("cuda_devices") != 0
        or runtime.get("numpy") != "2.4.6"
        or runtime.get("transformers") != "5.17.0"
        or runtime.get("tokenizers") != "0.23.2"
        or runtime.get("nvidia_packages")
    ):
        raise RuntimeError("conversion Python environment is not the pinned CPU-only runtime")
    return venv_python, runtime


def _checkout_llama_cpp(
    *,
    runtime_root: Path,
    log_root: Path,
    deadline: float,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    checkout = runtime_root / "llama.cpp"
    env = _safe_environment(runtime_root=runtime_root)
    if checkout.exists():
        raise FileExistsError("llama.cpp checkout already exists; use a fresh runtime directory")
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            LLAMA_CPP_REPOSITORY,
            str(checkout),
        ],
        label="llama-source-clone",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    )
    _run(
        ["git", "-C", str(checkout), "fetch", "--depth=1", "origin", LLAMA_CPP_REVISION],
        label="llama-source-fetch",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    )
    _run(
        ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
        label="llama-source-checkout",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    )
    revision = _run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        label="llama-source-revision",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    ).splitlines()[-1]
    dirty = _run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        label="llama-source-status",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    )
    if revision != LLAMA_CPP_REVISION or dirty:
        raise RuntimeError("llama.cpp checkout does not match the pinned clean revision")
    cmake_path = shutil.which("cmake")
    if cmake_path is None:
        raise RuntimeError("CMake is unavailable in the CPU conversion runtime")
    cmake_version = _run(
        [cmake_path, "--version"],
        label="cmake-version",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    ).splitlines()[0]
    files: dict[str, str] = {}
    for relative in CHECKOUT_FILES:
        path = checkout / relative
        if not path.is_file():
            raise RuntimeError("pinned llama.cpp checkout lacks a required conversion source")
        files[relative] = sha256_file(path)
    build = checkout / "build"
    cmake_env = {**env, "CMAKE_BUILD_PARALLEL_LEVEL": "2"}
    _run(
        [
            "cmake",
            "-S",
            str(checkout),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
            "-DGGML_CUDA=OFF",
            "-DGGML_VULKAN=OFF",
            "-DGGML_METAL=OFF",
            "-DGGML_SYCL=OFF",
            "-DGGML_OPENCL=OFF",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ],
        label="llama-cmake-configure",
        log_root=log_root,
        environment=cmake_env,
        deadline=deadline,
    )
    cache_path = build / "CMakeCache.txt"
    cxx_value = None
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
            cxx_value = line.partition("=")[2]
            break
    if not cxx_value:
        raise RuntimeError("CMake did not record the selected C++ compiler")
    cxx_path = Path(cxx_value).resolve(strict=True)
    cxx_version = _run(
        [str(cxx_path), "--version"],
        label="cxx-version",
        log_root=log_root,
        environment=env,
        deadline=deadline,
    ).splitlines()[0]
    _run(
        ["cmake", "--build", str(build), "--target", "llama-quantize", "-j", "2"],
        label="llama-quantizer-build",
        log_root=log_root,
        environment=cmake_env,
        deadline=deadline,
    )
    quantizer = build / "bin" / "llama-quantize"
    if not quantizer.is_file() or not os.access(quantizer, os.X_OK):
        raise RuntimeError("pinned CPU llama-quantize binary was not built")
    files["build/bin/llama-quantize"] = sha256_file(quantizer)
    tools = {
        "cmake_version": cmake_version,
        "cmake_binary_sha256": sha256_file(Path(cmake_path).resolve(strict=True)),
        "cxx_compiler": str(cxx_path),
        "cxx_version": cxx_version,
        "cxx_binary_sha256": sha256_file(cxx_path),
    }
    return checkout, files, tools


def _require_cpu_torch(python: Path, *, runtime_root: Path, logs: Path, deadline: float) -> None:
    script = (
        "import torch;"
        "assert torch.version.cuda is None;"
        "assert not torch.cuda.is_available();"
        "assert torch.cuda.device_count()==0"
    )
    _run(
        [str(python), "-c", script],
        label="cpu-only-guard",
        log_root=logs,
        environment=_safe_environment(runtime_root=runtime_root, python_path=python),
        deadline=deadline,
    )


def _load_runtime_contract(code_root: Path):
    return _contract_module(code_root)


def convert(
    *,
    input_root: Path,
    selection_path: Path,
    training_plan_path: Path,
    expected_selection_sha256: str,
    source_root: Path,
    source_kernel_reference: str,
    code_root: Path,
    output_root: Path,
    runtime_root: Path,
    session_seconds: float,
    reserve_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    if (
        not math.isfinite(session_seconds)
        or not math.isfinite(reserve_seconds)
        or not 0 < session_seconds <= MAX_SESSION_SECONDS
        or reserve_seconds < MINIMUM_RESERVE_SECONDS
        or reserve_seconds >= session_seconds
    ):
        raise ValueError("conversion session or finalization reserve is outside the frozen limit")
    if not _valid_source_kernel_reference(source_kernel_reference):
        raise ValueError("selected source kernel reference is invalid")
    deadline = started + session_seconds - reserve_seconds
    contract = _load_runtime_contract(code_root)
    if contract.LLAMA_CPP_REVISION != LLAMA_CPP_REVISION:
        raise RuntimeError("conversion contract and worker disagree on llama.cpp revision")
    lock_path = code_root / "kaggle" / "q25_fim_conversion_r1" / "requirements-conversion.lock"
    if not lock_path.is_file() or sha256_file(lock_path) != REQUIREMENTS_LOCK_SHA256:
        raise RuntimeError("CPU conversion dependency lock differs from the pinned file")
    if not input_root.is_dir() or not code_root.is_dir():
        raise FileNotFoundError("conversion input or code root is missing")
    output_root, runtime_root = _validate_io_roots(
        input_root=input_root,
        code_root=code_root,
        output_root=output_root,
        runtime_root=runtime_root,
    )
    _require_empty_or_missing(output_root, "conversion output")
    _require_empty_or_missing(runtime_root, "conversion runtime")

    verified = contract.verify_selection_bundle(
        input_root=input_root,
        selection_path=selection_path,
        training_plan_path=training_plan_path,
        expected_selection_sha256=expected_selection_sha256,
        source_root=source_root,
    )
    source_weight_bytes = verified.source_files["model.safetensors"]["bytes"]
    physical_parameter_count = max(
        verified.safetensors["physical_parameter_count"],
        contract.QWEN_GGUF_PHYSICAL_PARAMETER_COUNT,
    )
    f16_output_upper_bound = contract.projected_f16_gguf_bytes(
        source_safetensors_bytes=source_weight_bytes,
        physical_parameter_count=physical_parameter_count,
    )
    q4_output_upper_bound = contract.projected_q4_output_bytes(
        f16_gguf_bytes=f16_output_upper_bound
    )
    conversion_peak_bytes = contract.projected_conversion_peak_bytes(
        source_safetensors_bytes=source_weight_bytes,
        physical_parameter_count=physical_parameter_count,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    initial_roots = [input_root, code_root, output_root, runtime_root]
    projected = contract.preflight_storage(
        roots=initial_roots,
        additional_peak_bytes=RUNTIME_SETUP_PEAK_BYTES + conversion_peak_bytes,
        free_bytes=_minimum_free_bytes(output_root, runtime_root),
        max_artifact_bytes=MAX_ARTIFACT_BYTES,
        minimum_free_bytes=MINIMUM_FREE_BYTES,
    )
    output_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True)
    logs = output_root / "logs"
    state: dict[str, Any] = {
        "schema": "q25-fim-q4-conversion-run-v1",
        "status": "setup",
        "selection_sha256": verified.selection_sha256,
        "training_plan_sha256": verified.plan_sha256,
        "selected_arm": verified.selection["selected_arm"],
        "source_kernel_reference": source_kernel_reference,
        "source_mount_root": str(source_root.resolve(strict=True)),
        "source_export_manifest_sha256": verified.export_manifest_sha256,
        "source_fingerprint": verified.export_manifest["fingerprint"],
        "training_cursor": verified.expected_cursor,
        "source_files": verified.source_files,
        "safetensors": verified.safetensors,
        "tokenizer": verified.selection["tokenizer"],
        "model": verified.selection["model"],
        "conversion": {"format": "Q4_K_M", "gpu_enabled": False},
        "storage": {
            "artifact_cap_bytes": MAX_ARTIFACT_BYTES,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "preflight": projected,
            "projected_conversion_peak_bytes": conversion_peak_bytes,
            "f16_output_upper_bound_bytes": f16_output_upper_bound,
            "q4_output_upper_bound_bytes": q4_output_upper_bound,
        },
    }
    atomic_json(output_root / "conversion.json", state)
    try:
        uv_path, uv_binary_sha = _managed_uv(runtime_root, log_root=logs, deadline=deadline)
        python_path, runtime_identity = _install_python_runtime(
            uv_path=uv_path,
            runtime_root=runtime_root,
            lock_path=lock_path,
            log_root=logs,
            deadline=deadline,
        )
        runtime_root_bytes = contract.accounted_bytes([runtime_root])
        if runtime_root_bytes > RUNTIME_SETUP_PEAK_BYTES:
            raise RuntimeError("installed CPU runtime exceeded its declared setup storage bound")
        _require_cpu_torch(python_path, runtime_root=runtime_root, logs=logs, deadline=deadline)
        llama_root, converter_files, build_tools = _checkout_llama_cpp(
            runtime_root=runtime_root,
            log_root=logs,
            deadline=deadline,
        )
        runtime_root_bytes = contract.accounted_bytes([runtime_root])
        if runtime_root_bytes > RUNTIME_SETUP_PEAK_BYTES:
            raise RuntimeError("CPU runtime and converter exceeded their declared storage bound")
        converter_files["conversion/gguf-py/gguf/metadata.py"] = sha256_file(
            llama_root / "gguf-py" / "gguf" / "metadata.py"
        )
        worker_path = Path(__file__).resolve()
        contract_path = code_root / "src" / "tinycomplete" / "code_cpt" / "q25_fim_conversion.py"
        runtime_identity.update(
            {
                "uv_version": UV_VERSION,
                "uv_wheel_sha256": UV_WHEEL_SHA256,
                "uv_binary_sha256": uv_binary_sha,
                "python_executable_sha256": sha256_file(python_path.resolve(strict=True)),
                "requirements_lock_sha256": sha256_file(lock_path),
                "llama_cpp_revision": LLAMA_CPP_REVISION,
                "converter_source_sha256": converter_files,
                "build_tools": build_tools,
                "worker_source_sha256": sha256_file(worker_path),
                "contract_source_sha256": sha256_file(contract_path),
                "gguf_quantizer_sha256": converter_files["build/bin/llama-quantize"],
            }
        )
        state["runtime"] = runtime_identity
        state["status"] = "runtime_ready"
        state["storage"]["runtime_accounted_bytes"] = runtime_root_bytes
        atomic_json(output_root / "conversion.json", state)

        _require_cpu_torch(python_path, runtime_root=runtime_root, logs=logs, deadline=deadline)
        contract.verify_local_hf_tokenizer(verified.source_export, verified.selection["tokenizer"])
        staging = output_root / "conversion-input"
        staging.mkdir()
        for relative in verified.source_files:
            source_file = verified.source_export.joinpath(*Path(relative).parts)
            (staging / relative).symlink_to(source_file)
        f16_path = output_root / "selected-f16.gguf"
        q4_path = output_root / f"q25-{verified.selection['selected_arm']}-Q4_K_M.gguf"
        if q4_path.exists() or f16_path.exists():
            raise FileExistsError("conversion output file already exists")

        current = contract.preflight_storage(
            roots=[input_root, code_root, runtime_root, output_root],
            additional_peak_bytes=conversion_peak_bytes,
            free_bytes=_minimum_free_bytes(output_root, runtime_root),
            max_artifact_bytes=MAX_ARTIFACT_BYTES,
            minimum_free_bytes=MINIMUM_FREE_BYTES,
        )
        state["status"] = "converting_f16"
        state["storage"]["pre_conversion"] = current
        atomic_json(output_root / "conversion.json", state)
        convert_env = _safe_environment(runtime_root=runtime_root, python_path=python_path)
        _run(
            [
                str(python_path),
                str(llama_root / "convert_hf_to_gguf.py"),
                str(staging),
                "--outtype",
                "f16",
                "--outfile",
                str(f16_path),
            ],
            label="convert-f16-gguf",
            log_root=logs,
            environment=convert_env,
            deadline=deadline,
        )
        if not f16_path.is_file() or f16_path.stat().st_size <= 0:
            raise RuntimeError("F16 GGUF converter did not produce a nonempty file")
        f16_metadata, f16_tensors = contract.read_gguf_inspection(f16_path, llama_root)
        f16_verified = contract.validate_qwen_gguf(
            f16_metadata,
            f16_tensors,
            expected_file_type=1,
            precision_name="F16",
        )
        f16_upper_bound = contract.projected_f16_gguf_bytes(
            source_safetensors_bytes=source_weight_bytes,
            physical_parameter_count=max(
                f16_verified["physical_parameter_count"],
                contract.QWEN_GGUF_PHYSICAL_PARAMETER_COUNT,
            ),
        )
        if f16_path.stat().st_size > f16_upper_bound:
            raise RuntimeError("F16 GGUF exceeds its preflight storage estimate")
        q4_upper_bound = contract.projected_q4_output_bytes(f16_gguf_bytes=f16_path.stat().st_size)
        state["f16_intermediate"] = {
            "bytes": f16_path.stat().st_size,
            "sha256": sha256_file(f16_path),
            "inspection": f16_verified,
            "size_upper_bound_bytes": f16_upper_bound,
        }

        quantized_peak_bytes = q4_upper_bound + 64 * 1024**2
        before_quantize = contract.preflight_storage(
            roots=[input_root, code_root, runtime_root, output_root],
            additional_peak_bytes=quantized_peak_bytes,
            free_bytes=_minimum_free_bytes(output_root, runtime_root),
            max_artifact_bytes=MAX_ARTIFACT_BYTES,
            minimum_free_bytes=MINIMUM_FREE_BYTES,
        )
        state["status"] = "quantizing_q4"
        state["storage"]["pre_quantization"] = before_quantize
        atomic_json(output_root / "conversion.json", state)
        _run(
            [
                str(llama_root / "build" / "bin" / "llama-quantize"),
                str(f16_path),
                str(q4_path),
                "Q4_K_M",
            ],
            label="quantize-q4-k-m",
            log_root=logs,
            environment=convert_env,
            deadline=deadline,
        )
        if not q4_path.is_file() or q4_path.stat().st_size <= 0:
            raise RuntimeError("Q4 quantizer did not produce a nonempty file")
        q4_metadata, q4_tensors = contract.read_gguf_inspection(q4_path, llama_root)
        q4_verified = contract.validate_q4_gguf(q4_metadata, q4_tensors)
        if (
            q4_verified["physical_parameter_count"] != f16_verified["physical_parameter_count"]
            or q4_path.stat().st_size > q4_upper_bound
        ):
            raise RuntimeError("Q4 GGUF tensor storage differs from its preflight estimate")
        q4_digest = sha256_file(q4_path)
        state["q4_export"] = {
            "file": q4_path.name,
            "bytes": q4_path.stat().st_size,
            "sha256": q4_digest,
            "inspection": q4_verified,
            "size_upper_bound_bytes": q4_upper_bound,
        }
        state["status"] = "q4_verified_cleanup_pending"
        atomic_json(output_root / "conversion.json", state)

        if f16_path.is_symlink() or not f16_path.is_file():
            raise RuntimeError("worker-owned F16 intermediate is missing or unsafe to remove")
        if sha256_file(f16_path) != state["f16_intermediate"]["sha256"]:
            raise RuntimeError("worker-owned F16 intermediate changed after verification")
        f16_path.unlink()
        state.pop("f16_intermediate", None)
        final_storage = contract.preflight_storage(
            roots=[input_root, code_root, runtime_root, output_root],
            additional_peak_bytes=0,
            free_bytes=_minimum_free_bytes(output_root, runtime_root),
            max_artifact_bytes=MAX_ARTIFACT_BYTES,
            minimum_free_bytes=MINIMUM_FREE_BYTES,
        )
        state["storage"]["final"] = final_storage
        state["status"] = "complete"
        state["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output_root / "conversion.json", state)
        return state
    except BaseException:
        state["status"] = "failed"
        state["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output_root / "conversion.json", state)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-kernel-reference", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/q25_fim_conversion_r1"),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/kaggle/temp/q25_fim_conversion_r1"),
    )
    parser.add_argument("--session-seconds", type=float, required=True)
    parser.add_argument("--reserve-seconds", type=float, default=MINIMUM_RESERVE_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = convert(
            input_root=args.input_root,
            selection_path=args.selection,
            training_plan_path=args.training_plan,
            expected_selection_sha256=args.expected_selection_sha256,
            source_root=args.source_root,
            source_kernel_reference=args.source_kernel_reference,
            code_root=args.code_root,
            output_root=args.output_root,
            runtime_root=args.runtime_root,
            session_seconds=args.session_seconds,
            reserve_seconds=args.reserve_seconds,
        )
    except BaseException:
        print(
            "Q25 FIM conversion failed; inspect private conversion status and logs.",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_arm": result["selected_arm"],
                "q4_file": result["q4_export"]["file"],
                "q4_sha256": result["q4_export"]["sha256"],
                "q4_bytes": result["q4_export"]["bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
