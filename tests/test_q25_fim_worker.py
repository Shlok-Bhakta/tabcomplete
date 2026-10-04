from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "kaggle/q25_code_cpt_r2/run_fim.py"


def _load_worker() -> ModuleType:
    source = WORKER_PATH.read_text(encoding="utf-8")
    source = source.replace("__SESSION_JSON__", "{}")
    spec = importlib.util.spec_from_file_location("q25_fim_worker_test_module", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, str(WORKER_PATH), "exec"), module.__dict__)
    return module


def _sha(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


def _cursor(index: int) -> dict[str, int]:
    return {
        "next_example_index": index,
        "completed_updates": index,
        "attempted_updates": index,
        "skipped_updates": 0,
        "training_input_tokens": index * 1024,
        "supervised_target_tokens": index * 64,
        "epoch": 0,
    }


def _resume_identity(module: ModuleType, checkpoint: Path, index: int) -> dict[str, Any]:
    return {
        "fingerprint": _sha(b"stable-fim-run-fingerprint"),
        "checkpoint_sha256": module.sha256_file(checkpoint),
        "cursor": _cursor(index),
    }


def _write_attempt(
    module: ModuleType,
    output: Path,
    *,
    index: int,
    plan_sha256: str,
    input_manifest_sha256: str,
    arm: str,
    initializer: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    training = output / "training"
    training.mkdir(parents=True)
    checkpoint = training / f"checkpoint-{index}.safetensors"
    checkpoint.write_bytes(f"synthetic checkpoint {index}".encode())
    resume_identity = _resume_identity(module, checkpoint, index)
    module._write_json(
        checkpoint.with_suffix(checkpoint.suffix + ".complete.json"),
        {
            "fingerprint": resume_identity["fingerprint"],
            "sha256": resume_identity["checkpoint_sha256"],
        },
    )
    module._write_json(
        training / "latest.json",
        {
            "schema": "q25-fim-latest-checkpoint-v1",
            "path": checkpoint.name,
            "fingerprint": resume_identity["fingerprint"],
            "sha256": resume_identity["checkpoint_sha256"],
            "cursor": resume_identity["cursor"],
        },
    )
    identity = {
        "plan_sha256": plan_sha256,
        "arm": arm,
        "initializer": initializer,
        "corpus_metadata": {"metadata_sha256": _sha(b"fim-metadata")},
        "training_data": {"sha256": _sha(b"fim-training-data")},
    }
    module._write_json(
        training / "run_manifest.json",
        {"fingerprint": resume_identity["fingerprint"], "identity": identity},
    )
    if baseline is not None:
        module._write_json(output / "baseline" / "identity.json", baseline)
        for name, record in baseline["files"].items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = b"first-attempt baseline output: " + name.encode()
            assert len(content) == record["bytes"]
            assert _sha(content) == record["sha256"]
            path.write_bytes(content)
    return checkpoint, resume_identity


def _session(
    *,
    plan_sha256: str,
    input_manifest_sha256: str,
    arm: str,
    resume_source: str | None = None,
    resume_identity: dict[str, Any] | None = None,
    session_seconds: int = 3600,
    external_campaign_tokens: int = 0,
    attempt: int = 1,
) -> dict[str, Any]:
    return {
        "commit": "a" * 40,
        "plan_sha256": plan_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "arm": arm,
        "attempt": attempt,
        "session_seconds": session_seconds,
        "resume_source": resume_source,
        "resume_identity": resume_identity,
        "external_campaign_tokens": external_campaign_tokens,
    }


def _write_bundle(
    module: ModuleType,
    root: Path,
    *,
    external_campaign_tokens: int = 0,
    session_seconds: int = 10_800,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    input_root = root / "input"
    bundle = input_root / "fim-inputs"
    bundle.mkdir(parents=True)
    parent_cpt_sha = _sha(b"frozen parent CPT plan")
    preparation_sha = _sha(b"frozen FIM preparation plan")
    contents = {
        "train.jsonl": b'{"id":1}\n',
        "development.jsonl": b'{"id":2}\n',
        "causal200.jsonl": b'{"id":"causal"}\n',
        "line180.jsonl": b'{"id":"line"}\n',
    }
    metadata = {
        "raw_source_content_emitted": False,
        "sealed_test_accessed": False,
        "preparation_plan_sha256": preparation_sha,
        "parent_cpt_plan_sha256": parent_cpt_sha,
        "tokenizer_id": "Qwen/Qwen2.5-Coder-0.5B",
    }
    contents["corpus_metadata.json"] = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode()
    hashes = {name: _sha(value) for name, value in contents.items()}
    sizes = {name: len(value) for name, value in contents.items()}
    model_content = b"synthetic local initializer weights"
    model_sha = _sha(model_content)
    plan = {
        "schema": "q25-fim-training-plan-v1",
        "gpu_execution_authorized": True,
        "preparation_plan_sha256": preparation_sha,
        "parent_cpt_plan_sha256": parent_cpt_sha,
        "configuration": {
            "budget": {
                "maximum_campaign_input_tokens": 32_000_000,
                "session_seconds": 10_800,
                "new_artifact_bytes_cap": 4 * 1024**3,
                "minimum_free_bytes": 0,
                "runtime_setup_reserve_seconds": 1_800,
                "paid_compute": False,
                "automatic_renewal_use": False,
            },
            "training": {"max_input_tokens": 1_000_000},
            "runtime": module.EXPECTED_RUNTIME,
            "runtime_lock": {
                "runtime_lock_sha256": _sha(b"runtime lock"),
                "requirements_lock_sha256": _sha(b"requirements lock"),
                "bootstrap_uv_version": "0.12.3",
                "bootstrap_uv_wheel_sha256": module.UV_WHEEL_SHA256,
            },
        },
        "data": {
            "corpus_metadata_sha256": hashes["corpus_metadata.json"],
            "train": {"sha256": hashes["train.jsonl"], "bytes": sizes["train.jsonl"]},
            "development": {
                "sha256": hashes["development.jsonl"],
                "bytes": sizes["development.jsonl"],
            },
        },
        "initializers": {
            module.TRAIN_ARM: {
                "kind": "untouched_pretrained",
                "files": {"model.safetensors": {"sha256": model_sha, "bytes": len(model_content)}},
            }
        },
        "evaluation": {
            "attention_backend": "torch-efficient-sdpa-explicit-kv-repeat-v1",
            "fixtures": {
                "causal": {"sha256": hashes["causal200.jsonl"]},
                "line": {"sha256": hashes["line180.jsonl"]},
            },
        },
    }
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    contents["plan.json"] = plan_bytes
    for name, content in contents.items():
        (bundle / name).write_bytes(content)
    manifest = {
        "schema": "q25-fim-input-manifest-v1",
        "files": {
            name: {"bytes": len(content), "sha256": _sha(content)}
            for name, content in contents.items()
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = bundle / "input-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    model_dir = input_root / "base-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(model_content)
    session = _session(
        plan_sha256=_sha(plan_bytes),
        input_manifest_sha256=_sha(manifest_bytes),
        arm=module.TRAIN_ARM,
        session_seconds=session_seconds,
        external_campaign_tokens=external_campaign_tokens,
    )
    return input_root, bundle, session, plan


def _write_nvidia_inventory(
    host_site: Path,
    *,
    version_override: int | None = None,
    include_python_extension: bool = False,
) -> Path:
    host_site.mkdir(parents=True)
    requirements = host_site.parent / "requirements.lock"
    lines: list[str] = []
    for index in range(15):
        name = "nvidia-nvshmem-cu12" if index == 0 else f"nvidia-cuda-{index:03d}"
        lines.extend((f"{name}==1.0 \\", "    --hash=sha256:" + "a" * 64))
        component = "nvshmem" if index == 0 else f"cuda_{index:03d}"
        library = Path("nvidia") / component / "lib" / f"libnvidia_test_{index:03d}.so"
        library_path = host_site / library
        library_path.parent.mkdir(parents=True, exist_ok=True)
        elf = bytearray(20)
        elf[:6] = b"\x7fELF\x02\x01"
        elf[18:20] = (62).to_bytes(2, "little")
        library_path.write_bytes(elf)
        recorded_files = [library]
        if index == 0:
            plugin = Path("nvidia/nvshmem/lib/nvshmem_bootstrap_mpi.so.3")
            plugin_path = host_site / plugin
            plugin_path.write_bytes(elf)
            recorded_files.append(plugin)
            if include_python_extension:
                extension = Path("nvidia/nvshmem/lib/nvshmem_fake.cpython-313-x86_64-linux-gnu.so")
                (host_site / extension).write_bytes(elf)
                recorded_files.append(extension)
        pycache = library_path.parent.parent / "__pycache__"
        pycache.mkdir()
        (pycache / "test.cpython-313.pyc").write_bytes(b"host bytecode")

        dist_info_name = f"{name.replace('-', '_')}-1.0.dist-info"
        dist_info = host_site / dist_info_name
        dist_info.mkdir()
        actual_version = "2.0" if version_override == index else "1.0"
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {actual_version}\n\n",
            encoding="utf-8",
        )
        record = "\n".join(
            [
                *(f"{recorded_file.as_posix()},," for recorded_file in recorded_files),
                f"{dist_info_name}/METADATA,,",
                f"{dist_info_name}/RECORD,,",
                "",
            ]
        )
        (dist_info / "RECORD").write_text(record, encoding="utf-8")
    requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return requirements


def test_runtime_lock_verifies_report_sources_and_requirements_hashes() -> None:
    module = _load_worker()
    report_path = ROOT / module.RUNTIME_LOCK_RELATIVE
    report = json.loads(report_path.read_text(encoding="utf-8"))
    requirements = ROOT / report["requirements_lock"]["repo_relative_path"]
    plan = {
        "configuration": {
            "runtime": module.EXPECTED_RUNTIME,
            "runtime_lock": {
                "runtime_lock_sha256": _sha(report_path.read_bytes()),
                "requirements_lock_sha256": _sha(requirements.read_bytes()),
                "bootstrap_uv_version": report["bootstrap_uv_version"],
                "bootstrap_uv_wheel_sha256": report["bootstrap_uv_wheel_sha256"],
            },
        }
    }

    resolved_report, resolved_requirements = module.verify_runtime_lock_files(ROOT, plan)
    assert resolved_report == report
    assert resolved_requirements == requirements
    assert len(module.parse_pinned_requirements(resolved_requirements)) == 71

    mismatched = json.loads(json.dumps(plan))
    mismatched["configuration"]["runtime_lock"]["requirements_lock_sha256"] = _sha(b"wrong")
    with pytest.raises(module.WorkerError, match="runtime_requirements_lock_identity_invalid"):
        module.verify_runtime_lock_files(ROOT, mismatched)


def test_uv_command_uses_owned_installed_binary_despite_host_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    uv_site = tmp_path / "uv-site"
    owned_binary = uv_site / "bin" / "uv"
    owned_binary.parent.mkdir(parents=True)
    owned_payload = b"#!/bin/sh\nprintf '%s\\n' 'uv 0.12.3 (x86_64-unknown-linux-gnu)'\n"
    owned_binary.write_bytes(owned_payload)
    owned_binary.chmod(0o755)
    monkeypatch.setattr(module, "UV_BINARY_SHA256", _sha(owned_payload))

    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()
    host_uv = host_bin / "uv"
    host_uv.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'uv 0.12.9 (x86_64-unknown-linux-gnu)'\n",
        encoding="utf-8",
    )
    host_uv.chmod(0o755)

    resolved = module.verified_installed_uv_binary(uv_site)
    result = subprocess.run(
        [str(resolved), "--version"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": str(host_bin)},
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "uv 0.12.3 (x86_64-unknown-linux-gnu)"
    assert resolved == owned_binary


def test_uv_binary_identity_rejects_unpinned_target_executable(tmp_path: Path) -> None:
    module = _load_worker()
    uv_site = tmp_path / "uv-site"
    binary = uv_site / "bin" / "uv"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    with pytest.raises(module.WorkerError, match="uv_bootstrap_binary_identity_invalid"):
        module.verified_installed_uv_binary(uv_site)


def test_hashed_lock_filter_removes_exactly_verified_nvidia_packages(tmp_path: Path) -> None:
    module = _load_worker()
    source = ROOT / "reports/research/q25_code_cpt_r2/fim_runtime_requirements.lock"
    output = tmp_path / "requirements-without-nvidia.lock"
    source_versions = module.parse_pinned_requirements(source)

    removed = module.requirements_without_nvidia(source, output)
    filtered_versions = module.parse_pinned_requirements(output)
    assert len(removed) == 15
    assert removed == {name for name in source_versions if name.startswith("nvidia-")}
    assert filtered_versions == {
        name: version for name, version in source_versions.items() if name not in removed
    }
    filtered_text = output.read_text(encoding="utf-8")
    assert "--index-url https://pypi.org/simple" in filtered_text
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in filtered_text


def test_nvidia_inventory_links_only_exact_host_wheels(tmp_path: Path) -> None:
    module = _load_worker()
    host_site = tmp_path / "host-site"
    requirements = _write_nvidia_inventory(host_site)
    venv_site = tmp_path / "venv-site"
    venv_site.mkdir()
    inventory_path = tmp_path / "runtime_inventory.json"

    inventory = module.link_verified_nvidia_libraries(
        requirements,
        host_site,
        venv_site,
        inventory_path,
        host_python="3.13.15",
    )
    assert inventory["status"] == "verified_and_linked"
    assert inventory["expected_distribution_count"] == 15
    assert inventory["shared_library_bytes_reused"] == 16 * 20
    assert all(record["status"] == "verified" for record in inventory["distributions"])
    assert (venv_site / "nvidia/nvshmem/lib/libnvidia_test_000.so").is_symlink()
    assert (venv_site / "nvidia/nvshmem/lib/nvshmem_bootstrap_mpi.so.3").is_symlink()
    assert not (venv_site / "nvidia/nvshmem/__pycache__").exists()
    metadata_file = venv_site / "nvidia_nvshmem_cu12-1.0.dist-info/METADATA"
    assert metadata_file.is_symlink()
    assert json.loads(inventory_path.read_text(encoding="utf-8")) == inventory
    assert module._runtime_regular_bytes(venv_site) == 0


def test_nvidia_inventory_rejects_python_extension_even_for_vendor_lib_path(
    tmp_path: Path,
) -> None:
    module = _load_worker()
    host_site = tmp_path / "host-site"
    requirements = _write_nvidia_inventory(host_site, include_python_extension=True)
    venv_site = tmp_path / "venv-site"
    venv_site.mkdir()
    inventory_path = tmp_path / "runtime_inventory.json"

    with pytest.raises(module.WorkerError, match="runtime_nvidia_host_platform_inventory_mismatch"):
        module.link_verified_nvidia_libraries(
            requirements,
            host_site,
            venv_site,
            inventory_path,
            host_python="3.13.15",
        )

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    nvshmem = next(
        record for record in inventory["distributions"] if record["name"] == "nvidia-nvshmem-cu12"
    )
    assert nvshmem["status"] == "platform_file_inventory_mismatch"
    assert nvshmem["offending_relative_file"] == (
        "nvidia/nvshmem/lib/nvshmem_fake.cpython-313-x86_64-linux-gnu.so"
    )
    assert not (venv_site / "nvidia").exists()


def test_nvidia_version_mismatch_writes_inventory_and_fails_before_linking(
    tmp_path: Path,
) -> None:
    module = _load_worker()
    host_site = tmp_path / "host-site"
    requirements = _write_nvidia_inventory(host_site, version_override=4)
    venv_site = tmp_path / "venv-site"
    venv_site.mkdir()
    inventory_path = tmp_path / "runtime_inventory.json"

    with pytest.raises(module.WorkerError, match="runtime_nvidia_host_version_mismatch"):
        module.link_verified_nvidia_libraries(
            requirements,
            host_site,
            venv_site,
            inventory_path,
            host_python="3.13.15",
        )

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["status"] == "rejected"
    assert inventory["failure_reason"] == "runtime_nvidia_host_version_mismatch"
    assert not (venv_site / "nvidia").exists()


def test_venv_interpreter_must_resolve_under_owned_managed_python(tmp_path: Path) -> None:
    module = _load_worker()
    managed_root = tmp_path / "managed-python"
    target = managed_root / "cpython-3.11.15/bin/python3.11"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    interpreter_link = tmp_path / "venv/bin/python"
    interpreter_link.parent.mkdir(parents=True)
    interpreter_link.symlink_to(target)
    site_packages = tmp_path / "venv/lib/python3.11/site-packages"
    site_packages.mkdir(parents=True)

    assert (
        module.verified_managed_python311(interpreter_link, managed_root, site_packages) == target
    )

    external_interpreter = tmp_path / "host-python"
    external_interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    external_interpreter.chmod(0o755)
    interpreter_link.unlink()
    interpreter_link.symlink_to(external_interpreter)
    with pytest.raises(module.WorkerError, match="python311_venv_interpreter_not_owned"):
        module.verified_managed_python311(interpreter_link, managed_root, site_packages)


def test_runtime_probe_writes_parseable_identity_with_selected_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    expected = {
        "python": platform.python_version(),
        "torch": "mocktorch-cpu-test",
        "transformers": "mock-transformers",
        "bitsandbytes": "mock-bitsandbytes",
        "cuda_runtime": "12.8",
    }
    monkeypatch.setattr(module, "EXPECTED_RUNTIME", expected)
    monkeypatch.setattr(module, "_version_tuple", lambda _value: (2, 35))
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(module, "RUNTIME_ROOT", runtime_root)

    fake_site = tmp_path / "site"
    fake_site.mkdir()
    (fake_site / "torch.py").write_text(
        "__version__ = 'mocktorch-cpu-test'\n"
        "class _Version:\n    cuda = '12.8'\n"
        "class _Cuda:\n"
        "    def is_available(self): return True\n"
        "    def get_device_name(self, index): return 'Tesla T4 (mocked)'\n"
        "version = _Version()\ncuda = _Cuda()\n",
        encoding="utf-8",
    )
    for name, version in (
        ("transformers", expected["transformers"]),
        ("bitsandbytes", expected["bitsandbytes"]),
    ):
        info = fake_site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n",
            encoding="utf-8",
        )
    requirements = tmp_path / "runtime.lock"
    nvidia_names = [f"nvidia-cuda-{index:03d}" for index in range(15)]
    requirements.write_text(
        "".join(f"{name}==1.0 \\\n    --hash=sha256:{'a' * 64}\n" for name in nvidia_names),
        encoding="utf-8",
    )
    for name in nvidia_names:
        info = fake_site / f"{name.replace('-', '_')}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n\n",
            encoding="utf-8",
        )

    session = _session(
        plan_sha256=_sha(b"plan"),
        input_manifest_sha256=_sha(b"inputs"),
        arm=module.TRAIN_ARM,
    )
    worker = module.Worker(session)
    worker.out = output
    worker.python311 = Path(sys.executable)
    worker.runtime_requirements = requirements
    monkeypatch.setattr(worker, "check_storage", lambda **_kwargs: None)

    identity = worker._verify_python311_runtime(
        {},
        {
            **dict(__import__("os").environ),
            "PYTHONPATH": str(fake_site),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    identity_path = output / "runtime-identity.json"
    assert identity_path.read_bytes().endswith(b"\n")
    assert identity["python"] == expected["python"]
    assert identity["torch"] == expected["torch"]
    assert identity["nvidia"] == dict.fromkeys(nvidia_names, "1.0")


def test_initializer_probe_uses_selected_interpreter_and_emits_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    package_root = tmp_path / "packages/tinycomplete"
    (package_root / "code_cpt").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "code_cpt/__init__.py").write_text("", encoding="utf-8")
    (package_root / "code_cpt/q25_fim.py").write_text(
        "def verify_initializer(model, *, arm, entry):\n"
        "    return {'model': str(model), 'arm': arm, 'kind': entry['kind']}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RUNTIME_ROOT", runtime_root)
    worker = module.Worker(
        _session(
            plan_sha256=_sha(b"plan"),
            input_manifest_sha256=_sha(b"inputs"),
            arm=module.TRAIN_ARM,
        )
    )
    worker.out = output
    worker.python311 = Path(sys.executable)
    model = tmp_path / "initializer"
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(tmp_path / "packages"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    identity = worker._verify_initializer(
        model,
        arm=module.TRAIN_ARM,
        entry={"kind": "untouched_pretrained"},
        env=env,
    )
    assert identity == {
        "model": str(model),
        "arm": module.TRAIN_ARM,
        "kind": "untouched_pretrained",
    }


def test_baseline_is_carried_through_two_successive_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    plan_sha256 = _sha(b"frozen FIM plan")
    input_manifest_sha256 = _sha(b"frozen inputs")
    initializer = {
        "arm": module.TRAIN_ARM,
        "model_revision": "synthetic-pinned-revision",
        "files": {"model.safetensors": {"sha256": _sha(b"weights"), "bytes": 7}},
    }
    initializer_sha256 = module.canonical_sha256(initializer)

    # Attempt 1 has the original before-FIM evaluations plus a model export.
    initial_files: dict[str, dict[str, Any]] = {}
    initial_contents: dict[str, bytes] = {}
    for name in ("before/development/results.json", "before/line/results.json"):
        content = b"first-attempt baseline output: " + name.encode()
        initial_contents[name] = content
        initial_files[name] = {"bytes": len(content), "sha256": _sha(content)}
    baseline = {
        "schema": "q25-fim-baseline-v1",
        "arm": module.TRAIN_ARM,
        "plan_sha256": plan_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "initializer_identity_sha256": initializer_sha256,
        "measured_seconds": 17.25,
        "files": initial_files,
    }
    attempt1_input = tmp_path / "mount-attempt-1"
    attempt1_root = attempt1_input / "previous-attempt-1"
    _, resume1 = _write_attempt(
        module,
        attempt1_root,
        index=1,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        initializer=initializer,
        baseline=baseline,
    )
    export = attempt1_root / "training/inference-f16/model.safetensors"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"do not copy weights")
    session1 = _session(
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        resume_source="owner/fim-arm-attempt-1",
        resume_identity=resume1,
    )
    _, resolved_root1, resolved_baseline1 = module.resolve_verified_resume(
        attempt1_input,
        session1,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        initializer_sha256=initializer_sha256,
        arm=module.TRAIN_ARM,
    )
    assert resolved_root1 == attempt1_root
    assert resolved_baseline1 == baseline

    # Attempt 2 copies only the verified baseline outputs and preserves its identity.
    attempt2_root = tmp_path / "attempt-2-output"
    monkeypatch.setattr(module, "OUT", attempt2_root)
    worker2 = module.Worker(session1)
    worker2.out.mkdir(parents=True)
    worker2._record_baseline(
        initializer_sha256=initializer_sha256,
        measured_seconds=17.25,
        plan={"configuration": {"budget": {"new_artifact_bytes_cap": 2 * 1024**3}}},
        mounted_bytes=0,
        previous_root=resolved_root1,
        inherited=resolved_baseline1,
    )
    assert json.loads((attempt2_root / "baseline/identity.json").read_text()) == baseline
    reference2 = json.loads((attempt2_root / "baseline/reference.json").read_text())
    assert reference2["source"] == "owner/fim-arm-attempt-1"
    assert reference2["source_identity_sha256"] == module.canonical_sha256(baseline)
    for name, content in initial_contents.items():
        assert (attempt2_root / name).read_bytes() == content
    assert not (attempt2_root / "training/inference-f16/model.safetensors").exists()

    # Attempt 2's output becomes attempt 3's sole resume mount. The original baseline
    # must still verify there, and attempt 3 must carry the same files and identity.
    attempt2_input = tmp_path / "mount-attempt-2"
    attempt2_input.mkdir()
    mounted_attempt2 = attempt2_input / "previous-attempt-2"
    attempt2_root.rename(mounted_attempt2)
    checkpoint2, resume2 = _write_attempt(
        module,
        mounted_attempt2,
        index=2,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        initializer=initializer,
        baseline=None,
    )
    session2 = _session(
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        arm=module.TRAIN_ARM,
        resume_source="owner/fim-arm-attempt-2",
        resume_identity=resume2,
        attempt=2,
    )
    resolved_checkpoint2, resolved_root2, resolved_baseline2 = module.resolve_verified_resume(
        attempt2_input,
        session2,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        initializer_sha256=initializer_sha256,
        arm=module.TRAIN_ARM,
    )
    assert resolved_checkpoint2 == checkpoint2
    assert resolved_root2 == mounted_attempt2
    assert resolved_baseline2 == baseline

    attempt3_root = tmp_path / "attempt-3-output"
    monkeypatch.setattr(module, "OUT", attempt3_root)
    worker3 = module.Worker(session2)
    worker3.out.mkdir(parents=True)
    worker3._record_baseline(
        initializer_sha256=initializer_sha256,
        measured_seconds=17.25,
        plan={"configuration": {"budget": {"new_artifact_bytes_cap": 2 * 1024**3}}},
        mounted_bytes=0,
        previous_root=resolved_root2,
        inherited=resolved_baseline2,
    )
    assert json.loads((attempt3_root / "baseline/identity.json").read_text()) == baseline
    reference3 = json.loads((attempt3_root / "baseline/reference.json").read_text())
    assert reference3["source"] == "owner/fim-arm-attempt-2"
    assert reference3["source_identity_sha256"] == module.canonical_sha256(baseline)
    for name, content in initial_contents.items():
        assert (attempt3_root / name).read_bytes() == content
    assert not (attempt3_root / "training/inference-f16/model.safetensors").exists()


def test_session_validation_rejects_invalid_budget_and_resume_identity() -> None:
    module = _load_worker()
    base = _session(
        plan_sha256=_sha(b"plan"),
        input_manifest_sha256=_sha(b"inputs"),
        arm=module.TRAIN_ARM,
    )
    assert module.validate_session(base) == base

    invalid_sessions = []
    too_long = dict(base, session_seconds=module.MAX_SESSION_SECONDS + 1)
    invalid_sessions.append((too_long, "session_identity_or_budget_invalid"))
    boolean_budget = dict(base, external_campaign_tokens=True)
    invalid_sessions.append((boolean_budget, "session_identity_or_budget_invalid"))
    missing_resume_identity = dict(base, resume_source="owner/fim-attempt")
    invalid_sessions.append((missing_resume_identity, "resume_identity_missing"))
    identity_without_source = dict(base, resume_identity={})
    invalid_sessions.append((identity_without_source, "resume_identity_without_source"))
    invalid_attempt = dict(base, attempt=True)
    invalid_sessions.append((invalid_attempt, "session_identity_or_budget_invalid"))
    missing_attempt = {key: value for key, value in base.items() if key != "attempt"}
    invalid_sessions.append((missing_attempt, "session_missing_required_fields"))

    for invalid, expected_reason in invalid_sessions:
        with pytest.raises(module.WorkerError) as error:
            module.validate_session(invalid)
        assert error.value.reason == expected_reason


def test_frozen_bundle_accepts_bound_files_and_rejects_hash_mismatch(tmp_path: Path) -> None:
    module = _load_worker()
    input_root, bundle, session, plan = _write_bundle(module, tmp_path)
    manifest = json.loads((bundle / "input-manifest.json").read_text())
    matched_root, matched_manifest = module.find_input_manifest(
        input_root, session["input_manifest_sha256"]
    )
    assert matched_root == bundle
    paths, loaded_plan = module.verify_input_bundle(matched_root, matched_manifest, session)
    assert loaded_plan == plan
    assert set(paths) == module.REQUIRED_INPUT_FILES

    (bundle / "train.jsonl").write_bytes(b'{"id":3}\n')
    with pytest.raises(module.WorkerError) as error:
        module.verify_input_bundle(bundle, manifest, session)
    assert error.value.reason == "input_file_hash_or_size_mismatch"


def test_raw_regressions_use_the_fim_attention_wrapper() -> None:
    module = _load_worker()
    paths = {
        "causal200.jsonl": Path("/frozen/causal200.jsonl"),
        "line180.jsonl": Path("/frozen/line180.jsonl"),
    }
    commands = module._regression_commands(
        python=Path("/runtime/python"),
        model=Path("/model"),
        paths=paths,
        output=Path("/output"),
        plan_sha256="a" * 64,
        alias="untouched-q25-after-fim",
    )
    assert [name for name, _ in commands] == ["regression-causal", "regression-line"]
    for _, command in commands:
        assert command[1].endswith("evaluate_q25_fim_regression.py")
        assert command[command.index("--attention-backend") + 1] == (
            "torch-efficient-sdpa-explicit-kv-repeat-v1"
        )


@pytest.mark.parametrize(
    "mutation,expected_reason",
    [
        ("missing_manifest_entry", "input_manifest_file_set_invalid"),
        ("unlisted_file", "input_bundle_contains_unlisted_file"),
        ("symlink", "input_manifest_file_missing_or_unsafe"),
    ],
)
def test_frozen_bundle_rejects_malformed_or_unsafe_inventory(
    tmp_path: Path, mutation: str, expected_reason: str
) -> None:
    module = _load_worker()
    _, bundle, session, _ = _write_bundle(module, tmp_path)
    manifest = json.loads((bundle / "input-manifest.json").read_text())
    if mutation == "missing_manifest_entry":
        manifest["files"].pop("line180.jsonl")
    elif mutation == "unlisted_file":
        (bundle / "extra.jsonl").write_text("{}\n")
    else:
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes((bundle / "train.jsonl").read_bytes())
        (bundle / "train.jsonl").unlink()
        (bundle / "train.jsonl").symlink_to(outside)

    with pytest.raises(module.WorkerError) as error:
        module.verify_input_bundle(bundle, manifest, session)
    assert error.value.reason == expected_reason


def test_run_stage_uses_only_time_left_after_reserved_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    session = _session(
        plan_sha256=_sha(b"plan"),
        input_manifest_sha256=_sha(b"inputs"),
        arm=module.TRAIN_ARM,
    )
    worker = module.Worker(session)
    worker.out = tmp_path / "worker"
    worker.out.mkdir()
    worker.deadline = 112.0
    ticks = itertools.count()
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(monotonic=lambda: 100.0 + next(ticks) * 0.25),
    )
    observed: dict[str, Any] = {}

    def fake_run(_command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert worker.run_stage(["mock-stage"], "mock", reserve_seconds=5) == 0
    assert observed["timeout"] == pytest.approx(7.0)
    assert worker.status["stages"][-1]["exit_code"] == 0


def test_run_stage_fails_closed_when_deadline_reserve_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_worker()
    session = _session(
        plan_sha256=_sha(b"plan"),
        input_manifest_sha256=_sha(b"inputs"),
        arm=module.TRAIN_ARM,
    )
    worker = module.Worker(session)
    worker.out = tmp_path / "worker"
    worker.out.mkdir()
    worker.deadline = 105.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 100.0))
    calls: list[list[str]] = []

    def capture_forbidden_stage(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", capture_forbidden_stage)

    with pytest.raises(module.WorkerError) as error:
        worker.run_stage(["must-not-run"], "reserved", reserve_seconds=5)
    assert error.value.reason == "session_deadline_reserve_reached"
    assert calls == []


@pytest.mark.parametrize(
    ("session_seconds", "expected_state", "training_expected"),
    [
        (10_800, "partial_checkpoint_preserved", True),
        (2_200, "training_deferred_insufficient_time", False),
    ],
)
def test_execute_persists_training_boundary_and_forwards_global_token_carry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_seconds: int,
    expected_state: str,
    training_expected: bool,
) -> None:
    module = _load_worker()
    carry_tokens = 9_876_543
    input_root, bundle, session, _ = _write_bundle(
        module,
        tmp_path,
        external_campaign_tokens=carry_tokens,
        session_seconds=session_seconds,
    )
    repo = tmp_path / "repo"
    fixture = repo / "data/benchmarks/code_completion_v2.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes((bundle / "causal200.jsonl").read_bytes())
    output = tmp_path / "output/q25_fim"
    monkeypatch.setattr(module, "INPUT_ROOT", input_root)
    monkeypatch.setattr(module, "OUT", output)
    monkeypatch.setattr(module, "REPO", repo)

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    monkeypatch.setattr(module, "RUNTIME_ROOT", runtime_root)
    fake_python = runtime_root / "venv/bin/python"

    def mock_checkout(_worker: Any, _plan: dict[str, Any]) -> None:
        return None

    def mock_prepare(
        worker: Any, _plan: dict[str, Any], _report: dict[str, Any], _requirements: Path
    ) -> None:
        worker.python311 = fake_python
        worker.runtime_requirements = tmp_path / "requirements.lock"
        worker.runtime_inventory = {"shared_library_bytes_reused": 0}

    def mock_runtime_probe(
        worker: Any, _plan: dict[str, Any], _env: dict[str, str]
    ) -> dict[str, Any]:
        worker.status["hardware"] = {
            "torch": module.EXPECTED_RUNTIME["torch"],
            "cuda_devices_visible": 1,
            "training_device": "Tesla T4 (mocked, no GPU used)",
            "world_size": 1,
            "other_devices_unused": True,
        }
        return {}

    def mock_initializer(
        _worker: Any, _model: Path, *, arm: str, entry: dict[str, Any], env: dict[str, str]
    ) -> dict[str, Any]:
        del env
        return {"arm": arm, "kind": entry["kind"], "files": entry["files"]}

    monkeypatch.setattr(module.Worker, "_checkout_repository", mock_checkout)
    monkeypatch.setattr(module.Worker, "_prepare_python311", mock_prepare)
    monkeypatch.setattr(module.Worker, "_verify_python311_runtime", mock_runtime_probe)
    monkeypatch.setattr(module.Worker, "_verify_initializer", mock_initializer)
    monkeypatch.setattr(
        module,
        "verify_runtime_lock_files",
        lambda _repo, _plan: ({}, tmp_path / "requirements.lock"),
    )

    calls: list[list[str]] = []
    training_started_at_launch: bool | None = None
    startup_status: dict[str, Any] | None = None

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        nonlocal startup_status, training_started_at_launch
        calls.append(command)
        if startup_status is None and command[:3] == [
            str(fake_python),
            "-m",
            "tinycomplete.code_cpt.q25_fim",
        ]:
            startup_status = json.loads((output / "worker-status.json").read_text())
        if "--attention-smoke" in command:
            result_dir = Path(command[command.index("--output") + 1])
            result_dir.mkdir(parents=True, exist_ok=True)
            tokens = 77
            smoke_report = {
                "schema": "q25-fim-attention-smoke-v1",
                "attention_backend": "torch-efficient-sdpa-explicit-kv-repeat-v1",
                "key_value_head_expansion": "explicit-repeat",
                "plan_sha256": session["plan_sha256"],
                "success": True,
                "query_tokens": tokens,
                "query_heads": 14,
                "key_value_heads": 2,
                "head_dim": 64,
                "dtype": "float16",
                "cuda_device": "Mock T4",
                "peak_allocated_bytes": 1024,
                "dense_fp32_attention_score_bytes": tokens * tokens * 14 * 4,
            }
            (result_dir / "attention-smoke.json").write_text(
                json.dumps(smoke_report) + "\n", encoding="utf-8"
            )
            return SimpleNamespace(returncode=0)
        if "--mode" in command and "--output" in command:
            result_dir = Path(command[command.index("--output") + 1])
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "result.json").write_text("{}\n")
            return SimpleNamespace(returncode=0)
        if "--execute" in command:
            training_status = json.loads((output / "worker-status.json").read_text())
            training_started_at_launch = training_status["training_started"]
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    worker = module.Worker(session)
    assert worker.execute() == 0

    assert worker.status["state"] == expected_state
    assert worker.status["commit"] == session["commit"]
    assert worker.status["attempt"] == session["attempt"]
    assert worker.status["training_started"] is training_expected
    assert worker.status["attention_smoke"]["query_tokens"] == 77
    assert startup_status is not None
    assert startup_status["training_started"] is False
    assert startup_status["commit"] == session["commit"]
    assert startup_status["attempt"] == session["attempt"]
    preflight = next(
        command
        for command in calls
        if "tinycomplete.code_cpt.q25_fim" in command and "--execute" not in command
    )
    assert any(
        stage["name"] == "trainer-preflight" and stage["exit_code"] == 0
        for stage in worker.status["stages"]
    )
    commands_to_check = [preflight]
    if training_expected:
        training = next(command for command in calls if "--execute" in command)
        commands_to_check.append(training)
        assert worker.status["training_exit_code"] == 1
        assert training_started_at_launch is True
    else:
        assert not any("--execute" in command for command in calls)
        assert training_started_at_launch is None
    for command in commands_to_check:
        assert command[0] == str(fake_python)
        token_flag = command.index("--external-campaign-tokens")
        assert command[token_flag + 1] == str(carry_tokens)
