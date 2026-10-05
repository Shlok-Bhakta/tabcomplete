"""Bounded Kaggle worker for one frozen Qwen2.5 FIM arm."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import time
import urllib.request
from pathlib import Path
from typing import Any

SESSION = __SESSION_JSON__  # type: ignore[name-defined]  # noqa: F821
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
OUT = WORK_ROOT / "q25_fim_r2"
REPO = Path("/kaggle/temp/tabcomplete-q25-code-cpt-r2")
RUNTIME_ROOT = Path("/kaggle/temp/tabcomplete-q25-code-cpt-r2-python311")
RUNTIME_LOCK_RELATIVE = Path("reports/research/q25_code_cpt_r2/fim_runtime_lock.json")
RUNTIME_REQUIREMENTS_RELATIVE = Path(
    "reports/research/q25_code_cpt_r2/fim_runtime_requirements.lock"
)
UV_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/8d/c4/97fdd4fca11d06633bb500849f70e4e6b201bcba3833894732e709be2d60/"
    "uv-0.12.3-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
UV_WHEEL_FILENAME = "uv-0.12.3-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_WHEEL_SHA256 = "1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1"
UV_WHEEL_BYTES = 22_346_263
UV_BINARY_SHA256 = "729d27dbea534ee540a2d3ef43a62fa1a10af7fcbb6d57a70d5859509f624578"
EXPECTED_RUNTIME = {
    "python": "3.11.15",
    "torch": "2.11.0+cu128",
    "transformers": "5.17.0",
    "bitsandbytes": "0.50.2",
    "cuda_runtime": "12.8",
}
STARTED_MONOTONIC_ENV = "TABCOMPLETE_FIM_STARTED_MONOTONIC"
PYTHON311_READY_ENV = "TABCOMPLETE_FIM_PYTHON311_READY"
MAX_SESSION_SECONDS = 10_800
MAX_CAMPAIGN_TOKENS = 32_000_000
MAX_ARM_TOKENS = 4_194_304
SCALE_PLAN_SCHEMA = "q25-completion-scale-training-plan-v1"
SCALE_BRANCH = "research/q25-completion-scale-r1"
SCALE_VARIANTS = {"repeat", "scaled"}
SCALE_MAX_CAMPAIGN_TOKENS = 10_000_000
SCALE_MAX_ARM_TOKENS = 8_000_000
MAX_ARTIFACT_BYTES = 12 * 1024**3
MINIMUM_FREE_BYTES = 2 * 1024**3
MINIMUM_FINAL_RESERVE_SECONDS = 1_800
TRAIN_ARM = "untouched_q25_to_fim"
CPT_ARM = "completed_cpt_q25_to_fim"
ARMS = (TRAIN_ARM, CPT_ARM)
REQUIRED_INPUT_FILES = {
    "plan.json",
    "train.jsonl",
    "development.jsonl",
    "corpus_metadata.json",
    "causal200.jsonl",
    "line180.jsonl",
}
SCALE_REQUIRED_INPUT_FILES = {
    "plan.json",
    "repeat_train.jsonl",
    "scaled_train.jsonl",
    "development_new.jsonl",
    "development_previous.jsonl",
    "corpus_metadata.json",
    "causal200.jsonl",
    "line180.jsonl",
}
ALLOWED_INPUT_EXTRAS = {"input-manifest.json", "dataset-metadata.json"}
CHECKPOINT_CURSOR_FIELDS = (
    "next_example_index",
    "completed_updates",
    "attempted_updates",
    "skipped_updates",
    "training_input_tokens",
    "supervised_target_tokens",
    "epoch",
)


class WorkerError(RuntimeError):
    """A sanitized worker failure with a stable reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise WorkerError("file_read_failed") from None
    return digest.hexdigest()


def verified_installed_uv_binary(uv_site: Path) -> Path:
    binary = uv_site / "bin" / "uv"
    try:
        resolved_root = uv_site.resolve(strict=True)
        resolved_binary = binary.resolve(strict=True)
        resolved_binary.relative_to(resolved_root)
    except (OSError, ValueError):
        raise WorkerError("uv_bootstrap_binary_path_invalid") from None
    if (
        uv_site.is_symlink()
        or binary.is_symlink()
        or not resolved_binary.is_file()
        or not os.access(resolved_binary, os.X_OK)
        or sha256_file(resolved_binary) != UV_BINARY_SHA256
    ):
        raise WorkerError("uv_bootstrap_binary_identity_invalid")
    return resolved_binary


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def parse_pinned_requirements(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s]+) \\$", line)
            if match:
                name = _normalized_distribution_name(match.group(1))
                if name in versions:
                    raise WorkerError("runtime_requirements_duplicate_package")
                versions[name] = match.group(2)
    except (OSError, UnicodeError):
        raise WorkerError("runtime_requirements_unreadable") from None
    if not versions:
        raise WorkerError("runtime_requirements_empty")
    return versions


def requirements_without_nvidia(path: Path, output: Path) -> set[str]:
    """Filter only verified host CUDA library wheels from the hashed lock."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise WorkerError("runtime_requirements_unreadable") from None
    header = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+) \\$", re.MULTILINE)
    matches = list(header.finditer(source))
    if not matches:
        raise WorkerError("runtime_requirements_empty")
    output_parts = [source[: matches[0].start()]]
    removed: set[str] = set()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        block = source[match.start() : end]
        name = _normalized_distribution_name(match.group(1))
        if name.startswith("nvidia-"):
            removed.add(name)
        else:
            output_parts.append(block)
    if len(removed) != 15:
        raise WorkerError("runtime_nvidia_lock_inventory_invalid")
    try:
        output.write_text("".join(output_parts), encoding="utf-8")
    except OSError:
        raise WorkerError("runtime_filtered_lock_write_failed") from None
    return removed


def _elf_is_x86_64(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(20)
    except OSError:
        return False
    if len(header) < 20 or header[:4] != b"\x7fELF" or header[4] != 2:
        return False
    endian = "<" if header[5] == 1 else ">" if header[5] == 2 else ""
    if not endian:
        return False
    return struct.unpack_from(f"{endian}H", header, 18)[0] == 62


def _runtime_regular_bytes(*roots: Path) -> int:
    """Count owned regular files, excluding deliberate links to shared CUDA files."""
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name for name in directories if not (current_path / name).is_symlink()
            ]
            for name in files:
                path = current_path / name
                if path.is_symlink():
                    continue
                try:
                    if path.is_file():
                        total += path.stat().st_size
                except OSError:
                    raise WorkerError("runtime_storage_inventory_failed") from None
    return total


def verified_managed_python311(
    interpreter_link: Path, managed_python_root: Path, site_packages: Path
) -> Path:
    try:
        resolved = interpreter_link.resolve(strict=True)
        resolved.relative_to(managed_python_root.resolve(strict=True))
    except (OSError, ValueError):
        raise WorkerError("python311_venv_interpreter_not_owned") from None
    if (
        not interpreter_link.is_symlink()
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
        or not site_packages.is_dir()
    ):
        raise WorkerError("python311_venv_inventory_invalid")
    return resolved


def verify_runtime_lock_files(repo: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    configuration = plan.get("configuration")
    if not isinstance(configuration, dict):
        raise WorkerError("frozen_fim_runtime_binding_missing")
    binding = configuration.get("runtime_lock")
    runtime = configuration.get("runtime")
    if (
        not isinstance(binding, dict)
        or not isinstance(runtime, dict)
        or runtime != EXPECTED_RUNTIME
        or binding.get("bootstrap_uv_version") != "0.12.3"
        or binding.get("bootstrap_uv_wheel_sha256") != UV_WHEEL_SHA256
        or not _is_sha256(binding.get("runtime_lock_sha256"))
        or not _is_sha256(binding.get("requirements_lock_sha256"))
    ):
        raise WorkerError("frozen_fim_runtime_binding_invalid")
    report_path = repo / RUNTIME_LOCK_RELATIVE
    if report_path.is_symlink() or not report_path.is_file():
        raise WorkerError("runtime_lock_report_missing")
    if sha256_file(report_path) != binding["runtime_lock_sha256"]:
        raise WorkerError("runtime_lock_report_hash_mismatch")
    report = _read_json(report_path, "runtime_lock_report_invalid")
    if (
        report.get("schema") != "q25-fim-python-runtime-lock-v1"
        or report.get("expected_versions") != EXPECTED_RUNTIME
        or report.get("bootstrap_uv_version") != "0.12.3"
        or report.get("bootstrap_uv_wheel_sha256") != UV_WHEEL_SHA256
        or report.get("bootstrap_uv_wheel_filename") != UV_WHEEL_FILENAME
        or report.get("bootstrap_uv_wheel_bytes") != UV_WHEEL_BYTES
    ):
        raise WorkerError("runtime_lock_report_identity_invalid")
    bootstrap_policy = report.get("bootstrap_policy")
    if (
        not isinstance(bootstrap_policy, dict)
        or bootstrap_policy.get("runtime_root") != str(RUNTIME_ROOT)
        or bootstrap_policy.get("deadline_includes_setup") is not True
        or bootstrap_policy.get("maximum_setup_seconds") != 1_800
        or bootstrap_policy.get("minimum_finalization_reserve_seconds") != 1_800
        or bootstrap_policy.get("pip_uv_cache") is not False
        or bootstrap_policy.get("model_weight_downloads") is not False
        or bootstrap_policy.get("fallback_to_host_python") is not False
        or bootstrap_policy.get("bytecode_writes") is not False
    ):
        raise WorkerError("runtime_lock_bootstrap_policy_invalid")
    requirements = report.get("requirements_lock")
    if not isinstance(requirements, dict):
        raise WorkerError("runtime_requirements_lock_identity_missing")
    relative = Path(str(requirements.get("repo_relative_path", "")))
    if (
        relative != RUNTIME_REQUIREMENTS_RELATIVE
        or relative.is_absolute()
        or ".." in relative.parts
        or requirements.get("sha256") != binding["requirements_lock_sha256"]
        or isinstance(requirements.get("bytes"), bool)
        or not isinstance(requirements.get("bytes"), int)
        or requirements.get("package_count") != 71
    ):
        raise WorkerError("runtime_requirements_lock_identity_invalid")
    lock_path = repo / relative
    if (
        lock_path.is_symlink()
        or not lock_path.is_file()
        or lock_path.stat().st_size != requirements["bytes"]
        or sha256_file(lock_path) != binding["requirements_lock_sha256"]
    ):
        raise WorkerError("runtime_requirements_lock_hash_mismatch")
    versions = parse_pinned_requirements(lock_path)
    if (
        len(versions) != requirements["package_count"]
        or versions.get("torch") != EXPECTED_RUNTIME["torch"]
    ):
        raise WorkerError("runtime_requirements_lock_package_count_mismatch")
    source_requirements = report.get("source_requirements")
    if not isinstance(source_requirements, dict):
        raise WorkerError("runtime_source_requirements_identity_missing")
    source_relative = Path(str(source_requirements.get("repo_relative_path", "")))
    if (
        source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative != Path("reports/research/q25_code_cpt_r2/fim_runtime_requirements.in")
        or not _is_sha256(source_requirements.get("sha256"))
        or isinstance(source_requirements.get("bytes"), bool)
        or not isinstance(source_requirements.get("bytes"), int)
    ):
        raise WorkerError("runtime_source_requirements_identity_invalid")
    source_path = repo / source_relative
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or source_path.stat().st_size != source_requirements["bytes"]
        or sha256_file(source_path) != source_requirements["sha256"]
    ):
        raise WorkerError("runtime_source_requirements_hash_mismatch")
    return report, lock_path


def _distribution_name(distribution: importlib.metadata.Distribution) -> str | None:
    try:
        name = distribution.metadata["Name"]
    except KeyError:
        return None
    return _normalized_distribution_name(name) if isinstance(name, str) else None


def link_verified_nvidia_libraries(
    requirements_path: Path,
    host_site: Path,
    venv_site: Path | None,
    inventory_path: Path,
    *,
    host_python: str,
) -> dict[str, Any]:
    """Link exact lock-matching NVIDIA shared-library wheels without copying them."""
    locked = parse_pinned_requirements(requirements_path)
    expected = {name: version for name, version in locked.items() if name.startswith("nvidia-")}
    inventory: dict[str, Any] = {
        "schema": "q25-fim-nvidia-runtime-inventory-v1",
        "host_python": host_python,
        "host_architecture": platform.machine(),
        "host_glibc": platform.libc_ver()[1] if platform.libc_ver()[0] == "glibc" else None,
        "expected_distribution_count": len(expected),
        "shared_library_bytes_reused": 0,
        "distributions": [],
        "status": "validating",
    }
    links: dict[Path, Path] = {}
    reused_realpaths: set[Path] = set()
    namespace_init: Path | None = None
    namespace_init_sha256: str | None = None
    error_reason: str | None = None
    if (
        len(expected) != 15
        or not host_site.is_dir()
        or (venv_site is not None and not venv_site.is_dir())
    ):
        error_reason = "runtime_nvidia_host_inventory_unavailable"
    host_distributions = list(importlib.metadata.distributions(path=[str(host_site)]))
    by_name: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in host_distributions:
        name = _distribution_name(distribution)
        if name is not None:
            by_name.setdefault(name, []).append(distribution)

    inventory["host_distribution_count"] = len(host_distributions)
    inventory["host_nvidia_distribution_count"] = sum(
        name.startswith("nvidia-") for name in by_name
    )
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        error_reason = error_reason or "runtime_nvidia_host_architecture_mismatch"

    for name, expected_version in sorted(expected.items()):
        candidates = by_name.get(name, [])
        actual_version = candidates[0].version if len(candidates) == 1 else None
        record: dict[str, Any] = {
            "name": name,
            "expected_version": expected_version,
            "actual_version": actual_version,
            "distribution_count": len(candidates),
            "shared_library_bytes": 0,
            "x86_64_elf_verified": False,
            "status": "unverified",
        }
        inventory["distributions"].append(record)
        if len(candidates) != 1 or actual_version != expected_version:
            record["status"] = "version_or_inventory_mismatch"
            error_reason = error_reason or "runtime_nvidia_host_version_mismatch"
            continue
        distribution = candidates[0]
        files = distribution.files
        if not files:
            record["status"] = "installed_file_inventory_missing"
            error_reason = error_reason or "runtime_nvidia_host_files_missing"
            continue
        components: set[str] = set()
        dist_info_names: set[str] = set()
        package_root = host_site / "nvidia"
        file_paths: list[Path] = []
        item_invalid = False
        for item in files:
            relative = Path(str(item))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                item_invalid = True
                break
            if "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            located = Path(str(distribution.locate_file(item)))
            try:
                resolved = located.resolve(strict=True)
                resolved.relative_to(host_site.resolve(strict=True))
            except (OSError, ValueError):
                item_invalid = True
                break
            first = relative.parts[0]
            if relative == Path("nvidia/__init__.py"):
                if namespace_init is not None and sha256_file(resolved) != namespace_init_sha256:
                    item_invalid = True
                    break
                namespace_init = resolved
                namespace_init_sha256 = sha256_file(resolved)
            elif first == "nvidia" and len(relative.parts) >= 2:
                component = relative.parts[1]
                components.add(component)
                try:
                    resolved.relative_to(package_root.resolve(strict=True))
                except (OSError, ValueError):
                    item_invalid = True
                    break
            elif first.endswith(".dist-info"):
                dist_info_names.add(first)
            else:
                item_invalid = True
                break
            if not located.is_file():
                item_invalid = True
                break
            if ".so" in relative.name:
                is_nvshmem_plugin = (
                    name == "nvidia-nvshmem-cu12"
                    and relative.parts[:3] == ("nvidia", "nvshmem", "lib")
                    and relative.name.startswith(("nvshmem_bootstrap_", "nvshmem_transport_"))
                )
                if (
                    ".cpython-" in relative.name
                    or ".abi3" in relative.name
                    or not (relative.name.startswith("lib") or is_nvshmem_plugin)
                    or not _elf_is_x86_64(resolved)
                ):
                    record["offending_relative_file"] = relative.as_posix()[:200]
                    item_invalid = True
                    break
                reused_realpaths.add(resolved)
            elif relative.suffix == ".pyd":
                item_invalid = True
                break
            file_paths.append(resolved)
            if venv_site is not None:
                destination = venv_site / relative
                previous_source = links.get(destination)
                if previous_source is not None and previous_source != resolved:
                    item_invalid = True
                    break
                links[destination] = resolved
        if item_invalid or not components or len(dist_info_names) != 1:
            record["status"] = "platform_file_inventory_mismatch"
            error_reason = error_reason or "runtime_nvidia_host_platform_inventory_mismatch"
            continue
        record["shared_library_bytes"] = sum(
            path.stat().st_size
            for path in file_paths
            if ".so" in path.name and path in reused_realpaths
        )
        record["x86_64_elf_verified"] = True
        record["status"] = "verified"
        dist_info_source = host_site / next(iter(dist_info_names))
        if dist_info_source.is_symlink() or not dist_info_source.is_dir():
            record["status"] = "unsafe_or_ambiguous_link_target"
            error_reason = error_reason or "runtime_nvidia_link_target_unsafe"

    inventory["shared_library_bytes_reused"] = sum(path.stat().st_size for path in reused_realpaths)
    if error_reason is None and all(
        record["status"] == "verified" for record in inventory["distributions"]
    ):
        if venv_site is not None:
            try:
                for destination, source in sorted(links.items(), key=lambda item: str(item[0])):
                    if destination.exists() or destination.is_symlink():
                        raise OSError("NVIDIA wheel target already exists")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(source)
            except OSError:
                error_reason = "runtime_nvidia_symlink_creation_failed"
    if error_reason is None:
        inventory["status"] = "verified_and_linked" if venv_site is not None else "verified"
    else:
        inventory["status"] = "rejected"
        inventory["failure_reason"] = error_reason
    _write_json(inventory_path, inventory)
    if error_reason is not None:
        raise WorkerError(error_reason)
    return inventory


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise WorkerError(reason) from None
    if not isinstance(value, dict):
        raise WorkerError(reason)
    return value


def validate_session(session: Any) -> dict[str, Any]:
    if not isinstance(session, dict):
        raise WorkerError("session_invalid")
    required = {
        "commit",
        "plan_sha256",
        "input_manifest_sha256",
        "arm",
        "attempt",
        "session_seconds",
        "resume_source",
        "external_campaign_tokens",
    }
    if not required.issubset(session):
        raise WorkerError("session_missing_required_fields")
    commit = session["commit"]
    seconds = session["session_seconds"]
    external_tokens = session["external_campaign_tokens"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit.lower())
        or not _is_sha256(session["plan_sha256"])
        or not _is_sha256(session["input_manifest_sha256"])
        or session["arm"] not in ARMS
        or isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or not MINIMUM_FINAL_RESERVE_SECONDS < seconds <= MAX_SESSION_SECONDS
        or isinstance(external_tokens, bool)
        or not isinstance(external_tokens, int)
        or not 0 <= external_tokens <= MAX_CAMPAIGN_TOKENS
    ):
        raise WorkerError("session_identity_or_budget_invalid")
    scale_variant = session.get("scale_variant")
    if scale_variant is not None and (
        scale_variant not in SCALE_VARIANTS or session["arm"] != TRAIN_ARM
    ):
        raise WorkerError("scale_session_identity_invalid")
    if scale_variant is not None and external_tokens > SCALE_MAX_CAMPAIGN_TOKENS:
        raise WorkerError("scale_session_token_carry_invalid")
    attempt = session["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise WorkerError("session_identity_or_budget_invalid")
    resume_source = session["resume_source"]
    resume_identity = session.get("resume_identity")
    if resume_source is None:
        if resume_identity is not None:
            raise WorkerError("resume_identity_without_source")
    else:
        if not isinstance(resume_source, str) or not resume_source.strip():
            raise WorkerError("resume_source_invalid")
        if not isinstance(resume_identity, dict):
            raise WorkerError("resume_identity_missing")
        cursor = resume_identity.get("cursor")
        if (
            not _is_sha256(resume_identity.get("fingerprint"))
            or not _is_sha256(resume_identity.get("checkpoint_sha256"))
            or not isinstance(cursor, dict)
            or any(
                isinstance(cursor.get(field), bool)
                or not isinstance(cursor.get(field), int)
                or cursor[field] < 0
                for field in CHECKPOINT_CURSOR_FIELDS
            )
        ):
            raise WorkerError("resume_identity_invalid")
    return session


def _safe_input_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
        raise WorkerError("input_manifest_path_invalid")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise WorkerError("input_manifest_file_missing_or_unsafe")
    return path


def find_input_manifest(input_root: Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    try:
        matches = [
            path
            for path in input_root.rglob("input-manifest.json")
            if path.is_file() and not path.is_symlink() and sha256_file(path) == expected_sha256
        ]
    except OSError:
        raise WorkerError("input_manifest_search_failed") from None
    if len(matches) != 1:
        raise WorkerError("frozen_input_bundle_missing_or_ambiguous")
    manifest_path = matches[0]
    manifest = _read_json(manifest_path, "input_manifest_invalid")
    return manifest_path.parent, manifest


def verify_input_bundle(
    root: Path, manifest: dict[str, Any], session: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) not in (
        REQUIRED_INPUT_FILES,
        SCALE_REQUIRED_INPUT_FILES,
    ):
        raise WorkerError("input_manifest_file_set_invalid")
    scale_bundle = set(files) == SCALE_REQUIRED_INPUT_FILES
    paths: dict[str, Path] = {}
    for name, identity in files.items():
        if not isinstance(identity, dict):
            raise WorkerError("input_manifest_file_identity_invalid")
        path = _safe_input_file(root, name)
        expected_bytes = identity.get("bytes")
        expected_sha = identity.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not _is_sha256(expected_sha)
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
        ):
            raise WorkerError("input_file_hash_or_size_mismatch")
        paths[name] = path

    allowed = set(files) | ALLOWED_INPUT_EXTRAS
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WorkerError("input_bundle_contains_symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in allowed:
            raise WorkerError("input_bundle_contains_unlisted_file")

    if sha256_file(paths["plan.json"]) != session["plan_sha256"]:
        raise WorkerError("frozen_plan_hash_mismatch")
    plan = _read_json(paths["plan.json"], "frozen_plan_invalid")
    verify_plan(plan, session, files)
    metadata = _read_json(paths["corpus_metadata.json"], "fim_corpus_metadata_invalid")
    if (
        metadata.get("raw_source_content_emitted") is not False
        or metadata.get("sealed_test_accessed") is not False
        or metadata.get("preparation_plan_sha256") != plan.get("preparation_plan_sha256")
        or metadata.get("parent_cpt_plan_sha256") != plan.get("parent_cpt_plan_sha256")
        or metadata.get("tokenizer_id") != "Qwen/Qwen2.5-Coder-0.5B"
        or (
            scale_bundle
            and metadata.get("schema") != "q25-completion-scale-corpus-v1"
        )
    ):
        raise WorkerError("fim_corpus_metadata_policy_mismatch")
    if scale_bundle != (plan.get("schema") == SCALE_PLAN_SCHEMA):
        raise WorkerError("frozen_scale_input_plan_mismatch")
    return paths, plan


def verify_scale_plan(plan: dict[str, Any], session: dict[str, Any], files: dict[str, Any]) -> None:
    if (
        plan.get("schema") != SCALE_PLAN_SCHEMA
        or plan.get("branch") != SCALE_BRANCH
        or plan.get("gpu_execution_authorized") is not True
        or plan.get("scale_variant") != session.get("scale_variant")
        or session.get("arm") != TRAIN_ARM
        or plan.get("base_commit") != session.get("commit")
    ):
        raise WorkerError("frozen_scale_plan_identity_invalid")
    configuration = plan.get("configuration")
    if not isinstance(configuration, dict):
        raise WorkerError("frozen_scale_plan_incomplete")
    budget = configuration.get("budget")
    training = configuration.get("training")
    runtime = configuration.get("runtime")
    runtime_lock = configuration.get("runtime_lock")
    data = plan.get("data")
    experiment = plan.get("experiment")
    initializers = plan.get("initializers")
    evaluation = plan.get("evaluation")
    if not all(
        isinstance(value, dict)
        for value in (budget, training, runtime_lock, data, experiment, initializers, evaluation)
    ):
        raise WorkerError("frozen_scale_plan_incomplete")
    assert isinstance(budget, dict)
    assert isinstance(training, dict)
    assert isinstance(data, dict)
    assert isinstance(runtime_lock, dict)
    assert isinstance(initializers, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(experiment, dict)
    expected_training = {
        "sequence_length": 1024,
        "effective_batch": 16,
        "microbatch_examples": 1,
        "checkpoint_every_updates": 64,
        "seed": 314159,
        "attention": "sdpa",
        "gradient_checkpointing": True,
        "master_weights": "fp32",
        "compute": "fp16",
        "optimizer": "AdamW8bit",
        "objective": "example_mean_response_only_FIM_target_and_EOS",
        "learning_rate": 1e-5,
        "warmup_fraction": 0.03,
        "cosine_floor_fraction": 0.1,
        "gradient_clip": 1.0,
        "weight_decay": 0.01,
        "initial_loss_scale": 128,
        "max_input_tokens": SCALE_MAX_ARM_TOKENS,
    }
    max_campaign = budget.get("maximum_campaign_input_tokens")
    aggregate_seconds = budget.get("aggregate_session_seconds")
    account_gpu_hours = budget.get("conservative_account_gpu_hours")
    quota_multiplier = budget.get("conservative_quota_multiplier")
    session_limit = budget.get("session_seconds")
    storage_cap = budget.get("new_artifact_bytes_cap")
    input_cap = training.get("max_input_tokens")
    if (
        any(training.get(key) != value for key, value in expected_training.items())
        or "epochs" in training
        or isinstance(max_campaign, bool)
        or not isinstance(max_campaign, int)
        or not 1 <= max_campaign <= SCALE_MAX_CAMPAIGN_TOKENS
        or isinstance(session_limit, bool)
        or not isinstance(session_limit, int)
        or not MINIMUM_FINAL_RESERVE_SECONDS < session["session_seconds"] <= session_limit
        or session_limit > MAX_SESSION_SECONDS
        or isinstance(aggregate_seconds, bool)
        or not isinstance(aggregate_seconds, int)
        or not 1 <= aggregate_seconds <= 21_600
        or isinstance(account_gpu_hours, bool)
        or not isinstance(account_gpu_hours, int)
        or not 1 <= account_gpu_hours <= 12
        or quota_multiplier != 2
        or isinstance(storage_cap, bool)
        or not isinstance(storage_cap, int)
        or not 1 <= storage_cap <= MAX_ARTIFACT_BYTES
        or isinstance(input_cap, bool)
        or not isinstance(input_cap, int)
        or not 1 <= input_cap <= SCALE_MAX_ARM_TOKENS
        or budget.get("paid_compute") is not False
        or budget.get("automatic_renewal_use") is not False
        or budget.get("minimum_finalization_reserve_seconds", 0)
        < MINIMUM_FINAL_RESERVE_SECONDS
        or budget.get("minimum_free_bytes", 0) < MINIMUM_FREE_BYTES
        or budget.get("runtime_setup_reserve_seconds") != 1_800
        or runtime != EXPECTED_RUNTIME
        or evaluation.get("attention_backend")
        != "torch-efficient-sdpa-explicit-kv-repeat-v1"
        or not _is_sha256(runtime_lock.get("runtime_lock_sha256"))
        or not _is_sha256(runtime_lock.get("requirements_lock_sha256"))
        or runtime_lock.get("bootstrap_uv_version") != "0.12.3"
        or runtime_lock.get("bootstrap_uv_wheel_sha256") != UV_WHEEL_SHA256
        or session["external_campaign_tokens"] > max_campaign
    ):
        raise WorkerError("frozen_scale_budget_or_training_invalid")
    variant_records = experiment.get("variants")
    selected_variant = session["scale_variant"]
    selected_record = (
        variant_records.get(selected_variant) if isinstance(variant_records, dict) else None
    )
    expected_variant = (
        {"distinct_states": 4096, "epochs": 2, "example_exposures": 8192}
        if selected_variant == "repeat"
        else {"distinct_states": 8192, "epochs": 1, "example_exposures": 8192}
    )
    if not isinstance(selected_record, dict) or any(
        selected_record.get(key) != value for key, value in expected_variant.items()
    ):
        raise WorkerError("frozen_scale_variant_counts_invalid")
    initializer = initializers.get(TRAIN_ARM)
    if (
        not isinstance(initializer, dict)
        or initializer.get("kind", initializer.get("initializer")) != "untouched_pretrained"
        or initializer.get("model_id") != "Qwen/Qwen2.5-Coder-0.5B"
        or initializer.get("revision") != "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
    ):
        raise WorkerError("frozen_scale_initializer_invalid")
    if not _is_sha256(data.get("corpus_metadata_sha256")) or data.get(
        "corpus_metadata_sha256"
    ) != files["corpus_metadata.json"].get("sha256"):
        raise WorkerError("frozen_scale_metadata_identity_mismatch")
    if (
        data.get("previous_training_sha256")
        != "341f2d54da2d3c64299c18a918049ded75235ed375137012df71a9ce737fd690"
        or data.get("previous_development_sha256")
        != "43c56d113a819256c7e175ef1f863b9f622a8119f4d3d01b24d455a010fed4ac"
    ):
        raise WorkerError("frozen_scale_previous_data_identity_mismatch")
    for split, filename, row_count in (
        ("repeat_train", "repeat_train.jsonl", 4096),
        ("scaled_train", "scaled_train.jsonl", 8192),
        ("development_new", "development_new.jsonl", 512),
        ("development_previous", "development_previous.jsonl", 240),
    ):
        record = data.get(split)
        file_identity = files.get(filename)
        if (
            not isinstance(record, dict)
            or not isinstance(file_identity, dict)
            or record.get("sha256") != file_identity.get("sha256")
            or record.get("bytes") != file_identity.get("bytes")
            or record.get("row_count") != row_count
            or any(
                isinstance(record.get(key), bool)
                or not isinstance(record.get(key), int)
                or record[key] < 0
                for key in ("input_tokens", "target_tokens")
            )
        ):
            raise WorkerError("frozen_scale_split_identity_mismatch")
    fixtures = evaluation.get("fixtures")
    if not isinstance(fixtures, dict):
        raise WorkerError("frozen_scale_fixtures_missing")
    for key, filename in (("causal", "causal200.jsonl"), ("line", "line180.jsonl")):
        record = fixtures.get(key)
        if not isinstance(record, dict) or record.get("sha256") != files[filename].get("sha256"):
            raise WorkerError("frozen_scale_fixture_identity_mismatch")


def verify_plan(plan: dict[str, Any], session: dict[str, Any], files: dict[str, Any]) -> None:
    if plan.get("schema") == SCALE_PLAN_SCHEMA:
        verify_scale_plan(plan, session, files)
        return
    if session.get("scale_variant") is not None:
        raise WorkerError("scale_session_requires_scale_plan")
    if plan.get("schema") != "q25-fim-training-plan-v1":
        raise WorkerError("frozen_fim_plan_schema_invalid")
    if plan.get("gpu_execution_authorized") is not True:
        raise WorkerError("frozen_fim_plan_not_authorized")
    configuration = plan.get("configuration")
    if not isinstance(configuration, dict):
        raise WorkerError("frozen_fim_plan_incomplete")
    budget = configuration.get("budget")
    training = configuration.get("training")
    data = plan.get("data")
    initializers = plan.get("initializers")
    evaluation = plan.get("evaluation")
    if not all(
        isinstance(value, dict) for value in (budget, training, data, initializers, evaluation)
    ):
        raise WorkerError("frozen_fim_plan_incomplete")
    assert isinstance(budget, dict)
    assert isinstance(training, dict)
    assert isinstance(data, dict)
    assert isinstance(initializers, dict)
    assert isinstance(evaluation, dict)
    max_campaign = budget.get(
        "maximum_campaign_input_tokens", budget.get("maximum_additional_training_input_tokens")
    )
    session_limit = budget.get("session_seconds")
    runtime_setup_reserve = budget.get("runtime_setup_reserve_seconds")
    storage_cap = budget.get("new_artifact_bytes_cap")
    arm_limit = training.get("max_input_tokens")
    runtime = configuration.get("runtime")
    runtime_lock = configuration.get("runtime_lock")
    attention_backend = evaluation.get("attention_backend")
    if (
        isinstance(max_campaign, bool)
        or not isinstance(max_campaign, int)
        or not 1 <= max_campaign <= MAX_CAMPAIGN_TOKENS
        or isinstance(session_limit, bool)
        or not isinstance(session_limit, int)
        or not MINIMUM_FINAL_RESERVE_SECONDS < session["session_seconds"] <= session_limit
        or isinstance(storage_cap, bool)
        or not isinstance(storage_cap, int)
        or not 1 <= storage_cap <= MAX_ARTIFACT_BYTES
        or isinstance(arm_limit, bool)
        or not isinstance(arm_limit, int)
        or not 1 <= arm_limit <= MAX_ARM_TOKENS
        or budget.get("paid_compute") is not False
        or budget.get("automatic_renewal_use") is not False
        or runtime_setup_reserve != 1_800
        or runtime != EXPECTED_RUNTIME
        or not isinstance(runtime_lock, dict)
        or attention_backend != "torch-efficient-sdpa-explicit-kv-repeat-v1"
        or not _is_sha256(runtime_lock.get("runtime_lock_sha256"))
        or not _is_sha256(runtime_lock.get("requirements_lock_sha256"))
        or runtime_lock.get("bootstrap_uv_version") != "0.12.3"
        or runtime_lock.get("bootstrap_uv_wheel_sha256") != UV_WHEEL_SHA256
        or session["external_campaign_tokens"] > max_campaign
    ):
        raise WorkerError("frozen_fim_budget_invalid")
    if not isinstance(initializers.get(session["arm"]), dict):
        raise WorkerError("selected_fim_initializer_missing")
    data_metadata = data.get("corpus_metadata_sha256")
    if not _is_sha256(data_metadata) or data_metadata != files["corpus_metadata.json"]["sha256"]:
        raise WorkerError("frozen_fim_metadata_identity_mismatch")
    for split, filename in (("train", "train.jsonl"), ("development", "development.jsonl")):
        record = data.get(split)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != files[filename]["sha256"]
            or record.get("bytes") != files[filename]["bytes"]
        ):
            raise WorkerError("frozen_fim_split_identity_mismatch")
    fixtures = evaluation.get("fixtures")
    if not isinstance(fixtures, dict):
        raise WorkerError("frozen_fim_fixtures_missing")
    for key, filename in (("causal", "causal200.jsonl"), ("line", "line180.jsonl")):
        record = fixtures.get(key)
        if not isinstance(record, dict) or record.get("sha256") != files[filename]["sha256"]:
            raise WorkerError("frozen_fim_fixture_identity_mismatch")


def find_initializer_path(input_root: Path, entry: dict[str, Any]) -> Path:
    kind = entry.get("kind")
    if kind == "untouched_pretrained":
        file_records = entry.get("files")
        weight_record = (
            file_records.get("model.safetensors") if isinstance(file_records, dict) else None
        )
        expected_sha: Any
        expected_bytes: Any
        if isinstance(weight_record, str):
            expected_sha = weight_record
            expected_bytes = None
        elif isinstance(weight_record, dict):
            expected_sha = weight_record.get("sha256")
            expected_bytes = weight_record.get("bytes")
        else:
            raise WorkerError("base_initializer_weight_identity_missing")
        if not _is_sha256(expected_sha):
            raise WorkerError("base_initializer_weight_hash_invalid")
        matches: list[Path] = []
        for path in input_root.rglob("model.safetensors"):
            if path.is_symlink() or not path.is_file():
                continue
            if expected_bytes is not None and path.stat().st_size != expected_bytes:
                continue
            if sha256_file(path) == expected_sha:
                matches.append(path.parent)
        if len(matches) != 1:
            raise WorkerError("base_initializer_missing_or_ambiguous")
        return matches[0]

    expected_manifest_sha = entry.get("artifact_manifest_sha256")
    if kind != "completed_cpt_export" or not _is_sha256(expected_manifest_sha):
        raise WorkerError("cpt_initializer_identity_invalid")
    matches = []
    for path in input_root.rglob("artifact_manifest.json"):
        if path.is_symlink() or not path.is_file():
            continue
        if sha256_file(path) == expected_manifest_sha:
            matches.append(path.parent)
    if len(matches) != 1:
        raise WorkerError("cpt_initializer_missing_or_ambiguous")
    return matches[0]


def _tree_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WorkerError("artifact_tree_contains_symlink")
        if path.is_file():
            paths.append(path)
    return paths


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in _tree_files(root))


def mounted_artifact_bytes(
    input_root: Path,
    *,
    initializer: Path,
    trainer_files: list[Path],
    resume_checkpoint: Path | None,
) -> int:
    counted = {path.resolve() for path in trainer_files}
    counted.update(path.resolve() for path in _tree_files(initializer))
    if resume_checkpoint is not None:
        marker = resume_checkpoint.with_suffix(resume_checkpoint.suffix + ".complete.json")
        counted.add(resume_checkpoint.resolve())
        counted.add(marker.resolve())
    total = 0
    for path in _tree_files(input_root):
        if path.resolve() not in counted:
            total += path.stat().st_size
    return total


def _verify_baseline(
    previous_root: Path,
    *,
    arm: str,
    plan_sha256: str,
    input_manifest_sha256: str,
    initializer_sha256: str,
) -> dict[str, Any]:
    if previous_root.is_symlink() or not previous_root.is_dir():
        raise WorkerError("resume_baseline_root_unsafe")
    baseline_dir = previous_root / "baseline"
    identity_path = previous_root / "baseline" / "identity.json"
    if baseline_dir.is_symlink() or identity_path.is_symlink() or not identity_path.is_file():
        raise WorkerError("resume_baseline_identity_missing")
    identity = _read_json(identity_path, "resume_baseline_identity_missing")
    if (
        identity.get("schema") != "q25-fim-baseline-v1"
        or identity.get("arm") != arm
        or identity.get("plan_sha256") != plan_sha256
        or identity.get("input_manifest_sha256") != input_manifest_sha256
        or identity.get("initializer_identity_sha256") != initializer_sha256
    ):
        raise WorkerError("resume_baseline_identity_mismatch")
    records = identity.get("files")
    if not isinstance(records, dict) or not records or not _has_baseline_modes(records):
        raise WorkerError("resume_baseline_files_missing")
    discovered: set[str] = set()
    before_root = previous_root / "before"
    if before_root.is_symlink() or not before_root.is_dir():
        raise WorkerError("resume_baseline_inventory_mismatch")
    for path in _tree_files(before_root):
        discovered.add(path.relative_to(previous_root).as_posix())
    if discovered != set(records):
        raise WorkerError("resume_baseline_inventory_mismatch")
    for name, record in records.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, dict)
            or not _is_sha256(record.get("sha256"))
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
        ):
            raise WorkerError("resume_baseline_file_identity_invalid")
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "before"
        ):
            raise WorkerError("resume_baseline_file_path_invalid")
        path = previous_root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
            raise WorkerError("resume_baseline_file_missing_or_changed")
        if sha256_file(path) != record["sha256"]:
            raise WorkerError("resume_baseline_file_missing_or_changed")
    return identity


def _has_baseline_modes(records: dict[str, Any]) -> bool:
    names = [name for name in records if isinstance(name, str)]
    return all(
        any(name.startswith(f"before/{mode}/") for name in names)
        for mode in ("development", "line")
    )


def resolve_verified_resume(
    input_root: Path,
    session: dict[str, Any],
    *,
    plan_sha256: str,
    input_manifest_sha256: str,
    initializer_sha256: str,
    arm: str,
) -> tuple[Path, Path, dict[str, Any]]:
    pointers = [
        path
        for path in input_root.rglob("latest.json")
        if path.is_file() and not path.is_symlink() and path.parent.name == "training"
    ]
    if len(pointers) != 1:
        raise WorkerError("resume_checkpoint_pointer_missing_or_ambiguous")
    pointer_path = pointers[0]
    pointer = _read_json(pointer_path, "resume_checkpoint_pointer_invalid")
    resume_identity = session["resume_identity"]
    checkpoint_name = pointer.get("path")
    if (
        pointer.get("schema") != "q25-fim-latest-checkpoint-v1"
        or not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
        or pointer.get("fingerprint") != resume_identity["fingerprint"]
        or pointer.get("sha256") != resume_identity["checkpoint_sha256"]
        or pointer.get("cursor") != resume_identity["cursor"]
    ):
        raise WorkerError("resume_checkpoint_identity_mismatch")
    checkpoint = pointer_path.parent / checkpoint_name
    marker_path = checkpoint.with_suffix(checkpoint.suffix + ".complete.json")
    if checkpoint.is_symlink() or marker_path.is_symlink() or not checkpoint.is_file():
        raise WorkerError("resume_checkpoint_missing_or_unsafe")
    marker = _read_json(marker_path, "resume_checkpoint_marker_missing")
    if (
        marker.get("fingerprint") != resume_identity["fingerprint"]
        or marker.get("sha256") != resume_identity["checkpoint_sha256"]
        or sha256_file(checkpoint) != marker.get("sha256")
    ):
        raise WorkerError("resume_checkpoint_content_mismatch")
    previous_root = pointer_path.parent.parent
    run_manifest = _read_json(
        previous_root / "training" / "run_manifest.json", "resume_run_manifest_missing"
    )
    identity = run_manifest.get("identity")
    corpus_identity = identity.get("corpus_metadata") if isinstance(identity, dict) else None
    training_identity = identity.get("training_data") if isinstance(identity, dict) else None
    if (
        run_manifest.get("fingerprint") != resume_identity["fingerprint"]
        or not isinstance(identity, dict)
        or identity.get("plan_sha256") != plan_sha256
        or identity.get("arm") != arm
        or canonical_sha256(identity.get("initializer")) != initializer_sha256
        or not isinstance(corpus_identity, dict)
        or not _is_sha256(corpus_identity.get("metadata_sha256"))
        or not isinstance(training_identity, dict)
        or not _is_sha256(training_identity.get("sha256"))
    ):
        raise WorkerError("resume_run_fingerprint_mismatch")
    baseline = _verify_baseline(
        previous_root,
        arm=arm,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        initializer_sha256=initializer_sha256,
    )
    return checkpoint, previous_root, baseline


def _baseline_inventory(output: Path) -> dict[str, dict[str, Any]]:
    before_root = output / "before"
    result: dict[str, dict[str, Any]] = {}
    for path in _tree_files(before_root):
        result[path.relative_to(output).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def trainer_command(
    *,
    python: Path,
    model: Path,
    paths: dict[str, Path],
    plan: Path,
    arm: str,
    output: Path,
    session_seconds: int,
    reserve_seconds: int,
    external_campaign_tokens: int,
    mounted_bytes: int,
    resume: Path | None,
    execute: bool,
    scale_variant: str | None = None,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "tinycomplete.code_cpt.q25_fim",
        "--model",
        str(model),
        "--train",
        str(paths["train.jsonl"]),
        "--development",
        str(paths["development.jsonl"]),
        "--data-metadata",
        str(paths["corpus_metadata.json"]),
        "--plan",
        str(plan),
        "--arm",
        arm,
        "--output",
        str(output),
        "--session-seconds",
        str(session_seconds),
        "--reserve-seconds",
        str(reserve_seconds),
        "--external-campaign-tokens",
        str(external_campaign_tokens),
        "--mounted-artifact-bytes",
        str(mounted_bytes),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if scale_variant is not None:
        command.extend(
            (
                "--scale-variant",
                scale_variant,
                "--repeat-train",
                str(paths["repeat_train.jsonl"]),
                "--scaled-train",
                str(paths["scaled_train.jsonl"]),
                "--historical-development",
                str(paths["development_previous.jsonl"]),
            )
        )
    if execute:
        command.append("--execute")
    return command


def _evaluation_command(
    *,
    python: Path,
    model: Path,
    input_path: Path,
    output: Path,
    plan: Path,
    alias: str,
    mode: str,
) -> list[str]:
    return [
        str(python),
        str(REPO / "scripts/evaluate_q25_fim.py"),
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--plan",
        str(plan),
        "--alias",
        alias,
        "--mode",
        mode,
    ]


def _attention_smoke_command(
    *,
    python: Path,
    model: Path,
    development: Path,
    line_suite: Path,
    output: Path,
    plan: Path,
) -> list[str]:
    return [
        str(python),
        str(REPO / "scripts/evaluate_q25_fim.py"),
        "--attention-smoke",
        "--model",
        str(model),
        "--input",
        str(development),
        "--line-input",
        str(line_suite),
        "--output",
        str(output),
        "--plan",
        str(plan),
        "--mode",
        "development",
    ]


def _regression_commands(
    *,
    python: Path,
    model: Path,
    paths: dict[str, Path],
    output: Path,
    plan_sha256: str,
    alias: str,
) -> list[tuple[str, list[str]]]:
    return [
        (
            "regression-causal",
            [
                str(python),
                str(REPO / "scripts/evaluate_q25_fim_regression.py"),
                "--kind",
                "causal",
                "--attention-backend",
                "torch-efficient-sdpa-explicit-kv-repeat-v1",
                "--model",
                str(model),
                "--suite",
                str(paths["causal200.jsonl"]),
                "--output",
                str(output / "regression" / "causal"),
                "--alias",
                alias,
                "--plan-sha",
                plan_sha256,
            ],
        ),
        (
            "regression-line",
            [
                str(python),
                str(REPO / "scripts/evaluate_q25_fim_regression.py"),
                "--kind",
                "line",
                "--attention-backend",
                "torch-efficient-sdpa-explicit-kv-repeat-v1",
                "--model",
                str(model),
                "--suite",
                str(paths["line180.jsonl"]),
                "--output",
                str(output / "regression" / "line"),
                "--alias",
                alias,
                "--plan-sha",
                plan_sha256,
            ],
        ),
    ]


class Worker:
    def __init__(self, session: dict[str, Any]) -> None:
        self.session = validate_session(session)
        self.started = time.monotonic()
        self.deadline = self.started + self.session["session_seconds"]
        self.status: dict[str, Any] = {
            "schema": (
                "q25-completion-scale-kaggle-worker-status-v1"
                if self.session.get("scale_variant") is not None
                else "q25-fim-kaggle-worker-status-v1"
            ),
            "state": "setup",
            "commit": self.session["commit"],
            "attempt": self.session["attempt"],
            "arm": self.session["arm"],
            **(
                {"scale_variant": self.session["scale_variant"]}
                if self.session.get("scale_variant") is not None
                else {}
            ),
            "plan_sha256": self.session["plan_sha256"],
            "input_manifest_sha256": self.session["input_manifest_sha256"],
            "training_started": False,
            "stages": [],
        }
        self.out = (
            WORK_ROOT / f"q25_completion_scale_r1-{self.session['scale_variant']}"
            if self.session.get("scale_variant") is not None
            else OUT
        )
        self.python311: Path | None = None
        self.runtime_lock: dict[str, Any] | None = None
        self.runtime_requirements: Path | None = None
        self.runtime_inventory: dict[str, Any] | None = None
        self.input_artifact_bytes = 0

    def _setup_stage(
        self,
        command: list[str],
        name: str,
        *,
        plan: dict[str, Any],
        env: dict[str, str] | None = None,
        timeout_seconds: float,
    ) -> int:
        setup_limit = int(plan["configuration"]["budget"]["runtime_setup_reserve_seconds"])
        setup_remaining = self.started + setup_limit - time.monotonic()
        if setup_remaining < 1:
            raise WorkerError("runtime_setup_deadline_reached")
        result = self.run_stage(
            command,
            name,
            env=env,
            timeout_seconds=min(timeout_seconds, setup_remaining),
            reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
        )
        self.check_storage(mounted_bytes=self.input_artifact_bytes, plan=plan)
        return result

    def _checkout_repository(self, plan: dict[str, Any]) -> None:
        if REPO.exists() and (REPO.is_symlink() or not (REPO / ".git").is_dir()):
            raise WorkerError("repository_checkout_path_not_clean")
        if not REPO.exists():
            branch = (
                SCALE_BRANCH
                if self.session.get("scale_variant") is not None
                else "research/q25-code-cpt-r2"
            )
            if (
                self._setup_stage(
                    [
                        "git",
                        "clone",
                        "--branch",
                        branch,
                        "https://github.com/Shlok-Bhakta/tabcomplete.git",
                        str(REPO),
                    ],
                    "checkout-repository",
                    plan=plan,
                    timeout_seconds=600,
                )
                != 0
            ):
                raise WorkerError("repository_checkout_failed")
        if (
            self._setup_stage(
                ["git", "-C", str(REPO), "checkout", self.session["commit"]],
                "pin-repository",
                plan=plan,
                timeout_seconds=120,
            )
            != 0
        ):
            raise WorkerError("repository_pin_failed")
        if (
            self._setup_stage(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                "verify-repository-commit",
                plan=plan,
                timeout_seconds=30,
            )
            != 0
        ):
            raise WorkerError("repository_commit_check_failed")
        revision_log = self.out / "logs" / "verify-repository-commit.log"
        try:
            revision = revision_log.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise WorkerError("repository_commit_check_failed") from None
        if revision != self.session["commit"]:
            raise WorkerError("repository_commit_mismatch")
        if self.session.get("scale_variant") is not None:
            source_identity = plan.get("source_identity")
            source_files = (
                source_identity.get("files") if isinstance(source_identity, dict) else None
            )
            if (
                not isinstance(source_identity, dict)
                or source_identity.get("commit") != revision
                or not isinstance(source_files, dict)
                or not source_files
            ):
                raise WorkerError("scale_repository_source_identity_invalid")
            for relative, record in source_files.items():
                path = Path(relative) if isinstance(relative, str) else Path("..")
                expected_sha = record.get("sha256") if isinstance(record, dict) else record
                expected_bytes = record.get("bytes") if isinstance(record, dict) else None
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or not _is_sha256(expected_sha)
                    or (
                        isinstance(record, dict)
                        and (
                            isinstance(expected_bytes, bool)
                            or not isinstance(expected_bytes, int)
                            or expected_bytes < 0
                        )
                    )
                ):
                    raise WorkerError("scale_repository_source_identity_invalid")
                source_path = REPO / path
                if (
                    any((REPO / Path(*path.parts[:index])).is_symlink()
                        for index in range(1, len(path.parts) + 1))
                    or not source_path.is_file()
                    or (expected_bytes is not None and source_path.stat().st_size != expected_bytes)
                    or sha256_file(source_path) != expected_sha
                ):
                    raise WorkerError("scale_repository_source_hash_mismatch")
        if self._runtime_artifact_bytes() < _runtime_regular_bytes(REPO):
            raise WorkerError("repository_storage_inventory_invalid")

    def _setup_environment(self) -> dict[str, str]:
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(RUNTIME_ROOT / "tmp"),
            "UV_NO_CACHE": "1",
            "UV_NO_PROGRESS": "1",
            "UV_PYTHON_INSTALL_DIR": str(RUNTIME_ROOT / "managed-python"),
        }
        environment.pop("PYTHONPATH", None)
        return environment

    def _download_uv_wheel(self, destination: Path, plan: dict[str, Any]) -> None:
        setup_limit = int(plan["configuration"]["budget"]["runtime_setup_reserve_seconds"])
        remaining = min(
            self.deadline - time.monotonic() - MINIMUM_FINAL_RESERVE_SECONDS,
            self.started + setup_limit - time.monotonic(),
        )
        if remaining < 1:
            raise WorkerError("runtime_setup_deadline_reached")
        digest = hashlib.sha256()
        received = 0
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(UV_WHEEL_URL, timeout=min(30, remaining)) as response:
                with temporary.open("xb") as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        if time.monotonic() >= self.started + setup_limit:
                            raise WorkerError("runtime_setup_deadline_reached")
                        received += len(chunk)
                        if received > UV_WHEEL_BYTES:
                            raise WorkerError("uv_bootstrap_wheel_size_mismatch")
                        digest.update(chunk)
                        handle.write(chunk)
            if received != UV_WHEEL_BYTES or digest.hexdigest() != UV_WHEEL_SHA256:
                raise WorkerError("uv_bootstrap_wheel_hash_mismatch")
            os.replace(temporary, destination)
        except WorkerError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise WorkerError("uv_bootstrap_download_failed") from None
        self.check_storage(mounted_bytes=self.input_artifact_bytes, plan=plan)

    def _prepare_python311(
        self, plan: dict[str, Any], runtime_report: dict[str, Any], requirements: Path
    ) -> None:
        if platform.machine().lower() not in {"x86_64", "amd64"}:
            raise WorkerError("runtime_platform_architecture_unsupported")
        libc_name, libc_version = platform.libc_ver()
        if libc_name != "glibc" or _version_tuple(libc_version) < (2, 35):
            raise WorkerError("runtime_platform_glibc_unsupported")
        if RUNTIME_ROOT.exists() and (
            RUNTIME_ROOT.is_symlink() or not RUNTIME_ROOT.is_dir() or any(RUNTIME_ROOT.iterdir())
        ):
            raise WorkerError("runtime_root_not_clean")
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        tmpdir = RUNTIME_ROOT / "tmp"
        tmpdir.mkdir(exist_ok=True)
        runtime_inventory_path = self.out / "runtime_inventory.json"
        host_site_value = sysconfig.get_paths().get("purelib")
        if not isinstance(host_site_value, str) or not host_site_value:
            raise WorkerError("runtime_host_site_packages_missing")
        host_site = Path(host_site_value)
        if not host_site.is_dir():
            raise WorkerError("runtime_host_site_packages_missing")
        try:
            self.runtime_inventory = link_verified_nvidia_libraries(
                requirements,
                host_site,
                None,
                runtime_inventory_path,
                host_python=platform.python_version(),
            )
        except WorkerError:
            raise
        self.status["runtime_inventory"] = {
            "status": self.runtime_inventory["status"],
            "expected_distribution_count": self.runtime_inventory["expected_distribution_count"],
            "shared_library_bytes_reused": self.runtime_inventory["shared_library_bytes_reused"],
        }
        self.save_status()
        self.check_storage(mounted_bytes=self.input_artifact_bytes, plan=plan)

        uv_site = RUNTIME_ROOT / "uv-site"
        uv_wheel = RUNTIME_ROOT / UV_WHEEL_FILENAME
        self._download_uv_wheel(uv_wheel, plan)
        if (
            self._setup_stage(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-input",
                    "--disable-pip-version-check",
                    "--no-warn-script-location",
                    "--no-cache-dir",
                    "--no-index",
                    "--no-deps",
                    "--target",
                    str(uv_site),
                    str(uv_wheel),
                ],
                "setup-uv-bootstrap",
                plan=plan,
                env=self._setup_environment(),
                timeout_seconds=180,
            )
            != 0
        ):
            raise WorkerError("uv_bootstrap_install_failed")
        uv_env = self._setup_environment()
        uv_binary = verified_installed_uv_binary(uv_site)
        uv = [str(uv_binary)]
        if (
            self._setup_stage(
                uv + ["--version"],
                "verify-uv-bootstrap",
                plan=plan,
                env=uv_env,
                timeout_seconds=30,
            )
            != 0
        ):
            raise WorkerError("uv_bootstrap_version_check_failed")
        try:
            uv_version_output = (
                (self.out / "logs" / "verify-uv-bootstrap.log").read_text(encoding="utf-8").strip()
            )
        except (OSError, UnicodeError):
            raise WorkerError("uv_bootstrap_version_check_failed") from None
        if uv_version_output != "uv 0.12.3 (x86_64-unknown-linux-gnu)":
            raise WorkerError("uv_bootstrap_version_mismatch")

        if (
            self._setup_stage(
                uv
                + [
                    "python",
                    "install",
                    "--managed-python",
                    "--install-dir",
                    str(RUNTIME_ROOT / "managed-python"),
                    "--no-cache",
                    "--no-progress",
                    "3.11.15",
                ],
                "setup-managed-python311",
                plan=plan,
                env=uv_env,
                timeout_seconds=600,
            )
            != 0
        ):
            raise WorkerError("python311_install_failed")
        venv = RUNTIME_ROOT / "venv"
        if (
            self._setup_stage(
                uv
                + [
                    "venv",
                    "--python",
                    "3.11.15",
                    "--managed-python",
                    "--no-python-downloads",
                    "--no-project",
                    "--no-config",
                    str(venv),
                ],
                "setup-python311-venv",
                plan=plan,
                env=uv_env,
                timeout_seconds=180,
            )
            != 0
        ):
            raise WorkerError("python311_venv_create_failed")
        python311 = venv / "bin" / "python"
        venv_site = venv / "lib" / "python3.11" / "site-packages"
        managed_python = RUNTIME_ROOT / "managed-python"
        verified_managed_python311(python311, managed_python, venv_site)
        self.runtime_inventory = link_verified_nvidia_libraries(
            requirements,
            host_site,
            venv_site,
            runtime_inventory_path,
            host_python=platform.python_version(),
        )
        self.status["runtime_inventory"] = {
            "status": self.runtime_inventory["status"],
            "expected_distribution_count": self.runtime_inventory["expected_distribution_count"],
            "shared_library_bytes_reused": self.runtime_inventory["shared_library_bytes_reused"],
        }
        self.save_status()

        filtered_lock = RUNTIME_ROOT / "requirements-without-reused-nvidia.lock"
        removed = requirements_without_nvidia(requirements, filtered_lock)
        expected_removed = {record["name"] for record in self.runtime_inventory["distributions"]}
        if removed != expected_removed:
            raise WorkerError("runtime_nvidia_filtered_lock_mismatch")
        if (
            self._setup_stage(
                uv
                + [
                    "pip",
                    "install",
                    "--python",
                    str(python311),
                    "--require-hashes",
                    "--no-deps",
                    "--no-cache",
                    "--only-binary",
                    ":all:",
                    "--no-config",
                    "--requirements",
                    str(filtered_lock),
                ],
                "setup-python311-packages",
                plan=plan,
                env=uv_env,
                timeout_seconds=1_200,
            )
            != 0
        ):
            raise WorkerError("locked_dependency_install_failed")

        shutil.rmtree(uv_site)
        uv_wheel.unlink(missing_ok=True)
        self.python311 = python311
        self.runtime_lock = runtime_report
        self.runtime_requirements = requirements
        self.status["runtime"] = {
            "python": EXPECTED_RUNTIME["python"],
            "torch": EXPECTED_RUNTIME["torch"],
            "transformers": EXPECTED_RUNTIME["transformers"],
            "bitsandbytes": EXPECTED_RUNTIME["bitsandbytes"],
            "cuda_runtime": EXPECTED_RUNTIME["cuda_runtime"],
            "requirements_lock_sha256": sha256_file(requirements),
            "nvidia_shared_library_bytes_reused": self.runtime_inventory[
                "shared_library_bytes_reused"
            ],
        }
        self.check_storage(mounted_bytes=self.input_artifact_bytes, plan=plan)

    def _verify_python311_runtime(
        self, plan: dict[str, Any], env: dict[str, str]
    ) -> dict[str, Any]:
        if self.python311 is None or self.runtime_requirements is None:
            raise WorkerError("python311_runtime_not_prepared")
        versions = parse_pinned_requirements(self.runtime_requirements)
        nvidia = {name: version for name, version in versions.items() if name.startswith("nvidia-")}
        expected_path = RUNTIME_ROOT / "runtime-probe-input.json"
        result_path = self.out / "runtime-identity.json"
        _write_json(expected_path, {"nvidia": nvidia})
        probe = (
            "import importlib.metadata as m,json,platform,sys,torch;"
            "from pathlib import Path;"
            "expected=json.loads(Path(sys.argv[1]).read_text());"
            "available=torch.cuda.is_available();"
            "identity={'python':platform.python_version(),'torch':torch.__version__,"
            "'cuda_runtime':torch.version.cuda,'transformers':m.version('transformers'),"
            "'bitsandbytes':m.version('bitsandbytes'),'architecture':platform.machine(),"
            "'glibc':platform.libc_ver()[1] if platform.libc_ver()[0]=='glibc' else None,"
            "'cuda_available':available,"
            "'gpu_name':torch.cuda.get_device_name(0) if available else None,"
            "'nvidia':{name:m.version(name) for name in expected['nvidia']}};"
            "Path(sys.argv[2]).write_text(json.dumps(identity,sort_keys=True)+chr(10))"
        )
        command = [str(self.python311), "-c", probe, str(expected_path), str(result_path)]
        if (
            self.run_stage(
                command,
                "verify-python311-runtime",
                env=env,
                reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
                timeout_seconds=300,
            )
            != 0
        ):
            raise WorkerError("python311_runtime_probe_failed")
        identity = _read_json(result_path, "python311_runtime_probe_output_invalid")
        if (
            {key: identity.get(key) for key in EXPECTED_RUNTIME} != EXPECTED_RUNTIME
            or identity.get("architecture") not in {"x86_64", "amd64"}
            or not identity.get("cuda_available")
            or not isinstance(identity.get("gpu_name"), str)
            or "T4" not in identity["gpu_name"]
            or identity.get("nvidia") != nvidia
            or _version_tuple(str(identity.get("glibc"))) < (2, 35)
        ):
            self.status["runtime_probe"] = {
                "identity_valid": False,
                "cuda_available": identity.get("cuda_available") is True,
                "nvidia_count": len(identity.get("nvidia", {})),
            }
            self.save_status()
            raise WorkerError("python311_runtime_identity_mismatch")
        self.status["runtime_probe"] = {
            "identity_valid": True,
            "python": identity["python"],
            "torch": identity["torch"],
            "cuda_runtime": identity["cuda_runtime"],
            "transformers": identity["transformers"],
            "bitsandbytes": identity["bitsandbytes"],
            "gpu_name": identity["gpu_name"],
            "nvidia_count": len(nvidia),
        }
        self.status["hardware"] = {
            "torch": identity["torch"],
            "cuda_devices_visible": 1,
            "training_device": identity["gpu_name"],
            "world_size": 1,
            "other_devices_unused": True,
        }
        self.save_status()
        self.check_storage(mounted_bytes=self.input_artifact_bytes, plan=plan)
        return identity

    def _verify_initializer(
        self, model: Path, *, arm: str, entry: dict[str, Any], env: dict[str, str]
    ) -> dict[str, Any]:
        if self.python311 is None:
            raise WorkerError("python311_runtime_not_prepared")
        input_path = RUNTIME_ROOT / "initializer-entry.json"
        output_path = self.out / "initializer-identity.json"
        _write_json(input_path, entry)
        code = (
            "import json,sys;from pathlib import Path;"
            "from tinycomplete.code_cpt.q25_fim import verify_initializer;"
            "result=verify_initializer(Path(sys.argv[1]),arm=sys.argv[2],"
            "entry=json.loads(Path(sys.argv[3]).read_text()));"
            "Path(sys.argv[4]).write_text(json.dumps(result,sort_keys=True)+chr(10))"
        )
        command = [
            str(self.python311),
            "-c",
            code,
            str(model),
            arm,
            str(input_path),
            str(output_path),
        ]
        if (
            self.run_stage(
                command,
                "verify-model-initializer",
                env=env,
                reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
                timeout_seconds=300,
            )
            != 0
        ):
            raise WorkerError("model_initializer_verification_failed")
        return _read_json(output_path, "model_initializer_identity_invalid")

    def _runtime_artifact_bytes(self) -> int:
        repo_bytes = _tree_bytes(REPO) if REPO.is_dir() else 0
        return repo_bytes + _runtime_regular_bytes(RUNTIME_ROOT)

    def save_status(self) -> None:
        self.status["elapsed_seconds"] = time.monotonic() - self.started
        _write_json(self.out / "worker-status.json", self.status)

    def run_stage(
        self,
        command: list[str],
        name: str,
        *,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        reserve_seconds: float = 30,
    ) -> int:
        remaining = self.deadline - time.monotonic()
        usable = remaining - reserve_seconds
        if usable < 1:
            raise WorkerError("session_deadline_reserve_reached")
        timeout = usable if timeout_seconds is None else min(timeout_seconds, usable)
        started = time.monotonic()
        log_path = self.out / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("wb") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    env=env,
                    cwd=REPO if REPO.is_dir() else None,
                    check=False,
                )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        self.status["stages"].append(
            {"name": name, "exit_code": code, "elapsed_seconds": time.monotonic() - started}
        )
        self.save_status()
        return code

    def check_storage(self, *, mounted_bytes: int, plan: dict[str, Any]) -> None:
        cap = int(plan["configuration"]["budget"]["new_artifact_bytes_cap"])
        output_bytes = _tree_bytes(self.out)
        runtime_bytes = self._runtime_artifact_bytes()
        current = mounted_bytes + output_bytes + runtime_bytes
        self.status["storage"] = {
            "mounted_bytes": mounted_bytes,
            "worker_output_bytes": output_bytes,
            "runtime_setup_bytes": runtime_bytes,
            "combined_bytes": current,
            "cap_bytes": cap,
        }
        if current > cap:
            raise WorkerError("new_artifact_bytes_cap_exceeded")
        free = min(
            shutil.disk_usage(self.out.parent).free,
            shutil.disk_usage(RUNTIME_ROOT.parent).free
            if RUNTIME_ROOT.parent.exists()
            else shutil.disk_usage(self.out.parent).free,
        )
        self.status["storage"]["free_bytes"] = free
        minimum_free_bytes = int(
            plan["configuration"]["budget"].get("minimum_free_bytes", MINIMUM_FREE_BYTES)
        )
        if free < minimum_free_bytes:
            raise WorkerError("minimum_free_space_not_available")
        self.save_status()

    def _run_evaluation(
        self,
        *,
        model: Path,
        input_path: Path,
        plan_path: Path,
        alias: str,
        mode: str,
        stage: str,
        env: dict[str, str],
        reserve_seconds: float,
    ) -> None:
        if self.python311 is None:
            raise WorkerError("python311_runtime_not_prepared")
        command = _evaluation_command(
            python=self.python311,
            model=model,
            input_path=input_path,
            output=self.out / stage,
            plan=plan_path,
            alias=alias,
            mode=mode,
        )
        if (
            self.run_stage(
                command,
                f"eval-{stage.replace('/', '-')}",
                env=env,
                reserve_seconds=reserve_seconds,
            )
            != 0
        ):
            raise WorkerError("fim_evaluation_failed")

    def _verify_repository_fixture(self, paths: dict[str, Path], plan: dict[str, Any]) -> None:
        repo_fixture = REPO / "data/benchmarks/code_completion_v2.jsonl"
        if not repo_fixture.is_file() or sha256_file(repo_fixture) != sha256_file(
            paths["causal200.jsonl"]
        ):
            raise WorkerError("repository_causal_fixture_differs_from_frozen_input")
        planned = plan["evaluation"]["fixtures"]["causal"]
        if sha256_file(repo_fixture) != planned.get("sha256"):
            raise WorkerError("repository_causal_fixture_differs_from_plan")

    def _record_baseline(
        self,
        *,
        initializer_sha256: str,
        measured_seconds: float,
        plan: dict[str, Any],
        mounted_bytes: int,
        previous_root: Path | None = None,
        inherited: dict[str, Any] | None = None,
    ) -> None:
        if inherited is not None:
            if previous_root is None:
                raise WorkerError("resume_baseline_source_missing")
            files = inherited.get("files")
            if (
                inherited.get("schema") != "q25-fim-baseline-v1"
                or inherited.get("arm") != self.session["arm"]
                or inherited.get("plan_sha256") != self.session["plan_sha256"]
                or inherited.get("input_manifest_sha256") != self.session["input_manifest_sha256"]
                or inherited.get("initializer_identity_sha256") != initializer_sha256
                or not isinstance(files, dict)
                or not files
                or not _has_baseline_modes(files)
            ):
                raise WorkerError("resume_baseline_files_missing")
            copy_bytes = 0
            for name, item in files.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(item, dict)
                    or not _is_sha256(item.get("sha256"))
                    or isinstance(item.get("bytes"), bool)
                    or not isinstance(item.get("bytes"), int)
                    or item["bytes"] < 0
                ):
                    raise WorkerError("resume_baseline_file_identity_invalid")
                copy_bytes += item["bytes"]
            cap = int(plan["configuration"]["budget"]["new_artifact_bytes_cap"])
            if mounted_bytes + _tree_bytes(self.out) + copy_bytes > cap:
                raise WorkerError("baseline_copy_exceeds_artifact_cap")
            for name, record in files.items():
                relative = Path(name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != "before"
                ):
                    raise WorkerError("resume_baseline_file_path_invalid")
                source = previous_root / relative
                destination = self.out / relative
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or source.stat().st_size != record["bytes"]
                    or sha256_file(source) != record["sha256"]
                ):
                    raise WorkerError("resume_baseline_file_missing_or_changed")
                if destination.exists() or destination.is_symlink():
                    raise WorkerError("baseline_copy_destination_exists")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if (
                    destination.stat().st_size != record["bytes"]
                    or sha256_file(destination) != record["sha256"]
                ):
                    raise WorkerError("baseline_copy_hash_mismatch")
            _write_json(self.out / "baseline" / "identity.json", inherited)
            record = {
                "schema": "q25-fim-baseline-reference-v1",
                "source": self.session["resume_source"],
                "source_identity_sha256": canonical_sha256(inherited),
                "measured_seconds": inherited["measured_seconds"],
                "files": inherited["files"],
            }
            _write_json(self.out / "baseline" / "reference.json", record)
            self.status["baseline"] = record
            self.status["baseline_source"] = "verified_prior_arm_output"
            self.check_storage(mounted_bytes=mounted_bytes, plan=plan)
            return
        files = _baseline_inventory(self.out)
        if not files or not _has_baseline_modes(files):
            raise WorkerError("baseline_evaluation_outputs_missing")
        record = {
            "schema": "q25-fim-baseline-v1",
            "arm": self.session["arm"],
            "plan_sha256": self.session["plan_sha256"],
            "input_manifest_sha256": self.session["input_manifest_sha256"],
            "initializer_identity_sha256": initializer_sha256,
            "measured_seconds": measured_seconds,
            "files": files,
        }
        _write_json(self.out / "baseline" / "identity.json", record)
        self.status["baseline"] = record
        self.status["baseline_source"] = "evaluated_frozen_initializer"

    def execute(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        self.save_status()
        try:
            input_dir, manifest = find_input_manifest(
                INPUT_ROOT, self.session["input_manifest_sha256"]
            )
            paths, plan = verify_input_bundle(input_dir, manifest, self.session)
            scale_variant = self.session.get("scale_variant")
            if scale_variant is not None:
                selected_name = (
                    "repeat_train.jsonl"
                    if scale_variant == "repeat"
                    else "scaled_train.jsonl"
                )
                paths["train.jsonl"] = paths[selected_name]
                paths["development.jsonl"] = paths["development_new.jsonl"]
                paths["historical_development.jsonl"] = paths["development_previous.jsonl"]
            plan_path = paths["plan.json"]
            arm = self.session["arm"]
            initializer_entry = plan["initializers"][arm]
            model = find_initializer_path(INPUT_ROOT, initializer_entry)
            resume_checkpoint: Path | None = None
            previous_root: Path | None = None
            inherited_baseline: dict[str, Any] | None = None
            total_mounted = _tree_bytes(INPUT_ROOT)
            self.input_artifact_bytes = total_mounted
            self.status["state"] = "verified_inputs"
            self.save_status()
            self.check_storage(mounted_bytes=total_mounted, plan=plan)

            self._checkout_repository(plan)
            self._verify_repository_fixture(paths, plan)
            runtime_report, runtime_requirements = verify_runtime_lock_files(REPO, plan)
            self.runtime_requirements = runtime_requirements
            self._prepare_python311(plan, runtime_report, runtime_requirements)
            if self.python311 is None:
                raise WorkerError("python311_runtime_not_prepared")

            env = {
                **os.environ,
                "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
                "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
                "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(self.out / "observability.jsonl"),
                "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "0",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            self._verify_python311_runtime(plan, env)
            initializer_identity = self._verify_initializer(
                model, arm=arm, entry=initializer_entry, env=env
            )
            initializer_sha256 = canonical_sha256(initializer_identity)
            if self.session["resume_source"] is not None:
                resume_checkpoint, previous_root, inherited_baseline = resolve_verified_resume(
                    INPUT_ROOT,
                    self.session,
                    plan_sha256=self.session["plan_sha256"],
                    input_manifest_sha256=self.session["input_manifest_sha256"],
                    initializer_sha256=initializer_sha256,
                    arm=arm,
                )
            trainer_files = [
                paths[name]
                for name in (
                    "plan.json",
                    "train.jsonl",
                    "development.jsonl",
                    "corpus_metadata.json",
                )
            ]
            if scale_variant is not None:
                trainer_files.extend(
                    paths[name]
                    for name in (
                        "repeat_train.jsonl",
                        "scaled_train.jsonl",
                        "development_previous.jsonl",
                    )
                )
            mounted_extra = mounted_artifact_bytes(
                INPUT_ROOT,
                initializer=model,
                trainer_files=trainer_files,
                resume_checkpoint=resume_checkpoint,
            )
            runtime_bytes = self._runtime_artifact_bytes()
            mounted_extra += runtime_bytes
            self.status["input"] = {
                "directory": str(input_dir),
                "file_count": len(manifest["files"]),
                "mounted_bytes": total_mounted,
                "mounted_extra_bytes": mounted_extra,
                "repository_bytes": _tree_bytes(REPO),
                "python_runtime_bytes": _runtime_regular_bytes(RUNTIME_ROOT),
                "nvidia_shared_library_bytes_reused": self.runtime_inventory[
                    "shared_library_bytes_reused"
                ]
                if self.runtime_inventory is not None
                else 0,
                "initializer_identity_sha256": initializer_sha256,
            }
            self.save_status()
            if canonical_sha256(initializer_identity) != initializer_sha256:
                raise WorkerError("initializer_identity_changed_after_verification")

            preflight_command = trainer_command(
                python=self.python311,
                model=model,
                paths=paths,
                plan=plan_path,
                arm=arm,
                output=self.out / "training",
                session_seconds=min(
                    self.session["session_seconds"],
                    int(plan["configuration"]["budget"]["session_seconds"]),
                ),
                reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
                external_campaign_tokens=self.session["external_campaign_tokens"],
                mounted_bytes=mounted_extra,
                resume=resume_checkpoint,
                execute=False,
                scale_variant=scale_variant,
            )
            if (
                self.run_stage(preflight_command, "trainer-preflight", env=env, reserve_seconds=60)
                != 0
            ):
                raise WorkerError("trainer_preflight_failed")

            smoke_output = self.out / "attention-smoke"
            smoke_command = _attention_smoke_command(
                python=self.python311,
                model=model,
                development=paths["development.jsonl"],
                line_suite=paths["line180.jsonl"],
                output=smoke_output,
                plan=plan_path,
            )
            if (
                self.run_stage(
                    smoke_command,
                    "attention-smoke",
                    env=env,
                    reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
                )
                != 0
            ):
                raise WorkerError("fim_attention_smoke_failed")
            smoke_path = smoke_output / "attention-smoke.json"
            smoke_report = _read_json(smoke_path, "fim_attention_smoke_report_invalid")
            smoke_tokens = smoke_report.get("query_tokens")
            smoke_peak_bytes = smoke_report.get("peak_allocated_bytes")
            if (
                smoke_report.get("schema") != "q25-fim-attention-smoke-v1"
                or smoke_report.get("attention_backend")
                != "torch-efficient-sdpa-explicit-kv-repeat-v1"
                or smoke_report.get("key_value_head_expansion") != "explicit-repeat"
                or smoke_report.get("plan_sha256") != self.session["plan_sha256"]
                or smoke_report.get("success") is not True
                or isinstance(smoke_tokens, bool)
                or not isinstance(smoke_tokens, int)
                or smoke_tokens < 1
                or smoke_report.get("query_heads") != 14
                or smoke_report.get("key_value_heads") != 2
                or smoke_report.get("head_dim") != 64
                or smoke_report.get("dtype") != "float16"
                or not isinstance(smoke_report.get("cuda_device"), str)
                or not smoke_report["cuda_device"]
                or isinstance(smoke_peak_bytes, bool)
                or not isinstance(smoke_peak_bytes, int)
                or smoke_peak_bytes < 0
                or smoke_report.get("dense_fp32_attention_score_bytes")
                != smoke_tokens * smoke_tokens * 14 * 4
            ):
                raise WorkerError("fim_attention_smoke_evidence_mismatch")
            self.status["attention_smoke"] = {
                "sha256": sha256_file(smoke_path),
                "query_tokens": smoke_tokens,
                "query_heads": 14,
                "key_value_heads": 2,
                "head_dim": 64,
                "key_value_head_expansion": "explicit-repeat",
                "peak_allocated_bytes": smoke_peak_bytes,
                "dense_fp32_attention_score_bytes": smoke_report[
                    "dense_fp32_attention_score_bytes"
                ],
            }
            self.save_status()

            alias = "untouched-q25" if arm == TRAIN_ARM else "completed-cpt-q25"
            if scale_variant is not None:
                alias = f"{alias}-completion-scale-{scale_variant}"
            before_started = time.monotonic()
            if inherited_baseline is None:
                self.status["state"] = "baseline_evaluation"
                self.save_status()
                baseline_modes = [
                    ("development", "development.jsonl"),
                    ("line", "line180.jsonl"),
                ]
                if scale_variant is not None:
                    baseline_modes.append(("development", "development_previous.jsonl"))
                for mode, filename in baseline_modes:
                    stage = (
                        "before/historical-development"
                        if filename == "development_previous.jsonl"
                        else f"before/{mode}"
                    )
                    self._run_evaluation(
                        model=model,
                        input_path=paths[filename],
                        plan_path=plan_path,
                        alias=f"{alias}-before-fim",
                        mode=mode,
                        stage=stage,
                        env=env,
                        reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
                    )
                before_seconds = time.monotonic() - before_started
                self._record_baseline(
                    initializer_sha256=initializer_sha256,
                    measured_seconds=before_seconds,
                    plan=plan,
                    mounted_bytes=total_mounted,
                )
            else:
                before_seconds = float(inherited_baseline["measured_seconds"])
                self._record_baseline(
                    initializer_sha256=initializer_sha256,
                    measured_seconds=before_seconds,
                    plan=plan,
                    mounted_bytes=total_mounted,
                    previous_root=previous_root,
                    inherited=inherited_baseline,
                )
            self.status["before_evaluation_seconds"] = before_seconds

            trainer_reserve = max(MINIMUM_FINAL_RESERVE_SECONDS, int(1.25 * before_seconds + 300))
            post_eval_reserve = max(MINIMUM_FINAL_RESERVE_SECONDS, int(2.5 * before_seconds + 300))
            remaining = int(self.deadline - time.monotonic())
            trainer_session_seconds = min(
                int(plan["configuration"]["budget"]["session_seconds"]),
                remaining - post_eval_reserve - 60,
            )
            self.status["timing_reserves"] = {
                "trainer_reserve_seconds": trainer_reserve,
                "post_evaluation_reserve_seconds": post_eval_reserve,
                "trainer_session_seconds": trainer_session_seconds,
            }
            if trainer_session_seconds <= trainer_reserve + 60:
                self.status["state"] = "training_deferred_insufficient_time"
                self.save_status()
                return 0

            worker_extra_output = _tree_bytes(self.out) - _tree_bytes(self.out / "training")
            trainer_mounted_bytes = mounted_extra + max(0, worker_extra_output)
            self.check_storage(mounted_bytes=total_mounted + max(0, worker_extra_output), plan=plan)
            training_command = trainer_command(
                python=self.python311,
                model=model,
                paths=paths,
                plan=plan_path,
                arm=arm,
                output=self.out / "training",
                session_seconds=trainer_session_seconds,
                reserve_seconds=trainer_reserve,
                external_campaign_tokens=self.session["external_campaign_tokens"],
                mounted_bytes=trainer_mounted_bytes,
                resume=resume_checkpoint,
                execute=True,
                scale_variant=scale_variant,
            )
            self.status["state"] = "training"
            self.status["training_started"] = True
            self.status["trainer_mounted_artifact_bytes"] = trainer_mounted_bytes
            self.save_status()
            training_exit = self.run_stage(
                training_command,
                "training",
                env=env,
                timeout_seconds=trainer_session_seconds + 30,
                reserve_seconds=post_eval_reserve,
            )
            self.status["training_exit_code"] = training_exit
            result_path = self.out / "training" / "run_result.json"
            if training_exit != 0 or not result_path.is_file():
                self.status["state"] = "partial_checkpoint_preserved"
                self.status["checkpoint_pointer_present"] = (
                    self.out / "training" / "latest.json"
                ).is_file()
                self.save_status()
                return 0
            result = _read_json(result_path, "training_result_invalid")
            self.status["training"] = {
                "status": result.get("status"),
                "fingerprint": result.get("fingerprint"),
                "logical_training_input_tokens": result.get("logical_training_input_tokens"),
                "actual_campaign_input_tokens": result.get("actual_campaign_input_tokens"),
                "cursor": result.get("cursor"),
                "training_updates": result.get("training_updates"),
            }
            self.check_storage(mounted_bytes=total_mounted, plan=plan)
            if result.get("status") != "complete":
                self.status["state"] = "partial_checkpoint_preserved"
                self.save_status()
                return 0

            export = self.out / "training" / "inference-f16"
            self._verify_fim_export(export, result, arm)
            self.status["state"] = "candidate_evaluation"
            self.save_status()
            after_started = time.monotonic()
            after_modes = [
                ("development", "development.jsonl"),
                ("line", "line180.jsonl"),
            ]
            if scale_variant is not None:
                after_modes.append(("development", "development_previous.jsonl"))
            for mode, filename in after_modes:
                stage = (
                    "after/historical-development"
                    if filename == "development_previous.jsonl"
                    else f"after/{mode}"
                )
                self._run_evaluation(
                    model=export,
                    input_path=paths[filename],
                    plan_path=plan_path,
                    alias=f"{alias}-after-fim",
                    mode=mode,
                    stage=stage,
                    env=env,
                    reserve_seconds=30,
                )
                self.check_storage(mounted_bytes=total_mounted, plan=plan)
            for name, command in _regression_commands(
                python=self.python311,
                model=export,
                paths=paths,
                output=self.out,
                plan_sha256=self.session["plan_sha256"],
                alias=f"{alias}-after-fim",
            ):
                if self.run_stage(command, name, env=env, reserve_seconds=30) != 0:
                    raise WorkerError("raw_regression_evaluation_failed")
                self.check_storage(mounted_bytes=total_mounted, plan=plan)
            self.status["after_evaluation_seconds"] = time.monotonic() - after_started
            self.status["state"] = "complete"
            self.save_status()
            return 0
        except WorkerError as exc:
            self.status.update(state="failed", error_reason=exc.reason)
            self.save_status()
            return 1
        except Exception as exc:
            self.status.update(state="failed", error_class=type(exc).__name__)
            self.save_status()
            return 1

    def _verify_fim_export(self, export: Path, result: dict[str, Any], arm: str) -> None:
        manifest_path = export / "artifact_manifest.json"
        manifest = _read_json(manifest_path, "fim_export_manifest_missing")
        if (
            manifest.get("schema") != "q25-fim-inference-f16-v1"
            or manifest.get("arm") != arm
            or manifest.get("fingerprint") != result.get("fingerprint")
            or manifest.get("training_cursor") != result.get("cursor")
        ):
            raise WorkerError("fim_export_identity_mismatch")
        records = manifest.get("files")
        if not isinstance(records, dict) or "model.safetensors" not in records:
            raise WorkerError("fim_export_file_manifest_missing")
        for name, record in records.items():
            if Path(name).name != name or not isinstance(record, dict):
                raise WorkerError("fim_export_file_path_invalid")
            path = export / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                raise WorkerError("fim_export_file_hash_mismatch")


def main() -> int:
    try:
        worker = Worker(SESSION)
        result_code = worker.execute()
    except WorkerError as exc:
        print(
            json.dumps({"state": "failed", "error_reason": exc.reason}, sort_keys=True),
            flush=True,
        )
        return 1
    except Exception as exc:
        print(
            json.dumps({"state": "failed", "error_class": type(exc).__name__}, sort_keys=True),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "state": worker.status.get("state", "finished"),
                "arm": worker.status.get("arm"),
                "exit_code": result_code,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
