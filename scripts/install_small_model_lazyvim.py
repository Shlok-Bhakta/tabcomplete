"""Install the selected local LazyVim predictor without replacing mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "tools/trajectory_collector/nvim"
DEFAULT_RUNTIME = ROOT.parent / "tabcomplete/outputs/tools/llama.cpp/build/bin/llama-server"
WIRE_VERSIONS = ("compact-next-edit-v1", "single-line-edit-v1")
FIM_PROTOCOL = "q25-fim-line-completion-v1"
FIM_ALIAS = "q25-fim"
FIM_LAYOUT = "q25-fim-psm-bounded-v2"
FIM_MODE = "remaining_logical_line_after_utf8_cursor"
FIM_PORT = 19093
FIM_MEMORY_MAX = 1500 * 1024 * 1024


def prepare_native_fim_profile(
    model: Path, tokenizer: Path, conversion: Path, output: Path
) -> dict:
    """Prepare an explicit candidate profile without changing the stable service."""
    from tinycomplete.code_cpt.q25_fim_conversion import (
        EOS_TOKEN_ID,
        FIM_MARKER_IDS,
        MODEL_ID,
        MODEL_REVISION,
        TOKENIZER_SHA256,
    )

    manifest = json.loads(conversion.read_text())
    expected_tokenizer = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "sha256": TOKENIZER_SHA256,
        "eos_token_id": EOS_TOKEN_ID,
        "fim_marker_ids": FIM_MARKER_IDS,
    }
    artifact = manifest.get("q4_export", {})
    if (
        manifest.get("schema") != "q25-fim-q4-conversion-run-v1"
        or manifest.get("status") != "complete"
        or manifest.get("conversion") != {"format": "Q4_K_M", "gpu_enabled": False}
        or manifest.get("tokenizer") != expected_tokenizer
        or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_export_manifest_sha256")))
        or not model.is_file()
        or sha(model) != artifact.get("sha256")
        or model.stat().st_size != artifact.get("bytes")
        or sha(tokenizer) != TOKENIZER_SHA256
    ):
        raise ValueError(
            "native FIM profile requires the verified selected Q4 and pinned tokenizer"
        )
    raw = json.loads(tokenizer.read_text())
    inventory = sorted(
        ({"id": entry["id"], "spelling": entry["content"]} for entry in raw["added_tokens"]),
        key=lambda entry: entry["id"],
    )
    vocab_ids = sorted(set(raw["model"]["vocab"].values()) | {entry["id"] for entry in inventory})
    vocab_digest = hashlib.sha256("".join(f"{value}\n" for value in vocab_ids).encode()).hexdigest()
    mode = "remaining_logical_line_after_utf8_cursor"
    fields = [
        MODEL_ID,
        MODEL_REVISION,
        TOKENIZER_SHA256,
        EOS_TOKEN_ID,
        FIM_MARKER_IDS["fim_prefix"],
        FIM_MARKER_IDS["fim_suffix"],
        FIM_MARKER_IDS["fim_middle"],
        mode,
        len(vocab_ids),
        vocab_digest,
    ]
    contract = "q25-fim-tokenizer-contract-v1\n" + "".join(f"{value}\n" for value in fields)
    contract += "".join(f"{entry['id']}\t{entry['spelling']}\n" for entry in inventory)
    tokenizer_profile = {
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_contract_sha256": hashlib.sha256(contract.encode()).hexdigest(),
        "tokenizer_vocab_size": len(vocab_ids),
        "tokenizer_vocab_ids_sha256": vocab_digest,
        "tokenizer_vocab_ids": vocab_ids,
        "eos_id": EOS_TOKEN_ID,
        "fim_prefix_id": fields[4],
        "fim_suffix_id": fields[5],
        "fim_middle_id": fields[6],
        "completion_mode": mode,
        "special_tokens": inventory,
    }
    protocol = "q25-fim-line-completion-v1"
    profile = {
        "artifact_manifest_sha256": manifest["source_export_manifest_sha256"],
        "tokenizer": tokenizer_profile,
    }
    registry = {
        "q25-fim": {
            "path": str(model.resolve()),
            "sha256": artifact["sha256"],
            "protocol": protocol,
            "fim_profile": profile,
        }
    }
    editor_profile = json.loads(json.dumps(profile))
    del editor_profile["tokenizer"]["tokenizer_vocab_ids"]
    spec = {
        "model_sha256": artifact["sha256"],
        "model_protocol": protocol,
        "output_tokens": 96,
        "fim_profile": editor_profile,
    }
    if output.is_symlink():
        raise ValueError("native FIM profile output must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    embedded_profile = {
        "schema": "tabcomplete-q25-fim-embedded-profile-v1",
        "model_sha256": artifact["sha256"],
        "serving_profile": profile,
    }
    for name, value in {
        "registry.json": registry,
        "editor-model.json": spec,
        "embedded-profile.json": embedded_profile,
    }.items():
        target = output / name
        content = json.dumps(value, sort_keys=True, indent=2) + "\n"
        if target.exists() and (target.is_symlink() or target.read_text() != content):
            raise ValueError("native FIM profile output already has another identity")
        target.write_text(content)
    return {
        "model_sha256": artifact["sha256"],
        "protocol": protocol,
        "registry": str(output / "registry.json"),
        "editor_model": str(output / "editor-model.json"),
        "embedded_profile": str(output / "embedded-profile.json"),
        "tokenizer_contract_sha256": tokenizer_profile["tokenizer_contract_sha256"],
        "activated": False,
        "automatic_personalization_enabled": False,
    }


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def owned_by_nix(path: Path) -> bool:
    if path.is_symlink() or str(path.resolve()).startswith("/nix/store/"):
        return True
    if path.exists():
        head = path.read_text()[:512].lower()
        return "home-manager" in head or "generated by nix" in head
    return False


def render_config(
    original: str,
    *,
    model_revision: str,
    model_alias: str,
    runtime_config_hash: str,
    experimental_automatic: bool,
    protocol_version: str = "compact-next-edit-v1",
    precision: str = "Q4_K_M",
) -> str:
    if protocol_version not in WIRE_VERSIONS:
        raise ValueError("unsupported selected-model action protocol")
    if precision not in {"Q4_K_M", "Q5_K_M"}:
        raise ValueError("unsupported selected-model weight precision")
    match = re.search(
        r'(?m)^(\s*dir\s*=\s*)"[^"]*/tools/trajectory_collector/nvim"(,\s*)$', original
    )
    if not match:
        raise ValueError("existing LazyVim trajectory plugin dir was not found exactly once")
    if len(re.findall(r"tools/trajectory_collector/nvim", original)) != 1:
        raise ValueError("ambiguous trajectory plugin configuration")
    replacement = match.group(1) + '"' + str(PLUGIN) + '"' + match.group(2)
    updated = original[: match.start()] + replacement + original[match.end() :]
    needle = 'require("tabcomplete_trajectory").setup({'
    if needle not in updated:
        raise ValueError("collector setup call not found")
    mode = "automatic" if experimental_automatic else "manual"
    line = (
        '\n      require("tabcomplete_trajectory.predict").setup({ '
        f'model = "{model_alias}", model_revision = "{model_revision}", '
        f'precision = "{precision}", adapter_identity = "full-weight", '
        f'runtime_config_hash = "{runtime_config_hash}", '
        f'protocol_version = "{protocol_version}", '
        f"automatic_prefix_guard = {str(protocol_version == 'single-line-edit-v1').lower()}, "
        f'mode = "{mode}", experimental_auto_opt_in = '
        f"{str(experimental_automatic).lower()}, automatic_quality_validated = false, "
        "automatic_personalization_enabled = false, persist_mode = true })"
    )
    if 'require("tabcomplete_trajectory.predict").setup' in updated:
        updated = re.sub(
            r'\n\s*require\("tabcomplete_trajectory.predict"\)\.setup\(\{[^\n]*\}\)',
            line,
            updated,
            count=1,
        )
    else:
        # Insert after the collector setup's closing line. Preserve all other
        # settings and existing mappings byte-for-byte.
        close = updated.find("\n      })", updated.index(needle))
        if close < 0:
            raise ValueError("collector setup closing line not found")
        close += len("\n      })")
        updated = updated[:close] + line + updated[close:]
    return updated


def render_service(model: Path, runtime: Path) -> str:
    for path in (model, runtime):
        if any(char.isspace() for char in str(path)):
            raise ValueError("service paths must not contain whitespace")
    return (
        "[Unit]\nDescription=TabComplete local next-edit predictor\n"
        "After=network.target\n\n[Service]\nType=simple\n"
        f"ExecStart={runtime} -m {model} --host 127.0.0.1 --port 19093 "
        "-t 4 -tb 4 -ngl 0 -c 2304 -b 256 -ub 64 -np 1 "
        "--cache-ram 128 --ctx-checkpoints 32 --no-cache-idle-slots --no-warmup\n"
        "Restart=on-failure\nRestartSec=3\nNoNewPrivileges=yes\n"
        "MemoryMax=1500M\n\n[Install]\nWantedBy=default.target\n"
    )


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in FIM model profile")
        result[key] = value
    return result


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _load_fim_spec(path: Path, model_sha: str) -> tuple[dict, str]:
    from tinycomplete.code_cpt.q25_fim_conversion import (
        EOS_TOKEN_ID,
        FIM_MARKER_IDS,
        MODEL_ID,
        MODEL_REVISION,
        TOKENIZER_SHA256,
    )

    try:
        raw = json.loads(path.read_text(), object_pairs_hook=_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("cannot read the FIM editor-model profile") from None
    if not isinstance(raw, dict) or set(raw) != {
        "model_sha256",
        "model_protocol",
        "output_tokens",
        "fim_profile",
    }:
        raise ValueError("invalid FIM editor-model profile shape")
    profile = raw["fim_profile"]
    tokenizer = profile.get("tokenizer") if isinstance(profile, dict) else None
    if (
        not isinstance(profile, dict)
        or not isinstance(tokenizer, dict)
        or raw["model_sha256"] != model_sha
        or raw["model_protocol"] != FIM_PROTOCOL
        or type(raw["output_tokens"]) is not int
        or raw["output_tokens"] != 96
        or set(profile) != {"artifact_manifest_sha256", "tokenizer"}
        or not _valid_sha(profile["artifact_manifest_sha256"])
        or tokenizer["tokenizer_id"] != MODEL_ID
        or tokenizer["tokenizer_revision"] != MODEL_REVISION
        or tokenizer["tokenizer_sha256"] != TOKENIZER_SHA256
        or tokenizer["eos_id"] != EOS_TOKEN_ID
        or tokenizer["fim_prefix_id"] != FIM_MARKER_IDS["fim_prefix"]
        or tokenizer["fim_suffix_id"] != FIM_MARKER_IDS["fim_suffix"]
        or tokenizer["fim_middle_id"] != FIM_MARKER_IDS["fim_middle"]
        or tokenizer["completion_mode"] != FIM_MODE
        or "tokenizer_vocab_ids" in tokenizer
        or type(tokenizer["tokenizer_vocab_size"]) is not int
        or tokenizer["tokenizer_vocab_size"] < 1
        or not _valid_sha(tokenizer["tokenizer_vocab_ids_sha256"])
        or not _valid_sha(tokenizer["tokenizer_contract_sha256"])
        or not isinstance(tokenizer["special_tokens"], list)
    ):
        raise ValueError("FIM editor-model profile does not match the selected artifact")
    tokens = tokenizer["special_tokens"]
    try:
        ids = [item["id"] for item in tokens]
        spellings = [item["spelling"] for item in tokens]
    except (KeyError, TypeError):
        raise ValueError("invalid FIM special-token inventory") from None
    if (
        any(type(token_id) is not int or token_id < 0 for token_id in ids)
        or ids != sorted(set(ids))
        or any(
            not isinstance(spelling, str)
            or not spelling.isascii()
            or not spelling
            or any(char in spelling for char in "\r\n\t")
            for spelling in spellings
        )
        or len(set(spellings)) != len(spellings)
    ):
        raise ValueError("invalid FIM special-token inventory")
    inventory = dict(zip(ids, spellings, strict=True))
    if any(
        inventory.get(token_id) != spelling
        for token_id, spelling in {
            EOS_TOKEN_ID: "<|endoftext|>",
            FIM_MARKER_IDS["fim_prefix"]: "<|fim_prefix|>",
            FIM_MARKER_IDS["fim_suffix"]: "<|fim_suffix|>",
            FIM_MARKER_IDS["fim_middle"]: "<|fim_middle|>",
        }.items()
    ):
        raise ValueError("FIM special-token inventory omits a required marker")
    return raw, sha(path)


def _lua_json(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    level = 0
    while "]" + "=" * level + "]" in encoded:
        level += 1
    opener, closer = "[" + "=" * level + "[", "]" + "=" * level + "]"
    return f"vim.json.decode({opener}{encoded}{closer})"


def render_native_fim_config(
    original: str, spec: dict, runtime_config_hash: str | None = None
) -> tuple[str, str]:
    if runtime_config_hash is not None and not _valid_sha(runtime_config_hash):
        raise ValueError("invalid observed runtime configuration digest")
    match = re.search(
        r'(?m)^(\s*dir\s*=\s*)"[^"]*/tools/trajectory_collector/nvim"(,\s*)$', original
    )
    if not match or len(re.findall(r"tools/trajectory_collector/nvim", original)) != 1:
        raise ValueError("existing LazyVim trajectory plugin dir was not found exactly once")
    updated = (
        original[: match.start()]
        + match.group(1)
        + '"'
        + str(PLUGIN)
        + '"'
        + match.group(2)
        + original[match.end() :]
    )
    needle = 'require("tabcomplete_trajectory").setup({'
    if updated.count(needle) != 1:
        raise ValueError("collector setup call was not found exactly once")
    options = [
        'url = "http://127.0.0.1:19093"',
        'backend = "rust-editor-v1"',
        f'model = "{FIM_ALIAS}"',
        f'model_revision = "{spec["model_sha256"]}"',
        'precision = "Q4_K_M"',
        'adapter_identity = "embedded-q25-fim"',
    ]
    if runtime_config_hash is not None:
        options.append(f'runtime_config_hash = "{runtime_config_hash}"')
    options.extend(
        [
            f'protocol_version = "{FIM_PROTOCOL}"',
            "single_line_input_tokens = 1024",
            "max_prompt_tokens = 1024",
            "target_prompt_tokens = 1024",
            'mode = "automatic"',
            "experimental_auto_opt_in = true",
            "automatic_quality_validated = false",
            "automatic_personalization_enabled = false",
            "automatic_normal_mode = true",
            'accept_key = "<M-l>"',
            'predict_key = "<M-p>"',
            'mode_state_path = vim.fn.stdpath("state") .. "/tabcomplete-q25-fim-mode.json"',
            f'allowed_models = {{["{FIM_ALIAS}"]={_lua_json(spec)}}}',
            "persist_mode = true",
        ]
    )
    setup = (
        '\n      require("tabcomplete_trajectory.predict").setup({ ' + ", ".join(options) + " })"
    )
    predict_setup = 'require("tabcomplete_trajectory.predict").setup'
    if predict_setup in updated:
        pattern = r'\n\s*require\("tabcomplete_trajectory.predict"\)\.setup\(\{[^\n]*\}\)'
        if len(re.findall(pattern, updated)) != 1:
            raise ValueError("existing predictor setup is ambiguous or multiline")
        updated = re.sub(pattern, setup, updated, count=1)
    else:
        close = updated.find("\n      })", updated.index(needle))
        if close < 0:
            raise ValueError("collector setup closing line not found")
        close += len("\n      })")
        updated = updated[:close] + setup + updated[close:]
    return updated, setup.strip()


def render_native_fim_service(runtime: Path) -> str:
    if any(char.isspace() for char in str(runtime)):
        raise ValueError("native FIM executable path must not contain whitespace")
    return (
        "[Unit]\nDescription=TabComplete experimental Q25 FIM predictor\n"
        "After=network.target\n\n[Service]\nType=simple\n"
        f"ExecStart={runtime} --host 127.0.0.1 --port {FIM_PORT} --threads 4 "
        "--prompt-threads 4 --context-size 2304 --context-layout cursor-last-v1 "
        "--input-tokens 1024 --batch-size 256 --microbatch-size 64 "
        "--output-tokens 96 --cache-type f16 --syntax-validation false\n"
        "Restart=on-failure\nRestartSec=3\nTimeoutStopSec=10\n"
        "MemoryMax=1500M\nMemorySwapMax=0\nCPUQuota=400%\n"
        "NoNewPrivileges=yes\nPrivateTmp=yes\nProtectSystem=strict\n"
        "ProtectHome=read-only\nUMask=0077\n"
        "Environment=TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT=0\n"
        "Environment=TABCOMPLETE_AUTOMATIC_TRAINING=0\n"
        "\n[Install]\nWantedBy=default.target\n"
    )


def _local_json(path: str) -> dict | list:
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{FIM_PORT}{path}")
        with urllib.request.urlopen(request, timeout=1.0) as response:
            if response.status != 200:
                raise ValueError("native FIM health endpoint returned an error")
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("native FIM identity response exceeds its bound")
        value = json.loads(raw)
        if not isinstance(value, (dict, list)):
            raise ValueError("native FIM identity response has the wrong shape")
        return value
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        raise RuntimeError("native FIM health endpoint is unavailable") from None


def _verify_native_health(identity: dict, spec: dict, model_sha: str) -> str:
    tokenizer = spec["fim_profile"]["tokenizer"]
    expected = {
        "status": "ok",
        "alias": FIM_ALIAS,
        "model_sha256": model_sha,
        "model_protocol": FIM_PROTOCOL,
        "model_embedded": True,
        "model_switch_supported": False,
        "model_storage": "executable-mmap",
        "backend": "llama.cpp CPU via Rust",
        "context_layout": FIM_LAYOUT,
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
        "completion_mode": FIM_MODE,
        "fim_profile": spec["fim_profile"],
        "tokenizer_id": tokenizer["tokenizer_id"],
        "tokenizer_revision": tokenizer["tokenizer_revision"],
        "tokenizer_sha256": tokenizer["tokenizer_sha256"],
        "tokenizer_contract_sha256": tokenizer["tokenizer_contract_sha256"],
        "tokenizer_vocab_size": tokenizer["tokenizer_vocab_size"],
        "tokenizer_vocab_ids_sha256": tokenizer["tokenizer_vocab_ids_sha256"],
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("native FIM health identity does not match the selected profile")
    runtime_hash = identity.get("runtime_config_hash")
    if not isinstance(runtime_hash, str) or not _valid_sha(runtime_hash):
        raise ValueError("native FIM health omitted its runtime configuration digest")
    return runtime_hash


def _systemctl(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["systemctl", "--user", *args], check=False, capture_output=True, text=True
        )
    except OSError:
        if check:
            raise RuntimeError("systemd user service operation failed") from None
        return subprocess.CompletedProcess(args, 1, "", "")
    if check and result.returncode != 0:
        raise RuntimeError("systemd user service operation failed")
    return result


def _systemd_properties(unit: str) -> dict[str, int]:
    result = _systemctl(
        ["show", unit, "--property=MainPID", "--property=MemoryCurrent", "--property=MemoryMax"]
    )
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"MainPID", "MemoryCurrent", "MemoryMax"}:
            try:
                values[key] = int(value)
            except ValueError:
                pass
    if set(values) != {"MainPID", "MemoryCurrent", "MemoryMax"}:
        raise ValueError("systemd omitted native FIM process or memory properties")
    if values["MainPID"] <= 0 or values["MemoryMax"] != FIM_MEMORY_MAX:
        raise ValueError("native FIM service process or memory limit is invalid")
    if values["MemoryCurrent"] > FIM_MEMORY_MAX:
        raise ValueError("native FIM process exceeds its memory budget")
    return values


def _process_executable(pid: int) -> Path:
    return Path(f"/proc/{pid}/exe")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _restore_native_files(
    config: Path,
    unit: Path,
    original_config: bytes,
    original_unit: bytes | None,
    unit_was_active: bool,
    unit_was_enabled: bool,
) -> None:
    _atomic_write(config, original_config)
    if original_unit is None:
        _systemctl(["disable", "--now", unit.name], check=False)
        unit.unlink(missing_ok=True)
    else:
        _atomic_write(unit, original_unit)
    _systemctl(["daemon-reload"])
    if original_unit is not None and unit_was_active:
        _systemctl(["restart", unit.name])
    elif original_unit is not None:
        _systemctl(["stop", unit.name], check=False)
    if original_unit is not None:
        _systemctl(["enable" if unit_was_enabled else "disable", unit.name])
        active = _systemctl(["is-active", unit.name], check=False).stdout.strip() == "active"
        if active != unit_was_active:
            raise RuntimeError("previous service state was not restored")
    else:
        if _systemctl(["is-active", unit.name], check=False).stdout.strip() == "active":
            raise RuntimeError("new predictor service did not stop during rollback")
        if _systemctl(["is-enabled", unit.name], check=False).stdout.strip() == "enabled":
            raise RuntimeError("new predictor service remained enabled during rollback")
    if config.read_bytes() != original_config:
        raise RuntimeError("previous editor config was not restored")
    if original_unit is not None and unit.read_bytes() != original_unit:
        raise RuntimeError("previous service file was not restored")
    if original_unit is None and unit.exists():
        raise RuntimeError("new service file was not removed")


def install_native_fim(
    config: Path,
    unit: Path,
    model: Path,
    runtime: Path,
    editor_model: Path,
    expected_model_sha: str,
    expected_runtime_sha: str,
    expected_model_bytes: int,
    *,
    dry_run: bool,
) -> dict:
    if owned_by_nix(config) or owned_by_nix(unit):
        raise RuntimeError("Nix/Home Manager owns this path; edit its source configuration")
    if (
        not _valid_sha(expected_model_sha)
        or not _valid_sha(expected_runtime_sha)
        or not model.is_file()
        or sha(model) != expected_model_sha
        or type(expected_model_bytes) is not int
        or not 1 <= expected_model_bytes <= 2 * 1024**3
        or model.stat().st_size != expected_model_bytes
        or not runtime.is_file()
        or not os.access(runtime, os.X_OK)
        or sha(runtime) != expected_runtime_sha
    ):
        raise ValueError("selected native FIM model or executable identity mismatch")
    spec, spec_sha = _load_fim_spec(editor_model, expected_model_sha)
    try:
        original_config = config.read_bytes()
        original_text = original_config.decode("utf-8")
    except (OSError, UnicodeError):
        raise ValueError("existing editor configuration is unavailable or invalid") from None
    if unit.exists() and not unit.is_file():
        raise ValueError("native FIM service path must be a regular file")
    unit_existed = unit.is_file()
    original_unit = unit.read_bytes() if unit_existed else None
    service = render_native_fim_service(runtime.resolve())
    preview, setup = render_native_fim_config(original_text, spec)
    if dry_run:
        return {
            "dry_run": True,
            "activated": False,
            "config_path": str(config),
            "config_sha256": hashlib.sha256(preview.encode()).hexdigest(),
            "candidate_setup": setup,
            "unit_path": str(unit),
            "unit_content": service,
            "unit_sha256": hashlib.sha256(service.encode()).hexdigest(),
            "runtime_config_hash": None,
            "runtime_config_hash_source": "verified /health response after service start",
            "model_sha256": expected_model_sha,
            "runtime_sha256": expected_runtime_sha,
            "editor_model_sha256": spec_sha,
            "model_bytes": expected_model_bytes,
            "memory_max_bytes": FIM_MEMORY_MAX,
        }
    if not unit.name.endswith(".service") or not config.is_file():
        raise ValueError("native FIM install requires an existing editor config and service unit")
    before_enabled = _systemctl(["is-enabled", unit.name], check=False).stdout.strip() == "enabled"
    before_active = _systemctl(["is-active", unit.name], check=False).stdout.strip() == "active"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    config_backup = config.with_name(f"{config.name}.backup-{stamp}")
    unit_backup = unit.with_name(f"{unit.name}.backup-{stamp}") if original_unit else None
    if (
        config_backup.exists()
        or config_backup.is_symlink()
        or (unit_backup is not None and (unit_backup.exists() or unit_backup.is_symlink()))
    ):
        raise ValueError("native FIM backup path already exists")
    shutil.copy2(config, config_backup)
    if unit_backup is not None:
        shutil.copy2(unit, unit_backup)
    try:
        _atomic_write(unit, service.encode())
        _systemctl(["daemon-reload"])
        _systemctl(["enable", "--now", unit.name])
        if before_active:
            _systemctl(["restart", unit.name])
        deadline = time.monotonic() + 20.0
        health = None
        while time.monotonic() < deadline:
            try:
                health = _local_json("/health")
                break
            except RuntimeError:
                time.sleep(0.25)
        if not isinstance(health, dict):
            raise RuntimeError("native FIM health verification timed out")
        runtime_config_hash = _verify_native_health(health, spec, expected_model_sha)
        slots = _local_json("/slots")
        if slots != [{"id": 0, "is_processing": False}]:
            raise ValueError("native FIM worker slot is not idle and singular")
        properties = _systemd_properties(unit.name)
        if sha(_process_executable(properties["MainPID"])) != expected_runtime_sha:
            raise ValueError(
                "running native FIM executable digest does not match the selected binary"
            )
        final_config, _ = render_native_fim_config(original_text, spec, runtime_config_hash)
        _atomic_write(config, final_config.encode())
    except Exception:
        try:
            _restore_native_files(
                config, unit, original_config, original_unit, before_active, before_enabled
            )
        except Exception:
            raise RuntimeError(
                "native FIM failed verification and the previous service could not be confirmed"
            ) from None
        raise RuntimeError(
            "native FIM service failed verification; previous config and service restored"
        ) from None
    return {
        "dry_run": False,
        "activated": True,
        "config_path": str(config),
        "config_backup": str(config_backup),
        "config_sha256": sha(config),
        "unit_path": str(unit),
        "unit_backup": str(unit_backup) if unit_backup else None,
        "unit_sha256": sha(unit),
        "runtime_config_hash": runtime_config_hash,
        "model_sha256": expected_model_sha,
        "runtime_sha256": expected_runtime_sha,
        "editor_model_sha256": spec_sha,
        "model_bytes": expected_model_bytes,
        "memory_max_bytes": FIM_MEMORY_MAX,
    }


def install(
    config: Path,
    unit: Path,
    model: Path,
    runtime: Path,
    expected_sha: str,
    model_alias: str,
    *,
    dry_run: bool,
    experimental_automatic: bool = False,
    expected_bytes: int = 397_807_232,
    protocol_version: str = "compact-next-edit-v1",
    precision: str = "Q4_K_M",
) -> dict:
    if owned_by_nix(config) or owned_by_nix(unit):
        raise RuntimeError("Nix/Home Manager owns this path; edit its source configuration")
    if not model.is_file() or sha(model) != expected_sha:
        raise ValueError("selected model artifact hash mismatch")
    if type(expected_bytes) is not int or not 1 <= expected_bytes <= 2 * 1024**3:
        raise ValueError("selected model size must be pinned within the local artifact budget")
    if model.stat().st_size != expected_bytes:
        raise ValueError("selected model artifact size mismatch")
    if not runtime.is_file():
        raise FileNotFoundError(runtime)
    original = config.read_text()
    service = render_service(model.resolve(), runtime.resolve())
    runtime_config_hash = hashlib.sha256(service.encode()).hexdigest()
    updated = render_config(
        original,
        model_revision=expected_sha,
        model_alias=model_alias,
        runtime_config_hash=runtime_config_hash,
        experimental_automatic=experimental_automatic,
        protocol_version=protocol_version,
        precision=precision,
    )
    if dry_run:
        return {
            "config": str(config),
            "unit": str(unit),
            "model_sha256": expected_sha,
            "mode": "automatic" if experimental_automatic else "manual",
            "runtime_config_hash": runtime_config_hash,
            "dry_run": True,
            "model_bytes": expected_bytes,
            "protocol_version": protocol_version,
            "precision": precision,
        }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = config.with_name(config.name + ".backup-" + stamp)
    backup.write_text(original)
    prior_unit = unit.read_text() if unit.exists() else None
    unit_backup = unit.with_name(unit.name + ".backup-" + stamp) if prior_unit else None
    if unit_backup and prior_unit:
        unit_backup.write_text(prior_unit)
    config.write_text(updated)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(service)
    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", unit.name], check=True, capture_output=True
        )
        if prior_unit != service:
            subprocess.run(
                ["systemctl", "--user", "restart", unit.name], check=True, capture_output=True
            )
    except (OSError, subprocess.CalledProcessError):
        config.write_text(original)
        if prior_unit is None:
            unit.unlink(missing_ok=True)
        else:
            unit.write_text(prior_unit)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False, capture_output=True)
        if prior_unit is not None:
            subprocess.run(
                ["systemctl", "--user", "restart", unit.name], check=False, capture_output=True
            )
        raise RuntimeError(
            "local predictor service failed to start; installer restored configuration"
        ) from None
    return {
        "config": str(config),
        "backup": str(backup),
        "unit_backup": str(unit_backup) if unit_backup else None,
        "unit": str(unit),
        "model_sha256": expected_sha,
        "mode": "automatic" if experimental_automatic else "manual",
        "runtime_config_hash": runtime_config_hash,
        "dry_run": False,
        "model_bytes": expected_bytes,
        "protocol_version": protocol_version,
        "precision": precision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--model-sha256")
    parser.add_argument("--model-bytes", type=int)
    parser.add_argument("--runtime-sha256")
    parser.add_argument("--editor-model-spec", type=Path)
    parser.add_argument("--protocol-version", choices=WIRE_VERSIONS, default=WIRE_VERSIONS[0])
    parser.add_argument("--precision", choices=("Q4_K_M", "Q5_K_M"), default="Q4_K_M")
    parser.add_argument("--model-alias")
    parser.add_argument("--prepare-native-fim-profile", action="store_true")
    parser.add_argument("--install-native-fim", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--conversion", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--experimental-automatic", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".config/nvim/lua/plugins/tabcomplete-trajectory.lua",
    )
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path.home() / ".config/systemd/user/tabcomplete-predictor.service",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.prepare_native_fim_profile and args.install_native_fim:
        parser.error("choose profile preparation or native FIM installation")
    if args.prepare_native_fim_profile:
        if any(value is None for value in (args.tokenizer, args.conversion, args.profile_output)):
            parser.error("native FIM preparation requires tokenizer, conversion and profile-output")
        print(
            json.dumps(
                prepare_native_fim_profile(
                    args.model,
                    args.tokenizer,
                    args.conversion,
                    args.profile_output,
                ),
                sort_keys=True,
            )
        )
        return
    if args.install_native_fim:
        if not all((args.model_sha256, args.runtime_sha256, args.editor_model_spec)):
            parser.error(
                "native FIM install requires model-sha256, runtime-sha256 and editor-model-spec"
            )
        if args.model_bytes is None:
            parser.error("native FIM install requires the selected GGUF model-bytes")
        result = install_native_fim(
            args.config,
            args.unit,
            args.model,
            args.runtime,
            args.editor_model_spec,
            args.model_sha256,
            args.runtime_sha256,
            args.model_bytes,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if not args.model_sha256 or not args.model_alias:
        parser.error("installation requires model-sha256 and model-alias")
    result = install(
        args.config,
        args.unit,
        args.model,
        args.runtime,
        args.model_sha256,
        args.model_alias,
        dry_run=args.dry_run,
        experimental_automatic=args.experimental_automatic,
        expected_bytes=(args.model_bytes if args.model_bytes is not None else 397_807_232),
        protocol_version=args.protocol_version,
        precision=args.precision,
    )
    print(result)


if __name__ == "__main__":
    main()
