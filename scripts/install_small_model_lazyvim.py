"""Install the selected local LazyVim predictor without replacing mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "tools/trajectory_collector/nvim"
DEFAULT_RUNTIME = ROOT.parent / "tabcomplete/outputs/tools/llama.cpp/build/bin/llama-server"
WIRE_VERSIONS = ("compact-next-edit-v1", "single-line-edit-v1")


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
    for name, value in {"registry.json": registry, "editor-model.json": spec}.items():
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
    parser.add_argument("--model-bytes", type=int, default=397_807_232)
    parser.add_argument("--protocol-version", choices=WIRE_VERSIONS, default=WIRE_VERSIONS[0])
    parser.add_argument("--precision", choices=("Q4_K_M", "Q5_K_M"), default="Q4_K_M")
    parser.add_argument("--model-alias")
    parser.add_argument("--prepare-native-fim-profile", action="store_true")
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
        expected_bytes=args.model_bytes,
        protocol_version=args.protocol_version,
        precision=args.precision,
    )
    print(result)


if __name__ == "__main__":
    main()
