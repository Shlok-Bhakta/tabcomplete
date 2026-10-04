#!/usr/bin/env python3
"""Synthetic footer tests for the opt-in FIM executable appender."""

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

builder = Path(__file__).resolve().parents[1] / "embed-fim-model.py"
MAGIC = b"TABCOMPLETEGGUF1"
MAX_METADATA = 2 * 1024 * 1024
TOKEN_IDS = {
    "eos": (151643, "<|endoftext|>"),
    "fim_prefix": (151659, "<|fim_prefix|>"),
    "fim_suffix": (151661, "<|fim_suffix|>"),
    "fim_middle": (151660, "<|fim_middle|>"),
}


def serving_profile():
    vocab_ids = list(range(151662))
    vocab_digest = hashlib.sha256("".join(f"{value}\n" for value in vocab_ids).encode()).hexdigest()
    tokenizer = {
        "tokenizer_id": "synthetic/q25-fim",
        "tokenizer_revision": "synthetic-revision",
        "tokenizer_sha256": "a" * 64,
        "tokenizer_contract_sha256": "",
        "tokenizer_vocab_size": len(vocab_ids),
        "tokenizer_vocab_ids_sha256": vocab_digest,
        "tokenizer_vocab_ids": vocab_ids,
        "eos_id": TOKEN_IDS["eos"][0],
        "fim_prefix_id": TOKEN_IDS["fim_prefix"][0],
        "fim_suffix_id": TOKEN_IDS["fim_suffix"][0],
        "fim_middle_id": TOKEN_IDS["fim_middle"][0],
        "completion_mode": "remaining_logical_line_after_utf8_cursor",
        "special_tokens": [
            {"id": token_id, "spelling": spelling}
            for token_id, spelling in sorted(TOKEN_IDS.values())
        ],
    }
    tokenizer["tokenizer_contract_sha256"] = tokenizer_contract_digest(tokenizer)
    return {"artifact_manifest_sha256": "b" * 64, "tokenizer": tokenizer}


def tokenizer_contract_digest(tokenizer):
    canonical = (
        "q25-fim-tokenizer-contract-v1\n"
        f"{tokenizer['tokenizer_id']}\n"
        f"{tokenizer['tokenizer_revision']}\n"
        f"{tokenizer['tokenizer_sha256']}\n"
        f"{tokenizer['eos_id']}\n"
        f"{tokenizer['fim_prefix_id']}\n"
        f"{tokenizer['fim_suffix_id']}\n"
        f"{tokenizer['fim_middle_id']}\n"
        f"{tokenizer['completion_mode']}\n"
        f"{tokenizer['tokenizer_vocab_size']}\n"
        f"{tokenizer['tokenizer_vocab_ids_sha256']}\n"
    )
    canonical += "".join(
        f"{token['id']}\t{token['spelling']}\n"
        for token in sorted(tokenizer["special_tokens"], key=lambda item: item["id"])
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def profile_wrapper(model_bytes):
    return {
        "schema": "tabcomplete-q25-fim-embedded-profile-v1",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "serving_profile": serving_profile(),
    }


def write_profile(path, wrapper):
    path.write_text(json.dumps(wrapper, separators=(",", ":")), encoding="utf-8")


def command(engine, model, profile, output):
    return [
        sys.executable,
        str(builder),
        "--engine",
        str(engine),
        "--model",
        str(model),
        "--profile",
        str(profile),
        "--output",
        str(output),
    ]


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    engine = root / "engine"
    model = root / "model.gguf"
    profile = root / "profile.json"
    output = root / "candidate"
    engine.write_bytes(b"\x7fELF" + b"e" * 4100)
    model.write_bytes(b"GGUFsynthetic-q25-fim-model")
    wrapper = profile_wrapper(model.read_bytes())
    write_profile(profile, wrapper)

    subprocess.run(command(engine, model, profile, output), check=True, capture_output=True)
    data = output.read_bytes()
    assert data[-16:] == MAGIC
    metadata_size = struct.unpack("<I", data[-20:-16])[0]
    assert 4096 < metadata_size <= MAX_METADATA
    metadata_start = len(data) - 20 - metadata_size
    metadata = json.loads(data[metadata_start : metadata_start + metadata_size])
    assert data[: engine.stat().st_size] == engine.read_bytes()
    assert metadata["alias"] == "q25-fim"
    assert metadata["protocol"] == "q25-fim-line-completion-v1"
    assert metadata["output_tokens"] == 96
    assert metadata["context_layout"] == "q25-fim-psm-bounded-v2"
    assert metadata["fim_profile"] == wrapper["serving_profile"]
    assert data[metadata["offset"] : metadata["offset"] + metadata["length"]] == model.read_bytes()
    assert (
        hashlib.sha256(
            data[metadata["offset"] : metadata["offset"] + metadata["length"]]
        ).hexdigest()
        == metadata["sha256"]
    )
    assert output.stat().st_mode & 0o111
    assert not output.with_suffix(".gguf").exists()

    bad_profile = root / "bad-profile.json"
    bad = profile_wrapper(model.read_bytes())
    bad["serving_profile"]["tokenizer"]["tokenizer_contract_sha256"] = "0" * 64
    write_profile(bad_profile, bad)
    bad_output = root / "bad-profile-output"
    assert (
        subprocess.run(
            command(engine, model, bad_profile, bad_output), capture_output=True
        ).returncode
        != 0
    )
    assert not bad_output.exists() and not bad_output.with_suffix(".partial").exists()

    bad_inventory = profile_wrapper(model.read_bytes())
    bad_inventory["serving_profile"]["tokenizer"]["tokenizer_vocab_ids"][1] = 0
    bad_inventory_path = root / "bad-inventory.json"
    write_profile(bad_inventory_path, bad_inventory)
    bad_inventory_output = root / "bad-inventory-output"
    assert (
        subprocess.run(
            command(engine, model, bad_inventory_path, bad_inventory_output), capture_output=True
        ).returncode
        != 0
    )
    assert not bad_inventory_output.exists()

    bad_order = profile_wrapper(model.read_bytes())
    bad_order["serving_profile"]["tokenizer"]["special_tokens"].reverse()
    bad_order_path = root / "bad-control-token-order.json"
    write_profile(bad_order_path, bad_order)
    bad_order_output = root / "bad-control-token-order-output"
    assert (
        subprocess.run(
            command(engine, model, bad_order_path, bad_order_output), capture_output=True
        ).returncode
        != 0
    )
    assert not bad_order_output.exists()

    tampered_model = root / "tampered.gguf"
    tampered_model.write_bytes(model.read_bytes() + b"x")
    tampered_output = root / "tampered-output"
    assert (
        subprocess.run(
            command(engine, tampered_model, profile, tampered_output), capture_output=True
        ).returncode
        != 0
    )
    assert not tampered_output.exists() and not tampered_output.with_suffix(".partial").exists()

    oversized_profile = root / "oversized-profile.json"
    oversized_profile.write_bytes(b" " * (MAX_METADATA + 1))
    oversized_output = root / "oversized-output"
    assert (
        subprocess.run(
            command(engine, model, oversized_profile, oversized_output), capture_output=True
        ).returncode
        != 0
    )
    assert not oversized_output.exists()

    duplicate_profile = root / "duplicate-profile.json"
    duplicate_profile.write_text(
        '{"schema":"tabcomplete-q25-fim-embedded-profile-v1",'
        '"schema":"tabcomplete-q25-fim-embedded-profile-v1"}',
        encoding="utf-8",
    )
    duplicate_output = root / "duplicate-output"
    assert (
        subprocess.run(
            command(engine, model, duplicate_profile, duplicate_output), capture_output=True
        ).returncode
        != 0
    )
    assert not duplicate_output.exists()

    collision_output = root / "collision-output"
    preexisting_partial = collision_output.with_suffix(".partial")
    preexisting_partial.write_bytes(b"keep-existing-file")
    assert (
        subprocess.run(
            command(engine, model, profile, collision_output), capture_output=True
        ).returncode
        != 0
    )
    assert preexisting_partial.read_bytes() == b"keep-existing-file"
    assert not collision_output.exists()

    over_metadata = profile_wrapper(model.read_bytes())
    over_tokenizer = over_metadata["serving_profile"]["tokenizer"]
    oversized_special = {"id": 1, "spelling": "x"}
    over_tokenizer["special_tokens"].append(oversized_special)
    over_tokenizer["special_tokens"].sort(key=lambda item: item["id"])
    over_tokenizer["tokenizer_contract_sha256"] = tokenizer_contract_digest(over_tokenizer)
    calibration_profile = root / "calibration-profile.json"
    write_profile(calibration_profile, over_metadata)
    calibration_output = root / "calibration-output"
    subprocess.run(
        command(engine, model, calibration_profile, calibration_output),
        check=True,
        capture_output=True,
    )
    calibration_data = calibration_output.read_bytes()
    calibration_size = struct.unpack("<I", calibration_data[-20:-16])[0]
    calibration_output.unlink()
    padding = MAX_METADATA - calibration_size + 1
    oversized_special["spelling"] = "x" * (padding + 1)
    over_tokenizer["tokenizer_contract_sha256"] = tokenizer_contract_digest(over_tokenizer)
    over_metadata_path = root / "over-metadata.json"
    write_profile(over_metadata_path, over_metadata)
    assert over_metadata_path.stat().st_size <= MAX_METADATA
    over_metadata_output = root / "over-metadata-output"
    assert (
        subprocess.run(
            command(engine, model, over_metadata_path, over_metadata_output),
            capture_output=True,
        ).returncode
        != 0
    )
    assert (
        not over_metadata_output.exists()
        and not over_metadata_output.with_suffix(".partial").exists()
    )

print("FIM embedded appender: identity, full vocabulary, tampering, caps, no sidecar PASS")
