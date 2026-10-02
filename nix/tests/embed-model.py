#!/usr/bin/env python3
"""Deterministic appender regression with small synthetic files, no weights."""

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

builder = Path(__file__).resolve().parents[1] / "embed-model.py"
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    engine = root / "engine"
    model = root / "model"
    engine.write_bytes(b"\x7fELF" + b"e" * 4100)
    model.write_bytes(b"GGUFtest")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    for variant, alias, protocol, cap in [
        ("qwen", "q25", "single-line-edit-v1", 64),
        ("sweep", "sweep", "sweep-full-file-v1", 192),
    ]:
        output = root / variant
        command = [
            sys.executable,
            str(builder),
            "--engine",
            str(engine),
            "--model",
            str(model),
            "--sha256",
            digest,
            "--variant",
            variant,
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True)
        data = output.read_bytes()
        assert data[-16:] == b"TABCOMPLETEGGUF1"
        length = struct.unpack("<I", data[-20:-16])[0]
        metadata = json.loads(data[-20 - length : -20])
        assert metadata["offset"] == 8192
        assert data[: engine.stat().st_size] == engine.read_bytes()
        assert data[8192:8200] == model.read_bytes()
        assert metadata["length"] == 8 and metadata["sha256"] == digest
        assert (metadata["alias"], metadata["protocol"], metadata["output_tokens"]) == (
            alias,
            protocol,
            cap,
        )
        assert metadata["microbatch_size"] == 64
        assert output.stat().st_mode & 0o111
        assert subprocess.run(command, capture_output=True).returncode != 0
    bad = root / "bad"
    command = [
        sys.executable,
        str(builder),
        "--engine",
        str(engine),
        "--model",
        str(model),
        "--sha256",
        "0" * 64,
        "--variant",
        "qwen",
        "--output",
        str(bad),
    ]
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not bad.exists() and not bad.with_suffix(".partial").exists()
print("embedded appender: two profiles, alignment, hash, atomic failure PASS")
