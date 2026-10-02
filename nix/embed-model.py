#!/usr/bin/env python3
"""Append a verified private GGUF to an already finalized executable.

No weights enter Rust compilation, public source, downloads, or runtime extraction.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

MAGIC = b"TABCOMPLETEGGUF1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--variant", choices=("qwen", "sweep"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 4096 <= args.engine.stat().st_size <= 64 * 1024 * 1024:
        parser.error("engine size outside bounds")
    if not 4 <= args.model.stat().st_size <= 1024 * 1024 * 1024:
        parser.error("model size outside bounds")
    if args.engine.open("rb").read(4) != b"\x7fELF":
        parser.error("engine must be a finalized ELF executable")
    metadata = dict(
        version=1,
        sha256=args.sha256,
        alias="q25" if args.variant == "qwen" else "sweep",
        protocol="single-line-edit-v1" if args.variant == "qwen" else "sweep-full-file-v1",
        output_tokens=64 if args.variant == "qwen" else 192,
        context_size=2304,
        input_tokens=1024,
        batch_size=256,
        microbatch_size=64,
        threads=4,
        cache_type="f16",
        context_layout="cursor-last-v1",
    )
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    try:
        with (
            args.engine.open("rb") as engine,
            args.model.open("rb") as model,
            temporary.open("xb") as output,
        ):
            shutil.copyfileobj(engine, output, 65536)
            offset = (output.tell() + 4095) // 4096 * 4096
            if offset > 64 * 1024 * 1024 or offset < 4096:
                raise ValueError("embedded executable offset outside bounds")
            output.write(b"\0" * (offset - output.tell()))
            digest = hashlib.sha256()
            length = 0
            header = model.read(4)
            if header != b"GGUF":
                raise ValueError("model is not GGUF")
            model.seek(0)
            while chunk := model.read(65536):
                output.write(chunk)
                digest.update(chunk)
                length += len(chunk)
            if not 4 <= length <= 1024 * 1024 * 1024 or digest.hexdigest() != args.sha256:
                raise ValueError("embedded model identity mismatch")
            metadata.update(offset=offset, length=length)
            payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            if len(payload) > 4096:
                raise ValueError("embedded metadata outside bounds")
            output.write(payload)
            output.write(struct.pack("<I", len(payload)))
            output.write(MAGIC)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o755)
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
