#!/usr/bin/env python3
"""CPU-qualify and export one hash-bound public-source Muse candidate.

Every input is a local public-source artifact. The command never downloads a
model/tokenizer, calls a provider, or starts training. The output row remains
unaccepted by itself; training must verify its sealed acceptance package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from tinycomplete.one_line.muse_acceptance_package import export_muse_acceptance_package


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("input JSON has duplicate keys")
        parsed[key] = value
    return parsed


def _read(path: Path, *, maximum: int = 4 * 1024 * 1024) -> tuple[bytes, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("input artifact is absent, unbounded, or not a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_strict_pairs)
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("input artifact is not strict JSON") from None
    return payload, value


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent
    if parent.is_symlink():
        raise ValueError("output directory cannot be a symlink")
    if not parent.exists():
        parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(parent, 0o700)
    status = parent.stat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise ValueError("output directory ownership is invalid")
    if stat.S_IMODE(status.st_mode) & 0o077:
        raise ValueError("output directory is not owner-only")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".muse-row-", dir=parent)
        temporary = Path(name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise ValueError("output row already exists")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _tokenizer(directory: Path, expected_sha256: str) -> Any:
    tokenizer_json = directory / "tokenizer.json"
    if tokenizer_json.is_symlink() or not tokenizer_json.is_file():
        raise ValueError("local tokenizer.json is absent")
    actual = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("local tokenizer hash differs from the frozen plan")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            directory,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception:
        raise ValueError("pinned local tokenizer could not be loaded") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-row", type=Path, required=True)
    parser.add_argument("--objective-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--role-evidence", type=Path, required=True)
    parser.add_argument("--file-license-review", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json-sha256", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-row", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _candidate_bytes, candidate = _read(args.candidate)
        _source_bytes, source = _read(args.source_row)
        objective_bytes, objective = _read(args.objective_manifest)
        split_bytes, split = _read(args.split_manifest)
        role_bytes, role = _read(args.role_evidence)
        license_bytes, license_review = _read(args.file_license_review)
        if not all(
            isinstance(item, dict)
            for item in (candidate, source, objective, split, role, license_review)
        ):
            raise ValueError("input artifact must contain a JSON object")
        tokenizer = _tokenizer(args.tokenizer_dir, args.tokenizer_json_sha256)
        result = export_muse_acceptance_package(
            package_root=args.package_root,
            candidate=candidate,
            source_row=source,
            tokenizer=tokenizer,
            objective_manifest=objective_bytes,
            split_manifest=split_bytes,
            role_evidence=role_bytes,
            file_license_review=license_bytes,
        )
        if not result.accepted or result.acceptance_package_ref is None:
            print(json.dumps({"accepted": False, "reason": result.reason}, sort_keys=True))
            return 2
        metadata = source.get("authoring_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("source metadata is absent")
        row = {
            **(result.candidate_row or candidate),
            "source_path": metadata.get("source_path"),
            "source_license_sha256": metadata.get("license_sha256"),
            "source_sha256": metadata.get("source_sha256"),
            "acceptance_package_ref": result.acceptance_package_ref,
        }
        _write_private_json(args.output_row, row)
        print(
            json.dumps(
                {
                    "accepted": True,
                    "candidate_id": candidate["id"],
                    "package_manifest_sha256": result.acceptance_package_ref["manifest_sha256"],
                    "row_sha256": hashlib.sha256(args.output_row.read_bytes()).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return 0
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        print("Muse acceptance export failed: invalid or unverified local evidence")
        return 2


if __name__ == "__main__":
    sys.exit(main())
