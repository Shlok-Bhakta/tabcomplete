#!/usr/bin/env python3
"""Refuse startup unless the configured external filesystem is mounted."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DIRECTORIES = (
    "clickhouse",
    "clickhouse-user-scripts",
    "keeper",
    "keeper-image-state",
    "metastore",
    "collector-wal",
    "model-artifacts",
    "gateway-state",
    "backups",
)
MARKER = ".volume-identity.json"


@dataclass(frozen=True)
class MountInfo:
    target: Path
    source: str
    fstype: str
    uuid: str
    available_bytes: int
    writable: bool


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def inspect_mount(root: Path, mount_info_file: Path | None) -> MountInfo:
    if mount_info_file is not None:
        raw = json.loads(mount_info_file.read_text(encoding="utf-8"))
        return MountInfo(
            target=Path(raw["target"]),
            source=str(raw["source"]),
            fstype=str(raw["fstype"]),
            uuid=str(raw["uuid"]),
            available_bytes=int(raw["available_bytes"]),
            writable=bool(raw["writable"]),
        )
    ancestor = _existing_ancestor(root)
    process = subprocess.run(
        [
            "findmnt",
            "--json",
            "--bytes",
            "--target",
            str(ancestor),
            "--output",
            "TARGET,SOURCE,FSTYPE,AVAIL,UUID",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    row = json.loads(process.stdout)["filesystems"][0]
    target = Path(row["target"])
    available = shutil.disk_usage(target).free
    return MountInfo(
        target=target,
        source=str(row["source"]),
        fstype=str(row["fstype"]),
        uuid=str(row.get("uuid") or ""),
        available_bytes=available,
        writable=os.access(ancestor, os.W_OK),
    )


def validate(
    root: Path,
    mount: MountInfo,
    *,
    expected_source: str,
    expected_uuid: str,
    minimum_free_bytes: int,
) -> None:
    resolved_target = mount.target.resolve()
    resolved_root = root.resolve(strict=False)
    if resolved_target == Path("/") or not resolved_root.is_relative_to(resolved_target):
        raise RuntimeError("configured storage root is not on the expected external mount")
    if mount.source != expected_source:
        raise RuntimeError(
            f"storage source mismatch: expected {expected_source}, found {mount.source}"
        )
    if mount.uuid != expected_uuid:
        raise RuntimeError(f"storage UUID mismatch: expected {expected_uuid}, found {mount.uuid}")
    if not mount.writable:
        raise RuntimeError("external storage mount is not writable by the deployment user")
    if mount.available_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"external storage has {mount.available_bytes} free bytes; "
            f"{minimum_free_bytes} required"
        )


def marker_value(mount: MountInfo) -> dict[str, str | int]:
    return {
        "schema_version": 1,
        "source": mount.source,
        "filesystem_uuid": mount.uuid,
        "mount_target": str(mount.target),
    }


def prepare(root: Path, mount: MountInfo) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    for name in DIRECTORIES:
        (root / name).mkdir(mode=0o750, exist_ok=True)
    marker = root / MARKER
    payload = json.dumps(marker_value(mount), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".volume-identity-", dir=root)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, marker)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def check(root: Path, mount: MountInfo) -> None:
    marker = root / MARKER
    if not marker.is_file():
        raise RuntimeError("storage identity marker is missing; refusing internal-disk fallback")
    actual = json.loads(marker.read_text(encoding="utf-8"))
    expected = marker_value(mount)
    if actual != expected:
        raise RuntimeError("storage identity marker does not match the mounted filesystem")
    missing = [name for name in DIRECTORIES if not (root / name).is_dir()]
    if missing:
        raise RuntimeError(f"required storage directories are missing: {', '.join(missing)}")


def directory_bytes(path: Path) -> int:
    total = 0
    for directory, _, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(directory) / filename).stat().st_size
            except FileNotFoundError:
                pass
    return total


def component_sizes(root: Path) -> dict[str, int]:
    # The deployment user cannot read private PostgreSQL/Keeper files directly.
    # Read only this application's verified external bind; never inspect global Docker data.
    image = "postgres:16@sha256:a85daf0dbd5e79586e850e3fe4b21b796799828ad015ce2166aeb98cc24da61c"
    process = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--tmpfs",
            "/var/lib/postgresql/data:rw,size=1m",
            "--memory",
            "64m",
            "--cpus",
            "0.25",
            "--pids-limit",
            "32",
            "--mount",
            f"type=bind,source={root},target=/data,readonly",
            image,
            "du",
            "-sb",
            *[f"/data/{name}" for name in DIRECTORIES],
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    return {
        Path(path).name: int(size)
        for size, path in (line.split(None, 1) for line in process.stdout.splitlines())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "check", "health"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=100 * 2**30)
    parser.add_argument("--mount-info-file", type=Path)
    parser.add_argument("--write-health", action="store_true")
    args = parser.parse_args()
    try:
        mount = inspect_mount(args.root, args.mount_info_file)
        validate(
            args.root,
            mount,
            expected_source=args.expected_source,
            expected_uuid=args.expected_uuid,
            minimum_free_bytes=args.minimum_free_bytes,
        )
        if args.command == "prepare":
            prepare(args.root, mount)
        else:
            check(args.root, mount)
        result: dict[str, object] = {
            "checked_at": int(time.time() * 1000),
            "ok": True,
            "source": mount.source,
            "filesystem_uuid": mount.uuid,
            "mount_target": str(mount.target),
            "filesystem_type": mount.fstype,
            "available_bytes": mount.available_bytes,
        }
        if args.command == "health":
            sizes = component_sizes(args.root)
            result["components"] = sizes
            result["clickhouse_bytes"] = sizes["clickhouse"]
            result["artifact_bytes"] = sizes["model-artifacts"]
            result["collector_wal_bytes"] = sizes["collector-wal"]
            result["total_bytes"] = sum(sizes.values())
            result["high_water_bytes"] = int(
                os.environ.get("TABCOMPLETE_OBS_HIGH_WATER_BYTES", str(100 * 2**30))
            )
            result["high_water_warning"] = int(result["total_bytes"]) >= result["high_water_bytes"]
            if args.write_health:
                destination = args.root / "gateway-state" / "health.json"
                fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".health-")
                with os.fdopen(fd, "w") as handle:
                    json.dump(result, handle)
                os.chmod(temporary, 0o644)
                os.replace(temporary, destination)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as error:
        print(f"storage guard failed: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
