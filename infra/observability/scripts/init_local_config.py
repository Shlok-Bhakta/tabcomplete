#!/usr/bin/env python3
"""Create restricted machine-local deployment configuration without printing secrets."""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path


def write_restricted(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--volume-source", required=True)
    parser.add_argument("--volume-uuid", required=True)
    parser.add_argument("--tailscale-ip", required=True)
    parser.add_argument("--kiwi-dns", required=True)
    parser.add_argument("--crabcake-dns", required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    environment = args.directory / ".env"
    admin = args.directory / ".admin.env"
    upload = args.directory / ".artifact-upload-token"
    if environment.exists() or admin.exists() or upload.exists():
        raise SystemExit("local configuration already exists; refusing to overwrite")
    allowed = ",".join(
        [
            "kiwi",
            args.kiwi_dns.rstrip("."),
            args.tailscale_ip,
            args.crabcake_dns.rstrip("."),
            "localhost",
            "127.0.0.1",
        ]
    )
    pending_key = "pending-" + secrets.token_urlsafe(32)
    upload_token = secrets.token_urlsafe(48)
    write_restricted(
        environment,
        "\n".join(
            [
                f"TABCOMPLETE_OBS_DATA_ROOT={args.storage_root}",
                f"TABCOMPLETE_OBS_VOLUME_SOURCE={args.volume_source}",
                f"TABCOMPLETE_OBS_VOLUME_UUID={args.volume_uuid}",
                f"TABCOMPLETE_OBS_TAILSCALE_IP={args.tailscale_ip}",
                "TABCOMPLETE_OBS_GATEWAY_PORT=9090",
                "TABCOMPLETE_OBS_SIGNOZ_PORT=8080",
                "TABCOMPLETE_OBS_MCP_PORT=8000",
                "TABCOMPLETE_OBS_OTLP_GRPC_PORT=4317",
                "TABCOMPLETE_OBS_OTLP_HTTP_PORT=4318",
                f"TABCOMPLETE_OBS_ALLOWED_HOSTS={allowed}",
                f"SIGNOZ_API_KEY={pending_key}",
                f"TABCOMPLETE_OBS_POSTGRES_PASSWORD={secrets.token_hex(32)}",
                f"TABCOMPLETE_ARTIFACT_UPLOAD_TOKEN={upload_token}",
                "",
            ]
        ),
    )
    write_restricted(
        admin,
        "\n".join(
            [
                "SIGNOZ_ADMIN_EMAIL=admin@tabcomplete.local",
                f"SIGNOZ_ADMIN_PASSWORD={secrets.token_urlsafe(32)}",
                "SIGNOZ_ADMIN_NAME=TabComplete Administrator",
                "SIGNOZ_ORGANIZATION=TabComplete",
                "",
            ]
        ),
    )
    write_restricted(upload, upload_token + "\n")
    print(f"created restricted local configuration in {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
