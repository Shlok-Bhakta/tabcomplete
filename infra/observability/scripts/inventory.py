"""Write an uncommitted deployment inventory without environment values or credentials."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from bootstrap_signoz import atomic_replace, read_environment
from storage_guard import check, inspect_mount, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    env = read_environment(args.directory / ".env")
    root = Path(env["TABCOMPLETE_OBS_DATA_ROOT"])
    mount = inspect_mount(root, None)
    validate(
        root,
        mount,
        expected_source=env["TABCOMPLETE_OBS_VOLUME_SOURCE"],
        expected_uuid=env["TABCOMPLETE_OBS_VOLUME_UUID"],
        minimum_free_bytes=100 * 2**30,
    )
    check(root, mount)
    ids = subprocess.check_output(
        [
            "docker",
            "ps",
            "-q",
            "--filter",
            "label=com.docker.compose.project=tabcomplete-observability",
        ],
        text=True,
    ).split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
    inventory = []
    for item in containers:
        mounts = [
            {"type": m["Type"], "source": m["Source"], "destination": m["Destination"]}
            for m in item["Mounts"]
        ]
        if any(m["type"] == "volume" for m in mounts):
            raise RuntimeError("Unexpected anonymous or named volume")
        ports = item["NetworkSettings"]["Ports"]
        for bindings in ports.values():
            for binding in bindings or []:
                if binding["HostIp"] != env["TABCOMPLETE_OBS_TAILSCALE_IP"]:
                    raise RuntimeError("Unexpected host port binding")
        inventory.append(
            {
                "name": item["Name"],
                "image": item["Config"]["Image"],
                "image_id": item["Image"],
                "state": item["State"]["Status"],
                "health": item["State"].get("Health", {}).get("Status", "no_healthcheck"),
                "mounts": mounts,
                "ports": ports,
            }
        )
    report = {
        "schema_version": 1,
        "architecture": platform.machine(),
        "os": platform.platform(),
        "storage": {
            "root": str(root),
            "source": mount.source,
            "filesystem_uuid": mount.uuid,
            "mount_target": str(mount.target),
            "free_bytes": mount.available_bytes,
        },
        "tailscale_ip": env["TABCOMPLETE_OBS_TAILSCALE_IP"],
        "containers": inventory,
        "secrets_omitted": True,
    }
    atomic_replace(args.directory / "deployment-manifest.json", json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "containers_checked": len(inventory),
                "explicit_mounts": True,
                "tailscale_only_bindings": True,
                "manifest": "deployment-manifest.json",
            }
        )
    )


if __name__ == "__main__":
    main()
