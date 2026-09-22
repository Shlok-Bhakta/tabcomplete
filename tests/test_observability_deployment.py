from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).parents[1]
GUARD = REPOSITORY / "infra" / "observability" / "scripts" / "storage_guard.py"


def _run_guard(tmp_path: Path, command: str, *, uuid: str = "volume-uuid"):
    mount = tmp_path / "external"
    mount.mkdir(exist_ok=True)
    root = mount / "docker" / "tabcomplete-observability"
    probe = tmp_path / "mount.json"
    probe.write_text(
        json.dumps(
            {
                "target": str(mount),
                "source": "/dev/test-external",
                "fstype": "ext4",
                "uuid": uuid,
                "available_bytes": 4 * 2**40,
                "writable": True,
            }
        ),
        encoding="utf-8",
    )
    process = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            command,
            "--root",
            str(root),
            "--expected-source",
            "/dev/test-external",
            "--expected-uuid",
            "volume-uuid",
            "--mount-info-file",
            str(probe),
            "--minimum-free-bytes",
            str(100 * 2**30),
        ],
        text=True,
        capture_output=True,
    )
    return root, process


def test_missing_volume_marker_fails_without_fallback(tmp_path: Path) -> None:
    root, process = _run_guard(tmp_path, "check")
    assert process.returncode != 0
    assert "identity marker" in process.stderr
    assert not root.exists()


def test_prepare_and_check_storage_with_disposable_mount(tmp_path: Path) -> None:
    root, prepared = _run_guard(tmp_path, "prepare")
    assert prepared.returncode == 0, prepared.stderr
    checked = _run_guard(tmp_path, "check")[1]
    assert checked.returncode == 0, checked.stderr
    expected = {
        "clickhouse",
        "clickhouse-user-scripts",
        "keeper",
        "metastore",
        "collector-wal",
        "model-artifacts",
        "gateway-state",
        "backups",
    }
    assert expected <= {path.name for path in root.iterdir() if path.is_dir()}


def test_wrong_volume_identity_fails(tmp_path: Path) -> None:
    _, process = _run_guard(tmp_path, "prepare", uuid="wrong-volume")
    assert process.returncode != 0
    assert "UUID" in process.stderr


def test_rendered_compose_has_only_explicit_bind_storage_and_tailnet_ports() -> None:
    compose_path = REPOSITORY / "infra" / "observability" / "pours" / "deployment" / "compose.yaml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    assert "volumes" not in compose
    for service in compose["services"].values():
        for volume in service.get("volumes", []):
            if isinstance(volume, dict) and str(volume.get("target", "")).startswith("/var/lib"):
                assert volume["type"] == "bind"
                assert volume["bind"]["create_host_path"] is False
        for port in service.get("ports", []):
            assert not str(port).startswith("0.0.0.0:")
            assert "TABCOMPLETE_OBS_TAILSCALE_IP" in str(port)
