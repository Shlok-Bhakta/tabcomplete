#!/usr/bin/env python3
"""Bounded, read-only monitoring CLI; maintenance is explicitly local."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    client_path = Path(
        os.environ.get(
            "TABCOMPLETE_OBSERVABILITY_CLIENT_CONFIG",
            str(Path.home() / ".config/tabcomplete-observability/client.json"),
        )
    )
    client = json.loads(client_path.read_text()) if client_path.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=os.environ.get("TABCOMPLETE_MONITOR_URL", client.get("gateway"))
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "runs", "run", "failures", "trace", "request", "compare", "doctor"):
        child = sub.add_parser(command)
        child.add_argument("--json", action="store_true")
        child.add_argument("--since", default="24h")
        child.add_argument("--limit", type=int, default=100)
        child.add_argument("--cursor", default="0")
        if command in ("run", "trace", "request"):
            child.add_argument("identifier")
        if command == "failures":
            child.add_argument("--run", required=True)
        if command == "compare":
            child.add_argument("left")
            child.add_argument("right")
    importer = sub.add_parser("import-offline")
    importer.add_argument("path", type=Path)
    sync = sub.add_parser("sync-artifacts")
    sync.add_argument("path", type=Path)
    importer.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            os.environ.get("TABCOMPLETE_IMPORT_LEDGER", ".tabcomplete-observability/imports.sqlite")
        ),
    )
    provision = sub.add_parser("provision")
    provision.add_argument("--directory", type=Path, required=True)
    provision.add_argument("--signoz-url", required=True)
    args = parser.parse_args(argv)
    result: dict = {}
    try:
        if args.command == "sync-artifacts":
            from tinycomplete.observability.artifacts import sync_artifacts

            result = {"schema_version": 1, **sync_artifacts(args.path)}
            print(json.dumps(result, sort_keys=True))
            return 3 if result["failed"] else 0
        if args.command == "import-offline":
            if client.get("import_host"):
                if args.path.stat().st_size > 256 * 2**20:
                    raise ValueError("bundle exceeds limit")
                directory = client["deployment_directory"]
                command = shlex.join(
                    [
                        directory + "/.venv/bin/python",
                        directory + "/scripts/import_offline_remote.py",
                        "--directory",
                        directory,
                    ]
                )
                imported = subprocess.run(
                    ["ssh", client["import_host"], command],
                    input=args.path.read_bytes(),
                    capture_output=True,
                    timeout=120,
                )
                if imported.returncode:
                    raise RuntimeError("remote import failed")
                print(json.dumps(json.loads(imported.stdout), sort_keys=True, indent=2))
                return 0
            from tinycomplete.observability.offline import import_bundle

            result = import_bundle(
                args.path,
                args.ledger,
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318"),
            )
        elif args.command == "provision":
            return subprocess.call(
                [
                    sys.executable,
                    str(ROOT / "infra/observability/scripts/provision.py"),
                    "--directory",
                    str(args.directory),
                    "--url",
                    args.signoz_url,
                ]
            )
        else:
            if not args.url:
                parser.error("set TABCOMPLETE_MONITOR_URL or --url to the private-tailnet gateway")
            path = {"status": "status", "doctor": "status", "runs": "runs"}.get(args.command)

            def encode(value: str) -> str:
                return quote(value, safe="")

            if args.command in ("run", "trace", "request"):
                path = (
                    {"run": "runs", "trace": "traces", "request": "requests"}[args.command]
                    + "/"
                    + encode(args.identifier)
                )
            elif args.command == "failures":
                path = "runs/" + encode(args.run) + "/failures"
            elif args.command == "compare":
                path = "compare/" + encode(args.left) + "/" + encode(args.right)
            response = httpx.get(
                args.url.rstrip("/") + "/api/v1/" + str(path),
                params={"since": args.since, "limit": args.limit, "cursor": args.cursor},
                timeout=12,
                follow_redirects=False,
            )
            if response.status_code != 200:
                print(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "error": "monitor_query_failed",
                            "http_status": response.status_code,
                        }
                    )
                )
                return 4 if response.status_code == 404 else 3
            result = response.json()
            if args.command == "doctor":
                infra_response = httpx.get(
                    args.url.rstrip("/") + "/api/v1/infrastructure", timeout=12
                )
                infra_response.raise_for_status()
                infrastructure = infra_response.json()
                result["checks"] = {
                    "gateway": "reachable",
                    "signoz": result.get("status"),
                    "storage": infrastructure.get("storage", {}),
                    "collector": infrastructure.get("collectors", {}),
                    "mcp": "use observability_verify.py for a known-run tool query",
                }
            if args.command in ("run", "trace", "request") and not result.get("data"):
                print(json.dumps(result, sort_keys=True))
                return 4
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, httpx.HTTPError, RuntimeError) as exc:
        # Exception messages may contain authenticated URLs or payloads.
        print(json.dumps({"schema_version": 1, "error": type(exc).__name__}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
