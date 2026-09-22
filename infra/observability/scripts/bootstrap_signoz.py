#!/usr/bin/env python3
"""Bootstrap SigNoz without exposing administrator or service credentials."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def read_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    access_token: str | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if access_token is not None:
        headers["Authorization"] = f"Bearer {access_token}"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"SigNoz {path} returned HTTP {error.code}") from None
    parsed = json.loads(content) if content else None
    if isinstance(parsed, dict) and parsed.get("status") == "error":
        raise RuntimeError(f"SigNoz {path} returned an application error")
    if isinstance(parsed, dict) and parsed.get("status") == "success" and "data" in parsed:
        return parsed.get("data")
    return parsed


def atomic_replace(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def replace_environment_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    output: list[str] = []
    for line in lines:
        if line.startswith(key + "="):
            output.append(f"{key}={value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key}={value}")
    atomic_replace(path, "\n".join(output) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    environment_path = args.directory / ".env"
    admin_path = args.directory / ".admin.env"
    state_path = args.directory / ".bootstrap-state.json"
    if state_path.exists():
        print("SigNoz bootstrap already completed")
        return 0

    admin = read_environment(admin_path)
    registration = request_json(
        args.url,
        "/api/v1/register",
        method="POST",
        body={
            "name": admin["SIGNOZ_ADMIN_NAME"],
            "email": admin["SIGNOZ_ADMIN_EMAIL"],
            "password": admin["SIGNOZ_ADMIN_PASSWORD"],
            "orgDisplayName": admin["SIGNOZ_ORGANIZATION"],
            "orgName": "tabcomplete",
        },
    )
    if not isinstance(registration, dict):
        raise RuntimeError("SigNoz registration returned an unexpected response")
    org_id = str(registration.get("orgId") or registration.get("orgID") or "")
    if not org_id:
        raise RuntimeError("SigNoz registration did not return an organization ID")
    session = request_json(
        args.url,
        "/api/v2/sessions/email_password",
        method="POST",
        body={
            "email": admin["SIGNOZ_ADMIN_EMAIL"],
            "password": admin["SIGNOZ_ADMIN_PASSWORD"],
            "orgId": org_id,
        },
    )
    if not isinstance(session, dict) or not session.get("accessToken"):
        raise RuntimeError("SigNoz login did not return an access token")
    token = str(session["accessToken"])
    accounts = request_json(args.url, "/api/v1/service_accounts", access_token=token)
    account = next((item for item in accounts if item.get("name") == "tabcomplete-monitor"), None)
    if account is None:
        created = request_json(
            args.url,
            "/api/v1/service_accounts",
            method="POST",
            body={"name": "tabcomplete-monitor"},
            access_token=token,
        )
        account_id = str(created["id"])
    else:
        account_id = str(account["id"])
    roles = request_json(args.url, "/api/v1/roles", access_token=token)
    viewer = next((item for item in roles if item.get("name") == "signoz-viewer"), None)
    if viewer is None:
        raise RuntimeError("SigNoz viewer role was not found")
    current_roles = request_json(
        args.url, f"/api/v1/service_accounts/{account_id}/roles", access_token=token
    )
    if not any(item.get("id") == viewer["id"] for item in current_roles):
        request_json(
            args.url,
            "/api/v1/service_account_roles",
            method="POST",
            body={"serviceAccountId": account_id, "roleId": viewer["id"]},
            access_token=token,
        )
    key = request_json(
        args.url,
        f"/api/v1/service_accounts/{account_id}/keys",
        method="POST",
        body={"name": "tabcomplete-monitor", "expiresAt": 0},
        access_token=token,
    )
    if not isinstance(key, dict) or not key.get("key"):
        raise RuntimeError("SigNoz did not return the new service account key")
    replace_environment_value(environment_path, "SIGNOZ_API_KEY", str(key["key"]))
    atomic_replace(
        state_path,
        json.dumps(
            {
                "schema_version": 1,
                "organization_id": org_id,
                "service_account_id": account_id,
                "service_account_key_id": str(key.get("id", "")),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print("SigNoz administrator and read-only service identity are configured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
