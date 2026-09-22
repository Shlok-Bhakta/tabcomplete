#!/usr/bin/env python3
"""Idempotent local maintenance using SigNoz v0.142.1's dashboard v2 API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bootstrap_signoz import atomic_replace, read_environment, request_json


def dashboard(definition: dict) -> dict:
    panels, items = {}, []
    for i, panel in enumerate(definition["panels"]):
        fields = [
            {"name": key, "fieldContext": "resource" if key == "host.name" else "span"}
            for key in ["timestamp", "name", "trace_id", *panel["fields"]]
        ]
        query = {
            "name": "A",
            "signal": "traces",
            "filter": {"expression": "service.name = 'tabcomplete' AND (" + panel["filter"] + ")"},
            "selectFields": fields,
            "order": [{"key": {"name": "timestamp"}, "direction": "desc"}],
            "limit": 100,
        }
        panels[panel["id"]] = {
            "kind": "Panel",
            "spec": {
                "display": {"name": panel["title"]},
                "plugin": {"kind": "signoz/ListPanel", "spec": {"selectFields": fields}},
                "queries": [
                    {
                        "kind": "raw",
                        "spec": {"plugin": {"kind": "signoz/BuilderQuery", "spec": query}},
                    }
                ],
            },
        }
        items.append(
            {
                "x": 0,
                "y": i * 6,
                "width": 12,
                "height": 6,
                "content": {"$ref": "#/spec/panels/" + panel["id"]},
            }
        )
    return {
        "schemaVersion": "v6",
        "name": definition["name"],
        "tags": [],
        "spec": {
            "display": {"name": definition["title"], "description": definition["description"]},
            "duration": "24h",
            "variables": [],
            "panels": panels,
            "layouts": [{"kind": "Grid", "spec": {"items": items}}],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    admin = read_environment(args.directory / ".admin.env")
    state = json.loads((args.directory / ".bootstrap-state.json").read_text())
    token = request_json(
        args.url,
        "/api/v2/sessions/email_password",
        method="POST",
        body={
            "email": admin["SIGNOZ_ADMIN_EMAIL"],
            "password": admin["SIGNOZ_ADMIN_PASSWORD"],
            "orgId": state["organization_id"],
        },
    )["accessToken"]
    definition = json.loads((args.directory / "dashboards/runs-and-models.json").read_text())
    body = dashboard(definition)
    # Stable name lookup is supported by v2; IDs remain stable across updates.
    listing = request_json(args.url, "/api/v2/dashboards", access_token=token)
    entries = (
        listing
        if isinstance(listing, list)
        else listing.get("dashboards", listing.get("items", []))
    )
    existing = next((d for d in entries if d.get("name") == body["name"]), None)
    if existing:
        updated = body
        created = request_json(
            args.url,
            "/api/v2/dashboards/" + existing["id"],
            method="PUT",
            body=updated,
            access_token=token,
        )
    else:
        created = request_json(
            args.url, "/api/v2/dashboards", method="POST", body=body, access_token=token
        )
    panel_results = {}
    for name, panel in body["spec"]["panels"].items():
        query = panel["spec"]["queries"][0]["spec"]["plugin"]["spec"]
        end = int(time.time() * 1000)
        request_json(
            args.url,
            "/api/v5/query_range",
            method="POST",
            access_token=token,
            body={
                "start": end - 86400000,
                "end": end,
                "requestType": "raw",
                "compositeQuery": {"queries": [{"type": "builder_query", "spec": query}]},
            },
        )
        panel_results[name] = "query_verified"
    retention = {}
    policy = json.loads((args.directory / "config/retention.json").read_text())
    for signal in ("logs", "traces", "metrics"):
        days = policy[f"{signal}_days"]
        if not isinstance(days, int) or not 1 <= days <= 3650:
            raise ValueError("retention days must be an integer from 1 through 3650")
        hours = days * 24
        if signal == "logs":
            current = request_json(args.url, "/api/v2/settings/ttl?type=logs", access_token=token)
            if current.get("default_ttl_days") == days:
                retention[signal] = current
                continue
            request_json(
                args.url,
                "/api/v2/settings/ttl",
                method="POST",
                access_token=token,
                body={"type": "logs", "defaultTTLDays": days, "ttlConditions": []},
            )
            retention[signal] = request_json(
                args.url, "/api/v2/settings/ttl?type=logs", access_token=token
            )
            continue
        current = request_json(args.url, f"/api/v1/settings/ttl?type={signal}", access_token=token)
        if current.get(f"{signal}_ttl_duration_hrs") == hours:
            retention[signal] = current
            continue
        request_json(
            args.url,
            f"/api/v1/settings/ttl?type={signal}&duration={hours}h",
            method="POST",
            access_token=token,
        )
        retention[signal] = request_json(
            args.url, f"/api/v1/settings/ttl?type={signal}", access_token=token
        )
    report = {
        "schema_version": 1,
        "dashboard_id": created.get("id", existing["id"] if existing else None),
        "panels": panel_results,
        "retention": retention,
    }
    atomic_replace(args.directory / ".provision-state.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        # No credential-bearing response bodies or URLs in error output.
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "message": str(error)
                    if isinstance(error, RuntimeError)
                    else "provisioning failed",
                }
            )
        )
        raise SystemExit(3) from None
