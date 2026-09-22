#!/usr/bin/env python3
"""Read-only reconciliation of a smoke result against gateway, artifacts, and MCP."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--mcp", required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    run = result["run_id"]
    client = httpx.Client(timeout=15)
    response = client.get(
        args.gateway + "/api/v1/runs/" + run, params={"since": "14d", "limit": 200}
    )
    response.raise_for_status()
    rows = response.json()["data"]
    names = {row["name"] for row in rows}
    assert {"run.start", "run.heartbeat", "run.summary", "model.generate", "eval.case"} <= names
    quality = {
        r["tabcomplete.case_id"]: r
        for r in rows
        if r["name"] == "eval.case" and r.get("tabcomplete.quality.functional")
    }
    assert len(quality) == result["expected"]["cases"]
    assert (
        sum(r["tabcomplete.quality.functional"] == "pass" for r in quality.values())
        == result["expected"]["functional_pass"]
    )
    retrieved = set()
    for row in rows:
        for kind in ("input", "output"):
            digest = row.get(f"tabcomplete.artifact.{kind}.sha256")
            if digest and digest not in retrieved:
                payload = client.get(args.gateway + "/api/v1/artifacts/" + digest)
                payload.raise_for_status()
                assert hashlib.sha256(payload.content).hexdigest() == digest
                retrieved.add(digest)
    real = result.get("real_model")
    if real:
        model = next(
            r
            for r in rows
            if r.get("tabcomplete.request_id") == real["request_id"]
            and r["name"] == "model.generate"
        )
        payload = client.get(
            args.gateway + "/api/v1/artifacts/" + model["tabcomplete.artifact.output.sha256"]
        )
        assert payload.text == real["text"]
    headers = {"Accept": "application/json, text/event-stream"}
    response = client.post(
        args.mcp,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "tabcomplete-verification", "version": "1"},
            },
        },
    )
    response.raise_for_status()
    end = int(time.time() * 1000)
    query = {
        "schemaVersion": "v5",
        "start": end - 86400000,
        "end": end,
        "requestType": "raw",
        "compositeQuery": {
            "queries": [
                {
                    "type": "builder_query",
                    "spec": {
                        "name": "A",
                        "signal": "traces",
                        "filter": {
                            "expression": "service.name = 'tabcomplete' "
                            f"AND tabcomplete.run_id = '{run}'"
                        },
                        "selectFields": [{"name": "tabcomplete.run_id"}, {"name": "name"}],
                        "limit": 100,
                        "order": [{"key": {"name": "timestamp"}, "direction": "desc"}],
                    },
                }
            ]
        },
    }
    response = client.post(
        args.mcp,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "signoz_execute_builder_query", "arguments": {"query": query}},
        },
    )
    response.raise_for_status()
    assert not response.json().get("result", {}).get("isError")
    assert run in response.text
    print(
        json.dumps(
            {
                "run_id": run,
                "quality_cases": len(quality),
                "functional_pass": result["expected"]["functional_pass"],
                "unique_artifacts_verified": len(retrieved),
                "mcp": "known_run_returned",
                "trace_ids": sorted({r["trace_id"] for r in rows}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
