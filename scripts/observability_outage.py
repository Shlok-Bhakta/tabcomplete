#!/usr/bin/env python3
"""Temporarily stop only this deployment's ingester, then prove bounded replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import httpx

from tinycomplete.observability.bootstrap import current_runtime
from tinycomplete.observability.context import RunContext
from tinycomplete.observability.spans import operation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kiwi-host", required=True)
    parser.add_argument("--gateway", required=True)
    args = parser.parse_args()
    run = RunContext.new()
    target = "tabcomplete-observability-ingester-1"
    stopped = False
    try:
        subprocess.run(
            ["ssh", args.kiwi_host, "docker", "stop", target],
            check=True,
            capture_output=True,
            timeout=30,
        )
        stopped = True
        started = time.perf_counter()
        with run.activate():
            for index in range(20):
                with operation(
                    "outage.fixture",
                    attributes={
                        "tabcomplete.case_id": f"outage-{index}",
                        "tabcomplete.terminal_event_id": f"{run.run_id}:{index}",
                    },
                ):
                    pass
        elapsed = time.perf_counter() - started
        assert current_runtime().force_flush()
        time.sleep(7)
    finally:
        if stopped:
            subprocess.run(
                ["ssh", args.kiwi_host, "docker", "start", target],
                check=True,
                capture_output=True,
                timeout=30,
            )
    for _attempt in range(24):
        response = httpx.get(
            args.gateway + "/api/v1/runs/" + run.run_id,
            params={"since": "1h", "limit": 100},
            timeout=12,
        )
        response.raise_for_status()
        rows = response.json()["data"]
        unique = {
            r["tabcomplete.terminal_event_id"]
            for r in rows
            if r.get("tabcomplete.terminal_event_id")
        }
        if len(unique) == 20:
            print(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "expected_events": 20,
                        "unique_replayed_events": 20,
                        "raw_rows": len(rows),
                        "workload_seconds_during_outage": elapsed,
                    }
                )
            )
            return
        time.sleep(3)
    raise SystemExit("Replay did not complete inside verification window")


if __name__ == "__main__":
    main()
