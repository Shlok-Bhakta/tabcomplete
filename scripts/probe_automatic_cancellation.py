"""Verify whether closing one local SSE client releases the pinned server slot."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from benchmark_automatic_runtime import prompt


def slot(url: str) -> dict:
    with urllib.request.urlopen(url + "/slots", timeout=5) as response:
        return json.load(response)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:19093")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not slot(args.url)["is_processing"], "slot must be idle before probe"
    body = json.dumps({"prompt": prompt(1024, 2), "n_predict": 96, "temperature": 0,
                       "stream": True, "cache_prompt": False, "id_slot": 0}).encode()
    request = urllib.request.Request(args.url + "/completion", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    token_observed = False
    with urllib.request.urlopen(request, timeout=30) as response:
        for line in response:
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                if event.get("content"):
                    token_observed = True
                    break
    closed = time.perf_counter()
    observations = []
    idle_at = None
    for _ in range(100):
        state = slot(args.url)
        elapsed = time.perf_counter() - closed
        observations.append({"seconds_after_close": elapsed,
                             "is_processing": state["is_processing"],
                             "id_task": state.get("id_task")})
        if not state["is_processing"]:
            idle_at = elapsed
            break
        time.sleep(0.05)
    result = {"observed_at": datetime.now(UTC).isoformat(), "token_observed": token_observed,
              "time_to_first_token_seconds": closed - started,
              "slot_idle_after_close_seconds": idle_at, "observations": observations,
              "model": "selected q25 Q4_K_M; synthetic source"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"token_observed": token_observed,
                      "slot_idle_after_close_seconds": idle_at}))


if __name__ == "__main__":
    main()
