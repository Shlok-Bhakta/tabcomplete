"""Bounded, synthetic changed-state replay for the already selected local model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

TRANSITIONS = (
    "initial_open", "append", "near_cursor_replace", "earlier_edit",
    "reject_then_diverge", "typed_prefix", "file_a_b_a", "no_edit_or_cap",
)
SUITE_VERSION = 2  # v1 used too many filler lines and stopped at HTTP 400.


def post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=40) as response:
        return json.load(response)


def prompt(bucket: int, index: int) -> str:
    transition = TRANSITIONS[index]
    path = (
        "/synthetic/repo/b.py"
        if transition == "file_a_b_a" and index % 2
        else "/synthetic/repo/a.py"
    )
    lines = [f"# public synthetic context {n}: value_{n} = {n} + 1" for n in range(bucket // 21)]
    before = "\n".join(lines)
    if transition == "earlier_edit":
        lines[1] = "# public synthetic context 1: value_1 = 9 + 1"
        before = "\n".join(lines)
    history = (
        "<actual-recent-edit start=3 end=3>\nx\n</actual-recent-edit>\n"
        if transition in {"append", "reject_then_diverge", "typed_prefix"}
        else "<recent-edit unavailable>\n"
    )
    region = "result = old" if transition == "near_cursor_replace" else ""
    after = "\n# end of public synthetic fixture"
    return (
        f"<repo {path}>\n<filetype python>\n{history}<file {path}>\n"
        f"{before}\nresult = [[EDIT]]{region}[[/EDIT]]{after}\n</file>\n"
        f"<P {path} {len(before) + 10}>\n"
        "Return one compact next-edit action: N\\n for no edit or R\\n "
        "followed by exact replacement text. End with EOS.\n"
    )


def memory(pid: int) -> dict:
    out = {}
    path = Path(f"/proc/{pid}/smaps_rollup")
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"Rss", "Pss", "Anonymous", "Swap"}:
            out[key.lower() + "_bytes"] = int(value.split()[0]) * 1024
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            out["high_water_rss_bytes"] = int(line.split()[1]) * 1024
    return out


def pressure() -> dict:
    return {kind: Path(f"/proc/pressure/{kind}").read_text().strip()
            for kind in ("memory", "io")}


def vmstat() -> dict:
    fields = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in {"pswpin", "pswpout", "pgmajfault"}:
            fields[key] = int(value)
    return fields


def completion(url: str, text: str, n_predict: int, cache: bool) -> dict:
    body = json.dumps({"prompt": text, "n_predict": n_predict, "temperature": 0,
                       "stream": True, "cache_prompt": cache, "id_slot": 0}).encode()
    req = urllib.request.Request(
        url + "/completion", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    chunks: list[str] = []
    tokens = 0
    first_chunk = first_token = eight = sixteen = final = None
    with urllib.request.urlopen(req, timeout=45) as response:
        for raw_line in response:
            if first_chunk is None:
                first_chunk = time.perf_counter() - started
            if not raw_line.startswith(b"data: "):
                continue
            event = json.loads(raw_line[6:])
            if event.get("stop"):
                final = event
                break
            piece = event.get("content", "")
            if piece:
                chunks.append(piece)
                tokens += 1
                elapsed = time.perf_counter() - started
                first_token = first_token or elapsed
                if tokens >= 8 and eight is None:
                    eight = elapsed
                if tokens >= 16 and sixteen is None:
                    sixteen = elapsed
    elapsed = time.perf_counter() - started
    assert final is not None, "runtime omitted terminal event"
    output = "".join(chunks)
    return {"completed_seconds": elapsed, "first_chunk_seconds": first_chunk,
            "first_token_seconds": first_token, "eight_tokens_seconds": eight,
            "sixteen_tokens_seconds": sixteen, "output_events": tokens,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "stop_type": final.get("stop_type"), "timings": final.get("timings", {})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:19093")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 4:
        raise ValueError("repetitions must be 1..4")
    cases = []
    for bucket in (512, 1024, 2048):
        for index, transition in enumerate(TRANSITIONS):
            text = prompt(bucket, index)
            tokens = post(args.url + "/tokenize", {"content": text, "add_special": False})["tokens"]
            if len(tokens) + 96 > 2304:
                raise ValueError(f"fixture exceeds context reserve: {bucket}/{transition}")
            cases.append((bucket, transition, text, len(tokens)))
    records = []
    before = {"memory": memory(args.pid), "pressure": pressure(), "vmstat": vmstat()}
    started = time.perf_counter()
    for rep in range(args.repetitions):
        for bucket, transition, text, token_count in cases:
            cap = 96 if transition == "no_edit_or_cap" else 16
            result = completion(args.url, text, cap, True)
            records.append({"repetition": rep, "bucket_target": bucket,
                            "transition": transition, "actual_prompt_tokens": token_count,
                            "n_predict": cap, **result, "post_memory": memory(args.pid)})
    repeats = []
    for bucket in (512, 1024, 2048):
        text = prompt(bucket, 0)
        repeats.append({"bucket": bucket, "fresh": completion(args.url, text, 16, False),
                        "cached": completion(args.url, text, 16, True)})
    after = {"memory": memory(args.pid), "pressure": pressure(), "vmstat": vmstat()}
    result = {"schema_version": 1, "suite_version": SUITE_VERSION, "label": args.label,
              "observed_at": datetime.now(UTC).isoformat(), "host": "crabcake",
              "pid": args.pid, "repetitions": args.repetitions,
              "elapsed_seconds": time.perf_counter() - started,
              "changed_state_completed_median_seconds": statistics.median(
                  row["completed_seconds"] for row in records),
              "changed_state_completed_p95_seconds": sorted(
                  row["completed_seconds"] for row in records)[
                      max(0, int(0.95 * len(records) + 0.999999) - 1)],
              "records": records, "same_prompt_repeats": repeats,
              "before": before, "after": after}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "label", "elapsed_seconds", "changed_state_completed_median_seconds",
        "changed_state_completed_p95_seconds")}))


if __name__ == "__main__":
    main()
