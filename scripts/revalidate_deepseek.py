#!/usr/bin/env python3
"""Revalidate stored DeepSeek records against the corrected base (raw file).

Reconstructs each request deterministically from state_id "s{seed}-{k}",
re-runs validate_response on the stored candidates, and rewrites accepted /
rejected in place (raw response fields untouched). Prints a shift report.
Usage: uv run python scripts/revalidate_deepseek.py
"""

from __future__ import annotations

import json
import sys

IN_PATH = "data/generated/teacher_deepseek.jsonl"


def main() -> int:
    from tinycomplete.teacher.base import Candidate
    from tinycomplete.teacher.generate import FIXTURE_SOURCES, build_request
    from tinycomplete.teacher.validate import validate_response

    with open(IN_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    moved_to_accept, moved_to_reject = 0, 0
    for record in records:
        if not record.get("ok"):
            continue
        seed_s, k_s = record["state_id"][1:].split("-")
        seed, k = int(seed_s), int(k_s)
        source = FIXTURE_SOURCES[(seed + k) % len(FIXTURE_SOURCES)]
        request = build_request(
            source, seed * 100003 + k, record["num_candidates"], record["state_id"]
        )
        cands = tuple(
            Candidate(action=c["action"], replacement=c["replacement"])
            for c in record["accepted"] + record["rejected"]
        )
        old_acc = len(record["accepted"])
        accepted, rejected = validate_response(cands, request)
        record["accepted"] = accepted
        record["rejected"] = rejected
        moved_to_accept += max(0, len(accepted) - old_acc)
        moved_to_reject += max(0, len(rejected) - (len(cands) - old_acc))

    with open(IN_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    acc = sum(len(r["accepted"]) for r in records if r.get("ok"))
    rej = sum(len(r["rejected"]) for r in records if r.get("ok"))
    print(f"states={len(records)} accepted={acc} rejected={rej}")
    print(f"moved_to_accept={moved_to_accept} moved_to_reject={moved_to_reject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
