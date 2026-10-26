"""Score a next-edit engine configuration on replayed cases.

For each case the engine is reset, fed the edit history as timed observe
calls, then asked for a prediction. The first proposed edit (the one the
editor shows) is applied to the state and compared with what the user typed
next:

  exact    applying the edit reproduces the user's next state
  correct  every predicted changed hunk appears in the user's next change
           (a strict subset still saves typing)
  wrong    an edit was shown that the user did not make
  empty    nothing was shown

Usage:
    python3 -I run_eval.py --engine PATH --model GGUF --cases cases.jsonl \
        --name label [--limit N] -- [engine flags...]
"""

import argparse
import difflib
import json
import statistics
import subprocess
import sys


def apply_edit(content, edit):
    data = content.encode("utf-8")
    out = data[: edit["start_byte"]] + edit["text"].encode("utf-8") + data[edit["end_byte"] :]
    return out.decode("utf-8")


def hunks(a, b):
    sm = difflib.SequenceMatcher(None, a.split("\n"), b.split("\n"), autojunk=False)
    al, bl = a.split("\n"), b.split("\n")
    out = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            out.add((i1, i2, "\n".join(bl[j1:j2])))
    return out


def score(case, resp):
    edits = resp.get("edits") or []
    if not edits:
        return "empty"
    predicted = apply_edit(case["content"], edits[0])
    if predicted == case["target"]:
        return "exact"
    if predicted.split() == case["target"].split():
        return "exact"  # whitespace-equivalent
    ph = hunks(case["content"], predicted)
    th = hunks(case["content"], case["target"])
    if ph and ph <= th:
        return "correct"
    return "wrong"


def replay_lines(case, jump=True):
    yield {"reset": {}}
    for h in case["history"]:
        cursor = h.get("cursor") or {"row": 0, "col": 0}
        yield {
            "observe": {
                "path": case["path"],
                "content": h["content"],
                "cursor": {"line": cursor["row"], "col": cursor["col"]},
                "ts_ms": h["ts_ms"],
            }
        }
    yield {
        "predict": {
            "request_id": case["id"],
            "path": case["path"],
            "content": case["content"],
            "cursor": case["cursor"],
            "ts_ms": case["ts_ms"],
            "jump": jump,
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-jump", action="store_true")
    ap.add_argument("flags", nargs="*")
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(args.cases)]
    if args.limit:
        cases = cases[: args.limit]
    stdin = "".join(json.dumps(line) + "\n" for c in cases for line in replay_lines(c, not args.no_jump))
    cmd = [args.engine, "--model", args.model, *args.flags, "replay"]
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    responses = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    by_id = {r["request_id"]: r for r in responses}
    counts = {"exact": 0, "correct": 0, "wrong": 0, "empty": 0}
    latencies, prompt_tokens, kinds = [], [], {"local": 0, "jump": 0}
    rows = []
    for c in cases:
        r = by_id.get(c["id"], {})
        s = score(c, r)
        counts[s] += 1
        latencies.append(r.get("timings", {}).get("total_ms", 0))
        prompt_tokens.append(r.get("context", {}).get("prompt_tokens", 0))
        if r.get("edits"):
            kinds[r.get("kind", "local")] = kinds.get(r.get("kind", "local"), 0) + 1
        rows.append({"id": c["id"], "score": s, "kind": r.get("kind"), "exit": r.get("exit_reason"),
                     "edit": (r.get("edits") or [None])[0], "completion": r.get("completion"),
                     "ms": r.get("timings", {}).get("total_ms")})
    n = len(cases)
    shown = counts["exact"] + counts["correct"] + counts["wrong"]
    summary = {
        "name": args.name,
        "flags": args.flags,
        "n": n,
        **counts,
        "exact_rate": round(counts["exact"] / n, 3),
        "useful_rate": round((counts["exact"] + counts["correct"]) / n, 3),
        "precision": round((counts["exact"] + counts["correct"]) / shown, 3) if shown else None,
        "shown_rate": round(shown / n, 3),
        "kinds": kinds,
        "p50_ms": round(statistics.median(latencies)),
        "p90_ms": round(sorted(latencies)[int(0.9 * (n - 1))]),
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens)),
    }
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
