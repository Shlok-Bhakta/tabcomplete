"""Rescore evaluation rows against where the user ended up.

The next burst alone undercounts useful predictions: a model that proposes
the whole line the user finishes two bursts later is right, just early. For
each shown edit, compare the predicted state P with every future state F
(the next bursts within a minute; Zeta cases have only the target):

  exact_next    P equals the very next state
  exact_future  P equals some future state
  helpful       P is strictly closer (character edit distance) than the
                current state C to some future state: accepting saves typing
  harmful       P is farther from every future state than C is

Usage:
    python3 -I rescore.py --cases cases.jsonl rows1.jsonl [rows2.jsonl ...]
    python3 -I rescore.py --cases cases.jsonl --gate -1.5 rows.jsonl
"""

import argparse
import json
import statistics


def apply_edit(content, edit):
    data = content.encode("utf-8")
    return (data[: edit["start_byte"]] + edit["text"].encode("utf-8") + data[edit["end_byte"] :]).decode("utf-8")


def trim3(a, b, c):
    p = 0
    m = min(len(a), len(b), len(c))
    while p < m and a[p] == b[p] == c[p]:
        p += 1
    s = 0
    while s < m - p and a[-1 - s] == b[-1 - s] == c[-1 - s]:
        s += 1
    return a[p : len(a) - s], b[p : len(b) - s], c[p : len(c) - s]


def lev(a, b, cap=4000):
    if len(a) > cap or len(b) > cap:
        return abs(len(a) - len(b)) + min(len(a), len(b))  # rough bound for huge regions
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def same(a, b):
    return a == b or a.split() == b.split()


def score_row(case, row, gate):
    edit = row.get("edit")
    conf = (row.get("confidence") or {}).get("min_logprob")
    if not edit or (gate is not None and conf is not None and (row.get("confidence") or {}).get("changed_tokens", 0) > 0 and conf < gate):
        return {"shown": False}
    c = case["content"]
    p = apply_edit(c, edit)
    futures = case.get("future") or [case["target"]]
    out = {"shown": True, "exact_next": same(p, case["target"]), "exact_future": any(same(p, f) for f in futures)}
    best_saved, all_worse = None, True
    for f in futures:
        tc, tp, tf = trim3(c, p, f)
        dc, dp = lev(tc, tf), lev(tp, tf)
        if dc == 0:
            continue
        saved = (dc - dp) / dc
        best_saved = saved if best_saved is None else max(best_saved, saved)
        if dp <= dc:
            all_worse = False
    out["helpful"] = best_saved is not None and best_saved > 0
    out["harmful"] = all_worse
    out["saved"] = max(best_saved or 0.0, 0.0)
    return out


def summarize(cases, rows, gate):
    n = len(rows)
    scored = [score_row(cases[r["id"]], r, gate) for r in rows if r["id"] in cases]
    shown = [s for s in scored if s["shown"]]
    k = len(shown)
    lat = [r["ms"] for r in rows if r.get("ms")]
    return {
        "n": n,
        "shown": round(k / n, 3),
        "exact_next": round(sum(s["exact_next"] for s in shown) / n, 3),
        "exact_future": round(sum(s["exact_future"] for s in shown) / n, 3),
        "helpful": round(sum(s["helpful"] for s in shown) / n, 3),
        "harmful": round(sum(s["harmful"] for s in shown) / n, 3),
        "precision": round(sum(s["helpful"] for s in shown) / k, 3) if k else None,
        "saved_per_case": round(sum(s["saved"] for s in shown) / n, 3),
        "p50_ms": round(statistics.median(lat)) if lat else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, action="append")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("rows", nargs="+")
    args = ap.parse_args()
    cases = {}
    for path in args.cases:
        for line in open(path):
            c = json.loads(line)
            cases[c["id"]] = c
    for path in args.rows:
        rows = [json.loads(l) for l in open(path)]
        name = path.rsplit("/", 1)[-1].replace(".rows.jsonl", "")
        print(json.dumps({"name": name, **summarize(cases, rows, args.gate)}))


if __name__ == "__main__":
    main()
