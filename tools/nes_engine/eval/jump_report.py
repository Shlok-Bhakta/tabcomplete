"""Jump-stage report for run_eval-jump.py rows (and optional --all-out lines).

Usage: python3 -I jump_report.py NAME.rows.jsonl [NAME.all.jsonl]
"""
import json
import sys


def q(v, p):
    v = sorted(v)
    return round(v[int(p * (len(v) - 1))]) if v else None


rows = [json.loads(l) for l in open(sys.argv[1])]
name = sys.argv[1].rsplit("/", 1)[-1].replace(".rows.jsonl", "")
ms = [r["ms"] or 0 for r in rows]
ran = [r for r in rows if ((r.get("context") or {}).get("jump") or {}).get("outcome") in ("edit", "none", "cancelled")]
outcomes = {}
reasons = {}
for r in rows:
    j = (r.get("context") or {}).get("jump")
    if j:
        outcomes[j["outcome"]] = outcomes.get(j["outcome"], 0) + 1
        reasons[j["reason"]] = reasons.get(j["reason"], 0) + 1
js = [r["stages"][-1] for r in ran if r.get("stages")]
out = {
    "name": name,
    "n": len(rows),
    "p50_ms": q(ms, 0.5),
    "p90_ms": q(ms, 0.9),
    "mean_ms": round(sum(ms) / len(ms)),
    "jump_runs": len(ran),
    "outcomes": outcomes,
    "reasons": reasons,
    "jump_req_p50_ms": q([r["ms"] for r in ran], 0.5),
    "jump_req_p90_ms": q([r["ms"] for r in ran], 0.9),
    "jstage_eval_p50": q([s["evaluated_prompt_tokens"] for s in js], 0.5),
    "jstage_eval_mean": round(sum(s["evaluated_prompt_tokens"] for s in js) / len(js)) if js else None,
    "jstage_prefill_p50": q([s["prefill_ms"] for s in js], 0.5),
    "jstage_decode_p50": q([s["decode_ms"] for s in js], 0.5),
    "jstage_ms_sum": round(sum(s["prefill_ms"] + s["decode_ms"] for s in js)),
    "jstage_ms_per_eval_tok": round(sum(s["prefill_ms"] for s in js) / max(1, sum(s["evaluated_prompt_tokens"] for s in js)), 2),
    "scratch": sum(1 for s in js if s.get("scratch")),
}
if len(sys.argv) > 2:
    lines = [json.loads(l) for l in open(sys.argv[2])]
    # The request right after a jump-stage request (the cache it finds).
    after = []
    for a, b in zip(lines, lines[1:]):
        j = a.get("jump") or {}
        if j.get("outcome") in ("edit", "none") and b.get("ms") and not b.get("prefetched"):
            st = (b.get("stages") or [{}])[0]
            after.append((b["ms"], st.get("evaluated_prompt_tokens") or 0))
    out["next_after_jump_n"] = len(after)
    out["next_after_jump_p50_ms"] = q([a for a, _ in after], 0.5)
    out["next_after_jump_eval_p50"] = q([e for _, e in after], 0.5)
    out["next_after_jump_mean_ms"] = round(sum(a for a, _ in after) / len(after)) if after else None
print(json.dumps(out))
