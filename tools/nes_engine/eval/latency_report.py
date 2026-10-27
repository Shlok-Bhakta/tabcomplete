"""Latency breakdown for evaluation rows (where the time goes).

Usage: python3 -I latency_report.py rows.jsonl [...]
"""
import json
import statistics as st
import sys


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))]


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    total, pre, dec, ev, gen, batches, acc, prop = [], [], [], [], [], [], 0, 0
    for r in rows:
        stages = r.get("stages") or []
        if not stages or not r.get("ms"):
            continue
        s = stages[0]
        total.append(r["ms"])
        pre.append(sum(x["prefill_ms"] for x in stages))
        dec.append(sum(x["decode_ms"] for x in stages))
        ev.append(sum(x["evaluated_prompt_tokens"] for x in stages))
        gen.append(sum(x["generated_tokens"] for x in stages))
        batches.append(sum(x["decode_batches"] for x in stages))
        acc += sum(x["draft_accepted"] for x in stages)
        prop += sum(x["draft_proposed"] for x in stages)
    name = path.rsplit("/", 1)[-1].replace(".rows.jsonl", "")
    print(json.dumps({
        "name": name,
        "p50_ms": round(st.median(total)), "p90_ms": round(pct(total, 0.9)),
        "prefill_p50": round(st.median(pre)), "decode_p50": round(st.median(dec)),
        "eval_tokens_p50": round(st.median(ev)), "gen_tokens_p50": round(st.median(gen)),
        "batches_p50": round(st.median(batches)),
        "draft_accept": round(acc / max(1, prop), 3),
        "ms_per_gen_token": round(sum(dec) / max(1, sum(gen)), 1),
    }))
