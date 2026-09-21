# Repair semantically ineffective next-edit benchmark states

Read `input.jsonl`. Each line contains a synthetic executable benchmark case plus a prior
`current_region` that incorrectly passed the hidden behavioral test. Write one replacement suggestion per
input line to `repairs.jsonl`, in the same order, with exactly this schema:

```json
{"id":"copied id","current_region":"new plausible buggy region","recent_edits":["one compact diff-like edit","optional second edit"]}
```

Requirements:

- Inspect the hidden test in each case and choose a subtle semantic bug that definitely makes at least one
  assertion fail while keeping `prefix + current_region + suffix` parseable and compilable.
- `current_region` must be non-empty, differ from both `expected` and `prior_current_region`, and preserve
  indentation and language syntax.
- Do not alter the gold code, tests, commands, paths, context, or ids.
- Output JSONL only, with every id exactly once and no prose or markdown fences.
- Before finishing, mechanically validate JSON, count, ordering, and id uniqueness.
