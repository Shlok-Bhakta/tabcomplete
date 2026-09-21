# Next-edit benchmark author task

Read `input.jsonl`. It contains synthetic executable code-completion benchmark cases.
Write exactly one corresponding JSON object per input line to `suggestions.jsonl`, in the same order.
Do not modify any other file.

Each output object must have exactly:

```json
{"id":"copied input id","current_region":"plausible current region","recent_edits":["one compact diff-like edit","optional second edit"]}
```

For each input case, `expected` is the correct code currently inserted between `prefix` and `suffix`.
Create a realistic next-edit state by replacing that correct region with a plausible but behaviorally wrong
`current_region`. The future edit will replace `current_region` with the original `expected`.

Requirements:

- Preserve the input `id` exactly.
- `current_region` must differ from `expected` and must be non-empty.
- `prefix + current_region + suffix` should parse and compile when practical, but should fail at least one
  hidden behavioral edge case. Prefer subtle semantic mistakes over syntax damage.
- Keep the same indentation and rough shape as `expected`.
- Do not include markdown fences or prose in `suggestions.jsonl`.
- `recent_edits` should contain one or two concise, diff-like earlier edits that make the intended correction
  plausible without revealing the exact answer. Use only information already present in this synthetic case.
- Do not change hidden tests, context files, commands, paths, or the gold code.
- Use standard JSON escaping and ensure every line parses independently.

Before finishing, verify that every input id appears exactly once and every output line is valid JSON.
