# Synthetic Data Report

Date: 2026-09-16/17. Paid gate: `ALLOW_PAID_SYNTHETIC=1` set in-process for
the DeepSeek run below (user-authorized); all other runs fake.

## Paid API usage

- OpenRouter requests: 0, spend: $0.00 (no key present)
- DeepSeek requests: 2000 states / ~6000 candidates, estimated spend: **$0.78**
- Total paid examples: 2000 usable (2230 quota counted incl. 230 crash-lost, recovered)
- Key: `DEEPSEEK_API_KEY` from `~/.hermes/.env` (process env only, never logged)

## DeepSeek pricing (recorded 2026-09-17, source: api-docs.deepseek.com/quick_start/pricing)

Model `deepseek-flash` (verified via `GET /models`; `deepseek-chat` NOT listed):
peak $0.30/1M in (cache miss), $1.20/1M out; off-peak half. Run happened
00:16–00:3x UTC Thursday = off-peak, but accounting conservatively uses peak
+ 50% margin. Balance before run: $4.10 topped-up.

## DeepSeek teacher findings (stages A+B)

`uv run python scripts/deepseek_stage_a.py --states 60 --seed 3`
(10 stage-A + 50 stage-B, 3 candidates/state, fixture-only inputs):

- request ok-rate: 60/60 (100%); schema parse: 180/180 (100%, gate was 90%)
- accepted 148 / rejected-with-reasons 32 (rejects kept for preference data)
- actions: 84 replace / 64 noop — NOOP used freely, as instructed
- replace size median 12 chars / max 66 — small likely edits, no broad rewrites
- reject reasons: syntax errors 21, identical-to-region 9, empty-replace 2
- usage: 29,014 prompt + 3,240 output tokens
- terms check: DeepSeek Open Platform ToS §4.2 explicitly permits training
  other models (distillation). Inputs were fixture/synthetic code only.

Pipeline fixes Stage A forced (before scaling): thinking mode disabled
(`thinking: {type: disabled}` — default thinking starves JSON `content`),
JSON-object extraction fallback in `parse_payload`, explicit schema skeleton
+ example embedded in the teacher prompt (json_object mode has no strict
schema). One accounting fix: failed requests no longer consume example quota.

## Muse vs DeepSeek comparison

Pending: no OpenRouter key, so no overlapping Muse labels exist yet. The 60
DeepSeek states (seeds logged in the JSONL) are ready for an overlap run when
a key is available.

## Fake-provider pipeline (full path, $0)

`uv run tinycomplete synthetic --provider fake --states 12 --seed 1`:
12/12 ok, 20 accepted / 4 rejected. Artifacts gitignored under
`data/generated/`.

## Fake-provider pipeline (full path, $0)

`uv run tinycomplete synthetic --provider fake --states 12 --seed 1
--out data/generated/teacher_fake.jsonl`

- states: 12 (10 stage-A + 2 stage-B), request ok-rate 12/12
- accepted candidates: 20, rejected (stored with reasons): 4
- spend: $0.0000; budget file `data/generated/budget.json` untouched by fakes
- artifacts (gitignored): `data/generated/teacher_fake.jsonl`,
  `data/generated/teacher_fake.summary.json`

## Validation enforced per candidate

JSON/schema parse, region applicability, UTF-8, changed-unless-noop,
tree-sitter error-count check (python), rewrite size caps (8 KB absolute,
4× region), anti-reversal of the preceding edit, anti-duplication of the
region. Rejects are kept for future preference data.

## Staging policy status

A (10) ✅ 100% → B (50×3) ✅ 100% schema → C ✅ complete 2026-09-17:
**2000 states, 1999 ok, 4688 accepted / 1311 rejected**,
spend **$0.78** (cap $2.00), quota 2230/2230 counted (2000 usable + 230
lost to a summary-crash before file write; crash fixed, states recovered
with fresh seeds). Reject mix: identical-to-region, syntax errors,
duplicates, empty-replace. Training export `train_deepseek.jsonl`: 4688
records (2731 replace / 1957 noop), Qwen token median 144 / p95 203 /
max 250, zero truncation at 2048.
