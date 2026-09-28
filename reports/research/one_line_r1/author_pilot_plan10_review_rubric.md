# Plan-10/11 public author pilot audit rubric

Run `uv run python scripts/audit_one_line_author_pilot_plan10.py` only after
`author_pilot_combined_plan11.json` reports 99 durable rows and one named failed
source ID. Plan 10 stopped after seven durable rows. Its eighth request had a
known usage overrun and no candidate row or retry. Plan 11 covers the remaining
92 untouched seeds. The script checks both raw artifact hashes and the frozen
plan, protocol, source, and tokenizer identities before opening raw rows. It
requires 99 distinct durable IDs plus the failed ID to partition the 100 frozen
seeds, and refuses any row marked accepted for training.

The script reports structural yield, exact history/action replay, q25 student
input and supervised target token counts, provider token use, N/D/R/I balance,
source-group concentration, and the author's unreviewed objective kinds. The
objective kinds are not verified mechanisms. Its 12-case sample contains only
`candidate_preflight` rows. It ranks their source IDs by
`SHA256(b"plan10-human-audit-v1\0" + source_id.encode("utf-8"))` and takes
the first 12. This rule was fixed before reading public pilot results. Rejected
rows remain in the structural denominator and cannot enter the candidate-only
context review. The failed request is a separate censored denominator. A small
sample cannot establish general task quality.

For each sampled row, first read only its entry in
`author_pilot_combined_plan11_blind_sample.jsonl`. This is the exact bounded student
prompt. It contains no author-only focus, objective, or target action. Record:

1. Which target action, if any, the current source and recent edit support.
2. The exact visible source or history lines that support that judgment.
3. Whether keep or a different one-line action is equally plausible.

Use `clear` only when the target need and operation follow from the visible
state without author-only hints. Use `ambiguous` when multiple reasonable next
edits remain. Use `unsupported` when the prompt supplies no positive reason for
the proposed task. After writing that blind judgment, reveal the candidate's
action, intent evidence, and objective from the raw artifact. Check whether the
declared action matches the blind judgment and whether its objective is
independently testable. A statement of intent alone is not an executable test.
Record failures and disagreements without revising the sample or source data.

All public rows remain candidates with `accepted_training=0`. The 3/4
synthetic smoke checked the author format only. The same Muse model may fill
author, solver, and reviewer roles in separate sessions. Their actor IDs show
role isolation, not independent model families or human review.
