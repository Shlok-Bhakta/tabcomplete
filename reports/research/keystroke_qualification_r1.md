# Public keystroke sequence qualification r1

This CPU-only audit tests whether a public editor log can supply authentic
next-edit states for a future bounded training pilot. It did not train or
deploy a model. The installed q25 Q4 editor model remains unchanged.

## Source and access

The source is the processed `keystrokes.csv` from [2019 CS1 Keystroke
Data](https://doi.org/10.7910/DVN/6BPCXN), datafile 6157950, 455,078,162
bytes, SHA-256
`b54d8cf56b7719ae0ec9ecf19e5b3788168f087bd4a50238dca1e06f6bfe7955`.
The current [DataCite record](https://api.datacite.org/dois/10.7910%2FDVN%2F6BPCXN)
reports version 5.2 and CC0 1.0. The [publisher README](https://dataverse.harvard.edu/api/access/datafile/12130010)
says the raw logger file contains corruption and should not be used for code
reconstruction. Only the processed CSV and README were downloaded. Both are
kept outside Git in a private research directory.

The publisher warns that deidentification may have missed identifying
keystrokes. The audit never prints participant identifiers or source text.
Its email/URL checks are screening signals, not a privacy clearance. No
student-derived state was uploaded to Kaggle or another provider.
The [publisher paper](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/581/158)
describes identifying text that can appear briefly and then be deleted, so
checking only final buffers cannot clear a participant for training.

## Exact replay

The [frozen first plan](keystroke_qualification_r1/plan.json) verifies the
source hash and treats the publisher's linear source position as a character
index. It replays each file edit only when the claimed deleted text exactly
matches the current reconstructed buffer. Other event types do not modify
that buffer.

The [full census](keystroke_qualification_r1/full_census.json) read 5,130,297
events from 487 pseudonymous subjects and 3,028 subject-assignment-section
groups. Of 5,039,755 file edits, 5,039,734 replayed exactly. One mismatch
made the next 20 edits in that group unreconstructable; 3,027 groups remained
reconstructable at end. There were 1,922,410 same-buffer edit gaps of at least
250 ms. Twelve groups matched an email or URL pattern during sampled and final
buffer checks. Neither replay consistency nor this pattern count proves that
the data is safe or predictive.

The full scan first stopped on Python's default 128 KiB CSV field limit while
reading a large non-source field. The bounded limit was raised to 16 MiB,
and the full scan then passed without dropping rows. The initial
[100,000-row prefix](keystroke_qualification_r1/prefix_100k.json) is retained
as a diagnostic, not extrapolated to the full file.

## Paused-state yield

The [second plan](keystroke_qualification_r1/plan_bursts.json) freezes a
250 ms pause boundary, a cursor inferred from the preceding edit, an exact
first-edit cursor match, an unchanged pre-cursor prefix, and a single-line
net change with at most 96 UTF-8 response bytes. It does not derive no-edit
labels from inactivity. The first implementation was interrupted before
output because a Python per-character suffix scan was too slow. The same
rule was run with a bounded C-level suffix comparison and its new prefix
guard recorded in the frozen plan. The
[burst audit](keystroke_qualification_r1/burst_audit.json) found 1,925,437
bursts, 1,254,906 pattern-clean candidates, and candidates from 482 subjects.

The [third plan](keystroke_qualification_r1/plan_lengths.json) froze length
diagnostics before the [length scan](keystroke_qualification_r1/burst_lengths.json).
Of those 1,254,906 candidates, 638,203 have a one-byte replacement fragment,
321,864 have two to three bytes, and 288,454 have at least four bytes.
Only 13,826 have at least 16 bytes. At the changed-span level, 1,246,682
are insertions, 6,385 deletions, and 1,839 replacements. These are span
types, not the existing whole-line action labels. A separate 558 candidate
states had an email or URL pattern and were excluded from the pattern-clean
count. Pattern cleaning alone is insufficient for training clearance.

An independent contract audit found that these changed-span counts do not
match the installed editor's action region. The installed compact wire
replaces the suffix of the cursor line. The
[fourth plan](keystroke_qualification_r1/plan_serving.json) requires exact
whole-buffer reconstruction from that suffix replacement and breaks bursts
at intervening run, submit, and task-switch events. The
[serving audit](keystroke_qualification_r1/serving_audit.json) found
1,252,064 pattern-clean, exact serving-region candidates after those rules;
one otherwise eligible case changed bytes outside the serving region.
There were 59,011 non-edit event boundaries. Of the 1,252,064 exact-region
cases, 4,619 replacement suffixes exceeded 96 UTF-8 bytes. None has yet
passed the selected tokenizer's exact 96-token output budget.

## Decision

The source has enough replayable multi-character edits to justify a small
qualification sample. It has only beginner Python coursework, five
assignments, uncertain unlogged cursor movement, no grounded no-edit
decisions, and a publisher privacy warning. A natural next burst is an
observed action, not proof that it was inferable from the paused input state.
The current editor uses a compact replacement of the suffix of the cursor
line, while the separate one-line research pilot uses whole-line actions;
an extractor must choose one contract and replay it byte-exactly.

Before another GPU allocation: sample under a subject-disjoint split,
exclude all states from privacy-flagged subjects, review temporary/deleted
text as well as current buffers, verify tokenizer and action lengths, audit
the source/target context for leakage, and blind-review a fixed set of
targets for inferability. Keep any no-edit examples explicitly synthetic or
from genuine editor feedback. Do not use assignment text from the textbook
or the survey file. A private Kaggle dataset would still transfer the data
to another service; the current plan forbids that upload. The
[handoff](keystroke_qualification_r1/HANDOFF.md) records the commands and
remaining gates.
