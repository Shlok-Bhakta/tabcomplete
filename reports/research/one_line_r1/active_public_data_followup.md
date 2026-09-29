# Public edit data follow-up, 2026-09-28–29 UTC

This continuation tested whether two low-cost public sources could supply the
one-line editor states needed for the frozen one-line campaign. It did not change
the installed model, editor, collector, campaign gate, or sealed tests. No GPU
session, model download, teacher call, or training update was used.

## Fixed source probes

The [Zeta mapping plans](zeta_mapping_plan_v3.json) pinned the public dataset
revision `0f41422bc56fa0f9a5b3096e7e6f42905497d793` and mapped the first
100 train rows into the current editor action shape. [The result](zeta_mapping_results_v3.json)
found 11 exact one-line transformations, but only six target the recorded cursor
row and preserve the prefix before the cursor, as the installed suffix-only
editor region requires. The mapping did not reconstruct edit history or establish
repository, commit, and per-file license identity. These are diagnostic rows,
not accepted labels.

The [active-Git plan](active_git_sequence_plan_v1.json) pinned one MIT-root
repository per target language before reading recent diffs. The bounded
[screen](active_git_sequence_results_v1.json) inspected 128 first-parent commit
pairs in each repository. It found 69 structural pairs in 29 files: 20 in
Pydantic, 2 in Vue core, 35 in Tokio, and 12 in Gin. Eleven Gin pairs came
from repeated `Oupps` → `Oops` string spelling changes in one test file; the
rule also catches URL strings and primitive type substitutions. The scan spent
851.1 seconds and its four partial Git checkouts occupy about 6.3 MiB. Git
shows that two lines changed in one commit; it does not show their editor order
or the user's intermediate state. The structural yield gate passed, but the
result remains source-yield evidence, not a set of training examples.

## Blind review of a fixed subset

The [blind plan v1](active_git_blind_review_plan_v1.json) fixed nine IDs before
prompt construction. The [v2 revision](active_git_blind_review_plan_v2.json)
pins Gin's historical MIT license SHA-256
`03458b6d5828e1be1127ca2adf122572eb574fc47b56190c3b38203b8b2a98d0`;
the only difference from the repository-head license is a later copyright-line
update. The selected IDs and prompt rules did not change. A fresh reviewer saw
only the current line, 15 nearby lines on either side, and a disclosed
**synthetic** earlier one-line edit. It did not see the gold lines or Git data.

The reviewer supplied exact target lines for all **9/9** fixed examples;
eight are code and one is a documentation URL/comment. The
[predictions](active_git_blind_review_predictions_v2.json) were compared byte
for byte with the pinned Git child lines. The blind prompt artifact SHA-256 is
`1ff47c88775171284f39558b0f4af91ddffe602796d34ad9dacbf54cc8513c54`;
the separate local gold artifact SHA-256 is
`2336d21f076fedac91d9a53fa6837d0d07f9cb64fac73d6079fff439c305efb8`.
This was a deliberately small, structurally selected sample, not an estimate
of general inferability. The same substitution was shown in the prior edit by
construction, and real edit timing and intent remain unknown. All nine source
files lack an explicit SPDX notice in their first 2,048 bytes; root MIT license
hashes were checked at the historical child commits, but file-level scope needs
a separate review before a training decision.

## Data and training decision

Accepted train/development/test labels remain **0/0/0**. The frozen main-run
gate remains 20,000 accepted training states, 100 independent source groups,
and 25 reviewed mechanisms. The active Git screen has four source groups and
one narrow repeated-substitution mechanism; Zeta has six directly applyable
diagnostic rows without source identity. Neither route meets the gate. A
disposable fake-teacher CPU preflight produced 80 synthetic plumbing records;
its tokenizer check found median 153, p95 205, and maximum 231 input tokens.
Those records were not trained on and are not quality evidence. No Kaggle or
Colab GPU time was consumed, despite free quota being available.

The live crabcake predictor was checked read-only. `tabcomplete-predictor.service`
still runs CPU-only `llama-server` on `127.0.0.1:19093` with four generation and
prompt threads, context 2,304, one slot, batch 256, microbatch 64, and 128 MiB
optional cache. `/health` returned `ok`. The deployed adapted q25-coder Q4_K_M
file still hashes to
`a45bc50aa7c74a03e7bdcade90315b052b31675a97631ed5c96df953160fd626`.
This health check is not a new latency or model-quality measurement.

The existing collector database is `/mnt/ssd/collector-data/collector.sqlite`.
A read-only metadata snapshot found 70 prediction requests, 55 shown proposals,
22 explicit accepts, 24 explicit rejects, three implicit typing dismissals,
four typed matches, and one partial match. Event flags mark 20 accepts and 21
explicit rejects synthetic. The remaining two accepts and three rejects are
unclassified, not verified human feedback. The latest collector event in that
snapshot was 2026-09-25 20:17 UTC. No current human preference label is counted
from these records, and automatic personalization remains disabled. The live
predictor hardcodes `human_verified=false` on requests, while the exporter
requires `human_verified=true` for a defensible pair. No production path sets
it true. A key mapping can identify the software path used for a decision, but
Neovim cannot prove a physical human pressed it. A later reviewer-confirmation
event keyed to an exact outcome is needed before those outcomes can be
promoted; legacy untagged events stay unknown.

## Verification and next useful step

The two new source-probe test modules passed 6/6 cases. Blind gold comparison
passed 9/9 fixed lines. The full repository suite passed **414 tests, one
skipped** after linking two pre-existing hash-pinned, ignored historical author
artifacts into the isolated worktree. Ruff passed repository-wide, formatting
passed on changed scripts, and mypy passed on 81 source files. A type-only
annotation edit to the Zeta probe followed its v3 result; the v3 plan retains
the exact historical runner hash and no Zeta output was regenerated. Scientific
results are the JSON files above; SigNoz had no recent research run in the
24-hour query, so no monitoring run ID is asserted.

The useful next experiment is to obtain actual chronological editor states or
another source with observed edit order and an independent objective, then
audit license scope and group-disjoint splits before considering training. The
installed automatic collector can generate such evidence when the editor is
used; an unclassified event or a synthetic smoke decision must not be promoted
to a human preference pair.
