# Keystroke qualification handoff

Branch: `research/keystroke-qualification-r1`, based on the pushed one-line
pilot commit `26e6d6e86e8fb8dbd04652ee669166b61cd899be`.

The private source is
`/mnt/ssd/tabcomplete-keystroke-qualification-r1/raw/keystrokes.csv`.
It is the processed public CS1 log, not the corrupted raw logger file.
The source SHA-256 is
`b54d8cf56b7719ae0ec9ecf19e5b3788168f087bd4a50238dca1e06f6bfe7955`.
Do not commit or print participant source text, and do not upload unreviewed
rows to a training service. No GPU session or model-weight download was
used in this qualification.

Reproduce the aggregate scans with:

```bash
uv run python scripts/audit_keystroke_qualification.py \
  --csv /mnt/ssd/tabcomplete-keystroke-qualification-r1/raw/keystrokes.csv \
  --plan reports/research/keystroke_qualification_r1/plan.json \
  --output /tmp/keystroke-full-census-recheck.json

uv run python scripts/audit_keystroke_bursts.py \
  --csv /mnt/ssd/tabcomplete-keystroke-qualification-r1/raw/keystrokes.csv \
  --plan reports/research/keystroke_qualification_r1/plan_lengths.json \
  --output /tmp/keystroke-burst-lengths-recheck.json
```

The exact replay matched 5,039,734 of 5,039,755 file edits. The paused
screen found 288,454 pattern-clean one-line changes of at least four UTF-8
response bytes across 476 subjects, but nearly all changes are span
insertions. These are candidate observations, not accepted model labels.
After correcting for run/submit/task-switch boundaries and the installed
editor's suffix-region replacement, 1,252,064 candidates replayed exactly
under that region. The 288,454 figure remains a changed-fragment length
screen, not a serving response length or a token-budgeted training count.
No full prompt/response dataset has been constructed or independently
reviewed.

The next bounded step is a deterministic 32-case blind review and a
subject-disjoint small extraction. Exclude all states belonging to any
subject with a privacy flag, because a single flagged group can contaminate
other states from the same subject. The current email/URL screen is incomplete:
the publisher paper documents ephemeral identifiers in deleted text. Apply
stronger checks on reconstructed and deleted text, inspect source rights for
assignment templates, and retain `external_training_upload=false` until a
reviewed minimal export exists. A private Kaggle dataset is still an external
transfer. Avoid no-edit labels from pauses.
Use assignment p8 as a separate task transfer diagnostic and do not inspect
it for tuning. Keep the installed q25 Q4 model and editor running unchanged
until a candidate beats it on paired task and resource measurements.

Verification in this worktree: the full Python suite passed with 443 tests
and one skip, Ruff passed, and mypy passed on both audit scripts. Historical
ignored plan-10/11 pilot files were copied from the prior isolated worktree
solely so the existing tests could read their pinned local artifacts; they
are not part of this commit. The new research directory uses 455,081,636
bytes of source/README files, below its 2 GiB cap.

The full aggregate results, plan revisions, and SHA-256 values are in the
[report](../keystroke_qualification_r1.md) and
[artifact manifest](artifact_manifest.json). The isolated worktree has no
relationship to the main worktree's uncommitted changes.
