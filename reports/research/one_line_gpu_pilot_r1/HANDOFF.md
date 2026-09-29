# One-line GPU pilot r1 handoff

Branch: `research/one-line-gpu-pilot-r1`. The private Kaggle scientific worker
ran commit `cb4b18872416a6ee1ccae31d3e35703d9c9f8ab9` under plan revision 4,
SHA-256 `cd796a2efb535bea32c64e98d82dbb38b297b51173d22319083201c6c7654c20`.
Later branch commits repair packaging and post-evaluation verification code;
they did not alter that worker's model outputs.

The completed pilot trained the untouched q25 base for one pass over 720
Continue Instinct TypeScript rows, 538,275 nonpadding input tokens, and
23 applied updates. It produced a verified checkpoint and F16 export. The
heldout path-disjoint development result was 9/210 byte-exact after-states,
all output actions were replacements, and synthetic keep accuracy was 0/4.
**Do not promote or deploy this export.** The current desktop Q4 model and
automatic editor configuration were left intact.

The worker's final status is `failed` only because it compared the exported
tokenizer JSON byte hash with the original tokenizer JSON byte hash.
`save_pretrained` rewrote the file. All 210 frozen development prompts had
identical text and token IDs under the original and export tokenizers, with
matching EOS, full vocabulary ID map, and special-token map. Encoding and
decoding also matched on all 210 observed adapted action strings. The
checkpoint, marker, export, and both
evaluation files passed offline hash verification. The updated worker uses
semantic parity and the export's own hash for future runs. The raw failed
status is preserved in the private output; do not relabel it as a clean
worker success.

Private Kaggle identifiers:

- Input dataset: `shlokbhakta/tc-oline-instinct-r1-retry-inputs`.
- Evaluated kernel: `shlokbhakta/tabcomplete-one-line-instinct-pilot-r1-retry`.
- First failed setup kernel: `shlokbhakta/tabcomplete-one-line-instinct-pilot-r1`.

Local private artifacts are under
`/mnt/ssd/tabcomplete-one-line-gpu-pilot-r1/`. The downloaded checkpoint and
export weights are in `output-attempt2-artifacts`; small scientific results
and tokenizer files are in `output-attempt2-small`. Source train/dev JSONL
remain in `prepared/instinct-v1-formatted`. These files are intentionally
ignored by Git. The committed [artifact manifest](artifact_manifest.json)
contains exact hashes and sizes. No data or weights were published.

The main plan-11 campaign remains data gated. Its 20,000 reviewed training
states, 100 source groups, and 25 mechanisms remain unmet; this one-project
pilot does not count as satisfying them. The next useful work is better
replayable, licensed, diverse observed edit data and defensible no-edit
states. More GPU passes over the same narrow set alone are unlikely to
establish general next-edit quality. The gated DECODE corpus requires manual
access approval and was not accessed. Preserve the sealed native Instinct
test split.

A read-only source scout identified the [2019 CS1 Keystroke Data](https://doi.org/10.7910/DVN/6BPCXN)
as a possible authentic insertion/deletion sequence source. Its current
DataCite metadata declares CC0 1.0, but the publisher warns that
deidentification may have missed identifying keystrokes. It covers novice
coursework rather than the target repository workflow and has no defensible
no-edit labels. The local Harvard Dataverse metadata endpoint returned an
AWS WAF challenge; the public README was accessible separately. No keystroke
CSV was downloaded or used. The README says the processed CSV supports
file-edit reconstruction and the raw logger file has corruption. Qualify
privacy handling, replay yield, and split before making a new training plan.

Verification completed in this worktree after the verifier repair:
438 Python tests passed and one skipped; Ruff and mypy passed on pilot files.
Re-run the repository suite and focused lint after any further changes.
Offline artifact verification command:

```bash
uv run python scripts/run_one_line_gpu_pilot.py verify-output \
  /mnt/ssd/tabcomplete-one-line-gpu-pilot-r1/output-attempt2-small \
  --plan-sha256 cd796a2efb535bea32c64e98d82dbb38b297b51173d22319083201c6c7654c20
```

Quota after the run was 0.41 of 45 GPU-hours used, 44.59 hours remaining,
observed 2026-09-29 07:31 UTC; no active job. This is an observation,
not a future balance guarantee. Refresh authenticated quota and active-job
status before any new allocation. No automatic use of renewed quota is
authorized by the frozen pilot plan.
