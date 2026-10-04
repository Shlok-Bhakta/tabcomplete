# Qwen2.5 code CPT r2 handoff

Worktree: `/home/crabcake/Projects/tabcomplete-q25-code-cpt-r2`.
Branch: `research/q25-code-cpt-r2`, from `0047e18`.
Private artifacts: `/mnt/ssd/tabcomplete-q25-code-cpt-r2`.

Read `plan.json` and the scientific report before resuming. Preserve the running
ThinkPad/editor configuration and stable weights. Do not consume a renewed
Kaggle allocation automatically or download a different model.

The controller freezes identities, stages the CPU-prepared corpus, uploads private
inputs, submits one bounded session, and verifies retrieved complete checkpoints:

```bash
uv run python scripts/run_q25_code_cpt.py --bundle --upload
uv run python scripts/run_q25_code_cpt.py --execute
uv run python scripts/run_q25_code_cpt.py --watch --attempt 1
uv run python scripts/run_q25_code_cpt.py --collect --attempt 1
```

A later attempt must refer to the immediately previous exited campaign kernel
with a verified complete checkpoint. Preserve the fixed total learning-rate
horizon and batch order. Do not start again at example zero when resuming.
An ambiguous submission must be reconciled against authenticated kernel status;
never retry a push blindly.

CPU preparation is complete. The immutable private input dataset is
`shlokbhakta/tabcomplete-q25-code-cpt-r2-inputs`. It contains 7,872,512 training
input tokens and 131,072 development tokens. The worker also mounts the existing
`shlokbhakta/tabcomplete-one-line-instinct-pilot-r1-inputs` dataset for the exact
untouched Qwen weights. No new weights were downloaded.

The trainer preflight passed and resolves 481 updates. Actual corpus files live
in `/mnt/ssd/tabcomplete-q25-code-cpt-r2/corpus`. The data and trainer entry points
are `scripts/prepare_q25_code_cpt.py` and `python -m tinycomplete.code_cpt.q25`.
Do not regenerate the corpus or upload replacement files for a resumed run.

Use the existing shared development environment when running verification:
`UV_PROJECT_ENVIRONMENT=/home/crabcake/Projects/tabcomplete-product-r2/.venv`.
CPU PyTorch is available through
`/home/crabcake/Projects/tabcomplete/.venv/lib/python3.11/site-packages`; include
it after `src:scripts` in `PYTHONPATH` for trainer tests. Local CUDA is unavailable.

Do not describe an allocation or a model gain as completed based on this handoff
alone. Commit and push the campaign code before submission; the controller
enforces that the remote branch matches the local commit.

Attempt 1 is running at the actual Kaggle reference
`shlokbhakta/tabcomplete-q25-code-cpt-r2-attempt-1`, launched from `fce293d`.
The short requested ID was not created; do not retry submission under that ID.
Inspect the actual reference, then collect attempt 1 only after COMPLETE/ERROR.
The job receipt and fresh quota observation are committed beside this file.
No second notebook may be allocated while it remains active.

The optional watcher retrieves terminal output without starting another job. It
stops after three consecutive transport failures or its observation deadline.
The GPU worker still enforces its own deadline if the watcher disconnects. Watch
state is operational evidence, not a completed training-token count. Five watcher
tests and the submission URL test passed at that milestone. The later controller
suite has 78 passing tests.

## Matched completion follow-up

Keep the active CPT trainer and shared training/evaluation helpers unchanged while
attempt 1 runs or resumes. The worker pins `fce293d`; checkpoint fingerprints bind
trainer source, data order, runtime and model identity. Changes to the controller
and a new FIM trainer do not change that active worker.

The broader user-authorized budget is in `campaign_budget.json`. The controller
shares a 32-million processed-token ceiling across CPT and both FIM arms, keeps
one allocation active, reserves future FIM time, and refuses renewal consumption.
CPT still has its own 12-million processed-token cap and one source pass.

FIM preparation revision 3 is active. Its private corpus is
`/mnt/ssd/tabcomplete-q25-code-cpt-r2/fim/corpus-r3`, with 4,096 training states,
240 development states, 1,776,908 training input tokens and 34,405 supervised
response/EOS tokens per arm. Revisions 1 and 2 remain historical. Revision 3
rejects tokenizer normalization mismatches before selection; no FIM model outputs
preceded that correction. Every selected prompt and target round-trips exactly.

After the CPT allocation exits, retrieve and inspect its scientific outputs. A
partial pass resumes from its verified complete state, never from example zero.
Only a complete 481-update CPT pass may become the second FIM initializer. Then:

```bash
uv run python scripts/run_q25_code_cpt.py --freeze-fim --cpt-attempt 1
uv run python scripts/run_q25_code_cpt.py --phase fim --bundle --upload
```

The freeze step binds the actual CPT runtime and export. The upload mounts only
the completed export in a new private dataset, with no optimizer states. After
remote path/size verification it removes the owned temporary staged weight copy;
the original research export and training checkpoint remain intact. A retry of
a verified upload does not recopy the weights.

Run CPU preflight against the frozen full plan for both initializers. Commit and
push the completed source and upload receipts before GPU allocation. Then launch
one arm, watch and collect it before launching the other:

```bash
uv run python scripts/run_q25_code_cpt.py --phase fim --arm untouched_q25_to_fim --execute
uv run python scripts/run_q25_code_cpt.py --phase fim --arm untouched_q25_to_fim --watch
uv run python scripts/run_q25_code_cpt.py --phase fim --arm completed_cpt_q25_to_fim --execute
uv run python scripts/run_q25_code_cpt.py --phase fim --arm completed_cpt_q25_to_fim --watch
```

Each FIM allocation has a three-hour deadline including setup, evaluations and
saving. It records before/after development and FIM line outputs, plus raw causal
and raw line regressions after training. Resumes carry the original before-FIM
baseline files across every attempt. The developer baseline is synthetic source
completion, not observed next-edit intent. Compare paired cases and resample
repository groups with the frozen seed; an inconclusive small sample is not
model equivalence.

`verification-followup.json` records the last completed full Python gate.
Additional worker and upload tests have their own actual results; do not add
their counts to that historical full-suite count. The original Bun services were
not modified. `evaluation_environment.json` freezes the existing Docker images;
the gold-reference executor check passed 200 cases, which is environment evidence
and not a model score. `gold_monitoring_verification.json` records actual paginated
SigNoz retrieval. Import the GPU worker's offline bundle after collection.

The stable ThinkPad editor remains on its existing trained N/R protocol. A FIM
candidate must use a separately versioned serving contract; its PSM weights must
not be silently served through the old N/R prompt. Do not promote a new weight
artifact merely because CPT loss falls. Automatic personalization stays disabled.

The final follow-up gate passed 966 Python tests with two warnings, Ruff, and
mypy across 90 source files. Gateway Bun tests passed 16 cases; collector Bun
tests passed 59 cases. Both service type checks passed. Worker checks cover 10
CPU cases, including explicit persisted training-start status. A zero-training
FIM failure can retry with immediate allocation lineage and its unchanged
initializer or older verified checkpoint. An unknown post-start state cannot
claim zero work. Upload staging rejects all unlisted files before copying.
