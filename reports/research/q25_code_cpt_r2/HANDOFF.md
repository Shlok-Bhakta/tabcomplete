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

Attempt 1 completed at the actual Kaggle reference
`shlokbhakta/tabcomplete-q25-code-cpt-r2-attempt-1`, launched from `fce293d`.
The verified final cursor contains 481 completed updates and 7,872,512 input
tokens, with no skips or replayed tail. The full state and FP16 inference export
are collected in the private artifact directory. Do not resume this completed
source pass. The short requested ID was not created; do not retry under that ID.
Refresh authenticated quota and active jobs before either FIM allocation.

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

The freeze step binds the actual CPT export and the new matched Python 3.11.15
dependency lock. CPT actually ran Python 3.13.15; this deviation is recorded and
must not be carried forward. The upload mounts only
the completed export in a new private dataset, with no optimizer states. After
remote path/size verification it removes the owned temporary staged weight copy;
the original research export and training checkpoint remain intact. A retry of
a verified upload does not recopy the weights.

Full FIM plan revision 3 is active. Original revision 1 and its unused private
input staging/receipt are preserved under `fim_training_plan.r1.json` and
`/mnt/ssd/tabcomplete-q25-code-cpt-r2/fim/history/plan-r1`. CPU preflight fixed
ordinary reserved FIM token handling and semantically equivalent Transformers 5
config serialization before any FIM generation. The revised private input dataset
is `shlokbhakta/tabcomplete-q25-fim-r2-inputs-r3`; both initializer preflights
pass. Revision 2 remains in `fim_training_plan.r2.json` and `fim/history/plan-r2`.
Revision 3 fixes a runtime-report checksum typo found by CPU tests before any
FIM generation. The wheels, versions, examples, weights and tokenizer IDs stay
unchanged.

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

## Completion allocation history

Attempt 1 of the untouched arm failed before training after 12.176 worker
seconds: all 15 NVIDIA versions matched, but NVSHMEM failed the strict file
layout check. `fim-verified-untouched_q25_to_fim-1.json` is a verified zero-work
receipt. No completion quality outputs exist. Fix the vendor layout check on
CPU, preserve the frozen lock, and retry with attempt 2 and
`--resume-source shlokbhakta/tc-q25-fim-r2-0-a1`. This records authorization
lineage while retaining no checkpoint source, since training never started.
Never relaunch attempt 1 or count it as trained. Refresh quota before allocation.

The serving audit confirmed that the new PSM checkpoint needs a separate
`q25-fim-line-completion-v1` route, with an exact cursor-to-line-ending range.
Its target includes original LF/CRLF, or no newline at EOF; trim/repair would
change the trained contract. The native converter revision `f072b103` reads
normalized RoPE theta 1,000,000 correctly, but has no direct Q4 writer. Account
for temporary F16 and Q4 artifacts even when tmpfs holds the intermediate.
If local cap headroom is insufficient, use an owned CPU-only research conversion
job attached to the selected training output, then retrieve only its Q4 artifact.
Do not quantize/deploy until actual paired completion results support selection.

Attempt 2 then passed the CUDA inventory but stopped at pinned uv verification:
`python -m uv` found Kaggle host uv 0.12.9, not the verified wheel 0.12.3.
Its zero-work receipt is `fim-verified-untouched_q25_to_fim-2.json`; no FIM
training occurred. Fix/test explicit native uv invocation before attempt 3.
Use attempt 3 with `--resume-source shlokbhakta/tc-q25-fim-r2-0-a2`, not the
older attempt. The inherited checkpoint source remains null.

## FIM evaluator revision 4

Attempt 3 reached the pinned Python 3.11.15 runtime and completed 240
unadapted development cases. Full line case 111, Rust
`rust/af541fc7d7054c4f6dc5`, has 13,699 input tokens. Native GQA made
PyTorch 2.11 use math attention on the SM75 T4, materializing a 9.79 GiB
FP32 score tensor and exhausting memory. The worker stopped after 674.293
seconds with persisted `training_started=false`; no training tokens were spent.
The failed run and trace were imported and queried in SigNoz. Details are in
`fim_evaluation_failure_untouched_a3.json`.

Revision 4 freezes explicit KV-head repetition plus supported memory-efficient
SDPA for evaluation. It preserves full prompts, source fixtures, targets,
scoring, model/tokenizer, ordered training examples and trainer settings.
Earlier evaluation outputs remain historical; both arms rerun their comparisons.
`fim_training_plan-r3.json` preserves the old plan. The private input dataset
is now `shlokbhakta/tabcomplete-q25-fim-r2-inputs-r4`. Do not reuse the old
input manifest or claim that a numerical backend change is identical execution.

Attempt 4 must use `--resume-source shlokbhakta/tc-q25-fim-r2-0-a3` for
authorization lineage, while the actual checkpoint source remains null. The
runner permits this evaluator-only revision only after validating the exact
archived plan and an explicit zero-work receipt. Changed training, prompts,
fixtures, scoring, processed tokens or carried checkpoint state fail closed.
Refresh live quota and run all gates before allocating. The three failed FIM
allocations still reserve their full three-hour limits in the shared ledger.

The detached completion preview prototype is commit `703fdf2` on
`prototype/q25-fim-preview-r1`, in the separate preview worktree. Its Rust and
Lua adapters are tested, but have no serving route or deployment. Actual
terminal token ID 151643, complete special-token inventory and selected artifact
identity must be bound before integration. Do not serve FIM weights as the old
N/R next-edit protocol.

The latest ThinkPad SSH attempt timed out. No new target deployment or hardware
measurement was performed. Read-only feedback export observed 284 proposal
records across 88 sessions, eight collection-gap intervals and zero defensible
preference pairs. The private export stays outside Git. Counts mix synthetic
and other records and do not establish human acceptance. Personalization stays
disabled. Public single-recorder Rust/Svelte traces replayed exactly but lack
cursor/intent evidence; use them as mechanics fixtures, not training-ready
human preference data.
