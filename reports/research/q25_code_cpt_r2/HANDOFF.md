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
tests and the submission URL test passed; the controller suite now has 32 tests.
