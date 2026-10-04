# Qwen2.5 code CPT r2 handoff

Worktree: `/home/crabcake/Projects/tabcomplete-q25-code-cpt-r2`.
Branch: `research/q25-code-cpt-r2`, from `0047e18`.
Private artifacts: `/mnt/ssd/tabcomplete-q25-code-cpt-r2`.

## Current execution at 2026-10-04 10:16 UTC

CPT and both matched FIM arms remain complete; direct FIM remains selected.
Do not retrain. Native Q4, real inference/latency and editor/SQLite delivery are
still pending. Stable desktop/ThinkPad configurations remain unchanged.

Root HEAD `b1bacdd` includes explicit CPU conversion revisions 1 through 4.
R2 failed after verified source setup at uv pip-prefix lookup; worker fixed by
extracting only the exact executable from its SHA-pinned wheel. R3 then verified
CPU Python3.11.15/Torch2.11.0+cpu, installed dependencies, cloned exact
llama.cpp `f072b103714dfa1eee531f80b24512faf38e3dd2`, built quantizer and reached
F16 conversion. It failed at 440.692 worker seconds because the pinned converter
imports sentencepiece before testing the absent tokenizer.model and falling back
to GPT2/BPE. Actual private converter traceback confirms ModuleNotFoundError for
sentencepiece. All logs/failed manifests are under `fim/conversion-failure-r3`;
SigNoz query again returned no_runs. No additional training or GPU allocation.

R2/R3 terminal CPU charges are strictly validated 333/561 seconds covering
submission to observed terminal plus 60 seconds. Original 10,800-second job
reservations remain untouched; both R1 reservations remain fully charged.
R4 gets separate plan/ref/paths and one explicit attempt only. Another three-hour
CPU reservation fits the unchanged twenty-hour cap; no automatic retry.

Campaign audit agent is adding hash-pinned sentencepiece to the CPU lock and
worker inventory. It MUST run the actual pinned converter's --vocab-only route
on selected local HF config/tokenizer before another allocation. No new weights.
Await its worker/lock/test commit, merge, freeze R4 plan, independently test,
commit and push exact clean source, refresh live quota/active jobs, then explicitly
submit `--execute-conversion --conversion-revision4`. Watch/collect matching R4.
Do not reuse R3 fingerprints after lock/source changes or retune fixtures.

Current verification at pushed source `31dad32`: Python1208, Lua86, collector59,
analysis38, gateway16 passed; Ruff/mypy93 and all TS checks passed. Root's newer
R4 controller has 173 focused passes plus mypy/Ruff. Full gate must rerun after
worker/lock integration. Replay hard deadline, sampling coverage, honest warm
cache start and Lua prompt blob vs binding hashes are merged. Native case SHA
remains 57cef82cc1a8cc5d4ddfdc25201b7e54f7f33ec6d2860acf14166bb5537127bc.

Selected native profile/embedding must use actual completed Q4 only, never old
stable weights. Generic release binary SHA remains d9912725b657c4806e01c592537133222a59dfc79f0d25083e2c1d9ae0fb4074.
Private artifact total about11.44GB of12GiB leaves space for Q4+embedded ELF.
Stop only the owned old predictor before loading candidate, restore on exit.
Freeze native quality/latency plans after final source/model/binary/PID identities.
Run240 exact+EOS cases and144 changed-state/cache latency requests, then real
synthetic Neovim accept/dismiss/match/undo/navigation and existing-DB replay/blob
verification. Prepared fixture is fim/native/editor-smoke, case4154,273-bytePython.
Existing SQLite online backup retained; no migration/wipe needed. Data audit agent
is ready to audit exact session/prediction IDs, projections and blobs read-only.
ThinkPad approved SSH alias still times out; retry later without scanning hosts.
Automatic personalization stays disabled. Continue through measured native
verification and safe experimental desktop deployment before a completion claim.

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

## Diagnostic repair after attempt 4

Attempt 4 reached the pinned runtime but failed before attention ran. The new
smoke subprocess reset CUDA peak-memory statistics before initializing CUDA.
Its worker elapsed time was 105.938 seconds and the verified receipt records
zero training tokens and no carried checkpoint. The fix explicitly initializes
CUDA and selects device 0 before resetting statistics or allocating tensors.
The deterministic CPU test verifies that order. Kernel support remains unknown
until the repaired diagnostic runs on the T4.

Keep the revision 4 plan and private input manifest unchanged. This diagnostic
repair changes no generation, trainer, prompt, fixture, scoring or data policy.
Attempt 5 must use immediate lineage
`--resume-source shlokbhakta/tc-q25-fim-r2-0-a4`; its effective checkpoint source
remains null. Run the full verification gates, commit and push before allocating,
and refresh live quota. The last recorded quota is historical, not a fresh fact.

The controller settlement ledger may discount only independently verified
terminal zero-training failures. It charges the rounded-up interval from
controller submission through terminal observation plus 60 seconds, including
queueing, setup, evaluations and failures. Active, unknown or trained work keeps
its full deadline reservation. Hashes bind jobs, watches, receipts, worker status,
archived plans and input manifests. The 20 session-hour, 40 conservative account
GPU-hour and 32 million processed-token limits remain unchanged. The specific
user authorization is recorded in `campaign_authorization_audit.json`; do not
edit AGENTS.md or silently increase caps to make a retry fit.

The final attempt-5 repair gate passed 1,037 Python tests with two warnings,
Ruff, mypy across 92 source files, 16 gateway Bun tests, 59 collector Bun tests
and both TypeScript checks. `verification-fim-attempt5.json` records actual log
and source hashes. The session ledger validates all four zero-work failures at
1,294 seconds total and remains idempotent on a repeated explicit settlement.
Manifest validation binds the archived plan, exact approved file set and staged
file hashes; stray training artifacts invalidate a zero-work claim.

Attempt 5 was submitted as `shlokbhakta/tc-q25-fim-r2-0-a5`, pinned to source
commit `52a31de7ea1af7b125ddac53a6ff4d2b71a46e9f`. Fresh quota observed
2026-10-04 05:44:34 UTC was 42.45 remaining account GPU-hours, 2.55 used,
renewal 2026-10-10, and no active jobs before submission. The existing observer
uses `--phase fim --arm untouched_q25_to_fim --attempt 5 --watch`. Watch only
this allocation; do not retry a running or uncollected job. Detailed training
progress is not available from the terminal-only artifact export.

The campaign-owned preview Rust build cache moved into
`/mnt/ssd/tabcomplete-q25-code-cpt-r2/fim-preview-build-cache`, with a symlink
from its original preview-worktree target path. The existing storage limiter
now counts it. `owned_build_storage.json` records 6.987 GB of total artifacts
including the 2.559 GB cache at that observation. Reuse the cache for required
tests; remove only this owned temporary cache afterward if needed for checkpoint
retrieval. Coordinate with the integration agent before removing an active cache.
Never remove existing research checkpoints to create headroom.

The source-only integration branch is `prototype/q25-fim-integration-r1`,
based on preview adapter `703fdf2`. It implements a separate research route,
not a replacement of the stable Qwen profile. FIM prompts use no added special
tokens, exact PSM, a 96-token ceiling including EOS, cursor-to-line-ending
replacement and request/context/model identity checks. The conversion agent
owns the separate preview worktree's CPU-only conversion helper. Neither is
deployed or selected yet. The exact current stable artifact's header and native
source show no BOS insertion mismatch; see `stable_bos_audit.json`.

## Untouched completion arm completed

Attempt 5 completed the full 4,096-state pass with 256 updates, zero skips and
1,776,908 input / 34,405 supervised response/EOS tokens. The verified complete
checkpoint SHA is `6f82f28a83c366b782bdb0629b2810a08bbf69fbf43d889a48be0fd94aafb6e2`.
Development exact+EOS improved from 47/240 to 130/240; observed EOS rose from
122/240 to 240/240. FIM line exact rose from 79/180 to 85/180; that historical
line diagnostic accepts newline stops and is not the native EOS serving gate.
Worker elapsed time was 2,677.064 seconds, including 1,375.464 seconds for the
training subprocess and before/after/regression evaluation. Pinned Python
3.11.15 and efficient attention completed successfully, including the unchanged
13,699-token line fixture. These are synthetic completion results, not general
next-edit accuracy or human acceptance. See `fim_results_untouched_a5.json`.

Next refresh authenticated quota and launch the CPT-initialized arm sequentially.
Preserve its exact frozen plan, order and runtime. Do not retrain the completed
untouched pass, select before the matched comparison, or change the stable editor.
The checkpoint was collected locally; conversion/export retrieval must continue
to respect the aggregate artifact cap and remove only owned temporary build
cache when safe. Generic functional replay and source-syntax diagnostics remain
pending. Automatic personalization remains disabled.

The CPT-initialized arm is submitted as `shlokbhakta/tc-q25-fim-r2-1-a1`,
pinned to `99bcfd3f6baab8c8ff59086b238e816ba3927969`, at 06:35:16 UTC.
Fresh pre-submit quota observed 06:34:38 UTC was 41.71 remaining account
GPU-hours, 3.29 used, no active jobs and the unchanged October 10 renewal.
Its observer uses `--phase fim --arm completed_cpt_q25_to_fim --attempt 1 --watch`.
Do not launch another GPU allocation while this one is running.

CPU source diagnostics for the untouched FIM arm completed: parser regressions
against originally passing source declined from 94/238 to 11/238. The strict
post-FIM raw causal suite passed 11/200 functional, 19/200 compile and 22/200
parse cases with all eight frozen Docker image IDs unchanged. Do not call the
small causal difference a clean paired gain over the earlier runtime.
The worker offline bundle imported 4,102 spans; a second import returned duplicate
with zero spans. The training run was fully paginated through four pages and
357 unique spans, including its completed state. Full worker bundle spans include
separate evaluation runs. See the diagnostic and monitoring records.

The FIM route commits `703fdf2` and `1948c196` have been cherry-picked into the
campaign branch. Root independently reran the Neovim suite: 85 passed, zero failed.
Agent verification also passed 29 Rust tests, but selected-model/native Unicode
parity and actual SQLite integration remain pending. Preserve those distinctions.
The shared debug Cargo cache was removed after its owner confirmed tests finished.
Only owned temporary build artifacts were removed; all research checkpoints remain.
A smaller release-only native build cache is now under the campaign artifact root
and therefore counted by its cap.

## Both FIM arms complete; direct FIM selected for Q4 verification

Second kernel `shlokbhakta/tc-q25-fim-r2-1-a1` completed; verified receipt
`fim-verified-completed_cpt_q25_to_fim-1.json` binds the full 4,096-state cursor.
Direct FIM scored 130/240 versus CPT→FIM 127/240 exact+EOS. Three CPT gains and
six losses yield -1.25pp with repo-bootstrap CI [-3.78,+1.25]pp; no gain or
equivalence established. Selected conversion input is untouched arm a5 only.
`fim_conversion/selection.json` and `fim_quality_comparison.json` are root's
manual immutable evidence. Don't rerun either completed arm or overwrite stable.

CPU-only conversion controller is being finished by campaign audit agent. Its
source must be committed/pushed before allocation, with pinned CPU worker,
selected kernel-source attachment, immutable config dataset, 3h deadline and
30min finalization reserve. Count all attached optimizer states in worker disk
budget; local collection keeps only Q4+compact manifests, never FP16 weights.
Root artifact total11.45GB leaves1.435GB under12GiB cap, enough for selectedQ4
and compact release build, not another unquantized export or debugcache.

Native evaluator request order must tokenize first, prepare FIM context last,
then immediately complete; `/tokenize` invalidates one-shot prepared binding.
Trainer audit agent is fixing strict resume evidence, raw tokenizer identity,
source digest and actual PID/listening-port attestation in root evaluator/tests.
Data audit agent is completing bounded640/256 context route, retained NFC guard,
Lua source-range verification and bound malformed-output terminal evidence.
Merge and independently test those before freezing selected native evaluation.
Actual Q4 quality, latency, collector/Neovim smoke and installer remain pending.
ThinkPad known SSH alias still unreachable at latest check; no activation claimed.
Automatic personalization stays disabled. Continue through conversion and local
verified preview; do not stop after the matched pilot.


## Native delivery checkpoint at 2026-10-04 08:35 UTC

Both trained arms remain complete and selected direct FIM remains unchanged.
CPU conversion attempt 1, `shlokbhakta/tc-q25-fim-q4-conversion-r1`, pinned to
`c5ef7b0`, failed in its launcher at 0.798 seconds. Its pulled source is
byte-identical to the staged launcher; remote metadata confirms both expected
attachments and GPU disabled. No worker conversion manifest exists. The actual
SigNoz run query returned `no_runs`. Preserve `fim-conversion-failed-1.json`,
job/watch/quota and private log. An explicit audited retry is being implemented;
never blindly push the original reference again or erase its receipt.

The integrated controller requires branch `research/q25-code-cpt-r2`. Its exact
five plan-hashed worker files are embedded in the single uploaded script, with
bounded nested-mount discovery. The temporary root must be safely created before
extracting those files. The worker's source hashes, selected export, tokenizer,
training plan and fixtures are unchanged.

Native evaluator correction is committed as `1302b8b`: it now reproduces the
original 640/256 training crops and complete Rust context-v2 digest. Root reran
32 focused tests and actual 240-case preparation. The case SHA remains
`57cef82cc1a8cc5d4ddfdc25201b7e54f7f33ec6d2860acf14166bb5537127bc`.
The old full-file checker mismatched 83 prompts; no native model outputs existed
before correction. Do not reuse the obsolete context-v1 names or digest.

Opt-in executable packaging merged as `b48f245`. Root independently passed 40
Rust tests and the synthetic appender/tampering/cap tests, then rebuilt release.
The root generic engine SHA is
`d9912725b657c4806e01c592537133222a59dfc79f0d25083e2c1d9ae0fb4074`.
This differs from the agent's historical build hash; attest the actual binary.
The installer profile helper emits `embedded-profile.json` with full sorted
vocabulary/control IDs, alongside compact `editor-model.json` and registry.
The separate opt-in Nix candidate leaves Qwen/Sweep defaults unchanged.
Actual selected-GGUF mapping, quality, inference latency and SQLite/Neovim smoke
are still pending. Never call synthetic footer tests actual model inference.

Next continue the explicit CPU retry, retrieve only Q4 and compact manifests,
freeze actual native identities, run all 240 cases, then the bounded changed-state
replay and real synthetic editor/collector smoke. Temporarily stop only the owned
old predictor before loading the candidate and restore it on exit. ThinkPad's
known SSH alias still times out; don't claim installation there. Automatic
personalization remains disabled.
