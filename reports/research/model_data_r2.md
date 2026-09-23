# Model and data research R2

Status: **research evaluation complete; local 32k timing still running**. Both
matched pilots are checkpoint-verified. Their strict and line results fail the
registered promotion rule, so neither checkpoint is promoted and the sealed
research test remains closed. The active playground checkpoint has not changed.
The registered context job ended with runtime blocks; its valid partial scores
are reported below.

## Evidence available so far

The required initial comparison completed all 200 corrected causal fixtures on
T4 using Transformers 5.5 and Torch 2.10, FP16 text-only models, raw causal
prompts, greedy decoding, and the unchanged 96-token ceiling. Saved predictions
were executed on Crabcake in the existing network-disabled containers.

| Model / runtime | Causal passes | Line exact match | Returned-line latency | Peak RAM | Verified context | Status |
|---|---:|---:|---|---|---|---|
| P12, Transformers FP16 / T4 | 9/200 | 20/180 | 0.548 s median T4 line call | Not measured as host RSS | 2k parity gate failed; no R2 long-context score | Executed strict/line; context blocked |
| Qwen2.5-Coder, Transformers FP16 / T4 | 11/200 | 16/180 | 0.182 s median T4 line call | Not measured as host RSS | 25 scored 32k-family cases, including five genuine far dependencies; later OOM | Executed strict/line; partial context |
| D12, Transformers FP16 / T4 | 5/200 | 20/180 | 0.583 s median T4 line call | Not measured as host RSS | 25 scored 32k-family cases, including five genuine far dependencies; later OOM | Verified historical strict reuse; partial context |
| Qwen2.5-Coder, native Q4_K_M / Kiwi CPU | 7/200 | 18/180 | 2.641 s median across 180 line cases, Kiwi two-thread CPU | 1.405 GiB in separate Crabcake 2k grid | Line/strict fixtures; not a long-context intelligence claim | Strict and corrected line executed |
| P12, native Q4_K_M / Kiwi CPU | 8/200 | 19/180 | 3.694 s median across 180 line cases, Kiwi two-thread CPU | 3.033 GiB in separate Crabcake 2k grid | Crabcake 2k and 8k timing grids executed | Strict and corrected line executed |
| Base Qwen3.5, Transformers FP16 / T4 | 4/200 | 18/180 | 0.410 s median T4 line call | Unknown | Efficient attention kernel unavailable on T4 | Executed strict/line; context blocked |
| R2_STANDARD, Transformers FP16 / T4 | 4/200 | 19/180 | 0.580 s median T4 line call | Unknown | No registered long-context score | Executed; not promoted |
| R2_FILTERED, Transformers FP16 / T4 | 5/200 | 19/180 | 0.548 s median T4 line call | Unknown | No registered long-context score | Executed; not promoted |
| Granite H-350M, Transformers FP16 / T4 | 3/200 | Incomplete, 1/180 generated | Unknown | Unknown | No registered long-context score | Strict executed; line OOM |
| Granite H-350M, native Q4_K_M / Kiwi CPU | 0/200 | 6/180 | 2.725 s median across 180 line cases, Kiwi two-thread CPU | 1.981 GiB in separate Crabcake 2k grid | Line fixtures and 1,171–2,005 actual input tokens in the 2k grid | Strict and corrected line executed |

These strict counts are same-task comparisons under the frozen 200-case protocol.
Qwen2.5 FP16 leads at 11/200, with P12 at 9/200, but their paired difference
remains uncertain. Qwen2.5 uses fewer parameters and has a 0.182-second median
T4 line call here; its native Q4 result is 7/200 strict and 18/180 exact lines,
close to P12 Q4's 8/200 and 19/180 on the same task. That makes Qwen2.5 the
most useful smaller-model research direction. It is not an established
replacement for P12, and none of these causal tasks measures next-edit quality.
Base Qwen3.5 and Granite trail on strict quality in this run. The two trained
pilots regress on strict and line tests, so neither is a promotion candidate.

Strict functional passes by language, from the same 200 cases:

| Language (cases) | P12 | Qwen2.5 | D12 | Qwen3.5 base | Standard | Filtered | Granite FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C (22) | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| C++ (22) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C# (22) | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| Go (22) | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| Java (22) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| JavaScript (22) | 1 | 2 | 0 | 0 | 0 | 0 | 2 |
| Python (23) | 5 | 8 | 4 | 0 | 2 | 2 | 0 |
| Rust (22) | 1 | 0 | 0 | 1 | 1 | 1 | 0 |
| TypeScript (23) | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| Total (200) | 9 | 11 | 5 | 4 | 4 | 5 | 3 |

Exact line continuations by language, 20 cases per language:

| Language | P12 | Qwen2.5 | D12 | Qwen3.5 base | Standard | Filtered | Granite Q4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| C++ | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| C# | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Go | 5 | 3 | 5 | 4 | 4 | 4 | 1 |
| Java | 3 | 2 | 3 | 3 | 3 | 3 | 0 |
| JavaScript | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| Python | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| Rust | 3 | 2 | 3 | 2 | 3 | 3 | 2 |
| TypeScript | 3 | 3 | 3 | 3 | 3 | 3 | 1 |
| Total (180) | 20 | 16 | 20 | 18 | 19 | 19 | 6 |

The complete per-language strict and line tables, including all three native
Q4 models, are in `model_data_r2/baseline_evaluations/per-language.json`. Most
strict passes occur in Python; language-specific rates here have only 22 or 23
cases each and are descriptive.

The context notebook finished its 1,380-second observed session with four
runtime blocks. P12 observed the efficient-attention operator and identical
short greedy output, but its short target NLL difference was 0.003134, above
the registered 0.002 limit. Base Qwen3.5 had no available efficient-attention
kernel on the T4 at the parity gate. D12 and Qwen2.5 passed parity, with
0.001828 and 0.000023 mean absolute target NLL differences respectively, and
identical short greedy output. Each saved all 25 cases in the 32k family before
later CUDA memory errors during 16k cases. Within that family, five cases per
model carry a genuine far dependency. D12 preferred the reference target in
3/5 at actual dependency distances 31,932–32,054 tokens; Qwen2.5 did so in
4/5 at 30,820–30,939 tokens. Other 32k-family cases include near, absent and
short controls; the family label alone does not mean a 32k prompt. No model
completed the registered context suite or produced its planned 96-token
generation records. These partial likelihood preferences do not establish
long-context coding ability. The operator, head geometry, parity values,
actual token positions, scores and runtime errors are recorded in
`model_data_r2/long_context/implementation_record.json` and the collected
context artifacts.

P12's clean 2k deployment grid used a Ryzen 5 PRO 3400GE on Crabcake, four
threads, single-request concurrency, and 20 identical-source prompts with two
repetitions. Median total request time was 22.089 seconds; this includes prompt
processing and up to 32 output tokens, **not** pure decode time or a stopped-line
latency. The first process-cold request took 24.618 seconds. File caches were not
flushed, so this is not a cold-disk measurement. All 20 output pairs matched.
Prompt-cache reuse was zero. Do not compare these CPU measurements with the T4
generation durations or with Kiwi's two-thread quality-run timings.

On the same clean 2k source grid, Qwen2.5's median total request was 20.723
seconds and Granite's was 9.722 seconds. Each has 40 observations and identical
paired outputs. Tokenization differs: Qwen2.5 used 1,984 input tokens in every
case, while Granite used 1,171–2,005. These are identified-host, same-source
deployment observations, not equal-token throughput or laptop/phone estimates.

Qwen2.5 versus P12 had nine new passes, seven lost passes, and two shared passes.
The difference is +1 percentage point, with paired-bootstrap 95% interval
[-3, +5] points and exact paired binomial p=0.804. This does not establish a
quality winner. The same token ceiling also permits different amounts of text:
P12 reached it in 146/200 cases and Qwen2.5 in 106/200. Character/byte counts are
preserved beside each prediction.

P12's completed 8k Crabcake grid has 40 fixed 32-token requests, a median total
time of 111.592 seconds, p95 of 118.596 seconds, and peak server RSS of
4,160,626,688 bytes. Actual inputs range from 8,445 to 8,993 tokens. This is
latency and memory evidence for that host and runtime; it does not score whether
the model used distant context correctly.

The Qwen2.5 8k grid has 40 completed requests with a median total time of
114.217 seconds, peak server RSS of 2,858,635,264 bytes, and 8,128 actual
input tokens per request. A 685 MB transfer to Kiwi overlapped
both repetitions of `runtime-8192-07` around 03:56 to 03:57 UTC on 2026-09-23.
The later Granite F16 diagnostic hashed that local model file during the
`runtime-8192-15` pair. The filtered checkpoint download overlapped the final
`runtime-8192-19` pair. These I/O effects were not measured. All original
observations stay in the 40-request summary, with the overlaps recorded in
`model_data_r2/local_inference/q25_8k_transfer_overlap.json`. Removing those
three pairs only for a descriptive sensitivity check changes the median from
114.217 to 114.247 seconds; the official summary retains all 40 requests.

Granite's completed 8k Crabcake grid has 40 requests, a 36.927-second median,
44.986-second p95, and 2,550,587,392-byte peak server RSS. Its actual input
range is 6,592 to 8,076 tokens, below the other models' 8k-grid token counts.
The intentional pause between Granite request pairs is recorded outside request
durations. These measurements compare fixed source prompts and output ceilings,
not equal token counts or distant-context correctness. The per-bucket source is
`model_data_r2/local_inference/summary.json`.

Qwen2.5 Q4 versus its own FP16 checkpoint had two gains and six losses. Runtime
and precision both change in that comparison; it is not a pure quantization
ablation. Five lost cases involve extra generated code that breaks compilation;
the remaining loss is a negative-input GCD logic error. See
`model_data_r2/failure_audit/q25_quantized_changes.json` for actual assertions and
compiler diagnostics. No official output was repaired or gold-truncated.

P12 Q4 versus its own FP16 control has four gains, five losses and four shared
passes, a −0.5-point difference with paired-bootstrap 95% interval [−3.5, +2.5]
points. Its five losses fail compilation after extra methods/functions or emitted
file markup. The gain/loss audit is saved in
`model_data_r2/failure_audit/p12_quantized_changes.json`. One control case,
`python/100`, passes with an empty FP16 completion: strict functional success in
this fixture does not prove a useful nonempty suggestion. The original fixture,
suffix and tests remain unchanged, and no oracle stopping or repair is applied.

All 200 newly generated P12 completions exactly reproduced their R1 counterparts.
D12 reuse additionally verifies its weight fingerprint, tokenizer, suite hash,
decoding and protocol. Reused results are explicitly historical; no request
telemetry was invented for them.

The original line fixture admitted generated Doxygen indexes and embedded byte
arrays. Plan revisions 10–11 record that source-only audit bug after Qwen2.5 Q4 had
scored 21/180 on the original fixture. That result is superseded, not silently
rescored. Twelve files were replaced by deterministic ordering within the same
held-out source pool. The corrected 180-case fixture hash is
`2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead`.
All reference restorations pass syntax checks; exact-content and repository-alias
training overlap remain zero. No training policy, training array, strict causal
case, or model-dependent stopping rule changed. Raw earlier outputs are retained.
The intermediate correction missed generated-tool signatures; its partial P12
run was cancelled before inspecting its quality. Final exclusions include ANTLR,
gRPC, JNAerator, AutoRust and Swagger signatures, generated suffixes, documentation
indexes and embedded byte arrays. They are heuristics, not perfect provenance.

On that corrected fixture, P12 Q4 and Qwen2.5 Q4 share 15 exact continuations;
Qwen2.5 gains three and loses four. Its difference is −0.56 percentage points,
paired-bootstrap 95% interval [−3.33, +2.22] points, exact paired p=1.0.
Syntax-valid insertions are 151/180 and 136/180 respectively. These are not
functional pass counts. The fixed 20-case source-hash-selected audit contains
underdetermined imports, names, constants and inline-comment remainders; exact
mismatch alone does not establish wrong logic. Empty responses and syntax
failures are labeled separately. The quality-run latency values in the table
are single observations per variable-length source case, not the repeated,
controlled runtime grid or evidence of a large speed advantage.

Granite Q4 matches 6/180 lines exactly and yields 80/180 syntax-valid insertions.
Against P12 Q4, it gains two exact matches, loses 15, and shares four. The
paired difference is -7.22 percentage points, with a 95% paired-bootstrap
interval of [-11.67, -3.33] points and nominal exact paired p=0.00235. Its
20-case audit includes repeated output. The registered unquantized Granite line
run failed after one case with a 13.50 GiB allocation request on the T4, so
its 180-case FP16 line score is unavailable. The later fixed 20-case native F16
diagnostic below is exploratory and cannot replace it.

The Granite Q4 strict run passed 0/200 unchanged functional cases. All 200
predictions were judged in the network-disabled containers: 189 failed compile,
10 reached tests and failed, and one reached tests and timed out. There were no
evaluator error statuses. This is a poor result for that runtime and precision,
but the registered unquantized Granite control is needed to localize the cause.
Against P12 native Q4 on the same 200 cases and host, Granite has no gains and
eight lost passes, a -4-point paired difference with 95% bootstrap interval
[-7, -1.5] points and nominal exact paired p=0.0078125. Model weights,
tokenizers, and architecture differ, so this same-runtime contrast does not
identify one mechanism.

The independent Transformers FP16 Granite strict run passed 3/200. It gained
three passes relative to native Q4's 0/200 on the same strict suite, but runtime
and precision changed together and the FP16 line phase failed. The measured
native F16 sample and strict FP16 control make precision or conversion
sensitivity plausible; neither isolates it. The failed line run has one saved
prediction, a completed-case telemetry count of one, and an error span for the
second case. It is not scored as 0/180.

An explicitly exploratory Granite native F16 diagnostic used the same 20 frozen
audit IDs on Kiwi with two threads. F16 matched 3/20 lines exactly and passed
syntax on 19/20; Q4 matched 0/20 and passed syntax on 13/20. F16 gained three
exact matches and six syntax passes without a loss in this sample. One Rust Q4
completion repeated `DERP` until the cap where F16 restored the exact line.
These post-outcome diagnostic results implicate precision or conversion
sensitivity. They do not replace the official 180-case Q4 result. The
independent Transformers FP16 strict control is complete, while its line phase
failed after one case. The case list, paired changes, hashes and run ID are in
`model_data_r2/models/granite-native-f16-diagnostic-result.json`.

## Frozen data experiment

Both arms use the same new frozen upstream candidate pool and identical packed
language allocations. Each contains exactly 5,013,504 input tokens. Standard
selects 4,642 files; filtered selects 6,306. All source identities, known-exclusion
checks, packed fingerprints, rejection counts, parser coverage, repository
concentrations, and deterministic samples are under `model_data_r2/data_audit/`.

| Language | Candidate files / retained pool | Selected files standard / filtered | Packed input tokens per arm | Largest repository % standard / filtered |
|---|---:|---:|---:|---:|
| C | 12,544 / 11,721 | 94 / 371 | 301,056 | 35.54 / 1.90 |
| C++ | 5,218 / 4,922 | 220 / 513 | 501,760 | 39.42 / 1.67 |
| C# | 3,966 / 3,744 | 450 / 490 | 301,056 | 2.46 / 1.92 |
| Go | 5,305 / 4,801 | 240 / 371 | 401,408 | 7.72 / 1.94 |
| Java | 4,311 / 4,089 | 548 / 584 | 501,760 | 3.51 / 1.99 |
| JavaScript | 4,784 / 4,028 | 414 / 555 | 501,760 | 8.76 / 1.84 |
| Python | 2,690 / 2,534 | 821 / 954 | 1,001,472 | 7.43 / 1.82 |
| Rust | 4,120 / 3,780 | 265 / 368 | 501,760 | 17.84 / 1.94 |
| Shell | 827 / 730 | 477 / 687 | 249,856 | 12.30 / 1.97 |
| TypeScript | 4,081 / 3,641 | 1,113 / 1,413 | 751,616 | 7.28 / 1.80 |

All selected files received parser checks; 353 standard-arm files failed that
syntax check, versus zero filtered-arm files by construction. This is not proof
that all rejected files are invalid in their intended dialect. Rejected raw
source was not tokenized or retained, so raw pre-filter candidate-token totals
are explicitly unknown. Retained-pool token totals, source-length quantiles,
test-file shares, reason counts, and known duplicate rates are in the reconciled
audit. Packed counts include EOS boundaries and can differ from full selected
file token totals when the last file is cut at the fixed budget.

The largest standard-arm repository contributes about 39.4% of C++ tokens.
Filtered keeps every language's largest canonical repository below 2%. Added
syntax and repository-cap filtering do substantial work; added exact/near
duplicate rejection did not remove selected candidates in this pool. That is a
measured treatment description, not a declaration of better code quality.

All 200 deterministic sample identities were checked. The 180 line fixtures
have no source-hash or repository-alias overlap with either selected training
arm. Original foundation pretraining is unknown, and old Stage-1 provenance is
incomplete. Parser rejection can reflect upstream language-label mismatches:
one standard C sample is actually a generated Objective-C header. Repository
license metadata also differs from some file notices. The audit preserves those
limitations and does not claim newly certified permissive-only data. Checkpoints
and packed source remain private.

A post-hoc coverage audit using the broader diagnostic markers flags 147/4,642
standard files and 107/6,306 filtered files. These were not the frozen treatment's
markers, so the audit does not change either arm. Likewise, upstream language
mislabels can cause false syntax rejection, as the Objective-C example shows.
Conclusions concern the policy actually applied, not a perfect generated-code or
language-aware validity filter. These are limitations to fix in a future corpus
revision, not grounds to relabel this experiment's data after training began.

| Candidate | Parent | Data policy | Added pilot tokens | Dev NLL change | Functional wins/losses | Line-result change | Decision |
|---|---|---|---:|---|---|---|---|
| R2_STANDARD | Verified P12 | Existing Stage-1 policy on fresh pool | 5,013,504 | −0.005340 (1.034572 → 1.029232), aggregate fresh dev | 1 gain, 6 losses; 4/200 vs 9/200 | 19/180 vs 20/180; 0 gains, 1 loss | Retain for research; do not promote |
| R2_FILTERED | Verified P12 | Frozen extra filters and repository balancing | 5,013,504 | −0.006722 (1.034572 → 1.027850), aggregate fresh dev | 1 gain, 5 losses; 5/200 vs 9/200 | 19/180 vs 20/180; 0 gains, 1 loss | Retain for research; do not promote |

Both schedules completed 153 successful updates, 32,768 input tokens/update,
five-update warmup to 3e-6 and cosine decay to 3e-7 at update 153. Each scored
5,011,056 targets, with zero skipped updates, loss-scale overflows, or nonfinite
values. Filtered's aggregate fresh dev NLL is 0.001382 lower than standard's;
repository-paired uncertainty and functional quality are now measured. Its
final general-text diagnostic is 2.505032, versus standard's 2.507587. There is no
schedule search, FIM objective, next-edit adaptation, paid teacher, or automatic
deployment promotion.

The matched parent general-text NLL is 2.486807. Standard's final value is
0.836% higher and filtered's is 0.733% higher. Both remain below the existing
5% sustained-regression trigger of 2.611148, and neither pilot recorded two
consecutive guard failures. The general guard passes; strict and line quality
are the promotion blockers.

Both arms improve matched fresh development loss relative to P12 across 629
repositories. Standard's token-weighted NLL difference is -0.005319 with a
2,000-replicate paired 95% interval [-0.006844, -0.003892]; filtered's is
-0.006666 with interval [-0.007544, -0.005795]. Filtered is lower than standard
by -0.001347 on the same repository and token identities, interval
[-0.002475, -0.000333]. Balanced-language intervals are also below zero. The
base Qwen3.5 comparison is +0.005448 against P12 on that same tokenizer and
source, with interval [+0.000416, +0.013758]. These are within-tokenizer
comparisons; Qwen2.5 and Granite use different tokenizers.

Fresh held-out NLL by language, with the same scored token counts for all three
Qwen3.5 checkpoints:

| Language | P12 | Standard | Filtered |
|---|---:|---:|---:|
| C | 1.090076 | 1.085739 | 1.084674 |
| C++ | 0.466103 | 0.464576 | 0.459192 |
| C# | 0.851537 | 0.845329 | 0.845253 |
| Go | 1.677929 | 1.667279 | 1.667931 |
| Java | 0.934518 | 0.930302 | 0.928608 |
| JavaScript | 1.149572 | 1.145994 | 1.144097 |
| Python | 1.176552 | 1.172487 | 1.171343 |
| Rust | 1.005188 | 0.998444 | 0.995719 |
| TypeScript | 0.959672 | 0.952942 | 0.953831 |
| Overall code | 1.034572 | 1.029232 | 1.027850 |
| General text | 2.486807 | 2.507587 | 2.505032 |

Filtered is lower on seven of nine code languages; standard is lower on Go and
TypeScript. Both have higher general-text NLL than P12 but pass the registered
general guard. These NLL values are diagnostics, not observed code acceptance.

The strict evaluator had all 200 cases available for each pilot. The pilots'
shared gain is `rust/11`, where both finish an assertion that P12 cuts off.
Their losses are mostly added code that ends mid-statement at the unchanged
96-token cap. Both also add an interactive `input()` call in `python/stable_0`,
which raises EOFError in the frozen test. The full manual audit retains each raw
completion and the parse, compile and test records in
`model_data_r2/failure_audit/pilot_strict_changes.json`. Both line runs lose
`go/0450b7530994492d6379`: P12 restores `testing"`, while both pilots return
`fmt"` after `package leetcode` and `import "`. The reference's suffix uses
`testing.T`; the scorer counts one exact loss and does not infer a functional
line-test result. There are no offsetting exact gains. Even though the paired
strict intervals include zero, the registered no-regression gate fails on the
observed strict and line scores. No provisional checkpoint was locked, and the
sealed test slice remains unopened.

## Execution corrections and accounting

Three standard-session attempts stopped before a full pilot began. Two consumed
zero training tokens because setup contracts failed. The third completed the
3-update smoke, 8-update continuous control and 5 resumed updates, consuming
524,288 input tokens. It reached absolute update 8 and block 128, but refused to
save the resumed checkpoint with 7.20 GiB free against an unchanged 8 GiB guard.
The retry preserved its verified smoke restart checkpoint and removed only the
redundant disposable smoke inference export before holding three optimizer
states. Its repeated numerical restart gate passed, and the full standard pilot
completed all 153 successful updates with no skipped updates or nonfinite values.
The final inference export and both ranks' complete restart states were downloaded
and hash-verified, including preservation of the original MTP sidecar. The final
model SHA-256 is `4e7a1d50ee34b7aef85abb8f054fa807e9a1fed1eb7396f81631c06f241e8810`.
Restart matched the declared numerical tolerance, not bit identity: the largest
observed compared loss difference was 0.0000177622, and both branches reached
consumed block 128 with matching scheduler/scaler state. Full comparison evidence
is retained. The filtered arm completed under the same 153-update schedule. Its
final model SHA-256 is `8f09c13c841594b1721b6f263e0f4aafa3e70c1bf27f2d3c6de39cb50881735f`.
Both ranks' restart state, the inference export, and the original MTP sidecar
passed the controller's hash and completeness checks. The frozen filtered
corpus fingerprint is
`504775fa359354b427d3ed6a17c357f5d7320770ba6f0b477d2e21282e3da7b6`.

The verified total, including completed failed-session work and all repeated
checks, is 11,173,888 input tokens, below the 12 million cap. Before attempt 4,
authenticated Kaggle quota showed 23.16 account hours remaining at
2026-09-23 01:37:23 UTC; before the filtered arm it showed 21.64 hours at
03:17:31 UTC. After the filtered arm completed and before the additional
evaluation was submitted, the authenticated reading was 20.31 account hours
remaining at 05:01:59 UTC, renewing 2026-09-26 00:00 UTC. The separate ten-hour
aggregate session limit includes setup, failures, evaluation and finalization.
The evaluation and context sessions ended with observed submission-to-terminal
upper bounds of 4,692.454 and 1,380.272 seconds. Adding all eight campaign
sessions gives 22,511.596 seconds, or 6.2533 conservative wall-hours, below the
10-hour cap. The post-job authenticated account balance was 26.19 GPU hours
used and 18.81 remaining, renewing 2026-09-26. That account balance is not a
substitute for the campaign ledger. No further GPU job was submitted after the
registered context job.

Additional corrections are recorded in `model_data_r2/environment/runtime_corrections.json`:

- The converter needed its supported `--no-mtp` option because P12's text
  configuration advertises an MTP layer absent from its weights. Original
  checkpoints and the original sidecar were not changed.
- Two old orphaned Rust fixture containers each consumed one CPU core for more
  than a day. Their fixture directories were already deleted. Only those exact
  disposable containers were stopped. Earlier contended latency measurements
  remain saved but are excluded from the clean-host comparison. A real timeout
  control verified scoped container cleanup; unrelated services remain running.
- Imported restart telemetry exposed repeated heartbeat IDs after Python RNG
  restoration. A tested OpenTelemetry SDK ID-generator extension now uses OS
  entropy without consuming or replaying the scientific RNG.
- The first filtered-artifact download stopped with Kaggle CLI exit 1 after a
  partial transfer. A read-only collection retry succeeded, and the controller
  independently re-collected and verified the checkpoint before submitting the
  additional evaluation. No training attempt was repeated.

Plan revisions record these implementation corrections, including the explicit
post-outcome line-fixture correction. Model choices, training corpus treatments,
training schedules, strict causal prompts, ceilings and selection criteria were
not changed after comparison outcomes.

## Reproduction and continuation

The campaign-owned checkpoint/corpus paths must exist in ignored artifact storage.
The full command advances through the registered GPU jobs, waits for terminal
status, verifies pilot checkpoints and refreshes quota before further submission:

```sh
uv run python scripts/run_model_data_campaign.py \
  --config configs/research/model_data_r2.yaml --execute
```

`--once` performs one verified transition; `--stage local` resumes the isolated
CPU grid only when model, prompt, runtime, host and precision fingerprints match.
No automatic retry of a failed training arm is authorized by the runner. A failure
requires diagnosis and token-ledger reconciliation first. Scientific source must
be committed and pushed; collection may update report files without changing the
pinned worker code. GPU terminal status alone is not a completion declaration:
prediction judging, paired analysis, telemetry reconciliation and local runtime
measurements must also finish.

## Telemetry and verification

The existing private gateway and local collector are used; no monitoring
infrastructure was added. The initial offline baseline bundle was imported and
all 599 deduplicated captured payloads transferred successfully, including an
empty response. A real full output was retrieved and hash-checked.

Baseline run IDs:

- P12: `run-a8e7551d-31f3-46e3-8e9c-28faeff52b3b`.
- Qwen2.5: `run-b6d37693-5bc7-4d43-a8a2-42b055636766`.
- Resumed training control: `r2-r2_standard-resumed`, trace
  `43cd20c45fb303c3470d655b8abc072e`. The gateway returned its five real progress
  records through update 8, marked historical/offline. Its missing final restart
  state is reported as a storage failure, not model-quality evidence.
- Full standard pilot: `r2-r2_standard-R2_STANDARD`. Its rank-zero import yielded
  412 distinct spans, including 153 progress records, three validation records,
  and a completed terminal record. The last progress record matches 5,013,504
  input tokens and 153 successful updates. All records are historical/offline;
  the failure query was empty. Four additional rank-local model/checkpoint spans
  were imported separately, without duplicating global training progress.
- Full filtered pilot: `r2-r2_filtered-R2_FILTERED`. Its two rank bundles yielded
  425 spans across three query pages, including 153 progress records and three
  validation records. The last progress record matches the scientific
  5,013,504 pilot input tokens and 153 successful updates. The separate smoke
  bundles imported 62 spans. The full-run failure query returned no records;
  all imported spans retain their historical timestamps.

The additional evaluation and context work imported 21 offline bundles with
6,479 spans, including one Granite failure bundle already imported and
deduplicated. Thirteen actual run IDs were queried with pagination and failure
queries. Every complete strict run has 200 distinct `eval.case` spans and a
200-of-200 completed summary; every complete line run has 180 and a 180-of-180
summary. Granite line has one scientific prediction, one completed case in its
failed summary, and two queried error spans on the second case. Qwen2.5 and D12
context runs have 111 and 102 saved score rows; their `model.score` spans number
223 and 205, two per completed row plus the failed attempt. This is telemetry
reconciliation, not an alternate scoring source. All 1,492 evaluation and 164
context captured payloads synced with no failures. One remote payload from each
bundle was fetched and matched its local bytes and SHA-256. Run IDs, page counts
and scientific counts are in `model_data_r2/telemetry_index.json` and its
referenced reconciliation file.

After the plot filter change, repository verification passed 227/227 tests in
24.76 seconds, Ruff, and mypy across 106 source files. The commands and logs
are in `model_data_r2/environment/verification-window.json`. The registered
GPU jobs are terminal. The 48-file `model_data_r2/artifact_manifest.json` passed
hash verification; the local 32k runtime grid remains in progress.

The research recommendation is to investigate Qwen2.5-Coder as the smaller
local-code candidate, using a future matched next-edit evaluation before any
replacement decision. Neither R2 pilot meets the registered promotion rule;
P12 remains the checkpoint control and the sealed test slice stays closed. The
T480s was not reachable through its existing authorized alias; no T480s or phone
result is claimed.

## External specifications

The [Qwen2.5-Coder model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
declares a 0.49B pretrained causal model and 32,768-token context. The
[Granite H-350M model card](https://huggingface.co/ibm-granite/granite-4.0-h-350m-base)
describes a Mamba2/attention hybrid. Published benchmark results motivate inclusion;
they are not substituted for this campaign's same-task results. Native FIM support
is not used. [PyTorch 2.10 SDPA documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend-dependent GQA support; the context experiment must verify the
executed operator and short-input parity rather than infer either from `sdpa`.
