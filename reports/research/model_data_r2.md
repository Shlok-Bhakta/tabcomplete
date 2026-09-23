# Model and data research R2

Status: **in progress**. This is an evidence ledger, not a completed campaign or
promotion decision. The standard-policy pilot is complete and verified; the
filtered pilot is executing, with remaining GPU evaluations still ahead. The active
playground checkpoint has not changed. The sealed research test remains closed.

## Evidence available so far

The required initial comparison completed all 200 corrected causal fixtures on
T4 using Transformers 5.5 and Torch 2.10, FP16 text-only models, raw causal
prompts, greedy decoding, and the unchanged 96-token ceiling. Saved predictions
were executed on Crabcake in the existing network-disabled containers.

| Model / runtime | Causal passes | Line exact match | Returned-line latency | Peak RAM | Verified context | Status |
|---|---:|---:|---|---|---|---|
| P12, Transformers FP16 / T4 | 9/200 | Pending | Not measured by non-streaming baseline | Not measured as host RSS | Short causal fixtures | Executed |
| Qwen2.5-Coder, Transformers FP16 / T4 | 11/200 | Pending | Not measured by non-streaming baseline | Not measured as host RSS | Short causal fixtures | Executed |
| D12, Transformers FP16 / T4 | 5/200 | Pending | Unknown | Unknown | Historical short controls only | Verified historical strict-suite reuse |
| Qwen2.5-Coder, native Q4_K_M / Kiwi CPU | 7/200 | 18/180 | 2.641 s median across 180 line cases, Kiwi two-thread CPU | 1.405 GiB in separate Crabcake 2k grid | Line/strict fixtures; not a long-context intelligence claim | Strict and corrected line executed |
| P12, native Q4_K_M / Kiwi CPU | 200 predictions saved; judging pending | 19/180 | 3.694 s median across 180 line cases, Kiwi two-thread CPU | 3.033 GiB in separate Crabcake 2k grid | Crabcake 2k and 8k timing grids executed | Quality partly judged |
| Base Qwen3.5 | Pending | Pending | Unknown | Unknown | Pending bounded context run | Not yet executed in R2 |
| Granite H-350M, native Q4_K_M / Crabcake CPU | Pending | Pending | Saved in 2k grid; quality pending | 1.981 GiB peak server RSS in 2k grid | 1,171–2,005 actual input tokens | Native runtime executed; capability evaluation pending |

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

Qwen2.5 Q4 versus its own FP16 checkpoint had two gains and six losses. Runtime
and precision both change in that comparison; it is not a pure quantization
ablation. Five lost cases involve extra generated code that breaks compilation;
the remaining loss is a negative-input GCD logic error. See
`model_data_r2/failure_audit/q25_quantized_changes.json` for actual assertions and
compiler diagnostics. No official output was repaired or gold-truncated.

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

## Frozen data experiment

Both arms use the same new frozen upstream candidate pool and identical packed
language allocations. Each contains exactly 5,013,504 input tokens. Standard
selects 4,642 files; filtered selects 6,306. All source identities, known-exclusion
checks, packed fingerprints, rejection counts, parser coverage, repository
concentrations, and deterministic samples are under `model_data_r2/data_audit/`.

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
| R2_STANDARD | Verified P12 | Existing Stage-1 policy on fresh pool | 5,013,504 | −0.005340 (1.034572 → 1.029232), aggregate fresh dev | Pending | Pending | Research candidate; quality gates pending |
| R2_FILTERED | Verified P12 | Frozen extra filters and repository balancing | Running | Pending | Pending | Pending | No decision |

Both schedules remain 153 successful updates, 32,768 input tokens/update,
five-update warmup to 3e-6 and cosine decay to 3e-7 at update 153. There is no
schedule search, FIM objective, next-edit adaptation, paid teacher, or automatic
deployment promotion.

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
is retained. The filtered arm started at 2026-09-23 03:17:32 UTC.

The planned total, including completed failed-session work and all repeated
checks, is 11,173,888 input tokens, below the 12 million cap. Reservations remain
in the ledger until reconciled with scientific summaries. Before attempt 4,
authenticated Kaggle quota showed 23.16 account hours remaining at
2026-09-23 01:37:23 UTC; before the filtered arm it showed 21.64 hours at
03:17:31 UTC, renewing 2026-09-26 00:00 UTC. The separate ten-hour
aggregate session limit includes setup, failures, evaluation and finalization.

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

Repository tests passed 225/225 in 24.04 seconds at this checkpoint. Full lint
passed, and mypy reported no issues across 106 source files. Subsequent analysis
changes also passed all 16 R2-specific tests. Final checks and the final campaign
commit will be repeated after collection and analysis finish.

The next training recommendation and promotion decision remain open until the
registered pilots, paired evaluations and failure audits finish. The T480s was
not reachable through its existing authorized alias; no T480s or phone result is
claimed.

## External specifications

The [Qwen2.5-Coder model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
declares a 0.49B pretrained causal model and 32,768-token context. The
[Granite H-350M model card](https://huggingface.co/ibm-granite/granite-4.0-h-350m-base)
describes a Mamba2/attention hybrid. Published benchmark results motivate inclusion;
they are not substituted for this campaign's same-task results. Native FIM support
is not used. [PyTorch 2.10 SDPA documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend-dependent GQA support; the context experiment must verify the
executed operator and short-input parity rather than infer either from `sdpa`.
