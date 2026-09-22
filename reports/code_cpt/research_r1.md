# Causal CPT research campaign r1

## Decision

The matched pilots show that another 5,013,504 causal code tokens can improve fresh-code NLL,
but the learning-rate schedule matters more than the parent checkpoint in this small experiment.
Cosine decay improved both parents on fresh development data. The constant control improved P12
slightly in the token-weighted aggregate and made P5 worse. Repository-paired uncertainty leaves
D12 and D5 tied, and the corrected functional benchmark regressed for both. No arm is promoted or
extended. The new long-context diagnostic records the remaining dependency-use evidence below.

This campaign did not train FIM, next-edit, preference, reward, or quantized models. It kept
Qwen/Qwen3.5-0.8B-Base and the Stage-1 tokenizer/data contract fixed.

The five research questions resolve as follows:

- Further causal code training helps fresh development NLL under cosine decay, but all four new
  arms lost one or more strict functional passes. The quality result is mixed.
- P12 has the best post-training NLL point estimate through D12, but D12 versus D5 is unresolved
  under paired repository resampling. There is no supported parent winner.
- Early cosine decay beats the constant control for both parents on fresh development and
  historical MICRO NLL.
- The optimized trainer exports reloadable models and passes a same-world-size numerical restart
  test. The four pilots did not publish optimizer state, so they cannot be extended exactly.
- The new diagnostic separates target preference from generation failure at 2k. Genuine 32k
  dependency scoring remains blocked, so a training-induced long-context change is unresolved.

## Quota and jobs

The authenticated Kaggle quota read at 2026-09-21 18:46:21 UTC was 12.619809 hours used and
32.380191 hours remaining from a 45-hour allocation. Kaggle stated a reset time of
2026-09-26 00:00:00 UTC. The account meter showed that one T4x2 session wall-hour consumes about
one account quota-hour, although the session provides two physical accelerator-hours. The
campaign retained separate limits of 12 T4x2 wall-hours, 24 physical accelerator-hours, and
50 million new training input tokens.

The final authenticated reading was 20.73 hours used and 24.27 hours remaining. The campaign
therefore consumed about 8.11 account quota-hours and T4x2 wall-hours, or about 16.22 physical
accelerator-hours. The final reading is rounded to two decimal places by the CLI. No new quota
window was consumed.

| Kaggle job | Purpose | Final status |
|---|---|---|
| `tabcomplete-code-cpt-campaign-r1` | Four matched CPT pilots | Complete |
| `tabcomplete-code-cpt-resume-r1` | Eight-update restart comparison | Complete |
| `tabcomplete-code-cpt-context-probe-r1` | 2k to 32k training feasibility | Complete |
| `tabcomplete-code-cpt-evaluation-r1` | Repository bootstrap and corrected causal predictions | Complete |
| `tabcomplete-code-cpt-long-context-r1` v1 | Matched dependency diagnostic | Error after 99 seconds; ambiguous input path |
| `tabcomplete-code-cpt-long-context-r1` v2 | Matched dependency diagnostic | Partial; complete 2k matrix, 32k SDPA OOM |
| `tabcomplete-code-cpt-long-context-r1` v3 | Chunked-cache scoring check | Rejected by numerical verification |

The quota source and each observed reading are in `quota_plan.json` and
`reports/code_cpt/research_r1/quota_observations.json`. Kaggle did not expose a session runtime
limit through the authenticated account quota response, so that field remains unknown. No other
active kernel was consuming the account at the initial observation.

## Runtime and lineage

The production trainer now uses FSDP1 FULL_SHARD, FP32 master parameters, FP16 compute with
dynamic loss scaling initialized at 256, AdamW8bit, gradient clipping at 1.0, 2,048-token
sequences, one sequence per GPU, accumulation 8, SDPA, post-FSDP `torch.compile` in default mode
with `dynamic=False`, `BACKWARD_PRE`, and forward prefetch. Each optimizer update consumes 32,768
input tokens. A 98,304-token production smoke completed with finite updates, correct counters,
and a reloadable export.

`--init-from` now starts from weights with a fresh optimizer and scheduler. `--resume-from`
restores a training checkpoint. Metadata distinguishes inherited lineage tokens, added input and
scored tokens, attempted and successful updates, skips, and the exact next corpus block.

The campaign verified parent inventory, configuration, tokenizer files, tied embeddings, and
selected tensor values before training. The packed-token training data do not depend on tokenizer
serialization. Generation evaluations use the same canonical P12 tokenizer for every model
because the P5 and P12 directories contain equivalent token IDs but different tokenizer config
serializations.

| Parent | Inherited tokens | Model SHA-256 |
|---|---:|---|
| P5 | 5,013,504 | `d499c3fe2d720246c710c5c9e20653a4599aa25d77ebef871cb28431ccc888e8` |
| P12 | 11,993,088 | `d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43` |

## Frozen fresh corpus

All four arms consumed the same blocks in the same order. The corpus fingerprint is
`92f3a73fa43e937379ba04dc974489328dfb07020d2415b19ac0dd5815b7664a`. The manifest records
dataset and tokenizer revisions, repository and file identities, content hashes, language,
split, packing order, block counts, and token counts. It excludes known Stage-1 content,
evaluation repositories, benchmark fixtures, and reserved edit data using repository identity
and content hashes. Training contains 2,448 blocks and 5,013,504 tokens. The development and test
splits each contain 1,032,192 input tokens and 1,031,688 scored targets. Stage 1 did not preserve
its training repository and file manifests, so differently packed partial-document overlap cannot
be ruled out even though 5,859 exact old packed blocks and 2,405 known content hashes were
excluded. This separation says nothing about Qwen's original pretraining corpus.

## Matched pilots

Every arm completed 153 updates and 5,013,504 input tokens, with 5,011,056 scored target tokens.
There were no skipped updates, nonfinite values, or scaler overflows. The total was 20,054,016
new training input tokens and 612 successful optimizer updates. The four-arm job took
16,983.477 seconds, or 4.718 wall-hours.

The four arms account for the exact quality-comparison token total. Smoke, restart, and
feasibility probes kept total campaign training exposure below 22 million tokens even when every
configured OOM input is charged conservatively. This is well below the 50-million-token cap.

Updates 1 through 5 used linear warmup to 3e-6. Constant arms then held 3e-6. Decay arms used
cosine interpolation from 3e-6 at update 5 to 3e-7 at update 153 and would hold the floor after
that point.

| Arm | Parent | Schedule | Added tokens | Fresh dev NLL | Change from parent | General NLL | TPS | Model SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---|
| C12 | P12 | Constant | 5,013,504 | 1.033306 | -0.001266 | 2.503057 | 1680.73 | `ce3a666ad9eab501c6400a5b248d3532af758ad3e4bc0452e2098a19d58e2fb9` |
| C5 | P5 | Constant | 5,013,504 | 1.035386 | +0.003400 | 2.499958 | 1646.18 | `54cb68207afb20fa33897cef813c726e272ee3e778294ad164728a698d62315f` |
| D12 | P12 | Cosine | 5,013,504 | 1.029166 | -0.005406 | 2.510476 | 1689.17 | `ce0705a6ca265ee40eb7c65b4831b938d6af37c1b508d68cdace2b02ec6bdd8e` |
| D5 | P5 | Cosine | 5,013,504 | 1.030714 | -0.001273 | 2.502563 | 1690.82 | `47772f912bb7993ad31a2bdd8f1bc4f8afd961101cf26dd30b1bc0a4344e72fc` |

P5 started at fresh development NLL 1.031986 and P12 at 1.034572. General-text NLL rose by
less than 1 percent in all arms, below the 5 percent pause guard. D12 improved every evaluated
language. D5 improved eight of nine, with Go regressing by 0.013662 NLL. C12's small aggregate
gain depended heavily on Go while six languages regressed. C5 regressed overall.

The historical MICRO result still has its original interpretation. P5 has the lowest observed
Stage-1 MICRO NLL. P12's 9/200 corrected causal passes versus P5's 7/200 is the highest observed
functional count, but the five paired wins and three losses do not establish that P12 is better.
Nothing here proves that P12 was overtrained.

## Restart test

The first optimizer-state approach failed while gathering bitsandbytes state. The bounded second
attempt used distributed checkpoint model state plus rank-local AdamW8bit state for the same
world size and wrap layout. Continuous eight-update training and four updates plus process exit,
reload, and four more updates consumed the same 128 blocks once and ended at 262,144 input tokens.
Optimizer structure, scheduler, scaler, counters, and next block agreed.

The result passed the declared numerical tolerances but was not bitwise identical. Maximum loss
difference was 2.897e-5. Maximum model parameter difference was 1.055e-4 with mean difference at
most 5.53e-8. Maximum optimizer-state difference was 4.351e-5. This proves numerically faithful
same-world-size restart for the tested layout, not elastic resharding.

The four production pilots saved verified inference exports but did not publish optimizer states.
Extending one of those exact pilot trajectories across a new process would therefore require a
reset or replay. The campaign does neither, so no conditional 25-million-token extension was
launched.

## Longer-context training feasibility

The probes used genuinely long, coherent synthetic repository prompts. The known 2,048-token
runtime completed one update with a peak of 11.635 GiB on each T4. Every 4,096, 8,192, 16,384,
and 32,768-token attempt ran out of memory. Each length also failed its single allowed retry with
activation checkpointing and reduced prefetch. The 32k probe used the specified 65,536-token
update and is only a feasibility result. On this T4x2 stack, 2k is the only proven training length.

## Evaluation repairs

Next-edit protocol v2 distinguishes `no_edit`, deletion through an empty replacement, and a
nonempty replacement. It applies exact UTF-8 byte offsets and keeps whitespace. Tests cover no
edit, deletion, insertion, replacement, whitespace-only text, Unicode offsets, malformed or
truncated output, and the 96-token ceiling. The frozen controls give 200/200 functional gold
actions, 40/40 no-edit recall for unchanged behavior-preserving cases, and 0/160 passes for
unchanged edit-required cases. The six defective v1 cases that passed unchanged no longer pass.
No next-edit model was trained, and next-edit results do not rank these causal checkpoints.

The corrected causal suite remains 200 cases with SHA-256
`ed28739b302e4c0d3f9f45e859e7ccd68a1e8594a62eb7b13ee8425b92a439a4`.

## Development uncertainty and untouched test

The paired bootstrap resampled the same 629 repository identities 2,000 times. Differences below
are candidate minus comparison model, so negative values are better. D12 and D5 each improved
their own parent with the balanced-language 95 percent interval below zero. Their direct interval
crossed zero, so the declared result is a tie.

| Comparison | Balanced-language difference | 95 percent interval | Token-weighted difference | 95 percent interval |
|---|---:|---:|---:|---:|
| C12 minus P12 | -0.001282 | [-0.001764, +0.002138] | -0.001284 | [-0.005258, +0.002141] |
| C5 minus P5 | +0.003340 | [+0.001814, +0.004892] | +0.003340 | [+0.001091, +0.005664] |
| D12 minus P12 | -0.005395 | [-0.006094, -0.003581] | -0.005395 | [-0.007682, -0.003757] |
| D5 minus P5 | -0.001305 | [-0.003412, -0.000665] | -0.001304 | [-0.003593, +0.002017] |
| D12 minus D5 | -0.001531 | [-0.002656, +0.000717] | -0.001532 | [-0.005525, +0.001241] |

Because selection was not unique, the untouched test split remained sealed. There is no final-test
number to report, by design.

## Corrected causal function

All models used the same canonical P12 tokenizer, greedy decoding, and 96-token limit. The slight
P5 parse-count change from the historical table comes from removing its tokenizer-serialization
confound; its functional count remains 7/200.

| Model | Parse | Compile | Functional |
|---|---:|---:|---:|
| P5 | 30/200 | 17/200 | 7/200 |
| P12 | 26/200 | 19/200 | 9/200 |
| C12 | 31/200 | 23/200 | 8/200 |
| C5 | 30/200 | 17/200 | 6/200 |
| D12 | 31/200 | 20/200 | 5/200 |
| D5 | 23/200 | 18/200 | 4/200 |

C12 had four wins, five losses, and four shared passes against P12. C5 had four wins, five losses,
and two shared passes against P5. D12 had zero wins, four losses, and five shared passes against
P12. D5 had one win, four losses, and three shared passes against P5. The counts are too small for
a broad model-quality conclusion, but neither cosine arm passes the campaign's no-regression gate.

## Long-context dependency diagnostic

The versioned suite has constants, enums, signatures, fields, and nested configuration under
short-control, long-near, long-far, absent, and counterfactual conditions. Version 2 completed all
25 conditions at approximately 2k for Base, P5, P12, D12, and D5. It also completed the short
control for the 32k group, whose prompt is only about 73 tokens. It did not score a genuinely long
32k prompt.

At 2k every model had the same correct-versus-distractor pattern: 4/5 short controls, 4/5
long-near, 4/5 long-far, 2/5 absent, and 5/5 counterfactual. Distant preference conditioned on
nearby success was 4/4 for every model, with one unresolved competence group each. This small
matrix detects basic dependency preference, but it shows no checkpoint separation at 2k.

Strict generation remained poor. No model produced an exact target continuation. Isolated
execution passed 1/25 for Base, 1/25 for P5, 2/25 for P12, 1/25 for D12, and 0/25 for D5. Most
generations reached the 96-token cap. These results separate target preference from stopping and
generation failure instead of treating both as one context score.

The first genuinely long 32k condition made SDPA request 30.61 GiB on a 14.56 GiB T4 for every
model. A bounded chunked-cache attempt was accepted only if it matched full selected-logit scoring
on a short case. It failed that check in the pinned Kaggle runtime by 16.87 to 18.71 NLL and was
discarded. A local CPU checkpoint check had differed by only 2.86e-6, which confirms that the gate
caught a runtime-specific incompatibility rather than silently accepting it. The campaign stopped
there. It cannot claim that long-context dependency use was preserved or changed.

## Artifacts and verification

Campaign metrics live in `reports/code_cpt/research_r1/campaign_metrics/`. Full inference exports
were downloaded to the ignored local artifact tree at
`artifacts/code_cpt/research_r1/campaign_full/`. Each downloaded model hash matched the remote
completion record, and every export loaded for a deterministic CPU smoke generation. The original
MTP sidecar SHA-256 is
`94a8afbe3c204a6e82669193a1cd5107277a30529deb2b54d4d63720f8f2097c`.

The repository has machine-readable resume, context-probe, quota, corpus, selection, benchmark,
and long-context records beside the plots in `reports/code_cpt/research_r1/`. The compact hash
index is `reports/code_cpt/research_r1/artifact_manifest.json`.

## Verification

The final local checks passed 175 tests with one expected skip. Ruff passed the repository, and
Mypy reported no issues across 73 source and script files. The campaign entry point completed a
dry run and detected the existing pilot as reusable, so it will not resubmit the four training
arms on a later `--execute` invocation.

## Recommendation

Keep the cosine schedule for the next bounded causal-CPT recipe test, but do not promote or extend
either r1 arm. D12 is the best aggregate development result and D5 also improves its parent, yet
their paired comparison is unresolved and both lose strict functional passes. The next causal run
should revise the data or stopping behavior before spending a larger token budget. A model-base
comparison remains worthwhile because this campaign only tested Qwen3.5-0.8B. Prepare next-edit
supervision later under protocol v2; causal CPT did not learn that separate action contract.
