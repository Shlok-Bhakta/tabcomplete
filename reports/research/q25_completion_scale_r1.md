# Q25 completion training: repetition versus new states

Status: CPU preparation and implementation in progress. No GPU session has been
allocated for this experiment yet. The desktop's served model is unchanged.

## Why run this experiment

The completed direct FIM pilot trained the approved Qwen2.5-Coder-0.5B Base for
one pass over 4,096 public, synthetic completion states. It consumed 1,776,908
nonpadding input tokens and supervised 34,405 response/EOS tokens. Development
exact completion with observed EOS rose from 47/240 to 130/240. Response NLL
fell from 1.030 to 0.534 with the same tokenizer. The first and last sixteen
training updates averaged losses of 0.662 and 0.535.

A post-hoc lexical breakdown of that historical development slice found 200
content-bearing targets, with exact/EOS completion improving from36 to94.
The other40 targets contained only punctuation and whitespace, improving
from11 to36. This is a coarse text classification, not a functional check.
It shows that the aggregate gain was not entirely closing punctuation. The new
comparison reports these strata under a plan frozen before its outputs.

Those measurements do not establish a compute bottleneck. There is no
intermediate development curve, and a source-completion target does not reveal
what a human wanted to edit. The audited collector export contains no defensible
preference pairs yet. More public completion training remains possible without
waiting for those observations.

## Frozen comparison

Two candidates start independently from the exact untouched pretrained Q25
checkpoint. Each receives 8,192 example exposures and 512 optimizer updates.

| Variant | Distinct training states | Passes | Exposures |
| --- | ---: | ---: | ---: |
| repeat | 4,096 | 2 | 8,192 |
| scaled | 8,192 | 1 | 8,192 |

The first 4,096 states and their batch order are identical. The second stage
replays those states or uses 4,096 additional states. New states use the same
pinned public source pool and the same two variants per source document. Their
language, input-length and target-length distributions are matched as far as
supply permits. Actual token counts and distribution shortages are reported;
equal example exposures are not a claim of equal FLOPs.

A new, common development slice reserves whole repository-alias groups that the
previous 4,096-state FIM pilot never trained on. The prior 240-state development
slice is preserved as a secondary historical comparison. Original reserved
corpora, benchmark exclusions, licenses and sealed-test restrictions remain in
force. No model output informs this new split.

The full machine-readable CPU plan is
[q25_completion_scale_r1/preparation_plan-r2.json](q25_completion_scale_r1/preparation_plan-r2.json).
Revision 1 is preserved. Revision 2 strengthened the batch and token-distribution
controls before preparing examples or generating candidate outputs.

## Limits and current quota

Authenticated Kaggle observation at 2026-10-04 14:11:20 UTC found 41.08 of 45
account GPU-hours remaining, 3.92 used, no active jobs, and renewal at
2026-10-10 00:00:00. Every listed job status was verified. This is a fresh
observation, not a historical balance reused as current quota.

The new experiment allows six aggregate GPU-session wall-hours, one allocation
at a time, with three-hour session deadlines and a conservative charge of two
account GPU-hours per wall-hour. The aggregate additional nonpadding input-token
cap is 10 million, including discarded/replayed work. Each session reserves at
least thirty minutes for checkpoint saving and evaluation. The new artifact root
is capped at 12 GiB and must leave at least 2 GiB free. Existing research
checkpoints are preserved and counted separately. The old campaign's frozen
limits are unchanged.

No paid compute, new model weights, teacher calls, automatic quota-renewal use,
preference training, or automatic personalization is authorized by this plan.
The explicit user instruction authorizes this separate free Kaggle experiment.

## Deployment

The existing embedded Q4_K_M FIM predictor remains the stable editor provider.
Training candidates have separate paths, plans and process identities. Any
candidate improvement must survive fixed quality and regression checks before
an export is considered for deployment. A good result here establishes source
completion performance on these fixtures, not production next-edit accuracy.
