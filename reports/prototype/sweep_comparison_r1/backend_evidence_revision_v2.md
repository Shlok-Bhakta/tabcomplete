# CUDA inventory evidence after the fourth startup failure

The fourth worker stopped at `comparison_quality` after 1,896.346900338 seconds.
It produced zero scientific predictions. Its authenticated terminal ERROR was
observed at 2026-09-30 12:11:59.357066 UTC. The startup log independently lists
CUDA0 and CUDA1 as Tesla T4 devices, records use of CUDA0, and confirms 29/29
layers offloaded. Moving the verbosity flag first did not restore the separate
one-time `found N CUDA devices` message. The prior fix was insufficient.

The actual failing invocation and logs are preserved in `setup_failure_004`.
The observability CLI `runs --since 1h --json` returned `no_runs`; startup stopped
before the scientific run scope. No GPU request or trace ID is invented.

The new parser accepts either the explicit CUDA initializer count or the pinned
runtime's `common_param: - CUDA<N> : ...` device inventory. The latter is emitted
by actual backend-device enumeration in `common_params_print_info`. It does not
infer availability from requested arguments, generic CUDA build flags, host
buffers, or nvidia-smi alone. Both device availability and positive model-layer
offload remain mandatory. Contradictory initializer and inventory counts reject
the startup. The evidence records which source supplied the count.

Four regression cases cover the observed omitted initializer, duplicate device
lines without offload, CPU/host/argument false positives, and a contradictory
explicit zero-device log. Three failed before the fix. All 22 targeted runner
tests passed afterward, with Ruff passing. Applying the new parser to the exact
retained fourth log verifies two enumerated CUDA devices and 29/29 offloaded
layers. That re-interpretation is a placement diagnostic, not a newly completed
GPU benchmark.

The GPU comparison needs a new frozen plan before another invocation. Models,
weights, tokenizer, runtime revision, prompts, fixtures, and scoring rules stay
the same. Historical plans and the fourth invocation's original classification
remain unchanged. The worker and submission helper must both enforce any shorter
session deadline; changing only a report or provider timeout is insufficient.

The fourth conservative bound is 2,315.877205 seconds from its pre-submit quota
observation through authenticated terminal observation. Total prior bounds are
9,381.212425 seconds. A final 4,800-second session would bound the aggregate at
14,181.212425 seconds, leaving 218.787575 seconds under the same 14,400-second
cap. Its 1,200-second finalization reserve leaves a 3,600-second, 60-minute
work deadline that includes setup, compilation, inference and evaluation.
Fresh authenticated quota and absence of other active jobs are required at
submission. No automatic quota renewal or paid resources are permitted.

New runner SHA-256:
`0dff7c493df650f4d31182662c752cfa3ec82b963835f37dafec1f1a80c3c5b5`.
New test SHA-256:
`000aa553517f028a9ba078c6aa6c39fc976f537735c5a564a7b52ae9122485c2`.
