# Stage 1 Throughput Optimization Lab (T4x2) — FINAL

Status: **COMPLETE. Winner verified at 1M tokens. No long CPT run launched.**

## 1. Original baseline (control)

From `reports/code_cpt/overnight.md`: Qwen3.5-0.8B-Base, full-weight CPT,
2x T4, seq 2048, FSDP FULL_SHARD, FP32 masters, FP16 autocast, init scale 256,
MB1/GPU, accum 8 (32,768 tok/update), ckpt ON, bnb AdamW8bit, ~11.74 GiB/GPU,
~810 tok/s steady, 12M tokens, zero NaN/Inf/overflows.

Weekly projection: `tokens/week = steady_tok_s * 3600 * 45`.
Baseline: 810 * 3600 * 45 = **131,220,000 (~131M)**.

## 2. Environment

Kaggle: Python 3.12.13, torch 2.10.0+cu128, transformers 5.5.0,
accelerate 1.13.0, CUDA 12.8, 2x Tesla T4 (cc 7.5). Resolved optional pkgs:
`flash-linear-attention==0.5.2` OK, `liger-kernel==0.8.3` OK,
`causal-conv1d` FAILED (no wheel; source build fails), triton 3.6.0.
Model revision `dc7cdfe2...`, seed 271828, same corpus ordering throughout.

## 3. GDN backend discovery

Transformers 5.x resolves `fla`/`causal-conv1d` kernels at import time
(HF kernels -> package -> torch reference). Production installs neither, so
the 810 tok/s baseline ran `torch_chunk_gated_delta_rule` + `F.conv1d`
fallback on all 18 Gated DeltaNet layers. With `fla` installed the worker
records the FLA `ChunkGatedDeltaRuleFunction`; conv stays torch fallback
(causal-conv1d uninstallable). Attention: sdpa (6 full-attention layers).

## 4. Methodology

Fresh `torch.distributed.run --nproc_per_node=2` subprocess per candidate.
Stage A: 98,304 tokens (3 updates). Stage B: 524,288 (16 updates).
Stage C: 1,048,576 (32 updates). Ranked by steady-state tok/s (first update
excluded). Correctness gates: seq 2048, all params trainable, full 248,320
vocab, `training_tokens == optimizer_steps * tokens_per_update`, FP32/FP16
recipe preserved, NaN/overflow rejects. Full evidence in
`reports/code_cpt/throughput_lab/`.

## 5. Stage-A results (kernel v8, 14 candidates)

| Configuration | Steady tok/s | Speedup | Peak VRAM/GPU | Stable |
|---|---:|---:|---:|:---:|
| Baseline repro (exp00) | 781.5 | 0.96x | 11.74 GiB | yes |
| +FLA fast path (exp01) | 1123.7 | 1.39x | 11.75 GiB | yes |
| +FLA + torch.compile + nockpt + sync (exp08) | **1560.2** | **1.93x** | 12.31 GiB | yes |
| eager attention (exp05) | 818.8 | 1.01x | 11.75 GiB | yes, no effect |
| sync cleanup (exp06) | 784.3 | 0.97x | 11.74 GiB | yes, no effect |
| stock MB1 nockpt, FULL_SHARD or SGO | OOM | — | ~13.0 GiB | no |
| MB2 nockpt | OOM | — | ~13.1 GiB | no |
| liger/chunked CE (FSDP1) | incompatible | — | — | no (see 7) |
| fused torch AdamW under SGO | OOM (memory, inconclusive on bug) | — | ~13.0 GiB | no |

FLA alone measured 1.35–1.46x across four independent runs
(1092/1176/1179/1124). Eager-vs-SDPA and sync cleanup are within run noise
(T4 short-run variance ~±10%).

## 6. Stage-B finalists (524,288 tokens, kernel v9)

| Configuration | Steady tok/s | Full-process tok/s | Speedup | VRAM | Stable |
|---|---:|---:|---:|---:|:---:|
| Baseline (stageB_baseline) | 804.6 | 803.2 | 0.99x | 11.74 GiB | yes |
| FLA (stageB_fla) | 1096.2 | 696.6 | 1.35x | 11.75 GiB | yes |
| Winner (stageB_winner) | **1535.2** | 796.5 | **1.90x** | 12.31 GiB | yes |

All: 16/16 updates, tokens verified, weights updated, zero overflows,
grad-norm means ~2.55, loss trajectories identical across configs
(1.3076 -> 0.929). Full-process trails steady-state (model load + FLA
autotune + compile time amortize away on long runs).

## 7. Failed candidates (useful data)

* `causal-conv1d`: no prebuilt wheel for the Kaggle container; source build
  fails. Conv stays on the torch fallback; effect unmeasurable.
* Liger/chunked CE under FSDP1: calling the decoder submodule directly
  bypasses the outer FSDP unit's all-gather hook (embed stays sharded);
  `summon_full_params` + backward-inside leaves full-shaped `.grad`s that
  trip `RuntimeError: Cannot writeback when the gradient shape changes`
  (`[127140352]` vs `[248320, 1024]`) on the next FSDP forward. Chunked CE
  additionally conflicts with checkpoint recompute (`retain_graph` +
  recompute shape mismatch). Non-stock CE needs FSDP2/DTensor — untried.
  Worker now fails these fast with a clear message.
* Stock MB1/MB2 without checkpointing OOMs (~13 GB). Checkpoint removal is
  impossible WITHOUT the CE memory savings above — EXCEPT under
  torch.compile, which fits nockpt at 12.31 GiB (key interaction).
* Kineto profiler on the ckpt-ON baseline: SIGKILL (profiler overhead on a
  11.74 GB base). FLA profile captured fine.

## 8. Profiler findings (FLA geometry, 2 steps)

Self-CUDA: FLA `ChunkGatedDeltaRuleFunction` ~29% (fwd triton kernel 20.6%
alone), matmul backward (MLPs/projections) ~25%, FSDP `record_param_comms`
~20% + NCCL all-gather ~13% + FSDP forward overhead — **~35-40% of GPU time
is distributed communication**. That is why compile (kernel fusion, less
traffic) beats raw kernel swaps. Full table:
`throughput_lab/profile_fla_cuda_ops.txt`.

## 9. Stage-C winner confirmation (1,048,576 tokens, kernel v10)

| Configuration | Steady tok/s | Speedup | VRAM | Loss | Overflow | Weights |
|---|---:|---:|---:|---|---:|:---:|
| FLA control (stageC_fla) | 1139.4 | 1.41x | 11.75 GiB | 1.3077->1.4296 | 0 | updated, verified |
| Winner (stageC_winner) | **1632.7** | **2.02x** | 12.31 GiB | 1.3077->1.4291 | 0 | updated, verified |

32/32 updates each. Winner loss trajectory matches the FLA control to 4
significant figures (compile numerics clean). Grad norms normal
(2.60 mean / 3.12 max). Data wait 1.18%.

## 10. Optimization contribution

baseline 810 -> +FLA ~1120 (+38%) -> +compile + nockpt + sync ~1560-1630
(+38-45% more). Compile is load-bearing for nockpt (non-compiled nockpt
OOMs). Eager attention, sync cleanup, SGO, accum changes: no measurable
effect or OOM-blocked.

## 11. Recommended production configuration (VERIFIED)

* FSDP FULL_SHARD, FP32 masters, FP16 autocast, init scale 256, max norm 1.0
* MB1/GPU, accum 8, 32,768 tok/update, seq 2048, 1 worker
* bitsandbytes AdamW8bit (fused-torch path still unverified)
* `flash-linear-attention==0.5.2` installed (fast Gated DeltaNet path)
* gradient checkpointing OFF, torch.compile ON, sync-cleanup loop
* stock full-vocab causal CE, SDPA attention
* Expected steady throughput: **~1550-1630 tok/s (~1.9-2.0x vs 810)**
* Expected tokens/week at 45 h: **~250-264M** (1632.7 * 3600 * 45 =
  264,497,400), roughly double the 131M baseline
* Compile/load overhead (~50% of a 524k-token run's wall) amortizes away on
  multi-hour CPT runs; keep it OUT of steady-state accounting but budget it
  in wall-clock planning

Caveats: T4 short-run variance is ~±10%; the 1.9-2.0x band (not a point
estimate) is the honest claim. Fused-AdamW and FSDP2-CE remain open future
experiments. The next long CPT run is a separate decision — NOT launched.
