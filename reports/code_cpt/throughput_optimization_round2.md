# T4x2 Throughput Optimization — Round 2 (FINAL)

Status: **COMPLETE. New verified maximum: 1682.7 steady tok/s (2.08x vs 810,
1.03x vs the 1630 Round-1 control). No long CPT run launched.**

Round-1 results preserved in `reports/code_cpt/throughput_optimization.md`
and `reports/code_cpt/throughput_lab/`. This round extends the same reusable
lab (`kaggle/code_cpt_bench/`, `src/tinycomplete/code_cpt/bench.py`).
Round-2 evidence: `reports/code_cpt/throughput_lab_round2/`.

## 1. Round-1 1630 tok/s control

FLA + nockpt + torch.compile (post-FSDP, default, dynamic) + sync cleanup +
stock CE + FSDP1 FULL_SHARD + bnb AdamW8bit. Reproduced every session:
1517/1482/1516 steady (T4 session variance ±10% — all comparisons below are
within-session).

## 2. Winner reproduction

`r2_R0_winner_repro` passed in all three training sessions (1517.5,
1481.9, 1516.3). Loss/grad trajectories identical every run.

## 3. Compile placement

C1 (compile inner Qwen BEFORE FSDP, default mode): **1530.6 vs 1517.5
control (+0.9%)** — works, FSDP still wraps correctly, but no win over
compile-after-FSDP. **Post-FSDP wins by default (simpler, proven).**

## 4. Compile modes

* `reduce-overhead`, `max-autotune`: **InductorError**
  (`AssertionError: For s38 + 1, expected [s38] to have been codegen-ed`)
  in backward — cudagraphs + dynamic shapes + FSDP + FLA custom autograd
  are incompatible on torch 2.10. Dead.
* `max-autotune-no-cudagraphs`: 1511.6 — works, no win over default.
* `dynamic=False` (C5): **1591.4 (+4.9%), VRAM 12.32 vs 13.31 GiB** —
  static shapes are faster AND leaner (no dynamic cudagraph pool).
  Reproduced in Stage B: 1595.6 (+7.7%). **Promoted.**
* `fullgraph=True`: not run separately; graph-break count is 13 on the
  winner (FSDP/compile boundaries), so fullgraph would fail by construction.
* 13 graph breaks on all compiled configs (6 on the slower SGO run).

## 5. FSDP prefetch

* BACKWARD_PRE (F2): **1625.9 (+7.1%)** in A, **1616.6 (+9.1%)** in B.
  Reproduced. **Promoted.** (Accelerate 1.13 default is NO_PREFETCH —
  verified in installed source, not assumed.)
* BACKWARD_POST: 1574.5 (+3.8%) — weaker, dropped.
* forward_prefetch (F6): **1619.3 (+6.7%)** in A, **1606.0 (+8.4%)** in B.
  Reproduced. **Promoted.** The two mechanisms stack (Stage C).

## 6. Rate limiter / forward prefetch / wrap granularity

* `limit_all_gathers=false`: 1536.0 (+1.2%) — no effect. Keep `true`.
* Wrap granularity (`--fsdp-group-layers` implemented with key-remap
  bijection check): **not executed** — deprioritized with evidence:
  SHARD_GRAD_OP (fewer gathers) was 18% SLOWER, prefetch already hides
  comms, and nockpt budgets have no headroom for bigger units. Code stands
  ready for follow-up.
* Deferred grad sync (boundary-only, K=8): 1543.4 (+1.7%), loss_start
  bitwise-identical to control, grad norms match to 4e-05 — **gradient
  equivalence confirmed, but no meaningful win** (reduce-scatter is already
  well-overlapped). Not promoted.

## 7. NCCL topology

`nvidia-smi topo -m`: GPU0↔GPU1 = **PHB (PCIe, no NVLink)**. P2P access
available both directions. NCCL 2.27.5, CUDA 12.8.

## 8. NCCL microbenchmark (same 2 T4s, fp16)

| Collective | 8 MB | 62 MB (layer) | 256 MB | 508 MB (lm_head) |
|---|---:|---:|---:|---:|
| all_gather | 5.73 GB/s | 6.62 GB/s | 6.39 GB/s | 6.19 GB/s |
| reduce_scatter | 2.94 GB/s | 3.05 GB/s | 3.01 GB/s | 3.05 GB/s |

Flat across sizes = autotune healthy. Per-microstep FSDP traffic ≈ 3.7 GB
≈ 0.8 s of ~2.5 s (≈30% — consistent with the profiler). Forced
proto/algo/channel training sweeps skipped: autotune is already optimal for
our message sizes; a win there caps at ~3-6% end-to-end. No
TORCH_NCCL_HIGH_PRIORITY mechanism exists in installed torch (source
grep) — N1 skipped with evidence.

## 9. NCCL tuning result

Not run as training candidates per §8 (principled skip, microbench on
record). Default auto-selection stands.

## 10. FSDP2 findings (bounded probe, r2d_fsdp2_probe)

* Accelerate `fsdp_version=2` wraps the model; params become DTensor.
* bitsandbytes AdamW8bit **constructs** on DTensor params.
* `DTensor.full_tensor()` works ([248320, 1024] — the future fused-CE
  access pattern is viable in principle).
* **Forward OOMs**: 13.45 GiB allocated + 1.89 GiB (`logits.float()` fp32
  upcast in HF loss) where FSDP1 fits at 12.31. DTensor eager
  materialization is hungrier; FSDP2 needs its own memory program first
  (circular dependency with fused CE).
* Verdict: branch closed with evidence — FSDP1 combined config already at
  1683; migration cost unjustified this round.

## 11. Liger non-CE kernels (L5)

`apply_liger_kernel_to_qwen3_5(rms_norm=True, swiglu=True, CE off)` on the
winner: **both ranks SIGABRT (CUDA illegal memory access)** during triton
JIT (`'float' object has no attribute 'to'` + illegal address). Liger
triton kernels are incompatible with T4/sm75 in this stack — and they take
the whole process down, so this is a hard do-not-retry. (Compile likely
already fuses these pointwise ops anyway.)

## 12. FSDP2 fused CE

Not attempted — premise unavailable (§10). Mechanism for a future attempt
verified (`full_tensor()`); needs an FSDP2 memory program first.

## 13. FLA gate/beta fusion (G2/G3, custom 5.5.0-faithful shim)

* G2 loss parity 8.1e-05, G3 parity 3.5e-05 / 9.0e-05 — **math proven
  identical**, grad trajectories match control exactly.
* Perf: +2.7% / +2.2% / +0.2% across A/B/C — trending to zero at scale.
  **Not promoted** (100-line production shim for no measured gain).

## 14. Gradient sync

Boundary-only sync fits, is gradient-equivalent (bitwise loss_start,
4e-05 grad norms), but gains only +1.7-2.1%. Not promoted. OOM risk
documented as manageable (~1.6 GB full grads vs ~3.7 GB headroom).

## 15. Failed experiments

Cudagraph compile modes (InductorError), Liger (SIGABRT), FSDP2 forward
(OOM), SGO+ckpt (18% slower AND fewer graph breaks — less fusion),
expandable_segments allocator (-2.7%), wrap groups (unexecuted, §6).

## 16. Final ≥1M-token verification (kernel v13, 32 updates each)

| Configuration | Steady tok/s | vs 1630 | VRAM/GPU | Loss | Overflow | Weights |
|---|---:|---:|---:|---|---|:---:|
| Round-1 winner | 1516.3 | 0.93x | 13.31 GiB | 1.3077→1.4291 | 0 | updated, verified |
| **Combined** (static+PRE+forward) | **1682.6** | **1.03x** | 12.35 GiB | 1.3077→1.4291 | 0 | updated, verified |
| Combined+gate | 1685.5 | 1.03x | 12.35 GiB | 1.3076→1.4291 | 0 | updated, verified |

Combined vs control: **+11.0% at 1M tokens**. Gate adds +0.2% → excluded.
Data wait 2.97% (faster GPUs, loader still fine). Full-process 1563.5
(compile amortizes on long runs).

## 17. Production configuration (VERIFIED)

Round-1 winner, plus: `compile_dynamic=False`,
`BACKWARD_PRE`, `forward_prefetch=True`. Everything else identical
(FULL_SHARD, FP32/FP16, init 256, norm 1.0, MB1/acc8, bnb8bit, FLA 0.5.2,
nockpt, post-FSDP default compile, sync loop, stock CE, SDPA).

* New maximum verified steady tok/s: **1682.7 (2.08x vs 810)**
* Gains: static ~+6%, BACKWARD_PRE ~+8%, forward prefetch ~+7%
  (Stage-B reproduced; +11% combined at 1M tokens)
* Remaining bottleneck: ~25-30% matmul/backward compute (fused by
  inductor as far as cudagraph-free compilation allows), ~20% overlapped
  comms, rest FSDP wrapper overhead + 13 dynamo graph breaks that cannot be
  fused without fullgraph-incompatible surgery
* Pre-FSDP compile: works, no win. Mode winner: **default + dynamic=False**.
  BACKWARD_PRE: **yes**. Rate limiter: keep enabled. Grouping: untested.
  NCCL tuning: autotune stands. FSDP2: **no**. Fused CE under FSDP2: blocked
  on FSDP2 memory. Liger RMSNorm/SwiGLU: **no (crash)**. FLA gate fusion:
  parity-proven, perf-neutral, excluded. Reduced sync: equivalent but +2%,
  excluded.
* Tokens/week at 45 h: **1682.7 × 3600 × 45 = 272,581,400 (~273M)**

The next long CPT run is still a separate decision — NOT launched.
