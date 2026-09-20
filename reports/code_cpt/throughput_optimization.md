# Stage 1 Throughput Optimization Lab (T4x2)

Status: **lab built and locally verified; Kaggle measurements PENDING**.
No long CPT run was launched. No checkpoint was modified.
This report records the lab design, the backend discovery, the candidate
matrix, and exactly how to execute Stages A/B/C on Kaggle.

## 1. Original baseline (control)

From `reports/code_cpt/overnight.md`:

* Model: `Qwen/Qwen3.5-0.8B-Base`, revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, full-weight CPT
* 2x Tesla T4, sequence length 2048, FSDP `FULL_SHARD`
* FP32 master parameters, FP16 autocast, dynamic loss scaling (init 256)
* Microbatch 1/GPU, accumulation 8, 32,768 tokens/update
* Gradient checkpointing ON, bitsandbytes AdamW8bit, 1 DataLoader worker
* Peak allocated VRAM ~11.74 GiB/GPU, data wait ~0.77%
* Steady-state throughput **~810 tok/s aggregate**
* 11,993,088 tokens in ~4.11 h, zero NaN/Inf, zero loss-scale overflows
* Terminal Kaggle ERROR was optional FSDP resume-state serialization only;
  model and eval were already saved.

Baseline weekly projection at 45 Kaggle h/week:

```
tokens/week = steady_tok_s * 3600 * 45
baseline    = 810 * 3600 * 45 = 131,220,000 (~131M tokens/week)
```

## 2. Exact environment

* Kaggle: Python 3.12.13, PyTorch 2.10.0+cu128, Transformers 5.5.0,
  Accelerate 1.13.0, CUDA 12.8, 2x Tesla T4 (compute capability 7.5)
* Local lab verification: Python 3.11, torch 2.14.0+cpu,
  transformers 5.17.0 (CPU-only; harness logic, schema, and plots only)
* Seed 271828, corpus `train_blocks.npy` (exact 2048-token packed blocks),
  token ordering from block 0, model revision pinned as above
* Production `kaggle/code_cpt_train/run.py` installs exactly
  `transformers==5.5.0`, `accelerate`, `bitsandbytes` — notably **no**
  `flash-linear-attention`, **no** `causal-conv1d`, **no** `liger-kernel`

## 3. GDN backend discovery (Experiment 0 finding, code-verified)

Transformers 5.x (`modeling_qwen3_5.py`, confirmed in installed 5.17.0;
same mechanism in 5.5.0) resolves optimized kernels at **import time** via
`use_kernel_func_from_hub_with_fallback`:

* `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` <- package `fla`
* `causal_conv1d_fn` / `causal_conv1d_update` <- package `causal_conv1d`
* Priority: HF kernels -> original package -> pure-PyTorch reference,
  with a runtime warning when falling back to the reference
  ("correct but much slower").

Because the production kernel installs neither package, the 810 tok/s
baseline **almost certainly runs the PyTorch reference**
`torch_chunk_gated_delta_rule` plus the `F.conv1d` depthwise fallback on
all 18 Gated DeltaNet layers. The Transformers source notes the reference
path is "more than an order of magnitude" slower than the kernel on an
H100; the T4 ratio is unknown and is exactly what `exp01_fla_conv` measures.

The lab worker (`src/tinycomplete/code_cpt/bench.py`) records at startup:

* `fla` / `causal_conv1d` / `liger_kernel` / `triton` availability + versions
* the live module reprs and wrapper closures for the three kernel functions
* the actual `Qwen3_5GatedDeltaNet` module class and norm type
* `model.config._attn_implementation` (6 full-attention layers)

Definitive confirmation (fast-path vs fallback reprs) lands in
`backend_probe.json` on the first Kaggle run.

## 4. Benchmark methodology

* One Kaggle session where practical; fixed hardware, CUDA env, packages,
  dataset, model revision, seed, sequence length, token ordering.
* Each candidate runs in a **fresh** `torch.distributed.run --standalone
  --nproc_per_node=2` subprocess (FSDP/compile/kernel/CUDA state isolation).
* Stage A: 98,304 tokens = 3 x 32,768 complete updates (~2 min at 810 tok/s).
  `exp10_accum16` uses 131,072 = 2 x 65,536 updates (same reason).
  Budgets that would stop mid-update are refused by the orchestrator.
* Ranking by **steady-state tok/s** (first optimizer step excluded as
  warmup/JIT); startup, Triton/compile time recorded separately.
* Every candidate — including OOMs, import failures, NaN/overflow,
  FSDP incompatibilities — writes a result JSON with reproduction fields
  (config, backends, VRAM, loss/grad-norm stats, data-wait %, `tokens_verified`,
  `weights_updated`, `ce_parity_abs_diff` for fused CE).
* Correctness gates (never relaxed for speed): seq 2048, all params
  trainable (worker asserts), full 248,320 vocabulary, causal LM over the
  same tokens, `training_tokens == optimizer_steps * tokens_per_update`,
  FP32 masters / FP16 compute / init scale 256 / max grad norm 1.0,
  persistent overflow or NaN/Inf rejects the candidate.
* Data loader untouched (0.77% wait); revisited only if GPU work makes it
  significant.

## 5. Lab contents

* `src/tinycomplete/code_cpt/bench.py` — worker: `bench` + `probe`
  subcommands, backend probing, fused/liger CE with parity check,
  FULL_SHARD/SHARD_GRAD_OP, sdpa/eager, torch.compile, sync-cleanup mode,
  per-region timing, failure-as-data JSON.
* `kaggle/code_cpt_bench/run_bench.py` — orchestrator: optional-dep install
  (failures recorded, never fatal), backend probe, fresh-subprocess fan-out,
  `throughput_benchmarks.jsonl`/`.json`, auto-plots, console ranking.
* `kaggle/code_cpt_bench/candidates.json` — Stage-A matrix (13 candidates).
* `kaggle/code_cpt_bench/plot.py` — the five required figures + summary.
* `tests/test_bench.py` — alignment, schema, parser, orchestrator-guard,
  synthetic-data plot tests. Local: 97 passed + 6 bench tests green;
  `ruff check`, `ruff format --check`, `mypy` clean.

## 6. Stage-A candidate matrix (all PENDING Kaggle measurement)

| Candidate | Change vs baseline | Question |
|---|---|---|
| `exp00_baseline_repro` | none (torch fallback forced) | reproduce ~810 tok/s control |
| `exp01_fla_conv` | allow fla + causal-conv1d | DeltaNet fast-path effect |
| `exp02_fla_fusedce` | + Liger chunked full-vocab CE | CE throughput/memory/parity |
| `exp03_fla_fusedce_nockpt` | + checkpoint OFF at MB1 | recompute removal payoff |
| `exp04_fla_fusedce_nockpt_sgo` | + SHARD_GRAD_OP | less all-gather on T4 link |
| `exp05_attn_eager` | eager attention (control) | attention-backend sensitivity |
| `exp06_sync_cleanup` | boundary-only checks, known-tokens accounting | hot-loop sync cost |
| `exp07_fla_sync_cleanup` | sync cleanup on FLA geometry | stacking |
| `exp08_compile_winner` | torch.compile on winner | compile payoff vs overhead |
| `exp09_accum4` | accum 4 (16,384 tok/update) | optimizer amortization |
| `exp10_accum16` | accum 16, 131,072-token budget | amortization (do not select on this alone) |
| `exp11_fused_adamw_retry_sgo` | fused AdamW under SHARD_GRAD_OP only | does new FSDP dodge the broadcast bug |
| `exp12_mb2_nockpt_retry` | MB2 ckpt-OFF on best-memory geometry | does fused CE unlock MB2 |

Adaptive order: FLA -> fused CE -> no-ckpt -> SHARD_GRAD_OP -> attention ->
sync cleanup -> compile, retaining wins; one retest allowed for
memory-interacting losers. No Cartesian explosion.

## 7. Results

### Stage A — PENDING

Run on Kaggle:

```bash
python /kaggle/working/tabcomplete/kaggle/code_cpt_bench/run_bench.py
```

then paste the `plots/summary.json` ranking and the five figures here.
Per-candidate rows (tok/s, speedup, VRAM, stable) go in the table below.

### Stage B (finalists, 524,288 tokens) — PENDING

Rerun the top 3-5 stable Stage-A names with `--max-tokens 524288`
(record full-process, training-loop, and steady-state tok/s).

### Stage C (winner confirmation, 0.5-1M tokens) — PENDING

Top 1-2 configs at `--max-tokens 1048576`; verify throughput persists,
loss/grad-norms normal, no NaN/Inf, no persistent scaler overflow,
`weights_updated=true`, `tokens_verified=true`, causal LM unchanged.

### Final table — PENDING (template)

| Configuration | Tok/s | Speedup | Peak VRAM/GPU | Stable |
|---|---:|---:|---:|:---:|
| Original (~810) | ~810 | 1.00x | ~11.74 GiB | yes |
| ... | ... | ... | ... | ... |

### Profiler findings — PENDING

`torch.profiler` on baseline + likely winner only (warmup + few captured
steps): GDN/conv/MLP/LM-head/CE/attention/backward/FSDP-comm/optimizer
split; concise summary, no raw-trace dumps in Git.

## 8. Current answers (pre-Kaggle; measurement required to confirm)

* Original DeltaNet fast path: **no** — code inspection says the production
  dependency set cannot resolve `fla`/`causal-conv1d`; live confirmation via
  `backend_probe.json` on Kaggle.
* Effect of FLA / causal-conv1d / fused CE / MB1-no-ckpt / best FSDP /
  best attention / compile / sync cleanup: **unmeasured**; the matrix above
  isolates each one.
* Final tok/s, speedup vs 810, tokens/week, production config:
  **withheld until Stage C confirms a winner**. Do not launch the next
  long CPT run from this report alone.

## 9. How to finish the task

1. Attach the prepared-corpus kernel, launch the `code_cpt_bench` Kaggle
   kernel (T4x2, internet on), run Stage A.
2. Fill in Sections 7-8 from `throughput_benchmarks.json` + `plots/`.
3. Run Stage B (`--max-tokens 524288`) on the top 3-5, then Stage C
   (`--max-tokens 1048576`) on the top 1-2.
4. Promote the winner here as the verified production configuration with
   its `tokens/week = steady_tok_s * 3600 * 45` number — and stop.
   The next long CPT run is a separate decision, not part of this task.
