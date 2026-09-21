# TabComplete-Code Stage 1 Overnight Report

Result: **YES**. The best full-weight checkpoint reduced repository-held-out code
NLL from 1.112976 to 1.101055, a 1.071% reduction. All nine core languages improved.
The 5.014M-token snapshot beat both earlier snapshots and the 11.993M-token final
checkpoint, so it is the current `TabComplete-Code-0.8B` candidate.

## Repository / Environment

- Branch: `stage1/code-cpt`
- Training Git SHA: `5c50a13e8912cffd216a8212d290930cae14197a`
- Starting point: descendant of `ef12f31c6b04b58279a93be56ab0d53ac9958c44`
- Local development: Python 3.11, CPU-only
- Kaggle: Python 3.12.13, PyTorch 2.10.0+cu128, Transformers 5.5.0,
  Accelerate 1.13.0, CUDA 12.8
- GPUs: 2 Tesla T4
- Base model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Seed: 271828

## Hugging Face Authentication

PASS. `HfApi().whoami()` returned `WashableBowl700` locally and in Kaggle. The
read-only token was copied into a private Kaggle dataset without printing it or
placing it in Git. The training kernel used the prepared corpus and public base
model, so it did not attach the secret dataset.

## The Stack Dedup Access

PASS. The access probe streamed a nonempty record from every requested language.
The run used `bigcode/the-stack-dedup`, not the raw Stack or Stack v2.

## Dataset Revision

`17cad72c886a2858e08d4c349a00d6466f54df63`

Verified directories: `data/python`, `data/typescript`, `data/javascript`,
`data/java`, `data/cpp`, `data/rust`, `data/go`, `data/c`, `data/c-sharp`, and
`data/shell`.

## Validation Split

The split hashes the canonical repository name with SHA-256, reads the first eight
digest bytes as a big-endian integer, and takes modulo 1,000. Buckets 0 through 9
are validation-only. Paths do not affect assignment. The frozen manifest contains
171 validation repositories and 179 records. Each MICRO corpus has 16,384 input
tokens and 16,376 scored causal targets. Training checks every streamed record
against this repository-level split.

## Base Qwen Metrics

Untouched `Qwen/Qwen3.5-0.8B-Base` used FP16 autocast over FP32 master weights.

| Corpus | NLL | Evaluated tokens |
|---|---:|---:|
| Python | 1.263446 | 16,376 |
| TypeScript | 1.190239 | 16,376 |
| JavaScript | 1.231788 | 16,376 |
| Java | 0.920384 | 16,376 |
| C++ | 1.324023 | 16,376 |
| Rust | 1.011945 | 16,376 |
| Go | 0.852880 | 16,376 |
| C | 1.341754 | 16,376 |
| C# | 0.880329 | 16,376 |
| **Overall code** | **1.112976** | **147,384** |
| General text | 2.417105 | 16,376 |

The immutable baseline is in `reports/code_cpt/baseline.json`.

## Data Feeder Benchmarks

The direct streaming path used a 10,000-record shuffle buffer, cheap metadata-first
filters, token quotas, EOS file boundaries, and exact 2,048-token packing. It wrote
5,859 memory-mapped blocks containing 11,999,232 tokens. Tokenization sustained
238k to 316k tok/s by language and packing sustained 73k to 140k tok/s. Packing
efficiency was 99.84% to 100.00%. The main run waited for data only 0.773% of its
training time, so a rolling disk queue would add complexity without benefit.

| Workers | Training tok/s | Data wait | Stable |
|---:|---:|---:|:---:|
| 1 | 799.5 | 0.825% | yes |
| 2 | 798.1 | 0.821% | yes |
| 3 | 795.0 | 0.927% | yes |

One worker won by a small margin.

## GPU Topology Benchmarks

One-T4 full training reached backward but exposed BF16/GradScaler incompatibility.
Two-process replicated DDP then OOMed at sequence length 2,048. Two-GPU FSDP
`FULL_SHARD` fit and trained at about 800 tok/s. Plain DDP and the single-GPU path
were therefore rejected. No model parallelism or architectural change was used.

## Microbatch Benchmarks

Replicated DDP could not fit microbatch 1. FSDP fit microbatch 1 at 11.74 GiB per
GPU. Microbatch 2 without checkpointing OOMed, so larger values were not useful.
The selected effective update was 2 GPUs x 1 sequence x 2,048 tokens x 8 gradient
accumulation steps = 32,768 tokens.

## Gradient Checkpointing Benchmark

Checkpointing ON was the only stable 2,048-token full-weight configuration. The
no-checkpoint microbatch-2 probe OOMed. The selected run kept checkpointing enabled.

## Optimizer Benchmark

Fused torch AdamW failed on the first FSDP optimizer step with a scalar broadcast
shape error. Bitsandbytes 8-bit AdamW completed every benchmark, pilot, and main
optimizer step. It used 11.74 GiB per GPU and sustained 799.5 tok/s in the selected
probe. This is an optimizer-state memory choice, not QAT.

| Configuration | GPUs | MB/GPU | Accum | Tokens/update | Checkpointing | Optimizer | VRAM/GPU | Data wait | Tok/s | Stable |
|---|---:|---:|---:|---:|:---:|---|---:|---:|---:|:---:|
| FSDP torch AdamW | 2 | 1 | 8 | 32,768 | on | fused AdamW | n/a | n/a | n/a | no |
| FSDP 8-bit AdamW | 2 | 1 | 8 | 32,768 | on | AdamW 8-bit | 11.74 GiB | 0.825% | 799.5 | yes |

## Selected Training Configuration

- Transformers plus Accelerate FSDP `FULL_SHARD`
- 2 T4 GPUs, FP32 master weights, FP16 autocast, dynamic loss scaling
- Sequence length 2,048; microbatch 1/GPU; accumulation 8
- 32,768 tokens per optimizer update
- Gradient checkpointing ON
- AdamW 8-bit, weight decay 0.01
- Maximum gradient norm 1.0
- Five warmup optimizer steps followed by constant LR
- One data worker, `prefetch_factor=2`

## Learning-Rate Sweep

Every pilot restarted from untouched Qwen and consumed the same first 524,288
training tokens. Lower NLL is better; deltas are against the frozen baseline.

| LR | Pilot tokens | Overall code delta | General delta | Stable |
|---:|---:|---:|---:|:---:|
| 3e-6 | 524,288 | -0.736% | +0.751% | yes |
| 1e-5 | 524,288 | +1.208% | -0.032% | yes |
| 3e-5 | 524,288 | +7.489% | +2.469% | yes |

All pilots had zero loss-scale overflows and no NaN/Inf. Numerical stability alone
was insufficient: 1e-5 and 3e-5 damaged held-out code NLL.

## Selected Learning Rate

`3e-6`. It was the only candidate with broad code improvement and no language
regression over 2%. The main run restarted from the untouched base checkpoint.

## Main Training Run

The run consumed 11,993,088 real nonpadding tokens in 2,928 forward/backward
microsteps and 366 optimizer steps. Training took 14,807.5 seconds, or 4.11 hours,
and the full process including loading and checkpoint evaluation took 4.17 hours.
End-to-end training throughput was 809.9 tok/s; steady-state throughput was 810.0
tok/s. The configured maximum token count stopped training before the wall deadline.

The Kaggle kernel status is `ERROR` only because Accelerate failed while serializing
the optional FSDP optimizer resume state after the final model and eval were saved.
The model run itself reached `max_tokens`. The repository now catches this
version-specific save failure, removes the partial state, and keeps the completed
snapshot. No training rerun is needed.

## Actual Language Token Mix

| Language | Tokens | Actual share |
|---|---:|---:|
| Python | 2,400,256 | 20.014% |
| TypeScript | 1,798,144 | 14.993% |
| JavaScript | 1,196,032 | 9.973% |
| Java | 1,200,128 | 10.007% |
| C++ | 1,200,128 | 10.007% |
| Rust | 1,200,128 | 10.007% |
| Go | 960,512 | 8.009% |
| C | 718,848 | 5.994% |
| C# | 718,848 | 5.994% |
| Shell | 600,064 | 5.003% |

## Training Stability

- NaN/Inf: 0
- Loss-scale overflows: 0
- Gradient norm: 1.832 minimum, 7.268 maximum, 366 recorded
- Peak allocated VRAM: 11.742 GiB on each T4
- Initial recorded update loss: 1.3076
- Final recorded update loss: 0.8758
- Data wait: 0.773%

## Checkpoints Produced

| Name | Actual training tokens | Optimizer steps | Safetensors opened locally |
|---|---:|---:|:---:|
| `tokens-001000000` | 1,015,808 | 31 | PASS |
| `tokens-002500000` | 2,523,136 | 77 | PASS |
| `tokens-005000000` | 5,013,504 | 153 | PASS |
| `final` | 11,993,088 | 366 | PASS |

Each checkpoint contains the causal model, tokenizer, config, training metadata,
MICRO metrics, and the original native-MTP sidecar. The partial 3.75 GiB FSDP
resume file is not a valid resume checkpoint and is not promoted.

## Per-Checkpoint Code NLL

| Metric | BASE | 1.016M | 2.523M | 5.014M | 11.993M |
|---|---:|---:|---:|---:|---:|
| Python | 1.263446 | 1.257290 | 1.254493 | 1.252599 | 1.257542 |
| TypeScript | 1.190239 | 1.171970 | 1.172090 | 1.170404 | 1.169318 |
| JavaScript | 1.231788 | 1.220170 | 1.219772 | 1.222726 | 1.219624 |
| Java | 0.920384 | 0.907480 | 0.906577 | 0.903226 | 0.906835 |
| C++ | 1.324023 | 1.312870 | 1.313892 | 1.310572 | 1.314603 |
| Rust | 1.011945 | 1.005888 | 1.006108 | 1.008771 | 1.007856 |
| Go | 0.852880 | 0.853776 | 0.851029 | 0.848386 | 0.846396 |
| C | 1.341754 | 1.327811 | 1.325276 | 1.323796 | 1.326134 |
| C# | 0.880329 | 0.869361 | 0.868685 | 0.869014 | 0.870801 |
| **Overall code** | **1.112976** | **1.102957** | **1.101991** | **1.101055** | **1.102123** |
| General text | 2.417105 | 2.447062 | 2.468746 | 2.483156 | 2.486868 |

At 5.014M tokens, all nine languages improved. Overall code NLL fell 1.071%.

## General-Language Regression

The promoted 5.014M checkpoint increased general-text NLL from 2.417105 to
2.483156, or 2.733%. The final checkpoint reached 2.486868, or 2.886% above base.
This is measurable forgetting but far below the 25% collapse gate.

## MTP Preservation Findings

Transformers 5.5.0 does not register the native Qwen3.5 MTP module in
`Qwen3_5ForCausalLM`, so causal CPT neither loaded nor trained those parameters.
Stock `save_pretrained()` would drop them while leaving `mtp_num_hidden_layers=1`.
Every snapshot therefore preserves the original 15 MTP tensors, 20,452,864 BF16
parameters and 39.0 MiB, in `mtp-original.safetensors` plus a manifest pinned to the
base revision. The config still has `mtp_num_hidden_layers=1`. Restoring usable MTP
generation will require a Qwen-specific adapter later. MTP was not trained tonight.

## Kaggle GPU Time Used

The successful v3 kernel used 4.864 wall-clock hours on two T4s, or 9.728 T4
GPU-hours. This includes 6.3 minutes of configuration probes, 35.2 minutes of LR
pilots, and 4.17 hours for the main process. Earlier failed smoke probes consumed a
small additional amount and stopped quickly; they did not approach the weekly quota.

## Checkpoint Pull / Local Reload

PASS. Kaggle CLI pulled all four model snapshots, MICRO evaluations, logs, loss and
gradient histories, environment metadata, and the failed partial resume state. All
four model safetensors files opened locally and contained 321 tensors. The promoted
5.014M checkpoint loaded on CPU as `Qwen3_5ForCausalLM`, produced finite logits of
shape `[1, 9, 248320]`, and deterministically completed `def add(a, b): return` with
`a + b`. Parameter count was 752,393,024, excluding the preserved MTP sidecar.

## Best Current Checkpoint

`outputs/kaggle/code_cpt_train_v3/code_cpt_run/main/snapshots/tokens-005000000`

Model safetensors SHA-256:
`d499c3fe2d720246c710c5c9e20653a4599aa25d77ebef871cb28431ccc888e8`.
It wins on the primary held-out metric even though the final checkpoint is newer.

## Known Problems

- Accelerate 1.13.0 cannot serialize this FSDP plus 8-bit-Adam optimizer state. The
  final model is valid, but the partial resume file is not. Future kernels now treat
  resume serialization as optional and report failure without losing snapshots.
- The fused torch AdamW FSDP path has a scalar-state broadcast bug in this package
  combination.
- The Python 3.12 Kaggle runtime differs from the repository's Python 3.11 local
  environment.
- MICRO is a strong leakage-resistant thermometer, not a complete product
  benchmark. The post-run executable suites below reduce this uncertainty but are
  still small and synthetic.

## Post-run Functional Evaluation

The frozen 200-case causal completion suite executes hidden behavioral tests in a
network-disabled container. Version 2 corrects C/C++ fixtures so compiled binaries
are actually run. Base passed 2/200, 5.014M F16 passed 7/200, 11.993M F16 passed
9/200, and 5.014M Q4_K_M passed 2/200. The 11.993M checkpoint is significantly
better than Base on the paired outcomes (`p=0.039`), but its 9 versus 7 comparison
with 5.014M is not decisive (`p=0.727`).

A 31-case causal code-output suite found 2/27 parsable code continuations for Base,
9/27 for both 5.014M and 11.993M F16, and 7/27 for 5.014M Q4_K_M. A 25-case
repository-coherent long-context suite scored Base 2/25, 5.014M F16 1/25, and
11.993M F16 3/25. The 12M passes included 16k and 32k prompts, so no catastrophic
context loss was detected, but all three passes used the same enum dependency type
and broad long-context quality remains unclear.

The separate 200-case marked-region next-edit suite scored 0 functional patches
for Base, 5.014M F16, 11.993M F16, and 5.014M Q4_K_M. Every model always emitted a
nonempty response and hit the 96-token cap; none recognized `NO_EDIT`. This is the
expected Stage boundary: causal CPT improved code competence but did not teach the
next-edit protocol.

## Next Recommended Experiment

Increase the executable causal and long-context suites enough to resolve the
5.014M-versus-11.993M disagreement, then run a small fresh-data CPT recipe
tournament with LR decay below 3e-6 and checkpoint gates. In parallel, begin Stage
2 from the promoted 5.014M foundation using explicit next-edit, `NO_EDIT`, stopping,
and repository-context supervision. Do not use FIM as a substitute for next-edit.

## Final Research Answer

**YES.** The best TabComplete-Code checkpoint is measurably better than untouched
Qwen3.5-0.8B-Base on unseen repository-held-out code. Overall code NLL improved
1.071%, every core language improved, generation remained non-degenerate, and the
checkpoint passed local safetensors and CPU reload checks. Training-loss reduction
was not used as the promotion criterion.
