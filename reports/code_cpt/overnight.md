# TabComplete-Code Stage 1 Overnight Report

Status: blocked at the mandatory Hugging Face credential gate. No GPU Stage-1 job has
started. Numerical results appear only after their verification commands pass.

## Repository / Environment

- Starting commit: `ef12f31c6b04b58279a93be56ab0d53ac9958c44`
- Stage-1 branch: `stage1/code-cpt`
- Local development: Python 3.11, CPU-only
- Kaggle CLI authentication: PASS
- Kaggle GPU quota before Stage 1: 0.41 hours used, 29.59 hours remaining
- Existing v8 runtime: 2 Tesla T4 GPUs, 15,360 MiB each

## Hugging Face Authentication

FAIL. `HF_TOKEN` is absent from the local environment and Hugging Face cache. Two
CPU-only Kaggle probes tried `UserSecretsClient().get_secret("HF_TOKEN")`. Both
failed inside Kaggle's secret service with HTTP 400 before any Hugging Face call.
No token value was printed or stored.

## The Stack Dedup Access

FAIL pending Kaggle secret attachment. Anonymous repository metadata is readable,
but a direct ranged request for a Python parquet shard returns `401 GatedRepo`.
No raw Stack or Stack v2 fallback was attempted.

## Dataset Revision

Pinned to `17cad72c886a2858e08d4c349a00d6466f54df63`.

Verified directories: `data/python`, `data/typescript`, `data/javascript`,
`data/java`, `data/cpp`, `data/rust`, `data/go`, `data/c`, `data/c-sharp`, and
`data/shell`.

## Validation Split

Implemented and covered by deterministic tests. The split hashes the canonical
repository name with SHA-256, reads the first eight digest bytes as a big-endian
integer, and takes modulo 1,000. Buckets 0 through 9 are validation-only. The file
path never affects assignment. Frozen validation token arrays, repository hashes,
file hashes, paths, and source hexshas will be saved by the CPU preparation job.

## Base Qwen Metrics

Not run. The mandatory gated-dataset access check has not passed, so the MICRO set
does not exist yet.

## Data Feeder Benchmarks

Not run. The direct streamer uses a 10,000-record shuffle buffer, metadata-first
filters, token-level language quotas, EOS boundaries, exact 2,048-token blocks, and
memory-mapped NumPy arrays. It records network and accepted-source MB/s, files/s,
tokenizer tokens/s, packer tokens/s, filter reasons, and packing efficiency.

## GPU Topology Benchmarks

Not run. The prepared kernel tests one T4 and true two-process DDP on two T4s.
The older v8 run reported two visible GPUs but trained with one data-parallel worker,
so it is not evidence for two-GPU scaling.

## Microbatch Benchmarks

## Gradient Checkpointing Benchmark

## Optimizer Benchmark

## Selected Training Configuration

## Learning-Rate Sweep

## Selected Learning Rate

## Main Training Run

## Actual Language Token Mix

## Training Stability

## Checkpoints Produced

## Per-Checkpoint Code NLL

## General-Language Regression

## MTP Preservation Findings

Transformers 5.17 does not register Qwen3.5's native MTP module in
`Qwen3_5ForCausalLM`. Ordinary causal training therefore does not update it. The
source checkpoint has 15 `mtp.*` tensors with 20,452,864 BF16 parameters, about
39.0 MiB. Stock `save_pretrained()` drops those tensors while retaining
`mtp_num_hidden_layers=1`, which is misleading.

Stage-1 snapshots save the full causal model as safetensors and copy the original
15 tensors into `mtp-original.safetensors` with a manifest. Tests verify that the
sidecar excludes ordinary model tensors. The restoration source is pinned to model
revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Transformers 5.17 still needs a
Qwen-specific config and layout adapter before `generate(use_mtp=True)` can use the
sidecar. MTP remains untrained as required.

## Kaggle GPU Time Used

0.00 hours for Stage 1. Both access probes were CPU-only.

## Checkpoint Pull / Local Reload

Not run. No Stage-1 checkpoint exists.

## Best Current Checkpoint

None.

## Known Problems

- Kaggle has no CLI support for attaching secrets to a kernel. The user must add
  and enable `HF_TOKEN` in the Kaggle notebook editor.
- The current Kaggle image uses Python 3.12. Local development and tests use the
  repository-required Python 3.11. Reinstalling the full CUDA stack solely to change
  the Kaggle interpreter would consume material setup time, so this runtime mismatch
  remains explicit.

## Next Recommended Experiment

Attach `HF_TOKEN` to `tabcomplete-code-cpt-access`, rerun the CPU access probe, then
launch the CPU corpus preparation job. Do not start the T4 kernel until every target
language yields a nonempty Stack Dedup record.
