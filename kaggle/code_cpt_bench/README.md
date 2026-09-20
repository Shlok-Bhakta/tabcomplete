# T4x2 throughput benchmark lab

Controlled training-throughput optimization lab for full-weight CPT of
`Qwen/Qwen3.5-0.8B-Base` on 2x Tesla T4. Does NOT launch a long CPT run.

## What it does

1. `run_bench.py` clones `stage1/code-cpt`, installs base deps
   (`transformers==5.5.0`, `accelerate`, `bitsandbytes`), then *attempts*
   optional speed deps (`flash-linear-attention`, `causal-conv1d`,
   `liger-kernel`). Optional-install failure is recorded, not fatal.
2. Experiment 0: `tinycomplete.code_cpt.bench probe` records the live GDN /
   conv / attention backend selection before any training.
3. Every candidate in `candidates.json` runs in a FRESH
   `torch.distributed.run --nproc_per_node=2` subprocess via
   `tinycomplete.code_cpt.bench bench`, writing one result JSON.
4. Results accumulate in `throughput_benchmarks.jsonl` (+ `.json`);
   `plot.py` renders the five required figures into `plots/`.

Shared controls: same T4 hardware, CUDA env, torch/transformers, dataset,
model revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, seed 271828,
sequence length 2048, token ordering from block 0.

## Run it (Kaggle GPU kernel, prepared-corpus kernel attached)

```bash
git clone --depth 1 --branch stage1/code-cpt \
  https://github.com/Shlok-Bhakta/tabcomplete.git /kaggle/working/tabcomplete
python /kaggle/working/tabcomplete/kaggle/code_cpt_bench/run_bench.py
```

Subset / override:

```bash
python run_bench.py --candidates exp00_baseline_repro,exp01_fla_conv
python run_bench.py --max-tokens 98304   # must stay update-aligned
python run_bench.py --skip-optional-deps # torch-fallback only
```

Stage B finalists: rerun the top 3-5 stable names with
`--max-tokens 524288`. Stage C winner confirmation: `--max-tokens 1048576`
(32 x 32768) on the top 1-2.

## Outputs

- `environment.json`, `backend_probe.json`
- `candidates/<name>/result.json` + `process.log`
- `throughput_benchmarks.jsonl` / `.json`
- `plots/plot{1,2,3,4,5}_*.png` + `plots/summary.json`
