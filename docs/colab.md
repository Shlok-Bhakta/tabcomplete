# Colab workflow (human-run, GPU lives in Colab)

Preferred GPU: **NVIDIA L4 24GB** (bf16, roomy for 0.8B LoRA at 2K context).
Fallback: **T4 16GB** (fp16 path; GDN layers are float16-grad-norm sensitive —
keep smokes short and watch for NaN; see reports/colab.md). No TPU (CUDA/Unsloth
stack), no A100/H100 until tokens-per-CU justify it.

## Steps

1. Open `notebooks/qwen35_colab.ipynb` in Colab with a GPU runtime (L4 preferred).
2. Cell 1: set `MODEL_ID`, steps, seed, and `COLAB_CU_PER_HOUR` from the
   pay-as-you-go panel.
3. Cell 2: confirm CUDA (`torch.cuda.is_available()` + `nvidia-smi`).
4. Cell 3: clone this repo, `pip install unsloth` (keeps Colab's CUDA torch —
   never install a CPU torch there), record versions into `reports/colab.md`.
5. Cell 4: build the fixture dataset, check counts + token quantiles.
6. Cell 5: LoRA smoke (`configs/qwen35_lora_smoke.yaml`, 100 steps @ 2048).
7. Cell 6: persist checkpoints (Drive mount optional; `/content` is ephemeral).
8. Cell 7: reload adapter, compare before/after NLL + sample outputs.
9. Cell 8: throughput report → tokens per compute unit.

## Context-length policy

Train short first: 1K–2K buckets dominate, then 2K–4K; 4K–8K minority;
8K–16K rare probes. Never train everything at 13K/32K — quadratic attention
burn is not justified when retrieval (Tree-sitter/LSP/recent edits) keeps
contexts dense. `MAX_SEQ_LENGTH = 2048` in the smoke configs.

## Local validation (no GPU)

`uv run python scripts/colab_smoke.py` checks both YAML configs, the
formatting pipeline, and real Qwen token stats. `bash scripts/smoke.sh`
runs the whole local gate.
