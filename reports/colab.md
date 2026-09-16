# Colab Harness Report

Status: harness built, **not yet run on GPU** (local machine is CPU-only;
no Colab runtime was launched from here — by design).

## Versions (verified 2026-09-16)

| package | local (CPU dev) | Colab GPU (fill on first run) |
|---|---|---|
| transformers | 5.17.0 (`Qwen3_5ForCausalLM`, `Qwen3_5TextConfig` OK) | requires v5+ per Unsloth docs |
| torch | 2.14.0+cpu (local only; never in Colab) | Colab stock CUDA torch |
| unsloth | not installed locally (CUDA-oriented) | `pip install unsloth`, record version |
| trl / peft / datasets | via unsloth in Colab | record versions |
| tokenizer | `Qwen/Qwen3.5-0.8B-Base` loads from hub (config/tokenizer only) | same |

Unsloth Qwen3.5 notes (current official docs, checked 2026-09-16):
- `FastLanguageModel.from_pretrained(model_name=...)` + `get_peft_model`
  (generic FastModel path; no dedicated FastQwen3_5 class needed for 0.8B).
- LoRA targets: q/k/v/o + gate/up/down; `use_gradient_checkpointing="unsloth"`.
- Gated-DeltaNet layers run on flash-linear-attention Triton kernels when
  available, else slow pure-PyTorch recurrence (matches our CPU probe).
- Known upstream issue: GDN float16 grad-norm NaNs during training
  (qwen3_5 in Unsloth FORCE_FLOAT32 handling) — prefer L4/bf16; T4 fp16
  smokes must watch loss for NaN.

## Artifacts

- `notebooks/qwen35_colab.ipynb` (config → runtime → deps → dataset →
  LoRA smoke → save → reload/eval → throughput)
- `src/tinycomplete/train/train.py` (lazy GPU imports; CPU-importable)
- `configs/qwen35_lora_smoke.yaml` (100 steps @ 2048, LoRA r16)
- `configs/qwen35_full_smoke.yaml` (15 steps @ 1024, full weights — OOM probe only)
- `scripts/colab_smoke.py` — local gate (configs + formatting + token stats)

## First GPU run checklist

1. L4 runtime, `COLAB_CU_PER_HOUR` filled in.
2. `pip install unsloth` versions pasted into the table above.
3. LoRA smoke loss decreases; checkpoint reloads; eval NLL before/after recorded.
4. Throughput: tokens/sec, peak VRAM, tokens per compute unit.
