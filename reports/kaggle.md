# Kaggle GPU Run Report — first real training, 2026-09-17

Kernel `shlokbhakta/tabcomplete-lora-smoke` v8, Tesla T4 x2, fp16.
Quota used: 0.41h of 30h. Dataset: 4688 teacher labels (DeepSeek flash).

## Result: LoRA smoke COMPLETE, all mechanics proven

- Model `Qwen/Qwen3.5-0.8B-Base` loads (Unsloth + transformers 5.x), tokenizer
  unwrapped from VL processor mapping, dataset formats, backward pass runs,
  checkpoints save at 25/50/75/100, adapter reloads.
- 100 optimizer steps, wall 728s (~12 min).
- Tokens actually seen ≈ 100 × 2 × 4 × ~150 ≈ **120k** (~165 tok/s).
  (An earlier report computed dataset×steps = 67M; fixed in `train.py` to
  steps×batch×accum×mean. `tokens_per_second` in the v8 output is inflated —
  recompute as ~165 tok/s.)
- Peak VRAM **1.34 GB** — 0.8B LoRA is tiny on T4; headroom for longer
  contexts/bigger batches.
- Adapter: 6.4M params (r16), 25.6 MB safetensors, mean abs 0.006 (non-zero —
  weights genuinely moved from init).
- Eval NLL base vs lora on 3 probes: identical to 3 decimals (0.486/0.559/0.543).
  Expected, not alarming: base already ~0.5 NLL on trivially predictable
  synthetic snippets (near floor), 100 steps is a mechanics smoke, not a
  capability run. Per-step loss curve was not captured in fetched logs (gap —
  next run should `trainer.log_history` to a JSON artifact).
- No NaN observed (fp16 GDN watch: clear for this smoke).

## Path to a real signal

Longer run (1–3k steps) + harder validation slice (git-mined trajectories,
multi-line blocks) + log_history artifact. Then Colab L4/bf16 comparison for
tokens-per-compute-unit. Artifacts: kernel outputs (adapter + checkpoints)
pulled to local evidence; adapter not committed to git (25 MB, regenerable).
