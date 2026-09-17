# Kaggle GPU path (agent-driven, no browser needed)

Kaggle exposes an **official CLI/API** where `kernels push` uploads AND
executes a notebook on a GPU worker — this is the one cloud path the local
agent can actually drive itself. Free quota: ~30h GPU/week (T4 x2 = 32GB),
sessions run for hours. Our 100-step 0.8B LoRA smoke should take well under
an hour on T4.

Caveats vs Colab L4: T4 is **fp16-only** (no bf16). Qwen3.5 GDN layers are
float16-grad-norm sensitive (Unsloth FORCE_FLOAT32 notes), so the kernel uses
`precision="fp16"` and the eval cell watches loss/NLL for NaN. If NLL explodes,
the answer is L4/bf16 in Colab, not more T4 steps.

## What the human must do (once)

1. Create a Kaggle account, **verify phone number** (mandatory for GPU).
2. `kaggle.com/settings` → API → Create New Token → downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (`chmod 600`) on THIS machine and say go.
   (Never paste the key into chat; the agent only needs the file in place.)

## What the agent then does

```bash
pip install kaggle  # or uv tool / uvx
export KAGGLE_CONFIG_DIR=~/.kaggle
kaggle kernels list --mine                       # auth check (no secret printed)
# 1. snapshot training data as a dataset
cp data/generated/train_deepseek.jsonl kaggle/dataset/
cd kaggle/dataset && kaggle datasets push -p .  # first push creates it
# 2. push + execute the kernel (replace YOUR_USERNAME in both metadata files)
kaggle kernels push -p kaggle/kernel
kaggle kernels status <user>/tabcomplete-lora-smoke
kaggle kernels output <user>/tabcomplete-lora-smoke -p outputs/kaggle --force
```

Kernel: `kaggle/kernel/qwen35_lora_smoke.ipynb` (metadata pins
`machine_shape: NvidiaTeslaT4` — `enable_gpu` alone lands on a P100 that
modern torch can't use). Internet enabled for HF weights + unsloth install.

## Cost

$0 (free tier). No cloud billing involved. If quota is exhausted, the push
fails loudly — fall back to the Colab notebook (`notebooks/qwen35_colab.ipynb`).
