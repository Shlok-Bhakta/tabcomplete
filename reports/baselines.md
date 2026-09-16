# Baselines (planned, not run)

Primary: `Qwen/Qwen3.5-0.8B-Base`. Qwen infrastructure first by policy —
no baseline downloads/training happened overnight.

## Future controls

| model | params | why |
|---|---|---|
| `tiiuae/Falcon-H1-0.5B-Base` | 0.5B | hybrid Mamba/attention control |
| `ibm-granite/granite-4.0-h-350m-base` | 0.35B | hybrid Mamba control |
| RWKV-7 ~0.4B official checkpoint | 0.4B | pure-recurrent control |

## Harness modularity

Swap points already isolated: `src/tinycomplete/model/tiny_qwen.py`
(`build_tiny_config`/`build_tiny_model`), `src/tinycomplete/train/train.py`
(`MODEL_ID`, `load_model_for_training`), and the FIM sentinels in
`src/tinycomplete/data/schema.py`. Adding a baseline = new config +
model-id override; metrics (`eval/metrics.py`) and data generators are
model-agnostic.

## Gate for adding baselines

Only after the Qwen LoRA smoke shows decreasing loss. Compare on the same
frozen validation split with `eval/metrics.py` + tokens-per-compute-unit
from `train.throughput_report`.
