# Executable next-edit benchmark

## What it measures

`next_edit_v1` is a frozen 200-case benchmark for the Stage-2 marked-region
contract. The model receives the current file, a byte-precise editable region,
recent edit events, visible repository context, and a prediction point. It must
return only the replacement text. An exactly empty response means `NO_EDIT` and
preserves the selected region.

This is next-edit prediction, not fill-in-the-middle training or evaluation. The
editable region is supplied to the model, so this version measures edit action and
replacement quality rather than discovering an arbitrary edit location.

- Suite: `data/benchmarks/next_edit_v1.jsonl`
- Cases: 200 (160 replacement, 40 `NO_EDIT`)
- Repository-context cases: 20
- Languages: Python, TypeScript, JavaScript, Java, C++, Rust, Go, C, and C#
- SHA-256: `81efc7fff7706fefa50f376617e677888a055e08ddcb2cc27a4bebad00405607`
- Decoding: greedy, 96-token maximum, raw replacement only

Every candidate is applied to a fresh fixture and scored for action choice, exact
replacement, syntax, compilation or type checking, and hidden behavior. Generated
code runs in the same network-disabled container sandbox documented in
`reports/code_benchmark.md`.

## Controls

The gold replacement passes all 200 cases at every gate. The unchanged-file
control gets all 40 `NO_EDIT` choices right and no replacement choice right. It
nevertheless passes 46 hidden tests because 40 cases intentionally require no edit
and six seeded defects do not affect those particular behavioral assertions. This
is why action accuracy and exact edit accuracy are reported separately from final
file validity.

| Control | Action | Exact | Parse | Compile | Hidden tests | Valid patch |
|---|---:|---:|---:|---:|---:|---:|
| Gold edit | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 |
| Unchanged file | 40/200 | 40/200 | 200/200 | 198/200 | 46/200 | 46/200 |

## Model results

| Model | Action | Exact | Parse | Compile | Hidden tests | Valid patch | Mean character edit distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base F16 | 160/200 | 0/200 | 2/200 | 1/200 | 0/200 | 0/200 | 268.060 |
| TabComplete-Code 5.014M F16 | 160/200 | 0/200 | 0/200 | 0/200 | 0/200 | 0/200 | 263.955 |
| TabComplete-Code 5.014M Q4_K_M | 160/200 | 0/200 | 1/200 | 3/200 | 0/200 | 0/200 | 253.690 |

All three models always emitted a nonempty replacement, so their 80% action
accuracy is merely the 160/200 replacement-class prior. None recognized a
`NO_EDIT` case. Every response hit the 96-token cap, and none passed a hidden
behavioral test after insertion.

## Interpretation

Stage-1 causal continued pretraining improved held-out code NLL and ordinary causal
completion, but it did **not** teach this structured next-edit protocol. The F16
checkpoint is not better than untouched Qwen on this benchmark. Q4's three compile
passes do not constitute functional success because all three still fail hidden
tests.

This negative result is useful: Stage 2 needs explicit next-edit examples,
`NO_EDIT` supervision, replacement stopping behavior, and repository-context
conditioning. The benchmark is ready to measure that specialization without
changing the Stage-1 conclusion. A later version should add edit-location
discovery and larger real repository tasks.

## Reproduction

```bash
uv run python scripts/generate_next_edit_predictions.py \
  --suite data/benchmarks/next_edit_v1.jsonl \
  --server-url http://127.0.0.1:8080 \
  --server-model local-model --model-revision IMMUTABLE_WEIGHT_SHA \
  --max-new-tokens 96 --workers 1 \
  --output outputs/next_edit_benchmark/model/predictions.jsonl

uv run python scripts/evaluate_next_edit_benchmark.py \
  --suite data/benchmarks/next_edit_v1.jsonl \
  --predictions outputs/next_edit_benchmark/model/predictions.jsonl \
  --backend container --workers 1 \
  --output-dir outputs/next_edit_benchmark/model/results
```
