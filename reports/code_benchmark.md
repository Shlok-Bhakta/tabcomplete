# Executable code-completion benchmark

## Purpose

This suite measures whether a causal completion works, not whether it matches one
reference string. Each prediction is inserted between a fixed prefix and suffix,
parsed, compiled or type-checked, and run against hidden behavioral tests. Exact
match remains a diagnostic metric.

The suite is synthetic and was authored after the Stage-1 training corpus was
frozen, so The Stack Dedup cannot contain these exact fixtures. It does not replace
larger public benchmarks, but it avoids their immediate contamination problem.

## Frozen suite

- File: `data/benchmarks/code_completion_v2.jsonl`
- SHA-256: `ed28739b302e4c0d3f9f45e859e7ccd68a1e8594a62eb7b13ee8425b92a439a4`
- Cases: 200
- Core languages: Python 23, TypeScript 23, and 22 each for JavaScript, Java,
  C++, Rust, Go, C, and C#
- Multi-file repository-context cases: 20
- Every case has a parse check, compile or type check, and hidden behavioral test
- Categories: algorithms, control flow, data structures, error handling, parsing,
  repository APIs, and standard-library use

DeepSeek Flash authored the initial public/synthetic fixtures for an estimated
$0.3830. Muse workers repaired fixture infrastructure and incorrect edge-case
oracles. No private source code or Stage-2 teacher records were sent to either
model. The final JSONL, not the generator's nondeterministic output, is the frozen
benchmark.

## Isolation

Predicted code runs only through the explicit `container` backend by default. The
runner uses rootless Podman when available, then Docker. It disables networking,
uses a read-only container root, drops Linux capabilities, sets
`no-new-privileges`, limits memory, CPU, processes, file size, and open files, and
mounts only a fresh per-case fixture directory. Images must already exist because
the runner sets `--pull=never`.

`trusted-host` exists for deliberately trusted unit-test fixtures. Do not use it
for model output.

## Gold verification

The final merged suite passed all 200 parse checks, all 200 compile or type checks,
and all 200 hidden behavioral tests with zero timeouts. Version 2 corrects the C
and C++ fixtures so a successful compilation is followed by execution and a stdout
oracle; version 1 only linked those binaries. Go uses writable offline
workspace caches because the container's `/tmp` mount is intentionally `noexec`.
Its 90-second watchdog accommodates cold standard-library compilation; normal cases
finish much sooner.

```bash
uv run python scripts/evaluate_code_benchmark.py \
  --suite data/benchmarks/code_completion_v2.jsonl \
  --gold --backend container --workers 1 \
  --output-dir outputs/code_benchmark/gold-final
```

## Model protocol

The model sees sorted repository context files followed by the target prefix. It
never sees the expected insertion, suffix, or hidden tests. Generation is greedy
with temperature zero and a fixed maximum token count. Prediction JSONL is flushed
after every case and resumes by case ID. A metadata sidecar binds each run to the
suite hash, model hash, prompt protocol, decoding cap, and worker count. Resuming
with a different model or suite is refused.

```bash
uv run python scripts/generate_code_predictions.py \
  --model-path /path/to/model --max-new-tokens 96 \
  --output outputs/code_benchmark/model/predictions.jsonl

uv run python scripts/evaluate_code_benchmark.py \
  --predictions outputs/code_benchmark/model/predictions.jsonl \
  --backend container --workers 1 \
  --output-dir outputs/code_benchmark/model/results
```

An OpenAI-compatible local server, including llama.cpp or Ollama, can replace the
Transformers loader with `--server-url`, `--server-model`, and an immutable
`--model-revision`. All measured runs used llama.cpp build `f072b10`, four
concurrent server slots, 96 maximum new tokens, and the same raw causal prompt.

This is a strict insertion score. There is no gold-aware truncation, suffix
matching, parser-guided repair, or test-driven retry. Most predictions reached the
96-token cap and continued beyond the useful insertion, so this suite currently
measures completion selection and stopping behavior as well as code knowledge.

## Scoreboard

| Model | Precision | Exact | Parse | Compile | Hidden tests | Median latency |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | F16 GGUF CPU | 0/200 | 20/200 | 8/200 | 2/200 | 16.58 s |
| TabComplete-Code 5.014M | F16 GGUF CPU | 0/200 | 31/200 | 17/200 | 7/200 | 16.28 s |
| TabComplete-Code 11.993M | F16 GGUF CPU | 0/200 | 27/200 | 19/200 | 9/200 | 21.10 s |
| TabComplete-Code 5.014M | Q4_K_M CPU | 0/200 | 32/200 | 15/200 | 2/200 | 12.43 s |

F16 Stage-1 improved every aggregate functional gate over base: parse rate rose
from 10.0% to 15.5%, compile rate from 4.0% to 8.5%, and hidden-test pass rate from
1.0% to 3.5%. It had six base-to-Stage wins, one regression, and one shared pass.
The exact paired two-sided McNemar p-value is 0.125, so 200 cases and only seven
discordant functional outcomes are still not enough to call this statistically decisive.
The result is directionally positive and agrees with held-out NLL, but the
functional claim remains **UNCLEAR**.

The 11.993M checkpoint reached 9/200 hidden-test passes. Against Base it had eight
unique wins, one unique loss, and one shared pass (exact paired two-sided McNemar
`p=0.039`). Against 5.014M it had five unique wins and three losses (`p=0.727`).
This makes 11.993M clearly better than Base on this suite, but not decisively better
than 5.014M. The longer checkpoint also has slightly worse frozen code NLL
(1.102123 versus 1.101055), so the 5.014M NLL winner remains promoted while
11.993M is retained as a functional-eval candidate.

Q4_K_M cut median per-request latency by 23.7% and reduced the deployable model to
541,903,264 bytes, but hidden-test pass rate fell back to 1.0%. With only two Q4
passes this is also noisy, but Q4 is not promoted as quality-equivalent to F16.
No model passed a TypeScript, JavaScript, Java, Rust, or C# hidden test under this
strict protocol. Stage-1 F16 passed one of 20 repository-context cases; base and Q4
passed none.

## GGUF and MTP verification

The Stage-1 causal snapshot stores 15 original native-MTP tensors in a sidecar.
`scripts/prepare_qwen35_gguf_source.py` verifies and indexes that sidecar with the
trained language model before conversion. It also verifies that FSDP's duplicated
`lm_head.weight` is bit-identical to the tied token embedding and removes the
duplicate from the export without changing values.

| Artifact | Tensors | Bytes | SHA-256 |
|---|---:|---:|---|
| F16 GGUF | 335 | 1,557,662,112 | `649b11696a1d3a9b8a89a5795b9ae96f7f68862d4002e397deab635b01571c2b` |
| Q4_K_M GGUF | 335 | 541,903,264 | `3bdb86c93d9756cd0ac3fd7820adb6b23acabb98e0afd85825a8470ec6ab3abb` |

Both artifacts load in llama.cpp and produce deterministic raw completions. The MTP
block is present but reported unused during ordinary causal verification. Stage 1
did not train or enable MTP.

## Browser playground

`tinycomplete-playground` serves the checked-in editor UI and can use either local
Transformers checkpoints or an OpenAI-compatible local server. The Q4 launch is:

```bash
llama-server -m outputs/models/tabcomplete-code-5m-q4_k_m-final.gguf \
  --host 127.0.0.1 --port 8091 --reasoning off
uv run tinycomplete-playground --server-url http://127.0.0.1:8091 \
  --server-model tabcomplete-stage1-q4_k_m \
  --server-label "TabComplete Code 0.8B Q4_K_M"
```

The editor shows ghost text, accepts with Tab, rejects with Escape, attaches sorted
repository-context files, switches models, and stores explicit feedback locally.
