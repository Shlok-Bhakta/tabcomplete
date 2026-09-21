# Stage-1 auxiliary evaluation axes

Held-out per-language NLL remains the primary Stage-1 metric. These smaller frozen
suites probe two product-relevant failure modes that token loss alone cannot show:
whether causal output starts as code and remains syntactically usable, and whether
short-context CPT damages use of distant repository facts.

## Code-output specialization

`code_output_v1` contains 31 prompts: 27 code contexts across the nine core
languages and four Markdown controls. It does not instruct the model to answer a
question. Each prompt is an unfinished document, matching the causal objective.
The evaluator measures immediate code, conversational preambles, Markdown fences,
and whether the completed code parses.

- Suite: `data/benchmarks/code_output_v1.jsonl`
- SHA-256: `c28a46bf7920591e78760888ed0d838cd836397626b82eef395af3f27a1e5168`
- Decoding: greedy, 96-token maximum

| Model | Immediate code | Preamble | Fence in code | Parse/code-output pass |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base F16 | 27/27 | 0/27 | 1/27 | 2/27 (7.4%) |
| TabComplete-Code 5.014M F16 | 27/27 | 0/27 | 0/27 | 9/27 (33.3%) |
| TabComplete-Code 11.993M F16 | 27/27 | 0/27 | 0/27 | 9/27 (33.3%) |
| TabComplete-Code 5.014M Q4_K_M | 27/27 | 0/27 | 0/27 | 7/27 (25.9%) |

Both F16 Stage-1 checkpoints produce materially more syntactically complete causal
code than base on this small suite, while 11.993M does not improve on 5.014M. The
5M model still hits the generation cap in 23 of 31 cases, so stopping behavior
remains weak. The sample is too small for a standalone promotion claim, but its
direction agrees with code NLL and the 200-case executable completion benchmark.

## Long-context repository use

`long_context_v1` is a deterministic 25-case Python benchmark at approximate
prompt sizes 2k, 4k, 8k, 16k, and 32k tokens. Each context is a coherent synthetic
repository module with one distant dependency that must be retrieved and used at
the completion point. Dependency types cover constants, enums, function
signatures, dataclass fields, and nested configuration. Needle depth rotates across
12%, 31%, 50%, 69%, and 88%.

- Spec: `data/benchmarks/long_context_v1.spec.json`
- Materialized suite fingerprint:
  `23bb7fc9709787b879d7ee4deaf9360f877d6d0b4840823ac0c4c0a28a8bd523`
- Cases: 25
- Gold control: 25/25
- Blind constant-zero control: 0/25

| Prompt size | Base F16 | 5.014M F16 | 11.993M F16 |
|---|---:|---:|---:|
| 2k | 1/5 | 1/5 | 0/5 |
| 4k | 0/5 | 0/5 | 1/5 |
| 8k | 1/5 | 0/5 | 0/5 |
| 16k | 0/5 | 0/5 | 1/5 |
| 32k | 0/5 | 0/5 | 1/5 |
| **Overall** | **2/25** | **1/25** | **3/25** |

All three 11.993M passes are enum-dependency cases. The suite therefore finds no
catastrophic loss of 16k/32k use after short-context CPT, but it does not establish
broad long-context improvement either. Base, 5M, and 12M differ by only one or two
passes, all models remain poor overall, and continuation/stopping failures dominate.
The result is best read as **no detected context collapse; comparative quality
unclear**.

This axis is intentionally a retrieval-and-use test, not a password needle. It is
also diagnostic rather than a broad long-context benchmark: all fixtures are
Python and each dependency has one expected use. Future versions should add the
other core languages and multi-file symbol resolution.
