# Untouched Qwen3.5-0.8B-Base MICRO baseline

Model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
Tokenizer revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
Precision: fp16 autocast over fp32 master weights
Seed: 271828
GPU: Tesla T4

| Corpus | NLL | Evaluated tokens |
|---|---:|---:|
| Python | 1.263446 | 16,376 |
| TypeScript | 1.190239 | 16,376 |
| JavaScript | 1.231788 | 16,376 |
| Java | 0.920384 | 16,376 |
| C++ | 1.324023 | 16,376 |
| Rust | 1.011945 | 16,376 |
| Go | 0.852880 | 16,376 |
| C | 1.341754 | 16,376 |
| C# | 0.880329 | 16,376 |
| Overall code | 1.112976 | 147,384 |
| General text | 2.417105 | 16,376 |

Package versions:

- `numpy==2.0.2`
- `torch==2.10.0+cu128`
- `transformers==5.5.0`
