# TabComplete-Code 0.8B Stage-1 preview

This prerelease contains the experimental Q4_K_M deployment build of the best
Stage-1 code continued-pretraining checkpoint.

- Base: `Qwen/Qwen3.5-0.8B-Base` at revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Training: 5,013,504 code tokens from `bigcode/the-stack-dedup`
- Objective: ordinary causal language modeling only
- Held-out code NLL: 1.112976 base to 1.101055 Stage-1, a 1.071% reduction
- Languages improved by held-out NLL: all nine core languages
- General-text NLL: 2.417105 to 2.483156, a 2.733% increase
- Q4_K_M size: 541,903,264 bytes
- Q4_K_M SHA-256:
  `3bdb86c93d9756cd0ac3fd7820adb6b23acabb98e0afd85825a8470ec6ab3abb`

The native 15-tensor MTP block is preserved in the GGUF, but Stage 1 did not train
or enable MTP. llama.cpp ordinary causal inference reports those tensors unused.

The executable 200-case benchmark is promising for F16 but not conclusive. F16
Stage-1 passed 7 hidden-test cases versus 3 for base. Q4_K_M passed 3. The strict
protocol exposes weak completion stopping behavior, and Q4 is not yet established
as quality-equivalent to F16. This build is for local testing, not the final
next-edit model.

Run with a current llama.cpp build:

```bash
llama-server -m TabComplete-Code-0.8B-5M-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8091 --reasoning off
```

See `reports/code_cpt/overnight.md` and `reports/code_benchmark.md` for the full
training and evaluation evidence.
