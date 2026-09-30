# Sweep comparison allocation amendment 3

`plan-v3.json` preserves the comparison and runtime identity from revision 2. It records one failed pre-inference build attempt and authorizes at most one retry, for two total attempts.

- Revision 2 comparison plan SHA-256: `53256921e9135b432ed029e7a083c3ee12c27619ce200c4be4c5b91dc5200b51`
- Revision 2 plan file SHA-256: `8b6f781d153b58284c3c994774cc5cc82bbb9cfbc1615a51f0079080e7573a97`
- Allocation amendment 3 self-hash: `26781d965031fa3c486a771d4b75f8a08851b8aa4b82eaa095e87ad22e89bdb6`
- Allocation amendment 3 plan file SHA-256: `ecaada053a6c407c354562208c969b19c22ee54aa282b7d2296635550d3e5511`
- Runner SHA-256: `2a896dc411928ad0392444232999f7eeb24da3ba7f6dbb680750dab8ed4b85e0`
- Test SHA-256: `8402b086b8e63144ab9968464adef4a88d6a1163b788a36e311d85a7885ad63b`
- Next-edit prompt-only fixture SHA-256: `bc483e98938d6920d4323dac886585330d27eb0fb075debf9cf2e0dc57cf75e3`

Attempt 1 failed during `runtime_build` after 54.991668 worker-session seconds. The captured configure output shows CUDA 12.8.93 and architecture 75 were detected, then CMake failed because `CUDA::cuda_driver` was unresolved. The attempt directory contains setup and build logs plus status only; model inference did not start and no comparison predictions were produced. Its sorted output-file manifest has SHA-256 `2793525842a0ee1e61628e2f1505e316c04eb317baceed0b3d78823843524c02`. The status and configure log hashes are recorded separately in the JSON plan. Kaggle allocation start/stop wall time is not in that worker record and must be reconciled from the authenticated quota observations before retry.

The retry keeps the pinned llama.cpp revision, model files, prompts, fixtures, scoring, decoding, and output budgets. Its build prerequisite is to use only an already-installed driver library discovered through `ldconfig -p`; if no usable library is found, the pinned CMake source supports `GGML_CUDA_NO_VMM=ON` to avoid the direct driver link while retaining the CUDA backend. No driver or runtime installation is authorized. The selected build branch and actual flags must be recorded before inference, and both precisions must use the same resulting runtime build.

The amendment allows two total attempts, one at most 7,200 seconds each, with a 1,200-second finalization reserve, one active GPU job, and a strict 14,400-second aggregate GPU-session wall cap including setup, compilation, evaluation, saving, and failures. Attempt 1 counts toward that cap. No third attempt, blind submission retry, paid compute, or automatic renewal consumption is allowed. The v2 quota observation is historical only: refresh authenticated quota and verify all job statuses immediately before retry; stop if quota or cumulative wall time cannot be reconciled.

`load_and_verify_plan` accepted `plan-v3.json` with the frozen runner, test, strict fixture, line fixture, next-edit suite, prompt fixture, and pinned runtime hashes. No model outputs were generated while making this amendment. `plan.json` and `plan-v2.json` remain unchanged.
