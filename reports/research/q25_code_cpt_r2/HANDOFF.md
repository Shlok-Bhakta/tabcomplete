# Qwen2.5 completion campaign handoff

Worktree `/home/crabcake/Projects/tabcomplete-q25-code-cpt-r2`, branch
`research/q25-code-cpt-r2`. Private artifacts
`/mnt/ssd/tabcomplete-q25-code-cpt-r2`. Preserve the user-dirty original worktree.

## Current state at 2026-10-04 12:23:56 UTC

CPT, both matched FIM arms, selected Q4 conversion, 240 native development cases,
and two 144-request CPU replays are complete. Do not retrain, requantize, change
fixtures, extend the registered budget, or download another model.

Direct FIM scored130/240 exact+EOS; CPT→FIM127/240. Paired difference is
inconclusive. Native direct-FIM Q4 scored126/240 exact+valid+EOS;239 raw EOS,
235 valid actions, one cap and four line-format errors. Joint runtime/precision
comparison, not a pure quantization ablation. See native_quality_comparison.json.

Four generation/four prompt threads selected. Changed-state median7806.5ms,
p95 8275.998ms; identical-repeat median217.917ms,p95 446.503ms. Two threads were
slower and changed24/144 actions. Cached/fresh outputs can differ; no bitwise
consistency or general quality claim. Peak sampledRSS630923264B, no predictor
swap. Hostswap-in882pages and177major faults were observed in t4; this does not
establish predictor thrashing. See native_runtime_selection.json.

## Active deployment and remaining blocker

Desktop installer activated the selected service on loopback19093, CPU only,
ctx2304,input1024,output96,b256,ub64,F16KV,one slot,no saved snapshots,
MemoryMax1500M/SwapMax0/CPUQuota400%. Actual installed config/backup receipt:
`fim/native/installed-native-configuration.json`. Runtime hash
373ca1316a99065ae9e5c503db2ab898efa98abeabe002d18983ca1165102637.
ModelSHA ff43d25913e982c3580614ad0528722d9b261b6575d4e06349f9b8509b368682,
491399808B. Embedded ELF SHA
9639694abb385ace116a659602b70f0d79a19e868b746641afda42278d69543a,
501920617B. It directly maps its own GGUF region, no extraction or second model.

Automatic experimental opt-in is configured, quality false, training false,
normal-mode opt-in, Alt+l accept, Alt+p force. **Editor verification is pending.**
The real headless smoke exposed a cold-start bootstrap bug: buffer_state needs
`current_model_identity._fim_token_contract`, but setup never fetches it. Thus
zero model requests occurred, not an invalid model response. Diagnostic session
72fdfe3c-13f5-4271-8d8e-67fc769cee00 has six contiguous anchor/delta/session events.
Private status evidence is `fim/native/editor-failure-diagnostic.json`.

Trainer audit agent owns an isolated Lua bootstrap fix and deterministic tests.
Await its commit, inspect/cherry-pick, run Lua suite, then rerun actual selected
model Neovim smoke. No native benchmark rerun is needed for a Lua-only startup
fix, but record its separate editor verification revision. Do not end at install.

## Collector and feedback

Existing collector recovered with exact cached image/configuration and original
SQLite `/mnt/ssd/collector-data/collector.sqlite`, schema/projectionv3. Endpoint
`http://100.100.163.102:8787/healthz`. Original stuck container preserved for
rollback. Consistent8.56MB DB backups passed integrity checks. See private
collector-network-recovery-20261004.json and native_collector_compatibility.json.
Deployed five server sources match this branch. FIM-specific hashes/kind remain
in raw events; generic outcome/context projection works without DDL.

Actual-success session/prediction IDs remain pending. Give exact IDs to data audit
agent to verify all request/action/display/decision/later-delta/blob links, replay,
deduplication, sequence gaps and delayed/censored outcomes. All scripted decisions
must be synthetic, never claimed as human acceptance. Exporter readiness remains
false; automatic preference training stays disabled.

## Completed verification and limits

Current Python code:1217 pytest passes,two warnings; Ruff passes;mypy195 source
files. Collector Bun59,analysis38,gateway16 pass; all TypeScript checks pass.
Rust45 release tests/build pass. New Lua tests await the bootstrap fix.
Native plan/results/replay private paths are `fim/native/quality-t4` and
`replay-t4`,`replay-t2`; frozen plans bind model/binary/source/tokenizer/process.

Fresh quota at12:11:37UTC:41.08/45 account GPU-hours remaining,3.92 used,
renewal2026-10-10T00:00:00,55 verified statuses,no active jobs. Budget ledger
70588/72000 reserved wall seconds,1412 left. Actual inputtokens11426328 of32M.
Artifacttree12514423946/12884901888B,includingcache/temp/binary/staging;
370477942B headroom. Do not create another embedded ELF or local FP16/optimizer
copy. Preserve checkpoints. See campaign_final_budget_audit.json.

ThinkPad approved alias still timed out; no new laptop installation or benchmark.
Retry only `ssh thinkpad`, no tailnet scan. Nix/Home Manager configuration must be
declarative; Nix is absent locally. Tested explicit candidate interface is
`nix/fim-candidate.nix`, not a claim of evaluated/activated laptop deployment.

## Remaining sequence

1. Merge Lua bootstrap fix; actual Lua gates and selected-model smoke.
2. Audit exact SQLite IDs, blob hashes, replay and idempotent retry.
3. Verify installed actual LazyVim config/module paths, mappings, restart-persisted
   automatic experimental mode, request validity and service binary/arguments.
4. Update authoritative report/artifact manifest/HANDOFF with actual evidence,
   feedback readiness, rollback paths and access/visual/latency limitations.
5. Commit and push source and compact scientific evidence, never model weights
   or personal raw data. Do not replace stable laptop artifacts silently.
