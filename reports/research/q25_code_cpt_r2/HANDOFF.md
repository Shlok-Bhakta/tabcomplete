# Qwen2.5 completion campaign handoff

Worktree `/home/crabcake/Projects/tabcomplete-q25-code-cpt-r2`, branch
`research/q25-code-cpt-r2`. Private artifacts
`/mnt/ssd/tabcomplete-q25-code-cpt-r2`. Preserve the user-dirty original worktree.

## Completed and verified

Raw CPT, both matched completion arms, selected Q4 conversion, 240 native
quality cases, two 144-request CPU replays, editor startup/reload corrections,
actual selected-model headless previews/acceptance/typing/undo/navigation,
installed full LazyVim restart and existing SQLite reconstruction are complete.
No further allocation is running. Do not extend this frozen campaign, retrain,
requantize, change evaluation fixtures, publish weights/data or delete checkpoints.

Direct FIM scored 130/240 exact+EOS versus its untouched baseline 47/240.
CPT→FIM scored 127/240; the direct versus CPT difference is inconclusive.
Raw CPT did not improve causal functional success, 10/200 before and after.
Native direct FIM Q4 scored 126/240 exact+valid+EOS, 235 valid, 239 raw EOS,
one cap and four line-format errors. HF FP16 versus native Q4 changes both
runtime and precision. Synthetic completion results do not establish human
next-edit quality. All training used public/synthetic licensed data, not teachers.

## Running desktop installation

Crabcake CPU, Ryzen 5 PRO 3400GE, four physical/eight logical cores. Service
`tabcomplete-predictor.service` on loopback port 19093 runs the embedded Rust executable
at `fim/native/tabcomplete-q25-fim`. It directly maps its own GGUF region.

- GGUF SHA `ff43d25913e982c3580614ad0528722d9b261b6575d4e06349f9b8509b368682`,
  491399808 bytes, Q4_K_M.
- ELF SHA `9639694abb385ace116a659602b70f0d79a19e868b746641afda42278d69543a`,
  501920617 bytes.
- Installed runtime hash `373ca1316a99065ae9e5c503db2ab898efa98abeabe002d18983ca1165102637`.
- Four generation/four prompt threads, context 2,304/input 1,024/output 96,
  batch 256/microbatch 64, F16 KV, one slot, zero saved snapshots, syntax gate off.
- MemoryMax 1500M, SwapMax 0, CPUQuota 400%. No parallel model or model download.
- Automatic experimental mode persists; quality uncalibrated, training disabled.
- Alt+l accepts; Alt+p forces inference. `:TabCompleteMode off` stops suggestions.
  `manual`, `shadow`, `automatic`, `:TabCompleteStatus` and prediction commands work.

Four-thread changed-state median 7,806.500 ms/p95 8,275.998 ms. Identical repeats
217.917 ms/446.503 ms are not ordinary changing-state latency. Startup 981.441 ms,
model load 494.372 ms; caches were not flushed. Peak/final replay RSS 630,923,264 bytes,
PSS 627,230,720 bytes, predictor swap 0. Host swap-in 882 pages/major faults 177 do not prove
predictor thrashing. Saved snapshots are disabled. KV token prefixes were
verified, but fresh/cache actions differed 6/48 and changed/repeat 8/48.
No exact numerical repeatability or quality benefit is claimed. Prefill dominates.

The approved ThinkPad alias timed out again. No new laptop installation or
performance claim. Preserve its working old model/config. `nix/fim-candidate.nix`
is the explicit declarative interface; Nix is absent locally. No visual or normal
human-editing workload inspection was performed. See native_access_verification.json.

## Editor bugs fixed and tested

The real smoke initially made zero requests because FIM setup did not fetch
its required tokenizer identity. Lua now performs one coalesced async refresh,
bounded retries, generation guards and latest-state automatic debounce. Explicit
requests retain their origin and cancel after navigation. Off prevents callbacks
from restarting prediction. A second correction preserves the FIM protocol when
the plugin entry calls setup without arguments. Source commits `a34c8ce` and `3acc1c4`.

The pinned native tokenizer compatibility fix remains required: ordinary HF token
128247 `</s>` is inferred as native EOG, but only the exact pinned alias is accepted.
FIM stops only at declared EOS 151643. All other controls remain strict; legacy
EOS behavior is unchanged. Native plans were frozen after this correction.

## Feedback verification and provenance

Existing DB `/mnt/ssd/collector-data/collector.sqlite`, schema/projectionv3;
collector endpoint `http://100.100.163.102:8787`. No new database or destructive
migration. Recovery used the exact cached container image/config and preserved
original rollback container and consistent DB backups. Source compatibility proof
is native_collector_compatibility.json.

Verified synthetic session `30243bc9-bd3a-4c73-9d05-a1ea052b5fd1` has 47 contiguous
events, four prediction projections, 15 anchors/eight deltas, final 276-byte replay,
zero unanchored/mismatched changes. Five context/raw/action blobs decompress to
correct SHA-256/byte counts. Resolved file ID 1443 belongs to the session repository.

- Accepted `d6b9a0c1-7407-472b-a896-5071fb959a92`: request 7/generated 9/shown 10,
  acceptance 12; insertion 11 reversed byte-for-byte at 13 by the test undo command.
- Divergent typing `a624a1b9-cbf0-462c-abf3-2642c3aa53f3`: shown 20,
  dismissal 22 links actual delta 21.
- Typed match `c42aeff2-1856-4721-8ceb-f85f240885da`: shown 29,
  dismissal 31 links actual delta 30. It is not negative feedback.
- Navigation `8f0675fe-1c51-427b-901f-e93f4495a90e`: request 35,
  buffer leave 36/cancelled unseen 37, no generation/display.

Reverse-order redelivery skipped all 47 acknowledged events and left the full
projection hash unchanged. These scripted outcomes are synthetic/unreviewed,
human_verified=false. A second successful session
`79446d50-be2a-4372-a349-fb080b7749e6` dispatched actual `<M-l>` through Neovim.
Raw context prompts were identical in these short-file requests; cached display
latencies 464/458 ms must not be advertised as general changed-context performance.

Current exporter: 96 observed sessions/292 proposals, zero candidate/defensible
preference pairs, eight historical gaps outside the verified session. Five/30s
unobserved windows are censored, next save is observed. Automatic personalization
and perpetual training remain disabled. No RL claim. See native_feedback_*.json.

## Gates and monitoring

1217 Python tests/two warnings, Ruff, mypy 195 files; Rust 45; Neovim 93 plus
separate automatic/SSE and explicit-review scripts; collector server 59,
analysis 38, gateway 16 and all three TypeScript checks pass. Later changes are Lua
only; Python/Rust/server sources remain the versions tested. Actual full installed
LazyVim initialization and persisted mode/commands/mappings pass in fresh processes.

Training SigNoz runs `run-73c890ceb76492f727219afff8f0ce0c` and shared FIM
`run-af49853a32ad56891191cf2d46ec9cee` have imported/query/dedup receipts. The two
FIM attempt IDs distinguish arms. Native per-request SigNoz traces were not
captured; scientific JSON and SQLite are authoritative. No fake trace claim.

## Final budgets and rollback

Quota observed 2026-10-04T13:05:02UTC: 41.08/45 account GPU-hours remaining,
3.92 used, renewal2026-10-10. 55 statuses verified, including two authenticated 404,
no active jobs. Training input 11,426,328/32M; reserved wall 70,588/72,000 s,
1,412 s remaining. Conservative GPU reservation 20.7189/40 h is not measured usage.
Artifact tree 12,516,972,641/12,884,901,888 bytes with 367,929,247 bytes headroom; includes temporary
files/cache/embedded binary. Free SSD remains above 2 GiB. No extra allocation or
large artifact fits automatically. Preserve full training states and old models.

Installed config and unit paths/hashes/backups are recorded in
native_installed_configuration.json. To roll back, restore both recorded backups,
`systemctl --user daemon-reload`, restart `tabcomplete-predictor.service`, then
restart Neovim. Old binary/model paths are preserved. Immediate UI stop is
`:TabCompleteMode off`.

Next research should address actual editor intent and the measured prefill
bottleneck under a new frozen plan, with clean authorized data and target access.
Further raw-code pretraining has not earned another extension from these results.
