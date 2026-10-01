# Rust editor handoff

## Serving product

The ThinkPad runs one CPU-only Rust engine with llama.cpp bindings. Qwen Q4 is the default; Sweep Q4 is an optional picker. Automatic experimental display requires explicit acceptance to edit. Personalization training is disabled.

Commands are `:TabCompletePredict`, `:TabCompleteAccept`, `:TabCompleteReject`, `:TabCompleteMode`, `:TabCompleteStatus`, and `:TabCompleteModel`. Acceptance uses Alt+l. `:TabCompleteMode off` cancels pending work and clears suggestions. Modes persist in the dedicated LazyVim state file.

## Source and installation

Branch `prototype/rust-editor-r1` starts from product commit d847f7b4917d802302c57cfefebdf13fadd00958. Rust and the collector plugin are bundled with immutable Nix model symlinks. Model weights are private and never committed or published. The actual ThinkPad configuration has a focused pinned package/feature/import; unrelated staged and unstaged work is preserved. No full system or Home Manager activation was run.

See `installed_configuration.json` and `ROLLBACK.md` for exact files, service settings, private backups, source pin, and guarded rollback. Preserve the stable Nix store paths and existing collector database.

## Measurements

The fixed ThinkPad comparison selected Qwen. Trained-layout changed-state latency was 2,404 ms median and 3,683 ms p95. The selected cursor-last layout confirmation was 874 ms median and 2,689 ms p95; identical repeats were 160/196 ms. Qwen retained about 556 MiB RSS. Sweep changed-state latency was 12,458/16,212 ms and retained about 941 MiB. Neither measurement establishes human task accuracy.

Related development checks returned 9/24 exact actions for Qwen and 1/24 for Sweep. Qwen has poor insertion/no-edit behavior. Prefix-preserving automatic display filters unsafe removals; it does not improve model quality. Keep quality validation false.

`plan*.json`, `selection.json`, `serving_confirmation.json`, and `runtime_experiments.jsonl` retain the comparison identities. The last scientific confirmation omitted the observability enable flag. Its result files are authoritative and have no invented run ID. Earlier screened runs have imported telemetry and verified payload retrieval.

## Collection and integrity

The authoritative SQLite database remains `/mnt/ssd/collector-data/collector.sqlite` on Crabcake through the existing owned collector API. Schema versions 1, 2, and 3 are unchanged. Backups use SQLite's WAL-consistent backup API.

Keep raw observations immutable. An initial typed partial match can later reach a full match through actual linked deltas. Reloads and navigation break typing continuity; they are not rejections. Gaps and unobserved delayed outcomes remain censored. Scripted outcomes are synthetic and unreviewed, never human acceptance or preference training evidence.

The first real audit found missing `:edit!` reload capture. A collector fix adds whole-buffer reload deltas with explicit origin and format metadata. The first deployment of that fix still loaded old Lua through a Neovim bytecode cache collision: old/new Nix specs had the same size and timestamp. Installation now needs scoped cache invalidation and actual loaded-source checks. Historical failed audits remain in `feedback_verification/`.

## Completion verification

The final installed-source editor run passed eight of eight scenarios. Main session `3b127a5a-24e9-40bf-813f-58d14a8b873e` has 473 events and six projections, 23 anchors, 31 deltas, zero replay mismatches, and zero unanchored deltas. Hashes, resolved file identities, proposal links, acceptance-once, undo, divergent typing, and eventual exact typing all verify. Restart-only session `c00d6eb1-3b45-45c3-9532-f9e1c0722569` replays cleanly but does not contain every main-session scenario. Duplicate delivery of 473 and 41 events adds no rows and leaves database snapshots unchanged. See `feedback_projection_verification.json`, `replay_results.json`, and the detailed audit files.

The real first request took 2,814 ms from final input to the shown event, including 250 ms debounce. `feedback_verification/editor_latency.json` defines the timing stages. Headless verification had no visual UI attached; outcomes are synthetic and unreviewed. Existing open editors were not killed. Start a fresh normal LazyVim session for the installed plugin.

The saved q25 alias survives an explicit service restart. The current runtime hash is `6e58a95c8f5fba5581784636cf5e042a33a14e93b7c764f681eb2ff4c10c54ba`. Permanent Nix code is pinned to `f96d32f7e79945aea432a64ee88775313a58a713`; report-only commits may follow without changing that tested pin. Scoped evaluation passes and no full activation was run.

The version-6 private feedback export has 75 sessions and 107 proposals but zero defensible preference pairs. Four historical gaps remain excluded. Readiness is false and automatic training stays disabled. Do not promote synthetic scripted decisions into human preferences, reinterpret all later work as corrections, or mutate the served model.
