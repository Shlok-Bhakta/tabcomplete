# Normal cursor feedback handoff

Branch `prototype/rust-editor-format-r2`, code pin
`653fcbe8b2a8fe8f8f11c0e4b56be0b7b99b7494`.

The ThinkPad's declarative source pin and normal-mode option are updated. Package
build and full Home Manager evaluation pass. The fetched runtime source excludes
root reports through `postFetch`; hash is
`sha256-mZXjxNCIynly4nCNhiGvTwV4IwjaUdC+09TsTIy4/fI=`. Native executables and weights
are reused exactly. Active service and unrelated Nix changes are preserved.

User activation remains required: usual `nrs`, then restart Neovim. Target
AGENTS.md reserves activation for the user. Do not call it active before that.
Normal and insert cursor pauses debounce 250 ms. Alt+l accepts. Off cancels all
work. Explicit prediction now works in normal mode while automatic mode stays on.
Status gives the specific block reason rather than a generic UI-busy message.

Normal-mode next edits can affect source before the cursor; the typing-prefix
filter still protects insert-mode input. All exact range, UTF-8, stale state,
syntax, completion ownership, focus, and explicit-acceptance checks remain.

Navigation dismissal is a neutral editor observation, not incorrectness or a
negative training label. Unseen cancellation and normal-mode editor changes
also remain neutral. The existing exporter excludes those from negative pairs.
Do not interpret any ordinary rejection as a universal code-correctness judgment.
No scalar rewards, automatic personalization, weight updates, or new models.

Actual normal-mode model/SQLite smoke: session
`07af538a-2194-4afd-862d-527b247a807a`, 22 events, two displays, one navigation
dismissal, one mapped acceptance. All scripted synthetic, no human review or GUI
claim. Eight anchors and one delta replay exactly, zero gaps or mismatches.
Navigation proposal `83377f18-663d-4cbb-9e4c-a175992f5c31` has no rejection event or
candidate preference pair. First smoke exposed the typing-prefix filter problem;
it was preserved and the policy deliberately corrected, without changing fixtures.

Tests: 831 Python, 76 full Lua locally and on target, new explicit/normal scripts,
existing automatic/shared-slot scripts, 59 Bun collector, 38 analysis, both
TypeScript checks, Ruff and mypy. Model quality remains experimental.

Rollback the two owned Nix files from
`/home/shlok/.local/state/tabcomplete-install-backups/normal-cursor-r1-20261002T045752Z`
and use normal user activation. Code rollback needs no SQLite restore.
