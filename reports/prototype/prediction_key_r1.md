# Explicit prediction and experimental syntax opt-out

Code pin `c24773fd68a3c20b0136dc64ff276bf2b2856103` on
`prototype/rust-editor-format-r2`.

Alt+p requests a prediction in normal or insert mode. Alt+l accepts a valid
preview. Existing occupied maps are preserved; Alt+Shift+p is the prediction
fallback and `:TabCompleteStatus` reports the actual binding. Explicit requests
report start and completion, including no change and invalid output. Automatic
requests remain quiet. Requests do not change the persisted editor mode.

`services.tabcomplete.syntaxValidation = false` disables the optional Rust
syntax filter. The prepared ThinkPad configuration sets it false as requested
for testing. This permits syntax-breaking source suggestions without changing
wire decoding, termination, range, UTF-8, stale state, or acceptance checks.
Runtime health, request records, generated records, and status expose the policy.
No weights, precision, training, or automatic feedback classification changed.

## Findings

SSH initially succeeded. The actual ThinkPad service was running embedded Qwen
Q4_K_M, SHA-256
`4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb`.
Its installed plugin had normal-mode automatic prediction enabled. Alt+p was
unoccupied in both normal and insert maps in the actual LazyVim configuration.

Actual collector session `c85bfa3e-2987-416d-84a9-c63f0dbd2b1c` contained model
requests withheld by `rust_syntax_regression`, plus unseen requests cancelled by
editing/navigation. Example request `c6b786f7-71a0-4e05-a3e7-5d6390fb25b7` completed
with a syntax rejection. The CLI request query returned `no_runs`; collector
records supplied the failure evidence. The generic client error previously hid
the syntax reason.

The ThinkPad subsequently disconnected. Tailscale reported `Online=false` and
bounded SSH attempts timed out. Its Nix files and active service were not
modified in this iteration. The prepared [configuration patch](prediction_key_r1/thinkpad-config.patch)
changes only the two owned Nix files. Its filtered source hash was computed by
NAR serialization and checked against the previous known source hash as a control.
Applying the patch to an exact disposable copy of the observed files passes.
Target Nix evaluation/build and activation remain unverified until reconnection.

## Verification

831 Python tests, 76 full Lua tests, 19 Rust tests, 59 Bun collector tests, and
38 Bun analysis tests pass. New standalone mapping tests cover normal and insert
callbacks, no edit, malformed output, focus, off, occupied keys, and reconfiguration.
The normal automatic regression script also passes. Ruff, mypy on 84 source
files, both TypeScript checks, and application release Clippy pass. Vendored
wrapper warnings remain. The first local native build needed the installed GCC
header directory supplied to bindgen; the corrected offline build passes.

The same selected model ran on crabcake in a temporary CPU diagnostic service
with syntax filtering disabled. This used its existing GGUF through file mmap,
not the ThinkPad's embedded executable. Headless Neovim displayed two actual
model proposals, invalidated one neutrally on navigation, then requested the
other through Alt+p and accepted through Alt+l. All decisions were synthetic,
not human feedback. The existing SQLite database recorded two projections, one
acceptance, one navigation dismissal, and no negative preference pair. Eight
anchors and one delta reconstruct exactly 505 bytes with no gaps or mismatches.
See [verification](prediction_key_r1/verification.json) and
[test receipts](prediction_key_r1/tests.json). An initial diagnostic used alias
`default`, failed the existing allowlist check, and was rerun with the correct
single-entry q25 registry. No model was substituted.

## Activation and rollback

When the ThinkPad reconnects, back up its two owned Nix files and active unit,
apply the patch, build/evaluate the resulting Home Manager package, and verify
the compiled service accepts `--syntax-validation false`. The target's AGENTS.md
reserves `nrs` for the user. After user activation, restart Neovim and verify
health reports `syntax_validation=false`, status reports `<M-p>`, and the running
unit uses the new arguments. Do not claim the old running service has changed.

Rollback restores those two backed-up Nix files through normal user activation.
No database restore is needed. Automatic training remains disabled.
