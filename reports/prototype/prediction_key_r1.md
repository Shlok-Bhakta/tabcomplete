# Explicit prediction and experimental syntax opt-out

Latest status: the ThinkPad reconnected. Its declarative patch is applied,
formatted, built, and verified. Activation is reserved for the user. See
[ThinkPad installation](prediction_key_r1/thinkpad_installation.json).

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

## ThinkPad reconnection

After Wi-Fi returned, SSH succeeded. The two owned Nix files and active service
unit were backed up, the existing SQLite database received a consistent backup,
and the patch was applied and formatted with nixfmt. Unrelated working files
remained byte-identical. Model payload hashing verifies the unchanged Qwen GGUF.

The actual Home Manager configuration passes all 124 assertions. Nix built the
new native engine and both embedded executable wrappers using the existing
local model files. The selected package is
`/nix/store/sn2q9gnhsv9lsg55s7dn1v5y3v2la9d7-tabcomplete-qwen-0.1.0`. The selected
executable is 500,579,298 bytes, SHA-256
`80e17c659f4099a68603766c855f116ace5b906bc95df3633c4c8fc0f87d5fbb`. Its embedded
model digest remains `4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb`.
The built plugin passes the new prediction mapping tests; the full source Lua
suite passes 76 tests on the ThinkPad. The Nix native check passes 19 Rust tests.

A real selected-model headless smoke on the ThinkPad displayed two proposals,
requested one with Alt+p, and accepted one with Alt+l. It uses the new plugin
against the existing active engine; the new engine's syntax opt-out is not yet
active. All decisions are scripted synthetic. Session
`c1ab1d20-2db5-49c3-804e-911993e2fcf2` stores exact linked request/display/action
payloads in the existing collector. Eight anchors plus one delta reconstruct
505 bytes exactly, with no mismatches or gaps. Navigation stays neutral and no
rejection event is fabricated. Payload hashes match. SigNoz's request query
returns no_runs; collector evidence remains authoritative for this smoke.

The active unit and user plugin symlink were preserved. User activation remains
required: run nrs, reload the user systemd manager and restart tabcomplete-engine,
then restart Neovim. The expected service arguments contain
`--syntax-validation false`. Do not describe that flag as active before checking
the running health endpoint after activation.

A preparation script initially substituted its fixture marker inside an
environment variable name; it failed before executing and was corrected. A
read-only artifact verifier initially assumed the wrong footer size, then used
the format's actual magic length and passed. The fixtures and model were
unchanged.
