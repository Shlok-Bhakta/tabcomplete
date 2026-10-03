# Prediction key handoff

Code pin `c24773fd68a3c20b0136dc64ff276bf2b2856103`, branch
`prototype/rust-editor-format-r2`. Alt+p triggers explicit prediction in both
normal and insert mode. Alt+l accepts. Explicit completion reports no edit and
invalid output rather than silently leaving no preview.

User explicitly wants unpolished model output now. Disable optional syntax
filtering with `services.tabcomplete.syntaxValidation = false`; keep exact range,
staleness, UTF-8, terminal/wire validity, and explicit acceptance checks. The
prepared `thinkpad-config.patch` also pins the new source and declares predictKey.
Source hash `sha256-7QZsKJoE9k+Jy1i0bh3kdCFu/jqzmdoLk972CEAFT+M=` excludes reports
through the existing postFetch hook. Control serialization exactly reproduced
the previous known filtered-source hash.

Historical first attempt: ThinkPad was accessible at initial inspection, then went offline on Tailscale.
Bounded SSH retries timed out. No remote configuration, service, or activation
was changed. Do not call Alt+p or syntax opt-out installed there yet. On reconnect,
back up owned files/unit, check and apply the prepared patch to ~/nixos-config,
build/evaluate without activation, and preserve unrelated dirty work. Target
AGENTS.md reserves nrs for the user. Weights and precision remain unchanged;
only engine policy and matching plugin change. Native rebuild uses pinned offline
dependencies and existing local weights, with no model download.

Local crabcake actual-model/SQLite smoke passed, external-file-mmap diagnostic
backend, syntax_validation=false. Session b17e39f3-864c-4dbf-b2bb-dd4a0fbcdafd.
Explicit Alt+p proposal 23c59eb9-4cb0-4440-a669-eb1586366722 displayed and accepted
once through Alt+l. Eight anchors plus one delta replay to 505 bytes exactly.
All decisions synthetic. Existing database and projection preserved. The temporary
local diagnostic service was stopped after verification.

Tests: 831 Python, 76 Lua, 19 Rust, 59 collector, 38 analysis; new mapping script,
normal automatic script, Ruff, mypy84, both TS checks, application Clippy.
Target build/activation and visual inspection remain pending, not passing claims.

## Reconnected and ready for activation

The user connected Wi-Fi. We backed up and patched the two owned Nix files,
formatted them, verified unrelated files unchanged, and built the package above.
All 124 Home Manager assertions pass. Built plugin mapping test and full Lua76
pass on target, and Nix's Rust19 check passes. Exact embedded Qwen hash verified.
See thinkpad_installation.json and thinkpad_verification.json. The selected new
binary digest is 80e17c659f4099a68603766c855f116ace5b906bc95df3633c4c8fc0f87d5fbb.

Target actual-model smoke session c1ab1d20-2db5-49c3-804e-911993e2fcf2: two
requests/displays, one explicit Alt+p, one Alt+l acceptance, neutral navigation.
Old active engine plus new plugin, synthetic decisions, exact SQLite replay.

Activation has NOT occurred. The target AGENTS.md requires the user to run nrs.
Afterward reload/restart the user tabcomplete-engine service and restart Neovim;
verify health syntax_validation=false, actual running binary and arguments,
and loaded plugin path before calling this active. Current old unit and plugin
symlink were preserved. No additional weight download or training.
