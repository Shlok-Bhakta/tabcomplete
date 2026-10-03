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

ThinkPad was accessible at initial inspection, then went offline on Tailscale.
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
