# Normal-mode suggestions and neutral navigation feedback

This extends the existing automatic editor plugin on `prototype/rust-editor-format-r2`.
Model weights, native inference, existing SQLite schema, and training policy are unchanged.

## Interaction

`automatic_normal_mode=true`, exposed as Home Manager `automaticNormalMode`, adds the
same 250 ms debounce after normal-mode cursor movement, edits, buffer entry, and
return from insert mode. One request runs at a time, with only the latest state
eligible. Unchanged idle states and no-edit outputs do not repeat. The plugin's
accepted edit and resulting cursor event do not start another request.

Normal-mode proposals can change bytes before the cursor. The insert-mode typed
prefix filter remains in place while typing. Normal-mode display policy is
`normal-mode-next-edit-v1`; this change does not improve model accuracy.
Only complete, valid one-line actions are eligible. Buffer/range/hash/cursor
staleness, UTF-8 boundaries, excluded buffers, completion UI, snippets, focus,
resource backoff, and the Rust syntax guard remain enforced. Operator-pending,
visual, replace, terminal, and command-line modes do not trigger inference.

Alt+l accepts in normal or insert mode. Moving the cursor dismisses the preview.
Nothing applies, saves, executes, or trains automatically. `:TabCompleteMode off`
cancels pending work. `:TabCompletePredict` now makes an explicit request from
normal mode without changing the persisted automatic mode. Previously that
command incorrectly enforced the insert-mode automatic gate and reported the
generic "UI busy" message. A failing deterministic regression reproduced that
bug before the fix. Status now reports the specific `automatic_block_reason`.

## Feedback semantics

A shown proposal closed by navigation is `dismissed_navigation`, with
`outcome_source=editor_observation`. An unseen cancelled request is
`cancelled_unseen`. Neither means the model was incorrect or the user disliked
the suggestion. Ordinary normal-mode changes are `dismissed_editor_change`, not
implicit typing rejection. Correlation with an old insert-mode key cannot change
that classification. Raw proposals, state identities, and later deltas remain
available in the existing collector for analysis.

The exporter already excludes these observations from negative preference
pairs. Five new SQLite tests compare them against a same-state explicit-rejection
control. Only the explicitly rejected, reviewed proposal becomes a candidate
dispreferred action. No reward or correctness score is fabricated. Automatic
personalization remains disabled.

## Actual ThinkPad verification

The source-loaded updated plugin used the actual existing Qwen service and
collector on ThinkPad Neovim 0.12.3. It ran in actual normal mode without a mode
stub. Two actual model proposals displayed: cursor movement closed the first
as neutral navigation, the explicit command requested the second without
switching mode, and the normal-mode Alt+l mapping accepted it. Acceptance did not
create a third request. All interactions are scripted and synthetic, not human
feedback or general quality evidence. No GUI inspection is claimed.

Session: `07af538a-2194-4afd-862d-527b247a807a`.
Navigation proposal: `83377f18-663d-4cbb-9e4c-a175992f5c31`.
Accepted proposal: `0ad450a0-0ec3-46a8-989d-f3bc5e9a3a14`.
The production database contains 22 events for the session. Eight anchors and
one accepted-edit delta replay byte-exactly to the final 501 bytes, with no gaps
or mismatches. Navigation has no `prediction_rejected` event and no candidate
preference pair. SQLite integrity is okay. No migration or database reset is needed.

The first real smoke failed because the old typing-prefix filter suppressed a
normal-mode proposal. That failure is preserved. The policy was deliberately
changed for normal mode, versioned, and rerun. No fixture or model was changed
to rescue the result.

## Tests and activation

Repository Python: 831 passed, two deprecation warnings. Exporter tests: 14 passed.
Neovim full suite: 76 passed locally and on ThinkPad. Explicit-command,
normal-mode, existing automatic-mode, and shared-slot tests passed. Collector
and analysis Bun suites passed 59 and 38 tests; both TypeScript checks passed.
Ruff passed. Mypy checked 83 source files with no errors. Native Rust and weights
were not changed by this iteration.

The target configuration now enables `automaticNormalMode=true` declaratively.
The target's AGENTS.md reserves `nrs` for the user. This work does not activate
a generation or imperatively overwrite the installed plugin. Apply the updated
configuration with the usual `nrs`, then restart Neovim. The existing service and
current editor sessions remain running until that activation/restart. Backup,
build/evaluation, and final source-pin receipts live in the accompanying directory.

The tested code pin is `653fcbe8b2a8fe8f8f11c0e4b56be0b7b99b7494`. Its filtered
runtime source has NAR hash `sha256-mZXjxNCIynly4nCNhiGvTwV4IwjaUdC+09TsTIy4/fI=`.
The initial unfiltered fetch failed fixed-output reference checks because reports
contained actual Nix store paths. The installed source fetch removes only root
`reports` during `postFetch`, with the hash recomputed for that filtered tree.
The committed reports remain authoritative and preserved. The corrected actual
Nix package build passes, and full Home Manager evaluation has zero failed
assertions with normal mode enabled. Built-plugin explicit and normal-mode tests
also pass. Both executable payload store paths and hashes are exactly reused
from the previous generation; no extra model payload copies or weights were made.

Only the two previously owned Nix files changed, with unrelated-file hashes
unchanged. The active service unit is unchanged and its service remains active.
Configuration backup is
`/home/shlok/.local/state/tabcomplete-install-backups/normal-cursor-r1-20261002T045752Z`.
Restore its two owned files and perform the user's usual activation to roll back.
No database restore is required. A consistent database backup is recorded in
`database_backup.json`. The new editor generation still requires user activation
and Neovim restart; this report does not claim it is active in existing sessions.
