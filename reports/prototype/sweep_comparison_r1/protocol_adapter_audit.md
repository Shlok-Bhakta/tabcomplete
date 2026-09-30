# Sweep whole-file protocol adapter audit

**Scope:** Read-only inspection for a possible adapter, conditional on measured Sweep quality and target-device resource results. No source, model, service, or Neovim configuration was changed. No benchmark prediction outputs were read and no tests were run for this audit.

## Which source is installed

The active Neovim plugin spec is the regular file `/home/crabcake/.config/nvim/lua/plugins/tabcomplete-trajectory.lua` (SHA-256 `1259aa039251128965879309b891494aac4d2378948d16a7f53323a4a01b0dba`). Its LazyVim `dir` points directly to:

`/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/nvim`

That source directory is a normal directory, not a symlink; `readlink -f` returns the same path. No separate `tabcomplete` or `trajectory` plugin directory appeared under `~/.local/share/nvim/lazy`. This confirms the configured plugin source, not which modules an already-running Neovim process has loaded.

The installed plugin worktree is branch `prototype/automatic-feedback-r1`, HEAD `9dc5b1cef6c922bb65a9e59d206dee578cbc3e2a`. The requested active worktree is `/home/crabcake/Projects/tabcomplete-product-r2`, branch `prototype/product-r2`, HEAD `a681d5c4ba88388742b4217a838d840ed5c257a5`. The user's main checkout is a third worktree, branch `stage1/code-cpt`, HEAD `fc0c2db6330f53fd3da893a85d60cbc011641471`; it does not contain the predictor or SSE modules inspected here.

| Inspected path | SHA-256 |
|---|---|
| Installed config: `/home/crabcake/.config/nvim/lua/plugins/tabcomplete-trajectory.lua` | `1259aa039251128965879309b891494aac4d2378948d16a7f53323a4a01b0dba` |
| Installed `/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/predict.lua` | `794bd912114b8cdd8dd34ee9f6498e00a03659056974d3a6227ba91bea792df1` |
| Installed `/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/sse.lua` | `f747ea7ae39a8d658d4398b7593c879fa344c2e66e11c24e444012952b8e0dd2` |
| Installed `/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/buffers.lua` | `90dec3af78ce3b5f50b9502a7e84112507a7f509564e2de3754e24b9dc0d8577` |
| Installed `/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/nvim/tests/predict.lua` | `3eff0e65d652a6b07951d8ea9ecfe7069113776a0bb8b5e1ab6ad0b37f97adcb` |
| Installed `/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/nvim/tests/automatic_predict.lua` | `78c84f27cf24ea06200b0cf822432b4fac24e24b45578905458f77e662ae72d4` |
| Product-r2 `/home/crabcake/Projects/tabcomplete-product-r2/tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/predict.lua` | `e93cf25dda86cd21df990fd241cef243f0df13ac0c0fc6f6fd76bc14480b953d` |
| Product-r2 `/home/crabcake/Projects/tabcomplete-product-r2/tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/sse.lua` | `f747ea7ae39a8d658d4398b7593c879fa344c2e66e11c24e444012952b8e0dd2` |
| Product-r2 `/home/crabcake/Projects/tabcomplete-product-r2/tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/buffers.lua` | `90dec3af78ce3b5f50b9502a7e84112507a7f509564e2de3754e24b9dc0d8577` |
| Product-r2 `/home/crabcake/Projects/tabcomplete-product-r2/tools/trajectory_collector/nvim/tests/predict.lua` | `3eff0e65d652a6b07951d8ea9ecfe7069113776a0bb8b5e1ab6ad0b37f97adcb` |
| Product-r2 `/home/crabcake/Projects/tabcomplete-product-r2/tools/trajectory_collector/nvim/tests/automatic_predict.lua` | `44da5e16ee03ca0e9ef025c036f9bbb11cd725055418e5dcabc600e27117e9e5` |
| Installed `/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/scripts/install_small_model_lazyvim.py` | `1cbed3c3dde75e80b19597f3ed475969574420b95e7368b2a4173f567fb3c9e9` |
| Product-r2 `/home/crabcake/Projects/tabcomplete-product-r2/scripts/install_small_model_lazyvim.py` | `1cbed3c3dde75e80b19597f3ed475969574420b95e7368b2a4173f567fb3c9e9` |
| Product-r2 `/home/crabcake/Projects/tabcomplete-product-r2/scripts/run_sweep_comparison.py` (prompt builder) | `2a896dc411928ad0392444232999f7eeb24da3ba7f6dbb680750dab8ed4b85e0` |
| Main checkout `/home/crabcake/Projects/tabcomplete/tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/buffers.lua` | `1c3de5ef8ea964fe65022703aa8f4c854f091baa7d12fa9cc854f9b23c54c673` |

The installed and product-r2 SSE and buffer modules, installer, and base prediction test are byte-identical. Their predictors differ only in explicit-decision review bookkeeping/API: product-r2 removes `review_last()` and its confirmation callback; its automatic-prediction test also removes one assertion about synthetic human review. The rest of the relevant prediction flow inspected here matches. The main checkout's buffer module is different and it lacks the predictor/SSE/installer, so it is not the installed baseline.

The installed config currently selects `q25-coder`, Q4_K_M, compact protocol, and automatic experimental mode. The existing installer is explicitly tied to the Q25 artifact hash and 397,807,232-byte size; it cannot install Sweep as-is.

## Existing flow and smallest adapter

In the installed predictor, `buffer_state()` at lines 141–209 uses the cursor-to-end-of-current-line suffix as its editable region, builds the compact `N\n` / `R\n...` prompt, and records the pre-state hash, file identity, cursor/range, prediction/request IDs, and event sequence. `still_current()` at lines 211–220 gates both display and acceptance. `finished()` at lines 378–445 checks request identity, cancellation, transport status, and current state before parsing. `show()` at lines 315–368 hides no-edit results, suppresses multiline automatic proposals, previews manual multiline edits, and draws a non-mutating inline preview for one-line changes. `accept()` at lines 549 onward performs the existing explicit buffer edit after another freshness check.

The SSE parser at installed `sse.lua` lines 4–62 handles chunk boundaries, malformed records, response-size bounds, and terminal events. Its current `finish()` only accepts EOS and only parses the compact action wire format. The request path in `predict.lua` lines 446–498 caps output at 96 tokens and reserves that same cap against the 2,304-token server context. The pinned Sweep prompt builder in product-r2 `scripts/run_sweep_comparison.py` lines 177–208 orders context files, recent diffs, original file, current file, then an updated-file header; it is a different protocol.

If Sweep quality and resource measurements justify an editor experiment, add a **versioned Sweep decoder** between completed SSE and the existing `show()` path. Keep the compact decoder for Q25; select protocol from explicit model/config identity, not by guessing from response text. Build Sweep’s exact pinned prompt format and tokenizer budget. Record a new prompt-policy and wire version. Preserve the current scheduling, request IDs, collector, stale checks, UI, and explicit acceptance path.

For decoding, treat the returned file as bytes. Derive the editable span from the serialized current file, then require the candidate’s fixed-length prefix and suffix outside that span to match exactly. Extract only the intervening bytes as the replacement. If the candidate is byte-identical to the current file—or the extracted replacement equals the original region—return canonical no-edit. Reject any outside-region change, invalid encoding/boundary, line-ending mismatch, malformed response, incomplete terminal, cap, or cancellation. Do not trim, strip a leading newline, remove fences, repair syntax, use a global diff, or turn a partial output into an edit. Keep automatic multiline output suppressed; manual preview may use the existing path only after exact range mapping succeeds.

A byte-offset implementation must not reuse `region_start` as-is: it is computed over LF-joined lines, while `buffers.canonical_bytes()` uses the buffer’s `fileformat` and final-newline option. That function also documents that empty and lone-newline files collapse to the same canonical empty bytes. Reject that ambiguous state for whole-file mapping unless a later design preserves the distinction explicitly.

For context history, include only edits reconstructable from collector deltas and validated byte-for-byte against the current state. Otherwise use an explicitly empty-history prompt case or skip; do not infer a previous full file from a diff description. The prompt builder must preserve the pinned field order and no-trailing-newline behavior. Do not normalize model output to compensate for prompt separators.

Keep collector identities and lifecycle events. Record `prediction_requested` with Sweep model/revision/hash, precision, runtime hash, context policy, file/range/pre-state identities, and `sweep-whole-file-v1`. Existing `prediction_generated`, shown/dismissed/accepted events, blobs, and event links can remain. The current `raw_response_hash` hashes the serialized SSE chunks; preserve that meaning or version it, and separately retain/hash extracted model text and canonical replacement action so payload retrieval is unambiguous. Use the existing content-addressed blob store; do not copy full files into outcome rows. Keep `human_verified=false` absent an explicit review event.

### Minimal test plan after quality gate

- Pure Lua mapping tests with synthetic buffers: exact no-edit, one-line insert/replace/delete, multibyte UTF-8, LF/CRLF, final newline, empty/lone-newline ambiguity, outside-region changes, malformed/incomplete/capped/cancelled outputs, and exact terminal stop.
- Golden prompt test matching Lua serving construction to product-r2's pinned Python builder.
- Headless Neovim test through existing `_request_impl` seam: valid single-line suggestion displays and explicit acceptance applies once; no-edit stays hidden; outside-range/stale/invalid outputs never display or apply; collector events retain the new protocol/model/context identities; undo and existing mappings remain unchanged.

These are protocol/integration tests, not evidence of Sweep quality. Do not enable or deploy this adapter unless the frozen quality evaluation and measured target resource results support it.
