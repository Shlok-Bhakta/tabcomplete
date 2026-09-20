# tabcomplete-trajectory (Neovim collector)

LazyVim-compatible Lua plugin that records local editing trajectories
(anchors + exact edit deltas + cursor/nav + keys + prediction outcomes) and
ships them to the trajectory collector server over HTTP+JSON.

Installable as a LazyVim plugin via `dir`:

```lua
{
  dir = vim.fn.expand("~/Projects/tabcomplete/tools/trajectory_collector/nvim"),
  opts = { server_url = "http://crabcake:8787" },
  config = function(_, opts)
    require("tabcomplete_trajectory").setup(opts)
  end,
}
```

The plugin never auto-starts (`plugin/` only defines commands); collection
begins on `setup()`. It never blocks editing: all HTTP uses async
`vim.system()` + curl with short timeouts, and failures fall back to an
on-disk spool.

## Protocol

- `protocol_version = 1`.
- Event envelope: `{protocol_version, event_id, session_id,
  sequence_number, timestamp_ms, event_type, file_id, cursor {row, col},
  mode, payload}`. `cursor` is **ZERO-based** (`nvim_win_get_cursor`
  row minus one).
- Event types: `session_start`, `session_end`, `repo_snapshot`,
  `buffer_open`, `buffer_close`, `buffer_enter`, `buffer_leave`,
  `buffer_write`, `edit_delta`, `cursor_move`, `file_jump`, `mode_change`,
  `key`, `heartbeat`, `prediction_requested`, `prediction_shown`,
  `prediction_accepted`, `prediction_partially_accepted`,
  `prediction_rejected`.
- `edit_delta` payload: `{start_row, old_end_row, new_end_row,
  deleted_text, inserted_text, cursor_after, changedtick}` — all rows
  ZERO-based, end-exclusive, matching `nvim_buf_attach` `on_lines` indices
  directly (see replay notes below).
- Endpoints (no auth headers): `POST /v1/session/start|end`,
  `POST /v1/events/batch`, `POST /v1/blobs/check|upload`,
  `POST /v1/repository/snapshot`.

## Config

```lua
require("tabcomplete_trajectory").setup({
  enabled = true,                                            -- false: commands only, no collection
  server_url = "http://crabcake:8787",                       -- default: $TABCOMPLETE_COLLECTOR_URL
  capture_keys = true,                                       -- vim.on_key raw-key events
  capture_repo_context = true,                               -- git metadata + repo snapshot
  max_repo_file_bytes = 1048576,                             -- 1 MiB; metadata only above
  cursor_debounce_ms = 50,
  batch_max_events = 100,                                    -- flush at 100 events
  batch_interval_ms = 1500,                                  -- or every 1500 ms
  spool_max_bytes = 268435456,                               -- 256 MiB offline spool cap
  periodic_anchor_every = 50,                                -- full anchor every N deltas/buffer
  request_timeout_ms = 5000,
  spool_retry_interval_ms = 30000,
})
```

A missing URL warns once via `vim.notify` and never crashes; events keep
queueing/spooling locally until a URL is configured.

Per-buffer opt-out: `vim.b.tabcomplete_trajectory_optout = true` (or
`vim.b.tabcomplete_exclude = true`).

## Commands

| Command | Alias | Action |
|---|---|---|
| `:TabCompleteCollectorStatus` | — | active/paused, session, URL, repo, file, queue/spool counts+size, last upload ok/error |
| `:TabCompleteCollectorFlush` | `:Flush` | flush the in-memory queue now |
| `:TabCompleteCollectorRescan` | `:RescanRepo` | re-detect repo, re-snapshot, re-anchor open buffers |
| `:TabCompleteCollectorPause` | `:Pause` | stop emitting (timers idle) |
| `:TabCompleteCollectorResume` | `:Resume` | resume emitting |

## Prediction logging (no reward computation)

```lua
local tc = require("tabcomplete_trajectory")
tc.log_prediction_requested({ prediction_id = "p1", provider = "x", model = "m",
  requested_at_ms = 1, max_output_tokens = 64, temperature = 0.2 })
tc.log_prediction_shown({ prediction_id = "p1", proposed_start = { row = 0, col = 0 },
  proposed_end = { row = 0, col = 5 }, proposed_text = "hello" })
tc.log_prediction_accepted({ prediction_id = "p1", accepted_chars = 5, accepted_lines = 1 })
tc.log_prediction_partially_accepted({ prediction_id = "p1", accepted_chars = 2, total_chars = 5 })
tc.log_prediction_rejected({ prediction_id = "p1", finish_reason = "dismissed" })
```

Accepted fields: `prediction_id, provider, model, model_revision,
requested_at_ms, responded_at_ms, latency_ms, context_ref, context_hash,
file, cursor, proposed_start, proposed_end, proposed_text,
max_output_tokens, temperature, confidence, logprob, finish_reason,
accepted_chars, accepted_lines, total_chars` (plus passthrough of any extra
keys). `file`/`cursor` default to the current buffer/cursor.

## Spool behaviour

- Spool dir: `~/.local/state/tabcomplete-trajectory/spool/`
  (`$XDG_STATE_HOME` respected). Machine id:
  `~/.local/state/tabcomplete-trajectory/machine-id` (generated once).
- Failed batches (including session start/end) are written as
  `batch-<ms>-<rand>.json`. Retry happens on startup, after every
  successful flush, and every `spool_retry_interval_ms`; oldest first,
  stopping at the first failure to preserve order.
- A spool file is deleted **only** on HTTP 2xx ACK. Corrupt files are
  dropped (they can never be delivered).
- Cap (`spool_max_bytes`, default 256 MiB): when exceeded, oldest batches
  are deleted until under cap — newest data is kept. A visible warning is
  shown; the collector never crashes.
- `VimLeavePre` sends `POST /v1/session/end` with a ≤2 s blocking attempt
  (exit path only, never the editing hot path); on failure the
  `session_end` envelope is spooled. All session ends carry `ended_at_ms`,
  so a missing `ended_at` is tolerated server-side.

## Replay-relevant design decisions

- **Indexing**: rows ZERO-based, ranges end-exclusive — the exact
  `(firstline, lastline, new_lastline)` triple from `nvim_buf_attach`.
- **Delta semantics**: `deleted_text` is `"\n".join` of shadow lines
  `[start_row, old_end_row)`; `inserted_text` is `"\n".join` of the fresh
  buffer lines `[start_row, new_end_row)`. The shadow is updated by the
  same splice, so consecutive deltas chain without re-reading the file
  (never a whole-file snapshot per keystroke).
- **Anchors**: `buffer_open`/`buffer_write` events carry `content_hash`
  (sha256 of canonical bytes) after `blobs/check|upload`; the stream is
  anchor, delta\*, anchor (periodic every `periodic_anchor_every` deltas,
  plus open/session-start/write/rescan anchors).
- **Byte reconstruction**: `"\n".join(lines)` plus one trailing newline
  when `eol` is set (`\r\n` when `fileformat == "dos"`; both recorded in
  the anchor payload). One ambiguity is inherent to Neovim's line model:
  `""` and `"\n"` on disk both read as a single empty line; we
  canonicalise that case to `""`.
- **`file_id`**: `repo:<root_name>:<relpath>` inside a git repo (stable
  across machines), else `file:<abspath>`.
- **Idempotency**: every envelope carries a unique `event_id`; servers
  should dedupe on it (session-end may arrive twice after a crash).
- **Secrets**: remote URLs are sanitised (userinfo stripped) before
  storage; secret paths, terminal/prompt/special buffers are never
  captured.

## Troubleshooting

- `server_url is not set ... spooled locally`: set `server_url` in setup
  or export `TABCOMPLETE_COLLECTOR_URL=http://crabcake:8787`.
- Status shows growing `spool_files`: server unreachable; check
  `last_err` in `:TabCompleteCollectorStatus`, verify
  `curl http://crabcake:8787/` from this host.
- No events for a buffer: check `buftype` (`:set buftype?` — terminal,
  prompt, nofile, quickfix, help are excluded), secret-path match, or
  `b:tabcomplete_trajectory_optout`.
- Run the headless test suite: from this directory,
  `nvim --headless -l tests/run.lua` (exit 0 = green).
