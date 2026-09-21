-- tabcomplete_trajectory.buffers
-- Per-buffer shadow text + exact edit capture via nvim_buf_attach line callbacks.
--
-- Delta semantics (byte-exact replay for fileformat=unix, eol set):
--   on_lines gives (firstline, lastline, new_lastline), all ZERO-based,
--   end-exclusive, directly matching nvim_buf_attach indices.
--   deleted_text = "\n".join(shadow[firstline+1 .. lastline])
--   inserted_text = "\n".join(new buffer lines [firstline, new_lastline))
--   shadow splice: remove (lastline-firstline) lines at firstline+1,
--   insert fresh lines. Emitted payload:
--   {start_row, old_end_row, new_end_row, deleted_text, inserted_text,
--    cursor_after, changedtick}
--   Replayer keeps a line array; for each edit_delta: splice; anchor events
--   (buffer_open/buffer_write with content_hash) reset the array. Byte
--   reconstruction: join(lines, "\n") .. (eol and "\n" or "").
local M = {}

M.shadows = {} -- bufnr -> {lines: string[], delta_count: integer, file: string}
M.attached = {} -- bufnr -> true
M.on_emit = nil -- function(envelope_fields) set by init
M.on_anchor = nil -- function(bufnr, reason) set by init

local function snapshot_lines(bufnr)
  local ok, lines = pcall(vim.api.nvim_buf_get_lines, bufnr, 0, -1, false)
  if ok and type(lines) == "table" then
    return lines
  end
  return {}
end

function M.get_shadow(bufnr)
  return M.shadows[bufnr]
end

function M.set_callbacks(emit, anchor)
  M.on_emit = emit
  M.on_anchor = anchor
end

function M.is_tracked(bufnr)
  return M.attached[bufnr] == true
end

--- Compute delta fields from shadow + fresh lines. Pure helper (unit-testable).
---@param shadow_lines string[] pre-change line array (1-based lua table)
---@param firstline integer zero-based start (inclusive)
---@param lastline integer zero-based old end (exclusive)
---@param new_lines string[] replacement lines
function M.compute_delta(shadow_lines, firstline, lastline, new_lines)
  local deleted = {}
  for i = firstline + 1, lastline do
    deleted[#deleted + 1] = shadow_lines[i] or ""
  end
  return {
    start_row = firstline,
    old_end_row = lastline,
    new_end_row = firstline + #new_lines,
    deleted_text = table.concat(deleted, "\n"),
    inserted_text = table.concat(new_lines or {}, "\n"),
  }
end

--- Splice a line array in place. Pure helper (unit-testable).
function M.splice_lines(shadow_lines, firstline, lastline, new_lines)
  local out = {}
  for i = 1, firstline do
    out[#out + 1] = shadow_lines[i]
  end
  for _, l in ipairs(new_lines or {}) do
    out[#out + 1] = l
  end
  for i = lastline + 1, #shadow_lines do
    out[#out + 1] = shadow_lines[i]
  end
  return out
end

--- Expected live line count after applying a delta. Pure helper (unit-testable).
---@param old_total integer shadow line count before the change
---@param firstline integer zero-based start (inclusive)
---@param lastline integer zero-based old end (exclusive)
---@param new_count integer number of replacement lines
function M.expected_total(old_total, firstline, lastline, new_count)
  return old_total - (lastline - firstline) + new_count
end

-- nvim_buf_attach on_lines signature:
--   on_lines(event, bufnr, changedtick, firstline, lastline, new_lastline, bytecount)
-- where event == "lines" and the row range is ZERO-based, end-exclusive.
local function on_lines_handler(event, bufnr, changedtick, firstline, lastline, new_lastline, bytecount)
  if event ~= "lines" then
    return
  end
  local util = require("tabcomplete_trajectory.util")
  if util.is_excluded_buffer(bufnr) then
    return
  end
  local shadow = M.shadows[bufnr]
  if not shadow then
    -- Not initialised (e.g. attached late): take a fresh snapshot, no delta.
    M.shadows[bufnr] = { lines = snapshot_lines(bufnr), delta_count = 0, file = vim.api.nvim_buf_get_name(bufnr) }
    return
  end
  local ok, fresh = pcall(vim.api.nvim_buf_get_lines, bufnr, firstline, new_lastline, false)
  if not ok then
    return
  end
  -- Divergence guard: the shadow supplies deleted_text, so any disagreement
  -- with the live buffer poisons every later row number (two production
  -- cases: a zero-text deletion that never happened, and a whitespace retype
  -- whose row content the buffer no longer had — both single-segment with
  -- consecutive changedticks, prime suspects being formatter/autopairs edits
  -- interleaving with the callback). The live line count must always agree
  -- with the delta's line-count effect; on mismatch, resync the shadow from
  -- the buffer and skip this delta instead of poisoning the shadow. (There
  -- is deliberately no deleted-text cross-check: post-change, the live
  -- buffer no longer contains the deleted lines by definition — the count is
  -- the only live oracle. Count-neutral divergence remains possible but is
  -- bounded by anchors and detectable offline via replay.) Skipping one
  -- delta loses one keystroke of evidence; emitting against a diverged
  -- shadow corrupts the whole file history.
  local live_total_ok, live_total = pcall(vim.api.nvim_buf_line_count, bufnr)
  if live_total_ok and live_total ~= M.expected_total(#shadow.lines, firstline, lastline, #fresh) then
    shadow.lines = snapshot_lines(bufnr)
    shadow.resync_count = (shadow.resync_count or 0) + 1
    require("tabcomplete_trajectory.config").warn_once(
      "phantom-delta-resync",
      "tabcomplete-trajectory: skipped a divergent buffer delta and resynced shadow (count="
        .. tostring(shadow.resync_count) .. ")"
    )
    return
  end
  local delta = M.compute_delta(shadow.lines, firstline, lastline, fresh)
  delta.changedtick = changedtick
  delta.bytecount = bytecount
  delta.cursor_after = util.cursor_zero()
  shadow.lines = M.splice_lines(shadow.lines, firstline, lastline, fresh)
  shadow.delta_count = (shadow.delta_count or 0) + 1
  if M.on_emit then
    local ok_e, err_e = pcall(M.on_emit, bufnr, "edit_delta", delta)
    if not ok_e then
      require("tabcomplete_trajectory.config").warn_once(
        "emit-edit-delta",
        "edit_delta emit failed: " .. tostring(err_e)
      )
    end
  end
  -- Periodic anchor every N deltas: full-content anchor so replay stays bounded.
  local cfg = require("tabcomplete_trajectory.config").options
  local every = cfg.periodic_anchor_every or 50
  if every > 0 and shadow.delta_count % every == 0 and M.on_anchor then
    pcall(M.on_anchor, bufnr, "periodic_anchor")
  end
end

--- Attach to a buffer and initialise its shadow. Idempotent.
---@return boolean attached
function M.attach(bufnr)
  if M.attached[bufnr] then
    return true
  end
  if not vim.api.nvim_buf_is_valid(bufnr) then
    return false
  end
  local util = require("tabcomplete_trajectory.util")
  if util.is_excluded_buffer(bufnr) then
    return false
  end
  if not vim.api.nvim_buf_is_loaded(bufnr) then
    return false
  end
  M.shadows[bufnr] = {
    lines = snapshot_lines(bufnr),
    delta_count = 0,
    file = vim.api.nvim_buf_get_name(bufnr),
  }
  local ok = pcall(vim.api.nvim_buf_attach, bufnr, false, {
    on_lines = on_lines_handler,
    on_detach = function(_, b)
      M.attached[b] = nil
      -- keep shadow for potential re-attach comparison; drop on wipeout via autocmd
    end,
  })
  if ok then
    M.attached[bufnr] = true
    return true
  end
  M.shadows[bufnr] = nil
  return false
end

function M.detach(bufnr)
  M.attached[bufnr] = nil
  M.shadows[bufnr] = nil
end

--- Refresh shadow from current buffer content (used after anchor/rescan).
function M.refresh_shadow(bufnr)
  if vim.api.nvim_buf_is_valid(bufnr) and vim.api.nvim_buf_is_loaded(bufnr) then
    local s = M.shadows[bufnr] or { delta_count = 0 }
    s.lines = snapshot_lines(bufnr)
    s.file = vim.api.nvim_buf_get_name(bufnr)
    M.shadows[bufnr] = s
  end
end

--- Canonical buffer bytes for hashing/upload (handles trailing newline + dos).
---@return content string, meta table {eol, fileformat, line_count}
function M.canonical_bytes(bufnr)
  local lines = snapshot_lines(bufnr)
  local eol = true
  local ff = "unix"
  if vim.api.nvim_buf_is_valid(bufnr) then
    local ok1, v = pcall(vim.api.nvim_get_option_value, "eol", { buf = bufnr })
    if ok1 then
      eol = v
    end
    local ok2, f = pcall(vim.api.nvim_get_option_value, "fileformat", { buf = bufnr })
    if ok2 and f then
      ff = f
    end
  end
  local sep = (ff == "dos") and "\r\n" or "\n"
  -- NOTE: nvim's line model cannot distinguish "" from "\n" on disk (both
  -- read as a single empty line). We canonicalise a single empty line to ""
  -- so empty files round-trip byte-exact; a lone-newline file replays as
  -- empty. All other files (the replay-relevant case) are byte-exact.
  local content
  if #lines == 0 or (#lines == 1 and lines[1] == "") then
    content = ""
  else
    content = table.concat(lines, sep)
    if eol then
      content = content .. sep
    end
  end
  return content, { eol = eol, fileformat = ff, line_count = #lines }
end

return M
