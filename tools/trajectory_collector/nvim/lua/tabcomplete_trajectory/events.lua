-- tabcomplete_trajectory.events
-- Protocol envelope construction. Pure: no I/O, no vim timers.
-- protocol_version=1; envelope {protocol_version, event_id, session_id,
-- sequence_number, timestamp_ms, event_type, file_id, cursor {row,col}
-- ZERO-based, mode, payload}.
local M = {}

M.PROTOCOL_VERSION = 1

local VALID_TYPES = {
  session_start = true,
  session_end = true,
  repo_snapshot = true,
  buffer_open = true,
  buffer_close = true,
  buffer_enter = true,
  buffer_leave = true,
  buffer_write = true,
  edit_delta = true,
  cursor_move = true,
  file_jump = true,
  mode_change = true,
  key = true,
  heartbeat = true,
  prediction_requested = true,
  prediction_shown = true,
  prediction_accepted = true,
  prediction_partially_accepted = true,
  prediction_rejected = true,
}

function M.is_valid_type(t)
  return VALID_TYPES[t] == true
end

--- Build one envelope. Caller owns sequence_number increment.
---@param opts table {event_type, session_id, sequence_number, file_id, cursor, mode, payload, timestamp_ms, event_id}
function M.make(opts)
  assert(opts.event_type, "event_type required")
  assert(M.is_valid_type(opts.event_type), "unknown event_type: " .. tostring(opts.event_type))
  local util = require("tabcomplete_trajectory.util")
  return {
    protocol_version = M.PROTOCOL_VERSION,
    event_id = opts.event_id or util.uuid(),
    session_id = opts.session_id,
    sequence_number = opts.sequence_number,
    timestamp_ms = opts.timestamp_ms or util.now_ms(),
    event_type = opts.event_type,
    file_id = opts.file_id or "",
    cursor = opts.cursor or { row = 0, col = 0 },
    mode = opts.mode or "n",
    payload = opts.payload or {},
  }
end

return M
