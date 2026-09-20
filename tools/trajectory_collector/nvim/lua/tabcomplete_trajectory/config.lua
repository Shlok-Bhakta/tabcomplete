-- tabcomplete_trajectory.config
-- Default options, setup() merge, warn-once helper.
local M = {}

M.defaults = {
  enabled = true,
  server_url = nil, -- defaults to $TABCOMPLETE_COLLECTOR_URL at setup time
  capture_keys = true,
  capture_repo_context = true,
  max_repo_file_bytes = 1048576, -- 1 MiB content cap; metadata only above
  cursor_debounce_ms = 50,
  batch_max_events = 100,
  batch_interval_ms = 1500,
  spool_max_bytes = 268435456, -- 256 MiB
  periodic_anchor_every = 50, -- full anchor every N edit deltas per buffer
  request_timeout_ms = 5000, -- short timeouts; never block editing
  spool_retry_interval_ms = 30000,
}

M.options = vim.tbl_deep_extend("force", {}, M.defaults)

local warned = {}

--- Warn once per key; never throws.
function M.warn_once(key, msg)
  if warned[key] then
    return
  end
  warned[key] = true
  vim.schedule(function()
    vim.notify("[tabcomplete-trajectory] " .. msg, vim.log.levels.WARN)
  end)
end

function M.reset_warnings()
  warned = {}
end

--- Merge user opts. Resolves server_url default from env. Never throws.
function M.setup(opts)
  M.options = vim.tbl_deep_extend("force", {}, M.defaults)
  if type(opts) == "table" then
    M.options = vim.tbl_deep_extend("force", M.options, opts)
  end
  if not M.options.server_url or M.options.server_url == "" then
    local env = os.getenv("TABCOMPLETE_COLLECTOR_URL")
    if env and env ~= "" then
      M.options.server_url = env
    end
  end
  return M.options
end

--- True when the collector may attempt network I/O.
function M.has_url()
  return M.options.server_url ~= nil and M.options.server_url ~= ""
end

return M
