-- tabcomplete_trajectory.spool
-- Offline spool dir + persisted machine id. All writes are plain JSON files;
-- a batch file is deleted only after a 2xx ACK (caller responsibility).
local M = {}

local function state_dir()
  local base = os.getenv("XDG_STATE_HOME")
  if not base or base == "" then
    base = vim.fn.expand("~/.local/state")
  end
  return base .. "/tabcomplete-trajectory"
end

function M.state_dir()
  return state_dir()
end

function M.spool_dir()
  return state_dir() .. "/spool"
end

function M.machine_id_path()
  return state_dir() .. "/machine-id"
end

function M.ensure_dirs()
  vim.fn.mkdir(state_dir(), "p")
  vim.fn.mkdir(M.spool_dir(), "p")
end

--- Load or generate-once the machine id. Never throws; returns string.
function M.machine_id()
  M.ensure_dirs()
  local path = M.machine_id_path()
  local f = io.open(path, "r")
  if f then
    local id = (f:read("*l") or ""):gsub("%s+", "")
    f:close()
    if id ~= "" then
      return id
    end
  end
  local id = require("tabcomplete_trajectory.util").uuid()
  local w, err = io.open(path, "w")
  if w then
    w:write(id .. "\n")
    w:close()
  else
    require("tabcomplete_trajectory.config").warn_once(
      "machine-id-write",
      "cannot persist machine-id (" .. tostring(err) .. "); using ephemeral id"
    )
  end
  return id
end

--- Write one failed batch (list of envelopes + meta) as a spool file.
---@param batch table list of event envelopes
---@param meta table|nil extra info (e.g. {kind="events"})
---@return string|nil path written
function M.write_batch(batch, meta)
  M.ensure_dirs()
  local stamp = tostring(require("tabcomplete_trajectory.util").now_ms())
  local rnd = tostring(math.random(100000, 999999))
  local path = M.spool_dir() .. "/batch-" .. stamp .. "-" .. rnd .. ".json"
  local payload = { version = 1, meta = meta or { kind = "events" }, events = batch }
  local ok, encoded = pcall(vim.json.encode, payload)
  if not ok then
    return nil
  end
  local f = io.open(path, "w")
  if not f then
    return nil
  end
  f:write(encoded)
  f:close()
  return path
end

--- List spool files sorted oldest-first.
function M.list()
  local dir = M.spool_dir()
  local entries = vim.fn.globpath(dir, "batch-*.json", false, true)
  table.sort(entries)
  return entries
end

--- Read and decode one spool file. Returns nil + err on failure.
function M.read(path)
  local f = io.open(path, "r")
  if not f then
    return nil, "open failed"
  end
  local content = f:read("*a")
  f:close()
  local ok, decoded = pcall(vim.json.decode, content)
  if not ok then
    return nil, "json decode failed"
  end
  return decoded, nil
end

function M.remove(path)
  os.remove(path)
end

--- Total bytes currently spooled.
function M.total_bytes()
  local total = 0
  for _, p in ipairs(M.list()) do
    local st = vim.uv.fs_stat(p)
    if st then
      total = total + (st.size or 0)
    end
  end
  return total
end

--- Enforce the spool cap: keep newest, delete oldest until under cap.
--- Returns (deleted_count, total_bytes). Never throws, never crashes.
function M.enforce_cap(max_bytes)
  local files = M.list()
  local sizes = {}
  local total = 0
  for _, p in ipairs(files) do
    local st = vim.uv.fs_stat(p)
    local sz = st and (st.size or 0) or 0
    sizes[p] = sz
    total = total + sz
  end
  local deleted = 0
  local i = 1
  while total > max_bytes and i <= #files do
    local p = files[i]
    os.remove(p)
    total = total - (sizes[p] or 0)
    deleted = deleted + 1
    i = i + 1
  end
  if deleted > 0 then
    require("tabcomplete_trajectory.config").warn_once(
      "spool-cap-cycle",
      string.format(
        "spool exceeded cap; dropped %d oldest batch(es), kept newest (%d bytes remain)",
        deleted,
        total
      )
    )
    -- allow the warning to fire again on a future overflow cycle
    -- (reset after it has been shown once per cycle is handled by counting)
  end
  return deleted, total
end

return M
