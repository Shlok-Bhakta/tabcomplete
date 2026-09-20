-- tabcomplete_trajectory.util
-- Pure helpers: ids, time, hashing, paths, exclusion predicates.
local M = {}

--- Current wall-clock time in milliseconds since epoch.
function M.now_ms()
  local ok, sec, usec = pcall(vim.uv.gettimeofday)
  if ok and sec then
    return math.floor(sec * 1000 + (usec or 0) / 1000)
  end
  return math.floor(os.time() * 1000)
end

--- Generate a v4-style UUID hex string (8-4-4-4-12).
function M.uuid()
  -- Prefer kernel randomness when available (deterministic format, not crypto-critical).
  local f = io.open("/proc/sys/kernel/random/uuid", "r")
  if f then
    local u = f:read("*l")
    f:close()
    if u and u:match("^%x%x%x%x%x%x%x%x%-%x%x%x%x%-%x%x%x%x%-%x%x%x%x%-%x%x%x%x%x%x%x%x%x%x%x%x$") then
      return u:lower()
    end
  end
  math.randomseed(math.random() + (vim.uv.hrtime() % 1000000))
  local function h(n)
    local s = {}
    for _ = 1, n do
      s[#s + 1] = string.format("%x", math.random(0, 15))
    end
    return table.concat(s)
  end
  return string.format("%s-%s-4%s-%s-%s", h(8), h(4), h(3), h(4), h(12))
end

--- sha256 hex of a string. Uses vim.fn.sha256 when available,
--- otherwise falls back to external sha256sum (async callers should avoid hot path).
function M.sha256hex(s)
  if vim.fn.exists("*sha256") == 1 then
    local ok, out = pcall(vim.fn.sha256, s)
    if ok and type(out) == "string" and #out == 64 then
      return out
    end
  end
  -- Fallback: write to temp file and hash with sha256sum (sync, anchor path only).
  local tmp = vim.fn.tempname()
  local f = io.open(tmp, "w")
  if not f then
    error("tabcomplete_trajectory: cannot hash without sha256 provider")
  end
  f:write(s)
  f:close()
  local handle = io.popen("sha256sum " .. vim.fn.shellescape(tmp) .. " 2>/dev/null")
  local out = handle and handle:read("*l") or nil
  if handle then
    handle:close()
  end
  os.remove(tmp)
  if out then
    local h = out:match("^(%x+)")
    if h and #h == 64 then
      return h
    end
  end
  error("tabcomplete_trajectory: sha256 unavailable")
end

--- Normalize an absolute path (resolve symlinks when possible, no fs traversal).
function M.abspath(path)
  if not path or path == "" then
    return ""
  end
  local ok, resolved = pcall(vim.uv.fs_realpath, path)
  if ok and resolved and resolved ~= "" then
    return resolved
  end
  return path
end

--- Compute a stable file_id for the protocol envelope.
--- Prefers repo-relative ids so replays are machine-independent.
---@param abspath string absolute file path
---@param repo_root string|nil git root or nil
---@param root_name string|nil repo root basename or nil
function M.file_id(abspath, repo_root, root_name)
  if repo_root and repo_root ~= "" and abspath:sub(1, #repo_root) == repo_root then
    local rel = abspath:sub(#repo_root + 2) -- strip root + sep
    if rel == "" then
      rel = "."
    end
    return "repo:" .. (root_name or vim.fn.fnamemodify(repo_root, ":t")) .. ":" .. rel
  end
  return "file:" .. abspath
end

--- Strip credentials from a git remote URL. Never logs or stores userinfo.
---@param url string
function M.sanitize_origin(url)
  if not url or url == "" then
    return ""
  end
  -- scp-like syntax: [user@]host:path
  local userhost, path = url:match("^([^@:/]+@)([^:]+:.+)$")
  if userhost and path then
    local host = path:match("^([^:]+):")
    -- userhost includes trailing @; drop userinfo, keep host:path
    local clean_host = url:gsub("^[^@]+@", "")
    return clean_host
  end
  -- scheme://[user[:pass]@]host/path
  local sanitized = url:gsub("^(%w+://)[^@/]*@", "%1")
  return sanitized
end

-- Basename patterns (matched against the file basename, case-sensitive).
local SECRET_BASENAMES = {
  "^%.env$",
  "^%.env%..+$",
  "%.pem$",
  "%.key$",
  "%.p12$",
  "%.pfx$",
  "%.crt$",
  "%.cer$",
  "^id_rsa$",
  "^id_ed25519$",
  "^credentials",
  "^secrets",
}
-- Path segment patterns (matched against any '/'-separated segment or full path).
local SECRET_SEGMENTS = {
  "/%.aws/",
  "/%.ssh/",
  "/%.git/",
  "^%.aws/",
  "^%.ssh/",
  "^%.git/",
}

--- True when a path must never have its content captured.
---@param path string absolute or relative path
function M.is_secret_path(path)
  if not path or path == "" then
    return false
  end
  local base = vim.fn.fnamemodify(path, ":t")
  for _, pat in ipairs(SECRET_BASENAMES) do
    if base:match(pat) then
      return true
    end
    -- also match "*credentials*" / "*secrets*" anywhere in basename
    if base:find("credentials", 1, true) or base:find("secrets", 1, true) then
      return true
    end
  end
  local norm = path:gsub("\\", "/")
  for _, pat in ipairs(SECRET_SEGMENTS) do
    if norm:match(pat) then
      return true
    end
    if norm:find("/.aws", 1, true) or norm:find("/.ssh", 1, true) then
      -- plain-substring fallback for odd prefixes
      return true
    end
  end
  return false
end

--- Buffer types that are never captured.
local EXCLUDED_BUFTYPES = {
  terminal = true,
  prompt = true,
  nofile = true,
  quickfix = true,
  help = true,
  popup = true,
}

--- True when a buffer must be excluded from capture.
---@param bufnr integer
function M.is_excluded_buffer(bufnr)
  if not vim.api.nvim_buf_is_valid(bufnr) then
    return true
  end
  local bt = vim.api.nvim_get_option_value("buftype", { buf = bufnr })
  if EXCLUDED_BUFTYPES[bt] then
    return true
  end
  -- Per-buffer opt-out: b.tabcomplete_trajectory_optout or b.tabcomplete_exclude
  local ok1, v1 = pcall(vim.api.nvim_buf_get_var, bufnr, "tabcomplete_trajectory_optout")
  if ok1 and v1 then
    return true
  end
  local ok2, v2 = pcall(vim.api.nvim_buf_get_var, bufnr, "tabcomplete_exclude")
  if ok2 and v2 then
    return true
  end
  local name = vim.api.nvim_buf_get_name(bufnr)
  if name and name ~= "" and M.is_secret_path(name) then
    return true
  end
  return false
end

--- Current cursor in ZERO-based {row, col} protocol form for a window.
function M.cursor_zero(winid)
  winid = winid or vim.api.nvim_get_current_win()
  local ok, pos = pcall(vim.api.nvim_win_get_cursor, winid)
  if not ok or not pos then
    return { row = 0, col = 0 }
  end
  return { row = math.max(0, pos[1] - 1), col = math.max(0, pos[2]) }
end

--- Current vim mode string (e.g. "n", "i", "v").
function M.current_mode()
  local ok, m = pcall(vim.api.nvim_get_mode)
  if ok and m and m.mode then
    return m.mode
  end
  return "n"
end

--- Detect NUL bytes in the first `sample` bytes (binary heuristic).
function M.has_nul_bytes(path, sample)
  sample = sample or 8192
  local f = io.open(path, "rb")
  if not f then
    return false
  end
  local chunk = f:read(sample)
  f:close()
  if not chunk then
    return false
  end
  return chunk:find("\0", 1, true) ~= nil
end

return M
