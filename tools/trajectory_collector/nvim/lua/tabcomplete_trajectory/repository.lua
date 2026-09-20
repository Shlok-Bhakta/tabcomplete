-- tabcomplete_trajectory.repository
-- Git root detection, sanitized metadata, file enumeration via
-- `git ls-files --cached --others --exclude-standard` (never crawls .git,
-- never follows symlinks outside the root). Content-capped at
-- max_repo_file_bytes; giant binaries (NUL-byte heuristic) store metadata only.
local M = {}

M.cache = {
  root = nil,
  root_name = nil,
  origin = nil,
  branch = nil,
  head = nil,
  dirty = false,
  scanned_at_ms = 0,
}

local function run_git(args, cwd, timeout_ms)
  timeout_ms = timeout_ms or 5000
  local argv = { "git", "-C", cwd }
  for _, a in ipairs(args) do
    argv[#argv + 1] = a
  end
  local res = vim.system(argv, { text = true, timeout = timeout_ms }):wait()
  if res.code ~= 0 then
    return nil, (res.stderr or ""):sub(1, 200)
  end
  return (res.stdout or ""):gsub("%s+$", ""), nil
end

-- Overridable in tests.
M._run_git = run_git

--- Detect git repo for a file/dir. Returns root or nil. Never throws.
function M.detect_root(start_path)
  local dir = start_path
  if not dir or dir == "" then
    dir = vim.fn.getcwd()
  end
  if vim.fn.filereadable(dir) == 1 then
    dir = vim.fn.fnamemodify(dir, ":h")
  end
  local out, err = M._run_git({ "rev-parse", "--show-toplevel" }, dir)
  if not out or out == "" then
    return nil
  end
  return out
end

--- Collect repo metadata. Sanitizes origin (strips credentials). Never throws.
function M.collect_metadata(root)
  local meta = {
    root = root,
    root_name = vim.fn.fnamemodify(root, ":t"),
    origin = "",
    branch = "",
    head = "",
    dirty = false,
  }
  local ok1, origin = pcall(M._run_git, { "remote", "get-url", "origin" }, root)
  if ok1 and origin and origin ~= "" then
    meta.origin = require("tabcomplete_trajectory.util").sanitize_origin(origin)
  end
  local ok2, branch = pcall(M._run_git, { "rev-parse", "--abbrev-ref", "HEAD" }, root)
  if ok2 and branch and branch ~= "" then
    meta.branch = branch
  end
  local ok3, head = pcall(M._run_git, { "rev-parse", "HEAD" }, root)
  if ok3 and head and head ~= "" then
    meta.head = head
  end
  local ok4, status = pcall(M._run_git, { "status", "--porcelain" }, root)
  if ok4 and status and status ~= "" then
    meta.dirty = true
  end
  return meta
end

--- Refresh (and cache) repo metadata for the current context.
function M.refresh(start_path)
  local util = require("tabcomplete_trajectory.util")
  local root = M.detect_root(start_path or vim.api.nvim_buf_get_name(0))
  if not root then
    M.cache = { root = nil, scanned_at_ms = util.now_ms() }
    return M.cache
  end
  local meta = M.collect_metadata(root)
  meta.scanned_at_ms = util.now_ms()
  M.cache = meta
  return M.cache
end

--- Enumerate tracked + untracked (non-ignored) files. One path per line.
--- Never crawls .git (git handles exclusion), never follows symlinks
--- outside root (we only stat inside root and skip symlinks pointing out).
function M.enumerate_files(root, timeout_ms)
  local out, err = M._run_git({ "ls-files", "--cached", "--others", "--exclude-standard", "-z" }, root)
  if not out then
    return {}, err or "git ls-files failed"
  end
  local files = {}
  for rel in (out .. "\0"):gmatch("([^%z]*)%z") do
    if rel ~= "" then
      files[#files + 1] = rel
    end
  end
  return files, nil
end

local function is_symlink_outside_root(root, rel)
  local full = root .. "/" .. rel
  local st = vim.uv.fs_lstat(full)
  if st and st.type == "link" then
    local target = vim.uv.fs_realpath(full)
    if target and target:sub(1, #root) ~= root then
      return true
    end
  end
  return false
end

--- Build snapshot entries: {path, size_bytes, content_hash?, truncated?, binary?, skipped?}.
--- Files over max_bytes or containing NUL bytes store metadata only.
function M.snapshot_entries(root, files, max_bytes)
  local util = require("tabcomplete_trajectory.util")
  local entries = {}
  for _, rel in ipairs(files) do
    local full = root .. "/" .. rel
    if not is_symlink_outside_root(root, rel) then
      local st = vim.uv.fs_stat(full)
      if st and st.type == "file" then
        local entry = { path = rel, size_bytes = st.size or 0 }
        if util.is_secret_path(full) then
          entry.skipped = "secret_path"
        elseif (st.size or 0) > (max_bytes or 1048576) then
          entry.truncated = true -- metadata only above cap
        elseif util.has_nul_bytes(full) then
          entry.binary = true -- metadata only for giant binaries
        else
          local f = io.open(full, "rb")
          if f then
            local content = f:read("*a")
            f:close()
            if content then
              entry.content_hash = util.sha256hex(content)
              entry.size_bytes = #content
            end
          end
        end
        entries[#entries + 1] = entry
      end
      -- sockets/devices never appear as type=="file": skipped implicitly
    end
  end
  return entries
end

--- Full snapshot payload for POST /v1/repository/snapshot.
function M.build_snapshot(root, max_bytes)
  local meta = M.collect_metadata(root)
  local files, err = M.enumerate_files(root)
  local entries = {}
  if not err then
    entries = M.snapshot_entries(root, files, max_bytes)
  end
  return {
    root_name = meta.root_name,
    origin = meta.origin,
    branch = meta.branch,
    head = meta.head,
    dirty = meta.dirty,
    files = entries,
    enumeration_error = err,
  }
end

return M
