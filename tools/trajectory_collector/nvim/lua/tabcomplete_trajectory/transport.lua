-- tabcomplete_trajectory.transport
-- Async HTTP via vim.system() + curl. Never blocks the editor, never uses
-- os.execute on any hot path. Treats only HTTP 2xx as ACK.
local M = {}

-- Test seam: tests may set M._post_impl = function(url, body, timeout_ms) -> ok, err
M._post_impl = nil
-- Test seam: M._curl_runner = function(argv, body, timeout_ms, cb) for unit tests.
M._curl_runner = nil

local function default_curl_runner(argv, body, timeout_ms, cb)
  local out = ""
  local proc
  local done = false
  local function finish(code, signal)
    if done then
      return
    end
    done = true
    cb(code, out, signal)
  end
  proc = vim.system(argv, { stdin = body, text = true }, function(res)
    out = res.stdout or ""
    finish(res.code, res.signal)
  end)
  -- Hard timeout guard: kill the child if it outlives timeout_ms + grace.
  local timer = vim.uv.new_timer()
  if timer then
    timer:start(timeout_ms + 1000, 0, function()
      timer:stop()
      timer:close()
      if proc and proc.kill then
        pcall(proc.kill, proc, "sigterm")
      end
      vim.schedule(function()
        finish(124, 0)
      end)
    end)
    -- wrap finish to stop the timer: poll via deferred close is handled in cb chain
  end
  return proc
end

--- Parse trailing HTTP code appended via curl -w "\n%{http_code}".
---@return body string, code integer|nil
function M.split_http_code(raw)
  if not raw or raw == "" then
    return "", nil
  end
  local code_str = raw:match("(%d%d%d)%s*$")
  local code = code_str and tonumber(code_str) or nil
  local body = raw:gsub("\n?%d%d%d%s*$", "", 1)
  return body, code
end

--- Async POST JSON. cb(ok:boolean, info:table|string).
function M.post_json_async(url, body_table, timeout_ms, cb)
  if M._post_impl then
    -- Mock contract: impl(url, body, timeout_ms) -> true[, info] on ACK,
    -- false[, err] on failure; a thrown error also counts as failure.
    local results = { pcall(M._post_impl, url, body_table, timeout_ms) }
    local ok = table.remove(results, 1)
    vim.schedule(function()
      if not ok then
        cb(false, tostring(results[1]))
      elseif results[1] == false then
        cb(false, results[2])
      else
        cb(true, results[2] or { mocked = true })
      end
    end)
    return
  end
  local ok_enc, body = pcall(vim.json.encode, body_table)
  if not ok_enc then
    vim.schedule(function()
      cb(false, "json encode failed")
    end)
    return
  end
  local argv = {
    "curl",
    "-sS",
    "-m",
    tostring(math.max(1, math.ceil((timeout_ms or 5000) / 1000))),
    "-X",
    "POST",
    "-H",
    "Content-Type: application/json",
    "--data-binary",
    "@-",
    "-w",
    "\n%{http_code}",
    url,
  }
  local runner = M._curl_runner or default_curl_runner
  local ok_run, err_run = pcall(runner, argv, body, timeout_ms or 5000, function(code, raw, signal)
    vim.schedule(function()
      if code ~= 0 then
        cb(false, { exit_code = code, signal = signal, stderr_tail = (raw or ""):sub(-500) })
        return
      end
      local response_body, http_code = M.split_http_code(raw or "")
      if http_code and http_code >= 200 and http_code < 300 then
        local ok_json, decoded = pcall(vim.json.decode, response_body)
        if ok_json and type(decoded) == "table" then
          decoded.http_code = http_code
          cb(true, decoded)
        else
          cb(true, { http_code = http_code })
        end
      else
        cb(false, { http_code = http_code, body_tail = (raw or ""):sub(-500) })
      end
    end)
  end)
  if not ok_run then
    vim.schedule(function()
      cb(false, tostring(err_run))
    end)
  end
end

return M
