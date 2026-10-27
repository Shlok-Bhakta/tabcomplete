-- Headless regression test: spooled session bodies are delivered and a
-- permanently rejected batch does not block later batches.
-- Run from tools/trajectory_collector/nvim/: nvim --headless -l tests/spool_retry.lua
local src = debug.getinfo(1, "S").source:sub(2)
local root = vim.fn.fnamemodify(vim.fn.fnamemodify(src, ":h"), ":h")
vim.opt.rtp:prepend(root)
local tmp_state = vim.fn.tempname() .. "-tcstate"
vim.fn.mkdir(tmp_state, "p")
vim.fn.setenv("XDG_STATE_HOME", tmp_state)
vim.fn.setenv("TABCOMPLETE_COLLECTOR_URL", "http://collector.test:8787")

local transport = require("tabcomplete_trajectory.transport")
local spool = require("tabcomplete_trajectory.spool")
local collector = require("tabcomplete_trajectory")
local config = require("tabcomplete_trajectory.config")
config.setup({ server_url = "http://collector.test:8787" })

local posts = {}
transport._curl_runner = function(argv, body, _timeout, cb)
  local url = argv[#argv]
  local decoded = vim.json.decode(body)
  posts[#posts + 1] = { url = url, body = decoded }
  if url:match("/v1/events/batch$") and decoded.session_id == "poison" then
    cb(0, '{"error":"not_found"}\n404')
  else
    cb(0, '{"ok":true}\n200')
  end
end

spool.write_blob("ab12", "aGVsbG8=")
spool.write_batch({}, { kind = "session", path = "/v1/session/start", body = { session_id = "s1", machine_id = "m" } })
spool.write_batch({ { session_id = "poison", event_id = "e0" } }, { kind = "events", path = "/v1/events/batch", session_id = "poison" })
spool.write_batch({ { session_id = "s1", event_id = "e1" } }, { kind = "events", path = "/v1/events/batch", session_id = "s1" })

local result
collector.retry_spool(function(ok, info) result = { ok, info } end)
vim.wait(2000, function() return result ~= nil end, 10)

local function check(cond, msg)
  if not cond then
    io.stderr:write("FAIL: " .. msg .. "\n")
    vim.cmd("cquit 1")
  end
end
check(result and result[1] == true, "retry finished: " .. vim.inspect(result))
check(#posts == 4, "four posts, got " .. #posts)
check(posts[1].url:match("/v1/blobs/upload$") and posts[1].body.sha256 == "ab12", "spooled blob uploaded first")
check(#spool.list_blobs() == 0, "blob spool drained")
check(posts[2].url:match("/v1/session/start$") and posts[2].body.session_id == "s1", "session start sent with its stored body")
check(#spool.list() == 0, "spool empty after retry")
check(#vim.fn.globpath(spool.spool_dir() .. "/rejected", "batch-*.json", false, true) == 1, "poison batch quarantined")
print("spool_retry: ok")
vim.cmd("qa!")
