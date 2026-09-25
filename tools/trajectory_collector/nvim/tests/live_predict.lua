-- Explicitly invoked synthetic integration run against the configured owned collector.
-- Model responses are stubs; this checks transport, replay and feedback semantics.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local collector_url = vim.env.TABCOMPLETE_COLLECTOR_URL
local fixture = vim.env.TABCOMPLETE_TEST_FILE
assert(collector_url and collector_url ~= "", "collector URL required")
assert(fixture and vim.fn.filereadable(fixture) == 1, "synthetic test file required")

vim.cmd("edit " .. vim.fn.fnameescape(fixture))
local collector = require("tabcomplete_trajectory").setup({
  server_url = collector_url, capture_keys = false, batch_interval_ms = 30000,
  spool_retry_interval_ms = 1000,
})
local predict = require("tabcomplete_trajectory.predict").setup({
  synthetic = true, model = "stub", model_revision = "synthetic-integration",
})
assert(vim.wait(10000, function() return collector.last_ok_ms ~= nil end),
  "collector session start was not acknowledged")
assert(vim.wait(10000, function()
  for _, event in ipairs(collector.queue) do
    if event.event_type == "buffer_open" and event.payload.path == fixture
        and event.payload.blob_uploaded then return true end
  end
  return false
end), "initial synthetic file anchor was not uploaded before editing")
local config = require("tabcomplete_trajectory.config")
local spool = require("tabcomplete_trajectory.spool")

local function stub(body)
  predict._request_impl = function(_, _, callback)
    callback({ code = 0, stdout =
      "data: " .. vim.json.encode({ content = body, stop = false }) .. "\n" ..
      "data: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos" }) .. "\n" })
    return { kill = function() end }
  end
end

local function shown(body)
  stub(body)
  assert(predict.predict())
  assert(vim.wait(3000, function() return predict.status().proposal_active end))
end

local function flushed()
  local complete, success = false, false
  collector.flush_now(function(ok)
    complete, success = true, ok
  end)
  assert(vim.wait(10000, function() return complete end), "collector flush timed out")
  return success
end

vim.api.nvim_win_set_cursor(0, { 1, 0 })
shown("R\ndef amount(items):")
assert(predict.accept())
vim.cmd("undo")
vim.cmd("write")

shown("R\ndef other(items):")
assert(predict.reject())

stub("N\n")
assert(predict.predict())
assert(vim.wait(3000, function() return predict.status().state == "no_edit" end))

local late
predict._request_impl = function(_, _, callback)
  late = callback
  return { kill = function() end }
end
assert(predict.predict())
vim.api.nvim_buf_set_lines(0, 0, 1, false, { "def human(items):" })
late({ code = 0, stdout = "" })
assert(vim.wait(3000, function() return not predict.status().in_flight end))
assert(not predict.status().proposal_active)
vim.cmd("write")

local javascript = fixture:gsub("example%.py$", "example.js")
local rust = fixture:gsub("example%.py$", "example.rs")
assert(vim.fn.filereadable(javascript) == 1 and vim.fn.filereadable(rust) == 1)
predict._request_impl = function(_, _, callback)
  late = callback
  return { kill = function() end }
end
assert(predict.predict())
vim.cmd("edit " .. vim.fn.fnameescape(javascript))
late({ code = 0, stdout = "" })
assert(vim.wait(3000, function() return not predict.status().in_flight end))
assert(not predict.status().proposal_active)
vim.cmd("edit " .. vim.fn.fnameescape(rust))
vim.cmd("edit " .. vim.fn.fnameescape(fixture))
collector.rescan()
assert(vim.wait(10000, function()
  for _, event in ipairs(collector.queue) do
    if (event.event_type == "buffer_open" or event.event_type == "buffer_write")
        and event.payload.blob_uploaded then return true end
  end
  return false
end), "no content-addressed anchor was uploaded")
local anchor_uploads = { succeeded = 0, failed = 0 }
for _, event in ipairs(collector.queue) do
  if event.event_type == "buffer_open" or event.event_type == "buffer_write" then
    local key = event.payload.blob_uploaded and "succeeded" or "failed"
    anchor_uploads[key] = anchor_uploads[key] + 1
  end
end
local anchor_error = collector.last_err

-- Fail one batch to the local closed port, then recover it from the spool.
config.options.server_url = "http://127.0.0.1:9"
shown("R\ndef rejected_during_outage(items):")
assert(predict.reject())
assert(not flushed(), "outage batch unexpectedly succeeded")
assert(#spool.list() > 0, "collector outage did not spool the batch")
config.options.server_url = collector_url
local retry_done, retry_ok = false, false
collector.retry_spool(function(ok)
  retry_done, retry_ok = true, ok
end)
assert(vim.wait(10000, function() return retry_done end), "spool retry timed out")
assert(retry_ok and #spool.list() == 0, "spool recovery failed")
assert(flushed())
local session_id = collector.session_id
collector.shutdown()
print(vim.json.encode({ session_id = session_id, synthetic = true,
  transport = "collector-live", anchor_uploads = anchor_uploads,
  anchor_error = anchor_error }))
