-- Explicit synthetic decisions against the installed local model and owned collector.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local collector_url = assert(vim.env.TABCOMPLETE_COLLECTOR_URL)
local predictor_url = assert(vim.env.TABCOMPLETE_PREDICTOR_URL)
local fixture = assert(vim.env.TABCOMPLETE_TEST_FILE)
local model = assert(vim.env.TABCOMPLETE_MODEL_ALIAS)
local revision = assert(vim.env.TABCOMPLETE_MODEL_SHA256)
assert(vim.fn.filereadable(fixture) == 1, "synthetic test file required")

vim.cmd("edit " .. vim.fn.fnameescape(fixture))
vim.cmd("runtime plugin/tabcomplete_predict.lua")
local collector = require("tabcomplete_trajectory").setup({
  server_url = collector_url, capture_keys = false, batch_interval_ms = 30000,
  spool_retry_interval_ms = 1000,
})
local predict = require("tabcomplete_trajectory.predict").setup({
  synthetic = true, url = predictor_url, model = model, model_revision = revision,
  precision = "Q4_K_M", adapter_identity = "full-weight", automatic_gates_passed = false,
})
assert(vim.wait(10000, function() return collector.last_ok_ms ~= nil end),
  "collector session start was not acknowledged")
assert(vim.wait(10000, function()
  for _, event in ipairs(collector.queue) do
    if event.event_type == "buffer_open" and event.payload.path == fixture
        and event.payload.blob_uploaded then return true end
  end
  return false
end), "initial anchor was not uploaded")

local outcomes = { requests = 0, shown = 0, accepted = 0, rejected = 0,
  no_edit = 0, invalid_or_outage = 0 }
for attempt = 1, 12 do
  local lines = vim.api.nvim_buf_get_lines(0, 0, -1, false)
  local row = ((attempt - 1) % math.min(#lines, 3)) + 1
  vim.api.nvim_win_set_cursor(0, { row, 0 })
  vim.cmd("TabCompletePredict")
  outcomes.requests = outcomes.requests + 1
  assert(vim.wait(35000, function() return not predict.status().in_flight end),
    "local model request did not complete")
  local status = predict.status()
  if status.proposal_active then
    outcomes.shown = outcomes.shown + 1
    if outcomes.accepted == 0 then
      vim.cmd("TabCompleteAccept")
      outcomes.accepted = outcomes.accepted + 1
      vim.cmd("undo")
    else
      vim.cmd("TabCompleteReject")
      outcomes.rejected = outcomes.rejected + 1
    end
  elseif status.state == "no_edit" then
    outcomes.no_edit = outcomes.no_edit + 1
  else
    outcomes.invalid_or_outage = outcomes.invalid_or_outage + 1
  end
  if outcomes.accepted > 0 and outcomes.rejected > 0 then break end
end
vim.cmd("write")
local flushed, ok = false, false
collector.flush_now(function(success) flushed, ok = true, success end)
assert(vim.wait(10000, function() return flushed end), "collector flush timed out")
assert(ok, "collector batch failed")
local session_id = collector.session_id
collector.shutdown()
print(vim.json.encode({ session_id = session_id, synthetic = true,
  model_response_kind = "actual-selected-local-model", outcomes = outcomes,
  automatic_mode = false }))
