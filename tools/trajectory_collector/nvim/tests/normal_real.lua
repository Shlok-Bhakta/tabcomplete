-- Actual local model and existing collector; scripted synthetic decisions only.
-- Run against the public smoke fixture in a disposable repository, in normal mode.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local fixture = assert(vim.env.TABCOMPLETE_AUTO_FIXTURE)
vim.cmd("filetype on")
vim.cmd("edit " .. vim.fn.fnameescape(fixture))
local collector = require("tabcomplete_trajectory").setup({
  server_url = assert(vim.env.TABCOMPLETE_COLLECTOR_URL), capture_keys = false,
  batch_interval_ms = 30000, periodic_anchor_every = 1,
})
assert(vim.wait(10000, function()
  for _, event in ipairs(collector.queue) do
    if event.event_type == "buffer_open" and event.payload.blob_uploaded then return true end
  end
  return false
end, 20), "initial anchor not uploaded")
vim.api.nvim_win_set_cursor(0, { 19, 10 })
assert(vim.api.nvim_get_mode().mode == "n", "test must use actual normal mode")
local predict = require("tabcomplete_trajectory.predict").setup({
  backend = "rust-editor-v1", url = "http://127.0.0.1:19094",
  mode = "automatic", experimental_auto_opt_in = true, automatic_normal_mode = true,
  automatic_prefix_guard = true, synthetic = true, persist_mode = false,
  model = "q25", model_revision = assert(vim.env.TABCOMPLETE_MODEL_SHA256),
  runtime_config_hash = assert(vim.env.TABCOMPLETE_RUNTIME_CONFIG_SHA256),
  debounce_ms = 250,
})
vim.cmd("runtime plugin/tabcomplete_predict.lua")
local function shown_after(previous)
  local found
  assert(vim.wait(25000, function()
    for i = #collector.queue, 1, -1 do
      local event = collector.queue[i]
      if event.event_type == "prediction_shown" and event.event_id ~= previous then
        found = event
        return predict.status().proposal_active
      end
    end
    return false
  end, 20), "actual model did not display a normal-mode proposal")
  return found
end
local first = shown_after(nil)
vim.api.nvim_win_set_cursor(0, { 18, 0 })
vim.api.nvim_exec_autocmds("CursorMoved", { buffer = 0 })
assert(not predict.status().proposal_active)
local dismissed
for _, event in ipairs(collector.queue) do
  if event.payload.prediction_id == first.payload.prediction_id then
    assert(event.event_type ~= "prediction_rejected", "navigation became negative feedback")
    if event.event_type == "prediction_dismissed" then dismissed = event end
  end
end
assert(dismissed and dismissed.payload.outcome == "dismissed_navigation")
vim.api.nvim_win_set_cursor(0, { 19, 10 })
local prediction_mapping = vim.fn.maparg("<M-p>", "n", false, true)
assert(type(prediction_mapping.callback) == "function")
prediction_mapping.callback()
local second = shown_after(first.event_id)
assert(predict.status().mode == "automatic")
local mapping = vim.fn.maparg("<M-l>", "n", false, true)
assert(type(mapping.callback) == "function")
mapping.callback()
assert(predict.status().counters.accepted == 1)
vim.api.nvim_exec_autocmds("TextChanged", { buffer = 0 })
vim.api.nvim_exec_autocmds("CursorMoved", { buffer = 0 })
vim.wait(400)
assert(predict.status().counters.requested == 2,
  "accepted edit or its cursor event triggered a third request")
assert(predict.set_mode("off"))
vim.cmd("write")
collector._test.anchor(vim.api.nvim_get_current_buf(), "write")
assert(vim.wait(10000, function()
  for _, event in ipairs(collector.queue) do
    if event.event_type == "buffer_write" and event.payload.blob_uploaded then return true end
  end
  return false
end, 20))
local done, success = false, false
collector.flush_now(function(ok) done, success = true, ok end)
assert(vim.wait(10000, function() return done end, 20) and success)
print(vim.json.encode({ synthetic = true, actual_normal_mode = true,
  session_id = collector.session_id,
  navigation_prediction_id = first.payload.prediction_id,
  accepted_prediction_id = second.payload.prediction_id,
  requested = predict.status().counters.requested, displayed = predict.status().counters.displayed,
  accepted = predict.status().counters.accepted }))
