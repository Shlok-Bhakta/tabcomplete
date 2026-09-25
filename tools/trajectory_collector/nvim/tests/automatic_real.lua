-- Real selected-model automatic smoke in a disposable repository.
-- All interactions are scripted and synthetic, never human feedback.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local fixture = assert(vim.env.TABCOMPLETE_AUTO_FIXTURE)
local collector_url = assert(vim.env.TABCOMPLETE_COLLECTOR_URL)
local revision = assert(vim.env.TABCOMPLETE_MODEL_SHA256)
vim.cmd("filetype on")
vim.cmd("edit " .. vim.fn.fnameescape(fixture))
local collector = require("tabcomplete_trajectory").setup({
  server_url = collector_url, capture_keys = false, batch_interval_ms = 30000,
  periodic_anchor_every = 1,
})
local predict = require("tabcomplete_trajectory.predict").setup({
  mode = "automatic", experimental_auto_opt_in = true,
  automatic_quality_validated = false, automatic_personalization_enabled = false,
  persist_mode = false, synthetic = true, model = "q25-coder", model_revision = revision,
  precision = "Q4_K_M", adapter_identity = "full-weight", debounce_ms = 250,
})
assert(predict.status().mode == "automatic")
assert(predict.status().acceptance_key == "<M-l>", "acceptance mapping changed")
assert(vim.fn.maparg("<M-l>", "i") ~= "", "Alt+l is not mapped")
assert(vim.wait(10000, function()
  for _, event in ipairs(collector.queue) do
    if event.event_type == "buffer_open" and event.payload.path == fixture
        and event.payload.blob_uploaded then return true end
  end
  return false
end, 20), "initial replay anchor was not uploaded")
local util = require("tabcomplete_trajectory.util")
local real_mode = util.current_mode
-- Headless -l has no terminal input stream. This seam represents insert-mode
-- typing; all text changes still use Neovim's real buffer API and collector.
util.current_mode = function() return "i" end
local function latest_shown()
  for i = #collector.queue, 1, -1 do
    local event = collector.queue[i]
    if event.event_type == "prediction_shown" then return event end
  end
end
local function trigger(line)
  vim.api.nvim_buf_set_lines(0, 2, 3, false, { line })
  vim.api.nvim_win_set_cursor(0, { 3, 4 })
  vim.api.nvim_exec_autocmds("TextChangedI", { buffer = 0 })
end
local function wait_shown(previous)
  assert(vim.wait(15000, function()
    local shown = latest_shown()
    return shown and shown.event_id ~= previous and predict.status().proposal_active
  end, 20), "real model did not display a single-line automatic proposal")
  return latest_shown()
end
local prior = latest_shown()
vim.api.nvim_buf_set_lines(0, 1, 2, false, { "    total = item + 267" })
vim.api.nvim_win_set_cursor(0, { 3, 4 })
vim.api.nvim_exec_autocmds("TextChangedI", { buffer = 0 })
local first = wait_shown(prior and prior.event_id)
assert(first.payload.active_buffer and first.payload.focused)
local accepted_prediction_id = first.payload.prediction_id
assert(predict.accept(), "acceptance command failed")
assert(vim.api.nvim_buf_get_lines(0, 2, 3, false)[1] ~= "    return value")
vim.cmd("undo")
assert(vim.api.nvim_buf_get_lines(0, 2, 3, false)[1] == "    return value")
prior = latest_shown()
trigger("    return value ")
local second = wait_shown(prior and prior.event_id)
local dismissed_prediction_id = second.payload.prediction_id
trigger("    pass")
assert(not predict.status().proposal_active, "typing did not dismiss the proposal")
prior = latest_shown()
trigger("    return value")
local third = wait_shown(prior and prior.event_id)
local typed_prediction_id = third.payload.prediction_id
local proposed = third.payload.proposed_text
local prefix = vim.api.nvim_buf_get_lines(0, 2, 3, false)[1]:sub(1, 4)
trigger(prefix .. proposed)
assert(not predict.status().proposal_active, "matching typing did not dismiss")
predict.set_mode("off")
vim.cmd("write")
collector._test.anchor(vim.api.nvim_get_current_buf(), "write")
assert(vim.wait(10000, function()
  for _, event in ipairs(collector.queue) do
    if event.event_type == "buffer_write" and event.payload.path == fixture
        and event.payload.blob_uploaded then return true end
  end
  return false
end), "final anchor was not uploaded")
local flushed, ok = false, false
collector.flush_now(function(success) flushed, ok = true, success end)
assert(vim.wait(10000, function() return flushed end) and ok, "collector flush failed")
util.current_mode = real_mode
print(vim.json.encode({ session_id = collector.session_id,
  accepted_prediction_id = accepted_prediction_id,
  dismissed_prediction_id = dismissed_prediction_id,
  typed_prediction_id = typed_prediction_id,
  prediction_count = predict.status().counters.requested,
  displayed_count = predict.status().counters.displayed,
  synthetic = true }))
