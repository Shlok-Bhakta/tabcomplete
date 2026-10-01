-- Real selected-model automatic smoke in a disposable repository.
-- All interactions are scripted and synthetic, never human feedback.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local fixture = assert(vim.env.TABCOMPLETE_AUTO_FIXTURE)
local collector_url = assert(vim.env.TABCOMPLETE_COLLECTOR_URL)
local revision = assert(vim.env.TABCOMPLETE_MODEL_SHA256)
local backend = vim.env.TABCOMPLETE_PREDICTOR_BACKEND or "llama-cpp-legacy"
local predictor_url = vim.env.TABCOMPLETE_PREDICTOR_URL
  or (backend == "rust-editor-v1" and "http://127.0.0.1:19094" or nil)
local case
if vim.env.TABCOMPLETE_AUTO_CASE then
  case = vim.json.decode(table.concat(vim.fn.readfile(vim.env.TABCOMPLETE_AUTO_CASE), "\n"))
end
vim.cmd("filetype on")
vim.cmd("edit " .. vim.fn.fnameescape(fixture))
local collector = require("tabcomplete_trajectory").setup({
  server_url = collector_url, capture_keys = false, batch_interval_ms = 30000,
  periodic_anchor_every = 1,
})
local predict = require("tabcomplete_trajectory.predict").setup({
  backend = backend, url = predictor_url,
  mode = "automatic", experimental_auto_opt_in = true,
  automatic_quality_validated = false, automatic_personalization_enabled = false,
  persist_mode = false, synthetic = true, model = "q25-coder", model_revision = revision,
  precision = vim.env.TABCOMPLETE_MODEL_PRECISION or "Q4_K_M",
  runtime_config_hash = assert(vim.env.TABCOMPLETE_RUNTIME_CONFIG_SHA256),
  adapter_identity = "full-weight", debounce_ms = 250,
  protocol_version = case and "single-line-edit-v1" or "compact-next-edit-v1",
  automatic_prefix_guard = case ~= nil,
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
  local row = case and case.target_row or 2
  vim.api.nvim_buf_set_lines(0, row, row + 1, false, { line })
  vim.api.nvim_win_set_cursor(0, { row + 1, case and math.min(case.cursor_col, #line) or 4 })
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
local original_line = case and case.original_line or "    return value"
local row = case and case.target_row or 2
if case then
  trigger(original_line)
else
  vim.api.nvim_buf_set_lines(0, 1, 2, false, { "    total = item + 267" })
  vim.api.nvim_win_set_cursor(0, { 3, 4 })
  vim.api.nvim_exec_autocmds("TextChangedI", { buffer = 0 })
end
local first = wait_shown(prior and prior.event_id)
assert(first.payload.active_buffer and first.payload.focused)
local accepted_prediction_id = first.payload.prediction_id
local acceptance = vim.fn.maparg("<M-l>", "i", false, true)
assert(type(acceptance.callback) == "function", "acceptance mapping has no callback")
acceptance.callback()
assert(predict.status().counters.accepted == 1, "installed acceptance key did not apply")
assert(vim.api.nvim_buf_get_lines(0, row, row + 1, false)[1] ~= original_line)
local acceptance_closed_once = 0
for _, event in ipairs(collector.queue) do
  if event.payload.prediction_id == accepted_prediction_id then
    assert(event.event_type ~= "prediction_dismissed", "acceptance also recorded dismissal")
    if event.event_type == "prediction_accepted" then acceptance_closed_once = acceptance_closed_once + 1 end
  end
end
assert(acceptance_closed_once == 1, "acceptance not recorded exactly once")
vim.cmd("undo")
assert(vim.api.nvim_buf_get_lines(0, row, row + 1, false)[1] == original_line)
prior = latest_shown()
trigger(original_line .. " ")
local second = wait_shown(prior and prior.event_id)
local dismissed_prediction_id = second.payload.prediction_id
trigger("    pass")
assert(not predict.status().proposal_active, "typing did not dismiss the proposal")
prior = latest_shown()
trigger(original_line)
local third = wait_shown(prior and prior.event_id)
local typed_prediction_id = third.payload.prediction_id
local proposed = third.payload.proposed_text
local prefix = case and "" or vim.api.nvim_buf_get_lines(0, 2, 3, false)[1]:sub(1, 4)
assert(not case or third.payload.action == "replace_line", "typed-match smoke requires replacement")
trigger(prefix .. proposed)
assert(not predict.status().proposal_active, "matching typing did not dismiss")
local navigation_prediction_id
if case then
  trigger(original_line .. "  ")
  assert(vim.wait(10000, function()
    for i = #collector.queue, 1, -1 do
      local event = collector.queue[i]
      if event.event_type == "prediction_requested" then
        if event.payload.prediction_id == typed_prediction_id then return false end
        navigation_prediction_id = event.payload.prediction_id
        return predict.status().in_flight
      end
    end
    return false
  end, 10), "fourth automatic request did not start")
  vim.cmd("edit " .. vim.fn.fnameescape(vim.fn.fnamemodify(fixture, ":h") .. "/sample.go"))
  assert(not predict.status().proposal_active, "file switch retained a proposal")
  predict.set_mode("off")
  vim.cmd("edit " .. vim.fn.fnameescape(fixture))
  assert(vim.wait(10000, function() return not predict.status().in_flight end, 20),
    "cancelled navigation request did not close")
  assert(not predict.status().proposal_active, "late response appeared after navigation")
end
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
  navigation_prediction_id = navigation_prediction_id,
  prediction_count = predict.status().counters.requested,
  displayed_count = predict.status().counters.displayed,
  synthetic = true }))
