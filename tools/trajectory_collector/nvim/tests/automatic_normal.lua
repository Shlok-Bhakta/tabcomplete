-- Normal-mode scheduling and safety. Stub inference; all decisions synthetic.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
vim.fn.setenv("XDG_STATE_HOME", vim.fn.tempname() .. "-normal-state")
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local util = require("tabcomplete_trajectory.util")
local predict = require("tabcomplete_trajectory.predict")
collector.started, collector.disabled, collector.paused = true, false, false
collector.session_id, collector.queue, collector.seq = "synthetic-normal-test", {}, 0
buffers.set_callbacks(function(buf, kind, payload)
  return collector.emit(buf, kind, payload)
end, function() end)
local file = vim.fn.tempname() .. ".py"
vim.fn.writefile({ "first", "second", "third" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
vim.bo.filetype = "python"
assert(buffers.attach(vim.api.nvim_get_current_buf()))
local editor_mode = "n"
util.current_mode = function() return editor_mode end
local count, callbacks = 0, {}
predict._request_impl = function(_, _, callback)
  count = count + 1
  callbacks[count] = callback
  return { kill = function() end }
end
predict.setup({ mode = "automatic", automatic_normal_mode = true,
  experimental_auto_opt_in = true, synthetic = true, persist_mode = false,
  model = "stub", debounce_ms = 40 })
local function move(row)
  vim.api.nvim_win_set_cursor(0, { row, 0 })
  vim.api.nvim_exec_autocmds("CursorMoved", { buffer = 0 })
end
local function response(index, wire)
  callbacks[index]({ code = 0, stdout = "data: "
    .. vim.json.encode({ content = wire, stop = false }) .. "\n\ndata: "
    .. vim.json.encode({ content = "", stop = true, stop_type = "eos" }) .. "\n\n" })
end
move(2)
vim.wait(10)
move(3)
assert(count == 0, "moving inside debounce sent obsolete state")
assert(vim.wait(500, function() return count == 1 end))
response(1, "N\n")
assert(vim.wait(500, function() return not predict.status().in_flight end))
vim.wait(100)
assert(count == 1, "normal idle after no-edit caused request loop")
move(2)
assert(vim.wait(500, function() return count == 2 end))
move(1)
response(2, "R\nstale")
assert(vim.wait(500, function() return count == 3 end))
assert(not predict.status().proposal_active, "cursor movement displayed stale result")
response(3, "R\nreplacement")
assert(vim.wait(500, function() return predict.status().proposal_active end))
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "first")
move(2)
local navigation
for _, event in ipairs(collector.queue) do
  assert(event.event_type ~= "prediction_rejected", "navigation invented explicit negative feedback")
  if event.event_type == "prediction_dismissed" then navigation = event.payload end
end
assert(navigation and navigation.outcome == "dismissed_navigation"
  and navigation.outcome_source == "editor_observation")
assert(vim.wait(500, function() return count == 4 end))
response(4, "R\nreplacement")
assert(vim.wait(500, function() return predict.status().proposal_active end))
assert(predict.accept())
assert(vim.api.nvim_buf_get_lines(0, 1, 2, false)[1] == "replacement")
vim.api.nvim_exec_autocmds("TextChanged", { buffer = 0 })
vim.api.nvim_exec_autocmds("CursorMoved", { buffer = 0 })
vim.wait(100)
assert(count == 4, "accepted plugin edit triggered automatic request")
for _, excluded_mode in ipairs({ "v", "no", "c", "t", "R" }) do
  editor_mode = excluded_mode
  vim.api.nvim_exec_autocmds("CursorMoved", { buffer = 0 })
  vim.wait(80)
  assert(count == 4, "excluded mode triggered prediction: " .. excluded_mode)
end
editor_mode = "n"
vim.api.nvim_buf_set_lines(0, 0, 1, false, { "user edit" })
vim.api.nvim_exec_autocmds("TextChanged", { buffer = 0 })
assert(vim.wait(500, function() return count == 5 end), "normal-mode edit did not schedule")
response(5, "R\nanother proposal")
assert(vim.wait(500, function() return predict.status().proposal_active end))
vim.api.nvim_buf_set_lines(0, 1, 2, false, { "ordinary normal edit" })
local editor_dismissal
for _, event in ipairs(collector.queue) do
  if event.event_type == "prediction_dismissed" then editor_dismissal = event.payload end
end
assert(editor_dismissal and editor_dismissal.outcome == "dismissed_editor_change",
  "normal-mode change was falsely classified as typing rejection")
vim.api.nvim_exec_autocmds("TextChanged", { buffer = 0 })
assert(vim.wait(500, function() return count == 6 end))
response(6, "N\n")
assert(vim.wait(500, function() return not predict.status().in_flight end))
move(2)
assert(predict.set_mode("off"))
vim.wait(100)
assert(count == 6, "off failed to cancel normal-mode debounce")
assert(predict.status().automatic_normal_mode)
vim.fn.delete(file)
print("normal-mode debounce, no-edit idle, stale cursor, acceptance, edit, excluded modes, and off passed")
