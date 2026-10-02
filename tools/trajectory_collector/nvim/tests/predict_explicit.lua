-- Explicit commands work in normal mode without changing automatic mode.
-- All requests, actions, and decisions are synthetic; transport is stubbed.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
vim.fn.setenv("XDG_STATE_HOME", vim.fn.tempname() .. "-explicit-state")
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local predict = require("tabcomplete_trajectory.predict")
vim.cmd("runtime plugin/tabcomplete_predict.lua")
collector.started, collector.disabled, collector.paused = true, false, false
collector.session_id, collector.queue, collector.seq = "synthetic-explicit-test", {}, 0
buffers.set_callbacks(function(buf, kind, payload)
  return collector.emit(buf, kind, payload)
end, function() end)
local file = vim.fn.tempname() .. ".py"
vim.fn.writefile({ "hello" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
vim.bo.filetype = "python"
assert(buffers.attach(vim.api.nvim_get_current_buf()))
predict.setup({ mode = "automatic", experimental_auto_opt_in = true, synthetic = true,
  persist_mode = false, model = "stub", debounce_ms = 10000 })
assert(vim.api.nvim_get_mode().mode == "n")
local requests = 0
predict._request_impl = function(_, _, callback)
  requests = requests + 1
  callback({ code = 0, stdout = 'data: {"content":"R\\nhi","stop":false}\n\n'
    .. 'data: {"content":"","stop":true,"stop_type":"eos"}\n\n' })
  return { kill = function() end }
end
vim.cmd("TabCompletePredict")
assert(vim.wait(1000, function() return predict.status().proposal_active end),
  "explicit command in normal mode was suppressed by automatic insert-mode gate")
assert(requests == 1 and predict.status().mode == "automatic")
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "hello",
  "explicit preview mutated source")
assert(predict.reject())
local allowed, reason = predict.predict()
assert(not allowed and reason:find("insert mode", 1, true),
  "background scheduling lost the insert-mode safeguard or diagnostic")
assert(predict.status().automatic_block_reason:find("insert mode", 1, true))
vim.g.tabcomplete_predictor_focus_lost = true
allowed, reason = predict.predict({ explicit = true })
assert(not allowed and reason:find("focus", 1, true))
vim.g.tabcomplete_predictor_focus_lost = false
local original_blink = package.loaded["blink.cmp"]
package.loaded["blink.cmp"] = { is_visible = function() return true end }
allowed, reason = predict.predict({ explicit = true })
assert(not allowed and reason:find("completion menu", 1, true))
package.loaded["blink.cmp"] = original_blink
assert(requests == 1, "blocked explicit request reached transport")
assert(predict.set_mode("off"))
allowed, reason = predict.predict({ explicit = true })
assert(not allowed and reason == "mode off")
assert(requests == 1)
vim.fn.delete(file)
print("explicit normal-mode command, preview safety, mode preservation, and UI diagnostics passed")
