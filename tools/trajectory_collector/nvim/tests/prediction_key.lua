-- Synthetic transport and decisions. Test the actual installed mapping callbacks.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
vim.fn.setenv("XDG_STATE_HOME", vim.fn.tempname() .. "-prediction-key-state")
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local util = require("tabcomplete_trajectory.util")
local predict = require("tabcomplete_trajectory.predict")
local messages = {}
vim.notify = function(message) messages[#messages + 1] = message end
collector.started, collector.disabled, collector.paused = true, false, false
collector.session_id, collector.queue, collector.seq = "synthetic-prediction-key", {}, 0
buffers.set_callbacks(function(buf, kind, payload) return collector.emit(buf, kind, payload) end,
  function() end)
local file = vim.fn.tempname() .. ".py"
vim.fn.writefile({ "hello" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
vim.bo.filetype = "python"
assert(buffers.attach(vim.api.nvim_get_current_buf()))
vim.cmd("runtime plugin/tabcomplete_predict.lua")
predict.setup({ mode = "automatic", experimental_auto_opt_in = true, synthetic = true,
  persist_mode = false, model = "stub", debounce_ms = 10000 })
local wire, requests = "R\nhi", 0
predict._request_impl = function(_, _, callback)
  requests = requests + 1
  callback({ code = 0, stdout = "data: " .. vim.json.encode({ content = wire, stop = false })
    .. '\n\ndata: {"content":"","stop":true,"stop_type":"eos"}\n\n' })
  return { kill = function() end }
end
local function invoke(editor_mode)
  local map = vim.fn.maparg(predict.status().prediction_key, editor_mode, false, true)
  assert(type(map.callback) == "function")
  map.callback()
  assert(vim.wait(1000, function() return not predict.status().in_flight end))
end
invoke("n")
assert(requests == 1 and predict.status().proposal_active)
assert(messages[#messages]:find("preview ready", 1, true))
assert(predict.status().mode == "automatic")
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "hello")
assert(predict.reject())
local original_mode = util.current_mode
util.current_mode = function() return "i" end -- mode seam, actual insert mapping callback
invoke("i")
assert(requests == 2 and predict.status().proposal_active)
assert(predict.reject())
wire = "N\n"
invoke("i")
assert(requests == 3 and not predict.status().proposal_active)
assert(messages[#messages]:find("model returned no change", 1, true))
wire = "bad header"
invoke("i")
assert(requests == 4 and not predict.status().proposal_active)
assert(messages[#messages]:find("malformed compact action", 1, true))
vim.g.tabcomplete_predictor_focus_lost = true
invoke("i")
assert(requests == 4 and messages[#messages]:find("focus", 1, true))
vim.g.tabcomplete_predictor_focus_lost = false
util.current_mode = original_mode
-- A changed option is installed even after plugin/default setup already ran.
vim.keymap.set({ "n", "i" }, "<F8>", "user mapping")
predict.setup({ predict_key = "<F8>" })
assert(predict.status().prediction_key == "<M-P>")
assert(vim.fn.maparg("<M-p>", "n") == "")
assert(vim.fn.maparg("<F8>", "n") == "user mapping")
assert(vim.fn.maparg("<F8>", "i") == "user mapping")
predict.setup({ predict_key = "<F9>" })
assert(predict.status().prediction_key == "<F9>")
assert(vim.fn.maparg("<M-P>", "n") == "")
assert(predict.set_mode("off"))
invoke("n")
assert(requests == 4 and messages[#messages]:find("mode off", 1, true))
vim.fn.delete(file)
print("prediction mappings passed: normal/insert, no-edit, invalid output, focus/off, occupied maps, reconfiguration")
