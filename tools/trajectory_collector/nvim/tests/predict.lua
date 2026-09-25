-- Headless safety checks with a stubbed model response. No quality claim.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local collector = require("tabcomplete_trajectory")
local predict = require("tabcomplete_trajectory.predict")
vim.cmd("runtime plugin/tabcomplete_predict.lua")
for _, command in ipairs({ "TabCompletePredict", "TabCompleteAccept", "TabCompleteReject",
  "TabCompleteMode", "TabCompleteStatus" }) do
  assert(vim.fn.exists(":" .. command) == 2)
end
collector.started = true
collector.disabled = false
collector.paused = false
collector.session_id = "synthetic-test-session"
collector.queue = {}
collector.seq = 0
predict.setup({ synthetic = true, model = "stub", model_revision = "stub", url = "http://127.0.0.1:9" })

local function response(body, stop_type)
  local chunks = { "data: " .. vim.json.encode({ content = body, stop = false }),
    "data: " .. vim.json.encode({ content = "", stop = true, stop_type = stop_type or "eos" }) }
  predict._request_impl = function(_, _, callback)
    callback({ code = 0, stdout = table.concat(chunks, "\n") .. "\n" })
    return { kill = function() end }
  end
end

local file = vim.fn.tempname() .. ".py"
vim.fn.writefile({ "hello" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
vim.bo.filetype = "python"
vim.api.nvim_win_set_cursor(0, { 1, 0 })

response("R\nhi")
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().proposal_active end))
assert(predict.accept())
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "hi")

-- Insert at the cursor, then undo the single buffer change.
vim.opt.virtualedit = "onemore"
vim.api.nvim_win_set_cursor(0, { 1, 2 })
response("R\n!")
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().proposal_active end))
assert(predict.accept())
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "hi!")
vim.cmd("undo")
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "hi")

-- Delete the declared region, then restore it with undo.
vim.api.nvim_win_set_cursor(0, { 1, 0 })
response("R\n")
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().proposal_active end))
assert(predict.accept())
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "")
vim.cmd("undo")
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "hi")

response("R\n")
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().proposal_active end))
assert(predict.reject())
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "hi")

response("N\n")
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().state == "no_edit" end))
assert(not predict.status().proposal_active)

response("R\nlate")
assert(predict.predict())
vim.api.nvim_buf_set_lines(0, 0, 1, false, { "human" })
assert(vim.wait(1000, function() return not predict.status().in_flight end))
assert(not predict.status().proposal_active)
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "human")

-- A response after a file switch must not become a proposal.
local delayed
predict._request_impl = function(_, _, callback)
  delayed = callback
  return { kill = function() end }
end
assert(predict.predict())
local other = vim.fn.tempname() .. ".js"
vim.fn.writefile({ "const value = 1;" }, other)
vim.cmd("edit " .. vim.fn.fnameescape(other))
delayed({ code = 0, stdout = "data: " .. vim.json.encode({ content = "R\nwrong", stop = false })
  .. "\ndata: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos" }) .. "\n" })
assert(vim.wait(1000, function() return not predict.status().in_flight end))
assert(not predict.status().proposal_active)
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "const value = 1;")

-- Service failure leaves the buffer intact.
predict._request_impl = function(_, _, callback)
  callback({ code = 7, stdout = "" })
  return { kill = function() end }
end
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().state == "model-service outage" end))
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "const value = 1;")

local kinds = {}
for _, event in ipairs(collector.queue) do kinds[event.event_type] = (kinds[event.event_type] or 0) + 1 end
assert(kinds.prediction_requested == 8)
assert(kinds.prediction_shown == 4)
assert(kinds.prediction_accepted == 3)
assert(kinds.prediction_rejected == 1)
local lifecycle = {}
for _, event in ipairs(collector.queue) do
  if event.event_type == "heartbeat" and event.payload.prediction_lifecycle then
    lifecycle[event.payload.prediction_lifecycle] = (lifecycle[event.payload.prediction_lifecycle] or 0) + 1
  end
end
assert(lifecycle.no_edit == 1)
assert(lifecycle.cancelled == 1)
assert(lifecycle.stale_response == 1)
assert(lifecycle.transport_failure == 1)
assert(not predict.set_mode("automatic"))
assert(predict.status().mode == "manual")
predict.setup({ expiry_ms = 30 })
response("R\nx")
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().proposal_active end))
assert(vim.wait(1000, function() return predict.status().state == "expired" end))
assert(not predict.status().proposal_active)
local expired = false
for _, event in ipairs(collector.queue) do
  if event.event_type == "heartbeat" and event.payload.prediction_lifecycle == "expired" then
    expired = true
  end
end
assert(expired)
vim.api.nvim_buf_set_lines(0, 0, 1, false, { "αβ" })
vim.api.nvim_exec_autocmds("TextChanged", { buffer = 0 })
vim.api.nvim_buf_set_lines(0, 0, 1, false, { "αγ" })
vim.api.nvim_exec_autocmds("TextChanged", { buffer = 0 })
local captured_prompt
predict._request_impl = function(state, _, callback)
  captured_prompt = state.prompt
  callback({ code = 7, stdout = "" })
  return { kill = function() end }
end
vim.api.nvim_win_set_cursor(0, { 1, 0 })
assert(predict.predict())
assert(captured_prompt:find("<actual-recent-edit start=2 end=4>", 1, true))
assert(captured_prompt:find("αγ", 1, true))
vim.fn.delete(file)
vim.fn.delete(other)
print("predict headless safety checks passed")
