-- Headless safety checks with a stubbed model response. No quality claim.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local collector = require("tabcomplete_trajectory")
local predict = require("tabcomplete_trajectory.predict")
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

local kinds = {}
for _, event in ipairs(collector.queue) do kinds[event.event_type] = (kinds[event.event_type] or 0) + 1 end
assert(kinds.prediction_requested == 4)
assert(kinds.prediction_shown == 2)
assert(kinds.prediction_accepted == 1)
assert(kinds.prediction_rejected == 1)
assert(not predict.set_mode("automatic"))
assert(predict.status().mode == "manual")
vim.fn.delete(file)
print("predict headless safety checks passed")
