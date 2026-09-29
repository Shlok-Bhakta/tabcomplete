-- Disposable in-memory editor test; no live collector or human feedback.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local predict = require("tabcomplete_trajectory.predict")
collector.started, collector.disabled, collector.paused = true, false, false
collector.session_id, collector.queue, collector.seq = "synthetic-review-test", {}, 0
buffers.set_callbacks(function(buf, kind, payload) return collector.emit(buf, kind, payload) end,
  function() end)
local file = vim.fn.tempname() .. ".py"
vim.fn.writefile({ "value = 1" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
vim.bo.filetype = "python"
assert(buffers.attach(vim.api.nvim_get_current_buf()))
predict.setup({ mode = "manual", synthetic = false, persist_mode = false, model = "stub" })
vim.cmd("runtime plugin/tabcomplete_predict.lua")
assert(vim.fn.exists(":TabCompleteReviewLast") == 2)
predict._request_impl = function(_, _, callback)
  vim.schedule(function()
    callback({ code = 0, stdout = "data: " .. vim.json.encode({ content = "R\nvalue = 2", stop = false })
      .. "\n\ndata: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos" })
      .. "\n\n" })
  end)
  return { kill = function() end }
end
assert(predict.predict())
assert(vim.wait(1000, function() return predict.status().proposal_active end))
assert(predict.accept())
predict._confirm_impl = function() return 2 end
assert(not predict.review_last(), "cancelled confirmation recorded a review")
predict._confirm_impl = function() return 1 end
vim.cmd("TabCompleteReviewLast")
assert(not predict.review_last(), "duplicate confirmation recorded")
local accepted, reviewed = nil, nil
for _, event in ipairs(collector.queue) do
  if event.event_type == "prediction_accepted" then accepted = event end
  if event.event_type == "prediction_reviewed" then reviewed = event end
end
assert(accepted and reviewed)
assert(reviewed.payload.resolution_event_id == accepted.event_id)
assert(reviewed.payload.prediction_id == accepted.payload.prediction_id)
assert(reviewed.payload.human_verified == true and reviewed.payload.synthetic == false)
assert(reviewed.sequence_number > accepted.sequence_number)
vim.fn.delete(file)
print("explicit review event linkage passed (scripted test only)")
