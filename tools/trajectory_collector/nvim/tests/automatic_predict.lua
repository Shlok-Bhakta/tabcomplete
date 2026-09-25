-- Deterministic automatic-state and SSE checks. Decisions are synthetic.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local util = require("tabcomplete_trajectory.util")
local sse = require("tabcomplete_trajectory.sse")
local predict = require("tabcomplete_trajectory.predict")
collector.started, collector.disabled, collector.paused = true, false, false
collector.session_id, collector.queue, collector.seq = "synthetic-auto-test", {}, 0
buffers.set_callbacks(function(buf, kind, payload) return collector.emit(buf, kind, payload) end, function() end)
util.current_mode = function() return "i" end
local file = vim.fn.tempname() .. ".py"
vim.fn.writefile({ "x" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
vim.bo.filetype = "python"
assert(buffers.attach(vim.api.nvim_get_current_buf()))
predict.setup({ mode = "automatic", experimental_auto_opt_in = true, synthetic = true,
  debounce_ms = 30, expiry_ms = 1000, persist_mode = false, model = "stub" })
assert(predict.status().mode == "automatic")
assert(predict.status().experimental_auto_opt_in)
assert(not predict.status().automatic_quality_validated)
local callbacks, count = {}, 0
predict._request_impl = function(state, _, callback)
  count = count + 1
  callbacks[count] = { state = state, callback = callback }
  return { kill = function() end }
end
local function response(n, wire, stop_type)
  callbacks[n].callback({ code = 0, stdout = "data: "
    .. vim.json.encode({ content = wire, stop = false }) .. "\n\ndata: "
    .. vim.json.encode({ content = "", stop = true, stop_type = stop_type or "eos" }) .. "\n\n" })
end
local function changed(text)
  vim.api.nvim_buf_set_lines(0, 0, 1, false, { text })
  vim.api.nvim_exec_autocmds("TextChangedI", { buffer = 0 })
end
changed("ab")
vim.wait(10)
changed("abc")
assert(count == 0, "typing inside debounce sent a request")
assert(vim.wait(500, function() return count == 1 end), "pause did not send one request")
response(1, "N\n")
assert(vim.wait(500, function() return predict.status().state == "model_no_edit" end))
vim.wait(100)
assert(count == 1, "no-edit caused an idle request loop")
changed("abcd")
assert(vim.wait(500, function() return count == 2 end))
changed("abcde")
assert(vim.wait(500, function() return count == 2 end), "overlapping requests")
response(2, "R\nstale")
assert(vim.wait(500, function() return count == 3 end), "newest state was not requested")
assert(not predict.status().proposal_active)
response(3, "R\nhello")
assert(vim.wait(500, function() return predict.status().proposal_active end))
changed("diverged")
assert(not predict.status().proposal_active)
local implicit
for _, event in ipairs(collector.queue) do
  if event.event_type == "prediction_dismissed" and event.payload.outcome == "rejected_implicit_typing" then
    implicit = event
  end
end
assert(implicit and implicit.payload.ended_by_event_id, "divergent typing lacked linked delta")
assert(vim.wait(500, function() return count == 4 end))
response(4, "R\nmatched")
assert(vim.wait(500, function() return predict.status().proposal_active end))
changed("matched")
local match
for _, event in ipairs(collector.queue) do
  if event.event_type == "prediction_dismissed" and event.payload.outcome == "typed_match" then match = event end
end
assert(match and match.payload.ended_by_event_id, "typed match was treated as a negative")
assert(vim.wait(500, function() return count == 5 end))
response(5, "R\naccepted")
assert(vim.wait(500, function() return predict.status().proposal_active end))
assert(predict.accept())
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "accepted")
vim.cmd("undo")
assert(vim.api.nvim_buf_get_lines(0, 0, 1, false)[1] == "matched", "acceptance undo ate earlier typing")
assert(predict.set_mode("off"))
assert(predict.status().mode == "off")
changed("off")
vim.wait(100)
assert(count == 5, "off scheduled a request")
local parser = sse.new(4096)
local stream = 'data: {"content":"R\\nπ","stop":false}\r\n\r\ndata: {"content":"","stop":true,"stop_type":"eos"}\r\n\r\n'
for i = 1, #stream do assert(sse.feed(parser, stream:sub(i, i))) end
local action = assert(sse.finish(parser))
assert(action.action == "replace" and action.text == "π")
local bad = sse.new(4096)
assert(sse.feed(bad, 'data: {"content":"R\\na","stop":false}\n\n'))
assert(not sse.finish(bad), "incomplete action passed")
local cap = sse.new(4)
assert(not sse.feed(cap, "12345"), "oversize SSE passed")
local stats = predict.status().counters
assert(stats.requested == 5 and stats.displayed == 3 and stats.accepted == 1)
assert(stats.rejected_implicit_typing == 1 and stats.typed_match == 1)
assert(stats.cancelled_unseen >= 1 and stats.model_no_edit == 1)
-- Late callbacks, focus loss, transport failure and off during debounce.
predict.setup({ mode = "automatic", experimental_auto_opt_in = true, synthetic = true,
  debounce_ms = 30, persist_mode = false })
local late = {}
predict._request_impl = function(state, _, callback)
  late[#late + 1] = { state = state, callback = callback }
  return { kill = function() end }
end
changed("focus-test")
assert(vim.wait(500, function() return #late == 1 end))
vim.api.nvim_exec_autocmds("FocusLost", { buffer = 0 })
late[1].callback({ code = 0, stdout = "data: " .. vim.json.encode({ content = "R\nstale", stop = false })
  .. "\n\ndata: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos" }) .. "\n\n" })
vim.wait(50)
assert(not predict.status().proposal_active, "focus loss displayed a stale response")
vim.api.nvim_exec_autocmds("FocusGained", { buffer = 0 })
changed("offline-test")
assert(vim.wait(500, function() return #late == 2 end))
late[2].callback({ code = 7, stdout = "" })
assert(vim.wait(500, function() return predict.status().state:find("model request failed") end))
assert(not predict.status().proposal_active, "model outage displayed a proposal")
changed("current-test")
assert(vim.wait(500, function() return #late == 3 end))
late[3].callback({ code = 0, stdout = "data: " .. vim.json.encode({ content = "R\nvalid", stop = false })
  .. "\n\ndata: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos" }) .. "\n\n" })
assert(vim.wait(500, function() return predict.status().proposal_active end))
late[1].callback({ code = 0, stdout = "data: " .. vim.json.encode({ content = "R\nlate", stop = false })
  .. "\n\ndata: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos" }) .. "\n\n" })
vim.wait(30)
assert(predict.status().proposal_active, "old callback disturbed current proposal")
for index = 4, 6 do
  changed("failure-" .. index)
  assert(vim.wait(500, function() return #late == index end))
  local failures_before = predict.status().counters.request_failed
  late[index].callback({ code = 7, stdout = "" })
  assert(vim.wait(500, function()
    return predict.status().counters.request_failed == failures_before + 1
  end))
end
assert(predict.status().backoff_until_ms > util.now_ms(), "persistent failures did not back off")
changed("during-backoff")
vim.wait(100)
assert(#late == 6, "backoff sent another model request")
assert(predict.set_mode("off"))
assert(not predict.status().proposal_active)
changed("off-timer-test")
vim.wait(100)
assert(#late == 6, "off allowed a waiting timer to send a request")
vim.fn.delete(file)
print("automatic prediction state machine and SSE passed")
