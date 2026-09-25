-- Synthetic external-holder test for two Neovim sessions sharing one model slot.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local util = require("tabcomplete_trajectory.util")
local predict = require("tabcomplete_trajectory.predict")
collector.started, collector.disabled, collector.paused = true, false, false
collector.session_id, collector.queue, collector.seq = "synthetic-shared-slot", {}, 0
buffers.set_callbacks(function(buf, kind, payload) return collector.emit(buf, kind, payload) end,
  function() end)
util.current_mode = function() return "i" end
local file = vim.fn.tempname() .. ".py"
vim.fn.writefile({ "def synthetic():", "    return 1" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
assert(buffers.attach(vim.api.nvim_get_current_buf()))
predict.setup({ mode = "automatic", experimental_auto_opt_in = true, synthetic = true,
  debounce_ms = 30, persist_mode = false })
vim.api.nvim_buf_set_lines(0, 1, 2, false, { "    return 2" })
vim.api.nvim_win_set_cursor(0, { 2, 4 })
vim.api.nvim_exec_autocmds("TextChangedI", { buffer = 0 })
vim.wait(200)
local function requests()
  local n = 0
  for _, event in ipairs(collector.queue) do
    if event.event_type == "prediction_requested" then n = n + 1 end
  end
  return n
end
assert(requests() == 0 and predict.status().state == "waiting for shared service slot",
  "request escaped occupied cross-process slot")
-- Release the external holder on a timer, allowing the Neovim event loop to
-- prove that one latest-state retry starts after the slot becomes free.
local release = vim.uv.new_timer()
release:start(500, 0, function()
  vim.schedule(function()
    os.remove("/tmp/tabcomplete-predictor-19093.lock/owner")
    vim.uv.fs_rmdir("/tmp/tabcomplete-predictor-19093.lock")
    release:close()
  end)
end)
assert(vim.wait(5000, function() return requests() == 1 end, 20),
  "latest state was not sent after the shared slot released: "
    .. vim.inspect({ status = predict.status(), requests = requests() }))
assert(requests() == 1, "shared-slot wait queued duplicate requests")
predict.set_mode("off")
vim.fn.delete(file)
print("shared slot arbitration passed")
