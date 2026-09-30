-- Synthetic two-process arbitration. All model transport is stubbed, and the
-- copied module uses a private test lock. Never touch the running predictor.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
local peer = vim.env.TABCOMPLETE_SLOT_TEST_PEER == "1"
local test_root = vim.env.TABCOMPLETE_SLOT_TEST_ROOT or (vim.fn.tempname() .. "-slot-test")
if not peer then assert(vim.uv.fs_mkdir(test_root, 448)) end
local lock_path = test_root .. "/slot.lock"
local marker = test_root .. "/peer-running"
local source = table.concat(vim.fn.readfile(root .. "/lua/tabcomplete_trajectory/predict.lua"), "\n")
local original = 'local lock_path = "/tmp/tabcomplete-predictor-19093.lock"'
local first, last = source:find(original, 1, true)
assert(first and not source:find(original, last + 1, true), "lock declaration changed")
source = source:sub(1, first - 1) .. "local lock_path = " .. string.format("%q", lock_path)
  .. source:sub(last + 1)
package.loaded["tabcomplete_trajectory.predict"] = assert(loadstring(source, "@synthetic-slot-copy"))()
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local util = require("tabcomplete_trajectory.util")
local predict = require("tabcomplete_trajectory.predict")
collector.started, collector.disabled, collector.paused = true, false, false
collector.session_id, collector.queue, collector.seq = peer and "synthetic-peer" or "synthetic-parent", {}, 0
collector.store_prediction_blob = function(_, callback) if callback then callback(true) end end
buffers.set_callbacks(function(buf, kind, payload) return collector.emit(buf, kind, payload) end,
  function() end)
util.current_mode = function() return "i" end
local native_system = vim.system
local generated = 0
vim.system = function(argv, options, callback)
  if argv[1] ~= "curl" then return native_system(argv, options, callback) end
  local endpoint = argv[#argv]
  assert(endpoint:find("http://127.0.0.1:1/", 1, true), "unexpected model endpoint")
  local killed = false
  if endpoint:match("/tokenize$") then
    vim.defer_fn(function()
      callback({ code = killed and 143 or 0, stdout = vim.json.encode({ tokens = { 1, 2, 3 } }) })
    end, 10)
  else
    assert(endpoint:match("/completion$"), "unexpected transport request")
    generated = generated + 1
    if peer then vim.fn.writefile({ "synthetic generation running" }, marker) end
    vim.defer_fn(function()
      if not killed then
        options.stdout(nil, "data: " .. vim.json.encode({ content = "N\n", stop = false })
          .. "\n\ndata: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos" }) .. "\n\n")
      end
      callback({ code = killed and 143 or 0, stdout = "" })
    end, peer and 1500 or 20)
  end
  return { kill = function() killed = true end }
end
local file = test_root .. (peer and "/peer.py" or "/parent.py")
vim.fn.writefile({ "def synthetic():", "    return 1" }, file)
vim.cmd("edit " .. vim.fn.fnameescape(file))
vim.bo.filetype = "python"
assert(buffers.attach(vim.api.nvim_get_current_buf()))
predict.setup({ mode = "automatic", experimental_auto_opt_in = true, synthetic = true,
  model = "stub", url = "http://127.0.0.1:1", debounce_ms = 30, persist_mode = false })
local function changed(text)
  vim.api.nvim_buf_set_lines(0, 1, 2, false, { text })
  vim.api.nvim_win_set_cursor(0, { 2, 4 })
  vim.api.nvim_exec_autocmds("TextChangedI", { buffer = 0 })
end
local function requests()
  local n = 0
  for _, event in ipairs(collector.queue) do
    if event.event_type == "prediction_requested" then n = n + 1 end
  end
  return n
end
if peer then
  changed("    return 2")
  assert(vim.wait(5000, function() return predict.status().state == "model_no_edit" end, 20))
  assert(requests() == 1 and generated == 1)
  assert(not vim.uv.fs_stat(lock_path), "peer failed to release its own lock")
  predict.set_mode("off")
  print("synthetic peer passed")
  return
end
local finished
local process = native_system({ vim.v.progpath, "--headless", "-u", "NONE", "-l",
  root .. "/tests/automatic_shared_slot.lua" },
  { text = true, env = { TABCOMPLETE_SLOT_TEST_PEER = "1", TABCOMPLETE_SLOT_TEST_ROOT = test_root } },
  function(result) finished = result end)
assert(vim.wait(3000, function() return vim.uv.fs_stat(marker) ~= nil end, 20), "peer never held slot")
changed("    return 2")
vim.wait(200)
assert(requests() == 0 and predict.status().state == "waiting for shared service slot",
  "request escaped occupied cross-process slot")
changed("    return 3")
assert(vim.wait(6000, function() return requests() == 1 and predict.status().state == "model_no_edit" end, 20),
  "latest state was not served after peer released slot")
assert(requests() == 1 and generated == 1, "shared-slot wait queued duplicate requests")
assert(vim.wait(2000, function() return finished ~= nil end, 20), "peer did not exit")
assert(finished.code == 0, "synthetic peer failed")
for _, event in ipairs(collector.queue) do
  if event.event_type == "prediction_requested" then
    assert(event.payload.pre_state_hash == util.sha256hex("def synthetic():\n    return 3\n"),
      "request used obsolete waiting state")
  end
end
predict.set_mode("off")
assert(not vim.uv.fs_stat(lock_path), "parent failed to release its own lock")
vim.system = native_system
vim.fn.delete(test_root, "rf")
print("synthetic two-process shared slot arbitration passed")
