-- Command surface only. Stub model switching; no network or model-quality claim.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
vim.cmd("runtime plugin/tabcomplete_predict.lua")
local predict = require("tabcomplete_trajectory.predict")
local selected, offered
local original_set_model = predict.set_model
predict.set_model = function(alias, callback)
  selected = alias
  callback(true)
  return true
end
vim.ui.select = function(items, options, callback)
  offered = items
  assert(options.prompt == "TabComplete model")
  callback("sweep")
end
vim.cmd("TabCompleteModel")
assert(#offered == 2 and offered[1] == "q25" and offered[2] == "sweep")
assert(selected == "sweep")
vim.cmd("TabCompleteModel q25")
assert(selected == "q25")
local aliases = vim.fn.getcompletion("TabCompleteModel ", "cmdline")
assert(vim.tbl_contains(aliases, "q25") and vim.tbl_contains(aliases, "sweep"))

local notifications = {}
vim.notify = function(message, level)
  notifications[#notifications + 1] = { message = message, level = level }
end
offered = nil
vim.ui.select = function()
  error("fixed-model picker must not offer unavailable aliases")
end
predict.set_model = original_set_model
predict.setup({ backend = "rust-editor-v1", mode = "manual", url = "http://127.0.0.1:19094",
  allowed_models = {
    q25 = { model_sha256 = string.rep("a", 64), model_protocol = "single-line-edit-v1", output_tokens = 64 },
    sweep = { model_sha256 = string.rep("b", 64), model_protocol = "sweep-full-file-v1", output_tokens = 192 },
  } })
local post_calls = 0
local get_calls = 0
predict._backend_post_impl = function()
  post_calls = post_calls + 1
  error("declarative model selection must not call the model-switch endpoint")
end
predict._backend_get_impl = function(path, callback)
  get_calls = get_calls + 1
  assert(path == "/health")
  callback(true, { status = "ok", alias = "q25", model_sha256 = string.rep("a", 64),
    model_protocol = "single-line-edit-v1", context_layout = "cursor-last-v1",
    runtime_config_hash = string.rep("c", 64), input_tokens = 1024, output_tokens = 64,
    context_size = 2304, model_embedded = true, model_switch_supported = false,
    model_selection = "declarative", model_storage = "executable-mmap" })
  return { kill = function() end }
end
vim.cmd("TabCompleteStatus")
assert(vim.wait(1000, function() return #notifications == 1 end), "status loaded the installed model identity")
assert(notifications[1].message:find('selected_model = "q25"', 1, true))
assert(notifications[1].message:find('model_selection = "declarative"', 1, true))
vim.cmd("TabCompleteModel")
assert(#notifications == 2, "fixed model selection is reported")
assert(not offered, "fixed model did not open an alternate-model picker")
local fixed_status = predict.status()
assert(fixed_status.selected_model == "q25")
assert(fixed_status.model_switch_supported == false)
assert(fixed_status.model_selection == "declarative")
assert(fixed_status.model_storage == "executable-mmap")
assert(notifications[2].message:find("Nix model variant", 1, true))
local fixed_aliases = predict.model_aliases()
assert(#fixed_aliases == 1 and fixed_aliases[1] == "q25")
local fixed_completion = vim.fn.getcompletion("TabCompleteModel ", "cmdline")
assert(#fixed_completion == 1 and fixed_completion[1] == "q25")
vim.cmd("TabCompleteModel sweep")
assert(#notifications == 3 and notifications[3].message:find("rebuild the package", 1, true))
assert(get_calls == 1, "model status and picker share the loaded health identity")
assert(post_calls == 0, "fixed model selection does not try an external model registry")
print("model picker command checks passed")
