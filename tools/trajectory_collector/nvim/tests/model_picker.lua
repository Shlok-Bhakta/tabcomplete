-- Command surface only. Stub model switching; no network or model-quality claim.
local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
vim.opt.rtp:prepend(root)
vim.cmd("runtime plugin/tabcomplete_predict.lua")
local predict = require("tabcomplete_trajectory.predict")
local selected, offered
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
print("model picker command checks passed")
