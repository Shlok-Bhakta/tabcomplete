local predict = require("tabcomplete_trajectory.predict").setup()

vim.api.nvim_create_user_command("TabCompletePredict", function()
  local ok, err = predict.predict()
  if not ok then vim.notify("TabComplete: " .. tostring(err), vim.log.levels.WARN) end
end, {})
vim.api.nvim_create_user_command("TabCompleteAccept", function()
  local ok, err = predict.accept()
  if not ok then vim.notify("TabComplete: " .. tostring(err), vim.log.levels.WARN) end
end, {})
vim.api.nvim_create_user_command("TabCompleteReject", function()
  local ok, err = predict.reject()
  if not ok then vim.notify("TabComplete: " .. tostring(err), vim.log.levels.WARN) end
end, {})
vim.api.nvim_create_user_command("TabCompleteMode", function(args)
  if args.args == "" then
    vim.notify("TabComplete mode: " .. predict.status().mode)
    return
  end
  local ok, err = predict.set_mode(args.args)
  if not ok then vim.notify("TabComplete: " .. tostring(err), vim.log.levels.WARN) end
end, { nargs = "?", complete = function() return { "manual", "automatic", "shadow", "off" } end })
vim.api.nvim_create_user_command("TabCompleteStatus", function()
  vim.notify(vim.inspect(predict.status()))
end, {})
