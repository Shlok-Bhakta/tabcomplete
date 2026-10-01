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
vim.api.nvim_create_user_command("TabCompleteReviewLast", function()
  local ok, err = predict.review_last()
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
vim.api.nvim_create_user_command("TabCompleteModel", function(args)
  local function select(alias)
    if not alias then return end
    local ok, err = predict.set_model(alias, function(switched, message)
      if not switched then
        vim.notify("TabComplete: " .. tostring(message), vim.log.levels.WARN)
      else
        vim.notify("TabComplete model: " .. alias)
      end
    end)
    if not ok then vim.notify("TabComplete: " .. tostring(err), vim.log.levels.WARN) end
  end
  if args.args == "" then
    vim.ui.select(predict.model_aliases(), { prompt = "TabComplete model" }, select)
  else
    select(args.args)
  end
end, { nargs = "?", complete = function() return predict.model_aliases() end })
vim.api.nvim_create_user_command("TabCompleteStatus", function()
  vim.notify(vim.inspect(predict.status()))
end, {})
