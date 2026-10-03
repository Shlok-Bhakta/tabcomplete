local predict = require("tabcomplete_trajectory.predict").setup()

vim.api.nvim_create_user_command("TabCompletePredict", function()
  predict.predict_explicit()
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
  local function with_identity(callback)
    local status = predict.status()
    if status.backend ~= "rust-editor-v1" or status.model_switch_supported ~= nil then
      callback(status)
      return
    end
    local started = predict.refresh_model_identity(function(ok, result)
      if not ok then
        vim.notify("TabComplete: " .. tostring(result), vim.log.levels.WARN)
        return
      end
      callback(predict.status())
    end)
    if not started then vim.notify("TabComplete: model identity unavailable", vim.log.levels.WARN) end
  end
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
    with_identity(function(status)
      if status.model_switch_supported == false then
        vim.notify("TabComplete selected model: " .. tostring(status.selected_model)
          .. " (declarative; choose a Nix model variant and rebuild)")
        return
      end
      vim.ui.select(predict.model_aliases(), { prompt = "TabComplete model" }, select)
    end)
  else
    local alias = args.args
    with_identity(function(status)
      if status.model_switch_supported == false then
        local ok, err = predict.set_model(alias)
        if not ok then vim.notify("TabComplete: " .. tostring(err), vim.log.levels.WARN) end
        return
      end
      select(alias)
    end)
  end
end, { nargs = "?", complete = function()
  local status = predict.status()
  if status.backend == "rust-editor-v1" and status.model_switch_supported == nil then return {} end
  return predict.model_aliases()
end })
vim.api.nvim_create_user_command("TabCompleteStatus", function()
  local function report()
    vim.notify(vim.inspect(predict.status()))
  end
  local status = predict.status()
  if status.backend == "rust-editor-v1" and status.model_switch_supported == nil then
    local started = predict.refresh_model_identity(function(ok, result)
      if not ok then
        vim.notify("TabComplete: " .. tostring(result), vim.log.levels.WARN)
        return
      end
      report()
    end)
    if not started then vim.notify("TabComplete: model identity unavailable", vim.log.levels.WARN) end
    return
  end
  report()
end, {})
