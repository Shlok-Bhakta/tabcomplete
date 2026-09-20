-- tabcomplete_trajectory plugin entry (Vim plugin/ directory).
-- LazyVim-compatible: this file only defines user commands. Collection starts
-- when the user calls require("tabcomplete_trajectory").setup({...}), e.g.:
--
--   { dir = ".../tools/trajectory_collector/nvim", opts = { server_url = "http://crabcake:8787" } }
--
-- Never auto-starts, never throws when the module is absent.
if vim.g.tabcomplete_trajectory_loaded then
  return
end
vim.g.tabcomplete_trajectory_loaded = 1

local function with_module(fn)
  local ok, mod = pcall(require, "tabcomplete_trajectory")
  if ok and mod then
    fn(mod)
  else
    vim.notify("[tabcomplete-trajectory] module not on runtimepath", vim.log.levels.WARN)
  end
end

local function safe_create(name, rhs, desc)
  pcall(vim.api.nvim_create_user_command, name, rhs, { desc = desc })
end

safe_create("TabCompleteCollectorStatus", function()
  with_module(function(m)
    m.define_commands()
    local s = m.status()
    vim.notify(
      string.format(
        "[tabcomplete-trajectory] active=%s session=%s queue=%d",
        tostring(s.active),
        tostring(s.session_id),
        s.queue_depth
      ),
      vim.log.levels.INFO
    )
  end)
end, "Show trajectory collector status")
safe_create("TabCompleteCollectorFlush", function()
  with_module(function(m)
    m.flush_now(function()
    end)
  end)
end, "Flush queued trajectory events now")
safe_create("TabCompleteCollectorRescan", function()
  with_module(function(m)
    m.rescan()
  end)
end, "Re-detect repo and re-anchor buffers")
safe_create("TabCompleteCollectorPause", function()
  with_module(function(m)
    m.pause()
  end)
end, "Pause trajectory collection")
safe_create("TabCompleteCollectorResume", function()
  with_module(function(m)
    m.resume()
  end)
end, "Resume trajectory collection")
safe_create("Flush", function()
  with_module(function(m)
    m.flush_now(function()
    end)
  end)
end, "Flush queued trajectory events now")
safe_create("RescanRepo", function()
  with_module(function(m)
    m.rescan()
  end)
end, "Re-detect repo and re-anchor buffers")
safe_create("Pause", function()
  with_module(function(m)
    m.pause()
  end)
end, "Pause trajectory collection")
safe_create("Resume", function()
  with_module(function(m)
    m.resume()
  end)
end, "Resume trajectory collection")
