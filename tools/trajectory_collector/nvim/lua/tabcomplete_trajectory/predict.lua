-- Local next-edit preview. Uses the existing trajectory collector for evidence.
local M = {}
local util = require("tabcomplete_trajectory.util")
local collector = require("tabcomplete_trajectory")

local ns = vim.api.nvim_create_namespace("TabCompletePredict")
local allowed_modes = { manual = true, automatic = true, shadow = true, off = true }
local mode = "manual"
local opts = {
  url = vim.env.TABCOMPLETE_PREDICTOR_URL or "http://127.0.0.1:19093",
  model = "unconfigured",
  model_revision = "unconfigured",
  precision = "Q4_K_M",
  adapter_identity = "none",
  debounce_ms = 200,
  expiry_ms = 30000,
  synthetic = false,
  automatic_gates_passed = false,
}
local generation = 0
local pending = nil
local active = nil
local timer = nil
local expiry_timer = nil
local applying = false
local last_status = "idle"
local group = nil
local tracked_content = {}
local recent_edit = {}
M._request_impl = nil -- headless integration seam; never used by the installed client

local function utf8_boundary(line, col)
  if col < 0 or col > #line then return false end
  if col == #line then return true end
  local byte = line:byte(col + 1)
  return byte < 0x80 or byte >= 0xC0
end

local function content_of(bufnr)
  if not vim.api.nvim_buf_is_valid(bufnr) then return nil end
  local lines = vim.api.nvim_buf_get_lines(bufnr, 0, -1, false)
  local content = table.concat(lines, "\n")
  if vim.bo[bufnr].endofline then content = content .. "\n" end
  return content
end

local function capture_actual_edit(bufnr)
  local after = content_of(bufnr)
  if not after then return end
  local before = tracked_content[bufnr]
  tracked_content[bufnr] = after
  if not before or before == after then return end
  local prefix = 0
  while prefix < math.min(#before, #after) and before:byte(prefix + 1) == after:byte(prefix + 1) do
    prefix = prefix + 1
  end
  while prefix > 0 and ((before:byte(prefix + 1) or 0) >= 0x80
      and (before:byte(prefix + 1) or 0) < 0xC0
      or (after:byte(prefix + 1) or 0) >= 0x80 and (after:byte(prefix + 1) or 0) < 0xC0) do
    prefix = prefix - 1
  end
  local suffix = 0
  while suffix < math.min(#before - prefix, #after - prefix)
      and before:byte(#before - suffix) == after:byte(#after - suffix) do
    suffix = suffix + 1
  end
  local old_end, new_end = #before - suffix, #after - suffix
  while suffix > 0 and (not utf8_boundary(before, old_end)
      or not utf8_boundary(after, new_end)) do
    suffix = suffix - 1
    old_end, new_end = #before - suffix, #after - suffix
  end
  local inserted = after:sub(prefix + 1, new_end)
  if before:sub(1, prefix) .. inserted .. before:sub(old_end + 1) ~= after then return end
  recent_edit[bufnr] = { start_byte = prefix, end_byte = old_end, text = inserted }
end

local function close_preview()
  if active and active.preview_win and vim.api.nvim_win_is_valid(active.preview_win) then
    vim.api.nvim_win_close(active.preview_win, true)
  end
  if active and vim.api.nvim_buf_is_valid(active.bufnr) then
    vim.api.nvim_buf_clear_namespace(active.bufnr, ns, 0, -1)
  end
end

local function lifecycle(state, outcome, reason)
  if not state then return end
  collector.emit(state.bufnr, "heartbeat", {
    prediction_id = state.prediction_id, prediction_lifecycle = outcome,
    reason = reason, recorded_at_ms = util.now_ms(), synthetic = opts.synthetic,
  })
end

local function clear(reason)
  generation = generation + 1
  if expiry_timer then expiry_timer:stop() end
  if pending then lifecycle(pending.state, "cancelled", reason or "cleared") end
  if active and reason ~= "accepted" and reason ~= "explicitly rejected" then
    lifecycle(active, reason == "expired" and "expired" or "invalidated", reason or "cleared")
  end
  if pending and pending.process then
    pcall(pending.process.kill, pending.process, "sigterm")
  end
  pending = nil
  close_preview()
  active = nil
  last_status = reason or "idle"
end

local function buffer_state()
  local bufnr = vim.api.nvim_get_current_buf()
  if util.is_excluded_buffer(bufnr) or not vim.api.nvim_buf_is_valid(bufnr) then
    return nil, "excluded buffer"
  end
  local path = vim.api.nvim_buf_get_name(bufnr)
  if path == "" then
    return nil, "unnamed buffer"
  end
  local cursor = vim.api.nvim_win_get_cursor(0)
  local row, col = cursor[1] - 1, cursor[2]
  local line = vim.api.nvim_buf_get_lines(bufnr, row, row + 1, false)[1]
  if not line or not utf8_boundary(line, col) then
    return nil, "cursor is not on a UTF-8 boundary"
  end
  local lines = vim.api.nvim_buf_get_lines(bufnr, 0, -1, false)
  local content = content_of(bufnr)
  local before = table.concat(vim.list_slice(lines, math.max(1, row - 79), row), "\n")
  if before ~= "" then before = before .. "\n" end
  before = before .. line:sub(1, col)
  local after_lines = vim.list_slice(lines, row + 2, math.min(#lines, row + 41))
  local after = line:sub(#line + 1)
  if #after_lines > 0 then after = after .. "\n" .. table.concat(after_lines, "\n") end
  local region = line:sub(col + 1)
  local prior_lines = table.concat(vim.list_slice(lines, 1, row), "\n")
  if row > 0 then prior_lines = prior_lines .. "\n" end
  local region_start = #prior_lines + col
  local filetype = vim.bo[bufnr].filetype
  local history = "<recent-edit unavailable>\n"
  local latest = recent_edit[bufnr]
  if latest then
    history = "<actual-recent-edit start=" .. latest.start_byte .. " end=" .. latest.end_byte
      .. ">\n" .. latest.text .. "\n</actual-recent-edit>\n"
  end
  local prompt = "<repo " .. path .. ">\n<filetype " .. filetype .. ">\n"
    .. history .. "<file " .. path .. ">\n" .. before .. "[[EDIT]]" .. region .. "[[/EDIT]]"
    .. after .. "\n</file>\n<P " .. path .. " " .. region_start .. ">\n"
    .. "Return one compact next-edit action: N\\n for no edit or R\\n followed by exact replacement text. End with EOS.\n"
  return {
    bufnr = bufnr, path = path, row = row, start_col = col, end_col = #line,
    region = region, changedtick = vim.api.nvim_buf_get_changedtick(bufnr),
    content_hash = util.sha256hex(content), prompt = prompt,
    context_hash = util.sha256hex(prompt), requested_at_ms = util.now_ms(),
    prediction_id = util.uuid(), filetype = filetype,
  }
end

local function still_current(state)
  if vim.api.nvim_get_current_buf() ~= state.bufnr or not vim.api.nvim_buf_is_valid(state.bufnr) then
    return false
  end
  if vim.api.nvim_buf_get_name(state.bufnr) ~= state.path then return false end
  if vim.api.nvim_buf_get_changedtick(state.bufnr) ~= state.changedtick then return false end
  local line = vim.api.nvim_buf_get_lines(state.bufnr, state.row, state.row + 1, false)[1]
  if not line or line:sub(state.start_col + 1, state.end_col) ~= state.region then return false end
  if not utf8_boundary(line, state.start_col) or not utf8_boundary(line, state.end_col) then
    return false
  end
  local lines = vim.api.nvim_buf_get_lines(state.bufnr, 0, -1, false)
  local content = table.concat(lines, "\n")
  if vim.bo[state.bufnr].endofline then content = content .. "\n" end
  return util.sha256hex(content) == state.content_hash
end

local function parse_sse(raw)
  local pieces, final = {}, nil
  for line in raw:gmatch("[^\r\n]+") do
    if line:sub(1, 6) == "data: " then
      local ok, event = pcall(vim.json.decode, line:sub(7))
      if ok and type(event) == "table" then
        if event.stop then final = event else pieces[#pieces + 1] = event.content or "" end
      end
    end
  end
  if not final then return nil, "missing completion event" end
  if final.stop_type ~= "eos" then return nil, "incomplete: " .. tostring(final.stop_type) end
  local raw_action = table.concat(pieces)
  if raw_action == "N\n" then return { action = "no_edit" }, nil end
  if raw_action:sub(1, 2) == "R\n" then
    return { action = "replace", text = raw_action:sub(3) }, nil
  end
  return nil, "malformed compact action"
end

local function show(state, action)
  if action.action == "no_edit" then
    lifecycle(state, "no_edit", "explicit model no-edit")
    last_status = "no_edit"
    return
  end
  if action.text == state.region then
    lifecycle(state, "unchanged", "replacement matches editable region")
    last_status = "unchanged replacement"
    return
  end
  state.action = action
  state.shown_at_ms = util.now_ms()
  active = state
  local multiline = action.text:find("\n", 1, true) ~= nil
  if multiline then
    local lines = { "TabComplete replacement preview", "Original: " .. state.region,
      "Proposed:" }
    vim.list_extend(lines, vim.split(action.text, "\n", { plain = true }))
    local scratch = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_lines(scratch, 0, -1, false, lines)
    vim.bo[scratch].modifiable = false
    state.preview_win = vim.api.nvim_open_win(scratch, false, {
      relative = "cursor", row = 1, col = 0, width = math.min(100, math.max(35, vim.o.columns - 8)),
      height = math.min(#lines, 12), style = "minimal", border = "single", focusable = false,
    })
  else
    local label = state.region == "" and action.text or ("[replace: " .. action.text .. "]")
    if action.text == "" then label = "[delete: " .. state.region .. "]" end
    vim.api.nvim_buf_set_extmark(state.bufnr, ns, state.row, state.start_col, {
      virt_text = { { label, "Comment" } }, virt_text_pos = "eol", hl_mode = "combine",
    })
  end
  collector.log_prediction_shown({
    prediction_id = state.prediction_id, proposed_start = { row = state.row, col = state.start_col },
    proposed_end = { row = state.row, col = state.end_col }, proposed_text = action.text,
    action = action.action, shown_at_ms = state.shown_at_ms, context_hash = state.context_hash,
    pre_state_hash = state.content_hash, synthetic = opts.synthetic,
  })
  if not expiry_timer then expiry_timer = vim.uv.new_timer() end
  expiry_timer:stop()
  local prediction_id = state.prediction_id
  expiry_timer:start(opts.expiry_ms, 0, function()
    vim.schedule(function()
      if active and active.prediction_id == prediction_id then clear("expired") end
    end)
  end)
  last_status = "proposal shown"
end

function M.predict()
  if mode == "off" then return false, "mode off" end
  clear("requesting")
  local state, err = buffer_state()
  if not state then last_status = err; return false, err end
  local current_generation = generation
  collector.log_prediction_requested({
    prediction_id = state.prediction_id, provider = "tabcomplete-local",
    model = opts.model, model_revision = opts.model_revision, precision = opts.precision,
    adapter_identity = opts.adapter_identity, requested_at_ms = state.requested_at_ms,
    context_hash = state.context_hash, pre_state_hash = state.content_hash,
    max_output_tokens = 96, temperature = 0, wire_version = "compact-next-edit-v1",
    synthetic = opts.synthetic, human_verified = not opts.synthetic, mode = mode,
  })
  local body = vim.json.encode({ prompt = state.prompt, n_predict = 96, temperature = 0,
    stream = true, cache_prompt = true, id_slot = 0 })
  local function finished(result)
      vim.schedule(function()
        if current_generation ~= generation or not pending or pending.state ~= state then return end
        pending = nil
        if result.code ~= 0 then
          lifecycle(state, result.code == 28 and "timeout" or "transport_failure",
            "model service returned exit code " .. tostring(result.code))
          last_status = "model-service outage"
          return
        end
        if not still_current(state) then
          lifecycle(state, "stale_response", "editor state changed before response")
          last_status = "stale response dropped"
          return
        end
        local action, parse_err = parse_sse(result.stdout or "")
        if not action then
          lifecycle(state, "invalid_output", parse_err)
          last_status = parse_err
          return
        end
        state.responded_at_ms = util.now_ms()
        if mode == "shadow" then
          lifecycle(state, "shadow_completed", "proposal was not displayed")
          last_status = "shadow response"
          return
        end
        show(state, action)
      end)
  end
  local process
  if M._request_impl then
    process = M._request_impl(state, body, finished)
  else
    process = vim.system({ "curl", "-sS", "-N", "-m", "30", "-X", "POST",
      "-H", "Content-Type: application/json", "--data-binary", "@-",
      opts.url .. "/completion" }, { stdin = body, text = true }, finished)
  end
  pending = { process = process, state = state }
  return true
end

function M.accept()
  if not active or not active.action then return false, "no active proposal" end
  local state = active
  if not still_current(state) then clear("stale proposal rejected"); return false, "stale proposal" end
  local action = state.action
  if action.action ~= "replace" then return false, "no edit" end
  local replacement = vim.split(action.text, "\n", { plain = true })
  -- Setting the same local value closes any prior undo block without erasing it.
  vim.bo[state.bufnr].undolevels = vim.bo[state.bufnr].undolevels
  applying = true
  vim.api.nvim_buf_set_text(state.bufnr, state.row, state.start_col, state.row, state.end_col,
    replacement)
  tracked_content[state.bufnr] = content_of(state.bufnr)
  collector.log_prediction_accepted({ prediction_id = state.prediction_id,
    accepted_chars = vim.fn.strchars(action.text), accepted_lines = #replacement,
    total_chars = vim.fn.strchars(action.text), synthetic = opts.synthetic,
    accepted_at_ms = util.now_ms() })
  applying = false
  clear("accepted")
  return true
end

function M.reject()
  if not active then return false, "no active proposal" end
  local prediction_id = active.prediction_id
  collector.log_prediction_rejected({ prediction_id = prediction_id,
    finish_reason = "explicit_user_reject", synthetic = opts.synthetic,
    rejected_at_ms = util.now_ms() })
  clear("explicitly rejected")
  return true
end

function M.set_mode(next_mode)
  if not allowed_modes[next_mode] then return false, "invalid mode" end
  if next_mode == "automatic" and not opts.automatic_gates_passed then
    return false, "automatic mode gates have not passed"
  end
  clear("mode changed")
  mode = next_mode
  return true
end

function M.status()
  return { mode = mode, state = last_status, model = opts.model,
    revision = opts.model_revision, precision = opts.precision,
    in_flight = pending ~= nil, proposal_active = active ~= nil,
    automatic_gates_passed = opts.automatic_gates_passed,
    automatic_personalization_enabled = false }
end

function M.setup(options)
  opts = vim.tbl_deep_extend("force", opts, options or {})
  clear("idle")
  if group then pcall(vim.api.nvim_del_augroup_by_id, group) end
  group = vim.api.nvim_create_augroup("TabCompletePredict", { clear = true })
  for _, bufnr in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_is_loaded(bufnr) then tracked_content[bufnr] = content_of(bufnr) end
  end
  vim.api.nvim_create_autocmd({ "TextChanged", "TextChangedI" }, {
    group = group, callback = function(args)
      if not applying then
        capture_actual_edit(args.buf)
        clear("invalidated by editor change")
      end
    end,
  })
  vim.api.nvim_create_autocmd({ "BufLeave", "BufWipeout" }, {
    group = group, callback = function(args)
      clear("invalidated by file switch")
      if args.event == "BufWipeout" then
        tracked_content[args.buf] = nil
        recent_edit[args.buf] = nil
      end
    end,
  })
  vim.api.nvim_create_autocmd("BufEnter", {
    group = group, callback = function(args)
      tracked_content[args.buf] = content_of(args.buf)
    end,
  })
  vim.api.nvim_create_autocmd({ "TextChangedI", "CursorMovedI" }, {
    group = group, callback = function()
      if mode ~= "automatic" then return end
      if not timer then timer = vim.uv.new_timer() end
      timer:stop()
      timer:start(opts.debounce_ms, 0, function() vim.schedule(M.predict) end)
    end,
  })
  return M
end

return M
