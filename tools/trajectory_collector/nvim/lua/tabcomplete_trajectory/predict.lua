-- Local next-edit suggestions. Inference is automatic only after explicit opt-in;
-- every buffer mutation still requires an acceptance key.
local M = {}
local util = require("tabcomplete_trajectory.util")
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local repository = require("tabcomplete_trajectory.repository")
local sse = require("tabcomplete_trajectory.sse")

local ns = vim.api.nvim_create_namespace("TabCompletePredict")
local allowed_modes = { manual = true, automatic = true, shadow = true, off = true }
local opts = {
  url = vim.env.TABCOMPLETE_PREDICTOR_URL or "http://127.0.0.1:19093",
  model = "unconfigured", model_revision = "unconfigured", precision = "Q4_K_M",
  adapter_identity = "none", runtime_config_hash = "unconfigured",
  context_policy_version = "compact-next-edit-context-v1", mode = "manual",
  experimental_auto_opt_in = false, automatic_quality_validated = false,
  automatic_personalization_enabled = false, debounce_ms = 250, expiry_ms = 30000,
  max_buffer_bytes = 1048576, max_prompt_tokens = 2048, target_prompt_tokens = 1024,
  max_response_bytes = 262144, synthetic = false, persist_mode = false,
  accept_key = "<M-l>", dismiss_key = "<M-BS>",
}
local mode = "manual"
local phase = "idle"
local pending, active, debounce, expiry, wait_timer, group
local latest_wanted = false
local applying = false
local recent_edit = {}
local accepted_tick = {}
local last_fingerprint = nil
local last_status = "idle"
local last_error = nil
local last_latency_ms = nil
local consecutive_failures, backoff_until_ms = 0, 0
local counters = { requested = 0, displayed = 0, accepted = 0, cancelled_unseen = 0,
  rejected_explicit = 0, rejected_implicit_typing = 0, typed_match = 0,
  typed_partial_match = 0, dismissed_navigation = 0, expired = 0,
  model_no_edit = 0, request_failed = 0, invalid_output = 0, stale_discarded = 0 }
local mapping_key = nil
local lock_path = "/tmp/tabcomplete-predictor-19093.lock"
M._request_impl = nil -- deterministic test seam
M._tokenize_impl = nil -- deterministic test seam

local function now() return util.now_ms() end
local function utf8_boundary(line, col)
  if col < 0 or col > #line then return false end
  if col == #line then return true end
  local byte = line:byte(col + 1)
  return byte < 0x80 or byte >= 0xC0
end
local function content_of(buf)
  local content, _ = buffers.canonical_bytes(buf)
  return content
end
local function mode_path()
  return vim.fn.stdpath("state") .. "/tabcomplete-predictor-mode.json"
end
local function save_mode()
  if not opts.persist_mode then return end
  local path = mode_path()
  vim.fn.mkdir(vim.fn.fnamemodify(path, ":h"), "p")
  local file = io.open(path, "w")
  if file then
    file:write(vim.json.encode({ mode = mode, experimental_auto_opt_in = opts.experimental_auto_opt_in }))
    file:close()
  end
end
local function load_mode()
  if not opts.persist_mode then return nil end
  local file = io.open(mode_path(), "r")
  if not file then return nil end
  local bytes = file:read(256)
  file:close()
  local ok, saved = pcall(vim.json.decode, bytes or "")
  if ok and type(saved) == "table" and allowed_modes[saved.mode] then
    if saved.mode ~= "automatic" or (saved.experimental_auto_opt_in and opts.experimental_auto_opt_in) then
      return saved.mode
    end
  end
  return nil
end
local function stop_timer(timer)
  if timer then timer:stop() end
end
local function close_preview()
  if active and active.preview_win and vim.api.nvim_win_is_valid(active.preview_win) then
    vim.api.nvim_win_close(active.preview_win, true)
  end
  if active and vim.api.nvim_buf_is_valid(active.bufnr) then
    vim.api.nvim_buf_clear_namespace(active.bufnr, ns, 0, -1)
  end
end
local function emit(state, kind, payload)
  payload = payload or {}
  payload.prediction_id = state.prediction_id
  payload.request_id = state.request_id
  payload.synthetic = opts.synthetic
  payload.file = state.path
  return collector.emit(state.bufnr, kind, payload)
end
local function lifecycle(state, outcome, reason)
  if not state then return end
  emit(state, "heartbeat", { prediction_lifecycle = outcome, reason = reason,
    recorded_at_ms = now() })
end
local function release_lock(request)
  if request and request.has_lock then
    local owner = io.open(lock_path .. "/owner", "r")
    local token = owner and owner:read("*a") or ""
    if owner then owner:close() end
    if token == request.state.request_id then
      os.remove(lock_path .. "/owner")
      vim.uv.fs_rmdir(lock_path)
    end
    request.has_lock = false
  end
end
local function acquire_lock(state)
  local ok = vim.uv.fs_mkdir(lock_path, 448)
  if not ok then
    local stat = vim.uv.fs_stat(lock_path)
    if stat and stat.mtime and os.time() - stat.mtime.sec > 40 then
      os.remove(lock_path .. "/owner")
      vim.uv.fs_rmdir(lock_path)
      ok = vim.uv.fs_mkdir(lock_path, 448)
    end
  end
  if not ok then return false end
  local owner = io.open(lock_path .. "/owner", "w")
  if not owner then vim.uv.fs_rmdir(lock_path); return false end
  owner:write(state.request_id)
  owner:close()
  return true
end
local function state_fingerprint(state)
  return util.sha256hex(table.concat({ state.path, state.repo_identity, state.content_hash,
    tostring(state.row), tostring(state.start_col), tostring(state.changedtick) }, "\0"))
end
local function buffer_state()
  local buf = vim.api.nvim_get_current_buf()
  if not vim.api.nvim_buf_is_valid(buf) or util.is_excluded_buffer(buf) then
    return nil, "excluded buffer"
  end
  if vim.g.tabcomplete_predictor_paused or vim.b[buf].tabcomplete_predictor_off then
    return nil, "buffer opted out"
  end
  local path = vim.api.nvim_buf_get_name(buf)
  if path == "" then return nil, "unnamed buffer" end
  local count = vim.api.nvim_buf_line_count(buf)
  local bytes = vim.api.nvim_buf_get_offset(buf, count)
  if bytes > opts.max_buffer_bytes then return nil, "automatic prediction skipped: large buffer" end
  local pos = vim.api.nvim_win_get_cursor(0)
  local row, col = pos[1] - 1, pos[2]
  local line = vim.api.nvim_buf_get_lines(buf, row, row + 1, false)[1]
  if not line or not utf8_boundary(line, col) then return nil, "invalid UTF-8 cursor" end
  local lines = vim.api.nvim_buf_get_lines(buf, 0, -1, false)
  local before = table.concat(vim.list_slice(lines, math.max(1, row - 79), row), "\n")
  if before ~= "" then before = before .. "\n" end
  before = before .. line:sub(1, col)
  local after_lines = vim.list_slice(lines, row + 2, math.min(#lines, row + 41))
  local after = line:sub(#line + 1)
  if #after_lines > 0 then after = after .. "\n" .. table.concat(after_lines, "\n") end
  local region = line:sub(col + 1)
  local prior = table.concat(vim.list_slice(lines, 1, row), "\n")
  if row > 0 then prior = prior .. "\n" end
  local region_start = #prior + col
  local history = "<recent-edit unavailable>\n"
  local latest = recent_edit[buf]
  if latest then
    local prefix = table.concat(vim.list_slice(lines, 1, latest.start_row), "\n")
    if latest.start_row > 0 then prefix = prefix .. "\n" end
    local old, new = latest.deleted_text, latest.inserted_text
    local common = 0
    while common < math.min(#old, #new) and old:byte(common + 1) == new:byte(common + 1) do
      common = common + 1
    end
    while common > 0 and (not utf8_boundary(old, common) or not utf8_boundary(new, common)) do
      common = common - 1
    end
    local suffix = 0
    while suffix < math.min(#old - common, #new - common)
        and old:byte(#old - suffix) == new:byte(#new - suffix) do suffix = suffix + 1 end
    while suffix > 0 and (not utf8_boundary(old, #old - suffix)
        or not utf8_boundary(new, #new - suffix)) do suffix = suffix - 1 end
    history = "<actual-recent-edit start=" .. (#prefix + common) .. " end="
      .. (#prefix + #old - suffix) .. ">\n" .. new:sub(common + 1, #new - suffix)
      .. "\n</actual-recent-edit>\n"
  end
  local filetype = vim.bo[buf].filetype
  local prompt = "<repo " .. path .. ">\n<filetype " .. filetype .. ">\n"
    .. history .. "<file " .. path .. ">\n" .. before .. "[[EDIT]]" .. region
    .. "[[/EDIT]]" .. after .. "\n</file>\n<P " .. path .. " " .. region_start .. ">\n"
    .. "Return one compact next-edit action: N\\n for no edit or R\\n followed by exact replacement text. End with EOS.\n"
  local cache = repository.cache
  local identity = cache.root and (cache.root .. ":" .. (cache.head or "")) or path
  local state = {
    bufnr = buf, path = path, row = row, start_col = col, end_col = #line,
    prefix_line = line:sub(1, col), region = region,
    changedtick = vim.api.nvim_buf_get_changedtick(buf),
    repo_identity = identity, content_hash = util.sha256hex(content_of(buf)),
    prompt = prompt, context_hash = util.sha256hex(prompt),
    requested_at_ms = now(), prediction_id = util.uuid(), request_id = util.uuid(),
    filetype = filetype, file_identity = util.file_id(util.abspath(path), cache.root, cache.root_name),
    pre_state_sequence = collector.seq, cursor = { row = row, col = col },
  }
  state.fingerprint = state_fingerprint(state)
  return state
end
local function still_current(state)
  if not vim.api.nvim_buf_is_valid(state.bufnr) or vim.api.nvim_get_current_buf() ~= state.bufnr then return false end
  if vim.api.nvim_buf_get_name(state.bufnr) ~= state.path then return false end
  if vim.api.nvim_buf_get_changedtick(state.bufnr) ~= state.changedtick then return false end
  local pos = vim.api.nvim_win_get_cursor(0)
  if pos[1] - 1 ~= state.row or pos[2] ~= state.start_col then return false end
  local line = vim.api.nvim_buf_get_lines(state.bufnr, state.row, state.row + 1, false)[1]
  if not line or line:sub(state.start_col + 1, state.end_col) ~= state.region then return false end
  if not utf8_boundary(line, state.start_col) or not utf8_boundary(line, state.end_col) then return false end
  return util.sha256hex(content_of(state.bufnr)) == state.content_hash
end
local function can_auto()
  if (mode ~= "automatic" and mode ~= "shadow") or vim.g.tabcomplete_predictor_paused then return false end
  if now() < backoff_until_ms then return false end
  if not util.current_mode():find("^i") then return false end
  if vim.fn.pumvisible() == 1 then return false end
  if vim.snippet and vim.snippet.active and vim.snippet.active() then return false end
  if vim.g.tabcomplete_predictor_focus_lost then return false end
  return true
end
local function classify_delta(state, delta)
  local visible_ms = math.max(0, now() - state.shown_at_ms)
  local result = "dismissed_editor_change"
  local matched_bytes = 0
  local key = collector.last_key_event
  local key_correlated = opts.synthetic or (key and key.bufnr == state.bufnr
    and key.mode and key.mode:find("^i") and now() - key.timestamp_ms <= 250)
  if delta and delta.start_row <= state.row and delta.new_end_row > state.row
      and util.current_mode():find("^i") and key_correlated then
    local new_line = vim.api.nvim_buf_get_lines(state.bufnr, state.row, state.row + 1, false)[1] or ""
    local suffix = new_line:sub(state.start_col + 1)
    local wanted = state.action.text or ""
    if wanted ~= "" and suffix:sub(1, #wanted) == wanted then
      result = "typed_match"
      matched_bytes = #wanted
    elseif wanted ~= "" then
      local common = 0
      while common < math.min(#wanted, #suffix)
          and wanted:byte(common + 1) == suffix:byte(common + 1) do common = common + 1 end
      if common > 0 and utf8_boundary(wanted, common) then
        result = "typed_partial_match"
        matched_bytes = common
      else
        result = "rejected_implicit_typing"
      end
    else
      result = "rejected_implicit_typing"
    end
  end
  return result, visible_ms, matched_bytes, key_correlated and key or nil
end
local function dismiss(reason, delta, envelope)
  if not active then return end
  local state = active
  local outcome, visible_ms, matched_bytes, key
  if reason == "editor_change" then outcome, visible_ms, matched_bytes, key = classify_delta(state, delta)
  elseif reason == "explicit" then outcome = "rejected_explicit"
  elseif reason == "expired" then outcome = "expired"
  elseif reason == "mode_off" or reason == "mode_changed" then outcome = "cancelled_by_mode"
  else outcome = "dismissed_navigation" end
  visible_ms = visible_ms or math.max(0, now() - state.shown_at_ms)
  emit(state, "prediction_dismissed", { outcome = outcome,
    outcome_source = key and "key_correlated_buffer_delta" or "editor_observation",
    shown_at_ms = state.shown_at_ms, dismissed_at_ms = now(), visible_duration_ms = visible_ms,
    low_exposure = visible_ms < 150,
    ended_by_event_id = envelope and envelope.event_id or nil,
    ended_by_sequence = envelope and envelope.sequence_number or nil,
    matching_prefix_bytes = matched_bytes,
    diverged_after_prefix = false,
    delayed_matching_needed = outcome == "typed_partial_match",
    correlated_key_event_id = key and key.event_id or nil,
    input_delta = delta and { start_row = delta.start_row, old_end_row = delta.old_end_row,
      new_end_row = delta.new_end_row } or nil, active_buffer = true,
    focused = not vim.g.tabcomplete_predictor_focus_lost })
  if outcome == "rejected_explicit" then
    emit(state, "prediction_rejected", { finish_reason = "explicit_user_reject", rejected_at_ms = now() })
  end
  counters[outcome] = (counters[outcome] or 0) + 1
  close_preview()
  active = nil
  stop_timer(expiry)
  phase = "idle"
  last_status = outcome
end
local function invalidate(reason)
  stop_timer(debounce)
  if active then
    dismiss(reason == "mode_off" and "mode_off"
      or reason == "mode changed" and "mode_changed" or "navigation")
  end
  if pending and not pending.obsolete then
    pending.obsolete = true
    lifecycle(pending.state, "cancelled_unseen", reason)
    counters.cancelled_unseen = counters.cancelled_unseen + 1
    if pending.process then
      pcall(pending.process.kill, pending.process, "sigterm")
    end
  end
  phase = pending and "request_running" or "idle"
end
local function show(state, action)
  if action.action == "no_edit" then
    lifecycle(state, "model_no_edit", "model emitted N")
    counters.model_no_edit = counters.model_no_edit + 1
    last_status = "model_no_edit"
    return
  end
  if action.text == state.region then
    lifecycle(state, "model_no_edit", "unchanged replacement")
    counters.model_no_edit = counters.model_no_edit + 1
    last_status = "unchanged replacement"
    return
  end
  if mode == "automatic" and action.text:find("\n", 1, true) then
    lifecycle(state, "automatic_multiline_suppressed", "manual preview required")
    last_status = "multiline action suppressed"
    return
  end
  state.action = action
  state.shown_at_ms = now()
  local multiline = action.text:find("\n", 1, true) ~= nil
  if multiline then
    local lines = { "TabComplete replacement preview", "Original: " .. state.region, "Proposed:" }
    vim.list_extend(lines, vim.split(action.text, "\n", { plain = true }))
    local scratch = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_lines(scratch, 0, -1, false, lines)
    vim.bo[scratch].modifiable = false
    state.preview_win = vim.api.nvim_open_win(scratch, false, { relative = "cursor", row = 1, col = 0,
      width = math.min(100, math.max(35, vim.o.columns - 8)), height = math.min(#lines, 12),
      style = "minimal", border = "single", focusable = false })
  else
    local label = state.region == "" and action.text or ("[replace: " .. action.text .. "]")
    if action.text == "" then label = "[delete: " .. state.region .. "]" end
    vim.api.nvim_buf_set_extmark(state.bufnr, ns, state.row, state.start_col,
      { virt_text = { { label, "Comment" } }, virt_text_pos = "eol", hl_mode = "combine" })
  end
  active = state
  phase = "suggestion_displayed"
  local shown = emit(state, "prediction_shown", { proposed_start = { row = state.row, col = state.start_col },
    proposed_end = { row = state.row, col = state.end_col }, proposed_text = action.text,
    action = action.action, shown_at_ms = state.shown_at_ms, context_hash = state.context_hash,
    pre_state_hash = state.content_hash, active_buffer = true,
    focused = not vim.g.tabcomplete_predictor_focus_lost, display_policy = mode,
    action_blob_hash = state.action_blob_hash })
  state.shown_event_id = shown and shown.event_id
  counters.displayed = counters.displayed + 1
  if not expiry then expiry = vim.uv.new_timer() end
  stop_timer(expiry)
  local id = state.prediction_id
  expiry:start(opts.expiry_ms, 0, function()
    vim.schedule(function() if active and active.prediction_id == id then dismiss("expired") end end)
  end)
  last_status = "proposal shown"
end
local function schedule_retry()
  if not wait_timer then wait_timer = vim.uv.new_timer() end
  stop_timer(wait_timer)
  wait_timer:start(100, 0, function()
    vim.schedule(function()
      if latest_wanted and not pending and can_auto() then M.predict() end
    end)
  end)
end
local function finished(request, result)
  vim.schedule(function()
    if pending ~= request then return end
    pending = nil
    release_lock(request)
    phase = "idle"
    local state = request.state
    if request.obsolete then
      last_status = "stale response discarded"
      counters.stale_discarded = counters.stale_discarded + 1
    elseif request.parser and request.parser.error then
      lifecycle(state, "invalid_output", request.parser.error)
      counters.invalid_output = counters.invalid_output + 1
      last_status = request.parser.error
      consecutive_failures = consecutive_failures + 1
    elseif result.code ~= 0 then
      lifecycle(state, "request_failed", "model transport failure")
      counters.request_failed = counters.request_failed + 1
      last_error = "model request failed (exit " .. tostring(result.code) .. ")"
      last_status = last_error
      consecutive_failures = consecutive_failures + 1
    elseif not still_current(state) then
      lifecycle(state, "invalidated_unseen", "editor state changed before display")
      counters.stale_discarded = counters.stale_discarded + 1
      last_status = "stale response discarded"
    else
      local action, raw_or_err = sse.finish(request.parser)
      if not action then
        lifecycle(state, "invalid_output", raw_or_err)
        counters.invalid_output = counters.invalid_output + 1
        last_status = raw_or_err
        consecutive_failures = consecutive_failures + 1
      else
        consecutive_failures, backoff_until_ms = 0, 0
        state.responded_at_ms = now()
        local raw_response = table.concat(request.parser.raw_chunks)
        state.raw_response_hash = util.sha256hex(raw_response)
        collector.store_prediction_blob(raw_response, function() end)
        if action.action == "replace" then
          state.action_blob_hash = action.text == "" and state.raw_response_hash
            or util.sha256hex(action.text)
          if action.text ~= "" then collector.store_prediction_blob(action.text, function() end) end
        else
          state.action_blob_hash = state.raw_response_hash
        end
        emit(state, "prediction_generated", { raw_response_hash = state.raw_response_hash,
          action_blob_hash = state.action_blob_hash, canonical_action = action.action,
          stop_type = request.parser.terminal.stop_type, context_hash = state.context_hash,
          prompt_tokens = state.prompt_tokens, context_expanded = state.prompt_tokens
            and state.prompt_tokens > opts.target_prompt_tokens or false,
          first_chunk_at_ms = request.first_chunk_at_ms,
          first_token_at_ms = request.first_token_at_ms,
          completed_at_ms = state.responded_at_ms,
          model_elapsed_ms = state.responded_at_ms - state.requested_at_ms,
          backend_timings = request.parser.terminal.timings })
        last_latency_ms = state.responded_at_ms - state.requested_at_ms
        if mode == "shadow" then
          lifecycle(state, "shadow_only", "proposal was not displayed")
          last_status = "shadow response"
        else show(state, action) end
      end
    end
    if consecutive_failures >= 3 then
      backoff_until_ms = now() + math.min(60000, 1000 * (2 ^ math.min(6, consecutive_failures - 3)))
    end
    if latest_wanted and (mode == "automatic" or mode == "shadow") then schedule_retry() end
  end)
end
local function start_generation(request)
  local state = request.state
  request.kind = "generation"
  request.parser = sse.new(opts.max_response_bytes)
  local body = vim.json.encode({ prompt = state.prompt, n_predict = 96, temperature = 0,
    stream = true, cache_prompt = true, id_slot = 0, n_keep = 0 })
  local function chunk(data)
    if data and data ~= "" then
      if not request.first_chunk_at_ms then request.first_chunk_at_ms = now() end
      local ok = sse.feed(request.parser, data)
      if not ok and request.process then pcall(request.process.kill, request.process, "sigterm") end
      if request.parser.first_token and not request.first_token_at_ms then request.first_token_at_ms = now() end
    end
  end
  if M._request_impl then
    request.process = M._request_impl(state, body, function(result)
      chunk(result.stdout or "")
      finished(request, result)
    end, chunk)
  else
    request.process = vim.system({ "curl", "-sS", "-f", "-N", "-m", "30", "-X", "POST",
      "-H", "Content-Type: application/json", "--data-binary", "@-", opts.url .. "/completion" },
      { stdin = body, text = false, stdout = function(_, data) chunk(data) end,
        stderr = false, timeout = 31000 }, function(result) finished(request, result) end)
  end
  phase = "request_running"
end
local function tokenize(request)
  local state = request.state
  if M._request_impl then start_generation(request); return end
  request.kind = "tokenize"
  local body = vim.json.encode({ content = state.prompt, add_special = false })
  request.process = vim.system({ "curl", "-sS", "-f", "-m", "5", "-X", "POST",
    "-H", "Content-Type: application/json", "--data-binary", "@-", opts.url .. "/tokenize" },
    { stdin = body, text = true, timeout = 6000 }, function(result)
      vim.schedule(function()
        if pending ~= request then return end
        if request.obsolete then finished(request, { code = 0 }); return end
        local ok, decoded = pcall(vim.json.decode, result.stdout or "")
        if result.code ~= 0 or not ok or type(decoded) ~= "table" or type(decoded.tokens) ~= "table" then
          finished(request, { code = result.code ~= 0 and result.code or 1 }); return
        end
        state.prompt_tokens = #decoded.tokens
        if state.prompt_tokens > opts.max_prompt_tokens or state.prompt_tokens + 96 > 2304 then
          lifecycle(state, "invalidated_unseen", "prompt token budget exceeded")
          request.obsolete = true
          finished(request, { code = 0 })
          last_status = "prompt token budget exceeded"
          return
        end
        start_generation(request)
      end)
    end)
end
function M.predict()
  if mode == "off" then return false, "mode off" end
  if (mode == "automatic" or mode == "shadow") and not can_auto() then
    return false, "automatic UI is busy or unfocused"
  end
  stop_timer(debounce)
  local state, err = buffer_state()
  if not state then last_status = err; latest_wanted = false; return false, err end
  if (mode == "automatic" or mode == "shadow") and state.fingerprint == last_fingerprint then
    latest_wanted = false; return false, "unchanged state already requested"
  end
  if pending then
    latest_wanted = true
    last_status = "waiting for active request"
    schedule_retry()
    return true
  end
  if active then dismiss("navigation") end
  if not M._request_impl and not acquire_lock(state) then
    latest_wanted = true
    last_status = "waiting for shared service slot"
    schedule_retry()
    return true
  end
  latest_wanted = false
  last_fingerprint = state.fingerprint
  local request = { state = state, kind = "tokenize", has_lock = not M._request_impl }
  pending = request -- set before an immediate test callback can run
  phase = "request_running"
  collector.anchor_prediction(state.bufnr)
  collector.store_prediction_blob(state.prompt, function() end)
  local ev = emit(state, "prediction_requested", {
    provider = "tabcomplete-local", model = opts.model, model_revision = opts.model_revision,
    model_gguf_sha256 = opts.model_revision, precision = opts.precision,
    adapter_identity = opts.adapter_identity, runtime_config_hash = opts.runtime_config_hash,
    context_policy_version = opts.context_policy_version, requested_at_ms = state.requested_at_ms,
    context_hash = state.context_hash, context_blob_hash = state.context_hash,
    pre_state_hash = state.content_hash, pre_state_sequence = state.pre_state_sequence,
    file_identity = state.file_identity, cursor = state.cursor,
    editable_range = { start_row = state.row, start_col = state.start_col,
      end_row = state.row, end_col = state.end_col },
    max_output_tokens = 96, temperature = 0, wire_version = "compact-next-edit-v1",
    human_verified = false, mode = mode, display_policy = mode,
  })
  state.request_event_id = ev and ev.event_id
  counters.requested = counters.requested + 1
  tokenize(request)
  return true
end
function M.accept()
  if not active or not active.action then return false, "no active proposal" end
  local state = active
  if not still_current(state) then dismiss("navigation"); return false, "stale proposal" end
  if state.action.action ~= "replace" then return false, "no edit" end
  local replacement = vim.split(state.action.text, "\n", { plain = true })
  vim.bo[state.bufnr].undolevels = vim.bo[state.bufnr].undolevels
  applying = true
  vim.api.nvim_buf_set_text(state.bufnr, state.row, state.start_col, state.row, state.end_col, replacement)
  accepted_tick[state.bufnr] = vim.api.nvim_buf_get_changedtick(state.bufnr)
  local delta_sequence = collector.seq
  emit(state, "prediction_accepted", { accepted_at_ms = now(), shown_event_id = state.shown_event_id,
    accepted_chars = vim.fn.strchars(state.action.text), accepted_lines = #replacement,
    total_chars = vim.fn.strchars(state.action.text), applied_through_sequence = delta_sequence,
    visible_duration_ms = now() - state.shown_at_ms })
  applying = false
  counters.accepted = counters.accepted + 1
  close_preview()
  active = nil
  stop_timer(expiry)
  phase = "idle"
  last_status = "accepted"
  return true
end
function M.reject()
  if not active then return false, "no active proposal" end
  dismiss("explicit")
  return true
end
function M.set_mode(next_mode)
  if not allowed_modes[next_mode] then return false, "invalid mode" end
  if next_mode == "automatic" and not (opts.experimental_auto_opt_in or opts.automatic_quality_validated) then
    return false, "automatic mode requires explicit opt-in"
  end
  invalidate(next_mode == "off" and "mode_off" or "mode changed")
  mode = next_mode
  save_mode()
  if mode == "off" then
    latest_wanted = false
    if pending and pending.process then pcall(pending.process.kill, pending.process, "sigterm") end
  end
  last_status = next_mode
  return true
end
function M.status()
  local cs = collector.status()
  return { mode = mode, stage = phase, state = last_status, model = opts.model,
    revision = opts.model_revision, precision = opts.precision,
    automatic_state = mode == "automatic" and "automatic experimental" or "inactive",
    quality = opts.automatic_quality_validated and "validated" or "uncalibrated",
    experimental_auto_opt_in = opts.experimental_auto_opt_in,
    automatic_quality_validated = opts.automatic_quality_validated,
    automatic_personalization_enabled = false,
    in_flight = pending ~= nil, proposal_active = active ~= nil,
    acceptance_key = mapping_key or opts.accept_key, last_latency_ms = last_latency_ms,
    last_error = last_error, collector_connected = cs.last_ok_ms ~= nil,
    backoff_until_ms = backoff_until_ms,
    collector_state = cs.last_ok_ms and "connected" or (cs.spool_files > 0 and "spooling" or "unverified"),
    collector_spool_files = cs.spool_files, counters = vim.deepcopy(counters) }
end
local function debounce_changed()
  if not can_auto() then return end
  stop_timer(debounce)
  if not debounce then debounce = vim.uv.new_timer() end
  phase = "debounce_pending"
  debounce:start(opts.debounce_ms, 0, function()
    vim.schedule(function()
      if can_auto() then M.predict() else phase = "idle" end
    end)
  end)
end
local function install_mapping()
  if mapping_key then return end
  local key = opts.accept_key
  if vim.fn.maparg(key, "i") ~= "" or vim.fn.maparg(key, "n") ~= "" then
    key = "<M-;>"
  end
  if vim.fn.maparg(key, "i") == "" and vim.fn.maparg(key, "n") == "" then
    vim.keymap.set({ "i", "n" }, key, function() M.accept() end,
      { desc = "Accept TabComplete proposal", silent = true })
    mapping_key = key
  else
    mapping_key = "command only; preferred keys occupied"
  end
  if vim.fn.maparg(opts.dismiss_key, "i") == "" then
    vim.keymap.set("i", opts.dismiss_key, function() M.reject() end,
      { desc = "Dismiss TabComplete proposal", silent = true })
  end
end
local function refresh_repo_async(buf)
  if not vim.api.nvim_buf_is_valid(buf) then return end
  local path = vim.api.nvim_buf_get_name(buf)
  if path == "" then return end
  local directory = vim.fn.fnamemodify(path, ":h")
  vim.system({ "git", "-C", directory, "rev-parse", "--show-toplevel", "HEAD" },
    { text = true, timeout = 3000 }, function(result)
      vim.schedule(function()
        if not vim.api.nvim_buf_is_valid(buf) or vim.api.nvim_buf_get_name(buf) ~= path then return end
        if result.code ~= 0 then return end
        local lines = vim.split((result.stdout or ""):gsub("%s+$", ""), "\n")
        if #lines < 2 then return end
        local old = (repository.cache.root or "") .. ":" .. (repository.cache.head or "")
        local fresh = lines[1] .. ":" .. lines[2]
        if old ~= fresh then
          repository.cache.root, repository.cache.head = lines[1], lines[2]
          repository.cache.root_name = vim.fn.fnamemodify(lines[1], ":t")
          repository.cache.origin = ""
          last_fingerprint = nil
          invalidate("repository changed")
        end
      end)
    end)
end
function M.setup(options)
  opts = vim.tbl_deep_extend("force", opts, options or {})
  if group then pcall(vim.api.nvim_del_augroup_by_id, group) end
  invalidate("setup")
  recent_edit = {}
  last_fingerprint = nil
  mode = load_mode() or opts.mode
  if mode == "automatic" and not (opts.experimental_auto_opt_in or opts.automatic_quality_validated) then
    mode = "manual"
  end
  group = vim.api.nvim_create_augroup("TabCompletePredict", { clear = true })
  buffers.on_delta = function(buf, delta, envelope)
    if applying then return end
    recent_edit[buf] = delta
    if active and active.bufnr == buf then dismiss("editor_change", delta, envelope) end
    if pending and pending.state.bufnr == buf then invalidate("editor change before display") end
  end
  vim.api.nvim_create_autocmd({ "TextChangedI", "CursorMovedI" }, {
    group = group, callback = function()
      if applying then return end
      local buf = vim.api.nvim_get_current_buf()
      if accepted_tick[buf] == vim.api.nvim_buf_get_changedtick(buf) then
        accepted_tick[buf] = nil
        return
      end
      if active then dismiss("navigation") end
      if pending then invalidate("new insert input") end
      debounce_changed()
    end,
  })
  vim.api.nvim_create_autocmd({ "BufLeave", "BufWipeout", "FocusLost", "InsertLeave" }, {
    group = group, callback = function(args)
      if args.event == "FocusLost" then vim.g.tabcomplete_predictor_focus_lost = true end
      invalidate("navigation")
      if args.event == "BufWipeout" then
        recent_edit[args.buf] = nil
        accepted_tick[args.buf] = nil
      end
    end,
  })
  vim.api.nvim_create_autocmd("FocusGained", {
    group = group, callback = function()
      vim.g.tabcomplete_predictor_focus_lost = false
      refresh_repo_async(vim.api.nvim_get_current_buf())
    end,
  })
  vim.api.nvim_create_autocmd({ "BufEnter", "BufWritePost", "DirChanged" }, {
    group = group, callback = function(args)
      refresh_repo_async(args.buf or vim.api.nvim_get_current_buf())
    end,
  })
  install_mapping()
  save_mode()
  phase = "idle"
  last_status = mode == "automatic" and "automatic experimental" or mode
  return M
end
return M
