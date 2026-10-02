-- Local next-edit suggestions. Inference is automatic only after explicit opt-in;
-- every buffer mutation still requires an acceptance key.
local M = {}
local util = require("tabcomplete_trajectory.util")
local collector = require("tabcomplete_trajectory")
local buffers = require("tabcomplete_trajectory.buffers")
local repository = require("tabcomplete_trajectory.repository")
local sse = require("tabcomplete_trajectory.sse")
local single_line_v1 = require("tabcomplete_trajectory.single_line_v1")

local ns = vim.api.nvim_create_namespace("TabCompletePredict")
local allowed_modes = { manual = true, automatic = true, shadow = true, off = true }
local default_allowed_models = {
  q25 = { model_sha256 = "4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb",
    model_protocol = "single-line-edit-v1", output_tokens = 64 },
  sweep = { model_sha256 = "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4",
    model_protocol = "sweep-full-file-v1", output_tokens = 192 },
}
local opts = {
  url = vim.env.TABCOMPLETE_PREDICTOR_URL or "http://127.0.0.1:19093",
  backend = "llama-cpp-legacy", allowed_models = default_allowed_models,
  model = "unconfigured", model_revision = "unconfigured", precision = "Q4_K_M",
  adapter_identity = "none", runtime_config_hash = "unconfigured",
  context_policy_version = "compact-next-edit-context-v1", mode = "manual",
  protocol_version = "compact-next-edit-v1", single_line_input_tokens = 1024,
  automatic_prefix_guard = false,
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
local last_decision = nil
local model_switching = false
local consecutive_failures, backoff_until_ms = 0, 0
local counters = { requested = 0, displayed = 0, accepted = 0, cancelled_unseen = 0,
  rejected_explicit = 0, rejected_implicit_typing = 0, typed_match = 0,
  typed_partial_match = 0, dismissed_navigation = 0, expired = 0,
  model_no_edit = 0, request_failed = 0, invalid_output = 0, stale_discarded = 0,
  automatic_policy_suppressed = 0, automatic_ui_suppressed = 0 }
local mapping_key = nil
local lock_path = "/tmp/tabcomplete-predictor-19093.lock"
M._request_impl = nil -- deterministic test seam
M._tokenize_impl = nil -- deterministic test seam
M._backend_post_impl = nil -- deterministic Rust backend RPC seam
M._backend_get_impl = nil -- deterministic Rust health RPC seam
M._confirm_impl = nil -- deterministic test seam; production uses vim.fn.confirm
local model_identity_refresh_generation = 0

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
  return opts.mode_state_path or (vim.fn.stdpath("state") .. "/tabcomplete-predictor-mode.json")
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
local function is_sha256(value)
  return type(value) == "string" and #value == 64 and value:match("^%x+$") ~= nil
end
local function is_integer(value)
  return type(value) == "number" and value % 1 == 0
end
local function model_spec(alias)
  local configured = opts.allowed_models and opts.allowed_models[alias]
  if type(configured) ~= "table" then return nil end
  local layouts
  if alias == "q25" then
    layouts = {
      ["trained-v2"] = "single-line-context-v2",
      ["cursor-last-v1"] = "single-line-cursor-last-context-v1",
    }
  elseif alias == "sweep" then
    layouts = { ["sweep-window-v1"] = "sweep-window-context-v1" }
  else
    layouts = {}
  end
  return {
    model_sha256 = configured.model_sha256 or configured.sha256,
    model_protocol = configured.model_protocol or configured.protocol,
    output_tokens = configured.output_tokens or configured.max_output_tokens,
    context_layout_policies = layouts,
  }
end
local function expected_context_layout(identity, spec)
  local layout = identity.context_layout
  if layout == nil then
    layout = identity.alias == "q25" and "trained-v2"
      or (identity.alias == "sweep" and "sweep-window-v1" or nil)
  end
  local policy = layout and spec.context_layout_policies[layout]
  if not policy then return nil, nil end
  return layout, policy
end
local function validate_model_identity(identity, expected_alias)
  if type(identity) ~= "table" or identity.status ~= "ok" then
    return nil, "missing Rust model identity"
  end
  if identity.model_embedded ~= nil and type(identity.model_embedded) ~= "boolean" then
    return nil, "Rust embedded-model identity is invalid"
  end
  if identity.model_switch_supported ~= nil and type(identity.model_switch_supported) ~= "boolean" then
    return nil, "Rust model-switch identity is invalid"
  end
  if identity.model_selection ~= nil and type(identity.model_selection) ~= "string" then
    return nil, "Rust model-selection identity is invalid"
  end
  if identity.model_storage ~= nil and type(identity.model_storage) ~= "string" then
    return nil, "Rust model-storage identity is invalid"
  end
  local spec = model_spec(identity.alias)
  if not spec then return nil, "Rust model alias is not allowlisted" end
  if expected_alias and identity.alias ~= expected_alias then
    return nil, "Rust model alias does not match the requested model"
  end
  if identity.model_sha256 ~= spec.model_sha256 or not is_sha256(identity.model_sha256) then
    return nil, "Rust model digest does not match the allowlist"
  end
  if identity.model_protocol ~= spec.model_protocol then
    return nil, "Rust model protocol does not match the allowlist"
  end
  if not is_sha256(identity.runtime_config_hash) then
    return nil, "Rust runtime configuration digest is invalid"
  end
  if not is_integer(identity.output_tokens) or identity.output_tokens ~= spec.output_tokens then
    return nil, "Rust output-token cap does not match the allowlist"
  end
  if not is_integer(identity.input_tokens) or identity.input_tokens < 1
      or identity.input_tokens > opts.single_line_input_tokens then
    return nil, "Rust input-token budget is outside the configured limit"
  end
  if not is_integer(identity.context_size) or identity.context_size < identity.input_tokens + identity.output_tokens then
    return nil, "Rust model context is smaller than its configured token budgets"
  end
  local context_layout = expected_context_layout(identity, spec)
  if not context_layout then return nil, "Rust model context layout is not allowlisted" end
  local validated = vim.deepcopy(identity)
  validated.context_layout_legacy = identity.context_layout == nil
  validated.context_layout = context_layout
  return validated, nil, spec
end
local function record_request(state)
  collector.anchor_prediction(state.bufnr)
  collector.store_prediction_blob(state.prompt, function() end)
  local is_v1 = opts.protocol_version == single_line_v1.WIRE_VERSION
  local identity = state.model_identity or opts.current_model_identity or {}
  local rust = opts.backend == "rust-editor-v1"
  local ev = emit(state, "prediction_requested", {
    provider = "tabcomplete-local", model = rust and identity.alias or opts.model,
    model_revision = rust and identity.model_sha256 or opts.model_revision,
    model_gguf_sha256 = rust and identity.model_sha256 or opts.model_revision,
    model_alias = rust and identity.alias or nil,
    model_protocol = rust and identity.model_protocol or nil,
    context_layout = rust and identity.context_layout or nil,
    precision = opts.precision,
    adapter_identity = opts.adapter_identity,
    runtime_config_hash = rust and identity.runtime_config_hash or opts.runtime_config_hash,
    context_policy_version = state.context_policy_version
      or (is_v1 and single_line_v1.CONTEXT_POLICY_VERSION or opts.context_policy_version),
    requested_at_ms = state.requested_at_ms,
    context_hash = state.context_hash, context_blob_hash = state.context_hash,
    pre_state_hash = state.content_hash, pre_state_sequence = state.pre_state_sequence,
    file_identity = state.file_identity, cursor = state.cursor,
    editable_range = state.editable_range or { start_row = state.row, start_col = state.start_col,
      end_row = state.row, end_col = state.end_col },
    max_output_tokens = rust and identity.output_tokens
      or (is_v1 and single_line_v1.MAX_ACTION_TOKENS or 96),
    temperature = 0, wire_version = opts.protocol_version,
    human_verified = false, mode = mode,
    display_policy = mode == "automatic" and opts.automatic_prefix_guard
      and "prefix-preserving-completion-v1" or mode,
  })
  state.request_event_id = ev and ev.event_id
  counters.requested = counters.requested + 1
end
local function lifecycle(state, outcome, reason, extra)
  if not state then return end
  local payload = { prediction_lifecycle = outcome, reason = reason, recorded_at_ms = now() }
  if type(extra) == "table" then
    for key, value in pairs(extra) do payload[key] = value end
  end
  emit(state, "heartbeat", payload)
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
  return util.sha256hex(table.concat({ opts.protocol_version, state.path, state.repo_identity, state.content_hash,
    tostring(state.row), tostring(state.start_col), tostring(state.changedtick) }, "\0"))
end
local function normalized_filetype(filetype)
  if filetype == "python" then return "python" end
  if filetype == "typescript" or filetype == "typescriptreact" then return "typescript" end
  if filetype == "rust" then return "rust" end
  if filetype == "go" then return "go" end
  return nil
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
  local cache = repository.cache
  local identity = cache.root and (cache.root .. ":" .. (cache.head or "")) or path
  if opts.protocol_version == single_line_v1.WIRE_VERSION then
    local canonical_filetype = normalized_filetype(filetype)
    if not canonical_filetype then return nil, "single-line-edit-v1 does not support " .. filetype end
    local history_items = {}
    if latest and not latest.deleted_text:find("\n", 1, true)
        and not latest.inserted_text:find("\n", 1, true) then
      history_items[1] = { row = latest.start_row, old_text = latest.deleted_text,
        new_text = latest.inserted_text }
    end
    local contract_state = {
      file_id = path, filetype = canonical_filetype, source = content_of(buf), target_row = row,
      cursor_col = col, history = history_items, relevant = {},
    }
    local state_ok = pcall(single_line_v1.new_context, contract_state)
    if not state_ok then return nil, "buffer is outside the single-line-edit-v1 source contract" end
    local state = {
      bufnr = buf, path = path, row = row, start_col = col, end_col = #line,
      prefix_line = line:sub(1, col), region = line,
      changedtick = vim.api.nvim_buf_get_changedtick(buf), repo_identity = identity,
      content_hash = util.sha256hex(contract_state.source), source = contract_state.source,
      contract_state = contract_state, prompt = nil, context_hash = nil,
      requested_at_ms = now(), prediction_id = util.uuid(), request_id = util.uuid(),
      filetype = filetype, file_identity = util.file_id(util.abspath(path), cache.root, cache.root_name),
      pre_state_sequence = collector.seq, cursor = { row = row, col = col },
      editable_range = { start_row = row, start_col = 0, end_row = row, end_col = #line },
    }
    state.fingerprint = state_fingerprint(state)
    return state
  end
  local prompt = "<repo " .. path .. ">\n<filetype " .. filetype .. ">\n"
    .. history .. "<file " .. path .. ">\n" .. before .. "[[EDIT]]" .. region
    .. "[[/EDIT]]" .. after .. "\n</file>\n<P " .. path .. " " .. region_start .. ">\n"
    .. "Return one compact next-edit action: N\\n for no edit or R\\n followed by exact replacement text. End with EOS.\n"
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
  if not line then return false end
  if opts.protocol_version == single_line_v1.WIRE_VERSION then
    if #line ~= state.end_col or not utf8_boundary(line, state.start_col) then return false end
    return content_of(state.bufnr) == state.source
  end
  if line:sub(state.start_col + 1, state.end_col) ~= state.region then return false end
  if not utf8_boundary(line, state.start_col) or not utf8_boundary(line, state.end_col) then return false end
  return util.sha256hex(content_of(state.bufnr)) == state.content_hash
end
local function can_auto()
  if (mode ~= "automatic" and mode ~= "shadow") or vim.g.tabcomplete_predictor_paused then return false end
  if now() < backoff_until_ms then return false end
  if not util.current_mode():find("^i") then return false end
  if vim.fn.pumvisible() == 1 then return false end
  local blink = package.loaded["blink.cmp"]
  if blink and type(blink.is_visible) == "function" then
    local ok, visible = pcall(blink.is_visible)
    if not ok or visible then return false end
  end
  if vim.snippet and vim.snippet.active and vim.snippet.active() then return false end
  if vim.g.tabcomplete_predictor_focus_lost then return false end
  return true
end
local function classify_delta(state, delta)
  local visible_ms = math.max(0, now() - state.shown_at_ms)
  local result = "dismissed_editor_change"
  local matched_bytes = 0
  local key = collector.last_key_event
  local key_correlated = not (delta and delta.change_origin == "buffer_reload")
    and (opts.synthetic or (key and key.bufnr == state.bufnr
      and key.mode and key.mode:find("^i") and now() - key.timestamp_ms <= 250))
  if opts.protocol_version == single_line_v1.WIRE_VERSION then
    if delta and key_correlated and delta.start_row == state.row then
      local action = state.action
      local matched = false
      local offered = action.text or ""
      if action.kind == "replace_line" then
        matched = delta.old_end_row == state.row + 1 and delta.new_end_row == state.row + 1
          and delta.deleted_text == state.region and delta.inserted_text == offered
      elseif action.kind == "insert_before" then
        matched = delta.old_end_row == state.row and delta.new_end_row == state.row + 1
          and delta.deleted_text == "" and delta.inserted_text == offered
      elseif action.kind == "delete_line" then
        matched = delta.old_end_row == state.row + 1 and delta.new_end_row == state.row
          and delta.deleted_text == state.region and delta.inserted_text == ""
      end
      if matched then
        result = "typed_match"
        matched_bytes = #offered
      elseif action.kind ~= "delete_line" and offered ~= ""
          and delta.inserted_text ~= "" and offered:sub(1, #delta.inserted_text) == delta.inserted_text
          and delta.start_row == state.row then
        result = "typed_partial_match"
        matched_bytes = #delta.inserted_text
      else
        result = "rejected_implicit_typing"
      end
    end
    return result, visible_ms, matched_bytes, key_correlated and key or nil
  end
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
  elseif reason == "completion_ui" then outcome = "dismissed_editor_change"
  elseif reason == "mode_off" or reason == "mode_changed" then outcome = "cancelled_by_mode"
  else outcome = "dismissed_navigation" end
  visible_ms = visible_ms or math.max(0, now() - state.shown_at_ms)
  local dismissed = emit(state, "prediction_dismissed", { outcome = outcome, reason = reason,
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
    if dismissed then
      last_decision = { state = state, outcome = outcome, event_id = dismissed.event_id,
        session_id = collector.session_id, synthetic = opts.synthetic, reviewed = false }
    end
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
      or reason == "mode changed" and "mode_changed"
      or reason == "completion UI" and "completion_ui" or "navigation")
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
function M.preserves_typed_prefix(prefix, action)
  if action.kind == "keep" or action.kind == "insert_before" then return true end
  if action.kind == "delete_line" then return false end
  return action.kind == "replace_line" and type(action.text) == "string"
    and action.text:sub(1, #prefix) == prefix
end
local function automatic_ui_ready(state)
  if mode ~= "automatic" or can_auto() then return true end
  lifecycle(state, "invalidated_unseen", "automatic UI became busy or unfocused")
  counters.automatic_ui_suppressed = counters.automatic_ui_suppressed + 1
  last_status = "automatic suggestion skipped: UI busy or unfocused"
  return false
end
local function diff_preview_highlights()
  local dark = vim.o.background == "dark"
  vim.api.nvim_set_hl(0, "TabCompleteDiffAdd", {
    default = true,
    fg = dark and "#c7f6d0" or "#174d24",
    bg = dark and "#234d2e" or "#c7f6d0",
  })
  vim.api.nvim_set_hl(0, "TabCompleteDiffDelete", {
    default = true,
    fg = dark and "#ffd7dc" or "#8f1d2c",
    bg = dark and "#632c35" or "#ffd7dc",
    strikethrough = true,
  })
end
local function deleted_highlights()
  return "TabCompleteDiffDelete"
end
local function preview_virtual_line(bufnr, row, above, chunks)
  vim.api.nvim_buf_set_extmark(bufnr, ns, row, 0, {
    virt_lines = { chunks },
    virt_lines_above = above,
  })
end
local function preview_inline(bufnr, row, col, text)
  if text == "" then return end
  vim.api.nvim_buf_set_extmark(bufnr, ns, row, col, {
    virt_text = { { text, "TabCompleteDiffAdd" } },
    virt_text_pos = "inline",
    hl_mode = "combine",
    right_gravity = false,
  })
end
local function highlight_deleted_range(bufnr, row, start_col, end_col)
  if end_col <= start_col then return end
  vim.api.nvim_buf_set_extmark(bufnr, ns, row, start_col, {
    end_row = row,
    end_col = end_col,
    hl_group = deleted_highlights(),
    hl_mode = "combine",
    right_gravity = false,
    end_right_gravity = true,
  })
end
local function common_prefix_bytes(old_text, new_text)
  local count = 0
  local limit = math.min(#old_text, #new_text)
  while count < limit and old_text:byte(count + 1) == new_text:byte(count + 1) do
    count = count + 1
  end
  while count > 0 and (not utf8_boundary(old_text, count) or not utf8_boundary(new_text, count)) do
    count = count - 1
  end
  return count
end
local function common_suffix_bytes(old_text, new_text, prefix_bytes)
  local limit = math.min(#old_text - prefix_bytes, #new_text - prefix_bytes)
  local count = 0
  while count < limit and old_text:byte(#old_text - count) == new_text:byte(#new_text - count) do
    count = count + 1
  end
  while count > 0 and (not utf8_boundary(old_text, #old_text - count)
      or not utf8_boundary(new_text, #new_text - count)) do
    count = count - 1
  end
  return count
end
local function prefix_completion_text(state, old_text, new_text)
  local cursor_col = state.contract_state.cursor_col
  if not utf8_boundary(old_text, cursor_col) then return nil end
  local prefix = old_text:sub(1, cursor_col)
  local suffix = old_text:sub(cursor_col + 1)
  if #new_text < #prefix + #suffix
      or new_text:sub(1, #prefix) ~= prefix
      or (suffix ~= "" and new_text:sub(-#suffix) ~= suffix) then
    return nil
  end
  return new_text:sub(#prefix + 1, #new_text - #suffix)
end
local function preview_single_line_action(state, action)
  diff_preview_highlights()
  local bufnr, row = state.bufnr, state.row
  if action.kind == "insert_before" then
    local text = action.text == "" and "[empty line]" or action.text
    preview_virtual_line(bufnr, row, true, {
      { "+ ", "TabCompleteDiffAdd" }, { text, "TabCompleteDiffAdd" },
    })
    return
  end
  if action.kind == "delete_line" then
    if state.region == "" then
      preview_virtual_line(bufnr, row, false, {
        { "- [empty line]", deleted_highlights() },
      })
    else
      highlight_deleted_range(bufnr, row, 0, #state.region)
    end
    return
  end

  local old_text, new_text = state.region, action.text
  local insertion = prefix_completion_text(state, old_text, new_text)
  if insertion ~= nil then
    preview_inline(bufnr, row, state.contract_state.cursor_col, insertion)
    return
  end

  local prefix = common_prefix_bytes(old_text, new_text)
  local suffix = common_suffix_bytes(old_text, new_text, prefix)
  local old_end = #old_text - suffix
  local new_end = #new_text - suffix
  local inserted = new_text:sub(prefix + 1, new_end)
  highlight_deleted_range(bufnr, row, prefix, old_end)
  preview_inline(bufnr, row, old_end, inserted)
end
local function show(state, action)
  if opts.protocol_version == single_line_v1.WIRE_VERSION then
    if action.kind == "keep" then
      lifecycle(state, "model_no_edit", "model emitted N")
      counters.model_no_edit = counters.model_no_edit + 1
      last_status = "model_no_edit"
      return
    end
    if action.kind == "replace_line" and action.text == state.region then
      lifecycle(state, "model_no_edit", "unchanged line replacement")
      counters.model_no_edit = counters.model_no_edit + 1
      last_status = "unchanged replacement"
      return
    end
    if not automatic_ui_ready(state) then return end
    if mode == "automatic" and opts.automatic_prefix_guard
        and not M.preserves_typed_prefix(state.prefix_line, action) then
      lifecycle(state, "automatic_policy_suppressed", "proposal would remove text before cursor")
      counters.automatic_policy_suppressed = counters.automatic_policy_suppressed + 1
      last_status = "automatic completion skipped: typed prefix would change"
      return
    end
    action.action = action.kind
    state.action = action
    state.action_range = single_line_v1.action_range(state.contract_state, action)
    state.shown_at_ms = now()
    preview_single_line_action(state, action)
    active = state
    phase = "suggestion_displayed"
    local shown = emit(state, "prediction_shown", {
      proposed_start = { row = state.action_range.start_row, col = state.action_range.start_col },
      proposed_end = { row = state.action_range.end_row, col = state.action_range.end_col },
      proposed_start_byte = state.action_range.start_byte,
      proposed_end_byte = state.action_range.end_byte,
      proposed_range_end_exclusive = state.action_range.end_exclusive,
      proposed_range_includes_terminator = state.action_range.includes_terminator,
      proposed_text = action.text, action = action.kind, shown_at_ms = state.shown_at_ms,
      context_hash = state.context_hash, pre_state_hash = state.content_hash,
      active_buffer = true, ui_attached = #vim.api.nvim_list_uis() > 0, focused = not vim.g.tabcomplete_predictor_focus_lost,
      display_policy = mode, action_blob_hash = state.action_blob_hash,
      wire_version = single_line_v1.WIRE_VERSION,
    })
    state.shown_event_id = shown and shown.event_id
    counters.displayed = counters.displayed + 1
    if not expiry then expiry = vim.uv.new_timer() end
    stop_timer(expiry)
    local id = state.prediction_id
    expiry:start(opts.expiry_ms, 0, function()
      vim.schedule(function() if active and active.prediction_id == id then dismiss("expired") end end)
    end)
    last_status = "proposal shown"
    return
  end
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
  if not automatic_ui_ready(state) then return end
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
    pre_state_hash = state.content_hash, active_buffer = true, ui_attached = #vim.api.nvim_list_uis() > 0,
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
local start_generation
local function finished(request, result)
  vim.schedule(function()
    if pending ~= request then return end
    if not request.obsolete and opts.backend == "rust-editor-v1"
        and request.kind == "generation" and result.code == 22
        and (request.generation_attempts or 0) < 5 then
      request.generation_attempts = (request.generation_attempts or 0) + 1
      request.process = nil
      vim.defer_fn(function()
        if pending == request then
          if request.obsolete then finished(request, { code = 0 })
          else start_generation(request) end
        end
      end, 250)
      return
    end
    pending = nil
    release_lock(request)
    phase = "idle"
    local state = request.state
    if request.obsolete then
      last_status = "stale response discarded"
      counters.stale_discarded = counters.stale_discarded + 1
    elseif request.protocol_error then
      lifecycle(state, "invalid_output", request.protocol_error)
      counters.invalid_output = counters.invalid_output + 1
      last_status = request.protocol_error
      consecutive_failures = consecutive_failures + 1
    elseif request.context_error then
      lifecycle(state, "invalidated_unseen", request.context_error)
      last_status = request.context_error
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
      local is_v1 = opts.protocol_version == single_line_v1.WIRE_VERSION
      local is_rust = opts.backend == "rust-editor-v1"
      if is_rust and request.parser.terminal and #request.parser.raw_chunks > 0 then
        local raw_response = table.concat(request.parser.raw_chunks)
        if #raw_response <= request.parser.max_bytes then
          state.raw_response_hash = util.sha256hex(raw_response)
          collector.store_prediction_blob(raw_response, function() end)
        end
      end
      local action, raw_or_err
      if is_rust then action, raw_or_err = sse.finish_rust(request.parser, {
        model_identity = state.model_identity, contract_state = state.contract_state, window = state.window })
      elseif is_v1 then action, raw_or_err = sse.finish_single_line(request.parser)
      else action, raw_or_err = sse.finish(request.parser) end
      if not action then
        local evidence
        if is_rust and state.raw_response_hash then
          evidence = { raw_response_hash = state.raw_response_hash }
          local validation = request.parser.terminal and request.parser.terminal.action_validation
          if type(validation) == "table" then evidence.action_validation = vim.deepcopy(validation) end
        end
        lifecycle(state, "invalid_output", raw_or_err, evidence)
        counters.invalid_output = counters.invalid_output + 1
        last_status = raw_or_err
        consecutive_failures = consecutive_failures + 1
      else
        consecutive_failures, backoff_until_ms = 0, 0
        state.responded_at_ms = now()
        local raw_response = table.concat(request.parser.raw_chunks)
        if not state.raw_response_hash then
          state.raw_response_hash = util.sha256hex(raw_response)
          collector.store_prediction_blob(raw_response, function() end)
        end
        if is_v1 then
          action.text = action.text or ""
          local canonical_action = vim.json.encode({ kind = action.kind, text =
            (action.kind == "keep" or action.kind == "delete_line") and vim.NIL or action.text })
          state.action_blob_hash = util.sha256hex(canonical_action)
          collector.store_prediction_blob(canonical_action, function() end)
          emit(state, "prediction_generated", { raw_response_hash = state.raw_response_hash,
            action_blob_hash = state.action_blob_hash, canonical_action = action.kind,
            stop_type = request.parser.terminal.stop_type, context_hash = state.context_hash,
            context_policy_version = state.context_policy_version or single_line_v1.CONTEXT_POLICY_VERSION,
            model_alias = is_rust and state.model_identity.alias or nil,
            model_sha256 = is_rust and state.model_identity.model_sha256 or nil,
            model_protocol = is_rust and state.model_identity.model_protocol or nil,
            context_layout = is_rust and state.model_identity.context_layout or nil,
            runtime_config_hash = is_rust and state.model_identity.runtime_config_hash or nil,
            wire_version = single_line_v1.WIRE_VERSION,
            prompt_tokens = state.prompt_tokens,
            max_output_tokens = is_rust and state.model_identity.output_tokens or single_line_v1.MAX_ACTION_TOKENS,
            first_chunk_at_ms = request.first_chunk_at_ms,
            first_token_at_ms = request.first_token_at_ms,
            first_token_observation = request.parser.first_token_source,
            first_text_at_ms = request.first_text_at_ms,
            completed_at_ms = state.responded_at_ms,
            model_elapsed_ms = state.responded_at_ms - state.requested_at_ms,
            backend_timings = request.parser.terminal.timings })
        elseif action.action == "replace" then
          state.action_blob_hash = action.text == "" and state.raw_response_hash
            or util.sha256hex(action.text)
          if action.text ~= "" then collector.store_prediction_blob(action.text, function() end) end
        else
          state.action_blob_hash = state.raw_response_hash
        end
        if not is_v1 then
          emit(state, "prediction_generated", { raw_response_hash = state.raw_response_hash,
            action_blob_hash = state.action_blob_hash, canonical_action = action.action,
            stop_type = request.parser.terminal.stop_type, context_hash = state.context_hash,
            prompt_tokens = state.prompt_tokens, context_expanded = state.prompt_tokens
            and state.prompt_tokens > opts.target_prompt_tokens or false,
            first_chunk_at_ms = request.first_chunk_at_ms,
            first_token_at_ms = request.first_token_at_ms,
            first_token_observation = request.parser.first_token_source,
            first_text_at_ms = request.first_text_at_ms,
            completed_at_ms = state.responded_at_ms,
            model_elapsed_ms = state.responded_at_ms - state.requested_at_ms,
            backend_timings = request.parser.terminal.timings })
        end
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
start_generation = function(request)
  local state = request.state
  request.kind = "generation"
  request.parser = sse.new(opts.max_response_bytes)
  local is_v1 = opts.protocol_version == single_line_v1.WIRE_VERSION
  local body_table
  if opts.backend == "rust-editor-v1" then
    body_table = { prompt = state.prompt, n_predict = state.model_identity.output_tokens,
      window = state.window, repository_identity = state.repo_identity, cache_prompt = true }
  else
    body_table = { prompt = state.prompt,
      n_predict = is_v1 and single_line_v1.MAX_ACTION_TOKENS or 96, temperature = 0,
      stream = true, cache_prompt = true, id_slot = 0, n_keep = 0 }
  end
  local body = vim.json.encode(body_table)
  local function chunk(data)
    if data and data ~= "" then
      if not request.first_chunk_at_ms then request.first_chunk_at_ms = now() end
      local ok = sse.feed(request.parser, data)
      if not ok and request.process then pcall(request.process.kill, request.process, "sigterm") end
      if request.parser.first_token and not request.first_token_at_ms then request.first_token_at_ms = now() end
      if request.parser.first_text and not request.first_text_at_ms then request.first_text_at_ms = now() end
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
local function count_prompt_tokens(request, prompt, callback)
  request.kind = "tokenize"
  local function complete(result)
    vim.schedule(function()
      if pending ~= request then return end
      if request.obsolete then finished(request, { code = 0 }); return end
      local ok, decoded = pcall(vim.json.decode, result.stdout or "")
      if result.code ~= 0 or not ok or type(decoded) ~= "table"
          or type(decoded.tokens) ~= "table" then
        finished(request, { code = result.code ~= 0 and result.code or 1 })
        return
      end
      if request.process then request.process = nil end
      callback(#decoded.tokens)
    end)
  end
  local body = vim.json.encode({ content = prompt, add_special = true, parse_special = true })
  if M._tokenize_impl then
    request.process = M._tokenize_impl(prompt, true, true, complete)
  else
    request.process = vim.system({ "curl", "-sS", "-f", "-m", "5", "-X", "POST",
      "-H", "Content-Type: application/json", "--data-binary", "@-", opts.url .. "/tokenize" },
      { stdin = body, text = true, timeout = 6000 }, complete)
  end
end
local function tokenize_single_line(request)
  local state = request.state
  local context = single_line_v1.new_context(state.contract_state)
  local input_budget = opts.single_line_input_tokens
  if type(input_budget) ~= "number" or input_budget % 1 ~= 0 or input_budget < 1
      or input_budget + single_line_v1.MAX_ACTION_TOKENS > single_line_v1.MAX_TOTAL_TOKENS then
    request.context_error = "single-line input token budget is outside the validated total context"
    finished(request, { code = 0 })
    return
  end
  local function context_failure(reason)
    request.context_error = reason
    finished(request, { code = 0 })
  end
  local function fit_prompt(prompt, on_fit)
    count_prompt_tokens(request, prompt, function(count)
      if count > input_budget or count + single_line_v1.MAX_ACTION_TOKENS
          > single_line_v1.MAX_TOTAL_TOKENS then
        on_fit(false, count)
      else
        on_fit(true, count)
      end
    end)
  end
  local function take_next()
    local candidate = context:next_prompt()
    if not candidate then
      local prompt = context:prompt()
      fit_prompt(prompt, function(fits, count)
        if not fits then context_failure("mandatory target and markers exceed input budget")
        else
          state.prompt, state.prompt_tokens = prompt, count
          state.context_hash = util.sha256hex(prompt)
          request.context_included = context:included()
          record_request(state)
          start_generation(request)
        end
      end)
      return
    end
    fit_prompt(candidate, function(fits, count)
      context:resolve(fits)
      if fits then request.last_context_count = count end
      take_next()
    end)
  end
  local initial = context:prompt()
  fit_prompt(initial, function(fits, count)
    if not fits then context_failure("mandatory target and markers exceed input budget")
    else
      request.last_context_count = count
      take_next()
    end
  end)
end
local function shadow_source(buf, shadow)
  local lines = shadow and shadow.lines
  if type(lines) ~= "table" then return nil end
  local ok_ff, fileformat = pcall(vim.api.nvim_get_option_value, "fileformat", { buf = buf })
  local ok_eol, eol = pcall(vim.api.nvim_get_option_value, "eol", { buf = buf })
  local separator = ok_ff and fileformat == "dos" and "\r\n" or "\n"
  local total = math.max(0, #lines - 1) * #separator
  if ok_eol and eol and #lines > 0 then total = total + #separator end
  for _, line in ipairs(lines) do
    total = total + #line
    if total > 65536 then return nil end
  end
  local source = table.concat(lines, separator)
  if ok_eol and eol and #lines > 0 then source = source .. separator end
  return source
end
local function editor_context_buffers(state)
  local root = repository.cache.root
  if not root or root == "" then return {} end
  local root_abs = util.abspath(root):gsub("/+$", "")
  if root_abs == "" then return {} end
  local candidates = {}
  local current_path = util.abspath(state.path)
  for _, buf in ipairs(vim.api.nvim_list_bufs()) do
    if buf ~= state.bufnr and vim.api.nvim_buf_is_valid(buf) and vim.api.nvim_buf_is_loaded(buf)
        and not util.is_excluded_buffer(buf) and not vim.b[buf].tabcomplete_predictor_off
        and buffers.is_tracked(buf) then
      local path = vim.api.nvim_buf_get_name(buf)
      local full_path = util.abspath(path)
      local rel = full_path:sub(1, #root_abs + 1) == root_abs .. "/"
        and full_path:sub(#root_abs + 2) or nil
      local shadow = buffers.get_shadow(buf)
      if rel and rel ~= "" and full_path ~= current_path and shadow and shadow.file == path then
        local source = shadow_source(buf, shadow)
        if source and #source <= 65536 then
          local infos = vim.fn.getbufinfo({ bufnr = buf })
          local recency = infos[1] and infos[1].lastused or buf
          candidates[#candidates + 1] = { path = rel, source = source,
            recency = tonumber(recency) or buf, bufnr = buf }
        end
      end
    end
  end
  table.sort(candidates, function(a, b)
    if a.recency == b.recency then return a.bufnr > b.bufnr end
    return a.recency > b.recency
  end)
  local result, seen = {}, {}
  for _, item in ipairs(candidates) do
    if not seen[item.path] then
      result[#result + 1] = { path = item.path, source = item.source, recency = item.recency }
      seen[item.path] = true
      if #result == 8 then break end
    end
  end
  return result
end
local function post_rust_json(path, body, callback)
  if M._backend_post_impl then
    local called = false
    local function complete(ok, response)
      if called then return end
      called = true
      vim.schedule(function() callback(ok == true, response) end)
    end
    local ok, process = pcall(M._backend_post_impl, path, vim.deepcopy(body), complete)
    if not ok then complete(false, { error = "backend test transport failed" }) end
    return process
  end
  local ok_encode, encoded = pcall(vim.json.encode, body)
  if not ok_encode then
    vim.schedule(function() callback(false, { error = "request encoding failed" }) end)
    return nil
  end
  local base = opts.url:gsub("/+$", "")
  local timeout_ms = path == "/v1/model" and 180000 or 30000
  return vim.system({ "curl", "-sS", "-m", tostring(math.ceil(timeout_ms / 1000)), "-X", "POST", "-H",
    "Content-Type: application/json", "--data-binary", "@-", "-w", "\n%{http_code}",
    base .. path }, { stdin = encoded, text = true, stderr = false, timeout = timeout_ms + 1000 }, function(result)
      local raw = result.stdout or ""
      local code_text = raw:match("(%d%d%d)%s*$")
      local http_code = code_text and tonumber(code_text) or nil
      local response_body = code_text and raw:gsub("\n?%d%d%d%s*$", "", 1) or ""
      if result.code ~= 0 then
        vim.schedule(function() callback(false, { exit_code = result.code, http_code = http_code }) end)
      elseif http_code and http_code >= 200 and http_code < 300 then
        local decoded_ok, decoded = pcall(vim.json.decode, response_body)
        vim.schedule(function()
          callback(decoded_ok and type(decoded) == "table", decoded_ok and decoded or
            { http_code = http_code, error = "invalid backend JSON" })
        end)
      else
        vim.schedule(function() callback(false, { http_code = http_code }) end)
      end
    end)
end
local function get_rust_json(path, callback)
  if M._backend_get_impl then
    local called = false
    local function complete(ok, response)
      if called then return end
      called = true
      vim.schedule(function() callback(ok == true, response) end)
    end
    local ok, process = pcall(M._backend_get_impl, path, complete)
    if not ok then complete(false, { error = "backend test transport failed" }) end
    return process
  end
  local base = opts.url:gsub("/+$", "")
  return vim.system({ "curl", "-sS", "-m", "5", "-X", "GET", "-w", "\n%{http_code}",
    base .. path }, { text = true, stderr = false, timeout = 6000 }, function(result)
      local raw = result.stdout or ""
      local code_text = raw:match("(%d%d%d)%s*$")
      local http_code = code_text and tonumber(code_text) or nil
      local response_body = code_text and raw:gsub("\n?%d%d%d%s*$", "", 1) or ""
      if result.code ~= 0 or not http_code or http_code < 200 or http_code >= 300 then
        vim.schedule(function() callback(false, { exit_code = result.code, http_code = http_code }) end)
        return
      end
      local decoded_ok, decoded = pcall(vim.json.decode, response_body)
      vim.schedule(function()
        callback(decoded_ok and type(decoded) == "table", decoded_ok and decoded or
          { http_code = http_code, error = "invalid backend JSON" })
      end)
    end)
end
local function validate_editor_context(state, response)
  if type(response) ~= "table" or type(response.prompt) ~= "string"
      or not is_integer(response.prompt_tokens) or response.prompt_tokens < 1
      or type(response.context_hash) ~= "string"
      or not is_sha256(response.context_hash)
      or util.sha256hex(response.prompt) ~= response.context_hash then
    return nil, "Rust editor context response failed prompt validation"
  end
  local identity, identity_err = validate_model_identity(response.model_identity)
  if not identity then return nil, identity_err end
  if response.model_protocol ~= identity.model_protocol then
    return nil, "Rust editor context model protocol mismatch"
  end
  if response.prompt_tokens > math.min(opts.single_line_input_tokens, identity.input_tokens)
      or response.prompt_tokens + identity.output_tokens > identity.context_size then
    return nil, "Rust editor context exceeds the configured token budget"
  end
  local expected_layout, expected_policy = expected_context_layout(identity, model_spec(identity.alias))
  if response.context_layout ~= nil and response.context_layout ~= expected_layout then
    return nil, "Rust editor context layout does not match the model identity"
  end
  if response.context_policy_version ~= expected_policy then
    return nil, "Rust editor context policy does not match the model protocol"
  end
  if type(response.selected_buffers) ~= "table" or #response.selected_buffers > 8 then
    return nil, "Rust editor context selected an invalid buffer set"
  end
  local requested = {}
  for _, buffer in ipairs(state.context_buffers or {}) do requested[buffer.path] = true end
  local selected_count, selected_paths = 0, {}
  for index, path in pairs(response.selected_buffers) do
    if not is_integer(index) or index < 1 or index > #response.selected_buffers then
      return nil, "Rust editor context selected a malformed buffer array"
    end
    selected_count = selected_count + 1
    if type(path) ~= "string" or not requested[path] or selected_paths[path] then
      return nil, "Rust editor context selected an unrequested buffer"
    end
    selected_paths[path] = true
  end
  if selected_count ~= #response.selected_buffers then
    return nil, "Rust editor context selected a sparse buffer array"
  end
  if identity.model_protocol == "sweep-full-file-v1" then
    local window = response.window
    if type(window) ~= "table" or type(window.source) ~= "string"
        or not is_integer(window.target_row) or window.target_row < 0
        or not is_integer(window.start_row) or window.start_row < 0 then
      return nil, "Rust Sweep context window is invalid"
    end
    local source_ok, source_lines = pcall(single_line_v1.physical_lines, state.contract_state.source)
    local window_ok, window_lines = pcall(single_line_v1.physical_lines, window.source)
    if not source_ok or not window_ok then return nil, "Rust Sweep context contains invalid line endings" end
    if #window_lines == 0 or window.target_row >= #window_lines
        or window.start_row + window.target_row ~= state.contract_state.target_row
        or window.start_row + #window_lines > #source_lines then
      return nil, "Rust Sweep context window does not cover the target line"
    end
    for i, line in ipairs(window_lines) do
      local original = source_lines[window.start_row + i]
      if not original or original.content ~= line.content or original.terminator ~= line.terminator then
        return nil, "Rust Sweep context window diverges from the editor source"
      end
    end
  elseif response.window ~= nil and response.window ~= vim.NIL then
    return nil, "Rust single-line context unexpectedly included a window"
  end
  local window
  if response.window ~= nil and response.window ~= vim.NIL then window = response.window end
  return {
    prompt = response.prompt, prompt_tokens = response.prompt_tokens,
    context_hash = response.context_hash, context_policy_version = response.context_policy_version,
    model_identity = identity, model_protocol = response.model_protocol, context_layout = expected_layout,
    window = window,
    selected_buffers = response.selected_buffers,
  }
end
local function tokenize_rust(request)
  local state = request.state
  state.context_buffers = editor_context_buffers(state)
  local body = { state = state.contract_state, buffers = state.context_buffers,
    repository_identity = state.repo_identity }
  local attempts = 0
  local function attempt()
    attempts = attempts + 1
    request.kind = "context"
    request.process = post_rust_json("/v1/editor/context", body, function(ok, response)
      if pending ~= request then return end
      if request.obsolete then finished(request, { code = 0 }); return end
      request.process = nil
      if not ok and response and response.http_code == 409 and attempts < 120 then
        vim.defer_fn(function()
          if pending == request then
            if request.obsolete then finished(request, { code = 0 })
            else attempt() end
          end
        end, 250)
        return
      end
      if not ok then
        finished(request, { code = 1 })
        return
      end
      local context, err = validate_editor_context(state, response)
      if not context then
        request.protocol_error = err
        finished(request, { code = 0 })
        return
      end
      if not still_current(state) then
        finished(request, { code = 0 })
        return
      end
      state.prompt, state.prompt_tokens = context.prompt, context.prompt_tokens
      state.context_hash, state.context_policy_version = context.context_hash, context.context_policy_version
      state.model_identity, state.model_protocol = context.model_identity, context.model_protocol
      state.context_layout = context.context_layout
      state.window, state.selected_buffers = context.window, context.selected_buffers
      opts.current_model_identity = context.model_identity
      opts.model = context.model_identity.alias
      opts.model_revision = context.model_identity.model_sha256
      opts.model_protocol = context.model_identity.model_protocol
      opts.runtime_config_hash = context.model_identity.runtime_config_hash
      opts.context_policy_version = context.context_policy_version
      record_request(state)
      start_generation(request)
    end)
  end
  attempt()
end
local function tokenize(request)
  if opts.backend == "rust-editor-v1" then
    tokenize_rust(request)
    return
  end
  if opts.protocol_version == single_line_v1.WIRE_VERSION then
    tokenize_single_line(request)
    return
  end
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
  local request = { state = state, kind = "tokenize",
    has_lock = not M._request_impl and not M._tokenize_impl }
  pending = request -- set before an immediate test callback can run
  phase = "request_running"
  collector.anchor_prediction(state.bufnr)
  if opts.protocol_version ~= single_line_v1.WIRE_VERSION then record_request(state) end
  tokenize(request)
  return true
end
function M.accept()
  if not active or not active.action then return false, "no active proposal" end
  local state = active
  if not still_current(state) then dismiss("navigation"); return false, "stale proposal" end
  if opts.protocol_version == single_line_v1.WIRE_VERSION then
    if state.action.kind == "keep" then return false, "no edit" end
    applying = true
    local applied, apply_err = single_line_v1.apply_to_buffer(state.bufnr, state.contract_state, state.action)
    applying = false
    if not applied then
      last_status = "acceptance blocked: " .. tostring(apply_err)
      return false, last_status
    end
    accepted_tick[state.bufnr] = vim.api.nvim_buf_get_changedtick(state.bufnr)
    local delta_sequence = collector.seq
    local accepted_text = state.action.text or ""
    local accepted = emit(state, "prediction_accepted", { accepted_at_ms = now(), shown_event_id = state.shown_event_id,
      accepted_chars = vim.fn.strchars(accepted_text),
      accepted_lines = state.action.kind == "delete_line" and 0 or 1,
      total_chars = vim.fn.strchars(accepted_text), applied_through_sequence = delta_sequence,
      visible_duration_ms = now() - state.shown_at_ms,
      action = state.action.kind, wire_version = single_line_v1.WIRE_VERSION,
      editable_range = state.action_range,
      proposed_start_byte = state.action_range.start_byte,
      proposed_end_byte = state.action_range.end_byte })
    if accepted then
      last_decision = { state = state, outcome = "accepted", event_id = accepted.event_id,
        session_id = collector.session_id, synthetic = opts.synthetic, reviewed = false }
    end
    counters.accepted = counters.accepted + 1
    close_preview()
    active = nil
    stop_timer(expiry)
    phase = "idle"
    last_status = "accepted"
    return true
  end
  if state.action.action ~= "replace" then return false, "no edit" end
  local replacement = vim.split(state.action.text, "\n", { plain = true })
  vim.bo[state.bufnr].undolevels = vim.bo[state.bufnr].undolevels
  applying = true
  vim.api.nvim_buf_set_text(state.bufnr, state.row, state.start_col, state.row, state.end_col, replacement)
  accepted_tick[state.bufnr] = vim.api.nvim_buf_get_changedtick(state.bufnr)
  local delta_sequence = collector.seq
  local accepted = emit(state, "prediction_accepted", { accepted_at_ms = now(), shown_event_id = state.shown_event_id,
    accepted_chars = vim.fn.strchars(state.action.text), accepted_lines = #replacement,
    total_chars = vim.fn.strchars(state.action.text), applied_through_sequence = delta_sequence,
    visible_duration_ms = now() - state.shown_at_ms })
  if accepted then
    last_decision = { state = state, outcome = "accepted", event_id = accepted.event_id,
      session_id = collector.session_id, synthetic = opts.synthetic, reviewed = false }
  end
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
function M.review_last()
  local decision = last_decision
  if not decision or decision.reviewed then return false, "no unreviewed explicit decision" end
  if decision.synthetic or opts.synthetic then
    return false, "synthetic decisions cannot be human-reviewed"
  end
  if decision.session_id ~= collector.session_id then return false, "decision belongs to another session" end
  local confirm = M._confirm_impl or vim.fn.confirm
  local label = decision.outcome == "accepted" and "accepted" or "explicitly rejected"
  local answer = confirm("Confirm you personally reviewed the " .. label
    .. " TabComplete proposal " .. decision.state.prediction_id:sub(1, 8) .. "?",
    "&Confirm\n&Cancel", 2)
  if answer ~= 1 then return false, "review cancelled" end
  local event = emit(decision.state, "prediction_reviewed", {
    resolution_event_id = decision.event_id, outcome = decision.outcome,
    review_source = "explicit_editor_confirmation", human_verified = true,
    reviewed_at_ms = now(),
  })
  if not event then return false, "collector unavailable; review not recorded" end
  decision.reviewed = true
  last_status = "decision review recorded"
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
function M.set_model(alias, callback)
  if opts.backend ~= "rust-editor-v1" then return false, "model switching requires the Rust backend" end
  if not model_spec(alias) then return false, "model alias is not allowlisted" end
  local selected_identity = opts.current_model_identity
  if selected_identity and selected_identity.model_switch_supported == false then
    return false, "model selection is declarative; choose the " .. alias
      .. " Nix model variant and rebuild the package"
  end
  if model_switching then return false, "model switch already in progress" end
  local previous_mode = mode
  model_switching = true
  M.set_mode("off") -- cancels in-flight work before the service model request
  last_status = "switching model"
  local attempts = 0
  local function restore(ok, message)
    model_switching = false
    M.set_mode(previous_mode)
    last_status = ok and ("model switched to " .. alias) or ("model switch failed: " .. message)
    if callback then callback(ok, message) end
  end
  local function attempt()
    attempts = attempts + 1
    post_rust_json("/v1/model", { alias = alias }, function(ok, identity)
      if not ok and identity and identity.http_code == 409 and attempts < 120 then
        vim.defer_fn(attempt, 250)
        return
      end
      if not ok then restore(false, "Rust service did not accept the model switch"); return end
      local validated, err = validate_model_identity(identity, alias)
      if not validated then restore(false, err); return end
      opts.current_model_identity = validated
      opts.model = validated.alias
      opts.model_revision = validated.model_sha256
      opts.model_protocol = validated.model_protocol
      opts.runtime_config_hash = validated.runtime_config_hash
      local _, expected_policy = expected_context_layout(validated, model_spec(validated.alias))
      opts.context_policy_version = expected_policy
      last_fingerprint = nil
      restore(true, nil)
    end)
  end
  attempt()
  return true
end
function M.refresh_model_identity(callback)
  if opts.backend ~= "rust-editor-v1" then return false, "model identity requires the Rust backend" end
  model_identity_refresh_generation = model_identity_refresh_generation + 1
  local generation = model_identity_refresh_generation
  get_rust_json("/health", function(ok, identity)
    if generation ~= model_identity_refresh_generation or opts.backend ~= "rust-editor-v1" then
      if callback then callback(false, "model identity refresh became stale") end
      return
    end
    if not ok then
      if callback then callback(false, "cannot read the installed Rust model identity") end
      return
    end
    local validated, err = validate_model_identity(identity)
    if not validated then
      if callback then callback(false, err) end
      return
    end
    opts.current_model_identity = validated
    opts.model = validated.alias
    opts.model_revision = validated.model_sha256
    opts.model_protocol = validated.model_protocol
    opts.runtime_config_hash = validated.runtime_config_hash
    local _, expected_policy = expected_context_layout(validated, model_spec(validated.alias))
    opts.context_policy_version = expected_policy
    if callback then callback(true, validated) end
  end)
  return true
end
function M.model_aliases()
  local identity = opts.current_model_identity
  if identity and identity.model_switch_supported == false then
    return identity.alias and { identity.alias } or {}
  end
  local aliases = {}
  for alias in pairs(opts.allowed_models or {}) do aliases[#aliases + 1] = alias end
  table.sort(aliases)
  return aliases
end
function M.status()
  local cs = collector.status()
  local identity = opts.current_model_identity
  local model_selection = identity and identity.model_selection or nil
  if identity and identity.model_switch_supported == false and model_selection == nil then
    model_selection = "declarative"
  end
  return { mode = mode, stage = phase, state = last_status, model = opts.model,
    backend = opts.backend,
    selected_model = identity and identity.alias or opts.model,
    model_alias = identity and identity.alias or nil,
    model_embedded = identity and identity.model_embedded,
    model_switch_supported = identity and identity.model_switch_supported,
    model_selection = model_selection,
    model_storage = identity and identity.model_storage,
    model_protocol = identity and identity.model_protocol
      or opts.model_protocol,
    context_layout = identity and identity.context_layout or nil,
    protocol_version = opts.protocol_version,
    context_policy_version = identity and opts.context_policy_version
      or (opts.protocol_version == single_line_v1.WIRE_VERSION
        and single_line_v1.CONTEXT_POLICY_VERSION or opts.context_policy_version),
    input_token_budget = opts.protocol_version == single_line_v1.WIRE_VERSION
      and opts.single_line_input_tokens or opts.max_prompt_tokens,
    revision = opts.model_revision, precision = opts.precision,
    automatic_state = mode == "automatic" and "automatic experimental" or "inactive",
    quality = opts.automatic_quality_validated and "validated" or "uncalibrated",
    experimental_auto_opt_in = opts.experimental_auto_opt_in,
    automatic_quality_validated = opts.automatic_quality_validated,
    automatic_display_policy = opts.automatic_prefix_guard
      and "prefix-preserving-completion-v1" or "next-edit",
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
  local previous_backend = opts.backend
  opts = vim.tbl_deep_extend("force", opts, options or {})
  model_identity_refresh_generation = model_identity_refresh_generation + 1
  if opts.backend ~= previous_backend then opts.current_model_identity = nil end
  if opts.backend == "rust-editor-v1" then
    if not (options and options.url) and previous_backend ~= "rust-editor-v1" then
      opts.url = vim.env.TABCOMPLETE_PREDICTOR_URL or "http://127.0.0.1:19094"
    end
    opts.protocol_version = single_line_v1.WIRE_VERSION
  end
  opts.allowed_models = opts.allowed_models or default_allowed_models
  if opts.protocol_version ~= "compact-next-edit-v1"
      and opts.protocol_version ~= single_line_v1.WIRE_VERSION then
    error("unsupported TabComplete prediction protocol")
  end
  local port = opts.url:match(":(%d+)%s*$") or "19093"
  lock_path = "/tmp/tabcomplete-predictor-" .. port .. ".lock"
  if group then pcall(vim.api.nvim_del_augroup_by_id, group) end
  invalidate("setup")
  recent_edit = {}
  last_fingerprint = nil
  mode = load_mode() or opts.mode
  if mode == "automatic" and not (opts.experimental_auto_opt_in or opts.automatic_quality_validated) then
    mode = "manual"
  end
  group = vim.api.nvim_create_augroup("TabCompletePredict", { clear = true })
  vim.api.nvim_create_autocmd("ColorScheme", {
    group = group,
    callback = diff_preview_highlights,
  })
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
  vim.api.nvim_create_autocmd("User", {
    group = group, pattern = { "BlinkCmpMenuOpen", "BlinkCmpMenuClose" },
    callback = function(args)
      if args.match == "BlinkCmpMenuOpen" then
        latest_wanted = false
        invalidate("completion UI")
      else
        debounce_changed()
      end
    end,
  })
  vim.api.nvim_create_autocmd("CursorMoved", {
    group = group, callback = function()
      if applying or util.current_mode():find("^i") then return end
      if active or pending then invalidate("navigation") end
    end,
  })
  vim.api.nvim_create_autocmd({ "BufLeave", "BufWipeout", "FocusLost", "InsertLeave" }, {
    group = group, callback = function(args)
      if applying then return end
      -- Scratch buffers used for exact edit validation are unrelated to the
      -- source proposal. Their removal must not close its feedback outcome.
      if args.event == "BufWipeout" and args.buf ~= vim.api.nvim_get_current_buf()
          and (not active or args.buf ~= active.bufnr)
          and (not pending or args.buf ~= pending.state.bufnr) then return end
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
