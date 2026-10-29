-- tabcomplete_nes: next-edit suggestions from the local tabcomplete-nes engine.
--
-- The engine (Rust, llama.cpp, Sweep next-edit model) owns all context: edit
-- history, workspace retrieval and the prompt. This client sends buffer
-- snapshots, renders the proposed edit, and reports the suggestion lifecycle
-- to the trajectory collector when it is running.
--
-- With `stream`, the engine sends provisional ghost text while the cursor
-- line is still being decoded; it is drawn at once and replaced (or cleared)
-- by the final answer.
--
-- Keys: <Tab> jumps to a distant edit, then accepts (LazyVim's `ai_nes` hook in
-- insert mode, a fallback-preserving mapping in normal mode). <M-l> accepts,
-- <M-BS> dismisses, <M-p> requests a prediction explicitly.

local M = {}

local ns = vim.api.nvim_create_namespace("tabcomplete_nes")

M.config = {
  url = "http://127.0.0.1:19094",
  enabled = true,
  -- Short debounces: a new request kills the previous one and the engine
  -- drops stale work at its next batch, so waiting only delays the
  -- suggestion after the typing pause.
  insert_debounce_ms = 25,
  normal_debounce_ms = 60,
  normal_mode = true,
  jump = true,
  request_timeout_ms = 20000,
  max_file_bytes = 2 * 1024 * 1024,
  jump_distance = 6,
  tab = true,
  accept_key = "<M-l>",
  dismiss_key = "<M-BS>",
  predict_key = "<M-p>",
  capture_content = true,
  -- Precompute the suggestion that would follow accepting the shown one.
  prefetch = true,
  -- Stream predictions: show provisional ghost text before decoding ends.
  stream = true,
  clipboard_window_ms = 120000,
  exclude_filetypes = {
    "TelescopePrompt", "snacks_picker_input", "snacks_input", "neo-tree", "lazy", "mason", "help",
    "gitcommit", "gitrebase", "oil", "minifiles", "noice", "notify", "qf", "trouble",
  },
}

local state = {
  seq = 0,
  timer = nil,
  job = nil,
  suggestion = nil,
  yank = nil,
  health = nil,
  expected_cursor = nil,
  applying = false,
  stats = { requested = 0, shown = 0, accepted = 0, dismissed = 0, errors = 0 },
}

local function now_ms()
  local sec, usec = vim.uv.gettimeofday()
  return math.floor(sec * 1000 + usec / 1000)
end

local function uuid()
  local f = io.open("/proc/sys/kernel/random/uuid", "r")
  if f then
    local u = f:read("*l")
    f:close()
    if u then return u end
  end
  return string.format("%08x-%04x", math.random(0, 0x7fffffff), math.random(0, 0xffff))
end

local function collector()
  local ok, c = pcall(require, "tabcomplete_trajectory")
  if ok and c and c.started then return c end
  return nil
end

local function emit(bufnr, kind, payload)
  local c = collector()
  if not c then return nil end
  local ok, ev = pcall(c.emit, bufnr, kind, payload)
  return ok and ev or nil
end

local function store_blob(text)
  local c = collector()
  if not c or not M.config.capture_content or not text or text == "" then return nil end
  local ok, hash = pcall(vim.fn.sha256, text)
  if not ok then return nil end
  pcall(c.store_prediction_blob, text, function() end)
  return hash
end

local function is_secret(path)
  local ok, util = pcall(require, "tabcomplete_trajectory.util")
  if ok and util.is_secret_path then
    local ok2, secret = pcall(util.is_secret_path, path)
    return ok2 and secret
  end
  local base = path:match("[^/]+$") or path
  return base:match("^%.env") ~= nil or base:match("%.pem$") ~= nil or base:match("%.key$") ~= nil
end

local function eligible(bufnr)
  if not M.config.enabled or not vim.api.nvim_buf_is_valid(bufnr) then return false end
  local bo = vim.bo[bufnr]
  if bo.buftype ~= "" or not bo.modifiable or bo.readonly then return false end
  if vim.tbl_contains(M.config.exclude_filetypes, bo.filetype) then return false end
  local name = vim.api.nvim_buf_get_name(bufnr)
  if name == "" or is_secret(name) then return false end
  return true
end

local function buffer_content(bufnr)
  local lines = vim.api.nvim_buf_get_lines(bufnr, 0, -1, false)
  local content = table.concat(lines, "\n")
  if vim.bo[bufnr].eol or vim.bo[bufnr].fixeol then content = content .. "\n" end
  return content
end

local function root_for(bufnr)
  local ok, root = pcall(vim.fs.root, bufnr, { ".git", ".hg" })
  if ok and root then return root end
  return nil
end

local function post(path, body, cb)
  local ok, json = pcall(vim.json.encode, body)
  if not ok then return nil end
  local timeout = math.max(1, math.ceil(M.config.request_timeout_ms / 1000))
  local ok_sys, job = pcall(vim.system, {
    "curl", "-sS", "-m", tostring(timeout), "-H", "Content-Type: application/json",
    "--data-binary", "@-", M.config.url .. path,
  }, { stdin = json, text = true }, function(res)
    if not cb then return end
    vim.schedule(function()
      if res.code ~= 0 then
        cb(nil, (res.stderr ~= "" and res.stderr) or ("curl exit " .. res.code))
        return
      end
      local ok_dec, decoded = pcall(vim.json.decode, res.stdout)
      if not ok_dec or type(decoded) ~= "table" then
        cb(nil, "invalid engine response")
        return
      end
      cb(decoded, nil)
    end)
  end)
  return ok_sys and job or nil
end

-- POST with a newline-delimited JSON response: `on_event` gets each decoded
-- line as it arrives, `on_done` the transport error (or nil) at the end.
local function post_stream(path, body, on_event, on_done)
  local ok, json = pcall(vim.json.encode, body)
  if not ok then return nil end
  local timeout = math.max(1, math.ceil(M.config.request_timeout_ms / 1000))
  local pending = ""
  local ok_sys, job = pcall(vim.system, {
    "curl", "-sS", "-N", "-m", tostring(timeout), "-H", "Content-Type: application/json",
    "--data-binary", "@-", M.config.url .. path .. "?stream=1",
  }, {
    stdin = json,
    text = true,
    stdout = function(_, data)
      if not data then return end
      pending = pending .. data
      while true do
        local nl = pending:find("\n", 1, true)
        if not nl then break end
        local line = pending:sub(1, nl - 1)
        pending = pending:sub(nl + 1)
        if line ~= "" then
          vim.schedule(function()
            local ok_dec, ev = pcall(vim.json.decode, line)
            if ok_dec and type(ev) == "table" then on_event(ev) end
          end)
        end
      end
    end,
  }, function(res)
    -- Runs after stdout has closed, so every event was scheduled before it.
    vim.schedule(function()
      -- An engine without streaming answers with one unterminated JSON body.
      if pending ~= "" then
        local ok_dec, ev = pcall(vim.json.decode, pending)
        pending = ""
        if ok_dec and type(ev) == "table" then on_event(ev) end
      end
      if res.code ~= 0 then
        on_done((res.stderr ~= "" and res.stderr) or ("curl exit " .. res.code))
      else
        on_done(nil)
      end
    end)
  end)
  return ok_sys and job or nil
end

local function refresh_health()
  vim.system({ "curl", "-sS", "-m", "2", M.config.url .. "/health" }, { text = true }, function(res)
    if res.code ~= 0 then return end
    local ok, decoded = pcall(vim.json.decode, res.stdout)
    if ok and type(decoded) == "table" then state.health = decoded end
  end)
end

-- Rendering ---------------------------------------------------------------

local function clear_marks(bufnr)
  if bufnr and vim.api.nvim_buf_is_valid(bufnr) then
    vim.api.nvim_buf_clear_namespace(bufnr, ns, 0, -1)
  end
end

local function split_text(text)
  return vim.split(text, "\n", { plain = true })
end

-- Buffer text currently covered by an edit range.
local function range_text(bufnr, e)
  local count = vim.api.nvim_buf_line_count(bufnr)
  if e.start_line >= count then return "" end
  local end_line, end_col = e.end_line, e.end_col
  if end_line >= count then
    end_line = count - 1
    end_col = #(vim.api.nvim_buf_get_lines(bufnr, end_line, end_line + 1, false)[1] or "")
  end
  local ok, parts = pcall(vim.api.nvim_buf_get_text, bufnr, e.start_line, e.start_col, end_line, end_col, {})
  if not ok then return nil end
  local text = table.concat(parts, "\n")
  if e.end_line >= count and e.end_line > e.start_line then text = text .. "\n" end
  return text
end

local function virt_lines(lines, hl)
  local out = {}
  for _, l in ipairs(lines) do
    out[#out + 1] = { { l == "" and " " or l, hl } }
  end
  return out
end

local function cursor_pos()
  local c = vim.api.nvim_win_get_cursor(0)
  return c[1] - 1, c[2]
end

local function is_far(e)
  local row = cursor_pos()
  if math.abs(e.start_line - row) > M.config.jump_distance then return true end
  local top, bot = vim.fn.line("w0") - 1, vim.fn.line("w$") - 1
  return e.start_line < top or e.start_line > bot
end

-- Classify and draw one edit. Returns the display kind.
local function render(bufnr, s)
  clear_marks(bufnr)
  local e = s.edits[s.idx]
  local count = vim.api.nvim_buf_line_count(bufnr)
  local old = s.replaced or ""
  local new_lines = split_text(e.text)
  local display
  local row, col = cursor_pos()
  local function mark(line, c, opts)
    line = math.min(line, count - 1)
    local text = vim.api.nvim_buf_get_lines(bufnr, line, line + 1, false)[1] or ""
    pcall(vim.api.nvim_buf_set_extmark, bufnr, ns, line, math.min(c, #text), opts)
  end
  if old == "" then
    if e.start_line >= count or (e.start_col == 0 and e.text:sub(-1) == "\n") then
      -- Whole new lines.
      local lines = vim.deepcopy(new_lines)
      if lines[#lines] == "" then table.remove(lines) end
      if e.start_line >= count then
        mark(count - 1, 0, { virt_lines = virt_lines(lines, "TabCompleteNesGhost") })
      else
        mark(e.start_line, 0, { virt_lines = virt_lines(lines, "TabCompleteNesGhost"), virt_lines_above = true })
      end
      display = "insert_lines"
    else
      local first = table.remove(new_lines, 1)
      mark(e.start_line, e.start_col, {
        virt_text = { { first, "TabCompleteNesGhost" } },
        virt_text_pos = "inline",
      })
      if #new_lines > 0 then
        mark(e.start_line, e.start_col, { virt_lines = virt_lines(new_lines, "TabCompleteNesGhost") })
      end
      display = (e.start_line == row and e.start_col == col) and "ghost" or "insert_inline"
    end
  else
    -- Replacement: strike the old range, show the new text below it.
    local end_line = math.min(e.end_line, count - 1)
    local end_col = e.end_line >= count and #(vim.api.nvim_buf_get_lines(bufnr, end_line, end_line + 1, false)[1] or "") or e.end_col
    pcall(vim.api.nvim_buf_set_extmark, bufnr, ns, e.start_line, e.start_col, {
      end_row = end_line, end_col = end_col, hl_group = "TabCompleteNesDelete",
    })
    local preview = {}
    -- Show complete lines so a mid-line replacement reads naturally.
    local first_line = vim.api.nvim_buf_get_lines(bufnr, e.start_line, e.start_line + 1, false)[1] or ""
    local last_line = vim.api.nvim_buf_get_lines(bufnr, end_line, end_line + 1, false)[1] or ""
    local full = first_line:sub(1, e.start_col) .. e.text .. last_line:sub(end_col + 1)
    preview = split_text(full)
    if e.text:sub(-1) == "\n" and e.end_col == 0 and preview[#preview] == last_line then
      table.remove(preview)
    end
    local anchor = (e.end_col == 0 and e.end_line > e.start_line) and e.end_line - 1 or end_line
    mark(anchor, 0, { virt_lines = virt_lines(preview, "TabCompleteNesAdd") })
    display = "replace"
  end
  if s.far and not s.jumped then
    mark(row, 0, {
      virt_text = { { string.format("  ⇥ Tab: edit at line %d", e.start_line + 1), "TabCompleteNesJump" } },
      virt_text_pos = "eol",
    })
    display = "jump_" .. display
  end
  return display
end

-- Suggestion lifecycle ------------------------------------------------------

local announce

local function dismiss(reason)
  local s = state.suggestion
  if not s then return end
  -- The user saw the provisional text: record it as shown before it goes.
  if s.provisional then announce(s.bufnr, s) end
  state.suggestion = nil
  clear_marks(s.bufnr)
  state.stats.dismissed = state.stats.dismissed + 1
  emit(s.bufnr, reason == "explicit" and "prediction_rejected" or "prediction_dismissed", {
    prediction_id = s.prediction_id,
    request_id = s.request_id,
    reason = reason,
    shown_event_id = s.shown_event_id,
    visible_ms = s.shown_at and (now_ms() - s.shown_at) or nil,
    typed_through = s.typed_through,
    protocol = "sweep-next-edit-v1",
  })
end

local prefetch_after_accept
local target_line

-- Draw `s` and make it the current suggestion. Returns false when its range
-- no longer exists.
local function draw(bufnr, s)
  s.replaced = range_text(bufnr, s.edits[s.idx])
  if s.replaced == nil then return false end
  local first_edit = s.edits[s.idx]
  s.target = target_line(bufnr, first_edit)
  s.target_row = first_edit.start_line
  s.far = M.config.jump and is_far(s.edits[s.idx])
  state.suggestion = s
  s.drawn_at = s.drawn_at or now_ms()
  s.display = render(bufnr, s)
  return true
end

local function show(bufnr, s)
  if not draw(bufnr, s) then return end
  announce(bufnr, s)
  prefetch_after_accept(bufnr, s)
end

-- Record a drawn suggestion as shown. A provisional suggestion is announced
-- only when the user acts on it (accept, type through, dismiss); a final
-- answer that extends it is announced with the time it first appeared.
announce = function(bufnr, s)
  s.provisional = nil
  s.shown_at = now_ms()
  state.stats.shown = state.stats.shown + 1
  local e = s.edits[s.idx]
  local ev = emit(bufnr, "prediction_shown", {
    prediction_id = s.prediction_id,
    request_id = s.request_id,
    edit_index = s.idx,
    display = s.display,
    kind = s.kind,
    jump_reason = s.jump_reason,
    proposed_start = { row = e.start_line, col = e.start_col },
    proposed_end = { row = e.end_line, col = e.end_col },
    proposed_start_byte = e.start_byte,
    proposed_end_byte = e.end_byte,
    proposed_range_end_exclusive = true,
    proposed_text = M.config.capture_content and e.text or nil,
    replaced_text = M.config.capture_content and s.replaced or nil,
    shown_at_ms = s.shown_at,
    first_drawn_at_ms = s.drawn_at,
    provisional = s.partial or nil,
    file = vim.api.nvim_buf_get_name(bufnr),
    protocol = "sweep-next-edit-v1",
    synthetic = false,
  })
  s.shown_event_id = ev and ev.event_id or nil
end

-- Predict what accepting `s` will leave in the buffer (mirrors apply_edit
-- and the cursor placement in accept) and ask the engine to compute the
-- next suggestion for that state in the background.
prefetch_after_accept = function(bufnr, s)
  if not M.config.prefetch or not s.content then return end
  local e = s.edits[s.idx]
  if not e.start_byte or e.start_line >= vim.api.nvim_buf_line_count(bufnr) then return end
  local after = s.content:sub(1, e.start_byte) .. e.text .. s.content:sub(e.end_byte + 1)
  local lines = split_text(e.text)
  local row = e.start_line + #lines - 1
  local col = #lines == 1 and (e.start_col + #lines[1]) or #lines[#lines]
  if #lines > 1 and lines[#lines] == "" then
    row = row - 1
    col = #lines[#lines - 1]
  end
  if not vim.api.nvim_get_mode().mode:match("^i") then
    local line = vim.split(after, "\n", { plain = true })[row + 1] or ""
    if col >= #line and #line > 0 then col = #line - 1 end
  end
  post("/v1/prefetch", {
    path = s.path,
    root = s.root,
    content = after,
    cursor = { line = row, col = col },
    diagnostics = s.diagnostics,
    clipboard = s.clipboard,
    jump = M.config.jump,
  }, nil)
end

local schedule_request

local function apply_edit(bufnr, e)
  local count = vim.api.nvim_buf_line_count(bufnr)
  local lines = split_text(e.text)
  if e.start_line >= count then
    -- Append after the final line (insertion at EOF past the last newline).
    if lines[#lines] == "" then table.remove(lines) end
    vim.api.nvim_buf_set_lines(bufnr, count, count, false, lines)
    return count + #lines - 1, #(lines[#lines] or "")
  end
  local end_line, end_col = e.end_line, e.end_col
  if end_line >= count then
    end_line = count - 1
    end_col = #(vim.api.nvim_buf_get_lines(bufnr, end_line, end_line + 1, false)[1] or "")
    if lines[#lines] == "" then table.remove(lines) end
  end
  vim.api.nvim_buf_set_text(bufnr, e.start_line, e.start_col, end_line, end_col, lines)
  local last_row = e.start_line + #lines - 1
  local last_col = #lines == 1 and (e.start_col + #lines[1]) or #lines[#lines]
  if #lines > 1 and lines[#lines] == "" then
    last_row = last_row - 1
    last_col = #lines[#lines - 1]
  end
  return last_row, last_col
end

local function accept()
  local s = state.suggestion
  if not s then return false end
  local bufnr = s.bufnr
  if not vim.api.nvim_buf_is_valid(bufnr) or vim.api.nvim_buf_get_changedtick(bufnr) ~= s.changedtick then
    dismiss("stale")
    return false
  end
  if s.provisional then announce(bufnr, s) end
  state.suggestion = nil
  clear_marks(bufnr)
  local e = s.edits[s.idx]
  vim.schedule(function()
    if not vim.api.nvim_buf_is_valid(bufnr) or vim.api.nvim_buf_get_changedtick(bufnr) ~= s.changedtick then return end
    state.applying = true
    local ok, row, col = pcall(apply_edit, bufnr, e)
    state.applying = false
    if not ok then
      state.stats.errors = state.stats.errors + 1
      return
    end
    if vim.api.nvim_get_current_buf() == bufnr then
      local mode = vim.api.nvim_get_mode().mode
      local line = vim.api.nvim_buf_get_lines(bufnr, row, row + 1, false)[1] or ""
      if not mode:match("^i") and col >= #line and #line > 0 then col = #line - 1 end
      state.expected_cursor = { row, col }
      pcall(vim.api.nvim_win_set_cursor, 0, { row + 1, col })
    end
    state.stats.accepted = state.stats.accepted + 1
    emit(bufnr, "prediction_accepted", {
      prediction_id = s.prediction_id,
      request_id = s.request_id,
      edit_index = s.idx,
      shown_event_id = s.shown_event_id,
      accepted_at_ms = now_ms(),
      visible_ms = s.shown_at and (now_ms() - s.shown_at) or nil,
      jumped = s.jumped or false,
      display = s.display,
      kind = s.kind,
      protocol = "sweep-next-edit-v1",
    })
    -- Chain: ask for the next edit right away.
    schedule_request(bufnr, "after_accept", 0)
  end)
  return true
end

local function jump()
  local s = state.suggestion
  if not s then return false end
  local e = s.edits[s.idx]
  s.jumped = true
  vim.schedule(function()
    if state.suggestion ~= s then return end
    local count = vim.api.nvim_buf_line_count(s.bufnr)
    local row = math.min(e.start_line, count - 1)
    state.expected_cursor = { row, e.start_col }
    pcall(vim.api.nvim_win_set_cursor, 0, { row + 1, e.start_col })
    s.display = render(s.bufnr, s)
    emit(s.bufnr, "prediction_jumped", {
      prediction_id = s.prediction_id,
      request_id = s.request_id,
      to = { row = row, col = e.start_col },
      protocol = "sweep-next-edit-v1",
    })
  end)
  return true
end

--- Tab handler: jump to a distant edit, otherwise accept. Returns true when
--- a suggestion consumed the key.
function M.tab()
  local s = state.suggestion
  if not s or s.bufnr ~= vim.api.nvim_get_current_buf() then return false end
  if s.far and not s.jumped then return jump() end
  return accept()
end

M.accept = accept

function M.dismiss()
  if state.suggestion then
    dismiss("explicit")
    return true
  end
  return false
end

-- Requests ------------------------------------------------------------------

local function diagnostics(bufnr)
  local out = {}
  for _, d in ipairs(vim.diagnostic.get(bufnr)) do
    if #out >= 20 then break end
    out[#out + 1] = {
      line = d.lnum,
      col = d.col,
      severity = ({ "error", "warning", "info", "hint" })[d.severity] or "info",
      message = (d.message or ""):sub(1, 300),
    }
  end
  return out
end

local function request(bufnr, trigger)
  if not eligible(bufnr) or vim.api.nvim_get_current_buf() ~= bufnr then return end
  local content = buffer_content(bufnr)
  if #content > M.config.max_file_bytes then return end
  state.seq = state.seq + 1
  local seq = state.seq
  if state.job then pcall(state.job.kill, state.job, 15) end
  -- A provisional suggestion belongs to the request just cancelled.
  if state.suggestion and state.suggestion.provisional then
    clear_marks(state.suggestion.bufnr)
    state.suggestion = nil
  end
  local row, col = cursor_pos()
  local tick = vim.api.nvim_buf_get_changedtick(bufnr)
  local prediction_id = uuid()
  local request_id = uuid()
  local path = vim.api.nvim_buf_get_name(bufnr)
  local clipboard = nil
  if state.yank and now_ms() - state.yank.at < M.config.clipboard_window_ms then clipboard = state.yank.text end
  state.stats.requested = state.stats.requested + 1
  emit(bufnr, "prediction_requested", {
    prediction_id = prediction_id,
    request_id = request_id,
    trigger = trigger,
    cursor = { row = row, col = col },
    changedtick = tick,
    file = path,
    model = state.health and state.health.model or nil,
    model_sha256 = state.health and state.health.model_sha256 or nil,
    protocol = "sweep-next-edit-v1",
    requested_at_ms = now_ms(),
  })
  local started = now_ms()
  local root = root_for(bufnr)
  local diags = diagnostics(bufnr)
  local body = {
    request_id = request_id,
    path = path,
    root = root,
    content = content,
    cursor = { line = row, col = col },
    diagnostics = diags,
    clipboard = clipboard,
    -- The jump search costs a second model call; skip it while typing.
    jump = M.config.jump and trigger ~= "insert",
  }
  -- The provisional suggestion drawn for this request, if any.
  local early = nil
  local function current()
    return seq == state.seq and vim.api.nvim_buf_is_valid(bufnr) and vim.api.nvim_get_current_buf() == bufnr
      and vim.api.nvim_buf_get_changedtick(bufnr) == tick
  end
  local function on_provisional(ev)
    local e = ev.edits and ev.edits[1]
    if not e or not current() then return end
    local r2, c2 = cursor_pos()
    if r2 ~= row or c2 ~= col or e.start_line ~= row or e.start_col ~= col then return end
    if early then
      -- Still ours and untouched: grow it in place.
      if state.suggestion ~= early or not early.provisional then return end
      early.edits = { e }
      draw(bufnr, early)
      return
    end
    dismiss("superseded")
    early = {
      bufnr = bufnr,
      changedtick = tick,
      edits = { e },
      idx = 1,
      kind = "local",
      prediction_id = prediction_id,
      request_id = request_id,
      typed_through = 0,
      content = content,
      path = path,
      root = root,
      diagnostics = diags,
      clipboard = clipboard,
      provisional = true,
      partial = true,
    }
    draw(bufnr, early)
  end
  local function finish(resp, err)
    if seq == state.seq then state.job = nil end
    -- The provisional text is only a preview of this answer: drop it unless
    -- the answer extends it.
    local drawn_at = nil
    if early and state.suggestion == early and early.provisional then
      drawn_at = early.drawn_at
      state.suggestion = nil
      clear_marks(bufnr)
    end
    local latency = now_ms() - started
    if err or not resp then
      if seq == state.seq then
        state.stats.errors = state.stats.errors + 1
        emit(bufnr, "prediction_generated", {
          prediction_id = prediction_id, request_id = request_id, status = "error",
          error = tostring(err):sub(1, 200), latency_ms = latency, protocol = "sweep-next-edit-v1",
        })
      end
      return
    end
    if seq == state.seq then
      state.last_response = { prefetched = resp.prefetched or false, latency_ms = latency, status = resp.status, trigger = trigger }
    end
    local prompt_hash = store_blob(resp.prompt)
    local completion_hash = store_blob(resp.completion)
    emit(bufnr, "prediction_generated", {
      prediction_id = prediction_id,
      request_id = request_id,
      status = resp.status,
      exit_reason = resp.exit_reason,
      kind = resp.kind,
      jump_reason = resp.jump_reason,
      edits = #(resp.edits or {}),
      withheld = #(resp.withheld or {}),
      withheld_edit = M.config.capture_content and (resp.withheld or {})[1] or nil,
      confidence = resp.confidence,
      finish_reason = resp.finish_reason,
      latency_ms = latency,
      superseded = seq ~= state.seq or nil,
      engine_timings = resp.timings,
      context = resp.context,
      context_blob_hash = prompt_hash,
      raw_response_hash = completion_hash,
      prefetched = resp.prefetched or nil,
      prompt_sha256 = resp.prompt_sha256,
      model = resp.model,
      model_sha256 = resp.model_sha256,
      protocol = resp.protocol,
    })
    if seq ~= state.seq or resp.status ~= "ok" or not resp.edits or #resp.edits == 0 then return end
    -- The user already accepted, typed through or dismissed the preview.
    if early and not early.provisional then return end
    if not current() then return end
    local r2, c2 = cursor_pos()
    if r2 ~= row or c2 ~= col then return end
    dismiss("superseded")
    local final = resp.edits[1]
    local p = early and early.edits[1]
    local extends = p and final.start_byte == p.start_byte and final.end_byte == p.end_byte
      and final.text:sub(1, #p.text) == p.text
    show(bufnr, {
      bufnr = bufnr,
      changedtick = tick,
      edits = resp.edits,
      idx = 1,
      kind = resp.kind,
      jump_reason = resp.jump_reason,
      prediction_id = prediction_id,
      request_id = request_id,
      typed_through = 0,
      content = content,
      path = path,
      root = root,
      diagnostics = diags,
      clipboard = clipboard,
      drawn_at = extends and drawn_at or nil,
    })
  end
  if M.config.stream then
    local finished = false
    state.job = post_stream("/v1/predict", body, function(ev)
      if finished then return end
      if ev.event == "provisional" then
        on_provisional(ev)
      elseif ev.event == "final" or (ev.event == nil and ev.status) then
        finished = true
        finish(ev, nil)
      elseif ev.event == "error" then
        finished = true
        finish(nil, ev.error or "engine error")
      end
    end, function(err)
      if finished then return end
      finished = true
      finish(nil, err or "stream ended without a final event")
    end)
  else
    state.job = post("/v1/predict", body, finish)
  end
end

schedule_request = function(bufnr, trigger, delay)
  if state.timer then
    state.timer:stop()
  else
    state.timer = vim.uv.new_timer()
  end
  state.timer:start(delay, 0, vim.schedule_wrap(function()
    request(bufnr, trigger)
  end))
end

-- If the user types exactly the next characters of a ghost suggestion, keep
-- it (shortened) instead of discarding it.
-- The line a single-line suggestion would produce, or nil for other edits.
target_line = function(bufnr, e)
  local line = vim.api.nvim_buf_get_lines(bufnr, e.start_line, e.start_line + 1, false)[1]
  if not line then return nil end
  if e.end_line == e.start_line and not e.text:find("\n") then
    return line:sub(1, e.start_col) .. e.text .. line:sub(e.end_col + 1)
  end
  -- Whole-line replacement: [row, 0] .. [row + 1, 0] with one new line.
  if e.start_col == 0 and e.end_line == e.start_line + 1 and e.end_col == 0 then
    local body = e.text:match("^([^\n]*)\n$")
    return body
  end
  return nil
end

-- Keep a suggestion alive while the user types toward it: if the current
-- line still agrees with the suggested line before and after the cursor,
-- the rest of the difference becomes ghost text at the cursor (no engine
-- round trip).
local function try_type_through(bufnr)
  local s = state.suggestion
  if not s or s.bufnr ~= bufnr or not s.target then return false end
  local row, col = cursor_pos()
  if row ~= s.target_row then return false end
  local line = vim.api.nvim_buf_get_lines(bufnr, row, row + 1, false)[1] or ""
  local target = s.target
  local before, rest = line:sub(1, col), line:sub(col + 1)
  if target:sub(1, #before) ~= before or #target < #line then return false end
  if rest ~= "" and target:sub(-#rest) ~= rest then return false end
  local ghost = target:sub(#before + 1, #target - #rest)
  if s.provisional then announce(bufnr, s) end
  s.typed_through = (s.typed_through or 0) + 1
  if ghost == "" then
    -- The user typed the whole suggestion.
    state.suggestion = nil
    clear_marks(bufnr)
    emit(bufnr, "prediction_partially_accepted", {
      prediction_id = s.prediction_id,
      request_id = s.request_id,
      reason = "typed_through",
      keystrokes = s.typed_through,
      protocol = "sweep-next-edit-v1",
    })
    return true
  end
  s.edits[s.idx] = { start_line = row, start_col = col, end_line = row, end_col = col, text = ghost }
  s.replaced = ""
  s.changedtick = vim.api.nvim_buf_get_changedtick(bufnr)
  s.content = nil -- offsets no longer match the request snapshot
  s.display = render(bufnr, s)
  return true
end

local function observe(bufnr, opts)
  if not eligible(bufnr) then return end
  local content = buffer_content(bufnr)
  if #content > M.config.max_file_bytes then return end
  local row, col = cursor_pos()
  post("/v1/observe", {
    path = vim.api.nvim_buf_get_name(bufnr),
    root = root_for(bufnr),
    content = content,
    cursor = { line = row, col = col },
    saved = opts.saved or false,
    warm = opts.warm or false,
  }, nil)
end

-- Setup ---------------------------------------------------------------------

local function define_highlights()
  local function link(name, target, extra)
    local spec = vim.tbl_extend("force", { link = target, default = true }, extra or {})
    vim.api.nvim_set_hl(0, name, spec)
  end
  link("TabCompleteNesGhost", "Comment")
  link("TabCompleteNesAdd", "DiffAdd")
  link("TabCompleteNesJump", "Special")
  local del = vim.api.nvim_get_hl(0, { name = "DiffDelete", link = false })
  vim.api.nvim_set_hl(0, "TabCompleteNesDelete", { default = true, strikethrough = true, fg = del.fg, bg = del.bg })
end

local function hook_lazyvim()
  local lv = rawget(_G, "LazyVim")
  if type(lv) == "table" and lv.cmp and lv.cmp.actions then
    lv.cmp.actions.ai_nes = function()
      return M.tab()
    end
    return true
  end
  return false
end

local function map_keys()
  local cfg = M.config
  if cfg.tab then
    vim.keymap.set("n", "<Tab>", function()
      if M.tab() then return "" end
      return "<Tab>"
    end, { expr = true, desc = "Next edit: jump/accept" })
    local function blink_present()
      return package.loaded["blink.cmp"] ~= nil or hook_lazyvim()
    end
    vim.api.nvim_create_autocmd("User", {
      pattern = "VeryLazy",
      once = true,
      callback = function()
        if not blink_present() then
          vim.keymap.set("i", "<Tab>", function()
            if M.tab() then return "" end
            return "<Tab>"
          end, { expr = true, desc = "Next edit: jump/accept" })
        end
      end,
    })
    hook_lazyvim()
  end
  if cfg.accept_key and cfg.accept_key ~= "" then
    vim.keymap.set({ "i", "n" }, cfg.accept_key, function()
      if not M.tab() then return end
    end, { desc = "Next edit: accept" })
  end
  if cfg.dismiss_key and cfg.dismiss_key ~= "" then
    vim.keymap.set({ "i", "n" }, cfg.dismiss_key, function()
      M.dismiss()
    end, { desc = "Next edit: dismiss" })
  end
  if cfg.predict_key and cfg.predict_key ~= "" then
    vim.keymap.set({ "i", "n" }, cfg.predict_key, function()
      request(vim.api.nvim_get_current_buf(), "manual")
    end, { desc = "Next edit: predict" })
  end
end

local function autocmds()
  local group = vim.api.nvim_create_augroup("tabcomplete_nes", { clear = true })
  vim.api.nvim_create_autocmd({ "TextChangedI", "TextChangedP" }, {
    group = group,
    callback = function(args)
      if state.applying then return end
      if try_type_through(args.buf) then return end
      dismiss("typed_over")
      schedule_request(args.buf, "insert", M.config.insert_debounce_ms)
    end,
  })
  vim.api.nvim_create_autocmd("TextChanged", {
    group = group,
    callback = function(args)
      if state.applying then return end
      dismiss("buffer_changed")
      if M.config.normal_mode then schedule_request(args.buf, "normal", M.config.normal_debounce_ms) end
    end,
  })
  vim.api.nvim_create_autocmd("InsertLeave", {
    group = group,
    callback = function(args)
      dismiss("insert_leave")
      if M.config.normal_mode then schedule_request(args.buf, "insert_leave", M.config.normal_debounce_ms) end
    end,
  })
  vim.api.nvim_create_autocmd({ "CursorMoved", "CursorMovedI" }, {
    group = group,
    callback = function()
      local s = state.suggestion
      if not s then return end
      local row, col = cursor_pos()
      local exp = state.expected_cursor
      if exp and exp[1] == row and exp[2] == col then
        state.expected_cursor = nil
        return
      end
      local e = s.edits[s.idx]
      -- Insert-mode typing moves the cursor; TextChangedI handles that case.
      if s.display == "ghost" and row == e.start_line and col == e.start_col then return end
      if vim.api.nvim_buf_get_changedtick(s.bufnr) ~= s.changedtick then return end
      dismiss("cursor_moved")
    end,
  })
  vim.api.nvim_create_autocmd({ "BufLeave", "WinLeave" }, {
    group = group,
    callback = function()
      dismiss("buffer_left")
    end,
  })
  vim.api.nvim_create_autocmd("BufEnter", {
    group = group,
    callback = function(args)
      observe(args.buf, { warm = true })
    end,
  })
  vim.api.nvim_create_autocmd("BufWritePost", {
    group = group,
    callback = function(args)
      observe(args.buf, { saved = true })
    end,
  })
  vim.api.nvim_create_autocmd("TextYankPost", {
    group = group,
    callback = function()
      local ev = vim.v.event
      if ev.operator == "y" and ev.regcontents and #ev.regcontents > 0 then
        state.yank = { text = table.concat(ev.regcontents, "\n"):sub(1, 4000), at = now_ms() }
      end
    end,
  })
  vim.api.nvim_create_autocmd("ColorScheme", { group = group, callback = define_highlights })
end

function M.status()
  return {
    enabled = M.config.enabled,
    url = M.config.url,
    engine = state.health,
    suggestion = state.suggestion and { kind = state.suggestion.kind, display = state.suggestion.display } or nil,
    collector = collector() ~= nil,
    stats = vim.deepcopy(state.stats),
    last_response = state.last_response,
  }
end

function M.setup(opts)
  M.config = vim.tbl_deep_extend("force", M.config, opts or {})
  define_highlights()
  map_keys()
  autocmds()
  refresh_health()
  vim.api.nvim_create_user_command("TabCompleteNes", function(args)
    local sub = args.args
    if sub == "enable" then
      M.config.enabled = true
    elseif sub == "disable" then
      M.config.enabled = false
      dismiss("disabled")
    elseif sub == "toggle" then
      M.config.enabled = not M.config.enabled
      if not M.config.enabled then dismiss("disabled") end
    else
      refresh_health()
      vim.notify(vim.inspect(M.status()), vim.log.levels.INFO)
      return
    end
    vim.notify("TabCompleteNes " .. (M.config.enabled and "enabled" or "disabled"))
  end, { nargs = "?", complete = function() return { "status", "enable", "disable", "toggle" } end })
  return M
end

M._test = {
  state = state,
  request = request,
  render = render,
  apply_edit = apply_edit,
  show_for_test = function(bufnr, s) show(bufnr, s) end,
  post_stream = post_stream,
  try_type_through = function(bufnr) return try_type_through(bufnr) end,
}

return M
