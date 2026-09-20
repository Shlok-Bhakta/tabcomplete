-- tabcomplete_trajectory (init)
-- LazyVim-compatible collector: session lifecycle, event queue + batching,
-- anchors (buffer content -> sha256 -> blobs/check|upload), repo snapshots,
-- cursor/nav/key capture, offline spool, prediction logging, user commands.
local M = {}

local config = require("tabcomplete_trajectory.config")
local events = require("tabcomplete_trajectory.events")
local util = require("tabcomplete_trajectory.util")
local spool = require("tabcomplete_trajectory.spool")
local transport = require("tabcomplete_trajectory.transport")
local buffers = require("tabcomplete_trajectory.buffers")
local repository = require("tabcomplete_trajectory.repository")

M.session_id = nil
M.machine_id = nil
M.seq = 0
M.queue = {}
M.paused = false
M.disabled = false
M.started = false
M.timers = {}
M.last_ok_ms = nil
M.last_err = nil
M.current_file_id = nil
M.pending_cursor = nil
M.cursor_timer = nil
M.key_ns = nil
M.augroup = nil

-- Forward declarations for callbacks registered before definition.
local flush_now, retry_spool, post_to, anchor_buffer

--- Absolute path of a buffer ("" for scratch/unnamed).
local function buf_abspath(bufnr)
  local name = vim.api.nvim_buf_get_name(bufnr)
  if not name or name == "" then
    return ""
  end
  return util.abspath(name)
end

--- Envelope-level file_id for a buffer.
local function buf_file_id(bufnr)
  local abs = buf_abspath(bufnr)
  if abs == "" then
    return ""
  end
  local root = repository.cache.root
  return util.file_id(abs, root, repository.cache.root_name)
end

--- POST a JSON body to server_url .. path. Never throws, never blocks.
--- cb(ok:boolean, info). Warns once (never crashes) when URL is missing.
function post_to(path, body, cb)
  if not config.has_url() then
    config.warn_once("missing-url", "server_url is not set and $TABCOMPLETE_COLLECTOR_URL is empty; events are spooled locally")
    if cb then
      vim.schedule(function()
        cb(false, "missing server_url")
      end)
    end
    return
  end
  local url = config.options.server_url:gsub("/+$", "") .. path
  transport.post_json_async(url, body, config.options.request_timeout_ms or 5000, function(ok, info)
    if ok then
      M.last_ok_ms = util.now_ms()
    else
      local detail = type(info) == "table" and vim.inspect(info) or tostring(info)
      M.last_err = ("POST %s failed: %s"):format(path, detail)
    end
    if cb then
      cb(ok, info)
    end
  end)
end

--- Enqueue one event. Returns the envelope or nil when paused/disabled.
--- Edit deltas are attributed with the buffer's absolute path (replay groups
--- by payload.path; envelope file_id stays a client-local display id).
local function emit_from_buffers(bufnr, event_type, payload)
  if event_type == "edit_delta" and type(payload) == "table" and payload.path == nil then
    local abs = buf_abspath(bufnr)
    if abs ~= "" then
      payload.path = abs
    end
  end
  return M.emit(bufnr, event_type, payload)
end
function M.emit(bufnr, event_type, payload)
  if M.paused or M.disabled or not M.started then
    return nil
  end
  local ok_buf = true
  local cursor = { row = 0, col = 0 }
  local mode = util.current_mode()
  if bufnr and vim.api.nvim_buf_is_valid(bufnr) then
    local ok_c, c = pcall(util.cursor_zero)
    if ok_c then
      cursor = c
    end
  else
    bufnr = vim.api.nvim_get_current_buf()
    ok_buf = false
  end
  if not ok_buf and bufnr and vim.api.nvim_buf_is_valid(bufnr) then
    local ok_c, c = pcall(util.cursor_zero)
    if ok_c then
      cursor = c
    end
  end
  M.seq = M.seq + 1
  local env = events.make({
    event_type = event_type,
    session_id = M.session_id,
    sequence_number = M.seq,
    file_id = (bufnr and vim.api.nvim_buf_is_valid(bufnr)) and buf_file_id(bufnr) or "",
    cursor = cursor,
    mode = mode,
    payload = payload or {},
  })
  M.queue[#M.queue + 1] = env
  if #M.queue >= (config.options.batch_max_events or 100) then
    flush_now()
  end
  return env
end

--- Flush the in-memory queue as one POST /v1/events/batch.
--- On failure the batch is spooled to disk (deleted only on ACK).
function flush_now(cb)
  if M.paused or M.disabled then
    if cb then
      cb(false, "paused/disabled")
    end
    return
  end
  if #M.queue == 0 then
    if cb then
      cb(true, "empty")
    end
    return
  end
  local batch = M.queue
  M.queue = {}
  post_to("/v1/events/batch", {
    protocol_version = events.PROTOCOL_VERSION,
    session_id = M.session_id,
    machine_id = M.machine_id,
    events = batch,
  }, function(ok, info)
    if ok then
      retry_spool()
    else
      spool.write_batch(batch, { kind = "events", path = "/v1/events/batch" })
      spool.enforce_cap(config.options.spool_max_bytes or 268435456)
    end
    if cb then
      cb(ok, info)
    end
  end)
end

M.flush_now = flush_now

--- Retry spooled batches oldest-first; stop at first failure to keep order.
--- A spool file is deleted only after its POST is ACKed (HTTP 2xx).
function retry_spool(cb)
  if not config.has_url() then
    if cb then
      cb(false, "missing server_url")
    end
    return
  end
  local files = spool.list()
  if #files == 0 then
    if cb then
      cb(true, "empty")
    end
    return
  end
  local idx = 1
  local function next()
    if idx > #files then
      if cb then
        cb(true, "done")
      end
      return
    end
    local path = files[idx]
    local decoded, err = spool.read(path)
    if not decoded then
      -- Corrupt spool file: drop it (cannot be delivered) and continue.
      spool.remove(path)
      idx = idx + 1
      next()
      return
    end
    local post_path = (decoded.meta and decoded.meta.path) or "/v1/events/batch"
    local body = {
      protocol_version = events.PROTOCOL_VERSION,
      session_id = decoded.meta and decoded.meta.session_id or M.session_id,
      machine_id = M.machine_id,
      events = decoded.events or {},
    }
    if decoded.meta and decoded.meta.kind == "session" then
      body = decoded.body or body
    end
    post_to(post_path, body, function(ok)
      if ok then
        spool.remove(path)
        idx = idx + 1
        next()
      else
        if cb then
          cb(false, "post failed for " .. path)
        end
      end
    end)
  end
  next()
end

M.retry_spool = retry_spool

--- Upload content bytes via blobs/check|upload if the hash is unknown.
--- Server contract: check {hashes:[...]}, upload {sha256, content_base64}.
--- cb(ok, sha_hex).
local function ensure_blob(content, cb)
  local sha = util.sha256hex(content)
  post_to("/v1/blobs/check", { hashes = { sha } }, function(ok, info)
    -- Server answers which hashes it already has; the mock/test path and
    -- minimal servers may not implement `missing`, in which case we upload
    -- when the check itself failed, and skip only on explicit confirmation.
    if ok and type(info) == "table" and info.present then
      if cb then
        cb(true, sha)
      end
      return
    end
    local b64 = nil
    if vim.base64 and vim.base64.encode then
      local ok_b, out = pcall(vim.base64.encode, content)
      if ok_b and type(out) == "string" and out ~= "" then
        b64 = out
      end
    end
    if not b64 then
      if cb then
        cb(false, sha)
      end
      return
    end
    post_to("/v1/blobs/upload", { sha256 = sha, content_base64 = b64 }, function(ok2)
      if cb then
        cb(ok2, sha)
      end
    end)
  end)
end

--- Anchor a buffer: full canonical content -> sha256 -> blobs/check|upload,
--- then emit buffer_open/buffer_write carrying content_hash. Resets the
--- shadow to current content. Representation: anchor, delta*, anchor.
--- The upload is async: if the buffer was reused for another file (e.g. an
--- unnamed setup buffer later loaded with a real file) or edited while the
--- upload was in flight, the anchor is DROPPED — emitting it would bind a
--- stale hash to live content and corrupt replay from that point.
function anchor_buffer(bufnr, reason)
  if M.paused or M.disabled or not M.started then
    return
  end
  if not vim.api.nvim_buf_is_valid(bufnr) then
    return
  end
  if util.is_excluded_buffer(bufnr) then
    return
  end
  local content, meta = buffers.canonical_bytes(bufnr)
  local path_at_trigger = buf_abspath(bufnr)
  ensure_blob(content, function(ok, sha)
    if not vim.api.nvim_buf_is_valid(bufnr) then
      return
    end
    if util.is_excluded_buffer(bufnr) then
      return
    end
    buffers.refresh_shadow(bufnr)
    local now_content = buffers.canonical_bytes(bufnr)
    if buf_abspath(bufnr) ~= path_at_trigger or now_content ~= content then
      return
    end
    local abs = path_at_trigger
    local etype = (reason == "write" or reason == "periodic_anchor") and "buffer_write" or "buffer_open"
    if reason == "periodic_anchor" then
      etype = "buffer_write"
    end
    M.emit(bufnr, etype, {
      path = abs,
      content_hash = sha,
      content_bytes = #content,
      blob_uploaded = ok,
      line_count = meta.line_count,
      eol = meta.eol,
      fileformat = meta.fileformat,
      reason = reason,
    })
  end)
end

--- Snapshot repo context and POST /v1/repository/snapshot (best effort).
--- Server contract: {protocol_version, repo_id, root_name, origin_url,
--- files:[{relative_path, extension, sha256, source}]}. Entries without a
--- content hash (truncated/binary/secret-skipped) are manifest-excluded here;
--- their absence from the manifest IS the signal (never upload such bytes).
local function snapshot_repo(reason)
  if not config.options.capture_repo_context then
    return
  end
  local root = repository.cache.root
  if not root then
    return
  end
  local ok, snap = pcall(repository.build_snapshot, root, config.options.max_repo_file_bytes)
  if not ok then
    M.last_err = "repo snapshot failed: " .. tostring(snap)
    return
  end
  local repo_id = "repo:" .. (snap.root_name or "unknown")
  local files = {}
  for _, e in ipairs(snap.files or {}) do
    if e.content_hash and e.path and e.path ~= "" and e.path:sub(1, 1) ~= "/" then
      local ext = e.path:match("(%.[^./]+)$")
      files[#files + 1] = {
        relative_path = e.path,
        extension = ext,
        sha256 = e.content_hash,
        source = "repo_scan",
      }
    end
  end
  post_to("/v1/repository/snapshot", {
    protocol_version = events.PROTOCOL_VERSION,
    repo_id = repo_id,
    root_name = snap.root_name,
    origin_url = snap.origin,
    files = files,
  }, function(ok)
    if ok then
      M.emit(0, "repo_snapshot", {
        root_name = snap.root_name,
        branch = snap.branch,
        head = snap.head,
        dirty = snap.dirty,
        file_count = #files,
      })
    end
  end)
end

--- Flush any debounced-but-unsent cursor position immediately.
local function flush_pending_cursor()
  if M.pending_cursor then
    local p = M.pending_cursor
    M.pending_cursor = nil
    if M.cursor_timer then
      pcall(function()
        M.cursor_timer:stop()
      end)
    end
    if p.file_id ~= M.current_file_id then
      local from = M.current_file_id or ""
      M.current_file_id = p.file_id
      M.emit(p.bufnr, "file_jump", { from_file = from, to_file = p.file_id, position = { row = p.row, col = p.col } })
    else
      M.emit(p.bufnr, "cursor_move", { position = { row = p.row, col = p.col } })
    end
  end
end

local function on_cursor_moved()
  if M.paused or M.disabled or not M.started then
    return
  end
  local bufnr = vim.api.nvim_get_current_buf()
  if util.is_excluded_buffer(bufnr) then
    return
  end
  local c = util.cursor_zero()
  M.pending_cursor = { bufnr = bufnr, row = c.row, col = c.col, file_id = buf_file_id(bufnr) }
  if not M.cursor_timer then
    M.cursor_timer = vim.uv.new_timer()
  end
  local ms = config.options.cursor_debounce_ms or 50
  pcall(function()
    M.cursor_timer:stop()
    M.cursor_timer:start(ms, 0, function()
      vim.schedule(flush_pending_cursor)
    end)
  end)
end

local function on_buf_enter()
  if M.paused or M.disabled or not M.started then
    return
  end
  local bufnr = vim.api.nvim_get_current_buf()
  if util.is_excluded_buffer(bufnr) then
    return
  end
  flush_pending_cursor()
  if not buffers.is_tracked(bufnr) then
    if buffers.attach(bufnr) then
      M.current_file_id = buf_file_id(bufnr)
      anchor_buffer(bufnr, "buffer_open")
    end
  else
    local fid = buf_file_id(bufnr)
    if fid ~= M.current_file_id then
      local from = M.current_file_id or ""
      M.current_file_id = fid
      M.emit(bufnr, "file_jump", { from_file = from, to_file = fid })
    else
      local c = util.cursor_zero()
      M.emit(bufnr, "cursor_move", { position = { row = c.row, col = c.col }, reason = "buffer_enter" })
    end
    M.emit(bufnr, "buffer_enter", {})
  end
end

local function setup_autocmds()
  if M.augroup then
    pcall(vim.api.nvim_del_augroup_by_id, M.augroup)
    M.augroup = nil
  end
  local group = vim.api.nvim_create_augroup("TabCompleteTrajectory", { clear = true })
  M.augroup = group

  vim.api.nvim_create_autocmd({ "BufReadPost", "BufNewFile" }, {
    group = group,
    callback = function(args)
      if M.paused or M.disabled or not M.started then
        return
      end
      if buffers.attach(args.buf) then
        M.current_file_id = buf_file_id(args.buf)
        anchor_buffer(args.buf, "buffer_open")
      end
    end,
  })
  vim.api.nvim_create_autocmd("BufEnter", { group = group, callback = on_buf_enter })
  vim.api.nvim_create_autocmd("BufLeave", {
    group = group,
    callback = function(args)
      if M.paused or M.disabled or not M.started then
        return
      end
      flush_pending_cursor()
      if not util.is_excluded_buffer(args.buf) then
        M.emit(args.buf, "buffer_leave", {})
      end
    end,
  })
  vim.api.nvim_create_autocmd({ "BufWipeout", "BufDelete" }, {
    group = group,
    callback = function(args)
      if not util.is_excluded_buffer(args.buf) then
        M.emit(args.buf, "buffer_close", {})
      end
      buffers.detach(args.buf)
    end,
  })
  vim.api.nvim_create_autocmd("BufWritePost", {
    group = group,
    callback = function(args)
      if M.paused or M.disabled or not M.started then
        return
      end
      if config.options.capture_repo_context then
        repository.refresh(buf_abspath(args.buf))
      end
      anchor_buffer(args.buf, "write")
    end,
  })
  vim.api.nvim_create_autocmd({ "CursorMoved", "CursorMovedI" }, {
    group = group,
    callback = on_cursor_moved,
  })
  vim.api.nvim_create_autocmd("ModeChanged", {
    group = group,
    callback = function(args)
      if M.paused or M.disabled or not M.started then
        return
      end
      flush_pending_cursor()
      -- args.match is "from:to", e.g. "n:i".
      local from, to = args.match:match("^([^:]+):([^:]+)$")
      M.emit(vim.api.nvim_get_current_buf(), "mode_change", { from = from or "", to = to or "" })
    end,
  })
  vim.api.nvim_create_autocmd("VimLeavePre", {
    group = group,
    callback = function()
      M.shutdown()
    end,
  })
end

local function setup_key_capture()
  if M.key_ns then
    pcall(vim.on_key, nil, M.key_ns)
    M.key_ns = nil
  end
  if not config.options.capture_keys then
    return
  end
  M.key_ns = vim.on_key(function(key, typed, buf)
    if M.paused or M.disabled or not M.started then
      return
    end
    local bufnr
    if type(buf) == "number" and buf ~= 0 and vim.api.nvim_buf_is_valid(buf) then
      bufnr = buf
    else
      bufnr = vim.api.nvim_get_current_buf()
    end
    if util.is_excluded_buffer(bufnr) then
      return
    end
    local ok_kt, normalized = pcall(vim.fn.keytrans, key)
    if not ok_kt then
      normalized = key
    end
    local c = util.cursor_zero()
    M.seq = M.seq + 1
    local env = events.make({
      event_type = "key",
      session_id = M.session_id,
      sequence_number = M.seq,
      file_id = buf_file_id(bufnr),
      cursor = c,
      mode = util.current_mode(),
      payload = { key = normalized, typed = typed == 1 or typed == true },
    })
    M.queue[#M.queue + 1] = env
    if #M.queue >= (config.options.batch_max_events or 100) then
      flush_now()
    end
  end, nil)
end

local function start_timers()
  for _, t in ipairs(M.timers) do
    pcall(function()
      t:stop()
      t:close()
    end)
  end
  M.timers = {}
  local batch = vim.uv.new_timer()
  batch:start(config.options.batch_interval_ms or 1500, config.options.batch_interval_ms or 1500, function()
    vim.schedule(function()
      if not M.paused and not M.disabled then
        flush_now()
      end
    end)
  end)
  M.timers[#M.timers + 1] = batch
  local spool_t = vim.uv.new_timer()
  local retry_ms = config.options.spool_retry_interval_ms or 30000
  spool_t:start(retry_ms, retry_ms, function()
    vim.schedule(function()
      if not M.paused and not M.disabled and #M.queue == 0 then
        retry_spool()
      end
    end)
  end)
  M.timers[#M.timers + 1] = spool_t
  local hb = vim.uv.new_timer()
  hb:start(60000, 60000, function()
    vim.schedule(function()
      if not M.paused and not M.disabled and M.started then
        M.emit(vim.api.nvim_get_current_buf(), "heartbeat", { queue_depth = #M.queue })
      end
    end)
  end)
  M.timers[#M.timers + 1] = hb
end

--- Public setup. Never throws, never crashes on missing URL.
function M.setup(opts)
  config.setup(opts)
  M.paused = false
  M.disabled = false
  M.started = false
  M.seq = 0
  M.queue = {}
  M.current_file_id = nil
  M.pending_cursor = nil

  buffers.set_callbacks(emit_from_buffers, anchor_buffer)

  M.define_commands()
  if not config.options.enabled then
    M.disabled = true
    return M
  end
  if not config.has_url() then
    config.warn_once("missing-url", "server_url is not set and $TABCOMPLETE_COLLECTOR_URL is empty; events are spooled locally")
  end

  spool.ensure_dirs()
  M.machine_id = spool.machine_id()
  M.session_id = util.uuid()
  M.started = true

  setup_autocmds()
  setup_key_capture()
  start_timers()

  repository.refresh(vim.api.nvim_buf_get_name(0))
  local started_at = util.now_ms()
  local start_body = {
    session_id = M.session_id,
    machine_id = M.machine_id,
    started_at = os.date("!%Y-%m-%dT%H:%M:%SZ", math.floor(started_at / 1000)),
    started_at_ms = started_at,
    protocol_version = events.PROTOCOL_VERSION,
    nvim_version = vim.inspect(vim.version()),
  }
  post_to("/v1/session/start", start_body, function(ok)
    if not ok then
      spool.write_batch({}, { kind = "session", path = "/v1/session/start", body = start_body })
    end
  end)
  -- Session start is also an event in the batch stream (idempotent via event_id).
  M.seq = M.seq + 1
  M.queue[#M.queue + 1] = events.make({
    event_type = "session_start",
    session_id = M.session_id,
    sequence_number = M.seq,
    file_id = "",
    cursor = { row = 0, col = 0 },
    mode = "n",
    payload = { machine_id = M.machine_id, started_at_ms = started_at },
  })

  snapshot_repo("session_start")
  -- Anchor every already-open listed buffer.
  for _, bufnr in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_is_loaded(bufnr) and not util.is_excluded_buffer(bufnr) then
      if buffers.attach(bufnr) then
        anchor_buffer(bufnr, "session_start")
      end
    end
  end
  retry_spool()
  return M
end

--- Best-effort session end. Blocking ≤2s per POST only on the exit path; on
--- failure the session_end envelope is spooled synchronously (tolerate
--- missing ended_at server-side: we always send ended_at_ms). The final
--- session_end envelope is delivered in its own synchronous batch first so
--- the event stream always closes even when the process exits immediately.
local function sync_post(path, body_table)
  local ok_enc, encoded = pcall(vim.json.encode, body_table)
  if not ok_enc then
    return false
  end
  local url = config.options.server_url:gsub("/+$", "") .. path
  local ok_run, res = pcall(function()
    return vim.system({
      "curl",
      "-sS",
      "-m",
      "2",
      "-X",
      "POST",
      "-H",
      "Content-Type: application/json",
      "--data-binary",
      "@-",
      url,
    }, { stdin = encoded, text = true, timeout = 2500 }):wait()
  end)
  return ok_run and res ~= nil and res.code == 0
end

function M.shutdown()
  if not M.started then
    return
  end
  M.started = false
  flush_pending_cursor()
  pcall(flush_now)
  local ended_at = util.now_ms()
  M.seq = M.seq + 1
  local env = events.make({
    event_type = "session_end",
    session_id = M.session_id,
    sequence_number = M.seq,
    file_id = "",
    cursor = { row = 0, col = 0 },
    mode = "n",
    payload = { ended_at_ms = ended_at },
  })
  local end_body = {
    session_id = M.session_id,
    machine_id = M.machine_id,
    ended_at_ms = ended_at,
    protocol_version = events.PROTOCOL_VERSION,
  }
  if config.has_url() then
    local batch_ok = sync_post("/v1/events/batch", {
      protocol_version = events.PROTOCOL_VERSION,
      session_id = M.session_id,
      machine_id = M.machine_id,
      events = { env },
    })
    if not batch_ok then
      spool.ensure_dirs()
      spool.write_batch({ env }, { kind = "events", path = "/v1/events/batch" })
    end
    local end_ok = sync_post("/v1/session/end", end_body)
    if not end_ok then
      spool.ensure_dirs()
      spool.write_batch({ env }, { kind = "session", path = "/v1/session/end", body = end_body })
    end
  else
    spool.ensure_dirs()
    spool.write_batch({ env }, { kind = "session", path = "/v1/session/end", body = end_body })
  end
end

--- Re-detect repo, re-snapshot, re-anchor open buffers.
function M.rescan()
  repository.refresh(vim.api.nvim_buf_get_name(0))
  snapshot_repo("rescan")
  for _, bufnr in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_is_loaded(bufnr) and not util.is_excluded_buffer(bufnr) then
      buffers.attach(bufnr)
      anchor_buffer(bufnr, "rescan")
    end
  end
  return repository.cache
end

function M.pause()
  M.paused = true
end

function M.resume()
  M.paused = false
end

-- Prediction logging. Each accepts a table of documented fields, fills
-- file/cursor defaults from the current buffer, and enqueues the matching
-- event. No reward computation is performed here.
local PREDICTION_FIELDS = {
  "prediction_id",
  "provider",
  "model",
  "model_revision",
  "requested_at_ms",
  "responded_at_ms",
  "latency_ms",
  "context_ref",
  "context_hash",
  "file",
  "cursor",
  "proposed_start",
  "proposed_end",
  "proposed_text",
  "max_output_tokens",
  "temperature",
  "confidence",
  "logprob",
  "finish_reason",
  "accepted_chars",
  "accepted_lines",
  "total_chars",
}

local function prediction_payload(opts)
  local payload = {}
  opts = opts or {}
  for _, k in ipairs(PREDICTION_FIELDS) do
    if opts[k] ~= nil then
      payload[k] = opts[k]
    end
  end
  -- passthrough for any additional provider-specific fields
  for k, v in pairs(opts) do
    if payload[k] == nil then
      payload[k] = v
    end
  end
  if payload.file == nil then
    local abs = buf_abspath(vim.api.nvim_get_current_buf())
    if abs ~= "" then
      payload.file = abs
    end
  end
  if payload.cursor == nil then
    payload.cursor = util.cursor_zero()
  end
  return payload
end

function M.log_prediction_requested(opts)
  return M.emit(vim.api.nvim_get_current_buf(), "prediction_requested", prediction_payload(opts))
end

function M.log_prediction_shown(opts)
  return M.emit(vim.api.nvim_get_current_buf(), "prediction_shown", prediction_payload(opts))
end

function M.log_prediction_accepted(opts)
  return M.emit(vim.api.nvim_get_current_buf(), "prediction_accepted", prediction_payload(opts))
end

function M.log_prediction_partially_accepted(opts)
  return M.emit(vim.api.nvim_get_current_buf(), "prediction_partially_accepted", prediction_payload(opts))
end

function M.log_prediction_rejected(opts)
  return M.emit(vim.api.nvim_get_current_buf(), "prediction_rejected", prediction_payload(opts))
end

--- Status snapshot (table) for :TabCompleteCollectorStatus and tests.
function M.status()
  local spool_files = spool.list()
  local spool_bytes = 0
  for _, p in ipairs(spool_files) do
    local st = vim.uv.fs_stat(p)
    if st then
      spool_bytes = spool_bytes + (st.size or 0)
    end
  end
  local cur = vim.api.nvim_get_current_buf()
  local curfile = vim.api.nvim_buf_is_valid(cur) and vim.api.nvim_buf_get_name(cur) or ""
  return {
    active = (not M.paused) and (not M.disabled) and M.started,
    paused = M.paused,
    disabled = M.disabled,
    session_id = M.session_id,
    machine_id = M.machine_id,
    server_url = config.options.server_url or "",
    repo_root = repository.cache.root,
    repo_branch = repository.cache.branch,
    repo_head = repository.cache.head,
    repo_dirty = repository.cache.dirty,
    current_file = curfile,
    queue_depth = #M.queue,
    spool_files = #spool_files,
    spool_bytes = spool_bytes,
    last_ok_ms = M.last_ok_ms,
    last_err = M.last_err,
    sequence_number = M.seq,
  }
end

local function echo_status()
  local s = M.status()
  local lines = {
    ("active=%s paused=%s session=%s"):format(tostring(s.active), tostring(s.paused), tostring(s.session_id)),
    ("url=%s"):format(s.server_url ~= "" and s.server_url or "(unset)"),
    ("repo=%s branch=%s dirty=%s"):format(tostring(s.repo_root), tostring(s.repo_branch), tostring(s.repo_dirty)),
    ("file=%s"):format(s.current_file ~= "" and s.current_file or "(none)"),
    ("queue=%d spool_files=%d spool_bytes=%d"):format(s.queue_depth, s.spool_files, s.spool_bytes),
    ("last_ok=%s last_err=%s"):format(tostring(s.last_ok_ms), tostring(s.last_err)),
  }
  vim.notify("[tabcomplete-trajectory]\n" .. table.concat(lines, "\n"), vim.log.levels.INFO)
end

function M.define_commands()
  local function safe_create(name, fn, desc)
    pcall(vim.api.nvim_create_user_command, name, fn, { desc = desc })
  end
  safe_create("TabCompleteCollectorStatus", echo_status, "Show trajectory collector status")
  safe_create("TabCompleteCollectorFlush", function()
    flush_now(function(ok, info)
      vim.notify(
        "[tabcomplete-trajectory] flush " .. (ok and "ok" or ("failed: " .. tostring(info and (info.http_code or info) or "?"))),
        ok and vim.log.levels.INFO or vim.log.levels.WARN
      )
    end)
  end, "Flush queued trajectory events now")
  safe_create("TabCompleteCollectorRescan", function()
    local c = M.rescan()
    vim.notify("[tabcomplete-trajectory] rescan: " .. tostring(c and c.root), vim.log.levels.INFO)
  end, "Re-detect repo and re-anchor buffers")
  safe_create("TabCompleteCollectorPause", function()
    M.pause()
    vim.notify("[tabcomplete-trajectory] paused", vim.log.levels.INFO)
  end, "Pause trajectory collection")
  safe_create("TabCompleteCollectorResume", function()
    M.resume()
    vim.notify("[tabcomplete-trajectory] resumed", vim.log.levels.INFO)
  end, "Resume trajectory collection")
  -- Short aliases requested by the collection spec.
  safe_create("Flush", function()
    flush_now(function()
    end)
  end, "Flush queued trajectory events now")
  safe_create("RescanRepo", function()
    M.rescan()
  end, "Re-detect repo and re-anchor buffers")
  safe_create("Pause", function()
    M.pause()
  end, "Pause trajectory collection")
  safe_create("Resume", function()
    M.resume()
  end, "Resume trajectory collection")
end

--- Reset in-memory state. Test-only helper (not a user command).
function M._reset_for_tests()
  M.session_id = "test-session"
  M.machine_id = "test-machine"
  M.seq = 0
  M.queue = {}
  M.paused = false
  M.disabled = false
  M.started = true
  M.last_ok_ms = nil
  M.last_err = nil
  M.current_file_id = nil
  M.pending_cursor = nil
  buffers.shadows = {}
  buffers.attached = {}
  buffers.set_callbacks(emit_from_buffers, function()
  end)
end

-- Deterministic test hooks (not user commands).
M._test = {
  buf_file_id = function(bufnr)
    return buf_file_id(bufnr)
  end,
  set_pending_cursor = function(p)
    M.pending_cursor = p
  end,
  flush_cursor = function()
    flush_pending_cursor()
  end,
  anchor = function(bufnr, reason)
    anchor_buffer(bufnr, reason)
  end,
}

return M
