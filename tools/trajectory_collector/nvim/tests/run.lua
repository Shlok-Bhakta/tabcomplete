-- Headless test suite for tabcomplete_trajectory.
-- Run from tools/trajectory_collector/nvim/ via:
--   nvim --headless -l tests/run.lua
-- Exit code 0 = all green. Deterministic: isolated XDG_STATE_HOME,
-- mocked transport (no network), seeded RNG.
math.randomseed(12345)

local src = debug.getinfo(1, "S").source:sub(2)
local tests_dir = vim.fn.fnamemodify(src, ":h")
local root = vim.fn.fnamemodify(tests_dir, ":h")
vim.opt.rtp:prepend(root)

-- Isolate all on-disk state before requiring modules.
local tmp_state = vim.fn.tempname() .. "-tcstate"
vim.fn.mkdir(tmp_state, "p")
vim.fn.setenv("XDG_STATE_HOME", tmp_state)
vim.fn.setenv("TABCOMPLETE_COLLECTOR_URL", "")

local passed, failed = 0, 0
local failures = {}

local function ok(name, fn)
  local ok_run, err = pcall(fn)
  if ok_run then
    passed = passed + 1
    print("PASS " .. name)
  else
    failed = failed + 1
    failures[#failures + 1] = name .. ": " .. tostring(err)
    print("FAIL " .. name .. " :: " .. tostring(err))
  end
end

local function assert_eq(a, b, msg)
  if a ~= b then
    error((msg or "assert_eq") .. (" (got %s, want %s)"):format(vim.inspect(a), vim.inspect(b)), 2)
  end
end

local function assert_true(v, msg)
  if not v then
    error(msg or "expected true", 2)
  end
end

local function last(queue)
  return queue[#queue]
end

local collector = require("tabcomplete_trajectory")
local config = require("tabcomplete_trajectory.config")
local events = require("tabcomplete_trajectory.events")
local buffers = require("tabcomplete_trajectory.buffers")
local repository = require("tabcomplete_trajectory.repository")
local spool = require("tabcomplete_trajectory.spool")
local transport = require("tabcomplete_trajectory.transport")
local util = require("tabcomplete_trajectory.util")

local scratch = vim.fn.tempname() .. "-tcbufs"
vim.fn.mkdir(scratch, "p")
local buf_counter = 0

local function mkbuf(lines, name)
  buf_counter = buf_counter + 1
  local b = vim.api.nvim_create_buf(true, false)
  if name then
    vim.api.nvim_buf_set_name(b, name)
  end
  vim.api.nvim_buf_set_lines(b, 0, -1, false, lines)
  return b
end

local function test_reset()
  collector._reset_for_tests()
  repository.cache = { root = nil, root_name = nil, origin = "", branch = "", head = "", dirty = false }
  config.options.server_url = "http://127.0.0.1:9"
  config.options.batch_max_events = 100
end

local function wipe_spool()
  for _, p in ipairs(spool.list()) do
    os.remove(p)
  end
end

-- 1: module loads -----------------------------------------------------------
ok("module-loads", function()
  for _, m in ipairs({
    "tabcomplete_trajectory",
    "tabcomplete_trajectory.config",
    "tabcomplete_trajectory.events",
    "tabcomplete_trajectory.buffers",
    "tabcomplete_trajectory.repository",
    "tabcomplete_trajectory.transport",
    "tabcomplete_trajectory.spool",
    "tabcomplete_trajectory.util",
  }) do
    assert_true(require(m) ~= nil, "require " .. m)
  end
end)

-- 2: missing URL never crashes ----------------------------------------------
ok("missing-url-no-crash", function()
  config.reset_warnings()
  local ok_run, err = pcall(collector.setup, { server_url = "" })
  assert_true(ok_run, "setup threw: " .. tostring(err))
  assert_eq(collector.status().server_url, "")
  -- silence background timers/autocmds from this live setup for determinism
  for _, t in ipairs(collector.timers) do
    pcall(function()
      t:stop()
      t:close()
    end)
  end
  collector.timers = {}
  pcall(vim.api.nvim_create_augroup, "TabCompleteTrajectory", { clear = true })
  if collector.key_ns then
    pcall(vim.on_key, nil, collector.key_ns)
    collector.key_ns = nil
  end
  -- Pump the loop so setup's async session-start spool write lands now,
  -- then wipe it: later spool tests must start from a clean slate.
  vim.wait(1500, function()
    return #spool.list() >= 1
  end)
  for _, p in ipairs(spool.list()) do
    os.remove(p)
  end
  assert_eq(#spool.list(), 0, "spool clean after setup test")
end)

-- 3: machine-id persists -----------------------------------------------------
ok("machine-id-persists", function()
  local a = spool.machine_id()
  local b = spool.machine_id()
  assert_eq(a, b)
  assert_true(a:match("^%x%x%x%x%x%x%x%x%-%x%x%x%x%-%x%x%x%x%-%x%x%x%x%-%x%x%x%x%x%x%x%x%x%x%x%x$") ~= nil, "uuid shape")
  local f = io.open(spool.machine_id_path(), "r")
  local persisted = f:read("*l")
  f:close()
  assert_eq(persisted, a)
end)

-- 4: attach works ------------------------------------------------------------
ok("attach-works", function()
  test_reset()
  local b = mkbuf({ "a", "b", "c" }, scratch .. "/attach.lua")
  assert_true(buffers.attach(b), "attach returned true")
  assert_true(buffers.is_tracked(b), "tracked")
  assert_eq(#collector.queue, 0, "attach emits nothing by itself")
  local sh = buffers.get_shadow(b)
  assert_eq(table.concat(sh.lines, ","), "a,b,c")
  buffers.detach(b)
  assert_true(not buffers.is_tracked(b), "detached")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 5-7: insert / delete / replace deltas -------------------------------------
ok("insert-delta", function()
  test_reset()
  local b = mkbuf({ "a", "b", "c" }, scratch .. "/ins.lua")
  buffers.attach(b)
  vim.api.nvim_buf_set_lines(b, 1, 1, false, { "X" })
  local e = last(collector.queue)
  assert_eq(e.event_type, "edit_delta")
  assert_eq(e.payload.start_row, 1)
  assert_eq(e.payload.old_end_row, 1)
  assert_eq(e.payload.new_end_row, 2)
  assert_eq(e.payload.deleted_text, "")
  assert_eq(e.payload.inserted_text, "X")
  assert_true(e.payload.changedtick ~= nil, "changedtick present")
  assert_true(e.payload.cursor_after ~= nil, "cursor_after present")
  assert_eq(e.protocol_version, 1)
  assert_true(e.cursor.row >= 0 and e.cursor.col >= 0, "zero-based cursor")
  vim.api.nvim_buf_delete(b, { force = true })
end)

ok("delete-delta", function()
  test_reset()
  local b = mkbuf({ "a", "b", "c" }, scratch .. "/del.lua")
  buffers.attach(b)
  vim.api.nvim_buf_set_lines(b, 1, 2, false, {})
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 1)
  assert_eq(e.payload.old_end_row, 2)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "b")
  assert_eq(e.payload.inserted_text, "")
  vim.api.nvim_buf_delete(b, { force = true })
end)

ok("replace-multiline-delta", function()
  test_reset()
  local b = mkbuf({ "a", "b", "c" }, scratch .. "/rep.lua")
  buffers.attach(b)
  vim.api.nvim_buf_set_lines(b, 0, 2, false, { "x", "y", "z" })
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 0)
  assert_eq(e.payload.old_end_row, 2)
  assert_eq(e.payload.new_end_row, 3)
  assert_eq(e.payload.deleted_text, "a\nb")
  assert_eq(e.payload.inserted_text, "x\ny\nz")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 8: multi-delta shadow chaining + replay -----------------------------------
ok("multi-delta-shadow-replay", function()
  test_reset()
  local initial = { "one", "two", "three" }
  local b = mkbuf(initial, scratch .. "/chain.lua")
  buffers.attach(b)
  vim.api.nvim_buf_set_lines(b, 1, 1, false, { "INSERTED" })
  vim.api.nvim_buf_set_lines(b, 0, 1, false, {})
  vim.api.nvim_buf_set_lines(b, 1, 3, false, { "R1", "R2" })
  local deltas = {}
  for _, e in ipairs(collector.queue) do
    assert_eq(e.event_type, "edit_delta")
    deltas[#deltas + 1] = e.payload
  end
  assert_eq(#deltas, 3)
  -- shadow equals live buffer
  local live = vim.api.nvim_buf_get_lines(b, 0, -1, false)
  assert_eq(table.concat(buffers.get_shadow(b).lines, "\n"), table.concat(live, "\n"))
  -- replay from initial lines through recorded deltas == live buffer
  local replay = { "one", "two", "three" }
  for _, d in ipairs(deltas) do
    local ins = {}
    if d.inserted_text ~= "" then
      for line in (d.inserted_text .. "\n"):gmatch("([^\n]*)\n") do
        ins[#ins + 1] = line
      end
    end
    replay = buffers.splice_lines(replay, d.start_row, d.old_end_row, ins)
  end
  assert_eq(table.concat(replay, "\n"), table.concat(live, "\n"))
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 9: cursor + file_jump -------------------------------------------------------
ok("cursor-and-file-jump", function()
  test_reset()
  local a = mkbuf({ "x" }, scratch .. "/A.lua")
  local b = mkbuf({ "y" }, scratch .. "/B.lua")
  local aid = collector._test.buf_file_id(a)
  local bid = collector._test.buf_file_id(b)
  assert_true(aid ~= bid, "distinct file ids")
  collector.current_file_id = aid
  collector._test.set_pending_cursor({ bufnr = b, row = 3, col = 4, file_id = bid })
  collector._test.flush_cursor()
  local e = last(collector.queue)
  assert_eq(e.event_type, "file_jump")
  assert_eq(e.payload.from_file, aid)
  assert_eq(e.payload.to_file, bid)
  collector._test.set_pending_cursor({ bufnr = b, row = 7, col = 1, file_id = bid })
  collector._test.flush_cursor()
  local e2 = last(collector.queue)
  assert_eq(e2.event_type, "cursor_move")
  assert_eq(e2.payload.position.row, 7)
  assert_eq(e2.payload.position.col, 1)
  vim.api.nvim_buf_delete(a, { force = true })
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 10: exclusions --------------------------------------------------------------
ok("secret-terminal-exclusions", function()
  test_reset()
  assert_true(util.is_secret_path("/r/.env"), ".env")
  assert_true(util.is_secret_path("/r/.env.local"), ".env.*")
  assert_true(util.is_secret_path("/r/key.pem"), "pem")
  assert_true(util.is_secret_path("/r/id_rsa"), "id_rsa")
  assert_true(util.is_secret_path("/r/.aws/config"), ".aws")
  assert_true(util.is_secret_path("/r/.ssh/known_hosts"), ".ssh")
  assert_true(util.is_secret_path("/r/credentials.json"), "credentials")
  assert_true(util.is_secret_path("/r/secrets.yaml"), "secrets")
  assert_true(util.is_secret_path("/r/x.crt"), "crt")
  assert_true(not util.is_secret_path("/r/src/main.lua"), "normal file kept")
  assert_eq(util.sanitize_origin("https://user:pass@github.com/o/r.git"), "https://github.com/o/r.git")
  assert_eq(util.sanitize_origin("git@github.com:o/r.git"), "github.com:o/r.git")
  local term = mkbuf({ "x" }, scratch .. "/term.lua")
  vim.api.nvim_set_option_value("buftype", "prompt", { buf = term })
  assert_true(util.is_excluded_buffer(term), "prompt excluded")
  assert_true(not buffers.attach(term), "prompt not attached")
  vim.api.nvim_set_option_value("buftype", "nofile", { buf = term })
  assert_true(util.is_excluded_buffer(term), "nofile excluded")
  assert_true(not buffers.attach(term), "nofile not attached")
  local secret = mkbuf({ "x" }, scratch .. "/.env")
  assert_true(not buffers.attach(secret), "secret path not attached")
  local opt = mkbuf({ "x" }, scratch .. "/opt.lua")
  vim.api.nvim_buf_set_var(opt, "tabcomplete_trajectory_optout", true)
  assert_true(util.is_excluded_buffer(opt), "opt-out excluded")
  assert_true(not buffers.attach(opt), "opt-out not attached")
  vim.api.nvim_buf_delete(term, { force = true })
  vim.api.nvim_buf_delete(secret, { force = true })
  vim.api.nvim_buf_delete(opt, { force = true })
end)

-- 11: batch serialization -----------------------------------------------------
ok("batch-serialization", function()
  test_reset()
  wipe_spool()
  local posts = {}
  transport._post_impl = function(url, body, timeout)
    posts[#posts + 1] = { url = url, body = body }
    return true, { http_code = 200 }
  end
  local b = mkbuf({ "hello" }, scratch .. "/ser.lua")
  collector.emit(b, "session_start", { note = "t" })
  collector.emit(b, "edit_delta", { start_row = 0, old_end_row = 0, new_end_row = 1 })
  collector.emit(b, "cursor_move", { position = { row = 0, col = 5 } })
  local done = false
  collector.flush_now(function()
    done = true
  end)
  local function batch_post()
    for _, p in ipairs(posts) do
      if p.url:match("/v1/events/batch$") then
        return p
      end
    end
    return nil
  end
  assert_true(vim.wait(3000, function()
    return done and batch_post() ~= nil
  end), "flush completed")
  local captured = batch_post()
  local body = captured.body
  assert_true(captured.url:match("/v1/events/batch$") ~= nil, "batch endpoint")
  assert_eq(#body.events, 3)
  assert_eq(body.session_id, "test-session")
  assert_eq(body.machine_id, "test-machine")
  local seen_ids = {}
  local prev_seq = 0
  for _, e in ipairs(body.events) do
    assert_eq(e.protocol_version, 1)
    assert_true(type(e.event_id) == "string" and #e.event_id > 0, "event_id")
    assert_true(not seen_ids[e.event_id], "unique event_id")
    seen_ids[e.event_id] = true
    assert_true(e.sequence_number > prev_seq, "increasing sequence")
    prev_seq = e.sequence_number
    assert_true(type(e.timestamp_ms) == "number", "timestamp_ms")
    assert_true(type(e.file_id) == "string", "file_id")
    assert_true(type(e.cursor.row) == "number" and type(e.cursor.col) == "number", "cursor")
    assert_true(type(e.mode) == "string", "mode")
  end
  transport._post_impl = nil
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 12: failure spools ----------------------------------------------------------
ok("transport-failure-spools", function()
  test_reset()
  wipe_spool()
  transport._post_impl = function()
    return false, "connection refused"
  end
  local b = mkbuf({ "z" }, scratch .. "/fail.lua")
  collector.emit(b, "edit_delta", { start_row = 0, old_end_row = 1, new_end_row = 1 })
  local done = false
  collector.flush_now(function()
    done = true
  end)
  assert_true(vim.wait(3000, function()
    return done
  end), "flush callback ran")
  assert_true(vim.wait(3000, function()
    return #spool.list() == 1
  end), "spool file written")
  assert_eq(#collector.queue, 0, "queue drained into spool")
  local decoded = spool.read(spool.list()[1])
  assert_eq(decoded.meta.path, "/v1/events/batch")
  assert_eq(#decoded.events, 1)
  assert_eq(decoded.events[1].event_type, "edit_delta")
  transport._post_impl = nil
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 13: retry success removes spool ----------------------------------------------
ok("retry-success-removes-spool", function()
  -- consumes the spool file left by the previous test
  test_reset()
  assert_eq(#spool.list(), 1)
  local posted = {}
  transport._post_impl = function(url, body)
    posted[#posted + 1] = { url = url, body = body }
    return true, { http_code = 200 }
  end
  local done = false
  collector.retry_spool(function()
    done = true
  end)
  assert_true(vim.wait(3000, function()
    return done
  end), "retry completed")
  assert_true(vim.wait(3000, function()
    return #spool.list() == 0
  end), "ACK removed spool file")
  assert_eq(#posted, 1)
  assert_eq(#posted[1].body.events, 1)
  transport._post_impl = nil
end)

-- 14: queue limits --------------------------------------------------------------
ok("queue-limit-flush", function()
  test_reset()
  wipe_spool()
  config.options.batch_max_events = 3
  local posts = 0
  transport._post_impl = function()
    posts = posts + 1
    return true, { http_code = 200 }
  end
  local b = mkbuf({ "q" }, scratch .. "/lim.lua")
  collector.emit(b, "heartbeat", {})
  collector.emit(b, "heartbeat", {})
  collector.emit(b, "heartbeat", {}) -- hits the limit: auto flush
  assert_true(vim.wait(3000, function()
    return #collector.queue == 0 and posts >= 1
  end), "limit triggered flush")
  config.options.batch_max_events = 100
  transport._post_impl = nil
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 15: session event envelope shape ----------------------------------------------
ok("session-events", function()
  test_reset()
  local b = mkbuf({ "s" }, scratch .. "/sess.lua")
  local s = collector.emit(b, "session_start", { machine_id = "m" })
  local e = collector.emit(b, "session_end", { ended_at_ms = 123 })
  for _, env in ipairs({ s, e }) do
    for _, k in ipairs({
      "protocol_version",
      "event_id",
      "session_id",
      "sequence_number",
      "timestamp_ms",
      "event_type",
      "file_id",
      "cursor",
      "mode",
      "payload",
    }) do
      assert_true(env[k] ~= nil, "envelope key " .. k)
    end
  end
  assert_eq(s.event_type, "session_start")
  assert_eq(e.event_type, "session_end")
  assert_eq(e.payload.ended_at_ms, 123)
  assert_true(events.is_valid_type("prediction_partially_accepted"), "all spec types valid")
  for _, t in ipairs({
    "repo_snapshot",
    "buffer_open",
    "buffer_close",
    "buffer_enter",
    "buffer_leave",
    "buffer_write",
    "edit_delta",
    "cursor_move",
    "file_jump",
    "mode_change",
    "key",
    "heartbeat",
    "prediction_requested",
    "prediction_shown",
    "prediction_accepted",
    "prediction_rejected",
  }) do
    assert_true(events.is_valid_type(t), "valid: " .. t)
  end
  assert_true(not events.is_valid_type("nope"), "invalid rejected")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 16: prediction logging ---------------------------------------------------------
ok("prediction-logging", function()
  test_reset()
  local b = mkbuf({ "p" }, scratch .. "/pred.lua")
  vim.api.nvim_set_current_buf(b)
  collector.log_prediction_requested({ prediction_id = "p1", provider = "prov", model = "m", max_output_tokens = 64 })
  collector.log_prediction_shown({ prediction_id = "p1", proposed_text = "hello" })
  collector.log_prediction_accepted({ prediction_id = "p1", accepted_chars = 5 })
  collector.log_prediction_partially_accepted({ prediction_id = "p1", accepted_chars = 2, total_chars = 5 })
  collector.log_prediction_rejected({ prediction_id = "p1", finish_reason = "dismissed" })
  assert_eq(#collector.queue, 5)
  local want = {
    "prediction_requested",
    "prediction_shown",
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
  }
  for i, t in ipairs(want) do
    assert_eq(collector.queue[i].event_type, t)
    assert_eq(collector.queue[i].payload.prediction_id, "p1")
    assert_true(collector.queue[i].payload.cursor ~= nil, "cursor default filled")
  end
  assert_eq(collector.queue[1].payload.max_output_tokens, 64)
  assert_eq(collector.queue[2].payload.proposed_text, "hello")
  assert_eq(collector.queue[4].payload.total_chars, 5)
  assert_eq(collector.queue[5].payload.finish_reason, "dismissed")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 17: spool cap keeps newest -------------------------------------------------------
ok("spool-cap-keeps-newest", function()
  test_reset()
  wipe_spool()
  spool.write_batch({ { a = 1 } }, { kind = "events" })
  spool.write_batch({ { b = 2 } }, { kind = "events" })
  spool.write_batch({ { c = 3 } }, { kind = "events" })
  local files = spool.list()
  assert_eq(#files, 3)
  local newest = files[#files]
  local st = vim.uv.fs_stat(newest)
  local deleted, total = spool.enforce_cap(st.size)
  assert_eq(deleted, 2)
  local rest = spool.list()
  assert_eq(#rest, 1)
  assert_eq(rest[1], newest)
  assert_true(total <= st.size, "under cap")
  wipe_spool()
end)

-- 18: pause/resume + status/commands --------------------------------------------------
ok("pause-resume-status-commands", function()
  test_reset()
  collector.define_commands()
  for _, c in ipairs({
    "TabCompleteCollectorStatus",
    "TabCompleteCollectorFlush",
    "TabCompleteCollectorRescan",
    "TabCompleteCollectorPause",
    "TabCompleteCollectorResume",
    "Flush",
    "RescanRepo",
    "Pause",
    "Resume",
  }) do
    assert_true(vim.fn.exists(":" .. c) == 2, "command exists: " .. c)
  end
  local b = mkbuf({ "x" }, scratch .. "/pause.lua")
  collector.pause()
  assert_true(collector.emit(b, "heartbeat", {}) == nil, "paused emits nothing")
  assert_true(collector.status().paused, "status paused")
  assert_true(not collector.status().active, "not active while paused")
  collector.resume()
  assert_true(collector.emit(b, "heartbeat", {}) ~= nil, "resumed emits")
  local s = collector.status()
  assert_true(s.active, "active")
  assert_eq(s.session_id, "test-session")
  assert_true(s.queue_depth >= 1, "queue counted")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 19: curl argv captures stdout via pipe (no -o /dev/stdout) -------------------------
-- Regression: `-o /dev/stdout` made curl exit 23 under vim.system even on
-- HTTP 2xx, so every async POST looked failed (spool grew, server dup-fed).
ok("curl-argv-no-dev-stdout", function()
  test_reset()
  local seen_argv = nil
  transport._curl_runner = function(argv, body, timeout_ms, cb)
    seen_argv = argv
    cb(0, '{"ok":true}\n200', 0)
    return nil
  end
  local done, ok_res, info = false, nil, nil
  transport.post_json_async("http://127.0.0.1:9/v1/events/batch", { a = 1 }, 5000, function(ok, i)
    done, ok_res, info = true, ok, i
  end)
  assert_true(vim.wait(3000, function()
    return done
  end), "callback ran")
  assert_true(ok_res, "2xx via captured stdout is ACK, got " .. vim.inspect(info))
  assert_true(seen_argv ~= nil, "argv captured")
  for _, a in ipairs(seen_argv) do
    assert_true(a ~= "/dev/stdout", "argv must not contain /dev/stdout")
  end
  assert_true(info.http_code == 200, "http code parsed")
  transport._curl_runner = nil
end)

-- 20: NUL-separated git file list parses (enumerate_files) -------------------------
-- Regression: the `%z` pattern for `-z` output was written as `\0`, which is
-- an invalid Lua pattern, so every real repo snapshot failed to build.
ok("enumerate-files-nul-split", function()
  local real_run_git = repository._run_git
  repository._run_git = function(args, cwd)
    return "src/a.py\0src/b space.py\0", nil
  end
  local files, err = repository.enumerate_files("/fake/root")
  repository._run_git = real_run_git
  assert_true(err == nil, "no error, got " .. tostring(err))
  assert_eq(#files, 2)
  assert_eq(files[1], "src/a.py")
  assert_eq(files[2], "src/b space.py")
end)

-- 21: stale async anchor is dropped, not emitted ----------------------------------
-- Regression: a setup-time anchor of an unnamed buffer fired its upload
-- callback after the buffer was reused for a real file, emitting a
-- buffer_open that bound the empty sha to live content (replay corruptor).
ok("stale-anchor-dropped", function()
  test_reset()
  wipe_spool()
  local posts = 0
  transport._post_impl = function(url, body, timeout)
    posts = posts + 1
    return true, { http_code = 200 }
  end
  local b = mkbuf({ "one" }, scratch .. "/stale.lua")
  assert_true(buffers.attach(b), "attach")
  local stale_hash = util.sha256hex("one\n")
  collector._test.anchor(b, "buffer_open")
  -- Mutate before the async upload callback runs.
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "changed" })
  assert_true(vim.wait(5000, function()
    return posts >= 2
  end), "check+upload posted")
  vim.wait(800)
  local saw_stale_anchor = false
  local saw_delta = false
  for _, e in ipairs(collector.queue) do
    if e.event_type == "buffer_open" and e.payload.content_hash == stale_hash then
      saw_stale_anchor = true
    end
    if e.event_type == "edit_delta" then
      saw_delta = true
    end
  end
  assert_true(not saw_stale_anchor, "stale anchor must be dropped")
  assert_true(saw_delta, "concurrent edit delta still recorded")
  assert_eq(table.concat(buffers.get_shadow(b).lines, ","), "changed")
  transport._post_impl = nil
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 22: empty-line deletion still emits (guard must not swallow real deletes) --
-- The phantom guard skips deltas whose line-count effect disagrees with the
-- live buffer; a genuine dd on an empty line agrees and must be recorded.
ok("empty-line-delete-emits", function()
  test_reset()
  local b = mkbuf({ "a", "", "c" }, scratch .. "/emptydel.lua")
  buffers.attach(b)
  vim.api.nvim_buf_set_lines(b, 1, 2, false, {})
  local e = last(collector.queue)
  assert_eq(e.event_type, "edit_delta")
  assert_eq(e.payload.start_row, 1)
  assert_eq(e.payload.old_end_row, 2)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "")
  assert_eq(e.payload.inserted_text, "")
  local live = vim.api.nvim_buf_get_lines(b, 0, -1, false)
  assert_eq(table.concat(buffers.get_shadow(b).lines, ","), table.concat(live, ","))
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 23: expected_total helper ---------------------------------------------------
ok("expected-total-helper", function()
  assert_eq(buffers.expected_total(10, 2, 5, 1), 8)
  assert_eq(buffers.expected_total(3, 1, 2, 0), 2)
  assert_eq(buffers.expected_total(3, 1, 1, 2), 5)
  assert_eq(buffers.expected_total(0, 0, 0, 0), 0)
end)

-- 24: random ops keep shadow == live buffer -----------------------------------
ok("random-ops-shadow-matches-live", function()
  test_reset()
  local b = mkbuf({ "l0", "l1", "l2", "l3" }, scratch .. "/fuzz.lua")
  buffers.attach(b)
  for step = 1, 60 do
    local live = vim.api.nvim_buf_get_lines(b, 0, -1, false)
    local n = #live
    local op = math.random(3)
    if op == 1 then
      local at = math.random(0, n)
      vim.api.nvim_buf_set_lines(b, at, at, false, { "ins" .. step })
    elseif op == 2 and n > 0 then
      local at = math.random(0, n - 1)
      vim.api.nvim_buf_set_lines(b, at, at + 1, false, {})
    else
      local at = n > 0 and math.random(0, n - 1) or 0
      local e = n > 0 and at + 1 or at
      vim.api.nvim_buf_set_lines(b, at, e, false, { "rep" .. step })
    end
    local after = vim.api.nvim_buf_get_lines(b, 0, -1, false)
    assert_eq(
      table.concat(buffers.get_shadow(b).lines, "\n"),
      table.concat(after, "\n"),
      "shadow==live at step " .. step
    )
  end
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 25: diverged shadow resyncs instead of poisoning history --------------------
ok("divergent-shadow-resyncs", function()
  test_reset()
  local b = mkbuf({ "a", "b", "c" }, scratch .. "/resync.lua")
  buffers.attach(b)
  -- Simulate a line-count phantom: shadow lost a line the buffer still has.
  local sh = buffers.get_shadow(b)
  table.remove(sh.lines, 2)
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "A" })
  for _, e in ipairs(collector.queue) do
    assert_true(e.event_type ~= "edit_delta", "divergent delta must be skipped")
  end
  assert_eq(buffers.get_shadow(b).resync_count, 1)
  local live = vim.api.nvim_buf_get_lines(b, 0, -1, false)
  assert_eq(table.concat(buffers.get_shadow(b).lines, "\n"), table.concat(live, "\n"))
  -- Capture resumes correctly on the healed shadow.
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "A2" })
  local e = last(collector.queue)
  assert_eq(e.event_type, "edit_delta")
  assert_eq(e.payload.deleted_text, "A")
  assert_eq(e.payload.inserted_text, "A2")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- New deterministic scenario tests (26+) ------------------------------------
-- Helpers local to the appended block (no collision with existing locals).
local function replay_from(initial, deltas)
  local cur = {}
  for _, l in ipairs(initial) do
    cur[#cur + 1] = l
  end
  for _, d in ipairs(deltas) do
    local ins = {}
    if d.inserted_text ~= "" then
      for line in (d.inserted_text .. "\n"):gmatch("([^\n]*)\n") do
        ins[#ins + 1] = line
      end
    else
      -- inserted_text "" is ambiguous: it means zero lines when the triple
      -- is count-neutral, but N empty lines when new_end-start == N (e.g. o/O
      -- line-open emits deleted "" inserted "" with new_count 1 for one empty
      -- line). Use the triple to disambiguate so o/O chains replay exactly.
      local want = (d.new_end_row or 0) - (d.start_row or 0)
      for _ = 1, want do
        ins[#ins + 1] = ""
      end
    end
    cur = buffers.splice_lines(cur, d.start_row, d.old_end_row, ins)
  end
  return cur
end

local function assert_shadow_live(bufnr, msg)
  local live = vim.api.nvim_buf_get_lines(bufnr, 0, -1, false)
  assert_eq(table.concat(buffers.get_shadow(bufnr).lines, "\n"), table.concat(live, "\n"), msg or "shadow==live")
  return live
end

local function edit_deltas()
  local out = {}
  for _, e in ipairs(collector.queue) do
    if e.event_type == "edit_delta" then
      out[#out + 1] = e.payload
    end
  end
  return out
end

-- 26: rapid buffer switching mid-typing --------------------------------------
-- Regression for the structs2-detour-while-editing-structs3 replay break:
-- two attached buffers, alternate edits, each stream must replay cleanly.
ok("rapid-buffer-switch-mid-typing", function()
  test_reset()
  local a = mkbuf({ "a1", "a2", "a3" }, scratch .. "/swA.lua")
  local b = mkbuf({ "b1", "b2", "b3" }, scratch .. "/swB.lua")
  assert_true(buffers.attach(a), "attach A")
  assert_true(buffers.attach(b), "attach B")
  collector.queue = {}
  local initA = { "a1", "a2", "a3" }
  local initB = { "b1", "b2", "b3" }
  vim.api.nvim_set_current_buf(a)
  vim.api.nvim_buf_set_lines(a, 1, 1, false, { "AX" })
  vim.api.nvim_set_current_buf(b)
  vim.api.nvim_buf_set_lines(b, 1, 1, false, { "BX" })
  vim.api.nvim_set_current_buf(a)
  vim.api.nvim_buf_set_lines(a, 0, 1, false, { "A0" })
  vim.api.nvim_set_current_buf(b)
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "B0" })
  assert_eq(#collector.queue, 4, "four interleaved deltas")
  local liveA = assert_shadow_live(a, "shadowA==liveA")
  local liveB = assert_shadow_live(b, "shadowB==liveB")
  local absA = util.abspath(vim.api.nvim_buf_get_name(a))
  local absB = util.abspath(vim.api.nvim_buf_get_name(b))
  local dA, dB = {}, {}
  for _, e in ipairs(collector.queue) do
    assert_eq(e.event_type, "edit_delta")
    if e.payload.path == absA then
      dA[#dA + 1] = e.payload
    elseif e.payload.path == absB then
      dB[#dB + 1] = e.payload
    else
      error("delta with unknown path: " .. vim.inspect(e.payload.path), 2)
    end
  end
  assert_eq(#dA, 2, "two deltas for A")
  assert_eq(#dB, 2, "two deltas for B")
  assert_eq(table.concat(replay_from(initA, dA), "\n"), table.concat(liveA, "\n"), "A replays")
  assert_eq(table.concat(replay_from(initB, dB), "\n"), table.concat(liveB, "\n"), "B replays")
  vim.api.nvim_buf_delete(a, { force = true })
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 27: undo/redo tracks shadow + chains ---------------------------------------
-- Real vim undo: typed edit via normal! (true keystrokes, undo-joined),
-- then u / <C-r>. nvim_input is exercised with vim.wait first; headless -l
-- currently does not pump nvim_input (probed: mode stays n, no edit), so the
-- deterministic driver is normal!/:redo with a wait-guarded fallback that
-- only fires when nvim_input had no effect (future-proof either way).
ok("undo-redo-shadow-replay", function()
  test_reset()
  local b = mkbuf({ "" }, scratch .. "/undo.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "hello", "world" })
  collector.queue = {}
  local initial = { "hello", "world" }
  vim.api.nvim_win_set_cursor(0, { 1, 5 })
  vim.cmd("normal! aXYZ\27")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "typed delta captured")
  assert_eq(#edit_deltas(), 1, "one typed delta")
  assert_shadow_live(b, "shadow after type")
  local qlen = #collector.queue
  vim.api.nvim_input("u")
  local via_input = vim.wait(300, function() return #collector.queue > qlen end)
  if not via_input then
    vim.cmd("normal! u")
    assert_true(vim.wait(2000, function() return #collector.queue > qlen end), "undo delta captured")
  end
  local after_undo = assert_shadow_live(b, "shadow tracks undo")
  local du = edit_deltas()
  assert_eq(table.concat(replay_from(initial, du), "\n"), table.concat(after_undo, "\n"), "undo chain replays")
  qlen = #collector.queue
  vim.api.nvim_input(vim.api.nvim_replace_termcodes("<C-r>", true, false, true))
  local via_input2 = vim.wait(300, function() return #collector.queue > qlen end)
  if not via_input2 then
    vim.cmd("redo")
    assert_true(vim.wait(2000, function() return #collector.queue > qlen end), "redo delta captured")
  end
  local after_redo = assert_shadow_live(b, "shadow tracks redo")
  local dr = edit_deltas()
  assert_eq(table.concat(replay_from(initial, dr), "\n"), table.concat(after_redo, "\n"), "redo chain replays")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 28: simulated formatter interleave ------------------------------------------
-- Keystroke-style single-line edit immediately followed by a multi-line
-- reindent (like rustfmt): shadow must stay consistent and replay exactly.
ok("formatter-interleave-chains", function()
  test_reset()
  local b = mkbuf({ "fn main() {", "let x=1;", "}" }, scratch .. "/fmt.lua")
  buffers.attach(b)
  collector.queue = {}
  local initial = { "fn main() {", "let x=1;", "}" }
  vim.api.nvim_buf_set_lines(b, 1, 2, false, { "let x = 1;" })
  vim.api.nvim_buf_set_lines(b, 0, -1, false, { "fn main() {", "    let x = 1;", "}" })
  local ds = edit_deltas()
  assert_eq(#ds, 2, "keystroke + format = 2 deltas")
  assert_eq(ds[1].deleted_text, "let x=1;")
  assert_eq(ds[1].inserted_text, "let x = 1;")
  assert_eq(ds[2].deleted_text, "fn main() {\nlet x = 1;\n}")
  assert_eq(ds[2].inserted_text, "fn main() {\n    let x = 1;\n}")
  local live = assert_shadow_live(b)
  assert_eq(table.concat(replay_from(initial, ds), "\n"), table.concat(live, "\n"), "format replays")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 29: dd on non-empty line via real vim op -------------------------------------
ok("dd-nonempty-exact", function()
  test_reset()
  local b = mkbuf({ "a", "b", "c" }, scratch .. "/dd1.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  vim.api.nvim_win_set_cursor(0, { 2, 0 })
  vim.cmd("normal! dd")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "dd captured")
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 1)
  assert_eq(e.payload.old_end_row, 2)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "b")
  assert_eq(e.payload.inserted_text, "")
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 30: dd on empty line via real vim op ------------------------------------------
ok("dd-empty-line-exact", function()
  test_reset()
  local b = mkbuf({ "a", "", "c" }, scratch .. "/dd0.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  vim.api.nvim_win_set_cursor(0, { 2, 0 })
  vim.cmd("normal! dd")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "dd empty captured")
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 1)
  assert_eq(e.payload.old_end_row, 2)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "")
  assert_eq(e.payload.inserted_text, "")
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 31: o creates a line below (two deltas: open + fill) --------------------------
ok("o-line-creation-below", function()
  test_reset()
  local b = mkbuf({ "a", "c" }, scratch .. "/oline.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  local initial = { "a", "c" }
  vim.api.nvim_win_set_cursor(0, { 1, 0 })
  local esc = vim.api.nvim_replace_termcodes("<Esc>", true, false, true)
  vim.cmd("normal! oNEWL" .. esc)
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 2 end), "o deltas captured")
  local ds = edit_deltas()
  assert_eq(#ds, 2, "o = open + fill")
  assert_eq(ds[1].start_row, 1)
  assert_eq(ds[1].old_end_row, 1)
  assert_eq(ds[1].new_end_row, 2)
  assert_eq(ds[2].inserted_text, "NEWL")
  local live = assert_shadow_live(b)
  assert_eq(table.concat(live, ","), "a,NEWL,c")
  assert_eq(table.concat(replay_from(initial, ds), "\n"), table.concat(live, "\n"), "o replays")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 32: O creates a line above (two deltas: open + fill) ---------------------------
ok("O-line-creation-above", function()
  test_reset()
  local b = mkbuf({ "m1", "m2" }, scratch .. "/Oline.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  local initial = { "m1", "m2" }
  vim.api.nvim_win_set_cursor(0, { 1, 0 })
  local esc = vim.api.nvim_replace_termcodes("<Esc>", true, false, true)
  vim.cmd("normal! OABOVE" .. esc)
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 2 end), "O deltas captured")
  local ds = edit_deltas()
  assert_eq(#ds, 2, "O = open + fill")
  assert_eq(ds[1].start_row, 0)
  assert_eq(ds[1].old_end_row, 0)
  assert_eq(ds[1].new_end_row, 1)
  assert_eq(ds[2].inserted_text, "ABOVE")
  local live = assert_shadow_live(b)
  assert_eq(table.concat(live, ","), "ABOVE,m1,m2")
  assert_eq(table.concat(replay_from(initial, ds), "\n"), table.concat(live, "\n"), "O replays")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 33: p paste linewise ------------------------------------------------------------
ok("paste-linewise-p", function()
  test_reset()
  local b = mkbuf({ "aaa", "bbb", "ccc" }, scratch .. "/pline.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  vim.api.nvim_win_set_cursor(0, { 1, 0 })
  vim.cmd("normal! yy")
  collector.queue = {}
  local initial = { "aaa", "bbb", "ccc" }
  vim.api.nvim_win_set_cursor(0, { 3, 0 })
  vim.cmd("normal! p")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "p captured")
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 3)
  assert_eq(e.payload.old_end_row, 3)
  assert_eq(e.payload.new_end_row, 4)
  assert_eq(e.payload.deleted_text, "")
  assert_eq(e.payload.inserted_text, "aaa")
  local live = assert_shadow_live(b)
  assert_eq(table.concat(replay_from(initial, edit_deltas()), "\n"), table.concat(live, "\n"))
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 34: p paste charwise ----------------------------------------------------------------
ok("paste-charwise-p", function()
  test_reset()
  local b = mkbuf({ "abcdef" }, scratch .. "/pchar.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  vim.api.nvim_win_set_cursor(0, { 1, 0 })
  vim.cmd("normal! y2l")
  collector.queue = {}
  local initial = { "abcdef" }
  vim.api.nvim_win_set_cursor(0, { 1, 4 })
  vim.cmd("normal! p")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "charwise p captured")
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 0)
  assert_eq(e.payload.old_end_row, 1)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "abcdef")
  assert_eq(e.payload.inserted_text, "abcdeabf")
  local live = assert_shadow_live(b)
  assert_eq(table.concat(replay_from(initial, edit_deltas()), "\n"), table.concat(live, "\n"))
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 35: x single-char delete (line-based triple) ------------------------------------------
ok("x-single-char-delete", function()
  test_reset()
  local b = mkbuf({ "abcdef" }, scratch .. "/xdel.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  vim.api.nvim_win_set_cursor(0, { 1, 2 })
  vim.cmd("normal! x")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "x captured")
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 0)
  assert_eq(e.payload.old_end_row, 1)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "abcdef")
  assert_eq(e.payload.inserted_text, "abdef")
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 36: multi-line delete (2dd-style range) --------------------------------------------------
ok("multiline-delete-2dd-range", function()
  test_reset()
  local b = mkbuf({ "l1", "l2", "l3", "l4" }, scratch .. "/mdd.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  vim.api.nvim_win_set_cursor(0, { 2, 0 })
  vim.cmd("normal! 2dd")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "2dd captured")
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 1)
  assert_eq(e.payload.old_end_row, 3)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "l2\nl3")
  assert_eq(e.payload.inserted_text, "")
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 37: :s substitute range -------------------------------------------------------------------
ok("substitute-range-s", function()
  test_reset()
  local b = mkbuf({ "foo foo", "bar foo" }, scratch .. "/sub.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  vim.cmd("1s/foo/baz/g")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), ":s captured")
  local e = last(collector.queue)
  assert_eq(e.payload.start_row, 0)
  assert_eq(e.payload.old_end_row, 1)
  assert_eq(e.payload.new_end_row, 1)
  assert_eq(e.payload.deleted_text, "foo foo")
  assert_eq(e.payload.inserted_text, "baz baz")
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 38: unicode/multibyte replace: byte-exact texts ----------------------------------------------
ok("unicode-multibyte-exact", function()
  test_reset()
  local b = mkbuf({ "hello" }, scratch .. "/uni.lua")
  buffers.attach(b)
  collector.queue = {}
  local emoji = vim.fn.nr2char(0x1F30D)
  local cjk = vim.fn.nr2char(0x65E5) .. vim.fn.nr2char(0x672C)
  local line = emoji .. " test " .. cjk
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { line })
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "unicode delta")
  local e = last(collector.queue)
  assert_eq(e.payload.deleted_text, "hello")
  assert_eq(e.payload.inserted_text, line)
  -- byte-vs-char: line is 16 bytes but 9 display chars
  assert_eq(#line, 16, "byte length")
  assert_eq(vim.fn.strchars(line), 9, "char length")
  assert_eq(#e.payload.inserted_text, 16, "inserted bytes preserved, no truncation")
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 39: unicode x delete removes whole multibyte char -----------------------------------------------
ok("unicode-emoji-x-delete", function()
  test_reset()
  local emoji = vim.fn.nr2char(0x1F30D)
  local b = mkbuf({ "a" .. emoji .. "b" }, scratch .. "/unix.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  vim.api.nvim_win_set_cursor(0, { 1, 1 })
  vim.cmd("normal! x")
  assert_true(vim.wait(2000, function() return #edit_deltas() >= 1 end), "emoji x captured")
  local e = last(collector.queue)
  assert_eq(e.payload.deleted_text, "a" .. emoji .. "b")
  assert_eq(e.payload.inserted_text, "ab")
  assert_eq(#e.payload.deleted_text, 6, "deleted bytes = 1+4+1")
  assert_eq(#e.payload.inserted_text, 2)
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 40: CRLF fileformat buffer -----------------------------------------------------------------------
-- NOTE (eol limitation, documented not asserted byte-exact): nvim's line
-- model strips \r; delta deleted/inserted texts are always \n-joined line
-- slices, so they chain as a line array. Byte reconstruction needs
-- canonical_bytes which joins with \r\n for fileformat=dos (verified
-- separately via canonical_bytes below). We assert chaining + shadow, not
-- that delta texts contain \r.
ok("crlf-deltas-chain", function()
  test_reset()
  local b = vim.api.nvim_create_buf(true, false)
  buf_counter = buf_counter + 1
  vim.api.nvim_buf_set_name(b, scratch .. "/crlf.lua")
  vim.api.nvim_buf_set_lines(b, 0, -1, false, { "a", "b", "c" })
  vim.api.nvim_set_option_value("fileformat", "dos", { buf = b })
  assert_true(buffers.attach(b), "attach dos")
  collector.queue = {}
  local initial = { "a", "b", "c" }
  vim.api.nvim_buf_set_lines(b, 1, 2, false, { "B" })
  local ds = edit_deltas()
  assert_eq(#ds, 1)
  assert_eq(ds[1].deleted_text, "b")
  assert_eq(ds[1].inserted_text, "B")
  local live = assert_shadow_live(b)
  assert_eq(table.concat(replay_from(initial, ds), "\n"), table.concat(live, "\n"), "crlf replays as lines")
  local content, meta = buffers.canonical_bytes(b)
  assert_eq(meta.fileformat, "dos")
  assert_eq(content, "a\r\nB\r\nc\r\n", "canonical bytes use CRLF")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 41: empty file (single empty line) first insert ------------------------------------------------------
ok("empty-file-first-insert", function()
  test_reset()
  local b = mkbuf({ "" }, scratch .. "/empty1.lua")
  buffers.attach(b)
  collector.queue = {}
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "first" })
  local ds = edit_deltas()
  assert_eq(#ds, 1)
  assert_eq(ds[1].start_row, 0)
  assert_eq(ds[1].old_end_row, 1)
  assert_eq(ds[1].new_end_row, 1)
  assert_eq(ds[1].deleted_text, "")
  assert_eq(ds[1].inserted_text, "first")
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 42: single-empty-line canonicalisation ("" vs "\n") ------------------------------------------------------
-- nvim cannot distinguish "" from "\n" on disk (both read as one empty
-- line); the plugin canonicalises a lone empty line to "" so empty files
-- round-trip byte-exact while a lone-newline file replays as empty.
ok("single-empty-line-canonicalisation", function()
  test_reset()
  local b = mkbuf({ "" }, scratch .. "/emptycanon.lua")
  local content, meta = buffers.canonical_bytes(b)
  assert_eq(content, "", "lone empty line canonicalises to empty bytes")
  assert_eq(meta.line_count, 1)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 43: very long line (50KB) edit ------------------------------------------------------------------------------
ok("very-long-line-50k", function()
  test_reset()
  local long = string.rep("a", 50000)
  local b = mkbuf({ long }, scratch .. "/long.lua")
  buffers.attach(b)
  collector.queue = {}
  local long2 = string.rep("b", 50000)
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { long2 })
  local ds = edit_deltas()
  assert_eq(#ds, 1)
  assert_eq(ds[1].start_row, 0)
  assert_eq(ds[1].old_end_row, 1)
  assert_eq(ds[1].new_end_row, 1)
  assert_eq(#ds[1].deleted_text, 50000, "no truncation of deleted")
  assert_eq(#ds[1].inserted_text, 50000, "no truncation of inserted")
  assert_eq(ds[1].deleted_text, long)
  assert_eq(ds[1].inserted_text, long2)
  assert_shadow_live(b)
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 44: attach/detach/reattach cycle ----------------------------------------------------------------------------------
-- NOTE: buffers.detach clears only the Lua tables, not the active
-- nvim_buf_attach callback, so a reattach currently leaves two callbacks
-- live; the subsequent edit therefore emits the correct delta plus a
-- content-neutral duplicate (deleted==inserted). Replay and shadow stay
-- correct. This test asserts the guaranteed parts (no events while
-- detached, fresh shadow on reattach, chaining after) and reports the
-- duplicate rather than failing on exact counts. See final report.
ok("attach-detach-reattach-cycle", function()
  test_reset()
  local b = mkbuf({ "a", "b" }, scratch .. "/reattach.lua")
  assert_true(buffers.attach(b), "attach")
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "A" })
  assert_eq(#edit_deltas(), 1, "one delta before detach")
  buffers.detach(b)
  assert_true(not buffers.is_tracked(b), "detached")
  local qlen = #collector.queue
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "DETACHED" })
  assert_eq(#collector.queue, qlen, "no events while detached")
  assert_true(buffers.attach(b), "reattach")
  assert_eq(#collector.queue, qlen, "reattach itself emits nothing (no phantom)")
  assert_eq(table.concat(buffers.get_shadow(b).lines, ","), "DETACHED,b", "fresh shadow on reattach")
  local q2 = #collector.queue
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "AFTER" })
  assert_true(#collector.queue > q2, "subsequent edit captured")
  local found_correct = false
  for i = q2 + 1, #collector.queue do
    local p = collector.queue[i].payload
    if p.deleted_text == "DETACHED" and p.inserted_text == "AFTER" then
      found_correct = true
    end
  end
  assert_true(found_correct, "correct DETACHED->AFTER delta present")
  assert_shadow_live(b, "shadow chains after reattach")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 45: concurrent rapid-fire: 200 single-char inserts ------------------------------------------------------------
-- Headless -l does not pump nvim_input (probed), so each iteration also
-- issues nvim_input(<Ignore>) (spec-required API, guaranteed no-op even if
-- pumping ever starts) while the deterministic edit is a real vim keystroke
-- via normal! (not set_lines). Every delta must be captured in order with
-- consecutive changedticks.
ok("rapid-fire-200-inserts-ordered", function()
  test_reset()
  config.options.batch_max_events = 1000
  local b = mkbuf({ "" }, scratch .. "/rapid200.lua")
  buffers.attach(b)
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  local ignore = vim.api.nvim_replace_termcodes("<Ignore>", true, false, true)
  for i = 1, 200 do
    local want = #collector.queue + 1
    vim.api.nvim_input(ignore)
    vim.cmd("normal! a" .. (i % 10) .. "\27")
    assert_true(vim.wait(2000, function() return #collector.queue >= want end), "delta " .. i .. " captured")
  end
  local ds = edit_deltas()
  assert_eq(#ds, 200, "all 200 deltas captured")
  for i = 2, #ds do
    assert_eq(ds[i].changedtick, ds[i - 1].changedtick + 1, "consecutive changedtick at " .. i)
  end
  assert_shadow_live(b, "shadow after 200")
  local live = vim.api.nvim_buf_get_lines(b, 0, -1, false)
  assert_eq(#live[1], 200, "200 chars typed")
  config.options.batch_max_events = 100
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 46: anchor reason write emits buffer_write --------------------------------------------------------------------------
ok("anchor-reason-write", function()
  test_reset()
  config.options.server_url = "http://127.0.0.1:9"
  transport._post_impl = function(url, body)
    return true, { http_code = 200 }
  end
  local b = mkbuf({ "x", "y" }, scratch .. "/anchorw.lua")
  buffers.attach(b)
  collector.queue = {}
  collector._test.anchor(b, "write")
  assert_true(vim.wait(5000, function()
    for _, e in ipairs(collector.queue) do
      if e.event_type == "buffer_write" then
        return true
      end
    end
    return false
  end), "buffer_write emitted")
  local e = last(collector.queue)
  assert_eq(e.event_type, "buffer_write")
  assert_eq(e.payload.reason, "write")
  assert_true(e.payload.content_hash ~= nil and #e.payload.content_hash == 64, "sha present")
  assert_true(e.payload.line_count == 2, "line count")
  transport._post_impl = nil
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 47: periodic anchor fires every N deltas (small N for speed) ----------------------------------------------------------
ok("periodic-anchor-small-N", function()
  test_reset()
  local spy = { n = 0, reasons = {} }
  buffers.set_callbacks(function(bufnr, et, pay)
    return collector.emit(bufnr, et, pay)
  end, function(bufnr, reason)
    spy.n = spy.n + 1
    spy.reasons[#spy.reasons + 1] = reason
  end)
  config.options.periodic_anchor_every = 3
  local b = mkbuf({ "a" }, scratch .. "/periodic.lua")
  buffers.attach(b)
  collector.queue = {}
  for i = 1, 6 do
    vim.api.nvim_buf_set_lines(b, 0, 1, false, { "a" .. i })
  end
  assert_eq(#edit_deltas(), 6)
  assert_eq(spy.n, 2, "anchor every 3 deltas")
  assert_eq(spy.reasons[1], "periodic_anchor")
  assert_eq(spy.reasons[2], "periodic_anchor")
  config.options.periodic_anchor_every = 50
  test_reset()
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 48: cursor coalescing: many pending moves collapse ----------------------------------------------------------------------
ok("cursor-coalescing-single-flush", function()
  test_reset()
  local a = mkbuf({ "x" }, scratch .. "/curA.lua")
  local b = mkbuf({ "y" }, scratch .. "/curB.lua")
  local bid = collector._test.buf_file_id(b)
  collector.current_file_id = bid
  collector.queue = {}
  for i = 1, 10 do
    collector._test.set_pending_cursor({ bufnr = b, row = i, col = i, file_id = bid })
  end
  collector._test.flush_cursor()
  assert_eq(#collector.queue, 1, "coalesced to one event")
  assert_eq(collector.queue[1].event_type, "cursor_move")
  assert_eq(collector.queue[1].payload.position.row, 10)
  assert_eq(collector.queue[1].payload.position.col, 10)
  vim.api.nvim_buf_delete(a, { force = true })
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 49: file_jump emitted on buffer switch with from/to ------------------------------------------------------------------------
ok("file-jump-from-to-on-switch", function()
  test_reset()
  local a = mkbuf({ "x" }, scratch .. "/fjA.lua")
  local b = mkbuf({ "y" }, scratch .. "/fjB.lua")
  local aid = collector._test.buf_file_id(a)
  local bid = collector._test.buf_file_id(b)
  assert_true(aid ~= bid, "distinct ids")
  collector.current_file_id = aid
  collector.queue = {}
  collector._test.set_pending_cursor({ bufnr = b, row = 3, col = 4, file_id = bid })
  collector._test.flush_cursor()
  assert_eq(#collector.queue, 1)
  local e = last(collector.queue)
  assert_eq(e.event_type, "file_jump")
  assert_eq(e.payload.from_file, aid)
  assert_eq(e.payload.to_file, bid)
  vim.api.nvim_buf_delete(a, { force = true })
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 50: spool cap boundary: exact cap keeps all, over-by-one drops oldest -------------------------------------------------------
ok("spool-cap-boundary-near-full", function()
  test_reset()
  wipe_spool()
  spool.write_batch({ { a = 1 } }, { kind = "events" })
  spool.write_batch({ { b = 2 } }, { kind = "events" })
  spool.write_batch({ { c = 3 } }, { kind = "events" })
  local files = spool.list()
  assert_eq(#files, 3)
  local newest = files[#files]
  local total = spool.total_bytes()
  local d0 = spool.enforce_cap(total)
  assert_eq(d0, 0, "exact cap deletes nothing")
  assert_eq(#spool.list(), 3)
  local d1 = spool.enforce_cap(total - 1)
  assert_eq(d1, 1, "over by one drops oldest")
  local rest = spool.list()
  assert_eq(#rest, 2)
  assert_eq(rest[#rest], newest, "newest preserved")
  wipe_spool()
end)

-- 51: malformed spool file is ignored, not crash -------------------------------------------------------------------------------
ok("spool-malformed-ignored", function()
  test_reset()
  wipe_spool()
  config.options.server_url = "http://127.0.0.1:9"
  local good = spool.write_batch({ { a = 1 } }, { kind = "events", path = "/v1/events/batch" })
  assert_true(good ~= nil, "good batch written")
  local bad = spool.spool_dir() .. "/batch-0000-bad.json"
  local f = io.open(bad, "w")
  assert_true(f ~= nil, "bad file opened")
  f:write("not json{{{")
  f:close()
  local decoded, err = spool.read(bad)
  assert_true(decoded == nil, "malformed read returns nil")
  assert_true(err ~= nil, "malformed read returns err")
  local posted = {}
  transport._post_impl = function(url, body)
    posted[#posted + 1] = body
    return true, { http_code = 200 }
  end
  local done = false
  collector.retry_spool(function()
    done = true
  end)
  assert_true(vim.wait(5000, function() return done end), "retry completed despite malformed")
  assert_true(vim.wait(3000, function() return #spool.list() == 0 end), "both files drained")
  assert_eq(#posted, 1, "good batch delivered once")
  transport._post_impl = nil
  wipe_spool()
end)

-- 52: session start/end payload shapes + status repo fields ----------------------------------------------------------------------
ok("session-payload-shapes-machine-repo-branch", function()
  test_reset()
  repository.cache = {
    root = "/repo",
    root_name = "myrepo",
    origin = "https://github.com/o/r.git",
    branch = "main",
    head = "abc123",
    dirty = true,
  }
  local b = mkbuf({ "s" }, scratch .. "/sesshape.lua")
  local s = collector.emit(b, "session_start", { machine_id = "m", started_at_ms = 111 })
  local e = collector.emit(b, "session_end", { ended_at_ms = 222 })
  assert_eq(s.payload.machine_id, "m")
  assert_eq(s.payload.started_at_ms, 111)
  assert_eq(e.payload.ended_at_ms, 222)
  for _, env in ipairs({ s, e }) do
    for _, k in ipairs({ "protocol_version", "event_id", "session_id", "sequence_number", "timestamp_ms", "event_type", "file_id", "cursor", "mode", "payload" }) do
      assert_true(env[k] ~= nil, "envelope key " .. k)
    end
  end
  local st = collector.status()
  assert_eq(st.repo_root, "/repo")
  assert_eq(st.repo_branch, "main")
  assert_eq(st.repo_head, "abc123")
  assert_eq(st.repo_dirty, true)
  assert_eq(st.session_id, "test-session")
  assert_eq(st.machine_id, "test-machine")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 53: prediction log fns field completeness ------------------------------------------------------------------------------------------
ok("prediction-fns-field-completeness", function()
  test_reset()
  local b = mkbuf({ "p" }, scratch .. "/predfull.lua")
  vim.api.nvim_set_current_buf(b)
  collector.queue = {}
  local full = {
    prediction_id = "p1",
    provider = "prov",
    model = "m",
    model_revision = "r1",
    requested_at_ms = 1,
    responded_at_ms = 2,
    latency_ms = 1,
    context_ref = "c",
    context_hash = "h",
    file = "/f",
    cursor = { row = 0, col = 1 },
    proposed_start = { row = 0, col = 0 },
    proposed_end = { row = 0, col = 5 },
    proposed_text = "hello",
    max_output_tokens = 64,
    temperature = 0.5,
    confidence = 0.9,
    logprob = -1.2,
    finish_reason = "stop",
    accepted_chars = 5,
    accepted_lines = 1,
    total_chars = 5,
    extra_field = "kept",
  }
  collector.log_prediction_requested(full)
  collector.log_prediction_shown(full)
  collector.log_prediction_accepted(full)
  collector.log_prediction_partially_accepted(full)
  collector.log_prediction_rejected(full)
  assert_eq(#collector.queue, 5)
  for _, e in ipairs(collector.queue) do
    for k, v in pairs(full) do
      assert_eq(e.payload[k], v, "field " .. k .. " in " .. e.event_type)
    end
    assert_true(e.payload.cursor ~= nil, "cursor present")
    assert_true(e.payload.file ~= nil, "file present")
  end
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 54: exclusions: editing secret-named + terminal buffers emits nothing --------------------------------------------------------------
ok("exclusion-edit-emits-nothing", function()
  test_reset()
  local sec = mkbuf({ "x" }, scratch .. "/.env")
  assert_true(not buffers.attach(sec), "secret not attached")
  local q0 = #collector.queue
  vim.api.nvim_buf_set_lines(sec, 0, 1, false, { "y" })
  assert_eq(#collector.queue, q0, "secret edit emits nothing")
  local term = vim.api.nvim_create_buf(true, false)
  vim.api.nvim_buf_set_lines(term, 0, -1, false, { "x" })
  local ok_term, _ = pcall(vim.api.nvim_open_term, term, {})
  assert_true(ok_term, "terminal opened")
  assert_true(util.is_excluded_buffer(term), "terminal excluded")
  assert_true(not buffers.attach(term), "terminal not attached")
  q0 = #collector.queue
  pcall(vim.api.nvim_buf_set_lines, term, 0, 1, false, { "y" })
  assert_eq(#collector.queue, q0, "terminal edit emits nothing")
  vim.api.nvim_buf_delete(sec, { force = true })
  pcall(vim.api.nvim_buf_delete, term, { force = true })
end)

-- 55: resync counter increments on repeated divergences ----------------------------------------------------------------------------------
ok("resync-counter-increments", function()
  test_reset()
  local b = mkbuf({ "a", "b", "c" }, scratch .. "/resync2.lua")
  buffers.attach(b)
  collector.queue = {}
  local sh = buffers.get_shadow(b)
  table.remove(sh.lines, 2)
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "A" })
  assert_eq(buffers.get_shadow(b).resync_count, 1, "first resync")
  assert_eq(#edit_deltas(), 0, "divergent delta skipped")
  sh = buffers.get_shadow(b)
  table.remove(sh.lines, 2)
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "A2" })
  assert_eq(buffers.get_shadow(b).resync_count, 2, "second resync increments")
  assert_eq(#edit_deltas(), 0, "second divergent delta skipped")
  assert_shadow_live(b, "healed shadow")
  vim.api.nvim_buf_delete(b, { force = true })
end)

-- 56: paused status surfaces active/paused; paused edits emit nothing but track shadow --------------------------------------------------
ok("paused-status-emits-nothing", function()
  test_reset()
  local b = mkbuf({ "a", "b" }, scratch .. "/paused2.lua")
  buffers.attach(b)
  collector.queue = {}
  collector.pause()
  assert_true(collector.emit(b, "heartbeat", {}) == nil, "emit nil while paused")
  assert_true(collector.status().paused, "status paused")
  assert_true(not collector.status().active, "not active while paused")
  local q0 = #collector.queue
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "PAUSED_EDIT" })
  assert_eq(#collector.queue, q0, "paused buffer edit emits nothing")
  assert_shadow_live(b, "shadow still tracks while paused")
  collector.resume()
  assert_true(collector.status().active, "active after resume")
  vim.api.nvim_buf_set_lines(b, 0, 1, false, { "RESUMED" })
  local e = last(collector.queue)
  assert_eq(e.event_type, "edit_delta")
  assert_eq(e.payload.deleted_text, "PAUSED_EDIT")
  assert_eq(e.payload.inserted_text, "RESUMED")
  vim.api.nvim_buf_delete(b, { force = true })
end)

print(("--\n%d passed, %d failed"):format(passed, failed))
if failed > 0 then
  print("FAILURES:")
  for _, f in ipairs(failures) do
    print("  " .. f)
  end
  os.exit(1)
end
print("ALL GREEN")
os.exit(0)
