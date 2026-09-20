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
