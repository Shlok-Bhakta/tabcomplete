-- Headless test: prediction telemetry through the trajectory collector.
-- Checks the recorded fields (withheld edits, latency, engine identity,
-- request context) and that every shown suggestion ends in exactly one
-- terminal outcome. With COLLECTOR_URL set, events go to that collector
-- server instead of a mocked transport (see collector_e2e.sh).
-- Run: NES_CLIENT=tools/nes_engine/nvim COLLECTOR_CLIENT=tools/trajectory_collector/nvim \
--   nvim --headless --clean -l tools/nes_engine/nvim/tests/telemetry.lua
vim.opt.runtimepath:prepend(vim.env.NES_CLIENT)
vim.opt.runtimepath:prepend(vim.env.COLLECTOR_CLIENT)
local here = vim.fs.dirname(debug.getinfo(1, "S").source:sub(2))
local dir = vim.fn.tempname()
vim.fn.mkdir(dir, "p")
vim.fn.setenv("XDG_STATE_HOME", dir .. "/state")
local E2E = vim.env.COLLECTOR_URL

local port
local server = vim.system({ "python3", here .. "/fake_stream_server.py", dir .. "/server.log" }, {
  stdout = function(_, data)
    local p = data and data:match("PORT (%d+)")
    if p then port = tonumber(p) end
  end,
})
assert(vim.wait(5000, function() return port ~= nil end), "fake server did not start")

local collector = require("tabcomplete_trajectory")
local events = require("tabcomplete_trajectory.events")
local transport = require("tabcomplete_trajectory.transport")
local spool = require("tabcomplete_trajectory.spool")
local posts = {}
if not E2E then
  -- Mocked collector server: ACK everything, report every blob missing.
  transport._post_impl = function(url, body)
    posts[#posts + 1] = { url = url, body = body }
    if url:match("/v1/blobs/check$") then return true, { http_code = 200, missing = body.hashes } end
    return true, { http_code = 200 }
  end
end
collector.setup({ server_url = E2E or "http://collector.invalid", capture_keys = false, capture_repo_context = false })
assert(collector.started, "collector started")

-- Every event the collector accepts into its queue, in order.
local log = {}
local emit = collector.emit
collector.emit = function(bufnr, kind, payload)
  local ev = emit(bufnr, kind, payload)
  if ev then log[#log + 1] = ev end
  return ev
end

local nes = require("tabcomplete_nes").setup({
  url = "http://127.0.0.1:" .. port, prefetch = false, normal_mode = false, stream = true,
})
local T = nes._test
vim.o.virtualedit = "onemore"
local function wait(what, f, ms) assert(vim.wait(ms or 3000, f, 10), "timed out waiting for " .. what) end
wait("engine health", function() return T.state.health ~= nil end)

local function buffer(name, lines)
  vim.cmd("enew!")
  local buf = vim.api.nvim_get_current_buf()
  vim.api.nvim_buf_set_name(buf, dir .. "/" .. name .. ".rs")
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines or { "fn main() {", "    let x = foo(", "}" })
  vim.api.nvim_win_set_cursor(0, { 2, 16 })
  return buf
end

local function find(kind, pred)
  for i = #log, 1, -1 do
    local ev = log[i]
    if ev.event_type == kind and (not pred or pred(ev.payload)) then return ev.payload end
  end
  return nil
end
local function by_id(id) return function(p) return p.prediction_id == id end end
local function shown_now()
  local s = T.state.suggestion
  return s and not s.provisional and s.shown_event_id and s or nil
end

-- 1. Requested/generated/shown fields, then an accept. -----------------------
local buf = buffer("t_accept")
vim.diagnostic.set(vim.api.nvim_create_namespace("telemetry_test"), buf,
  { { lnum = 1, col = 12, severity = vim.diagnostic.severity.ERROR, message = "cannot find `foo`" } })
T.state.yank = { text = "yanked text", at = T.now_ms() - 1000 }
T.schedule_request(buf, "normal", 20)
wait("shown", shown_now)
local s = shown_now()
local req = find("prediction_requested", by_id(s.prediction_id))
assert(req.trigger == "normal" and req.debounce_ms == 20, vim.inspect(req))
assert(req.trigger_at_ms <= req.requested_at_ms and req.requested_at_ms - req.trigger_at_ms >= 0, vim.inspect(req))
local content = table.concat(vim.api.nvim_buf_get_lines(buf, 0, -1, false), "\n") .. "\n"
assert(req.content_sha256 == vim.fn.sha256(content) and req.content_bytes == #content, "content hash")
assert(type(req.filetype) == "string" and req.line_count == 3 and req.mode == "n", vim.inspect(req))
assert(req.diagnostics_count == 1 and req.diagnostics_severity.error == 1 and #req.diagnostics_hash == 64, vim.inspect(req))
assert(req.clipboard_present and req.clipboard_chars == 11 and req.clipboard_age_ms >= 1000, vim.inspect(req))
assert(#req.clipboard_hash == 64 and req.clipboard_hash == vim.fn.sha256("yanked text"), "clipboard hash")
assert(req.engine and req.engine.settings_hash == "fedcba9876543210", "engine on request")
assert(vim.deep_equal(req.engine_args, { "--model", "m.gguf", "serve" }), "engine args on first request")
local gen = find("prediction_generated", by_id(s.prediction_id))
assert(gen.shown == true and gen.not_shown_reason == nil, vim.inspect(gen))
assert(gen.engine.build_id == "0123456789abcdef" and gen.gate_threshold == -1.3, vim.inspect(gen.engine))
local lat = gen.latency
assert(lat.total_ms == 210 and lat.prefill_ms == 80.3 and lat.decode_ms == 120.5, vim.inspect(lat))
assert(lat.prompt_tokens == 700 and lat.cached_tokens == 650 and lat.evaluated_prompt_tokens == 50, vim.inspect(lat))
assert(lat.generated_tokens == 20 and lat.stages == 1 and lat.first_edit_ms == 150, vim.inspect(lat))
assert(gen.client_latency.request_ms >= 0 and gen.client_latency.trigger_to_request_ms >= 0, vim.inspect(gen.client_latency))
local st = gen.engine_timings.stages[1]
assert(#st.batches == 32 and st.batches_dropped == 8, "per-batch log capped")
assert(#gen.context_blob_hash == 64, "prompt blob")
local shown = find("prediction_shown", by_id(s.prediction_id))
assert(shown.keystroke_to_display_ms >= shown.request_to_display_ms and shown.request_to_display_ms >= 0, vim.inspect(shown))
assert(shown.trigger_at_ms == req.trigger_at_ms, "shown keeps the trigger time")
assert(nes.accept(), "accepted")
wait("accepted event", function() return find("prediction_accepted", by_id(s.prediction_id)) end)
local acc = find("prediction_accepted", by_id(s.prediction_id))
assert(acc.outcome == "accepted" and acc.accepted_chars == 5 and acc.accepted_lines == 1, vim.inspect(acc))
assert(acc.shown_event_id == s.shown_event_id and acc.visible_ms >= 0, vim.inspect(acc))
vim.diagnostic.reset(nil, buf)
T.state.yank = nil
vim.wait(300) -- the after-accept request

-- 2. Explicit rejection; a second dismissal adds nothing. --------------------
buf = buffer("t_reject")
T.request(buf, "manual")
wait("shown", shown_now)
s = shown_now()
local second = find("prediction_requested", by_id(s.prediction_id))
assert(second.engine_args == nil, "engine args recorded once per settings hash")
assert(second.diagnostics_count == 0 and second.clipboard_present == false, vim.inspect(second))
assert(nes.dismiss(), "dismissed")
local rej = find("prediction_rejected", by_id(s.prediction_id))
assert(rej.outcome == "rejected_explicit" and rej.reason == "explicit", vim.inspect(rej))
T.dismiss("cursor_moved")

-- 3. Cursor moved. ------------------------------------------------------------
buf = buffer("t_cursor")
T.request(buf, "manual")
wait("shown", shown_now)
s = shown_now()
T.dismiss("cursor_moved")
local d = find("prediction_dismissed", by_id(s.prediction_id))
assert(d.outcome == "dismissed" and d.reason == "cursor_moved" and d.outcome_source == "editor_observation", vim.inspect(d))

-- 4. Leaving the buffer (the next scenario's :enew). --------------------------
buf = buffer("t_leave")
T.request(buf, "manual")
wait("shown", shown_now)
local left = shown_now()

-- 5. Typed through to the end: a partial acceptance. --------------------------
buf = buffer("t_typed")
d = find("prediction_dismissed", by_id(left.prediction_id))
assert(d and d.outcome == "buffer_left", vim.inspect(d))
T.request(buf, "manual")
wait("shown", shown_now)
s = shown_now()
vim.api.nvim_buf_set_text(buf, 1, 16, 1, 16, { "bar);" })
vim.api.nvim_win_set_cursor(0, { 2, 21 })
assert(T.try_type_through(buf), "typed through")
local pa = find("prediction_partially_accepted", by_id(s.prediction_id))
assert(pa.outcome == "partially_accepted" and pa.reason == "typed_through" and pa.accepted_chars == 5, vim.inspect(pa))
assert(pa.remaining_chars == 0, vim.inspect(pa))

-- 6. Typed part of it, then diverged. -----------------------------------------
buf = buffer("t_diverge")
T.request(buf, "manual")
wait("shown", shown_now)
s = shown_now()
vim.api.nvim_buf_set_text(buf, 1, 16, 1, 16, { "ba" })
vim.api.nvim_win_set_cursor(0, { 2, 18 })
assert(T.try_type_through(buf), "kept while typing toward it")
vim.api.nvim_buf_set_text(buf, 1, 18, 1, 18, { "z" })
vim.api.nvim_win_set_cursor(0, { 2, 19 })
assert(not T.try_type_through(buf), "divergent")
T.dismiss("typed_over")
d = find("prediction_dismissed", by_id(s.prediction_id))
assert(d.reason == "typed_over" and d.typed_through == 1 and d.total_chars == 5 and d.remaining_chars == 3, vim.inspect(d))

-- 7. A newer suggestion supersedes the shown one. -----------------------------
buf = buffer("t_super")
T.request(buf, "manual")
wait("shown", shown_now)
local old = shown_now()
T.request(buf, "manual")
wait("new suggestion", function() local x = shown_now(); return x and x ~= old end)
d = find("prediction_dismissed", by_id(old.prediction_id))
assert(d.outcome == "superseded", vim.inspect(d))
T.dismiss("cursor_moved")

-- 8. A withheld edit, with what it would have replaced. -----------------------
buf = buffer("t_withheld")
T.request(buf, "manual")
wait("withheld answer", function()
  return find("prediction_generated", function(p) return p.withheld_reason == "low_confidence" end)
end)
gen = find("prediction_generated", function(p) return p.withheld_reason == "low_confidence" end)
assert(gen.withheld == 1 and gen.shown == false and gen.not_shown_reason == "empty", vim.inspect(gen))
local w = gen.withheld_edits[1]
assert(w.text == "bar(" and w.replaced_text == "foo(" and w.start_byte == 24, vim.inspect(w))
assert(gen.gate_threshold == -1.3 and gen.confidence.min_logprob == -0.5, vim.inspect(gen))

-- 9. A request cancelled by a newer one is closed as cancelled. ---------------
buf = buffer("t_slow")
T.request(buf, "manual")
local first_id = log[#log].payload.prediction_id
vim.wait(100)
T.request(buf, "manual")
wait("cancel record", function() return find("prediction_generated", by_id(first_id)) end)
gen = find("prediction_generated", by_id(first_id))
assert(gen.status == "cancelled" and gen.not_shown_reason == "superseded" and gen.superseded, vim.inspect(gen))
wait("slow answer shown", shown_now)
T.dismiss("cursor_moved")

-- 10. Jump to a distant edit, then accept: the jump is on the terminal event.
local lines = {}
for i = 1, 40 do lines[i] = "line " .. i end
buf = buffer("t_jump", lines)
vim.api.nvim_win_set_cursor(0, { 1, 0 })
T.show_for_test(buf, {
  bufnr = buf, changedtick = vim.api.nvim_buf_get_changedtick(buf), idx = 1, kind = "jump",
  prediction_id = "jump-pred", request_id = "jump-req", typed_through = 0,
  edits = { { start_line = 30, start_col = 0, end_line = 30, end_col = 4, text = "LINE",
    start_byte = 0, end_byte = 4 } },
})
assert(T.state.suggestion.far, "far edit")
assert(nes.tab(), "jump")
vim.wait(50)
assert(nes.tab(), "accept after jump")
wait("accepted after jump", function() return find("prediction_accepted", by_id("jump-pred")) end)
acc = find("prediction_accepted", by_id("jump-pred"))
assert(acc.jumped == true and acc.jumped_at_ms, vim.inspect(acc))

-- 11. Provisional text accepted: announced, then accepted. --------------------
buf = buffer("t_stream")
T.request(buf, "manual")
wait("provisional", function() local x = T.state.suggestion; return x and x.provisional end)
s = T.state.suggestion
assert(nes.accept(), "accepted provisional")
wait("accepted", function() return find("prediction_accepted", by_id(s.prediction_id)) end)
shown = find("prediction_shown", by_id(s.prediction_id))
assert(shown.provisional == true and shown.keystroke_to_display_ms ~= nil, vim.inspect(shown))
vim.wait(400)

-- 12. Accept races a buffer change: closed as accept_stale. -------------------
buf = buffer("t_stale")
T.request(buf, "manual")
wait("shown", shown_now)
s = shown_now()
assert(nes.accept(), "accept started")
vim.api.nvim_buf_set_lines(buf, 0, 1, false, { "fn other() {" })
wait("stale accept closed", function() return find("prediction_dismissed", by_id(s.prediction_id)) end)
assert(find("prediction_dismissed", by_id(s.prediction_id)).reason == "accept_stale")
assert(not find("prediction_accepted", by_id(s.prediction_id)), "not accepted")

-- 13. Payload bounds. ----------------------------------------------------------
local big = T.withheld_edits({ { text = string.rep("x", 5000), start_byte = 0, end_byte = 3000,
  start_line = 0, start_col = 0, end_line = 0, end_col = 0 } }, string.rep("y", 4000))
assert(#big[1].text == 2000 and big[1].text_truncated and #big[1].replaced_text == 2000 and big[1].replaced_truncated)
assert(big[1].text_chars == 5000)
local p = T.bounded({ keep = 1, context = string.rep("c", 40000), engine_timings = "t" }, { "engine_timings", "context" })
assert(p.keep == 1 and p.context == nil and p.engine_timings == nil and vim.deep_equal(p.dropped_fields, { "engine_timings", "context" }), vim.inspect(p))
p = T.bounded({ keep = 1, context = "small" }, { "context" })
assert(p.context == "small" and p.dropped_fields == nil)

-- 14. Editor exit closes the visible suggestion before the session ends. -----
buf = buffer("t_exit")
T.request(buf, "manual")
wait("shown", shown_now)
s = shown_now()
vim.wait(200)
collector.shutdown()
d = find("prediction_dismissed", by_id(s.prediction_id))
assert(d and d.outcome == "editor_exit", "editor exit recorded")

-- Invariants over everything recorded. ----------------------------------------
local terminal = { prediction_accepted = true, prediction_partially_accepted = true,
  prediction_rejected = true, prediction_dismissed = true }
local shown_ids, closes, requested, generated = {}, {}, {}, {}
for i, ev in ipairs(log) do
  assert(events.is_valid_type(ev.event_type), "invalid type " .. ev.event_type)
  if i > 1 then assert(ev.sequence_number == log[i - 1].sequence_number + 1, "sequence gap at " .. i) end
  local ok_enc, json = pcall(vim.json.encode, ev)
  assert(ok_enc and #json < 32768, "event encodes within the budget: " .. ev.event_type)
  local id = type(ev.payload) == "table" and ev.payload.prediction_id
  if ev.event_type == "prediction_shown" then
    assert(not shown_ids[id], "shown once: " .. id)
    shown_ids[id] = i
  elseif terminal[ev.event_type] then
    assert(shown_ids[id] and shown_ids[id] < i, "terminal after shown: " .. id)
    closes[id] = (closes[id] or 0) + 1
    assert(ev.payload.outcome and ev.payload.reason, "outcome and reason: " .. vim.inspect(ev.payload))
  elseif ev.event_type == "prediction_requested" then
    requested[id] = true
  elseif ev.event_type == "prediction_generated" then
    assert(not generated[id], "generated once: " .. id)
    generated[id] = true
  end
end
local n_shown = 0
for id in pairs(shown_ids) do
  n_shown = n_shown + 1
  assert(closes[id] == 1, "exactly one terminal for " .. id .. ": " .. tostring(closes[id]))
end
for id in pairs(requested) do assert(generated[id], "every request closed: " .. id) end
assert(n_shown >= 12, "shown " .. n_shown)

if E2E then
  -- Everything was delivered: nothing spooled or quarantined.
  local function drained()
    return #spool.list() == 0 and #vim.fn.globpath(spool.spool_dir() .. "/rejected", "*", false, true) == 0
  end
  assert(drained(), "spool: " .. vim.inspect(spool.list()))
  local counts = {}
  for _, ev in ipairs(log) do counts[ev.event_type] = (counts[ev.event_type] or 0) + 1 end
  counts.session_end = 1
  local f = assert(io.open(vim.env.E2E_OUT, "w"))
  f:write(vim.json.encode({ session_id = collector.session_id, counts = counts, total_with_queue_start = log[#log].sequence_number + 1 }))
  f:close()
else
  local blob_uploads = 0
  for _, post in ipairs(posts) do
    if post.url:match("/v1/blobs/upload$") and post.body.sha256 == req.diagnostics_hash then blob_uploads = blob_uploads + 1 end
  end
  assert(blob_uploads == 1, "diagnostics blob uploaded once")
end

server:kill(15)
print("telemetry: ok (" .. #log .. " events, " .. n_shown .. " shown)")
vim.cmd("qa!")
