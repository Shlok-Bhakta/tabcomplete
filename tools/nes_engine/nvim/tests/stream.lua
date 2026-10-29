-- Headless test: streamed provisional ghost text is drawn at once, then
-- extended, replaced or cleared by the final answer; stale streams are dropped.
-- Run: NES_CLIENT=tools/nes_engine/nvim nvim --headless --clean -l tools/nes_engine/nvim/tests/stream.lua
vim.opt.runtimepath:prepend(vim.env.NES_CLIENT)
local here = vim.fs.dirname(debug.getinfo(1, "S").source:sub(2))
local dir = vim.fn.tempname()
vim.fn.mkdir(dir, "p")
local log = dir .. "/server.log"

local port
local server = vim.system({ "python3", here .. "/fake_stream_server.py", log }, {
  stdout = function(_, data)
    local p = data and data:match("PORT (%d+)")
    if p then port = tonumber(p) end
  end,
})
assert(vim.wait(5000, function() return port ~= nil end), "fake server did not start")

local nes = require("tabcomplete_nes").setup({ url = "http://127.0.0.1:" .. port, prefetch = false, normal_mode = false, stream = true })
local T = nes._test
local ns = vim.api.nvim_create_namespace("tabcomplete_nes")
vim.o.virtualedit = "onemore"

local function buffer(name)
  vim.cmd("enew!")
  local buf = vim.api.nvim_get_current_buf()
  vim.api.nvim_buf_set_name(buf, dir .. "/" .. name .. ".rs")
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "fn main() {", "    let x = foo(", "}" })
  vim.api.nvim_win_set_cursor(0, { 2, 16 })
  return buf
end

-- Inline ghost text drawn in the buffer, or nil.
local function ghost(buf)
  for _, m in ipairs(vim.api.nvim_buf_get_extmarks(buf, ns, 0, -1, { details = true })) do
    local vt = m[4].virt_text
    if vt and m[4].virt_text_pos == "inline" then return vt[1][1] end
  end
  return nil
end

local function marks(buf)
  return #vim.api.nvim_buf_get_extmarks(buf, ns, 0, -1, {})
end

local function wait(what, f, ms)
  assert(vim.wait(ms or 3000, f, 10), "timed out waiting for " .. what)
end

local function suggestion() return T.state.suggestion end

-- Provisional text grows in place and the final answer extends it.
local buf = buffer("extend")
local shown = T.state.stats.shown
T.request(buf, "manual")
wait("provisional", function() return ghost(buf) == "ba" end)
local s = suggestion()
assert(s and s.provisional and s.display == "ghost", vim.inspect(s))
local drawn = s.drawn_at
assert(T.state.stats.shown == shown, "provisional text is not announced as shown")
wait("growth", function() return ghost(buf) == "bar" end)
assert(suggestion() == s, "grown in place")
wait("final", function() return ghost(buf) == "bar);" end)
s = suggestion()
assert(s and not s.provisional and s.edits[1].text == "bar);", vim.inspect(s))
assert(s.drawn_at == drawn, "final keeps the time the text first appeared")
assert(T.state.stats.shown == shown + 1, "shown once")

-- An empty final answer clears the provisional text.
buf = buffer("retract")
T.request(buf, "manual")
wait("provisional", function() return ghost(buf) == "ba" end)
wait("retraction", function() return suggestion() == nil and marks(buf) == 0 end)

-- A different final edit replaces it.
buf = buffer("differs")
T.request(buf, "manual")
wait("provisional", function() return ghost(buf) == "ba" end)
wait("replacement", function() local x = suggestion(); return x and x.display == "replace" end)
assert(ghost(buf) == nil, "provisional ghost text removed")
assert(suggestion().drawn_at ~= drawn and suggestion().edits[1].text == "let y")

-- Dismissing the provisional text keeps the final answer from showing.
buf = buffer("dismiss")
T.request(buf, "manual")
wait("provisional", function() return ghost(buf) == "ba" end)
assert(nes.dismiss(), "dismissed")
assert(marks(buf) == 0)
vim.wait(600)
assert(suggestion() == nil and marks(buf) == 0, "final not shown after dismissal")

-- Accepting the provisional text inserts what was shown.
buf = buffer("accept")
T.request(buf, "manual")
wait("provisional", function() return ghost(buf) == "ba" end)
assert(nes.accept(), "accepted")
wait("applied", function() return vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1] == "    let x = foo(ba" end)
vim.wait(600)
assert(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1] == "    let x = foo(ba", "final not applied over it")

-- A new request cancels the stream of the previous one.
buf = buffer("cancel")
T.request(buf, "manual")
local first = T.state.job
vim.wait(100)
T.request(buf, "manual")
assert(T.state.job ~= first, "new job")
local stale = vim.wait(900, function() return marks(buf) > 0 or suggestion() ~= nil end, 5)
assert(not stale, "stale stream drawn: " .. vim.inspect(suggestion()))

-- An engine without streaming still works.
buf = buffer("plain")
T.request(buf, "manual")
wait("plain final", function() return ghost(buf) == "bar);" end)

local entries = vim.tbl_map(vim.json.decode, vim.fn.readfile(log))
local gone = vim.tbl_filter(function(e) return e.gone end, entries)
assert(#gone == 1 and gone[1].key == "cancel#1", "server saw the cancelled client hang up: " .. vim.inspect(entries))
assert(vim.iter(entries):all(function(e) return e.gone or e.stream end), "requests stream")

server:kill(15)
print("stream: ok")
vim.cmd("qa!")
