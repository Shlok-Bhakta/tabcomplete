-- Headless test: typing toward a single-line suggestion keeps it as ghost text.
-- Run: NES_CLIENT=tools/nes_engine/nvim nvim --headless --clean -l tools/nes_engine/nvim/tests/type_through.lua
vim.opt.runtimepath:prepend(vim.env.NES_CLIENT)
local nes = require("tabcomplete_nes").setup({ url = "http://127.0.0.1:1", prefetch = false })
local T = nes._test
vim.o.virtualedit = "onemore"
vim.cmd("enew"); local buf = vim.api.nvim_get_current_buf()
vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "fn main() {", "    let x = ", "}" })
vim.api.nvim_win_set_cursor(0, { 2, 12 })
-- Pretend the engine proposed a whole-line rewrite of line 2.
local s = { bufnr = buf, changedtick = vim.api.nvim_buf_get_changedtick(buf), idx = 1,
  edits = { { start_line = 1, start_col = 0, end_line = 2, end_col = 0, text = "    let x = compute(42);\n" } } }
T.state.suggestion = nil
T.show_for_test(buf, s)
assert(T.state.suggestion and T.state.suggestion.target == "    let x = compute(42);", "target recorded")
-- User types "co" at the cursor.
vim.api.nvim_buf_set_text(buf, 1, 12, 1, 12, { "co" })
vim.api.nvim_win_set_cursor(0, { 2, 14 })
assert(T.try_type_through(buf), "suggestion kept")
local e = T.state.suggestion.edits[1]
assert(e.text == "mpute(42);" and e.start_col == 14, vim.inspect(e))
-- A divergent keystroke drops it.
vim.api.nvim_buf_set_text(buf, 1, 14, 1, 14, { "z" })
vim.api.nvim_win_set_cursor(0, { 2, 15 })
assert(not T.try_type_through(buf), "divergent typing not kept")
print("type-through: ok")
vim.cmd("qa!")
