-- Reference adapter for the trained single-line-edit-v1 wire/context contract.
-- It is opt-in; the legacy compact-next-edit-v1 path remains the default.
local M = {}

M.WIRE_VERSION = "single-line-edit-v1"
M.CONTEXT_POLICY_VERSION = "single-line-context-v2"
M.MAX_ACTION_TOKENS = 64
M.DEFAULT_INPUT_TOKENS = 1024
M.MAX_TOTAL_TOKENS = 2048

local function is_boundary(text, column)
  if type(text) ~= "string" or type(column) ~= "number" or column < 0 or column > #text then
    return false
  end
  if column == 0 or column == #text then return true end
  local byte = text:byte(column + 1)
  return byte < 0x80 or byte >= 0xC0
end

local function physical_lines(source)
  if type(source) ~= "string" then error("source must be a string") end
  local lines, start, index = {}, 1, 1
  while index <= #source do
    local byte = source:byte(index)
    if byte == 13 then
      if source:byte(index + 1) ~= 10 then error("lone CR is outside the v1 line model") end
      lines[#lines + 1] = {
        content = source:sub(start, index - 1), terminator = "\r\n",
      }
      index = index + 2
      start = index
    elseif byte == 10 then
      lines[#lines + 1] = { content = source:sub(start, index - 1), terminator = "\n" }
      index = index + 1
      start = index
    else
      index = index + 1
    end
  end
  if start <= #source then
    lines[#lines + 1] = { content = source:sub(start), terminator = "" }
  end
  return lines
end

M.physical_lines = physical_lines

-- Describe the exact source-byte interval changed by an action. Neovim row/column
-- endpoints cannot represent a final newline after the last physical line, so
-- byte offsets remain authoritative for separators and EOF deletions.
function M.action_range(state, action)
  local lines = physical_lines(state.source)
  local row = state.target_row
  if type(row) ~= "number" or row % 1 ~= 0 or row < 0 or row > #lines then
    error("action target row is outside the file")
  end
  if action.kind == "keep" then return nil end
  local start_byte = 0
  for index = 1, row do
    local line = lines[index]
    if line then start_byte = start_byte + #line.content + #line.terminator end
  end
  local start_row, start_col = row, 0
  local end_row, end_col = row, 0
  local end_byte = start_byte
  local includes_terminator = false
  if action.kind == "insert_before" then
    end_col = 0
  elseif action.kind == "replace_line" then
    local line = lines[row + 1]
    if not line then error("replace requires an existing target line") end
    end_col = #line.content
    end_byte = start_byte + #line.content
  elseif action.kind == "delete_line" then
    local line = lines[row + 1]
    if not line then error("delete requires an existing target line") end
    end_byte = start_byte + #line.content + #line.terminator
    includes_terminator = line.terminator ~= ""
    if line.terminator ~= "" and row + 1 < #lines then
      end_row, end_col = row + 1, 0
    else
      end_col = #line.content
    end
  else
    error("unsupported single-line-edit action")
  end
  return {
    start_row = start_row,
    start_col = start_col,
    end_row = end_row,
    end_col = end_col,
    start_byte = start_byte,
    end_byte = end_byte,
    end_exclusive = true,
    includes_terminator = includes_terminator,
  }
end

local function predominant_terminator(lines)
  local lf, crlf, first = 0, 0, nil
  for _, line in ipairs(lines) do
    if line.terminator == "\n" then lf = lf + 1 end
    if line.terminator == "\r\n" then crlf = crlf + 1 end
    if first == nil and line.terminator ~= "" then first = line.terminator end
  end
  if lf == crlf then return first or "\n" end
  return crlf > lf and "\r\n" or "\n"
end

local function validate_state(state)
  if type(state) ~= "table" or type(state.file_id) ~= "string" or state.file_id == "" then
    error("file identity is required")
  end
  local supported = { python = true, typescript = true, rust = true, go = true }
  if not supported[state.filetype] then error("unsupported single-line-edit filetype") end
  local lines = physical_lines(state.source)
  if type(state.target_row) ~= "number" or state.target_row % 1 ~= 0
      or state.target_row < 0 or state.target_row > #lines then
    error("target row is outside the file")
  end
  if type(state.cursor_col) ~= "number" or state.cursor_col % 1 ~= 0 then
    error("cursor byte column must be an integer")
  end
  if state.target_row == #lines then
    if state.cursor_col ~= 0 then error("EOF insertion cursor column must be zero") end
  else
    local content = lines[state.target_row + 1].content
    if not is_boundary(content, state.cursor_col) then
      error("cursor byte column is outside a UTF-8 boundary")
    end
  end
  state.history = state.history or {}
  state.relevant = state.relevant or {}
  if type(state.history) ~= "table" or type(state.relevant) ~= "table" then
    error("history and relevant must be arrays")
  end
  for _, edit in ipairs(state.history) do
    if type(edit) ~= "table" or type(edit.row) ~= "number" or edit.row % 1 ~= 0
        or edit.row < 0 or type(edit.old_text) ~= "string" or type(edit.new_text) ~= "string" then
      error("invalid recent edit")
    end
  end
  for _, snippet in ipairs(state.relevant) do
    if type(snippet) ~= "string" then error("relevant context must contain strings") end
  end
  return lines
end

local function quote(text)
  return vim.json.encode(text)
end

local function render(state, lines, rows, history, relevant)
  local selected_rows = {}
  for row in pairs(rows) do selected_rows[#selected_rows + 1] = row end
  table.sort(selected_rows)
  local out = {
    "<single-line-edit-v1>",
    "Actions: N, D, R\t, I\t. R/I prefixes take one source line of text.",
    "The gap after R/I is a literal tab. No CR, LF, or explanation; end with tokenizer EOS.",
    "File: " .. quote(state.file_id),
    "Filetype: " .. state.filetype,
    "Target row (zero based): " .. tostring(state.target_row),
    "Cursor byte column: " .. tostring(state.cursor_col),
    "Physical line count: " .. tostring(#lines),
    "Current source lines (JSON strings; indices are original rows):",
  }
  for _, row in ipairs(selected_rows) do
    local line = lines[row + 1]
    local terminator = line.terminator == "\r\n" and "CRLF"
      or line.terminator == "\n" and "LF" or "EOF"
    out[#out + 1] = tostring(row) .. " [" .. terminator .. "] " .. quote(line.content)
  end
  if state.target_row == #lines then
    out[#out + 1] = tostring(state.target_row) .. " [APPEND AT EOF]"
  end
  out[#out + 1] = "Recent edits, oldest to newest (JSON old and new text):"
  for _, index in ipairs(history) do
    local edit = state.history[index]
    out[#out + 1] = tostring(index - 1) .. " row=" .. tostring(edit.row)
      .. " old=" .. quote(edit.old_text) .. " new=" .. quote(edit.new_text)
  end
  out[#out + 1] = "Relevant definitions and imports (JSON strings):"
  for _, index in ipairs(relevant) do
    out[#out + 1] = tostring(index - 1) .. " " .. quote(state.relevant[index])
  end
  out[#out + 1] = "</single-line-edit-v1>"
  out[#out + 1] = "Action:"
  return table.concat(out, "\n")
end

local function candidate_items(state, lines, rows)
  local items = {}
  local function row_item(row)
    if row >= 0 and row < #lines and not rows[row] then
      items[#items + 1] = { bucket = "rows", index = row }
    end
  end
  for distance = 1, 3 do
    row_item(state.target_row - distance)
    row_item(state.target_row + distance)
  end
  for index = #state.history, 1, -1 do
    items[#items + 1] = { bucket = "history", index = index }
  end
  for index = 1, #state.relevant do
    items[#items + 1] = { bucket = "relevant", index = index }
  end
  for distance = 4, 20 do
    row_item(state.target_row - distance)
    row_item(state.target_row + distance)
  end
  return items
end

function M.new_context(state)
  local lines = validate_state(state)
  local rows, history, relevant = {}, {}, {}
  if state.target_row < #lines then rows[state.target_row] = true end
  local items = candidate_items(state, lines, rows)
  local cursor = 1
  local pending
  local ctx = {}
  function ctx:prompt()
    return render(state, lines, rows, history, relevant)
  end
  function ctx:next_prompt()
    if pending then error("previous context candidate is unresolved") end
    local item = items[cursor]
    if not item then return nil end
    cursor = cursor + 1
    local bucket = item.bucket == "rows" and rows or item.bucket == "history" and history or relevant
    if item.bucket == "rows" then
      bucket[item.index] = true
    else
      bucket[#bucket + 1] = item.index
      table.sort(bucket)
    end
    pending = item
    return self:prompt()
  end
  function ctx:resolve(fits)
    if not pending then error("no context candidate is pending") end
    if not fits then
      local item = pending
      if item.bucket == "rows" then
        rows[item.index] = nil
      else
        local bucket = item.bucket == "history" and history or relevant
        for index = #bucket, 1, -1 do
          if bucket[index] == item.index then table.remove(bucket, index); break end
        end
      end
    end
    pending = nil
  end
  function ctx:next_item()
    return items[cursor]
  end
  function ctx:included()
    return { rows = rows, history = history, relevant = relevant }
  end
  return ctx
end

function M.render_full_context(state)
  local ctx = M.new_context(state)
  while true do
    local next_prompt = ctx:next_prompt()
    if not next_prompt then return ctx:prompt() end
    ctx:resolve(true)
  end
end

function M.serialize_bounded(state, token_count, max_input_tokens)
  if type(token_count) ~= "function" then error("token counter is required") end
  max_input_tokens = max_input_tokens or M.DEFAULT_INPUT_TOKENS
  if type(max_input_tokens) ~= "number" or max_input_tokens < 1
      or max_input_tokens > M.MAX_TOTAL_TOKENS then
    error("input token limit is outside the validated range")
  end
  local ctx = M.new_context(state)
  local current = ctx:prompt()
  local count = token_count(current)
  if count > max_input_tokens then error("mandatory target and control markers exceed input budget") end
  while true do
    local prompt = ctx:next_prompt()
    if not prompt then break end
    count = token_count(prompt)
    ctx:resolve(count <= max_input_tokens)
  end
  current = ctx:prompt()
  count = token_count(current)
  return { text = current, input_tokens = count, included = ctx:included() }
end

function M.encode_action(action)
  if type(action) ~= "table" then error("action is required") end
  if action.kind == "keep" then return "N" end
  if action.kind == "delete_line" then return "D" end
  if action.kind == "replace_line" or action.kind == "insert_before" then
    if type(action.text) ~= "string" or action.text:find("[\r\n]") then
      error("line action requires one physical line of text")
    end
    return (action.kind == "replace_line" and "R\t" or "I\t") .. action.text
  end
  error("unsupported single-line-edit action")
end

function M.decode_action(text, stop_type, generated_tokens)
  if stop_type ~= "eos" then return nil, "action did not end with tokenizer EOS" end
  if type(text) ~= "string" then return nil, "action is not text" end
  if generated_tokens ~= nil and (type(generated_tokens) ~= "number"
      or generated_tokens % 1 ~= 0 or generated_tokens < 1
      or generated_tokens > M.MAX_ACTION_TOKENS) then
    return nil, "action exceeded the 64-token ceiling"
  end
  if text == "N" then return { kind = "keep" } end
  if text == "D" then return { kind = "delete_line" } end
  local prefix, kind
  if text:sub(1, 2) == "R\t" then prefix, kind = "R\t", "replace_line" end
  if text:sub(1, 2) == "I\t" then prefix, kind = "I\t", "insert_before" end
  if not prefix then return nil, "malformed single-line-edit action" end
  local value = text:sub(#prefix + 1)
  if value:find("[\r\n]") then return nil, "action contains more than one physical line" end
  return { kind = kind, text = value }
end

function M.apply_action(state, action)
  local lines = physical_lines(state.source)
  local row = state.target_row
  if action.kind == "keep" then return state.source end
  if action.kind == "replace_line" or action.kind == "delete_line" then
    if row >= #lines then error("replace/delete require an existing target line") end
  elseif action.kind ~= "insert_before" then
    error("unsupported single-line-edit action")
  end
  if action.kind == "replace_line" then
    if type(action.text) ~= "string" or action.text:find("[\r\n]") then
      error("replace requires one physical line")
    end
    local line = lines[row + 1]
    local terminator = line.terminator
    if action.text == "" and terminator == "" then terminator = predominant_terminator(lines) end
    lines[row + 1] = { content = action.text, terminator = terminator }
  elseif action.kind == "delete_line" then
    table.remove(lines, row + 1)
  elseif action.kind == "insert_before" then
    if type(action.text) ~= "string" or action.text:find("[\r\n]") then
      error("insert requires one physical line")
    end
    if row < #lines then
      local style = lines[row + 1].terminator
      if style == "" then style = predominant_terminator(lines) end
      table.insert(lines, row + 1, { content = action.text, terminator = style })
    else
      if #lines > 0 and lines[#lines].terminator == "" then
        lines[#lines].terminator = predominant_terminator(lines)
      end
      local term = action.text == "" and predominant_terminator(lines) or ""
      lines[#lines + 1] = { content = action.text, terminator = term }
    end
  end
  local out = {}
  for _, line in ipairs(lines) do out[#out + 1] = line.content .. line.terminator end
  return table.concat(out)
end

function M.apply_to_buffer(bufnr, state, action)
  local expected = M.apply_action(state, action)
  local buffers = require("tabcomplete_trajectory.buffers")
  if buffers.canonical_bytes(bufnr) ~= state.source then
    return false, "buffer bytes no longer match the single-line-edit state"
  end
  local eol = vim.api.nvim_get_option_value("eol", { buf = bufnr })
  local fileformat = vim.api.nvim_get_option_value("fileformat", { buf = bufnr })
  local separator = fileformat == "dos" and "\r\n" or "\n"
  local empty_file_insert = state.source == "" and state.target_row == 0
    and action.kind == "insert_before"
  if expected ~= "" then
    local expected_eol = expected:sub(-#separator) == separator
    if expected_eol ~= eol and not (empty_file_insert and not expected_eol) then
      return false, "action would change the buffer's final newline state"
    end
    local without_crlf = expected:gsub("\r\n", "")
    if without_crlf:find("\r", 1, true) then
      return false, "action would create unsupported line endings"
    end
    local without_lf = expected:gsub("\r\n", "")
    if fileformat == "dos" and without_lf:find("\n", 1, true) then
      return false, "action would mix LF and CRLF line endings"
    elseif fileformat ~= "dos" and expected:find("\r\n", 1, true) then
      return false, "action would mix LF and CRLF line endings"
    end
  end
  local row = state.target_row
  if action.kind == "keep" then return true end
  local old_lines = vim.api.nvim_buf_get_lines(bufnr, 0, -1, false)
  local function mutate(target)
    if empty_file_insert then vim.api.nvim_set_option_value("eol", false, { buf = target }) end
    if action.kind == "replace_line" then
      vim.api.nvim_buf_set_lines(target, row, row + 1, false, { action.text })
    elseif action.kind == "delete_line" then
      vim.api.nvim_buf_set_lines(target, row, row + 1, false, {})
    elseif action.kind == "insert_before" then
      if empty_file_insert then
        vim.api.nvim_buf_set_lines(target, 0, -1, false, { action.text })
      else
        vim.api.nvim_buf_set_lines(target, row, row, false, { action.text })
      end
    end
  end
  local probe = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_buf_set_lines(probe, 0, -1, false, old_lines)
  vim.api.nvim_set_option_value("fileformat", fileformat, { buf = probe })
  vim.api.nvim_set_option_value("eol", eol, { buf = probe })
  mutate(probe)
  local probe_matches = buffers.canonical_bytes(probe) == expected
  vim.api.nvim_buf_delete(probe, { force = true })
  if not probe_matches then
    return false, "Neovim cannot represent the exact single-line-edit result"
  end
  if empty_file_insert then
    vim.api.nvim_set_option_value("eol", false, { buf = bufnr })
  end
  vim.api.nvim_set_option_value("undolevels",
    vim.api.nvim_get_option_value("undolevels", { buf = bufnr }), { buf = bufnr })
  mutate(bufnr)
  local actual = buffers.canonical_bytes(bufnr)
  if actual ~= expected then
    pcall(vim.api.nvim_buf_set_lines, bufnr, 0, -1, false, old_lines)
    pcall(vim.api.nvim_set_option_value, "eol", eol, { buf = bufnr })
    return false, "Neovim byte result differs from the single-line-edit contract"
  end
  return true
end

return M
