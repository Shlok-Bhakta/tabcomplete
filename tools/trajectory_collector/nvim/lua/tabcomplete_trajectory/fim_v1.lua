-- Detached Qwen PSM completion adapter. It is not selected by the legacy
-- predictor or by any model alias until an artifact identity is frozen.
local M = {}

M.WIRE_VERSION = "q25-fim-line-completion-v1"
M.CONTEXT_POLICY_VERSION = "q25-fim-psm-cursor-to-line-end-v1"
M.CONTEXT_LAYOUT = "q25-fim-psm-v1"
M.MAX_OUTPUT_TOKENS = 96

M.EOS_TOKEN_ID = 151643
M.FIM_PREFIX_TOKEN_ID = 151659
M.FIM_MIDDLE_TOKEN_ID = 151660
M.FIM_SUFFIX_TOKEN_ID = 151661

M.EOS = "<|endoftext|>"
M.FIM_PREFIX = "<|fim_prefix|>"
M.FIM_SUFFIX = "<|fim_suffix|>"
M.FIM_MIDDLE = "<|fim_middle|>"

local single_line_v1 = require("tabcomplete_trajectory.single_line_v1")

local function integer(value, minimum)
  return type(value) == "number" and value % 1 == 0 and value >= minimum
end

local function is_sha256(value)
  return type(value) == "string" and #value == 64 and value:match("^%x+$") ~= nil
end

local function valid_utf8(text)
  local index = 1
  while index <= #text do
    local first = text:byte(index)
    if first <= 0x7f then
      index = index + 1
    elseif first >= 0xc2 and first <= 0xdf then
      local second = text:byte(index + 1)
      if not second or second < 0x80 or second > 0xbf then return false end
      index = index + 2
    elseif first >= 0xe0 and first <= 0xef then
      local second, third = text:byte(index + 1), text:byte(index + 2)
      if not second or not third or second < 0x80 or second > 0xbf
          or third < 0x80 or third > 0xbf then return false end
      if (first == 0xe0 and second < 0xa0) or (first == 0xed and second > 0x9f) then return false end
      index = index + 3
    elseif first >= 0xf0 and first <= 0xf4 then
      local second, third, fourth = text:byte(index + 1), text:byte(index + 2), text:byte(index + 3)
      if not second or not third or not fourth or second < 0x80 or second > 0xbf
          or third < 0x80 or third > 0xbf or fourth < 0x80 or fourth > 0xbf then return false end
      if (first == 0xf0 and second < 0x90) or (first == 0xf4 and second > 0x8f) then return false end
      index = index + 4
    else
      return false
    end
  end
  return true
end

local function is_boundary(text, column)
  if column == 0 or column == #text then return true end
  local byte = text:byte(column + 1)
  return byte < 0x80 or byte >= 0xc0
end

local function validate_token_contract(contract)
  if type(contract) ~= "table" or contract.eos_id ~= M.EOS_TOKEN_ID
      or contract.fim_prefix_id ~= M.FIM_PREFIX_TOKEN_ID
      or contract.fim_suffix_id ~= M.FIM_SUFFIX_TOKEN_ID
      or contract.fim_middle_id ~= M.FIM_MIDDLE_TOKEN_ID
      or not is_sha256(contract.tokenizer_sha256) then
    return nil, "FIM tokenizer identity mismatch"
  end
  if type(contract.special_tokens) ~= "table" then
    return nil, "missing complete tokenizer special-token inventory"
  end
  local by_id, by_spelling = {}, {}
  local count = 0
  for key, token in pairs(contract.special_tokens) do
    if not integer(key, 1) or type(token) ~= "table"
        or not integer(token.id, 0) or type(token.spelling) ~= "string"
        or token.spelling == "" or not valid_utf8(token.spelling) then
      return nil, "invalid tokenizer special-token inventory"
    end
    if by_id[token.id] or by_spelling[token.spelling] then
      return nil, "duplicate tokenizer special token"
    end
    by_id[token.id] = token.spelling
    by_spelling[token.spelling] = token.id
    count = count + 1
  end
  for index = 1, count do
    if type(contract.special_tokens[index]) ~= "table" then
      return nil, "sparse special-token inventory"
    end
  end
  for id, spelling in pairs({
    [M.EOS_TOKEN_ID] = M.EOS,
    [M.FIM_PREFIX_TOKEN_ID] = M.FIM_PREFIX,
    [M.FIM_SUFFIX_TOKEN_ID] = M.FIM_SUFFIX,
    [M.FIM_MIDDLE_TOKEN_ID] = M.FIM_MIDDLE,
  }) do
    if by_id[id] ~= spelling or by_spelling[spelling] ~= id then
      return nil, "required FIM/EOS token missing from special-token inventory"
    end
  end
  return { by_id = by_id, by_spelling = by_spelling }
end

function M.new_token_contract(config)
  local inventory = validate_token_contract(config)
  if not inventory then error("invalid FIM tokenizer contract") end
  return {
    eos_id = config.eos_id,
    fim_prefix_id = config.fim_prefix_id,
    fim_suffix_id = config.fim_suffix_id,
    fim_middle_id = config.fim_middle_id,
    tokenizer_sha256 = config.tokenizer_sha256:lower(),
    special_tokens = vim.deepcopy(config.special_tokens),
    _by_id = inventory.by_id,
    _by_spelling = inventory.by_spelling,
  }
end

local function reject_special_spelling(text, contract, label)
  for spelling in pairs(contract.by_spelling) do
    if text:find(spelling, 1, true) then
      return nil, label .. " contains a tokenizer special-token spelling"
    end
  end
  return true
end

-- Build exact PSM context. `model_hole_range` includes the physical line
-- ending; `apply_range` is the whole line content interval used by the existing
-- canonical buffer applier.
function M.prepare(source, target_row, cursor_col, contract)
  if type(source) ~= "string" or #source > 1024 * 1024 or source:find("\0", 1, true)
      or not valid_utf8(source) then
    return nil, "source outside the FIM byte contract"
  end
  if not integer(target_row, 0) or not integer(cursor_col, 0) then
    return nil, "invalid FIM cursor"
  end
  local valid_contract, contract_err = validate_token_contract(contract)
  if not valid_contract then return nil, contract_err end
  local special_ok, special_err = reject_special_spelling(source, valid_contract, "source")
  if not special_ok then return nil, special_err end

  local ok, lines = pcall(single_line_v1.physical_lines, source)
  if not ok then return nil, "source has an unsupported physical line ending" end
  local virtual_empty_file = source == ""
  local line_start, content_end, line_end, content, terminator
  if virtual_empty_file then
    if target_row ~= 0 or cursor_col ~= 0 then return nil, "target outside empty file" end
    line_start, content_end, line_end, content, terminator = 0, 0, 0, "", ""
  else
    local line = lines[target_row + 1]
    if not line then return nil, "target row outside source" end
    local offset = 0
    for index = 1, target_row do
      offset = offset + #lines[index].content + #lines[index].terminator
    end
    line_start = offset
    content = line.content
    if cursor_col > #content or not is_boundary(content, cursor_col) then
      return nil, "cursor is outside a UTF-8 boundary"
    end
    content_end = line_start + #content
    terminator = line.terminator
    line_end = content_end + #terminator
  end
  local cursor_byte = line_start + cursor_col
  local prompt = M.FIM_PREFIX .. source:sub(1, cursor_byte)
    .. M.FIM_SUFFIX .. source:sub(line_end + 1) .. M.FIM_MIDDLE
  local ending_name = terminator == "\n" and "LF"
    or (terminator == "\r\n" and "CRLF" or "EOF")
  return {
    prompt = prompt,
    model_protocol = M.WIRE_VERSION,
    context_policy_version = M.CONTEXT_POLICY_VERSION,
    context_layout = M.CONTEXT_LAYOUT,
    tokenizer_sha256 = contract.tokenizer_sha256,
    source = source,
    target_row = target_row,
    cursor_col = cursor_col,
    prefix_before_cursor = content:sub(1, cursor_col),
    line_ending = ending_name,
    line_ending_bytes = terminator,
    model_hole_range = { start_byte = cursor_byte, end_byte = line_end, end_exclusive = true },
    apply_range = { start_byte = line_start, end_byte = content_end, end_exclusive = true },
    virtual_empty_file = virtual_empty_file,
  }
end

local function valid_sampled_ids(ids, predicted, contract)
  if type(ids) ~= "table" or not integer(predicted, 0) then return nil end
  local count, max_index = 0, 0
  for index, id in pairs(ids) do
    if not integer(index, 1) or not integer(id, 0)
        or contract.by_id[id] then return nil end
    count = count + 1
    max_index = math.max(max_index, index)
  end
  if count ~= max_index or count ~= predicted then return nil end
  return true
end

-- Decode plain model text. The FIM completion is never interpreted as the
-- legacy N/R wire; one matching final line ending is removed only to create the
-- canonical whole-line action consumed by single_line_v1.apply_to_buffer.
function M.decode_completion(prepared, response, contract, expected_identity)
  if type(prepared) ~= "table" or prepared.model_protocol ~= M.WIRE_VERSION
      or type(response) ~= "table" or type(expected_identity) ~= "table" then
    return nil, "missing FIM protocol state"
  end
  local valid_contract, contract_err = validate_token_contract(contract)
  if not valid_contract then return nil, contract_err end
  if expected_identity.model_protocol ~= M.WIRE_VERSION
      or expected_identity.output_tokens ~= M.MAX_OUTPUT_TOKENS
      or not is_sha256(expected_identity.model_sha256)
      or not is_sha256(expected_identity.tokenizer_sha256)
      or expected_identity.tokenizer_sha256:lower() ~= contract.tokenizer_sha256
      or prepared.tokenizer_sha256 ~= contract.tokenizer_sha256 then
    return nil, "FIM model identity is not frozen"
  end
  if response.model_protocol ~= expected_identity.model_protocol
      or response.model_sha256 ~= expected_identity.model_sha256 then
    return nil, "FIM response model identity mismatch"
  end
  if response.stop_type ~= "eos" or response.terminal_token_id ~= M.EOS_TOKEN_ID then
    return nil, "FIM response did not terminate with the exact EOS token"
  end
  if not valid_sampled_ids(response.sampled_token_ids, response.tokens_predicted, valid_contract) then
    return nil, "FIM sampled token inventory is invalid"
  end
  local generated_steps = response.tokens_predicted + 1
  if generated_steps > M.MAX_OUTPUT_TOKENS then
    return nil, "FIM response exceeds the EOS-inclusive output cap"
  end
  local raw = response.raw_text
  if type(raw) ~= "string" or raw:find("\0", 1, true) or not valid_utf8(raw) then
    return nil, "FIM response text is invalid"
  end
  local special_ok, special_err = reject_special_spelling(raw, valid_contract, "completion")
  if not special_ok then return nil, special_err end

  local ending = prepared.line_ending_bytes
  local body
  if ending == "" then
    if raw:find("[\r\n]") then return nil, "EOF completion contains a line ending" end
    body = raw
  else
    if raw:sub(-#ending) ~= ending then return nil, "completion line ending mismatch" end
    body = raw:sub(1, #raw - #ending)
    if body:find("[\r\n]") then
      return nil, "completion contains more than its one declared line ending"
    end
  end
  local action = {
    kind = prepared.virtual_empty_file and "insert_before" or "replace_line",
    text = prepared.prefix_before_cursor .. body,
  }
  return {
    wire_version = M.WIRE_VERSION,
    raw_text = raw,
    stripped_line_ending = ending,
    action = action,
    model_hole_range = vim.deepcopy(prepared.model_hole_range),
    apply_range = vim.deepcopy(prepared.apply_range),
    generated_steps = generated_steps,
  }
end

-- Reuse the existing byte-exact scratch-buffer check and canonical mutation.
function M.apply_to_buffer(bufnr, prepared, decoded)
  if type(decoded) ~= "table" or decoded.wire_version ~= M.WIRE_VERSION
      or type(decoded.action) ~= "table" then
    return false, "invalid FIM decoded action"
  end
  local state = {
    source = prepared.source,
    target_row = prepared.target_row,
    cursor_col = prepared.cursor_col,
  }
  return single_line_v1.apply_to_buffer(bufnr, state, decoded.action)
end

return M
