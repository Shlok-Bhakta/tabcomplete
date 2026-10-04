-- Incremental llama.cpp completion SSE decoder. Bytes remain untouched until JSON decoding.
local M = {}

function M.new(max_bytes)
  return { buffer = "", pieces = {}, raw_chunks = {}, bytes = 0, max_bytes = max_bytes or 262144,
    terminal = nil, error = nil, first_token = false, first_text = false,
    first_token_source = nil, model_tokens = 0, has_token_ids = false,
    sampled_token_ids = {}, events = 0 }
end

function M.feed(parser, chunk)
  if parser.error or parser.terminal then return false, parser.error or "data after terminal event" end
  if type(chunk) ~= "string" then return false, "invalid SSE chunk" end
  parser.bytes = parser.bytes + #chunk
  if parser.bytes > parser.max_bytes then parser.error = "SSE response too large"; return false, parser.error end
  parser.raw_chunks[#parser.raw_chunks + 1] = chunk
  parser.buffer = parser.buffer .. chunk
  while true do
    local start, finish = parser.buffer:find("\r?\n\r?\n")
    if not start then break end
    if parser.terminal then parser.error = "SSE data after terminal event"; return false, parser.error end
    local record = parser.buffer:sub(1, start - 1)
    parser.buffer = parser.buffer:sub(finish + 1)
    local data = {}
    for line in (record .. "\n"):gmatch("(.-)\n") do
      line = line:gsub("\r$", "")
      if line:sub(1, 5) == "data:" then
        local value = line:sub(6)
        if value:sub(1, 1) == " " then value = value:sub(2) end
        data[#data + 1] = value
      end
    end
    if #data > 0 then
      local ok, event = pcall(vim.json.decode, table.concat(data, "\n"))
      if not ok or type(event) ~= "table" or type(event.stop) ~= "boolean" then
        parser.error = "malformed SSE event"; return false, parser.error
      end
      parser.events = parser.events + 1
      if event.stop then
        parser.terminal = event
      elseif type(event.content) == "string" then
        parser.pieces[#parser.pieces + 1] = event.content
        if event.tokens ~= nil then
          if type(event.tokens) ~= "table" then
            parser.error = "invalid SSE token ids"; return false, parser.error
          end
          parser.has_token_ids = true
          local count = 0
          for index, token in pairs(event.tokens) do
            if type(index) ~= "number" or index % 1 ~= 0 or index < 1
                or type(token) ~= "number" or token % 1 ~= 0 or token < 0 then
              parser.error = "invalid SSE token ids"; return false, parser.error
            end
            count = count + 1
          end
          if count ~= #event.tokens then
            parser.error = "invalid SSE token ids"; return false, parser.error
          end
          parser.model_tokens = parser.model_tokens + count
          for index = 1, #event.tokens do
            parser.sampled_token_ids[#parser.sampled_token_ids + 1] = event.tokens[index]
          end
          if count > 0 and not parser.first_token then
            parser.first_token = true
            parser.first_token_source = "sampled_token_ids"
          end
        elseif event.content ~= "" and not parser.first_token then
          parser.first_token = true
          parser.first_token_source = "observed_text_fallback"
        end
        if event.content ~= "" then parser.first_text = true end
      else
        parser.error = "missing SSE content"; return false, parser.error
      end
    end
  end
  return true
end

function M.finish(parser)
  if parser.error then return nil, parser.error end
  if not parser.terminal then return nil, "missing terminal SSE event" end
  if parser.terminal.stop_type ~= "eos" then
    return nil, "incomplete: " .. tostring(parser.terminal.stop_type)
  end
  local raw = table.concat(parser.pieces)
  if raw == "N\n" then return { action = "no_edit" }, raw end
  if raw:sub(1, 2) == "R\n" then
    return { action = "replace", text = raw:sub(3) }, raw
  end
  return nil, "malformed compact action"
end

function M.finish_single_line(parser)
  if parser.error then return nil, parser.error end
  if not parser.terminal then return nil, "missing terminal SSE event" end
  if parser.terminal.stop_type ~= "eos" then
    return nil, "incomplete: " .. tostring(parser.terminal.stop_type)
  end
  local raw = table.concat(parser.pieces)
  local predicted = parser.terminal.tokens_predicted
  local action, err = require("tabcomplete_trajectory.single_line_v1").decode_action(
    raw, parser.terminal.stop_type, predicted)
  if not action then return nil, err end
  return action, raw
end

local function dense_nonnegative_ids(ids, maximum)
  if type(ids) ~= "table" then return nil end
  local count, max_index = 0, 0
  for index, id in pairs(ids) do
    if type(index) ~= "number" or index % 1 ~= 0 or index < 1
        or type(id) ~= "number" or id % 1 ~= 0 or id < 0 or id > 2147483647 then
      return nil
    end
    count = count + 1
    max_index = math.max(max_index, index)
  end
  if count ~= max_index or count > maximum then return nil end
  return true
end

local function exact_fim_ids(ids, contract)
  if type(ids) ~= "table" then return false end
  local expected = {
    eos = contract.eos_id,
    fim_prefix = contract.fim_prefix_id,
    fim_suffix = contract.fim_suffix_id,
    fim_middle = contract.fim_middle_id,
  }
  local count = 0
  for key, value in pairs(ids) do
    if expected[key] == nil or value ~= expected[key] then return false end
    count = count + 1
  end
  return count == 4
end

local function valid_digest(value)
  return type(value) == "string" and #value == 64 and value:match("^%x+$") ~= nil
end

local function exact_object_fields(value, allowed, expected_count)
  if type(value) ~= "table" then return false end
  local count = 0
  for key in pairs(value) do
    if not allowed[key] then return false end
    count = count + 1
  end
  return count == expected_count
end

local function valid_fim_timings(value, predicted)
  local expected = {
    cache_n = true, prompt_n = true, prompt_ms = true,
    predicted_n = true, predicted_ms = true, total_ms = true,
  }
  if not exact_object_fields(value, expected, 6) then return false end
  for key, number in pairs(value) do
    if type(number) ~= "number" or number < 0 or number ~= number
        or number == math.huge or number == -math.huge
        or (key:match("_n$") and number % 1 ~= 0) then
      return false
    end
  end
  return value.predicted_n == predicted
end

local function validate_fim_terminal_binding(terminal, parser, expected, identity)
  local fim = require("tabcomplete_trajectory.fim_v1")
  local contract = expected and expected.fim_token_contract
  local prepared = expected and expected.fim_prepared
  local profile = identity and identity.fim_profile
  local profile_tokenizer = type(profile) == "table" and profile.tokenizer or nil
  if type(expected) ~= "table" or type(identity) ~= "table"
      or type(prepared) ~= "table" or type(contract) ~= "table"
      or type(profile) ~= "table" or type(profile_tokenizer) ~= "table"
      or not valid_digest(profile.artifact_manifest_sha256)
      or not valid_digest(identity.model_sha256) or identity.model_protocol ~= fim.WIRE_VERSION
      or identity.output_tokens ~= fim.MAX_OUTPUT_TOKENS
      or type(expected.request_id) ~= "string" or expected.request_id == ""
      or not valid_digest(expected.context_hash)
      or expected.completion_mode ~= fim.COMPLETION_MODE
      or terminal.request_id ~= expected.request_id
      or terminal.context_hash ~= expected.context_hash
      or terminal.completion_mode ~= fim.COMPLETION_MODE
      or terminal.model_protocol ~= identity.model_protocol
      or terminal.model_sha256 ~= identity.model_sha256
      or terminal.context_layout ~= identity.context_layout
      or terminal.artifact_manifest_sha256 ~= profile.artifact_manifest_sha256
      or terminal.tokenizer_id ~= identity.tokenizer_id
      or terminal.tokenizer_revision ~= identity.tokenizer_revision
      or terminal.tokenizer_sha256 ~= identity.tokenizer_sha256
      or terminal.tokenizer_contract_sha256 ~= identity.tokenizer_contract_sha256
      or terminal.tokenizer_vocab_size ~= identity.tokenizer_vocab_size
      or terminal.tokenizer_vocab_ids_sha256 ~= identity.tokenizer_vocab_ids_sha256
      or not valid_digest(identity.tokenizer_sha256)
      or not valid_digest(identity.tokenizer_contract_sha256)
      or not valid_digest(identity.tokenizer_vocab_ids_sha256)
      or terminal.tokenizer_sha256 ~= contract.tokenizer_sha256
      or terminal.tokenizer_contract_sha256 ~= contract.tokenizer_contract_sha256
      or identity.tokenizer_sha256 ~= contract.tokenizer_sha256
      or identity.tokenizer_contract_sha256 ~= contract.tokenizer_contract_sha256
      or identity.tokenizer_vocab_size ~= contract.tokenizer_vocab_size
      or identity.tokenizer_vocab_ids_sha256 ~= contract.tokenizer_vocab_ids_sha256
      or prepared.tokenizer_sha256 ~= contract.tokenizer_sha256
      or prepared.tokenizer_contract_sha256 ~= contract.tokenizer_contract_sha256
      or profile_tokenizer.tokenizer_sha256 ~= contract.tokenizer_sha256
      or profile_tokenizer.tokenizer_contract_sha256 ~= contract.tokenizer_contract_sha256
      or profile_tokenizer.tokenizer_vocab_size ~= contract.tokenizer_vocab_size
      or profile_tokenizer.tokenizer_vocab_ids_sha256 ~= contract.tokenizer_vocab_ids_sha256
      or profile_tokenizer.eos_id ~= contract.eos_id
      or profile_tokenizer.fim_prefix_id ~= contract.fim_prefix_id
      or profile_tokenizer.fim_suffix_id ~= contract.fim_suffix_id
      or profile_tokenizer.fim_middle_id ~= contract.fim_middle_id
      or not exact_fim_ids(identity.fim_token_ids, contract)
      or not exact_fim_ids(terminal.fim_token_ids, contract)
      or not dense_nonnegative_ids(terminal.sampled_token_ids, fim.MAX_OUTPUT_TOKENS)
      or not vim.deep_equal(terminal.sampled_token_ids, parser.sampled_token_ids)
      or terminal.tokens_predicted ~= #terminal.sampled_token_ids
      or parser.model_tokens ~= terminal.tokens_predicted
      or (terminal.tokens_predicted > 0 and not parser.has_token_ids) then
    return nil, "Rust FIM terminal context or sampled-token identity mismatch"
  end

  local token_id = terminal.terminal_token_id
  if terminal.stop_type == "eos" then
    if token_id ~= contract.eos_id then
      return nil, "Rust FIM terminal EOS identity mismatch"
    end
  elseif terminal.stop_type == "control" then
    if type(token_id) ~= "number" or token_id % 1 ~= 0
        or token_id < 0 or token_id == contract.eos_id
        or type(contract._by_id) ~= "table" or contract._by_id[token_id] == nil then
      return nil, "Rust FIM terminal control identity mismatch"
    end
  elseif terminal.stop_type == "limit" then
    if token_id ~= nil and token_id ~= vim.NIL then
      return nil, "Rust FIM limit terminal unexpectedly has a token ID"
    end
  else
    return nil, "Rust FIM terminal stop type is invalid"
  end

  local validation = terminal.action_validation
  if type(validation) ~= "table" or validation.policy ~= "q25-fim-completion-v1" then
    return nil, "Rust FIM terminal validation policy mismatch"
  end
  if validation.status == "invalid" then
    if not exact_object_fields(validation,
          { policy = true, status = true, code = true }, 3)
        or not fim.INVALID_COMPLETION_CODES[validation.code]
        or not valid_fim_timings(terminal.timings, terminal.tokens_predicted)
        or terminal.canonical_action ~= vim.NIL
        or (validation.code == "fim_missing_eos"
          and (terminal.stop_type ~= "limit" or #terminal.sampled_token_ids ~= fim.MAX_OUTPUT_TOKENS))
        or (validation.code == "fim_terminal_not_eos"
          and terminal.stop_type ~= "control")
        or (validation.code ~= "fim_missing_eos" and validation.code ~= "fim_terminal_not_eos"
          and terminal.stop_type ~= "eos")
        or (validation.code == "fim_output_cap_exceeded"
          and #terminal.sampled_token_ids < fim.MAX_OUTPUT_TOKENS) then
      return nil, "Rust FIM invalid-completion terminal is inconsistent"
    end
    return true, "invalid FIM completion: " .. validation.code
  end
  if not exact_object_fields(validation, { policy = true, status = true }, 2)
      or not valid_fim_timings(terminal.timings, terminal.tokens_predicted)
      or validation.status ~= "not_applicable" or terminal.stop_type ~= "eos"
      or terminal.tokens_predicted + 1 > fim.MAX_OUTPUT_TOKENS then
    return nil, "Rust FIM terminal is not a complete validated completion"
  end
  return true
end

-- Validate the Rust service terminal metadata and its canonical one-line action.
-- Qwen's raw wire is independently decoded; Sweep's raw text remains the full
-- generated file while the service supplies its strict target-line mapping.
function M.finish_rust(parser, expected)
  if parser.error then return nil, parser.error end
  if not parser.terminal then return nil, "missing terminal SSE event" end
  local terminal = parser.terminal
  local identity = expected and (expected.model_identity or expected)
  if type(identity) ~= "table" or terminal.model_sha256 ~= identity.model_sha256 then
    return nil, "Rust terminal model digest mismatch"
  end
  if terminal.model_protocol ~= identity.model_protocol then
    return nil, "Rust terminal model protocol mismatch"
  end
  local context_layout = identity.context_layout
  local legacy_layout = identity.context_layout_legacy == true or context_layout == nil
  if context_layout ~= nil then
    if terminal.context_layout ~= nil and terminal.context_layout ~= context_layout then
      return nil, "Rust terminal context layout mismatch"
    end
    if not legacy_layout and terminal.context_layout ~= context_layout then
      return nil, "Rust terminal context layout mismatch"
    end
  end
  local is_fim = identity.model_protocol == "q25-fim-line-completion-v1"
  if is_fim then
    local bound, fim_err = validate_fim_terminal_binding(terminal, parser, expected, identity)
    if not bound then return nil, fim_err end
    if fim_err then return nil, fim_err, true end
  elseif terminal.stop_type ~= "eos" then
    return nil, "incomplete: " .. tostring(terminal.stop_type)
  end
  local predicted = terminal.tokens_predicted
  if type(predicted) ~= "number" or predicted % 1 ~= 0 or predicted < 0
      or type(identity.output_tokens) ~= "number" or predicted > identity.output_tokens then
    return nil, "Rust terminal token count exceeds the configured cap"
  end
  if parser.has_token_ids and parser.model_tokens ~= predicted then
    return nil, "Rust terminal token count disagrees with sampled token IDs"
  end
  local canonical = terminal.canonical_action
  if type(canonical) ~= "table" then
    local validation = terminal.action_validation
    if type(validation) == "table" and validation.status == "rejected"
        and validation.reason == "rust_syntax_regression" then
      return nil, "prediction withheld: model introduced a Rust syntax error"
    end
    return nil, "missing Rust canonical action"
  end
  if canonical.text == nil then return nil, "Rust canonical action omitted its text field" end
  for key in pairs(canonical) do
    if key ~= "kind" and key ~= "text" then return nil, "unexpected Rust canonical action field" end
  end
  local kind = canonical.kind
  local text = canonical.text
  if text == vim.NIL then text = nil end
  local action
  if kind == "keep" or kind == "delete_line" then
    if text ~= nil then return nil, "Rust no-text action included replacement text" end
    action = { kind = kind }
  elseif kind == "replace_line" or kind == "insert_before" then
    if type(text) ~= "string" or text:find("[\r\n]") or text:find("\0", 1, true) then
      return nil, "Rust canonical action is not a single-line edit"
    end
    action = { kind = kind, text = text }
  else
    return nil, "unsupported Rust canonical action"
  end
  if identity.model_protocol == "q25-fim-line-completion-v1" then
    local fim = require("tabcomplete_trajectory.fim_v1")
    local decoded, err = fim.decode_completion(expected.fim_prepared, {
      model_protocol = terminal.model_protocol,
      model_sha256 = terminal.model_sha256,
      stop_type = terminal.stop_type,
      terminal_token_id = terminal.terminal_token_id,
      raw_text = table.concat(parser.pieces),
      sampled_token_ids = terminal.sampled_token_ids,
      tokens_predicted = terminal.tokens_predicted,
    }, expected.fim_token_contract, identity)
    if not decoded then return nil, err, true end
    if decoded.action.kind ~= action.kind or decoded.action.text ~= action.text then
      return nil, "Rust FIM canonical action disagrees with the PSM completion", true
    end
    action = decoded.action
  elseif identity.model_protocol == "single-line-edit-v1" then
    local raw_action, err = require("tabcomplete_trajectory.single_line_v1").decode_action(
      table.concat(parser.pieces), "eos", predicted)
    if not raw_action then return nil, err end
    if raw_action.kind ~= action.kind or raw_action.text ~= action.text then
      return nil, "Rust canonical action disagrees with Qwen wire text"
    end
  elseif identity.model_protocol ~= "sweep-full-file-v1" then
    return nil, "unsupported Rust model protocol"
  elseif not expected.window then
    return nil, "Rust Sweep action has no validated context window"
  end
  if expected and expected.contract_state
      and identity.model_protocol ~= "q25-fim-line-completion-v1" then
    local ok, err = pcall(require("tabcomplete_trajectory.single_line_v1").action_range,
      expected.contract_state, action)
    if not ok then return nil, "Rust action is outside the editor state contract: " .. tostring(err) end
  end
  return action, table.concat(parser.pieces)
end

return M
