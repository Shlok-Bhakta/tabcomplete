return function(ok, assert_eq, assert_true)
  local fim_v1 = require("tabcomplete_trajectory.fim_v1")
  local buffers = require("tabcomplete_trajectory.buffers")

  local function contract()
    local profile = {
      tokenizer_id = "synthetic/fim-fixture",
      tokenizer_revision = "synthetic-revision",
      completion_mode = fim_v1.COMPLETION_MODE,
      eos_id = 151643,
      fim_prefix_id = 151659,
      fim_middle_id = 151660,
      fim_suffix_id = 151661,
      tokenizer_sha256 = string.rep("a", 64),
      tokenizer_vocab_ids = { 17, 151643, 151644, 151645, 151659, 151660, 151661 },
      special_tokens = {
        { id = 151643, spelling = "<|endoftext|>" },
        { id = 151644, spelling = "<|im_start|>" },
        { id = 151645, spelling = "<|im_end|>" },
        { id = 151659, spelling = "<|fim_prefix|>" },
        { id = 151660, spelling = "<|fim_middle|>" },
        { id = 151661, spelling = "<|fim_suffix|>" },
      },
    }
    profile.tokenizer_vocab_size = #profile.tokenizer_vocab_ids
    profile.tokenizer_vocab_ids_sha256 = fim_v1.tokenizer_vocab_ids_sha256(profile.tokenizer_vocab_ids)
    profile.tokenizer_contract_sha256 = fim_v1.tokenizer_contract_sha256(profile)
    return fim_v1.new_token_contract(profile)
  end

  local function identity()
    return {
      model_protocol = fim_v1.WIRE_VERSION,
      model_sha256 = string.rep("a", 64),
      tokenizer_sha256 = string.rep("a", 64),
      tokenizer_contract_sha256 = contract().tokenizer_contract_sha256,
      tokenizer_vocab_size = contract().tokenizer_vocab_size,
      tokenizer_vocab_ids_sha256 = contract().tokenizer_vocab_ids_sha256,
      fim_profile = {
        artifact_manifest_sha256 = string.rep("b", 64),
        tokenizer = {
          tokenizer_contract_sha256 = contract().tokenizer_contract_sha256,
        },
      },
      output_tokens = fim_v1.MAX_OUTPUT_TOKENS,
    }
  end

  local function response(raw, ids, overrides)
    local result = {
      model_protocol = fim_v1.WIRE_VERSION,
      model_sha256 = string.rep("a", 64),
      stop_type = "eos",
      terminal_token_id = 151643,
      raw_text = raw,
      sampled_token_ids = ids or { 17 },
      tokens_predicted = ids and #ids or 1,
    }
    for key, value in pairs(overrides or {}) do result[key] = value end
    return result
  end

  local function mkbuf(lines, fileformat, eol)
    local bufnr = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_lines(bufnr, 0, -1, false, lines)
    vim.api.nvim_set_option_value("fileformat", fileformat or "unix", { buf = bufnr })
    vim.api.nvim_set_option_value("eol", eol == true, { buf = bufnr })
    return bufnr
  end

  ok("fim-golden-psm-lf-unicode-byte-ranges", function()
    local prepared, err = fim_v1.prepare("α=1\nnext()\n", 0, 2, contract())
    assert_true(prepared ~= nil, tostring(err))
    assert_eq(prepared.prompt,
      "<|fim_prefix|>α<|fim_suffix|>next()\n<|fim_middle|>")
    assert_eq(prepared.model_hole_range.start_byte, 2)
    assert_eq(prepared.model_hole_range.end_byte, 5)
    assert_true(prepared.model_hole_range.end_exclusive)
    assert_eq(prepared.apply_range.start_byte, 0)
    assert_eq(prepared.apply_range.end_byte, 4)
    assert_eq(prepared.line_ending, "LF")
    assert_eq(prepared.model_protocol, "q25-fim-line-completion-v1")
  end)

  ok("fim-bounded-context-v2-reconstructs-source-without-tokenizing-locally", function()
    local source = "old line\nleft🙂right\r\nsuffix"
    local token_contract = contract()
    local base = assert(fim_v1.prepare(source, 1, 8, token_contract))
    local prepared = vim.deepcopy(base)
    prepared.prefix_range = { start_byte = 9, end_byte = 17, end_exclusive = true }
    prepared.prefix_token_count = 5
    prepared.suffix_range = { start_byte = 24, end_byte = 30, end_exclusive = true }
    prepared.suffix_token_count = 6
    prepared.prompt = "<|fim_prefix|>left🙂<|fim_suffix|>suffix<|fim_middle|>"
    local response_value = {
      context_policy_version = fim_v1.CONTEXT_POLICY_VERSION,
      context_layout = fim_v1.CONTEXT_LAYOUT,
      tokenizer_sha256 = base.tokenizer_sha256,
      tokenizer_contract_sha256 = base.tokenizer_contract_sha256,
      target_row = 1,
      cursor_col = 8,
      prompt = prepared.prompt,
      prompt_tokens = 14,
      prefix_range = vim.deepcopy(prepared.prefix_range),
      prefix_context_tokens = prepared.prefix_token_count,
      suffix_range = vim.deepcopy(prepared.suffix_range),
      suffix_context_tokens = prepared.suffix_token_count,
      model_hole_range = vim.deepcopy(base.model_hole_range),
      apply_range = vim.deepcopy(base.apply_range),
      line_ending = base.line_ending,
    }
    local verified, err = fim_v1.verify_prepared(source, 1, 8, token_contract, response_value)
    assert_true(verified ~= nil, tostring(err))
    assert_eq(verified.prompt, prepared.prompt)
    assert_eq(verified.prefix_range.start_byte, 9)
    assert_eq(verified.prefix_range.end_byte, 17)
    assert_eq(verified.suffix_range.start_byte, 24)
    assert_eq(verified.suffix_range.end_byte, 30)
    assert_eq(verified.apply_range.start_byte, base.apply_range.start_byte)
    assert_eq(verified.apply_range.end_byte, base.apply_range.end_byte)

    local tampered = vim.deepcopy(response_value)
    tampered.prefix_range.start_byte = 15
    assert_true(not fim_v1.verify_prepared(source, 1, 8, token_contract, tampered),
      "a crop inside the emoji is rejected")
    tampered = vim.deepcopy(response_value)
    tampered.prefix_range.end_byte = 16
    assert_true(not fim_v1.verify_prepared(source, 1, 8, token_contract, tampered),
      "prefix must end at the original cursor")
    tampered = vim.deepcopy(response_value)
    tampered.prompt = tampered.prompt .. "altered"
    assert_true(not fim_v1.verify_prepared(source, 1, 8, token_contract, tampered),
      "prompt must be reconstructed from source ranges")
    tampered = vim.deepcopy(response_value)
    tampered.prefix_context_tokens = fim_v1.PREFIX_CONTEXT_TOKEN_LIMIT + 1
    assert_true(not fim_v1.verify_prepared(source, 1, 8, token_contract, tampered),
      "per-side token limit is enforced")
  end)

  ok("fim-crlf-completion-reuses-canonical-buffer-applier", function()
    local source = "a🙂b\r\ntail\r\n"
    local prepared, err = fim_v1.prepare(source, 0, 5, contract())
    assert_true(prepared ~= nil, tostring(err))
    assert_eq(prepared.prompt,
      "<|fim_prefix|>a🙂<|fim_suffix|>tail\r\n<|fim_middle|>")
    assert_eq(prepared.model_hole_range.start_byte, 5)
    assert_eq(prepared.model_hole_range.end_byte, 8)
    assert_eq(prepared.apply_range.start_byte, 0)
    assert_eq(prepared.apply_range.end_byte, 6)
    assert_true(prepared.apply_range.end_exclusive)
    local decoded = fim_v1.decode_completion(prepared, response("value\r\n"), contract(), identity())
    assert_true(decoded ~= nil, "valid CRLF completion decoded")
    assert_eq(decoded.raw_text, "value\r\n")
    assert_eq(decoded.stripped_line_ending, "\r\n")
    assert_eq(decoded.action.kind, "replace_line")
    assert_eq(decoded.action.text, "a🙂value")
    assert_eq(decoded.model_hole_range.end_byte, 8)
    assert_eq(decoded.apply_range.end_byte, 6)
    local mapped = require("tabcomplete_trajectory.single_line_v1").action_range(
      { source = source, target_row = 0 }, decoded.action)
    assert_eq(mapped.start_byte, decoded.apply_range.start_byte)
    assert_eq(mapped.end_byte, decoded.apply_range.end_byte)
    local bufnr = mkbuf({ "a🙂b", "tail" }, "dos", true)
    local applied, apply_err = fim_v1.apply_to_buffer(bufnr, prepared, decoded)
    assert_true(applied, tostring(apply_err))
    assert_eq(buffers.canonical_bytes(bufnr), "a🙂value\r\ntail\r\n")
    vim.api.nvim_buf_delete(bufnr, { force = true })
  end)

  ok("fim-eof-has-no-synthesized-newline-and-keeps-old-wire-literal", function()
    local prepared = assert(fim_v1.prepare("ab", 0, 2, contract()))
    local decoded = assert(fim_v1.decode_completion(
      prepared, response("R\tvalue"), contract(), identity()))
    assert_eq(decoded.raw_text, "R\tvalue")
    assert_eq(decoded.action.text, "abR\tvalue")
    assert_eq(decoded.stripped_line_ending, "")
    local bufnr = mkbuf({ "ab" }, "unix", false)
    local applied, apply_err = fim_v1.apply_to_buffer(bufnr, prepared, decoded)
    assert_true(applied, tostring(apply_err))
    assert_eq(buffers.canonical_bytes(bufnr), "abR\tvalue")
    vim.api.nvim_buf_delete(bufnr, { force = true })
  end)

  ok("fim-empty-file-uses-validated-insertion", function()
    local prepared = assert(fim_v1.prepare("", 0, 0, contract()))
    assert_eq(prepared.prompt, "<|fim_prefix|><|fim_suffix|><|fim_middle|>")
    local decoded = assert(fim_v1.decode_completion(
      prepared, response("fn main() {}"), contract(), identity()))
    assert_eq(decoded.action.kind, "insert_before")
    local mapped = require("tabcomplete_trajectory.single_line_v1").action_range(
      { source = "", target_row = 0 }, decoded.action)
    assert_eq(mapped.start_byte, decoded.apply_range.start_byte)
    assert_eq(mapped.end_byte, decoded.apply_range.end_byte)
    local bufnr = mkbuf({ "" }, "unix", false)
    local applied, apply_err = fim_v1.apply_to_buffer(bufnr, prepared, decoded)
    assert_true(applied, tostring(apply_err))
    assert_eq(buffers.canonical_bytes(bufnr), "fn main() {}")
    vim.api.nvim_buf_delete(bufnr, { force = true })
  end)

  ok("fim-eos-at-step-96-valid-and-incomplete-or-late-eos-rejected", function()
    local prepared = assert(fim_v1.prepare("x\n", 0, 1, contract()))
    local ids = {}
    for index = 1, 95 do ids[index] = 17 end
    local decoded, err = fim_v1.decode_completion(
      prepared, response("\n", ids), contract(), identity())
    assert_true(decoded ~= nil, tostring(err))
    assert_eq(decoded.generated_steps, 96)
    ids[96] = 17
    local late = fim_v1.decode_completion(
      prepared, response("\n", ids), contract(), identity())
    assert_true(late == nil, "EOS after the cap rejected")
    local no_eos = fim_v1.decode_completion(prepared,
      response("\n", {}, { terminal_token_id = nil, stop_type = "limit" }), contract(), identity())
    assert_true(no_eos == nil, "missing EOS rejected")
  end)

  ok("fim-rejects-wrong-eog-added-controls-and-model-identity-mismatch", function()
    local prepared = assert(fim_v1.prepare("x", 0, 1, contract()))
    for _, bad in ipairs({
      response("text", { 17 }, { terminal_token_id = 151645 }),
      response("", {}, { terminal_token_id = 151645 }),
      response("text", { 151659 }),
      response("text", { 18 }),
      response("<|im_end|>", { 17 }),
      response("text", { 17 }, { model_protocol = "single-line-edit-v1" }),
      response("text", { 17 }, { model_sha256 = string.rep("b", 64) }),
    }) do
      local decoded = fim_v1.decode_completion(prepared, bad, contract(), identity())
      assert_true(decoded == nil, "mismatched FIM response rejected")
    end
    local wrong_tokenizer = identity()
    wrong_tokenizer.tokenizer_sha256 = string.rep("b", 64)
    assert_true(fim_v1.decode_completion(
      prepared, response("text"), contract(), wrong_tokenizer) == nil,
      "tokenizer identity must match the frozen inventory")
    local as_legacy = fim_v1.decode_completion(prepared,
      response("R\ttext"), contract(), {
        model_protocol = "single-line-edit-v1",
        model_sha256 = string.rep("a", 64),
        output_tokens = 64,
      })
    assert_true(as_legacy == nil, "FIM never falls back to legacy action parsing")
  end)

  ok("fim-rejects-special-source-invalid-cursor-and-newline-repair", function()
    for _, invalid in ipairs({
      { "x<|fim_prefix|>y", 0, 0 },
      { "x<|im_start|>y", 0, 0 },
      { "a\rb", 0, 0 },
      { "🙂", 0, 1 },
      { string.char(0xc0, 0xaf), 0, 0 },
      { "x\n", 1, 0 },
    }) do
      local prepared = fim_v1.prepare(invalid[1], invalid[2], invalid[3], contract())
      assert_true(prepared == nil, "invalid FIM source rejected")
    end
    local crlf = assert(fim_v1.prepare("x\r\n", 0, 1, contract()))
    for _, raw in ipairs({ "body\n", "body\r", "body\r\n\r\n" }) do
      local decoded = fim_v1.decode_completion(crlf,
        response(raw), contract(), identity())
      assert_true(decoded == nil, "completion EOL is never repaired")
    end
    local eof = assert(fim_v1.prepare("x", 0, 1, contract()))
    assert_true(fim_v1.decode_completion(eof, response("body\n"), contract(), identity()) == nil,
      "EOF completion cannot add a newline")
    assert_true(fim_v1.decode_completion(eof, response(string.char(0xff)), contract(), identity()) == nil,
      "invalid UTF-8 output rejected")
  end)

  ok("fim-rejects-stale-buffer-through-canonical-applier", function()
    local prepared = assert(fim_v1.prepare("old\n", 0, 1, contract()))
    local decoded = assert(fim_v1.decode_completion(
      prepared, response("new\n"), contract(), identity()))
    local bufnr = mkbuf({ "changed" }, "unix", true)
    local applied = fim_v1.apply_to_buffer(bufnr, prepared, decoded)
    assert_true(not applied, "stale source does not apply")
    assert_eq(buffers.canonical_bytes(bufnr), "changed\n")
    vim.api.nvim_buf_delete(bufnr, { force = true })
  end)
end
