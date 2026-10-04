return function(ok, assert_eq, assert_true)
  local collector = require("tabcomplete_trajectory")
  local buffers = require("tabcomplete_trajectory.buffers")
  local predict = require("tabcomplete_trajectory.predict")
  local fim = require("tabcomplete_trajectory.fim_v1")
  local util = require("tabcomplete_trajectory.util")

  local vocab_ids = { 17, 151643, 151644, 151645, 151659, 151660, 151661 }
  local tokenizer = {
    tokenizer_id = "synthetic/q25-fim-editor-fixture",
    tokenizer_revision = "synthetic-commit",
    tokenizer_sha256 = string.rep("a", 64),
    tokenizer_vocab_size = #vocab_ids,
    tokenizer_vocab_ids_sha256 = fim.tokenizer_vocab_ids_sha256(vocab_ids),
    eos_id = fim.EOS_TOKEN_ID,
    fim_prefix_id = fim.FIM_PREFIX_TOKEN_ID,
    fim_suffix_id = fim.FIM_SUFFIX_TOKEN_ID,
    fim_middle_id = fim.FIM_MIDDLE_TOKEN_ID,
    completion_mode = fim.COMPLETION_MODE,
    special_tokens = {
      { id = 151643, spelling = "<|endoftext|>" },
      { id = 151644, spelling = "<|im_start|>" },
      { id = 151645, spelling = "<|im_end|>" },
      { id = 151659, spelling = "<|fim_prefix|>" },
      { id = 151660, spelling = "<|fim_middle|>" },
      { id = 151661, spelling = "<|fim_suffix|>" },
    },
  }
  tokenizer.tokenizer_contract_sha256 = fim.tokenizer_contract_sha256(tokenizer)
  local profile = {
    artifact_manifest_sha256 = string.rep("b", 64),
    tokenizer = tokenizer,
  }
  local model = {
    model_sha256 = string.rep("c", 64),
    model_protocol = fim.WIRE_VERSION,
    output_tokens = fim.MAX_OUTPUT_TOKENS,
    fim_profile = profile,
  }
  local function identity()
    return {
      status = "ok",
      alias = "q25-fim-synthetic",
      model_sha256 = model.model_sha256,
      model_protocol = model.model_protocol,
      fim_profile = vim.deepcopy(profile),
      completion_mode = fim.COMPLETION_MODE,
      tokenizer_id = tokenizer.tokenizer_id,
      tokenizer_revision = tokenizer.tokenizer_revision,
      tokenizer_sha256 = tokenizer.tokenizer_sha256,
      tokenizer_contract_sha256 = tokenizer.tokenizer_contract_sha256,
      tokenizer_vocab_size = tokenizer.tokenizer_vocab_size,
      tokenizer_vocab_ids_sha256 = tokenizer.tokenizer_vocab_ids_sha256,
      runtime_config_hash = string.rep("d", 64),
      input_tokens = 1024,
      output_tokens = fim.MAX_OUTPUT_TOKENS,
      context_size = 1120,
      context_layout = fim.CONTEXT_LAYOUT,
      model_embedded = false,
      model_switch_supported = true,
    }
  end

  ok("q25-fim-editor-binds-profile-context-and-terminal-before-preview", function()
    collector.started = true
    collector.disabled = false
    collector.paused = false
    collector.session_id = "synthetic-q25-fim-session"
    collector.queue = {}
    collector.seq = 0

    local inventory_fetches = 0
    local installed_identity = identity()
    predict.setup({ backend = "rust-editor-v1", protocol_version = fim.WIRE_VERSION,
      url = "http://127.0.0.1:19094", mode = "manual", synthetic = true,
      allowed_models = { ["q25-fim-synthetic"] = model },
      single_line_input_tokens = 1024 })
    assert_eq(predict.status().protocol_version, fim.WIRE_VERSION)
    assert_eq(predict.status().context_policy_version, fim.CONTEXT_POLICY_VERSION)
    local aliases = predict.model_aliases()
    assert_eq(#aliases, 1)
    assert_eq(aliases[1], "q25-fim-synthetic")

    predict._backend_get_impl = function(path, callback)
      if path == "/health" then
        callback(true, identity())
      elseif path == "/v1/fim-tokenizer" then
        inventory_fetches = inventory_fetches + 1
        callback(true, {
          alias = installed_identity.alias,
          model_sha256 = installed_identity.model_sha256,
          artifact_manifest_sha256 = profile.artifact_manifest_sha256,
          tokenizer = vim.deepcopy(tokenizer),
          tokenizer_vocab_ids = vim.deepcopy(vocab_ids),
        })
      else
        error("unexpected Rust identity endpoint " .. tostring(path))
      end
      return { kill = function() end }
    end

    local refreshed
    assert_true(predict.refresh_model_identity(function(success, result)
      refreshed = success and result or false
    end), "first frozen model identity refresh started")
    assert_true(vim.wait(1000, function() return refreshed ~= nil end),
      "first frozen identity refresh completed")
    assert_true(refreshed ~= false, "frozen identity accepted")
    assert_eq(inventory_fetches, 1, "complete known-token inventory fetched once")

    refreshed = nil
    assert_true(predict.refresh_model_identity(function(success, result)
      refreshed = success and result or false
    end), "cached model identity refresh started")
    assert_true(vim.wait(1000, function() return refreshed ~= nil end),
      "cached model identity refresh completed")
    assert_true(refreshed ~= false, "cached identity accepted")
    assert_eq(inventory_fetches, 1, "same model and tokenizer contract reuse inventory")

    local buf = vim.api.nvim_create_buf(true, false)
    local path = vim.fn.tempname() .. ".py"
    vim.api.nvim_buf_set_name(buf, path)
    vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "old" })
    vim.bo[buf].filetype = "python"
    vim.api.nvim_set_option_value("fileformat", "unix", { buf = buf })
    vim.api.nvim_set_option_value("eol", true, { buf = buf })
    vim.api.nvim_set_current_buf(buf)
    assert_true(buffers.attach(buf), "synthetic FIM editor buffer attached")
    vim.api.nvim_win_set_cursor(0, { 1, 0 })

    local prepared_context
    predict._backend_post_impl = function(endpoint, body, callback)
      if endpoint == "/v1/editor/context" then
        assert_eq(body.completion_mode, fim.COMPLETION_MODE)
        assert_true(type(body.request_id) == "string" and body.request_id ~= "",
          "context request has a unique request ID")
        prepared_context = assert(fim.prepare(body.state.source, body.state.target_row,
          body.state.cursor_col, refreshed._fim_token_contract))
        assert_eq(prepared_context.prompt, "<|fim_prefix|><|fim_suffix|><|fim_middle|>")
        local context_hash = fim.context_digest(body.request_id, body.state.source,
          body.state.target_row, body.state.cursor_col, prepared_context.prompt,
          tokenizer.tokenizer_contract_sha256)
        callback(true, {
          prompt = prepared_context.prompt,
          prompt_tokens = 5,
          context_hash = context_hash,
          context_policy_version = fim.CONTEXT_POLICY_VERSION,
          context_layout = fim.CONTEXT_LAYOUT,
          model_protocol = fim.WIRE_VERSION,
          model_identity = identity(),
          selected_buffers = {},
          request_id = body.request_id,
          completion_mode = fim.COMPLETION_MODE,
          tokenizer_sha256 = tokenizer.tokenizer_sha256,
          tokenizer_contract_sha256 = tokenizer.tokenizer_contract_sha256,
          target_row = body.state.target_row,
          cursor_col = body.state.cursor_col,
          model_hole_range = vim.deepcopy(prepared_context.model_hole_range),
          apply_range = vim.deepcopy(prepared_context.apply_range),
          line_ending = prepared_context.line_ending,
        })
      else
        error("unexpected Rust POST endpoint " .. tostring(endpoint))
      end
      return { kill = function() end }
    end
    predict._request_impl = function(state, body, callback)
      local request = vim.json.decode(body)
      assert_eq(request.n_predict, fim.MAX_OUTPUT_TOKENS, "FIM generation keeps the 96-token cap")
      assert_eq(request.prompt, prepared_context.prompt, "generation uses the exact prepared PSM prompt")
      assert_eq(request.completion_mode, fim.COMPLETION_MODE)
      assert_eq(request.request_id, state.request_id)
      assert_eq(request.context_hash, state.context_hash)
      local wire = {
        { content = "new\n", tokens = { 17 }, stop = false },
        { content = "", stop = true, stop_type = "eos", tokens_predicted = 1,
          canonical_action = { kind = "replace_line", text = "new" },
          model_protocol = fim.WIRE_VERSION, model_sha256 = model.model_sha256,
          context_layout = fim.CONTEXT_LAYOUT, terminal_token_id = fim.EOS_TOKEN_ID,
          sampled_token_ids = { 17 }, request_id = state.request_id,
          context_hash = state.context_hash, completion_mode = fim.COMPLETION_MODE,
          tokenizer_sha256 = tokenizer.tokenizer_sha256,
          tokenizer_contract_sha256 = tokenizer.tokenizer_contract_sha256 },
      }
      local chunks = {}
      for _, event in ipairs(wire) do
        chunks[#chunks + 1] = "data: " .. vim.json.encode(event) .. "\n\n"
      end
      callback({ code = 0, stdout = table.concat(chunks) })
      return { kill = function() end }
    end

    assert_true(predict.predict({ explicit = true }), "explicit FIM request started")
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end),
      "verified PSM completion became a proposal")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 0, 1, false)[1], "old",
      "completion preview does not mutate the source")
    local requested, generated, shown
    for _, event in ipairs(collector.queue) do
      if event.event_type == "prediction_requested" then requested = event.payload end
      if event.event_type == "prediction_generated" then generated = event.payload end
      if event.event_type == "prediction_shown" then shown = event.payload end
    end
    assert_true(requested and generated and shown, "FIM lifecycle events were recorded")
    assert_eq(requested.wire_version, fim.WIRE_VERSION)
    assert_eq(requested.context_policy_version, fim.CONTEXT_POLICY_VERSION)
    assert_eq(requested.max_output_tokens, fim.MAX_OUTPUT_TOKENS)
    assert_eq(generated.wire_version, fim.WIRE_VERSION)
    assert_eq(shown.wire_version, fim.WIRE_VERSION)
    assert_eq(shown.action, "replace_line")
    assert_eq(shown.proposed_text, "new")

    assert_true(predict.accept(), "explicit acceptance uses the existing canonical line applier")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 0, 1, false)[1], "new")
    local accepted
    for _, event in ipairs(collector.queue) do
      if event.event_type == "prediction_accepted" then accepted = event.payload end
    end
    assert_true(accepted ~= nil, "accepted decision recorded")
    assert_eq(accepted.wire_version, fim.WIRE_VERSION)

    predict._backend_get_impl = nil
    predict._backend_post_impl = nil
    predict._request_impl = nil
    predict.setup({ backend = "llama-cpp-legacy", protocol_version = "compact-next-edit-v1",
      url = "http://127.0.0.1:9", mode = "manual", synthetic = true, allowed_models = {} })
    buffers.detach(buf)
    vim.api.nvim_buf_delete(buf, { force = true })
  end)
end
