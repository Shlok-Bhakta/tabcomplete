-- Cold-start and outage checks for the selected FIM tokenizer identity.
-- All editor states and transport responses are synthetic.
return function(ok, assert_eq, assert_true)
  local collector = require("tabcomplete_trajectory")
  local buffers = require("tabcomplete_trajectory.buffers")
  local predict = require("tabcomplete_trajectory.predict")
  local fim = require("tabcomplete_trajectory.fim_v1")
  local util = require("tabcomplete_trajectory.util")

  local vocab_ids = { 17, 151643, 151644, 151645, 151659, 151660, 151661 }
  local tokenizer = {
    tokenizer_id = "synthetic/q25-fim-editor-fixture",
    tokenizer_revision = "synthetic-revision",
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
  local profile = { artifact_manifest_sha256 = string.rep("b", 64), tokenizer = tokenizer }
  local model = {
    model_sha256 = string.rep("c", 64),
    model_protocol = fim.WIRE_VERSION,
    output_tokens = fim.MAX_OUTPUT_TOKENS,
    fim_profile = profile,
  }
  local identity_variant = 0
  local function fresh_model_identity()
    identity_variant = identity_variant + 1
    local model_digit = string.format("%x", identity_variant)
    local artifact_digit = string.format("%x", identity_variant + 1)
    model.model_sha256 = string.rep(model_digit, 64)
    profile.artifact_manifest_sha256 = string.rep(artifact_digit, 64)
  end
  local token_config = vim.deepcopy(tokenizer)
  token_config.tokenizer_vocab_ids = vim.deepcopy(vocab_ids)
  local token_contract = assert(fim.new_token_contract(token_config))
  local previous_current_mode = util.current_mode
  util.current_mode = function() return "i" end

  local active_buf, active_path, active_service
  local opened_buffers, opened_paths = {}, {}
  local function identity()
    return {
      status = "ok",
      alias = "q25-fim-synthetic",
      model_sha256 = model.model_sha256,
      model_protocol = fim.WIRE_VERSION,
      fim_profile = vim.deepcopy(profile),
      completion_mode = fim.COMPLETION_MODE,
      tokenizer_id = tokenizer.tokenizer_id,
      tokenizer_revision = tokenizer.tokenizer_revision,
      tokenizer_sha256 = tokenizer.tokenizer_sha256,
      tokenizer_contract_sha256 = tokenizer.tokenizer_contract_sha256,
      tokenizer_vocab_size = tokenizer.tokenizer_vocab_size,
      tokenizer_vocab_ids_sha256 = tokenizer.tokenizer_vocab_ids_sha256,
      fim_token_ids = {
        eos = fim.EOS_TOKEN_ID,
        fim_prefix = fim.FIM_PREFIX_TOKEN_ID,
        fim_suffix = fim.FIM_SUFFIX_TOKEN_ID,
        fim_middle = fim.FIM_MIDDLE_TOKEN_ID,
      },
      runtime_config_hash = string.rep("d", 64),
      input_tokens = 1024,
      output_tokens = fim.MAX_OUTPUT_TOKENS,
      context_size = 1120,
      context_layout = fim.CONTEXT_LAYOUT,
      model_embedded = false,
      model_switch_supported = true,
    }
  end
  local function inventory()
    return {
      alias = "q25-fim-synthetic",
      model_sha256 = model.model_sha256,
      artifact_manifest_sha256 = profile.artifact_manifest_sha256,
      tokenizer = vim.deepcopy(tokenizer),
      tokenizer_vocab_ids = vim.deepcopy(vocab_ids),
    }
  end
  local function new_service(health_mode, tokenizer_mode)
    local service = {
      health_mode = health_mode or "hold",
      tokenizer_mode = tokenizer_mode or "auto",
      health_calls = 0,
      tokenizer_calls = 0,
      pending_health = {},
      pending_tokenizer = {},
      context_calls = 0,
      completion_calls = 0,
      completion_sources = {},
    }
    function service:resolve_health(index, success)
      local callback = table.remove(self.pending_health, index or 1)
      assert_true(callback ~= nil, "a health refresh was waiting")
      if success == false then callback(false, {}) else callback(true, identity()) end
    end
    function service:resolve_tokenizer(index, success)
      local callback = table.remove(self.pending_tokenizer, index or 1)
      assert_true(callback ~= nil, "a tokenizer fetch was waiting")
      if success == false then callback(false, {}) else callback(true, inventory()) end
    end
    function service:release_all()
      while #self.pending_health > 0 do
        self:resolve_health(1, false)
      end
      while #self.pending_tokenizer > 0 do
        self:resolve_tokenizer(1, false)
      end
    end
    predict._backend_get_impl = function(path, callback)
      if path == "/health" then
        service.health_calls = service.health_calls + 1
        if service.health_mode == "hold" then
          service.pending_health[#service.pending_health + 1] = callback
        elseif service.health_mode == "fail" then
          callback(false, {})
        else
          callback(true, identity())
        end
      elseif path == "/v1/fim-tokenizer" then
        service.tokenizer_calls = service.tokenizer_calls + 1
        if service.tokenizer_mode == "hold" then
          service.pending_tokenizer[#service.pending_tokenizer + 1] = callback
        elseif service.tokenizer_mode == "fail" then
          callback(false, {})
        else
          callback(true, inventory())
        end
      else
        error("unexpected Rust identity endpoint")
      end
      return { kill = function() end }
    end
    predict._backend_post_impl = function(endpoint, body, callback)
      if endpoint ~= "/v1/editor/context" then
        error("unexpected Rust POST endpoint")
      end
      service.context_calls = service.context_calls + 1
      local state = body.state
      local prepared = assert(fim.prepare(state.source, state.target_row,
        state.cursor_col, token_contract))
      local context_hash = fim.context_digest(body.request_id, state.source,
        state.target_row, state.cursor_col, prepared)
      callback(true, {
        prompt = prepared.prompt,
        prompt_tokens = 3,
        prefix_context_tokens = 0,
        suffix_context_tokens = 0,
        prefix_range = vim.deepcopy(prepared.prefix_range),
        suffix_range = vim.deepcopy(prepared.suffix_range),
        context_hash = context_hash,
        context_policy_version = fim.CONTEXT_POLICY_VERSION,
        context_layout = fim.CONTEXT_LAYOUT,
        model_protocol = fim.WIRE_VERSION,
        model_identity = identity(),
        selected_buffers = {},
        request_id = body.request_id,
        completion_mode = fim.COMPLETION_MODE,
        filetype_training_scope = fim.filetype_training_scope(state.filetype),
        tokenizer_sha256 = tokenizer.tokenizer_sha256,
        tokenizer_contract_sha256 = tokenizer.tokenizer_contract_sha256,
        target_row = state.target_row,
        cursor_col = state.cursor_col,
        model_hole_range = vim.deepcopy(prepared.model_hole_range),
        apply_range = vim.deepcopy(prepared.apply_range),
        line_ending = prepared.line_ending,
      })
      return { kill = function() end }
    end
    predict._request_impl = function(state, _, callback)
      service.completion_calls = service.completion_calls + 1
      service.completion_sources[#service.completion_sources + 1] = state.contract_state.source
      local terminal = {
        content = "", stop = true, stop_type = "eos", tokens_predicted = 1,
        canonical_action = { kind = "replace_line", text = "new" },
        action_validation = { policy = "q25-fim-completion-v1", status = "not_applicable" },
        model_protocol = fim.WIRE_VERSION, model_sha256 = model.model_sha256,
        context_layout = fim.CONTEXT_LAYOUT, terminal_token_id = fim.EOS_TOKEN_ID,
        sampled_token_ids = { 17 }, request_id = state.request_id,
        context_hash = state.context_hash, completion_mode = fim.COMPLETION_MODE,
        tokenizer_sha256 = tokenizer.tokenizer_sha256,
        tokenizer_contract_sha256 = tokenizer.tokenizer_contract_sha256,
        artifact_manifest_sha256 = profile.artifact_manifest_sha256,
        tokenizer_id = tokenizer.tokenizer_id,
        tokenizer_revision = tokenizer.tokenizer_revision,
        tokenizer_vocab_size = tokenizer.tokenizer_vocab_size,
        tokenizer_vocab_ids_sha256 = tokenizer.tokenizer_vocab_ids_sha256,
        fim_token_ids = vim.deepcopy(identity().fim_token_ids),
        timings = { cache_n = 0, prompt_n = 3, prompt_ms = 1,
          predicted_n = 1, predicted_ms = 1, total_ms = 2 },
      }
      local chunks = {
        "data: " .. vim.json.encode({ content = "new\n", tokens = { 17 }, stop = false }) .. "\n\n",
        "data: " .. vim.json.encode(terminal) .. "\n\n",
      }
      callback({ code = 0, stdout = table.concat(chunks) })
      return { kill = function() end }
    end
    active_service = service
    return service
  end
  local function open_buffer(text)
    vim.cmd("enew")
    active_buf = vim.api.nvim_get_current_buf()
    active_path = vim.fn.tempname() .. ".py"
    opened_buffers[#opened_buffers + 1] = active_buf
    opened_paths[#opened_paths + 1] = active_path
    vim.api.nvim_buf_set_name(active_buf, active_path)
    vim.api.nvim_buf_set_lines(active_buf, 0, -1, false, { text })
    vim.bo[active_buf].filetype = "python"
    vim.api.nvim_set_option_value("fileformat", "unix", { buf = active_buf })
    vim.api.nvim_set_option_value("eol", true, { buf = active_buf })
    vim.api.nvim_win_set_cursor(0, { 1, 0 })
    assert_true(buffers.attach(active_buf), "synthetic source buffer attached")
  end
  local function setup(mode, debounce_ms)
    collector.started, collector.disabled, collector.paused = true, false, false
    collector.session_id, collector.queue, collector.seq = "synthetic-fim-bootstrap", {}, 0
    buffers.set_callbacks(function(buf, kind, payload)
      return collector.emit(buf, kind, payload)
    end, function(buf) collector.anchor_prediction(buf) end)
    predict.setup({
      backend = "rust-editor-v1",
      protocol_version = fim.WIRE_VERSION,
      url = "http://127.0.0.1:19094",
      mode = mode,
      experimental_auto_opt_in = mode == "automatic",
      automatic_quality_validated = false,
      automatic_personalization_enabled = false,
      debounce_ms = debounce_ms or 5,
      expiry_ms = 1000,
      synthetic = true,
      persist_mode = false,
      allowed_models = { ["q25-fim-synthetic"] = model },
      single_line_input_tokens = 1024,
    })
  end
  local function cleanup()
    pcall(predict.set_mode, "off")
    if active_service then active_service:release_all() end
    vim.wait(100, function() return false end)
    predict._backend_get_impl = nil
    predict._backend_post_impl = nil
    predict._request_impl = nil
    predict.setup({ backend = "llama-cpp-legacy",
      protocol_version = "compact-next-edit-v1", mode = "manual",
      url = "http://127.0.0.1:9", persist_mode = false, allowed_models = {} })
    for _, buf in ipairs(opened_buffers) do
      if vim.api.nvim_buf_is_valid(buf) then
        pcall(buffers.detach, buf)
        pcall(vim.api.nvim_buf_delete, buf, { force = true })
      end
    end
    for _, path in ipairs(opened_paths) do os.remove(path) end
    opened_buffers, opened_paths = {}, {}
    active_buf, active_path, active_service = nil, nil, nil
  end
  local function case(name, fn)
    ok(name, function()
      local success, err = xpcall(fn, debug.traceback)
      cleanup()
      if not success then error(err) end
    end)
  end

  case("fim-automatic-cold-start-refreshes-then-uses-latest-buffer", function()
    fresh_model_identity()
    open_buffer("before identity")
    local service = new_service("hold", "auto")
    setup("automatic", 5)
    assert_eq(service.health_calls, 1, "setup started one identity refresh")
    vim.api.nvim_buf_set_lines(active_buf, 0, 1, false, { "latest editor state" })
    vim.api.nvim_exec_autocmds("TextChangedI", { buffer = active_buf })
    vim.wait(30, function() return false end)
    assert_eq(service.context_calls, 0, "no context request before tokenizer identity")
    service:resolve_health(1, true)
    assert_true(vim.wait(1500, function() return service.completion_calls == 1 end),
      "automatic prediction resumed after identity validation")
    assert_eq(service.health_calls, 1, "automatic startup did not duplicate health requests")
    assert_eq(service.completion_sources[1], buffers.canonical_bytes(active_buf),
      "prediction re-read the latest buffer after identity became ready")
  end)

  case("fim-manual-first-request-waits-for-identity-without-extra-command", function()
    fresh_model_identity()
    open_buffer("manual state")
    local service = new_service("hold", "auto")
    setup("manual", 5)
    assert_true(predict.predict_explicit(), "Alt+p request was queued during identity startup")
    assert_eq(service.health_calls, 1, "manual request joined setup identity refresh")
    assert_eq(service.context_calls, 0, "manual request waits for the tokenizer contract")
    service:resolve_health(1, true)
    assert_true(vim.wait(1500, function() return service.completion_calls == 1 end),
      "the queued manual request started after identity validation")
    assert_eq(service.completion_sources[1], buffers.canonical_bytes(active_buf),
      "manual prediction uses the current buffer state")
  end)

  case("fim-manual-bootstrap-cancels-request-after-buffer-navigation", function()
    fresh_model_identity()
    open_buffer("requested file")
    local requested_buf = active_buf
    local service = new_service("hold", "auto")
    setup("manual", 5)
    assert_true(predict.predict_explicit(), "Alt+p request was queued during identity startup")
    open_buffer("different current file")
    assert_true(active_buf ~= requested_buf, "navigation selected another buffer")
    service:resolve_health(1, true)
    vim.wait(100, function() return false end)
    assert_eq(service.context_calls, 0, "stale explicit request did not use the new buffer")
    assert_eq(service.completion_calls, 0, "stale explicit request did not start generation")
    assert_true(predict.status().state:find("canceled after editor navigation", 1, true) ~= nil,
      "status explains why the queued manual request was canceled")
  end)

  case("fim-off-prevents-a-delayed-identity-callback-from-starting-prediction", function()
    fresh_model_identity()
    open_buffer("off state")
    local service = new_service("auto", "hold")
    setup("automatic", 5)
    assert_true(vim.wait(1000, function() return service.tokenizer_calls == 1 end),
      "health identity reached the tokenizer fetch")
    assert_true(predict.set_mode("off"), "off mode selected during tokenizer fetch")
    service:resolve_tokenizer(1, true)
    vim.wait(100, function() return false end)
    assert_eq(predict.status().mode, "off")
    assert_eq(service.context_calls, 0, "off callback did not create a context request")
    assert_eq(service.completion_calls, 0, "off callback did not restart prediction")
    assert_eq(service.health_calls, 1, "off callback did not retry identity refresh")
  end)

  case("fim-stale-setup-waits-for-old-fetch-then-refreshes-current-setup", function()
    fresh_model_identity()
    open_buffer("before new setup")
    local service = new_service("auto", "hold")
    setup("automatic", 5)
    assert_true(vim.wait(1000, function() return service.tokenizer_calls == 1 end),
      "first setup reached tokenizer fetch")
    setup("automatic", 5)
    assert_eq(service.health_calls, 1, "new setup waited for the active identity request")
    vim.api.nvim_buf_set_lines(active_buf, 0, 1, false, { "current setup state" })
    service.health_mode = "hold"
    service:resolve_tokenizer(1, true)
    assert_true(vim.wait(1000, function() return service.health_calls == 2 end),
      "stale fetch completed before the new setup refreshed identity")
    assert_eq(predict.status().model_alias, nil,
      "stale tokenizer callback did not install identity for the new setup")
    assert_eq(service.context_calls, 0, "stale setup did not start an editor request")
    service:resolve_health(1, true)
    assert_true(vim.wait(1500, function() return service.completion_calls == 1 end),
      "current setup predicted after its identity refresh")
    assert_eq(service.health_calls, 2, "only one refresh ran at a time")
    assert_eq(service.completion_sources[1], buffers.canonical_bytes(active_buf),
      "current setup read its latest buffer state")
  end)

  case("fim-identity-outage-uses-bounded-retry-and-explicit-retry", function()
    fresh_model_identity()
    open_buffer("offline state")
    local service = new_service("fail", "auto")
    setup("manual", 5)
    assert_true(vim.wait(4000, function()
      return predict.status().state == "model identity unavailable; press Alt+p to retry"
    end), "identity outage reached its bounded terminal status")
    assert_eq(service.health_calls, 3, "initial health plus two bounded retries")
    vim.wait(700, function() return false end)
    assert_eq(service.health_calls, 3, "offline startup does not loop")
    assert_true(predict.predict_explicit(), "Alt+p explicitly restarted identity discovery")
    vim.wait(50, function() return false end)
    assert_eq(service.health_calls, 4, "explicit retry started one new identity request")
    assert_true(predict.status().state:find("model identity unavailable", 1, true) ~= nil,
      "status retains a bounded identity error")
  end)

  case("fim-plugin-entry-preserves-protocol-and-registers-commands", function()
    fresh_model_identity()
    open_buffer("plugin entry state")
    local service = new_service("auto", "auto")
    setup("manual", 5)
    assert_true(vim.wait(1000, function()
      return predict.status().model_alias == "q25-fim-synthetic"
    end), "FIM identity is ready before the plugin entry reload")
    assert_eq(predict.status().protocol_version, fim.WIRE_VERSION)

    local plugin_path = vim.api.nvim_get_runtime_file("plugin/tabcomplete_predict.lua", false)[1]
    assert_true(plugin_path ~= nil, "prediction plugin entry is on the runtime path")
    dofile(plugin_path)

    local status = predict.status()
    assert_eq(status.protocol_version, fim.WIRE_VERSION,
      "no-argument setup retained the selected FIM protocol")
    assert_eq(status.model_protocol, fim.WIRE_VERSION,
      "no-argument setup retained the validated FIM identity")
    assert_eq(status.selected_model, "q25-fim-synthetic",
      "no-argument setup retained the selected model")
    assert_eq(service.health_calls, 1, "plugin setup did not start a second identity refresh")
    assert_eq(service.tokenizer_calls, 1, "plugin setup retained the validated tokenizer contract")
    assert_true(vim.api.nvim_get_commands({}).TabCompletePredict ~= nil,
      "plugin entry registered the explicit prediction command")
    assert_true(vim.api.nvim_get_commands({}).TabCompleteStatus ~= nil,
      "plugin entry registered the status command")
  end)

  cleanup()
  util.current_mode = previous_current_mode
end
