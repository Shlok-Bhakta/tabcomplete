return function(ok, assert_eq, assert_true)
  local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
  vim.opt.rtp:prepend(root)
  local collector = require("tabcomplete_trajectory")
  local predict = require("tabcomplete_trajectory.predict")
  local buffers = require("tabcomplete_trajectory.buffers")
  local adapter = require("tabcomplete_trajectory.single_line_v1")

  ok("single-line-v1-predict-path-is-explicit-and-legacy-default-remains", function()
    assert_eq(predict.status().protocol_version, "compact-next-edit-v1")
    predict.setup({ protocol_version = adapter.WIRE_VERSION, mode = "manual", synthetic = true,
      model = "stub", model_revision = "stub", url = "http://127.0.0.1:9" })
    assert_eq(predict.status().protocol_version, adapter.WIRE_VERSION)
    assert_eq(predict.status().context_policy_version, adapter.CONTEXT_POLICY_VERSION)
    assert_eq(predict.status().mode, "manual")
  end)

  collector.started = true
  collector.disabled = false
  collector.paused = false
  collector.session_id = "synthetic-v1-test-session"
  collector.queue = {}
  collector.seq = 0
  buffers.set_callbacks(function(buf, kind, payload)
    return collector.emit(buf, kind, payload)
  end, function() end)

  local function set_model_action(wire, stop_type, predicted)
    local token_prompts = {}
    predict._tokenize_impl = function(prompt, add_special, parse_special, callback)
      assert_true(add_special, "single-line contract must count tokenizer special tokens")
      assert_true(parse_special, "single-line contract must parse FIM and control markers")
      token_prompts[#token_prompts + 1] = prompt
      local tokens = {}
      for i = 1, #prompt + 1 do tokens[i] = i end
      callback({ code = 0, stdout = vim.json.encode({ tokens = tokens }) })
      return { kill = function() end }
    end
    predict._request_impl = function(state, body, callback)
      local request = vim.json.decode(body)
      assert_eq(request.n_predict, 64, "single-line action cap")
      assert_eq(request.prompt, token_prompts[#token_prompts], "generated prompt was token-counted")
      assert_eq(state.prompt_tokens, #request.prompt + 1, "exact test tokenizer count")
      local events = {
        "data: " .. vim.json.encode({ content = wire, stop = false }),
        "data: " .. vim.json.encode({ content = "", stop = true,
          stop_type = stop_type or "eos", tokens_predicted = predicted or 1 }),
      }
      callback({ code = 0, stdout = table.concat(events, "\n\n") .. "\n\n" })
      return { kill = function() end }
    end
    return token_prompts
  end

  local path = vim.fn.tempname() .. ".py"
  local file = io.open(path, "wb")
  assert_true(file ~= nil, "temp file opened")
  file:write("before\nold = 'α'\nafter\n")
  file:close()
  vim.cmd("edit " .. vim.fn.fnameescape(path))
  vim.bo.filetype = "python"
  local buf = vim.api.nvim_get_current_buf()
  vim.api.nvim_set_option_value("fileformat", "unix", { buf = buf })
  vim.api.nvim_set_option_value("eol", true, { buf = buf })
  assert_true(buffers.attach(buf), "prediction buffer attached")
  vim.api.nvim_win_set_cursor(0, { 2, 8 })

  ok("single-line-v1-request-records-exact-prompt-and-history-before-display", function()
    vim.api.nvim_buf_set_lines(buf, 1, 2, false, { "old = 'β'" })
    vim.api.nvim_win_set_cursor(0, { 2, 9 })
    local prompts = set_model_action("R\tnew = 'λ'", "eos", 6)
    assert_true(predict.predict(), "manual v1 request accepted")
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end),
      "completed suggestion became visible")
    assert_true(#prompts >= 1, "model request required one or more exact tokenizer calls")
    assert_true(prompts[#prompts]:find("<single-line-edit-v1>", 1, true) ~= nil, "v1 prompt header")
    assert_true(prompts[#prompts]:find('old="old = \'α\'" new="old = \'β\'"', 1, true) ~= nil,
      "actual buffer delta included in history")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'",
      "proposal does not mutate source")
    local request_event
    for _, event in ipairs(collector.queue) do
      if event.event_type == "prediction_requested" then request_event = event end
    end
    assert_true(request_event ~= nil, "request event recorded")
    assert_eq(request_event.payload.wire_version, adapter.WIRE_VERSION)
    assert_eq(request_event.payload.context_policy_version, adapter.CONTEXT_POLICY_VERSION)
    assert_eq(request_event.payload.max_output_tokens, 64)
    assert_eq(request_event.payload.editable_range.start_col, 0)
    assert_eq(request_event.payload.editable_range.end_col, #"old = 'β'")
    assert_true(request_event.payload.context_hash == require("tabcomplete_trajectory.util")
      .sha256hex(prompts[#prompts]), "recorded context hash matches exact generation prompt")
    local shown_event
    for _, event in ipairs(collector.queue) do
      if event.event_type == "prediction_shown" then shown_event = event end
    end
    assert_true(shown_event ~= nil, "replacement display range recorded")
    assert_eq(shown_event.payload.proposed_start_byte, 7, "replacement source byte start")
    assert_eq(shown_event.payload.proposed_end_byte, 17, "replacement source byte end")
    assert_eq(shown_event.payload.proposed_start.row, 1, "replacement start row")
    assert_eq(shown_event.payload.proposed_end.row, 1, "replacement end row")
    local scratch = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_delete(scratch, { force = true })
    assert_true(predict.status().proposal_active, "unrelated scratch deletion preserves proposal")
    local accepted_id = shown_event.payload.prediction_id
    assert_true(predict.accept(), "explicit acceptance applied replacement")
    local accepted_count = 0
    for _, event in ipairs(collector.queue) do
      if event.payload.prediction_id == accepted_id then
        assert_true(event.event_type ~= "prediction_dismissed",
          "acceptance validation must not record navigation dismissal")
        if event.event_type == "prediction_accepted" then accepted_count = accepted_count + 1 end
      end
    end
    assert_eq(accepted_count, 1, "acceptance records one terminal decision")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "new = 'λ'")
    vim.cmd("silent undo")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'",
      "acceptance undo restores prior text only")
  end)

  ok("rust-editor-context-rpc-validates-identity-and-switches-model-safely", function()
    local util = require("tabcomplete_trajectory.util")
    local repository = require("tabcomplete_trajectory.repository")
    local prior_repo = vim.deepcopy(repository.cache)
    local rust_root = vim.fn.tempname() .. "-rust-context"
    vim.fn.mkdir(rust_root, "p")
    vim.api.nvim_buf_set_name(buf, rust_root .. "/main.py")
    buffers.refresh_shadow(buf)
    repository.cache.root, repository.cache.head, repository.cache.root_name = rust_root, "test-head", "rust-context"
    local peers = {}
    for i = 1, 10 do
      local peer = vim.api.nvim_create_buf(true, false)
      vim.api.nvim_buf_set_name(peer, rust_root .. "/context-" .. i .. ".py")
      vim.api.nvim_buf_set_lines(peer, 0, -1, false, { "def helper_" .. i .. "():", "    return " .. i })
      vim.bo[peer].filetype = "python"
      assert_true(buffers.attach(peer), "same-repository peer attached")
      peers[#peers + 1] = peer
    end
    local outside = vim.api.nvim_create_buf(true, false)
    vim.api.nvim_buf_set_name(outside, rust_root .. "-other/context.py")
    vim.api.nvim_buf_set_lines(outside, 0, -1, false, { "def outside():" })
    assert_true(buffers.attach(outside), "outside peer attached for scope check")
    peers[#peers + 1] = outside
    local excluded = vim.api.nvim_create_buf(true, false)
    vim.api.nvim_buf_set_name(excluded, rust_root .. "/.env")
    vim.api.nvim_buf_set_lines(excluded, 0, -1, false, { "SECRET=fixture" })
    assert_true(not buffers.attach(excluded), "excluded peer is not tracked")
    peers[#peers + 1] = excluded
    local oversized = vim.api.nvim_create_buf(true, false)
    vim.api.nvim_buf_set_name(oversized, rust_root .. "/oversized.py")
    vim.api.nvim_buf_set_lines(oversized, 0, -1, false, { string.rep("x", 65537) })
    assert_true(buffers.attach(oversized), "oversized peer attached for cap check")
    peers[#peers + 1] = oversized

    local allowed_models = {
      q25 = { model_sha256 = "4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb",
        model_protocol = "single-line-edit-v1", output_tokens = 64 },
      sweep = { model_sha256 = "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4",
        model_protocol = "sweep-full-file-v1", output_tokens = 192 },
    }
    local q25_identity = { status = "ok", alias = "q25", model_sha256 = allowed_models.q25.model_sha256,
      model_protocol = allowed_models.q25.model_protocol, runtime_config_hash = string.rep("a", 64),
      input_tokens = 1024, context_size = 2304, output_tokens = 64 }
    local sweep_identity = { status = "ok", alias = "sweep", model_sha256 = allowed_models.sweep.model_sha256,
      model_protocol = allowed_models.sweep.model_protocol, runtime_config_hash = string.rep("b", 64),
      input_tokens = 1024, context_size = 2304, output_tokens = 192 }
    local response_layout_override
    local switch_alias = "sweep"
    local queue_start = #collector.queue
    predict.setup({ backend = "rust-editor-v1", url = "http://127.0.0.1:19094", mode = "manual",
      synthetic = true, allowed_models = allowed_models, single_line_input_tokens = 1024 })
    assert_eq(predict.status().protocol_version, adapter.WIRE_VERSION,
      "Rust backend retains canonical single-line action codec")
    vim.api.nvim_win_set_cursor(0, { 2, 0 })
    local raw_sse, context_attempts, completion_attempts = nil, 0, 0
    predict._tokenize_impl = function() error("Rust path must not call Lua tokenizer") end
    predict._backend_post_impl = function(endpoint, body, callback)
      if endpoint == "/v1/editor/context" then
        context_attempts = context_attempts + 1
        assert_eq(body.repository_identity, rust_root .. ":test-head", "repository identity sent")
        assert_eq(body.state.file_id, rust_root .. "/main.py", "exact editor state sent")
        assert_eq(#body.buffers, 8, "context buffer count capped at eight")
        for _, context_buffer in ipairs(body.buffers) do
          assert_true(context_buffer.path:match("^context%-%d+%.py$") ~= nil,
            "only same-repository, non-excluded buffers are sent")
          assert_true(#context_buffer.source <= 65536, "context source stays under byte cap")
          assert_true(type(context_buffer.recency) == "number", "context recency is explicit")
        end
        if context_attempts == 1 then
          callback(false, { http_code = 409 })
          return { kill = function() end }
        end
        local is_sweep = predict.status().model_alias == "sweep"
        local prompt = is_sweep and "<sweep-context-fixture>" or "<rust-context-fixture>"
        local identity = is_sweep and sweep_identity or q25_identity
        local layout = identity.context_layout
          or (is_sweep and "sweep-window-v1" or "trained-v2")
        local policy = ({ ["trained-v2"] = "single-line-context-v2",
          ["cursor-last-v1"] = "single-line-cursor-last-context-v1",
          ["sweep-window-v1"] = "sweep-window-context-v1" })[layout]
        local context_response = { prompt = prompt, prompt_tokens = 13, context_hash = util.sha256hex(prompt),
          context_policy_version = policy,
          model_protocol = identity.model_protocol,
          window = is_sweep and { source = body.state.source, target_row = body.state.target_row, start_row = 0 }
            or vim.NIL,
          selected_buffers = { body.buffers[1].path }, model_identity = identity }
        if identity.context_layout ~= nil then context_response.context_layout = identity.context_layout end
        if response_layout_override ~= nil then context_response.context_layout = response_layout_override end
        callback(true, context_response)
      elseif endpoint == "/v1/model" then
        assert_eq(predict.status().mode, "off", "mode is off before picker request")
        assert_eq(body.alias, switch_alias, "model alias request")
        local fields = 0
        for key in pairs(body) do
          fields = fields + 1
          assert_eq(key, "alias", "model switch sends only the allowlisted alias")
        end
        assert_eq(fields, 1, "model switch has no extra request fields")
        callback(true, switch_alias == "sweep" and sweep_identity or q25_identity)
      else
        error("unexpected Rust endpoint " .. tostring(endpoint))
      end
      return { kill = function() end }
    end
    predict._request_impl = function(state, body, callback)
      completion_attempts = completion_attempts + 1
      local completion = vim.json.decode(body)
      assert_eq(completion.repository_identity, rust_root .. ":test-head", "completion cache identity sent")
      assert_eq(completion.cache_prompt, true, "prompt cache enabled")
      assert_eq(completion.temperature, nil, "legacy sampling fields omitted")
      assert_eq(completion.stream, nil, "Rust streaming uses the SSE response, not a body flag")
      local is_sweep = predict.status().model_alias == "sweep"
      if is_sweep then
        assert_eq(completion.prompt, "<sweep-context-fixture>", "Rust-owned Sweep prompt sent unchanged")
        assert_eq(completion.n_predict, 192, "configured Sweep output cap sent")
        assert_eq(completion.window.source, state.source, "validated exact Sweep window sent")
        assert_eq(completion.window.target_row, state.row, "Sweep target row is window-relative")
        assert_eq(completion.window.start_row, 0, "Sweep window starting row sent")
        local lines = adapter.physical_lines(completion.window.source)
        lines[completion.window.target_row + 1].content = "sweep"
        local full_file = table.concat(vim.tbl_map(function(line)
          return line.content .. line.terminator
        end, lines))
        local events = {
          "data: " .. vim.json.encode({ content = full_file, stop = false,
            tokens = { 42, 43, 44, 45, 46, 47, 48, 49, 50 } }),
          "data: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos", tokens_predicted = 9,
            canonical_action = { kind = "replace_line", text = "sweep" },
            model_protocol = "sweep-full-file-v1", model_sha256 = sweep_identity.model_sha256,
            context_layout = "sweep-window-v1",
            timings = { cache_n = 0, prompt_n = 13, prompt_ms = 1, predicted_n = 9,
              predicted_ms = 3, total_ms = 4 } }),
        }
        raw_sse = table.concat(events, "\n\n") .. "\n\n"
        callback({ code = 0, stdout = raw_sse })
        return { kill = function() end }
      end
      assert_eq(completion.prompt, "<rust-context-fixture>", "Rust-owned prompt sent unchanged")
      assert_eq(completion.n_predict, 64, "configured Qwen output cap sent")
      if completion_attempts == 1 then
        callback({ code = 22, stdout = "" }) -- curl reports HTTP 409 as a bounded failure.
        return { kill = function() end }
      end
      local events = {
        "data: " .. vim.json.encode({ content = "R\tbackend", stop = false, tokens = { 17 } }),
        "data: " .. vim.json.encode({ content = "", stop = false, tokens = { 18, 19 } }),
        "data: " .. vim.json.encode({ content = "", stop = true, stop_type = "eos", tokens_predicted = 3,
          canonical_action = { kind = "replace_line", text = "backend" },
          model_protocol = "single-line-edit-v1", model_sha256 = q25_identity.model_sha256,
          context_layout = q25_identity.context_layout,
          timings = { cache_n = 2, prompt_n = 11, prompt_ms = 1, predicted_n = 3,
            predicted_ms = 2, total_ms = 3 } }),
      }
      raw_sse = table.concat(events, "\n\n") .. "\n\n"
      callback({ code = 0, stdout = raw_sse })
      return { kill = function() end }
    end
    assert_true(predict.predict(), "Rust backend prediction accepted")
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end),
      "Rust response became a proposal")
    assert_eq(context_attempts, 2, "busy context request retried once before generation")
    assert_eq(completion_attempts, 2, "busy generation slot retried without duplicate request event")
    local prediction_id, requested, generated
    for i = queue_start + 1, #collector.queue do
      local event = collector.queue[i]
      if event.event_type == "prediction_shown" and not prediction_id then
        prediction_id = event.payload.prediction_id
      end
    end
    assert_true(prediction_id ~= nil, "Rust proposal has a prediction id")
    for _, event in ipairs(collector.queue) do
      if event.payload.prediction_id == prediction_id then
        if event.event_type == "prediction_requested" then requested = event end
        if event.event_type == "prediction_generated" then generated = event end
      end
    end
    assert_true(requested ~= nil and generated ~= nil, "Rust request and generation evidence recorded")
    assert_eq(requested.payload.model_alias, "q25")
    assert_eq(requested.payload.model_protocol, "single-line-edit-v1")
    assert_eq(requested.payload.context_layout, "trained-v2",
      "legacy identity and context response infer the trained-v2 layout")
    assert_eq(requested.payload.wire_version, adapter.WIRE_VERSION,
      "canonical action codec remains separate from model protocol")
    assert_eq(requested.payload.context_policy_version, "single-line-context-v2")
    assert_eq(requested.payload.max_output_tokens, 64)
    assert_eq(requested.payload.runtime_config_hash, q25_identity.runtime_config_hash)
    assert_eq(generated.payload.model_protocol, "single-line-edit-v1")
    assert_eq(generated.payload.max_output_tokens, 64)
    assert_eq(generated.payload.first_token_observation, "sampled_token_ids")
    assert_true(generated.payload.first_token_at_ms ~= nil, "sampled token timing recorded")
    assert_true(generated.payload.first_text_at_ms ~= nil, "first visible text timing distinguished")
    assert_eq(generated.payload.context_layout, "trained-v2", "generated event records effective layout")
    assert_eq(generated.payload.raw_response_hash, util.sha256hex(raw_sse),
      "raw SSE response, including terminal metadata, is retained")

    assert_eq(predict.status().context_layout, "trained-v2", "legacy effective layout is visible in status")
    assert_eq(predict.status().context_policy_version, "single-line-context-v2")
    assert_true(predict.set_mode("off"), "legacy proposal cancelled before layout cases")
    assert_true(predict.set_mode("manual"), "manual mode restored for layout cases")

    local cursor_queue_start = #collector.queue
    q25_identity.context_layout = "cursor-last-v1"
    assert_true(predict.predict(), "cursor-last context request accepted")
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end),
      "cursor-last response became a proposal")
    local cursor_request, cursor_generated
    for i = cursor_queue_start + 1, #collector.queue do
      local event = collector.queue[i]
      if event.event_type == "prediction_requested" then cursor_request = event end
      if event.event_type == "prediction_generated" then cursor_generated = event end
    end
    assert_true(cursor_request ~= nil and cursor_generated ~= nil, "cursor-last events recorded")
    assert_eq(cursor_request.payload.context_layout, "cursor-last-v1")
    assert_eq(cursor_request.payload.context_policy_version, "single-line-cursor-last-context-v1")
    assert_eq(cursor_generated.payload.context_layout, "cursor-last-v1")
    assert_eq(cursor_generated.payload.context_policy_version, "single-line-cursor-last-context-v1")
    assert_eq(predict.status().context_layout, "cursor-last-v1")
    assert_eq(predict.status().context_policy_version, "single-line-cursor-last-context-v1")
    assert_true(predict.set_mode("off"), "cursor-last proposal cancelled before validation failures")
    assert_true(predict.set_mode("manual"), "manual mode restored after cursor-last case")

    local invalid_before = predict.status().counters.invalid_output
    response_layout_override = "trained-v2"
    assert_true(predict.predict(), "mismatched response layout request accepted")
    assert_true(vim.wait(1000, function() return not predict.status().in_flight end),
      "mismatched response layout rejected")
    assert_eq(predict.status().counters.invalid_output, invalid_before + 1,
      "response layout must match the validated identity")
    response_layout_override = nil

    q25_identity.context_layout = "unknown-layout-v9"
    assert_true(predict.predict(), "unknown identity layout request accepted")
    assert_true(vim.wait(1000, function() return not predict.status().in_flight end),
      "unknown identity layout rejected")
    assert_eq(predict.status().counters.invalid_output, invalid_before + 2,
      "unknown model layout cannot pass identity validation")
    q25_identity.context_layout = "cursor-last-v1"

    local switched, switch_error
    assert_true(predict.set_model("sweep", function(ok, err)
      switched, switch_error = ok, err
    end), "allowlisted model switch started")
    assert_eq(predict.status().mode, "off", "switch cancels proposals before model loading")
    assert_true(vim.wait(1000, function() return switched ~= nil end), "model switch completed")
    assert_true(switched, tostring(switch_error))
    assert_eq(predict.status().mode, "manual", "previous mode restored after switch")
    assert_eq(predict.status().model_alias, "sweep")
    assert_eq(predict.status().model_protocol, "sweep-full-file-v1")
    assert_eq(predict.status().context_layout, "sweep-window-v1")
    assert_eq(predict.status().context_policy_version, "sweep-window-context-v1")
    assert_true(not predict.status().automatic_quality_validated,
      "model switching does not mark quality validated")
    local terminal_count, terminal = 0, nil
    local accepted_count, rejected_count = 0, 0
    for i = queue_start + 1, #collector.queue do
      local event = collector.queue[i]
      if event.payload.prediction_id == prediction_id then
        if event.event_type == "prediction_dismissed" or event.event_type == "prediction_accepted"
            or event.event_type == "prediction_partially_accepted" or event.event_type == "prediction_rejected" then
          terminal_count, terminal = terminal_count + 1, event
        end
        if event.event_type == "prediction_accepted" then accepted_count = accepted_count + 1 end
        if event.event_type == "prediction_rejected" then rejected_count = rejected_count + 1 end
      end
    end
    assert_eq(terminal_count, 1, "switch records one terminal proposal cancellation")
    assert_eq(terminal.event_type, "prediction_dismissed")
    assert_eq(terminal.payload.outcome, "cancelled_by_mode")
    assert_eq(accepted_count, 0, "model switch is not acceptance")
    assert_eq(rejected_count, 0, "model switch is not rejection")

    local sweep_queue_start = #collector.queue
    assert_true(predict.predict(), "Sweep backend prediction accepted")
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end),
      "Sweep canonical action became a proposal")
    assert_eq(context_attempts, 6, "Sweep context request used the backend endpoint")
    assert_eq(completion_attempts, 4, "Sweep generation completed once")
    local sweep_id, sweep_request, sweep_generated
    for i = sweep_queue_start + 1, #collector.queue do
      local event = collector.queue[i]
      if event.event_type == "prediction_shown" then sweep_id = event.payload.prediction_id end
    end
    for i = sweep_queue_start + 1, #collector.queue do
      local event = collector.queue[i]
      if event.payload.prediction_id == sweep_id then
        if event.event_type == "prediction_requested" then sweep_request = event end
        if event.event_type == "prediction_generated" then sweep_generated = event end
      end
    end
    assert_true(sweep_request ~= nil and sweep_generated ~= nil, "Sweep evidence recorded")
    assert_eq(sweep_request.payload.model_alias, "sweep")
    assert_eq(sweep_request.payload.model_protocol, "sweep-full-file-v1")
    assert_eq(sweep_request.payload.context_policy_version, "sweep-window-context-v1")
    assert_eq(sweep_request.payload.context_layout, "sweep-window-v1")
    assert_eq(sweep_request.payload.wire_version, adapter.WIRE_VERSION)
    assert_eq(sweep_request.payload.max_output_tokens, 192)
    assert_eq(sweep_generated.payload.model_protocol, "sweep-full-file-v1")
    assert_eq(sweep_generated.payload.raw_response_hash, util.sha256hex(raw_sse),
      "raw Sweep full-file SSE response retained")
    assert_true(predict.set_mode("off"), "Sweep proposal cancelled before teardown")

    switch_alias = "sweep"
    sweep_identity.context_layout = "sweep-window-v1"
    local sweep_layout_switched, sweep_layout_error
    assert_true(predict.set_model("sweep", function(ok, err)
      sweep_layout_switched, sweep_layout_error = ok, err
    end), "explicit Sweep layout switch started")
    assert_true(vim.wait(1000, function() return sweep_layout_switched ~= nil end),
      "explicit Sweep layout switch completed")
    assert_true(sweep_layout_switched, tostring(sweep_layout_error))
    assert_eq(predict.status().context_layout, "sweep-window-v1")
    assert_eq(predict.status().context_policy_version, "sweep-window-context-v1")

    sweep_identity.context_layout = "unknown-layout-v9"
    local bad_sweep_switch, bad_sweep_error
    assert_true(predict.set_model("sweep", function(ok, err)
      bad_sweep_switch, bad_sweep_error = ok, err
    end), "unknown Sweep layout switch request started")
    assert_true(vim.wait(1000, function() return bad_sweep_switch ~= nil end),
      "unknown Sweep layout switch completed")
    assert_true(not bad_sweep_switch and bad_sweep_error ~= nil, "unknown Sweep layout rejected")
    assert_eq(predict.status().context_layout, "sweep-window-v1",
      "failed switch retains previous validated Sweep layout")
    sweep_identity.context_layout = nil

    switch_alias = "q25"
    local q25_switched, q25_switch_error
    assert_true(predict.set_model("q25", function(ok, err)
      q25_switched, q25_switch_error = ok, err
    end), "cursor-last Q25 model switch started")
    assert_true(vim.wait(1000, function() return q25_switched ~= nil end),
      "cursor-last Q25 model switch completed")
    assert_true(q25_switched, tostring(q25_switch_error))
    assert_eq(predict.status().model_alias, "q25")
    assert_eq(predict.status().context_layout, "cursor-last-v1")
    assert_eq(predict.status().context_policy_version, "single-line-cursor-last-context-v1",
      "model switch derives policy from the validated identity layout")

    predict._backend_post_impl = nil
    predict._request_impl = nil
    predict._tokenize_impl = nil
    predict.setup({ backend = "llama-cpp-legacy", protocol_version = adapter.WIRE_VERSION,
      url = "http://127.0.0.1:9", mode = "manual", synthetic = true,
      allowed_models = allowed_models, automatic_prefix_guard = false })
    for _, peer in ipairs(peers) do
      if vim.api.nvim_buf_is_valid(peer) then vim.api.nvim_buf_delete(peer, { force = true }) end
    end
    repository.cache = prior_repo
    vim.api.nvim_buf_set_name(buf, path)
    buffers.refresh_shadow(buf)
    vim.api.nvim_win_set_cursor(0, { 2, 0 })
  end)

  ok("rust-specific-mode-state-persists-off-without-reading-legacy-state", function()
    local mode_path = vim.fn.tempname() .. "-rust-mode.json"
    predict.setup({ backend = "rust-editor-v1", protocol_version = adapter.WIRE_VERSION,
      mode = "manual", persist_mode = true, mode_state_path = mode_path, synthetic = true })
    assert_true(predict.set_mode("off"), "off mode set")
    predict.setup({ backend = "rust-editor-v1", protocol_version = adapter.WIRE_VERSION,
      mode = "manual", persist_mode = true, mode_state_path = mode_path, synthetic = true })
    assert_eq(predict.status().mode, "off", "off mode restored from Rust-specific state file")
    assert_true(predict.set_mode("manual"), "manual mode restored for remaining tests")
    predict.setup({ backend = "llama-cpp-legacy", protocol_version = adapter.WIRE_VERSION,
      mode = "manual", persist_mode = false, mode_state_path = nil, synthetic = true })
    os.remove(mode_path)
  end)

  ok("rust-sse-requires-eos-model-match-and-canonical-single-line-action", function()
    local sse = require("tabcomplete_trajectory.sse")
    local function parser_for(raw, terminal)
      local parser = sse.new()
      local content = "data: " .. vim.json.encode({ content = raw, stop = false }) .. "\n\n"
      local finish = "data: " .. vim.json.encode(terminal) .. "\n\n"
      assert_true(sse.feed(parser, content), "raw SSE content accepted")
      assert_true(sse.feed(parser, finish), "SSE terminal accepted")
      return parser
    end
    local q25 = { model_sha256 = "4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb",
      model_protocol = "single-line-edit-v1", output_tokens = 64 }
    local qwen_terminal = { content = "", stop = true, stop_type = "eos", tokens_predicted = 3,
      model_protocol = q25.model_protocol, model_sha256 = q25.model_sha256,
      canonical_action = { kind = "replace_line", text = "same" } }
    local action = assert(sse.finish_rust(parser_for("R\tsame", qwen_terminal), { model_identity = q25 }))
    assert_eq(action.kind, "replace_line")
    assert_eq(action.text, "same")

    local mismatch = vim.deepcopy(qwen_terminal)
    mismatch.canonical_action = { kind = "replace_line", text = "different" }
    assert_true(not sse.finish_rust(parser_for("R\tsame", mismatch), { model_identity = q25 }),
      "Qwen canonical action must match raw wire decoder")
    local cutoff = vim.deepcopy(qwen_terminal)
    cutoff.stop_type = "limit"
    assert_true(not sse.finish_rust(parser_for("R\tsame", cutoff), { model_identity = q25 }),
      "Rust terminal must report EOS")
    local wrong_model = vim.deepcopy(qwen_terminal)
    wrong_model.model_sha256 = string.rep("0", 64)
    assert_true(not sse.finish_rust(parser_for("R\tsame", wrong_model), { model_identity = q25 }),
      "Rust terminal digest must match context identity")

    local cursor_last_q25 = vim.deepcopy(q25)
    cursor_last_q25.context_layout = "cursor-last-v1"
    local cursor_last_terminal = vim.deepcopy(qwen_terminal)
    cursor_last_terminal.context_layout = "cursor-last-v1"
    assert_true(sse.finish_rust(parser_for("R\tsame", cursor_last_terminal),
      { model_identity = cursor_last_q25 }) ~= nil,
      "explicit Q25 identity and terminal layouts match")
    local missing_layout_terminal = vim.deepcopy(cursor_last_terminal)
    missing_layout_terminal.context_layout = nil
    assert_true(not sse.finish_rust(parser_for("R\tsame", missing_layout_terminal),
      { model_identity = cursor_last_q25 }),
      "explicit Q25 identity requires a terminal layout")
    local wrong_layout_terminal = vim.deepcopy(cursor_last_terminal)
    wrong_layout_terminal.context_layout = "trained-v2"
    assert_true(not sse.finish_rust(parser_for("R\tsame", wrong_layout_terminal),
      { model_identity = cursor_last_q25 }),
      "explicit Q25 identity rejects a different terminal layout")

    local legacy_validated_q25 = vim.deepcopy(q25)
    legacy_validated_q25.context_layout = "trained-v2"
    legacy_validated_q25.context_layout_legacy = true
    assert_true(sse.finish_rust(parser_for("R\tsame", qwen_terminal),
      { model_identity = legacy_validated_q25 }) ~= nil,
      "legacy validated identity still accepts an older terminal without layout")
    local inconsistent_legacy_terminal = vim.deepcopy(qwen_terminal)
    inconsistent_legacy_terminal.context_layout = "cursor-last-v1"
    assert_true(not sse.finish_rust(parser_for("R\tsame", inconsistent_legacy_terminal),
      { model_identity = legacy_validated_q25 }),
      "a supplied terminal layout still matches the normalized legacy layout")

    local sweep = { model_sha256 = "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4",
      model_protocol = "sweep-full-file-v1", output_tokens = 192 }
    local full_file = "before\nnew\nafter\n"
    local sweep_terminal = { content = "", stop = true, stop_type = "eos", tokens_predicted = 9,
      model_protocol = sweep.model_protocol, model_sha256 = sweep.model_sha256,
      canonical_action = { kind = "replace_line", text = "new" } }
    local sweep_action, raw = sse.finish_rust(parser_for(full_file, sweep_terminal), {
      model_identity = sweep, window = { source = "before\nold\nafter\n", target_row = 1, start_row = 0 } })
    assert_eq(sweep_action.kind, "replace_line")
    assert_eq(sweep_action.text, "new")
    assert_eq(raw, full_file, "Sweep raw full-file response remains untrimmed")
    assert_true(not sse.finish_rust(parser_for(full_file, sweep_terminal), { model_identity = sweep }),
      "Sweep action requires its validated window")

    local explicit_sweep = vim.deepcopy(sweep)
    explicit_sweep.context_layout = "sweep-window-v1"
    local explicit_sweep_terminal = vim.deepcopy(sweep_terminal)
    explicit_sweep_terminal.context_layout = "sweep-window-v1"
    assert_true(sse.finish_rust(parser_for(full_file, explicit_sweep_terminal), {
      model_identity = explicit_sweep,
      window = { source = "before\nold\nafter\n", target_row = 1, start_row = 0 },
    }) ~= nil, "explicit Sweep identity and terminal layouts match")
    explicit_sweep_terminal.context_layout = nil
    assert_true(not sse.finish_rust(parser_for(full_file, explicit_sweep_terminal), {
      model_identity = explicit_sweep,
      window = { source = "before\nold\nafter\n", target_row = 1, start_row = 0 },
    }), "explicit Sweep identity requires a terminal layout")

    local token_only = sse.new()
    assert_true(sse.feed(token_only, "data: " .. vim.json.encode({ content = "", stop = false,
      tokens = { 99 } }) .. "\n\n"), "sampled token with no UTF-8 text fragment is valid")
    assert_true(token_only.first_token, "token ID records first sampled token")
    assert_true(not token_only.first_text, "empty fragment does not claim observed text")
    assert_eq(token_only.first_token_source, "sampled_token_ids")
    assert_eq(token_only.model_tokens, 1)
  end)

  ok("single-line-v1-manual-normal-cursor-navigation-dismisses-once", function()
    vim.api.nvim_win_set_cursor(0, { 2, 0 })
    set_model_action("R\tnew", "eos", 3)
    assert_true(predict.predict(), "manual v1 navigation request accepted")
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end),
      "proposal became visible before navigation")
    local prediction_id
    for i = #collector.queue, 1, -1 do
      local event = collector.queue[i]
      if event.event_type == "prediction_shown" then
        prediction_id = event.payload.prediction_id
        break
      end
    end
    assert_true(prediction_id ~= nil, "visible proposal has a prediction id")

    vim.cmd("normal! l")
    -- `-l` does not consistently dispatch CursorMoved in headless runs.
    vim.api.nvim_exec_autocmds("CursorMoved", { buffer = 0 })
    assert_true(vim.wait(500, function() return not predict.status().proposal_active end),
      "normal cursor movement dismissed the proposal")

    local terminal_count, terminal = 0, nil
    local accepted_count, rejected_count = 0, 0
    for _, event in ipairs(collector.queue) do
      if event.payload.prediction_id == prediction_id then
        if event.event_type == "prediction_dismissed"
            or event.event_type == "prediction_accepted"
            or event.event_type == "prediction_partially_accepted"
            or event.event_type == "prediction_rejected" then
          terminal_count = terminal_count + 1
          terminal = event
        end
        if event.event_type == "prediction_accepted" then accepted_count = accepted_count + 1 end
        if event.event_type == "prediction_rejected" then rejected_count = rejected_count + 1 end
      end
    end
    assert_eq(terminal_count, 1, "navigation records one terminal decision")
    assert_true(terminal ~= nil, "navigation terminal event recorded")
    assert_eq(terminal.event_type, "prediction_dismissed")
    assert_eq(terminal.payload.outcome, "dismissed_navigation")
    assert_eq(accepted_count, 0, "navigation is not acceptance")
    assert_eq(rejected_count, 0, "navigation is not rejection")
  end)

  ok("single-line-v1-insert-delete-and-no-edit-semantics", function()
    vim.api.nvim_win_set_cursor(0, { 2, 0 })
    set_model_action("I\t# before", "eos", 3)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end))
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'")
    assert_true(predict.accept())
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 3, false)[1], "# before")
    vim.cmd("silent undo")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'")

    set_model_action("D", "eos", 1)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end))
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'")
    assert_true(predict.accept())
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "after")
    local shown_event, accepted_event
    for _, event in ipairs(collector.queue) do
      if event.event_type == "prediction_shown" then shown_event = event end
      if event.event_type == "prediction_accepted" then accepted_event = event end
    end
    assert_true(shown_event ~= nil and accepted_event ~= nil, "delete range events recorded")
    assert_eq(shown_event.payload.action, "delete_line")
    assert_eq(shown_event.payload.proposed_start_byte, 7, "delete source byte start")
    assert_eq(shown_event.payload.proposed_end_byte, 18, "delete includes the LF terminator")
    assert_eq(shown_event.payload.proposed_start.row, 1, "delete starts at target row")
    assert_eq(shown_event.payload.proposed_end.row, 2, "middle-line delete ends at next row")
    assert_eq(shown_event.payload.proposed_end.col, 0, "middle-line delete ends before next row")
    assert_eq(accepted_event.payload.editable_range.start_byte, 7,
      "acceptance repeats the exact source byte start")
    assert_eq(accepted_event.payload.editable_range.end_byte, 18,
      "acceptance repeats the exact source byte end")
    vim.cmd("silent undo")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'")

    set_model_action("N", "eos", 1)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return not predict.status().in_flight end))
    assert_true(not predict.status().proposal_active, "no-edit creates no visible proposal")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'")
  end)

  ok("single-line-v1-empty-file-insertion-is-one-undoable-acceptance", function()
    vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "" })
    vim.api.nvim_set_option_value("eol", true, { buf = buf })
    vim.bo[buf].filetype = "go"
    vim.api.nvim_win_set_cursor(0, { 1, 0 })
    set_model_action("I\tpackage main", "eos", 4)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end))
    assert_eq(require("tabcomplete_trajectory.buffers").canonical_bytes(buf), "",
      "preview did not modify empty source")
    assert_true(predict.accept())
    assert_eq(require("tabcomplete_trajectory.buffers").canonical_bytes(buf), "package main")
    vim.cmd("silent undo")
    assert_eq(require("tabcomplete_trajectory.buffers").canonical_bytes(buf), "",
      "undo restores the empty file")
    vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "before", "old = 'β'", "after" })
    vim.api.nvim_set_option_value("eol", true, { buf = buf })
    vim.bo[buf].filetype = "python"
    vim.api.nvim_win_set_cursor(0, { 2, 0 })
  end)

  ok("single-line-v1-rejects-cutoff-cap-stale-and-unrepresentable-edits", function()
    set_model_action("R\tinvalid", "length", 2)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return not predict.status().in_flight end))
    assert_true(not predict.status().proposal_active, "length termination cannot display")

    set_model_action("R\tinvalid", "eos", 65)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return not predict.status().in_flight end))
    assert_true(not predict.status().proposal_active, "65 generated tokens cannot display")

    set_model_action("R\tstale", "eos", 4)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end))
    vim.api.nvim_buf_set_lines(buf, 1, 2, false, { "typed while shown" })
    assert_true(not predict.accept(), "stale proposal must not apply")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "typed while shown")

    vim.api.nvim_buf_set_lines(buf, 1, 2, false, { "restore" })
    vim.api.nvim_win_set_cursor(0, { 2, 0 })
    set_model_action("D", "eos", 1)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end))
    local old_eol = vim.api.nvim_get_option_value("eol", { buf = buf })
    -- The delete is eligible only if it preserves the current final-newline state.
    assert_true(predict.accept())
    vim.cmd("silent undo")
    assert_eq(vim.api.nvim_get_option_value("eol", { buf = buf }), old_eol,
      "undo and acceptance preserve EOL option")
  end)

  ok("automatic-prefix-policy-preserves-unicode-and-does-not-label-suppression-rejection", function()
    assert_true(predict.preserves_typed_prefix("    return λ", { kind = "replace_line", text = "    return λ + 1" }))
    assert_true(not predict.preserves_typed_prefix("    return λ", { kind = "replace_line", text = "    return x" }))
    assert_true(not predict.preserves_typed_prefix("", { kind = "delete_line" }))
    assert_true(predict.preserves_typed_prefix("x", { kind = "insert_before", text = "# before" }))
    local util = require("tabcomplete_trajectory.util")
    local original_mode = util.current_mode
    util.current_mode = function() return "i" end
    predict.setup({ protocol_version = adapter.WIRE_VERSION, mode = "automatic",
      experimental_auto_opt_in = true, automatic_prefix_guard = true, persist_mode = false })
    vim.api.nvim_win_set_cursor(0, { 2, 3 })
    set_model_action("D", "eos", 1)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return not predict.status().in_flight end))
    assert_true(not predict.status().proposal_active, "automatic deletion must be suppressed")
    local last = collector.queue[#collector.queue]
    assert_eq(last.event_type, "heartbeat")
    assert_eq(last.payload.prediction_lifecycle, "automatic_policy_suppressed")
    assert_true(predict.status().counters.automatic_policy_suppressed > 0)
    predict.set_mode("manual")
    set_model_action("D", "eos", 1)
    assert_true(predict.predict())
    assert_true(vim.wait(1000, function() return predict.status().proposal_active end),
      "manual deletion preview remains available")
    predict.reject()
    util.current_mode = original_mode
  end)

  predict._request_impl = nil
  predict._tokenize_impl = nil
  predict.setup({ protocol_version = "compact-next-edit-v1", mode = "manual", synthetic = true,
    automatic_prefix_guard = false })
  vim.fn.delete(path)
end
