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
    assert_true(predict.accept(), "explicit acceptance applied replacement")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "new = 'λ'")
    vim.cmd("silent undo")
    assert_eq(vim.api.nvim_buf_get_lines(buf, 1, 2, false)[1], "old = 'β'",
      "acceptance undo restores prior text only")
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

  predict._request_impl = nil
  predict._tokenize_impl = nil
  predict.setup({ protocol_version = "compact-next-edit-v1", mode = "manual", synthetic = true })
  vim.fn.delete(path)
end
