return function(ok, assert_eq, assert_true)
  local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")
  local repo_root = vim.fn.fnamemodify(root .. "/../../..", ":p")
  local fixture_path = repo_root .. "/tests/fixtures/single_line_edit_v1_golden.json"
  local fixture = vim.json.decode(table.concat(vim.fn.readfile(fixture_path), "\n"))
  local contract = require("tabcomplete_trajectory.single_line_v1")
  local buffers = require("tabcomplete_trajectory.buffers")

  ok("single-line-v1-context-golden-parity", function()
    assert_eq(contract.WIRE_VERSION, fixture.wire_version)
    assert_eq(contract.CONTEXT_POLICY_VERSION, fixture.context_policy_version)
    for _, case in ipairs(fixture.contexts) do
      assert_eq(contract.render_full_context(case.state), case.expected_prompt, case.name)
    end
  end)

  ok("single-line-v1-bounded-context-selection-parity", function()
    for _, case in ipairs(fixture.bounded_contexts) do
      local actual = contract.serialize_bounded(case.state, function(text)
        return #text + 1 -- Same UTF-8 byte count plus one special token as the Python fixture.
      end, case.max_input_tokens)
      assert_eq(actual.text, case.expected_prompt, case.name)
      assert_eq(actual.input_tokens, case.expected_input_tokens, case.name .. " tokens")
      local actual_rows = {}
      for row in pairs(actual.included.rows) do actual_rows[#actual_rows + 1] = row end
      table.sort(actual_rows)
      assert_true(vim.deep_equal(actual_rows, case.expected_rows), case.name .. " rows")
      assert_eq(#actual.included.history, case.expected_history_count, case.name .. " history")
      assert_eq(#actual.included.relevant, case.expected_relevant_count, case.name .. " relevant")
    end
  end)

  ok("single-line-v1-wire-decode-and-byte-application-parity", function()
    for _, case in ipairs(fixture.actions) do
      assert_eq(contract.encode_action(case.action), case.wire, case.name .. " wire")
      assert_eq(contract.apply_action(case.state, case.action), case.after_source,
        case.name .. " after source")
    end
    for _, case in ipairs(fixture.decode_cases) do
      local action, err = contract.decode_action(case.wire, case.stop_type, case.generated_tokens)
      if case.status == "ok" then
        assert_true(action ~= nil, case.name .. " should decode: " .. tostring(err))
        assert_eq(action.kind, case.action.kind, case.name .. " kind")
        if case.action.text == vim.NIL then
          assert_true(action.text == nil, case.name .. " should have no text")
        else
          assert_eq(action.text, case.action.text, case.name .. " text")
        end
      else
        assert_true(action == nil, case.name .. " must be rejected")
      end
    end
  end)

  ok("single-line-v1-action-ranges-match-lf-crlf-and-eof-byte-effects", function()
    local cases = {
      { name = "middle-lf-delete", source = "before\nremove\nend\n", fileformat = "unix",
        eol = true, row = 1, kind = "delete_line", start_byte = 7, end_byte = 14,
        end_row = 2, end_col = 0, includes_terminator = true },
      { name = "middle-crlf-delete", source = "before\r\nremove\r\nend\r\n",
        fileformat = "dos", eol = true, row = 1, kind = "delete_line", start_byte = 8,
        end_byte = 16, end_row = 2, end_col = 0, includes_terminator = true },
      { name = "eof-lf-delete", source = "before\nremove\n", fileformat = "unix",
        eol = true, row = 1, kind = "delete_line", start_byte = 7, end_byte = 14,
        end_row = 1, end_col = 6, includes_terminator = true },
      { name = "eof-noeol-delete", source = "before\nremove", fileformat = "unix",
        eol = false, row = 1, kind = "delete_line", start_byte = 7, end_byte = 13,
        end_row = 1, end_col = 6, includes_terminator = false },
      { name = "unicode-replace", source = "α\nβeta", fileformat = "unix", eol = false,
        row = 0, kind = "replace_line", start_byte = 0, end_byte = 2,
        end_row = 0, end_col = 2, includes_terminator = false },
      { name = "insert-zero-width", source = "α\nβeta", fileformat = "unix", eol = false,
        row = 1, kind = "insert_before", start_byte = 3, end_byte = 3,
        end_row = 1, end_col = 0, includes_terminator = false },
    }
    for _, case in ipairs(cases) do
      local action = case.kind == "delete_line" and { kind = case.kind }
        or case.kind == "replace_line" and { kind = case.kind, text = "Ω" }
        or { kind = case.kind, text = "inserted" }
      local state = { file_id = "src/range.py", filetype = "python", source = case.source,
        target_row = case.row, cursor_col = 0, history = {}, relevant = {} }
      local range = contract.action_range(state, action)
      assert_true(range ~= nil, case.name .. " range exists")
      assert_eq(range.start_byte, case.start_byte, case.name .. " start byte")
      assert_eq(range.end_byte, case.end_byte, case.name .. " exclusive end byte")
      assert_eq(range.end_row, case.end_row, case.name .. " editor end row")
      assert_eq(range.end_col, case.end_col, case.name .. " editor end byte column")
      assert_eq(range.includes_terminator, case.includes_terminator,
        case.name .. " terminator coverage")

      if case.name == "middle-lf-delete" or case.name == "middle-crlf-delete"
          or case.name == "eof-lf-delete" or case.name == "unicode-replace" then
        local buf = vim.api.nvim_create_buf(false, true)
        local source_lines = contract.physical_lines(case.source)
        local buffer_lines = {}
        for _, line in ipairs(source_lines) do buffer_lines[#buffer_lines + 1] = line.content end
        vim.api.nvim_buf_set_lines(buf, 0, -1, false, buffer_lines)
        vim.api.nvim_set_option_value("fileformat", case.fileformat, { buf = buf })
        vim.api.nvim_set_option_value("eol", case.eol, { buf = buf })
        vim.api.nvim_set_option_value("undolevels", -1, { buf = buf })
        vim.api.nvim_set_option_value("undolevels", 1000, { buf = buf })
        assert_eq(buffers.canonical_bytes(buf), case.source, case.name .. " source bytes")
        local expected = contract.apply_action(state, action)
        local applied, reason = contract.apply_to_buffer(buf, state, action)
        assert_true(applied, case.name .. " actual buffer application: " .. tostring(reason))
        assert_eq(buffers.canonical_bytes(buf), expected, case.name .. " applied bytes")
        vim.api.nvim_buf_delete(buf, { force = true })
      end
    end
  end)

  ok("single-line-v1-neovim-line-application-and-undo", function()
    local actions = {
      { kind = "replace_line", text = "λ = 'changed'" },
      { kind = "delete_line" },
      { kind = "insert_before", text = "# inserted ✓" },
    }
    for _, action in ipairs(actions) do
      local buf = vim.api.nvim_create_buf(false, true)
      vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "alpha", "βeta", "omega" })
      vim.api.nvim_set_option_value("fileformat", "dos", { buf = buf })
      vim.api.nvim_set_option_value("eol", true, { buf = buf })
      vim.api.nvim_set_option_value("undolevels", -1, { buf = buf })
      vim.api.nvim_set_option_value("undolevels", 1000, { buf = buf })
      local source = buffers.canonical_bytes(buf)
      assert_eq(source, "alpha\r\nβeta\r\nomega\r\n", "CRLF source")
      local state = {
        file_id = "src/app.py", filetype = "python", source = source,
        target_row = 1, cursor_col = 2, history = {}, relevant = {},
      }
      local expected = contract.apply_action(state, action)
      local applied, reason = contract.apply_to_buffer(buf, state, action)
      assert_true(applied, tostring(reason))
      assert_eq(buffers.canonical_bytes(buf), expected, action.kind .. " bytes")
      vim.api.nvim_set_current_buf(buf)
      vim.cmd("silent undo")
      assert_eq(buffers.canonical_bytes(buf), source, action.kind .. " clean undo")
      vim.api.nvim_buf_delete(buf, { force = true })
    end
  end)

  ok("single-line-v1-empty-file-insert-and-unrepresentable-eol-transition", function()
    local empty = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_lines(empty, 0, -1, false, { "" })
    vim.api.nvim_set_option_value("eol", true, { buf = empty })
    vim.api.nvim_set_option_value("fileformat", "unix", { buf = empty })
    vim.api.nvim_set_option_value("undolevels", -1, { buf = empty })
    vim.api.nvim_set_option_value("undolevels", 1000, { buf = empty })
    local empty_state = { file_id = "src/empty.go", filetype = "go", source = "",
      target_row = 0, cursor_col = 0, history = {}, relevant = {} }
    assert_true(contract.apply_to_buffer(empty, empty_state,
      { kind = "insert_before", text = "package main" }))
    assert_eq(buffers.canonical_bytes(empty), "package main")
    vim.api.nvim_set_current_buf(empty)
    vim.cmd("silent undo")
    assert_eq(buffers.canonical_bytes(empty), "", "empty-file insertion undo is byte exact")
    vim.api.nvim_buf_delete(empty, { force = true })

    local no_eol = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_lines(no_eol, 0, -1, false, { "first", "last" })
    vim.api.nvim_set_option_value("eol", false, { buf = no_eol })
    local source = buffers.canonical_bytes(no_eol)
    local state = { file_id = "src/no-eol.py", filetype = "python", source = source,
      target_row = 1, cursor_col = 0, history = {}, relevant = {} }
    local applied = contract.apply_to_buffer(no_eol, state, { kind = "delete_line" })
    assert_true(not applied, "deletion that changes final-newline state is blocked")
    assert_eq(buffers.canonical_bytes(no_eol), source, "blocked delete leaves bytes untouched")
    vim.api.nvim_buf_delete(no_eol, { force = true })
  end)

  ok("single-line-v1-application-refuses-stale-buffer", function()
    local buf = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "before" })
    local state = {
      file_id = "src/stale.py", filetype = "python", source = "before",
      target_row = 0, cursor_col = 3, history = {}, relevant = {},
    }
    vim.api.nvim_buf_set_lines(buf, 0, 1, false, { "changed" })
    local before = buffers.canonical_bytes(buf)
    local applied = contract.apply_to_buffer(buf, state, { kind = "delete_line" })
    assert_true(not applied, "stale action was applied")
    assert_eq(buffers.canonical_bytes(buf), before, "stale buffer changed")
    vim.api.nvim_buf_delete(buf, { force = true })
  end)
end
