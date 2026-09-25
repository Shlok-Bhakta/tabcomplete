-- Incremental llama.cpp completion SSE decoder. Bytes remain untouched until JSON decoding.
local M = {}

function M.new(max_bytes)
  return { buffer = "", pieces = {}, raw_chunks = {}, bytes = 0, max_bytes = max_bytes or 262144,
    terminal = nil, error = nil, first_token = false, events = 0 }
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
        if event.content ~= "" then parser.first_token = true end
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

return M
