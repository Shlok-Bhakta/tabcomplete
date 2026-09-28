# OpenCode structured-output isolation design (read-only investigation)

Investigated installed OpenCode **v1.18.31** without another provider request. The earlier JSON-schema smoke is inconclusive about Muse tool support: its config used `permission: "deny"` and legacy `tools: {"*": false}`, which also hid OpenCode's `StructuredOutput` tool. OpenCode then returned `StructuredOutputError` after a plain-text response. This proves that smoke did not yield structured output; it does not prove the Go Responses route rejects tool calls.

## Proposed isolated configuration

Start the existing loopback server from an empty temporary working directory, retain `--pure`, disable project config, and use a fresh temporary config home while retaining only the existing credential data path. Set the final config to:

```json
{
  "model": "opencode-go/muse-spark-1.3-contributor",
  "small_model": "opencode-go/muse-spark-1.3-contributor",
  "permission": {"*": "deny", "StructuredOutput": "allow"},
  "plugin": [],
  "mcp": {}
}
```

Do **not** include legacy `tools: {"*": false}`. Send a fresh session with `format: {"type":"json_schema","schema":...}` and no per-message `tools` overrides. Before any real provider call, assert from the prepared request that the **only offered tool** is `StructuredOutput` and `toolChoice` is `required`; fail closed otherwise. Keep the existing exact-model, public/synthetic-source, usage-ledger, token, cost, and timeout guards.

The rule order matters. `Permission.fromConfig` preserves config property order; `Permission.disabled` uses the last matching rule. With the catch-all deny first and exact `StructuredOutput` allow second, ordinary tools are removed from the provider request while that one output tool survives. `SessionPrompt` adds `StructuredOutput` only for a JSON-schema request, then `LLMRequestPrep.resolveTools` applies permission filtering. The old string `"deny"` normalized to `{"*":"deny"}` and removed that tool as well. Per-message `tools: {}` creates no override.

`--pure` disables external plugins, but not all built-in plugins. A temporary config home and disabled project config prevent user/project instructions, tools, MCP servers, and agents from entering this calibration process. OpenCode's permission system constrains model-callable tools; it is **not an operating-system network sandbox**. The provider request itself needs network access. If an absolute egress guarantee is required, constrain the process to the approved OpenCode Go endpoint separately and verify it does not need ancillary startup fetches.

## Muse Go route status

The installed model catalog declares `muse-spark-1.3-contributor` active, uses `@ai-sdk/openai` at `https://opencode.ai/zen/go/v1`, and reports `capabilities.toolcall: true`. OpenCode v1.18.31 sets `strict: false` on tools for this AI SDK provider family, and its JSON-schema mode asks for `toolChoice: "required"`. These facts establish that OpenCode can *offer* the tool to the Go Responses route. They do not establish that Muse will call it or that the route accepts the exact schema. A single authorized public/synthetic probe would be needed later; this investigation made none. If the route rejects tools or emits plain text, treat that as unsupported for the campaign and retain the prior result without a text-to-JSON rescue.

The v1.18.31 response field is **`info.structured`**, not `info.structured_output`: `SessionPrompt` assigns `handle.message.structured`, [the assistant schema](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/schema/src/v1/session.ts) declares `structured`, and [the session HTTP handler](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/opencode/src/server/routes/instance/httpapi/handlers/session.ts) serializes the returned `WithParts` as JSON. The teacher wrapper now reads `info.structured` and requires it when `format=json_schema` was requested. Plain text or an old-name field cannot rescue a missing structured result. This was verified only with deterministic mock responses; no additional provider call was made.

## Deterministic mock tests before a provider probe

1. Unit-test permission ordering with the actual v1.18.31 rules: `*` deny followed by `StructuredOutput` allow leaves exactly that tool; reversed order and legacy wildcard `tools=false` are rejected by the wrapper's configuration check.
2. Use a loopback mock of the Go Responses API to capture one prepared request. Assert one offered tool named `StructuredOutput`, `tool_choice=required`, exact model ID, fixed instruction hash, and no file/shell/web/MCP tools. This should inspect bytes sent to the mock, not only config text.
3. Return a valid tool call with schema-conforming arguments. Assert `info.structured_output` is present and the ledger settles once. Return plain text, malformed arguments, a disallowed tool call, and an HTTP tool-unsupported error in separate cases; each must fail without parsing prose or falling back to another model.
4. Launch with a disposable config home containing a deliberately hostile project config/plugin fixture outside the temp workspace. Assert it contributes no instructions or tools. Verify no source or secret appears in mock request or exception text.

## Source evidence

- [OpenCode v1.18.31 prompt implementation](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/opencode/src/session/prompt.ts): `StructuredOutput` insertion, required tool choice, and error on missing structured result.
- [OpenCode v1.18.31 request preparation](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/opencode/src/session/llm/request.ts): permission filtering and `strict: false` for `@ai-sdk/openai` tools.
- [OpenCode v1.18.31 permission implementation](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/opencode/src/permission/index.ts): last-rule precedence and disabled-tool filtering.
- [OpenCode v1.18.31 config loader](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/opencode/src/config/config.ts): global, project, and inline config loading; legacy `tools` conversion.
- [OpenCode permissions documentation](https://opencode.ai/docs/permissions/): catch-all plus per-tool rules and last-match semantics.
