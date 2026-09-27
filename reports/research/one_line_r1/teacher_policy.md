# Hosted teacher boundary and synthetic calibration

Observed 2026-09-26/27: `opencode --version` returned `1.18.31`; the read-only
model listing included `opencode-go/muse-spark-1.3-contributor`. The user reports
direct approval from the provider for this small student campaign. That approval
is **user-reported**, not independently verified here. The ordinary
[OpenCode Terms of Use](https://opencode.ai/legal/terms-of-service), effective
2026-08-15, otherwise restrict programmatic Output extraction and competing
model development. The frozen plan revision 4 records the reported exception
and permits only public or synthetic source through this route. No private,
sealed, or secret-bearing input is authorized.

The [Go documentation](https://opencode.ai/docs/go/) lists the exact Muse model
and Responses endpoint, describes typical coding-agent traffic, and publishes
5-hour, weekly, and monthly usage sublimits. The
[server documentation](https://opencode.ai/docs/server/) supports a loopback
headless server. `OpenCodeTeacherClient` uses that server in an isolated empty
directory, pins primary and small models to Muse, disables plugins and agent
tools, and creates one session per role request. It returns final content,
session/message IDs, exact model identity, token fields, finish reason, and
reported cost. A locked append-only ledger reserves upper-bound tokens before
the request and settles reported usage afterward. Failed or ambiguous calls
retain their reservation. No paid fallback or other model route is implemented.
The provider account's separate Zen-balance fallback setting was not verified;
the one smoke consumed far below the documented Go sublimit.

One synthetic calibration request completed under frozen plan revision 4:

| Field | Observed |
| --- | ---: |
| Session | `ses_f1f2f656dffegGNyaNZaa17zZl` |
| Message | `msg_0e0d09ddc001Kvvhcw94vwawC1` |
| Prompt SHA-256 | `57ea7c3e953efb39eff33021584ac7995a6c221dd3e6e9588427726b6814a980` |
| Output SHA-256 | `19c26fd657a76ce0e5fb0766dfce2742c935b7bc72390e66f833835776c0a9d1` |
| Input tokens | 2,072 |
| Visible output tokens | 32 |
| Reasoning tokens | 323 |
| Generated tokens counted against output cap | 355 |
| Reported total tokens | 2,427 |
| Cache read/write tokens | 0 / 0 |
| Reported cost | $0.0002782 |
| Finish | `stop` |

The reported cost matches the published input/output prices when reasoning is
included with generated output; this is an inference from this request's fields,
not a claim about undocumented billing rules. The initial ledger settlement
recorded 32 visible output tokens. An append-only correction event raises the
budget count to 355 generated tokens, preserving the original events.

The request included a JSON-schema `format` body, but the response lacked
`structured_output` metadata and contained narration before the JSON object.
The strict candidate parser rejects it as `invalid_extra_text`. No student
label was accepted from this smoke. The full synthetic response and hashes are
in `teacher_calibration.json`; the ledger is `teacher_usage.jsonl`. No bulk
teacher calls have started.

## Frozen 16-case calibration, plan revision 5

The fixed synthetic development suite was sent exactly once per case using
`calibration_prompts` and the pinned local q25 tokenizer. The provider returned
all 16 messages with finish `stop`; q25 EOS was not observed in any provider
message. The 64-token scope check counted the raw wire encoded with q25 plus
one hypothetical q25 EOS. No output hit that scope cap. The evaluator did not
extract embedded actions from prose or repair a malformed response.

| Method | Strict valid wire | Verified edit success | Correct keep |
| --- | ---: | ---: | ---: |
| Gold control | 16/16 | 12/12 | 4/4 |
| Keep control | 16/16 | 0/12 | 4/4 |
| Trivial control | 16/16 | 0/12 | 4/4 |
| Muse raw output | 6/16 | 5/12 | 1/4 |

Ten outputs were malformed. All ten included prose before any apparent action;
two also contained line breaks. The 95% structural pass gate in the frozen
teacher protocol fails, so this calibration does not justify the 100-case
teacher pilot or any 1,000-case expansion. No bulk teacher calls followed.

The 16 requests reported 34,659 input tokens and 11,176 generated tokens
including reasoning; their reported costs sum to $0.0057011. The append-only
ledger totals, including the earlier one-request smoke, are 17 calls, 36,731
input tokens, and 11,531 generated output tokens. Per-case usage, prompt/output
hashes, strict statuses, controls, and the raw-artifact SHA-256 are in
`teacher_calibration_pass_v1.json`. Raw outputs are in the ignored artifact
`artifacts/research/one_line_r1/teacher_calibration_plan5.jsonl` (SHA-256
`5402903f8cfd4b5e2eebb484ab7dec3855a0f07d33a8a293da08bdb2bec407fb`).

## Frozen final-block calibration, plan revision 6

After plan 5 failed, the plan registered a provider-only final-action block. The
16 development states and student N/D/R/I action remained unchanged. The
system instruction was read verbatim from the hashed protocol v2 file. The
strict parser required one terminal `<FINAL_ACTION>` block on complete lines,
exactly one action line inside it, no carriage return or trailing text, and a
q25 wire-plus-EOS length within 64 tokens. The historic plan 5 outputs were
not rescored. All 16 new provider messages finished `stop`; none observed an
actual q25 EOS, which remains a distinct student requirement.

| Method | Valid final block | Verified edit success | Correct keep | False-positive keep edits |
| --- | ---: | ---: | ---: | ---: |
| Muse, plan 6 | 13/16 | 7/12 | 2/4 | 1/4 |

All three malformed messages lacked the frozen newline before the opening tag.
The observed 81.25% structural rate missed the 95% gate, and the fixed
correct-minus-three-times-incorrect utility was -2 on this small suite. This
is a prompt-format diagnostic on the same synthetic development cases, not an
independent held-out quality estimate. No 100-source pilot ran and no accepted
training labels were produced.

This pass used 35,971 reported input tokens and 14,451 generated tokens
including reasoning; its reported cost was $0.0064873. Cumulative usage is
33 calls, 72,702 input tokens, 25,982 generated tokens, and $0.0124666
reported cost. The [plan 6 result](teacher_calibration_pass_v2.json) includes
per-case hashes, failures, controls, and usage. Its ignored raw artifact is
`artifacts/research/one_line_r1/teacher_calibration_plan6.jsonl`, SHA-256
`69e162f616bf890ad163fb29edd3884a083edd5c705b52936c89949b0634fb33`.
