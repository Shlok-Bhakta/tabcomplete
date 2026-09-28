# Fresh32 Muse calibration: focused error analysis

This reviews the frozen synthetic development suite at fixture SHA-256
`7ba1b86df983fe30009a10746ba343e38963b034e5300dc77dc4aaa2df634177`
and the recorded plan-7 result in `teacher_calibration_pass_v3_fresh32.json`.
It does not change the suite, extract a different answer, or rescore any case.

| Gold action | Cases | Verified edit successes | Failures |
| --- | ---: | ---: | ---: |
| Keep | 8 | 8 correct keeps | 0 false edits |
| Delete | 8 | 8 | 0 |
| Replace | 8 | 3 | 5 |
| Insert | 8 | 2 | 6 |

The recorded edit-required result is **13/24**, with **0/8** false edits on keep
cases. All 32 *extracted* actions are valid and explicitly terminated. The
provider's native text had surrounding narration or `<FINAL_ACTION>` tags;
the artifact marks **32/32 raw responses malformed** under the direct wire
parser. Thus the 32/32 result depends on the frozen extraction protocol. The
frozen result certifies **zero alternative-valid** outputs. The observations
below do not promote plausible alternatives to passes.

| Case | Recorded failure and visible evidence | Interpretation |
| --- | --- | --- |
| f03 Python replace | A recent `customer_key` edit changed `-` to `:`; `order_key` still uses `-`. Muse kept it. | Propagating a sibling edit is plausible, but no visible contract says both keys share a delimiter. Intent is ambiguous. |
| f05 Python insert | Comment requests case-insensitive display order. Muse replaced `return names` with `return sorted(names, key=str.lower)` rather than inserting `names.sort(key=str.casefold)`. | Plausible nonmutating solution, but `lower` and `casefold` differ for some Unicode and the frozen insertion objective does not accept it. |
| f06 Python insert | Comment requires an event after `store.write(record)`; Muse kept the return line. | Clear missed edit despite visible intent. |
| f09 TypeScript replace | Recent `gross` change uses `Math.round`; `net` still uses `Math.trunc`. Muse kept `net`. | Different rounding between gross and net could be intentional; transfer intent is ambiguous. |
| f11 TypeScript replace | Recent `primary` change uses `|`; `secondary` still uses `,`. Muse kept it and explicitly reasoned that the formats may differ. | Visible history was considered, but propagation is not compelled by source. |
| f13 TypeScript insert | Comment requires ascending rank. Muse returned `[...rows].sort(...)` on the target line instead of inserting `rows.sort(...)`. | Plausible behavior with different mutation semantics; fails the frozen insertion objective. |
| f14 TypeScript insert | Comment requires a save event after `cache.set`; Muse kept the return line. | Clear missed edit despite visible intent. |
| f19 Rust replace | Recent `primary` separator change uses `:`; `secondary` retains `-`. Muse kept it. | No visible shared-format contract; transfer intent is ambiguous. |
| f21 Rust insert | Comment requires ascending values. Muse replaced the target expression with `items.sort(); items` rather than inserting `items.sort_unstable();`. | Plausible one-line behavioral solution for `Vec<i32>`, but fails the frozen insertion objective. |
| f27 Go replace | Recent `primary` separator change uses `|`; `secondary` retains `,`. Muse kept it and explicitly said the formats may differ. | Visible history was considered; propagation remains ambiguous. |
| f30 Go insert | Comment requires a save event; Muse inserted `append(events, value)` without assigning the returned slice or adding `"saved:"`. | Wrong event content and no change to the returned slice length: a concrete code-semantics error. |

The failure pattern is five sibling-history transfer keeps, three ordering
solutions with a different action or operation, two missed event insertions,
and one incorrect Go event insertion. None of these prompts appears truncated:
the recorded failing prompts are 173–259 Qwen tokenizer tokens against a
1,024-token input budget; the prompt builder includes the target, neighboring
source, and the single history edit. Their stored prompt hashes were checked
against the frozen fixture by the calibration runner. This supports a task
ambiguity or model-decision explanation, rather than a context omission claim.

**One change for a future suite revision:** make shared-policy intent explicit
in the visible source of sibling-transfer cases, for example a comment stating
that both serializers use the same delimiter. This would test following recent
history under a stated rule. The present cases and scores should remain frozen.

Limits: These are 32 short, synthetic development states, not held-out real
editing quality. A plausible alternative above has not passed an independently
registered check. The observability CLI returned `no_runs` for request
`cal7-f03`; this analysis relies on the scientific result and raw artifact, not
on an unavailable SigNoz trace.
