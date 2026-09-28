# Plan-10/11 public author candidate audit

The completed combined pilot has 99 durable rows from 100 frozen public-source
seeds. Seven rows came from plan 10 and 92 from plan 11. The eighth plan-10
request, `public-source/b13cc3a6e67cae9e931a9694`, had a settled usage
overrun and remains censored with no candidate or retry. The audit verified
both raw artifact hashes, plan-10 protocol identity, plan-11 continuation spec
identity, prompt and source hashes, phase-specific request IDs, and the exact
99-plus-one source partition. Training acceptance remains **0**.

| Audit measure | Result |
| --- | ---: |
| Candidate preflight / rejected / censored | 77 / 22 / 1 |
| Exact history and action replay among candidates | 77/77 |
| Student input tokens, q25 | 50,489 total; median 653; range 406–1,017 |
| Candidate target tokens including EOS, q25 | 1,086 total; median 14; range 2–37 |
| Potential nonpadding student tokens | 51,575; none used for training |
| Provider-reported input / generated tokens on 99 durable calls | 492,521 / 289,308 |
| Keep / delete / replace / insert candidates | 2 / 6 / 50 / 19 |
| Distinct candidate source repos and connected groups | 77 / 77 |
| Verified mechanism categories | 0 |
| Accepted training rows | 0 |

Twenty of the 22 rejections had an invalid objective-kind field. One reversed
the prior intended replacement; one had history that did not replay. Every
candidate mechanism is still the generic `unreviewed_teacher_author`. The
author's own objective-kind strings are not verified mechanism labels; `fix`
appears 19 times and `consistency` 17 times. The bounded student contexts omit
a median of 35 physical source rows, with a maximum of 483. This makes
student-visible evidence a separate review question from byte-exact replay.

## Frozen 12-case review

The [blind notes](author_pilot_combined_plan11_blind_notes.md) were written
from the exact student prompts before viewing the authored actions or
objectives. Cases 10 and 11 reused judgments from the earlier seven-row blind
review. Six prompts supported a core action or keep; six left multiple
plausible next edits. This sample is conditional on candidate preflight and is
too small to estimate acceptance or general edit quality.

| Sample | Authored action | Context and objective finding |
| ---: | --- | --- |
| 1 | Change `i <= len(list(Brand))` to `<`. | No behavioral fix: enumeration only produces indices below the length, so both predicates pass the same elements. The author's claimed off-by-one objective was already true before the action. |
| 2 | Rename one `a.getClient(ctx)` call to `a.getServiceClient(ctx)`. | The target rename is clearly supported, but 24 other `a.getClient(` calls remain after the edit and the old helper definition is gone. The proposed one-line action is an incomplete file repair; a Go parse check cannot establish method resolution. |
| 3 | Replace direct deep-dive route registration with `routes.push(...)`. | The visible `routes` array is never applied to the router. Removing `router.use('/deepdive', ...)` may remove a working endpoint. The objective merely checks the proposed text, not route behavior. |
| 4 | Insert `defer stem.Close()` before `stem.Exec`. | Closing a prepared statement on the success path is plausible, but the preceding `Prepare` error branch only logs and continues. The new defer and existing Exec still encounter an invalid `stem` on that path. The objective's all-path safety claim is unsupported. |
| 5 | Reinsert `this.typingAnimate()` immediately after the latest edit removed it. | This reverses the latest visible user action. The author supplies no new intent to restore animation; its checks simply demand the removed call. |
| 6 | Construct `Variant::Content` in the `VariantType::Content` conversion arm. | The variant-kind repair has clear source evidence. The exact `.into()` conversion and round-trip claim need an independent Rust build and behavioral test; the author supplied only prose checks. |
| 7 | Append a `console.assert(result === 5, ...)` at EOF. | The assertion follows arithmetic in the sample code, but nothing in the prior edit or target context requests an assertion. It is an invented next task, not an inferable user edit. |
| 8 | Rename `vt.setVar` to `vt.setVariable`. | Getter rename history does not prove a setter rename; the `VarTree` API is outside the student prompt. The objective is textual and does not type-check the proposed method. |
| 9 | Rename `defer out.Flush()` to `defer w.Flush()`. | The target rename is clearly supported, but two `Fprintln(out, ...)` calls remain after it. The author's stated go-build objective is not met by this one-line patch. |
| 10 | Add `connect=False` to the development Mongo registration. | Production recently gained the argument, but matching development behavior is a choice the visible source does not settle. The objective checks only argument presence and indentation. |
| 11 | Keep the `mkdir` import. | Student-visible history supports keep: a prior call now uses `mkdir`. The author's statement that compilation proves no runtime `NameError` is not a valid check. |
| 12 | Add `true` to a second `fillHeader` call. | The meaning of the flag and whether the two push methods need identical header policy are absent. The objective checks matching arity, not request behavior. |

The clearest action evidence is in samples 2, 6, and 9, plus the keep control
in sample 11. Samples 2 and 9 still leave visible or independently countable
stale references. Samples 1, 3, 4, 5, and 7 show substantive objective gaps;
samples 8, 10, and 12 depend on unshown API or user intent. The review does
not assign training labels, replace the author's objectives, or alter the
frozen sample. An independently authored executable check is still needed
before any candidate is accepted.

The [audit JSON](author_pilot_combined_plan11_audit.json) records aggregate
counts and all 12 SHA-selected IDs. The [blind prompt artifact](author_pilot_combined_plan11_blind_sample.jsonl)
contains exactly the student-visible prompts used here. The same Muse model
may author, solve, and review under different sessions. Those session IDs
show role isolation, not independent model families or human review.
