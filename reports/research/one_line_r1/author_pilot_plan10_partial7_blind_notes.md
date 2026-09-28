# Plan-10 partial seven: blind context notes

These judgments use only the seven exact student prompts in
`author_pilot_plan10_partial7_blind.jsonl`, before revealing the author action,
author-only focus, intent evidence, or objective. The SHA ordering is fixed by
the audit script. "Likely" describes a human hypothesis, not a training label.

| Blind order | Source suffix | Student-visible evidence | Blind judgment |
| --- | --- | --- | --- |
| 1 | `79d01114` | The recent production registration added `connect=False`; the target development registration lacks it. | Ambiguous. Adding `connect=False` to the target is plausible, but production and development paths may intentionally differ. |
| 2 | `44bcb0af` | The target imports `mkdir`; recent history changed a distant call from `os.mkdir` to `mkdir`. | Keep is the visible explanation. Removing or changing the import would conflict with the new call shown in history. |
| 3 | `d50d6e8f` | The constructor default changed from `""` to `None`; the target assigns the parameter to an attribute. | Keep is the visible explanation. No target change follows from the new default alone. |
| 4 | `0739697c` | The target foreign key was changed from `CASCADE` to `SET_NULL` but has no `null=True`. | Likely replace target to make the field nullable; exact migration policy is not established by this snippet alone. |
| 5 | `c27ecab9` | Recent history assigned the `word_search` result, but the target renders a template without it. | Ambiguous. Passing `result` to the template is plausible, yet `result` is unbound when form validation fails. One target-line edit may be insufficient. |
| 6 | `ee11ee23` | The preceding `C21_PT` correction acts on `grace_clm` and now says cosine coefficient. The target `S21_PT` correction also acts on `grace_clm`, while the visible return has `grace_slm`. | Likely replace target to act on `grace_slm`; adding a sine comment is plausible but secondary. |
| 7 | `28dca222` | Recent history changed `L_model_forward` to return three values; the target call unpacks two. | Likely replace target's unpacking with three values. This resolves a visible arity mismatch. |

The seven prompts come from public Python snippets. This is a partial review of
seven candidates, not a score for the planned 99-row pilot. The eighth source
is excluded after a settled usage overrun and has no candidate row.
