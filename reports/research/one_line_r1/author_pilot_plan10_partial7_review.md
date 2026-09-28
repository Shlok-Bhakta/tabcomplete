# Plan-10 first-seven candidate review

This is a partial audit of seven settled Muse author rows. The eighth source,
`public-source/b13cc3a6e67cae9e931a9694`, had a settled usage overrun and no
candidate row or retry. The other 92 seeds have no output in this review. The
planned 99-row audit remains separate. Provider calls are paused. Training
acceptance remains **0**.

The audit checked the raw artifact hash
`2b0c6ea5af9971f19c1eb73e4808a0895da690d99676328fdbac479def43e167`
against the [usage incident](author_pilot_plan10_usage_incident.json). It
replayed every declared prior edit against its pinned public source, applied
every target action byte-exactly, checked before and after hashes, and retokenized
the student prompt and target. All seven passed those mechanical checks. All
seven before and after files also parse as Python 3.11 ASTs. No project tests or
functional objectives were executed, and parsing does not establish behavior.

| Partial yield | Observed |
| --- | ---: |
| Settled candidate rows | 7 |
| Excluded usage-overrun request | 1 |
| Exact replay and student token checks | 7/7 |
| Python 3.11 AST parse, before and after | 7/7 |
| Keep / replace / delete / insert | 1 / 6 / 0 / 0 |
| Student input tokens, q25 | 4,951 total; 491–1,016 per row |
| Target tokens including EOS, q25 | 105 total; 2–35 per row |
| Source repositories | 7 distinct among 7 rows |
| Verified mechanism categories | 0; all seven are `unreviewed_teacher_author` |
| Accepted training rows | 0 |

I read the [blind prompt notes](author_pilot_plan10_partial7_blind_notes.md)
before revealing the candidate actions and author objectives. The table below
compares those notes with the authored actions. It is an audit judgment, not a
score or a training label.

| Source suffix | Blind inference from student context | Authored action | Objective and ambiguity finding |
| --- | --- | --- | --- |
| `79d01114` | A development call might receive the `connect=False` argument recently added to the production call, but the paths may intentionally differ. | Replace the development call to add `connect=False`. | Ambiguous. Checks only require the proposed token and indentation; they do not establish that development connection behavior should match production. |
| `44bcb0af` | Keep the `mkdir` import because the recent edit changed a distant call to use `mkdir`. | Keep. | Student context supports keep. The objective's claim that Python compilation proves no `NameError` is unsound: compilation does not execute name lookup. The visible history does support the import's use. |
| `d50d6e8f` | Keep the assignment. A new `None` default alone does not show that the attribute must remain an empty string. | Replace with `error_response or ""`. | The author assumes an unstated string invariant. Its tests encode that assumption and omit explicit `None`, `0`, or `False` inputs, which the replacement also changes. The action is not inferable from the student prompt. |
| `0739697c` | Add `null=True` to a foreign key whose deletion policy just changed to `SET_NULL`. | Add both `null=True` and `blank=True`. | The core nullability repair is supported. Django says `SET_NULL` requires `null=True`; `blank=True` instead changes form validation and is an extra behavior choice. The author's checks do not test that choice or run the migration. [Django field reference](https://docs.djangoproject.com/en/5.2/ref/models/fields/). |
| `c27ecab9` | The newly assigned search result is unused, but a one-line remedy is unclear because the form-invalid path never defines `result`. | Replace the HTML template return with `return jsonify(result)`. | This can reference an unbound local when validation fails and changes the endpoint's response type. The objective checks only the chosen line and variable use, not either failure. Candidate is not supported as a safe one-line next edit. |
| `ee11ee23` | Move the `S21_PT` correction from `grace_clm` to the visible `grace_slm` array. | Replace the target with `grace_slm[2, 1, i] -= S21_PT`. | The action is well supported by the visible C/S pairing. The objective names further rows 119–120, which were omitted from the bounded student prompt; its checks are static relationships, not an independent numerical test of the correction. |
| `28dca222` | The recent return now has three values; the target still unpacks two. | Replace target with `AL, caches, _ = L_model_forward(`. | Strong visible arity evidence. A stub returning three values could check the unpack, but the author supplied only prose checks and no executable regression test. |

The [partial audit JSON](author_pilot_plan10_partial7_audit.json) and [blind
prompt artifact](author_pilot_plan10_partial7_blind.jsonl) retain the complete
IDs and hashes. This seven-row sample has no deletion or insertion candidates
and cannot support a general author quality estimate. The same Muse model may
fill author, solver, and reviewer roles in separate sessions; actor IDs show
role isolation rather than independent model families or human review. The
synthetic smoke was a format gate. No candidate is a training label yet.

The observability CLI request for the excluded request
`author10-public-fbd464d6ff93ee68d286cb77` returned `no_runs`. The local
scientific artifact and incident record are the evidence for this review;
there is no SigNoz trace to add operational detail.
