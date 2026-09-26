# One-line edit evaluation design

Contract: `single-line-edit-v1`, suite revision 3. This design scores model output only after the reference `EditState` and `EditAction` contract has validated the gold edit. It does not define training data or inspect a sealed case. `src/tinycomplete/one_line/evaluate.py` is the executable scorer.

## Inputs and isolation

The model receives only `serialize_state(state)`. Scoring metadata (`gold_action`, `after_source`, split, source repository, mechanism, objective checker, ambiguity) stays outside that serializer. The evaluator accepts exact raw response bytes decoded as UTF-8 text, an actual EOS/termination observation, generated-token count, and optional calibrated confidence. It calls the shared codec with the 64-token cap. A cutoff or malformed header is invalid, even if a partial response resembles a valid action. A decoded action must also apply at the specified row. No payload trimming or multiline repair is performed.

`EvaluationCase` verifies that the gold action produces `after_source`. A keep case must preserve the source; an edit-required case must change it. When an independent objective check is attached, it must accept gold. For an unambiguous edit-required case it must reject the unchanged source. These checks make a gold/unchanged control meaningful and expose trivial or unsound tests before model scoring. Objective checks should be deterministic structural assertions or existing network-disabled sandbox checks prepared without showing the solver gold patches. The scorer does not run arbitrary generated instructions on the host.

## Outcome accounting

Report valid serialization, explicit EOS, cap hits, action choice, byte-exact after state, output tokens, and output bytes separately. For edit-required cases, exact after state is a verified success. A different after state is successful only when an independently checkable objective accepts it. Without such a check, a nonexact valid edit has `unknown` success, not a hard zero. Cases declared ambiguous remain in an exploratory count and outside hard success denominators. For keep cases, recall requires an explicit valid keep action. A valid edit action on a keep case is a false-positive edit. Malformed output harms keep recall and is counted separately from false-positive edits.

Aggregate output keeps edit-required and keep denominators separate. It reports action breakdowns for insertion, replacement, deletion, and keep; mechanism, language, source-type, and input-length strata; and utility with a predeclared incorrect-edit cost of 3. Raw correct, incorrect, false-positive, and unknown counts remain visible. An always-keep model earns no edit-required success. The display calibration function accepts development rows only, uses verified precision, reports unknown displayed actions separately, and penalizes those unknowns conservatively when selecting a threshold. It never fits a threshold on test scores. Confidence must be a finite probability in `[0,1]`; no confidence means no threshold calibration.

Controls are gold, keep, seeded random action, and a deliberately simple rule. The simple rule copies a unique exact substitution from the latest *visible* history entry onto the selected line; otherwise it keeps. It does not read gold actions or metadata. Gold must score as valid and successful. Keep should fail the intended edit criterion on unambiguous edit-required cases; objective-check validation enforces this where a check exists. A high model score against only weak controls does not establish broad next-edit quality.

## Uncertainty and paired comparison

`clustered_interval` resamples repositories or generated-task families, rather than treating near-duplicate rows as independent. Report both views when enough clusters exist, including the cluster count and sample count. `paired_outcomes` matches unique case IDs and metadata across two candidates, reports second-candidate wins, losses, ties, and a repository-clustered interval for the success-rate difference. A confidence interval containing zero is unresolved evidence; it is not equivalence. Rows with unknown edit success cannot enter a verified-success comparison, and their count remains visible in the aggregate.

## Sealed test gate

The suite manifest contains case IDs, their digest, and any frozen content hashes. Before test cases are opened, a locked development selection must identify the selected artifact and exact suite-manifest SHA-256. `claim_sealed_evaluation` writes an exclusive, fsynced claim file. Repeating the claim fails, including after an interrupted run; a repeat requires a documented suite/plan revision. Batch scoring rejects test rows without a claim, rejects a mixture of test and non-test rows, and verifies the claimed case-ID digest. Direct single-case scoring also rejects unclaimed test rows. The claim is a local procedural guard, not a security boundary against someone intentionally bypassing Python code.

No teacher solver output or sealed content enters training, LR choice, abstention calibration, or case-authoring feedback. The current OpenCode output-use restriction in the frozen schema excludes OpenCode output from student labels and automated benchmark scoring without a separate compatible rights basis.

## CPU verification

`uv run pytest -q tests/test_one_line_evaluate.py`: 9 passed. `uv run ruff check src/tinycomplete/one_line/evaluate.py tests/test_one_line_evaluate.py`: passed. `uv run mypy src/tinycomplete/one_line/evaluate.py`: passed. These verify the scorer and guards with synthetic in-memory cases; they are not model-quality evidence.
