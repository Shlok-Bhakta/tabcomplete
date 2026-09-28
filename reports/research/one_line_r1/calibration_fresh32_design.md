# Fresh 32-case development calibration

The [fixture](calibration_fresh32_cases.json) is a separate `single-line-edit-v1` suite at contract revision 3. It contains 32 original synthetic development states: eight per Python, TypeScript, Rust, and Go; eight each for keep (`N`), delete (`D`), replace (`R`), and insert (`I`). It does not modify the first 16-case calibration, the shared evaluator, the four public Git pilot repositories, or TabComplete's sealed test data.

The [manifest](calibration_fresh32_manifest.json) pins the fixture SHA-256 `7ba1b86df983fe30009a10746ba343e38963b034e5300dc77dc4aaa2df634177` and each canonical case hash. The manifest SHA-256 is `66f61346ba32b618e7814b024e2b63a3a46deaab14e13c957fe6841ac8113689`. A changed fixture or row is rejected by `load_fresh32_cases`; an altered test split is rejected. These hashes were fixed before any provider/model quality call on this suite.

## Task coverage and visible evidence

The prior 16 cases focused on identifier suffix renames, missing guards, and duplicate-line removal. These cases cover rounding, delimiter choice, saturation, nonnegative computation, ordering before return, event recording, removal of debug output, and removal of an assignment that overwrites a processed value. Eight counterfactual pairs share the exact file identity, source, target row, and cursor. Their histories differ: one member shows an earlier sibling operation change, while the keep member shows a latest actual edit that restored the current target line. Neutral file identities avoid naming the action or mechanism in the model prompt.

The source comments and prior edits make the intended property visible. Scoring metadata, gold actions, objectives, families, and after states are excluded from `fresh32_prompts`. The objective checks use the installed tree-sitter grammar plus the applicable source-grounded rule: a transformed operation on the target, preservation after a visible reversal, one inserted operation in the required order, or removal of a stale line. The new operations and forbidden lines are checked independently of byte-exact gold matching. These checks are structural and do not prove full runtime behavior, external type correctness, or human usefulness.

Use `load_fresh32_cases(fixture_path, manifest_path)`, then `fresh32_prompts(cases, q25_tokenizer)` to construct answer-free model input. Preserve raw response, actual EOS observation, and generated-token count. Use `score_fresh32_predictions(cases, predictions)` for a complete 32-case pass with the shared 64-token wire scorer. The provider was not called during case construction or validation.

## Local controls

Gold passed all 32 objectives. Unchanged source failed all 24 edit-required objectives. A wrong `BROKEN` replacement failed all 32 objectives. The always-keep control scored zero edit-required successes and recalled all eight keep cases. A nonexact but structurally valid replacement passed as an alternative-valid edit. All before/after sources parsed without tree-sitter errors. On the pinned q25 tokenizer, input prompts used 173–259 tokens (median 207), all histories fit, and gold response plus EOS used 2–29 tokens (median 5), below the 64-token ceiling.

`uv run pytest -q tests/test_one_line_calibration_fresh32.py` passed five tests. Ruff and mypy passed for the new validator and test. These results verify the fixture and scorer wiring only; no model quality result is claimed.
