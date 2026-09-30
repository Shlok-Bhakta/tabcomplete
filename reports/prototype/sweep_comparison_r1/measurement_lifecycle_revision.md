# Measurement lifecycle correction after attempt six

Frozen plan 10 generated 760 code-control predictions and 48 Q8 next-edit
predictions. Scoring completed for all 808 available records under the original
source identities. Missing Q4 next-edit output remains unknown.

The next-edit runner read final host/GPU measurements before leaving the Server
context. Those attributes are populated by `Server.__exit__`. Move only the
measurement-file write after that exit, retaining the enclosing observed run.
No model, tokenizer, fixture, prompt, decoding, output mapping or scoring rule
changes. This does not recreate the missing historical measurement file.

The regression fake now creates final measurements on context exit, matching
the real server lifecycle. Before the fix, one test failed and 25 passed. After
the fix, all 26 passed. Scoped Ruff and mypy passed.

Plan 10 and its scoring manifest retain the historical source hashes. Any
future inference must freeze a fresh campaign/scoring plan with corrected
source and test hashes. No new allocation is included in this correction.
