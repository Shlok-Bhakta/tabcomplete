# Changing-state replay and input-header experiment

Frozen on 2026-09-30 before reading this replay's prediction content.

The first local replay used 24 independent source snapshots, with an immediate
same-prompt repeat after each request. It was not a chronological editor replay.
Its actual input lengths also did not cover the intended 2,048-token bucket.
Those results and their original prompt policy remain preserved.

The new suite contains 24 ordered synthetic editor states in three trajectories.
It validates actual byte edits, cursor positions, history reconstruction, and
A-to-B-to-A file switches. It uses fixed nearby synthetic modules to cover
approximately 512, 1,024, and 2,048 input tokens. Per-model actual tokenizer
lengths, token-prefix identities, and backend reused/recomputed counts are
recorded. Two repetitions and two precisions produce 96 new requests. It does
not reinterpret synthetic events as human feedback.

The baseline's completed outputs all began with a newline. A diagnostic that
ignored that byte still found incompatible immutable suffixes or missing file
content in most cases. No baseline output was repaired or counted as a valid
action. This supports testing an input construction hypothesis, not promoting
the model or claiming that the sole problem was a separator.

The new `publisher-header-lf-input-v2` policy verifies the original publisher
prompt and appends exactly one LF after its final updated-file header. It records
the original and effective prompt hashes. All other input information and field
ordering remain fixed. Full-file mapping remains byte-exact, and generated text
is never trimmed. This is a separate prompt experiment, not a pure cache or
precision ablation. Its new fixtures also prevent a matched quality comparison
against the old snapshots.

## Preserved revisions and execution budget

- `changing_state_plan.json` preserves the original unexecuted control plan.
- Revision 3 started, completed seven Q4 requests, and was interrupted because
  artifact preparation overlapped the timing window. Its files are preserved.
  The whole 65.019949-second invocation is charged; its subset is not quality
  evidence for this comparison.
- Revision 4 records that interruption. It was not executed. An audit found
  duplicate budget metadata that still held old values.
- Revision 5 reconciles both budget fields and charges an additional
  1.000638-second validation. It is the active frozen plan.

The 5,400-second local campaign ledger charges 1,696.506838 seconds for the first
replay, 8.606580 seconds for separate tokenizer preflights and an identity-check
allowance, 65.019949 seconds for the interruption, and 1.000638 seconds for the
additional validation. Revision 5 has 3,628.865995 seconds remaining, floors its
invocation cap to 3,628 seconds, and reserves 600 seconds for finalization. Its
request deadline is 3,028 seconds from invocation start, including identity and
input validation. No weights, runtime settings, training, live service, or
installed editor configuration changed for this experiment.

Before execution, 21 targeted tests, Ruff, and scoped mypy passed. The read-only
audit verified the LF input construction and interruption accounting. Full
repository checks wait until timed CPU inference finishes to avoid contention.

Active plan file SHA-256:
`ae465ebb19a83354d0054f3bf29b8bae32474a9264b6c1a8ced05260806bd494`.
Canonical plan identity:
`06f1e61035bc7eb4578827b10c75e4fd2ecb129796facb67fe1e238dc48bb415`.
