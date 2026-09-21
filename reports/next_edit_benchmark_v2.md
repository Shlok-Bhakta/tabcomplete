# Next-edit benchmark protocol v2

Protocol v2 repairs the action contract while preserving the v1 suite and predictions as
historical artifacts. It does not train a next-edit model and is not used to rank causal-CPT
checkpoints.

The canonical actions are:

- `{"action":"no_edit"}`: leave the marked region unchanged.
- `{"action":"replace","text":""}`: delete a nonempty marked region, or insert nothing at
  an empty region.
- `{"action":"replace","text":"replacement code"}`: replace or insert the exact text.

Replacement text is applied at UTF-8 byte offsets without trimming whitespace. Malformed JSON,
truncated output, token-cap termination, overgeneration, wrong action choice, and wrong code are
reported separately. The product generation ceiling remains 96 tokens.

## Frozen controls

The v2 suite contains 200 cases. Its SHA-256 is
`69c42041f499ac5839195573bfbe691565d013ca132e8e183b5fc3fa61f0a0ff`.

| Control | Action choice | No-edit recall | Edit-required functional | False-positive edits | Malformed | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| Gold action | 200/200 | 40/40 | 160/160 | 0 | 0 | 0 |
| Unchanged/no-edit | 40/200 | 40/40 | 0/160 | 0 | 0 | 0 |
| Malformed/truncated | 0/200 | 0/40 | 0/160 | 0 | 100 | 100 |

The current executable evaluator therefore rejects every unchanged edit-required fixture and
accepts every gold edit. All 40 behavior-preserving fixtures pass under `no_edit`. The earlier
report's six unchanged defect passes do not reproduce under this versioned contract and current
container images.

Action choice is reported by class, not as a misleading aggregate: an always-replace policy would
score 80% on this suite while having zero demonstrated no-edit recall, so that number alone is not
treated as useful success.

Deletion behavior is covered by deterministic protocol tests, including deletion of a nonempty
region, insertion at an empty region, whitespace-only replacement, Unicode byte offsets,
malformed/truncated responses, and token-cap output. This suite has no gold deletion cases, so its
deletion success rate is correctly reported as unavailable rather than zero.

Machine-readable controls are in `reports/code_cpt/research_r1/next_edit_controls/`.
