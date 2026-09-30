# Local Q4/Q8 next-edit result interpretation

This revision interprets the frozen local CPU replay. It adds no predictions and does not alter or repair any result. The JSON companion contains the full score-artifact SHA-256 manifest and observability IDs.

## Result

The replay used 24 fixed public/synthetic states across seven languages, with two changed-state repetitions per precision. These are not 24 live editor trajectories and contain no human acceptance evidence.

| Precision | Valid EOS, below cap | Byte-safe mapped actions | Mapped clear-intent cases | Interpretation |
|---|---:|---:|---:|---|
| Q4_K_M | 48/48 | 0/48 | 0/11 cases | No output could be applied under the frozen whole-file-to-edit-region mapper. Functional behavior is unassessed because no mapped patch reached the source oracle. |
| Q8_0 | 48/48 | 0/48 | 0/11 cases | Same mapping failure. This does not establish equivalence with Q4_K_M. |

The frozen score records zero exact-reference and clear-intent functional successes because no response mapped into the editable range. No functional source checks ran for those outputs. The no-edit false-positive count of zero is also not a zero false-positive rate: no action mapped, so that rate is undefined.

## Why mapping failed

Every raw response began with LF; none of the corresponding source files began with LF. The official score did not strip it. For diagnosis only, an in-memory check skipped that byte and then compared the exact immutable prefix and suffix. This was not a score adjustment.

| Diagnostic category after the LF-only check | Q4_K_M states | Q8_0 states |
|---|---:|---:|
| Leading LF alone explains the out-of-range mismatch; in-range replacement remains | 2/24 | 2/24 |
| Prefix preserved, but immutable suffix changed or absent | 15/24 | 18/24 |
| Response too short to contain both immutable spans | 7/24 | 3/24 |
| Enough total bytes, but neither immutable span matches | 0/24 | 1/24 |

The two separator-only states are the same for both precisions. All responses had an observed EOS and none hit the output cap. Short responses therefore indicate an incomplete whole-file body under this mapping contract, not a runtime token-cap cutoff. For suffix mismatches, these records cannot distinguish intentional out-of-range changes from omitted body. The official result remains 0/48 mapped actions for each precision.

The frozen prompt builder ends at the updated-file path header without a trailing LF. A future input-only separator variant would require a new frozen prompt plan. This report does not test that variant and does not normalize output text.

## Repeats and null actions

Raw response hashes matched across the two changed-state repetitions in 24/24 cases for each precision. In the separate immediate same-prompt cache probes, raw response hashes matched in 46/48 pairs per precision. Those probes have zero quality weight.

The score file reports 48 `identical_actions` for the immediate repeats and 24 per-case `identical_action_count`. Those counts compare `None == None` because all outputs were unmapped. Valid mapped-action comparisons are 0, so action consistency is undefined. Only raw response hash consistency is meaningful here.

## Telemetry verification

The baseline offline bundle imported 500 spans; the controller score bundle imported two. The existing observability CLI returned `state=ok` for the fully paginated baseline run, the controller run, a genuine request lookup, and its generation trace. The baseline run contained 192 `request.start` and 192 `model.generate` spans; all 192 local replay request IDs were recovered through unique case/precision/request-kind/timestamp joins. The controller scoring run contains only start/summary spans and no inference requests.

Example linkage: `python/stable_11` / `q4_k_m` / repetition `0` → request `request-968a672e-06ef-48ea-a855-17ee4f51267d` → generation trace `eb64ded111f1a34a96ae0515192b658d`. The request lookup returned both `request.start` and `model.generate`; the trace lookup returned the matching generation span. The request-start and generation spans share the request ID but have separate trace IDs.

Content capture was disabled in the baseline plan and metadata, and the scorer set capture content to `0`. The telemetry bundles have null input/output preview fields and provide no prompt/completion payload. Frozen public/synthetic local prediction files were read for byte-boundary diagnosis; no raw text was copied into this report. The report stores IDs and hashes only. Gateway `last_ingested_at` was null, so ingestion time remains unknown. No metric labels were added and request UUIDs were not used as metric labels.

## Artifacts and limits

The JSON companion binds the exact scoring-plan identity, inputs, producer/scorer source hashes, predictions, score outputs, measurements, telemetry bundles, and model hashes. It preserves the original scoring implementation identity even though later working-tree source changes exist. No raw prediction text is copied here.

This fixed synthetic benchmark result does not establish human acceptance, next-edit generalization, Q4/Q8 equivalence, or production quality. Neither precision produced a safe mapped action in this suite.
