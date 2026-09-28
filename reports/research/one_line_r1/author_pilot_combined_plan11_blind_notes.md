# Plan-11 combined pilot: frozen 12-case blind notes

I read only `author_pilot_combined_plan11_blind_sample.jsonl` before opening
authored actions, objectives, or focus tags. The audit chose these 12 candidate
source IDs by the predeclared SHA rule. Cases 10 and 11 overlap the earlier
seven-row review, so their judgments below carry forward the prior blind notes.
These are hypotheses from student-visible context, not accepted labels.

| Sample | Source suffix | Student-visible reason | Blind judgment |
| ---: | --- | --- | --- |
| 1 | `0ac93e6d` | Recent edit replaced a simple enumeration of `Brand` with a list comprehension that also enumerates and checks `i <= len(list(Brand))`. The bound appears redundant, but the edit was just made. | Ambiguous. Simplifying the new line is plausible; keep is also plausible, and replacing it with the old logic could reverse the latest edit. |
| 2 | `83503c23` | Method definition changed from `getClient` to `getServiceClient`; the target still calls `a.getClient(ctx)`. | Clear replacement of the target call with `a.getServiceClient(ctx)`. |
| 3 | `c5145b75` | The incident route became a `routes` array entry, while the target deep-dive route is still registered with `router.use`. No loop applies `routes` in the visible file. | Ambiguous. Migrating the target to the array may be intended, but the array is currently unused; keep is safer from this context. |
| 4 | `8d66c254` | Prior transaction-begin failure now returns. Prepare and Exec errors still merely log, and the target executes `stem`. | Ambiguous. Error handling needs work, but a safe one-line change at the target is not established. The preceding Prepare error branch may need the repair. |
| 5 | `c712598c` | Recent edit deliberately replaced `this.typingAnimate()` with a log call. The target is the following closing brace. | Clear keep. Inserting the removed call would undo the latest visible edit without new intent. |
| 6 | `f24cfbbc` | A `VariantType::Content` branch constructs `Variant::String`; recent history confirms a distinct `Variant::Content` representation. | Likely replace the target's variant kind. The exact inner conversion type is not shown. |
| 7 | `e3334f02` | Target is append at EOF of a small TypeScript teaching example. Recent history changed an existing `soma` call; it implies no particular new line. | Clear keep; no source-grounded insertion target is visible. |
| 8 | `1b6c9917` | Getter changed from `getVar` to `getVariable`; setter still calls `setVar`. The `VarTree` API is not visible. | Ambiguous. A matching setter rename is possible, but getter and setter names need not change together. |
| 9 | `b4a2039a` | Writer local was renamed from `out` to `w`; target still says `defer out.Flush()`. | Clear replacement to `defer w.Flush()`. |
| 10 | `79d01114` | Production Mongo registration recently gained `connect=False`; development registration lacks it. | Ambiguous. Matching it is plausible, but production and development connection behavior may differ. This is the earlier blind judgment. |
| 11 | `44bcb0af` | Recent history changed a distant call from `os.mkdir` to `mkdir`; target imports `mkdir`. | Clear keep. This is the earlier blind judgment. |
| 12 | `9946e471` | A different push method's `fillHeader` call gained `true`; target `BroadcastPush` still calls it without `true`. The flag's meaning is absent. | Ambiguous. Propagation may be intended, but the two methods may need different header behavior. |

The blind pass found six cases with a supported core action or keep, and six
with multiple plausible next edits. This is a small conditional sample of
structurally valid candidates, not a quality estimate for all 100 source
attempts.
