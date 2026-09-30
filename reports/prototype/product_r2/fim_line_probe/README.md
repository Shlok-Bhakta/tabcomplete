# Qwen line-completion diagnostic

The first two cases for each language in the unchanged public 180-case line
suite were selected before generation. The untouched Qwen2.5-Coder-0.5B Q4
artifact and pinned native CPU runtime received raw source prefixes and the
publisher's prefix/suffix/middle format. The latter supplies suffix information,
so this measures a task/prompt difference, not a systems-only optimization.
The format follows the [publisher example](https://github.com/QwenLM/Qwen3-Coder/blob/main/examples/Qwen2.5-Coder-fim.py).

## Actual v1 results

Seventy model requests ran, with two repetitions. One Python FIM input contained
3,233 tokens and exceeded the fixed 2,976-token input budget. It was skipped both
times without inference or truncation. Seventeen distinct cases support a paired
comparison: FIM exactly matched six, raw-prefix completion matched one, and all
five disagreements favored FIM. Repetitions do not double the quality sample.
This small ordered diagnostic does not establish general edit quality.

FIM request latency was 2,253.6 ms median and 27,837.3 ms p95 across its 34 actual
requests. Raw-prefix latency was 1,089.7 ms median and 15,742.6 ms p95 across 36
requests. These are completed native requests, excluding editor debounce/UI.
There was no prompt-cache reuse. Retained and sampled peak predictor RSS were
506,478,592 bytes, PSS 500,186,112 bytes, and process swap zero. Host swap-in
increased two pages and swap-out did not increase. The installed editor service
remained running, and no competing compiler or benchmark ran during this window.
Its separate CPU/memory use was not sampled in this diagnostic.

## Reporting correction

Version 1 placed completion metadata assignment outside the non-overflow branch.
The two skipped records incorrectly inherited the preceding request's native
metadata and were labeled completed. They have no scoring fields. Preserve those
historical records and summary hashes. `results-v1.json` reports the actual 70
model requests and excludes the two over-budget rows from latency and quality.
No output text was repaired or regenerated. The existing score function is
unchanged.

The regression test requires an over-budget row to skip inference and contain
no preceding response, termination or outcome. Version 2 returns skipped rows
before generation and distinguishes planned slots from actual model requests.
A new source-bound plan is required for further execution. The selected-model
v2 plan uses exactly the existing adapted Q4 artifact; it does not change the
installed service or weights.

## Selected-model v2 results

The corrected runner completed 70 actual requests and recorded two budget skips.
FIM matched four of the 17 paired cases exactly, versus six for the untouched
model. This small diagnostic does not establish equivalence or a general training
regression. Selected-model FIM request latency was 2,169.4 ms median and
27,291.0 ms p95 across 34 requests, with explicit native termination in all 34.
Peak and retained RSS were 505,356,288 bytes. PSS was 324,715,520 bytes; the selected
weight file also backs the existing resident editor service, so shared file pages
lower PSS. This is not evidence that adaptation reduced predictor RAM. All actual
requests reported backend cache_n zero.

These results support investigating completion prompting, but they do not yet
justify a useful-product claim. The current editor configuration remains intact.

The useful-product objective remains open. No model is promoted by this report.
