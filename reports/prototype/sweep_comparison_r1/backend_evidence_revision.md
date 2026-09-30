# Backend evidence and allocation revision

The first two allocations generated no quality predictions. Attempt 1 failed during CUDA configuration. Attempt 2 compiled the CUDA runtime and loaded the model, but its placement-evidence guard failed because verbosity 3 suppressed GGML INFO messages. The pinned runtime maps those messages to verbosity 4.

The revised runner uses verbosity 4 and writes startup arguments, device snapshots, and parsed placement evidence before checking the same mandatory CUDA-device and offloaded-layer guards. Debug verbosity 5 and per-token instrumentation remain disabled. This change does not establish that the second allocation used CPU inference; its backend placement was unverified.

The memory field formerly called `file_private_bytes` summed Private_Clean and Private_Dirty, which also includes anonymous memory. New measurements name this `private_bytes` and retain separate Pss_File and Pss_Anon observations. Historical measurements remain intact. Legacy provider methods and scorer type narrowing also receive compatibility fixes; the detailed generation and scoring paths retain their original behavior.

Model artifacts, tokenizer, runtime revision, prompt, fixtures, decoding, output caps, termination checks, action mapping, and scoring rules are unchanged. A new allocation amendment allows one final attempt, only after both previous terminal errors are authenticated and their conservative wall bounds plus the new two-hour deadline fit the four-hour cap. Quota, renewal, active jobs, source identity, and unique notebook identity are checked again before submission.

Plan v6 is preserved. Its top-level revision number 3 was rejected by the benchmark loader, which supports revision 2. Plan v7 corrects that interface field before submission and retains allocation amendment 6. Neither plan was used to generate comparison outputs before this correction. Scoring plans are likewise versioned before reading outputs.
