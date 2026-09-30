# Sweep comparison allocation amendment 4

`plan-v4.json` preserves `plan-v3.json` and adds authenticated quota and terminal-status evidence for attempt 1. It corrects the v3 retry gate: exact provider allocation wall time is unknown, so the budget uses a conservative time upper bound and the separately observed quota delta. Unknown exact provider time alone does not block the one remaining retry.

- Comparison revision 2 self-hash: `53256921e9135b432ed029e7a083c3ee12c27619ce200c4be4c5b91dc5200b51`
- Allocation amendment 3 self-hash: `26781d965031fa3c486a771d4b75f8a08851b8aa4b82eaa095e87ad22e89bdb6`
- Allocation amendment 3 file SHA-256: `ecaada053a6c407c354562208c969b19c22ee54aa282b7d2296635550d3e5511`
- Allocation amendment 4 self-hash: `c340bb87b64c3af927be00f02497d43459a7f417a2fbc82eb1c5feccf5f5eb4b`
- Allocation amendment 4 file SHA-256: `3c2b264ea6aa7ef17004cb08ecf89092eef741a398b10a20ea91130a5aeb55f0`

The pre-submit authenticated quota observation was at `2026-09-30T08:17:13.958025Z`. An authenticated terminal `ERROR` status was observed at `2026-09-30T08:39:46.187999Z`. This gives a conservative upper bound of 1,352.229974 seconds for attempt 1, including time before launch, queue/setup, execution, and status observation. It is not the exact provider GPU-session duration. The two quota snapshots report a 0.02 GPU-hour increase; their 0.01-hour rounding means this is not converted into exact seconds. Their source hashes and the terminal-observation hash are in the plan.

One retry may use at most 7,200 session seconds, including its 1,200-second finalization reserve. The conservative attempt-1 bound plus that maximum retry is 8,552.229974 seconds, below the 14,400-second campaign cap by 5,847.770026 seconds. The cap includes failed work and setup. A fresh authenticated quota and job-status check is still required immediately before retry; automatic renewal use remains disabled. There is no third attempt.

The retry orchestration source hash is `882ef00fe0c8f457c921d08a6578521a69636653a797cea6cb054720da43b050`. The focused current tests passed: 39 tests in 0.67 seconds with `uv run --frozen --no-sync pytest -q tests/test_sweep_kaggle_submission.py tests/test_sweep_kaggle_worker.py`. The model runner, model, runtime, prompts, fixtures, and scoring identities remain those frozen in revision 2. `load_and_verify_plan` accepted v4 with those exact inputs. No model outputs were created while preparing the amendment, and no retry was submitted by this task.
