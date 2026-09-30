# Sweep comparison allocation amendment 5

`plan-v5.json` preserves v3 and v4 and pins the final retry worker source after its provenance-recording changes. It does not change the model, inference runtime revision, prompts, fixtures, scoring, or limits.

- Comparison revision 2 self-hash: `53256921e9135b432ed029e7a083c3ee12c27619ce200c4be4c5b91dc5200b51`
- Allocation amendment 4 self-hash: `c340bb87b64c3af927be00f02497d43459a7f417a2fbc82eb1c5feccf5f5eb4b`
- Allocation amendment 4 file SHA-256: `3c2b264ea6aa7ef17004cb08ecf89092eef741a398b10a20ea91130a5aeb55f0`
- Allocation amendment 5 self-hash: `f3c508b85af695da9ed2087d84659c949a58223e2d9dabbd325a6c355fb4b163`
- Allocation amendment 5 file SHA-256: `bbb9091b9f2ba91107364f4084ac90f93fddbfd511f777bf78bd3373f120ed15`

The current worker SHA-256 is `e4e2522b99c1462284207c9fb3b42409d2cb2c86fa4a9b010177587c7130e414`. It records the actual CMake driver flags and records optional version-probe failures as unavailable. The preparation source and its tests are unchanged from v4. The focused submission/worker test command passed all 39 tests.

The v4 accounting remains in force: attempt 1 has a conservative 1,352.229974-second upper bound from the authenticated pre-submit quota timestamp to authenticated terminal `ERROR`. Exact provider wall time remains unknown. Adding the single permitted 7,200-second retry gives an 8,552.229974-second worst-case bound against the 14,400-second cap. A fresh authenticated quota and job-status gate is still required immediately before retry; no additional attempt or automatic renewal use is permitted.

`load_and_verify_plan` accepted v5 with the frozen runner, test, model, fixture, and runtime identities. No comparison prediction was generated while creating this amendment; v4 remains byte-for-byte unchanged.
