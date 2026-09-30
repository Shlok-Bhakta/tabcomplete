# Source verification execution inputs

`test_commit_sequence_source_verification.v2.py.txt` is the byte-for-byte test source used by the completed source-verification plan v2. Its SHA-256 is `71ddc45b128129a921e9eb2529fb712ee07c3487df40a8bd651950c92e1d4556`, matching the frozen plan's `test_sha256`.

The current test includes a portability-only change: its plan-construction check now uses synthetic files under pytest's temporary directory and stubs the `gh --version` call. The historical plan and result are unchanged. The v2 input loader is expected to reject the current test-source hash; the completed execution must not be resumed or rewritten under changed inputs. A future execution needs a newly frozen plan.
