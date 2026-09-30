# Commit sequence source verification

The bounded exact-source check covered the frozen train/development review queue
from queue plan v4. The queue remained unchanged at 96 train and 32 development
rows; no reserved evaluation rows were read.

The verifier fetched each queued child commit and required a single parent. It
fetched both exact file snapshots, compared them with the queue's expected
SHA-256 hashes, and checked root and nearest-ancestor license evidence plus
source SPDX headers at both revisions. A candidate passed only when source
bytes matched, the sensitive-text filter passed, and one allowlisted SPDX
identifier matched the dataset's license claim. Ambiguous, missing, or
unverified evidence was skipped.

## Results

- 97 of 128 candidates passed exact parent/child source and license verification: 71 train and 26 development.
- 31 were skipped: 5 ambiguous nearest licenses, 5 missing source/repository responses, and 21 unknown or mismatched license evidence.
- Verified language counts: Python 33, TypeScript 27, Go 25, Rust 12.
- Verified action counts: 39 replacements, 30 insertions, 28 deletions.
- License identifiers: MIT 63, Apache-2.0 19, BSD-3-Clause 10, BSD-2-Clause 4, ISC 1.
- Five distinct train groups are marked for possible later review. They are not training labels or evidence of inferability.
- No human chronology, inferability, or objective correctness was established. Training acceptance remains zero.

The run took 908.459 seconds and received 4,142,368 response-body bytes against
the 20 MiB cap. Private output occupied 752,240 bytes, including the final
manifest, against the 256 MiB cap. Private directories were mode 0700 and files
mode 0600. The final integrity pass rechecked all stored source/license hashes.

## Reproducibility and privacy

Plan v2 was frozen before any GitHub request. Its raw SHA-256 is
`26af54a339fff90434c42b2f28a86e518e4d4d2336b12b1f15d948d1232e2ebd`; its
canonical identity is
`eb586ae0d2c88a4e0be704085a6b7426ad71510930aa19842c0bd56f34658fd1`. Plan v1
is preserved as a failed preflight: its loader compared the frozen inputs to a
plan object that still included its embedded identity, so it stopped before
authentication, network access, or output.

The verifier source SHA-256 is
`e9560e8d1be633c0056982512f4a69cec81c31306bc1cb7359090bd8c01fee55`; targeted
tests passed (6 tests), Ruff passed, and mypy passed. The private output is at
`/mnt/ssd/tabcomplete-product-r2/commitpackft/source-verification-v1`:

- Manifest SHA-256: `3b3f5de211e8836b4e7fed067af7f794c51182f746267d6f4da09c4c8bbc1b0e`
- Candidate results SHA-256: `fe993904622cb44090c07c42775d621ea82050959435c6004b85e720e2e1efd9`

Raw source snapshots and license text remain on the private SSD. The repository
contains only aggregate qualification counts and hashes. There were no teacher
or model-provider calls, weight downloads, or training runs.

The available provider usage ledger has request IDs and evidence hashes but no
candidate/source identity field. It contains no direct match to these five
candidate IDs, but historical candidate-level request overlap is unknown
without that mapping.
