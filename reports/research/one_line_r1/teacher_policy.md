# Hosted teacher boundary

Observed 2026-09-26: `opencode --version` returned `1.18.31`; the read-only
`opencode models opencode-go` listing included
`opencode-go/muse-spark-1.3-contributor`. No inference request was made.

The [OpenCode Terms of Use](https://opencode.ai/legal/terms-of-service),
effective 2026-08-15, prohibit automatic/programmatic extraction of Output in
section 40(5) and using Output to develop AI models competing with the service
or third-party models in section 40(7). The proposed automated teacher-label and
benchmark-scoring route for this local code model is blocked. Ownership language
in section 65 does not remove those use restrictions. This is an operational
reading of the current terms; a separately documented compatible rights basis
would require a new policy review and code change.

`teacher.py` has no network/provider path. It rejects all OpenCode egress for
this campaign, including public and synthetic prompts. It separately rejects
private or sealed source and obvious credential patterns before evaluating the
output-use restriction. The pattern scan is only a backstop; provenance review
would still be needed for any future external route.

The append-only usage ledger enforces the campaign caps of 25,000 calls,
60 million input tokens, and 20 million output tokens using locked reservations.
It rejects duplicate request IDs, malformed prior records, missing reported
usage, and actual usage above the reservation. Reservations count their maximum
until settled. Ledger accounting never authorizes a provider request.

Provider calls: **0**. Provider training labels: **0**. Provider-scored cases:
**0**. Public-source edit history and local deterministic validation are the
available data route under this campaign plan.
