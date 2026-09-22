# Bounded queries

The gateway's `queryPayload` builds SigNoz v5 raw trace queries. It caps pages at
200 rows, offsets at 10,000, windows at 90 days, and backend requests at 8 seconds.
No browser-provided SQL, filter expression, or file path reaches SigNoz.

Native dashboard queries are defined in `../dashboards/runs-and-models.json` and
compiled to SigNoz v0.142.1 dashboard schema v6 by `../scripts/provision.py`.
These query the same operation names and attribute schema as the gateway.

Domain filters use `tabcomplete.run_id`, `tabcomplete.request_id`, and `trace_id`.
Deduplicate completed cases by `tabcomplete.terminal_event_id`, falling back to
trace/span ID for old records. Different suite hashes and output protocols must
not share a quality denominator. Raw telemetry counts are not scientific totals.
