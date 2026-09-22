#!/usr/bin/env python3
"""Record an existing summary as historical, without inventing unobserved timings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinycomplete.observability.bootstrap import initialize_observability
from tinycomplete.observability.config import ObservabilityConfig
from tinycomplete.observability.context import RunContext
from tinycomplete.observability.spans import operation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.bundle.exists():
        raise SystemExit("Bundle exists; import it instead of creating new timestamps")
    payload = args.summary.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    summary = json.loads(payload)
    identity = RunContext.new(
        campaign_id="historical-research-r1", run_id="historical-" + digest[:24]
    )
    runtime = initialize_observability(
        ObservabilityConfig(enabled=True, mode="offline", offline_bundle=args.bundle)
    )
    attrs = {
        "tabcomplete.run.state": "completed",
        "tabcomplete.historical": True,
        "tabcomplete.timestamp.kind": "summary_import_time",
        "tabcomplete.source.summary.sha256": digest,
        "tabcomplete.source.summary.file": args.summary.name,
        "tabcomplete.phase": "historical_summary",
        "tabcomplete.terminal_event_id": identity.run_id + ":summary",
    }
    for source, target in [
        ("additional_input_tokens", "input_tokens"),
        ("additional_scored_target_tokens", "scored_tokens"),
        ("successful_optimizer_updates", "successful_updates"),
        ("skipped_optimizer_updates", "skipped_updates"),
    ]:
        if source in summary:
            attrs["tabcomplete.training." + target] = summary[source]
    with runtime.activate(), identity.activate(), operation("run.summary", attributes=attrs):
        pass
    runtime.shutdown()
    print(
        json.dumps(
            {
                "run_id": identity.run_id,
                "bundle": str(args.bundle),
                "source_sha256": digest,
                "timestamps": "import record time; source execution timestamps unavailable",
            }
        )
    )


if __name__ == "__main__":
    main()
