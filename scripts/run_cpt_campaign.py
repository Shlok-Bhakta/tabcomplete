#!/usr/bin/env python3
"""Submit, monitor, collect, and verify the bounded research-r1 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinycomplete.code_cpt.campaign import CampaignOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    orchestrator = CampaignOrchestrator(repository, args.config)
    result = orchestrator.execute() if args.execute else orchestrator.validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
