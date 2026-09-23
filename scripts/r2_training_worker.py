"""Invoke the production trainer with separate offline files for each owned rank."""

import os
from pathlib import Path

if os.environ.get("TABCOMPLETE_OBSERVABILITY_MODE") == "offline":
    bundle = Path(os.environ["TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE"])
    rank = int(os.environ.get("RANK", "0"))
    os.environ["TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE"] = str(
        bundle.with_name(f"{bundle.stem}-rank-{rank}{bundle.suffix}")
    )

from tinycomplete.code_cpt.train import main  # noqa: E402
from tinycomplete.observability.bootstrap import current_runtime  # noqa: E402
from tinycomplete.observability.spans import operation  # noqa: E402

if __name__ == "__main__":
    try:
        with operation("campaign.phase", attributes={"tabcomplete.phase": "training.worker"}):
            main()
    finally:
        current_runtime().shutdown()
