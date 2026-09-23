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

if __name__ == "__main__":
    main()
