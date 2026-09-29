"""Run CPU preflight, explicitly submit, or verify output for the private pilot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "kaggle/one_line_gpu_pilot_r1"
sys.path.insert(0, str(PILOT))

from build_pilot import main  # noqa: E402

if __name__ == "__main__":
    main()
