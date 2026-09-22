"""Add the existing restricted upload credential to gateway deployment configuration."""

import argparse
from pathlib import Path

from bootstrap_signoz import replace_environment_value

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
replace_environment_value(
    args.directory / ".env",
    "TABCOMPLETE_ARTIFACT_UPLOAD_TOKEN",
    (args.directory / ".artifact-upload-token").read_text().strip(),
)
print("Gateway upload credential configured; no credential output")
