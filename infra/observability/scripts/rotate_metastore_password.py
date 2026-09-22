"""Replace Foundry's initial local-only database password without printing it."""

import argparse
import secrets
import subprocess
from pathlib import Path

from bootstrap_signoz import read_environment, replace_environment_value

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
environment = args.directory / ".env"
current = read_environment(environment)
if "TABCOMPLETE_OBS_POSTGRES_PASSWORD" not in current:
    password = secrets.token_hex(32)
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "tabcomplete-observability-metastore-postgres-0",
            "psql",
            "-U",
            "signoz",
            "-d",
            "signoz",
            "-v",
            "ON_ERROR_STOP=1",
        ],
        input=f"ALTER ROLE signoz PASSWORD '{password}';\n",
        text=True,
        capture_output=True,
        check=True,
    )
    replace_environment_value(environment, "TABCOMPLETE_OBS_POSTGRES_PASSWORD", password)
print("Restricted metastore password configured")
