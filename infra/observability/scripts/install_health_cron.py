"""Install this deployment's storage check without replacing other cron jobs."""

import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
directory = args.directory.resolve()
if any(character in str(directory) for character in " \n\r\t%"):
    raise SystemExit("Deployment directory must not contain shell/cron separators")
existing = subprocess.run(["crontab", "-l"], text=True, capture_output=True)
if existing.returncode not in (0, 1):
    raise SystemExit("Cannot inspect existing crontab")
tag = "# tabcomplete-observability-health"
lines = [line for line in existing.stdout.splitlines() if not line.endswith(tag)]
lines.append(f"* * * * * {directory}/scripts/storage-health.sh >/dev/null 2>&1 {tag}")
subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)
print("Storage health check installed every minute; other cron entries preserved")
