"""SSH/stdin offline import with its canonical ledger on verified Kiwi storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from bootstrap_signoz import read_environment
from storage_guard import check, inspect_mount, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    environment = read_environment(args.directory / ".env")
    root = Path(environment["TABCOMPLETE_OBS_DATA_ROOT"])
    mount = inspect_mount(root, None)
    validate(
        root,
        mount,
        expected_source=environment["TABCOMPLETE_OBS_VOLUME_SOURCE"],
        expected_uuid=environment["TABCOMPLETE_OBS_VOLUME_UUID"],
        minimum_free_bytes=100 * 2**30,
    )
    check(root, mount)
    payload = sys.stdin.buffer.read(256 * 2**20 + 1)
    if len(payload) > 256 * 2**20:
        raise ValueError("Offline bundle exceeds limit")
    digest = hashlib.sha256(payload).hexdigest()
    spool = root / "gateway-state" / "offline-imports"
    spool.mkdir(mode=0o700, exist_ok=True)
    bundle = spool / (digest + ".jsonl")
    if not bundle.exists():
        with os.fdopen(
            os.open(bundle, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
        ) as handle:
            handle.write(payload)
    sys.path.insert(0, str(args.directory / "python-runtime"))
    from tinycomplete.observability.offline import import_bundle

    result = import_bundle(
        bundle,
        root / "gateway-state" / "imports.sqlite",
        f"http://{environment['TABCOMPLETE_OBS_TAILSCALE_IP']}:"
        f"{environment.get('TABCOMPLETE_OBS_OTLP_HTTP_PORT', '4318')}",
    )
    bundle.unlink()  # Only this content-addressed temporary upload; source outputs are untouched.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        raise SystemExit(3) from None
