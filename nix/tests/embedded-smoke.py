#!/usr/bin/env python3
"""Local sequential integration check: real mmap models, synthetic source only."""

import argparse
import hashlib
import json
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--port", type=int, default=19194)
args = parser.parse_args()
records = []
for variant, alias, digest, protocol, cap in [
    (
        "qwen",
        "q25",
        "4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb",
        "single-line-edit-v1",
        64,
    ),
    (
        "sweep",
        "sweep",
        "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4",
        "sweep-full-file-v1",
        192,
    ),
]:
    executable = (args.directory / f"tabcomplete-{variant}").resolve()
    base = f"http://127.0.0.1:{args.port}"

    def request(path, payload=None, base=base):
        data = None if payload is None else json.dumps(payload).encode()
        query = urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(query, timeout=90) as response:
            return response.read()

    log = args.directory / f"embedded-{variant}-smoke.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            [str(executable), "--port", str(args.port)], stdout=output, stderr=output
        )
        started = time.monotonic()
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{variant}: executable failed, inspect local log")
                try:
                    health = json.loads(request("/health"))
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() - started > 90:
                        raise RuntimeError(f"{variant}: health timeout") from None
                    time.sleep(0.1)
            assert (
                health["alias"],
                health["model_sha256"],
                health["model_protocol"],
                health["output_tokens"],
            ) == (alias, digest, protocol, cap)
            assert health["model_embedded"] and not health["model_switch_supported"]
            assert health["model_storage"] == "executable-mmap"
            assert health["context_size"] == 2304 and health["input_tokens"] == 1024
            assert health["microbatch_size"] == 64
            models = json.loads(request("/v1/models"))
            assert list(models["models"]) == [alias]
            try:
                request("/v1/model", {"alias": alias})
                raise AssertionError("embedded switching accepted")
            except urllib.error.HTTPError as error:
                assert error.code == 405
            state = dict(
                file_id="synthetic.rs",
                filetype="rust",
                source="fn main() { let value = 1; }\n",
                target_row=0,
                cursor_col=0,
                history=[],
                relevant=[],
            )
            prepared = json.loads(request("/v1/editor/context", {"state": state, "buffers": []}))
            generation = dict(
                prompt=prepared["prompt"], n_predict=8, repository_identity="embedded-smoke"
            )
            if prepared.get("window") is not None:
                generation["window"] = prepared["window"]
            events = request("/completion", generation).decode()
            terminal = [
                json.loads(row[6:]) for row in events.splitlines() if row.startswith("data: ")
            ][-1]
            assert terminal["stop"] and terminal["stop_type"] != "error"
            assert terminal["model_sha256"] == digest
            assert "action_validation" in terminal
            mappings = []
            for row in Path(f"/proc/{process.pid}/maps").read_text().splitlines():
                fields = row.split(maxsplit=5)
                if len(fields) == 6 and fields[5] == str(executable) and fields[1] == "r--s":
                    begin, end = fields[0].split("-")
                    mappings.append(
                        dict(
                            mapped_bytes=int(end, 16) - int(begin, 16),
                            offset=int(fields[2], 16),
                            permissions=fields[1],
                        )
                    )
            assert mappings and sum(item["mapped_bytes"] for item in mappings) > 100_000_000
            descriptors = []
            for descriptor in Path(f"/proc/{process.pid}/fd").iterdir():
                try:
                    descriptors.append(str(descriptor.readlink()))
                except FileNotFoundError:
                    pass
            assert str(executable) in descriptors
            assert not any(path.endswith(".gguf") for path in descriptors)
            status = Path(f"/proc/{process.pid}/status").read_text().splitlines()
            memory = {
                row.split(":", 1)[0]: row.split(":", 1)[1].strip()
                for row in status
                if row.startswith(("VmRSS:", "VmHWM:", "RssFile:", "RssAnon:"))
            }
            binary_digest = hashlib.file_digest(executable.open("rb"), "sha256").hexdigest()
            records.append(
                dict(
                    variant=variant,
                    pid=process.pid,
                    executable=str(executable),
                    bytes=executable.stat().st_size,
                    binary_sha256=binary_digest,
                    health=health,
                    model_mappings=mappings,
                    memory=memory,
                    generation={
                        key: terminal.get(key)
                        for key in [
                            "stop_type",
                            "tokens_predicted",
                            "canonical_action",
                            "action_validation",
                            "timings",
                        ]
                    },
                    external_gguf_descriptors=0,
                    one_process=True,
                )
            )
            print(
                f"{variant}: embedded hash/defaults/mmap/switch rejection/synthetic decode PASS",
                flush=True,
            )
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    assert process.poll() is not None
result = dict(
    schema_version=1,
    records=records,
    total_executable_bytes=sum(record["bytes"] for record in records),
    production_mutated=False,
)
assert result["total_executable_bytes"] < 2 * 1024**3
(args.directory / "embedded-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
