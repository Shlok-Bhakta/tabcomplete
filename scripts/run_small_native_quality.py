"""Run the existing R2 suites against one pinned local Q4 server at a time."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "tabcomplete/outputs/tools/llama.cpp/build/bin/llama-server"
LINE = (
    ROOT.parent
    / "tabcomplete-model-data-r2/artifacts/research/model_data_r2/frozen-corpora"
    / "causal_line_v1-r3.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19095)
    parser.add_argument("--context-tokens", type=int, default=2304)
    parser.add_argument("--line-only", action="store_true")
    parser.add_argument("--suite-revision", default="small-q4-v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.model.is_file() or not LINE.is_file():
        raise FileNotFoundError("model or frozen line suite missing")
    command = [
        str(RUNTIME),
        "-m",
        str(args.model.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-t",
        "4",
        "-tb",
        "4",
        "-ngl",
        "0",
        "-c",
        str(args.context_tokens),
        "-b",
        "256",
        "-ub",
        "64",
        "-np",
        "1",
        "--cache-ram",
        "128",
        "--no-cache-idle-slots",
        "--no-warmup",
    ]
    url = f"http://127.0.0.1:{args.port}"
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(900):
                if server.poll() is not None:
                    raise RuntimeError("native server exited before readiness")
                try:
                    if httpx.get(url + "/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("native server load exceeded 180 seconds")
            environment = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
            }
            command = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/run_r2_quantized_quality.py",
                "--url",
                url,
                "--alias",
                args.alias,
                "--model",
                str(args.model.resolve()),
                "--line-suite",
                str(LINE),
                "--output",
                str(args.output),
                "--hardware",
                "crabcake-desktop-CPU",
                "--campaign-id",
                "small-model-prototype-r1",
                "--plan",
                "reports/research/small_model_prototype_r1/plan.json",
                "--threads",
                "4",
                "--context-tokens",
                str(args.context_tokens),
                "--suite-revision",
                args.suite_revision,
            ]
            if args.line_only:
                command.append("--line-only")
            with (args.output / "runner.log").open("w") as run_log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=run_log,
                    stderr=subprocess.STDOUT,
                    timeout=7200,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError("native quality generation failed; inspect private runner log")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
