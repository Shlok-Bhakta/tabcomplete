#!/usr/bin/env python3
"""DeepSeek staged generation launcher (Stage A default: 10 states).

Reads DEEPSEEK_API_KEY from ~/.hermes/.env ONLY into process memory — the key
never appears in argv, logs, reports, or git. Sets ALLOW_PAID_SYNTHETIC=1
in-process (scoped to this run); Budget caps still enforced.
Usage: uv run python scripts/deepseek_stage_a.py [--states 10 --seed 1]
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

HERMES_ENV = os.path.expanduser("~/.hermes/.env")


def load_hermes_env() -> None:
    try:
        with open(HERMES_ENV, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not found in ~/.hermes/.env", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="data/generated/teacher_deepseek.jsonl")
    parser.add_argument("--stage", default="")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    load_hermes_env()
    os.environ["ALLOW_PAID_SYNTHETIC"] = "1"

    import asyncio

    import httpx

    async def info() -> int:
        headers = {"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]}
        async with httpx.AsyncClient(timeout=30.0) as client:
            models = (await client.get("https://api.deepseek.com/models", headers=headers)).json()
            print("models:", sorted(m.get("id") for m in models.get("data", [])))
            try:
                bal = (
                    await client.get("https://api.deepseek.com/user/balance", headers=headers)
                ).json()
                print("balance:", bal)
            except Exception as exc:
                print("balance lookup failed:", type(exc).__name__)
        print("utcnow:", datetime.datetime.now(datetime.UTC).isoformat())
        return 0

    if args.list_models:
        return asyncio.run(info())

    from tinycomplete.teacher.generate import run_generation

    summary = run_generation(
        "deepseek",
        max_states=args.states,
        seed=args.seed,
        out_path=args.out,
        stage_override=args.stage,
        **({"model": args.model} if args.model else {}),
    )
    print(
        f"states={summary['states']} accepted={summary['accepted']} "
        f"rejected={summary['rejected']} spent~=${summary['spent_usd']:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
