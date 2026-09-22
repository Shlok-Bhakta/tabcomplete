"""Produce code-completion v2 with C/C++ tests that execute and verify behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinycomplete.eval.code_benchmark import BenchmarkCase, load_suite

EXPECTED_STDOUT = {
    "c/stable_unique_id_2": "3\n",
    "c/stable_unique_id_3": "5\n",
    "c/stable_unique_id_4": "123\n",
    "c/stable_unique_id_6": "30\n",
    "c/stable_unique_id_7": "two\n",
    "c/stable_unique_id_8": "5\n",
}

EXPECTED_REPLACEMENTS = {
    "c/stable_13": (
        "int64_t g = gcd(a, b);\n"
        "    int64_t value = (a / g) * b;\n"
        "    return value < 0 ? -value : value;"
    )
}


def upgrade(case: BenchmarkCase) -> BenchmarkCase:
    check = case.check
    updates: dict[str, object] = {}
    if case.id in EXPECTED_STDOUT:
        updates["expected_stdout"] = EXPECTED_STDOUT[case.id]
    if case.language in {"c", "cpp"} and check.test and not check.run:
        command = list(check.test)
        if "-o" in command:
            command[command.index("-o") + 1] = ".tabcomplete-test-bin"
            updates.update(test=command, run=["./.tabcomplete-test-bin"])
    return case.model_copy(
        update={
            "check": check.model_copy(update=updates),
            "expected": EXPECTED_REPLACEMENTS.get(case.id, case.expected),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/benchmarks/code_completion_v1.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/benchmarks/code_completion_v2.jsonl")
    )
    args = parser.parse_args()
    cases = [upgrade(case) for case in load_suite(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(mode="json"), sort_keys=True) + "\n")
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {len(cases)} cases sha256={digest}")


if __name__ == "__main__":
    main()
