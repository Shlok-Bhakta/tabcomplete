"""Author the fixed synthetic code benchmark with DeepSeek Flash.

The API key is read only from DEEPSEEK_API_KEY. Responses contain synthetic
fixtures and are validated before the final JSONL is written.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import httpx

from tinycomplete.eval.code_benchmark import BenchmarkCase, Prediction, evaluate_prediction
from tinycomplete.teacher.budget import Budget, estimate_cost_usd

MODEL = "deepseek-flash"
URL = "https://api.deepseek.com/chat/completions"
LANGUAGES = ("python", "typescript", "javascript", "java", "cpp", "rust", "go", "c", "csharp")
IMAGES = {
    "python": "docker.io/library/python:3.12-slim",
    "typescript": "localhost/tabcomplete-typescript-bench:5.9.2",
    "javascript": "docker.io/library/node:22-bookworm-slim",
    "java": "docker.io/library/eclipse-temurin:21-jdk",
    "cpp": "docker.io/library/gcc:14",
    "rust": "docker.io/library/rust:1.85-slim",
    "go": "docker.io/library/golang:1.24-bookworm",
    "c": "docker.io/library/gcc:14",
    "csharp": "mcr.microsoft.com/dotnet/sdk:9.0",
}


def prompt(language: str, start: int, count: int, repository_context: bool) -> str:
    kind = "multi-file repository-context" if repository_context else "single-file"
    return f"""Create {count} synthetic {kind} causal code-completion benchmark cases for {language}.
Return one JSON object with a `cases` array and no prose.

Each case must match this schema exactly:
{{
  "id": "{language}/stable_unique_id",
  "language": "{language}",
  "path": "safe/relative/source/path",
  "prefix": "source text strictly before the cursor",
  "suffix": "fixed source text after the insertion",
  "expected": "the exact insertion that solves the task",
  "context_files": {{"safe/relative/context/path": "content"}},
  "check": {{
    "compile": ["executable", "arg"],
    "test": ["executable", "arg"],
    "files": {{"safe/relative/hidden_test_path": "test content"}},
    "timeout_seconds": 10,
    "container_image": "{IMAGES[language]}"
  }},
  "category": "one of control_flow, data_structures, algorithms, standard_library, parsing, error_handling, repository_api",
  "repository_context": {str(repository_context).lower()}
}}

Rules:
- IDs in this batch use numeric indices {start} through {start + count - 1}.
- Code and tests are newly authored and license-safe. Do not copy public benchmark tasks.
- The model under test sees context files and prefix only, never expected, suffix, or hidden tests.
- `prefix + expected + suffix` must be complete, idiomatic, parseable {language}.
- Hidden tests must distinguish plausible wrong implementations and include edge cases.
- Use only the standard library and deterministic behavior. No network, clocks, randomness, subprocesses, or filesystem access outside the fixture.
- Commands are direct argv arrays, never shell strings. Compile/type-check where the language supports it, then run behavioral tests.
- Keep the expected insertion under 120 tokens and the full fixture compact.
- Vary task shape and category. Do not make every answer a one-line arithmetic expression.
- For repository-context cases, expected must depend on an API, type, or constant in context_files. Include at least two context files.
- JSON strings must escape newlines correctly.
"""


def validate_batch(raw: object, language: str, count: int, repository_context: bool) -> list[BenchmarkCase]:
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError("response must contain a cases array")
    normalized = []
    for item in raw["cases"]:
        if not isinstance(item, dict):
            raise ValueError("every case must be an object")
        item = dict(item)
        item["language"] = language
        item["repository_context"] = repository_context
        check = dict(item.get("check", {}))
        check["container_image"] = IMAGES[language]
        item["check"] = check
        normalized.append(item)
    cases = [BenchmarkCase.model_validate(item) for item in normalized]
    if len(cases) != count:
        raise ValueError(f"expected {count} cases, received {len(cases)}")
    with tempfile.TemporaryDirectory(prefix="tabcomplete-benchmark-validate-") as directory:
        root = Path(directory)
        for index, case in enumerate(cases):
            result = evaluate_prediction(
                case,
                Prediction(case_id=case.id, completion=case.expected),
                work_root=root / str(index),
                execution_backend="none",
            )
            if result.parse.status != "pass":
                raise ValueError(f"gold completion does not parse: {case.id}")
    return cases


def generate_batch(
    client: httpx.Client,
    key: str,
    budget: Budget,
    language: str,
    start: int,
    count: int,
    repository_context: bool,
) -> list[BenchmarkCase]:
    last_error = "unknown validation failure"
    for _ in range(3):
        if not budget.can_spend(0.025, count):
            raise RuntimeError("benchmark authoring budget exhausted before API request")
        response = client.post(
            URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You author executable code benchmarks and output strict JSON."},
                    {"role": "user", "content": prompt(language, start, count, repository_context)},
                ],
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "temperature": 0.2,
                "max_tokens": 16_000,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek API returned HTTP {response.status_code}")
        body = response.json()
        usage = body.get("usage", {})
        cost = estimate_cost_usd(
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            price_in_per_1k=0.0003,
            price_out_per_1k=0.0012,
        )
        budget.record(cost, count)
        try:
            content = body["choices"][0]["message"]["content"]
            return validate_batch(json.loads(content), language, count, repository_context)
        except (KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)[:200]
    raise RuntimeError(f"DeepSeek produced three invalid {language} batches: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/code_completion_v1.jsonl"))
    parser.add_argument("--budget", type=Path, default=Path("data/generated/benchmark_budget.json"))
    parser.add_argument("--spend-cap", type=float, default=5.0)
    parser.add_argument("--repair-existing", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is absent")
    budget = Budget(
        path=str(args.budget),
        spend_cap=args.spend_cap,
        example_cap=2_000,
        candidate_cap=1,
    )
    cases: list[BenchmarkCase] = []
    with httpx.Client(timeout=180) as client:
        if args.repair_existing:
            raw_cases = [json.loads(line) for line in args.output.read_text().splitlines() if line]
            invalid = [index for index, case in enumerate(raw_cases) if not str(case.get("expected", "")).strip()]
            groups: dict[tuple[str, bool], list[int]] = {}
            for index in invalid:
                case = raw_cases[index]
                group_key = (str(case["language"]), bool(case.get("repository_context")))
                groups.setdefault(group_key, []).append(index)
            next_id = 500
            for (language, repository_context), indices in groups.items():
                replacements = generate_batch(
                    client,
                    key,
                    budget,
                    language,
                    next_id,
                    len(indices),
                    repository_context,
                )
                next_id += len(indices)
                for index, replacement in zip(indices, replacements, strict=True):
                    raw_cases[index] = replacement.model_dump(mode="json")
                print(f"repaired {len(indices)} {language} cases")
            cases = [BenchmarkCase.model_validate(case) for case in raw_cases]
        else:
            for language in LANGUAGES:
                for start in (0, 10):
                    batch = generate_batch(client, key, budget, language, start, 10, False)
                    cases.extend(batch)
                    print(f"validated {language} single-file {start + 1}-{start + 10}")
            for batch_index in range(2):
                for offset in range(10):
                    language = LANGUAGES[(batch_index * 10 + offset) % len(LANGUAGES)]
                    cases.extend(generate_batch(client, key, budget, language, 100 + batch_index * 10 + offset, 1, True))
                    print(f"validated {language} repository-context {batch_index * 10 + offset + 1}/20")
    ids = [case.id for case in cases]
    if len(cases) != 200 or len(set(ids)) != 200:
        raise RuntimeError("final suite must contain exactly 200 unique cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(mode="json"), sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"wrote {len(cases)} validated cases; estimated spend=${budget.state.spent_usd:.4f}")


if __name__ == "__main__":
    main()
