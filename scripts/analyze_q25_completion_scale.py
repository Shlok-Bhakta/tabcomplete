"""Paired analysis for the frozen completion repetition/data experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_q25_fim import paired_development


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def bucket(value: int, limits: list[int]) -> str:
    for limit in limits:
        if value <= limit:
            return f"up_to_{limit}"
    raise ValueError("observation exceeds the frozen bin limits")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "exact_and_terminated": sum(row["exact_and_terminated"] for row in rows),
        "exact": sum(row["exact"] for row in rows),
        "terminated": sum(row["terminated"] for row in rows),
        "cap_hits": sum(row["reached_token_ceiling"] for row in rows),
        "valid_output_tokens": sum(row["valid_output_tokens"] for row in rows),
        "output_token_counts": dict(Counter(row["output_tokens"] for row in rows)),
    }


def analyze(
    corpus: list[dict[str, Any]],
    repeat: list[dict[str, Any]],
    scaled: list[dict[str, Any]],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    sources = {f"fim-development-{row['id']}": row for row in corpus}
    if len(sources) != len(corpus):
        raise ValueError("duplicate development source identity")
    for predictions in (repeat, scaled):
        if len(predictions) != len(sources) or {row["case_id"] for row in predictions} != set(
            sources
        ):
            raise ValueError("predictions do not cover the complete frozen development slice")
        for row in predictions:
            source = sources[row["case_id"]]
            if row["context_sha256"] != source["prompt_sha256"] or (
                row["repository"] != source["repository_identity_sha256"]
            ):
                raise ValueError("prediction does not belong to its source state")

    # Preserve original request identity checks above, then cluster uncertainty
    # by the whole transitive alias component when the frozen corpus supplies it.
    group_field = "repository_group_sha256"
    grouped = any(group_field in source for source in sources.values())
    if grouped:
        for source in sources.values():
            group = source.get(group_field)
            if not isinstance(group, str) or len(group) != 64 or any(
                character not in "0123456789abcdef" for character in group
            ):
                raise ValueError("development repository component identity is missing or invalid")
        repeat = [
            {**row, "repository": sources[row["case_id"]][group_field]} for row in repeat
        ]
        scaled = [
            {**row, "repository": sources[row["case_id"]][group_field]} for row in scaled
        ]

    strata: dict[str, dict[str, str]] = {}
    for case_id, source in sources.items():
        response = source["input_ids"][source["prompt_tokens"] : -1]
        target = tokenizer.decode(
            response, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if hashlib.sha256(target.encode()).hexdigest() != source["target_sha256"]:
            raise ValueError("target token IDs do not reconstruct the frozen source target")
        punctuation = all(character.isspace() or character in "[]{}(),;:" for character in target)
        strata[case_id] = {
            "language": source["language"],
            "mode": source["mode"],
            "response_tokens_including_eos": bucket(len(response) + 1, [8, 16, 32, 64, 96]),
            "input_tokens": bucket(len(source["input_ids"]), [128, 256, 512, 768, 1024]),
            "target_content": "punctuation_only" if punctuation else "content_bearing",
        }
    comparisons: dict[str, dict[str, Any]] = {}
    for dimension in next(iter(strata.values())):
        values = sorted({state[dimension] for state in strata.values()})
        comparisons[dimension] = {}
        for value in values:
            first = [row for row in repeat if strata[row["case_id"]][dimension] == value]
            second = [row for row in scaled if strata[row["case_id"]][dimension] == value]
            comparisons[dimension][value] = {
                "repeat": summarize(first),
                "scaled": summarize(second),
                "paired": paired_development(first, second, metric="exact_and_terminated"),
            }
    return {
        "schema": "q25-completion-scale-paired-analysis-v1",
        "primary": paired_development(repeat, scaled, metric="exact_and_terminated"),
        "repeat": summarize(repeat),
        "scaled": summarize(scaled),
        "strata": comparisons,
        "bootstrap_group_identity": (
            "transitive_repository_alias_component" if grouped else "historical_primary_repository"
        ),
        "general_human_edit_quality_established": False,
        "automatic_personalization_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--repeat-results", type=Path, required=True)
    parser.add_argument("--scaled-results", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    paths = (args.development, args.repeat_results, args.scaled_results)
    result = analyze(*(read_rows(path) for path in paths), tokenizer=tokenizer)
    result["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(
            ("development", "repeat_results", "scaled_results"), paths, strict=True
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["primary"]))


if __name__ == "__main__":
    main()
