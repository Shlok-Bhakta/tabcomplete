"""Summarize dependency use separately from basic task and generation failures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def rate(rows: list[dict], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in args.scores:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_model[row["model_label"]].append(row)
    summary = {}
    for model, rows in sorted(by_model.items()):
        conditions = sorted({row["condition"] for row in rows})
        condition_summary = {}
        for condition in conditions:
            selected = [row for row in rows if row["condition"] == condition]
            condition_summary[condition] = {
                "cases": len(selected),
                "correct_preference_rate": rate(selected, "correct_preferred"),
                "mean_nll_margin": sum(
                    row["distractor_score"]["target_nll_mean"]
                    - row["correct"]["target_nll_mean"]
                    for row in selected
                )
                / len(selected),
                "generation_exact_rate": rate(selected, "generation_exact"),
                "generation_truncation_rate": rate(selected, "generation_truncated"),
            }
        groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            key = (row["family"], row["requested_context_tokens"], row["seed"])
            groups[key][row["condition"]] = row
        nearby_solved = []
        distant_given_near = []
        unresolved = 0
        counterfactual_pairs = []
        for group in groups.values():
            short = group.get("short_control")
            near = group.get("long_near")
            far = group.get("long_far")
            counterfactual = group.get("counterfactual")
            if short and near:
                competence = short["correct_preferred"] and near["correct_preferred"]
                nearby_solved.append(competence)
                if competence and far:
                    distant_given_near.append(far["correct_preferred"])
                elif not competence:
                    unresolved += 1
            if far and counterfactual:
                counterfactual_pairs.append(
                    far["correct_preferred"] and counterfactual["correct_preferred"]
                )
        summary[model] = {
            "cases": len(rows),
            "by_condition": condition_summary,
            "nearby_competence_rate": (
                sum(nearby_solved) / len(nearby_solved) if nearby_solved else None
            ),
            "distant_preference_conditioned_on_nearby_success": (
                sum(distant_given_near) / len(distant_given_near)
                if distant_given_near
                else None
            ),
            "unresolved_task_competence_groups": unresolved,
            "counterfactual_preference_change_rate": (
                sum(counterfactual_pairs) / len(counterfactual_pairs)
                if counterfactual_pairs
                else None
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
