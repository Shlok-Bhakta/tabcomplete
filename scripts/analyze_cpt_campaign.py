"""Build machine-readable campaign analysis and the six declared plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LANGUAGES = [
    "python",
    "typescript",
    "javascript",
    "java",
    "cpp",
    "rust",
    "go",
    "c",
    "csharp",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def functional_passes(path: Path) -> dict[str, bool]:
    return {row["case_id"]: row["test"]["status"] == "pass" for row in load_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--functional-root", type=Path)
    parser.add_argument("--long-context-summary", type=Path)
    parser.add_argument("--quota-observations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign = load_json(args.campaign_root / "campaign_progress.json")
    parents = {
        label: load_json(args.campaign_root / "parents" / label / "fresh_development.json")[
            "metrics"
        ]
        for label in ("P5", "P12")
    }
    arms = {row["name"]: row for row in campaign["completed_arms"]}
    final = {
        label: load_json(args.campaign_root / "arms" / label / "final" / "micro_eval.json")
        for label in arms
    }
    development_curves = {}
    for label, arm in arms.items():
        rows = [{"additional_tokens": 0, "nll": parents[arm["parent"]]["overall_code"]["nll"]}]
        for path in sorted((args.campaign_root / "arms" / label / "evaluations").glob("*.json")):
            value = load_json(path)
            rows.append(
                {
                    "additional_tokens": value["actual_training_tokens"],
                    "nll": value["metrics"]["overall_code"]["nll"],
                }
            )
        rows.append(
            {
                "additional_tokens": arm["additional_input_tokens"],
                "nll": final[label]["overall_code"]["nll"],
            }
        )
        development_curves[label] = rows

    fig, axis = plt.subplots(figsize=(8, 5))
    for label, rows in development_curves.items():
        axis.plot(
            [row["additional_tokens"] / 1_000_000 for row in rows],
            [row["nll"] for row in rows],
            marker="o",
            label=f"{label} ({arms[label]['parent']})",
        )
    axis.set(xlabel="Additional campaign input tokens (millions)", ylabel="Fresh dev NLL")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "development_nll.png", dpi=160)
    plt.close(fig)

    per_language = {
        label: {
            language: final[label][language]["nll"] - parents[arm["parent"]][language]["nll"]
            for language in LANGUAGES
        }
        for label, arm in arms.items()
    }
    x = np.arange(len(LANGUAGES))
    width = 0.19
    fig, axis = plt.subplots(figsize=(12, 5))
    for index, label in enumerate(arms):
        axis.bar(
            x + (index - 1.5) * width,
            [per_language[label][language] for language in LANGUAGES],
            width,
            label=label,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, LANGUAGES, rotation=30, ha="right")
    axis.set_ylabel("NLL change from own parent (lower is better)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "per_language_changes.png", dpi=160)
    plt.close(fig)

    schedule = {}
    for parent in ("P5", "P12"):
        schedule[parent] = {}
        for kind in ("constant", "cosine"):
            label = next(
                label
                for label, arm in arms.items()
                if arm["parent"] == parent and arm["lr_schedule"] == kind
            )
            schedule[parent][kind] = final[label]["overall_code"]["nll"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, parent in zip(axes, ("P5", "P12"), strict=True):
        axis.bar(list(schedule[parent]), list(schedule[parent].values()))
        axis.axhline(parents[parent]["overall_code"]["nll"], color="black", linestyle="--")
        axis.set_title(parent)
        axis.set_ylabel("Fresh dev NLL")
    fig.tight_layout()
    fig.savefig(args.output_dir / "schedule_by_parent.png", dpi=160)
    plt.close(fig)

    functional = {}
    if args.functional_root:
        passes = {
            path.parent.name: functional_passes(path)
            for path in args.functional_root.glob("*/results.jsonl")
        }
        labels = [label for label in ("P5", "P12", *arms) if label in passes]
        counts = {label: sum(passes[label].values()) for label in labels}
        paired = {}
        for label, arm in arms.items():
            parent = arm["parent"]
            if label not in passes or parent not in passes:
                continue
            shared = sorted(set(passes[label]) & set(passes[parent]))
            paired[label] = {
                "wins": sum(passes[label][case] and not passes[parent][case] for case in shared),
                "losses": sum(not passes[label][case] and passes[parent][case] for case in shared),
            }
        functional = {"counts": counts, "paired_vs_parent": paired}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(list(counts), list(counts.values()))
        axes[0].set_ylabel("Functional passes / 200")
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(
            list(paired),
            [paired[label]["wins"] for label in paired],
            label="wins",
        )
        axes[1].bar(
            list(paired),
            [-paired[label]["losses"] for label in paired],
            label="losses",
        )
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "functional_counts_and_pairs.png", dpi=160)
        plt.close(fig)

    long_context = load_json(args.long_context_summary) if args.long_context_summary else {}
    if long_context:
        conditions = ["short_control", "long_near", "long_far", "absent", "counterfactual"]
        labels = list(long_context)
        x = np.arange(len(conditions))
        width = 0.8 / len(labels)
        fig, axis = plt.subplots(figsize=(10, 5))
        for index, label in enumerate(labels):
            axis.bar(
                x + (index - (len(labels) - 1) / 2) * width,
                [
                    long_context[label]["by_condition"].get(condition, {}).get(
                        "correct_preference_rate", 0
                    )
                    for condition in conditions
                ],
                width,
                label=label,
            )
        axis.set_xticks(x, conditions, rotation=25, ha="right")
        axis.set_ylabel("Correct-vs-distractor preference rate")
        axis.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "long_context_conditions.png", dpi=160)
        plt.close(fig)

    quota = load_json(args.quota_observations) if args.quota_observations else []
    if quota:
        baseline = quota[0]["used_gpu_hours"]
        consumed = [row["used_gpu_hours"] - baseline for row in quota]
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(range(len(quota)), consumed, marker="o", label="GPU-hours")
        axis.plot(
            range(len(quota)),
            [value / 2 for value in consumed],
            marker="x",
            label="T4x2 wall-hours",
        )
        axis.set(xlabel="Quota observation", ylabel="Campaign consumption")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "quota_and_wall_time.png", dpi=160)
        plt.close(fig)

    analysis = {
        "schema_version": 1,
        "development_curves": development_curves,
        "per_language_change_from_parent": per_language,
        "schedule_by_parent": schedule,
        "selection": load_json(args.selection) if args.selection else None,
        "functional": functional,
        "long_context": long_context,
        "quota_observations": quota,
        "original_lineage_tokens": {
            label: load_json(args.campaign_root / "arms" / label / "summary.json")[
                "parent_training_tokens"
            ]
            for label in arms
        },
        "additional_campaign_tokens": {
            label: arm["additional_input_tokens"] for label, arm in arms.items()
        },
    }
    (args.output_dir / "campaign_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
