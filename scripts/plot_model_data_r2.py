"""Plot observed scientific results only; never substitute prepared runs for results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/model_data_r2"
ARTIFACTS = ROOT / "artifacts/research/model_data_r2"
PLOTS = REPORT / "plots"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / (name + ".png"), dpi=150)
    fig.savefig(PLOTS / (name + ".svg"))
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    causal = {}
    for path in sorted((REPORT / "baseline_evaluations").glob("*/results.jsonl")):
        records = rows(path)
        if len(records) != 200 or len({r["case_id"] for r in records}) != 200:
            continue
        causal[path.parent.name] = records
    if causal:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = list(causal)
        counts = [sum(row["test"]["status"] == "pass" for row in causal[k]) for k in labels]
        ax.bar(labels, counts, color="#5474aa")
        ax.set(
            ylabel="Strict passes / 200", title="Corrected causal suite · raw prompt · 96-token cap"
        )
        ax.bar_label(ax.containers[0])
        finish(fig, "causal-functional")
        languages = sorted({r["language"] for records in causal.values() for r in records})
        matrix = [
            [
                sum(
                    r["language"] == language and r["test"]["status"] == "pass"
                    for r in causal[model]
                )
                for language in languages
            ]
            for model in labels
        ]
        fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.6)))
        view = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set(
            xticks=range(len(languages)),
            xticklabels=languages,
            yticks=range(len(labels)),
            yticklabels=labels,
            title="Per-language strict pass counts",
        )
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                ax.text(x, y, str(value), ha="center", va="center")
        fig.colorbar(view, ax=ax, label="cases")
        finish(fig, "causal-by-language")
    line = {}
    for path in sorted((REPORT / "baseline_evaluations").glob("*/line/results.jsonl")):
        records = rows(path)
        if len(records) == 180 and len({r["case_id"] for r in records}) == 180:
            line[path.parent.parent.name] = records
    if line:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            list(line),
            [sum(r["exact"] for r in records) for records in line.values()],
            color="#588564",
        )
        ax.set(ylabel="Exact continuations / 180", title="causal_line_v1 · first-newline/EOS stop")
        ax.bar_label(ax.containers[0])
        finish(fig, "line-exact")
    comparisons = []
    for path in sorted((REPORT / "paired_comparisons").glob("*-vs-p12.json")):
        record = json.loads(path.read_text())
        comparisons.append((record["second"], len(record["wins"]), len(record["losses"])))
    if comparisons:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(comparisons))
        ax.bar(x - 0.2, [r[1] for r in comparisons], 0.4, label="new passes", color="#588564")
        ax.bar(x + 0.2, [-r[2] for r in comparisons], 0.4, label="lost passes", color="#ac5858")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(xticks=x, xticklabels=[r[0] for r in comparisons], ylabel="Paired cases vs P12")
        ax.legend()
        finish(fig, "paired-functional")
    measurements = []
    for path in sorted((ARTIFACTS / "local-inference").glob("*/measurements.jsonl")):
        if (path.parent / "exclusion.json").exists():
            continue
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        grouped = defaultdict(list)
        for row in rows(path):
            grouped[row["bucket"]].append(row)
        for bucket, records in grouped.items():
            ids = {(r["case_id"], r["repetition"]) for r in records}
            if len(records) != 40 or len(ids) != 40:
                continue
            observed = [
                r["returned_line_seconds"]
                for r in records
                if r["returned_line_seconds"] is not None
            ]
            measurements.append(
                {
                    "model": metadata["model_alias"],
                    "host": metadata["host"],
                    "precision": metadata["precision"],
                    "threads": metadata["threads"],
                    "bucket": bucket,
                    "n": len(records),
                    "prompts": len({r["case_id"] for r in records}),
                    "total_median_seconds": float(np.median([r["total_seconds"] for r in records])),
                    "total_p95_seconds": float(
                        np.quantile([r["total_seconds"] for r in records], 0.95)
                    ),
                    "returned_line_median_seconds": float(np.median(observed))
                    if observed
                    else None,
                    "returned_line_observed_n": len(observed),
                    "peak_resident_bytes": max(r["peak_resident_bytes"] for r in records),
                    "source": str(path.relative_to(ROOT)),
                    "warm_context": False,
                }
            )
    (REPORT / "local_inference").mkdir(parents=True, exist_ok=True)
    (REPORT / "local_inference/summary.json").write_text(json.dumps(measurements, indent=2) + "\n")
    if measurements:
        for host in {r["host"] for r in measurements}:
            records = [r for r in measurements if r["host"] == host]
            fig, ax = plt.subplots(figsize=(8, 4))
            for model in sorted({r["model"] for r in records}):
                subset = sorted(
                    [r for r in records if r["model"] == model], key=lambda r: r["bucket"]
                )
                ax.plot(
                    [r["bucket"] for r in subset],
                    [r["total_median_seconds"] for r in subset],
                    "o-",
                    label=model,
                )
            ax.set(
                xlabel="Approximate context bucket (own token counts saved)",
                ylabel="Median request seconds",
                title=f"{host} CPU · Q4_K_M · 4 threads · uncached prompt, 32-token cap",
            )
            ax.legend()
            finish(fig, "local-latency-" + host)
            for key, axis_label, divisor, plot_name in (
                ("total_median_seconds", "Median 2k request seconds", 1, "quality-latency"),
                ("peak_resident_bytes", "Peak resident GiB", 2**30, "quality-memory"),
            ):
                matched = [
                    (r, causal.get(r["model"] + "-q4")) for r in records if r["bucket"] == 2048
                ]
                matched = [(r, quality) for r, quality in matched if quality is not None]
                if not matched:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                for record, quality in matched:
                    value = sum(r["test"]["status"] == "pass" for r in quality)
                    ax.scatter(record[key] / divisor, value)
                    ax.annotate(
                        record["model"],
                        (record[key] / divisor, value),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )
                ax.set(
                    xlabel=axis_label,
                    ylabel="Q4 strict causal passes / 200",
                    title=host + " CPU · Q4_K_M",
                )
                finish(fig, plot_name + "-" + host)
    quota = []
    for path in sorted((REPORT / "quota").glob("20*.json")):
        record = json.loads(path.read_text())
        if record.get("used_account_hours") is not None:
            quota.append(record)
    if quota:
        from datetime import datetime

        started = datetime.fromisoformat(quota[0]["observed_at"])
        initial = quota[0]["used_account_hours"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            [
                (datetime.fromisoformat(r["observed_at"]) - started).total_seconds() / 3600
                for r in quota
            ],
            [r["used_account_hours"] - initial for r in quota],
            "o-",
        )
        ax.set(
            xlabel="Elapsed campaign hours",
            ylabel="Observed account quota-hours consumed",
            title="Authenticated Kaggle meter · setup and failures included",
        )
        finish(fig, "quota-consumption")


if __name__ == "__main__":
    main()
