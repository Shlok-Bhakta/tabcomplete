"""Automatic benchmark plots for the T4x2 throughput lab.

Reads ``throughput_benchmarks.json`` (a list of worker result dicts) and
renders the five required figures. Failed candidates are excluded from
throughput figures but listed in the summary table. No timing category is
invented: the step-time breakdown uses only worker-measured forward /
backward / optimizer / data-wait seconds, with the remainder labelled
synchronization+misc.
"""

from __future__ import annotations

import json
from pathlib import Path

BASELINE_TOKS = 810.0
ROUND2_CONTROL_TOKS = 1630.0


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("status") == "pass"
            and (r.get("steady_state_tokens_per_second") or 0) > 0]


def _control(results: list[dict]) -> float:
    """Round-2 control: prefer an explicit round-2 winner repro candidate."""
    for marker in ("r2_R0_winner_repro", "stageC_winner", "stageB_winner"):
        for row in results:
            if row.get("name") == marker and row.get("steady_state_tokens_per_second"):
                return float(row["steady_state_tokens_per_second"])
    return ROUND2_CONTROL_TOKS


def render_all(results_path: Path, plots_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(results_path)
    good = sorted(stable(results),
                  key=lambda r: r["steady_state_tokens_per_second"], reverse=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    names = [r["name"] for r in good]
    steady = [float(r["steady_state_tokens_per_second"]) for r in good]

    # Plot 1 — throughput by configuration (steady-state tok/s).
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["tab:red" if "baseline" in n else "tab:blue" for n in names]
    ax.bar(names, steady, color=colors)
    ax.axhline(BASELINE_TOKS, color="red", linestyle="--", label="baseline 810 tok/s")
    ax.set_ylabel("steady-state tok/s")
    ax.set_title("Throughput by configuration (steady-state)")
    ax.tick_params(axis="x", rotation=30)
    for n, v in zip(names, steady):
        ax.text(n, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "plot1_throughput_by_config.png", dpi=150)
    plt.close(fig)
    made.append("plot1_throughput_by_config.png")

    # Plot 2 — speedup vs baseline.
    fig, ax = plt.subplots(figsize=(12, 4))
    speedups = [v / BASELINE_TOKS for v in steady]
    ax.bar(names, speedups, color=colors)
    ax.axhline(1.0, color="red", linestyle="--", label="baseline 1.00x")
    ax.set_ylabel("speedup vs baseline")
    ax.set_title("Speedup vs baseline (candidate / 810)")
    ax.tick_params(axis="x", rotation=30)
    for n, v in zip(names, speedups):
        ax.text(n, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "plot2_speedup_vs_baseline.png", dpi=150)
    plt.close(fig)
    made.append("plot2_speedup_vs_baseline.png")

    # Plot 3 — throughput vs VRAM.
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in good:
        vram = r.get("peak_allocated_vram_gib_max")
        if vram is None and r.get("peak_allocated_vram_gib"):
            vals = r["peak_allocated_vram_gib"]
            vram = max(vals) if vals else None
        if vram is None:
            continue
        ax.scatter(vram, float(r["steady_state_tokens_per_second"]), s=80)
        ax.annotate(r["name"], (vram, float(r["steady_state_tokens_per_second"])),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("peak allocated VRAM / GPU (GiB)")
    ax.set_ylabel("steady-state tok/s")
    ax.set_title("Throughput vs VRAM (memory/compute tradeoff)")
    fig.tight_layout()
    fig.savefig(plots_dir / "plot3_throughput_vs_vram.png", dpi=150)
    plt.close(fig)
    made.append("plot3_throughput_vs_vram.png")

    # Plot 4 — step-time breakdown (measured regions only).
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = ["forward", "backward", "optimizer", "data wait", "sync+misc"]
    bottoms = [0.0] * len(good)
    series = []
    for r in good:
        total = float(r.get("training_seconds") or 0)
        fwd = float(r.get("forward_seconds") or 0)
        bwd = float(r.get("backward_seconds") or 0)
        opt = float(r.get("optimizer_seconds") or 0)
        dw = float(r.get("data_wait_seconds") or 0)
        misc = max(0.0, total - (fwd + bwd + opt + dw))
        series.append([fwd, bwd, opt, dw, misc])
    x = range(len(good))
    for idx, label in enumerate(labels):
        vals = [s[idx] for s in series]
        ax.bar([good[i]["name"] for i in x], vals, bottom=list(bottoms), label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("seconds (training window)")
    ax.set_title("Step-time breakdown (measured regions; remainder = sync+misc)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "plot4_step_time_breakdown.png", dpi=150)
    plt.close(fig)
    made.append("plot4_step_time_breakdown.png")

    # Plot 5 — optimization contribution (stacking chain, best-effort order).
    chain_order = ["exp00_baseline_repro", "exp01_fla_conv", "exp02_fla_fusedce",
                   "exp03_fla_fusedce_nockpt", "exp04_fla_fusedce_nockpt_sgo"]
    by_name = {r["name"]: r for r in good}
    chain = [by_name[n] for n in chain_order if n in by_name]
    if len(chain) >= 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([r["name"] for r in chain],
                [float(r["steady_state_tokens_per_second"]) for r in chain],
                marker="o")
        ax.set_ylabel("steady-state tok/s")
        ax.set_title("Optimization contribution (baseline -> FLA -> fused CE -> "
                     "no-ckpt -> SHARD_GRAD_OP)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(plots_dir / "plot5_optimization_stacking.png", dpi=150)
        plt.close(fig)
        made.append("plot5_optimization_stacking.png")

    summary = {
        "results": len(results), "stable": len(good),
        "failed": [r.get("name") for r in results if r.get("status") != "pass"],
        "ranking": [{k: r.get(k) for k in (
            "name", "steady_state_tokens_per_second", "speedup_vs_baseline",
            "peak_allocated_vram_gib_max", "loss_start", "loss_end",
            "gradient_norm_mean", "loss_scale_overflows")} for r in good],
        "plots": made,
    }
    (plots_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")

    # Round-2 plots: leaderboard vs control, tok/s vs VRAM with Round-2
    # fields, compile-mode comparison, and verification waterfall.
    control = _control(results)
    names2 = [r["name"] for r in good]
    steady2 = [float(r["steady_state_tokens_per_second"]) for r in good]

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["tab:red" if v >= control else "tab:blue" for v in steady2]
    ax.bar(names2, steady2, color=colors)
    ax.axhline(control, color="red", linestyle="--",
               label=f"round-2 control {control:.0f} tok/s")
    ax.set_ylabel("steady-state tok/s")
    ax.set_title("Round-2 leaderboard vs control")
    ax.tick_params(axis="x", rotation=30)
    for n, v in zip(names2, steady2):
        ax.text(n, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "r2_plot1_leaderboard.png", dpi=150)
    plt.close(fig)
    made.append("r2_plot1_leaderboard.png")

    compile_rows = [r for r in good if r.get("torch_compile")]
    if len(compile_rows) >= 2:
        fig, ax = plt.subplots(figsize=(11, 4))
        labels = [f"{r['name']} ({r.get('compile_placement')}/{r.get('compile_mode')})"
                  for r in compile_rows]
        vals = [float(r["steady_state_tokens_per_second"]) for r in compile_rows]
        ax.bar(labels, vals, color="tab:green")
        ax.axhline(control, color="red", linestyle="--")
        ax.set_ylabel("steady-state tok/s")
        ax.set_title("Compile placement/mode comparison")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(plots_dir / "r2_plot2_compile_modes.png", dpi=150)
        plt.close(fig)
        made.append("r2_plot2_compile_modes.png")

    waterfall = [r for n in ("r2_R0_winner_repro", "r2_C1_precompile",
                             "r2_F_best", "r2_G_best", "r2_L5_liger",
                             "r2_combined")
                 for r in good if r["name"] == n]
    if len(waterfall) >= 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([r["name"] for r in waterfall],
                [float(r["steady_state_tokens_per_second"]) for r in waterfall],
                marker="o")
        ax.axhline(control, color="red", linestyle="--")
        ax.set_ylabel("steady-state tok/s")
        ax.set_title("Round-2 verification waterfall (independently verified wins only)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(plots_dir / "r2_plot3_waterfall.png", dpi=150)
        plt.close(fig)
        made.append("r2_plot3_waterfall.png")

    (plots_dir / "summary.json").write_text(
        json.dumps({**summary, "plots": made, "round2_control": control},
                   indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return made


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render_all(args.results, args.plots_dir))
