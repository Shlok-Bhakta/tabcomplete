"""Paired scientific-file analysis. Missing experiments remain missing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


def rows(path):
    return {r["case_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def paired(left, right, *, line=False):
    if set(left) != set(right):
        raise ValueError("paired case identities differ")
    ids = sorted(left)
    success = (lambda r: r["exact"]) if line else (lambda r: r["test"]["status"] == "pass")
    wins = [k for k in ids if success(right[k]) and not success(left[k])]
    losses = [k for k in ids if success(left[k]) and not success(right[k])]
    shared = [k for k in ids if success(left[k]) and success(right[k])]
    discordant = len(wins) + len(losses)
    exact_p = (
        min(
            1,
            2
            * sum(math.comb(discordant, k) for k in range(min(len(wins), len(losses)) + 1))
            / 2**discordant,
        )
        if discordant
        else 1.0
    )
    delta = np.array([int(success(right[k])) - int(success(left[k])) for k in ids])
    rng = np.random.default_rng(928173)
    samples = [float(delta[rng.integers(0, len(ids), len(ids))].mean()) for _ in range(2000)]
    return {
        "cases": len(ids),
        "first_passes": len(shared) + len(losses),
        "second_passes": len(shared) + len(wins),
        "wins": wins,
        "losses": losses,
        "shared_passes": shared,
        "exact_paired_binomial_two_sided_p": exact_p,
        "difference_second_minus_first": float(delta.mean()),
        "paired_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "bootstrap_samples": 2000,
        "bootstrap_seed": 928173,
    }


def historical_failure_audit(predictions, root):
    results = root / "reports/code_cpt/research_r1/causal_functional"
    before, after = rows(results / "P12/results.jsonl"), rows(results / "D12/results.jsonl")
    predictions_by_model = {m: rows(predictions / (m + ".jsonl")) for m in ("P12", "D12")}
    explanation = {
        "python/stable_0": (
            "overgeneration",
            ["wrong symbol/API/type"],
            "The extra generated main() requests interactive input; execution raises EOFError. "
            "The number-classification prefix itself handles the three signs.",
        ),
        "typescript/0": (
            "overgeneration",
            ["length truncation", "syntax error"],
            "After closing classifyNumber, D12 adds classifyString and reaches the cap inside "
            "an else-if expression. The compiler reports TS1109/TS1128.",
        ),
        "javascript/501": (
            "overgeneration",
            ["length truncation", "syntax error"],
            "After implementing the Stack methods, D12 adds example calls and ends at "
            "stack.push(7. Node reports a missing closing parenthesis.",
        ),
        "rust/2": (
            "overgeneration",
            ["length truncation", "syntax error"],
            "D12 completes gcd, adds another gcd function, then starts a third before the cap. "
            "The unchanged fixture suffix yields an unexpected closing delimiter.",
        ),
    }
    report = []
    for case_id in paired(before, after)["losses"]:
        primary, secondary, evidence = explanation[case_id]
        report.append(
            {
                "case_id": case_id,
                "comparison": "historical D12 versus P12",
                "primary": primary,
                "secondary": secondary,
                "evidence": evidence,
                "checks": {k: after[case_id][k] for k in ("parse", "compile", "test")},
                "raw_predictions": {m: predictions_by_model[m][case_id] for m in ("P12", "D12")},
                "official_score_repaired": False,
                "prefix_analysis": "diagnostic only, no rescoring",
            }
        )
    save(root / "reports/research/model_data_r2/failure_audit/historical_d12_losses.json", report)


def reuse_verified_d12(root):
    from tinycomplete.eval.code_generation import file_sha256, model_weight_fingerprint

    historical = root.parent / "tabcomplete-cpt-recipe-r1"
    directory = (
        historical
        / "artifacts/code_cpt/research_r1/evaluation/code_cpt_evaluation_r1/causal_predictions"
    )
    checkpoint = (
        historical
        / "artifacts/code_cpt/research_r1/campaign_full/code_cpt_campaign_r1/arms/D12/final"
    )
    metadata = json.loads((directory / "D12.jsonl.metadata.json").read_text())
    assert metadata["model_revision"] == model_weight_fingerprint(checkpoint)
    assert metadata["tokenizer_sha256"] == file_sha256(checkpoint / "tokenizer.json")
    assert metadata["suite_sha256"] == file_sha256(
        root / "data/benchmarks/code_completion_v2.jsonl"
    )
    assert metadata["protocol"] == "causal-context-v1" and metadata["provider"] == "transformers"
    assert metadata["decoding"] == {"do_sample": False, "temperature": 0}
    assert metadata["max_new_tokens"] == 96 and metadata["workers"] == 1
    old = rows(directory / "P12.jsonl")
    new = rows(
        root / "artifacts/research/model_data_r2/baseline/model_data_r2_baseline/q35-p12.jsonl"
    )
    assert set(old) == set(new) and len(new) == 200
    assert all(old[key]["completion"] == row["completion"] for key, row in new.items())
    source = root / "reports/code_cpt/research_r1/causal_functional/D12"
    assert set(rows(source / "results.jsonl")) == set(rows(directory / "D12.jsonl")) == set(new)
    destination = root / "reports/research/model_data_r2/baseline_evaluations/q35-d12"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "results.jsonl"):
        shutil.copy2(source / name, destination / name)
    save(
        destination / "historical-reuse.json",
        {
            "historical": True,
            "new_predictions_generated": False,
            "weight_sha256": file_sha256(checkpoint / "model.safetensors"),
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "original_metadata": metadata,
            "prediction_sha256": file_sha256(directory / "D12.jsonl"),
            "result_sha256": file_sha256(source / "results.jsonl"),
            "control_reproduction": "All 200 R2 P12 raw completions exactly reproduce R1 P12",
            "backend": "Transformers 5.5, Torch 2.10, text causal model, FP16 on T4",
            "request_telemetry_fabricated": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-predictions", type=Path)
    parser.add_argument("--reuse-d12", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = root / "reports/research/model_data_r2"
    baseline = report / "baseline_evaluations"
    if args.reuse_d12:
        reuse_verified_d12(root)
    if args.historical_predictions:
        historical_failure_audit(args.historical_predictions, root)
    parent_path = baseline / "q35-p12/results.jsonl"
    if parent_path.exists():
        parent = rows(parent_path)
        for candidate in sorted(baseline.iterdir()):
            path = candidate / "results.jsonl"
            if not path.exists() or candidate.name == "q35-p12":
                continue
            comparison = paired(parent, rows(path))
            comparison.update(
                first="q35-p12",
                second=candidate.name,
                first_results_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                second_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                protocol="causal-context-v1",
                output_ceiling_tokens=96,
            )
            save(report / "paired_comparisons" / (candidate.name + "-vs-p12.json"), comparison)
            if candidate.name.endswith("-q4"):
                own = baseline / candidate.name.removesuffix("-q4") / "results.jsonl"
                if own.exists():
                    quantization = paired(rows(own), rows(path))
                    quantization.update(
                        first=candidate.name.removesuffix("-q4"),
                        second=candidate.name,
                        caveat="Own checkpoint comparison changes runtime and precision together",
                    )
                    save(
                        report
                        / "paired_comparisons"
                        / (candidate.name + "-vs-own-unquantized.json"),
                        quantization,
                    )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in comparison.items()
                        if k not in ("wins", "losses", "shared_passes")
                    }
                )
            )


if __name__ == "__main__":
    main()
