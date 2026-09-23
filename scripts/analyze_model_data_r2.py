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


def compare_lines(baseline, report):
    """Compare only complete, identity-matched versions of the line diagnostic."""
    available = {}
    for path in sorted(baseline.glob("*/line/summary.json")):
        metadata = json.loads(path.read_text())
        records = rows(path.with_name("results.jsonl"))
        if len(records) != 180 or metadata["total"] != 180:
            raise ValueError("incomplete line evaluation")
        available[path.parent.parent.name] = (records, metadata)
    for alias, (records, metadata) in available.items():
        controls = ["q35-p12"]
        if alias.endswith("-q4"):
            controls.append(alias.removesuffix("-q4"))
        if alias == "q25-coder-q4":
            controls.append("q35-p12-q4")
        for control in dict.fromkeys(controls):
            if control not in available or control == alias:
                continue
            before, control_metadata = available[control]
            if metadata["suite_sha256"] != control_metadata["suite_sha256"]:
                raise ValueError("cannot compare different line fixtures")
            comparison = paired(before, records, line=True)
            comparison.update(
                first=control,
                second=alias,
                suite_sha256=metadata["suite_sha256"],
                protocol="causal_line_v1",
                precision_and_hardware_first=control_metadata["metadata"],
                precision_and_hardware_second=metadata["metadata"],
                caveat="Exact continuation is diagnostic, not functional correctness",
            )
            save(report / "paired_comparisons" / f"line-{alias}-vs-{control}.json", comparison)


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


def compare_development(root, evaluation):
    from tinycomplete.code_cpt.eval import paired_repository_bootstrap

    parent_path = (
        root
        / "artifacts/research/model_data_r2/standard-attempt1/model_data_r2_pilot"
        / "parent-fresh_repositories.json"
    )
    if not parent_path.exists():
        return
    parent = json.loads(parent_path.read_text())
    parent_keys = {(row["repository"], row["language"]): row["tokens"] for row in parent}
    for alias in ("R2_STANDARD", "R2_FILTERED", "q35-base"):
        path = evaluation / alias / "fresh_repositories.json"
        if not path.exists():
            continue
        candidate = json.loads(path.read_text())
        keys = {(row["repository"], row["language"]): row["tokens"] for row in candidate}
        if keys != parent_keys:
            raise ValueError("development source boundaries/counts differ: " + alias)
        comparison = paired_repository_bootstrap(parent, candidate, samples=2000, seed=271828)
        comparison.update(
            first="q35-p12",
            second=alias,
            parent_repository_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            candidate_repository_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            tokenizer_comparison="same pinned Qwen3.5 tokenizer only",
            scored_boundaries="Provenance-attributed target tokens; no missing repos dropped",
        )
        save(
            root
            / "reports/research/model_data_r2/paired_comparisons"
            / (alias + "-development-vs-p12.json"),
            comparison,
        )


def audit_line_sample(root):
    report = root / "reports/research/model_data_r2"
    fixture = root / "artifacts/research/model_data_r2/frozen-corpora/causal_line_v1-r3.jsonl"
    cases = {r["id"]: r for r in map(json.loads, fixture.read_text().splitlines())}
    selected = sorted(
        cases, key=lambda key: hashlib.sha256(("928173:" + key).encode()).hexdigest()
    )[:20]
    save(
        report / "failure_audit/line-sample-definition.json",
        {
            "suite_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "ordering": "First 20 IDs by SHA256('928173:' + case_id), independent of outcomes",
            "case_ids": selected,
            "same_sample_for_every_model": True,
            "passes_retained": True,
            "semantic_oracle": "None: syntax validity does not establish semantic correctness",
        },
    )
    for path in (report / "baseline_evaluations").glob("*/line/results.jsonl"):
        predictions = rows(path)
        if set(predictions) != set(cases):
            raise ValueError("line audit fixture identities differ: " + path.parent.parent.name)
        audit = []
        for identifier in selected:
            row = predictions[identifier]
            if row["exact"]:
                label, evidence = "pass", "Exact reference continuation; no quality failure."
            elif row["syntax"] == "fail":
                label, evidence = (
                    "syntax error",
                    "Insertion fails syntax; reference restoration passes.",
                )
            elif not row["returned_text"]:
                label, evidence = (
                    "premature stopping",
                    "Empty returned line for a nonempty reference.",
                )
            elif row["fallback_cap"]:
                label, evidence = "length truncation", "Recorded 96-token fallback limit reached."
            else:
                label, evidence = (
                    "unresolved",
                    (
                        "Exact mismatch with valid syntax. No executable semantic oracle; "
                        "a different valid continuation is not necessarily wrong logic."
                    ),
                )
            case = cases[identifier]
            audit.append(
                {
                    "case_id": identifier,
                    "primary": label,
                    "evidence": evidence,
                    "left_context_tail": case["source_before"][-240:],
                    "reference": row["reference"],
                    "raw_response": row["raw_response"],
                    "returned_text": row["returned_text"],
                    "syntax": row["syntax"],
                    "prefix_characters": row["longest_exact_character_prefix"],
                    "finish_reason": row["finish_reason"],
                    "official_score_repaired": False,
                }
            )
        save(
            report / "failure_audit" / (path.parent.parent.name + "-line-fixed-sample.json"), audit
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-predictions", type=Path)
    parser.add_argument("--reuse-d12", action="store_true")
    parser.add_argument("--evaluation-directory", type=Path)
    parser.add_argument("--audit-line-sample", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = root / "reports/research/model_data_r2"
    baseline = report / "baseline_evaluations"
    compare_lines(baseline, report)
    if args.audit_line_sample:
        audit_line_sample(root)
    compare_development(
        root,
        args.evaluation_directory
        or (root / "artifacts/research/model_data_r2/evaluation/model_data_r2_evaluation"),
    )
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
