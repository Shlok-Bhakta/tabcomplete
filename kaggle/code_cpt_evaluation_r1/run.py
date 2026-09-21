"""Repository-paired development evaluation and causal predictions for CPT pilots."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/cpt-recipe-r1"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_evaluation_r1")
SESSION_LIMIT_SECONDS = 6 * 60 * 60
FINALIZATION_RESERVE_SECONDS = 30 * 60
PARENTS = {
    "P5": "d499c3fe2d720246c710c5c9e20653a4599aa25d77ebef871cb28431ccc888e8",
    "P12": "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43",
}
CAUSAL_SUITE_SHA256 = "ed28739b302e4c0d3f9f45e859e7ccd68a1e8594a62eb7b13ee8425b92a439a4"


def run(command: list[str], *, name: str, environment: dict[str, str]) -> float:
    started = time.time()
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env={**os.environ, **environment},
    )
    elapsed = time.time() - started
    log = OUTPUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit code {process.returncode}")
    return elapsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern}, found {matches}")
    return matches[0]


def evaluate(
    model_path: Path,
    corpus: Path,
    output: Path,
    expected_sha256: str,
    environment: dict[str, str],
    *,
    split: str,
    repository_only: bool,
) -> float:
    command = [
        sys.executable,
        "-m",
        "tinycomplete.code_cpt.train",
        "checkpoint-eval",
        "--checkpoint",
        str(model_path),
        "--corpus-dir",
        str(corpus),
        "--output",
        str(output),
        "--expected-sha256",
        expected_sha256,
        "--split",
        split,
    ]
    if repository_only:
        command.append("--repository-only")
    return run(command, name=f"evaluate-{output.stem}", environment=environment)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    environment = {"PYTHONPATH": str(CHECKOUT / "src"), "TOKENIZERS_PARALLELISM": "false"}
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==5.5.0",
            "accelerate==1.13.0",
        ],
        name="pip",
        environment={},
    )
    run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        name="git",
        environment={},
    )
    sys.path.insert(0, str(CHECKOUT / "src"))
    progress_path = find_one("**/campaign_progress.json")
    campaign_root = progress_path.parent
    campaign = load_json(progress_path)
    if campaign["status"] != "complete":
        raise RuntimeError("campaign source has no completion marker")
    corpus_candidates = []
    for path in Path("/kaggle/input").glob("**/corpus_metadata.json"):
        metadata = load_json(path)
        if (
            metadata.get("campaign") == "code_cpt_research_r1"
            and metadata.get("corpus_fingerprint") == campaign["corpus_fingerprint"]
        ):
            corpus_candidates.append(path.parent)
    if len(corpus_candidates) != 1:
        raise RuntimeError(f"expected one frozen corpus, found {corpus_candidates}")
    corpus = corpus_candidates[0]

    models: dict[str, dict] = {}
    for parent, digest in PARENTS.items():
        model_path = find_one(f"**/{parent}/training_metadata.json").parent
        models[parent] = {"path": model_path, "sha256": digest, "parent": None}
    for arm in campaign["completed_arms"]:
        model_path = campaign_root / "arms" / arm["name"] / "final"
        digest = sha256_file(model_path / "model.safetensors")
        if digest != arm["model_sha256"]:
            raise RuntimeError(f"campaign model hash differs for {arm['name']}")
        models[arm["name"]] = {
            "path": model_path,
            "sha256": digest,
            "parent": arm["parent"],
        }

    record = {
        "schema_version": 1,
        "status": "running",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True
        ).strip(),
        "models": {name: {k: str(v) for k, v in value.items()} for name, value in models.items()},
        "development_evaluations": [],
        "causal_predictions": [],
    }
    progress_output = OUTPUT / "evaluation_progress.json"

    for label, specification in models.items():
        destination = OUTPUT / "development" / f"{label}.json"
        elapsed = evaluate(
            specification["path"],
            corpus,
            destination,
            specification["sha256"],
            environment,
            split="micro",
            repository_only=True,
        )
        record["development_evaluations"].append(
            {"model": label, "elapsed_seconds": elapsed, "output": str(destination)}
        )
        progress_output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        gc.collect()

    from tinycomplete.code_cpt.eval import paired_repository_bootstrap

    repository_rows = {
        label: load_json(OUTPUT / "development" / f"{label}_repositories.json")
        for label in models
    }
    comparisons = {}
    aggregate_nll = {
        "P5": load_json(campaign_root / "parents" / "P5" / "fresh_development.json")[
            "metrics"
        ]["overall_code"]["nll"],
        "P12": load_json(campaign_root / "parents" / "P12" / "fresh_development.json")[
            "metrics"
        ]["overall_code"]["nll"],
    }
    for arm in campaign["completed_arms"]:
        label = arm["name"]
        parent = arm["parent"]
        aggregate_nll[label] = load_json(
            campaign_root / "arms" / label / "final" / "micro_eval.json"
        )["overall_code"]["nll"]
        comparisons[f"{label}-minus-{parent}"] = paired_repository_bootstrap(
            repository_rows[parent], repository_rows[label]
        )

    eligible = []
    for arm in campaign["completed_arms"]:
        label = arm["name"]
        parent = arm["parent"]
        paired = comparisons[f"{label}-minus-{parent}"]
        if (
            aggregate_nll[label] < aggregate_nll[parent]
            and paired["balanced_language_95ci"][1] < 0
        ):
            eligible.append(label)
    candidate = min(eligible, key=aggregate_nll.get) if eligible else None
    unique = candidate
    if candidate is not None:
        for challenger in eligible:
            if challenger == candidate:
                continue
            comparison = paired_repository_bootstrap(
                repository_rows[challenger], repository_rows[candidate]
            )
            comparisons[f"{candidate}-minus-{challenger}"] = comparison
            if comparison["balanced_language_95ci"][1] >= 0:
                unique = None

    selection = {
        "development_metric": "balanced-language NLL with paired repository bootstrap",
        "aggregate_nll": aggregate_nll,
        "comparisons": comparisons,
        "eligible_improvements": eligible,
        "selected_candidate": unique,
        "result": "unique" if unique else ("tie" if eligible else "no-improvement"),
        "untouched_test_opened": unique is not None,
    }
    (OUTPUT / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if unique is not None:
        for label in [models[unique]["parent"], unique]:
            destination = OUTPUT / "test" / f"{label}.json"
            evaluate(
                models[label]["path"],
                corpus,
                destination,
                models[label]["sha256"],
                environment,
                split="test",
                repository_only=False,
            )

    suite = CHECKOUT / "data" / "benchmarks" / "code_completion_v2.jsonl"
    if sha256_file(suite) != CAUSAL_SUITE_SHA256:
        raise RuntimeError("corrected causal suite hash differs from the frozen protocol")
    for arm in campaign["completed_arms"]:
        if time.time() - started + FINALIZATION_RESERVE_SECONDS >= SESSION_LIMIT_SECONDS:
            record["causal_predictions"].append(
                {"model": arm["name"], "status": "skipped-session-reserve"}
            )
            continue
        label = arm["name"]
        destination = OUTPUT / "causal_predictions" / f"{label}.jsonl"
        elapsed = run(
            [
                sys.executable,
                str(CHECKOUT / "scripts" / "generate_code_predictions.py"),
                "--suite",
                str(suite),
                "--model-path",
                str(models[label]["path"]),
                "--device",
                "cuda",
                "--max-new-tokens",
                "96",
                "--output",
                str(destination),
            ],
            name=f"causal-predictions-{label}",
            environment=environment,
        )
        record["causal_predictions"].append(
            {"model": label, "status": "complete", "elapsed_seconds": elapsed}
        )
        progress_output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    record["status"] = "complete"
    record["elapsed_seconds"] = time.time() - started
    progress_output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("campaign evaluation complete")


if __name__ == "__main__":
    main()
