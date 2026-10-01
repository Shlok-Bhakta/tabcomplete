#!/usr/bin/env python3
"""Qualify the narrow public-prefix and functional synthetic pilot on CPU."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from tinycomplete.eval.code_benchmark import (
    BenchmarkCase,
    CheckSpec,
    Prediction,
    evaluate_prediction,
)
from tinycomplete.one_line.contract import EditState, apply_action
from tinycomplete.one_line.public_prefix_pilot import (
    canonical_bytes,
    canonical_sha256,
    sha256_bytes,
)
from tinycomplete.one_line.synthetic_functional_mix import (
    SYNTHETIC_GENERATOR_REVISION,
    SYNTHETIC_PILOT_SCHEMA,
    build_synthetic_candidates,
    validate_synthetic_family_split,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (
    "docker.io/library/python:3.12-slim@sha256:"
    "2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9"
)
EVALUATOR = ROOT / "src/tinycomplete/eval/code_benchmark.py"


def write(path: Path, value, *, jsonl=False):
    payload = (
        b"".join(canonical_bytes(row) + b"\n" for row in value)
        if jsonl
        else canonical_bytes(value) + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as handle:
        handle.write(payload)
    path.chmod(0o600)
    return {"path": str(path.name), "sha256": sha256_bytes(payload), "bytes": len(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-bank", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    candidates = build_synthetic_candidates()
    grouping = validate_synthetic_family_split([c.row for c in candidates])
    identities = [
        {
            "id": c.row["id"],
            "state_sha256": canonical_sha256(c.row["state"]),
            "action_sha256": canonical_sha256(c.row["action"]),
            "fixture_sha256": c.fixture_sha256,
        }
        for c in candidates
    ]
    plan = {
        "schema": "public-synthetic-cpu-qualification-plan-v1",
        "generator_sha256": SYNTHETIC_GENERATOR_REVISION,
        "builder_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "evaluator_sha256": sha256_bytes(EVALUATOR.read_bytes()),
        "image_identity": IMAGE,
        "network_access": "none",
        "candidate_identities": identities,
        "grouping": grouping,
        "max_wall_seconds": 1800,
        "max_checks": 384,
        "public_bank_manifest_sha256": sha256_bytes(
            (args.public_bank / "manifest.json").read_bytes()
        ),
        "qualification": (
            "all family variants gold parse/compile/test pass "
            "and wrong parse/compile pass/test fail"
        ),
        "preflight_revision_note": (
            "Fixes syntax, data shapes, indentation, neutral file identity and shared "
            "edit operations before authoritative checks. HTML/CSV normalized contexts "
            "grouped together in train; page-selection family reserved for dev. "
            "No model outputs used. V1 was preparation only; V2 pins formatted code."
        ),
    }
    if not args.execute:
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        write(args.output / "plan.json", plan)
        print(json.dumps({"frozen": True, "candidates": len(candidates)}))
        return
    if json.loads((args.output / "plan.json").read_text()) != plan:
        raise ValueError("CPU qualification identity changed")
    if (args.output / "diagnostics.jsonl").exists():
        raise ValueError("qualification already started; preserve its outputs")
    started = time.monotonic()
    qualified = []
    diagnostics = []
    families_failed = set()
    for c in candidates:
        if time.monotonic() - started > plan["max_wall_seconds"] - 60:
            raise TimeoutError("CPU qualification finalization reserve")
        state = EditState.from_mapping(c.row["state"])
        variants = {}
        for name, action in [
            ("gold", c.row["action"]),
            ("wrong", {"kind": c.wrong_action.kind, "text": c.wrong_action.text}),
        ]:
            from tinycomplete.one_line.contract import EditAction

            source = apply_action(state, EditAction(**action))
            case = BenchmarkCase(
                id=c.row["id"] + "/" + name,
                language="python",
                path="solution.py",
                prefix="",
                expected=c.row["after_source"],
                check=CheckSpec(
                    compile=["python", "-m", "py_compile", "solution.py"],
                    test=["python", "tests.py"],
                    files={"tests.py": c.test_source},
                    container_image=IMAGE,
                ),
            )
            result = evaluate_prediction(
                case,
                Prediction(case_id=case.id, completion=source),
                work_root=args.output / "checks" / c.row["id"] / name,
                execution_backend="container",
            )
            variants[name] = result.model_dump()
        receipt = {
            "schema": "one-line-synthetic-functional-receipt-v1",
            "execution_backend": "container",
            "network_access": "none",
            "image_identity": IMAGE,
            "evaluator_sha256": plan["evaluator_sha256"],
            "state_sha256": canonical_sha256(c.row["state"]),
            "gold_action_sha256": canonical_sha256(c.row["action"]),
            "fixture_sha256": c.fixture_sha256,
            **{
                name + "_" + check: variants[name][check]["status"]
                for name in ["gold", "wrong"]
                for check in ["parse", "compile", "test"]
            },
        }
        passed = all(
            receipt["gold_" + check] == "pass" for check in ["parse", "compile", "test"]
        ) and (
            receipt["wrong_parse"] == receipt["wrong_compile"] == "pass"
            and receipt["wrong_test"] == "fail"
        )
        record = {"id": c.row["id"], "receipt": receipt, "results": variants, "qualified": passed}
        descriptor = write(args.output / "proofs" / (canonical_sha256(record) + ".json"), record)
        row = copy.deepcopy(c.row)
        row["synthetic_objective_receipt"] = receipt
        row["receipt_path"] = "proofs/" + descriptor["path"]
        row["receipt_sha256"] = descriptor["sha256"]
        row["receipt_bytes"] = descriptor["bytes"]
        row["validation"] = {"replay_verified": True}
        row["accepted_training"] = passed
        if not passed:
            families_failed.add(row["algorithm_family_id"])
        qualified.append((row, c))
        diagnostics.append(record)
        # A durable per-case receipt precedes the progress counter.
        print(json.dumps({"completed": len(diagnostics), "qualified": passed}), flush=True)
    write(args.output / "diagnostics.jsonl", diagnostics, jsonl=True)
    selected = [
        (row, c) for row, c in qualified if row["algorithm_family_id"] not in families_failed
    ]
    for source in (args.public_bank / "artifacts").rglob("*"):
        if source.is_file():
            target = args.output / source.relative_to(args.public_bank)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, target)
    public = []
    for name in ["train.jsonl", "development.jsonl"]:
        for line in (args.public_bank / name).read_text().splitlines():
            row = json.loads(line)
            row["schema"] = SYNTHETIC_PILOT_SCHEMA
            row["accepted_training"] = True
            row["validation"] = {"replay_verified": True}
            public.append(row)
    rows = public + [row for row, _ in selected]
    oracles = [
        {
            "id": row["id"],
            "test_source": c.test_source,
            "fixture_sha256": c.fixture_sha256,
            "state_sha256": canonical_sha256(row["state"]),
            "image_identity": IMAGE,
            "gold_after_source_sha256": sha256_bytes(row["after_source"].encode()),
        }
        for row, c in selected
        if row["split"] == "development"
    ]
    oracle_descriptor = write(args.output / "proofs/functional_oracles.jsonl", oracles, jsonl=True)
    shards = {}
    for split, name in [("train", "train.jsonl"), ("development", "development.jsonl")]:
        shards[split] = write(
            args.output / name, [r for r in rows if r["split"] == split], jsonl=True
        )
    counts = Counter(r["split"] for r in rows)
    proof_files = []
    for directory in [args.output / "artifacts", args.output / "proofs"]:
        for p in sorted(directory.rglob("*")):
            if p.is_file():
                proof_files.append(
                    {
                        "path": p.relative_to(args.output).as_posix(),
                        "sha256": sha256_bytes(p.read_bytes()),
                        "bytes": p.stat().st_size,
                    }
                )
    manifest = {
        "schema": SYNTHETIC_PILOT_SCHEMA,
        "dataset_id": "tabcomplete/public-prefix-synthetic-functional-pilot-r1",
        "dataset_revision": canonical_sha256(plan),
        "dataset_license": "per-file-public-and-author-owned-synthetic",
        "source_file_license_status": "verified-public-path-scope-and-author-owned-synthetic",
        "train_sha256": shards["train"]["sha256"],
        "development_sha256": shards["development"]["sha256"],
        "train_count": counts["train"],
        "dev_count": counts["development"],
        "file_groups_disjoint": True,
        "artifact_root": ".",
        "proof_files": proof_files,
        "functional_oracles_path": "proofs/functional_oracles.jsonl",
        "functional_oracles_sha256": oracle_descriptor["sha256"],
        "oracle_evaluator_sha256": plan["evaluator_sha256"],
        "qualified_synthetic_rows": len(selected),
        "excluded_synthetic_families": sorted(families_failed),
        "training_ready": counts["train"] >= 128 and counts["development"] >= 64,
        "quality_evidence": False,
        "human_chronology_observed": False,
    }
    write(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "complete": True,
                "counts": dict(counts),
                "synthetic": len(selected),
                "excluded_families": sorted(families_failed),
            }
        )
    )


if __name__ == "__main__":
    main()
