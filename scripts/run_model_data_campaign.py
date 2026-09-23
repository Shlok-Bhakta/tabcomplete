"""Bounded R2 stages, immutable preregistration, and explicit Kaggle accounting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/model_data_r2"
ARTIFACTS = ROOT / "artifacts/research/model_data_r2"


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(args, timeout=60):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"command {args[0]} failed with exit {result.returncode}")
    return result.stdout


def observe_quota():
    try:
        rows = json.loads(command(["kaggle", "quota", "--format", "json"]))
        gpu = next(r for r in rows if r["resource"] == "GPU")
        quota = {
            "used_account_hours": float(gpu["used"].rstrip("h")),
            "remaining_account_hours": float(gpu["remaining"].rstrip("h")),
            "renewal": gpu["refreshAt"],
            "units": "account quota-hours",
        }
    except Exception:
        quota = {"used_account_hours": None, "remaining_account_hours": None, "renewal": None}
    listed = csv.DictReader(
        io.StringIO(command(["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"]))
    )
    statuses = []
    for row in listed:
        try:
            status = command(["kaggle", "kernels", "status", row["ref"]]).strip()
        except RuntimeError:
            status = "unknown: status request failed"
        statuses.append({"ref": row["ref"], "status": status, "last_run": row.get("lastRunTime")})
    quota.update(
        observed_at=now(),
        source="authenticated kaggle CLI quota and kernel status",
        jobs=statuses,
        active_jobs=[r for r in statuses if any(s in r["status"] for s in ("RUNNING", "QUEUED"))],
    )
    write_json(REPORT / "quota" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + ".json"), quota)
    return quota


def freeze(config_path):
    config = yaml.safe_load(config_path.read_text())
    target = REPORT / "preregistered_plan.json"
    if target.exists():
        plan = json.loads(target.read_text())
        if plan["configuration_sha256"] != digest(config_path):
            raise ValueError("Frozen plan differs: register a documented new plan revision")
        return plan
    quota = observe_quota()
    models = {}
    for alias, spec in config["models"].items():
        if "hub" not in spec:
            continue
        info = httpx.get(
            f"https://huggingface.co/api/models/{spec['hub']}/revision/{spec['revision']}",
            timeout=30,
        )
        info.raise_for_status()
        data = info.json()
        cfg = httpx.get(
            f"https://huggingface.co/{spec['hub']}/resolve/{spec['revision']}/config.json",
            follow_redirects=True,
            timeout=30,
        )
        cfg.raise_for_status()
        fields = cfg.json()
        models[alias] = {
            "id": spec["hub"],
            "revision": data["sha"],
            "license": data.get("cardData", {}).get("license"),
            "hub_total_parameters_including_unused_components": data.get("safetensors", {}).get(
                "total"
            ),
            "text_vocabulary_size": fields.get("text_config", fields).get("vocab_size"),
            "config": fields,
        }
    parent = Path(
        os.environ.get(
            "TABCOMPLETE_R2_PARENT",
            str(
                ROOT.parent / "tabcomplete/outputs/kaggle/code_cpt_train_v3/code_cpt_run/main/final"
            ),
        )
    )
    assert digest(parent / "model.safetensors") == config["models"]["q35-p12"]["model_sha256"]
    assert digest(parent / "tokenizer.json") == config["models"]["q35-p12"]["tokenizer_sha256"]
    assert (
        digest(ROOT / config["evaluation"]["causal"]["path"])
        == config["evaluation"]["causal"]["sha256"]
    )
    plan = {
        "schema_version": 1,
        "frozen_at": now(),
        "configuration_sha256": digest(config_path),
        "configuration": config,
        "resolved_models": models,
        "quota_at_freeze": quota,
        "integration": {
            "research": "9ab49df73a2cff9c8ba340f93f96f4ec1dc58271",
            "observability": "5513296edfb77d4ab2763a293e93969e6a769557",
        },
        "outcomes_inspected_before_freeze": "historical R1 only, no R2 generation or training",
        "context_runtime_inventory": "No separate context-runtime worktree or verified tiling",
        "thinkpad": "authorized existing alias timed out, not measured",
        "parent_bytes_verified": True,
        "sealed_test_opened": False,
    }
    write_json(target, plan)
    write_json(
        REPORT / "environment/local.json",
        {
            "observed_at": now(),
            "platform": platform.platform(),
            "cpu": command(["lscpu"]),
            "memory": command(["free", "-b"]),
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "python": sys.version,
            "local_cuda_assumed": False,
            "inference_threads": 4,
            "llama_cpp_revision": config["deployment"]["revision"],
        },
    )
    return plan


def submit_baseline(plan):
    state_path = REPORT / "baseline_evaluations/job.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != digest(REPORT / "preregistered_plan.json"):
            raise ValueError("job belongs to another plan")
        print(
            json.dumps(
                {
                    "existing_job": state["reference"],
                    "status": command(["kaggle", "kernels", "status", state["reference"]]).strip(),
                }
            )
        )
        return
    quota = observe_quota()
    if quota["active_jobs"]:
        raise RuntimeError("another notebook allocation is active")
    seconds = 7200
    if (
        quota["remaining_account_hours"] is not None
        and seconds / 3600 > quota["remaining_account_hours"]
    ):
        raise RuntimeError("insufficient verified quota")
    revision = command(["git", "rev-parse", "HEAD"]).strip()
    if command(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise RuntimeError("commit frozen code before submitting")
    folder = ARTIFACTS / "submissions/baseline"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(
        (ROOT / "kaggle/model_data_r2/run_baseline.py")
        .read_text()
        .replace("__CHECKOUT_COMMIT__", revision)
    )
    reference = "shlokbhakta/tabcomplete-model-data-r2-baseline"
    write_json(
        folder / "kernel-metadata.json",
        {
            "id": reference,
            "title": "tabcomplete-model-data-r2-baseline",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["shlokbhakta/tabcomplete-code-cpt-parents-r1"],
            "kernel_sources": [],
            "competition_sources": [],
        },
    )
    state = {
        "reference": reference,
        "submitted_at": now(),
        "deadline_seconds": seconds,
        "reserved_wall_hours": 2,
        "plan_sha256": digest(REPORT / "preregistered_plan.json"),
        "git_sha": revision,
        "quota_before": quota,
        "status": "submission_pending",
    }
    write_json(state_path, state)
    result = command(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(folder),
            "--timeout",
            str(seconds),
            "--accelerator",
            "NvidiaTeslaT4",
        ],
        timeout=180,
    )
    state.update(status="submitted", response=result.strip())
    write_json(state_path, state)
    print(json.dumps({"reference": reference, "status": "submitted", "deadline_seconds": seconds}))


def amend(config_path, reason):
    if not reason:
        raise ValueError("a recorded reason is required")
    path = REPORT / "preregistered_plan.json"
    old = json.loads(path.read_text())
    config = yaml.safe_load(config_path.read_text())
    if config["plan_revision"] != old["configuration"]["plan_revision"] + 1:
        raise ValueError("increment the plan revision exactly once")
    old_hash = digest(path)
    archive = REPORT / "plan_revisions" / (old_hash + ".json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, archive)
    revised = {
        **old,
        "frozen_at": now(),
        "configuration": config,
        "configuration_sha256": digest(config_path),
        "amendment": {
            "previous_plan_sha256": old_hash,
            "reason": reason,
            "new_line_or_training_outcomes_inspected": False,
        },
    }
    write_json(path, revised)
    print(json.dumps({"revision": config["plan_revision"], "sha256": digest(path)}))


def package_corpora():
    """Publish only the private, verified experiment inputs, never credentials or test arrays."""
    from tinycomplete.code_cpt.prepare import CORE_LANGUAGES, corpus_fingerprint
    from tinycomplete.eval.code_benchmark import _parse

    source = ARTIFACTS / "frozen-corpora"
    completion = json.loads((source / "completion.json").read_text())
    if digest(source / "causal_line_v1.jsonl") != completion["line_sha256"]:
        raise ValueError("line suite fingerprint mismatch")
    rows = [json.loads(line) for line in (source / "causal_line_v1.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 180
    for language in CORE_LANGUAGES:
        assert sum(r["language"] == language for r in rows) == 20
    for row in rows:
        original = row["source_before"] + row["reference"] + row["source_after"]
        assert hashlib.sha256(original.encode()).hexdigest() == row["source_sha256"]
        assert _parse(original, row["language"]).status == "pass"
    target = ARTIFACTS / "private-dataset"
    if (target / "dataset-metadata.json").exists():
        raise ValueError("dataset already packaged; validate its manifest instead of overwriting")
    target.mkdir(parents=True, exist_ok=True)
    for arm in ("R2_STANDARD", "R2_FILTERED"):
        origin = source / arm
        metadata = json.loads((origin / "corpus_metadata.json").read_text())
        assert (
            corpus_fingerprint(
                [
                    origin / "train_blocks.npy",
                    origin / "train_languages.npy",
                    *list((origin / "manifests").glob("*.jsonl")),
                ]
            )
            == (metadata["corpus_fingerprint"])
        )
        destination = target / arm
        destination.mkdir()
        for name in ("train_blocks.npy", "train_languages.npy", "corpus_metadata.json"):
            shutil.copy2(origin / name, destination / name)
        for name in ("micro", "manifests"):
            shutil.copytree(origin / name, destination / name)
    shutil.copytree(ARTIFACTS / "stage1-corpus/code_cpt_corpus/micro", target / "historical/micro")
    shutil.copy2(
        ARTIFACTS / "stage1-corpus/code_cpt_corpus/corpus_metadata.json",
        target / "historical/corpus_metadata.json",
    )
    shutil.copy2(source / "causal_line_v1.jsonl", target / "causal_line_v1.jsonl")
    shutil.copytree(source / "audit", REPORT / "data_audit", dirs_exist_ok=True)
    manifest = {
        "schema_version": 1,
        "frozen_at": now(),
        "plan_sha256": digest(REPORT / "preregistered_plan.json"),
        "preparation_code_sha256": digest(ROOT / "src/tinycomplete/code_cpt/model_data_r2.py"),
        "completion": completion,
        "files": [
            {"path": str(p.relative_to(target)), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(target.rglob("*"))
            if p.is_file()
        ],
    }
    write_json(target / "input-manifest.json", manifest)
    write_json(REPORT / "data_audit/frozen-input-manifest.json", manifest)
    write_json(
        target / "dataset-metadata.json",
        {
            "id": "shlokbhakta/tabcomplete-model-data-r2-inputs",
            "title": "TabComplete R2 private inputs",
            "licenses": [{"name": "other"}],
            "description": (
                "Private controlled causal research inputs. Public Stack-dedup source retains its "
                "original per-file licenses; source identities are preserved in manifests. "
                "No sealed test token arrays, credentials, or personal editor data included."
            ),
        },
    )
    print(
        json.dumps(
            {
                "packaged": True,
                "line_controls": len(rows),
                "manifest_sha256": digest(target / "input-manifest.json"),
            }
        )
    )


def collect_jobs():
    for path in REPORT.rglob("job.json"):
        state = json.loads(path.read_text())
        status = command(["kaggle", "kernels", "status", state["reference"]]).strip()
        state["last_status"] = status
        if not any(s in status for s in ("RUNNING", "QUEUED")):
            if "terminal_observed_at" not in state:
                state["terminal_observed_at"] = now()
                state["observed_wall_upper_bound_seconds"] = (
                    datetime.fromisoformat(state["terminal_observed_at"])
                    - datetime.fromisoformat(state["submitted_at"])
                ).total_seconds()
            state["terminal"] = True
        write_json(path, state)
        print(
            json.dumps(
                {
                    "reference": state["reference"],
                    "status": status,
                    "wall_upper_bound_seconds": state.get("observed_wall_upper_bound_seconds"),
                }
            )
        )


def submit_pilot(arm, attempt=1):
    suffix = "" if attempt == 1 else f"-attempt-{attempt}"
    reference = "shlokbhakta/tabcomplete-model-data-r2-" + arm.lower().replace("r2_", "") + suffix
    state_path = (
        REPORT / "pilots" / arm / ("" if attempt == 1 else f"attempt-{attempt}") / "job.json"
    )
    if state_path.exists():
        print(command(["kaggle", "kernels", "status", reference]).strip())
        return
    quota = observe_quota()
    if quota["active_jobs"]:
        raise RuntimeError("another notebook allocation is active")
    seconds = 9900
    if quota["remaining_account_hours"] is None:
        raise RuntimeError("quota must be refreshed before a subsequent session")
    if quota["remaining_account_hours"] < seconds / 3600:
        raise RuntimeError("insufficient observed account quota")
    first = json.loads((REPORT / "baseline_evaluations/job.json").read_text())
    if quota["renewal"] != first["quota_before"]["renewal"]:
        raise RuntimeError("automatic consumption of a renewed allocation is forbidden")
    prior = [json.loads(p.read_text()) for p in REPORT.rglob("job.json")]
    if (
        sum(p.get("observed_wall_upper_bound_seconds", p["deadline_seconds"]) for p in prior)
        + seconds
        > 36000
    ):
        raise RuntimeError("aggregate conservative session reservations exceed ten hours")
    revision = command(["git", "rev-parse", "HEAD"]).strip()
    if command(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise RuntimeError("commit scientific code before submission")
    folder = ARTIFACTS / "submissions" / (arm + suffix)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(
        (ROOT / "kaggle/model_data_r2/run_pilot.py")
        .read_text()
        .replace("__CHECKOUT_COMMIT__", revision)
        .replace("__ARM__", arm)
        .replace("__REUSE_RESTART__", "none")
        .replace(
            "__BASELINE_SHA__",
            digest(ARTIFACTS / "standard-attempt1/model_data_r2_pilot/parent-fresh.json"),
        )
    )
    write_json(
        folder / "kernel-metadata.json",
        {
            "id": reference,
            "title": reference.split("/")[1],
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [
                "shlokbhakta/tabcomplete-code-cpt-parents-r1",
                "shlokbhakta/tabcomplete-model-data-r2-inputs",
            ],
            "kernel_sources": ["shlokbhakta/tabcomplete-model-data-r2-standard"],
            "competition_sources": [],
        },
    )
    state = {
        "reference": reference,
        "submitted_at": now(),
        "deadline_seconds": seconds,
        "plan_sha256": digest(REPORT / "preregistered_plan.json"),
        "git_sha": revision,
        "inputs_sha256": digest(ARTIFACTS / "private-dataset/input-manifest.json"),
        "quota_before": quota,
        "status": "submission_pending",
    }
    write_json(state_path, state)
    response = command(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(folder),
            "--timeout",
            str(seconds),
            "--accelerator",
            "NvidiaTeslaT4",
        ],
        timeout=180,
    )
    state.update(status="submitted", response=response.strip())
    write_json(state_path, state)
    print(json.dumps({"reference": reference, "deadline_seconds": seconds}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/model_data_r2.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["freeze", "baseline", "amend", "package", "pilot", "collect"],
        default="baseline",
    )
    parser.add_argument("--reason")
    parser.add_argument("--arm", choices=["R2_STANDARD", "R2_FILTERED"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "amend":
        if not args.execute:
            raise ValueError("amend requires explicit --execute")
        amend(args.config, args.reason)
        return
    plan = freeze(args.config)
    if args.execute and args.stage == "collect":
        collect_jobs()
        return
    if args.execute and args.stage == "package":
        package_corpora()
        return
    if args.execute and args.stage == "pilot":
        if not args.arm:
            raise ValueError("pilot requires --arm")
        submit_pilot(args.arm, args.attempt)
        return
    if args.execute and args.stage == "baseline":
        submit_baseline(plan)
    else:
        print(
            json.dumps(
                {
                    "plan_sha256": digest(REPORT / "preregistered_plan.json"),
                    "frozen": True,
                    "execution_requested": args.execute,
                }
            )
        )


if __name__ == "__main__":
    main()
