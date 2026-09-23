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
import socket
import subprocess
import sys
import time
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


def published_scientific_revision():
    revision = command(["git", "rev-parse", "HEAD"]).strip()
    # Reports are updated by collection. Only scientific/deployment code must
    # remain identical to the published revision used inside the notebook.
    changed = command(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
            "kaggle",
            "configs",
            "pyproject.toml",
            "uv.lock",
            "AGENTS.md",
            "data/benchmarks",
            "reports/research/model_data_r2/preregistered_plan.json",
        ]
    )
    if changed.strip():
        raise RuntimeError("commit and push scientific code before submission")
    remote = command(["git", "ls-remote", "origin", "refs/heads/research/model-data-r2"])
    if remote.split()[0] != revision:
        raise RuntimeError("local scientific revision is not the published campaign branch tip")
    return revision


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


def amend(config_path, reason, outcomes_inspected):
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
            "new_line_or_training_outcomes_inspected": outcomes_inspected != "none",
            "outcomes_inspected": outcomes_inspected,
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


def package_baseline():
    source = ARTIFACTS / "standard-attempt1/model_data_r2_pilot/parent-fresh.json"
    report = json.loads(source.read_text())
    assert report["checkpoint_identity"]["weight_files"][0]["sha256"] == (
        "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43"
    )
    assert report["metrics"]["overall_code"]["tokens"] == 1031688
    destination = ARTIFACTS / "private-dataset"
    shutil.copy2(source, destination / "r2-parent-fresh.json")
    shutil.copy2(
        source.with_name("parent-fresh_repositories.json"),
        destination / "r2-parent-fresh_repositories.json",
    )
    addition = {
        "parent_report_sha256": digest(source),
        "frozen_corpus_manifest_sha256": digest(destination / "input-manifest.json"),
        "reason": "Transport previously verified parent metrics as explicit private dataset input",
        "training_arrays_changed": False,
    }
    write_json(REPORT / "data_audit/baseline-input-addition.json", addition)
    print(json.dumps(addition))


def submit_pilot(arm, attempt=1):
    suffix = "" if attempt == 1 else f"-attempt-{attempt}"
    reference = "shlokbhakta/tabcomplete-model-data-r2-" + arm.lower().replace("r2_", "") + suffix
    state_path = (
        REPORT / "pilots" / arm / ("" if attempt == 1 else f"attempt-{attempt}") / "job.json"
    )
    if state_path.exists():
        print(command(["kaggle", "kernels", "status", reference]).strip())
        return
    ledger_path = REPORT / "quota/training_token_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    reservation = 5_013_504 + (16 if arm == "R2_STANDARD" else 3) * 32768
    if sum(job["input_tokens"] for job in ledger["jobs"].values()) + reservation > 12_000_000:
        raise RuntimeError(
            "training token reservations exceed 12 million; reconcile actual evidence"
        )
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
    revision = published_scientific_revision()
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
            "kernel_sources": [],
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
    ledger["jobs"][reference] = {
        "input_tokens": reservation,
        "status": "reserved_upper_bound",
        "evidence": "Bounded worker absolute update limits, including smoke and restart work",
    }
    write_json(ledger_path, ledger)
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


def retrieve_pilot(arm, reference):
    """Verify bytes and complete restart state before declaring an arm finished."""
    status = command(["kaggle", "kernels", "status", reference])
    if "COMPLETE" not in status and "ERROR" not in status:
        raise RuntimeError("pilot is not terminal; leave its allocation undisturbed")
    destination = ARTIFACTS / "collected" / reference.split("/")[-1]
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < 18 * 2**30:
        raise RuntimeError("18 GiB free required before collecting complete pilot state")
    command(
        [
            "kaggle",
            "kernels",
            "output",
            reference,
            "-p",
            str(destination),
            "--file-pattern",
            "^model_data_r2_pilot/",
            "--page-size",
            "200",
            "-q",
        ],
        timeout=1800,
    )
    directory = destination / "model_data_r2_pilot"
    progress = json.loads((directory / "progress.json").read_text())
    if progress["status"] != "complete":
        raise RuntimeError("pilot artifacts retained but scientific completion gate did not pass")
    import runpy

    worker = runpy.run_path(str(ROOT / "kaggle/model_data_r2/run_pilot.py"), run_name="verify")
    summary = worker["verify_state"](directory / arm, 153, persist=False)
    assert summary["additional_input_tokens"] == 5_013_504
    actual_hash = digest(directory / arm / "final/model.safetensors")
    assert actual_hash == progress["final_model_sha256"]
    parent = Path(
        os.environ.get(
            "TABCOMPLETE_R2_PARENT",
            str(
                ROOT.parent / "tabcomplete/outputs/kaggle/code_cpt_train_v3/code_cpt_run/main/final"
            ),
        )
    )
    mtp_hash = digest(directory / arm / "final/mtp-original.safetensors")
    assert mtp_hash == digest(parent / "mtp-original.safetensors"), "original MTP sidecar changed"
    ledger_path = REPORT / "quota/training_token_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    consumed = sum(
        json.loads(path.read_text())["additional_input_tokens"]
        for path in directory.glob("*/summary.json")
    )
    assert consumed == progress["training_input_tokens"]
    assert consumed <= ledger["jobs"][reference]["input_tokens"]
    ledger["jobs"][reference] = {
        "input_tokens": consumed,
        "status": "verified",
        "evidence": "Summed per-process additional input tokens; verified complete checkpoints",
    }
    write_json(ledger_path, ledger)
    report = REPORT / "pilots" / arm
    for name in ("summary.json", "verified-checkpoint-manifest.json", "run_config.json"):
        write_json(report / name, json.loads((directory / arm / name).read_text()))
    if arm == "R2_STANDARD":
        verification = json.loads((directory / "restart-verification.json").read_text())
        assert verification["numerical_gate_passed"]
        write_json(report / "restart-verification.json", verification)
    write_json(
        report / "completion.json",
        {
            "reference": reference,
            "verified_at": now(),
            "model_sha256": actual_hash,
            "original_mtp_sha256": mtp_hash,
            "corpus_fingerprint": progress["corpus"]["corpus_fingerprint"],
            "additional_input_tokens_including_checks": consumed,
            "pilot_input_tokens": summary["additional_input_tokens"],
            "checkpoint_complete_and_hash_verified": True,
            "artifact_directory": str(directory.relative_to(ROOT)),
            "source_git_sha": progress["git_sha"],
        },
    )
    print(json.dumps({"arm": arm, "verified": True, "input_tokens": consumed}))


def submit_evaluation(*, context_only=False):
    # Use a job.json name so the common accounting collector cannot omit this session.
    path = REPORT / (
        "long_context/job.json" if context_only else "baseline_evaluations/additional/job.json"
    )
    if path.exists():
        state = json.loads(path.read_text())
        print(command(["kaggle", "kernels", "status", state["reference"]]).strip())
        return
    pilots = (
        []
        if context_only
        else [
            json.loads((REPORT / "pilots" / arm / "completion.json").read_text())
            for arm in ("R2_STANDARD", "R2_FILTERED")
        ]
    )
    assert all(p["checkpoint_complete_and_hash_verified"] for p in pilots)
    quota = observe_quota()
    seconds = 3600 if context_only else 9000
    if quota["active_jobs"] or quota["remaining_account_hours"] is None:
        raise RuntimeError("fresh quota and no other active allocation required")
    if quota["remaining_account_hours"] < seconds / 3600:
        raise RuntimeError("insufficient verified quota")
    first = json.loads((REPORT / "baseline_evaluations/job.json").read_text())
    if quota["renewal"] != first["quota_before"]["renewal"]:
        raise RuntimeError("renewed allocation consumption forbidden")
    prior = [json.loads(p.read_text()) for p in REPORT.rglob("job.json")]
    reserved = sum(p.get("observed_wall_upper_bound_seconds", p["deadline_seconds"]) for p in prior)
    if reserved + seconds > 36000:
        raise RuntimeError("ten-hour aggregate session budget exceeded")
    revision = published_scientific_revision()
    kind = "context" if context_only else "evaluation"
    line_fixture = json.loads((REPORT / "data_audit/line-fixture-correction.json").read_text())
    folder = ARTIFACTS / "submissions" / kind
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(
        (ROOT / "kaggle/model_data_r2" / ("run_" + kind + ".py"))
        .read_text()
        .replace("__CHECKOUT_COMMIT__", revision)
        .replace("__LINE_FIXTURE_FILENAME__", line_fixture["fixture_filename"])
        .replace("__LINE_FIXTURE_SHA__", line_fixture["corrected_sha256"])
        .replace(
            "__PILOT_EXPECTED__",
            json.dumps(
                {
                    arm: record["model_sha256"]
                    for arm, record in zip(("R2_STANDARD", "R2_FILTERED"), pilots, strict=False)
                }
            ),
        )
    )
    reference = "shlokbhakta/tabcomplete-model-data-r2-" + kind
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
            "kernel_sources": [p["reference"] for p in pilots]
            + ["shlokbhakta/tabcomplete-code-cpt-campaign-r1"]
            + (["shlokbhakta/tabcomplete-code-cpt-long-context-r1"] if context_only else []),
            "competition_sources": [],
        },
    )
    state = {
        "reference": reference,
        "submitted_at": now(),
        "deadline_seconds": seconds,
        "git_sha": revision,
        "plan_sha256": digest(REPORT / "preregistered_plan.json"),
        "quota_before": quota,
        "status": "submission_pending",
        "training_input_tokens": 0,
    }
    write_json(path, state)
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
    write_json(path, state)
    print(json.dumps({"reference": reference, "deadline_seconds": seconds}))


def run_local_grid():
    """One isolated CPU inference process at a time, resuming exact completed pairs."""
    runtime = ROOT.parent / "tabcomplete/outputs/tools/llama.cpp"
    prompts = ARTIFACTS / "local-prompts.jsonl"
    models = {
        "q35-p12": ("p12-text-Q4_K_M.gguf", "p12-q4-clean"),
        "q25-coder": ("q25-Q4_K_M.gguf", "q25-q4-clean"),
        "granite-h350": ("granite-Q4_K_M.gguf", "granite-q4-clean"),
    }
    # Wait only for the campaign's currently running isolated measurement process.
    # Never terminate or replace a process using the port.
    while True:
        with socket.socket() as probe:
            busy = probe.connect_ex(("127.0.0.1", 19091)) == 0
        if not busy:
            break
        print(json.dumps({"local_grid": "waiting_for_existing_isolated_process"}), flush=True)
        time.sleep(30)
    blocked = []
    for bucket in (2048, 8192, 32768):
        for alias, (model_name, output_name) in models.items():
            output = ARTIFACTS / "local-inference" / output_name
            measurements = output / "measurements.jsonl"
            if measurements.exists():
                metadata = json.loads((output / "metadata.json").read_text())
                assert metadata["model_sha256"] == digest(ARTIFACTS / model_name)
                assert metadata["prompts_sha256"] == digest(prompts)
                assert metadata["runtime_sha"] == "f072b103714dfa1eee531f80b24512faf38e3dd2"
                assert metadata["threads"] == 4 and metadata["concurrency"] == 1
                records = [json.loads(line) for line in measurements.read_text().splitlines()]
                selected = [r for r in records if r["bucket"] == bucket]
                if len(selected) == len({(r["case_id"], r["repetition"]) for r in selected}) == 40:
                    assert all(not r["truncated"] for r in selected)
                    assert all(r["server_timings"]["cache_n"] == 0 for r in selected)
                    print(
                        json.dumps(
                            {"model": alias, "bucket": bucket, "verified_cached_results": 40}
                        ),
                        flush=True,
                    )
                    continue
            cmd = [
                sys.executable,
                str(ROOT / "scripts/measure_r2_local.py"),
                "--model",
                str(ARTIFACTS / model_name),
                "--alias",
                alias,
                "--runtime",
                str(runtime),
                "--prompts",
                str(prompts),
                "--output",
                str(output),
                "--port",
                "19091",
                "--bucket",
                str(bucket),
            ]
            if measurements.exists():
                cmd.append("--resume")
            print(json.dumps({"model": alias, "bucket": bucket, "state": "started"}), flush=True)
            env = {
                **os.environ,
                "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
                "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "1",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
                "TABCOMPLETE_CAMPAIGN_ID": "tabcomplete-model-data-r2",
                "TABCOMPLETE_RUN_ID": "r2-local-" + alias + "-clean",
            }
            result = subprocess.run(cmd, cwd=ROOT, env=env)
            if result.returncode:
                blocked.append({"model": alias, "bucket": bucket, "exit_code": result.returncode})
                print(
                    json.dumps(
                        {
                            "model": alias,
                            "bucket": bucket,
                            "state": "runtime_blocked",
                            "exit_code": result.returncode,
                        }
                    ),
                    flush=True,
                )
                # Other candidates remain independent; never assign a zero quality score.
                continue
    write_json(
        REPORT / "local_inference/grid-status.json",
        {
            "status": "partial" if blocked else "complete",
            "blocked": blocked,
            "observed_at": now(),
            "prompts_sha256": digest(prompts),
        },
    )
    if blocked:
        raise RuntimeError(
            "local grid has blocked stages; retained independent completed measurements"
        )


def advance_campaign(plan):
    """Advance the actual bounded jobs; never call a submitted job a completed experiment."""
    baseline = REPORT / "baseline_evaluations/job.json"
    if not baseline.exists():
        submit_baseline(plan)
        return False
    collect_jobs()
    active = [
        json.loads(path.read_text())
        for path in REPORT.rglob("job.json")
        if not json.loads(path.read_text()).get("terminal")
    ]
    if active:
        print(
            json.dumps({"campaign": "running", "jobs": [s["reference"] for s in active]}),
            flush=True,
        )
        return False
    # Resume only verified scientific data, not merely existing output filenames.
    suite = ROOT / "data/benchmarks/code_completion_v2.jsonl"
    case_ids = {json.loads(line)["id"] for line in suite.read_text().splitlines()}
    for alias in ("q35-p12", "q25-coder"):
        path = ARTIFACTS / "baseline/model_data_r2_baseline" / (alias + ".jsonl")
        metadata = json.loads(path.with_suffix(".jsonl.metadata.json").read_text())
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert metadata["suite_sha256"] == digest(suite)
        assert metadata["protocol"] == "causal-context-v1" and metadata["max_new_tokens"] == 96
        assert len(records) == len(case_ids) == 200
        assert {row["case_id"] for row in records} == case_ids
    for arm in ("R2_STANDARD", "R2_FILTERED"):
        completed = REPORT / "pilots" / arm / "completion.json"
        if completed.exists():
            record = json.loads(completed.read_text())
            final = ROOT / record["artifact_directory"] / arm / "final/model.safetensors"
            assert record["checkpoint_complete_and_hash_verified"]
            assert digest(final) == record["model_sha256"]
            continue
        jobs = [
            json.loads(path.read_text()) for path in (REPORT / "pilots" / arm).rglob("job.json")
        ]
        if jobs:
            latest = max(jobs, key=lambda job: job["submitted_at"])
            if "COMPLETE" not in latest["last_status"]:
                raise RuntimeError(
                    "pilot requires diagnosis, not an automatic training retry: " + arm
                )
            retrieve_pilot(arm, latest["reference"])
        else:
            submit_pilot(arm)
        return False
    for context_only, relative in (
        (False, "baseline_evaluations/additional/job.json"),
        (True, "long_context/job.json"),
    ):
        if not (REPORT / relative).exists():
            submit_evaluation(context_only=context_only)
            return False
    print(
        json.dumps(
            {
                "campaign": "gpu_jobs_terminal",
                "scientific_completion": False,
                "next": (
                    "Retrieve evaluation/context artifacts, judge predictions, "
                    "reconcile telemetry, and finish local measurements and paired analysis"
                ),
            }
        ),
        flush=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/model_data_r2.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stage",
        choices=[
            "freeze",
            "baseline",
            "amend",
            "package",
            "package-baseline",
            "pilot",
            "collect",
            "retrieve-pilot",
            "evaluation",
            "local",
            "context",
            "campaign",
        ],
        default="campaign",
    )
    parser.add_argument("--reason")
    parser.add_argument("--outcomes-inspected", default="none")
    parser.add_argument("--arm", choices=["R2_STANDARD", "R2_FILTERED"])
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--reference")
    parser.add_argument(
        "--once", action="store_true", help="Advance one verified campaign transition"
    )
    args = parser.parse_args()
    if args.stage == "amend":
        if not args.execute:
            raise ValueError("amend requires explicit --execute")
        amend(args.config, args.reason, args.outcomes_inspected)
        return
    plan = freeze(args.config)
    if args.execute and args.stage == "campaign":
        while True:
            terminal = advance_campaign(plan)
            if terminal or args.once:
                return
            time.sleep(30)
    if args.execute and args.stage == "local":
        run_local_grid()
        return
    if args.execute and args.stage == "retrieve-pilot":
        if not args.arm or not args.reference:
            raise ValueError("retrieve-pilot requires --arm and --reference")
        retrieve_pilot(args.arm, args.reference)
        return
    if args.execute and args.stage == "evaluation":
        submit_evaluation()
        return
    if args.execute and args.stage == "context":
        submit_evaluation(context_only=True)
        return
    if args.execute and args.stage == "package-baseline":
        package_baseline()
        return
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
