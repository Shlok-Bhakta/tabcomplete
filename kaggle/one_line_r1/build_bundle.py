"""Prepare and verify a private, bounded Kaggle one-line SFT submission.

Preparation is CPU-only and does not call `kaggle datasets create` or
`kaggle kernels push`. The campaign controller owns those authorized writes
after it has rechecked quota and active jobs. No model files are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
)
REQUIRED_MODEL_FILES = frozenset({"model.safetensors", "config.json", "tokenizer.json"})
MAX_CAMPAIGN_BYTES = 12 * 1024**3
HEX_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
KAGGLE_REFERENCE = re.compile(r"[a-z0-9_-]+/[a-z0-9_-]+\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return value


def validate_data_gate(manifest: dict[str, Any], config: dict[str, Any], train_path: Path) -> int:
    """Refuse GPU packaging until the reviewed public data gate is met."""
    policy = config["data"]
    if manifest.get("accepted_train", 0) < policy["minimum_main_train"]:
        raise ValueError("accepted training-state gate is below 20,000")
    source_groups = manifest.get(
        "source_groups", manifest.get("source_repositories_with_accepted_rows", 0)
    )
    if source_groups < policy["minimum_source_groups"]:
        raise ValueError("training source-group diversity gate failed")
    mechanisms = manifest.get("mechanisms", manifest.get("mechanisms_with_accepted_rows", 0))
    if mechanisms < policy["minimum_mechanisms"]:
        raise ValueError("training mechanism diversity gate failed")
    for key, ceiling in (
        ("maximum_repository_fraction", policy["maximum_repository_fraction"]),
        ("maximum_template_fraction", policy["maximum_template_fraction"]),
        ("synthetic_repair_fraction", policy["maximum_synthetic_repair_fraction"]),
    ):
        measured = manifest.get(key)
        if measured is None or not 0 <= float(measured) <= ceiling:
            raise ValueError(f"training diversity gate failed: {key}")
    real_fraction = manifest.get("real_source_fraction")
    if real_fraction is None or (
        float(real_fraction) < policy["target_real_source_fraction"]
        and not manifest.get("real_source_exception")
    ):
        raise ValueError("real-source fraction lacks target or documented exception")
    expected = _require_sha(manifest.get("train_sha256"), "train_sha256")
    _require_sha(manifest.get("split_manifest_sha256"), "split_manifest_sha256")
    if sha256_file(train_path) != expected:
        raise ValueError("accepted training shard hash mismatch")
    planned = manifest.get("planned_training_input_tokens")
    if not isinstance(planned, int) or planned <= 0:
        raise ValueError("manifest lacks exact planned training input tokens")
    return planned


def validate_source_rows(train_path: Path, *, expected_count: int) -> None:
    """A second, cheap egress guard before private Kaggle dataset staging."""
    seen: set[str] = set()
    count = 0
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get("id")
            if not isinstance(identifier, str) or identifier in seen:
                raise ValueError("training shard contains duplicate or missing IDs")
            seen.add(identifier)
            source_type = str(row.get("source_type", ""))
            if source_type.startswith("git_"):
                provenance = row.get("provenance") or {}
                if not str(provenance.get("public_url", "")).startswith("https://"):
                    raise ValueError("Git training row lacks public source URL")
            elif not source_type.startswith("synthetic"):
                raise ValueError("training shard contains an unapproved source type")
            if not row.get("source_license") or not row.get("validation", {}).get(
                "inferability_reviewed"
            ):
                raise ValueError("training shard contains unreviewed or unlicensed source")
            if row.get("split", "train") != "train":
                raise ValueError("nontraining split entered Kaggle training input")
            count += 1
    if count != expected_count:
        raise ValueError("training row count differs from frozen manifest")


def check_campaign_budget(
    *,
    plan: dict[str, Any],
    ledger: dict[str, Any],
    quota: dict[str, Any],
    session_seconds: int,
    planned_tokens: int,
) -> None:
    limits = plan["budgets"]
    if session_seconds <= limits["minimum_session_finalization_minutes"] * 60:
        raise ValueError("session leaves no finalization reserve")
    if ledger["session_wall_seconds"] + session_seconds > (
        limits["maximum_kaggle_t4x2_session_wall_hours"] * 3600
    ):
        raise ValueError("campaign T4x2 wall-hour cap")
    if (
        ledger["training_input_tokens"] + planned_tokens
        > (limits["maximum_nonpadding_training_input_tokens"])
    ):
        raise ValueError("campaign training-token cap")
    if quota.get("active_jobs"):
        raise ValueError("another GPU notebook is active")
    original = plan["quota_at_freeze"]
    if quota.get("renewal") != original.get("renewal"):
        raise ValueError("renewed Kaggle allocation is not automatically authorized")
    remaining = quota.get("remaining")
    if remaining is None:
        if ledger["session_wall_seconds"] or session_seconds > 7200:
            raise ValueError("unknown quota permits one initial two-hour session only")
    elif float(remaining) * 3600 < session_seconds * 2:
        # Kaggle T4x2 may charge two account GPU-hours per session wall-hour.
        raise ValueError("conservative T4x2 account-quota check failed")


def _command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{args[0]} identity check failed")
    return result.stdout.strip()


def pushed_clean_commit(branch: str = "research/one-line-r1") -> str:
    if _command("git", "-C", str(ROOT), "status", "--porcelain"):  # includes untracked files
        raise ValueError("commit the campaign bundle before Kaggle packaging")
    commit = _command("git", "-C", str(ROOT), "rev-parse", "HEAD")
    remote = _command("git", "-C", str(ROOT), "ls-remote", "origin", f"refs/heads/{branch}")
    if not remote or remote.split()[0] != commit:
        raise ValueError("push the exact campaign commit before Kaggle packaging")
    return commit


def _stage(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError("bundle output path is already occupied")
    resolved_source = source.resolve(strict=True)
    try:
        os.link(resolved_source, destination)
    except OSError:
        shutil.copy2(resolved_source, destination)
    return {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def prepare_bundle(
    *,
    config_path: Path,
    plan_path: Path,
    ledger_path: Path,
    model_dir: Path,
    train_path: Path,
    data_manifest_path: Path,
    selection_path: Path,
    output: Path,
    dataset_id: str,
    kernel_id: str,
    commit: str,
    quota: dict[str, Any],
    session_seconds: int,
    resume: dict[str, str] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("submission bundle directory already exists")
    if (
        KAGGLE_REFERENCE.fullmatch(dataset_id) is None
        or KAGGLE_REFERENCE.fullmatch(kernel_id) is None
    ):
        raise ValueError("invalid Kaggle dataset or kernel reference")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("kernel code must be pinned to a Git commit")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plan = json_file(plan_path)
    ledger = json_file(ledger_path)
    if config["suite_revision"] != 3 or plan["suite_revision"] != 3:
        raise ValueError("suite revision changed")
    if sha256_file(config_path) != plan["config_sha256"]:
        raise ValueError("frozen campaign configuration changed")
    if config["student"]["initializer"] != "untouched_pretrained":
        raise ValueError("unapproved model initializer")
    data_manifest = json_file(data_manifest_path)
    planned_tokens = validate_data_gate(data_manifest, config, train_path)
    validate_source_rows(train_path, expected_count=data_manifest["accepted_train"])
    selection = json_file(selection_path)
    if selection.get("selected_peak_lr") not in (1e-5, 3e-5, 3e-6):
        raise ValueError("main run has no locked supported learning rate")
    _require_sha(selection.get("development_evidence_sha256"), "development_evidence_sha256")
    check_campaign_budget(
        plan=plan,
        ledger=ledger,
        quota=quota,
        session_seconds=session_seconds,
        planned_tokens=planned_tokens,
    )
    model_files = [name for name in ALLOWED_MODEL_FILES if (model_dir / name).is_file()]
    if not REQUIRED_MODEL_FILES.issubset(model_files):
        raise ValueError("required local q25 model files are missing")
    if sha256_file(model_dir / "model.safetensors") != config["student"]["weight_sha256"]:
        raise ValueError("untouched q25 model hash mismatch")
    if sha256_file(model_dir / "tokenizer.json") != config["student"]["tokenizer_sha256"]:
        raise ValueError("pinned q25 tokenizer hash mismatch")
    projected_bytes = sum((model_dir / name).stat().st_size for name in model_files)
    projected_bytes += train_path.stat().st_size + data_manifest_path.stat().st_size
    projected_bytes += selection_path.stat().st_size
    if projected_bytes > min(
        MAX_CAMPAIGN_BYTES, plan["budgets"]["maximum_new_persistent_local_research_bytes"]
    ):
        raise ValueError("dataset staging would exceed the research storage cap")
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < projected_bytes + 2 * 1024**3:
        raise OSError("insufficient local storage headroom for Kaggle staging")
    if resume is not None:
        _require_sha(resume.get("checkpoint_sha256"), "resume checkpoint SHA-256")
        if KAGGLE_REFERENCE.fullmatch(resume.get("kernel_source", "")) is None:
            raise ValueError("resume kernel source must be an owned Kaggle reference")
        _require_sha(resume.get("fingerprint"), "resume training fingerprint")
    dataset_dir = output / "dataset"
    kernel_dir = output / "kernel"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir()
    staged: dict[str, dict[str, Any]] = {}
    for name in model_files:
        staged[name] = _stage(model_dir / name, dataset_dir / name)
    staged["train.jsonl"] = _stage(train_path, dataset_dir / "train.jsonl")
    staged["data_manifest.json"] = _stage(data_manifest_path, dataset_dir / "data_manifest.json")
    staged["lr_selection.json"] = _stage(selection_path, dataset_dir / "lr_selection.json")
    input_manifest = {
        "schema": "one-line-kaggle-input-v1",
        "model_id": config["student"]["model_id"],
        "model_revision": config["student"]["revision"],
        "plan_sha256": sha256_file(plan_path),
        "config_sha256": sha256_file(config_path),
        "files": staged,
    }
    write_json(dataset_dir / "input-manifest.json", input_manifest)
    dataset_metadata = {
        "title": "TabComplete One Line R1 Private Inputs",
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "description": (
            "Private research bundle. Qwen2.5-Coder-0.5B retains its Apache-2.0 license; "
            "public source rows retain the per-row licenses in data_manifest.json. "
            "No model or source rights are relicensed by this metadata."
        ),
    }
    write_json(dataset_dir / "dataset-metadata.json", dataset_metadata)
    session = {
        "schema": "one-line-kaggle-session-v1",
        "commit": commit,
        "branch": "research/one-line-r1",
        "dataset_id": dataset_id,
        "input_manifest_sha256": sha256_file(dataset_dir / "input-manifest.json"),
        "plan_sha256": input_manifest["plan_sha256"],
        "config_sha256": input_manifest["config_sha256"],
        "train_sha256": staged["train.jsonl"]["sha256"],
        "selection_sha256": staged["lr_selection.json"]["sha256"],
        "session_seconds": session_seconds,
        "reserve_seconds": config["budget"]["minimum_session_finalization_minutes"] * 60,
        "external_campaign_tokens": ledger["training_input_tokens"],
        "resume": resume,
    }
    template = (Path(__file__).parent / "run.py").read_text(encoding="utf-8")
    (kernel_dir / "run.py").write_text(
        template.replace('"__SESSION_LITERAL__"', repr(json.dumps(session, sort_keys=True))),
        encoding="utf-8",
    )
    kernel_metadata = {
        "id": kernel_id,
        "title": kernel_id.split("/", 1)[1],
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_id],
        "kernel_sources": [resume["kernel_source"]] if resume else [],
        "competition_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", kernel_metadata)
    record = {
        "session": session,
        "dataset_metadata": dataset_metadata,
        "kernel_metadata": kernel_metadata,
        "dataset_logical_bytes": sum(item["bytes"] for item in staged.values()),
        "dataset_stage": str(dataset_dir),
        "kernel_stage": str(kernel_dir),
        "submission_commands": [
            f"kaggle datasets create -p {dataset_dir} -t",
            f"kaggle kernels push -p {kernel_dir} --timeout {session_seconds}",
        ],
    }
    write_json(output / "bundle-manifest.json", record)
    return record


def verify_pulled_output(path: Path, *, expected_fingerprint: str | None = None) -> dict[str, Any]:
    """Verify a downloaded Kaggle output before using it as a resume parent."""
    manifests = list(path.rglob("run_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("expected exactly one pulled run manifest")
    root = manifests[0].parent
    run_manifest = json_file(manifests[0])
    latest = json_file(root / "latest.json")
    result = json_file(root / "run_result.json")
    fingerprint = _require_sha(run_manifest.get("fingerprint"), "training fingerprint")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("pulled output belongs to another training identity")
    if result.get("fingerprint") != fingerprint:
        raise ValueError("run result and manifest identities disagree")
    checkpoint_name = Path(latest["checkpoint"]).name
    checkpoint = root / checkpoint_name
    marker = json_file(checkpoint.with_suffix(checkpoint.suffix + ".complete.json"))
    digest = sha256_file(checkpoint)
    if digest != latest["sha256"] or digest != marker["sha256"]:
        raise ValueError("pulled resumable checkpoint hash mismatch")
    if marker.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint belongs to another training identity")
    if latest.get("cursor") != result.get("cursor"):
        raise ValueError("latest checkpoint does not match the reported consumed position")
    return {
        "fingerprint": fingerprint,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "cursor": latest["cursor"],
        "status": result["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/research/one_line_r1.yaml")
    prepare.add_argument(
        "--plan", type=Path, default=ROOT / "reports/research/one_line_r1/plan.json"
    )
    prepare.add_argument(
        "--ledger", type=Path, default=ROOT / "reports/research/one_line_r1/quota_ledger.json"
    )
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--train", type=Path, required=True)
    prepare.add_argument("--data-manifest", type=Path, required=True)
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dataset-id", default="shlokbhakta/tabcomplete-one-line-r1-inputs")
    prepare.add_argument("--kernel-id", default="shlokbhakta/tabcomplete-one-line-r1-main-s01")
    prepare.add_argument("--session-minutes", type=int, default=120)
    prepare.add_argument("--resume-output", type=Path)
    prepare.add_argument("--resume-kernel-source")
    verify = sub.add_parser("verify-output")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expected-fingerprint")
    args = parser.parse_args()
    if args.command == "verify-output":
        print(
            json.dumps(
                verify_pulled_output(args.path, expected_fingerprint=args.expected_fingerprint),
                sort_keys=True,
            )
        )
        return
    # The accepted-data gate is checked before any quota call or submission.
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_data_gate(json_file(args.data_manifest), config, args.train)
    if args.resume_output is not None:
        if not args.resume_kernel_source:
            raise ValueError("resume output needs its exact Kaggle kernel source")
        parent = verify_pulled_output(args.resume_output)
        resume = {
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "fingerprint": parent["fingerprint"],
            "kernel_source": args.resume_kernel_source,
        }
    else:
        resume = None
    commit = pushed_clean_commit()
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_one_line_campaign import live_quota

    record = prepare_bundle(
        config_path=args.config,
        plan_path=args.plan,
        ledger_path=args.ledger,
        model_dir=args.model,
        train_path=args.train,
        data_manifest_path=args.data_manifest,
        selection_path=args.selection,
        output=args.output,
        dataset_id=args.dataset_id,
        kernel_id=args.kernel_id,
        commit=commit,
        quota=live_quota(),
        session_seconds=args.session_minutes * 60,
        resume=resume,
    )
    print(
        json.dumps(
            {
                "dataset_stage": record["dataset_stage"],
                "kernel_stage": record["kernel_stage"],
                "session": record["session"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
