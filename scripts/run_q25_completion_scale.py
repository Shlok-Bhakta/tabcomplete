"""Freeze and orchestrate the bounded Q25 completion-scale comparison.

CPU preparation and remote file inspection are separate from the explicit
``--execute`` action. The two variants use separate private inputs, kernels,
outputs, receipts, and one shared campaign ledger.
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
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/q25_completion_scale_r1"
ARTIFACTS = Path("/mnt/ssd/tabcomplete-q25-completion-scale-r1")
CONFIG = ROOT / "configs/research/q25_completion_scale_r1.yaml"
PREPARATION_PLAN = REPORT / "preparation_plan-r2.json"
ANALYSIS_PLAN = REPORT / "analysis_plan-r2.json"
TRAINING_PLANS = {
    "repeat": REPORT / "training_plan_repeat.json",
    "scaled": REPORT / "training_plan_scaled.json",
}
CORPUS = ARTIFACTS / "corpus-r3"
BASE_REPORT = ROOT / "reports/research/q25_code_cpt_r2"
BASE_PLAN = BASE_REPORT / "fim_training_plan.json"
RUNTIME_LOCK = BASE_REPORT / "fim_runtime_lock.json"
TRAIN_ARM = "untouched_q25_to_fim"
VARIANTS = ("repeat", "scaled")
ALLOCATION_ORDER = VARIANTS
DATASETS = {
    "repeat": "shlokbhakta/tabcomplete-q25-completion-scale-r1-repeat",
    "scaled": "shlokbhakta/tabcomplete-q25-completion-scale-r1-scaled",
}
KERNELS = {
    "repeat": "shlokbhakta/tc-q25-completion-scale-r1-repeat",
    "scaled": "shlokbhakta/tc-q25-completion-scale-r1-scaled",
}
BASE_DATASET = "shlokbhakta/tabcomplete-one-line-instinct-pilot-r1-inputs"
WORKER = ROOT / "kaggle/q25_code_cpt_r2/run_fim.py"
LINE_FIXTURE = Path(
    "/mnt/ssd/tabcomplete-preserved-research/model_data_r2/frozen-corpora/causal_line_v1-r3.jsonl"
)
PLAN_SCHEMA = "q25-completion-scale-training-plan-v1"
CORPUS_SCHEMA = "q25-completion-scale-corpus-v1"
INPUT_SCHEMA = "q25-completion-scale-input-v1"
LEDGER_SCHEMA = "q25-completion-scale-budget-v1"
SESSION_SECONDS = 10_800
AGGREGATE_SESSION_SECONDS = 21_600
CONSERVATIVE_ACCOUNT_GPU_HOURS = 12
QUOTA_MULTIPLIER = 2
CAMPAIGN_TOKEN_CAP = 10_000_000
DISCARDED_TOKEN_CAP = 2_000_000
ARM_TOKEN_CAP = 8_000_000
ARTIFACT_CAP_BYTES = 12 * 1024**3
MIN_FREE_BYTES = 2 * 1024**3
FINALIZATION_RESERVE_SECONDS = 1_800
MAX_OUTPUT_RESERVATION_BYTES = 4 * 1024**3
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
KAGGLE_REF = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+\Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("campaign JSON must be an object")
    return value


def cli(*args: str, timeout: int = 120) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} command failed (exit {result.returncode})")
    return result.stdout.strip()


def _campaign_module() -> Any:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_q25_code_cpt

    return run_q25_code_cpt


def quota() -> dict[str, Any]:
    """Reuse the existing authenticated Kaggle quota and status reader."""
    observation = _campaign_module().quota()
    # The shared live-quota CLI can leave old paginated rows as ``unknown``;
    # resolve them through the existing authenticated 404 check before gating.
    _pilot_module()._resolve_unknown_kernel_statuses(observation)
    return observation


def check_quota(plan: dict[str, Any], observation: dict[str, Any]) -> None:
    _campaign_module().check_quota(plan, observation)


def _pilot_module() -> Any:
    helper_dir = str(ROOT / "kaggle/one_line_gpu_pilot_r1")
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    import build_pilot

    return build_pilot


def _config() -> dict[str, Any]:
    configuration = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError("completion-scale config is invalid")
    configured_report = Path(configuration.get("report_root", ""))
    if not configured_report.is_absolute():
        configured_report = ROOT / configured_report
    configured_artifacts = Path(configuration.get("artifact_root", ""))
    configured_corpus = Path(configuration.get("corpus_dir", ""))
    if (
        configuration.get("schema") != "q25-completion-scale-config-v1"
        or configuration.get("branch") != "research/q25-completion-scale-r1"
        or configured_report.resolve() != REPORT.resolve()
        or configured_artifacts.resolve() != ARTIFACTS.resolve()
        or configured_corpus.resolve() != CORPUS.resolve()
        or configuration.get("preparation_plan") != PREPARATION_PLAN.relative_to(ROOT).as_posix()
        or configuration.get("preparation_plan_sha256") != digest(PREPARATION_PLAN)
    ):
        raise ValueError("completion-scale config differs from its frozen preparation plan")
    budget = configuration.get("budget")
    required_budget = {
        "aggregate_session_seconds": AGGREGATE_SESSION_SECONDS,
        "session_seconds": SESSION_SECONDS,
        "conservative_account_gpu_hours": CONSERVATIVE_ACCOUNT_GPU_HOURS,
        "conservative_quota_multiplier": QUOTA_MULTIPLIER,
        "maximum_campaign_input_tokens": CAMPAIGN_TOKEN_CAP,
        "maximum_discarded_replay_input_tokens": DISCARDED_TOKEN_CAP,
        "new_artifact_bytes_cap": ARTIFACT_CAP_BYTES,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "minimum_finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "automatic_renewal_use": False,
        "paid_compute": False,
        "gpu_allocation_authorized_by_this_plan": False,
    }
    if not isinstance(budget, dict) or any(
        budget.get(key) != value for key, value in required_budget.items()
    ):
        raise ValueError("completion-scale budget differs from its frozen authorization")
    return configuration


def _preparation() -> dict[str, Any]:
    preparation = read_json(PREPARATION_PLAN)
    if (
        preparation.get("schema") != "q25-completion-scale-preparation-plan-v1"
        or preparation.get("plan_revision") != 2
        or preparation.get("gpu_execution_authorized") is not None
        or preparation.get("budget", {}).get("gpu_allocation_authorized_by_this_plan") is not False
    ):
        raise ValueError("completion-scale preparation plan is not the frozen CPU-only revision")
    return preparation


def _safe_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("campaign input is missing or unsafe")


def _artifact_tree_bytes(root: Path = ARTIFACTS) -> int:
    if not root.exists():
        return 0
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign artifact root is unsafe")
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("campaign artifact tree contains a symbolic link")
        if path.is_file():
            total += path.stat().st_size
    return total


def _check_storage(*, projected_bytes: int = 0) -> None:
    if projected_bytes < 0:
        raise ValueError("projected storage cannot be negative")
    present = _artifact_tree_bytes()
    if present + projected_bytes > ARTIFACT_CAP_BYTES:
        raise OSError("completion-scale artifact cap would be exceeded")
    usage_root = ARTIFACTS if ARTIFACTS.exists() else ARTIFACTS.parent
    if shutil.disk_usage(usage_root).free < projected_bytes + MIN_FREE_BYTES:
        raise OSError("completion-scale free-space reserve would be exhausted")


def _source_files() -> dict[str, str]:
    paths = (
        CONFIG,
        PREPARATION_PLAN,
        BASE_PLAN,
        ROOT / "scripts/run_q25_completion_scale.py",
        ROOT / "scripts/analyze_q25_completion_scale.py",
        WORKER,
        ROOT / "src/tinycomplete/code_cpt/q25_fim.py",
        ROOT / "src/tinycomplete/code_cpt/q25.py",
        ROOT / "src/tinycomplete/eval/q25_fim_attention.py",
        ROOT / "scripts/evaluate_q25_fim.py",
        ROOT / "scripts/evaluate_q25_fim_regression.py",
        ROOT / "scripts/prepare_q25_completion_scale.py",
        ANALYSIS_PLAN,
        ROOT / "reports/research/q25_code_cpt_r2/fim_runtime_requirements.lock",
        RUNTIME_LOCK,
    )
    result: dict[str, str] = {}
    for path in paths:
        _safe_regular_file(path)
        result[path.relative_to(ROOT).as_posix()] = digest(path)
    return result


def _corpus_records(preparation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = CORPUS / "corpus_metadata.json"
    _safe_regular_file(metadata_path)
    metadata = read_json(metadata_path)
    expected_plan_sha = digest(PREPARATION_PLAN)
    if (
        metadata.get("schema") != CORPUS_SCHEMA
        or metadata.get("preparation_plan_sha256") != expected_plan_sha
        or metadata.get("parent_cpt_plan_sha256") != preparation.get("parent_cpt_plan_sha256")
        or metadata.get("tokenizer_sha256") != preparation.get("model", {}).get("tokenizer_sha256")
        or metadata.get("raw_source_content_emitted") is not False
        or metadata.get("sealed_test_accessed") is not False
    ):
        raise ValueError("prepared completion-scale corpus provenance differs")
    splits = metadata.get("splits")
    files = metadata.get("files")
    expected_names = {
        "repeat_train": "repeat_train.jsonl",
        "scaled_train": "scaled_train.jsonl",
        "development_new": "development_new.jsonl",
        "development_previous": "development_previous.jsonl",
    }
    if not isinstance(splits, dict) or set(splits) != set(expected_names):
        raise ValueError("prepared completion-scale corpus split set differs")
    if not isinstance(files, dict):
        raise ValueError("prepared completion-scale corpus has no file ledger")
    records: dict[str, Any] = {}
    for key, filename in expected_names.items():
        split = splits.get(key)
        file_record = files.get(filename)
        path = CORPUS / filename
        _safe_regular_file(path)
        if not isinstance(split, dict) or not isinstance(file_record, dict):
            raise ValueError("prepared completion-scale corpus record is incomplete")
        record = {
            "file": filename,
            "sha256": split.get("sha256"),
            "bytes": split.get("bytes"),
            "row_count": split.get("row_count"),
            "input_tokens": split.get("input_tokens"),
            "target_tokens": split.get("target_tokens"),
        }
        if (
            split.get("file") != filename
            or file_record.get("sha256") != record["sha256"]
            or file_record.get("bytes") != record["bytes"]
            or not isinstance(record["sha256"], str)
            or HEX_SHA256.fullmatch(record["sha256"]) is None
            or type(record["bytes"]) is not int
            or type(record["row_count"]) is not int
            or type(record["input_tokens"]) is not int
            or type(record["target_tokens"]) is not int
            or min(
                record["bytes"],
                record["row_count"],
                record["input_tokens"],
                record["target_tokens"],
            )
            < 0
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError(f"prepared {key} corpus identity differs")
        records[key] = record
    if (
        records["repeat_train"]["sha256"] != preparation.get("previous_training", {}).get("sha256")
        or records["development_previous"]["sha256"]
        != preparation.get("previous_development", {}).get("sha256")
        or records["repeat_train"]["row_count"] != 4096
        or records["scaled_train"]["row_count"] != 8192
        or records["development_new"]["row_count"] != 512
        or records["development_previous"]["row_count"] != 240
    ):
        raise ValueError("prepared completion-scale split counts or preserved baselines differ")
    return metadata, records


def _plan_common_fields(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key not in {"scale_variant", "frozen_at"}}


def freeze(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("unknown completion-scale variant")
    configuration = _config()
    preparation = _preparation()
    metadata, records = _corpus_records(preparation)
    parent = read_json(BASE_PLAN)
    if parent.get("schema") != "q25-fim-training-plan-v1" or not parent.get(
        "gpu_execution_authorized"
    ):
        raise ValueError("the verified Q25 FIM runtime/training plan is unavailable")
    runtime_lock_document = read_json(RUNTIME_LOCK)
    runtime_lock_expected = preparation.get("runtime_lock", {})
    if digest(RUNTIME_LOCK) != runtime_lock_expected.get("sha256"):
        raise ValueError("pinned Q25 runtime lock differs from the preparation plan")
    if _config().get("model") != preparation.get("model"):
        raise ValueError("Q25 base model identity differs from the frozen plan")

    initializer = deepcopy(parent["initializers"][TRAIN_ARM])
    training = deepcopy(parent["configuration"]["training"])
    training.pop("epochs", None)
    training["max_input_tokens"] = ARM_TOKEN_CAP
    budget = {
        "aggregate_session_seconds": AGGREGATE_SESSION_SECONDS,
        "conservative_account_gpu_hours": CONSERVATIVE_ACCOUNT_GPU_HOURS,
        "runtime_setup_reserve_seconds": configuration["budget"]["runtime_setup_reserve_seconds"],
        "session_seconds": SESSION_SECONDS,
        "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "minimum_finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "maximum_additional_training_input_tokens": ARM_TOKEN_CAP,
        "maximum_campaign_input_tokens": CAMPAIGN_TOKEN_CAP,
        "maximum_discarded_replay_input_tokens": DISCARDED_TOKEN_CAP,
        "new_artifact_bytes_cap": ARTIFACT_CAP_BYTES,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "paid_compute": False,
        "automatic_renewal_use": False,
        "conservative_quota_multiplier": QUOTA_MULTIPLIER,
        "quota_renewal": configuration["budget"]["quota_renewal"],
    }
    source_files = _source_files()
    fixture_paths = {
        "causal": ROOT / "data/benchmarks/code_completion_v2.jsonl",
        "line": LINE_FIXTURE,
    }
    fixture_records = {}
    for name, path in fixture_paths.items():
        _safe_regular_file(path)
        old_record = parent["evaluation"]["fixtures"][name]
        if digest(path) != old_record.get("sha256") or path.stat().st_size != old_record.get(
            "bytes"
        ):
            raise ValueError(f"frozen {name} evaluation fixture differs")
        fixture_records[name] = deepcopy(old_record)
    evaluation = deepcopy(parent["evaluation"])
    evaluation["source_syntax"]["development_splits"] = [
        "development_new", "development_previous"
    ]
    evaluation.update(
        {
            "fixtures": fixture_records,
            "development": (
                "512 new repository-disjoint synthetic source states; exact-and-terminated primary"
            ),
            "historical_development": "240 preserved prior states; secondary historical comparison",
            "primary": preparation["evaluation"]["primary"],
            "secondary": preparation["evaluation"]["secondary"],
            "max_output_tokens": preparation["evaluation"]["max_output_tokens"],
            "decoding": preparation["evaluation"]["decoding"],
            "automatic_editor_promotion": False,
            "automatic_personalization_enabled": False,
            "stable_served_model_unchanged": True,
            "general_next_edit_quality_established": False,
            "selection": preparation["evaluation"]["decision"],
        }
    )
    evaluation["paired_analysis"] = {
        **deepcopy(parent["evaluation"]["paired_analysis"]),
        "primary": "development_new exact_and_terminated",
        "secondary": ["development_previous", "line exact", "raw causal functional pass"],
        "analysis_plan_sha256": digest(ANALYSIS_PLAN),
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_revision": 1,
        "branch": configuration["branch"],
        "base_commit": cli("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "frozen_at": datetime.now(UTC).isoformat(),
        "gpu_execution_authorized": True,
        "scale_variant": variant,
        "preparation_plan_sha256": digest(PREPARATION_PLAN),
        "parent_cpt_plan_sha256": preparation["parent_cpt_plan_sha256"],
        "parent_cpt_runtime_observation": deepcopy(parent.get("parent_cpt_runtime_observation")),
        "configuration": {
            "training": training,
            "runtime": deepcopy(parent["configuration"]["runtime"]),
            "runtime_lock": deepcopy(parent["configuration"]["runtime_lock"]),
            "budget": budget,
        },
        "data": {
            "corpus_metadata_sha256": digest(CORPUS / "corpus_metadata.json"),
            **records,
            "previous_training_sha256": preparation["previous_training"]["sha256"],
            "previous_development_sha256": preparation["previous_development"]["sha256"],
        },
        "experiment": deepcopy(preparation["experiment"]),
        "initializers": {TRAIN_ARM: initializer},
        "evaluation": evaluation,
        "analysis_plan_sha256": digest(ANALYSIS_PLAN),
        "source_identity": {
            "commit": cli("git", "-C", str(ROOT), "rev-parse", "HEAD"),
            "branch": configuration["branch"],
            "files": source_files,
            "runtime_lock_sha256": digest(RUNTIME_LOCK),
            "requirements_lock_sha256": parent["configuration"]["runtime_lock"][
                "requirements_lock_sha256"
            ],
        },
        "runtime_lock_file_sha256": digest(RUNTIME_LOCK),
        "runtime_lock_schema": runtime_lock_document.get("schema"),
    }
    if plan["base_commit"] != plan["source_identity"]["commit"]:
        raise ValueError("source commit changed during plan freeze")
    target = TRAINING_PLANS[variant]
    if target.exists():
        frozen = read_json(target)
        comparison = dict(plan)
        comparison["frozen_at"] = frozen.get("frozen_at")
        if frozen != comparison:
            raise ValueError("frozen completion-scale plan differs; create an explicit revision")
        plan = frozen
    else:
        save(target, plan)
    peer = TRAINING_PLANS["scaled" if variant == "repeat" else "repeat"]
    if peer.exists() and _plan_common_fields(read_json(peer)) != _plan_common_fields(plan):
        raise ValueError("paired scale plans do not share an immutable common identity")
    return plan


def load_plan(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("unknown completion-scale variant")
    plan = read_json(TRAINING_PLANS[variant])
    _config()
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("scale_variant") != variant
        or plan.get("gpu_execution_authorized") is not True
        or plan.get("preparation_plan_sha256") != digest(PREPARATION_PLAN)
        or plan.get("base_commit") != plan.get("source_identity", {}).get("commit")
        or plan.get("branch") != "research/q25-completion-scale-r1"
        or plan.get("analysis_plan_sha256") != digest(ANALYSIS_PLAN)
        or plan.get("source_identity", {}).get("files") != _source_files()
    ):
        raise ValueError("frozen completion-scale plan identity differs")
    for relative, expected in plan.get("source_identity", {}).get("files", {}).items():
        path = ROOT / relative
        _safe_regular_file(path)
        if digest(path) != expected:
            raise ValueError("frozen completion-scale source code changed")
    if digest(RUNTIME_LOCK) != plan.get("runtime_lock_file_sha256"):
        raise ValueError("frozen completion-scale runtime lock changed")
    return plan


def _bundle_paths(plan: dict[str, Any]) -> dict[str, Path]:
    return {
        "plan.json": TRAINING_PLANS[plan["scale_variant"]],
        "repeat_train.jsonl": CORPUS / "repeat_train.jsonl",
        "scaled_train.jsonl": CORPUS / "scaled_train.jsonl",
        "development_new.jsonl": CORPUS / "development_new.jsonl",
        "development_previous.jsonl": CORPUS / "development_previous.jsonl",
        "corpus_metadata.json": CORPUS / "corpus_metadata.json",
        "causal200.jsonl": ROOT / "data/benchmarks/code_completion_v2.jsonl",
        "line180.jsonl": LINE_FIXTURE,
    }


def build_bundle(plan: dict[str, Any]) -> Path:
    variant = plan.get("scale_variant")
    if variant not in VARIANTS or plan != load_plan(variant):
        raise ValueError("bundle requires the exact frozen variant plan")
    output = ARTIFACTS / f"input-bundle-{variant}"
    output.mkdir(parents=True, exist_ok=True)
    sources = _bundle_paths(plan)
    allowed = set(sources) | {"input-manifest.json", "dataset-metadata.json"}
    if any(
        path.name not in allowed or path.is_symlink() or not path.is_file()
        for path in output.iterdir()
    ):
        raise ValueError("completion-scale input staging contains an unapproved artifact")
    expected_files = {
        "repeat_train.jsonl": plan["data"]["repeat_train"],
        "scaled_train.jsonl": plan["data"]["scaled_train"],
        "development_new.jsonl": plan["data"]["development_new"],
        "development_previous.jsonl": plan["data"]["development_previous"],
        "corpus_metadata.json": {"sha256": plan["data"]["corpus_metadata_sha256"]},
        "causal200.jsonl": plan["evaluation"]["fixtures"]["causal"],
        "line180.jsonl": plan["evaluation"]["fixtures"]["line"],
    }
    for name, source in sources.items():
        _safe_regular_file(source)
        expected_record = expected_files.get(name)
        if expected_record is not None and digest(source) != expected_record["sha256"]:
            raise ValueError("completion-scale input differs from the frozen plan")
    projected = sum(
        source.stat().st_size for name, source in sources.items() if not (output / name).exists()
    )
    _check_storage(projected_bytes=projected)
    file_records: dict[str, dict[str, Any]] = {}
    for name, source in sources.items():
        target = output / name
        source_sha = digest(source)
        if target.exists() and (target.is_symlink() or digest(target) != source_sha):
            raise ValueError("immutable completion-scale staged input differs")
        if not target.exists():
            shutil.copy2(source, target)
        file_records[name] = {"sha256": source_sha, "bytes": target.stat().st_size}
    save(
        output / "input-manifest.json",
        {
            "schema": INPUT_SCHEMA,
            "scale_variant": variant,
            "plan_sha256": digest(TRAINING_PLANS[variant]),
            "model_dataset": BASE_DATASET,
            "files": file_records,
        },
    )
    save(
        output / "dataset-metadata.json",
        {
            "id": DATASETS[variant],
            "title": f"TabComplete Q25 completion scale r1 {variant} inputs",
            "licenses": [{"name": "other"}],
        },
    )
    _check_storage()
    return output


def _remote_input_expected(output: Path) -> dict[str, dict[str, int]]:
    manifest = read_json(output / "input-manifest.json")
    expected = {name: {"bytes": item["bytes"]} for name, item in manifest["files"].items()}
    expected["input-manifest.json"] = {"bytes": (output / "input-manifest.json").stat().st_size}
    return expected


def upload_bundle(plan: dict[str, Any]) -> dict[str, Any]:
    variant = plan["scale_variant"]
    output = build_bundle(plan)
    manifest_path = output / "input-manifest.json"
    marker = ARTIFACTS / f"dataset-submission-{variant}.json"
    expected = _remote_input_expected(output)
    pilot = _pilot_module()
    if marker.exists():
        receipt = read_json(marker)
        if receipt.get("dataset") != DATASETS[variant] or receipt.get(
            "input_manifest_sha256"
        ) != digest(manifest_path):
            raise ValueError("completion-scale dataset receipt differs from staged inputs")
    else:
        refs = pilot._csv_refs(
            ["kaggle", "datasets", "list", "--mine", "--page-size", "100", "--csv"]
        )
        if DATASETS[variant] in refs:
            raise ValueError("completion-scale dataset already exists without this receipt")
        result = cli("kaggle", "datasets", "create", "-p", str(output), "-t", timeout=900)
        if "Your private Dataset is being created." not in result:
            raise RuntimeError("Kaggle did not confirm private completion-scale dataset creation")
        receipt = {
            "dataset": DATASETS[variant],
            "variant": variant,
            "state": "created",
            "input_manifest_sha256": digest(manifest_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
        save(marker, receipt)
    verification = pilot.wait_for_remote_inputs(DATASETS[variant], expected)
    _check_storage()
    receipt = {**receipt, "state": "verified", "remote_verification": verification}
    save(marker, receipt)
    return receipt


def _planned_variant_input_tokens(plan: dict[str, Any]) -> int:
    variant = plan["scale_variant"]
    split = "repeat_train" if variant == "repeat" else "scaled_train"
    one_pass = plan["data"][split]["input_tokens"]
    planned = one_pass * (2 if variant == "repeat" else 1)
    if type(planned) is not int or planned <= 0 or planned > ARM_TOKEN_CAP:
        raise ValueError("frozen variant training tokens exceed the per-arm budget")
    return planned


def _budget_ledger(plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    token_counts = {variant: _planned_variant_input_tokens(plans[variant]) for variant in VARIANTS}
    if sum(token_counts.values()) + DISCARDED_TOKEN_CAP > CAMPAIGN_TOKEN_CAP:
        raise RuntimeError("frozen variant exposure and replay reservations exceed campaign tokens")
    plan_hashes = {variant: digest(TRAINING_PLANS[variant]) for variant in VARIANTS}
    return {
        "schema": LEDGER_SCHEMA,
        "preparation_plan_sha256": digest(PREPARATION_PLAN),
        "plan_sha256": plan_hashes,
        "aggregate_reserved_session_seconds": AGGREGATE_SESSION_SECONDS,
        "conservative_account_gpu_hours_reserved": CONSERVATIVE_ACCOUNT_GPU_HOURS,
        "conservative_quota_multiplier": QUOTA_MULTIPLIER,
        "maximum_campaign_input_tokens": CAMPAIGN_TOKEN_CAP,
        "maximum_discarded_replay_input_tokens": DISCARDED_TOKEN_CAP,
        "planned_variant_input_tokens": token_counts,
        "variants": {
            variant: {
                "reserved_session_seconds": SESSION_SECONDS,
                "reserved_input_tokens": token_counts[variant],
                "state": "reserved",
                "reference": KERNELS[variant],
                "processed_input_tokens_conservative": None,
            }
            for variant in VARIANTS
        },
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _load_or_create_ledger(plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = REPORT / "campaign_budget.json"
    expected = _budget_ledger(plans)
    if not path.exists():
        save(path, expected)
        return expected
    ledger = read_json(path)
    for key in (
        "schema",
        "preparation_plan_sha256",
        "plan_sha256",
        "aggregate_reserved_session_seconds",
        "conservative_account_gpu_hours_reserved",
        "conservative_quota_multiplier",
        "maximum_campaign_input_tokens",
        "maximum_discarded_replay_input_tokens",
        "planned_variant_input_tokens",
    ):
        if ledger.get(key) != expected.get(key):
            raise ValueError("completion-scale campaign budget ledger differs from frozen plans")
    if set(ledger.get("variants", {})) != set(VARIANTS):
        raise ValueError("completion-scale campaign budget ledger has an invalid variant set")
    for variant in VARIANTS:
        record = ledger["variants"][variant]
        expected_reference = KERNELS[variant]
        if record.get("state") != "reserved":
            job_path = REPORT / f"job-{variant}.json"
            _safe_regular_file(job_path)
            expected_reference = _job_reference(read_json(job_path), variant)
        if (
            record.get("reserved_session_seconds") != SESSION_SECONDS
            or record.get("reserved_input_tokens")
            != expected["planned_variant_input_tokens"][variant]
            or record.get("reference") != expected_reference
            or record.get("state")
            not in {
                "reserved", "submission_pending", "submitted", "submission_unknown", "collected"
            }
        ):
            raise ValueError("completion-scale budget ledger variant reservation differs")
        processed = record.get("processed_input_tokens_conservative")
        if processed is not None and (type(processed) is not int or processed < 0):
            raise ValueError("completion-scale budget ledger token total is invalid")
        if record.get("state") == "collected" and processed is None:
            raise ValueError("collected completion-scale reservation lacks token accounting")
    return ledger


def _remote_history_refs() -> set[str]:
    return _pilot_module()._csv_refs(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"]
    )


def _has_unresolved_job() -> bool:
    for path in REPORT.glob("job-*.json"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("completion-scale job receipt is unsafe")
        status = read_json(path).get("status")
        if status in {"submission_pending", "submitted", "submission_unknown"}:
            return True
    return False


def _job_reference(job: dict[str, Any], variant: str) -> str:
    """Bind a canonical Kaggle URL to its recorded requested kernel identity."""
    reference = job.get("reference")
    if (
        variant not in VARIANTS
        or not isinstance(reference, str)
        or KAGGLE_REF.fullmatch(reference) is None
        or reference.split("/", 1)[0] != KERNELS[variant].split("/", 1)[0]
        or (
            reference != KERNELS[variant]
            and job.get("requested_reference") != KERNELS[variant]
        )
    ):
        raise ValueError("completion-scale kernel reference differs from its allocation receipt")
    return reference


def _mark_collected(job_path: Path, job: dict[str, Any], record: dict[str, Any]) -> None:
    variant = record.get("variant")
    if variant not in VARIANTS or record.get("reference") != _job_reference(job, variant):
        raise ValueError("completion-scale terminal receipt has an unknown variant")
    job.update(
        status="collected",
        terminal_status=record.get("status"),
        verified_output_sha256=digest(REPORT / f"verified-{variant}.json"),
    )
    save(job_path, job)
    ledger_path = REPORT / "campaign_budget.json"
    ledger = read_json(ledger_path)
    variant_record = ledger.get("variants", {}).get(variant)
    if not isinstance(variant_record, dict):
        raise ValueError("completion-scale budget ledger lacks its collected variant")
    variant_record.update(
        state="collected",
        processed_input_tokens_conservative=record["processed_input_tokens_conservative"],
        training_status=record["training_status"],
        verified_output_sha256=job["verified_output_sha256"],
    )
    ledger["updated_at"] = datetime.now(UTC).isoformat()
    save(ledger_path, ledger)


def _git_identity(plan: dict[str, Any]) -> tuple[str, str]:
    status = cli("git", "-C", str(ROOT), "status", "--porcelain")
    allowed_report_files = {
        "training_plan_repeat.json",
        "training_plan_scaled.json",
        "campaign_budget.json",
        "quota-before-repeat.json",
        "quota-before-scaled.json",
        "job-repeat.json",
        "job-scaled.json",
        "watch-repeat.json",
        "watch-scaled.json",
        "verified-repeat.json",
        "verified-scaled.json",
    }
    report_prefix = REPORT.relative_to(ROOT).as_posix() + "/"
    for line in status.splitlines():
        if len(line) < 4:
            raise RuntimeError("working tree status is malformed")
        changed_path = line[3:]
        if " -> " in changed_path:
            raise RuntimeError("source rename is not permitted after plan freeze")
        if (
            not changed_path.startswith(report_prefix)
            or Path(changed_path).name not in allowed_report_files
        ):
            raise RuntimeError("commit all completion-scale source changes before allocation")
    branch = cli("git", "-C", str(ROOT), "branch", "--show-current")
    commit = cli("git", "-C", str(ROOT), "rev-parse", "HEAD")
    plan_commit = plan.get("base_commit")
    if (
        branch != plan["branch"]
        or not isinstance(plan_commit, str)
        or not HEX_SHA1.fullmatch(plan_commit)
    ):
        raise RuntimeError("working tree branch or frozen completion-scale commit differs")
    for relative, expected in plan.get("source_identity", {}).get("files", {}).items():
        source = ROOT / relative
        _safe_regular_file(source)
        if digest(source) != expected:
            raise RuntimeError("completion-scale source file changed after plan freeze")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", plan_commit, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("frozen completion-scale code commit is not an ancestor of HEAD")
    remote = cli("git", "-C", str(ROOT), "ls-remote", "origin", f"refs/heads/{branch}")
    if not remote or remote.split()[0] != commit:
        raise RuntimeError("push the frozen completion-scale source branch before allocation")
    return branch, commit


def submit(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("unknown completion-scale variant")
    plans = {item: load_plan(item) for item in VARIANTS}
    if _plan_common_fields(plans["repeat"]) != _plan_common_fields(plans["scaled"]):
        raise ValueError("paired completion-scale plans do not share a common frozen identity")
    plan = plans[variant]
    ledger_path = REPORT / "campaign_budget.json"
    ledger = _load_or_create_ledger(plans)
    variant_record = ledger["variants"][variant]
    if variant_record["state"] != "reserved" or (REPORT / f"job-{variant}.json").exists():
        raise ValueError("completion-scale variant was already submitted or reserved")
    preceding = [
        name
        for name in ALLOCATION_ORDER
        if ALLOCATION_ORDER.index(name) < ALLOCATION_ORDER.index(variant)
    ]
    external_tokens = 0
    for earlier in preceding:
        previous_record = ledger["variants"][earlier]
        verified_path = REPORT / f"verified-{earlier}.json"
        if previous_record["state"] != "collected" or not verified_path.exists():
            raise RuntimeError("collect the preceding paired allocation before starting this arm")
        verified = read_json(verified_path)
        earlier_job = read_json(REPORT / f"job-{earlier}.json")
        if (
            verified.get("training_status")
            not in {"complete", "incomplete", "no_training_executed"}
            or type(verified.get("processed_input_tokens_conservative")) is not int
            or verified["processed_input_tokens_conservative"] < 0
            or verified.get("reference") != _job_reference(earlier_job, earlier)
        ):
            raise RuntimeError(
                "preceding completion-scale arm lacks bounded verified terminal accounting"
            )
        external_tokens += int(verified["processed_input_tokens_conservative"])
    planned = _planned_variant_input_tokens(plan)
    if external_tokens + planned + DISCARDED_TOKEN_CAP > CAMPAIGN_TOKEN_CAP:
        raise RuntimeError("remaining campaign token reservation is insufficient")
    if variant_record["reference"] in _remote_history_refs():
        raise ValueError("completion-scale kernel reference already exists")
    if not (ARTIFACTS / f"dataset-submission-{variant}.json").exists():
        raise FileNotFoundError("verified completion-scale input upload is required")
    submission = read_json(ARTIFACTS / f"dataset-submission-{variant}.json")
    manifest = ARTIFACTS / f"input-bundle-{variant}/input-manifest.json"
    if (
        submission.get("state") != "verified"
        or submission.get("input_manifest_sha256") != digest(manifest)
        or read_json(manifest).get("plan_sha256") != digest(TRAINING_PLANS[variant])
    ):
        raise ValueError("completion-scale private input receipt differs from the frozen bundle")
    _git_identity(plan)
    observation = quota()
    check_quota(plan, observation)
    minimum_remaining = (SESSION_SECONDS / 3600) * QUOTA_MULTIPLIER
    pending_account_hours = sum(
        SESSION_SECONDS / 3600 * QUOTA_MULTIPLIER
        for item in ledger["variants"].values()
        if item["state"] != "collected"
    )
    remaining = observation.get("remaining")
    if (
        not isinstance(remaining, (int, float))
        or isinstance(remaining, bool)
        or float(remaining) < max(minimum_remaining, pending_account_hours)
    ):
        raise RuntimeError("live Kaggle GPU quota is below this session's reserved amount")
    if _has_unresolved_job():
        raise RuntimeError("another completion-scale kernel may still be active or unresolved")

    kernel = ARTIFACTS / f"kernel-{variant}"
    kernel.mkdir(parents=True, exist_ok=True)
    if any(
        path.name not in {"run.py", "kernel-metadata.json"} or path.is_symlink()
        for path in kernel.iterdir()
    ):
        raise ValueError("completion-scale kernel staging contains an unapproved file")
    session = {
        "commit": plan["base_commit"],
        "plan_sha256": digest(TRAINING_PLANS[variant]),
        "input_manifest_sha256": digest(manifest),
        "arm": TRAIN_ARM,
        "attempt": 1,
        "session_seconds": SESSION_SECONDS,
        "resume_source": None,
        "external_campaign_tokens": external_tokens,
        "scale_variant": variant,
    }
    worker = WORKER.read_text(encoding="utf-8")
    if worker.count("__SESSION_JSON__") != 1:
        raise ValueError("completion-scale worker session placeholder is missing or ambiguous")
    (kernel / "run.py").write_text(
        worker.replace("__SESSION_JSON__", repr(session)), encoding="utf-8"
    )
    save(
        kernel / "kernel-metadata.json",
        {
            "id": KERNELS[variant],
            "title": KERNELS[variant].split("/", 1)[1].replace("-", " "),
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [DATASETS[variant], BASE_DATASET],
            "kernel_sources": [],
        },
    )
    _check_storage()
    receipt_path = REPORT / f"job-{variant}.json"
    job = {
        **session,
        "reference": KERNELS[variant],
        "dataset": DATASETS[variant],
        "quota_observation": _sanitize_quota(observation),
        "planned_variant_input_tokens": planned,
        "conservative_reserved_session_seconds": SESSION_SECONDS,
        "conservative_reserved_account_gpu_hours": SESSION_SECONDS / 3600 * QUOTA_MULTIPLIER,
        "status": "submission_pending",
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    variant_record.update(state="submission_pending", plan_sha256=session["plan_sha256"])
    ledger["updated_at"] = datetime.now(UTC).isoformat()
    save(ledger_path, ledger)
    save(REPORT / f"quota-before-{variant}.json", _sanitize_quota(observation))
    save(receipt_path, job)
    try:
        response = cli(
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(kernel),
            "--timeout",
            str(SESSION_SECONDS),
            "--accelerator",
            "NvidiaTeslaT4",
            timeout=240,
        )
        urls = re.findall(
            r"https://www\.kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", response
        )
        resolved = urls[-1] if urls else KERNELS[variant]
        if resolved != KERNELS[variant]:
            job["requested_reference"] = KERNELS[variant]
        job["reference"] = resolved
        _job_reference(job, variant)
        job.update(status="submitted", reference=resolved)
        variant_record.update(state="submitted", reference=resolved)
    except Exception as exc:
        job.update(status="submission_unknown", error_class=type(exc).__name__)
        variant_record.update(state="submission_unknown")
        save(receipt_path, job)
        ledger["updated_at"] = datetime.now(UTC).isoformat()
        save(ledger_path, ledger)
        raise
    save(receipt_path, job)
    ledger["updated_at"] = datetime.now(UTC).isoformat()
    save(ledger_path, ledger)
    return job


def _sanitize_quota(observation: dict[str, Any]) -> dict[str, Any]:
    allowed = ("observed_at", "remaining", "renewal", "units", "source")
    result = {key: observation.get(key) for key in allowed if key in observation}
    for name in ("active_jobs", "job_statuses"):
        rows = observation.get(name)
        if isinstance(rows, list):
            cleaned = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                cleaned.append(
                    {
                        key: row[key]
                        for key in (
                            "reference",
                            "status",
                            "not_found_verified",
                            "http_status",
                            "verified_at",
                            "verification_source",
                        )
                        if key in row
                    }
                )
            result[name] = cleaned
    return result


def collect(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("unknown completion-scale variant")
    plan = load_plan(variant)
    job_path = REPORT / f"job-{variant}.json"
    job = read_json(job_path)
    reference = _job_reference(job, variant)
    saved_verification = REPORT / f"verified-{variant}.json"
    if saved_verification.is_file():
        prior = read_json(saved_verification)
        if (
            job.get("status") not in {"submitted", "collected"}
            or prior.get("reference") != job.get("reference")
            or prior.get("plan_sha256") != job.get("plan_sha256")
            or prior.get("input_manifest_sha256") != job.get("input_manifest_sha256")
            or prior.get("reference") != reference
            or job.get("plan_sha256") != digest(TRAINING_PLANS[variant])
        ):
            raise ValueError("collected completion-scale receipt identity differs")
        _mark_collected(job_path, job, prior)
        return prior
    if job.get("status") == "collected":
        raise ValueError("collected completion-scale output receipt is missing or changed")
    if (
        job.get("status") != "submitted"
        or job.get("scale_variant") != variant
        or job.get("arm") != TRAIN_ARM
        or job.get("attempt") != 1
        or job.get("plan_sha256") != digest(TRAINING_PLANS[variant])
        or job.get("input_manifest_sha256")
        != digest(ARTIFACTS / f"input-bundle-{variant}/input-manifest.json")
        or job.get("commit") != plan.get("base_commit")
        or job.get("reference") != reference
    ):
        raise ValueError("completion-scale allocation receipt differs from its frozen plan")
    status = cli("kaggle", "kernels", "status", job["reference"], timeout=60)
    if not any(terminal in status for terminal in ("COMPLETE", "ERROR")):
        return {"status": status, "checkpoint_verified": False}
    output = ARTIFACTS / f"output-{variant}"
    output.mkdir(parents=True, exist_ok=True)
    _check_storage(projected_bytes=MAX_OUTPUT_RESERVATION_BYTES)
    cli(
        "kaggle",
        "kernels",
        "output",
        job["reference"],
        "-p",
        str(output),
        "-q",
        "--file-pattern",
        r"\.(json|jsonl|log)$",
        timeout=300,
    )
    pointers = list(output.glob("**/training/latest.json"))
    worker_paths = list(output.glob("**/worker-status.json"))
    if len(worker_paths) != 1 or worker_paths[0].is_symlink():
        raise ValueError("completion-scale output must contain one safe worker status")
    worker = read_json(worker_paths[0])
    expected_worker = {
        "schema": "q25-completion-scale-kaggle-worker-status-v1",
        "commit": job["commit"],
        "attempt": 1,
        "arm": TRAIN_ARM,
        "scale_variant": variant,
        "plan_sha256": job["plan_sha256"],
        "input_manifest_sha256": job["input_manifest_sha256"],
    }
    if any(worker.get(key) != value for key, value in expected_worker.items()):
        raise ValueError("completion-scale worker status identity differs")
    run_result_paths = list(output.glob("**/training/run_result.json"))
    updates_paths = list(output.glob("**/training/updates.jsonl"))
    if len(run_result_paths) > 1 or len(updates_paths) > 1:
        raise ValueError("completion-scale output contains ambiguous training results")
    if any(path.is_symlink() for path in (*run_result_paths, *updates_paths)):
        raise ValueError("completion-scale training output contains a symbolic link")
    result = read_json(run_result_paths[0]) if run_result_paths else {}
    committed = 0
    checkpoint_record: dict[str, Any] | None = None
    cursor: dict[str, Any] = {}
    pointer: dict[str, Any] = {}
    if pointers:
        if len(pointers) != 1 or pointers[0].is_symlink():
            raise ValueError("completion-scale output has ambiguous checkpoint pointers")
        pointer = read_json(pointers[0])
        filename = pointer.get("path") or pointer.get("checkpoint")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or re.fullmatch(r"resume-step-[0-9]+\.pt", filename) is None
        ):
            raise ValueError("completion-scale checkpoint pointer filename is invalid")
        filename = Path(filename).name
        marker_path = pointers[0].parent / f"{filename}.complete.json"
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError("completion-scale checkpoint completion marker is missing")
        marker = read_json(marker_path)
        from tinycomplete.code_cpt.q25 import canonical_sha256

        manifest_path = pointers[0].parent / "run_manifest.json"
        _safe_regular_file(manifest_path)
        manifest = read_json(manifest_path)
        manifest_identity = manifest.get("identity")
        declared_initializer = plan["initializers"][TRAIN_ARM]
        expected_initializer = {
            **declared_initializer,
            "files": {
                name: declared_initializer["files"][name]
                for name in ("config.json", "model.safetensors", "tokenizer.json")
            },
        }
        if (
            pointer.get("schema") != "q25-fim-latest-checkpoint-v1"
            or marker.get("version") != 1
            or not isinstance(pointer.get("fingerprint"), str)
            or HEX_SHA256.fullmatch(pointer["fingerprint"]) is None
            or manifest.get("fingerprint") != pointer.get("fingerprint")
            or not isinstance(manifest_identity, dict)
            or canonical_sha256(manifest_identity) != pointer.get("fingerprint")
            or manifest_identity.get("schema") != "q25-completion-scale-resume-v1"
            or manifest_identity.get("plan_sha256") != job["plan_sha256"]
            or manifest_identity.get("scale_variant") != variant
            or manifest_identity.get("arm") != TRAIN_ARM
            or manifest_identity.get("initializer") != expected_initializer
            or (result and result.get("identity") != manifest_identity)
            or (result and result.get("fingerprint") != pointer.get("fingerprint"))
            or marker.get("fingerprint") != pointer.get("fingerprint")
            or marker.get("sha256") != pointer.get("sha256")
            or not isinstance(marker.get("sha256"), str)
            or HEX_SHA256.fullmatch(marker["sha256"]) is None
        ):
            raise ValueError("completion-scale checkpoint marker differs from its pointer")
        _check_storage(projected_bytes=MAX_OUTPUT_RESERVATION_BYTES)
        cli(
            "kaggle",
            "kernels",
            "output",
            job["reference"],
            "-p",
            str(output),
            "-q",
            "--file-pattern",
            re.escape(filename) + r"$",
            timeout=900,
        )
        checkpoint = pointers[0].parent / filename
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or not 0 < checkpoint.stat().st_size <= MAX_OUTPUT_RESERVATION_BYTES
            or digest(checkpoint) != marker["sha256"]
        ):
            raise ValueError("completion-scale checkpoint hash differs from its marker")
        cursor_record = pointer.get("cursor")
        cursor_fields = (
            "next_example_index",
            "completed_updates",
            "attempted_updates",
            "skipped_updates",
            "training_input_tokens",
            "supervised_target_tokens",
            "epoch",
        )
        if (
            not isinstance(cursor_record, dict)
            or any(
                type(cursor_record.get(key)) is not int or cursor_record[key] < 0
                for key in cursor_fields
            )
            or pointer.get("fingerprint") != marker.get("fingerprint")
            or int(filename.removeprefix("resume-step-").removesuffix(".pt"))
            != cursor_record["attempted_updates"]
            or cursor_record["attempted_updates"] > 512
            or cursor_record["next_example_index"] > 8192
            or cursor_record["completed_updates"] + cursor_record["skipped_updates"]
            != cursor_record["attempted_updates"]
        ):
            raise ValueError("completion-scale checkpoint cursor or identity is invalid")
        cursor = cursor_record
        committed = cursor["training_input_tokens"]
        checkpoint_record = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": marker["sha256"],
            "fingerprint": pointer["fingerprint"],
            "run_manifest_sha256": digest(manifest_path),
            "payload_reload_verified": False,
        }
    update_tokens: list[int] = []
    truncated = False
    if updates_paths:
        content = updates_paths[0].read_text(encoding="utf-8")
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if (
                    index != len(lines) - 1
                    or content.endswith("\n")
                    or result.get("status") == "complete"
                ):
                    raise ValueError("completion-scale update log has a malformed record") from None
                truncated = True
                break
            value = row.get("cumulative_input_tokens") if isinstance(row, dict) else None
            if type(value) is not int or value < 0:
                raise ValueError("completion-scale update record lacks token totals")
            if update_tokens and value < update_tokens[-1]:
                raise ValueError("completion-scale update token totals moved backwards")
            update_tokens.append(value)
    elif committed or status.find("COMPLETE") < 0:
        if (
            pointers
            or result
            or worker.get("training_started") is not False
            or worker.get("scale_variant") != variant
            or worker.get("state")
            not in {
                "setup",
                "verified_inputs",
                "baseline_evaluation",
                "training_deferred_insufficient_time",
                "failed",
            }
        ):
            raise ValueError("completion-scale update log is missing; work cannot be bounded")
    clean_training = any(
        isinstance(stage, dict) and stage.get("name") == "training" and stage.get("exit_code") == 0
        for stage in worker.get("stages", [])
    )
    no_training_executed = (
        not pointers
        and not update_tokens
        and not result
        and worker.get("training_started") is False
    )
    if no_training_executed:
        forbidden_names = {"latest.json", "updates.jsonl", "run_result.json", "training.log"}
        if any(
            path.is_file()
            and (
                path.name in forbidden_names
                or (path.name.startswith("resume-step-") and path.suffix == ".pt")
                or path.suffix == ".safetensors"
            )
            for path in output.rglob("*")
        ):
            raise ValueError("zero-work completion-scale receipt has training artifacts")
    latest_logged = max([committed, *update_tokens])
    unlogged = 0 if clean_training or no_training_executed else 16 * 1024
    discarded = latest_logged - committed + unlogged
    own_processed = committed + discarded
    external = int(job.get("external_campaign_tokens", 0))
    total_processed = external + own_processed
    if total_processed > CAMPAIGN_TOKEN_CAP:
        raise RuntimeError("completion-scale processed input token cap was exceeded")
    prior_discarded = 0
    for earlier in ALLOCATION_ORDER[: ALLOCATION_ORDER.index(variant)]:
        earlier_receipt_path = REPORT / f"verified-{earlier}.json"
        if not earlier_receipt_path.is_file():
            raise ValueError("completion-scale token carry lacks an earlier verified receipt")
        earlier_receipt = read_json(earlier_receipt_path)
        earlier_job = read_json(REPORT / f"job-{earlier}.json")
        if earlier_receipt.get("reference") != _job_reference(earlier_job, earlier):
            raise ValueError("completion-scale earlier token receipt has a different reference")
        prior_discarded += int(earlier_receipt.get("discarded_input_tokens_conservative", 0))
    if prior_discarded + discarded > DISCARDED_TOKEN_CAP:
        raise RuntimeError("completion-scale discarded/replay token reservation was exceeded")
    expected_train_tokens = _planned_variant_input_tokens(plan)
    cursor = pointer.get("cursor", {})
    identity = result.get("identity") if isinstance(result, dict) else None
    complete = (
        status.find("COMPLETE") >= 0
        and result.get("status") == "complete"
        and result.get("cursor") == cursor
        and cursor.get("completed_updates") == 512
        and cursor.get("attempted_updates") == 512
        and cursor.get("skipped_updates") == 0
        and committed == expected_train_tokens
        and clean_training
        and worker.get("state") == "complete"
        and result.get("scale_variant") == variant
        and isinstance(identity, dict)
        and identity.get("plan_sha256") == job["plan_sha256"]
        and identity.get("scale_variant") == variant
        and identity.get("arm") == TRAIN_ARM
    )
    if status.find("COMPLETE") >= 0 and not complete and not no_training_executed:
        raise ValueError("Kaggle completed without the exact frozen 512-update training result")
    record = {
        "schema": "q25-completion-scale-verified-output-v1",
        "reference": job["reference"],
        "status": status,
        "variant": variant,
        "attempt": 1,
        "plan_sha256": job["plan_sha256"],
        "input_manifest_sha256": job["input_manifest_sha256"],
        "commit": job["commit"],
        "training_status": "complete"
        if complete
        else "no_training_executed"
        if no_training_executed
        else "incomplete",
        "trainer_status": result.get("status"),
        "checkpoint_verified": bool(pointers),
        "checkpoint": checkpoint_record,
        "cursor": cursor,
        "training_input_tokens": committed,
        "discarded_logged_tail_tokens": latest_logged - committed,
        "discarded_input_tokens_conservative": discarded,
        "unlogged_inflight_input_token_reservation": unlogged,
        "processed_input_tokens_conservative": own_processed,
        "processed_campaign_input_tokens_conservative": total_processed,
        "truncated_last_update_record": truncated,
        "output_root": str(output),
        "observed_at": datetime.now(UTC).isoformat(),
    }
    save(REPORT / f"verified-{variant}.json", record)
    _mark_collected(job_path, job, record)
    _check_storage()
    return record


def watch(variant: str, *, poll_seconds: float = 30) -> dict[str, Any]:
    if variant not in VARIANTS or not 1 <= poll_seconds <= 60:
        raise ValueError("completion-scale watch variant or interval is invalid")
    job = read_json(REPORT / f"job-{variant}.json")
    if job.get("plan_sha256") != digest(TRAINING_PLANS[variant]):
        raise ValueError("completion-scale observer plan identity differs")
    deadline = datetime.fromisoformat(job["submitted_at"].replace("Z", "+00:00")).timestamp()
    deadline += SESSION_SECONDS + 900
    failures = 0
    while time.time() < deadline:
        observed = {
            "reference": job["reference"],
            "observed_at": datetime.now(UTC).isoformat(),
            "automatic_allocation": False,
        }
        try:
            status = cli("kaggle", "kernels", "status", job["reference"], timeout=60)
            observed["status"] = status
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            failures += 1
            observed.update(error_class=type(exc).__name__, consecutive_failures=failures)
            save(REPORT / f"watch-{variant}.json", observed)
            if failures >= 3:
                raise RuntimeError(
                    "completion-scale observer lost connection; no retry was launched"
                ) from None
            time.sleep(poll_seconds)
            continue
        save(REPORT / f"watch-{variant}.json", observed)
        if any(value in status for value in ("COMPLETE", "ERROR")):
            return collect(variant)
        time.sleep(poll_seconds)
    raise TimeoutError("completion-scale observer deadline reached; inspect the existing job")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--bundle", action="store_true")
    actions.add_argument("--upload", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--collect", action="store_true")
    actions.add_argument("--watch", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    result: dict[str, Any]
    if args.freeze:
        result = {"training_plan": freeze(args.variant)}
    else:
        plan = load_plan(args.variant)
        if args.bundle:
            result = {"input_bundle": str(build_bundle(plan))}
        elif args.upload:
            result = {"dataset_upload": upload_bundle(plan)}
        elif args.execute:
            result = {"job": submit(args.variant)}
        elif args.collect:
            result = {"verified_output": collect(args.variant)}
        else:
            result = {"verified_output": watch(args.variant, poll_seconds=args.poll_seconds)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
