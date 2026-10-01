"""Local-artifact, exact-resume q25 single-line SFT worker for an allocated Kaggle job.

This entry point never allocates a notebook or downloads weights. Run --inspect
for CPU-only preparation, then --execute only inside an explicitly allocated
CUDA session whose quota/deadline has already been checked by the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION
from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    apply_action,
    decode_action,
    encode_action,
)
from tinycomplete.one_line.pilot_data import (
    CONSTRUCTIVE,
    INSTINCT,
    LICENSE_MIXED,
    LICENSE_MIXED_HISTORY,
    PUBLIC_SYNTHETIC,
    _strict_json,
    license_mixed_artifact_root,
    policy_for_schema,
    validate_aggregate_budget,
    validate_constructive_manifest,
    validate_constructive_review,
    validate_constructive_splits,
    validate_license_mixed_manifest,
    validate_license_mixed_review,
    validate_license_mixed_splits,
    validate_pilot_row,
)
from tinycomplete.one_line.train import (
    CosineUpdateSchedule,
    EncodedExample,
    TrainingCursor,
    batch_order_sha256,
    bucketed_batches,
    encode_training_row,
    load_resume_checkpoint,
    save_resume_checkpoint,
    token_counts,
    train_encoded,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
FIXTURE_ACTION_KINDS = ("keep", "replace_line", "insert_before", "delete_line")
FIXTURE_DIAGNOSTIC_ORDINALS = (0, 5, 10, 15)
FIXTURE_V2_PLAN_SCHEMA = "single-line-disposable-training-fixture-v2"
FIXTURE_V1_PLAN_SCHEMA = "single-line-disposable-training-fixture-v1"
FIXTURE_VIABILITY_SCHEMA = "single-line-disposable-implementation-viability-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    return (
        sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if path.exists()
        else 0
    )


def session_deadline_from_invocation(started_monotonic: float, session_minutes: float) -> float:
    """Include artifact preparation and model loading in the allocated session."""
    if started_monotonic < 0 or session_minutes <= 0:
        raise ValueError("session deadline needs a valid invocation time and positive duration")
    return started_monotonic + session_minutes * 60


def finalization_reserve_seconds(reserve_minutes: float, last_save_seconds: float) -> float:
    """Keep the larger of the frozen reserve and observed checkpoint margin."""
    if reserve_minutes < 0 or last_save_seconds < 0:
        raise ValueError("finalization reserve values cannot be negative")
    return max(reserve_minutes * 60, last_save_seconds * 2 + 60)


def remaining_training_seconds(
    deadline_monotonic: float, now_monotonic: float, reserve_seconds: float
) -> float:
    """Return time available for updates after preserving checkpoint reserve."""
    if reserve_seconds < 0:
        raise ValueError("finalization reserve cannot be negative")
    return deadline_monotonic - now_monotonic - reserve_seconds


def disposable_fixture_diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select four fixed examples of each answer-cued action for a bounded check."""
    by_kind = {
        kind: [row for row in rows if row.get("action", {}).get("kind") == kind]
        for kind in FIXTURE_ACTION_KINDS
    }
    if any(len(group) < max(FIXTURE_DIAGNOSTIC_ORDINALS) + 1 for group in by_kind.values()):
        raise ValueError("disposable fixture must provide sixteen examples of every action kind")
    return [
        by_kind[kind][ordinal]
        for kind in FIXTURE_ACTION_KINDS
        for ordinal in FIXTURE_DIAGNOSTIC_ORDINALS
    ]


def disposable_fixture_implementation_viability(
    observations: list[dict[str, Any]], *, training_complete: bool
) -> dict[str, Any]:
    """Score codec execution separately from model quality or intent inference."""
    by_kind = {
        kind: [row for row in observations if row.get("gold_action") == kind]
        for kind in FIXTURE_ACTION_KINDS
    }
    per_action: dict[str, Any] = {}
    failures: list[str] = []
    for kind, group in by_kind.items():
        count = len(group)
        valid = sum(row.get("valid_action") is True for row in group)
        terminated = sum(row.get("terminated_by_eos") is True for row in group)
        exact = sum(row.get("exact_action") is True for row in group)
        per_action[kind] = {
            "examples": count,
            "valid_actions": valid,
            "terminated_by_eos": terminated,
            "exact_actions": exact,
            "minimum_exact_actions": 3,
        }
        if count != len(FIXTURE_DIAGNOSTIC_ORDINALS):
            failures.append(f"{kind}:expected_4_observations")
        elif min(valid, terminated, exact) < 3:
            failures.append(f"{kind}:below_3_of_4_decode_threshold")
    if not training_complete:
        failures.insert(0, "disposable_training_pass_incomplete")
    passed = not failures
    return {
        "schema": "single-line-disposable-implementation-viability-v1",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "scope": "answer-cued_disposable_codec_only",
        "quality_evidence": False,
        "training_complete": training_complete,
        "total_observations": len(observations),
        "per_action": per_action,
        "failure_reasons": failures,
        "criterion": (
            "For each N/R/I/D action, at least 3 of 4 fixed probes decode to the exact "
            "canonical action and terminate with EOS after one complete disposable pass."
        ),
    }


def disposable_fixture_supervision_by_action(
    rows: list[dict[str, Any]], encoded: list[EncodedExample], tokenizer: Any
) -> dict[str, Any]:
    """Audit per-action wire/EOS labels and prompt masking in the disposable pass."""
    if len(rows) != len(encoded):
        raise ValueError("disposable fixture rows and encoded examples differ in count")
    per_action: dict[str, Any] = {}
    for kind in FIXTURE_ACTION_KINDS:
        paired = [
            (row, example)
            for row, example in zip(rows, encoded, strict=True)
            if row.get("action", {}).get("kind") == kind
        ]
        exact_target_count = 0
        masked_prompt_count = 0
        target_tokens = 0
        for row, example in paired:
            action = EditAction(**row["action"])
            response_ids = tokenizer.encode(encode_action(action), add_special_tokens=False)
            expected_response = tuple(response_ids) + (tokenizer.eos_token_id,)
            target = example.input_ids[example.prompt_tokens :]
            labels = example.labels[example.prompt_tokens :]
            if target == expected_response and labels == expected_response:
                exact_target_count += 1
            if all(label == -100 for label in example.labels[: example.prompt_tokens]):
                masked_prompt_count += 1
            target_tokens += example.response_tokens
        per_action[kind] = {
            "examples": len(paired),
            "supervised_response_and_eos_tokens": target_tokens,
            "exact_wire_and_eos_targets": exact_target_count,
            "prompt_positions_masked": masked_prompt_count,
            "passed": (
                bool(paired)
                and exact_target_count == len(paired)
                and masked_prompt_count == len(paired)
            ),
        }
    return {
        "schema": "single-line-disposable-supervision-audit-v1",
        "passed": all(item["passed"] for item in per_action.values()),
        "quality_evidence": False,
        "per_action": per_action,
    }


def validate_disposable_fixture_plan(
    plan: dict[str, Any], *, phase: str, epochs: int, microbatch: int
) -> dict[str, Any] | None:
    """Validate the new implementation-only fixture plan without changing v1."""
    if phase != "fixture":
        return None
    fixture = plan.get("training", {}).get("disposable_fixture", {})
    if not isinstance(fixture, dict):
        raise ValueError("disposable fixture plan must be an object")
    schema = fixture.get("schema")
    if schema in (None, FIXTURE_V1_PLAN_SCHEMA):
        return None
    if schema != FIXTURE_V2_PLAN_SCHEMA:
        raise ValueError("unknown disposable fixture plan schema")
    fixed = {
        "examples": 64,
        "epochs": 1,
        "effective_batch_examples": 2,
        "microbatch_examples": 2,
        "peak_learning_rate": 1e-4,
        "expected_updates": 32,
        "quality_evidence": False,
        "implementation_viability_schema": FIXTURE_VIABILITY_SCHEMA,
        "decode_examples_per_action": 4,
        "minimum_exact_actions_per_action": 3,
        "eos_required": True,
    }
    if any(fixture.get(key) != value for key, value in fixed.items()):
        raise ValueError("disposable fixture v2 differs from its frozen implementation contract")
    if (
        epochs != 1
        or microbatch != 2
        or type(fixture.get("nonpadding_training_input_tokens")) is not int
        or fixture["nonpadding_training_input_tokens"] <= 0
    ):
        raise ValueError("disposable fixture v2 phase or token declaration is invalid")
    if fixture.get("initial_loss_scale", 256.0) not in (128.0, 256.0):
        raise ValueError("unsupported disposable fixture initial loss scale")
    if (
        plan.get("data", {}).get("schema") == LICENSE_MIXED_HISTORY.data_schema
        and fixture.get("initial_loss_scale") != 128.0
    ):
        raise ValueError("history pilot requires the frozen 128 initial loss scale")
    return fixture


def initial_loss_scale_for_run(
    *,
    plan: dict[str, Any],
    phase: str,
    disposable_fixture_plan: dict[str, Any] | None,
) -> float:
    """Use the v2 pilot's frozen stable scale; preserve the legacy default elsewhere."""
    if phase == "pilot" and plan.get("data", {}).get("schema") == PUBLIC_SYNTHETIC.data_schema:
        if plan["training"].get("initial_loss_scale") != 128.0:
            raise ValueError("source/functional pilot requires scale128")
        return 128.0
    if disposable_fixture_plan is not None:
        return float(disposable_fixture_plan.get("initial_loss_scale", 256.0))
    if phase == "pilot" and plan.get("data", {}).get("schema") == LICENSE_MIXED_HISTORY.data_schema:
        fixture = plan.get("training", {}).get("disposable_fixture", {})
        if not isinstance(fixture, dict) or fixture.get("initial_loss_scale") != 128.0:
            raise ValueError("history pilot requires the frozen 128 initial loss scale")
        return 128.0
    return 256.0


def verify_artifacts(model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    student = config["student"]
    if (
        student["model_id"] != MODEL_ID
        or student["revision"] != MODEL_REVISION
        or student["tokenizer_revision"] != MODEL_REVISION
        or student["initializer"] != "untouched_pretrained"
    ):
        raise ValueError("unapproved student identity or initializer")
    weights = model_dir / "model.safetensors"
    tokenizer = model_dir / "tokenizer.json"
    model_config = model_dir / "config.json"
    for path in (weights, tokenizer, model_config):
        if not path.is_file():
            raise FileNotFoundError(f"required local model artifact missing: {path.name}")
    if sha256_file(weights) != student["weight_sha256"]:
        raise ValueError("untouched pretrained weight SHA-256 mismatch")
    if sha256_file(tokenizer) != student["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")
    architecture = json.loads(model_config.read_text(encoding="utf-8"))
    if architecture.get("model_type") != "qwen2":
        raise ValueError("expected the approved Qwen2 architecture")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_weight_sha256": student["weight_sha256"],
        "tokenizer_sha256": student["tokenizer_sha256"],
        "config_sha256": sha256_file(model_config),
        "model_weight_bytes": weights.stat().st_size,
    }


def load_training_rows(
    path: Path,
    expected_sha256: str,
    *,
    phase: str,
    minimum_main_train: int,
    pilot_schema: str = INSTINCT.data_schema,
    expected_split: str = "train",
    package_root: Path | None = None,
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("training JSONL hash mismatch")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    mixed_license = pilot_schema in {
        LICENSE_MIXED.data_schema,
        LICENSE_MIXED_HISTORY.data_schema,
    }
    if expected_split != "train" and not (phase == "pilot" and expected_split == "development"):
        raise ValueError("only pilot validation can read a nontraining shard")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = (
                _strict_json(line.encode("utf-8"), label="training row")
                if mixed_license
                else json.loads(line)
            )
            if not isinstance(row, dict):
                raise ValueError("training JSONL row must be an object")
            if row.get("split", "train") != expected_split:
                raise ValueError("nontraining split entered the training shard")
            if row.get("source_type", "").startswith("opencode"):
                raise ValueError("OpenCode output cannot be used as a student label")
            if not row.get("source_license"):
                raise ValueError("training row lacks a source license")
            identifier = row.get("id")
            if mixed_license:
                candidate_id = row.get("candidate_id")
                if (
                    identifier is not None
                    and candidate_id is not None
                    and identifier != candidate_id
                ):
                    raise ValueError("LICENSE-MIXED candidate ID aliases disagree")
                identifier = candidate_id if candidate_id is not None else identifier
            if not isinstance(identifier, str) or identifier in ids:
                raise ValueError("training IDs must be unique strings")
            ids.add(identifier)
            state = EditState.from_mapping(row["state"])
            action = EditAction(**row["action"])
            if apply_action(state, action) != row["after_source"]:
                raise ValueError("training action does not reconstruct its after-state")
            if phase == "main" and not row.get("validation", {}).get("inferability_reviewed"):
                raise ValueError("main training requires inferability-reviewed states")
            if phase == "pilot":
                validate_pilot_row(
                    row,
                    policy_for_schema(pilot_schema),
                    package_root=package_root,
                )
            rows.append(row)
    if phase == "main" and len(rows) < minimum_main_train:
        raise ValueError("main training has fewer than 20,000 accepted states")
    if phase.startswith("probe") and len(rows) < 2048:
        raise ValueError("LR probes require the same first 2,048 accepted states")
    if phase == "fixture" and len(rows) < 64:
        raise ValueError("disposable fixture requires 64 states")
    if phase == "pilot" and expected_split == "train" and not 128 <= len(rows) <= 1024:
        raise ValueError("pilot requires 128 to 1,024 distinct edit states")
    if phase == "pilot" and expected_split == "development" and len(rows) < 64:
        raise ValueError("pilot requires at least 64 development states")
    return rows


def phase_rows(rows: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    if phase == "fixture":
        return rows[:64]
    if phase.startswith("probe"):
        return rows[:2048]
    return rows


def peak_learning_rate(phase: str, selection_path: Path | None) -> float:
    if phase == "probe_1e-5":
        return 1e-5
    if phase == "probe_3e-5":
        return 3e-5
    if phase == "fixture":
        return 1e-4
    if phase == "pilot":
        return 1e-5
    if selection_path is None:
        raise ValueError("main training requires a locked development LR selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = float(selection["selected_peak_lr"])
    if selected not in (1e-5, 3e-5, 3e-6) or not selection.get("development_evidence_sha256"):
        raise ValueError("unverified main-run learning-rate selection")
    return selected


def _storage_preflight(output: Path, *, predicted_write_bytes: int, cap_bytes: int) -> None:
    existing = directory_bytes(output)
    if existing + predicted_write_bytes > cap_bytes:
        raise OSError("campaign-owned output would exceed the declared storage cap")
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < predicted_write_bytes + 2 * 1024**3:
        raise OSError("insufficient free space plus 2 GiB headroom for checkpoint")


def _optimizer(model: Any) -> tuple[Any, str]:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        from bitsandbytes.optim import AdamW8bit
    except (ImportError, OSError):
        return torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.01), "torch.AdamW"
    return AdamW8bit(parameters, lr=1e-5, weight_decay=0.01), "bitsandbytes.AdamW8bit"


def _export_inference(
    model: Any,
    tokenizer: Any,
    destination: Path,
    cap_bytes: int,
    output_root: Path,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    import torch

    _storage_preflight(
        output_root,
        predicted_write_bytes=source_identity["model_weight_bytes"] * 2,
        cap_bytes=cap_bytes,
    )
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists() or destination.exists():
        raise FileExistsError("inference export path already exists")
    temporary.mkdir(parents=True)
    state = {
        name: (
            tensor.detach().to(device="cpu", dtype=torch.float16)
            if tensor.is_floating_point()
            else tensor.detach().cpu()
        )
        for name, tensor in model.state_dict().items()
    }
    model.save_pretrained(temporary, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    files = {
        str(path.relative_to(temporary)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file()
    }
    atomic_json(
        temporary / "artifact_manifest.json",
        {"source": source_identity, "files": files, "precision": "F16 export from FP32 masters"},
    )
    os.replace(temporary, destination)
    return {"path": str(destination), "files": files}


def main() -> None:
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--development-data", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--phase", choices=("fixture", "probe_1e-5", "probe_3e-5", "pilot", "main"), required=True
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-minutes", type=float, default=120)
    parser.add_argument("--reserve-minutes", type=float, default=15)
    parser.add_argument("--checkpoint-every-updates", type=int, default=100)
    parser.add_argument("--external-campaign-tokens", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    deadline_monotonic = session_deadline_from_invocation(invocation_started, args.session_minutes)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    disposable_fixture_plan = validate_disposable_fixture_plan(
        plan,
        phase=args.phase,
        epochs=args.epochs,
        microbatch=args.microbatch,
    )
    if config["suite_revision"] != 3 or plan["suite_revision"] != 3:
        raise ValueError("single-line suite revision mismatch")
    if config["context"]["serializer"] != CONTEXT_POLICY_VERSION:
        raise ValueError("context serializer identity changed")
    if args.epochs not in (1, 2) or args.epochs > config["training"]["maximum_epochs"]:
        raise ValueError("at most two frozen training epochs are permitted")
    if args.phase in ("fixture", "probe_1e-5", "probe_3e-5") and args.epochs != 1:
        raise ValueError("LR probes have one fixed pass")
    if args.phase == "pilot" and args.epochs != 1:
        raise ValueError("pilot has one fixed pass")
    if args.microbatch < 1 or args.microbatch > config["training"]["effective_batch_examples"]:
        raise ValueError("invalid microbatch")
    if args.session_minutes <= 0 or args.reserve_minutes < 15:
        raise ValueError("session deadline requires at least 15 minutes for finalization")
    if args.external_campaign_tokens < 0:
        raise ValueError("invalid prior campaign token count")
    pilot_policy = None
    data_manifest: dict[str, Any] | None = None
    pilot_artifact_root: Path | None = None
    development: list[dict[str, Any]] | None = None
    if args.phase == "pilot":
        pilot_policy = policy_for_schema(plan.get("data", {}).get("schema"))
        if (
            plan.get("schema") != pilot_policy.plan_schema
            or plan.get("training", {}).get("phase") != "pilot"
            or plan.get("training", {}).get("epochs") != 1
            or plan.get("training", {}).get("peak_learning_rate")
            not in ((1e-5, 3e-5) if pilot_policy is PUBLIC_SYNTHETIC else (1e-5,))
            or plan.get("training", {}).get("max_nonpadding_input_tokens") != 2_000_000
            or plan.get("budgets", {}).get("max_session_seconds") != 7200
            or plan.get("budgets", {}).get("reserve_seconds") != 1200
            or plan.get("student", {}).get("weight_sha256") != config["student"]["weight_sha256"]
            or plan.get("student", {}).get("tokenizer_sha256")
            != config["student"]["tokenizer_sha256"]
            or args.session_minutes > 120
            or args.reserve_minutes < 20
        ):
            raise ValueError("pilot plan or session budget differs from the frozen contract")
        if pilot_policy is CONSTRUCTIVE:
            fixture_tokens = (
                plan["training"]
                .get("disposable_fixture", {})
                .get("nonpadding_training_input_tokens")
            )
            if type(fixture_tokens) is not int or fixture_tokens <= 0:
                raise ValueError("constructive pilot lacks disposable fixture token accounting")
            validate_aggregate_budget(
                {
                    **plan["budgets"],
                    "prior_training_input_tokens": plan["budgets"]["prior_training_input_tokens"]
                    + fixture_tokens,
                },
                planned_tokens=plan["training"]["planned_nonpadding_input_tokens"],
                session_seconds=plan["budgets"]["max_session_seconds"],
                external_campaign_tokens=args.external_campaign_tokens,
            )
        if pilot_policy in (LICENSE_MIXED, LICENSE_MIXED_HISTORY, PUBLIC_SYNTHETIC):
            validate_aggregate_budget(
                plan["budgets"],
                planned_tokens=plan["training"]["planned_nonpadding_input_tokens"],
                session_seconds=plan["budgets"]["max_session_seconds"],
                external_campaign_tokens=args.external_campaign_tokens,
            )
    if args.phase in ("main", "pilot"):
        if args.data_manifest is None:
            raise ValueError("main/pilot run needs a frozen data manifest")
        manifest_payload = args.data_manifest.read_bytes()
        if args.phase == "pilot" and pilot_policy in (LICENSE_MIXED, LICENSE_MIXED_HISTORY):
            parsed_manifest = _strict_json(manifest_payload, label="data manifest")
        else:
            parsed_manifest = json.loads(manifest_payload)
        if not isinstance(parsed_manifest, dict):
            raise ValueError("data manifest must be a JSON object")
        data_manifest = parsed_manifest
        if args.phase == "pilot" and pilot_policy in (LICENSE_MIXED, LICENSE_MIXED_HISTORY):
            validate_license_mixed_manifest(data_manifest, policy=pilot_policy)
            if (
                plan.get("data", {}).get("manifest_sha256") != sha256_file(args.data_manifest)
                or plan.get("data", {}).get("train_sha256") != args.data_sha256
                or data_manifest.get("train_sha256") != args.data_sha256
            ):
                raise ValueError("LICENSE-MIXED data manifest is not pinned by the pilot plan")
            pilot_artifact_root = license_mixed_artifact_root(
                data_manifest, args.data_manifest.parent
            )
    if args.phase == "pilot" and pilot_policy is PUBLIC_SYNTHETIC:
        if args.data_manifest is None:
            raise ValueError("source/functional pilot needs its manifest")
        data_manifest = json.loads(args.data_manifest.read_text())
        if sha256_file(args.data_manifest) != plan["data"]["manifest_sha256"]:
            raise ValueError("source/functional manifest hash mismatch")
        pilot_artifact_root = args.data_manifest.parent
    source_identity = verify_artifacts(args.model, config)
    if sha256_file(args.config) != plan["config_sha256"]:
        raise ValueError("frozen configuration hash mismatch")
    rows = load_training_rows(
        args.data,
        args.data_sha256,
        phase=args.phase,
        minimum_main_train=config["data"]["minimum_main_train"],
        pilot_schema=plan.get("data", {}).get("schema", INSTINCT.data_schema),
        package_root=pilot_artifact_root,
    )
    if args.phase == "fixture":
        from tinycomplete.one_line.train import disposable_fixture_rows

        if canonical_sha256(rows) != canonical_sha256(disposable_fixture_rows()):
            raise ValueError("fixture rows differ from the disposable codec exercises")
    if args.phase in ("main", "pilot"):
        assert data_manifest is not None
        if args.phase == "main":
            if (
                data_manifest.get("accepted_train", 0) < config["data"]["minimum_main_train"]
                or data_manifest.get("train_sha256") != args.data_sha256
                or not data_manifest.get("split_manifest_sha256")
            ):
                raise ValueError("main data diversity/split manifest gate failed")
        elif args.phase == "pilot":
            assert pilot_policy is not None and args.data_manifest is not None
            if (
                data_manifest.get("schema") != pilot_policy.data_schema
                or data_manifest.get("train_sha256") != args.data_sha256
                or data_manifest.get("train_count") != len(rows)
                or data_manifest.get("dev_count", 0) < 64
                or not data_manifest.get("file_groups_disjoint")
                or plan.get("data", {}).get("train_sha256") != args.data_sha256
                or plan.get("data", {}).get("manifest_sha256") != sha256_file(args.data_manifest)
            ):
                raise ValueError("pilot data manifest gate failed")
        if args.phase == "pilot" and pilot_policy is CONSTRUCTIVE:
            validate_constructive_manifest(data_manifest)
            if args.development_data is None:
                raise ValueError("constructive pilot needs the frozen development shard")
            development = load_training_rows(
                args.development_data,
                plan["data"]["development_sha256"],
                phase="pilot",
                minimum_main_train=config["data"]["minimum_main_train"],
                pilot_schema=pilot_policy.data_schema,
                expected_split="development",
            )
            if data_manifest.get("development_sha256") != plan["data"][
                "development_sha256"
            ] or data_manifest.get("dev_count") != len(development):
                raise ValueError("constructive development shard differs from frozen manifest")
            validate_constructive_splits([*rows, *development])
            validate_constructive_review(
                data_manifest,
                [*rows, *development],
                args.data_manifest.parent / "independent_review.json",
            )
        elif args.phase == "pilot" and pilot_policy in (LICENSE_MIXED, LICENSE_MIXED_HISTORY):
            if args.development_data is None or pilot_artifact_root is None:
                raise ValueError("LICENSE-MIXED pilot needs its frozen development shard")
            development = load_training_rows(
                args.development_data,
                plan["data"]["development_sha256"],
                phase="pilot",
                minimum_main_train=config["data"]["minimum_main_train"],
                pilot_schema=pilot_policy.data_schema,
                expected_split="development",
                package_root=pilot_artifact_root,
            )
            if data_manifest.get("development_sha256") != plan["data"][
                "development_sha256"
            ] or data_manifest.get("dev_count") != len(development):
                raise ValueError("LICENSE-MIXED development shard differs from frozen manifest")
            validate_license_mixed_splits([*rows, *development])
    if args.phase == "pilot" and pilot_policy is PUBLIC_SYNTHETIC:
        from tinycomplete.one_line.public_synthetic_pilot import validate_manifest

        assert data_manifest is not None
        if args.development_data is None or pilot_artifact_root is None:
            raise ValueError("source/functional pilot needs its development data")
        development = load_training_rows(
            args.development_data,
            plan["data"]["development_sha256"],
            phase="pilot",
            minimum_main_train=config["data"]["minimum_main_train"],
            pilot_schema=pilot_policy.data_schema,
            expected_split="development",
            package_root=pilot_artifact_root,
        )
        validate_manifest(data_manifest, [*rows, *development], package_root=pilot_artifact_root)
    chosen_rows = phase_rows(rows, args.phase)
    peak_lr = (
        plan["training"]["peak_learning_rate"]
        if pilot_policy is PUBLIC_SYNTHETIC
        else peak_learning_rate(args.phase, args.selection)
    )
    effective_batch_examples = (
        disposable_fixture_plan["effective_batch_examples"]
        if disposable_fixture_plan is not None
        else config["training"]["effective_batch_examples"]
    )
    output_cap = config["budget"]["maximum_new_persistent_local_research_bytes"]
    _storage_preflight(
        args.output,
        predicted_write_bytes=source_identity["model_weight_bytes"] * 4,
        cap_bytes=output_cap,
    )

    # Tokenizer loading uses only the verified local snapshot and no model stack.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if args.phase == "pilot" and pilot_policy in (LICENSE_MIXED, LICENSE_MIXED_HISTORY):
        assert args.data_manifest is not None and data_manifest is not None
        assert development is not None and pilot_artifact_root is not None
        validate_license_mixed_review(
            data_manifest,
            [*rows, *development],
            tokenizer=tokenizer,
            package_root=pilot_artifact_root,
            policy=pilot_policy,
        )
    encoded: list[EncodedExample] = []
    for row in chosen_rows:
        encoded.append(
            encode_training_row(
                tokenizer,
                EditState.from_mapping(row["state"]),
                EditAction(**row["action"]),
                max_input_tokens=config["context"]["default_input_tokens"],
                max_total_tokens=config["context"]["maximum_total_tokens"],
                max_action_tokens=config["context"]["maximum_response_tokens_including_eos"],
            )
        )
    batches = bucketed_batches(
        encoded,
        epochs=args.epochs,
        effective_batch=effective_batch_examples,
    )
    counts = token_counts(encoded)
    planned_tokens = counts["nonpadding_training_input_tokens"] * args.epochs
    fixture_supervision_audit: dict[str, Any] | None = None
    if disposable_fixture_plan is not None:
        if (
            len(encoded) != disposable_fixture_plan["examples"]
            or len(batches) != disposable_fixture_plan["expected_updates"]
            or planned_tokens != disposable_fixture_plan["nonpadding_training_input_tokens"]
        ):
            raise ValueError("disposable fixture v2 counts differ from the frozen phase plan")
        fixture_supervision_audit = disposable_fixture_supervision_by_action(
            chosen_rows, encoded, tokenizer
        )
        if not fixture_supervision_audit["passed"]:
            raise ValueError("disposable fixture v2 response/EOS supervision audit failed")
    if args.phase == "pilot" and planned_tokens > 2_000_000:
        raise ValueError("pilot planned input exposure exceeds 2,000,000 tokens")
    if args.phase == "pilot" and planned_tokens != plan["training"].get(
        "planned_nonpadding_input_tokens"
    ):
        raise ValueError("pilot tokenizer exposure differs from the frozen plan")
    if (
        args.external_campaign_tokens + planned_tokens
        > config["budget"]["maximum_nonpadding_training_input_tokens"]
    ):
        raise ValueError("planned input exposure exceeds campaign token ceiling")
    identity = {
        "source": source_identity,
        "suite_revision": 3,
        "context_policy": CONTEXT_POLICY_VERSION,
        "config_sha256": sha256_file(args.config),
        "plan_sha256": sha256_file(args.plan),
        "data_sha256": args.data_sha256,
        "order_sha256": batch_order_sha256(batches),
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "src/tinycomplete/one_line/contract.py",
                "src/tinycomplete/one_line/context.py",
                "src/tinycomplete/one_line/train.py",
                "scripts/train_one_line.py",
            )
        },
        "phase": args.phase,
        "peak_lr": peak_lr,
        "epochs": args.epochs,
        "microbatch": args.microbatch,
        "effective_batch": effective_batch_examples,
        "torch": _installed_version("torch"),
        "transformers": _installed_version("transformers"),
        "external_campaign_tokens": args.external_campaign_tokens,
    }
    fingerprint = canonical_sha256(identity)
    summary = {
        "fingerprint": fingerprint,
        "identity": identity,
        "examples": len(encoded),
        "token_counts_one_pass": counts,
        "planned_training_input_tokens": planned_tokens,
        "planned_updates": len(batches),
        "effective_batch_examples": effective_batch_examples,
        "status": "inspected" if not args.execute else "prepared",
    }
    if disposable_fixture_plan is not None:
        identity["disposable_fixture_plan"] = disposable_fixture_plan
        summary["disposable_fixture_preflight"] = {
            "plan_schema": FIXTURE_V2_PLAN_SCHEMA,
            "implementation_viability_schema": FIXTURE_VIABILITY_SCHEMA,
            "supervision_audit": fixture_supervision_audit,
            "implementation_viability_status": "pending_gpu_fixture_execution",
            "quality_evidence": False,
        }
        fingerprint = canonical_sha256(identity)
        summary["fingerprint"] = fingerprint
        summary["identity"] = identity
    if not args.execute:
        print(json.dumps(summary, sort_keys=True))
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA execution requires an explicitly allocated GPU session")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("this worker supports one active training rank only")
    if args.session_minutes <= args.reserve_minutes:
        raise ValueError("session deadline leaves no training time")
    if args.output.exists() and args.resume is None:
        raise FileExistsError("existing output requires an exact-resume checkpoint")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.float()
    if model.config.model_type != "qwen2":
        raise ValueError("model architecture mismatch after load")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    device = torch.device("cuda:0")
    model.to(device)
    # One small, declared Qwen2 final-norm parameter proves an actual update.
    # Sampling occurs only outside the training hot loop.
    fixture_before = (
        model.model.norm.weight.detach().float().cpu().clone() if args.phase == "fixture" else None
    )
    optimizer, optimizer_name = _optimizer(model)
    scheduler = CosineUpdateSchedule(
        optimizer,
        peak_lr=peak_lr,
        total_updates=len(batches),
        warmup_fraction=config["training"]["warmup_fraction"],
        floor_fraction=config["training"]["final_lr_fraction"],
    )
    initial_loss_scale = initial_loss_scale_for_run(
        plan=plan,
        phase=args.phase,
        disposable_fixture_plan=disposable_fixture_plan,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=initial_loss_scale, growth_interval=2000)
    identity["initial_loss_scale"] = initial_loss_scale
    identity["optimizer"] = optimizer_name
    fingerprint = canonical_sha256(identity)
    # Persist the resolved optimizer identity before the first update.
    summary["fingerprint"] = fingerprint
    summary["identity"] = identity
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("resolved optimizer differs from resumable run")
    elif args.resume is not None:
        raise ValueError("resume output has no frozen run manifest")
    else:
        atomic_json(manifest_path, summary)
    cursor = TrainingCursor()
    if args.resume is not None:
        cursor = TrainingCursor(
            **load_resume_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                expected_fingerprint=fingerprint,
            )
        )
    latest_path: Path | None = args.resume
    log_path = args.output / "updates.jsonl"
    last_save_seconds = 0.0

    reserve_seconds = finalization_reserve_seconds(args.reserve_minutes, last_save_seconds)
    training_window_seconds = remaining_training_seconds(
        deadline_monotonic, time.monotonic(), reserve_seconds
    )
    if training_window_seconds <= 0:
        summary.update(
            {
                "status": "deadline_stop_before_training",
                "cursor": asdict(cursor),
                "invocation_elapsed_seconds": time.monotonic() - invocation_started,
                "training_window_seconds_at_start": training_window_seconds,
                "finalization_reserve_seconds": reserve_seconds,
                "latest_checkpoint": str(latest_path) if latest_path else None,
            }
        )
        atomic_json(args.output / "run_result.json", summary)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "cursor": asdict(cursor),
                    "result": str(args.output / "run_result.json"),
                },
                sort_keys=True,
            )
        )
        return

    def save(cursor_at_boundary: TrainingCursor) -> None:
        nonlocal latest_path, last_save_seconds
        started = time.monotonic()
        predicted = max(
            source_identity["model_weight_bytes"] * 4,
            latest_path.stat().st_size if latest_path else 0,
        )
        _storage_preflight(args.output, predicted_write_bytes=predicted, cap_bytes=output_cap)
        checkpoint = args.output / f"resume-step-{cursor_at_boundary.attempted_updates:06d}.pt"
        if checkpoint.exists():
            return
        marker = save_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            fingerprint=fingerprint,
            next_example_index=cursor_at_boundary.next_example_index,
            completed_updates=cursor_at_boundary.completed_updates,
            attempted_updates=cursor_at_boundary.attempted_updates,
            skipped_updates=cursor_at_boundary.skipped_updates,
            training_input_tokens=cursor_at_boundary.training_input_tokens,
            supervised_target_tokens=cursor_at_boundary.supervised_target_tokens,
            epoch=min(args.epochs, cursor_at_boundary.next_example_index // len(encoded)),
        )
        atomic_json(
            args.output / "latest.json",
            {
                "checkpoint": str(checkpoint),
                "sha256": marker["sha256"],
                "cursor": asdict(cursor_at_boundary),
            },
        )
        if (
            latest_path is not None
            and latest_path != checkpoint
            and latest_path.parent == args.output
        ):
            best_path = args.output / "best.json"
            best = json.loads(best_path.read_text()) if best_path.exists() else {}
            if best.get("checkpoint") != str(latest_path):
                latest_path.unlink(missing_ok=True)
                latest_path.with_suffix(latest_path.suffix + ".complete.json").unlink(
                    missing_ok=True
                )
        latest_path = checkpoint
        last_save_seconds = time.monotonic() - started

    def log_update(record: dict[str, int | float | bool]) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    run = train_encoded(
        model,
        encoded,
        batches,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        pad_token_id=tokenizer.eos_token_id,
        cursor=cursor,
        microbatch_examples=args.microbatch,
        max_input_tokens=config["budget"]["maximum_nonpadding_training_input_tokens"],
        external_campaign_tokens=args.external_campaign_tokens,
        deadline_monotonic=deadline_monotonic,
        finalization_reserve_seconds=lambda: finalization_reserve_seconds(
            args.reserve_minutes, last_save_seconds
        ),
        checkpoint_every_updates=args.checkpoint_every_updates,
        on_checkpoint=save,
        on_update=log_update,
    )
    summary.update(
        {
            "status": run.status,
            "cursor": asdict(run.cursor),
            "elapsed_seconds": run.elapsed_seconds,
            "optimizer": optimizer_name,
            "latest_checkpoint": str(latest_path) if latest_path else None,
        }
    )
    if fixture_before is not None:
        fixture_after = model.model.norm.weight.detach().float().cpu()
        delta = (fixture_after - fixture_before).abs()
        generated_observations = []
        model.eval()
        # Four deliberately exposed codec exercises. Actual greedy stopping is
        # observed separately from correctly supervising EOS in the loss.
        diagnostic_rows = (
            disposable_fixture_diagnostic_rows(chosen_rows)
            if disposable_fixture_plan is not None
            else chosen_rows[:4]
        )
        diagnostic_ids = {row["id"] for row in diagnostic_rows}
        for row, example in zip(chosen_rows, encoded, strict=True):
            if row["id"] not in diagnostic_ids:
                continue
            ids = torch.tensor([example.input_ids[: example.prompt_tokens]], device=device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(
                    ids,
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )[0, ids.shape[1] :].tolist()
            terminated = bool(generated and generated[-1] == tokenizer.eos_token_id)
            wire = cast(
                str,
                tokenizer.decode(
                    generated[:-1] if terminated else generated,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            )
            decoded = decode_action(wire, terminated=terminated, generated_tokens=len(generated))
            generated_observations.append(
                {
                    "id": row["id"],
                    "gold_action": row["action"]["kind"],
                    "wire": wire,
                    "generated_tokens": len(generated),
                    "terminated_by_eos": terminated,
                    "valid_action": decoded.action is not None,
                    "exact_action": decoded.action == EditAction(**row["action"]),
                }
            )
        fixture_summary = {
            "quality_evidence": False,
            "generation_quality_status": "four_exposed_training_exercises_not_quality_evidence",
            "greedy_generation_observations": generated_observations,
            "response_and_eos_positions_supervised": all(
                example.labels[: example.prompt_tokens] == (-100,) * example.prompt_tokens
                and example.labels[-1] == tokenizer.eos_token_id
                for example in encoded
            ),
            "observed_parameter": "model.norm.weight",
            "changed_parameter_elements": int((delta > 0).sum().item()),
            "maximum_absolute_parameter_delta": float(delta.max().item()),
            "initial_parameter_sha256": hashlib.sha256(
                fixture_before.numpy().tobytes()
            ).hexdigest(),
            "final_parameter_sha256": hashlib.sha256(fixture_after.numpy().tobytes()).hexdigest(),
        }
        if disposable_fixture_plan is not None:
            training_complete = (
                run.status == "complete"
                and run.cursor.completed_updates == disposable_fixture_plan["expected_updates"]
                and run.cursor.skipped_updates == 0
            )
            viability = disposable_fixture_implementation_viability(
                generated_observations,
                training_complete=training_complete,
            )
            fixture_summary.update(
                {
                    "plan_schema": FIXTURE_V2_PLAN_SCHEMA,
                    "effective_batch_examples": effective_batch_examples,
                    "microbatch_examples": args.microbatch,
                    "planned_updates": len(batches),
                    "supervision_audit": fixture_supervision_audit,
                    "generation_quality_status": (
                        "answer_cued_implementation_viability_only_no_quality_evidence"
                    ),
                    "implementation_viability": viability,
                }
            )
        summary["disposable_fixture"] = fixture_summary
        atomic_json(args.output / "run_result.json", summary)
        if (
            run.status != "complete"
            or run.cursor.completed_updates != len(batches)
            or run.cursor.skipped_updates
            or not summary["disposable_fixture"]["changed_parameter_elements"]
            or not summary["disposable_fixture"]["response_and_eos_positions_supervised"]
            or (
                disposable_fixture_plan is not None
                and not summary["disposable_fixture"]["implementation_viability"]["passed"]
            )
        ):
            raise RuntimeError("disposable fixture failed actual update or supervision checks")
    if run.status == "complete":
        summary["inference_export"] = _export_inference(
            model,
            tokenizer,
            args.output / "inference-f16",
            output_cap,
            args.output,
            source_identity,
        )
    atomic_json(args.output / "run_result.json", summary)
    print(
        json.dumps(
            {
                "status": run.status,
                "cursor": asdict(run.cursor),
                "result": str(args.output / "run_result.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
