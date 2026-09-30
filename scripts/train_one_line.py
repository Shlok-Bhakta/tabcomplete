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
from typing import Any

import yaml

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION
from tinycomplete.one_line.contract import EditAction, EditState, apply_action
from tinycomplete.one_line.pilot_data import (
    CONSTRUCTIVE,
    INSTINCT,
    policy_for_schema,
    validate_aggregate_budget,
    validate_constructive_manifest,
    validate_constructive_review,
    validate_constructive_splits,
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
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("training JSONL hash mismatch")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    if expected_split != "train" and not (phase == "pilot" and expected_split == "development"):
        raise ValueError("only pilot validation can read a nontraining shard")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "train") != expected_split:
                raise ValueError("nontraining split entered the training shard")
            if row.get("source_type", "").startswith("opencode"):
                raise ValueError("OpenCode output cannot be used as a student label")
            if not row.get("source_license"):
                raise ValueError("training row lacks a source license")
            identifier = row.get("id")
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
                validate_pilot_row(row, policy_for_schema(pilot_schema))
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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
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
    if args.phase == "pilot":
        pilot_policy = policy_for_schema(plan.get("data", {}).get("schema"))
        if (
            plan.get("schema") != pilot_policy.plan_schema
            or plan.get("training", {}).get("phase") != "pilot"
            or plan.get("training", {}).get("epochs") != 1
            or plan.get("training", {}).get("peak_learning_rate") != 1e-5
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
    source_identity = verify_artifacts(args.model, config)
    if sha256_file(args.config) != plan["config_sha256"]:
        raise ValueError("frozen configuration hash mismatch")
    rows = load_training_rows(
        args.data,
        args.data_sha256,
        phase=args.phase,
        minimum_main_train=config["data"]["minimum_main_train"],
        pilot_schema=plan.get("data", {}).get("schema", INSTINCT.data_schema),
    )
    if args.phase == "fixture":
        from tinycomplete.one_line.train import disposable_fixture_rows

        if rows != disposable_fixture_rows():
            raise ValueError("fixture rows differ from the disposable codec exercises")
    if args.phase in ("main", "pilot"):
        if args.data_manifest is None:
            raise ValueError("main/pilot run needs a frozen data manifest")
        data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
        if args.phase == "main":
            if (
                data_manifest.get("accepted_train", 0) < config["data"]["minimum_main_train"]
                or data_manifest.get("train_sha256") != args.data_sha256
                or not data_manifest.get("split_manifest_sha256")
            ):
                raise ValueError("main data diversity/split manifest gate failed")
        elif (
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
    chosen_rows = phase_rows(rows, args.phase)
    peak_lr = peak_learning_rate(args.phase, args.selection)
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
        effective_batch=config["training"]["effective_batch_examples"],
    )
    counts = token_counts(encoded)
    planned_tokens = counts["nonpadding_training_input_tokens"] * args.epochs
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
        "effective_batch": config["training"]["effective_batch_examples"],
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
        "status": "inspected" if not args.execute else "prepared",
    }
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
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0, growth_interval=2000)
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

    deadline = time.monotonic() + args.session_minutes * 60
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
        deadline_monotonic=deadline,
        finalization_reserve_seconds=lambda: max(
            args.reserve_minutes * 60, last_save_seconds * 2 + 60
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
        summary["disposable_fixture"] = {
            "quality_evidence": False,
            "generation_quality_status": "not_measured_by_training_mechanics_check",
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
        atomic_json(args.output / "run_result.json", summary)
        if (
            run.status != "complete"
            or run.cursor.completed_updates != len(batches)
            or run.cursor.skipped_updates
            or not summary["disposable_fixture"]["changed_parameter_elements"]
            or not summary["disposable_fixture"]["response_and_eos_positions_supervised"]
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
