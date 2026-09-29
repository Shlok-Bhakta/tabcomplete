"""Private Kaggle T4 worker for the frozen one-line q25 pilot.

The controller prepares and uploads the private input bundle separately. This
worker verifies every input, runs a bounded untouched-model baseline, invokes
the existing pilot-only trainer once, verifies its resumable checkpoint and
F16 export, then evaluates that export on the disjoint development shard.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SESSION_LITERAL = "__SESSION_LITERAL__"
SESSION = json.loads(SESSION_LITERAL)
STARTED = time.monotonic()
HARD_DEADLINE = STARTED + int(SESSION["session_seconds"])
RESERVE_SECONDS = int(SESSION["reserve_seconds"])
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
REPO = WORK_ROOT / "tabcomplete"
OUT = WORK_ROOT / "one_line_gpu_pilot_r1"
TRAIN_OUT = OUT / "training"
MAX_EVAL_SECONDS = 15 * 60
REQUIRED_INPUTS = {
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "train.jsonl",
    "development.jsonl",
    "data_manifest.json",
    "config.yaml",
    "plan.json",
}
OPTIONAL_MODEL_FILES = {
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def remaining_seconds() -> float:
    return HARD_DEADLINE - time.monotonic()


def remaining_before_reserve() -> float:
    return remaining_seconds() - RESERVE_SECONDS


def _run(
    command: list[str],
    *,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1.0, min(timeout, remaining_seconds() - 15)),
    )


def stage(
    command: list[str],
    label: str,
    *,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    finalization: bool = False,
) -> None:
    available = remaining_seconds() if finalization else remaining_before_reserve()
    if available < 90:
        raise TimeoutError("pilot session reached its finalization boundary")
    started = time.monotonic()
    try:
        result = _run(command, timeout=min(timeout, available - 30), cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        status = load(OUT / "worker-status.json")
        status["last_stage"] = {
            "stage": label,
            "state": "timeout",
            "timeout_seconds": min(timeout, available - 30),
            "elapsed_seconds": time.monotonic() - started,
        }
        status["state_before_failure"] = status.get("state")
        save(OUT / "worker-status.json", status)
        raise
    # Kaggle artifacts stay private. Logs may include code or generated text and
    # are deliberately neither streamed nor printed by this worker.
    (OUT / f"{label}.log").write_text((result.stdout + result.stderr)[-96_000:], encoding="utf-8")
    record = {"stage": label, "elapsed_seconds": time.monotonic() - started}
    status = load(OUT / "worker-status.json")
    status.setdefault("stages", []).append(record)
    status["last_stage"] = {
        **record,
        "return_code": result.returncode,
        "state": "complete" if result.returncode == 0 else "failed",
    }
    save(OUT / "worker-status.json", status)
    if result.returncode:
        raise RuntimeError(f"{label} failed; inspect its private Kaggle log")
    print(json.dumps(record, sort_keys=True), flush=True)


def _safe_input_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    matches: list[Path] = []
    for candidate in INPUT_ROOT.rglob("input-manifest.json"):
        if sha(candidate) == SESSION["input_manifest_sha256"]:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("expected exactly one attached manifest matching the frozen input hash")
    manifest_path = matches[0]
    manifest = load(manifest_path)
    if manifest.get("schema") != "one-line-instinct-pilot-input-v1":
        raise ValueError("unrecognized pilot input schema")
    if manifest.get("plan_sha256") != SESSION["plan_sha256"]:
        raise ValueError("attached plan identity differs from the frozen session")
    branch, commit = manifest.get("branch"), manifest.get("commit")
    if branch != SESSION["branch"] or commit != SESSION["commit"]:
        raise ValueError("input bundle was not built from the frozen pushed commit")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("input manifest has no file identities")
    names = set(files)
    expected_names = REQUIRED_INPUTS | OPTIONAL_MODEL_FILES
    if not REQUIRED_INPUTS <= names or not names <= expected_names:
        raise ValueError("input package contains missing or unapproved files")
    for candidate in manifest_path.parent.rglob("*"):
        if (
            candidate.is_file()
            and candidate.suffix.casefold()
            in {
                ".safetensors",
                ".bin",
                ".pt",
                ".pth",
                ".gguf",
            }
            and candidate.name != "model.safetensors"
        ):
            raise ValueError("input package contains an unapproved model or training artifact")
    for name, identity in files.items():
        file_path = manifest_path.parent / name
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError("an approved input file is missing or is a symlink")
        if file_path.stat().st_size != identity.get("bytes") or sha(file_path) != identity.get(
            "sha256"
        ):
            raise ValueError("attached input file hash or size mismatch")
    data_manifest = load(manifest_path.parent / "data_manifest.json")
    plan = load(manifest_path.parent / "plan.json")
    if data_manifest.get("schema") != "one-line-instinct-pilot-v1":
        raise ValueError("training data manifest schema mismatch")
    if (
        data_manifest.get("dataset_id") != "continuedev/instinct-data"
        or data_manifest.get("dataset_revision") != SESSION["dataset_revision"]
        or data_manifest.get("dataset_license") != SESSION["dataset_license"]
        or data_manifest.get("source_file_license_status") != "unverified"
        or data_manifest.get("train_count") != SESSION["train_count"]
        or data_manifest.get("dev_count") != SESSION["development_count"]
        or data_manifest.get("file_groups_disjoint") is not True
    ):
        raise ValueError("pilot data provenance or split metadata differs from the frozen plan")
    for key in ("train_sha256", "development_sha256"):
        expected = SESSION.get(key)
        if expected is not None and data_manifest.get(key) != expected:
            raise ValueError("training data manifest differs from the frozen session")
    if sha(manifest_path.parent / "data_manifest.json") != SESSION["manifest_sha256"]:
        raise ValueError("training data manifest hash differs from the frozen session")
    data_plan = plan.get("data", {})
    student = plan.get("student", {})
    training = plan.get("training", {})
    budgets = plan.get("budgets", {})
    if (
        plan.get("schema") != "one-line-instinct-pilot-plan-v1"
        or plan.get("branch") != SESSION["branch"]
        or plan.get("base_commit") != SESSION["base_commit"]
        or plan.get("suite_revision") != 3
        or plan.get("config_sha256") != SESSION["config_sha256"]
        or student.get("model_id") != SESSION["model_id"]
        or student.get("revision") != SESSION["model_revision"]
        or student.get("weight_sha256") != SESSION["model_weight_sha256"]
        or student.get("tokenizer_sha256") != SESSION["tokenizer_sha256"]
        or student.get("config_sha256") != SESSION["model_config_sha256"]
        or data_plan.get("dataset_id") != "continuedev/instinct-data"
        or data_plan.get("dataset_revision") != SESSION["dataset_revision"]
        or data_plan.get("dataset_license") != SESSION["dataset_license"]
        or data_plan.get("source_file_license_status") != "unverified"
        or data_plan.get("train_count") != SESSION["train_count"]
        or data_plan.get("dev_count") != SESSION["development_count"]
        or data_plan.get("file_groups_disjoint") is not True
        or training.get("phase") != "pilot"
        or training.get("epochs") != 1
        or training.get("max_nonpadding_input_tokens") != 2_000_000
        or training.get("planned_nonpadding_input_tokens", 2_000_001) > 2_000_000
        or budgets.get("max_session_seconds") != SESSION["session_seconds"]
        or budgets.get("reserve_seconds") != SESSION["reserve_seconds"]
        or budgets.get("max_new_storage_bytes") != 12 * 1024**3
        or budgets.get("quota_gpu_hours_multiplier") != 2
        or budgets.get("no_automatic_renewal") is not True
    ):
        raise ValueError("frozen pilot plan violates the model, data, or budget contract")
    if sha(manifest_path.parent / "train.jsonl") != SESSION["train_sha256"]:
        raise ValueError("training shard differs from the frozen session")
    if sha(manifest_path.parent / "development.jsonl") != SESSION["development_sha256"]:
        raise ValueError("development shard differs from the frozen session")
    if sha(manifest_path.parent / "plan.json") != SESSION["plan_sha256"]:
        raise ValueError("frozen plan hash mismatch")
    if sha(manifest_path.parent / "config.yaml") != SESSION["config_sha256"]:
        raise ValueError("frozen config hash mismatch")
    if files["model.safetensors"]["sha256"] != SESSION["model_weight_sha256"]:
        raise ValueError("input manifest does not pin the approved model weights")
    if files["tokenizer.json"]["sha256"] != SESSION["tokenizer_sha256"]:
        raise ValueError("input manifest does not pin the approved tokenizer")
    return manifest_path.parent, manifest


def _verify_model(directory: Path) -> None:
    config = load(directory / "config.json")
    if (
        config.get("model_type") != "qwen2"
        or sha(directory / "config.json") != SESSION["model_config_sha256"]
    ):
        raise ValueError("pilot model config identity or architecture mismatch")
    if sha(directory / "model.safetensors") != SESSION["model_weight_sha256"]:
        raise ValueError("untouched q25 model weight hash mismatch")
    if sha(directory / "tokenizer.json") != SESSION["tokenizer_sha256"]:
        raise ValueError("pinned q25 tokenizer hash mismatch")


def _clone_frozen_commit() -> None:
    if REPO.exists():
        raise FileExistsError("worker repository destination already exists")
    stage(
        [
            "git",
            "clone",
            "--branch",
            SESSION["branch"],
            "--single-branch",
            "--no-tags",
            "https://github.com/Shlok-Bhakta/tabcomplete.git",
            str(REPO),
        ],
        "clone",
        timeout=300,
    )
    stage(["git", "checkout", SESSION["commit"]], "checkout", cwd=REPO, timeout=120)
    actual = _run(["git", "rev-parse", "HEAD"], timeout=30, cwd=REPO)
    if actual.returncode or actual.stdout.strip() != SESSION["commit"]:
        raise ValueError("checked-out worker source commit differs from the frozen session")
    ancestry = _run(
        ["git", "merge-base", "--is-ancestor", SESSION["base_commit"], "HEAD"],
        timeout=30,
        cwd=REPO,
    )
    if ancestry.returncode:
        raise ValueError("pilot commit does not descend from its frozen base")
    if sha(REPO / "reports/research/one_line_gpu_pilot_r1/plan.json") != SESSION["plan_sha256"]:
        raise ValueError("pushed pilot plan does not match the attached frozen plan")
    if sha(REPO / "configs/research/one_line_r1.yaml") != SESSION["config_sha256"]:
        raise ValueError("pushed training config differs from the attached frozen config")


def _check_t4_and_logits_support() -> dict[str, str]:
    import torch
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("an explicitly allocated CUDA device is required")
    device_name = torch.cuda.get_device_name(0)
    if "t4" not in device_name.casefold():
        raise RuntimeError("pilot allocation is not the expected Nvidia T4")
    signature = inspect.signature(Qwen2ForCausalLM.forward)
    if "logits_to_keep" not in signature.parameters:
        raise RuntimeError("installed Qwen2 forward lacks logits_to_keep support")
    # Exercise the exact selected-position API on a disposable tiny Qwen2 model
    # before loading the candidate or paying for a training step.
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    smoke = Qwen2ForCausalLM(config).to("cuda:0").eval()  # type: ignore[arg-type]
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cuda:0")
    positions = torch.tensor([2], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        result = smoke(input_ids=ids, use_cache=False, logits_to_keep=positions)
    if tuple(result.logits.shape) != (1, 1, config.vocab_size):
        raise RuntimeError("Qwen2 selected-logit compatibility smoke returned an invalid shape")
    del result, smoke
    torch.cuda.empty_cache()
    return {"device": device_name, "torch": str(torch.__version__)}


def _offline_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONPATH": str(REPO / "src") + os.pathsep + str(REPO / "scripts"),
        }
    )
    return env


def _evaluation_command(
    *, model: Path, weight_hash: str, source_hash: str, dataset: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        str(REPO / "scripts/evaluate_one_line_gpu_pilot.py"),
        "--model",
        str(model),
        "--model-weight-sha256",
        weight_hash,
        "--source-weight-sha256",
        source_hash,
        "--development",
        str(dataset / "development.jsonl"),
        "--development-sha256",
        SESSION["development_sha256"],
        "--calibration-fixture",
        str(REPO / "reports/research/one_line_r1/calibration_cases.json"),
        "--calibration-manifest",
        str(REPO / "reports/research/one_line_r1/calibration_manifest.json"),
        "--output",
        str(output),
    ]


def _verify_training_output() -> tuple[dict[str, Any], Path]:
    result = load(TRAIN_OUT / "run_result.json")
    latest = load(TRAIN_OUT / "latest.json")
    checkpoint = TRAIN_OUT / Path(latest["checkpoint"]).name
    marker = load(checkpoint.with_suffix(checkpoint.suffix + ".complete.json"))
    digest = sha(checkpoint)
    if digest != latest.get("sha256") or digest != marker.get("sha256"):
        raise ValueError("resumable training checkpoint hash verification failed")
    if result.get("fingerprint") != marker.get("fingerprint"):
        raise ValueError("training and checkpoint fingerprints differ")
    if result.get("cursor") != latest.get("cursor"):
        raise ValueError("latest checkpoint does not cover the reported cursor")
    if result.get("identity", {}).get("phase") != "pilot":
        raise ValueError("trainer result is not from the pilot phase")
    if (
        result.get("identity", {}).get("source", {}).get("model_weight_sha256")
        != SESSION["model_weight_sha256"]
    ):
        raise ValueError("trainer initialized from a different model")
    return result, checkpoint


def _output_bytes() -> int:
    return sum(
        path.stat().st_size for path in OUT.rglob("*") if path.is_file() and not path.is_symlink()
    )


def _verify_evaluation(path: Path, *, model_sha: str) -> dict[str, Any]:
    result = load(path)
    identity = result.get("identity", {})
    if (
        identity.get("model_weight_sha256") != model_sha
        or identity.get("source_weight_sha256") != SESSION["model_weight_sha256"]
        or identity.get("development_sha256") != SESSION["development_sha256"]
        or identity.get("tokenizer_sha256") != SESSION["tokenizer_sha256"]
        or result.get("cases") != SESSION["development_count"]
        or "synthetic_calibration" not in result
    ):
        raise ValueError("pilot evaluation identity or case count mismatch")
    return {
        "cases": result["cases"],
        "valid_actions": result.get("valid_actions"),
        "explicit_termination": result.get("explicit_termination"),
        "exact_after": result.get("exact_after"),
        "synthetic_calibration": result.get("synthetic_calibration"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    status: dict[str, Any] = {
        "schema": "one-line-instinct-pilot-worker-status-v1",
        "state": "starting",
        "training_status": None,
        "plan_sha256": SESSION["plan_sha256"],
        "commit": SESSION["commit"],
        "base_commit": SESSION["base_commit"],
        "development_sha256": SESSION["development_sha256"],
        "development_count": SESSION["development_count"],
        "source_weight_sha256": SESSION["model_weight_sha256"],
        "model_weight_sha256": SESSION["model_weight_sha256"],
        "tokenizer_sha256": SESSION["tokenizer_sha256"],
        "model_config_sha256": SESSION["model_config_sha256"],
        "session_limit_seconds": SESSION["session_seconds"],
        "reserve_seconds": RESERVE_SECONDS,
        "phase": "pilot",
        "epochs": 1,
        "stages": [],
    }
    save(OUT / "worker-status.json", status)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    try:
        dataset, input_manifest = _safe_input_manifest(INPUT_ROOT)
        _verify_model(dataset)
        status["input_manifest_sha256"] = SESSION["input_manifest_sha256"]
        status["input_file_count"] = len(input_manifest["files"])
        save(OUT / "worker-status.json", status)

        stage(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "-q",
                "transformers==5.17.0",
                "bitsandbytes==0.50.2",
                "PyYAML==6.0.2",
                "tree-sitter-language-pack==1.20.0",
            ],
            "setup",
            timeout=8 * 60,
        )
        _clone_frozen_commit()
        env = _offline_environment()
        runtime = _check_t4_and_logits_support()
        status.update(runtime)
        status["state"] = "baseline_evaluation"
        save(OUT / "worker-status.json", status)
        stage(
            _evaluation_command(
                model=dataset,
                weight_hash=SESSION["model_weight_sha256"],
                source_hash=SESSION["model_weight_sha256"],
                dataset=dataset,
                output=OUT / "baseline-evaluation.json",
            ),
            "baseline-evaluation",
            timeout=MAX_EVAL_SECONDS,
            cwd=REPO,
            env=env,
        )
        status["baseline"] = _verify_evaluation(
            OUT / "baseline-evaluation.json", model_sha=SESSION["model_weight_sha256"]
        )
        save(OUT / "worker-status.json", status)

        if remaining_before_reserve() < 5 * 60:
            raise TimeoutError("baseline left too little training time before finalization reserve")
        status["state"] = "training"
        save(OUT / "worker-status.json", status)
        command = [
            sys.executable,
            str(REPO / "scripts/train_one_line.py"),
            "--config",
            str(dataset / "config.yaml"),
            "--plan",
            str(dataset / "plan.json"),
            "--model",
            str(dataset),
            "--data",
            str(dataset / "train.jsonl"),
            "--data-sha256",
            SESSION["train_sha256"],
            "--data-manifest",
            str(dataset / "data_manifest.json"),
            "--phase",
            "pilot",
            "--epochs",
            "1",
            "--output",
            str(TRAIN_OUT),
            "--session-minutes",
            str(max(1.0, remaining_seconds() / 60)),
            "--reserve-minutes",
            str(RESERVE_SECONDS / 60),
            "--checkpoint-every-updates",
            "25",
            "--execute",
        ]
        stage(
            command,
            "training",
            timeout=remaining_seconds() - RESERVE_SECONDS,
            cwd=REPO,
            env=env,
        )
        result, checkpoint = _verify_training_output()
        status["training_status"] = result["status"]
        status["training_result_fingerprint"] = result["fingerprint"]
        status["checkpoint_sha256"] = sha(checkpoint)
        status["training_input_tokens"] = result.get("cursor", {}).get("training_input_tokens")
        status["session_elapsed_seconds"] = time.monotonic() - STARTED

        if result["status"] == "deadline_stop":
            status["output_bytes"] = _output_bytes()
            if status["output_bytes"] > 12 * 1024**3:
                raise OSError("pilot output exceeded the 12 GiB campaign storage cap")
            status["state"] = "partial_checkpoint_preserved"
            status["quality_evaluation_status"] = "not_run_after_partial_training"
            save(OUT / "worker-status.json", status)
            print(
                json.dumps(
                    {"state": status["state"], "checkpoint_sha256": status["checkpoint_sha256"]},
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        if result["status"] != "complete":
            raise ValueError("pilot trainer returned an unsupported terminal status")

        export = result.get("inference_export")
        if not isinstance(export, dict):
            raise ValueError("complete pilot training did not produce an inference export")
        export_dir = Path(export["path"])
        if not export_dir.is_absolute() or not export_dir.exists():
            export_dir = TRAIN_OUT / "inference-f16"
        export_manifest = load(export_dir / "artifact_manifest.json")
        if (
            export_manifest.get("source", {}).get("model_weight_sha256")
            != SESSION["model_weight_sha256"]
        ):
            raise ValueError("inference export source identity mismatch")
        export_sha = sha(export_dir / "model.safetensors")
        status["export_weight_sha256"] = export_sha
        status["state"] = "adapted_evaluation"
        save(OUT / "worker-status.json", status)
        stage(
            _evaluation_command(
                model=export_dir,
                weight_hash=export_sha,
                source_hash=SESSION["model_weight_sha256"],
                dataset=dataset,
                output=OUT / "adapted-evaluation.json",
            ),
            "adapted-evaluation",
            timeout=MAX_EVAL_SECONDS,
            cwd=REPO,
            env=env,
            finalization=True,
        )
        status["adapted"] = _verify_evaluation(
            OUT / "adapted-evaluation.json", model_sha=export_sha
        )
        status["output_bytes"] = _output_bytes()
        if status["output_bytes"] > 12 * 1024**3:
            raise OSError("pilot output exceeded the 12 GiB campaign storage cap")
        status["state"] = "verified_complete"
        status["quality_evaluation_status"] = "complete"
        status["session_elapsed_seconds"] = time.monotonic() - STARTED
        save(OUT / "worker-status.json", status)
        print(
            json.dumps(
                {
                    "state": status["state"],
                    "training_status": status["training_status"],
                    "checkpoint_sha256": status["checkpoint_sha256"],
                    "export_weight_sha256": export_sha,
                    "elapsed_seconds": status["session_elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        # Error messages from commands can contain source or credentials. Keep
        # only a type code and stage in public notebook logs; details remain in
        # private bounded stage logs where available.
        status["failure_stage"] = status.get("state_before_failure") or status.get("state")
        status["state"] = "failed"
        status["failure_type"] = type(error).__name__
        status["session_elapsed_seconds"] = time.monotonic() - STARTED
        save(OUT / "worker-status.json", status)
        print(json.dumps({"state": "failed", "failure_type": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
