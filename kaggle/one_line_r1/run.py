"""Kaggle T4 worker for a single bounded one-line SFT session.

`build_bundle.py` substitutes a frozen JSON session literal before submission.
This worker never fetches model weights; the attached private dataset supplies
the exact approved q25 snapshot. Setup, failures, and checkpoint saving all
count against the session deadline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SESSION_LITERAL = "__SESSION_LITERAL__"
SESSION = json.loads(SESSION_LITERAL)
START = time.monotonic()
HARD_DEADLINE = START + SESSION["session_seconds"]
RESERVE = SESSION["reserve_seconds"]
INPUT_ROOT = Path("/kaggle/input")
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/one_line_r1")
TRAIN_OUT = OUT / "training"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def remaining_before_reserve() -> float:
    return HARD_DEADLINE - time.monotonic() - RESERVE


def stage(command: list[str], label: str, *, cwd: Path | None = None) -> None:
    remaining = remaining_before_reserve()
    if remaining < 60:
        raise TimeoutError("session finalization reserve reached during setup")
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1, remaining - 30),
    )
    # Logs are bounded and private. Do not print pip, Git, source, or model text.
    log = (result.stdout + result.stderr)[-64_000:]
    (OUT / (label + ".log")).write_text(log, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private Kaggle output log")
    print(json.dumps({"stage": label, "seconds": time.monotonic() - started}), flush=True)


def find_approved_input() -> tuple[Path, dict]:
    matches = []
    for candidate in INPUT_ROOT.rglob("input-manifest.json"):
        if sha(candidate) == SESSION["input_manifest_sha256"]:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("expected exactly one attached approved input manifest")
    directory = matches[0].parent
    manifest = load(matches[0])
    if manifest["schema"] != "one-line-kaggle-input-v1":
        raise ValueError("unrecognized dataset schema")
    if manifest["plan_sha256"] != SESSION["plan_sha256"]:
        raise ValueError("attached plan identity changed")
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("attached dataset file identity mismatch")
    if manifest["files"]["train.jsonl"]["sha256"] != SESSION["train_sha256"]:
        raise ValueError("training shard identity changed")
    if manifest["files"]["lr_selection.json"]["sha256"] != SESSION["selection_sha256"]:
        raise ValueError("locked LR selection identity changed")
    return directory, manifest


def find_resume_parent() -> Path | None:
    wanted = SESSION.get("resume")
    if wanted is None:
        return None
    matches = []
    for candidate in INPUT_ROOT.rglob("resume-step-*.pt"):
        if sha(candidate) == wanted["checkpoint_sha256"]:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("expected one attached exact-resume checkpoint")
    parent = matches[0]
    marker = load(parent.with_suffix(parent.suffix + ".complete.json"))
    if marker["sha256"] != wanted["checkpoint_sha256"]:
        raise ValueError("attached resume marker hash mismatch")
    if marker["fingerprint"] != wanted["fingerprint"]:
        raise ValueError("attached resume fingerprint mismatch")
    prior_manifest = load(parent.parent / "run_manifest.json")
    if prior_manifest["fingerprint"] != wanted["fingerprint"]:
        raise ValueError("attached run manifest fingerprint mismatch")
    TRAIN_OUT.mkdir(parents=True, exist_ok=False)
    for filename in (
        parent.name,
        parent.name + ".complete.json",
        "run_manifest.json",
    ):
        shutil.copy2(parent.parent / filename, TRAIN_OUT / filename)
    return TRAIN_OUT / parent.name


def verify_worker_result() -> dict:
    result = load(TRAIN_OUT / "run_result.json")
    latest = load(TRAIN_OUT / "latest.json")
    checkpoint = TRAIN_OUT / Path(latest["checkpoint"]).name
    marker = load(checkpoint.with_suffix(checkpoint.suffix + ".complete.json"))
    digest = sha(checkpoint)
    if digest != latest["sha256"] or digest != marker["sha256"]:
        raise ValueError("saved checkpoint failed post-process hash verification")
    if result["fingerprint"] != marker["fingerprint"]:
        raise ValueError("result and checkpoint identities disagree")
    if result["cursor"] != latest["cursor"]:
        raise ValueError("checkpoint does not cover the reported consumed position")
    return {
        "status": result["status"],
        "fingerprint": result["fingerprint"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": digest,
        "cursor": result["cursor"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "setup",
        "session_seconds_limit": SESSION["session_seconds"],
        "reserve_seconds": RESERVE,
        "plan_sha256": SESSION["plan_sha256"],
        "commit": SESSION["commit"],
    }
    save(OUT / "worker-status.json", status)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        dataset, _ = find_approved_input()
        # The established Kaggle worker path pins short-lived library setup,
        # then clones exactly the already-pushed public research commit.
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
            ],
            "setup",
        )
        stage(
            [
                "git",
                "clone",
                "--branch",
                SESSION["branch"],
                "--single-branch",
                "https://github.com/Shlok-Bhakta/tabcomplete.git",
                str(ROOT),
            ],
            "clone",
        )
        stage(["git", "checkout", SESSION["commit"]], "checkout", cwd=ROOT)
        if sha(ROOT / "reports/research/one_line_r1/plan.json") != SESSION["plan_sha256"]:
            raise ValueError("pushed research plan differs from submitted bundle")
        if sha(ROOT / "configs/research/one_line_r1.yaml") != SESSION["config_sha256"]:
            raise ValueError("pushed research configuration differs from submitted bundle")
        import torch
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("approved T4 CUDA device is unavailable")
        if "logits_to_keep" not in Qwen2ForCausalLM.forward.__code__.co_varnames:
            raise RuntimeError("installed Qwen2 forward lacks selected-position logits support")
        parent = find_resume_parent()
        if remaining_before_reserve() < 60:
            raise TimeoutError("setup consumed the training window")
        status["state"] = "training"
        status["torch"] = torch.__version__
        status["device"] = torch.cuda.get_device_name(0)
        save(OUT / "worker-status.json", status)
        command = [
            sys.executable,
            str(ROOT / "scripts/train_one_line.py"),
            "--config",
            str(ROOT / "configs/research/one_line_r1.yaml"),
            "--plan",
            str(ROOT / "reports/research/one_line_r1/plan.json"),
            "--model",
            str(dataset),
            "--data",
            str(dataset / "train.jsonl"),
            "--data-sha256",
            SESSION["train_sha256"],
            "--data-manifest",
            str(dataset / "data_manifest.json"),
            "--selection",
            str(dataset / "lr_selection.json"),
            "--phase",
            "main",
            "--epochs",
            "1",
            "--output",
            str(TRAIN_OUT),
            "--session-minutes",
            str((HARD_DEADLINE - time.monotonic()) / 60),
            "--reserve-minutes",
            str(RESERVE / 60),
            "--external-campaign-tokens",
            str(SESSION["external_campaign_tokens"]),
            "--checkpoint-every-updates",
            "100",
            "--execute",
        ]
        if parent is not None:
            command.extend(("--resume", str(parent)))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, HARD_DEADLINE - time.monotonic() - 30),
        )
        (OUT / "train.log").write_text((result.stdout + result.stderr)[-128_000:], encoding="utf-8")
        if result.returncode:
            raise RuntimeError("training worker failed; inspect private Kaggle log")
        status.update(verify_worker_result())
        status["training_process_seconds"] = time.monotonic() - started
        status["session_elapsed_seconds"] = time.monotonic() - START
        status["state"] = "verified"
        save(OUT / "worker-status.json", status)
        print(
            json.dumps(
                {
                    "state": "verified",
                    "status": status["status"],
                    "session_elapsed_seconds": status["session_elapsed_seconds"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        status["state"] = "failed"
        status["failure_type"] = type(error).__name__
        status["session_elapsed_seconds"] = time.monotonic() - START
        save(OUT / "worker-status.json", status)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
