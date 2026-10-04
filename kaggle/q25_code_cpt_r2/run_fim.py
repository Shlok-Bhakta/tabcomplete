"""Bounded Kaggle worker for one frozen Qwen2.5 FIM arm."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SESSION = __SESSION_JSON__  # type: ignore[name-defined]  # noqa: F821
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
OUT = WORK_ROOT / "q25_fim_r2"
REPO = Path("/kaggle/temp/tabcomplete-q25-code-cpt-r2")
MAX_SESSION_SECONDS = 10_800
MAX_CAMPAIGN_TOKENS = 32_000_000
MAX_ARM_TOKENS = 4_194_304
MAX_ARTIFACT_BYTES = 12 * 1024**3
MINIMUM_FREE_BYTES = 2 * 1024**3
MINIMUM_FINAL_RESERVE_SECONDS = 1_800
TRAIN_ARM = "untouched_q25_to_fim"
CPT_ARM = "completed_cpt_q25_to_fim"
ARMS = (TRAIN_ARM, CPT_ARM)
REQUIRED_INPUT_FILES = {
    "plan.json",
    "train.jsonl",
    "development.jsonl",
    "corpus_metadata.json",
    "causal200.jsonl",
    "line180.jsonl",
}
ALLOWED_INPUT_EXTRAS = {"input-manifest.json", "dataset-metadata.json"}
CHECKPOINT_CURSOR_FIELDS = (
    "next_example_index",
    "completed_updates",
    "attempted_updates",
    "skipped_updates",
    "training_input_tokens",
    "supervised_target_tokens",
    "epoch",
)


class WorkerError(RuntimeError):
    """A sanitized worker failure with a stable reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise WorkerError("file_read_failed") from None
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise WorkerError(reason) from None
    if not isinstance(value, dict):
        raise WorkerError(reason)
    return value


def validate_session(session: Any) -> dict[str, Any]:
    if not isinstance(session, dict):
        raise WorkerError("session_invalid")
    required = {
        "commit",
        "plan_sha256",
        "input_manifest_sha256",
        "arm",
        "attempt",
        "session_seconds",
        "resume_source",
        "external_campaign_tokens",
    }
    if not required.issubset(session):
        raise WorkerError("session_missing_required_fields")
    commit = session["commit"]
    seconds = session["session_seconds"]
    external_tokens = session["external_campaign_tokens"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit.lower())
        or not _is_sha256(session["plan_sha256"])
        or not _is_sha256(session["input_manifest_sha256"])
        or session["arm"] not in ARMS
        or isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or not MINIMUM_FINAL_RESERVE_SECONDS < seconds <= MAX_SESSION_SECONDS
        or isinstance(external_tokens, bool)
        or not isinstance(external_tokens, int)
        or not 0 <= external_tokens <= MAX_CAMPAIGN_TOKENS
    ):
        raise WorkerError("session_identity_or_budget_invalid")
    attempt = session["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise WorkerError("session_identity_or_budget_invalid")
    resume_source = session["resume_source"]
    resume_identity = session.get("resume_identity")
    if resume_source is None:
        if resume_identity is not None:
            raise WorkerError("resume_identity_without_source")
    else:
        if not isinstance(resume_source, str) or not resume_source.strip():
            raise WorkerError("resume_source_invalid")
        if not isinstance(resume_identity, dict):
            raise WorkerError("resume_identity_missing")
        cursor = resume_identity.get("cursor")
        if (
            not _is_sha256(resume_identity.get("fingerprint"))
            or not _is_sha256(resume_identity.get("checkpoint_sha256"))
            or not isinstance(cursor, dict)
            or any(
                isinstance(cursor.get(field), bool)
                or not isinstance(cursor.get(field), int)
                or cursor[field] < 0
                for field in CHECKPOINT_CURSOR_FIELDS
            )
        ):
            raise WorkerError("resume_identity_invalid")
    return session


def _safe_input_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
        raise WorkerError("input_manifest_path_invalid")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise WorkerError("input_manifest_file_missing_or_unsafe")
    return path


def find_input_manifest(input_root: Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    try:
        matches = [
            path
            for path in input_root.rglob("input-manifest.json")
            if path.is_file() and not path.is_symlink() and sha256_file(path) == expected_sha256
        ]
    except OSError:
        raise WorkerError("input_manifest_search_failed") from None
    if len(matches) != 1:
        raise WorkerError("frozen_input_bundle_missing_or_ambiguous")
    manifest_path = matches[0]
    manifest = _read_json(manifest_path, "input_manifest_invalid")
    return manifest_path.parent, manifest


def verify_input_bundle(
    root: Path, manifest: dict[str, Any], session: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_INPUT_FILES:
        raise WorkerError("input_manifest_file_set_invalid")
    paths: dict[str, Path] = {}
    for name, identity in files.items():
        if not isinstance(identity, dict):
            raise WorkerError("input_manifest_file_identity_invalid")
        path = _safe_input_file(root, name)
        expected_bytes = identity.get("bytes")
        expected_sha = identity.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not _is_sha256(expected_sha)
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
        ):
            raise WorkerError("input_file_hash_or_size_mismatch")
        paths[name] = path

    allowed = REQUIRED_INPUT_FILES | ALLOWED_INPUT_EXTRAS
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WorkerError("input_bundle_contains_symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in allowed:
            raise WorkerError("input_bundle_contains_unlisted_file")

    if sha256_file(paths["plan.json"]) != session["plan_sha256"]:
        raise WorkerError("frozen_plan_hash_mismatch")
    plan = _read_json(paths["plan.json"], "frozen_plan_invalid")
    verify_plan(plan, session, files)
    metadata = _read_json(paths["corpus_metadata.json"], "fim_corpus_metadata_invalid")
    if (
        metadata.get("raw_source_content_emitted") is not False
        or metadata.get("sealed_test_accessed") is not False
        or metadata.get("preparation_plan_sha256") != plan.get("preparation_plan_sha256")
        or metadata.get("parent_cpt_plan_sha256") != plan.get("parent_cpt_plan_sha256")
        or metadata.get("tokenizer_id") != "Qwen/Qwen2.5-Coder-0.5B"
    ):
        raise WorkerError("fim_corpus_metadata_policy_mismatch")
    return paths, plan


def verify_plan(plan: dict[str, Any], session: dict[str, Any], files: dict[str, Any]) -> None:
    if plan.get("schema") != "q25-fim-training-plan-v1":
        raise WorkerError("frozen_fim_plan_schema_invalid")
    if plan.get("gpu_execution_authorized") is not True:
        raise WorkerError("frozen_fim_plan_not_authorized")
    configuration = plan.get("configuration")
    if not isinstance(configuration, dict):
        raise WorkerError("frozen_fim_plan_incomplete")
    budget = configuration.get("budget")
    training = configuration.get("training")
    data = plan.get("data")
    initializers = plan.get("initializers")
    evaluation = plan.get("evaluation")
    if not all(
        isinstance(value, dict) for value in (budget, training, data, initializers, evaluation)
    ):
        raise WorkerError("frozen_fim_plan_incomplete")
    assert isinstance(budget, dict)
    assert isinstance(training, dict)
    assert isinstance(data, dict)
    assert isinstance(initializers, dict)
    assert isinstance(evaluation, dict)
    max_campaign = budget.get(
        "maximum_campaign_input_tokens", budget.get("maximum_additional_training_input_tokens")
    )
    session_limit = budget.get("session_seconds")
    storage_cap = budget.get("new_artifact_bytes_cap")
    arm_limit = training.get("max_input_tokens")
    if (
        isinstance(max_campaign, bool)
        or not isinstance(max_campaign, int)
        or not 1 <= max_campaign <= MAX_CAMPAIGN_TOKENS
        or isinstance(session_limit, bool)
        or not isinstance(session_limit, int)
        or not MINIMUM_FINAL_RESERVE_SECONDS < session["session_seconds"] <= session_limit
        or isinstance(storage_cap, bool)
        or not isinstance(storage_cap, int)
        or not 1 <= storage_cap <= MAX_ARTIFACT_BYTES
        or isinstance(arm_limit, bool)
        or not isinstance(arm_limit, int)
        or not 1 <= arm_limit <= MAX_ARM_TOKENS
        or budget.get("paid_compute") is not False
        or budget.get("automatic_renewal_use") is not False
        or session["external_campaign_tokens"] > max_campaign
    ):
        raise WorkerError("frozen_fim_budget_invalid")
    if not isinstance(initializers.get(session["arm"]), dict):
        raise WorkerError("selected_fim_initializer_missing")
    data_metadata = data.get("corpus_metadata_sha256")
    if not _is_sha256(data_metadata) or data_metadata != files["corpus_metadata.json"]["sha256"]:
        raise WorkerError("frozen_fim_metadata_identity_mismatch")
    for split, filename in (("train", "train.jsonl"), ("development", "development.jsonl")):
        record = data.get(split)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != files[filename]["sha256"]
            or record.get("bytes") != files[filename]["bytes"]
        ):
            raise WorkerError("frozen_fim_split_identity_mismatch")
    fixtures = evaluation.get("fixtures")
    if not isinstance(fixtures, dict):
        raise WorkerError("frozen_fim_fixtures_missing")
    for key, filename in (("causal", "causal200.jsonl"), ("line", "line180.jsonl")):
        record = fixtures.get(key)
        if not isinstance(record, dict) or record.get("sha256") != files[filename]["sha256"]:
            raise WorkerError("frozen_fim_fixture_identity_mismatch")


def find_initializer_path(input_root: Path, entry: dict[str, Any]) -> Path:
    kind = entry.get("kind")
    if kind == "untouched_pretrained":
        file_records = entry.get("files")
        weight_record = (
            file_records.get("model.safetensors") if isinstance(file_records, dict) else None
        )
        expected_sha: Any
        expected_bytes: Any
        if isinstance(weight_record, str):
            expected_sha = weight_record
            expected_bytes = None
        elif isinstance(weight_record, dict):
            expected_sha = weight_record.get("sha256")
            expected_bytes = weight_record.get("bytes")
        else:
            raise WorkerError("base_initializer_weight_identity_missing")
        if not _is_sha256(expected_sha):
            raise WorkerError("base_initializer_weight_hash_invalid")
        matches: list[Path] = []
        for path in input_root.rglob("model.safetensors"):
            if path.is_symlink() or not path.is_file():
                continue
            if expected_bytes is not None and path.stat().st_size != expected_bytes:
                continue
            if sha256_file(path) == expected_sha:
                matches.append(path.parent)
        if len(matches) != 1:
            raise WorkerError("base_initializer_missing_or_ambiguous")
        return matches[0]

    expected_manifest_sha = entry.get("artifact_manifest_sha256")
    if kind != "completed_cpt_export" or not _is_sha256(expected_manifest_sha):
        raise WorkerError("cpt_initializer_identity_invalid")
    matches = []
    for path in input_root.rglob("artifact_manifest.json"):
        if path.is_symlink() or not path.is_file():
            continue
        if sha256_file(path) == expected_manifest_sha:
            matches.append(path.parent)
    if len(matches) != 1:
        raise WorkerError("cpt_initializer_missing_or_ambiguous")
    return matches[0]


def _tree_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WorkerError("artifact_tree_contains_symlink")
        if path.is_file():
            paths.append(path)
    return paths


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in _tree_files(root))


def mounted_artifact_bytes(
    input_root: Path,
    *,
    initializer: Path,
    trainer_files: list[Path],
    resume_checkpoint: Path | None,
) -> int:
    counted = {path.resolve() for path in trainer_files}
    counted.update(path.resolve() for path in _tree_files(initializer))
    if resume_checkpoint is not None:
        marker = resume_checkpoint.with_suffix(resume_checkpoint.suffix + ".complete.json")
        counted.add(resume_checkpoint.resolve())
        counted.add(marker.resolve())
    total = 0
    for path in _tree_files(input_root):
        if path.resolve() not in counted:
            total += path.stat().st_size
    return total


def _verify_baseline(
    previous_root: Path,
    *,
    arm: str,
    plan_sha256: str,
    input_manifest_sha256: str,
    initializer_sha256: str,
) -> dict[str, Any]:
    if previous_root.is_symlink() or not previous_root.is_dir():
        raise WorkerError("resume_baseline_root_unsafe")
    baseline_dir = previous_root / "baseline"
    identity_path = previous_root / "baseline" / "identity.json"
    if baseline_dir.is_symlink() or identity_path.is_symlink() or not identity_path.is_file():
        raise WorkerError("resume_baseline_identity_missing")
    identity = _read_json(identity_path, "resume_baseline_identity_missing")
    if (
        identity.get("schema") != "q25-fim-baseline-v1"
        or identity.get("arm") != arm
        or identity.get("plan_sha256") != plan_sha256
        or identity.get("input_manifest_sha256") != input_manifest_sha256
        or identity.get("initializer_identity_sha256") != initializer_sha256
    ):
        raise WorkerError("resume_baseline_identity_mismatch")
    records = identity.get("files")
    if not isinstance(records, dict) or not records or not _has_baseline_modes(records):
        raise WorkerError("resume_baseline_files_missing")
    discovered: set[str] = set()
    before_root = previous_root / "before"
    if before_root.is_symlink() or not before_root.is_dir():
        raise WorkerError("resume_baseline_inventory_mismatch")
    for path in _tree_files(before_root):
        discovered.add(path.relative_to(previous_root).as_posix())
    if discovered != set(records):
        raise WorkerError("resume_baseline_inventory_mismatch")
    for name, record in records.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, dict)
            or not _is_sha256(record.get("sha256"))
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
        ):
            raise WorkerError("resume_baseline_file_identity_invalid")
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "before"
        ):
            raise WorkerError("resume_baseline_file_path_invalid")
        path = previous_root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
            raise WorkerError("resume_baseline_file_missing_or_changed")
        if sha256_file(path) != record["sha256"]:
            raise WorkerError("resume_baseline_file_missing_or_changed")
    return identity


def _has_baseline_modes(records: dict[str, Any]) -> bool:
    names = [name for name in records if isinstance(name, str)]
    return all(
        any(name.startswith(f"before/{mode}/") for name in names)
        for mode in ("development", "line")
    )


def resolve_verified_resume(
    input_root: Path,
    session: dict[str, Any],
    *,
    plan_sha256: str,
    input_manifest_sha256: str,
    initializer_sha256: str,
    arm: str,
) -> tuple[Path, Path, dict[str, Any]]:
    pointers = [
        path
        for path in input_root.rglob("latest.json")
        if path.is_file() and not path.is_symlink() and path.parent.name == "training"
    ]
    if len(pointers) != 1:
        raise WorkerError("resume_checkpoint_pointer_missing_or_ambiguous")
    pointer_path = pointers[0]
    pointer = _read_json(pointer_path, "resume_checkpoint_pointer_invalid")
    resume_identity = session["resume_identity"]
    checkpoint_name = pointer.get("path")
    if (
        pointer.get("schema") != "q25-fim-latest-checkpoint-v1"
        or not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
        or pointer.get("fingerprint") != resume_identity["fingerprint"]
        or pointer.get("sha256") != resume_identity["checkpoint_sha256"]
        or pointer.get("cursor") != resume_identity["cursor"]
    ):
        raise WorkerError("resume_checkpoint_identity_mismatch")
    checkpoint = pointer_path.parent / checkpoint_name
    marker_path = checkpoint.with_suffix(checkpoint.suffix + ".complete.json")
    if checkpoint.is_symlink() or marker_path.is_symlink() or not checkpoint.is_file():
        raise WorkerError("resume_checkpoint_missing_or_unsafe")
    marker = _read_json(marker_path, "resume_checkpoint_marker_missing")
    if (
        marker.get("fingerprint") != resume_identity["fingerprint"]
        or marker.get("sha256") != resume_identity["checkpoint_sha256"]
        or sha256_file(checkpoint) != marker.get("sha256")
    ):
        raise WorkerError("resume_checkpoint_content_mismatch")
    previous_root = pointer_path.parent.parent
    run_manifest = _read_json(
        previous_root / "training" / "run_manifest.json", "resume_run_manifest_missing"
    )
    identity = run_manifest.get("identity")
    corpus_identity = identity.get("corpus_metadata") if isinstance(identity, dict) else None
    training_identity = identity.get("training_data") if isinstance(identity, dict) else None
    if (
        run_manifest.get("fingerprint") != resume_identity["fingerprint"]
        or not isinstance(identity, dict)
        or identity.get("plan_sha256") != plan_sha256
        or identity.get("arm") != arm
        or canonical_sha256(identity.get("initializer")) != initializer_sha256
        or not isinstance(corpus_identity, dict)
        or not _is_sha256(corpus_identity.get("metadata_sha256"))
        or not isinstance(training_identity, dict)
        or not _is_sha256(training_identity.get("sha256"))
    ):
        raise WorkerError("resume_run_fingerprint_mismatch")
    baseline = _verify_baseline(
        previous_root,
        arm=arm,
        plan_sha256=plan_sha256,
        input_manifest_sha256=input_manifest_sha256,
        initializer_sha256=initializer_sha256,
    )
    return checkpoint, previous_root, baseline


def _baseline_inventory(output: Path) -> dict[str, dict[str, Any]]:
    before_root = output / "before"
    result: dict[str, dict[str, Any]] = {}
    for path in _tree_files(before_root):
        result[path.relative_to(output).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def trainer_command(
    *,
    model: Path,
    paths: dict[str, Path],
    plan: Path,
    arm: str,
    output: Path,
    session_seconds: int,
    reserve_seconds: int,
    external_campaign_tokens: int,
    mounted_bytes: int,
    resume: Path | None,
    execute: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tinycomplete.code_cpt.q25_fim",
        "--model",
        str(model),
        "--train",
        str(paths["train.jsonl"]),
        "--development",
        str(paths["development.jsonl"]),
        "--data-metadata",
        str(paths["corpus_metadata.json"]),
        "--plan",
        str(plan),
        "--arm",
        arm,
        "--output",
        str(output),
        "--session-seconds",
        str(session_seconds),
        "--reserve-seconds",
        str(reserve_seconds),
        "--external-campaign-tokens",
        str(external_campaign_tokens),
        "--mounted-artifact-bytes",
        str(mounted_bytes),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if execute:
        command.append("--execute")
    return command


def _evaluation_command(
    *, model: Path, input_path: Path, output: Path, plan: Path, alias: str, mode: str
) -> list[str]:
    return [
        sys.executable,
        str(REPO / "scripts/evaluate_q25_fim.py"),
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--plan",
        str(plan),
        "--alias",
        alias,
        "--mode",
        mode,
    ]


def _regression_commands(
    *, model: Path, paths: dict[str, Path], output: Path, plan_sha256: str, alias: str
) -> list[tuple[str, list[str]]]:
    return [
        (
            "regression-causal",
            [
                sys.executable,
                str(REPO / "scripts/evaluate_q25_code_cpt.py"),
                "--model",
                str(model),
                "--suite",
                str(paths["causal200.jsonl"]),
                "--output",
                str(output / "regression" / "causal"),
                "--alias",
                alias,
                "--plan-sha",
                plan_sha256,
            ],
        ),
        (
            "regression-line",
            [
                sys.executable,
                str(REPO / "scripts/evaluate_causal_line.py"),
                "--model",
                str(model),
                "--suite",
                str(paths["line180.jsonl"]),
                "--output",
                str(output / "regression" / "line"),
                "--alias",
                alias,
                "--plan-sha",
                plan_sha256,
            ],
        ),
    ]


class Worker:
    def __init__(self, session: dict[str, Any]) -> None:
        self.session = validate_session(session)
        self.started = time.monotonic()
        self.deadline = self.started + self.session["session_seconds"]
        self.status: dict[str, Any] = {
            "schema": "q25-fim-kaggle-worker-status-v1",
            "state": "setup",
            "commit": self.session["commit"],
            "attempt": self.session["attempt"],
            "arm": self.session["arm"],
            "plan_sha256": self.session["plan_sha256"],
            "input_manifest_sha256": self.session["input_manifest_sha256"],
            "training_started": False,
            "stages": [],
        }
        self.out = OUT

    def save_status(self) -> None:
        self.status["elapsed_seconds"] = time.monotonic() - self.started
        _write_json(self.out / "worker-status.json", self.status)

    def run_stage(
        self,
        command: list[str],
        name: str,
        *,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        reserve_seconds: float = 30,
    ) -> int:
        remaining = self.deadline - time.monotonic()
        usable = remaining - reserve_seconds
        if usable < 1:
            raise WorkerError("session_deadline_reserve_reached")
        timeout = usable if timeout_seconds is None else min(timeout_seconds, usable)
        started = time.monotonic()
        log_path = self.out / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("wb") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    env=env,
                    cwd=REPO if REPO.is_dir() else None,
                    check=False,
                )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        self.status["stages"].append(
            {"name": name, "exit_code": code, "elapsed_seconds": time.monotonic() - started}
        )
        self.save_status()
        return code

    def check_storage(self, *, mounted_bytes: int, plan: dict[str, Any]) -> None:
        cap = int(plan["configuration"]["budget"]["new_artifact_bytes_cap"])
        output_bytes = _tree_bytes(self.out)
        current = mounted_bytes + output_bytes
        self.status["storage"] = {
            "mounted_bytes": mounted_bytes,
            "worker_output_bytes": output_bytes,
            "combined_bytes": current,
            "cap_bytes": cap,
        }
        if current > cap:
            raise WorkerError("new_artifact_bytes_cap_exceeded")
        free = shutil.disk_usage(self.out.parent).free
        self.status["storage"]["free_bytes"] = free
        minimum_free_bytes = int(
            plan["configuration"]["budget"].get("minimum_free_bytes", MINIMUM_FREE_BYTES)
        )
        if free < minimum_free_bytes:
            raise WorkerError("minimum_free_space_not_available")
        self.save_status()

    def _run_evaluation(
        self,
        *,
        model: Path,
        input_path: Path,
        plan_path: Path,
        alias: str,
        mode: str,
        stage: str,
        env: dict[str, str],
        reserve_seconds: float,
    ) -> None:
        command = _evaluation_command(
            model=model,
            input_path=input_path,
            output=self.out / stage,
            plan=plan_path,
            alias=alias,
            mode=mode,
        )
        if (
            self.run_stage(
                command,
                f"eval-{stage.replace('/', '-')}",
                env=env,
                reserve_seconds=reserve_seconds,
            )
            != 0
        ):
            raise WorkerError("fim_evaluation_failed")

    def _verify_repository_fixture(self, paths: dict[str, Path], plan: dict[str, Any]) -> None:
        repo_fixture = REPO / "data/benchmarks/code_completion_v2.jsonl"
        if not repo_fixture.is_file() or sha256_file(repo_fixture) != sha256_file(
            paths["causal200.jsonl"]
        ):
            raise WorkerError("repository_causal_fixture_differs_from_frozen_input")
        planned = plan["evaluation"]["fixtures"]["causal"]
        if sha256_file(repo_fixture) != planned.get("sha256"):
            raise WorkerError("repository_causal_fixture_differs_from_plan")

    def _record_baseline(
        self,
        *,
        initializer_sha256: str,
        measured_seconds: float,
        plan: dict[str, Any],
        mounted_bytes: int,
        previous_root: Path | None = None,
        inherited: dict[str, Any] | None = None,
    ) -> None:
        if inherited is not None:
            if previous_root is None:
                raise WorkerError("resume_baseline_source_missing")
            files = inherited.get("files")
            if (
                inherited.get("schema") != "q25-fim-baseline-v1"
                or inherited.get("arm") != self.session["arm"]
                or inherited.get("plan_sha256") != self.session["plan_sha256"]
                or inherited.get("input_manifest_sha256") != self.session["input_manifest_sha256"]
                or inherited.get("initializer_identity_sha256") != initializer_sha256
                or not isinstance(files, dict)
                or not files
                or not _has_baseline_modes(files)
            ):
                raise WorkerError("resume_baseline_files_missing")
            copy_bytes = 0
            for name, item in files.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(item, dict)
                    or not _is_sha256(item.get("sha256"))
                    or isinstance(item.get("bytes"), bool)
                    or not isinstance(item.get("bytes"), int)
                    or item["bytes"] < 0
                ):
                    raise WorkerError("resume_baseline_file_identity_invalid")
                copy_bytes += item["bytes"]
            cap = int(plan["configuration"]["budget"]["new_artifact_bytes_cap"])
            if mounted_bytes + _tree_bytes(self.out) + copy_bytes > cap:
                raise WorkerError("baseline_copy_exceeds_artifact_cap")
            for name, record in files.items():
                relative = Path(name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != "before"
                ):
                    raise WorkerError("resume_baseline_file_path_invalid")
                source = previous_root / relative
                destination = self.out / relative
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or source.stat().st_size != record["bytes"]
                    or sha256_file(source) != record["sha256"]
                ):
                    raise WorkerError("resume_baseline_file_missing_or_changed")
                if destination.exists() or destination.is_symlink():
                    raise WorkerError("baseline_copy_destination_exists")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if (
                    destination.stat().st_size != record["bytes"]
                    or sha256_file(destination) != record["sha256"]
                ):
                    raise WorkerError("baseline_copy_hash_mismatch")
            _write_json(self.out / "baseline" / "identity.json", inherited)
            record = {
                "schema": "q25-fim-baseline-reference-v1",
                "source": self.session["resume_source"],
                "source_identity_sha256": canonical_sha256(inherited),
                "measured_seconds": inherited["measured_seconds"],
                "files": inherited["files"],
            }
            _write_json(self.out / "baseline" / "reference.json", record)
            self.status["baseline"] = record
            self.status["baseline_source"] = "verified_prior_arm_output"
            self.check_storage(mounted_bytes=mounted_bytes, plan=plan)
            return
        files = _baseline_inventory(self.out)
        if not files or not _has_baseline_modes(files):
            raise WorkerError("baseline_evaluation_outputs_missing")
        record = {
            "schema": "q25-fim-baseline-v1",
            "arm": self.session["arm"],
            "plan_sha256": self.session["plan_sha256"],
            "input_manifest_sha256": self.session["input_manifest_sha256"],
            "initializer_identity_sha256": initializer_sha256,
            "measured_seconds": measured_seconds,
            "files": files,
        }
        _write_json(self.out / "baseline" / "identity.json", record)
        self.status["baseline"] = record
        self.status["baseline_source"] = "evaluated_frozen_initializer"

    def execute(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        self.save_status()
        try:
            input_dir, manifest = find_input_manifest(
                INPUT_ROOT, self.session["input_manifest_sha256"]
            )
            paths, plan = verify_input_bundle(input_dir, manifest, self.session)
            plan_path = paths["plan.json"]
            arm = self.session["arm"]
            initializer_entry = plan["initializers"][arm]
            model = find_initializer_path(INPUT_ROOT, initializer_entry)
            resume_checkpoint: Path | None = None
            previous_root: Path | None = None
            inherited_baseline: dict[str, Any] | None = None
            total_mounted = _tree_bytes(INPUT_ROOT)
            self.status["state"] = "verified_inputs"
            self.save_status()
            self.check_storage(mounted_bytes=total_mounted, plan=plan)

            os.environ.update(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                HF_DATASETS_OFFLINE="1",
                HF_HUB_DISABLE_TELEMETRY="1",
                TOKENIZERS_PARALLELISM="false",
            )
            package_command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "-q",
                "transformers==5.17.0",
                "bitsandbytes==0.50.2",
                "PyYAML==6.0.2",
                "opentelemetry-api==1.44.0",
                "opentelemetry-sdk==1.44.0",
                "opentelemetry-exporter-otlp-proto-http==1.44.0",
                "pydantic>=2",
                "tree-sitter==0.25.2",
                "tree-sitter-language-pack==1.20.0",
                "httpx",
            ]
            if self.run_stage(package_command, "setup-packages", reserve_seconds=60) != 0:
                raise WorkerError("dependency_setup_failed")
            if (
                self.run_stage(
                    [
                        "git",
                        "clone",
                        "--branch",
                        "research/q25-code-cpt-r2",
                        "https://github.com/Shlok-Bhakta/tabcomplete.git",
                        str(REPO),
                    ],
                    "checkout-repository",
                    reserve_seconds=60,
                )
                != 0
            ):
                raise WorkerError("repository_checkout_failed")
            if (
                self.run_stage(
                    ["git", "-C", str(REPO), "checkout", self.session["commit"]],
                    "pin-repository",
                    reserve_seconds=60,
                )
                != 0
            ):
                raise WorkerError("repository_pin_failed")
            revision = subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if revision.returncode != 0 or revision.stdout.strip() != self.session["commit"]:
                raise WorkerError("repository_commit_mismatch")

            env = {
                **os.environ,
                "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}",
                "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
                "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
                "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(self.out / "observability.jsonl"),
                "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "0",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            sys.path.insert(0, str(REPO / "src"))
            self._verify_repository_fixture(paths, plan)
            import torch

            import tinycomplete.code_cpt.q25_fim as q25_fim

            if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
                raise WorkerError("intended_t4_backend_unavailable")
            hardware = {
                "torch": torch.__version__,
                "cuda_devices_visible": torch.cuda.device_count(),
                "training_device": torch.cuda.get_device_name(0),
                "world_size": 1,
                "other_devices_unused": True,
            }
            self.status["hardware"] = hardware
            initializer_identity = q25_fim.verify_initializer(
                model, arm=arm, entry=initializer_entry
            )
            initializer_sha256 = canonical_sha256(initializer_identity)
            if self.session["resume_source"] is not None:
                resume_checkpoint, previous_root, inherited_baseline = resolve_verified_resume(
                    INPUT_ROOT,
                    self.session,
                    plan_sha256=self.session["plan_sha256"],
                    input_manifest_sha256=self.session["input_manifest_sha256"],
                    initializer_sha256=initializer_sha256,
                    arm=arm,
                )
            trainer_files = [
                paths[name]
                for name in (
                    "plan.json",
                    "train.jsonl",
                    "development.jsonl",
                    "corpus_metadata.json",
                )
            ]
            mounted_extra = mounted_artifact_bytes(
                INPUT_ROOT,
                initializer=model,
                trainer_files=trainer_files,
                resume_checkpoint=resume_checkpoint,
            )
            self.status["input"] = {
                "directory": str(input_dir),
                "file_count": len(manifest["files"]),
                "mounted_bytes": total_mounted,
                "mounted_extra_bytes": mounted_extra,
                "initializer_identity_sha256": initializer_sha256,
            }
            self.save_status()
            if canonical_sha256(initializer_identity) != initializer_sha256:
                raise WorkerError("initializer_identity_changed_after_verification")

            preflight_command = trainer_command(
                model=model,
                paths=paths,
                plan=plan_path,
                arm=arm,
                output=self.out / "training",
                session_seconds=min(
                    self.session["session_seconds"],
                    int(plan["configuration"]["budget"]["session_seconds"]),
                ),
                reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
                external_campaign_tokens=self.session["external_campaign_tokens"],
                mounted_bytes=mounted_extra,
                resume=resume_checkpoint,
                execute=False,
            )
            if (
                self.run_stage(preflight_command, "trainer-preflight", env=env, reserve_seconds=60)
                != 0
            ):
                raise WorkerError("trainer_preflight_failed")

            alias = "untouched-q25" if arm == TRAIN_ARM else "completed-cpt-q25"
            before_started = time.monotonic()
            if inherited_baseline is None:
                self.status["state"] = "baseline_evaluation"
                self.save_status()
                for mode, filename in (
                    ("development", "development.jsonl"),
                    ("line", "line180.jsonl"),
                ):
                    self._run_evaluation(
                        model=model,
                        input_path=paths[filename],
                        plan_path=plan_path,
                        alias=f"{alias}-before-fim",
                        mode=mode,
                        stage=f"before/{mode}",
                        env=env,
                        reserve_seconds=MINIMUM_FINAL_RESERVE_SECONDS,
                    )
                before_seconds = time.monotonic() - before_started
                self._record_baseline(
                    initializer_sha256=initializer_sha256,
                    measured_seconds=before_seconds,
                    plan=plan,
                    mounted_bytes=total_mounted,
                )
            else:
                before_seconds = float(inherited_baseline["measured_seconds"])
                self._record_baseline(
                    initializer_sha256=initializer_sha256,
                    measured_seconds=before_seconds,
                    plan=plan,
                    mounted_bytes=total_mounted,
                    previous_root=previous_root,
                    inherited=inherited_baseline,
                )
            self.status["before_evaluation_seconds"] = before_seconds

            trainer_reserve = max(MINIMUM_FINAL_RESERVE_SECONDS, int(1.25 * before_seconds + 300))
            post_eval_reserve = max(MINIMUM_FINAL_RESERVE_SECONDS, int(2.5 * before_seconds + 300))
            remaining = int(self.deadline - time.monotonic())
            trainer_session_seconds = min(
                int(plan["configuration"]["budget"]["session_seconds"]),
                remaining - post_eval_reserve - 60,
            )
            self.status["timing_reserves"] = {
                "trainer_reserve_seconds": trainer_reserve,
                "post_evaluation_reserve_seconds": post_eval_reserve,
                "trainer_session_seconds": trainer_session_seconds,
            }
            if trainer_session_seconds <= trainer_reserve + 60:
                self.status["state"] = "training_deferred_insufficient_time"
                self.save_status()
                return 0

            worker_extra_output = _tree_bytes(self.out) - _tree_bytes(self.out / "training")
            trainer_mounted_bytes = mounted_extra + max(0, worker_extra_output)
            self.check_storage(mounted_bytes=total_mounted + max(0, worker_extra_output), plan=plan)
            training_command = trainer_command(
                model=model,
                paths=paths,
                plan=plan_path,
                arm=arm,
                output=self.out / "training",
                session_seconds=trainer_session_seconds,
                reserve_seconds=trainer_reserve,
                external_campaign_tokens=self.session["external_campaign_tokens"],
                mounted_bytes=trainer_mounted_bytes,
                resume=resume_checkpoint,
                execute=True,
            )
            self.status["state"] = "training"
            self.status["training_started"] = True
            self.status["trainer_mounted_artifact_bytes"] = trainer_mounted_bytes
            self.save_status()
            training_exit = self.run_stage(
                training_command,
                "training",
                env=env,
                timeout_seconds=trainer_session_seconds + 30,
                reserve_seconds=post_eval_reserve,
            )
            self.status["training_exit_code"] = training_exit
            result_path = self.out / "training" / "run_result.json"
            if training_exit != 0 or not result_path.is_file():
                self.status["state"] = "partial_checkpoint_preserved"
                self.status["checkpoint_pointer_present"] = (
                    self.out / "training" / "latest.json"
                ).is_file()
                self.save_status()
                return 0
            result = _read_json(result_path, "training_result_invalid")
            self.status["training"] = {
                "status": result.get("status"),
                "fingerprint": result.get("fingerprint"),
                "logical_training_input_tokens": result.get("logical_training_input_tokens"),
                "actual_campaign_input_tokens": result.get("actual_campaign_input_tokens"),
                "cursor": result.get("cursor"),
                "training_updates": result.get("training_updates"),
            }
            self.check_storage(mounted_bytes=total_mounted, plan=plan)
            if result.get("status") != "complete":
                self.status["state"] = "partial_checkpoint_preserved"
                self.save_status()
                return 0

            export = self.out / "training" / "inference-f16"
            self._verify_fim_export(export, result, arm)
            self.status["state"] = "candidate_evaluation"
            self.save_status()
            after_started = time.monotonic()
            for mode, filename in (
                ("development", "development.jsonl"),
                ("line", "line180.jsonl"),
            ):
                self._run_evaluation(
                    model=export,
                    input_path=paths[filename],
                    plan_path=plan_path,
                    alias=f"{alias}-after-fim",
                    mode=mode,
                    stage=f"after/{mode}",
                    env=env,
                    reserve_seconds=30,
                )
                self.check_storage(mounted_bytes=total_mounted, plan=plan)
            for name, command in _regression_commands(
                model=export,
                paths=paths,
                output=self.out,
                plan_sha256=self.session["plan_sha256"],
                alias=f"{alias}-after-fim",
            ):
                if self.run_stage(command, name, env=env, reserve_seconds=30) != 0:
                    raise WorkerError("raw_regression_evaluation_failed")
                self.check_storage(mounted_bytes=total_mounted, plan=plan)
            self.status["after_evaluation_seconds"] = time.monotonic() - after_started
            self.status["state"] = "complete"
            self.save_status()
            return 0
        except WorkerError as exc:
            self.status.update(state="failed", error_reason=exc.reason)
            self.save_status()
            return 1
        except Exception as exc:
            self.status.update(state="failed", error_class=type(exc).__name__)
            self.save_status()
            return 1

    def _verify_fim_export(self, export: Path, result: dict[str, Any], arm: str) -> None:
        manifest_path = export / "artifact_manifest.json"
        manifest = _read_json(manifest_path, "fim_export_manifest_missing")
        if (
            manifest.get("schema") != "q25-fim-inference-f16-v1"
            or manifest.get("arm") != arm
            or manifest.get("fingerprint") != result.get("fingerprint")
            or manifest.get("training_cursor") != result.get("cursor")
        ):
            raise WorkerError("fim_export_identity_mismatch")
        records = manifest.get("files")
        if not isinstance(records, dict) or "model.safetensors" not in records:
            raise WorkerError("fim_export_file_manifest_missing")
        for name, record in records.items():
            if Path(name).name != name or not isinstance(record, dict):
                raise WorkerError("fim_export_file_path_invalid")
            path = export / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                raise WorkerError("fim_export_file_hash_mismatch")


def main() -> int:
    try:
        worker = Worker(SESSION)
        result_code = worker.execute()
    except WorkerError as exc:
        print(
            json.dumps({"state": "failed", "error_reason": exc.reason}, sort_keys=True),
            flush=True,
        )
        return 1
    except Exception as exc:
        print(
            json.dumps({"state": "failed", "error_class": type(exc).__name__}, sort_keys=True),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "state": worker.status.get("state", "finished"),
                "arm": worker.status.get("arm"),
                "exit_code": result_code,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
