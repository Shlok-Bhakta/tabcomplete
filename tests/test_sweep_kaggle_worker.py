"""Budget and artifact identity safeguards before a Sweep GPU allocation."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "kaggle/sweep_comparison_r1/run.py"
SPEC = importlib.util.spec_from_file_location("sweep_kaggle_worker", PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def valid_spec():
    return {
        "schema": "sweep-comparison-kaggle-input-v1",
        "model_revision": worker.MODEL_REVISION,
        "model_sha256": worker.MODEL_SHA256,
        "runtime_revision": worker.RUNTIME_REVISION,
        "session_seconds": 7200,
        "reserve_seconds": 1200,
        "training_enabled": False,
        "commit": "a" * 40,
        "files": {"plan.json": "b" * 64},
        "runner_sha256": "c" * 64,
        "runner_arguments": ["--plan", "{input}/plan.json"],
        "runner_modes": ["download", "quantize", "quality", "next-edit"],
    }


@pytest.mark.parametrize("field,value", [
    ("model_sha256", "a" * 64),
    ("model_revision", "a" * 40),
    ("runtime_revision", "a" * 40),
    ("training_enabled", True),
    ("reserve_seconds", 0),
    ("session_seconds", 7201),
    ("runner_arguments", [None]),
    ("runner_arguments", ["--mode", "download"]),
    ("runner_modes", ["train"]),
    ("runner_sha256", "invalid"),
    ("commit", "invalid"),
    ("files", {"../secret": "b" * 64}),
])
def test_rejects_unapproved_identity_and_input_paths(field, value):
    spec = valid_spec()
    spec[field] = value
    with pytest.raises(ValueError):
        worker.validate_spec(spec)


def test_accepts_exact_authorized_spec():
    worker.validate_spec(valid_spec())


def test_finalization_reserve_is_not_inference_time():
    assert worker.remaining_seconds(worker.START + 5999) == 1
    with pytest.raises(TimeoutError):
        worker.remaining_seconds(worker.START + 6000)


def test_reexec_preserves_original_session_clock(monkeypatch):
    monkeypatch.setenv("TABCOMPLETE_SWEEP_SESSION_START", "100")
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    assert module.START == 100
    assert module.DEADLINE == 6100


def test_timeout_terminates_owned_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "OUT", tmp_path)
    monkeypatch.setattr(worker, "remaining_seconds", lambda: 0.5)
    pid_file = tmp_path / "child-pid"
    program = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(TimeoutError):
        worker.run([sys.executable, "-c", program, str(pid_file)], "timeout-test")
    assert pid_file.exists()
    child = int(pid_file.read_text())
    for _ in range(50):
        stat = Path(f"/proc/{child}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        os.kill(child, 9)
        pytest.fail("stage timeout left an owned child running")


def test_canonical_q4_staging_checks_identity_and_reuses(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    artifact = source / worker.Q4_FILE
    artifact.write_bytes(b"authorized-test-artifact")
    monkeypatch.setattr(worker, "Q4_BYTES", artifact.stat().st_size)
    monkeypatch.setattr(worker, "Q4_SHA256", worker.digest(artifact))
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(worker, "STORAGE_ROOT", scratch)
    worker.stage_canonical_q4(source, scratch)
    assert (scratch / worker.Q4_FILE).read_bytes() == artifact.read_bytes()
    worker.stage_canonical_q4(source, scratch)
    (scratch / worker.Q4_FILE).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="existing worker"):
        worker.stage_canonical_q4(source, scratch)


def test_bad_canonical_q4_never_stages(tmp_path):
    source = tmp_path / worker.Q4_FILE
    source.write_bytes(b"wrong model")
    scratch = tmp_path / "scratch"
    with pytest.raises(ValueError, match="identity"):
        worker.stage_canonical_q4(tmp_path, scratch)
    assert not scratch.exists()


def test_storage_budget_accounts_existing_files_before_more_work(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "STORAGE_ROOT", tmp_path)
    monkeypatch.setattr(worker, "STORAGE_CAP", 100)
    (tmp_path / "existing").write_bytes(b"x" * 80)
    with pytest.raises(RuntimeError, match="cap"):
        worker.check_storage(21)
    worker.check_storage(20)
