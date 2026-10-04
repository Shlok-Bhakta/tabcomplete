from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))

import build_pilot  # noqa: E402
import run_q25_code_cpt as campaign  # noqa: E402


def setup_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, Path]:
    artifacts = tmp_path / "artifacts"
    export = artifacts / "output-attempt-1/training/inference-f16"
    export.mkdir(parents=True)
    (export / "model.safetensors").write_bytes(b"approved synthetic weights")
    (export / "config.json").write_text("{}")
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": campaign.digest(p)} for p in export.iterdir()
    }
    manifest = export / "artifact_manifest.json"
    campaign.save(manifest, {"files": files})
    plan = {
        "initializers": {
            campaign.FIM_ARMS[1]: {
                "files": files,
                "artifact_manifest_sha256": campaign.digest(manifest),
            }
        },
        "configuration": {"budget": {"new_artifact_bytes_cap": 1024**3, "minimum_free_bytes": 0}},
    }
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(campaign, "REPORT", tmp_path / "report")
    monkeypatch.setattr(build_pilot, "_csv_refs", lambda _args: set())
    monkeypatch.setattr(
        campaign, "cli", lambda *_args, **_kwargs: "Your private Dataset is being created."
    )
    return plan, export, artifacts


def test_verified_upload_removes_only_staging_duplicate_and_resume_never_recopies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, export, artifacts = setup_upload(tmp_path, monkeypatch)
    checks = []

    def remote_check(dataset: str, files: dict[str, Any]) -> dict[str, Any]:
        checks.append((dataset, files))
        return {"verified": True}

    monkeypatch.setattr(build_pilot, "wait_for_remote_inputs", remote_check)
    assert campaign.upload_cpt_initializer(plan) == {"verified": True}
    staged = artifacts / "fim/cpt-initializer-bundle/model.safetensors"
    assert not staged.exists()
    assert (export / "model.safetensors").read_bytes() == b"approved synthetic weights"
    assert (artifacts / "fim/cpt-initializer-bundle/config.json").is_file()

    def cannot_copy(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("verified immutable dataset must not recopy weights")

    monkeypatch.setattr(campaign.shutil, "copy2", cannot_copy)
    assert campaign.upload_cpt_initializer(plan) == {"verified": True}
    assert len(checks) == 2
    assert not staged.exists()


def test_remote_verification_failure_keeps_staged_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, export, artifacts = setup_upload(tmp_path, monkeypatch)

    def remote_failure(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("synthetic remote verification failure")

    monkeypatch.setattr(build_pilot, "wait_for_remote_inputs", remote_failure)
    with pytest.raises(RuntimeError, match="synthetic remote"):
        campaign.upload_cpt_initializer(plan)
    assert (artifacts / "fim/cpt-initializer-bundle/model.safetensors").is_file()
    assert (export / "model.safetensors").is_file()


def test_receipt_cannot_authorize_a_different_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, export, artifacts = setup_upload(tmp_path, monkeypatch)
    campaign.save(
        artifacts / "fim/cpt-initializer-submission.json",
        {
            "dataset": "unapproved/other",
            "artifact_manifest_sha256": campaign.digest(export / "artifact_manifest.json"),
            "state": "verified",
        },
    )
    with pytest.raises(ValueError, match="identity differs"):
        campaign.upload_cpt_initializer(plan)
    assert (export / "model.safetensors").is_file()
