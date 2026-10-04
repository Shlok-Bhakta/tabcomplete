from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_q25_code_cpt as campaign  # noqa: E402


def export_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, Path]:
    arm = campaign.FIM_ARMS[0]
    artifacts = tmp_path / "artifacts"
    export = artifacts / f"fim/output-{arm}-1/training/inference-f16"
    export.mkdir(parents=True)
    (export / "model.safetensors").write_bytes(b"synthetic trained weights")
    files = {
        "model.safetensors": {"bytes": 25, "sha256": campaign.digest(export / "model.safetensors")}
    }
    cursor = {"training_input_tokens": 42, "completed_updates": 1}
    campaign.save(
        export / "artifact_manifest.json",
        {
            "schema": "q25-fim-inference-f16-v1",
            "arm": arm,
            "fingerprint": "identity",
            "training_cursor": cursor,
            "files": files,
        },
    )
    report = tmp_path / "report"
    campaign.save(
        report / f"fim-verified-{arm}-1.json",
        {
            "arm": arm,
            "checkpoint_verified": True,
            "training_status": "complete",
            "cursor": cursor,
            "fingerprint": "identity",
            "reference": "owner/kernel",
        },
    )
    monkeypatch.setattr(campaign, "ARTIFACTS", artifacts)
    monkeypatch.setattr(campaign, "REPORT", report)
    plan = {
        "data": {"train": {"input_tokens": 42}},
        "configuration": {
            "budget": {
                "new_artifact_bytes_cap": 1024**3,
                "minimum_free_bytes": 0,
            }
        },
    }
    return plan, export, report


def test_fim_export_verifies_complete_model_without_fetching_another_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, export, report = export_context(tmp_path, monkeypatch)

    def forbidden_download(*_args: Any, **_kwargs: Any) -> str:
        pytest.fail("all manifested files are already present")

    monkeypatch.setattr(campaign, "cli", forbidden_download)
    assert campaign.collect_fim_export(plan, campaign.FIM_ARMS[0], 1) == export
    assert (report / f"fim-export-verification-{campaign.FIM_ARMS[0]}-1.json").is_file()


@pytest.mark.parametrize("failure", ["partial", "wrong_arm", "changed_weights", "unsafe_path"])
def test_fim_export_rejects_unverified_or_changed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import json

    plan, export, report = export_context(tmp_path, monkeypatch)
    arm = campaign.FIM_ARMS[0]
    if failure == "partial":
        path = report / f"fim-verified-{arm}-1.json"
        row = json.loads(path.read_text())
        row["training_status"] = "paused"
        campaign.save(path, row)
    elif failure == "changed_weights":
        (export / "model.safetensors").write_bytes(b"changed")
    else:
        path = export / "artifact_manifest.json"
        row = json.loads(path.read_text())
        if failure == "wrong_arm":
            row["arm"] = campaign.FIM_ARMS[1]
        else:
            row["files"]["../private"] = row["files"]["model.safetensors"]
        campaign.save(path, row)
    with pytest.raises(ValueError):
        campaign.collect_fim_export(plan, arm, 1)
