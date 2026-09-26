"""Campaign limits must reject a projected action before resource use."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_one_line_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_one_line_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_storage_cap_counts_existing_campaign_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    report = root / "reports/research/one_line_r1"
    artifacts = root / "artifacts/research/one_line_r1"
    report.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (artifacts / "previous.bin").write_bytes(b"12345678")
    (report / "quota_ledger.json").write_text(
        json.dumps({"session_wall_seconds": 0, "training_input_tokens": 0})
    )
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "REPORT", report)
    plan = {"budgets": {
        "maximum_kaggle_t4x2_session_wall_hours": 24,
        "maximum_nonpadding_training_input_tokens": 100,
        "maximum_new_persistent_local_research_bytes": 10,
    }}
    campaign.budget_check(plan, new_bytes=2)
    with pytest.raises(RuntimeError, match="storage cap"):
        campaign.budget_check(plan, new_bytes=3)


def test_training_token_cap_includes_prior_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "reports/research/one_line_r1"
    report.mkdir(parents=True)
    (report / "quota_ledger.json").write_text(
        json.dumps({"session_wall_seconds": 0, "training_input_tokens": 90})
    )
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "REPORT", report)
    plan = {"budgets": {
        "maximum_kaggle_t4x2_session_wall_hours": 24,
        "maximum_nonpadding_training_input_tokens": 100,
        "maximum_new_persistent_local_research_bytes": 10,
    }}
    with pytest.raises(RuntimeError, match="input-token cap"):
        campaign.budget_check(plan, training_tokens=11)
