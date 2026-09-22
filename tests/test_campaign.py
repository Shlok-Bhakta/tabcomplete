from __future__ import annotations

from pathlib import Path

from tinycomplete.code_cpt.campaign import (
    CampaignLimits,
    _find_output_file,
    affordable_pilot_plan,
)


def test_affordable_plan_keeps_all_four_matched_arms_when_quota_fits() -> None:
    limits = CampaignLimits(
        max_wall_hours=12,
        max_gpu_hours=24,
        max_training_tokens=50_000_000,
        reserve_minutes=15,
    )

    plan = affordable_pilot_plan(
        limits=limits,
        quota_remaining_gpu_hours=20,
        campaign_wall_hours_used=0.5,
        estimated_seconds_per_full_arm=4_200,
    )

    assert plan["arms"] == ["C12", "C5", "D12", "D5"]
    assert plan["updates_per_arm"] == 153
    assert plan["new_training_tokens"] == 4 * 5_013_504


def test_affordable_plan_falls_back_to_matched_constant_pair() -> None:
    limits = CampaignLimits(12, 24, 50_000_000, 15)

    plan = affordable_pilot_plan(
        limits=limits,
        quota_remaining_gpu_hours=3.0,
        campaign_wall_hours_used=0.0,
        estimated_seconds_per_full_arm=4_200,
    )

    assert plan["arms"] == ["C12", "C5"]
    assert plan["updates_per_arm"] == 153


def test_affordable_plan_reduces_both_constant_arms_equally() -> None:
    limits = CampaignLimits(12, 24, 50_000_000, 15)

    plan = affordable_pilot_plan(
        limits=limits,
        quota_remaining_gpu_hours=1.5,
        campaign_wall_hours_used=0.0,
        estimated_seconds_per_full_arm=4_200,
    )

    assert plan["arms"] == ["C12", "C5"]
    assert 61 <= plan["updates_per_arm"] < 153
    assert plan["new_training_tokens"] == 2 * plan["updates_per_arm"] * 32_768


def test_affordable_plan_refuses_an_unusable_pair() -> None:
    limits = CampaignLimits(12, 24, 50_000_000, 15)

    plan = affordable_pilot_plan(
        limits=limits,
        quota_remaining_gpu_hours=0.5,
        campaign_wall_hours_used=0.0,
        estimated_seconds_per_full_arm=4_200,
    )

    assert plan["arms"] == []
    assert plan["reason"] == "a useful matched pair does not fit"


def test_output_discovery_prefers_kernel_artifact_over_cloned_report(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "code_cpt_evaluation_r1" / "selection.json"
    clone = tmp_path / "tabcomplete" / "reports" / "campaign" / "selection.json"
    artifact.parent.mkdir(parents=True)
    clone.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    clone.write_text("{}\n", encoding="utf-8")

    assert _find_output_file(tmp_path, "selection.json") == artifact
