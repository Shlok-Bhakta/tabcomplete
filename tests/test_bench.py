"""Tests for the T4x2 throughput benchmark lab (Stage 1, CPU-safe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH_MATRIX = REPO / "kaggle" / "code_cpt_bench" / "candidates.json"
sys.path.insert(0, str(REPO / "kaggle" / "code_cpt_bench"))
sys.path.insert(0, str(REPO / "src"))

SEQ_LEN = 2048
WORLD = 2
BASELINE_UPDATE = 32_768


def _load_matrix():
    return json.loads(BENCH_MATRIX.read_text(encoding="utf-8"))


def test_stage_token_budgets_align_to_complete_updates() -> None:
    assert 98_304 == 3 * BASELINE_UPDATE
    assert 524_288 == 16 * BASELINE_UPDATE
    assert 1_048_576 == 32 * BASELINE_UPDATE


def test_candidate_matrix_is_update_aligned() -> None:
    raw = _load_matrix()
    shared = raw["shared"]
    assert shared["max_tokens"] == 98_304
    names = [c["name"] for c in raw["candidates"]]
    assert len(names) == len(set(names)) and len(names) >= 10
    assert "exp00_baseline_repro" in names
    for candidate in raw["candidates"]:
        max_tokens = int(candidate.get("max_tokens", shared["max_tokens"]))
        update = (
            WORLD * int(candidate["microbatch"]) * SEQ_LEN * int(candidate["gradient_accumulation"])
        )
        assert max_tokens % update == 0, candidate["name"]
        assert candidate["fsdp_strategy"] in ("FULL_SHARD", "SHARD_GRAD_OP")


def test_bench_parser_round_trips_candidate_flags() -> None:
    from tinycomplete.code_cpt.bench import build_parser

    args = build_parser().parse_args(
        [
            "bench",
            "--corpus-dir",
            "corpus",
            "--output",
            "out.json",
            "--name",
            "exp03_fla_fusedce_nockpt",
            "--microbatch",
            "1",
            "--gradient-accumulation",
            "8",
            "--fused-ce",
            "auto",
            "--fsdp-strategy",
            "FULL_SHARD",
            "--no-gradient-checkpointing",
            "--sync-cleanup",
        ]
    )
    assert args.name == "exp03_fla_fusedce_nockpt"
    assert args.fused_ce == "auto"
    assert args.gradient_checkpointing is False
    assert args.sync_cleanup is True


def test_orchestrator_rejects_mid_update_budgets() -> None:
    from run_bench import validate_alignment

    _, _, error = validate_alignment(
        {"microbatch": 1, "gradient_accumulation": 8}, {"max_tokens": 98_304}
    )
    assert error is None
    _, _, error = validate_alignment(
        {"microbatch": 1, "gradient_accumulation": 8}, {"max_tokens": 100_000}
    )
    assert error is not None and "mid-update" in error


def test_fused_ce_mode_resolution() -> None:
    from tinycomplete.code_cpt.bench import resolve_fused_ce

    assert resolve_fused_ce("none") == (None, "stock")
    assert resolve_fused_ce("chunked") == (None, "chunked")
    # liger-kernel is not installed locally, so auto must degrade to chunked,
    # never to stock (which would silently drop the memory experiment).
    assert resolve_fused_ce("auto") == (None, "chunked")
    with pytest.raises(ImportError):
        resolve_fused_ce("liger")


def test_result_schema_has_reproduction_fields() -> None:
    from tinycomplete.code_cpt.bench import run_bench as _  # noqa: F401 (import check)

    required = {
        "name",
        "status",
        "training_tokens",
        "optimizer_steps",
        "wall_seconds",
        "training_seconds",
        "steady_state_seconds",
        "tokens_per_second",
        "steady_state_tokens_per_second",
        "speedup_vs_baseline",
        "peak_allocated_vram_gib",
        "loss_start",
        "gradient_norm_mean",
        "loss_scale_overflows",
        "nan_or_inf",
        "data_wait_percent",
        "gdn_backend",
        "causal_conv_backend",
        "attention_backend",
        "fused_cross_entropy",
        "gradient_checkpointing",
        "fsdp_strategy",
        "optimizer",
        "microbatch_per_gpu",
        "gradient_accumulation",
        "sequence_length",
        "torch_compile",
        "tokens_verified",
        "weights_updated",
    }
    import inspect

    source = inspect.getsource(__import__("tinycomplete.code_cpt.bench", fromlist=["run_bench"]))
    for field in required:
        assert f'"{field}"' in source, field


def test_plots_render_from_synthetic_results(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from plot import render_all

    results = []
    for i, name in enumerate(["exp00_baseline_repro", "exp01_fla_conv", "exp02_fla_fusedce"]):
        results.append(
            {
                "name": name,
                "status": "pass",
                "steady_state_tokens_per_second": 810.0 + i * 100,
                "speedup_vs_baseline": (810.0 + i * 100) / 810.0,
                "peak_allocated_vram_gib_max": 11.7 + i * 0.5,
                "peak_allocated_vram_gib": [11.7 + i * 0.5] * 2,
                "training_seconds": 120.0,
                "forward_seconds": 50.0,
                "backward_seconds": 55.0,
                "optimizer_seconds": 5.0,
                "data_wait_seconds": 1.0,
                "loss_start": 1.3,
                "loss_end": 1.1,
                "gradient_norm_mean": 3.0,
                "loss_scale_overflows": 0,
            }
        )
    results.append(
        {
            "name": "exp12_mb2_nockpt_retry",
            "status": "oom",
            "error_type": "CUDA_OOM",
            "steady_state_tokens_per_second": 0,
        }
    )
    path = tmp_path / "throughput_benchmarks.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    made = render_all(path, tmp_path / "plots")
    assert "plot1_throughput_by_config.png" in made
    assert "plot2_speedup_vs_baseline.png" in made
    assert "plot3_throughput_vs_vram.png" in made
    assert "plot4_step_time_breakdown.png" in made
    summary = json.loads((tmp_path / "plots" / "summary.json").read_text())
    assert summary["stable"] == 3
    assert summary["failed"] == ["exp12_mb2_nockpt_retry"]
