"""Evaluator: metric correctness + CPU latency smoke."""

import torch

from tinycomplete.eval.latency import benchmark, peak_vram_mb
from tinycomplete.eval.metrics import (
    aggregate,
    candidate_length,
    character_edit_distance,
    exact_match,
    noop_accuracy,
    normalized_exact_match,
    prefix_match_length,
    score_example,
    tree_sitter_parse_success,
    write_predictions,
)
from tinycomplete.model.tiny_qwen import build_tiny_model


def test_exact_and_normalized():
    assert exact_match("x = 1", "x = 1")
    assert not exact_match("x = 1 ", "x = 1")
    assert normalized_exact_match("x = 1 \n", "x = 1")
    assert not normalized_exact_match("x = 1", "x = 2")


def test_prefix_and_distance():
    assert prefix_match_length("abcdef", "abcxyz") == 3
    assert prefix_match_length("", "abc") == 0
    assert character_edit_distance("kitten", "sitting") == 3
    assert character_edit_distance("", "") == 0
    assert character_edit_distance("a", "") == 1
    assert candidate_length("héllo") == {"chars": 5, "bytes": 6}


def test_parse_and_noop():
    assert tree_sitter_parse_success("def f():\n    return 1\n")
    assert not tree_sitter_parse_success("def f(:\n")
    assert noop_accuracy([True, True, False], [True, False, False]) == 0.5
    assert noop_accuracy([False], [False]) is None


def test_score_and_aggregate_and_store(tmp_path):
    scores = [
        score_example("x = 1", "x = 1"),
        score_example("x = 2", "x = 1"),
    ]
    assert scores[0]["exact_match"] is True
    assert scores[1]["character_edit_distance"] == 1
    agg = aggregate(scores)
    assert agg["n"] == 2
    assert agg["exact_match_rate"] == 0.5
    assert agg["character_edit_distance_mean"] == 0.5
    path = str(tmp_path / "preds.jsonl")
    write_predictions(path, [{"pred": "x = 1", "gold": "x = 1", **scores[0]}])
    assert len(open(path, encoding="utf-8").read().strip().split("\n")) == 1


def test_latency_benchmark_cpu():
    model = build_tiny_model(seed=0)
    ids = torch.tensor([[10, 20, 30, 40]])
    stats = benchmark(model, ids, max_new_tokens=2)
    assert stats.prefill_tokens == 4
    assert stats.decoded_tokens == 2
    assert stats.wall_time > 0
    assert stats.prefill_tokens_per_second > 0
    assert stats.decode_tokens_per_second > 0
    assert stats.time_to_first_token > 0
    assert peak_vram_mb() is None  # CPU-only here
