from __future__ import annotations

import hashlib

import pytest
from analyze_q25_completion_scale import analyze


class Tokenizer:
    def decode(self, ids, **kwargs):
        return {1: "}", 2: "return value\n"}[ids[0]]


def fixture():
    corpus, predictions = [], []
    for index, target in enumerate(("}", "return value\n")):
        prompt_hash = hashlib.sha256(f"prompt-{index}".encode()).hexdigest()
        repository = hashlib.sha256(f"repo-{index}".encode()).hexdigest()
        corpus.append(
            {
                "id": 8192 + index,
                "input_ids": [5, index + 1, 9],
                "prompt_tokens": 1,
                "prompt_sha256": prompt_hash,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                "repository_identity_sha256": repository,
                "language": "rust",
                "mode": "whole_logical_line",
            }
        )
        predictions.append(
            {
                "case_id": f"fim-development-{8192 + index}",
                "context_sha256": prompt_hash,
                "repository": repository,
                "exact_and_terminated": False,
                "exact": False,
                "terminated": True,
                "reached_token_ceiling": False,
                "valid_output_tokens": True,
                "output_tokens": 2,
            }
        )
    return corpus, predictions


def test_complete_paired_results_keep_content_and_punctuation_denominators():
    corpus, repeat = fixture()
    scaled = [{**row, "exact": True, "exact_and_terminated": True} for row in repeat]
    result = analyze(corpus, repeat, scaled, tokenizer=Tokenizer())
    assert result["primary"]["first_successes"] == 0
    assert result["primary"]["second_successes"] == 2
    assert result["strata"]["target_content"]["punctuation_only"]["scaled"]["cases"] == 1
    assert result["strata"]["target_content"]["content_bearing"]["scaled"]["cases"] == 1


@pytest.mark.parametrize("changed", ["context_sha256", "repository", "case_id"])
def test_different_editor_state_cannot_enter_a_paired_comparison(changed):
    corpus, repeat = fixture()
    scaled = [dict(row) for row in repeat]
    scaled[0][changed] = "different-state"
    with pytest.raises(ValueError):
        analyze(corpus, repeat, scaled, tokenizer=Tokenizer())


def test_target_hash_mismatch_stops_analysis():
    corpus, repeat = fixture()
    corpus[0]["target_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target token IDs"):
        analyze(corpus, repeat, repeat, tokenizer=Tokenizer())
