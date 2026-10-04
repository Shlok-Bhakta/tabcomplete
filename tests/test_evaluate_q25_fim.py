"""Observed stopping and raw-token integrity are separate from decoded text."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_q25_fim import decoded_completion, development_case  # noqa: E402


class Tokenizer:
    eos_token_id = 9
    all_special_ids = [8, 9]
    pieces = {0: " ", 1: "\t", 2: "λ", 3: "\n", 8: "<|fim_middle|>", 9: "<eos>"}

    def get_vocab(self):
        return {text: index for index, text in self.pieces.items()}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces.get(index, "") for index in ids)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == "<|fim_middle|>λ":
            return [8, 2]
        raise ValueError("unexpected test prompt")


def test_eos_at_ceiling_terminates_without_trimming_unicode_or_whitespace() -> None:
    text, reason, evidence = decoded_completion(
        Tokenizer(), [0, 1, 2, 3, 9], ceiling=5, newline_stop=False
    )
    assert text == " \tλ\n"
    assert reason == "eos"
    assert evidence["ended_by_eos"] is True
    assert evidence["reached_token_ceiling"] is True
    assert evidence["truncated"] is False


@pytest.mark.parametrize("ids", [[8, 2, 9], [2, 9, 9], [2, 99, 9]])
def test_control_or_unknown_ids_are_invalid_even_if_decoded_text_looks_useful(ids) -> None:
    text, reason, evidence = decoded_completion(Tokenizer(), ids, ceiling=96, newline_stop=False)
    assert reason == "invalid_control_or_vocabulary"
    assert evidence["unexpected_special_token_ids"] or evidence["unknown_token_ids"]
    if 8 in ids:
        assert "<|fim_middle|>" in text


def test_a_parsable_cutoff_is_not_an_explicit_stop() -> None:
    text, reason, evidence = decoded_completion(Tokenizer(), [2, 3], ceiling=2, newline_stop=False)
    assert text == "λ\n"
    assert reason == "length"
    assert evidence["truncated"] is True
    assert evidence["ended_by_eos"] is False


def test_registered_newline_stop_and_empty_unverified_output_are_distinct() -> None:
    assert decoded_completion(Tokenizer(), [2, 3], ceiling=2, newline_stop=True)[1] == "newline"
    assert (
        decoded_completion(Tokenizer(), [], ceiling=96, newline_stop=True)[1] == "unverified_stop"
    )


def test_prepared_ids_reconstruct_exact_prompt_target_and_repository_identity() -> None:
    row = {
        "id": 12,
        "input_ids": [8, 2, 0, 1, 2, 3, 9],
        "prompt_tokens": 2,
        "prompt_sha256": hashlib.sha256("<|fim_middle|>λ".encode()).hexdigest(),
        "target_sha256": hashlib.sha256(" \tλ\n".encode()).hexdigest(),
        "repository_identity_sha256": "repository digest",
    }
    case = development_case(row, Tokenizer())
    assert case["id"] == "fim-development-12"
    assert case["prompt"] == "<|fim_middle|>λ"
    assert case["target"] == " \tλ\n"
    assert case["repository"] == "repository digest"
    row["target_sha256"] = "wrong"
    with pytest.raises(ValueError, match="exact FIM strings"):
        development_case(row, Tokenizer())


def test_development_target_requires_observed_ground_truth_eos() -> None:
    with pytest.raises(ValueError, match="declared EOS"):
        development_case({"input_ids": [2], "prompt_tokens": 0}, Tokenizer())


def test_paired_completion_bootstrap_preserves_repository_groups_and_case_weights() -> None:
    from evaluate_q25_fim import paired_development

    first = [
        {"case_id": "a", "repository": "repo1", "context_sha256": "1", "exact": False},
        {"case_id": "b", "repository": "repo1", "context_sha256": "2", "exact": False},
        {"case_id": "c", "repository": "repo2", "context_sha256": "3", "exact": True},
    ]
    second = [{**row, "exact": not row["exact"]} for row in first]
    result = paired_development(first, second, metric="exact")
    assert result["repository_groups"] == 2
    assert result["cases"] == 3
    assert result["wins"] == ["a", "b"]
    assert result["losses"] == ["c"]
    assert result["difference_second_minus_first"] == pytest.approx(1 / 3)
    assert result["paired_repository_bootstrap_95ci"] == [-1.0, 1.0]
    assert paired_development(first, second, metric="exact") == result


@pytest.mark.parametrize("failure", ["duplicate", "changed_context", "changed_repo", "non_boolean"])
def test_paired_completion_rejects_unmatched_or_ambiguous_evidence(failure: str) -> None:
    from evaluate_q25_fim import paired_development

    first = [{"case_id": "a", "repository": "repo", "context_sha256": "1", "exact": True}]
    second = [dict(first[0])]
    if failure == "duplicate":
        second.append(dict(second[0]))
    elif failure == "changed_context":
        second[0]["context_sha256"] = "different"
    elif failure == "changed_repo":
        second[0]["repository"] = "other"
    else:
        second[0]["exact"] = 1
    with pytest.raises(ValueError):
        paired_development(first, second, metric="exact")
