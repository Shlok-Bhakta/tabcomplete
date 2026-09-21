from __future__ import annotations

import re

import torch

from tinycomplete.eval.long_context_diagnostic import (
    CONDITIONS,
    build_diagnostic_family,
    selected_target_nll,
    target_logit_positions,
)


class RegexTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(re.findall(r"[A-Za-z_]+|\d+|[^\w\s]", text))))


def test_matched_family_has_all_dependency_conditions() -> None:
    cases = build_diagnostic_family(
        tokenizer=RegexTokenizer(),
        family="constant",
        target_tokens=2048,
        seed=17,
        tolerance_fraction=0.1,
    )

    assert {case.condition for case in cases} == set(CONDITIONS)
    by_condition = {case.condition: case for case in cases}
    assert by_condition["long_far"].dependency_distance_tokens > by_condition[
        "long_near"
    ].dependency_distance_tokens
    assert by_condition["absent"].dependency_token_position is None
    assert by_condition["counterfactual"].target != by_condition["long_far"].target
    assert by_condition["counterfactual"].target == by_condition["long_far"].distractor
    assert by_condition["short_control"].prompt_tokens < by_condition["long_near"].prompt_tokens


def test_independent_seeds_change_names_and_values() -> None:
    first = build_diagnostic_family(
        tokenizer=RegexTokenizer(), family="enum", target_tokens=2048, seed=10
    )
    second = build_diagnostic_family(
        tokenizer=RegexTokenizer(), family="enum", target_tokens=2048, seed=11
    )

    assert first[0].prompt != second[0].prompt
    assert first[0].target != second[0].target


def test_target_only_selected_logits_match_full_logit_scoring() -> None:
    torch.manual_seed(3)
    prompt_length = 5
    target_ids = torch.tensor([[2, 4, 1]])
    input_ids = torch.tensor([[6, 3, 5, 0, 2, 2, 4, 1]])
    full_logits = torch.randn(1, input_ids.shape[1], 8)
    positions = target_logit_positions(prompt_length, target_ids.shape[1], device=None)
    selected = full_logits[:, positions, :]

    selected_sum, selected_count = selected_target_nll(selected, target_ids)
    full_targets = input_ids[:, 1:]
    full_losses = torch.nn.functional.cross_entropy(
        full_logits[:, :-1, :].reshape(-1, 8),
        full_targets.reshape(-1),
        reduction="none",
    ).reshape_as(full_targets)
    expected = full_losses[:, prompt_length - 1 :].sum()

    assert positions.tolist() == [4, 5, 6]
    assert selected_count == 3
    assert torch.allclose(selected_sum, expected)
