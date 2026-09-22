from __future__ import annotations

import re
from types import SimpleNamespace

import torch

from tinycomplete.eval.code_benchmark import Prediction, evaluate_prediction
from tinycomplete.eval.long_context_diagnostic import (
    CONDITIONS,
    build_diagnostic_family,
    diagnostic_case_to_benchmark,
    greedy_generate_streamed,
    score_target_continuation_streamed,
    selected_target_nll,
    target_logit_positions,
    verify_streamed_scoring,
)


class RegexTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(re.findall(r"[A-Za-z_]+|\d+|[^\w\s]", text))))


class IntegerTokenizer:
    eos_token_id = 6

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [int(value) for value in text.split()]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        values = [token for token in token_ids if not skip_special_tokens or token != 6]
        return " ".join(str(value) for value in values)


class NextIntegerModel:
    def __call__(
        self,
        *,
        input_ids,
        past_key_values=None,
        use_cache: bool,
        logits_to_keep,
    ):
        del use_cache
        vocabulary = 11
        logits = torch.full((*input_ids.shape, vocabulary), -8.0, device=input_ids.device)
        next_ids = (input_ids + 1) % vocabulary
        logits.scatter_(2, next_ids.unsqueeze(-1), 8.0)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep, :]
        cached = (past_key_values or 0) + input_ids.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=cached)


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


def test_streamed_scoring_and_generation_match_full_short_input() -> None:
    model = NextIntegerModel()
    tokenizer = IntegerTokenizer()
    device = torch.device("cpu")

    comparison = verify_streamed_scoring(
        model,
        tokenizer,
        "1 2 3",
        "4 5",
        device,
        chunk_tokens=2,
    )
    score = score_target_continuation_streamed(
        model,
        tokenizer,
        "1 2 3",
        "4 5",
        device,
        chunk_tokens=2,
    )
    generation = greedy_generate_streamed(
        model,
        tokenizer,
        "1 2 3",
        device,
        max_new_tokens=5,
        chunk_tokens=2,
    )

    assert comparison["absolute_difference"] == 0
    assert score["target_tokens"] == 2
    assert generation == {
        "text": "4 5",
        "tokens": 3,
        "finish_reason": "eos_or_stop",
        "truncated": False,
        "streaming_chunk_tokens": 2,
    }


def test_diagnostic_gold_is_executable(tmp_path) -> None:
    for index, family in enumerate(("constant", "enum", "signature", "field", "config")):
        case = build_diagnostic_family(
            tokenizer=RegexTokenizer(),
            family=family,
            target_tokens=768,
            seed=100 + index,
            tolerance_fraction=0.1,
        )[1]
        benchmark = diagnostic_case_to_benchmark(case)
        result = evaluate_prediction(
            benchmark,
            Prediction(case_id=case.id, completion=case.target),
            work_root=tmp_path / family,
            execution_backend="trusted-host",
        )

        assert result.parse.status == "pass", family
        assert result.compile.status == "pass", family
        assert result.test.status == "pass", family
