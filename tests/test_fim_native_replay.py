from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_small_model import build_fim_latency_states, fim_replay_schedule


class FIMCharTokenizer:
    markers = (
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
    )

    def encode(self, text, add_special_tokens=False):
        ids = []
        index = 0
        while index < len(text):
            marker = next((item for item in self.markers if text.startswith(item, index)), None)
            if marker is not None:
                ids.append(100 + self.markers.index(marker))
                index += len(marker)
            else:
                ids.append(ord(text[index]) + 1000)
                index += 1
        return ids

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {
            "input_ids": [ord(char) + 1000 for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _case(case_id: str, repository: str, filename: str, filler: str) -> dict:
    from evaluate_q25_fim_native import expected_context

    tokenizer = FIMCharTokenizer()
    source = filler * 2600 + "\nvalue = old\n" + filler[::-1] * 2600 + "\n"
    case = {
        "case_id": case_id,
        "repository": repository,
        "state": {
            "file_id": filename,
            "filetype": "python",
            "source": source,
            "target_row": 1,
            "cursor_col": len(b"value = "),
            "history": [],
            "relevant": [],
        },
    }
    expected = expected_context(case, tokenizer)
    case["prompt"] = expected["prompt"]
    case["prompt_token_ids"] = tokenizer.encode(case["prompt"])
    return case


@pytest.fixture
def latency_states():
    return build_fim_latency_states(
        [
            _case("public-a", "a" * 64, "src/a.py", "p"),
            _case("public-b", "b" * 64, "src/b.py", "s"),
        ],
        FIMCharTokenizer(),
    )


def test_fim_latency_suite_has_24_bounded_source_states_and_required_transitions(latency_states):
    assert len(latency_states) == 24
    assert len({row["id"] for row in latency_states}) == 24
    for target in (512, 1024, 2048):
        rows = [row for row in latency_states if row["context_target_tokens"] == target]
        assert len(rows) == 8
        assert all(abs(row["full_context_tokens"] - target) <= 16 for row in rows)
        assert all(row["retained_prefix_tokens"] <= 640 for row in rows)
        assert all(row["retained_suffix_tokens"] <= 256 for row in rows)
        if target > 899:
            assert all(row["retained_prefix_tokens"] == 640 for row in rows)
            assert all(row["retained_suffix_tokens"] == 256 for row in rows)
        assert {row["operation"] for row in rows} == {
            "fresh_open",
            "append_utf8_at_cursor",
            "near_replace_before_cursor",
            "earlier_edit",
            "divergent_typing",
            "matching_prefix_suffix_change",
            "switch_A_to_B",
            "return_B_to_A",
        }


def test_utf8_cursor_and_matching_prefix_and_a_b_a_transition_are_exact(latency_states):
    for target in (512, 1024, 2048):
        rows = [row for row in latency_states if row["context_target_tokens"] == target]
        baseline, appended, _, _, _, matching, switched, returned = rows
        assert appended["state"]["cursor_col"] == baseline["state"]["cursor_col"] + len(
            "λ".encode()
        )
        assert "λ" in appended["state"]["source"]
        assert matching["prompt_sha256"] != baseline["prompt_sha256"]
        matching_prefix = matching["state"]["source"].encode()[
            : matching["prefix_range"]["end_byte"]
        ]
        baseline_prefix = baseline["state"]["source"].encode()[
            : baseline["prefix_range"]["end_byte"]
        ]
        assert matching_prefix == baseline_prefix
        assert switched["repository"] != matching["repository"]
        assert returned["repository"] == matching["repository"]
        assert returned["source_sha256"] == matching["source_sha256"]
        assert returned["prompt_sha256"] == matching["prompt_sha256"]


def test_suite_is_deterministic_and_requires_distinct_public_sources():
    tokenizer = FIMCharTokenizer()
    cases = [
        _case("public-a", "a" * 64, "src/a.py", "p"),
        _case("public-b", "b" * 64, "src/b.py", "s"),
    ]
    first = build_fim_latency_states(cases, tokenizer)
    second = build_fim_latency_states(list(reversed(cases)), tokenizer)
    assert first == second
    with pytest.raises(ValueError, match="distinct public repositories"):
        build_fim_latency_states(
            [cases[0], {**cases[1], "repository": cases[0]["repository"]}], tokenizer
        )


def test_replay_schedule_separates_fresh_pass_and_pairs_each_cached_repeat(latency_states):
    schedule = fim_replay_schedule(latency_states)
    assert len(schedule) == 144

    fresh = schedule[:48]
    expected_state_ids = [row["id"] for row in latency_states]
    assert [row[1]["id"] for row in fresh[:24]] == expected_state_ids
    assert [row[1]["id"] for row in fresh[24:]] == expected_state_ids
    assert all(row[0] == 0 for row in fresh[:24])
    assert all(row[0] == 1 for row in fresh[24:])
    assert all(
        condition == "fresh_state_cache_off" and not cached for _, _, condition, cached in fresh
    )

    cached = schedule[48:]
    previous_changed_prompt = None
    for index in range(0, len(cached), 2):
        changed, repeat = cached[index : index + 2]
        assert changed[0] == repeat[0] == index // 48
        assert changed[1]["id"] == expected_state_ids[(index // 2) % 24]
        assert changed[1] is repeat[1]
        assert changed[2:] == ("changed_editor_state_cache_on", True)
        assert repeat[2:] == ("identical_prompt_repeat_cache_on", True)
        assert repeat[1]["prompt_sha256"] == changed[1]["prompt_sha256"]
        if previous_changed_prompt is not None:
            assert changed[1]["prompt_sha256"] != previous_changed_prompt
        previous_changed_prompt = changed[1]["prompt_sha256"]
        if index + 2 < len(cached):
            assert repeat[1]["id"] != cached[index + 2][1]["id"]
