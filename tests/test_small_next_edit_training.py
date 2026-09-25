import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_small_next_edit import EFFECTIVE_BATCH, encode_rows  # noqa: E402


class TinyTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [ord(character) % 90 + 1 for character in text]
        return [98, *ids] if add_special_tokens else ids


def test_response_and_eos_only_are_supervised() -> None:
    rows = [{"prompt": "abc", "response": "R\nx"}, {"prompt": "é", "response": "N\n"}]
    encoded, counts = encode_rows(TinyTokenizer(), rows)
    assert counts["examples"] == 2
    assert counts["supervised_response_and_eos_tokens"] == 7
    for item in encoded:
        assert item["labels"][: item["prompt_tokens"]] == [-100] * item["prompt_tokens"]
        assert item["labels"][-1] == 99
        assert item["target_tokens"] > 1


def test_example_weighted_accumulation_matches_full_update() -> None:
    torch = pytest.importorskip("torch")
    assert EFFECTIVE_BATCH == 16
    observations = [torch.tensor([1.0, float(index % 4)]) for index in range(16)]
    targets = [torch.tensor(float(index % 3)) for index in range(16)]
    first = torch.nn.Linear(2, 1, bias=False)
    second = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        first.weight.fill_(0.25)
        second.weight.copy_(first.weight)
    for x, y in zip(observations, targets, strict=True):
        ((first(x).squeeze() - y) ** 2 / 16).backward()
    torch.stack(
        [(second(x).squeeze() - y) ** 2 for x, y in zip(observations, targets, strict=True)]
    ).mean().backward()
    assert torch.allclose(first.weight.grad, second.weight.grad, rtol=1e-6, atol=1e-6)
