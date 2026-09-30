from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_fim_line_probe.py"


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    spec = importlib.util.spec_from_file_location("fim_line_probe", PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_preserves_unicode_whitespace_and_excludes_reference(probe):
    case = {"source_before": "λ = ", "source_after": "\n\tprint(λ)\n", "reference": "SECRET_GOLD"}
    assert probe.prompt(case, "raw_prefix") == "λ = "
    assert probe.prompt(case, "fim_psm") == (
        "<|fim_prefix|>λ = <|fim_suffix|>\n\tprint(λ)\n<|fim_middle|>"
    )
    assert "SECRET_GOLD" not in probe.prompt(case, "fim_psm")
    with pytest.raises(ValueError, match="unknown prompt"):
        probe.prompt(case, "chat")


def test_selection_is_fixed_before_outputs_and_budget_bounded(probe):
    rows = [
        {"language": str(language), "id": f"{language}/{i}"}
        for language in range(9)
        for i in range(3)
    ]
    selected = probe.selected_cases(rows)
    assert [r["id"] for r in selected] == [
        f"{language}/{i}" for language in range(9) for i in range(2)
    ]
    assert len(selected) * 2 * 2 == probe.MAX_REQUESTS == 72
    assert probe.MAX_SECONDS == 900
    with pytest.raises(ValueError, match="nine languages"):
        probe.selected_cases(rows[:3])


def test_overflow_never_inherits_previous_request_outcome(probe):
    class Provider:
        last = {"raw_response": "previous request must not appear here"}

        def generate_line_detailed(self, *args):
            pytest.fail("over-budget prompt must not run inference")

    record = probe.generate_record({"id": "too-long"}, "fim_psm", 1, 2977, Provider())
    assert record == {
        "case_id": "too-long",
        "policy": "fim_psm",
        "repetition": 1,
        "input_tokens": 2977,
        "status": "input_budget_skipped",
    }
