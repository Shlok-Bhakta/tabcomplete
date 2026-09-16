"""Budget gate: caps, persistence, estimates, paid flag."""

import json
import os

import pytest

from tinycomplete.teacher.budget import (
    SAFETY_MARGIN,
    Budget,
    estimate_cost_usd,
)


def test_paid_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_PAID_SYNTHETIC", raising=False)
    budget = Budget(path=str(tmp_path / "b.json"))
    assert Budget.paid_enabled() is False
    assert budget.can_spend(0.01) is False


def test_caps_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_SYNTHETIC", "1")
    budget = Budget(path=str(tmp_path / "b.json"), spend_cap=1.0, example_cap=2)
    assert budget.can_spend(0.5, 1) is True
    budget.record(0.6, 1)
    assert budget.can_spend(0.5, 1) is False  # spend cap
    budget2 = Budget(path=str(tmp_path / "b2.json"), spend_cap=100.0, example_cap=1)
    budget2.record(0.01, 1)
    assert budget2.can_spend(0.01, 1) is False  # example cap


def test_persistence_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_SYNTHETIC", "1")
    path = str(tmp_path / "b.json")
    Budget(path=path).record(0.25, 3)
    reopened = Budget(path=path)
    assert reopened.state.spent_usd == pytest.approx(0.25)
    assert reopened.state.paid_examples == 3
    raw = json.load(open(path, encoding="utf-8"))
    assert set(raw) == {"spent_usd", "paid_examples", "updated_at"}


def test_estimate_adds_margin():
    exact = 1000 / 1000 * 0.01 + 500 / 1000 * 0.03
    assert estimate_cost_usd(1000, 500, 0.01, 0.03) == pytest.approx(exact * SAFETY_MARGIN)
    assert SAFETY_MARGIN >= 1.5


def test_from_env_defaults(monkeypatch):
    for var in ("MAX_TOTAL_SPEND_USD", "MAX_PAID_EXAMPLES", "MAX_CANDIDATES_PER_STATE"):
        monkeypatch.delenv(var, raising=False)
    budget = Budget.from_env(path=os.path.join("nowhere", "b.json"))
    assert budget.spend_cap == 2.00
    assert budget.example_cap == 2000
    assert budget.candidate_cap == 3
