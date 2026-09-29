"""Check structural Git yield rules without accepting inferred editor history."""

import runpy
from pathlib import Path

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/probe_active_git_sequence_yield.py")
)


def test_repeated_identifier_pair() -> None:
    atoms = MODULE["_atoms"]("a = alpha\nb = alpha\n", "a = beta\nb = beta\n")
    assert MODULE["_pairs"](atoms) == [(0, 1, ("alpha", "beta"))]


def test_formatting_only_change_has_no_pair() -> None:
    atoms = MODULE["_atoms"]("\ta = alpha\n\tb = alpha\n", "    a = alpha\n    b = alpha\n")
    assert MODULE["_pairs"](atoms) == []


def test_pair_must_be_within_eighty_rows() -> None:
    old = "a = alpha\n" + "middle\n" * 80 + "b = alpha\n"
    new = "a = beta\n" + "middle\n" * 80 + "b = beta\n"
    assert MODULE["_pairs"](MODULE["_atoms"](old, new)) == []
