"""Git synthetic trajectories: temp-repo extraction, determinism, provenance."""

import os
import subprocess

from tinycomplete.data.git_edits import (
    NOT_HUMAN_ORDER_NOTE,
    extract_commit_pair,
    mine_linear_history,
)
from tinycomplete.data.schema import Provenance


def _git(repo: str, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


def _fixture_repo(tmp_path) -> tuple[str, str, str]:
    repo = str(tmp_path / "repo")
    os.makedirs(repo)
    _git(repo, "init", "-q")
    with open(os.path.join(repo, "a.py"), "w") as f:
        f.write("import os\n\n\ndef f(x):\n    return x\n")
    with open(os.path.join(repo, "notes.txt"), "w") as f:
        f.write("not python\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "parent")
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    with open(os.path.join(repo, "a.py"), "w") as f:
        f.write("import os\nimport sys\n\n\ndef f(x):\n    y = x + 1\n    return y\n")
    with open(os.path.join(repo, "b.py"), "w") as f:
        f.write("CONST = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "child")
    child = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    return repo, parent, child


def test_extract_pair(tmp_path):
    repo, parent, child = _fixture_repo(tmp_path)
    edits = extract_commit_pair(repo, parent, child)
    paths = sorted(e.path for e in edits)
    assert paths == ["a.py", "b.py"], f"non-python files must be skipped, got {paths}"
    by_path = {e.path: e for e in edits}
    a = by_path["a.py"]
    assert a.provenance == Provenance.GIT_SYNTHETIC
    assert a.note == NOT_HUMAN_ORDER_NOTE
    assert len(a.hunks) >= 1
    assert any(h.added for h in a.hunks)
    assert any(h.removed for h in a.hunks)
    # enclosing syntax nodes identified for the function-body hunk
    assert any("function_definition" in h.enclosing for h in a.hunks)
    # intermediates converge on the child text
    assert a.intermediates, "expected synthetic intermediate states"
    assert a.intermediates[-1] == a.child_text
    assert a.intermediates[0] != a.parent_text
    # future changes exposed as labels
    assert any("y = x + 1" in line for h in a.hunks for line in h.added)


def test_deterministic_extraction(tmp_path):
    repo, parent, child = _fixture_repo(tmp_path)
    first = extract_commit_pair(repo, parent, child)
    second = extract_commit_pair(repo, parent, child)
    assert first == second


def test_mine_linear_history(tmp_path):
    repo, parent, child = _fixture_repo(tmp_path)
    edits = mine_linear_history(repo)
    assert {e.path for e in edits} == {"a.py", "b.py"}
    assert all(e.provenance.value == "git_synthetic" for e in edits)
