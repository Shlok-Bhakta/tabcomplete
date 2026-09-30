from __future__ import annotations

import runpy
from dataclasses import asdict
from pathlib import Path

import pytest

from tinycomplete.eval.code_benchmark import BenchmarkResult, CheckResult
from tinycomplete.one_line.contract import EditAction, EditState, apply_action, encode_action

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/evaluate_functional_pilot_predictions.py")
)


def _case(kind: str = "keep"):
    state = EditState("solution.py", "python", "value = 1\n", 0, 9)
    action = EditAction(kind)
    row = {
        "id": "case",
        "state": asdict(state),
        "action": asdict(action),
        "after_source": apply_action(state, action),
    }
    prediction = {
        "id": "case",
        "state_sha256": MODULE["state_digest"](row["state"]),
        "source_sha256": MODULE["digest"](state.source.encode()),
        "wire": encode_action(action),
        "terminated_by_eos": True,
        "generated_tokens": 2,
        "canonical_action": asdict(action),
    }
    fixture = {
        "id": "case",
        "test_source": "assert True\n",
        "fixture_sha256": MODULE["digest"](b"assert True\n"),
        "gold_after_source_sha256": MODULE["digest"](row["after_source"].encode()),
        "image_identity": "python:3.11@sha256:" + "a" * 64,
    }
    return row, prediction, fixture


def _success(case, prediction, *, work_root, execution_backend):
    assert execution_backend == "container"
    return BenchmarkResult(
        case_id=case.id,
        language="python",
        category="test",
        repository_context=False,
        exact_match=False,
        normalized_exact_match=False,
        nonempty=True,
        compile_configured=True,
        test_configured=True,
        parse=CheckResult(status="pass"),
        compile=CheckResult(status="pass"),
        test=CheckResult(status="pass"),
        working_tree_sha256="a" * 64,
    )


def test_always_keep_cannot_win_edit_required(tmp_path):
    row, prediction, fixture = _case("delete_line")
    keep = EditAction("keep")
    prediction.update(wire=encode_action(keep), canonical_action=asdict(keep))

    def evaluate(case, prediction, **kwargs):
        result = _success(case, prediction, **kwargs)
        return result.model_copy(update={"test": CheckResult(status="fail")})

    result = MODULE["score_predictions"](
        [row], [prediction], [fixture], work_root=tmp_path, evaluate=evaluate
    )
    assert result["counts"]["edit_required_exact"] == 0
    assert result["counts"]["functional_success"] == 0
    assert result["counts"]["nonempty_edit"] == 0


def test_gratuitous_behavior_preserving_edit_is_not_no_edit_recall(tmp_path):
    row, prediction, fixture = _case()
    action = EditAction("replace_line", "value = 1  # unnecessary")
    prediction.update(wire=encode_action(action), canonical_action=asdict(action))
    result = MODULE["score_predictions"](
        [row], [prediction], [fixture], work_root=tmp_path, evaluate=_success
    )
    assert result["counts"]["functional_success"] == 0
    assert result["counts"]["no_edit_false_edits"] == 1
    assert result["counts"]["no_edit_recalled"] == 0


def test_cutoff_never_executes_partial_parseable_action(tmp_path):
    row, prediction, fixture = _case()
    prediction.update(terminated_by_eos=False, generated_tokens=64, canonical_action=None)

    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete action was executed")

    result = MODULE["score_predictions"](
        [row], [prediction], [fixture], work_root=tmp_path, evaluate=forbidden
    )
    assert result["counts"]["valid"] == 0
    assert result["counts"]["functional_success"] == 0


@pytest.mark.parametrize("mutation", ["state", "fixture", "canonical", "coverage", "duplicate"])
def test_identity_or_coverage_mismatch_fails_closed(tmp_path, mutation):
    row, prediction, fixture = _case()
    fixtures = [fixture]
    predictions = [prediction]
    if mutation == "state":
        prediction["source_sha256"] = "b" * 64
    elif mutation == "fixture":
        fixture["test_source"] += "assert False\n"
    elif mutation == "canonical":
        prediction["canonical_action"] = {"kind": "delete_line", "text": None}
    elif mutation == "coverage":
        fixtures = []
    else:
        predictions.append(prediction)
    with pytest.raises(ValueError):
        MODULE["score_predictions"](
            [row], predictions, fixtures, work_root=tmp_path, evaluate=_success
        )
