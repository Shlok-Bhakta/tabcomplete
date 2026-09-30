from __future__ import annotations

import importlib.util
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_sweep_comparison.py"
_SPEC = importlib.util.spec_from_file_location("score_sweep_comparison", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
scorer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scorer)


def test_frozen_campaign_plan_and_all_source_inputs_validate() -> None:
    campaign = scorer._load_plan(scorer.CAMPAIGN_PLAN_DEFAULT)
    strict, line, next_edit, diagnostic, prompts = scorer._validate_inputs(
        campaign,
        scorer.STRICT_SUITE_DEFAULT,
        scorer.LINE_SUITE_DEFAULT,
        scorer.NEXT_EDIT_SUITE_DEFAULT,
        scorer.DIAGNOSTIC_PLAN_DEFAULT,
        scorer.NEXT_EDIT_INPUTS_DEFAULT,
    )

    assert len(strict) == 200
    assert len(line) == 180
    assert len(next_edit) == 200
    assert len(diagnostic["fixtures"]) == len(prompts) == 24
    assert (
        sum(fixture.get("source_case_id") is not None for fixture in diagnostic["fixtures"]) == 22
    )


def test_full_file_mapping_is_byte_exact_and_rejects_out_of_range_edits() -> None:
    current = "éx\n"
    valid = scorer.map_full_file(current, "évalue\n", 2, 3)
    assert valid == {
        "mapping": "within_editable_range",
        "out_of_range": False,
        "action": "replace",
        "replacement": "value",
    }

    assert scorer.map_full_file(current, "zvalue\n", 2, 3)["mapping"] == "out_of_range"
    assert scorer.map_full_file(current, "évalue\n", 1, 3)["mapping"] == ("invalid_utf8_boundary")


def test_podman_hex_image_id_is_normalized_to_immutable_reference() -> None:
    digest = "f5699e5c9724cd5bc9f639ba77608e169e8fb5675e230882683bc691dfc44622"
    assert scorer._canonical_image_id(digest) == "sha256:" + digest
    assert scorer._canonical_image_id("sha256:" + digest) == "sha256:" + digest
    with pytest.raises(ValueError):
        scorer._canonical_image_id("latest")


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"explicit_terminal": True, "finish_reason": "eos", "hit_token_cap": False}, True),
        ({"explicit_terminal": True, "finish_reason": "word", "hit_token_cap": False}, True),
        ({"explicit_terminal": False, "finish_reason": "eos", "hit_token_cap": False}, False),
        ({"explicit_terminal": True, "finish_reason": "length", "hit_token_cap": True}, False),
        ({"explicit_terminal": True, "finish_reason": "eos", "hit_token_cap": True}, False),
    ],
)
def test_next_edit_requires_actual_non_cutoff_terminal(
    row: dict[str, object], expected: bool
) -> None:
    assert scorer._row_terminal_valid(row) is expected


def test_case_cluster_bootstrap_is_paired_deterministic_and_reports_missing_pairs() -> None:
    q8 = {"a": 0.0, "b": 1.0, "c": None}
    q4 = {"a": 1.0, "b": 0.0, "c": 1.0}
    first = scorer._pairwise_bootstrap(q8, q4)
    second = scorer._pairwise_bootstrap(q8, q4)

    assert first == second
    assert first["n_paired_cases"] == 2
    assert first["n_skipped"] == 1
    assert first["q4_minus_q8"] == 0.0
    assert first["interpretation"] == "descriptive uncertainty interval; not an equivalence test"


def test_adapted_case_uses_observed_current_state_and_original_source_check() -> None:
    campaign = scorer._load_plan(scorer.CAMPAIGN_PLAN_DEFAULT)
    _, _, source_cases, diagnostic, prompts = scorer._validate_inputs(
        campaign,
        scorer.STRICT_SUITE_DEFAULT,
        scorer.LINE_SUITE_DEFAULT,
        scorer.NEXT_EDIT_SUITE_DEFAULT,
        scorer.DIAGNOSTIC_PLAN_DEFAULT,
        scorer.NEXT_EDIT_INPUTS_DEFAULT,
    )
    fixture = next(row for row in diagnostic["fixtures"] if row.get("source_case_id"))
    prompt_row = next(row for row in prompts if row["case_id"] == fixture["case_id"])
    source = next(case for case in source_cases if case.id == fixture["source_case_id"])
    image_ids = {
        case.check.container_image: "sha256:test"
        for case in source_cases
        if case.check.container_image
    }

    adapted = scorer._adapt_source_case(source, prompt_row, fixture, image_ids)
    assert adapted.current == prompt_row["current_content"]
    assert (
        adapted.region_text
        == prompt_row["current_content"]
        .encode()[prompt_row["editable_start_byte"] : prompt_row["editable_end_byte"]]
        .decode()
    )
    assert adapted.expected == source.expected
    assert adapted.check == source.check.model_copy(
        update={"container_image": image_ids.get(source.check.container_image)}
    )


def test_incomplete_result_is_unscored_and_raw_response_is_not_in_score_row() -> None:
    current = "x = 1\n"
    raw = "x = 2"
    prompt_row = {
        "current_content": current,
        "editable_start_byte": 4,
        "editable_end_byte": 5,
    }
    fixture = {
        "case_id": "python/example",
        "gold_action": "replace",
        "gold_text": "2",
        "target_assessment": {"functional_intent": "clear_functional_intent"},
    }
    row = {
        "raw_output": raw,
        "output_sha256": scorer.digest_bytes(raw.encode()),
        "context_eligible": True,
        "explicit_terminal": False,
        "finish_reason": "length",
        "hit_token_cap": True,
        "editable_range_mapping": {
            "mapping": "incomplete_or_unterminated_output_not_scored",
            "out_of_range": None,
        },
        "completed_response_ms": 5.0,
        "input_tokens": 7,
        "output_tokens": 3,
        "observability_ids": {"request_id": "synthetic-request"},
    }

    scored = scorer._score_next_edit_result(
        row=row,
        prompt_row=prompt_row,
        fixture=fixture,
        source_case=None,
        precision="q4_k_m",
        repetition=0,
        case_index=0,
        temp_root=Path("."),
        run_context=scorer.RunContext.new(campaign_id="score-test"),
    )

    assert scored["terminal_valid"] is False
    assert scored["predicted_action"] is None
    assert scored["clear_intent_functional_success"] is False
    assert "raw_output" not in scored
    assert scored["response_sha256"] == row["output_sha256"]


def test_over_context_case_counts_as_unavailable_not_as_a_prediction() -> None:
    fixture = {
        "case_id": "python/example",
        "gold_action": "no_edit",
        "gold_text": "",
        "target_assessment": {"functional_intent": "no_edit_plausible"},
    }
    row = {
        "context_eligible": False,
        "input_tokens": 8000,
        "observability_ids": None,
        "editable_range_mapping": {
            "mapping": "incomplete_or_unterminated_output_not_scored",
            "out_of_range": None,
        },
    }

    scored = scorer._score_next_edit_result(
        row=row,
        prompt_row={
            "current_content": "x = 1\n",
            "editable_start_byte": 4,
            "editable_end_byte": 5,
        },
        fixture=fixture,
        source_case=None,
        precision="q8_0",
        repetition=0,
        case_index=0,
        temp_root=Path("."),
        run_context=scorer.RunContext.new(campaign_id="score-test"),
    )

    assert scored["context_eligible"] is False
    assert scored["terminal_valid"] is False
    assert scored["predicted_action"] is None
    assert scored["no_edit_agreement"] is None
    assert scored["mapping_status"] == "context_ineligible_not_generated"
    assert scored["observability_ids"] is None
    scorer._validate_request_correlation(row, fixture["case_id"], request_expected=False)


def test_request_ids_are_mandatory_only_when_the_worker_made_a_request() -> None:
    case_id = "python/example"
    scorer._validate_request_correlation(
        {"observability_ids": {"case_id": case_id, "request_id": "request-1"}},
        case_id,
        request_expected=True,
    )
    scorer._validate_request_correlation(
        {"observability_ids": None}, case_id, request_expected=False
    )

    with pytest.raises(ValueError, match="missing its request correlation"):
        scorer._validate_request_correlation(
            {"observability_ids": None}, case_id, request_expected=True
        )
    with pytest.raises(ValueError, match="fabricated request correlation"):
        scorer._validate_request_correlation(
            {"observability_ids": {"case_id": case_id, "request_id": "fake"}},
            case_id,
            request_expected=False,
        )


def test_sigterm_after_case_checkpoint_preserves_row_and_marks_missing_unknown(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".attempt.incomplete"
    final = tmp_path / "attempt"
    raw = tmp_path / "raw-results"
    staging.mkdir()
    raw.mkdir()
    strict_cases = [
        scorer.BenchmarkCase(
            id="strict/done", language="python", path="a.py", prefix="", expected="x"
        ),
        scorer.BenchmarkCase(
            id="strict/pending", language="python", path="b.py", prefix="", expected="x"
        ),
    ]
    line_cases = [{"id": "line/not-reached"}]
    prompt_rows = [{"case_id": "edit/not-reached"}]
    checkpoint = staging / "strict" / "q8_0" / "results.jsonl"
    scorer._append_jsonl(
        checkpoint,
        {
            "case_id": "strict/done",
            "precision": "q8_0",
            "context_eligible": False,
            "scored": False,
        },
    )

    with pytest.raises(scorer.ScoringInterrupted):
        scorer._scoring_sigterm(signal.SIGTERM, None)

    plan = {
        "scoring_plan_sha256": "frozen-test-plan",
        "campaign": {"embedded_plan_sha256": "frozen-test-campaign"},
    }
    summary = scorer._preserve_interrupted_output(
        staging_root=staging,
        final_output_root=final,
        results_root=raw,
        plan=plan,
        strict_cases=strict_cases,
        line_cases=line_cases,
        prompt_rows=prompt_rows,
    )

    persisted = scorer._read_jsonl(final / "strict" / "q8_0" / "results.jsonl")
    assert persisted == [
        {
            "case_id": "strict/done",
            "precision": "q8_0",
            "context_eligible": False,
            "scored": False,
        }
    ]
    assert summary["state"] == "interrupted_partial"
    coverage = summary["partial_coverage"]
    assert coverage["completed_record_total"] == 1
    assert coverage["unknown_unscored_total"] == 2 * (2 + 1 + 2) - 1
    q8_strict = coverage["tasks"]["strict/q8_0"]
    assert q8_strict["known_unscored_ids"] == ["strict/done"]
    assert q8_strict["unknown_unscored_ids"] == ["strict/pending"]
    assert json.loads((final / "artifact_manifest.json").read_text())["state"] == (
        "interrupted_partial"
    )


def test_sigterm_controlled_exception_bypasses_evaluator_exception_handlers() -> None:
    assert issubclass(scorer.ScoringInterrupted, KeyboardInterrupt)
    try:
        scorer._scoring_sigterm(signal.SIGTERM, None)
    except Exception:
        pytest.fail("SIGTERM was swallowed by an ordinary evaluator exception handler")
    except scorer.ScoringInterrupted:
        pass


def test_unexpected_error_preserves_checkpoint_without_message_or_traceback(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".failed.incomplete"
    final = tmp_path / "failed"
    raw = tmp_path / "raw-results"
    staging.mkdir()
    raw.mkdir()
    strict_cases = [
        scorer.BenchmarkCase(
            id="strict/done", language="python", path="a.py", prefix="", expected="x"
        )
    ]
    checkpoint = staging / "strict" / "q8_0" / "results.jsonl"
    row = {"case_id": "strict/done", "precision": "q8_0", "scored": True}
    scorer._append_jsonl(checkpoint, row)

    scorer._preserve_failed_output(
        staging_root=staging,
        final_output_root=final,
        results_root=raw,
        plan={
            "scoring_plan_sha256": "frozen-test-plan",
            "campaign": {"embedded_plan_sha256": "frozen-test-campaign"},
        },
        strict_cases=strict_cases,
        line_cases=[],
        prompt_rows=[],
        error=ValueError("private prompt text must not be persisted"),
    )

    failure = json.loads((final / "failure_metadata.json").read_text())
    assert failure == {
        "schema": "sweep-comparison-failure-metadata-v1",
        "state": "failed_partial",
        "exception_type": "ValueError",
        "exception_message": "omitted",
        "traceback": "omitted",
    }
    assert scorer._read_jsonl(final / "strict" / "q8_0" / "results.jsonl") == [row]
    summary = json.loads((final / "summary.json").read_text())
    assert summary["state"] == "failed_partial"
    assert summary["partial_coverage"]["unknown_unscored_total"] == 1
def test_complete_mapped_action_uses_container_backend_and_preserves_public_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = scorer.NextEditCase(
        id="python/example",
        language="python",
        path="sample.py",
        current="value = 1\n",
        region_start=8,
        region_end=9,
        expected="2",
        action="replace",
    )
    current = source.current
    updated = "value = 2\n"
    captured: dict[str, object] = {}

    def fake_evaluate(case, prediction, *, work_root, execution_backend):
        captured["completion"] = prediction.completion
        captured["execution_backend"] = execution_backend
        return SimpleNamespace(
            parse=SimpleNamespace(status="pass", seconds=0.0),
            compile=SimpleNamespace(status="not_run", seconds=0.0),
            test=SimpleNamespace(status="not_run", seconds=0.0),
            compile_configured=False,
            test_configured=False,
        )

    monkeypatch.setattr(scorer, "evaluate_prediction", fake_evaluate)
    row = {
        "raw_output": updated,
        "output_sha256": scorer.digest_bytes(updated.encode()),
        "context_eligible": True,
        "explicit_terminal": True,
        "finish_reason": "eos",
        "hit_token_cap": False,
        "editable_range_mapping": scorer.map_full_file(current, updated, 8, 9),
        "completed_response_ms": 11.0,
        "input_tokens": 12,
        "output_tokens": 4,
        "observability_ids": {"request_id": "synthetic-request"},
    }
    fixture = {
        "case_id": source.id,
        "gold_action": "replace",
        "gold_text": "2",
        "target_assessment": {"functional_intent": "clear_functional_intent"},
    }

    scored = scorer._score_next_edit_result(
        row=row,
        prompt_row={
            "current_content": current,
            "editable_start_byte": 8,
            "editable_end_byte": 9,
        },
        fixture=fixture,
        source_case=source,
        precision="q4_k_m",
        repetition=0,
        case_index=0,
        temp_root=Path("."),
        run_context=scorer.RunContext.new(campaign_id="score-test"),
    )

    assert scored["terminal_valid"] is True
    assert scored["predicted_action"] == "replace"
    assert scored["exact_reference_match"] is True
    assert scored["clear_intent_functional_success"] is True
    assert scored["functional_oracle_available"] is True
    assert captured == {"completion": "2", "execution_backend": "container"}
    assert "raw_output" not in scored


def test_expected_source_suite_does_not_change_during_adaptation() -> None:
    rows = scorer._read_jsonl(scorer.NEXT_EDIT_INPUTS_DEFAULT)
    assert len(rows) == 24
    assert all("gold_action" not in row and "gold_text" not in row for row in rows)
    assert all(
        row["prompt_sha256"] == scorer.digest_bytes(scorer.build_sweep_prompt(row).encode())
        for row in rows
    )
