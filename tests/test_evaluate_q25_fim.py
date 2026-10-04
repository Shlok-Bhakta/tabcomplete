"""Observed stopping and raw-token integrity are separate from decoded text."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_q25_fim import (  # noqa: E402
    SOURCE_SYNTAX_PROTOCOL,
    _source_syntax_rows,
    decoded_completion,
    development_case,
    run_source_syntax_diagnostic,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_syntax_fixture(source: str = "# π\r\nvalue = 1\r\n"):
    source_bytes = source.encode("utf-8")
    target_bytes = b"value = 1"
    start = source_bytes.index(target_bytes)
    end = start + len(target_bytes)
    repository = _sha("repository")
    aliases = sorted((repository, _sha("repository-alias")))
    prompt_hash = _sha("<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>")
    row = {
        "id": 4096,
        "split": "development",
        "language": "python",
        "mode": "whole_logical_line",
        "prompt_format": "psm",
        "variant": 0,
        "source_content_sha256": _sha(source),
        "repository_identity_sha256": repository,
        "repository_alias_sha256": aliases,
        "source_path": "src/example.py",
        "region_start": start,
        "region_end": end,
        "prompt_sha256": prompt_hash,
        "target_sha256": _sha(target_bytes.decode("utf-8")),
    }
    document = SimpleNamespace(
        content=source,
        content_sha256=row["source_content_sha256"],
        repository_identity_sha256=repository,
        repository_alias_sha256=tuple(aliases),
        language="python",
        path="src/example.py",
    )
    completion = "value = 2 \t"
    prediction = {
        "case_id": "fim-development-4096",
        "repository": repository,
        "context_sha256": prompt_hash,
        "language": "python",
        "raw_response": completion,
        "completion_sha256": _sha(completion),
    }
    return row, document, prediction


class Tokenizer:
    eos_token_id = 9
    all_special_ids = [8, 9]
    pieces = {0: " ", 1: "\t", 2: "λ", 3: "\n", 8: "<|fim_middle|>", 9: "<eos>"}

    def get_vocab(self):
        return {text: index for index, text in self.pieces.items()}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces.get(index, "") for index in ids)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == "<|fim_middle|>λ":
            return [8, 2]
        raise ValueError("unexpected test prompt")


def test_eos_at_ceiling_terminates_without_trimming_unicode_or_whitespace() -> None:
    text, reason, evidence = decoded_completion(
        Tokenizer(), [0, 1, 2, 3, 9], ceiling=5, newline_stop=False
    )
    assert text == " \tλ\n"
    assert reason == "eos"
    assert evidence["ended_by_eos"] is True
    assert evidence["reached_token_ceiling"] is True
    assert evidence["truncated"] is False


@pytest.mark.parametrize("ids", [[8, 2, 9], [2, 9, 9], [2, 99, 9]])
def test_control_or_unknown_ids_are_invalid_even_if_decoded_text_looks_useful(ids) -> None:
    text, reason, evidence = decoded_completion(Tokenizer(), ids, ceiling=96, newline_stop=False)
    assert reason == "invalid_control_or_vocabulary"
    assert evidence["unexpected_special_token_ids"] or evidence["unknown_token_ids"]
    if 8 in ids:
        assert "<|fim_middle|>" in text


def test_regular_added_fim_control_tokens_are_still_invalid_outputs() -> None:
    tokenizer = Tokenizer()
    tokenizer.all_special_ids = [9]
    text, reason, evidence = decoded_completion(
        tokenizer, [8, 2, 9], ceiling=96, newline_stop=False
    )
    assert reason == "invalid_control_or_vocabulary"
    assert text == "<|fim_middle|>λ"
    assert evidence["unexpected_special_token_ids"] == [8]


def test_regular_added_repository_control_token_is_invalid() -> None:
    tokenizer = Tokenizer()
    tokenizer.pieces = {**Tokenizer.pieces, 7: "<|file_sep|>"}
    tokenizer.added_tokens_decoder = {7: object()}
    _, reason, evidence = decoded_completion(tokenizer, [7, 9], ceiling=96, newline_stop=False)
    assert reason == "invalid_control_or_vocabulary"
    assert evidence["unexpected_special_token_ids"] == [7]


def test_a_parsable_cutoff_is_not_an_explicit_stop() -> None:
    text, reason, evidence = decoded_completion(Tokenizer(), [2, 3], ceiling=2, newline_stop=False)
    assert text == "λ\n"
    assert reason == "length"
    assert evidence["truncated"] is True
    assert evidence["ended_by_eos"] is False


def test_registered_newline_stop_and_empty_unverified_output_are_distinct() -> None:
    assert decoded_completion(Tokenizer(), [2, 3], ceiling=2, newline_stop=True)[1] == "newline"
    assert (
        decoded_completion(Tokenizer(), [], ceiling=96, newline_stop=True)[1] == "unverified_stop"
    )


def test_prepared_ids_reconstruct_exact_prompt_target_and_repository_identity() -> None:
    row = {
        "id": 12,
        "input_ids": [8, 2, 0, 1, 2, 3, 9],
        "prompt_tokens": 2,
        "prompt_sha256": hashlib.sha256("<|fim_middle|>λ".encode()).hexdigest(),
        "target_sha256": hashlib.sha256(" \tλ\n".encode()).hexdigest(),
        "repository_identity_sha256": "repository digest",
    }
    case = development_case(row, Tokenizer())
    assert case["id"] == "fim-development-12"
    assert case["prompt"] == "<|fim_middle|>λ"
    assert case["target"] == " \tλ\n"
    assert case["repository"] == "repository digest"
    row["target_sha256"] = "wrong"
    with pytest.raises(ValueError, match="exact FIM strings"):
        development_case(row, Tokenizer())


def test_development_target_requires_observed_ground_truth_eos() -> None:
    with pytest.raises(ValueError, match="declared EOS"):
        development_case({"input_ids": [2], "prompt_tokens": 0}, Tokenizer())


def test_paired_completion_bootstrap_preserves_repository_groups_and_case_weights() -> None:
    from evaluate_q25_fim import paired_development

    first = [
        {"case_id": "a", "repository": "repo1", "context_sha256": "1", "exact": False},
        {"case_id": "b", "repository": "repo1", "context_sha256": "2", "exact": False},
        {"case_id": "c", "repository": "repo2", "context_sha256": "3", "exact": True},
    ]
    second = [{**row, "exact": not row["exact"]} for row in first]
    result = paired_development(first, second, metric="exact")
    assert result["repository_groups"] == 2
    assert result["cases"] == 3
    assert result["wins"] == ["a", "b"]
    assert result["losses"] == ["c"]
    assert result["difference_second_minus_first"] == pytest.approx(1 / 3)
    assert result["paired_repository_bootstrap_95ci"] == [-1.0, 1.0]
    assert paired_development(first, second, metric="exact") == result


@pytest.mark.parametrize("failure", ["duplicate", "changed_context", "changed_repo", "non_boolean"])
def test_paired_completion_rejects_unmatched_or_ambiguous_evidence(failure: str) -> None:
    from evaluate_q25_fim import paired_development

    first = [{"case_id": "a", "repository": "repo", "context_sha256": "1", "exact": True}]
    second = [dict(first[0])]
    if failure == "duplicate":
        second.append(dict(second[0]))
    elif failure == "changed_context":
        second[0]["context_sha256"] = "different"
    elif failure == "changed_repo":
        second[0]["repository"] = "other"
    else:
        second[0]["exact"] = 1
    with pytest.raises(ValueError):
        paired_development(first, second, metric="exact")


def test_source_syntax_splices_verbatim_utf8_and_crlf_and_keeps_only_hashes() -> None:
    row, document, prediction = _source_syntax_fixture()
    parsed: list[str] = []

    def parse_status(source: str, language: str, role: str) -> str:
        assert language == "python"
        parsed.append(source)
        return "pass" if role == "original" else "fail"

    records, summary = _source_syntax_rows(
        [row], [prediction], [document], parse_status=parse_status
    )
    assert parsed == ["# π\r\nvalue = 1\r\n", "# π\r\nvalue = 2 \t\r\n"]
    assert records[0]["original_source_parse_status"] == "pass"
    assert records[0]["generated_source_parse_status"] == "fail"
    assert records[0]["parser_regression"] is True
    assert summary["parser_regression_cases"] == 1
    assert summary["parser_regression_denominator"] == 1
    serialized = json.dumps({"records": records, "summary": summary})
    assert "value =" not in serialized
    assert "raw_response" not in records[0]


def test_source_syntax_uses_original_parse_status_as_control_for_invalid_input() -> None:
    row, document, prediction = _source_syntax_fixture()
    records, summary = _source_syntax_rows(
        [row],
        [prediction],
        [document],
        parse_status=lambda _source, _language, role: "fail" if role == "original" else "pass",
    )
    assert records[0]["original_source_parse_status"] == "fail"
    assert records[0]["generated_source_parse_status"] == "pass"
    assert records[0]["parser_regression"] is None
    assert summary["parser_regression_denominator"] == 0
    assert summary["parser_regression_cases"] == 0


def test_source_syntax_uses_tsx_grammar_and_reports_unavailable_explicitly() -> None:
    row, document, prediction = _source_syntax_fixture()
    row["language"] = "typescript"
    row["source_path"] = "src/component.tsx"
    document.language = "typescript"
    document.path = "src/component.tsx"
    prediction["language"] = "typescript"
    observed_grammars: list[str] = []

    def unavailable(_source: str, language: str, _role: str) -> str:
        observed_grammars.append(language)
        return "unavailable"

    records, summary = _source_syntax_rows(
        [row], [prediction], [document], parse_status=unavailable
    )
    assert observed_grammars == ["tsx", "tsx"]
    assert records[0]["parser_grammar"] == "tsx"
    assert records[0]["original_source_parse_status"] == "unavailable"
    assert records[0]["generated_source_parse_status"] == "unavailable"
    assert summary["original_source_parse_status_counts"] == {"unavailable": 1}
    assert summary["parser_regression_denominator"] == 0


@pytest.mark.parametrize(
    ("region_start", "region_end"),
    [(3, 11), (0, 10_000), (9, 8)],
)
def test_source_syntax_rejects_invalid_utf8_or_out_of_bounds_regions(
    region_start: int, region_end: int
) -> None:
    row, document, prediction = _source_syntax_fixture()
    row["region_start"] = region_start
    row["region_end"] = region_end
    with pytest.raises(ValueError, match="byte region|UTF-8"):
        _source_syntax_rows([row], [prediction], [document], parse_status=lambda *_: "pass")


def test_source_syntax_rejects_source_hash_mismatch() -> None:
    row, document, prediction = _source_syntax_fixture()
    row["source_content_sha256"] = _sha("different source")
    with pytest.raises(ValueError, match="identity differs"):
        _source_syntax_rows([row], [prediction], [document], parse_status=lambda *_: "pass")


@pytest.mark.parametrize("field", ["case_id", "repository", "context_sha256"])
def test_source_syntax_rejects_prediction_identity_mismatch(field: str) -> None:
    row, document, prediction = _source_syntax_fixture()
    prediction[field] = "mismatched identity"
    with pytest.raises(ValueError, match="identities differ|identity differs"):
        _source_syntax_rows([row], [prediction], [document], parse_status=lambda *_: "pass")


def test_source_syntax_cli_path_validates_frozen_files_without_a_model(
    tmp_path: Path, monkeypatch
) -> None:
    import prepare_q25_fim

    row, document, prediction = _source_syntax_fixture()
    parent_plan = tmp_path / "parent-cpt-plan.json"
    parent_plan.write_text('{"schema":"test-parent"}\n', encoding="utf-8")
    parent_sha = hashlib.sha256(parent_plan.read_bytes()).hexdigest()
    preparation_plan = tmp_path / "fim-preparation-plan.json"
    preparation_plan.write_text(
        json.dumps({"parent_cpt_plan_sha256": parent_sha}) + "\n", encoding="utf-8"
    )
    preparation_sha = hashlib.sha256(preparation_plan.read_bytes()).hexdigest()
    monkeypatch.setattr("evaluate_q25_fim.FIM_PREPARATION_PLAN", preparation_plan)

    development = tmp_path / "development.jsonl"
    development_bytes = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
    development.write_bytes(development_bytes)
    development_sha = hashlib.sha256(development_bytes).hexdigest()
    metadata = {
        "preparation_plan_sha256": preparation_sha,
        "parent_cpt_plan_sha256": parent_sha,
        "splits": {
            "development": {
                "file": development.name,
                "sha256": development_sha,
                "row_count": 1,
            }
        },
        "files": {development.name: {"sha256": development_sha, "bytes": len(development_bytes)}},
    }
    metadata_path = tmp_path / "corpus_metadata.json"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()

    gpu_results = tmp_path / "gpu" / "results.jsonl"
    gpu_results.parent.mkdir()
    gpu_results.write_text(json.dumps({**prediction, "exact": False}) + "\n", encoding="utf-8")
    model_sha = _sha("local model export")
    gpu_summary = {
        "plan_sha256": "pending",
        "cases": 1,
        "model_sha256": model_sha,
    }
    result_summary = gpu_results.parent / "summary.json"
    result_summary.write_text(json.dumps(gpu_summary) + "\n", encoding="utf-8")

    full_plan = {
        "schema": "q25-fim-training-plan-v1",
        "gpu_execution_authorized": True,
        "parent_cpt_plan_sha256": parent_sha,
        "preparation_plan_sha256": preparation_sha,
        "data": {
            "development": {
                "sha256": development_sha,
                "row_count": 1,
                "bytes": len(development_bytes),
            },
            "corpus_metadata_sha256": metadata_sha,
        },
        "evaluation": {
            "source_syntax": {"protocol": SOURCE_SYNTAX_PROTOCOL},
        },
    }
    plan_path = tmp_path / "fim-training-plan.json"
    plan_path.write_text(json.dumps(full_plan) + "\n", encoding="utf-8")
    gpu_summary["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    result_summary.write_text(json.dumps(gpu_summary) + "\n", encoding="utf-8")

    pool_hashes = {"python": {"sha256": _sha("pool bytes"), "sidecar_sha256": _sha("pool sidecar")}}
    monkeypatch.setattr(
        prepare_q25_fim,
        "_load_pinned_inputs",
        lambda _path: ({}, {"pool_hashes": pool_hashes}, [], [document]),
    )
    args = argparse.Namespace(
        mode="development",
        model=None,
        alias=None,
        predictions=gpu_results,
        parent_cpt_plan=parent_plan,
        plan=plan_path,
        input=development,
        output=tmp_path / "cpu-output",
    )

    summary = run_source_syntax_diagnostic(args)
    report_text = (args.output / "source-syntax-results.jsonl").read_text(encoding="utf-8")
    assert summary["cases"] == 1
    assert summary["model_sha256"] == model_sha
    assert summary["raw_source_pool_file_hashes"] == pool_hashes
    assert "value =" not in report_text
