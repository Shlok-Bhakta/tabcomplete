"""Executable code-completion benchmark contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinycomplete.eval.code_benchmark import (
    BenchmarkCase,
    Prediction,
    container_command,
    evaluate_prediction,
    load_suite,
    summarize_results,
)
from tinycomplete.eval.code_generation import (
    OpenAICompatibleGenerationProvider,
    build_causal_prompt,
    generate_predictions,
)


def python_case(**overrides) -> BenchmarkCase:
    values = {
        "id": "python/add",
        "language": "python",
        "path": "solution.py",
        "prefix": "def add(a, b):\n    ",
        "suffix": "\n",
        "expected": "return a + b",
        "context_files": {"README.md": "Implement integer addition.\n"},
        "check": {
            "compile": ["python3", "-m", "py_compile", "solution.py"],
            "test": ["python3", "tests.py"],
            "files": {
                "tests.py": (
                    "from solution import add\nassert add(2, 3) == 5\nassert add(-2, 2) == 0\n"
                )
            },
        },
    }
    values.update(overrides)
    return BenchmarkCase.model_validate(values)


def test_suite_rejects_duplicate_ids_and_unsafe_paths(tmp_path: Path):
    suite = tmp_path / "suite.jsonl"
    record = python_case().model_dump(mode="json")
    suite.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate benchmark id"):
        load_suite(suite)

    with pytest.raises(ValueError, match="relative path"):
        python_case(path="../escape.py")
    with pytest.raises(ValueError, match="relative path"):
        python_case(context_files={"/tmp/escape": "nope"})


def test_trusted_host_evaluation_compiles_and_runs_real_tests(tmp_path: Path):
    result = evaluate_prediction(
        python_case(),
        Prediction(case_id="python/add", completion="return a + b"),
        work_root=tmp_path,
        execution_backend="trusted-host",
    )
    assert result.exact_match is True
    assert result.parse.status == "pass"
    assert result.compile.status == "pass"
    assert result.test.status == "pass"
    assert result.working_tree_sha256


def test_wrong_but_valid_completion_is_not_confused_with_correctness(tmp_path: Path):
    result = evaluate_prediction(
        python_case(),
        Prediction(case_id="python/add", completion="return a - b"),
        work_root=tmp_path,
        execution_backend="trusted-host",
    )
    assert result.exact_match is False
    assert result.parse.status == "pass"
    assert result.compile.status == "pass"
    assert result.test.status == "fail"


def test_fixture_hash_does_not_include_compiler_outputs(tmp_path: Path):
    clean = evaluate_prediction(
        python_case(),
        Prediction(case_id="python/add", completion="return a + b"),
        work_root=tmp_path / "clean",
        execution_backend="none",
    )
    executed = evaluate_prediction(
        python_case(),
        Prediction(case_id="python/add", completion="return a + b"),
        work_root=tmp_path / "executed",
        execution_backend="trusted-host",
    )
    assert executed.test.status == "pass"
    assert clean.working_tree_sha256 == executed.working_tree_sha256


def test_execution_is_refused_without_a_sandbox(tmp_path: Path):
    result = evaluate_prediction(
        python_case(),
        Prediction(case_id="python/add", completion="return a + b"),
        work_root=tmp_path,
        execution_backend="none",
    )
    assert result.parse.status == "pass"
    assert result.compile.status == "not_run"
    assert result.test.status == "not_run"


def test_container_command_has_hard_isolation_and_no_implicit_pull(tmp_path: Path):
    command = container_command(
        runtime="podman",
        image="docker.io/library/python:3.12-slim",
        work_root=tmp_path,
        fixture_command=["python3", "tests.py"],
        timeout_seconds=3,
    )
    joined = " ".join(command)
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=all" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--pull=never" in command
    assert "--pids-limit=64" in command
    assert "--memory=768m" in command
    assert str(tmp_path.resolve()) in joined
    assert command[-2:] == ["python3", "tests.py"]


def test_timeout_and_output_are_bounded(tmp_path: Path):
    case = python_case(
        id="python/timeout",
        prefix="",
        suffix="",
        expected="while True: pass",
        check={"test": ["python3", "solution.py"], "timeout_seconds": 0.1},
    )
    result = evaluate_prediction(
        case,
        Prediction(case_id=case.id, completion=case.expected),
        work_root=tmp_path,
        execution_backend="trusted-host",
    )
    assert result.test.status == "timeout"


def test_summary_keeps_language_and_check_breakdowns(tmp_path: Path):
    good = evaluate_prediction(
        python_case(),
        Prediction(
            case_id="python/add",
            completion="return a + b",
            latency_seconds=1.0,
            generated_tokens=4,
        ),
        work_root=tmp_path / "good",
        execution_backend="trusted-host",
    )
    bad = evaluate_prediction(
        python_case(id="python/sub"),
        Prediction(
            case_id="python/sub",
            completion="return a - b",
            latency_seconds=3.0,
            generated_tokens=8,
        ),
        work_root=tmp_path / "bad",
        execution_backend="trusted-host",
    )
    summary = summarize_results([good, bad])
    assert summary["total"] == 2
    assert summary["exact_match_rate"] == 0.5
    assert summary["test_pass_rate"] == 0.5
    assert summary["median_latency_seconds"] == 2.0
    assert summary["median_generated_tokens"] == 6.0
    assert summary["by_language"]["python"]["total"] == 2
    assert summary["by_repository_context"]["standalone"]["total"] == 2
    assert summary["by_repository_context"]["repository_context"]["total"] == 0
    assert summary["by_category"]["unspecified"]["total"] == 2


def test_summary_counts_skipped_downstream_checks_as_failures(tmp_path: Path):
    good = evaluate_prediction(
        python_case(),
        Prediction(case_id="python/add", completion="return a + b"),
        work_root=tmp_path / "good",
        execution_backend="trusted-host",
    )
    compile_failure = evaluate_prediction(
        python_case(id="python/broken"),
        Prediction(case_id="python/broken", completion="return ("),
        work_root=tmp_path / "broken",
        execution_backend="trusted-host",
    )

    assert compile_failure.compile.status == "fail"
    assert compile_failure.test.status == "not_run"
    summary = summarize_results([good, compile_failure])
    assert summary["compile_pass_rate"] == 0.5
    assert summary["test_pass_rate"] == 0.5
    assert summary["compile_availability_rate"] == 1.0
    assert summary["test_availability_rate"] == 0.5


def test_causal_prompt_never_leaks_gold_or_suffix_and_target_is_last():
    case = python_case(
        expected="SECRET_GOLD",
        suffix="SECRET_SUFFIX",
        context_files={"z.py": "Z = 1", "a.py": "A = 2"},
    )
    prompt = build_causal_prompt(case)
    assert "SECRET_GOLD" not in prompt
    assert "SECRET_SUFFIX" not in prompt
    assert prompt.index('path="a.py"') < prompt.index('path="z.py"')
    assert prompt.endswith(case.prefix)


def test_prediction_generation_flushes_and_resumes(tmp_path: Path):
    class Provider:
        calls = 0

        def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
            self.calls += 1
            assert "SECRET_GOLD" not in prompt
            assert max_new_tokens == 17
            return "return a + b", 4

    case = python_case(expected="SECRET_GOLD")
    output = tmp_path / "predictions.jsonl"
    metadata = {
        "schema_version": 1,
        "suite_sha256": "fixture",
        "model_revision": "base-revision",
        "max_new_tokens": 17,
    }
    first = Provider()
    predictions = generate_predictions(
        [case], first, output, max_new_tokens=17, run_metadata=metadata
    )
    assert first.calls == 1
    assert predictions[0].completion == "return a + b"
    assert output.read_text(encoding="utf-8").count("\n") == 1

    resumed = Provider()
    assert (
        generate_predictions([case], resumed, output, max_new_tokens=17, run_metadata=metadata)
        == predictions
    )
    assert resumed.calls == 0

    with pytest.raises(ValueError, match="metadata does not match"):
        generate_predictions(
            [case],
            Provider(),
            output,
            max_new_tokens=17,
            run_metadata={**metadata, "model_revision": "different-model"},
        )


def test_server_generation_uses_configured_request_timeout(monkeypatch):
    seen: dict[str, float] = {}

    class Response:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict:
            return {"choices": [{"text": "pass"}], "usage": {"completion_tokens": 1}}

    def fake_post(url: str, *, json: dict, timeout: float):
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("tinycomplete.eval.code_generation.httpx.post", fake_post)
    provider = OpenAICompatibleGenerationProvider(
        "http://localhost:8080", "fixture", timeout_seconds=900
    )
    assert provider.generate("def f():\n    ", 4) == ("pass", 1)
    assert seen["timeout"] == 900

    with pytest.raises(ValueError, match="positive"):
        OpenAICompatibleGenerationProvider("http://localhost:8080", "fixture", timeout_seconds=0)


def test_prediction_resume_refuses_missing_metadata(tmp_path: Path):
    output = tmp_path / "predictions.jsonl"
    output.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata missing"):
        generate_predictions(
            [python_case()],
            object(),
            output,
            run_metadata={"schema_version": 1},
        )
