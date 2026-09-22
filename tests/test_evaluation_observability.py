from __future__ import annotations

import sys
from pathlib import Path

from tinycomplete.eval.code_benchmark import (
    BenchmarkCase,
    CheckSpec,
    Prediction,
    evaluate_prediction,
)
from tinycomplete.eval.long_context_diagnostic import score_target_continuation
from tinycomplete.observability.testing import memory_runtime


def test_executable_evaluation_records_parse_compile_and_execute(tmp_path: Path) -> None:
    runtime, exporter = memory_runtime()
    case = BenchmarkCase(
        id="eval-smoke",
        language="python",
        path="solution.py",
        prefix="",
        expected="print('ok')\n",
        check=CheckSpec(
            compile=[sys.executable, "-m", "py_compile", "solution.py"],
            test=[sys.executable, "solution.py"],
            expected_stdout="ok\n",
        ),
    )
    with runtime.activate():
        result = evaluate_prediction(
            case,
            Prediction(case_id=case.id, completion=case.expected),
            work_root=tmp_path / "work",
            execution_backend="trusted-host",
        )
    runtime.shutdown()
    assert result.parse.status == result.compile.status == result.test.status == "pass"
    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert {"eval.case", "eval.parse", "eval.compile", "eval.execute"} <= names
    case_span = next(span for span in spans if span.name == "eval.case")
    assert case_span.attributes["tabcomplete.quality.functional"] == "pass"
    assert case_span.attributes["tabcomplete.case_id"] == "eval-smoke"


class _FakeTensor:
    pass


def test_long_context_scoring_has_model_score_span(monkeypatch) -> None:
    runtime, exporter = memory_runtime()

    class Torch:
        long = object()
        float16 = object()

        class inference_mode:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class autocast(inference_mode):
            def __init__(self, **kwargs):
                pass

        @staticmethod
        def tensor(*args, **kwargs):
            return _FakeTensor()

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text == "prompt" else [3]

    class Logits:
        def float(self):
            return self

    class Model:
        def __call__(self, **kwargs):
            return type("Output", (), {"logits": Logits()})()

    class Nll:
        def item(self):
            return 2.5

    monkeypatch.setitem(sys.modules, "torch", Torch)
    monkeypatch.setattr(
        "tinycomplete.eval.long_context_diagnostic.target_logit_positions",
        lambda *args: type("Positions", (), {"__getitem__": lambda self, index: Nll()})(),
    )
    monkeypatch.setattr(
        "tinycomplete.eval.long_context_diagnostic.selected_target_nll",
        lambda *args: (Nll(), 1),
    )
    with runtime.activate():
        result = score_target_continuation(Model(), Tokenizer(), "prompt", "target", "cuda")
    runtime.shutdown()
    assert result["target_nll_mean"] == 2.5
    assert any(span.name == "model.score" for span in exporter.get_finished_spans())
