from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from tinycomplete.eval.code_benchmark import BenchmarkCase
from tinycomplete.eval.code_generation import DetailedGeneration, generate_predictions
from tinycomplete.observability.testing import memory_runtime


def test_http_missing_usage_preserves_scientific_fallback_but_telemetry_is_unknown(monkeypatch):
    import httpx

    from tinycomplete.eval.code_generation import OpenAICompatibleGenerationProvider

    def respond(url, **kwargs):
        assert kwargs["json"] == {
            "model": "fixture",
            "prompt": " original \n",
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        }
        return httpx.Response(
            200,
            json={"choices": [{"text": " output\n", "finish_reason": "stop"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", respond)
    runtime, exporter = memory_runtime()
    with runtime.activate():
        result = OpenAICompatibleGenerationProvider(
            "http://fixture.invalid", "fixture"
        ).generate_detailed(" original \n", 8)
    runtime.shutdown()
    assert result.text == " output\n" and result.tokens == 0 and not result.output_tokens_known
    span = next(s for s in exporter.get_finished_spans() if s.name == "model.generate")
    assert "gen_ai.usage.output_tokens" not in span.attributes
    assert "tabcomplete.timing.first_output_ms" not in span.attributes


class FakeProvider:
    def generate_detailed(self, prompt: str, max_new_tokens: int) -> DetailedGeneration:
        assert max_new_tokens == 8
        return DetailedGeneration(
            text="return 1\n",
            tokens=3,
            finish_reason="stop",
            input_tokens=5,
            usage_source="tokenizer",
        )


class SlowFakeProvider(FakeProvider):
    def generate_detailed(self, prompt: str, max_new_tokens: int) -> DetailedGeneration:
        time.sleep(0.04)
        return super().generate_detailed(prompt, max_new_tokens)


def test_prediction_runner_records_threaded_model_calls_and_artifacts(tmp_path: Path) -> None:
    runtime, exporter = memory_runtime()
    runtime.config = replace(
        runtime.config,
        capture_content=True,
        artifact_root=tmp_path / "artifacts",
        heartbeat_seconds=0.01,
    )
    cases = [
        BenchmarkCase(
            id=f"case-{index}",
            language="python",
            path=f"case_{index}.py",
            prefix="def f():\n    ",
            expected="return 1\n",
        )
        for index in range(2)
    ]
    metadata = {
        "protocol": "causal-context-v1",
        "suite_sha256": "suite-smoke",
        "provider": "fake",
        "model_source": "fixture-model",
        "model_revision": "weights-123",
        "max_new_tokens": 8,
        "workers": 2,
        "decoding": {"do_sample": False, "temperature": 0},
    }
    with runtime.activate():
        predictions = generate_predictions(
            cases,
            FakeProvider(),
            tmp_path / "predictions.jsonl",
            max_new_tokens=8,
            run_metadata=metadata,
            workers=2,
        )
    runtime.shutdown()
    assert [prediction.case_id for prediction in predictions] == ["case-0", "case-1"]
    spans = exporter.get_finished_spans()
    model_spans = [span for span in spans if span.name == "model.generate"]
    case_spans = [span for span in spans if span.name == "eval.case"]
    assert len(model_spans) == len(case_spans) == 2
    assert all(span.attributes["gen_ai.usage.input_tokens"] == 5 for span in model_spans)
    assert all(span.attributes["gen_ai.usage.output_tokens"] == 3 for span in model_spans)
    assert all(span.attributes["tabcomplete.usage.source"] == "tokenizer" for span in model_spans)
    assert all("tabcomplete.timing.first_output_ms" not in span.attributes for span in model_spans)
    for span in model_spans:
        assert span.attributes["tabcomplete.artifact.input.status"] == "captured"
        assert span.attributes["tabcomplete.artifact.output.status"] == "captured"
    assert {span.attributes["tabcomplete.case_id"] for span in case_spans} == {"case-0", "case-1"}
    assert any(span.name == "run.start" for span in spans)
    assert any(span.name == "run.summary" for span in spans)
    metadata_saved = (tmp_path / "predictions.jsonl.metadata.json").read_text(encoding="utf-8")
    assert "run_id" in metadata_saved


def test_collector_outage_does_not_fail_prediction(tmp_path: Path) -> None:
    from tinycomplete.observability.bootstrap import initialize_observability
    from tinycomplete.observability.config import ObservabilityConfig

    runtime = initialize_observability(
        ObservabilityConfig(
            enabled=True,
            mode="live",
            otlp_endpoint="http://127.0.0.1:1",
            export_timeout_seconds=0.05,
        )
    )
    case = BenchmarkCase(
        id="case-outage",
        language="python",
        path="case.py",
        prefix="def f():\n    ",
        expected="return 1\n",
    )
    with runtime.activate():
        result = generate_predictions(
            [case],
            FakeProvider(),
            tmp_path / "outage.jsonl",
            max_new_tokens=8,
            workers=1,
            run_metadata={
                "protocol": "smoke",
                "suite_sha256": "outage",
                "provider": "fake",
                "model_source": "fixture",
                "model_revision": "fixture",
                "max_new_tokens": 8,
                "workers": 1,
                "decoding": {"do_sample": False, "temperature": 0},
            },
        )
    runtime.shutdown()
    assert result[0].completion == "return 1\n"


def test_prediction_runner_emits_live_heartbeats(tmp_path: Path) -> None:
    runtime, exporter = memory_runtime()
    runtime.config = replace(runtime.config, heartbeat_seconds=0.01)
    case = BenchmarkCase(
        id="case-heartbeat",
        language="python",
        path="case.py",
        prefix="def f():\n    ",
        expected="return 1\n",
    )
    with runtime.activate():
        generate_predictions(
            [case],
            SlowFakeProvider(),
            tmp_path / "heartbeat.jsonl",
            max_new_tokens=8,
            workers=1,
            run_metadata={
                "protocol": "smoke",
                "suite_sha256": "heartbeat",
                "provider": "fake",
                "model_source": "fixture",
                "model_revision": "fixture",
                "max_new_tokens": 8,
                "workers": 1,
                "decoding": {"do_sample": False, "temperature": 0},
            },
        )
    runtime.shutdown()
    heartbeats = [span for span in exporter.get_finished_spans() if span.name == "run.heartbeat"]
    assert heartbeats
    assert heartbeats[-1].attributes["tabcomplete.cases.completed"] in {0, 1}


def test_prediction_runner_summarizes_serial_failure(tmp_path: Path) -> None:
    class BrokenProvider:
        def generate_detailed(self, prompt: str, max_new_tokens: int) -> DetailedGeneration:
            raise RuntimeError("expected fixture failure")

    runtime, exporter = memory_runtime()
    case = BenchmarkCase(
        id="case-failure",
        language="python",
        path="case.py",
        prefix="def f():\n    ",
        expected="return 1\n",
    )
    with runtime.activate(), pytest.raises(RuntimeError, match="fixture failure"):
        generate_predictions(
            [case],
            BrokenProvider(),
            tmp_path / "failure.jsonl",
            max_new_tokens=8,
            workers=1,
            run_metadata={
                "protocol": "smoke",
                "suite_sha256": "failure",
                "provider": "fake",
                "model_source": "fixture",
                "model_revision": "fixture",
                "max_new_tokens": 8,
                "workers": 1,
                "decoding": {"do_sample": False, "temperature": 0},
            },
        )
    runtime.shutdown()
    summary = [span for span in exporter.get_finished_spans() if span.name == "run.summary"]
    assert summary[-1].attributes["tabcomplete.run.state"] == "failed"
