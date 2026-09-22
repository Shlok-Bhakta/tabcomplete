from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tinycomplete.observability.artifacts import ArtifactStore, CaptureStatus
from tinycomplete.observability.config import ObservabilityConfig
from tinycomplete.observability.context import (
    RunContext,
    context_callable,
    current_run_context,
    ensure_persistent_run_identity,
    subprocess_environment,
)
from tinycomplete.observability.metrics import validate_metric_attributes
from tinycomplete.observability.offline import OfflineImportLedger
from tinycomplete.observability.spans import operation
from tinycomplete.observability.testing import memory_runtime


@pytest.mark.parametrize("rank", ["0", "1"])
def test_training_scalars_are_published_only_by_authoritative_rank(monkeypatch, rank):
    from tinycomplete.observability.hooks import training_progress

    recorded = []
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setattr(
        "tinycomplete.observability.hooks.record_metric", lambda *a, **k: recorded.append(a)
    )
    runtime, exporter = memory_runtime()
    with runtime.activate():
        training_progress({"loss": 1.2, "training_tokens": 100, "optimizer_step": 2})
    runtime.shutdown()
    assert len(recorded) == (3 if rank == "0" else 0)
    assert len(exporter.get_finished_spans()) == (1 if rank == "0" else 0)


def test_owned_http_trace_context_propagates_without_baggage():
    from opentelemetry import baggage

    from tinycomplete.observability.hooks import observed_http_request

    class Handler:
        headers = {
            "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
            "baggage": "private=do-not-propagate",
        }

        @observed_http_request
        def request(self):
            assert baggage.get_baggage("private") is None
            return "unchanged"

    runtime, exporter = memory_runtime()
    with runtime.activate():
        assert Handler().request() == "unchanged"
    runtime.shutdown()
    span = exporter.get_finished_spans()[0]
    assert span.context.trace_id == int("0123456789abcdef0123456789abcdef", 16)
    assert span.parent.span_id == int("0123456789abcdef", 16)


def test_artifact_sync_validates_hash_and_reports_failed_delivery(tmp_path, monkeypatch):
    import httpx

    from tinycomplete.observability.artifacts import sync_artifacts

    store = ArtifactStore(tmp_path / "cas")
    reference = store.capture_text("output", "  code\n", authorized=True)
    token = tmp_path / "token"
    token.write_text("synthetic-test-secret")
    monkeypatch.setenv("TABCOMPLETE_ARTIFACT_UPLOAD_URL", "http://monitor.invalid")
    monkeypatch.setenv("TABCOMPLETE_ARTIFACT_UPLOAD_TOKEN_FILE", str(token))
    seen = []

    def put(self, url, **kwargs):
        seen.append(kwargs["content"])
        return httpx.Response(200, request=httpx.Request("PUT", url))

    monkeypatch.setattr(httpx.Client, "put", put)
    assert sync_artifacts(store.root) == {"accepted": 1, "failed": 0}
    assert seen == [b"  code\n"]
    assert reference.sha256 is not None

    def fail(self, url, **kwargs):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(httpx.Client, "put", fail)
    assert sync_artifacts(store.root) == {"accepted": 0, "failed": 1}
    assert store.read_text(reference.sha256) == "  code\n"


def test_disabled_instrumentation_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABCOMPLETE_OBSERVABILITY_ENABLED", raising=False)
    config = ObservabilityConfig.from_env()
    assert not config.enabled
    with operation("model.generate") as span:
        assert not span.is_recording()


def test_context_survives_thread_pool_execution() -> None:
    runtime, exporter = memory_runtime()
    run = RunContext.new(campaign_id="campaign-smoke", run_id="run-smoke")
    with runtime.activate(), run.activate(), operation("campaign.phase") as parent:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            child_context = pool.submit(
                context_callable(lambda: current_run_context()),
            ).result()
        assert child_context == run
        assert parent.get_span_context().is_valid
    runtime.shutdown()
    assert [span.name for span in exporter.get_finished_spans()] == ["campaign.phase"]


def test_subprocess_correlation_only_propagates_safe_fields(tmp_path: Path) -> None:
    runtime, _ = memory_runtime()
    run = RunContext.new(campaign_id="campaign-smoke", run_id="run-smoke")
    with runtime.activate(), run.activate(), operation("campaign.phase"):
        env = subprocess_environment({"PATH": os.environ["PATH"], "SECRET_TOKEN": "do-not-copy"})
    assert "TRACEPARENT" in env
    assert env["TABCOMPLETE_RUN_ID"] == "run-smoke"
    assert "SECRET_TOKEN" not in env
    code = (
        "import os,json;print(json.dumps({k:v for k,v in os.environ.items() "
        "if k.startswith(('TRACE','TABCOMPLETE_'))}))"
    )
    child = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    values = json.loads(child.stdout)
    assert values["TABCOMPLETE_RUN_ID"] == "run-smoke"
    assert values["TRACEPARENT"].startswith("00-")
    runtime.shutdown()


@pytest.mark.parametrize(
    ("exception", "expected_type"),
    [
        (RuntimeError("provider failed with Bearer secret-value"), "RuntimeError"),
        (TimeoutError("slow"), "TimeoutError"),
        (concurrent.futures.CancelledError(), "CancelledError"),
    ],
)
def test_failures_record_sanitized_exception_and_retry(
    exception: BaseException, expected_type: str
) -> None:
    runtime, exporter = memory_runtime()
    with runtime.activate():
        with pytest.raises(type(exception)):
            with operation("model.generate", attributes={"tabcomplete.retry.number": 2}):
                raise exception
    runtime.shutdown()
    span = exporter.get_finished_spans()[0]
    assert span.attributes["tabcomplete.failure.type"] == expected_type
    assert span.attributes["tabcomplete.retry.number"] == 2
    assert "secret-value" not in str(span.events)


def test_unknown_usage_and_first_output_are_absent() -> None:
    runtime, exporter = memory_runtime()
    with (
        runtime.activate(),
        operation("model.generate", attributes={"gen_ai.operation.name": "text_completion"}),
    ):
        pass
    runtime.shutdown()
    attributes = exporter.get_finished_spans()[0].attributes
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes
    assert "tabcomplete.timing.first_output_ms" not in attributes


def test_artifacts_deduplicate_and_preserve_whitespace(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_payload_bytes=1024)
    content = "  def f():\n\treturn 1\n\n"
    first = store.capture_text("model-input", content, authorized=True)
    second = store.capture_text("model-input", content, authorized=True)
    assert first == second
    assert first.status is CaptureStatus.CAPTURED
    assert store.read_text(first.sha256) == content
    assert len(list(tmp_path.rglob("*.gz"))) == 1


def test_artifacts_omit_oversize_and_redact_secrets(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_payload_bytes=16)
    oversize = store.capture_text("model-input", "x" * 17, authorized=True, source_ref="a.jsonl")
    assert oversize.status is CaptureStatus.OMITTED_SIZE
    assert oversize.source_ref == "a.jsonl"
    redacted = ArtifactStore(tmp_path / "redacted", max_payload_bytes=1024).capture_text(
        "model-output", "Authorization: Bearer super-secret", authorized=True
    )
    assert redacted.status is CaptureStatus.REDACTED
    assert "super-secret" not in ArtifactStore(
        tmp_path / "redacted", max_payload_bytes=1024
    ).read_text(redacted.sha256 or "")


def test_artifact_hash_validation_blocks_path_traversal(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.read_text("../../etc/passwd")


def test_metric_attribute_allowlist_rejects_high_cardinality_values() -> None:
    validate_metric_attributes({"service": "eval", "task": "causal", "outcome": "pass"})
    for forbidden in ("run_id", "case_id", "trace_id", "prompt", "file_path"):
        with pytest.raises(ValueError, match=forbidden):
            validate_metric_attributes({forbidden: "value"})


def test_distributed_progress_is_authoritative_only_on_rank_zero() -> None:
    runtime, exporter = memory_runtime()
    with runtime.activate():
        with operation("training.progress", authoritative_rank=0, rank=0):
            pass
        with operation("training.progress", authoritative_rank=0, rank=1):
            pass
        with pytest.raises(RuntimeError):
            with operation("training.progress", authoritative_rank=0, rank=1):
                raise RuntimeError("rank-local failure")
    runtime.shutdown()
    spans = exporter.get_finished_spans()
    assert len([span for span in spans if span.name == "training.progress"]) == 2
    assert any(span.attributes.get("tabcomplete.rank") == 1 for span in spans)


def test_offline_import_ledger_rejects_duplicate_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.jsonl"
    bundle.write_text('{"schema_version":1}\n', encoding="utf-8")
    ledger = OfflineImportLedger(tmp_path / "ledger.sqlite")
    first = ledger.claim(bundle)
    second = ledger.claim(bundle)
    assert first.imported
    assert not second.imported
    assert first.bundle_sha256 == second.bundle_sha256


def test_run_id_persists_but_attempt_id_changes(tmp_path: Path) -> None:
    metadata_path = tmp_path / "run.metadata.json"
    metadata = {"suite_sha256": "abc", "protocol": "causal-context-v1"}
    first = ensure_persistent_run_identity(metadata_path, metadata)
    metadata_path.write_text(json.dumps(first.metadata), encoding="utf-8")
    second = ensure_persistent_run_identity(metadata_path, metadata)
    assert first.context.run_id == second.context.run_id
    assert first.context.campaign_id == second.context.campaign_id
    assert first.context.run_attempt_id != second.context.run_attempt_id


def test_run_identity_rejects_changed_scientific_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "run.metadata.json"
    first = ensure_persistent_run_identity(metadata_path, {"suite_sha256": "abc"})
    metadata_path.write_text(json.dumps(first.metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific metadata"):
        ensure_persistent_run_identity(metadata_path, {"suite_sha256": "changed"})
