from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from tinycomplete.observability.bootstrap import initialize_observability
from tinycomplete.observability.config import ObservabilityConfig
from tinycomplete.observability.offline import import_bundle
from tinycomplete.observability.spans import operation


def test_offline_replay_preserves_ids_and_deduplicates(tmp_path, monkeypatch):
    bundle = tmp_path / "offline.jsonl"
    runtime = initialize_observability(
        ObservabilityConfig(enabled=True, mode="offline", offline_bundle=bundle)
    )
    with runtime.activate(), operation("eval.case") as span:
        identity = span.get_span_context()
    runtime.shutdown()
    exported = []
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter.export",
        lambda self, spans: exported.extend(spans) or SpanExportResult.SUCCESS,
    )
    first = import_bundle(bundle, tmp_path / "ledger.sqlite", "http://127.0.0.1:1")
    second = import_bundle(bundle, tmp_path / "ledger.sqlite", "http://127.0.0.1:1")
    assert first["imported_spans"] == 1 and second["duplicate"] is True
    assert exported[0].context.trace_id == identity.trace_id
    assert exported[0].attributes["tabcomplete.historical"] is True
    assert exported[0].start_time == json.loads(bundle.read_text())["start_time_unix_nano"]


def test_offline_failed_export_does_not_claim_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "offline.jsonl"
    runtime = initialize_observability(
        ObservabilityConfig(enabled=True, mode="offline", offline_bundle=bundle)
    )
    with runtime.activate(), operation("run.summary"):
        pass
    runtime.shutdown()
    method = "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter.export"
    monkeypatch.setattr(method, lambda self, spans: SpanExportResult.FAILURE)
    with pytest.raises(RuntimeError):
        import_bundle(bundle, tmp_path / "ledger.sqlite", "http://127.0.0.1:1")
    monkeypatch.setattr(method, lambda self, spans: SpanExportResult.SUCCESS)
    assert (
        import_bundle(bundle, tmp_path / "ledger.sqlite", "http://127.0.0.1:1")["imported_spans"]
        == 1
    )


def test_dashboard_definition_is_stable_and_versioned():
    root = Path(__file__).parents[1] / "infra/observability"
    import sys

    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("provision", root / "scripts/provision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    definition = json.loads((root / "dashboards/runs-and-models.json").read_text())
    assert module.dashboard(definition) == module.dashboard(definition)
    assert module.dashboard(definition)["schemaVersion"] == "v6"
