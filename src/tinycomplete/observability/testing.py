"""Test-only in-memory OpenTelemetry runtime."""

from __future__ import annotations

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .bootstrap import ObservabilityRuntime
from .config import ObservabilityConfig


def memory_runtime() -> tuple[ObservabilityRuntime, InMemorySpanExporter]:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    config = ObservabilityConfig(enabled=True, mode="live", service_name="test")
    return ObservabilityRuntime(config=config, tracer_provider=provider), exporter
