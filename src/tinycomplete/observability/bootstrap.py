"""Process-scoped OpenTelemetry SDK lifecycle."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import os
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from .config import ObservabilityConfig

_active_runtime: contextvars.ContextVar[ObservabilityRuntime | None] = contextvars.ContextVar(
    "tabcomplete_observability_runtime", default=None
)


@dataclass
class ObservabilityRuntime:
    config: ObservabilityConfig
    tracer_provider: TracerProvider | None = None
    meter_provider: MeterProvider | None = None
    logger_provider: LoggerProvider | None = None

    @property
    def tracer(self):
        if self.tracer_provider is None:
            return trace.NoOpTracerProvider().get_tracer("tinycomplete.observability")
        return self.tracer_provider.get_tracer("tinycomplete.observability", "1")

    @contextlib.contextmanager
    def activate(self) -> Iterator[ObservabilityRuntime]:
        token = _active_runtime.set(self)
        try:
            yield self
        finally:
            _active_runtime.reset(token)

    def force_flush(self, timeout_millis: int = 5_000) -> bool:
        if self.tracer_provider is None:
            return True
        return bool(self.tracer_provider.force_flush(timeout_millis=timeout_millis))

    def shutdown(self) -> None:
        if self.tracer_provider is not None:
            self.tracer_provider.force_flush()
            self.tracer_provider.shutdown()
        for provider in (self.meter_provider, self.logger_provider):
            if provider is not None:
                provider.shutdown()


_disabled_runtime = ObservabilityRuntime(ObservabilityConfig())
_process_runtime: ObservabilityRuntime | None = None
_process_id = os.getpid()
_runtime_lock = threading.RLock()


def current_runtime() -> ObservabilityRuntime:
    global _process_runtime, _process_id
    active = _active_runtime.get()
    if active is not None:
        return active
    with _runtime_lock:
        if _process_runtime is None or _process_id != os.getpid():
            _process_id = os.getpid()
            try:
                _process_runtime = initialize_observability()
                atexit.register(_process_runtime.shutdown)
            except Exception:
                # Telemetry configuration or exporter failure cannot abort the workload.
                _process_runtime = _disabled_runtime
        return _process_runtime


def initialize_observability(
    config: ObservabilityConfig | None = None,
    *,
    span_exporter: SpanExporter | None = None,
) -> ObservabilityRuntime:
    config = config or ObservabilityConfig.from_env()
    if not config.enabled:
        return ObservabilityRuntime(config)
    resource = Resource(
        {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment.name": config.deployment_environment,
            "host.name": socket.gethostname(),
        }
    )
    provider = TracerProvider(resource=resource)
    if span_exporter is None:
        if config.mode == "offline":
            from .offline import OfflineSpanExporter

            span_exporter = OfflineSpanExporter(
                config.offline_bundle, max_bytes=config.offline_max_bytes
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            span_exporter = OTLPSpanExporter(
                endpoint=config.otlp_endpoint + "/v1/traces",
                timeout=config.export_timeout_seconds,
            )
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    runtime = ObservabilityRuntime(config=config, tracer_provider=provider)
    if config.mode == "live":
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import (
            DefaultAggregation,
            ExplicitBucketHistogramAggregation,
            View,
        )

        from .metrics import METRIC_UNITS, PERMITTED_METRIC_ATTRIBUTES

        # No run/process UUID resource labels. SDK views enforce label bounds at export.
        runtime.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=config.otlp_endpoint + "/v1/metrics",
                        timeout=config.export_timeout_seconds,
                    ),
                    export_interval_millis=15_000,
                )
            ],
            views=[
                View(
                    instrument_name=name,
                    attribute_keys=set(PERMITTED_METRIC_ATTRIBUTES),
                    aggregation=ExplicitBucketHistogramAggregation(
                        [1, 5, 10, 50, 100, 500, 1000, 5000, 15000, 60000, 180000, 600000]
                    )
                    if name in {"tabcomplete.model.duration", "tabcomplete.model.first_output"}
                    else DefaultAggregation(),
                )
                for name in METRIC_UNITS
            ],
        )
        runtime.logger_provider = LoggerProvider(resource=resource)
        runtime.logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(
                    endpoint=config.otlp_endpoint + "/v1/logs",
                    timeout=config.export_timeout_seconds,
                )
            )
        )
    return runtime
