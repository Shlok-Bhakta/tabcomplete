"""Shared OpenTelemetry instrumentation for TabComplete workloads."""

from .bootstrap import ObservabilityRuntime, initialize_observability
from .config import ObservabilityConfig

__all__ = ["ObservabilityConfig", "ObservabilityRuntime", "initialize_observability"]
