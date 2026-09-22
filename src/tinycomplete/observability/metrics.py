"""Low-cardinality metric dimensions and units."""

from __future__ import annotations

from collections.abc import Mapping

PERMITTED_METRIC_ATTRIBUTES = frozenset(
    {
        "service",
        "task",
        "language",
        "backend",
        "model_alias",
        "quantization",
        "outcome",
        "context_size_bucket",
        "suite_version",
        "protocol_version",
        "device_type",
        "rank_role",
        "token_type",
    }
)

METRIC_UNITS = {
    "tabcomplete.model.calls": "{call}",
    "tabcomplete.model.failures": "{failure}",
    "tabcomplete.model.tokens": "{token}",
    "tabcomplete.model.duration": "ms",
    "tabcomplete.model.first_output": "ms",
    "tabcomplete.model.active": "{call}",
    "tabcomplete.eval.cases": "{case}",
    "tabcomplete.training.input_tokens": "{token}",
    "tabcomplete.training.scored_tokens": "{token}",
    "tabcomplete.training.updates": "{update}",
    "tabcomplete.training.loss": "1",
    "tabcomplete.training.validation_nll": "1",
    "tabcomplete.training.gradient_norm": "1",
    "tabcomplete.training.skipped_updates": "{update}",
    "tabcomplete.training.learning_rate": "1",
}


def validate_metric_attributes(
    attributes: Mapping[str, object],
) -> dict[str, str | bool | int | float]:
    forbidden = sorted(set(attributes) - PERMITTED_METRIC_ATTRIBUTES)
    if forbidden:
        raise ValueError(f"metric attributes are not permitted: {', '.join(forbidden)}")
    if any(not isinstance(value, (str, bool, int, float)) for value in attributes.values()):
        raise ValueError("metric attribute values must be scalar")
    return {
        key: value
        for key, value in attributes.items()
        if isinstance(value, (str, bool, int, float))
    }


def context_size_bucket(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    for limit in (2_048, 4_096, 8_192, 16_384, 32_768):
        if tokens <= limit:
            return f"le_{limit}"
    return "gt_32768"


def record_metric(
    name: str, value: float, attributes: Mapping[str, object], *, kind: str = "counter"
) -> None:
    """Fail-open aggregate export; IDs and content are never metric dimensions."""
    from .bootstrap import current_runtime

    clean = validate_metric_attributes(attributes)
    runtime = current_runtime()
    if runtime.meter_provider is None:
        return
    try:
        meter = runtime.meter_provider.get_meter("tinycomplete.observability", "1")
        unit = METRIC_UNITS.get(name, "1")
        if kind == "histogram":
            meter.create_histogram(name, unit=unit).record(value, clean)
        elif kind == "updown":
            meter.create_up_down_counter(name, unit=unit).add(value, clean)
        elif kind == "gauge":
            meter.create_gauge(name, unit=unit).set(value, clean)
        else:
            meter.create_counter(name, unit=unit).add(value, clean)
    except Exception:
        return
