"""Small, shared span helpers with sanitized failure recording."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from .bootstrap import current_runtime
from .context import current_run_context

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"()(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,})"),
)


def sanitize_text(value: object, *, limit: int = 512) -> str:
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text[:limit]


def _attributes(values: Mapping[str, Any] | None) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            clean[key] = sanitize_text(value) if isinstance(value, str) else value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, bool, int, float)) for item in value
        ):
            clean[key] = [sanitize_text(item) if isinstance(item, str) else item for item in value]
    run = current_run_context()
    if run is not None:
        clean = {**run.attributes(), **clean}
    return clean


@contextlib.contextmanager
def operation(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    authoritative_rank: int | None = None,
    rank: int | None = None,
) -> Iterator[Span]:
    runtime = current_runtime()
    should_record = runtime.config.enabled and (
        authoritative_rank is None or rank is None or rank == authoritative_rank
    )
    base_attributes = _attributes(attributes)
    if rank is not None:
        base_attributes["tabcomplete.rank"] = rank
    if not should_record:
        try:
            yield trace.INVALID_SPAN
        except BaseException as exc:
            if runtime.config.enabled:
                with runtime.tracer.start_as_current_span(
                    name,
                    attributes=base_attributes,
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    _record_failure(span, exc)
            raise
        return
    with runtime.tracer.start_as_current_span(
        name,
        attributes=base_attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as exc:
            _record_failure(span, exc)
            raise


def _record_failure(span: Span, exception: BaseException) -> None:
    span.set_status(Status(StatusCode.ERROR))
    span.set_attribute("tabcomplete.failure.type", type(exception).__name__)
    span.set_attribute("tabcomplete.failure.message", sanitize_text(exception))
    if isinstance(exception, TimeoutError):
        span.set_attribute("tabcomplete.failure.timeout", True)
    if type(exception).__name__ in {"CancelledError", "Cancelled"}:
        span.set_attribute("tabcomplete.failure.cancelled", True)
    from .logging import emit_log

    emit_log(
        "operation.failure",
        {
            "tabcomplete.failure.type": type(exception).__name__,
            "tabcomplete.failure.message": sanitize_text(exception),
        },
        error=True,
    )
