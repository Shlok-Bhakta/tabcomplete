"""Sanitized structured logging helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from .context import current_run_context
from .spans import sanitize_text


def emit_log(event: str, fields: Mapping[str, Any], *, error: bool = False) -> None:
    import time

    from opentelemetry import trace
    from opentelemetry._logs import LogRecord, SeverityNumber

    from .bootstrap import current_runtime

    runtime = current_runtime()
    if runtime.logger_provider is None:
        return
    try:
        attributes = {
            k: sanitize_text(v) if isinstance(v, str) else v
            for k, v in fields.items()
            if v is not None
        }
        run = current_run_context()
        if run:
            attributes.update(run.attributes())
        span = trace.get_current_span().get_span_context()
        runtime.logger_provider.get_logger("tinycomplete.observability").emit(
            LogRecord(
                timestamp=time.time_ns(),
                observed_timestamp=time.time_ns(),
                trace_id=span.trace_id,
                span_id=span.span_id,
                trace_flags=span.trace_flags,
                severity_number=SeverityNumber.ERROR if error else SeverityNumber.INFO,
                severity_text="ERROR" if error else "INFO",
                body=event,
                attributes=attributes,
            )
        )
    except Exception:
        return


def structured_log(
    logger: logging.Logger, level: int, event: str, fields: Mapping[str, Any]
) -> None:
    payload: dict[str, Any] = {
        "event": event,
        **{key: sanitize_text(value) for key, value in fields.items()},
    }
    run = current_run_context()
    if run is not None:
        payload.update(run.attributes())
    logger.log(level, json.dumps(payload, sort_keys=True))
