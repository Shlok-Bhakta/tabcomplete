"""Shared boundary hooks; no additional tensor reads or per-token instrumentation."""

from __future__ import annotations

import contextlib
import functools
import os
import time
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.context import attach, detach
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .artifacts import ArtifactStore
from .bootstrap import current_runtime
from .context import RunContext, current_run_context
from .metrics import record_metric
from .spans import operation


def observed_http_request(function: Callable) -> Callable:
    """Accept W3C trace context on our HTTP server, never arbitrary baggage."""

    @functools.wraps(function)
    def call(handler, *args, **kwargs):
        if not current_runtime().config.enabled:
            return function(handler, *args, **kwargs)
        carrier = {
            key: handler.headers[key]
            for key in ("traceparent", "tracestate")
            if key in handler.headers
        }
        token = attach(TraceContextTextMapPropagator().extract(carrier))
        try:
            with operation("http.server.request"):
                return function(handler, *args, **kwargs)
        finally:
            detach(token)

    return call


def observed(name: str) -> Callable:
    def decorate(function: Callable) -> Callable:
        @functools.wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            if not current_runtime().config.enabled:
                return function(*args, **kwargs)
            with operation(name):
                return function(*args, **kwargs)

        return call

    return decorate


def observed_generation(function: Callable) -> Callable:
    """Direct provider calls and runner calls share one model span, never two."""

    @functools.wraps(function)
    def call(provider: Any, prompt: str, max_new_tokens: int) -> Any:
        runtime = current_runtime()
        if not runtime.config.enabled:
            return function(provider, prompt, max_new_tokens)
        labels = {"backend": type(provider).__name__}
        # The prediction runner owns the richer benchmark span and its capture.
        owned = getattr(trace.get_current_span(), "name", None) == "model.generate"
        if owned:
            return function(provider, prompt, max_new_tokens)
        context = current_run_context() or RunContext.new()
        if context.request_id is None:
            context = context.for_case("direct-provider")
        store = ArtifactStore(
            runtime.config.artifact_root,
            enabled=runtime.config.capture_content,
            max_payload_bytes=runtime.config.artifact_max_payload_bytes,
        )
        attributes = {
            "gen_ai.operation.name": "text_completion",
            "gen_ai.provider.name": type(provider).__name__,
            "gen_ai.request.model": str(getattr(provider, "model_name", "unknown")),
            "gen_ai.request.max_tokens": max_new_tokens,
            **store.capture_text(
                "model-input", prompt, authorized=runtime.config.capture_content
            ).attributes("input"),
        }
        with context.activate():
            with operation("request.start", attributes=attributes):
                pass
            with operation("model.generate", attributes=attributes) as span:
                with model_metrics(labels):
                    started = time.perf_counter()
                    result = function(provider, prompt, max_new_tokens)
                    elapsed = (time.perf_counter() - started) * 1000
                span.set_attribute("tabcomplete.timing.total_ms", elapsed)
                span.set_attribute("tabcomplete.timing.kind", "end_to_end")
                usage_metrics(result, labels)
                for key, field in [
                    ("gen_ai.usage.input_tokens", "input_tokens"),
                    ("gen_ai.usage.output_tokens", "tokens"),
                    ("tabcomplete.usage.source", "usage_source"),
                ]:
                    value = getattr(result, field, None)
                    if field == "tokens" and not getattr(result, "output_tokens_known", True):
                        continue
                    if value is not None:
                        span.set_attribute(key, value)
                if result.finish_reason is not None:
                    span.set_attribute("gen_ai.response.finish_reasons", [result.finish_reason])
                for key, value in (
                    store.capture_text(
                        "model-output", result.text, authorized=runtime.config.capture_content
                    )
                    .attributes("output")
                    .items()
                ):
                    span.set_attribute(key, value)
                return result

    return call


@contextlib.contextmanager
def model_metrics(labels: dict[str, str]):
    record_metric("tabcomplete.model.active", 1, labels, kind="updown")
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        record_metric("tabcomplete.model.failures", 1, labels)
        raise
    finally:
        record_metric("tabcomplete.model.active", -1, labels, kind="updown")
        record_metric("tabcomplete.model.calls", 1, labels)
        record_metric(
            "tabcomplete.model.duration",
            (time.perf_counter() - started) * 1000,
            labels,
            kind="histogram",
        )


def usage_metrics(generation, labels: dict[str, str]) -> None:
    for field, kind in [
        ("input_tokens", "input"),
        ("tokens", "output"),
        ("cache_tokens", "cache_read"),
    ]:
        value = getattr(generation, field, None)
        if field == "tokens" and not getattr(generation, "output_tokens_known", True):
            continue
        if value is not None:
            record_metric("tabcomplete.model.tokens", value, {**labels, "token_type": kind})
    first = getattr(generation, "first_output_ms", None)
    if first is not None:
        record_metric("tabcomplete.model.first_output", first, labels, kind="histogram")


def training_progress(record: object) -> None:
    """Called only where the authoritative trainer already writes scalar records."""
    if not current_runtime().config.enabled or not isinstance(record, dict):
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    mapping = {
        "loss": "loss",
        "gradient_norm": "gradient_norm",
        "learning_rate": "lr",
        "training_tokens": "input_tokens",
        "optimizer_step": "successful_updates",
        "event": "event",
        "consecutive": "consecutive_scaler_events",
    }
    attributes = {
        f"tabcomplete.training.{target}": record[source]
        for source, target in mapping.items()
        if source in record
    }
    with operation("training.progress", attributes=attributes, authoritative_rank=0, rank=0):
        pass
    for source, metric in {
        "loss": "loss",
        "gradient_norm": "gradient_norm",
        "learning_rate": "learning_rate",
        "training_tokens": "input_tokens",
        "optimizer_step": "updates",
    }.items():
        value = record.get(source)
        if isinstance(value, (int, float)):
            record_metric(
                "tabcomplete.training." + metric,
                value,
                {"rank_role": "authoritative"},
                kind="gauge",
            )
    if record.get("event") == "loss_scale_overflow":
        record_metric("tabcomplete.training.skipped_updates", 1, {"rank_role": "authoritative"})


def observed_next_edit(function: Callable) -> Callable:
    @functools.wraps(function)
    def call(case, prediction, **kwargs):
        if not current_runtime().config.enabled:
            return function(case, prediction, **kwargs)
        active = current_run_context() or RunContext.new()
        context = active if active.case_id == case.id else active.for_case(case.id)
        with context.activate(), operation("eval.next_edit") as span:
            result = function(case, prediction, **kwargs)
            for field in (
                "predicted_action",
                "gold_action",
                "action_correct",
                "deletion_success",
                "valid_patch",
                "parse_status",
                "functional_success",
            ):
                value = getattr(result, field, None)
                if value is not None:
                    span.set_attribute("tabcomplete.next_edit." + field, value)
            span.set_attribute(
                "tabcomplete.terminal_event_id",
                f"{context.run_id}:{context.case_id}:{context.case_attempt_id}:next-edit",
            )
            return result

    return call


def observed_completion(function: Callable) -> Callable:
    """Playground boundary hook. Provider arguments and returned Completion are untouched."""

    @functools.wraps(function)
    def call(provider, *, model, prompt, max_new_tokens):
        runtime = current_runtime()
        if not runtime.config.enabled:
            return function(provider, model=model, prompt=prompt, max_new_tokens=max_new_tokens)
        context = (current_run_context() or RunContext.new()).for_case("playground")
        store = ArtifactStore(runtime.config.artifact_root, enabled=runtime.config.capture_content)
        attrs = {
            "gen_ai.request.model": model,
            "gen_ai.request.max_tokens": max_new_tokens,
            "gen_ai.provider.name": type(provider).__name__,
            **store.capture_text(
                "model-input", prompt, authorized=runtime.config.capture_content
            ).attributes("input"),
        }
        with context.activate(), operation("model.generate", attributes=attrs) as span:
            with model_metrics({"backend": type(provider).__name__}):
                started = time.perf_counter()
                result = function(
                    provider, model=model, prompt=prompt, max_new_tokens=max_new_tokens
                )
                duration = (time.perf_counter() - started) * 1000
            span.set_attribute("tabcomplete.timing.total_ms", duration)
            span.set_attribute("tabcomplete.timing.kind", "end_to_end")
            if result.generated_tokens > 0:
                span.set_attribute("gen_ai.usage.output_tokens", result.generated_tokens)
            for key, value in (
                store.capture_text(
                    "model-output", result.text, authorized=runtime.config.capture_content
                )
                .attributes("output")
                .items()
            ):
                span.set_attribute(key, value)
            return result

    return call
