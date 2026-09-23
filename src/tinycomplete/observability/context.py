"""Stable run identities and explicit concurrency propagation."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ParamSpec, TypeVar

from opentelemetry.context import attach
from opentelemetry.propagate import extract, inject
from opentelemetry.propagators.textmap import Setter

P = ParamSpec("P")
R = TypeVar("R")


def _identifier(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


@dataclass(frozen=True)
class RunContext:
    campaign_id: str
    run_id: str
    run_attempt_id: str
    case_id: str | None = None
    case_attempt_id: str | None = None
    request_id: str | None = None
    workload: dict[str, str | int | float | bool] = field(default_factory=dict)

    @classmethod
    def new(cls, *, campaign_id: str | None = None, run_id: str | None = None) -> RunContext:
        return cls(
            campaign_id=campaign_id or _identifier("campaign"),
            run_id=run_id or _identifier("run"),
            run_attempt_id=_identifier("attempt"),
        )

    def for_case(self, case_id: str, *, request_id: str | None = None) -> RunContext:
        return RunContext(
            campaign_id=self.campaign_id,
            run_id=self.run_id,
            run_attempt_id=self.run_attempt_id,
            case_id=case_id,
            case_attempt_id=_identifier("case-attempt"),
            request_id=request_id or _identifier("request"),
            workload=self.workload,
        )

    def attributes(self) -> dict[str, str | int | float | bool]:
        values = {
            "tabcomplete.campaign_id": self.campaign_id,
            "tabcomplete.run_id": self.run_id,
            "tabcomplete.run_attempt_id": self.run_attempt_id,
            "tabcomplete.case_id": self.case_id,
            "tabcomplete.case_attempt_id": self.case_attempt_id,
            "tabcomplete.request_id": self.request_id,
        }
        return {
            **self.workload,
            **{key: value for key, value in values.items() if value is not None},
        }

    @contextlib.contextmanager
    def activate(self) -> Iterator[RunContext]:
        token = _run_context.set(self)
        try:
            yield self
        finally:
            _run_context.reset(token)


@dataclass(frozen=True)
class PersistentRunIdentity:
    metadata: dict[str, object]
    context: RunContext


def ensure_persistent_run_identity(
    metadata_path: Path, metadata: Mapping[str, object]
) -> PersistentRunIdentity:
    """Keep the logical run ID across resume and assign a new process attempt."""
    scientific = dict(metadata)
    scientific.pop("observability", None)
    saved: dict[str, object] | None = None
    if metadata_path.exists():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("prediction metadata must be a JSON object")
        saved = loaded
        saved_scientific = dict(saved)
        saved_scientific.pop("observability", None)
        if saved_scientific != scientific:
            raise ValueError("scientific metadata does not match existing output")
    observed = saved.get("observability") if saved is not None else None
    if isinstance(observed, dict):
        campaign_id = str(observed["campaign_id"])
        run_id = str(observed["run_id"])
    else:
        campaign_id = str(scientific.get("campaign_id") or _identifier("campaign"))
        run_id = _identifier("run")
    enriched: dict[str, object] = {
        **scientific,
        "observability": {"campaign_id": campaign_id, "run_id": run_id},
    }
    return PersistentRunIdentity(
        metadata=enriched,
        context=RunContext.new(campaign_id=campaign_id, run_id=run_id),
    )


_run_context: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "tabcomplete_run_context", default=None
)


def current_run_context() -> RunContext | None:
    current = _run_context.get()
    if current is None and os.environ.get("TABCOMPLETE_RUN_ID"):
        current = RunContext(
            campaign_id=os.environ.get("TABCOMPLETE_CAMPAIGN_ID", "unknown"),
            run_id=os.environ["TABCOMPLETE_RUN_ID"],
            run_attempt_id=os.environ.get("TABCOMPLETE_RUN_ATTEMPT_ID", _identifier("attempt")),
            case_id=os.environ.get("TABCOMPLETE_CASE_ID"),
            case_attempt_id=os.environ.get("TABCOMPLETE_CASE_ATTEMPT_ID"),
            request_id=os.environ.get("TABCOMPLETE_REQUEST_ID"),
        )
        _run_context.set(current)
        if os.environ.get("TRACEPARENT"):
            attach(extract({"traceparent": os.environ["TRACEPARENT"]}))
    return current


def context_callable(function: Callable[P, R]) -> Callable[P, R]:
    copied = contextvars.copy_context()

    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        return copied.run(function, *args, **kwargs)

    return call


class _CarrierSetter(Setter[dict[str, str]]):
    def set(self, carrier: dict[str, str], key: str, value: str) -> None:
        carrier[key] = value


_SAFE_PARENT_ENV = {
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "TMPDIR",
}


def subprocess_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a child environment without copying secrets or OpenTelemetry baggage."""
    source = os.environ if base is None else base
    environment = {key: value for key, value in source.items() if key in _SAFE_PARENT_ENV}
    carrier: dict[str, str] = {}
    inject(carrier, setter=_CarrierSetter())
    if traceparent := carrier.get("traceparent"):
        environment["TRACEPARENT"] = traceparent
    run = current_run_context()
    if run is not None:
        environment.update(
            {
                "TABCOMPLETE_CAMPAIGN_ID": run.campaign_id,
                "TABCOMPLETE_RUN_ID": run.run_id,
                "TABCOMPLETE_RUN_ATTEMPT_ID": run.run_attempt_id,
            }
        )
        if run.case_id:
            environment["TABCOMPLETE_CASE_ID"] = run.case_id
        if run.case_attempt_id:
            environment["TABCOMPLETE_CASE_ATTEMPT_ID"] = run.case_attempt_id
        if run.request_id:
            environment["TABCOMPLETE_REQUEST_ID"] = run.request_id
    return environment
