"""Opt-in repository pytest telemetry. Assertions and pytest exit codes are unchanged."""

from __future__ import annotations

import hashlib
import threading

import pytest

from tinycomplete.observability.bootstrap import current_runtime
from tinycomplete.observability.context import RunContext
from tinycomplete.observability.spans import operation

_run: RunContext | None = None
_completed = 0
_failed = 0
_stop = threading.Event()
_heartbeat: threading.Thread | None = None


def _progress():
    while not _stop.wait(15):
        if _run is not None:
            with (
                _run.activate(),
                operation(
                    "run.heartbeat",
                    attributes={
                        "tabcomplete.phase": "pytest",
                        "tabcomplete.cases.completed": _completed,
                        "tabcomplete.run.state": "active",
                    },
                ),
            ):
                pass


def pytest_sessionstart(session):
    global _run, _completed, _failed, _heartbeat
    if not current_runtime().config.enabled:
        return
    _run = RunContext.new()
    _completed = _failed = 0
    _stop.clear()
    _heartbeat = threading.Thread(target=_progress, daemon=True)
    _heartbeat.start()
    with (
        _run.activate(),
        operation(
            "run.start",
            attributes={"tabcomplete.phase": "pytest", "tabcomplete.run.state": "started"},
        ),
    ):
        pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if _run is None:
        yield
        return
    case = _run.for_case(hashlib.sha256(item.nodeid.encode()).hexdigest()[:24])
    with (
        case.activate(),
        operation(
            "eval.case",
            attributes={"tabcomplete.task": "pytest", "tabcomplete.case.name": item.nodeid},
        ),
    ):
        yield


def pytest_runtest_logreport(report):
    global _completed, _failed
    if report.when == "call":
        _completed += 1
        _failed += int(report.failed)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _run is None or not (report.when == "call" or report.failed):
        return
    # This hook still runs inside the protocol's case context. Do not copy
    # assertion text: it may contain secrets or private fixture contents.
    from tinycomplete.observability.context import current_run_context

    context = current_run_context()
    if context is None:
        return
    with operation(
        "eval.pytest.result",
        attributes={
            "tabcomplete.task": "pytest",
            "tabcomplete.case.name": item.nodeid,
            "tabcomplete.quality.functional": "fail"
            if report.failed
            else "pass"
            if report.passed
            else "skipped",
            "tabcomplete.failure.stage": report.when if report.failed else "none",
            "tabcomplete.terminal_event_id": (
                f"{context.run_id}:{context.case_id}:{context.case_attempt_id}:{report.when}"
            ),
        },
    ):
        pass


def pytest_sessionfinish(session, exitstatus):
    _stop.set()
    if _heartbeat is not None:
        _heartbeat.join(timeout=1)
    if _run is None:
        return
    with (
        _run.activate(),
        operation(
            "run.summary",
            attributes={
                "tabcomplete.phase": "pytest",
                "tabcomplete.run.state": "completed" if exitstatus == 0 else "failed",
                "tabcomplete.cases.completed": _completed,
                "tabcomplete.cases.failed": _failed,
                "tabcomplete.cases.planned": session.testscollected,
            },
        ),
    ):
        pass
    current_runtime().force_flush()
