"""Run lifecycle and heartbeat independent of long-lived work spans."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .bootstrap import current_runtime
from .context import RunContext, current_run_context
from .spans import operation


@contextlib.contextmanager
def run_scope(metadata_path: Path, phase: str):
    runtime = current_runtime()
    if not runtime.config.enabled:
        yield current_run_context()
        return
    inherited = current_run_context()
    rank = int(os.environ.get("RANK", "0"))
    try:
        if metadata_path.exists():
            stored = json.loads(metadata_path.read_text())
            run = RunContext.new(campaign_id=stored["campaign_id"], run_id=stored["run_id"])
        else:
            seed = hashlib.sha256(str(metadata_path.resolve()).encode()).hexdigest()[:32]
            run = RunContext.new(
                campaign_id=inherited.campaign_id if inherited else "campaign-" + seed,
                run_id="run-" + seed,
            )
            if rank == 0:
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with metadata_path.open("x") as handle:
                    json.dump({"campaign_id": run.campaign_id, "run_id": run.run_id}, handle)
    except (OSError, ValueError, KeyError):
        run = inherited or RunContext.new()
    stop = threading.Event()

    def emit(name, state):
        if rank != 0:
            return
        with (
            runtime.activate(),
            run.activate(),
            operation(
                name, attributes={"tabcomplete.phase": phase, "tabcomplete.run.state": state}
            ),
        ):
            pass

    def heartbeat():
        while not stop.wait(runtime.config.heartbeat_seconds):
            emit("run.heartbeat", "active")

    thread = threading.Thread(target=heartbeat, daemon=True, name="tabcomplete-run-heartbeat")
    state = "completed"
    with run.activate():
        emit("run.start", "started")
        thread.start()
        try:
            yield run
        except BaseException:
            state = "failed"
            raise
        finally:
            stop.set()
            thread.join(timeout=1)
            emit("run.summary", state)


def observed_run(metadata_path: Callable, phase: str):
    def decorate(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            if not current_runtime().config.enabled:
                return function(*args, **kwargs)
            with run_scope(metadata_path(*args, **kwargs), phase):
                return function(*args, **kwargs)

        return call

    return decorate


@contextlib.contextmanager
def evaluation_scope(args, protocol: str):
    runtime = current_runtime()
    if not runtime.config.enabled:
        yield
        return
    metadata_path = args.output_dir / "observability-run.json"
    source = {}
    predictions = getattr(args, "predictions", None)
    if predictions:
        try:
            source = json.loads(
                predictions.with_suffix(predictions.suffix + ".metadata.json").read_text()
            )
            if "observability" in source and not metadata_path.exists():
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps(source["observability"]))
        except (OSError, ValueError):
            pass
    with run_scope(metadata_path, "evaluation") as run:
        workload = {
            "tabcomplete.protocol": source.get("protocol", protocol),
            "tabcomplete.suite.sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "gen_ai.request.model": source.get(
                "model_source", "gold" if getattr(args, "gold", False) else "unknown"
            ),
            "tabcomplete.decoding": json.dumps(source.get("decoding", {}), sort_keys=True),
        }
        with replace(run, workload=workload).activate():
            yield


def context_map(executor, function, items):
    from .context import context_callable

    # Each task gets a distinct Context, retaining executor.map's ordered results.
    futures = [
        executor.submit(context_callable(lambda item=item: function(item))) for item in items
    ]
    for future in futures:
        yield future.result()
