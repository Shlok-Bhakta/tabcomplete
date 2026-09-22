"""Bounded offline span bundles and an idempotent import ledger."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class OfflineSpanExporter(SpanExporter):
    def __init__(self, path: Path, *, max_bytes: int):
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self.dropped = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                for span in spans:
                    context = span.context
                    if context is None:
                        continue
                    record = {
                        "schema_version": 1,
                        "historical": True,
                        "name": span.name,
                        "trace_id": f"{context.trace_id:032x}",
                        "span_id": f"{context.span_id:016x}",
                        "parent_span_id": (
                            f"{span.parent.span_id:016x}" if span.parent is not None else None
                        ),
                        "start_time_unix_nano": span.start_time,
                        "end_time_unix_nano": span.end_time,
                        "attributes": dict(span.attributes or {}),
                        "status": span.status.status_code.name,
                        "resource": dict(span.resource.attributes),
                        "events": [
                            {
                                "name": e.name,
                                "timestamp": e.timestamp,
                                "attributes": dict(e.attributes or {}),
                            }
                            for e in span.events
                        ],
                        "links": [
                            {
                                "trace_id": f"{link.context.trace_id:032x}",
                                "span_id": f"{link.context.span_id:016x}",
                                "attributes": dict(link.attributes or {}),
                            }
                            for link in span.links
                        ],
                    }
                    data = (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                    if handle.tell() + len(data) > self.max_bytes:
                        self.dropped += 1
                        continue
                    handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        return SpanExportResult.SUCCESS


@dataclass(frozen=True)
class ImportClaim:
    bundle_sha256: str
    imported: bool


class OfflineImportLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS imports (bundle_sha256 TEXT PRIMARY KEY, "
                "imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def claim(self, bundle: Path) -> ImportClaim:
        digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO imports(bundle_sha256) VALUES (?)", (digest,)
            )
            return ImportClaim(bundle_sha256=digest, imported=cursor.rowcount == 1)


def import_bundle(bundle: Path, ledger_path: Path, endpoint: str) -> dict[str, object]:
    """Replay observed spans only, retaining timestamps; commit ledger after export.

    A crash after OTLP acceptance but before the ledger commit can replay spans.
    Trace/span IDs remain unchanged so consumers can deduplicate them.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import Event
    from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags

    if bundle.stat().st_size > 256 * 2**20:
        raise ValueError("offline bundle exceeds 256 MiB")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    ledger = OfflineImportLedger(ledger_path)
    exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces", timeout=10)
    with ledger._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute("SELECT 1 FROM imports WHERE bundle_sha256=?", (digest,)).fetchone():
            return {
                "schema_version": 1,
                "bundle_sha256": digest,
                "duplicate": True,
                "imported_spans": 0,
            }
        spans = []
        count = 0

        def context(trace_id: str, span_id: str) -> SpanContext:
            return SpanContext(int(trace_id, 16), int(span_id, 16), False, TraceFlags(1))

        with bundle.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("schema_version") != 1:
                    raise ValueError("unsupported offline schema")
                ctx = context(record["trace_id"], record["span_id"])
                spans.append(
                    ReadableSpan(
                        name=record["name"],
                        context=ctx,
                        parent=context(record["trace_id"], record["parent_span_id"])
                        if record.get("parent_span_id")
                        else None,
                        resource=Resource(record["resource"]),
                        attributes={
                            **record["attributes"],
                            "tabcomplete.historical": True,
                            "tabcomplete.offline.bundle_sha256": digest,
                        },
                        start_time=record["start_time_unix_nano"],
                        end_time=record["end_time_unix_nano"],
                        status=Status(StatusCode[record["status"]]),
                        events=[
                            Event(e["name"], e["attributes"], e["timestamp"])
                            for e in record.get("events", [])
                        ],
                        links=[
                            Link(context(e["trace_id"], e["span_id"]), e["attributes"])
                            for e in record.get("links", [])
                        ],
                    )
                )
                count += 1
                if len(spans) == 128:
                    if exporter.export(spans) != SpanExportResult.SUCCESS:
                        raise RuntimeError("offline export failed; ledger not committed")
                    spans.clear()
        if spans and exporter.export(spans) != SpanExportResult.SUCCESS:
            raise RuntimeError("offline export failed; ledger not committed")
        connection.execute("INSERT INTO imports(bundle_sha256) VALUES (?)", (digest,))
    exporter.shutdown()
    return {
        "schema_version": 1,
        "bundle_sha256": digest,
        "duplicate": False,
        "imported_spans": count,
    }
