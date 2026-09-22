"""Deduplicated, size-bounded prompt and response storage."""

from __future__ import annotations

import atexit
import gzip
import hashlib
import os
import queue
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .spans import sanitize_text

_HASH = re.compile(r"^[0-9a-f]{64}$")


class CaptureStatus(StrEnum):
    CAPTURED = "captured"
    REDACTED = "redacted"
    OMITTED_SIZE = "omitted_size"
    OMITTED_UNAUTHORIZED = "omitted_unauthorized"
    DISABLED = "disabled"
    OMITTED_STORAGE = "omitted_storage"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str | None
    sha256: str | None
    bytes: int
    stored_bytes: int
    status: CaptureStatus
    kind: str
    preview: str
    source_ref: str | None = None

    def attributes(self, prefix: str) -> dict[str, str | int]:
        values = asdict(self)
        return {
            f"tabcomplete.artifact.{prefix}.{key}": str(value)
            if isinstance(value, StrEnum)
            else value
            for key, value in values.items()
            if value is not None
        }


class ArtifactStore:
    def __init__(self, root: Path, *, max_payload_bytes: int = 8 * 2**20, enabled: bool = True):
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        self.root = root
        self.max_payload_bytes = max_payload_bytes
        self.enabled = enabled

    def capture_text(
        self, kind: str, text: str, *, authorized: bool, source_ref: str | None = None
    ) -> ArtifactRef:
        if not self.enabled:
            return ArtifactRef(None, None, 0, 0, CaptureStatus.DISABLED, kind, "")
        try:
            return self._capture_text(kind, text, authorized=authorized, source_ref=source_ref)
        except (OSError, ValueError):
            return ArtifactRef(
                None,
                None,
                len(text.encode("utf-8")),
                0,
                CaptureStatus.OMITTED_STORAGE,
                kind,
                "",
                source_ref,
            )

    def _capture_text(
        self,
        kind: str,
        text: str,
        *,
        authorized: bool,
        source_ref: str | None = None,
    ) -> ArtifactRef:
        raw = text.encode("utf-8")
        preview = sanitize_text(text, limit=240) if self.enabled and authorized else ""
        if not self.enabled:
            return ArtifactRef(None, None, len(raw), 0, CaptureStatus.DISABLED, kind, preview)
        if not authorized:
            return ArtifactRef(
                None,
                None,
                len(raw),
                0,
                CaptureStatus.OMITTED_UNAUTHORIZED,
                kind,
                preview,
                source_ref,
            )
        if len(raw) > self.max_payload_bytes:
            return ArtifactRef(
                None,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                0,
                CaptureStatus.OMITTED_SIZE,
                kind,
                preview,
                source_ref,
            )
        sanitized = sanitize_text(text, limit=max(len(text), 1))
        status = CaptureStatus.REDACTED if sanitized != text else CaptureStatus.CAPTURED
        stored = sanitized.encode("utf-8")
        digest = hashlib.sha256(stored).hexdigest()
        destination = self._path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            cap = int(os.environ.get("TABCOMPLETE_ARTIFACT_CAP_BYTES", str(20 * 2**30)))
            size = (
                sum(p.stat().st_size for p in self.root.glob("*/*.gz")) if self.root.exists() else 0
            )
            if size + len(stored) > cap:
                return ArtifactRef(
                    None,
                    digest,
                    len(raw),
                    0,
                    CaptureStatus.OMITTED_STORAGE,
                    kind,
                    preview,
                    source_ref,
                )
            compressed = gzip.compress(stored, compresslevel=6, mtime=0)
            fd, temporary_name = tempfile.mkstemp(prefix=".artifact-", dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(compressed)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        _enqueue_transfer(destination, digest)
        return ArtifactRef(
            f"sha256:{digest}",
            digest,
            len(raw),
            destination.stat().st_size,
            status,
            kind,
            preview,
            source_ref,
        )

    def read_text(self, sha256: str) -> str:
        if not _HASH.fullmatch(sha256):
            raise ValueError("artifact hash must be 64 lowercase hexadecimal characters")
        path = self._path(sha256)
        return gzip.decompress(path.read_bytes()).decode("utf-8")

    def _path(self, sha256: str) -> Path:
        return self.root / sha256[:2] / f"{sha256}.txt.gz"


_uploads: queue.Queue[tuple[Path, str]] = queue.Queue(maxsize=128)
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


def _enqueue_transfer(path: Path, digest: str) -> None:
    global _worker
    if not os.environ.get("TABCOMPLETE_ARTIFACT_UPLOAD_URL"):
        return
    with _worker_lock:
        if _worker is None:
            _worker = threading.Thread(
                target=_transfer_worker, daemon=True, name="tabcomplete-artifact-transfer"
            )
            _worker.start()
            atexit.register(flush_artifact_transfers)
    try:
        _uploads.put_nowait((path, digest))
    except queue.Full:
        pass  # The local CAS file remains available for a later explicit sync.


def _transfer_worker() -> None:
    import httpx

    while True:
        path, digest = _uploads.get()
        try:
            token_path = os.environ.get("TABCOMPLETE_ARTIFACT_UPLOAD_TOKEN_FILE")
            if token_path:
                token = Path(token_path).read_text().strip()
                payload = gzip.decompress(path.read_bytes())
                httpx.put(
                    os.environ["TABCOMPLETE_ARTIFACT_UPLOAD_URL"].rstrip("/")
                    + "/internal/v1/artifacts/"
                    + digest,
                    content=payload,
                    headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "text/plain; charset=utf-8",
                    },
                    timeout=5,
                ).raise_for_status()
        except Exception:
            pass  # Keep local content. Never include tokens or payload in error logs.
        finally:
            _uploads.task_done()


def flush_artifact_transfers(timeout: float = 10) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while _uploads.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    return not _uploads.unfinished_tasks


def sync_artifacts(root: Path) -> dict[str, int]:
    """Explicit bounded-spool recovery; server dedup makes retries safe."""
    import httpx

    endpoint = os.environ["TABCOMPLETE_ARTIFACT_UPLOAD_URL"].rstrip("/")
    token = Path(os.environ["TABCOMPLETE_ARTIFACT_UPLOAD_TOKEN_FILE"]).read_text().strip()
    result = {"accepted": 0, "failed": 0}
    with httpx.Client(timeout=10) as client:
        for path in root.glob("*/*.txt.gz"):
            digest = path.name.removesuffix(".txt.gz")
            if not _HASH.fullmatch(digest):
                continue
            try:
                with gzip.open(path, "rb") as stream:
                    payload = stream.read(8 * 2**20 + 1)
                if len(payload) > 8 * 2**20 or hashlib.sha256(payload).hexdigest() != digest:
                    result["failed"] += 1
                    continue
                client.put(
                    endpoint + "/internal/v1/artifacts/" + digest,
                    content=payload,
                    headers={"Authorization": "Bearer " + token},
                ).raise_for_status()
                result["accepted"] += 1
            except (OSError, httpx.HTTPError):
                result["failed"] += 1
    return result
