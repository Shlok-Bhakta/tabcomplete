"""Dependency-light HTTP server for the local TabComplete playground."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from tinycomplete.observability.hooks import observed, observed_completion, observed_http_request


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    format: str


@dataclass(frozen=True)
class Completion:
    text: str
    generated_tokens: int
    latency_ms: float


class CompletionProvider(Protocol):
    def models(self) -> list[ModelInfo]: ...

    def complete(self, *, model: str, prompt: str, max_new_tokens: int) -> Completion: ...


class TransformersProvider:
    """Lazy CPU-safe Transformers provider for one or more HF checkpoints."""

    def __init__(self, model_paths: dict[str, str]) -> None:
        if not model_paths:
            raise ValueError("at least one model path is required")
        self.model_paths = dict(model_paths)
        self._loaded: dict[str, tuple[object, object]] = {}
        self._lock = threading.Lock()

    def models(self) -> list[ModelInfo]:
        return [
            ModelInfo(id=model_id, label=model_id, format="transformers")
            for model_id in self.model_paths
        ]

    @observed("model.load")
    def _load(self, model_id: str):
        if model_id not in self.model_paths:
            raise ValueError(f"unknown model: {model_id}")
        if model_id not in self._loaded:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = self.model_paths[model_id]
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=False,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.eval()
            self._loaded[model_id] = (model, tokenizer)
        return self._loaded[model_id]

    @observed_completion
    def complete(self, *, model: str, prompt: str, max_new_tokens: int) -> Completion:
        import torch

        with self._lock, torch.inference_mode():
            loaded, tokenizer = self._load(model)
            inputs = tokenizer(prompt, return_tensors="pt")
            started = time.perf_counter()
            output = loaded.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return Completion(
                text=text,
                generated_tokens=int(new_tokens.numel()),
                latency_ms=(time.perf_counter() - started) * 1000,
            )


class OpenAICompatibleProvider:
    """Proxy a local llama.cpp/Ollama completion server to the browser UI."""

    def __init__(self, server_url: str, model: str, label: str | None = None) -> None:
        self.url = server_url.rstrip("/") + "/v1/completions"
        self.model = model
        self.label = label or model

    def models(self) -> list[ModelInfo]:
        return [ModelInfo(id=self.model, label=self.label, format="openai-compatible")]

    @observed_completion
    def complete(self, *, model: str, prompt: str, max_new_tokens: int) -> Completion:
        if model != self.model:
            raise ValueError(f"unknown model: {model}")
        started = time.perf_counter()
        response = httpx.post(
            self.url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
        return Completion(
            text=str(body["choices"][0]["text"]),
            generated_tokens=int(body.get("usage", {}).get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class PlaygroundServer:
    TELEMETRY_ROUTES = {
        "/api/telemetry/session": "/v1/session/start",
        "/api/telemetry/session/end": "/v1/session/end",
        "/api/telemetry/events": "/v1/events/batch",
        "/api/telemetry/blobs/check": "/v1/blobs/check",
        "/api/telemetry/blobs/upload": "/v1/blobs/upload",
        "/api/telemetry/snapshot": "/v1/repository/snapshot",
    }

    def __init__(
        self,
        *,
        provider: CompletionProvider,
        feedback_path: Path,
        host: str = "127.0.0.1",
        port: int = 8765,
        collector_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.feedback_path = feedback_path
        self._feedback_lock = threading.Lock()
        self.host = host
        self.port = port
        raw = collector_url or os.environ.get("TABCOMPLETE_COLLECTOR_URL", "http://crabcake:8787")
        self.collector_url = raw.rstrip("/")

    @staticmethod
    def serialize_prompt(*, language: str, prefix: str, context_files: list[dict[str, str]]) -> str:
        pieces = []
        for item in sorted(context_files, key=lambda value: value["path"]):
            path = item["path"].replace('"', "")
            pieces.append(f'<file path="{path}">\n{item["content"]}\n</file>\n')
        pieces.append(f'<target language="{language}">\n{prefix}')
        return "".join(pieces)

    @staticmethod
    def serialize_next_edit_prompt(
        *,
        path: str,
        prefix: str,
        region: str,
        suffix: str,
        recent_edits: list[str],
        context_files: list[dict[str, str]],
    ) -> str:
        """Serialize the marked-region next-edit protocol used by the benchmark."""
        safe_path = path.replace("\n", "").replace("\r", "") or "untitled.txt"
        pieces = [f"<repo {safe_path}>\n"]
        pieces.extend(f"{edit}\n" for edit in recent_edits if edit)
        for item in sorted(context_files, key=lambda value: value["path"]):
            context_path = item["path"].replace("\n", "").replace("\r", "")
            pieces.append(f"<context {context_path}>\n{item['content']}\n</context>\n")
        byte_offset = len(prefix.encode("utf-8"))
        pieces.append(
            f"<file {safe_path}>\n{prefix}[[EDIT]]{region}[[/EDIT]]{suffix}\n</file>\n"
            f"<P {safe_path} {byte_offset}>\n"
        )
        return "".join(pieces)

    def _complete(self, payload: dict) -> dict:
        model = str(payload.get("model", ""))
        language = str(payload.get("language", "text"))
        prefix = str(payload.get("prefix", ""))
        context_files = payload.get("context_files", [])
        if not isinstance(context_files, list):
            raise ValueError("context_files must be a list")
        safe_context = []
        for item in context_files:
            if not isinstance(item, dict) or "path" not in item or "content" not in item:
                raise ValueError("each context file needs path and content")
            safe_context.append({"path": str(item["path"]), "content": str(item["content"])})
        max_new_tokens = max(1, min(256, int(payload.get("max_new_tokens", 96))))
        mode = str(payload.get("mode", "next_edit"))
        if mode == "next_edit":
            region = str(payload.get("region", ""))
            suffix = str(payload.get("suffix", ""))
            path = str(payload.get("path", "untitled.txt"))
            raw_recent_edits = payload.get("recent_edits", [])
            if not isinstance(raw_recent_edits, list):
                raise ValueError("recent_edits must be a list")
            prompt = self.serialize_next_edit_prompt(
                path=path,
                prefix=prefix,
                region=region,
                suffix=suffix,
                recent_edits=[str(edit) for edit in raw_recent_edits],
                context_files=safe_context,
            )
        elif mode == "autocomplete":
            prompt = self.serialize_prompt(
                language=language,
                prefix=prefix,
                context_files=safe_context,
            )
        else:
            raise ValueError("mode must be next_edit or autocomplete")
        return asdict(
            self.provider.complete(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        )

    def _feedback(self, payload: dict) -> dict:
        allowed = {"request_id", "action", "model", "latency_ms", "edited_text"}
        record = {key: payload[key] for key in allowed if key in payload}
        if record.get("action") not in {"accept", "reject", "edit"}:
            raise ValueError("feedback action must be accept, reject, or edit")
        record["timestamp"] = time.time()
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._feedback_lock, self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return {"saved": True}

    def _telemetry_status(self) -> dict:
        try:
            response = httpx.get(self.collector_url + "/healthz", timeout=5)
            body = response.json()
            version = body.get("protocol_version") if isinstance(body, dict) else None
            return {
                "collector_url": self.collector_url,
                "reachable": bool(response.status_code == 200),
                "protocol_version": version,
            }
        except Exception as exc:
            return {
                "collector_url": self.collector_url,
                "reachable": False,
                "error": type(exc).__name__,
            }

    def _collector_forward(self, route: str, payload: dict) -> tuple[int, dict]:
        """Proxy a telemetry body to the trajectory collector (tailnet-only).

        Never raises: collector outages surface as 502 JSON so the editor
        keeps working offline.
        """
        try:
            response = httpx.post(self.collector_url + route, json=payload, timeout=15)
            try:
                body = response.json()
            except ValueError:
                body = {"raw": response.text[:500]}
            if not isinstance(body, dict):
                body = {"value": body}
            return response.status_code, body
        except Exception as exc:
            return 502, {"error": "collector_unreachable", "message": type(exc).__name__}

    def build_http_server(self) -> ThreadingHTTPServer:
        app = self
        static_dir = Path(__file__).with_name("static")
        static_root = static_dir.resolve()
        index_path = static_dir / "index.html"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                return

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                policy = (
                    "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "script-src 'self' 'unsafe-inline'"
                )
                self.send_header("Content-Security-Policy", policy)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, value: object) -> None:
                self._send(status, json.dumps(value).encode(), "application/json")

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/":
                    if index_path.exists():
                        body = index_path.read_bytes()
                    else:
                        body = (
                            b"<!doctype html><title>TabComplete</title>"
                            b"<script>fetch('/api/complete')</script>"
                        )
                    self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
                elif path == "/api/models":
                    models = [asdict(model) for model in app.provider.models()]
                    self._json(HTTPStatus.OK, {"models": models})
                elif path == "/api/health":
                    self._json(HTTPStatus.OK, {"ok": True})
                elif path == "/api/telemetry/status":
                    self._json(HTTPStatus.OK, app._telemetry_status())
                else:
                    candidate = (static_dir / path.lstrip("/")).resolve()
                    if (
                        candidate.is_file()
                        and candidate != static_root
                        and static_root in candidate.parents
                    ):
                        content_type, _ = mimetypes.guess_type(str(candidate))
                        self._send(
                            HTTPStatus.OK,
                            candidate.read_bytes(),
                            content_type or "application/octet-stream",
                        )
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            @observed_http_request
            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size <= 0 or size > 2 * 2**20:
                        raise ValueError("request body must be between 1 byte and 2 MiB")
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError("JSON body must be an object")
                    if path == "/api/complete":
                        self._json(HTTPStatus.OK, app._complete(payload))
                    elif path == "/api/feedback":
                        self._json(HTTPStatus.OK, app._feedback(payload))
                    elif path in PlaygroundServer.TELEMETRY_ROUTES:
                        status, body = app._collector_forward(
                            PlaygroundServer.TELEMETRY_ROUTES[path], payload
                        )
                        self._json(status, body)
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                except (ValueError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__})

        return ThreadingHTTPServer((self.host, self.port), Handler)

    def serve_forever(self) -> None:
        server = self.build_http_server()
        print(f"TabComplete playground: http://{self.host}:{server.server_port}")
        try:
            server.serve_forever()
        finally:
            server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", action="append", metavar="ID=PATH")
    source.add_argument("--server-url")
    parser.add_argument("--server-model", default="local-model")
    parser.add_argument("--server-label")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--feedback", type=Path, default=Path("data/playground/feedback.jsonl"))
    parser.add_argument(
        "--collector-url",
        default=None,
        help="Trajectory collector base URL (default: $TABCOMPLETE_COLLECTOR_URL or http://crabcake:8787)",
    )
    args = parser.parse_args()
    if args.model:
        provider: CompletionProvider = TransformersProvider(
            dict(value.split("=", 1) for value in args.model)
        )
    else:
        provider = OpenAICompatibleProvider(args.server_url, args.server_model, args.server_label)
    PlaygroundServer(
        provider=provider,
        feedback_path=args.feedback,
        host=args.host,
        port=args.port,
        collector_url=args.collector_url,
    ).serve_forever()


if __name__ == "__main__":
    main()
