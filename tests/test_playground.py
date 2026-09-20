"""Real HTTP contract for the local browser playground."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from tinycomplete.playground.server import (
    Completion,
    ModelInfo,
    OpenAICompatibleProvider,
    PlaygroundServer,
)


class FakeProvider:
    def models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake/base", label="Fake base", format="test")]

    def complete(self, *, model: str, prompt: str, max_new_tokens: int) -> Completion:
        assert model == "fake/base"
        assert "def add(a, b):" in prompt
        assert max_new_tokens == 32
        return Completion(text="return a + b", generated_tokens=4, latency_ms=12.5)


def request_json(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_real_http_server_serves_ui_models_completion_and_feedback(tmp_path: Path):
    app = PlaygroundServer(
        provider=FakeProvider(),
        feedback_path=tmp_path / "feedback.jsonl",
        host="127.0.0.1",
        port=0,
    )
    server = app.build_http_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode()
        assert "TabComplete" in html
        assert "/api/complete" in html
        assert "/api/models" in html
        assert "/api/feedback" in html
        assert 'id="ghost"' in html
        assert 'e.key === "Tab"' in html
        assert 'e.key === "Escape"' in html
        assert "innerHTML" not in html
        for language in (
            "python",
            "typescript",
            "javascript",
            "java",
            "cpp",
            "rust",
            "go",
            "c",
            "csharp",
        ):
            assert f'value="{language}"' in html

        status, models = request_json(base + "/api/models")
        assert status == 200
        assert models["models"][0]["id"] == "fake/base"

        status, completion = request_json(
            base + "/api/complete",
            "POST",
            {
                "model": "fake/base",
                "language": "python",
                "prefix": "def add(a, b):\n    ",
                "suffix": "",
                "context_files": [{"path": "types.py", "content": "Number = int\n"}],
                "max_new_tokens": 32,
            },
        )
        assert status == 200
        assert completion == {
            "text": "return a + b",
            "generated_tokens": 4,
            "latency_ms": 12.5,
        }

        status, saved = request_json(
            base + "/api/feedback",
            "POST",
            {"request_id": "abc", "action": "accept", "model": "fake/base"},
        )
        assert status == 200
        assert saved == {"saved": True}
        record = json.loads((tmp_path / "feedback.jsonl").read_text())
        assert record["action"] == "accept"
        assert "source" not in record
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_prompt_serialization_puts_target_prefix_last(tmp_path: Path):
    provider = FakeProvider()
    app = PlaygroundServer(provider=provider, feedback_path=tmp_path / "feedback.jsonl")
    prompt = app.serialize_prompt(
        language="python",
        prefix="def target():\n    ",
        context_files=[
            {"path": "z.py", "content": "Z = 1\n"},
            {"path": "a.py", "content": "A = 2\n"},
        ],
    )
    assert prompt.index('<file path="a.py">') < prompt.index('<file path="z.py">')
    assert prompt.endswith('<target language="python">\ndef target():\n    ')


def test_openai_compatible_provider_proxies_local_completion(monkeypatch):
    class Response:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict:
            return {
                "choices": [{"text": "return a + b"}],
                "usage": {"completion_tokens": 4},
            }

    def fake_post(url: str, *, json: dict, timeout: int):
        assert url == "http://127.0.0.1:8080/v1/completions"
        assert json["model"] == "tabcomplete-q4"
        assert json["temperature"] == 0
        assert timeout == 300
        return Response()

    monkeypatch.setattr("tinycomplete.playground.server.httpx.post", fake_post)
    provider = OpenAICompatibleProvider("http://127.0.0.1:8080", "tabcomplete-q4", "TabComplete Q4")
    completion = provider.complete(
        model="tabcomplete-q4", prompt="def add(a, b):\n    ", max_new_tokens=32
    )
    assert provider.models()[0].format == "openai-compatible"
    assert completion.text == "return a + b"
    assert completion.generated_tokens == 4
    assert completion.latency_ms >= 0
