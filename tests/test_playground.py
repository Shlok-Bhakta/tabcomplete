"""Real HTTP contract for the local browser playground."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
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
        assert "[[EDIT]]" in prompt
        assert "[[/EDIT]]" in prompt
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
        assert "/api/telemetry/events" in html
        assert "vendor/codemirror.min.js" in html
        assert "ghost-widget" in html
        assert '"Tab"' in html
        assert "innerHTML" not in html
        assert 'class="card"' not in html
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
                "mode": "next_edit",
                "language": "python",
                "path": "src/math.py",
                "prefix": "def add(a, b):\n    ",
                "region": "return a - b",
                "suffix": "\n",
                "recent_edits": ["<I src/math.py 0> # fix arithmetic"],
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


def test_next_edit_prompt_marks_region_and_includes_editor_state(tmp_path: Path):
    app = PlaygroundServer(provider=FakeProvider(), feedback_path=tmp_path / "feedback.jsonl")
    prompt = app.serialize_next_edit_prompt(
        path="src/math.py",
        prefix="def add(a, b):\n    ",
        region="return a - b",
        suffix="\n",
        recent_edits=["<I src/math.py 0> # fix arithmetic"],
        context_files=[{"path": "src/types.py", "content": "Number = int\n"}],
    )
    assert prompt.startswith("<repo src/math.py>\n")
    assert "<I src/math.py 0> # fix arithmetic\n" in prompt
    assert "<context src/types.py>\nNumber = int\n\n</context>" in prompt
    assert "[[EDIT]]return a - b[[/EDIT]]" in prompt
    assert prompt.endswith("<P src/math.py 19>\n")


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


def test_telemetry_proxy_forwards_events_to_collector(tmp_path: Path, monkeypatch):
    seen: dict = {}

    class Response:
        status_code = 200

        def json(self) -> dict:
            return {"ingested": 1, "skipped_duplicate": 0, "total": 1}

    def fake_post(url: str, *, json: dict, timeout: int):
        seen["url"] = url
        seen["body"] = json
        return Response()

    monkeypatch.setattr("tinycomplete.playground.server.httpx.post", fake_post)
    app = PlaygroundServer(
        provider=FakeProvider(),
        feedback_path=tmp_path / "feedback.jsonl",
        collector_url="http://collector.test:8787",
        host="127.0.0.1",
        port=0,
    )
    server = app.build_http_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = request_json(
            base + "/api/telemetry/events",
            "POST",
            {"protocol_version": 1, "events": []},
        )
        assert status == 200
        assert body == {"ingested": 1, "skipped_duplicate": 0, "total": 1}
        assert seen["url"] == "http://collector.test:8787/v1/events/batch"
        assert seen["body"]["protocol_version"] == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_telemetry_status_reports_unreachable_collector(tmp_path: Path, monkeypatch):
    def fake_get(url: str, *, timeout: int):
        raise ConnectionError("down")

    monkeypatch.setattr("tinycomplete.playground.server.httpx.get", fake_get)
    app = PlaygroundServer(
        provider=FakeProvider(),
        feedback_path=tmp_path / "feedback.jsonl",
        collector_url="http://collector.test:8787",
    )
    status_info = app._telemetry_status()
    assert status_info["reachable"] is False
    assert status_info["collector_url"] == "http://collector.test:8787"


def test_static_vendor_files_are_served(tmp_path: Path):
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
        with urllib.request.urlopen(base + "/vendor/codemirror.min.js", timeout=5) as response:
            assert response.status == 200
            assert "CodeMirror" in response.read().decode()[:2000]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_inline_sha256_matches_hashlib_vectors(tmp_path: Path):
    """The page's pure-JS SHA-256 must agree with hashlib.

    Collector blob addressing depends on it: a wrong digest means the
    check-then-upload flow stores nothing and anchors dangle (blob_uploaded
    false). Runs the shipped function in node when available.
    """
    if shutil.which("node") is None:
        import pytest

        pytest.skip("node is required to execute the shipped JS")
    static_path = Path(__file__).parent.parent / "src" / "tinycomplete" / "playground" / "static"
    html = (static_path / "index.html").read_text(encoding="utf-8")
    assert "\x00" not in html
    match = re.search(
        r"function sha256ascii\(ascii\) \{.*?\n  \}\n  function sha256\(s\) \{[^\n]*\}",
        html,
        re.S,
    )
    assert match is not None
    cases = ["abc", "", "x" * 200, "héllo ✓\n", "a" * 55, "a" * 56, "a" * 64]
    vectors = [[text, hashlib.sha256(text.encode("utf-8")).hexdigest()] for text in cases]
    runner = tmp_path / "sha_check.js"
    runner.write_text(
        match.group(0)
        + "\nconst vectors = "
        + json.dumps(vectors)
        + ";\nlet failed = 0;\nfor (const [input, expected] of vectors) {\n"
        + "  const got = sha256(input);\n"
        + "  if (got !== expected) { failed += 1; console.error('MISMATCH'); }\n"
        + "}\nif (failed > 0) { process.exit(1); }\nconsole.log('sha256 ok');\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(runner)], capture_output=True, text=True, timeout=30, check=False
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
