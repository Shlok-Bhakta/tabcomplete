"""Teacher layer: prompt, fake pipeline, validation, adapters (mocked HTTP)."""

import hashlib
import json

import httpx
import pytest

from tinycomplete.teacher.base import (
    Candidate,
    EditableRegion,
    TeacherRequest,
    provider_from_name,
)
from tinycomplete.teacher.budget import Budget
from tinycomplete.teacher.deepseek import DeepSeekProvider
from tinycomplete.teacher.generate import build_request, run_generation
from tinycomplete.teacher.openrouter import OpenRouterProvider, TeacherError, sanitize_error
from tinycomplete.teacher.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_messages
from tinycomplete.teacher.validate import validate_candidate


def _request(n: int = 2) -> TeacherRequest:
    return TeacherRequest(
        state_id="s1",
        serialized_state="<F a.py>\n<S>\nx = 1\n</S>\n</F>\n<P a.py 5>\n",
        region=EditableRegion(path="a.py", start=0, end=5, text="x = 1"),
        repo_context="",
        recent_edits=("- replace [0,1): 'x' -> 'y'",),
        num_candidates=n,
    )


def test_prompt_semantics():
    assert "NOOP is valid" in SYSTEM_PROMPT
    assert "Do not include rationale" in SYSTEM_PROMPT
    assert "Do not undo the user's immediately preceding edit" in SYSTEM_PROMPT
    messages = build_messages(_request())
    assert messages[0]["role"] == "system"
    assert "<<<REGION\nx = 1\nREGION>>>" in messages[1]["content"]
    assert "exactly 2 distinct candidate(s)" in messages[1]["content"]
    assert set(RESPONSE_SCHEMA["properties"]) == {"candidates"}
    assert RESPONSE_SCHEMA["properties"]["candidates"]["items"]["properties"]["action"][
        "enum"
    ] == ["replace", "noop"]


def test_fake_provider_pipeline():
    import asyncio

    async def go():
        provider = provider_from_name("fake")
        return await provider.predict(_request())

    response = asyncio.run(go())
    assert response.provider == "fake"
    assert response.reported_cost_usd == 0.0
    assert len(response.candidates) == 2
    assert response.request_id == "fake-s1"
    assert len(response.raw_sha256) == 64
    assert response.validation.ok


def test_validation_rules():
    ok = validate_candidate(
        Candidate(action="replace", replacement="x = 2"),
        region_text="x = 1",
        full_text="x = 1\n",
        region_start=0,
        region_end=4,
    )
    assert ok.ok
    noop = validate_candidate(
        Candidate(action="noop", replacement=""),
        region_text="x = 1",
        full_text="x = 1\n",
        region_start=0,
        region_end=4,
    )
    assert noop.ok
    same = validate_candidate(
        Candidate(action="replace", replacement="x = 1"),
        region_text="x = 1",
        full_text="x = 1\n",
        region_start=0,
        region_end=4,
    )
    assert not same.ok
    reversal = validate_candidate(
        Candidate(action="replace", replacement="old"),
        region_text="new",
        full_text="new\n",
        region_start=0,
        region_end=3,
        prev_region_text="old",
    )
    assert not reversal.ok and any("reversal" in r for r in reversal.reasons)
    broken = validate_candidate(
        Candidate(action="replace", replacement="def f(:"),
        region_text="x = 1",
        full_text="x = 1\n",
        region_start=0,
        region_end=4,
    )
    assert not broken.ok and any("syntax" in r for r in broken.reasons)


def _openrouter_ok_transport() -> httpx.MockTransport:
    body = {
        "id": "gen-123",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "candidates": [
                                {"action": "replace", "replacement": "x = 2"},
                                {"action": "noop", "replacement": ""},
                            ]
                        }
                    )
                }
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content)
        assert payload["model"] == "meta/muse-spark-1.3-contributor"
        assert payload["response_format"]["type"] == "json_schema"
        assert "Bearer" not in json.dumps(payload)  # no secret in body
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_openrouter_payload_and_parse(tmp_path, monkeypatch):
    budget_path = tmp_path / "b.json"
    monkeypatch.setenv("ALLOW_PAID_SYNTHETIC", "1")
    import asyncio

    async def go():
        provider = OpenRouterProvider(api_key="fake-key", budget=Budget(path=str(budget_path)))
        payload = provider.build_payload(_request())
        assert payload["response_format"]["json_schema"]["schema"] == RESPONSE_SCHEMA
        async with httpx.AsyncClient(transport=_openrouter_ok_transport()) as client:
            original = provider._post
            calls = {"n": 0}

            async def counting_post(c, p, h):
                calls["n"] += 1
                assert h["Authorization"] == "Bearer fake-key"
                return await original(c, p, h)

            provider._post = counting_post  # type: ignore[method-assign]
            return await provider.predict(_request(), client=client), calls["n"]

    response, n = asyncio.run(go())
    assert n == 1
    assert response.prompt_tokens == 100
    assert response.request_id == "gen-123"
    assert response.candidates[0] == Candidate(action="replace", replacement="x = 2")
    assert len(response.raw_sha256) == 64


def test_retry_bounded_on_429(tmp_path, monkeypatch):
    budget_path = tmp_path / "b.json"
    monkeypatch.setenv("ALLOW_PAID_SYNTHETIC", "1")
    import asyncio

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(429, json={"error": "rate limited"})

    async def go():
        provider = OpenRouterProvider(api_key="fake-key", budget=Budget(path=str(budget_path)))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(TeacherError):
                await provider.predict(_request(), client=client)

    asyncio.run(go())
    assert attempts["n"] == 4, f"expected exactly 4 attempts, got {attempts['n']}"


def test_sanitize_error_redacts_credentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-123")
    text = sanitize_error(RuntimeError("boom sk-secret-123 Bearer sk-secret-123"))
    assert "sk-secret-123" not in text
    assert "[redacted]" in text


def test_deepseek_payload_shape(tmp_path):
    budget_path = tmp_path / "b.json"
    provider = DeepSeekProvider(api_key="fake-key", budget=Budget(path=str(budget_path)))
    payload = provider.build_payload(_request())
    assert payload["model"] == "deepseek-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}


def test_parse_extracts_json_from_prose():
    inner = '{"candidates": [{"action": "noop", "replacement": ""}]}'
    body = json.dumps({"choices": [{"message": {"content": f"Here you go:\n{inner}\n"}}]})
    cands = OpenRouterProvider.parse_payload(body, _request())
    assert cands == (Candidate(action="noop", replacement=""),)
    bad = json.dumps({"choices": [{"message": {"content": "no json here"}}]})
    with pytest.raises(TeacherError):
        OpenRouterProvider.parse_payload(bad, _request())


def test_paid_gate_blocks_without_flag(tmp_path, monkeypatch):
    budget_path = tmp_path / "b.json"
    monkeypatch.delenv("ALLOW_PAID_SYNTHETIC", raising=False)
    import asyncio

    async def go():
        provider = OpenRouterProvider(api_key="fake-key", budget=Budget(path=str(budget_path)))
        await provider.predict(_request())

    with pytest.raises(Exception, match="not enabled"):
        asyncio.run(go())


def test_staged_fake_generation(tmp_path):
    out = str(tmp_path / "teacher.jsonl")
    summary = run_generation("fake", max_states=12, seed=1, out_path=out)
    assert summary["states"] == 12
    assert summary["spent_usd"] == 0.0
    assert summary["accepted"] > 0
    lines = open(out, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 12
    stages = [json.loads(line)["stage"] for line in lines]
    assert stages[:10] == ["A"] * 10 and stages[10:] == ["B"] * 2
    for line in lines:
        record = json.loads(line)
        assert "raw_sha256" in record and "request_id" in record
        assert hashlib.sha256(record["raw_sha256"].encode()).hexdigest()  # well-formed hex


def test_build_request_from_fixture():
    source = "x = 1\n"
    request = build_request(source, seed=3, num_candidates=1, state_id="s3")
    assert request.region.text and request.region.text in source
    assert request.state_id == "s3"
