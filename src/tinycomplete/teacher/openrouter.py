"""OpenRouter teacher adapter (preferred cheap teacher: Muse Spark Contributor).

Paid calls require ALLOW_PAID_SYNTHETIC=1, budget headroom, and a verified
model id (see verify_model_id). Uses structured JSON output. Secrets come
from OPENROUTER_API_KEY at call time and never enter stored records.
"""

from __future__ import annotations

import json
import os
import time

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .base import Candidate, TeacherRequest, TeacherResponse, ValidationSummary, raw_sha256_of
from .budget import Budget
from .prompt import RESPONSE_SCHEMA, build_messages
from .validate import validate_response

__all__ = [
    "OpenRouterProvider",
    "TeacherError",
    "DEFAULT_MODEL",
    "MODELS_URL",
    "CHAT_URL",
    "sanitize_error",
]

DEFAULT_MODEL = "meta/muse-spark-1.3-contributor"
MODELS_URL = "https://openrouter.ai/api/v1/models"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class TeacherError(RuntimeError):
    pass


def sanitize_error(exc: BaseException) -> str:
    """One-line error description guaranteed free of credential material."""
    text = f"{type(exc).__name__}: {exc}"
    secrets = (
        os.environ.get("OPENROUTER_API_KEY", ""),
        os.environ.get("DEEPSEEK_API_KEY", ""),
    )
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "[redacted]")
    if "Bearer " in text:
        head, _, _ = text.partition("Bearer ")
        text = head + "Bearer [redacted]"
    return text[:500]


def _extract_json_object(text: str) -> str:
    """Return the largest {...} span; raises ValueError when absent."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response content")
    return text[start : end + 1]


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, TeacherError) and "retryable:" in str(exc):
        return True
    return False


class OpenRouterProvider:
    provider_name = "openrouter"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        budget: Budget | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self._api_key = api_key  # test-only; production reads env at call time
        self.budget = budget or Budget.from_env()
        self.timeout_s = timeout_s

    def _key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise TeacherError("OPENROUTER_API_KEY is absent")
        return key

    def build_payload(self, request: TeacherRequest) -> dict:
        return {
            "model": self.model,
            "messages": build_messages(request),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "next_edit_candidates",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        }

    @staticmethod
    def parse_payload(raw: str, request: TeacherRequest) -> tuple[Candidate, ...]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TeacherError(f"malformed JSON response: {exc}") from None
        try:
            items = data["choices"][0]["message"]["content"]
            if isinstance(items, str):
                items = _extract_json_object(items)
            cands = json.loads(items) if isinstance(items, str) else items
            return tuple(Candidate.model_validate(c) for c in cands["candidates"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise TeacherError(f"schema parse failure: {exc}") from None

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception(_retryable),
        reraise=True,
    )
    async def _post(
        self, client: httpx.AsyncClient, payload: dict, headers: dict
    ) -> httpx.Response:
        try:
            resp = await client.post(CHAT_URL, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError):
            raise
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            raise TeacherError(f"retryable: HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise TeacherError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp

    async def predict(
        self, request: TeacherRequest, client: httpx.AsyncClient | None = None
    ) -> TeacherResponse:
        if not Budget.paid_enabled():
            raise TeacherError("paid generation not enabled (ALLOW_PAID_SYNTHETIC=1 required)")
        if request.num_candidates > self.budget.candidate_cap:
            raise TeacherError("candidate count exceeds MAX_CANDIDATES_PER_STATE")
        key = self._key()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = self.build_payload(request)
        owned = client is None
        client = client or httpx.AsyncClient(timeout=self.timeout_s)
        try:
            resp = await self._post(client, payload, headers)
        except BaseException as exc:
            raise TeacherError(sanitize_error(exc)) from None
        finally:
            if owned:
                await client.aclose()
        # Never store headers; only the body hash and parsed fields.
        body = resp.text
        try:
            data = resp.json()
        except ValueError:
            raise TeacherError("malformed JSON response") from None
        usage = data.get("usage", {})
        candidates = self.parse_payload(body, request)
        _, rejected = validate_response(candidates, request)
        reasons: list[str] = [reason for record in rejected for reason in record["reasons"]]
        return TeacherResponse(
            provider=self.provider_name,
            model=self.model,
            timestamp=time.time(),
            state_id=request.state_id,
            candidates=candidates,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reported_cost_usd=None,
            request_id=data.get("id", ""),
            raw_sha256=raw_sha256_of(body),
            validation=ValidationSummary(ok=not reasons, reasons=tuple(reasons)),
        )

    @staticmethod
    async def verify_model_id(model: str, client: httpx.AsyncClient | None = None) -> dict:
        """Check the exact model id exists on OpenRouter before paid calls."""
        owned = client is None
        client = client or httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.get(MODELS_URL)
            resp.raise_for_status()
            models = resp.json().get("data", [])
        finally:
            if owned:
                await client.aclose()
        for entry in models:
            if entry.get("id") == model:
                return entry
        raise TeacherError(f"model id not listed on OpenRouter: {model}")
