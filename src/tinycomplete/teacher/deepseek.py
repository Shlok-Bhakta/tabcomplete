"""DeepSeek teacher adapter (SECOND teacher, small overlap only).

Uses the OpenAI-compatible chat API with JSON-object output. Same budget
gate and secret hygiene as the OpenRouter adapter. Fits under the same
MAX_TOTAL_SPEND_USD unless the user explicitly raises it.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import Candidate, TeacherRequest, TeacherResponse, ValidationSummary, raw_sha256_of
from .budget import Budget
from .openrouter import TeacherError, _retryable, sanitize_error
from .prompt import build_messages
from .validate import validate_response

__all__ = ["DeepSeekProvider", "DEFAULT_MODEL", "CHAT_URL"]

DEFAULT_MODEL = "deepseek-flash"  # verified via GET /models 2026-09-17 (cheap tier)
CHAT_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekProvider:
    provider_name = "deepseek"

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
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise TeacherError("DEEPSEEK_API_KEY is absent")
        return key

    def build_payload(self, request: TeacherRequest) -> dict:
        # Thinking disabled: reasoning goes to reasoning_content and starves
        # the JSON final answer; labeling needs strict JSON in content.
        # Toggle name verified in DeepSeek Thinking Mode docs (2026-09-17).
        return {
            "model": self.model,
            "messages": build_messages(request),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": 2000,
        }

    async def predict(
        self, request: TeacherRequest, client: httpx.AsyncClient | None = None
    ) -> TeacherResponse:
        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential,
        )

        if not Budget.paid_enabled():
            raise TeacherError("paid generation not enabled (ALLOW_PAID_SYNTHETIC=1 required)")
        if request.num_candidates > self.budget.candidate_cap:
            raise TeacherError("candidate count exceeds MAX_CANDIDATES_PER_STATE")
        key = self._key()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = self.build_payload(request)
        owned = client is None
        client = client or httpx.AsyncClient(timeout=self.timeout_s)

        @retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            retry=retry_if_exception(_retryable),
            reraise=True,
        )
        async def _post() -> httpx.Response:
            try:
                resp = await client.post(CHAT_URL, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError):
                raise
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                raise TeacherError(f"retryable: HTTP {resp.status_code}")
            if resp.status_code != 200:
                raise TeacherError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp

        try:
            resp = await _post()
        except BaseException as exc:
            raise TeacherError(sanitize_error(exc)) from None
        finally:
            if owned:
                await client.aclose()
        body = resp.text
        try:
            data = resp.json()
        except ValueError:
            raise TeacherError("malformed JSON response") from None
        candidates: tuple[Candidate, ...]
        try:
            from .openrouter import OpenRouterProvider

            candidates = OpenRouterProvider.parse_payload(body, request)
        except TeacherError:
            raise
        usage = data.get("usage", {})
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
