"""Fail-closed boundary for the proposed hosted teacher route.

OpenCode Terms of Use effective 2026-08-15 prohibit programmatic extraction of
Output and using Output to develop a competing model. No code in this module
invokes a provider. A later compatible route needs a separate policy change.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

MODEL_ID = "opencode-go/muse-spark-1.3-contributor"
TERMS_URL = "https://opencode.ai/legal/terms-of-service"
MAX_CALLS = 25_000
MAX_INPUT_TOKENS = 60_000_000
MAX_OUTPUT_TOKENS = 20_000_000


class TeacherPolicyError(ValueError):
    """A provider request must not be sent."""


class TeacherBudgetError(ValueError):
    """A provider usage reservation or report is not trustworthy."""


_SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16})\b|"
    r"^\s*(?:api[_-]?key|access[_-]?token|password)\s*[:=]",
    re.IGNORECASE | re.MULTILINE,
)


def assert_opencode_request_allowed(
    *,
    model_id: str,
    purpose: Literal["student_label", "automated_score", "calibration"],
    source_class: Literal["public", "synthetic", "private", "sealed"],
    prompt: str,
) -> None:
    """Reject unsafe input and the currently prohibited output-use route.

    The classifier is a second guard, never a replacement for source review.
    This function intentionally has no success path for hosted OpenCode models.
    """
    if source_class not in {"public", "synthetic"}:
        raise TeacherPolicyError("private or sealed source cannot leave the machine")
    if _SECRET.search(prompt):
        raise TeacherPolicyError("possible credential in provider input")
    if model_id != MODEL_ID:
        raise TeacherPolicyError("unapproved teacher model")
    if purpose not in {"student_label", "automated_score", "calibration"}:
        raise TeacherPolicyError("unapproved teacher purpose")
    raise TeacherPolicyError("OpenCode output-use terms block this campaign route")


@dataclass(frozen=True)
class UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class TeacherUsageLedger:
    """Append-only, locked token ledger for a future rights-compatible route.

    Reservations count their full upper bounds until settled. This ledger is
    accounting only: it does not grant permission to call any provider.
    """

    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _positive(value: int, name: str) -> int:
        if type(value) is not int or value < 0:
            raise TeacherBudgetError(f"invalid {name}")
        return value

    @staticmethod
    def _read(handle: TextIO) -> dict[str, dict[str, int]]:
        handle.seek(0)
        requests: dict[str, dict[str, int]] = {}
        for line in handle:
            try:
                event = json.loads(line)
                request_id = event["request_id"]
                kind = event["kind"]
                input_tokens = event["input_tokens"]
                output_tokens = event["output_tokens"]
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("request ID")
                TeacherUsageLedger._positive(input_tokens, "input tokens")
                TeacherUsageLedger._positive(output_tokens, "output tokens")
                if kind == "reserve" and request_id not in requests:
                    requests[request_id] = {
                        "reserved_input": input_tokens,
                        "reserved_output": output_tokens,
                    }
                elif (
                    kind == "settle"
                    and request_id in requests
                    and "input" not in requests[request_id]
                ):
                    requests[request_id]["input"] = input_tokens
                    requests[request_id]["output"] = output_tokens
                else:
                    raise ValueError("event order")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TeacherBudgetError("invalid teacher usage ledger") from exc
        return requests

    @staticmethod
    def _totals(requests: dict[str, dict[str, int]]) -> UsageTotals:
        return UsageTotals(
            calls=len(requests),
            input_tokens=sum(r.get("input", r["reserved_input"]) for r in requests.values()),
            output_tokens=sum(r.get("output", r["reserved_output"]) for r in requests.values()),
        )

    @staticmethod
    def _within_budget(totals: UsageTotals) -> None:
        if (totals.calls > MAX_CALLS or totals.input_tokens > MAX_INPUT_TOKENS
                or totals.output_tokens > MAX_OUTPUT_TOKENS):
            raise TeacherBudgetError("teacher usage cap exceeded")

    def totals(self) -> UsageTotals:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            return self._totals(self._read(handle))

    def reserve(self, request_id: str, *, input_tokens: int, max_output_tokens: int) -> None:
        """Reserve one completed-call slot and upper-bound tokens before a call."""
        if not isinstance(request_id, str) or not request_id or "\n" in request_id:
            raise TeacherBudgetError("invalid request ID")
        self._positive(input_tokens, "input tokens")
        self._positive(max_output_tokens, "output tokens")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            requests = self._read(handle)
            if request_id in requests:
                raise TeacherBudgetError("duplicate teacher request ID")
            requests[request_id] = {
                "reserved_input": input_tokens,
                "reserved_output": max_output_tokens,
            }
            self._within_budget(self._totals(requests))
            handle.seek(0, 2)
            handle.write(json.dumps({"kind": "reserve", "request_id": request_id,
                                     "input_tokens": input_tokens,
                                     "output_tokens": max_output_tokens}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def settle(self, request_id: str, *, input_tokens: int | None,
               output_tokens: int | None) -> None:
        """Record exact provider usage; unknown usage is never treated as zero."""
        if input_tokens is None or output_tokens is None:
            raise TeacherBudgetError("provider token usage missing")
        self._positive(input_tokens, "input tokens")
        self._positive(output_tokens, "output tokens")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            requests = self._read(handle)
            if request_id not in requests or "input" in requests[request_id]:
                raise TeacherBudgetError("missing or already settled reservation")
            reservation = requests[request_id]
            if (input_tokens > reservation["reserved_input"]
                    or output_tokens > reservation["reserved_output"]):
                raise TeacherBudgetError("actual usage exceeded reserved limit")
            requests[request_id] = {**reservation, "input": input_tokens,
                                    "output": output_tokens}
            self._within_budget(self._totals(requests))
            handle.seek(0, 2)
            handle.write(json.dumps({"kind": "settle", "request_id": request_id,
                                     "input_tokens": input_tokens,
                                     "output_tokens": output_tokens}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
