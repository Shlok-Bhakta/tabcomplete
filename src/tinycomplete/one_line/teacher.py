"""Bounded teacher boundary for the user-authorized Muse campaign.

The user reports direct provider approval for this specific student project.
That report is recorded as an authorization basis, not independently verified.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import httpx

from tinycomplete.one_line.contract import EditAction, decode_action

MODEL_ID = "opencode-go/muse-spark-1.3-contributor"
TERMS_URL = "https://opencode.ai/legal/terms-of-service"
AUTHORIZATION_BASIS = "user-reported direct provider approval on 2026-09-26"
MAX_CALLS = 25_000
MAX_INPUT_TOKENS = 60_000_000
MAX_OUTPUT_TOKENS = 20_000_000
FINAL_ACTION_OPEN = "<FINAL_ACTION>"
FINAL_ACTION_CLOSE = "</FINAL_ACTION>"
AUTHOR_CANDIDATE_OPEN = "<AUTHOR_CANDIDATE>"
AUTHOR_CANDIDATE_CLOSE = "</AUTHOR_CANDIDATE>"
MAX_AUTHOR_MESSAGE_BYTES = 16_384
MAX_AUTHOR_JSON_BYTES = 8_192
MAX_FAILURE_CAPTURE_BYTES = 64 * 1024
FAILURE_CAPTURE_ROOT = Path(
    "/mnt/ssd/tabcomplete-public-mechanism-pilot-r1-v2/transport-diagnostic-r1/failure-capture"
)
_FAILURE_STAGES = frozenset({"session.create", "session.prompt"})
_SAFE_CONTENT_TYPE = re.compile(
    r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+"
    r"(?:;\s*charset=[A-Za-z0-9._-]+)?"
)


class TeacherPolicyError(ValueError):
    """A provider request must not be sent."""


class TeacherBudgetError(ValueError):
    """A provider usage reservation or report is not trustworthy."""


class TeacherTransportError(RuntimeError):
    """A provider transport or response failed without exposing payloads."""

    def __init__(
        self, message: str, *, stage: str | None = None, request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.request_id = request_id


class TeacherResponseValidationError(TeacherTransportError):
    """A successful HTTP response could not establish usable output or usage.

    This is deliberately separate from transport failure. The request may have
    reached the provider, but this client cannot prove whether generation
    completed or what it consumed. Callers must keep the full reservation and
    must not retry it.
    """

    def __init__(
        self,
        *,
        stage: Literal["session.create", "session.prompt"],
        request_id: str,
        failure_kind: Literal[
            "invalid_json",
            "invalid_envelope",
            "provider_error",
            "invalid_model",
            "invalid_usage",
            "invalid_content",
        ],
    ) -> None:
        super().__init__(
            "OpenCode returned an unusable successful HTTP response",
            stage=stage,
            request_id=request_id,
        )
        self.failure_kind = failure_kind
        self.provider_completion: Literal["unknown"] = "unknown"
        self.usage_status: Literal["unknown"] = "unknown"
        self.retry_permitted = False


class TeacherCompletedResponseError(RuntimeError):
    """A provider completed, but the result cannot proceed as an ordinary role output."""

    def __init__(
        self,
        *,
        request_id: str,
        response: TeacherResponse,
        failure_status: Literal[
            "completed_budget_overrun",
            "completed_accounting_failure",
            "completed_response_persistence_failure",
        ],
        evidence_sha256: str | None,
        ledger_reconciled: bool,
    ) -> None:
        super().__init__("OpenCode completed a response that requires terminal handling")
        self.request_id = request_id
        self.response = response
        self.failure_status = failure_status
        self.evidence_sha256 = evidence_sha256
        self.ledger_reconciled = ledger_reconciled


class TeacherCandidateError(ValueError):
    """The complete teacher output is not a canonical candidate action."""


_SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16})\b|"
    r"^\s*(?:api[_-]?key|access[_-]?token|password)\s*[:=]",
    re.IGNORECASE | re.MULTILINE,
)


def parse_candidate_action(content: str) -> dict[str, str]:
    """Parse one complete JSON action; never mine JSON from surrounding prose."""
    try:
        action = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise TeacherCandidateError("candidate is not a complete JSON object") from exc
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise TeacherCandidateError("candidate action missing")
    name = action["action"]
    if name in {"keep", "delete_line"}:
        if set(action) != {"action"}:
            raise TeacherCandidateError("candidate has extra fields")
    elif name in {"replace_line", "insert_before"}:
        if (
            set(action) != {"action", "text"}
            or not isinstance(action["text"], str)
            or "\r" in action["text"]
            or "\n" in action["text"]
        ):
            raise TeacherCandidateError("candidate line text invalid")
    else:
        raise TeacherCandidateError("candidate action unknown")
    return action


@dataclass(frozen=True)
class ExtractedFinalAction:
    wire: str
    action: EditAction
    wire_plus_q25_eos_tokens: int


@dataclass(frozen=True)
class ExtractedAuthorCandidate:
    json_text: str
    value: dict[str, object]


def extract_author_candidate_block(
    content: str, *, provider_complete: bool
) -> ExtractedAuthorCandidate:
    """Read one complete terminal tagged author JSON object with no repair."""
    if not provider_complete:
        raise TeacherCandidateError("provider message incomplete")
    try:
        content_bytes = content.encode("utf-8")
    except UnicodeError as exc:
        raise TeacherCandidateError("author message is not UTF-8") from exc
    if len(content_bytes) > MAX_AUTHOR_MESSAGE_BYTES:
        raise TeacherCandidateError("author message exceeds byte cap")
    if "\r" in content:
        raise TeacherCandidateError("CR is outside author block protocol")
    if content.count(AUTHOR_CANDIDATE_OPEN) != 1 or content.count(AUTHOR_CANDIDATE_CLOSE) != 1:
        raise TeacherCandidateError("author block count invalid")
    start = content.index(AUTHOR_CANDIDATE_OPEN)
    opening = AUTHOR_CANDIDATE_OPEN + "\n"
    closing = "\n" + AUTHOR_CANDIDATE_CLOSE
    if not content.startswith(opening, start) or not content.endswith(closing):
        raise TeacherCandidateError("author block incomplete or has trailing text")
    json_text = content[start + len(opening) : -len(closing)]
    if not json_text or len(json_text.encode("utf-8")) > MAX_AUTHOR_JSON_BYTES:
        raise TeacherCandidateError("author JSON exceeds byte cap or is empty")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TeacherCandidateError("author JSON contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise TeacherCandidateError("author JSON contains nonfinite value")

    try:
        value = json.loads(
            json_text, object_pairs_hook=unique_pairs, parse_constant=reject_constant
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise TeacherCandidateError("author block is not strict JSON") from exc
    if not isinstance(value, dict):
        raise TeacherCandidateError("author JSON must be one object")
    return ExtractedAuthorCandidate(json_text=json_text, value=value)


def extract_final_action_block(
    content: str,
    *,
    provider_complete: bool,
    tokenizer: object,
) -> ExtractedFinalAction:
    """Read one v6 terminal block without searching prose for an action.

    Exact shape: optional prose ending in LF, opening tag, LF, one raw wire
    line, LF, closing tag, and end of message. Provider completion is separate
    from the hypothetical q25 EOS counted against the action scope.
    """
    if not provider_complete:
        raise TeacherCandidateError("provider message incomplete")
    if "\r" in content:
        raise TeacherCandidateError("CR is outside the final action block protocol")
    if content.count(FINAL_ACTION_OPEN) != 1 or content.count(FINAL_ACTION_CLOSE) != 1:
        raise TeacherCandidateError("final action block count invalid")
    start = content.index(FINAL_ACTION_OPEN)
    if start and content[start - 1] != "\n":
        raise TeacherCandidateError("opening tag must begin on a new line")
    opening = FINAL_ACTION_OPEN + "\n"
    closing = "\n" + FINAL_ACTION_CLOSE
    if not content.startswith(opening, start) or not content.endswith(closing):
        raise TeacherCandidateError("final action block incomplete or has trailing text")
    wire = content[start + len(opening) : -len(closing)]
    if not wire or "\n" in wire or "\r" in wire:
        raise TeacherCandidateError("final action must be one LF-free wire line")
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TeacherCandidateError("q25 tokenizer unavailable")
    token_ids = encode(wire, add_special_tokens=False)
    if not isinstance(token_ids, list):
        raise TeacherCandidateError("q25 token count unavailable")
    count = len(token_ids) + 1
    parsed = decode_action(wire, terminated=True, generated_tokens=count, max_tokens=64)
    if parsed.status != "ok" or parsed.action is None:
        raise TeacherCandidateError(f"final action wire {parsed.status}")
    return ExtractedFinalAction(wire, parsed.action, count)


def extract_final_action_block_v7(
    content: str,
    *,
    provider_complete: bool,
    tokenizer: object,
) -> ExtractedFinalAction:
    """Read the frozen v7 terminal block; prior prose need not end in LF."""
    if not provider_complete:
        raise TeacherCandidateError("provider message incomplete")
    if "\r" in content:
        raise TeacherCandidateError("CR is outside the final action block protocol")
    if content.count(FINAL_ACTION_OPEN) != 1 or content.count(FINAL_ACTION_CLOSE) != 1:
        raise TeacherCandidateError("final action block count invalid")
    start = content.index(FINAL_ACTION_OPEN)
    opening = FINAL_ACTION_OPEN + "\n"
    closing = "\n" + FINAL_ACTION_CLOSE
    if not content.startswith(opening, start) or not content.endswith(closing):
        raise TeacherCandidateError("final action block incomplete or has trailing text")
    wire = content[start + len(opening) : -len(closing)]
    if not wire or "\n" in wire:
        raise TeacherCandidateError("final action must be one LF-free wire line")
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TeacherCandidateError("q25 tokenizer unavailable")
    token_ids = encode(wire, add_special_tokens=False)
    if not isinstance(token_ids, list):
        raise TeacherCandidateError("q25 token count unavailable")
    count = len(token_ids) + 1
    parsed = decode_action(wire, terminated=True, generated_tokens=count, max_tokens=64)
    if parsed.status != "ok" or parsed.action is None:
        raise TeacherCandidateError(f"final action wire {parsed.status}")
    return ExtractedFinalAction(wire, parsed.action, count)


def assert_opencode_request_allowed(
    *,
    model_id: str,
    purpose: Literal["student_label", "automated_score", "calibration"],
    source_class: Literal["public", "synthetic", "private", "sealed"],
    prompt: str,
    authorization_basis: str,
) -> None:
    """Accept only the explicit reported exception and safe source classes.

    The classifier is a second guard, never a replacement for source review.
    """
    if source_class not in {"public", "synthetic"}:
        raise TeacherPolicyError("private or sealed source cannot leave the machine")
    if _SECRET.search(prompt):
        raise TeacherPolicyError("possible credential in provider input")
    if model_id != MODEL_ID:
        raise TeacherPolicyError("unapproved teacher model")
    if purpose not in {"student_label", "automated_score", "calibration"}:
        raise TeacherPolicyError("unapproved teacher purpose")
    if authorization_basis != AUTHORIZATION_BASIS:
        raise TeacherPolicyError("campaign-specific provider approval basis missing")


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
                    if (
                        input_tokens > requests[request_id]["reserved_input"]
                        or output_tokens > requests[request_id]["reserved_output"]
                    ):
                        raise ValueError("settlement exceeds reservation")
                    requests[request_id]["input"] = input_tokens
                    requests[request_id]["output"] = output_tokens
                elif (
                    kind == "settle_overrun"
                    and request_id in requests
                    and "input" not in requests[request_id]
                    and (
                        input_tokens > requests[request_id]["reserved_input"]
                        or output_tokens > requests[request_id]["reserved_output"]
                    )
                    and event.get("reason") == "persisted_provider_usage_exceeded_reservation"
                    and isinstance(event.get("evidence_sha256"), str)
                    and re.fullmatch(r"[a-f0-9]{64}", event["evidence_sha256"]) is not None
                ):
                    requests[request_id]["input"] = input_tokens
                    requests[request_id]["output"] = output_tokens
                    requests[request_id]["overrun"] = 1
                elif (
                    kind == "correct"
                    and request_id in requests
                    and "output" in requests[request_id]
                    and "corrected" not in requests[request_id]
                    and input_tokens == requests[request_id]["input"]
                    and event.get("previous_output_tokens") == requests[request_id]["output"]
                    and event.get("reason") == "include_reported_reasoning_tokens"
                ):
                    requests[request_id]["output"] = output_tokens
                    requests[request_id]["corrected"] = 1
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
        if (
            totals.calls > MAX_CALLS
            or totals.input_tokens > MAX_INPUT_TOKENS
            or totals.output_tokens > MAX_OUTPUT_TOKENS
        ):
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
            handle.write(
                json.dumps(
                    {
                        "kind": "reserve",
                        "request_id": request_id,
                        "input_tokens": input_tokens,
                        "output_tokens": max_output_tokens,
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def settle(
        self, request_id: str, *, input_tokens: int | None, output_tokens: int | None
    ) -> None:
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
            if (
                input_tokens > reservation["reserved_input"]
                or output_tokens > reservation["reserved_output"]
            ):
                raise TeacherBudgetError("actual usage exceeded reserved limit")
            requests[request_id] = {**reservation, "input": input_tokens, "output": output_tokens}
            self._within_budget(self._totals(requests))
            handle.seek(0, 2)
            handle.write(
                json.dumps(
                    {
                        "kind": "settle",
                        "request_id": request_id,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def settle_overrun(
        self,
        request_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        evidence_sha256: str,
    ) -> None:
        """Account for a completed, persisted provider response that exceeded its reservation.

        This is an audit correction for a single failed request, never permission
        to retry it or to turn its response into a training candidate.
        """
        self._positive(input_tokens, "input tokens")
        self._positive(output_tokens, "output tokens")
        if re.fullmatch(r"[a-f0-9]{64}", evidence_sha256) is None:
            raise TeacherBudgetError("invalid overrun evidence hash")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            requests = self._read(handle)
            row = requests.get(request_id)
            if row is None or "input" in row:
                raise TeacherBudgetError("missing or already settled reservation")
            if input_tokens <= row["reserved_input"] and output_tokens <= row["reserved_output"]:
                raise TeacherBudgetError("usage did not exceed reservation")
            row["input"] = input_tokens
            row["output"] = output_tokens
            row["overrun"] = 1
            self._within_budget(self._totals(requests))
            handle.seek(0, 2)
            handle.write(
                json.dumps(
                    {
                        "kind": "settle_overrun",
                        "request_id": request_id,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "evidence_sha256": evidence_sha256,
                        "reason": "persisted_provider_usage_exceeded_reservation",
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def correct_output_accounting(
        self,
        request_id: str,
        *,
        previous_output_tokens: int,
        generated_tokens_including_reasoning: int,
    ) -> None:
        """Append one auditable correction to a settled calibration record."""
        self._positive(previous_output_tokens, "previous output tokens")
        self._positive(generated_tokens_including_reasoning, "generated tokens")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            requests = self._read(handle)
            row = requests.get(request_id)
            if (
                row is None
                or "input" not in row
                or "corrected" in row
                or row["output"] != previous_output_tokens
                or generated_tokens_including_reasoning < previous_output_tokens
            ):
                raise TeacherBudgetError("invalid usage correction")
            row["output"] = generated_tokens_including_reasoning
            row["corrected"] = 1
            self._within_budget(self._totals(requests))
            handle.seek(0, 2)
            handle.write(
                json.dumps(
                    {
                        "kind": "correct",
                        "request_id": request_id,
                        "input_tokens": row["input"],
                        "previous_output_tokens": previous_output_tokens,
                        "output_tokens": generated_tokens_including_reasoning,
                        "reason": "include_reported_reasoning_tokens",
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())


@dataclass(frozen=True)
class TeacherResponse:
    content: str
    session_id: str
    response_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens_reported: int | None
    cached_read_tokens: int
    cached_write_tokens: int
    finish_reason: str | None
    cost_usd_reported: float | None


def _isolated_env(
    *, structured_output_only: bool = False, config_home: Path | None = None
) -> dict[str, str]:
    """Isolate OpenCode config while retaining only its existing credential store."""
    if config_home is None:
        raise TeacherPolicyError("OpenCode requires isolated config home")
    keys = ("PATH", "HOME", "USER", "SHELL", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    config_home.mkdir(parents=True, exist_ok=True)
    env["XDG_CONFIG_HOME"] = str(config_home)
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    config: dict[str, object] = {
        "model": MODEL_ID,
        "small_model": MODEL_ID,
        "permission": "deny",
        "tools": {"*": False},
        "plugin": [],
        "mcp": {},
    }
    if structured_output_only:
        config = {
            "model": MODEL_ID,
            "small_model": MODEL_ID,
            "permission": {"*": "deny", "StructuredOutput": "allow"},
            "plugin": [],
            "mcp": {},
        }
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    return env


class OpenCodeTeacherClient:
    """Owned, loopback-only OpenCode server with one role request at a time.

    The server uses the installed OpenCode credential store. Its agent tools and
    plugins are disabled, and both primary and auxiliary models are pinned.
    A failed response leaves the full ledger reservation in place.
    """

    def __init__(
        self,
        ledger: TeacherUsageLedger,
        *,
        startup_seconds: float = 20.0,
        structured_output_only: bool = False,
        failure_capture_dir: Path | None = None,
    ):
        self.ledger = ledger
        self.startup_seconds = startup_seconds
        self.structured_output_only = structured_output_only
        self.failure_capture_dir = None
        if failure_capture_dir is not None:
            capture_root = FAILURE_CAPTURE_ROOT.resolve(strict=False)
            capture_dir = Path(failure_capture_dir).resolve(strict=False)
            if not capture_dir.is_relative_to(capture_root):
                raise ValueError("failure capture path must stay under private capture root")
            self.failure_capture_dir = capture_dir
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None

    def __enter__(self) -> OpenCodeTeacherClient:
        self._temporary = tempfile.TemporaryDirectory(prefix="tabcomplete-teacher-")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        log_path = Path(self._temporary.name) / "server.log"
        log_file = log_path.open("wb")
        try:
            self._process = subprocess.Popen(
                ["opencode", "serve", "--pure", "--hostname", "127.0.0.1", "--port", str(port)],
                cwd=self._temporary.name,
                env=_isolated_env(
                    structured_output_only=self.structured_output_only,
                    config_home=Path(self._temporary.name) / "config",
                ),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()
        self._client = httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            timeout=2.0,
            trust_env=False,
        )
        deadline = time.monotonic() + self.startup_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self.__exit__(None, None, None)
                raise TeacherTransportError("OpenCode server exited before readiness")
            try:
                health = self._client.get("/global/health")
                if health.status_code == 200 and health.json().get("healthy") is True:
                    return self
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.1)
        self.__exit__(None, None, None)
        raise TeacherTransportError("OpenCode server readiness timed out")

    def __exit__(self, *_: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _capture_http_failure(
        self, response: httpx.Response, *, stage: str, request_id: str
    ) -> None:
        """Privately preserve a bounded non-200 body for an explicitly enabled diagnosis."""
        directory = self.failure_capture_dir
        if directory is None or stage not in _FAILURE_STAGES:
            return
        created = False
        descriptor: int | None = None
        target: Path | None = None
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory_info = directory.lstat()
            if (
                not stat.S_ISDIR(directory_info.st_mode)
                or directory_info.st_uid != os.getuid()
                or directory_info.st_mode & 0o077
            ):
                return
            body = response.content
            request_hash = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            target = directory / f"{request_hash}.{stage.replace('.', '-')}.json"
            content_type = response.headers.get("content-type", "")[:128]
            if not _SAFE_CONTENT_TYPE.fullmatch(content_type):
                content_type = ""
            capture = {
                "capture_version": 1,
                "route_stage": stage,
                "http_status": response.status_code,
                "request_id_sha256": request_hash,
                "content_type": content_type or None,
                "response_body_bytes": len(body),
                "response_body_sha256": hashlib.sha256(body).hexdigest(),
                "captured_body_bytes": min(len(body), MAX_FAILURE_CAPTURE_BYTES),
                "truncated": len(body) > MAX_FAILURE_CAPTURE_BYTES,
                "response_body_prefix_base64": base64.b64encode(
                    body[:MAX_FAILURE_CAPTURE_BYTES]
                ).decode("ascii"),
            }
            encoded = (json.dumps(capture, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            )
            directory_fd = os.open(
                directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                descriptor = os.open(
                    target.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                created = True
                if os.fstat(descriptor).st_mode & 0o077:
                    raise OSError("private capture file permissions unavailable")
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = None
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, ValueError, TypeError):
            if descriptor is not None:
                os.close(descriptor)
            if created and target is not None:
                try:
                    target.unlink()
                except OSError:
                    pass

    def _json_response(
        self,
        response: httpx.Response,
        *,
        stage: Literal["session.create", "session.prompt"],
        request_id: str,
    ) -> dict[str, object]:
        if response.status_code != 200:
            if stage is not None and request_id is not None:
                self._capture_http_failure(response, stage=stage, request_id=request_id)
            raise TeacherTransportError(
                f"OpenCode HTTP status {response.status_code}",
                stage=stage,
                request_id=request_id,
            )
        try:
            body = response.json()
        except ValueError:
            raise TeacherResponseValidationError(
                stage=stage,
                request_id=request_id,
                failure_kind="invalid_json",
            ) from None
        if not isinstance(body, dict):
            raise TeacherResponseValidationError(
                stage=stage,
                request_id=request_id,
                failure_kind="invalid_envelope",
            )
        return body

    def run_role(
        self,
        *,
        request_id: str,
        prompt: str,
        purpose: Literal["student_label", "automated_score", "calibration"],
        source_class: Literal["public", "synthetic", "private", "sealed"],
        authorization_basis: str,
        reserve_input_tokens: int = 100_000,
        reserve_output_tokens: int = 4_096,
        output_schema: dict[str, object] | None = None,
        system_instruction: str | None = None,
        persist_completed_response: Callable[[TeacherResponse], str] | None = None,
    ) -> TeacherResponse:
        """Run one isolated role; never retry after ambiguous transport failure."""
        assert_opencode_request_allowed(
            model_id=MODEL_ID,
            purpose=purpose,
            source_class=source_class,
            prompt=prompt,
            authorization_basis=authorization_basis,
        )
        if system_instruction is not None and _SECRET.search(system_instruction):
            raise TeacherPolicyError("possible credential in system instruction")
        if self.structured_output_only and output_schema is None:
            raise TeacherPolicyError("structured output mode requires a JSON schema")
        if self._client is None:
            raise TeacherTransportError(
                "OpenCode server is not running", stage="session.create", request_id=request_id
            )
        self.ledger.reserve(
            request_id, input_tokens=reserve_input_tokens, max_output_tokens=reserve_output_tokens
        )
        stage: Literal["session.create", "session.prompt"] = "session.create"
        try:
            session = self._json_response(
                self._client.post(
                    "/session",
                    json={"title": f"one-line-{purpose}-{request_id}"},
                    timeout=30.0,
                ),
                stage="session.create",
                request_id=request_id,
            )
            try:
                session_id = session["id"]
            except KeyError:
                raise TeacherResponseValidationError(
                    stage="session.create",
                    request_id=request_id,
                    failure_kind="invalid_envelope",
                ) from None
            if not isinstance(session_id, str):
                raise TeacherResponseValidationError(
                    stage="session.create",
                    request_id=request_id,
                    failure_kind="invalid_envelope",
                )
            message_body: dict[str, object] = {
                "model": {"providerID": "opencode-go", "modelID": "muse-spark-1.3-contributor"},
                "parts": [{"type": "text", "text": prompt}],
            }
            if not self.structured_output_only:
                message_body["tools"] = {}
            if system_instruction is not None:
                message_body["system"] = system_instruction
            if output_schema is not None:
                message_body["format"] = {"type": "json_schema", "schema": output_schema}
            stage = "session.prompt"
            response = self._json_response(
                self._client.post(
                    f"/session/{session_id}/message",
                    json=message_body,
                    timeout=120.0,
                ),
                stage="session.prompt",
                request_id=request_id,
            )
            try:
                info = response["info"]
                parts = response["parts"]
            except KeyError:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_envelope",
                ) from None
            if not isinstance(info, dict) or not isinstance(parts, list):
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_envelope",
                )
            try:
                model_id = f"{info['providerID']}/{info['modelID']}"
            except KeyError:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_envelope",
                ) from None
            if model_id != MODEL_ID:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_model",
                )
            if info.get("error") is not None:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="provider_error",
                )
            try:
                tokens = info["tokens"]
            except KeyError:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_usage",
                ) from None
            if not isinstance(tokens, dict):
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_usage",
                )
            try:
                input_tokens = tokens["input"]
                output_tokens = tokens["output"]
                reasoning_tokens = tokens["reasoning"]
            except KeyError:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_usage",
                ) from None
            total_tokens = tokens.get("total")
            cache = tokens.get("cache", {})
            if not isinstance(cache, dict):
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_usage",
                )
            cached_read = cache.get("read", 0)
            cached_write = cache.get("write", 0)
            try:
                self.ledger._positive(input_tokens, "input tokens")
                self.ledger._positive(output_tokens, "output tokens")
                self.ledger._positive(reasoning_tokens, "reasoning tokens")
                if total_tokens is not None:
                    self.ledger._positive(total_tokens, "total tokens")
                self.ledger._positive(cached_read, "cached read tokens")
                self.ledger._positive(cached_write, "cached write tokens")
            except TeacherBudgetError:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_usage",
                ) from None
            text_parts: list[str] = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    value = part.get("text")
                    if not isinstance(value, str):
                        raise TeacherResponseValidationError(
                            stage="session.prompt",
                            request_id=request_id,
                            failure_kind="invalid_content",
                        )
                    text_parts.append(value)
            structured = info.get("structured")
            if output_schema is not None:
                if not isinstance(structured, dict):
                    raise TeacherResponseValidationError(
                        stage="session.prompt",
                        request_id=request_id,
                        failure_kind="invalid_content",
                    )
                content = json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
            elif text_parts:
                content = "".join(text_parts)
            else:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_content",
                )
            try:
                response_id = info["id"]
            except KeyError:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_envelope",
                ) from None
            if not isinstance(response_id, str) or not response_id:
                raise TeacherResponseValidationError(
                    stage="session.prompt",
                    request_id=request_id,
                    failure_kind="invalid_envelope",
                )
            result = TeacherResponse(
                content=content,
                session_id=session_id,
                response_id=response_id,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens_reported=total_tokens,
                cached_read_tokens=cached_read,
                cached_write_tokens=cached_write,
                finish_reason=info.get("finish") if isinstance(info.get("finish"), str) else None,
                cost_usd_reported=(
                    float(info["cost"]) if isinstance(info.get("cost"), (int, float)) else None
                ),
            )
            metered_output_tokens = output_tokens + reasoning_tokens
            reservation_exceeded = (
                input_tokens > reserve_input_tokens
                or metered_output_tokens > reserve_output_tokens
            )
            evidence_sha256: str | None = None
            if persist_completed_response is not None:
                try:
                    evidence_sha256 = persist_completed_response(result)
                except Exception as exc:
                    raise TeacherCompletedResponseError(
                        request_id=request_id,
                        response=result,
                        failure_status="completed_response_persistence_failure",
                        evidence_sha256=None,
                        ledger_reconciled=False,
                    ) from exc
                if not isinstance(evidence_sha256, str) or re.fullmatch(
                    r"[a-f0-9]{64}", evidence_sha256
                ) is None:
                    raise TeacherCompletedResponseError(
                        request_id=request_id,
                        response=result,
                        failure_status="completed_response_persistence_failure",
                        evidence_sha256=None,
                        ledger_reconciled=False,
                    )
            if persist_completed_response is not None and evidence_sha256 is None:
                # Do not attempt settlement until a durable response+usage receipt exists.
                raise TeacherCompletedResponseError(
                    request_id=request_id,
                    response=result,
                    failure_status="completed_response_persistence_failure",
                    evidence_sha256=None,
                    ledger_reconciled=False,
                )
            if reservation_exceeded and evidence_sha256 is None:
                # Keep the full reservation when an overrun has no durable receipt.
                raise TeacherCompletedResponseError(
                    request_id=request_id,
                    response=result,
                    failure_status="completed_budget_overrun",
                    evidence_sha256=None,
                    ledger_reconciled=False,
                )
            try:
                self.ledger.settle(
                    request_id,
                    input_tokens=input_tokens,
                    output_tokens=metered_output_tokens,
                )
            except TeacherBudgetError as exc:
                if reservation_exceeded and evidence_sha256 is not None:
                    try:
                        self.ledger.settle_overrun(
                            request_id,
                            input_tokens=input_tokens,
                            output_tokens=metered_output_tokens,
                            evidence_sha256=evidence_sha256,
                        )
                        ledger_reconciled = True
                    except TeacherBudgetError:
                        ledger_reconciled = False
                    raise TeacherCompletedResponseError(
                        request_id=request_id,
                        response=result,
                        failure_status="completed_budget_overrun",
                        evidence_sha256=evidence_sha256,
                        ledger_reconciled=ledger_reconciled,
                    ) from exc
                if "cap exceeded" in str(exc):
                    raise TeacherCompletedResponseError(
                        request_id=request_id,
                        response=result,
                        failure_status="completed_accounting_failure",
                        evidence_sha256=evidence_sha256,
                        ledger_reconciled=False,
                    ) from exc
                raise TeacherCompletedResponseError(
                    request_id=request_id,
                    response=result,
                    failure_status="completed_accounting_failure",
                    evidence_sha256=evidence_sha256,
                    ledger_reconciled=False,
                ) from exc
            return result
        except (TeacherCompletedResponseError, TeacherResponseValidationError):
            raise
        except httpx.HTTPError:
            raise TeacherTransportError(
                "OpenCode request transport failed", stage=stage, request_id=request_id
            ) from None
        except (KeyError, TypeError, ValueError):
            # Unexpected response-shape failures are quarantined as unknown usage.
            raise TeacherResponseValidationError(
                stage=stage,
                request_id=request_id,
                failure_kind=("invalid_usage" if stage == "session.prompt" else "invalid_envelope"),
            ) from None
