from __future__ import annotations

import contextlib
import importlib.util
import json
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "measure_sweep_local.py"
_SPEC = importlib.util.spec_from_file_location("measure_sweep_local", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
local = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(local)


class _SSEHandler(BaseHTTPRequestHandler):
    events: list[dict[str, Any]] = []
    split_after_utf8_lead_byte = False
    delay_before_response_seconds = 0.0
    request_json: dict[str, Any] = {}

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        request_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).request_json = json.loads(request_body)
        payload = "".join(
            f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in self.events
        ).encode("utf-8")
        if self.delay_before_response_seconds:
            time.sleep(self.delay_before_response_seconds)
        if self.split_after_utf8_lead_byte:
            marker = "π".encode()
            byte_offset = payload.index(marker) + 1
            chunks = [payload[:byte_offset], payload[byte_offset : byte_offset + 1]]
            chunks.append(payload[byte_offset + 1 :])
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for chunk in chunks:
                    if not chunk:
                        continue
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # The client closes once it observes the terminal SSE event.
                return
        else:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                return


@pytest.fixture
def fake_server():
    _SSEHandler.events = []
    _SSEHandler.split_after_utf8_lead_byte = False
    _SSEHandler.delay_before_response_seconds = 0.0
    _SSEHandler.request_json = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _minimal_plan() -> dict[str, Any]:
    return {
        "runtime": {"server_settings": dict(local.SERVER_SETTINGS)},
        "replay": {
            "request_settings": dict(local.REQUEST_SETTINGS),
            "precision_order": ["q4_k_m", "q8_0"],
            "repetitions": 2,
            "fixture_count": len(local.EXPECTED_CASE_IDS),
            "fixture_order": list(local.EXPECTED_CASE_IDS),
            "expected_requests": len(local.EXPECTED_CASE_IDS) * 2 * 2 * 2,
            "request_order_per_fixture_per_repetition": [
                "changed_state",
                "immediate_same_prompt_repeat",
            ],
        },
        "limits": {
            "max_wall_seconds_including_preflight_and_finalization": local.MAX_WALL_SECONDS,
            "finalization_reserve_seconds": local.FINALIZATION_RESERVE_SECONDS,
            "request_work_deadline_seconds_from_invocation_start": (
                local.MAX_WALL_SECONDS - local.FINALIZATION_RESERVE_SECONDS
            ),
            "minimum_available_ram_bytes": 4 * 1024 * 1024 * 1024,
            "candidate_predictor_memory_bytes": int(1.5 * 1024**3),
        },
    }


def test_local_server_flags_are_cpu_only_and_bounded() -> None:
    argv = local.build_server_argv(Path("llama-server"), Path("model.gguf"), 12345)
    pairs = {argv[index]: argv[index + 1] for index in range(len(argv) - 1)}

    assert pairs["--ctx-size"] == "3072"
    assert pairs["--threads"] == pairs["--threads-batch"] == "4"
    assert pairs["--parallel"] == "1"
    assert pairs["--cache-ram"] == "0"
    assert pairs["--ctx-checkpoints"] == "0"
    assert pairs["--n-gpu-layers"] == "0"
    assert pairs["--cache-type-k"] == pairs["--cache-type-v"] == "f16"
    assert pairs["--spec-type"] == "none"
    assert "--no-cache-idle-slots" in argv
    assert "--no-context-shift" in argv
    assert "--no-repack" in argv


def test_frozen_plan_must_bind_bounded_runtime_and_cache_policy() -> None:
    plan = _minimal_plan()

    local.validate_plan_policy(plan)

    plan["runtime"]["server_settings"]["context_tokens"] = 8192
    with pytest.raises(ValueError, match="server configuration"):
        local.validate_plan_policy(plan)


def test_execute_entrypoint_uses_frozen_deadline_fields_before_model_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tinycomplete.observability.runs as runs

    plan = _minimal_plan()
    plan["schema"] = "sweep-local-replay-plan-v1"
    plan["inputs"] = {"prompt_bundle_sha256": "synthetic-prompt-bundle"}
    plan["outputs"] = {"root": str(tmp_path / "runs")}
    plan["plan_sha256"] = local.plan_digest(plan)
    fixtures: list[dict[str, Any]] = []

    monkeypatch.setattr(
        local,
        "load_and_verify_inputs",
        lambda *_args: (plan, {"plan_sha256": "synthetic-upstream"}, fixtures),
    )
    monkeypatch.setattr(local, "verify_static_identity", lambda *_args: {})
    monkeypatch.setattr(
        local,
        "_proc_kib_fields",
        lambda *_args: {"MemAvailable": 8 * 1024 * 1024 * 1024},
    )
    monkeypatch.setattr(local, "_find_q25_background_processes", lambda *_args: [])
    monkeypatch.setattr(local, "_host_snapshot", lambda *_args: {"background_model_processes": []})

    @contextlib.contextmanager
    def fake_run_scope(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(runs, "run_scope", fake_run_scope)

    class ModelStartReached(Exception):
        pass

    def stop_before_model(**_kwargs):
        raise ModelStartReached

    monkeypatch.setattr(local, "_server_run", stop_before_model)
    args = Namespace(
        plan=Path("plan.json"),
        upstream_plan=Path("upstream.json"),
        fixtures=Path("fixtures.jsonl"),
        runtime=Path("runtime"),
        execute=True,
    )

    with pytest.raises(ModelStartReached):
        local.execute(args)

    failure_files = list((tmp_path / "runs").glob("*/failure.json"))
    assert len(failure_files) == 1
    assert json.loads(failure_files[0].read_text())["error_type"] == "ModelStartReached"


def test_cpu_temperature_excludes_wifi_and_other_sensor_readings(monkeypatch) -> None:
    monkeypatch.setattr(
        local,
        "_temperature_readings",
        lambda: ({"hwmon:k10temp:Tctl": 53.5}, {"thermal:iwlwifi_1": 41.0}),
    )

    assert local._cpu_temperatures() == {"hwmon:k10temp:Tctl": 53.5}


def test_stream_records_actual_eos_and_backend_cache_counters(fake_server: str) -> None:
    _SSEHandler.split_after_utf8_lead_byte = False
    _SSEHandler.events = [
        {
            "content": "return 1\n",
            "tokens": [11, 12],
            "stop": True,
            "stop_type": "eos",
            "truncated": False,
            "tokens_cached": 80,
            "tokens_evaluated": 2,
            "timings": {"prompt_ms": 4.5},
        }
    ]

    result = local._capture_stream(fake_server, "synthetic prompt", timeout_seconds=2)

    assert result["raw_output"] == "return 1\n"
    assert result["actual_terminal_observed"] is True
    assert result["hit_output_cap"] is False
    assert result["server_tokens_cached"] == 80
    assert result["server_tokens_evaluated"] == 2
    assert result["cache_counter_evidence"] == "runtime_counters_present"
    assert result["raw_response_not_repaired"] is True
    assert _SSEHandler.request_json["cache_prompt"] is True
    assert _SSEHandler.request_json["n_predict"] == 512
    assert _SSEHandler.request_json["temperature"] == 0
    assert _SSEHandler.request_json["id_slot"] == 0


def test_stream_cap_is_not_promoted_to_terminal_or_repaired(fake_server: str) -> None:
    raw = "x" * 512
    _SSEHandler.events = [
        {
            "content": raw,
            "tokens": list(range(512)),
            "stop": True,
            "stop_type": "limit",
            "truncated": True,
        }
    ]

    result = local._capture_stream(fake_server, "synthetic prompt", timeout_seconds=2)

    assert result["raw_output"] == raw
    assert result["output_token_count"] == 512
    assert result["actual_terminal_observed"] is False
    assert result["hit_output_cap"] is True
    assert result["output_sha256"] == local.digest_bytes(raw.encode())


def test_stream_eof_without_terminal_is_not_no_edit(fake_server: str) -> None:
    _SSEHandler.split_after_utf8_lead_byte = False
    _SSEHandler.events = [{"content": "partial", "tokens": [1], "stop": False}]

    result = local._capture_stream(fake_server, "synthetic prompt", timeout_seconds=2)

    assert result["raw_output"] == "partial"
    assert result["stream_terminal_event_observed"] is False
    assert result["actual_terminal_observed"] is False
    assert result["hit_output_cap"] is False


def test_sse_preserves_utf8_split_across_http_chunks(fake_server: str) -> None:
    _SSEHandler.split_after_utf8_lead_byte = True
    _SSEHandler.events = [
        {
            "content": "π\n",
            "tokens": [31],
            "stop": True,
            "stop_type": "eos",
            "truncated": False,
        }
    ]

    result = local._capture_stream(fake_server, "synthetic prompt", timeout_seconds=2)

    assert result["raw_output"] == "π\n"
    assert result["actual_terminal_observed"] is True


def test_absolute_deadline_cancels_an_incomplete_request(fake_server: str) -> None:
    _SSEHandler.delay_before_response_seconds = 0.2
    deadline = time.monotonic() + 0.03
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        local._capture_stream(
            fake_server,
            "synthetic prompt",
            timeout_seconds=1,
            deadline_monotonic=deadline,
        )

    assert time.monotonic() - started < 0.18


def test_summary_uses_nearest_rank_p95_and_keeps_cache_conditions_separate() -> None:
    rows = [
        {
            "request_kind": "changed_state",
            "precision": "q4_k_m",
            "completed_response_ms": value,
            "actual_terminal_observed": True,
            "hit_output_cap": False,
            "server_tokens_cached": 0,
            "server_tokens_evaluated": 100,
        }
        for value in range(1, 49)
    ]
    rows.append(
        {
            "request_kind": "immediate_same_prompt_repeat",
            "precision": "q4_k_m",
            "completed_response_ms": 10,
            "actual_terminal_observed": False,
            "hit_output_cap": True,
            "server_tokens_cached": 100,
            "server_tokens_evaluated": 0,
        }
    )

    summary = local._summarize(rows)

    changed = summary["q4_k_m/changed_state"]
    repeated = summary["q4_k_m/immediate_same_prompt_repeat"]
    assert changed["requests"] == 48
    assert changed["latency_ms_p95_nearest_rank"] == 46
    assert changed["backend_tokens_cached_median"] == 0
    assert repeated["requests"] == 1
    assert repeated["cap_count"] == 1
    assert repeated["backend_tokens_cached_median"] == 100
