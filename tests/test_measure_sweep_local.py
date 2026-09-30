from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import threading
import time
from argparse import Namespace
from datetime import UTC, datetime, timedelta
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
        if self.path == "/tokenize":
            payload = json.dumps({"tokens": [7, 11, 13]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
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
    assert pairs["--log-verbosity"] == "4"
    assert "--no-cache-idle-slots" in argv
    assert "--no-context-shift" in argv
    assert "--no-repack" in argv


def test_frozen_plan_must_bind_bounded_runtime_and_cache_policy() -> None:
    plan = _minimal_plan()

    local.validate_plan_policy(plan)

    plan["runtime"]["server_settings"]["context_tokens"] = 8192
    with pytest.raises(ValueError, match="server configuration"):
        local.validate_plan_policy(plan)


def _changing_state_replay() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (
        Path(__file__).parents[1]
        / "reports/prototype/sweep_comparison_r1/changing_state_inputs.jsonl"
    )
    rows = local.read_jsonl(path)
    from collections import Counter

    replay = {
        "suite_mode": local.TRANSITION_SUITE_MODE,
        "fixture_order": [row["case_id"] for row in rows],
        "transition_type_counts": dict(Counter(row["transition_type"] for row in rows)),
        "context_modules_by_bucket": {
            bucket: list(
                next(
                    row["context_files"] for row in rows if row["nominal_context_bucket"] == bucket
                )
            )
            for bucket in ("512", "1024", "2048")
        },
        "context_file_sha256": {
            path: sha for row in rows for path, sha in row["context_file_sha256"].items()
        },
    }
    return rows, replay


def test_changing_state_fixtures_replay_exact_history_and_context_provenance() -> None:
    rows, replay = _changing_state_replay()

    local.validate_trajectory_fixtures(rows, replay)

    assert len(rows) == 24
    assert sum(row["transition_type"] == "file_switch_return" for row in rows) == 3
    assert rows[3]["prompt_sha256"] == rows[1]["prompt_sha256"]


def test_header_lf_input_policy_appends_one_lf_and_preserves_fixture_hashes() -> None:
    rows, _ = _changing_state_replay()
    sweep = local._sweep_runner_module()
    fixture = rows[0]
    original_hash = fixture["prompt_sha256"]
    current_hash = fixture["current_sha256"]
    source_prompt = sweep.build_sweep_prompt(fixture)

    model_input, input_hash = local.build_model_input_prompt(
        sweep, fixture, local.PUBLISHER_HEADER_LF_INPUT_POLICY
    )

    assert source_prompt.encode("utf-8") != model_input.encode("utf-8")
    assert model_input == source_prompt + "\n"
    assert model_input.endswith(f"<|file_sep|>updated/{fixture['file_path']}\n")
    assert input_hash == local.digest_bytes(model_input.encode("utf-8"))
    assert fixture["prompt_sha256"] == original_hash
    assert fixture["current_sha256"] == current_hash


def test_header_lf_input_policy_requires_the_unmodified_publisher_header() -> None:
    from types import SimpleNamespace

    fake_sweep = SimpleNamespace(build_sweep_prompt=lambda _fixture: "updated/header\n")
    fixture = {"case_id": "synthetic/case", "file_path": "main.py", "prompt_sha256": ""}

    with pytest.raises(ValueError, match="publisher prompt hash"):
        local.build_model_input_prompt(fake_sweep, fixture, local.PUBLISHER_HEADER_LF_INPUT_POLICY)

    fixture["prompt_sha256"] = local.digest_bytes(b"updated/header\n")
    with pytest.raises(ValueError, match="updated-file header"):
        local.build_model_input_prompt(fake_sweep, fixture, local.PUBLISHER_HEADER_LF_INPUT_POLICY)


def test_token_bucket_validation_binds_policy_prompt_hashes_and_actual_counts() -> None:
    fixtures = [{"case_id": "state-a", "nominal_context_bucket": "512"}]
    token_rows = [
        {
            "case_id": "state-a",
            "input_tokens": 511,
            "effective_input_prompt_sha256": "a" * 64,
        }
    ]
    replay = {
        "input_bucket_windows": {"512": [384, 768]},
        "effective_input_prompt_sha256_by_case": {"state-a": "a" * 64},
        "input_tokens_by_case": {"state-a": 511},
    }

    local.validate_input_token_buckets(fixtures, token_rows, replay)

    token_rows[0]["effective_input_prompt_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="prompt hashes"):
        local.validate_input_token_buckets(fixtures, token_rows, replay)
    token_rows[0]["effective_input_prompt_sha256"] = "a" * 64
    token_rows[0]["input_tokens"] = 510
    with pytest.raises(ValueError, match="token counts"):
        local.validate_input_token_buckets(fixtures, token_rows, replay)


def test_valid_action_latency_excludes_incomplete_and_out_of_range_outputs() -> None:
    completed = {
        "actual_terminal_observed": True,
        "hit_output_cap": False,
        "client_request_ms": 42.0,
    }

    invalid = local.response_action_measurements(
        completed, {"mapping": "out_of_range", "out_of_range": True}
    )
    assert invalid["completed_file_output"] is True
    assert invalid["completed_file_output_latency_ms"] == 42.0
    assert invalid["valid_canonical_action"] is False
    assert invalid["canonical_action"] is None
    assert invalid["valid_canonical_action_latency_ms"] is None

    valid = local.response_action_measurements(
        completed,
        {"mapping": "within_editable_range", "action": "replace", "replacement": "return 2"},
    )
    assert valid["valid_canonical_action"] is True
    assert valid["canonical_action"] == {"action": "replace", "text": "return 2"}
    assert valid["valid_canonical_action_latency_ms"] == 42.0

    capped = local.response_action_measurements(
        {**completed, "hit_output_cap": True},
        {"mapping": "within_editable_range", "action": "replace", "replacement": "x"},
    )
    assert capped["completed_file_output"] is False
    assert capped["valid_canonical_action"] is False


def test_changing_state_fixture_rejects_tampered_byte_delta_and_context() -> None:
    rows, replay = _changing_state_replay()
    rows[1]["synthetic_event"]["replacement_text"] = "fabricated"

    with pytest.raises(ValueError, match="byte delta"):
        local.validate_trajectory_fixtures(rows, replay)

    rows, replay = _changing_state_replay()
    rows[0]["context_files"]["src/editor/ranges.py"] += "\n# added after freeze\n"
    with pytest.raises(ValueError, match="context bytes"):
        local.validate_trajectory_fixtures(rows, replay)


def test_changing_state_plan_binds_remaining_campaign_budget_and_single_requests() -> None:
    rows, replay = _changing_state_replay()
    replay.update(
        fixture_count=24,
        expected_requests=96,
        request_order_per_fixture_per_repetition=["changed_state"],
        precision_order=["q4_k_m", "q8_0"],
        repetitions=2,
        request_settings=dict(local.REQUEST_SETTINGS),
        input_bucket_windows={"512": [384, 768], "1024": [900, 1450], "2048": [1728, 2304]},
        input_bucket_case_counts={"512": 8, "1024": 8, "2048": 8},
    )
    plan = {
        "schema": local.TRANSITION_PLAN_SCHEMA,
        "runtime": {"server_settings": dict(local.SERVER_SETTINGS)},
        "replay": replay,
        "limits": {
            "finalization_reserve_seconds": 600,
            "max_wall_seconds_including_preflight_and_finalization": 3700,
            "request_work_deadline_seconds_from_invocation_start": 3100,
            "minimum_available_ram_bytes": 4 * 1024**3,
            "candidate_predictor_memory_bytes": int(1.5 * 1024**3),
        },
        "campaign_budget": {
            "previous_run_wall_seconds": 1696.506838,
            "campaign_cap_seconds": 5400,
            "previous_run_summary_path": local.BASELINE_SUMMARY_PATH,
            "previous_run_summary_sha256": local.BASELINE_SUMMARY_SHA256,
            "tokenizer_preflight_wall_seconds": 3.296,
            "remaining_seconds_before_new_run": 3700.197162,
        },
    }

    local.validate_plan_policy(plan)

    campaign_budget = plan["campaign_budget"]
    campaign_budget["tokenizer_preflight_wall_seconds"] = 8.60658
    campaign_budget["interrupted_run_wall_seconds"] = 65.019949
    campaign_budget["pre_run_plan_validation_wall_seconds"] = 1.000638
    remaining = (
        local.MAX_WALL_SECONDS
        - campaign_budget["previous_run_wall_seconds"]
        - campaign_budget["tokenizer_preflight_wall_seconds"]
        - campaign_budget["interrupted_run_wall_seconds"]
        - campaign_budget["pre_run_plan_validation_wall_seconds"]
    )
    campaign_budget["remaining_seconds_before_new_run"] = remaining
    plan["limits"]["max_wall_seconds_including_preflight_and_finalization"] = int(remaining)
    plan["limits"]["request_work_deadline_seconds_from_invocation_start"] = int(remaining) - 600
    campaign_budget["max_wall_seconds_including_finalization"] = int(remaining)
    campaign_budget["request_work_deadline_seconds"] = int(remaining) - 600
    campaign_budget["interrupted_attempt"] = {"fixture": "synthetic"}

    local.validate_plan_policy(plan)
    assert plan["limits"]["max_wall_seconds_including_preflight_and_finalization"] == 3628

    campaign_budget["max_wall_seconds_including_finalization"] += 1
    with pytest.raises(ValueError, match="campaign budget max wall"):
        local.validate_plan_policy(plan)
    campaign_budget["max_wall_seconds_including_finalization"] -= 1

    replay["expected_requests"] = 192
    with pytest.raises(ValueError, match="24 transitions twice"):
        local.validate_plan_policy(plan)


def test_interrupted_attempt_artifact_hashes_and_wall_time_are_verified(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial-run"
    run_dir.mkdir()
    started = datetime(2026, 9, 30, 11, 30, 3, 640105, tzinfo=UTC)
    failed = started + timedelta(seconds=65, microseconds=19_949)
    metadata = {
        "plan_sha256": "prior-plan",
        "invocation_deadline_started_at_utc": started.isoformat(),
    }
    failure = {
        "plan_sha256": "prior-plan",
        "error_type": "KeyboardInterrupt",
        "completed_requests": 7,
        "failure_at_utc": failed.isoformat(),
    }
    metadata_path = run_dir / "metadata.json"
    failure_path = run_dir / "failure.json"
    predictions_path = run_dir / "predictions-q4_k_m.jsonl"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    predictions_path.write_text("{}\n" * 7, encoding="utf-8")
    reference = {
        "run_directory": str(run_dir),
        "plan_sha256": "prior-plan",
        "prior_plan_sha256": "prior-plan",
        "completed_q4_records": 7,
        "q8_request_count": 0,
        "wall_seconds": 65.019949,
        "artifact_sha256": {
            "metadata": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "failure": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
            "q4_predictions": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        },
    }
    plan = {"campaign_budget": {"interrupted_run_wall_seconds": 65.019949}}

    local._verify_interrupted_attempt(reference, plan)

    failure_path.write_text(json.dumps({**failure, "completed_requests": 6}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        local._verify_interrupted_attempt(reference, plan)


def test_token_prefix_uses_actual_token_ids_and_not_character_prefixes() -> None:
    assert local._common_token_prefix([7, 11, 13], [7, 11, 17]) == 2
    assert local._common_token_prefix([], [7]) == 0


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


def test_tokenize_returns_actual_token_ids_and_time(fake_server: str) -> None:
    token_ids, elapsed_ms = local._tokenize(fake_server, "synthetic prompt")

    assert token_ids == [7, 11, 13]
    assert elapsed_ms >= 0


def test_stream_observes_eight_and_sixteen_token_id_thresholds(fake_server: str) -> None:
    _SSEHandler.events = [
        {
            "content": "synthetic",
            "tokens": list(range(1, 17)),
            "stop": True,
            "stop_type": "eos",
            "truncated": False,
        }
    ]

    result = local._capture_stream(fake_server, "synthetic prompt", timeout_seconds=2)

    assert result["time_to_8_output_token_ids_ms"] == result["first_token_ids_ms"]
    assert result["time_to_16_output_token_ids_ms"] == result["first_token_ids_ms"]


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
